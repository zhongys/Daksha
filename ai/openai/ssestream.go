package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

type Decoder interface {
	Event() Event
	Next() bool
	Close() error
	Err() error
}

func NewDecoder(res *http.Response) Decoder {
	if res == nil || res.Body == nil {
		return nil
	}

	scn := bufio.NewScanner(res.Body)
	scn.Buffer(nil, bufio.MaxScanTokenSize<<9)
	return &eventStreamDecoder{rc: res.Body, scn: scn}
}

type Event struct {
	Type string
	Data []byte
}

// StreamError represents an error event that occurred during streaming,
// preserving the original event data for structured access.
type StreamError struct {
	Message string
	Event   Event
	Cause   error
}

func (e *StreamError) Error() string {
	return e.Message
}

func (e *StreamError) Unwrap() error {
	return e.Cause
}

// A base implementation of a Decoder for text/event-stream.
type eventStreamDecoder struct {
	evt Event
	rc  io.ReadCloser
	scn *bufio.Scanner
	err error
}

func (s *eventStreamDecoder) Next() bool {
	if s.err != nil {
		return false
	}

	event := ""
	data := bytes.NewBuffer(nil)

	for s.scn.Scan() {
		txt := s.scn.Bytes()

		// Dispatch event on an empty line
		if len(txt) == 0 {
			// SSE comments and empty blocks do not dispatch events. Reset the
			// event name so a comment-only heartbeat cannot leak state into the
			// next data block.
			if data.Len() == 0 {
				event = ""
				continue
			}
			s.evt = Event{
				Type: event,
				Data: data.Bytes(),
			}
			return true
		}

		// Split a string like "event: bar" into name="event" and value=" bar".
		name, value, _ := bytes.Cut(txt, []byte(":"))

		// Consume an optional space after the colon if it exists.
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}

		switch string(name) {
		case "":
			// A line in the form ": something" is a comment and should be ignored.
			continue
		case "event":
			event = string(value)
		case "data":
			_, s.err = data.Write(value)
			if s.err != nil {
				break
			}
			_, s.err = data.WriteRune('\n')
			if s.err != nil {
				break
			}
		}
	}

	if err := s.scn.Err(); err != nil {
		s.err = err
		return false
	}

	// Compatibility: some OpenAI-compatible endpoints omit the final empty
	// line. After a clean EOF, dispatch their pending data block as if it had
	// been terminated normally.
	if data.Len() > 0 {
		s.evt = Event{
			Type: event,
			Data: data.Bytes(),
		}
		return true
	}

	return false
}

func (s *eventStreamDecoder) Event() Event {
	return s.evt
}

func (s *eventStreamDecoder) Close() error {
	return s.rc.Close()
}

func (s *eventStreamDecoder) Err() error {
	return s.err
}

type Stream[T any] struct {
	decoder Decoder
	cur     T
	err     error
	done    bool
}

func NewStream[T any](decoder Decoder, err error) *Stream[T] {
	return &Stream[T]{
		decoder: decoder,
		err:     err,
	}
}

// Next returns false if the stream has ended or an error occurred.
// Call Stream.Current() to get the current value.
// Call Stream.Err() to get the error.
//
//	for stream.Next() {
//		data := stream.Current()
//	}
//
//	if stream.Err() != nil {
//		...
//	}
func (s *Stream[T]) Next() bool {
	if s.err != nil || s.done {
		return false
	}
	if s.decoder == nil {
		s.err = errors.New("openai: streaming response has no decoder")
		return false
	}

	for s.decoder.Next() {
		event := cloneEvent(s.decoder.Event())
		data := event.Data

		// An explicit SSE error event is terminal even when its payload is a
		// flat error object rather than the usual {"error": {...}} envelope.
		// Check the event name before [DONE] so `event: error` can never be
		// disguised as a clean terminator.
		if event.Type == "error" {
			apiErr := errorFromEventData(data)
			if apiErr == nil {
				apiErr = parseAPIErrorData(data, true)
				apiErr.Body = string(data)
			}
			if apiErr.Message == "" {
				apiErr.Message = "stream emitted an error event"
			}
			s.err = newAPIStreamError(event, apiErr)
			return false
		}

		if bytes.Equal(bytes.TrimSuffix(data, []byte("\n")), []byte("[DONE]")) {
			s.done = true
			return false
		}

		if apiErr := errorFromEventData(data); apiErr != nil {
			s.err = newAPIStreamError(event, apiErr)
			return false
		}

		trimmed := bytes.TrimSpace(data)
		if len(trimmed) == 0 {
			s.err = newDecodeStreamError(event, errors.New("empty streaming event data"))
			return false
		}
		if bytes.Equal(trimmed, []byte("null")) {
			s.err = newDecodeStreamError(event, errors.New("null streaming event data"))
			return false
		}

		var nxt T
		if err := json.Unmarshal(data, &nxt); err != nil {
			s.err = newDecodeStreamError(event, err)
			return false
		}
		s.cur = nxt
		return true
	}

	// decoder.Next() may be false because of an error
	s.err = s.decoder.Err()

	return false
}

func (s *Stream[T]) Current() T {
	return s.cur
}

func (s *Stream[T]) Err() error {
	return s.err
}

func (s *Stream[T]) Close() error {
	if s.decoder == nil {
		// already closed
		return nil
	}
	return s.decoder.Close()
}

func cloneEvent(event Event) Event {
	event.Data = append([]byte(nil), event.Data...)
	return event
}

func newAPIStreamError(event Event, apiErr *APIError) *StreamError {
	return &StreamError{
		Message: fmt.Sprintf("received error while streaming: %s", apiErr.Message),
		Event:   event,
		Cause:   apiErr,
	}
}

func newDecodeStreamError(event Event, cause error) *StreamError {
	return &StreamError{
		Message: fmt.Sprintf("failed to decode streaming event: %v", cause),
		Event:   event,
		Cause:   cause,
	}
}
