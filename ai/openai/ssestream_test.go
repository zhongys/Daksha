package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type terminalErrorReader struct{ err error }

func (r terminalErrorReader) Read([]byte) (int, error) { return 0, r.err }

func TestStreamSkipsCommentOnlySSEBlocks(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		": keepalive\n\n\n" +
			"data: {\"value\":1}\n\n" +
			"data: [DONE]\n\n",
	))}
	stream := NewStream[struct {
		Value int `json:"value"`
	}](NewDecoder(res), nil)
	defer stream.Close()

	if !stream.Next() {
		t.Fatalf("Next() = false, err = %v", stream.Err())
	}
	if got := stream.Current().Value; got != 1 {
		t.Fatalf("value = %d, want 1", got)
	}
	if stream.Next() {
		t.Fatal("Next() returned an event after [DONE]")
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

func TestStreamDoneDoesNotWaitForResponseBodyEOF(t *testing.T) {
	reader, writer := io.Pipe()
	stream := NewStream[struct{}](NewDecoder(&http.Response{Body: reader}), nil)
	defer stream.Close()
	defer writer.Close()

	written := make(chan error, 1)
	go func() {
		_, err := io.WriteString(writer, "data: [DONE]\n\n")
		written <- err
	}()

	next := make(chan bool, 1)
	go func() {
		next <- stream.Next()
	}()

	select {
	case got := <-next:
		if got {
			t.Fatal("Next() = true after [DONE], want false")
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Next() waited for response body EOF after [DONE]")
	}

	if err := <-written; err != nil {
		t.Fatalf("writing [DONE]: %v", err)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

func TestStreamRequiresExactDoneMarker(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		"data: [DONE] trailing-data\n\n",
	))}
	stream := NewStream[struct{}](NewDecoder(res), nil)
	defer stream.Close()

	if stream.Next() {
		t.Fatal("Next() accepted invalid JSON after a [DONE] prefix")
	}
	if stream.Err() == nil {
		t.Fatal("Err() = nil, want JSON decoding error for a non-exact [DONE] marker")
	}
	var streamErr *StreamError
	if !errors.As(stream.Err(), &streamErr) {
		t.Fatalf("Err() type = %T, want *StreamError", stream.Err())
	}
	if got := strings.TrimSpace(string(streamErr.Event.Data)); got != "[DONE] trailing-data" {
		t.Fatalf("StreamError event data = %q", got)
	}
}

func TestStreamExplicitErrorEventPreservesFlatAPIError(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		"event: error\n" +
			`data: {"message":"overloaded","type":"server_error","code":529,"param":"model"}` + "\n\n" +
			"data: {\"value\":1}\n\n" +
			"data: [DONE]\n\n",
	))}
	stream := NewStream[struct {
		Value int `json:"value"`
	}](NewDecoder(res), nil)
	defer stream.Close()

	if stream.Next() {
		t.Fatal("Next() returned a value for event: error")
	}
	var streamErr *StreamError
	if !errors.As(stream.Err(), &streamErr) {
		t.Fatalf("Err() = %T, want *StreamError", stream.Err())
	}
	if streamErr.Event.Type != "error" || !strings.Contains(string(streamErr.Event.Data), "overloaded") {
		t.Fatalf("preserved event = %#v", streamErr.Event)
	}
	var apiErr *APIError
	if !errors.As(stream.Err(), &apiErr) {
		t.Fatalf("Err() does not unwrap to *APIError: %v", stream.Err())
	}
	if apiErr.Message != "overloaded" || apiErr.Type != "server_error" || apiErr.Code != "529" || apiErr.Param != "model" {
		t.Fatalf("APIError = %#v", apiErr)
	}
	if stream.Next() {
		t.Fatal("stream continued to the success event after event: error")
	}
}

func TestStreamExplicitErrorEventAcceptsPlainText(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		"event: error\ndata: overloaded\n\n",
	))}
	stream := NewStream[struct{}](NewDecoder(res), nil)
	defer stream.Close()

	if stream.Next() {
		t.Fatal("Next() returned a value for event: error")
	}
	var apiErr *APIError
	if !errors.As(stream.Err(), &apiErr) || apiErr.Message != "overloaded" {
		t.Fatalf("Err() = %#v, want plain-text APIError", stream.Err())
	}
}

func TestStreamErrorEnvelopeUnwrapsAPIError(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		`data: {"error":{"message":"quota","type":"rate_limit","code":429}}` + "\n\n",
	))}
	stream := NewStream[struct{}](NewDecoder(res), nil)
	defer stream.Close()

	if stream.Next() {
		t.Fatal("Next() returned an error envelope as a value")
	}
	var streamErr *StreamError
	var apiErr *APIError
	if !errors.As(stream.Err(), &streamErr) || !errors.As(stream.Err(), &apiErr) {
		t.Fatalf("Err() chain = %T %v", stream.Err(), stream.Err())
	}
	if apiErr.Code != "429" || apiErr.Body == "" {
		t.Fatalf("APIError = %#v", apiErr)
	}
}

func TestStreamRejectsNullEvent(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader("data: null\n\n"))}
	stream := NewStream[struct{}](NewDecoder(res), nil)
	defer stream.Close()

	if stream.Next() {
		t.Fatal("Next() accepted a null event")
	}
	var streamErr *StreamError
	if !errors.As(stream.Err(), &streamErr) || !strings.Contains(streamErr.Error(), "null streaming event") {
		t.Fatalf("Err() = %#v, want null *StreamError", stream.Err())
	}
}

func TestStreamDispatchesFinalDataBlockAtEOF(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		"event: result\ndata: {\"value\":7}",
	))}
	stream := NewStream[struct {
		Value int `json:"value"`
	}](NewDecoder(res), nil)
	defer stream.Close()

	if !stream.Next() {
		t.Fatalf("Next() = false, err = %v", stream.Err())
	}
	if got := stream.Current().Value; got != 7 {
		t.Fatalf("value = %d, want 7", got)
	}
	if stream.Next() {
		t.Fatal("Next() returned the final event more than once")
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

func TestStreamScannerErrorWinsOverPendingData(t *testing.T) {
	wantErr := errors.New("read failed")
	reader := io.MultiReader(
		strings.NewReader("data: {\"value\":7}"),
		terminalErrorReader{err: wantErr},
	)
	stream := NewStream[struct {
		Value int `json:"value"`
	}](NewDecoder(&http.Response{Body: io.NopCloser(reader)}), nil)
	defer stream.Close()

	if stream.Next() {
		t.Fatal("Next() dispatched pending data after a scanner error")
	}
	if !errors.Is(stream.Err(), wantErr) {
		t.Fatalf("Err() = %v, want %v", stream.Err(), wantErr)
	}
}

func TestStreamingTimeoutLivesUntilResponseBodyCloses(t *testing.T) {
	allowChunk := make(chan struct{})
	requestDone := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(requestDone)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()

		// Do not send the first chunk until NewStreaming (and therefore Execute)
		// has returned. With the old deferred cancel, the request context wins
		// this select and the client can never receive the chunk.
		select {
		case <-allowChunk:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, "data: {\"id\":\"chunk-1\",\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()

	service := NewChatCompletionService(
		WithBaseURL(server.URL+"/"),
		// Keep the natural deadline well outside the Close assertion below,
		// so only Close-triggered cancellation can make the test pass.
		WithRequestTimeout(10*time.Second),
	)
	stream := service.NewStreaming(context.Background(), ChatCompletionNewParams{
		Model:    "m",
		Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
	})
	defer stream.Close()
	close(allowChunk)

	if !stream.Next() {
		t.Fatalf("Next() = false, err = %v", stream.Err())
	}
	if got := stream.Current().ID; got != "chunk-1" {
		t.Fatalf("chunk id = %q, want chunk-1", got)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	select {
	case <-requestDone:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("closing the stream did not cancel the request context")
	}
}
