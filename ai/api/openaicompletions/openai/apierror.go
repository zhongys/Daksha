package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// APIError represents an OpenAI-compatible error payload. It is returned for
// non-2xx responses, error envelopes received where a success payload was
// expected (including status 2xx), and, through StreamError.Unwrap, errors
// emitted mid-stream.
type APIError struct {
	// StatusCode is zero for errors that did not come from an HTTP response,
	// e.g. errors embedded in a streaming event.
	StatusCode int
	Message    string
	Type       string
	Code       string
	Param      string
	// Body is the raw response body, kept for logging and debugging.
	Body string
}

func (e *APIError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("openai-compatible API error: status=%d type=%s message=%s", e.StatusCode, e.Type, e.Message)
	}
	return fmt.Sprintf("openai-compatible API error: type=%s message=%s", e.Type, e.Message)
}

// ParseAPIError builds an APIError from a response body shaped like
// {"error": {"message": ..., "type": ..., "code": ..., "param": ...}}.
// Bodies that do not match the envelope still produce a usable error
// carrying the status code and raw body.
func ParseAPIError(statusCode int, body []byte) *APIError {
	apiErr := parseAPIErrorData(body, true)
	apiErr.StatusCode = statusCode
	apiErr.Body = string(body)
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(statusCode)
	}
	return apiErr
}

// errorFromEventData inspects a streaming event payload for an error
// envelope. It returns nil when the payload carries no error. Any non-null
// "error" key counts as an error, even without standard message/type fields.
func errorFromEventData(data []byte) *APIError {
	payload, ok := errorEnvelopePayload(data)
	if !ok {
		return nil
	}
	apiErr := parseAPIErrorValue(payload)
	apiErr.Body = string(data)
	if apiErr.Message == "" {
		apiErr.Message = compactErrorValue(payload)
	}
	return apiErr
}

func parseErrorEnvelope(data []byte) *APIError {
	payload, ok := errorEnvelopePayload(data)
	if !ok {
		return &APIError{}
	}
	return parseAPIErrorValue(payload)
}

// parseAPIErrorData accepts both the standard {"error": {...}} envelope and
// the flat error objects commonly carried by explicit SSE `event: error`
// records. Keeping the two shapes here gives HTTP and streaming callers the
// same tolerant field conversion.
func parseAPIErrorData(data []byte, allowFlat bool) *APIError {
	if payload, ok := errorEnvelopePayload(data); ok {
		return parseAPIErrorValue(payload)
	}
	if allowFlat {
		return parseAPIErrorValue(data)
	}
	return &APIError{}
}

func errorEnvelopePayload(data []byte) (json.RawMessage, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return nil, false
	}
	payload, ok := object["error"]
	if !ok || len(bytes.TrimSpace(payload)) == 0 || bytes.Equal(bytes.TrimSpace(payload), []byte("null")) {
		return nil, false
	}
	return payload, true
}

func parseAPIErrorValue(data []byte) *APIError {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil || fields == nil {
		return &APIError{Message: compactErrorValue(data)}
	}
	return &APIError{
		Message: compactErrorValue(fields["message"]),
		Type:    compactErrorValue(fields["type"]),
		Code:    compactErrorValue(fields["code"]),
		Param:   compactErrorValue(fields["param"]),
	}
}

// compactErrorValue converts the scalar variants seen in compatible APIs to
// strings without losing numeric precision. In particular, several providers
// emit an integer `code`, while OpenAI documents it as a string.
func compactErrorValue(raw json.RawMessage) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return ""
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err == nil {
		switch value := value.(type) {
		case string:
			return value
		case json.Number:
			return value.String()
		case bool:
			return fmt.Sprint(value)
		}
	}
	return strings.TrimSpace(string(raw))
}
