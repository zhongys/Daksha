package ai

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// APIError is the provider-neutral error returned when an OpenAI-compatible
// endpoint responds with a non-2xx status, or emits an error event mid-stream.
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
	apiErr := parseErrorEnvelope(body)
	apiErr.StatusCode = statusCode
	apiErr.Body = string(body)
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(statusCode)
	}
	return apiErr
}

// ErrorFromEventData inspects a streaming event payload for an error envelope.
// It returns nil when the payload carries no error. Any non-null "error" key
// counts as an error, even if it lacks the standard message/type fields.
func ErrorFromEventData(data []byte) *APIError {
	var probe struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(data, &probe) != nil || len(probe.Error) == 0 || string(probe.Error) == "null" {
		return nil
	}
	apiErr := parseErrorEnvelope(data)
	apiErr.Body = string(data)
	if apiErr.Message == "" {
		apiErr.Message = string(probe.Error)
	}
	return apiErr
}

func parseErrorEnvelope(data []byte) *APIError {
	var envelope struct {
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
			Param   string `json:"param"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil || envelope.Error == nil {
		return &APIError{}
	}
	return &APIError{
		Message: envelope.Error.Message,
		Type:    envelope.Error.Type,
		Code:    envelope.Error.Code,
		Param:   envelope.Error.Param,
	}
}
