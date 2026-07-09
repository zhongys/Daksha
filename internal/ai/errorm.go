package models

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type ModelsError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
	Body       string
}

func (e *ModelsError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("openai-compatible API error: status=%d message=%s", e.StatusCode, e.Message)
	}
	return fmt.Sprintf("openai-compatible API error: message=%s", e.Message)
}

func checkStatus(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return statusError(resp)
}

func statusError(resp *http.Response) error {
	bodyBytes, readErr := io.ReadAll(
		io.LimitReader(
			resp.Body,
			1024*1024,
		),
	)
	if readErr != nil {
		return fmt.Errorf(
			"read API error body: %w",
			readErr,
		)
	}
	apiErr := parseAPIError(bodyBytes)
	apiErr.StatusCode = resp.StatusCode
	apiErr.Body = string(bodyBytes)
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(resp.StatusCode)
	}
	return apiErr
}

func errorFromEventData(data []byte) error {
	apiErr := parseAPIError(data)
	if apiErr.Message == "" {
		return nil
	}
	apiErr.Body = string(data)
	return apiErr
}

func parseAPIError(data []byte) *ModelsError {
	var envelope struct {
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil || envelope.Error == nil {
		return &ModelsError{}
	}
	return &ModelsError{
		Message: envelope.Error.Message,
		Type:    envelope.Error.Type,
		Code:    envelope.Error.Code,
	}
}
