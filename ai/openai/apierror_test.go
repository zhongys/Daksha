package openai

import (
	"net/http"
	"testing"
)

func TestParseAPIErrorAcceptsEnvelopeAndFlatNumericFields(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "standard envelope",
			body: `{"error":{"message":"bad request","type":"invalid_request_error","code":9007199254740995,"param":7}}`,
		},
		{
			name: "flat error event payload",
			body: `{"message":"bad request","type":"invalid_request_error","code":9007199254740995,"param":7}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			apiErr := ParseAPIError(http.StatusBadRequest, []byte(test.body))
			if apiErr.Message != "bad request" || apiErr.Type != "invalid_request_error" {
				t.Fatalf("APIError = %#v", apiErr)
			}
			if apiErr.Code != "9007199254740995" || apiErr.Param != "7" {
				t.Fatalf("numeric fields = code %q, param %q", apiErr.Code, apiErr.Param)
			}
			if apiErr.Body != test.body {
				t.Fatalf("Body = %q, want %q", apiErr.Body, test.body)
			}
		})
	}
}

func TestParseAPIErrorAcceptsScalarEnvelope(t *testing.T) {
	apiErr := ParseAPIError(http.StatusServiceUnavailable, []byte(`{"error":"overloaded"}`))
	if apiErr.Message != "overloaded" {
		t.Fatalf("Message = %q, want overloaded", apiErr.Message)
	}
}
