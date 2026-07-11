package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestExecuteJSONRequest(t *testing.T) {
	type receivedRequest struct {
		Method        string
		Path          string
		Authorization string
		ContentType   string
		Body          map[string]any
	}

	received := make(chan receivedRequest, 1)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var decoded map[string]any
		if err := json.Unmarshal(body, &decoded); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		received <- receivedRequest{
			Method:        r.Method,
			Path:          r.URL.Path,
			Authorization: r.Header.Get("Authorization"),
			ContentType:   r.Header.Get("Content-Type"),
			Body:          decoded,
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "test-123",
			"status": "ok",
		})
	}))
	defer server.Close()

	requestBody := map[string]any{
		"name": "daksha",
	}

	var response struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}

	err := ExecuteNewRequest(
		context.Background(),
		http.MethodPost,
		"/v1/test",
		requestBody,
		&response,
		WithDefaultBaseURL(server.URL+"/"),
		RequestOptionFunc(func(cfg *RequestConfig) error {
			cfg.SetAPIKey("test-api-key")
			cfg.RequestTimeout = 3 * time.Second
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("ExecuteNewRequest() error = %v", err)
	}

	req := <-received

	if req.Method != http.MethodPost {
		t.Errorf("method = %q, want POST", req.Method)
	}

	if req.Path != "/v1/test" {
		t.Errorf("path = %q, want /v1/test", req.Path)
	}

	if req.Authorization != "Bearer test-api-key" {
		t.Errorf(
			"Authorization = %q, want %q",
			req.Authorization,
			"Bearer test-api-key",
		)
	}

	if req.ContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", req.ContentType)
	}

	if req.Body["name"] != "daksha" {
		t.Errorf("request body = %#v", req.Body)
	}

	if response.ID != "test-123" {
		t.Errorf("response.ID = %q, want test-123", response.ID)
	}

	if response.Status != "ok" {
		t.Errorf("response.Status = %q, want ok", response.Status)
	}
}

func TestExecuteTextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte("hello"))
	}))
	defer server.Close()

	var response string

	err := ExecuteNewRequest(
		context.Background(),
		http.MethodGet,
		"/text",
		nil,
		&response,
		WithDefaultBaseURL(server.URL+"/"),
	)
	if err != nil {
		t.Fatalf("ExecuteNewRequest() error = %v", err)
	}

	if response != "hello" {
		t.Errorf("response = %q, want hello", response)
	}
}

func TestEncodePathParam(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "single dot",
			input: ".",
			want:  "%2E",
		},
		{
			name:  "double dot",
			input: "..",
			want:  "%2E%2E",
		},
		{
			name:  "space",
			input: "hello world",
			want:  "hello%20world",
		},
		{
			name:  "slash",
			input: "a/b",
			want:  "a%2Fb",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := encodePathParam(tt.input)
			if got != tt.want {
				t.Errorf(
					"encodePathParam(%q) = %q, want %q",
					tt.input,
					got,
					tt.want,
				)
			}
		})
	}
}

func TestExecuteHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)

		_, _ = w.Write([]byte(`{
			"error": {
				"message": "invalid API key",
				"type": "authentication_error"
			}
		}`))
	}))
	defer server.Close()

	var response map[string]any

	err := ExecuteNewRequest(
		context.Background(),
		http.MethodGet,
		"/protected",
		nil,
		&response,
		WithDefaultBaseURL(server.URL+"/"),
	)

	if err == nil {
		t.Fatal("expected an error, got nil")
	}
}
