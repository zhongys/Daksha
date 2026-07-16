package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type httpDoerFunc func(*http.Request) (*http.Response, error)

func (f httpDoerFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

type trackingResponseBody struct {
	reader io.Reader
	closed bool
}

func (b *trackingResponseBody) Read(p []byte) (int, error) {
	if b.reader == nil {
		return 0, io.EOF
	}
	return b.reader.Read(p)
}
func (b *trackingResponseBody) Close() error {
	b.closed = true
	return nil
}

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

func TestExecuteRejectsNon2xxResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMultipleChoices)
		_, _ = w.Write([]byte(`{"error":{"message":"choose another endpoint","type":"redirect"}}`))
	}))
	defer server.Close()

	var response map[string]any
	err := ExecuteNewRequest(
		context.Background(),
		http.MethodGet,
		"/redirected",
		nil,
		&response,
		WithDefaultBaseURL(server.URL+"/"),
	)
	if err == nil {
		t.Fatal("expected non-2xx response to return an error")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("error type = %T, want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusMultipleChoices {
		t.Fatalf("status = %d, want %d", apiErr.StatusCode, http.StatusMultipleChoices)
	}
}

func TestExecuteClosesUnownedResponseBody(t *testing.T) {
	body := &trackingResponseBody{}
	err := ExecuteNewRequest(
		context.Background(),
		http.MethodGet,
		"/ignored",
		nil,
		nil,
		WithDefaultBaseURL("https://example.test/"),
		WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusNoContent,
				Header:     make(http.Header),
				Body:       body,
				Request:    req,
			}, nil
		})),
	)
	if err != nil {
		t.Fatalf("ExecuteNewRequest: %v", err)
	}
	if !body.closed {
		t.Fatal("unowned response body was not closed")
	}
}

func TestExecuteLeavesRawResponseBodyWithItsOwner(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T, body *trackingResponseBody)
	}{
		{
			name: "ResponseInto without decoded destination",
			run: func(t *testing.T, body *trackingResponseBody) {
				var raw *http.Response
				err := ExecuteNewRequest(
					context.Background(),
					http.MethodGet,
					"/raw",
					nil,
					nil,
					WithResponseInto(&raw),
					WithDefaultBaseURL("https://example.test/"),
					WithHTTPClient(responseDoer(body)),
				)
				if err != nil {
					t.Fatalf("ExecuteNewRequest: %v", err)
				}
				if raw == nil {
					t.Fatal("ResponseInto was not populated")
				}
				if body.closed {
					t.Fatal("Execute closed the body before its ResponseInto owner")
				}
				if err := raw.Body.Close(); err != nil {
					t.Fatalf("closing owned body: %v", err)
				}
			},
		},
		{
			name: "ResponseBodyInto raw response",
			run: func(t *testing.T, body *trackingResponseBody) {
				var raw *http.Response
				err := ExecuteNewRequest(
					context.Background(),
					http.MethodGet,
					"/raw",
					nil,
					&raw,
					WithDefaultBaseURL("https://example.test/"),
					WithHTTPClient(responseDoer(body)),
				)
				if err != nil {
					t.Fatalf("ExecuteNewRequest: %v", err)
				}
				if raw == nil {
					t.Fatal("raw response destination was not populated")
				}
				if body.closed {
					t.Fatal("Execute closed the body before its raw-response owner")
				}
				if err := raw.Body.Close(); err != nil {
					t.Fatalf("closing owned body: %v", err)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := &trackingResponseBody{}
			tt.run(t, body)
			if !body.closed {
				t.Fatal("response owner could not close the body")
			}
		})
	}
}

func TestExecuteConsumesBodyWhenResponseIntoOnlyObservesDecodedResponse(t *testing.T) {
	body := &trackingResponseBody{reader: bytes.NewBufferString(`{"ok":true}`)}
	var raw *http.Response
	var decoded struct {
		OK bool `json:"ok"`
	}
	err := ExecuteNewRequest(
		context.Background(),
		http.MethodGet,
		"/decoded",
		nil,
		&decoded,
		WithResponseInto(&raw),
		WithDefaultBaseURL("https://example.test/"),
		WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       body,
				Request:    req,
			}, nil
		})),
	)
	if err != nil {
		t.Fatalf("ExecuteNewRequest: %v", err)
	}
	if raw == nil {
		t.Fatal("ResponseInto was not populated")
	}
	if !decoded.OK {
		t.Fatal("decoded response was not populated")
	}
	if !body.closed {
		t.Fatal("decoded response body was not closed")
	}
}

func TestRequestConfigCloneWithBodyAndNilGetBody(t *testing.T) {
	body := io.NopCloser(bytes.NewBufferString("request body"))
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, "https://example.test/", body)
	if err != nil {
		t.Fatalf("NewRequestWithContext: %v", err)
	}
	if req.GetBody != nil {
		t.Fatal("test setup unexpectedly produced a GetBody function")
	}

	cfg := &RequestConfig{Request: req}
	clone := cfg.Clone(context.Background())
	if clone != nil {
		t.Fatal("Clone returned a config for a request whose body cannot be replayed")
	}
}

func TestExecuteClosesUnownedBodyOnEarlyErrors(t *testing.T) {
	t.Run("request context canceled after response", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		body := &trackingResponseBody{}
		err := ExecuteNewRequest(
			ctx,
			http.MethodGet,
			"/canceled",
			nil,
			nil,
			WithDefaultBaseURL("https://example.test/"),
			WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
				cancel()
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       body,
					Request:    req,
				}, nil
			})),
		)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("ExecuteNewRequest error = %v, want context.Canceled", err)
		}
		if !body.closed {
			t.Fatal("canceled request left its unowned response body open")
		}
	})

	t.Run("response and transport error", func(t *testing.T) {
		transportErr := errors.New("transport failed after response")
		body := &trackingResponseBody{}
		err := ExecuteNewRequest(
			context.Background(),
			http.MethodGet,
			"/failed",
			nil,
			nil,
			WithDefaultBaseURL("https://example.test/"),
			WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusBadGateway,
					Header:     make(http.Header),
					Body:       body,
					Request:    req,
				}, transportErr
			})),
		)
		if !errors.Is(err, transportErr) {
			t.Fatalf("ExecuteNewRequest error = %v, want transport error", err)
		}
		if !body.closed {
			t.Fatal("transport error left its unowned response body open")
		}
	})
}

func TestExecuteTransfersEarlyErrorResponseToRawOwner(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	body := &trackingResponseBody{}
	var raw *http.Response
	err := ExecuteNewRequest(
		ctx,
		http.MethodGet,
		"/canceled",
		nil,
		nil,
		WithResponseInto(&raw),
		WithDefaultBaseURL("https://example.test/"),
		WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
			cancel()
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       body,
				Request:    req,
			}, nil
		})),
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("ExecuteNewRequest error = %v, want context.Canceled", err)
	}
	if raw == nil || raw.Body == nil {
		t.Fatal("raw response owner was not populated on the early error path")
	}
	if body.closed {
		t.Fatal("Execute closed a response body transferred to ResponseInto")
	}
	if err := raw.Body.Close(); err != nil {
		t.Fatalf("closing owned response body: %v", err)
	}
}

func responseDoer(body io.ReadCloser) HTTPDoer {
	return httpDoerFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusNoContent,
			Header:     make(http.Header),
			Body:       body,
			Request:    req,
		}, nil
	})
}
