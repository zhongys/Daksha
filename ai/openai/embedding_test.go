package openai

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestEmbeddingNewParamsMarshal(t *testing.T) {
	tests := []struct {
		name    string
		params  EmbeddingNewParams
		want    []string
		notWant []string
	}{
		{
			name: "minimal params omit unset fields",
			params: EmbeddingNewParams{
				Input: []string{"hello"},
				Model: "text-embedding-v4",
			},
			want: []string{
				`"input":["hello"]`,
				`"model":"text-embedding-v4"`,
			},
			notWant: []string{"encoding_format", "dimensions"},
		},
		{
			name: "optional fields serialize when set",
			params: EmbeddingNewParams{
				Input:          []string{"a", "b"},
				Model:          "m",
				EncodingFormat: "float",
				Dimensions:     1024,
			},
			want: []string{
				`"input":["a","b"]`,
				`"encoding_format":"float"`,
				`"dimensions":1024`,
			},
		},
		{
			name: "extra fields merge and override",
			params: EmbeddingNewParams{
				Input:       []string{"a"},
				Model:       "m",
				ExtraFields: map[string]any{"model": "override", "text_type": "query"},
			},
			want:    []string{`"model":"override"`, `"text_type":"query"`},
			notWant: []string{`"model":"m"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.params)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			s := string(b)
			for _, w := range tt.want {
				if !strings.Contains(s, w) {
					t.Errorf("want substring %q in %s", w, s)
				}
			}
			for _, nw := range tt.notWant {
				if strings.Contains(s, nw) {
					t.Errorf("unwanted substring %q in %s", nw, s)
				}
			}
		})
	}
}

func TestEmbeddingServiceValidatesSuccessfulResponse(t *testing.T) {
	defaultRequest := EmbeddingNewParams{Input: []string{"a", "b"}, Model: "m", Dimensions: 2}
	tests := []struct {
		name    string
		request EmbeddingNewParams
		body    string
		wantErr bool
	}{
		{
			name:    "minimal compatible response",
			request: defaultRequest,
			body:    `{"data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}]}`,
		},
		{name: "null", request: defaultRequest, body: `null`, wantErr: true},
		{name: "empty object", request: defaultRequest, body: `{}`, wantErr: true},
		{
			name:    "error envelope with status 200",
			request: defaultRequest,
			body:    `{"error":{"message":"embedding unavailable","type":"server_error"}}`,
			wantErr: true,
		},
		{
			name:    "wrong response object",
			request: defaultRequest,
			body:    `{"object":"embedding","data":[{"index":0,"embedding":[0.1,0.2]},{"index":1,"embedding":[0.3,0.4]}]}`,
			wantErr: true,
		},
		{
			name:    "wrong item object",
			request: defaultRequest,
			body:    `{"data":[{"object":"list","index":0,"embedding":[0.1,0.2]},{"index":1,"embedding":[0.3,0.4]}]}`,
			wantErr: true,
		},
		{
			name:    "missing embedding",
			request: defaultRequest,
			body:    `{"data":[{"index":0,"embedding":[0.1,0.2]}]}`,
			wantErr: true,
		},
		{
			name:    "duplicate index",
			request: defaultRequest,
			body:    `{"data":[{"index":0,"embedding":[0.1,0.2]},{"index":0,"embedding":[0.3,0.4]}]}`,
			wantErr: true,
		},
		{
			name:    "empty vector",
			request: defaultRequest,
			body:    `{"data":[{"index":0,"embedding":[]},{"index":1,"embedding":[0.3,0.4]}]}`,
			wantErr: true,
		},
		{
			name:    "requested dimension mismatch",
			request: defaultRequest,
			body:    `{"data":[{"index":0,"embedding":[0.1]},{"index":1,"embedding":[0.3]}]}`,
			wantErr: true,
		},
		{
			name:    "inconsistent vector dimensions",
			request: EmbeddingNewParams{Input: []string{"a", "b"}, Model: "m"},
			body:    `{"data":[{"index":0,"embedding":[0.1]},{"index":1,"embedding":[0.3,0.4]}]}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			responseBody := &trackingResponseBody{reader: strings.NewReader(tt.body)}
			service := NewEmbeddingService(
				WithDefaultBaseURL("https://example.test/"),
				WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     http.Header{"Content-Type": []string{"application/json"}},
						Body:       responseBody,
						Request:    req,
					}, nil
				})),
			)
			res, err := service.New(t.Context(), tt.request)
			if (err != nil) != tt.wantErr {
				t.Fatalf("New error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr && res != nil {
				t.Fatalf("New returned response on error: %+v", res)
			}
			if !tt.wantErr && (res == nil || len(res.Data) != len(tt.request.Input)) {
				t.Fatalf("New response = %+v", res)
			}
			if !responseBody.closed {
				t.Fatal("response body was not closed")
			}
		})
	}
}
