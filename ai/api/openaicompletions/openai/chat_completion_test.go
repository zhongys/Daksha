package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestChatCompletionNewParamsMarshal(t *testing.T) {
	tests := []struct {
		name    string
		params  ChatCompletionNewParams
		want    []string // substrings that must appear
		notWant []string // substrings that must not appear
	}{
		{
			name: "minimal params omit unset fields",
			params: ChatCompletionNewParams{
				Model:    "deepseek-chat",
				Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
			},
			want: []string{
				`"model":"deepseek-chat"`,
				`"role":"user"`,
				`"content":"hi"`,
			},
			notWant: []string{
				"temperature", "stop", "tools", "tool_choice",
				"response_format", "stream_options", "logit_bias",
			},
		},
		{
			name: "optional scalars serialize when set",
			params: ChatCompletionNewParams{
				Model:       "qwen-max",
				Messages:    []ChatCompletionMessageParamUnion{UserMessage("hi")},
				Temperature: Float(0),
				MaxTokens:   Int(1024),
				Logprobs:    Bool(false),
			},
			want: []string{
				`"temperature":0`,
				`"max_tokens":1024`,
				`"logprobs":false`,
			},
		},
		{
			name: "extra fields merge into top level",
			params: ChatCompletionNewParams{
				Model:    "qwen-plus",
				Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
				ExtraFields: map[string]any{
					"enable_thinking": false,
					"thinking_budget": 100,
				},
			},
			want: []string{
				`"enable_thinking":false`,
				`"thinking_budget":100`,
				`"model":"qwen-plus"`,
			},
		},
		{
			name: "extra fields override standard fields",
			params: ChatCompletionNewParams{
				Model:       "m",
				Messages:    []ChatCompletionMessageParamUnion{UserMessage("hi")},
				ExtraFields: map[string]any{"model": "override"},
			},
			want:    []string{`"model":"override"`},
			notWant: []string{`"model":"m"`},
		},
		{
			name: "stop union scalar",
			params: ChatCompletionNewParams{
				Model:    "m",
				Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
				Stop:     ChatCompletionNewParamsStopUnion{OfString: String("\n")},
			},
			want: []string{`"stop":"\n"`},
		},
		{
			name: "stop union array",
			params: ChatCompletionNewParams{
				Model:    "m",
				Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
				Stop:     ChatCompletionNewParamsStopUnion{OfStringArray: []string{"a", "b"}},
			},
			want: []string{`"stop":["a","b"]`},
		},
		{
			name: "response format json schema",
			params: ChatCompletionNewParams{
				Model:    "m",
				Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
				ResponseFormat: ResponseFormatJSONSchema(ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "result",
					Schema: map[string]any{"type": "object"},
				}),
			},
			want: []string{
				`"response_format":{"type":"json_schema"`,
				`"name":"result"`,
			},
		},
		{
			name: "tool choice auto string",
			params: ChatCompletionNewParams{
				Model:      "m",
				Messages:   []ChatCompletionMessageParamUnion{UserMessage("hi")},
				ToolChoice: ChatCompletionToolChoiceOptionUnionParam{OfAuto: String("auto")},
			},
			want: []string{`"tool_choice":"auto"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.params)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			got := string(b)
			for _, w := range tt.want {
				if !strings.Contains(got, w) {
					t.Errorf("missing %q in %s", w, got)
				}
			}
			for _, nw := range tt.notWant {
				if strings.Contains(got, nw) {
					t.Errorf("unexpected %q in %s", nw, got)
				}
			}
		})
	}
}

func TestMessageConstructorsSetRoleAndType(t *testing.T) {
	tests := []struct {
		name string
		msg  ChatCompletionMessageParamUnion
		want []string
	}{
		{
			name: "system",
			msg:  SystemMessage("be brief"),
			want: []string{`"role":"system"`, `"content":"be brief"`},
		},
		{
			name: "developer",
			msg:  DeveloperMessage("be brief"),
			want: []string{`"role":"developer"`},
		},
		{
			name: "user with parts",
			msg: UserMessage([]ChatCompletionContentPartUnionParam{
				TextContentPart("look"),
				ImageContentPart(ChatCompletionContentPartImageImageURLParam{URL: "https://x/1.png"}),
			}),
			want: []string{
				`"role":"user"`,
				`"type":"text"`,
				`"type":"image_url"`,
				`"url":"https://x/1.png"`,
			},
		},
		{
			name: "assistant",
			msg:  AssistantMessage("ok"),
			want: []string{`"role":"assistant"`},
		},
		{
			name: "tool",
			msg:  ToolMessage("42", "call_1"),
			want: []string{`"role":"tool"`, `"tool_call_id":"call_1"`},
		},
		{
			name: "text part built by hand gets default type",
			msg: SystemMessage([]ChatCompletionContentPartTextParam{
				{Text: "raw"},
			}),
			want: []string{`"type":"text"`, `"text":"raw"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.msg)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			got := string(b)
			for _, w := range tt.want {
				if !strings.Contains(got, w) {
					t.Errorf("missing %q in %s", w, got)
				}
			}
		})
	}
}

func TestEmptyUnionsFailFast(t *testing.T) {
	tests := []struct {
		name string
		v    any
	}{
		{"message union", ChatCompletionMessageParamUnion{}},
		{"stop union", ChatCompletionNewParamsStopUnion{}},
		{"tool union", ChatCompletionToolUnionParam{}},
		{"tool choice union", ChatCompletionToolChoiceOptionUnionParam{}},
		{"content part union", ChatCompletionContentPartUnionParam{}},
		{"response format union", ChatCompletionNewParamsResponseFormatUnion{}},
		{"tool call union", ChatCompletionMessageToolCallUnionParam{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := json.Marshal(tt.v); err == nil {
				t.Errorf("expected error marshaling empty %T", tt.v)
			}
		})
	}
}

func TestEmptyUnionFieldsAreOmitted(t *testing.T) {
	// An unset union field in the params must be dropped by omitzero before
	// its MarshalJSON (which would error) is ever consulted.
	params := ChatCompletionNewParams{
		Model:    "m",
		Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
	}
	if _, err := json.Marshal(params); err != nil {
		t.Fatalf("marshal error: %v", err)
	}
}

func TestChatCompletionMessageUnmarshal(t *testing.T) {
	data := []byte(`{
		"role": "assistant",
		"content": "answer",
		"reasoning_content": "thinking...",
		"tool_calls": [
			{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
			{"id": "c2", "type": "custom", "custom": {"name": "g", "input": "raw"}}
		],
		"vendor_only_field": {"x": 1}
	}`)

	var msg ChatCompletionMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if msg.Content != "answer" {
		t.Errorf("Content = %q", msg.Content)
	}
	if msg.ReasoningContent != "thinking..." {
		t.Errorf("ReasoningContent = %q", msg.ReasoningContent)
	}
	if len(msg.ToolCalls) != 2 {
		t.Fatalf("ToolCalls len = %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].Type != "function" || msg.ToolCalls[0].Function.Name != "f" {
		t.Errorf("tool call 0 = %+v", msg.ToolCalls[0])
	}
	if msg.ToolCalls[1].Type != "custom" || msg.ToolCalls[1].Custom.Input != "raw" {
		t.Errorf("tool call 1 = %+v", msg.ToolCalls[1])
	}

	// Vendor extras are reachable through Raw.
	var ext struct {
		VendorOnly map[string]int `json:"vendor_only_field"`
	}
	if err := json.Unmarshal(msg.Raw, &ext); err != nil {
		t.Fatalf("unmarshal Raw: %v", err)
	}
	if ext.VendorOnly["x"] != 1 {
		t.Errorf("vendor_only_field = %+v", ext.VendorOnly)
	}
}

func TestToAssistantMessageParamRoundTrip(t *testing.T) {
	msg := ChatCompletionMessage{
		Role:             "assistant",
		Content:          "done",
		ReasoningContent: "secret thinking",
		ToolCalls: []ChatCompletionMessageToolCall{
			{ID: "c1", Type: "function", Function: ChatCompletionMessageToolCallFunction{Name: "f", Arguments: `{"a":1}`}},
		},
	}

	b, err := json.Marshal(msg.ToAssistantMessageParam())
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	got := string(b)

	for _, w := range []string{
		`"role":"assistant"`,
		`"content":"done"`,
		`"id":"c1"`,
		`"type":"function"`,
		`"name":"f"`,
	} {
		if !strings.Contains(got, w) {
			t.Errorf("missing %q in %s", w, got)
		}
	}
	// Reasoning must not be echoed back to the provider.
	if strings.Contains(got, "secret thinking") || strings.Contains(got, "reasoning") {
		t.Errorf("reasoning leaked into param: %s", got)
	}
}

func TestChunkDeltaUnmarshal(t *testing.T) {
	data := []byte(`{
		"id": "chunk-1",
		"object": "chat.completion.chunk",
		"choices": [{
			"index": 0,
			"delta": {"reasoning_content": "step 1", "vendor_field": true}
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 5,
			"total_tokens": 15,
			"prompt_cache_hit_tokens": 8
		}
	}`)

	var chunk ChatCompletionChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	delta := chunk.Choices[0].Delta
	if delta.ReasoningContent != "step 1" {
		t.Errorf("ReasoningContent = %q", delta.ReasoningContent)
	}

	var deltaExt struct {
		VendorField bool `json:"vendor_field"`
	}
	if err := json.Unmarshal(delta.Raw, &deltaExt); err != nil || !deltaExt.VendorField {
		t.Errorf("delta Raw = %s, err = %v", delta.Raw, err)
	}

	if chunk.Usage.TotalTokens != 15 {
		t.Errorf("TotalTokens = %d", chunk.Usage.TotalTokens)
	}
	var usageExt struct {
		CacheHit int64 `json:"prompt_cache_hit_tokens"`
	}
	if err := json.Unmarshal(chunk.Usage.Raw, &usageExt); err != nil || usageExt.CacheHit != 8 {
		t.Errorf("usage Raw = %s, err = %v", chunk.Usage.Raw, err)
	}
}

func TestChatCompletionUnmarshalKeepsRaw(t *testing.T) {
	data := []byte(`{"id":"x","model":"deepseek-reasoner","choices":[],"provider_meta":"p"}`)

	var res ChatCompletion
	if err := json.Unmarshal(data, &res); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if res.ID != "x" {
		t.Errorf("ID = %q", res.ID)
	}
	if !strings.Contains(string(res.Raw), `"provider_meta":"p"`) {
		t.Errorf("Raw = %s", res.Raw)
	}
}

func TestChatCompletionServiceValidatesSuccessfulResponse(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{
			name: "minimal compatible response",
			body: `{"id":"r","choices":[{"index":0,"finish_reason":"stop","message":{"content":"ok"}}]}`,
		},
		{name: "null", body: `null`, wantErr: true},
		{name: "empty object", body: `{}`, wantErr: true},
		{name: "empty choices", body: `{"choices":[]}`, wantErr: true},
		{name: "missing finish reason", body: `{"choices":[{"index":0,"message":{"content":"ok"}}]}`, wantErr: true},
		{
			name:    "duplicate choice index",
			body:    `{"choices":[{"index":0,"finish_reason":"stop"},{"index":0,"finish_reason":"stop"}]}`,
			wantErr: true,
		},
		{
			name:    "contradictory object",
			body:    `{"object":"chat.completion.chunk","choices":[{"index":0,"finish_reason":"stop"}]}`,
			wantErr: true,
		},
		{
			name:    "error envelope with status 200",
			body:    `{"error":{"message":"upstream failed","type":"server_error"}}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			responseBody := &trackingResponseBody{reader: strings.NewReader(tt.body)}
			service := NewChatCompletionService(
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
			res, err := service.New(t.Context(), ChatCompletionNewParams{
				Model: "m", Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
			})
			if (err != nil) != tt.wantErr {
				t.Fatalf("New error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr && res != nil {
				t.Fatalf("New returned response on error: %+v", res)
			}
			if !tt.wantErr && (res == nil || len(res.Choices) != 1) {
				t.Fatalf("New response = %+v", res)
			}
			if !responseBody.closed {
				t.Fatal("response body was not closed")
			}
		})
	}
}

func TestNewStreamingJSONRequiresEventStreamAndSetsWireFields(t *testing.T) {
	var requestBody []byte
	responseBody := &trackingResponseBody{reader: strings.NewReader(
		"data: {\"id\":\"r\",\"choices\":[]}\n\n",
	)}
	service := NewChatCompletionService(
		WithDefaultBaseURL("https://example.test/"),
		WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
			var err error
			requestBody, err = io.ReadAll(req.Body)
			if err != nil {
				return nil, err
			}
			if got := req.Header.Get("Accept"); got != "text/event-stream" {
				return nil, errors.New("streaming request did not ask for text/event-stream")
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header: http.Header{
					"Content-Type": []string{"text/event-stream; charset=utf-8"},
				},
				Body:    responseBody,
				Request: req,
			}, nil
		})),
	)

	raw := json.RawMessage(`{"model":"m","messages":[],"snapshot":"before"}`)
	stream := service.NewStreamingJSON(t.Context(), raw)
	if err := stream.Err(); err != nil {
		t.Fatalf("NewStreamingJSON: %v", err)
	}
	if !stream.Next() {
		t.Fatalf("Next = false, error = %v", stream.Err())
	}
	if got := stream.Current().ID; got != "r" {
		t.Fatalf("chunk ID = %q, want r", got)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !responseBody.closed {
		t.Fatal("stream response body was not closed")
	}

	var encoded map[string]any
	if err := json.Unmarshal(requestBody, &encoded); err != nil {
		t.Fatalf("request body = %q: %v", requestBody, err)
	}
	if encoded["stream"] != true || encoded["snapshot"] != "before" {
		t.Fatalf("request body = %s", requestBody)
	}
	if !bytes.Contains(requestBody, []byte(`"model":"m"`)) {
		t.Fatalf("request body = %s", requestBody)
	}
}

func TestNewStreamingRejectsUnexpectedContentTypeAndClosesBody(t *testing.T) {
	responseBody := &trackingResponseBody{reader: strings.NewReader(
		`{"error":{"message":"stream unavailable","type":"server_error"}}`,
	)}
	service := NewChatCompletionService(
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

	stream := service.NewStreaming(t.Context(), ChatCompletionNewParams{
		Model: "m", Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
	})
	var apiErr *APIError
	if !errors.As(stream.Err(), &apiErr) {
		t.Fatalf("stream error = %T %v, want *APIError", stream.Err(), stream.Err())
	}
	if apiErr.StatusCode != http.StatusOK || apiErr.Message != "stream unavailable" {
		t.Fatalf("APIError = %+v", apiErr)
	}
	if !responseBody.closed {
		t.Fatal("unexpected-content response body was not closed")
	}
	if stream.Next() {
		t.Fatal("stream produced a chunk after content-type error")
	}
}

func TestNewStreamingStrictContentTypeMatrix(t *testing.T) {
	for _, contentType := range []string{
		"",
		"text/plain",
		"application/x-ndjson",
		"text/event-stream+json",
		"text/event-stream; charset",
	} {
		name := contentType
		if name == "" {
			name = "missing"
		}
		t.Run(name, func(t *testing.T) {
			responseBody := &trackingResponseBody{reader: strings.NewReader("not an SSE response")}
			service := NewChatCompletionService(
				WithDefaultBaseURL("https://example.test/"),
				WithHTTPClient(httpDoerFunc(func(req *http.Request) (*http.Response, error) {
					header := make(http.Header)
					if contentType != "" {
						header.Set("Content-Type", contentType)
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     header,
						Body:       responseBody,
						Request:    req,
					}, nil
				})),
			)
			stream := service.NewStreaming(t.Context(), ChatCompletionNewParams{
				Model: "m", Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
			})
			if stream.Err() == nil {
				t.Fatal("NewStreaming accepted a non-event-stream media type")
			}
			if !responseBody.closed {
				t.Fatal("rejected response body was not closed")
			}
		})
	}
}

func TestNewStreamingRejectsMissingResponsePartsWithoutPanic(t *testing.T) {
	tests := []struct {
		name string
		do   httpDoerFunc
	}{
		{
			name: "nil response",
			do:   func(*http.Request) (*http.Response, error) { return nil, nil },
		},
		{
			name: "nil body",
			do: func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Request:    req,
				}, nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewChatCompletionService(
				WithDefaultBaseURL("https://example.test/"), WithHTTPClient(tt.do),
			)
			stream := service.NewStreaming(t.Context(), ChatCompletionNewParams{
				Model: "m", Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
			})
			if stream.Err() == nil {
				t.Fatal("NewStreaming succeeded")
			}
			if stream.Next() {
				t.Fatal("stream produced a chunk")
			}
		})
	}
}
