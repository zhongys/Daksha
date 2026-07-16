package openaicompletions

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/zhongys/Daksha/ai"
)

// sseServer returns an httptest server that writes the given SSE data lines.
func sseServer(t *testing.T, events []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			fmt.Fprintf(w, "data: %s\n\n", ev)
		}
	}))
}

func collectStream(t *testing.T, stream *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage],
) ([]ai.AssistantMessageEvent, *ai.AssistantMessage, error) {
	t.Helper()
	var events []ai.AssistantMessageEvent
	for ev := range stream.Events() {
		events = append(events, ev)
	}
	msg, err := stream.Result(context.Background())
	return events, msg, err
}

func eventTypes(events []ai.AssistantMessageEvent) []ai.AssistantMessageEventType {
	types := make([]ai.AssistantMessageEventType, 0, len(events))
	for _, ev := range events {
		types = append(types, ev.EventType())
	}
	return types
}

func sseTextChunk(t *testing.T, content, finishReason string) string {
	t.Helper()
	choice := map[string]any{
		"index": 0,
		"delta": map[string]any{"content": content},
	}
	if finishReason != "" {
		choice["finish_reason"] = finishReason
	}
	chunk := map[string]any{"id": "r", "choices": []any{choice}}
	encoded, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal SSE chunk: %v", err)
	}
	return string(encoded)
}

func TestStreamerFullTurn(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"resp-1","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think "}}]}`,
		`{"id":"resp-1","choices":[{"index":0,"delta":{"reasoning_content":"hard"}}]}`,
		`{"id":"resp-1","choices":[{"index":0,"delta":{"content":"The answer"}}]}`,
		`{"id":"resp-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}`,
		`{"id":"resp-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\": \"/tmp/x\"}"}}]}}]}`,
		`{"id":"resp-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"id":"resp-1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":60},"completion_tokens_details":{"reasoning_tokens":5}}}`,
		`[DONE]`,
	})
	defer server.Close()

	s := NewStreamer()
	stream := s.Stream(context.Background(),
		ai.Provider{Name: "deepseek", BaseURL: server.URL + "/", APIKey: "k"},
		ai.Model{Provider: "deepseek", ID: "deepseek-reasoner", Reasoning: true},
		ai.Prompt{Messages: []ai.Message{userText("hi")}},
		ai.StreamOptions{ReasoningEffort: "high"},
	)

	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	wantTypes := []ai.AssistantMessageEventType{
		ai.AssistantEventStart,
		ai.AssistantEventThinkingStart, ai.AssistantEventThinkingDelta, ai.AssistantEventThinkingDelta,
		ai.AssistantEventTextStart, ai.AssistantEventTextDelta,
		ai.AssistantEventToolCallStart, ai.AssistantEventToolCallDelta, ai.AssistantEventToolCallDelta,
		ai.AssistantEventTextEnd, ai.AssistantEventThinkingEnd, ai.AssistantEventToolCallEnd,
		ai.AssistantEventDone,
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, wantTypes) {
		t.Errorf("event sequence = %v, want %v", got, wantTypes)
	}

	// The mid-stream tool call delta must carry a best-effort partial parse.
	var midParse map[string]any
	for _, ev := range events {
		if d, ok := ev.(ai.ToolCallDeltaEvent); ok {
			tc := d.Partial.Content[2].(*ai.ToolCallContent)
			midParse = tc.Arguments
			break // first delta: partial JSON `{"pa`
		}
	}
	if midParse == nil {
		t.Fatal("no toolcall delta event")
	}

	if msg.StopReason != ai.StopReasonToolUse {
		t.Errorf("StopReason = %v", msg.StopReason)
	}
	if msg.ResponseId != "resp-1" {
		t.Errorf("ResponseId = %q", msg.ResponseId)
	}
	if len(msg.Content) != 3 {
		t.Fatalf("content blocks = %d", len(msg.Content))
	}

	thinking := msg.Content[0].(*ai.ThinkingContent)
	if thinking.Thinking != "think hard" || thinking.ThinkingSignature != "reasoning_content" {
		t.Errorf("thinking = %+v", thinking)
	}
	text := msg.Content[1].(*ai.TextContent)
	if text.Text != "The answer" {
		t.Errorf("text = %+v", text)
	}
	tc := msg.Content[2].(*ai.ToolCallContent)
	wantArgs := map[string]any{"path": "/tmp/x"}
	if tc.Id != "call_1" || tc.Name != "read_file" || !reflect.DeepEqual(tc.Arguments, wantArgs) {
		t.Errorf("tool call = %+v", tc)
	}

	// Usage: cached tokens carved out of input.
	wantUsage := ai.Usage{Input: 40, Output: 20, CacheRead: 60, Reasoning: 5, TotalTokens: 120}
	if msg.Usage != wantUsage {
		t.Errorf("usage = %+v, want %+v", msg.Usage, wantUsage)
	}
}

func TestStreamerValidatesFragmentedFinalJSON(t *testing.T) {
	server := sseServer(t, []string{
		sseTextChunk(t, "  {\"answer\":", ""),
		sseTextChunk(t, "\"Daksha\"} \n", "stop"),
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}},
		ai.StreamOptions{OutputFormat: ai.OutputFormat{Type: ai.OutputFormatJSONObject}},
	)
	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	wantTypes := []ai.AssistantMessageEventType{
		ai.AssistantEventStart,
		ai.AssistantEventTextStart,
		ai.AssistantEventTextDelta,
		ai.AssistantEventTextDelta,
		ai.AssistantEventTextEnd,
		ai.AssistantEventDone,
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, wantTypes) {
		t.Fatalf("event sequence = %v, want %v", got, wantTypes)
	}
	if msg.StopReason != ai.StopReasonStop {
		t.Fatalf("StopReason = %s, error = %q", msg.StopReason, msg.ErrorMessage)
	}
	text := msg.Content[0].(*ai.TextContent).Text
	if text != "  {\"answer\":\"Daksha\"} \n" || !json.Valid([]byte(text)) {
		t.Fatalf("final text = %q", text)
	}
}

func TestStreamerRejectsInvalidFinalJSON(t *testing.T) {
	schemaFormat := ai.OutputFormat{
		Type: ai.OutputFormatJSONSchema,
		JSONSchema: &ai.JSONSchema{
			Name: "answer", Schema: map[string]any{"type": "object"}, Strict: true,
		},
	}
	tests := []struct {
		name         string
		content      string
		finishReason string
		format       ai.OutputFormat
		wantError    string
	}{
		{
			name: "truncated json object", content: `{"answer":`, finishReason: "length",
			format: ai.OutputFormat{Type: ai.OutputFormatJSONObject}, wantError: "invalid final JSON",
		},
		{
			name: "trailing content under json schema", content: `{"answer":"ok"} trailing`, finishReason: "stop",
			format: schemaFormat, wantError: "invalid final JSON",
		},
		{
			name: "empty json object", finishReason: "stop",
			format: ai.OutputFormat{Type: ai.OutputFormatJSONObject}, wantError: "produced no JSON content",
		},
		{
			name: "array is not json object", content: `[]`, finishReason: "stop",
			format: ai.OutputFormat{Type: ai.OutputFormatJSONObject}, wantError: "invalid final JSON object",
		},
		{
			name: "null is not json object", content: `null`, finishReason: "stop",
			format: ai.OutputFormat{Type: ai.OutputFormatJSONObject}, wantError: "invalid final JSON object",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := sseServer(t, []string{
				sseTextChunk(t, tt.content, tt.finishReason),
				`[DONE]`,
			})
			defer server.Close()

			stream := NewStreamer().Stream(context.Background(),
				ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
				ai.Prompt{Messages: []ai.Message{userText("hi")}},
				ai.StreamOptions{OutputFormat: tt.format},
			)
			events, msg, err := collectStream(t, stream)
			if err != nil {
				t.Fatalf("Result: %v", err)
			}
			if msg.StopReason != ai.StopReasonError || !strings.Contains(msg.ErrorMessage, tt.wantError) {
				t.Fatalf("result = stop %s, error %q", msg.StopReason, msg.ErrorMessage)
			}
			for _, event := range events {
				if event.EventType() == ai.AssistantEventDone || event.EventType() == ai.AssistantEventTextEnd {
					t.Fatalf("invalid JSON published terminal success event %s", event.EventType())
				}
			}
			if len(events) == 0 || events[len(events)-1].EventType() != ai.AssistantEventError {
				t.Fatalf("event sequence = %v", eventTypes(events))
			}
			if tt.content != "" {
				text, ok := msg.Content[0].(*ai.TextContent)
				if !ok || text.Text != tt.content {
					t.Fatalf("partial content = %#v, want %q", msg.Content, tt.content)
				}
			}
		})
	}
}

func TestStreamerPreservesStructuredOutputRefusal(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"refusal":"I cannot "}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"refusal":"help with that."},"finish_reason":"stop"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}},
		ai.StreamOptions{OutputFormat: ai.OutputFormat{Type: ai.OutputFormatJSONObject}},
	)
	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if msg.StopReason != ai.StopReasonError || !strings.Contains(msg.ErrorMessage, "structured output refused") {
		t.Fatalf("message = stop %s, error %q", msg.StopReason, msg.ErrorMessage)
	}
	if got := msg.Diagnostics.Details["refusal"]; got != "I cannot help with that." {
		t.Fatalf("diagnostic refusal = %#v", got)
	}
	if len(events) == 0 || events[len(events)-1].EventType() != ai.AssistantEventError {
		t.Fatalf("event sequence = %v", eventTypes(events))
	}
}

func TestSnapshotOutputFormatDetachesSchema(t *testing.T) {
	nested := map[string]any{"type": "string"}
	original := ai.OutputFormat{
		Type: ai.OutputFormatJSONSchema,
		JSONSchema: &ai.JSONSchema{
			Name: "answer",
			Schema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"answer": nested},
				"required":   []string{"answer"},
			},
		},
	}
	snapshot, err := snapshotOutputFormat(original)
	if err != nil {
		t.Fatalf("snapshotOutputFormat: %v", err)
	}

	nested["type"] = "number"
	original.JSONSchema.Schema["required"].([]string)[0] = "changed"

	properties := snapshot.JSONSchema.Schema["properties"].(map[string]any)
	answer := properties["answer"].(map[string]any)
	if answer["type"] != "string" {
		t.Fatalf("snapshot answer schema = %#v", answer)
	}
	required := snapshot.JSONSchema.Schema["required"].([]any)
	if required[0] != "answer" {
		t.Fatalf("snapshot required = %#v", required)
	}
}

func TestStreamerStructuredOutputSkipsJSONValidationForToolCallTurn(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}},
		ai.StreamOptions{OutputFormat: ai.OutputFormat{Type: ai.OutputFormatJSONObject}},
	)
	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if msg.StopReason != ai.StopReasonToolUse {
		t.Fatalf("StopReason = %s, error = %q", msg.StopReason, msg.ErrorMessage)
	}
	if events[len(events)-1].EventType() != ai.AssistantEventDone {
		t.Fatalf("event sequence = %v", eventTypes(events))
	}
}

func TestStreamerTextOutputDoesNotValidateJSON(t *testing.T) {
	server := sseServer(t, []string{
		sseTextChunk(t, "plain text", "stop"),
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}},
		ai.StreamOptions{OutputFormat: ai.OutputFormat{Type: ai.OutputFormatText}},
	)
	_, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if msg.StopReason != ai.StopReasonStop || msg.Content[0].(*ai.TextContent).Text != "plain text" {
		t.Fatalf("message = %#v", msg)
	}
}

func TestStreamerInvalidOutputFormatIsEncodedInStream(t *testing.T) {
	stream := NewStreamer().Stream(context.Background(), ai.Provider{}, ai.Model{ID: "m"}, ai.Prompt{},
		ai.StreamOptions{OutputFormat: ai.OutputFormat{Type: "yaml"}},
	)
	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if len(events) != 1 || events[0].EventType() != ai.AssistantEventError {
		t.Fatalf("event sequence = %v", eventTypes(events))
	}
	if msg.StopReason != ai.StopReasonError || !strings.Contains(msg.ErrorMessage, "unknown output format") {
		t.Fatalf("message = stop %s, error %q", msg.StopReason, msg.ErrorMessage)
	}
}

func TestStreamerPartialsAreEventTimeSnapshots(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"reasoning_content":" more"}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"content":"hel"}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"inspect","arguments":"{\"config\":{\"path\":\"/tmp"}}]}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"/x\"}}"}}]},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})
	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	var thinkingStart ai.ThinkingStartEvent
	var thinkingDeltas []ai.ThinkingDeltaEvent
	var textStart ai.TextStartEvent
	var textDeltas []ai.TextDeltaEvent
	var toolStart ai.ToolCallStartEvent
	var toolDeltas []ai.ToolCallDeltaEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ThinkingStartEvent:
			thinkingStart = event
		case ai.ThinkingDeltaEvent:
			thinkingDeltas = append(thinkingDeltas, event)
		case ai.TextStartEvent:
			textStart = event
		case ai.TextDeltaEvent:
			textDeltas = append(textDeltas, event)
		case ai.ToolCallStartEvent:
			toolStart = event
		case ai.ToolCallDeltaEvent:
			toolDeltas = append(toolDeltas, event)
		}
	}

	thinkingAt := func(partial ai.AssistantMessage, index int) string {
		t.Helper()
		content, ok := partial.Content[index].(*ai.ThinkingContent)
		if !ok {
			t.Fatalf("content[%d] = %T, want *ai.ThinkingContent", index, partial.Content[index])
		}
		return content.Thinking
	}
	textAt := func(partial ai.AssistantMessage, index int) string {
		t.Helper()
		content, ok := partial.Content[index].(*ai.TextContent)
		if !ok {
			t.Fatalf("content[%d] = %T, want *ai.TextContent", index, partial.Content[index])
		}
		return content.Text
	}
	toolConfigAt := func(partial ai.AssistantMessage, index int) map[string]any {
		t.Helper()
		content, ok := partial.Content[index].(*ai.ToolCallContent)
		if !ok {
			t.Fatalf("content[%d] = %T, want *ai.ToolCallContent", index, partial.Content[index])
		}
		config, ok := content.Arguments["config"].(map[string]any)
		if !ok {
			t.Fatalf("tool arguments = %#v, want nested config", content.Arguments)
		}
		return config
	}

	if len(thinkingDeltas) != 2 || len(textDeltas) != 2 || len(toolDeltas) != 2 {
		t.Fatalf("delta counts: thinking=%d text=%d tool=%d", len(thinkingDeltas), len(textDeltas), len(toolDeltas))
	}
	if got := thinkingAt(thinkingStart.Partial, thinkingStart.ContentIndex); got != "" {
		t.Errorf("thinking start snapshot = %q, want empty", got)
	}
	if got := thinkingAt(thinkingDeltas[0].Partial, thinkingDeltas[0].ContentIndex); got != "think" {
		t.Errorf("first thinking snapshot = %q, want %q", got, "think")
	}
	if got := thinkingAt(thinkingDeltas[1].Partial, thinkingDeltas[1].ContentIndex); got != "think more" {
		t.Errorf("second thinking snapshot = %q, want %q", got, "think more")
	}
	if got := textAt(textStart.Partial, textStart.ContentIndex); got != "" {
		t.Errorf("text start snapshot = %q, want empty", got)
	}
	if got := textAt(textDeltas[0].Partial, textDeltas[0].ContentIndex); got != "hel" {
		t.Errorf("first text snapshot = %q, want %q", got, "hel")
	}
	if got := textAt(textDeltas[1].Partial, textDeltas[1].ContentIndex); got != "hello" {
		t.Errorf("second text snapshot = %q, want %q", got, "hello")
	}
	if args := toolStart.Partial.Content[toolStart.ContentIndex].(*ai.ToolCallContent).Arguments; len(args) != 0 {
		t.Errorf("tool start arguments = %#v, want empty", args)
	}
	if got := toolConfigAt(toolDeltas[0].Partial, toolDeltas[0].ContentIndex)["path"]; got != "/tmp" {
		t.Errorf("first tool snapshot path = %#v, want %q", got, "/tmp")
	}
	if got := toolConfigAt(toolDeltas[1].Partial, toolDeltas[1].ContentIndex)["path"]; got != "/tmp/x" {
		t.Errorf("second tool snapshot path = %#v, want %q", got, "/tmp/x")
	}

	// Nested JSON values must not alias later event snapshots or the result.
	toolConfigAt(toolDeltas[0].Partial, toolDeltas[0].ContentIndex)["path"] = "changed"
	if got := toolConfigAt(toolDeltas[1].Partial, toolDeltas[1].ContentIndex)["path"]; got != "/tmp/x" {
		t.Errorf("mutating first snapshot changed second snapshot to %#v", got)
	}
	finalConfig := msg.Content[2].(*ai.ToolCallContent).Arguments["config"].(map[string]any)
	if got := finalConfig["path"]; got != "/tmp/x" {
		t.Errorf("mutating event snapshot changed final result to %#v", got)
	}
}

func TestStreamerToolContinuationWithIDOmittedIndexRoutesByID(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[` +
			`{"index":0,"id":"call_a","function":{"name":"tool_a","arguments":"{\"value\":\""}}]}}]}`,
		// call_b first appears with an ID but no index. The omitted index
		// decodes to zero, which is already occupied by call_a.
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[` +
			`{"id":"call_b","function":{"name":"tool_b","arguments":"{\"value\":\"B\"}"}}]}}]}`,
		// Verify the ID conflict did not overwrite index zero's mapping.
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"A\"}"}}]},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})
	_, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("content blocks = %d, want 2", len(msg.Content))
	}

	callA := msg.Content[0].(*ai.ToolCallContent)
	callB := msg.Content[1].(*ai.ToolCallContent)
	if callA.Id != "call_a" || callA.Name != "tool_a" || callA.Arguments["value"] != "A" {
		t.Errorf("call A = %#v", callA)
	}
	if callB.Id != "call_b" || callB.Name != "tool_b" || callB.Arguments["value"] != "B" {
		t.Errorf("call B = %#v", callB)
	}
}

func TestStreamerReasoningFieldFallback(t *testing.T) {
	// OpenRouter-style `reasoning` field; signature must record the source.
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"reasoning":"hmm"}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	s := NewStreamer()
	stream := s.Stream(context.Background(),
		ai.Provider{Name: "openrouter", BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})

	_, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	thinking := msg.Content[0].(*ai.ThinkingContent)
	if thinking.Thinking != "hmm" || thinking.ThinkingSignature != "reasoning" {
		t.Errorf("thinking = %+v", thinking)
	}
}

func TestStreamerUnknownFinishReasonIsError(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"weird_reason"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	s := NewStreamer()
	stream := s.Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})

	events, msg, _ := collectStream(t, stream)
	last := events[len(events)-1]
	if last.EventType() != ai.AssistantEventError {
		t.Errorf("last event = %v", last.EventType())
	}
	if msg.StopReason != ai.StopReasonError || msg.ErrorMessage == "" {
		t.Errorf("msg = %+v", msg)
	}
}

func TestStreamerMissingFinishReasonIsError(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"content":"x"}}]}`,
		`[DONE]`,
	})
	defer server.Close()

	s := NewStreamer()
	stream := s.Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{
			OutputFormat: ai.OutputFormat{Type: ai.OutputFormatJSONObject},
		})

	_, msg, _ := collectStream(t, stream)
	if msg.StopReason != ai.StopReasonError || !strings.Contains(msg.ErrorMessage, "without finish_reason") {
		t.Errorf("message = stop %v, error %q", msg.StopReason, msg.ErrorMessage)
	}
}

func TestStreamerHTTPErrorEncodedInStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"auth"}}`))
	}))
	defer server.Close()

	s := NewStreamer()
	stream := s.Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})

	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result should carry the message, got err %v", err)
	}
	if len(events) != 1 || events[0].EventType() != ai.AssistantEventError {
		t.Errorf("events = %v", eventTypes(events))
	}
	if msg.StopReason != ai.StopReasonError {
		t.Errorf("StopReason = %v", msg.StopReason)
	}
}

func TestStreamerErrorEventInStream(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"content":"par"}}]}`,
		`{"error":{"message":"overloaded","type":"server_error"}}`,
	})
	defer server.Close()

	s := NewStreamer()
	stream := s.Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"}, ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})

	_, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if msg.StopReason != ai.StopReasonError {
		t.Errorf("StopReason = %v", msg.StopReason)
	}
	// Partial content survives on the error message.
	if text, ok := msg.Content[0].(*ai.TextContent); !ok || text.Text != "par" {
		t.Errorf("content = %+v", msg.Content)
	}
}
