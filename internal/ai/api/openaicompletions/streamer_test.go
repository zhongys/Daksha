package openaicompletions

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/zhongys/Daksha/internal/ai"
	"github.com/zhongys/Daksha/internal/ai/openai"
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

	s := NewStreamer(ai.Provider{Name: "deepseek", BaseURL: server.URL + "/"}, openai.WithAPIKey("k"))
	stream := s.Stream(context.Background(),
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

func TestStreamerReasoningFieldFallback(t *testing.T) {
	// OpenRouter-style `reasoning` field; signature must record the source.
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"reasoning":"hmm"}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	s := NewStreamer(ai.Provider{Name: "openrouter", BaseURL: server.URL + "/"}, openai.WithAPIKey("k"))
	stream := s.Stream(context.Background(), ai.Model{ID: "m"},
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

	s := NewStreamer(ai.Provider{BaseURL: server.URL + "/"}, openai.WithAPIKey("k"))
	stream := s.Stream(context.Background(), ai.Model{ID: "m"},
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

	s := NewStreamer(ai.Provider{BaseURL: server.URL + "/"}, openai.WithAPIKey("k"))
	stream := s.Stream(context.Background(), ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}}, ai.StreamOptions{})

	_, msg, _ := collectStream(t, stream)
	if msg.StopReason != ai.StopReasonError {
		t.Errorf("StopReason = %v, want error", msg.StopReason)
	}
}

func TestStreamerHTTPErrorEncodedInStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"auth"}}`))
	}))
	defer server.Close()

	s := NewStreamer(ai.Provider{BaseURL: server.URL + "/"}, openai.WithAPIKey("k"))
	stream := s.Stream(context.Background(), ai.Model{ID: "m"},
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

	s := NewStreamer(ai.Provider{BaseURL: server.URL + "/"}, openai.WithAPIKey("k"))
	stream := s.Stream(context.Background(), ai.Model{ID: "m"},
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
