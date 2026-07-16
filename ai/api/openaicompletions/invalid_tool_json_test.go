package openaicompletions

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/zhongys/Daksha/ai"
)

func TestStreamerRejectsInvalidFinalToolArguments(t *testing.T) {
	tests := []struct {
		name      string
		arguments string
	}{
		{name: "truncated", arguments: `{"path":`},
		{name: "trailing value", arguments: `{} {}`},
		{name: "null", arguments: `null`},
		{name: "non-object", arguments: `[]`},
		{name: "repairable invalid escape", arguments: `{"path":"\q"}`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := sseServer(t, []string{
				fmt.Sprintf(
					`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":%q}}]}}]}`,
					test.arguments,
				),
				`{"id":"r","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
				`[DONE]`,
			})
			defer server.Close()

			stream := NewStreamer().Stream(context.Background(),
				ai.Provider{BaseURL: server.URL + "/", APIKey: "k"},
				ai.Model{ID: "m", ToolCall: true},
				ai.Prompt{Messages: []ai.Message{userText("go")}},
				ai.StreamOptions{},
			)
			events, msg, err := collectStream(t, stream)
			if err != nil {
				t.Fatalf("Result: %v", err)
			}
			if msg.StopReason != ai.StopReasonError {
				t.Fatalf("StopReason = %q, want error", msg.StopReason)
			}
			if !strings.Contains(msg.ErrorMessage, "invalid final JSON arguments") {
				t.Fatalf("ErrorMessage = %q", msg.ErrorMessage)
			}

			var sawDelta, sawError bool
			for _, event := range events {
				switch event.EventType() {
				case ai.AssistantEventToolCallDelta:
					sawDelta = true // tolerant partials remain available to the UI
				case ai.AssistantEventError:
					sawError = true
				case ai.AssistantEventToolCallEnd, ai.AssistantEventDone:
					t.Fatalf("invalid arguments published executable terminal event %q", event.EventType())
				}
			}
			if !sawDelta || !sawError {
				t.Fatalf("events = %v, want partial delta followed by error", eventTypes(events))
			}
		})
	}
}
