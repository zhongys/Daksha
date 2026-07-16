package ai

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestCloneAssistantMessageDeepCopy(t *testing.T) {
	original := &AssistantMessage{
		Role: RoleAssistant,
		Content: []AssistantContent{
			&TextContent{Type: ContentTypeText, Text: "answer"},
			&JSONContent{Type: ContentTypeJSON, SchemaName: "result", Value: json.RawMessage(`{"ok":true}`)},
			&ThinkingContent{Type: ContentTypeThinking, Thinking: "reason", ThinkingSignature: "reasoning_content"},
			&ToolCallContent{
				Type: ContentTypeToolCall,
				Id:   "call_1",
				Name: "inspect",
				Arguments: map[string]any{
					"path": "/safe",
					"nested": map[string]any{
						"items": []any{
							map[string]any{
								"value": "original",
								"raw":   json.RawMessage(`{"n":1}`),
							},
						},
					},
				},
			},
		},
		Diagnostics: AssistantMessageDiagnostic{
			Type: "provider",
			Details: map[string]any{
				"nested": map[string]any{
					"items": []any{map[string]any{"status": "original"}},
				},
				"raw":   json.RawMessage(`{"diagnostic":true}`),
				"bytes": []byte("diagnostic bytes"),
			},
		},
		StopReason: StopReasonToolUse,
	}

	cloned := CloneAssistantMessage(original)
	if cloned == original {
		t.Fatal("clone returned the original message pointer")
	}
	if !reflect.DeepEqual(cloned, original) {
		t.Fatalf("clone differs before mutation:\nclone:    %#v\noriginal: %#v", cloned, original)
	}

	cloned.Content[0].(*TextContent).Text = "changed clone"
	cloned.Content[1].(*JSONContent).Value[2] = 'X'
	clonedTool := cloned.Content[3].(*ToolCallContent)
	clonedTool.Arguments["path"] = "/changed-clone"
	clonedItem := clonedTool.Arguments["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	clonedItem["value"] = "changed clone"
	clonedItem["raw"].(json.RawMessage)[2] = 'X'
	clonedDiagnostic := cloned.Diagnostics.Details["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	clonedDiagnostic["status"] = "changed clone"
	cloned.Diagnostics.Details["raw"].(json.RawMessage)[2] = 'X'
	cloned.Diagnostics.Details["bytes"].([]byte)[0] = 'X'

	if got := original.Content[0].(*TextContent).Text; got != "answer" {
		t.Fatalf("mutating cloned text changed original to %q", got)
	}
	if got := string(original.Content[1].(*JSONContent).Value); got != `{"ok":true}` {
		t.Fatalf("mutating cloned JSON changed original to %q", got)
	}
	originalTool := original.Content[3].(*ToolCallContent)
	if got := originalTool.Arguments["path"]; got != "/safe" {
		t.Fatalf("mutating cloned arguments changed original path to %#v", got)
	}
	originalItem := originalTool.Arguments["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	if got := originalItem["value"]; got != "original" {
		t.Fatalf("mutating cloned arguments changed original nested value to %#v", got)
	}
	if got := string(originalItem["raw"].(json.RawMessage)); got != `{"n":1}` {
		t.Fatalf("mutating cloned argument raw JSON changed original to %q", got)
	}
	originalDiagnostic := original.Diagnostics.Details["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	if got := originalDiagnostic["status"]; got != "original" {
		t.Fatalf("mutating cloned diagnostics changed original status to %#v", got)
	}
	if got := string(original.Diagnostics.Details["raw"].(json.RawMessage)); got != `{"diagnostic":true}` {
		t.Fatalf("mutating cloned diagnostic raw JSON changed original to %q", got)
	}
	if got := string(original.Diagnostics.Details["bytes"].([]byte)); got != "diagnostic bytes" {
		t.Fatalf("mutating cloned diagnostic bytes changed original to %q", got)
	}

	// Isolation is bidirectional: later producer mutations must not rewrite a
	// snapshot already handed to another component.
	original.Content[0].(*TextContent).Text = "changed original"
	originalTool.Arguments["path"] = "/changed-original"
	originalItem["value"] = "changed original"
	originalDiagnostic["status"] = "changed original"
	if got := cloned.Content[0].(*TextContent).Text; got != "changed clone" {
		t.Fatalf("mutating original text changed clone to %q", got)
	}
	if got := clonedTool.Arguments["path"]; got != "/changed-clone" {
		t.Fatalf("mutating original arguments changed cloned path to %#v", got)
	}
	if got := clonedItem["value"]; got != "changed clone" {
		t.Fatalf("mutating original arguments changed cloned nested value to %#v", got)
	}
	if got := clonedDiagnostic["status"]; got != "changed clone" {
		t.Fatalf("mutating original diagnostics changed cloned status to %#v", got)
	}
}

func TestCloneAssistantMessageNilAndTypedNilContent(t *testing.T) {
	if cloned := CloneAssistantMessage(nil); cloned != nil {
		t.Fatalf("CloneAssistantMessage(nil) = %#v, want nil", cloned)
	}

	var nilToolCall *ToolCallContent
	original := &AssistantMessage{
		Role:    RoleAssistant,
		Content: []AssistantContent{nilToolCall},
	}
	cloned := CloneAssistantMessage(original)
	if cloned == original {
		t.Fatal("clone returned the original message pointer")
	}
	toolCall, ok := cloned.Content[0].(*ToolCallContent)
	if !ok || toolCall != nil {
		t.Fatalf("typed-nil tool call = %#v (%T), want typed nil", cloned.Content[0], cloned.Content[0])
	}
}
