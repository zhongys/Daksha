package ai

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestUnmarshalMessagesRoundTrip(t *testing.T) {
	details := any(map[string]any{"path": "/tmp/x", "size": float64(42)})
	original := []Message{
		&UserMessage{
			Role: RoleUser,
			Content: []UserContent{
				&TextContent{Type: ContentTypeText, Text: "look at this"},
				&ImageContent{Type: ContentTypeImage, URL: "https://x/1.png"},
				&AudioContent{Type: ContentTypeAudio, Data: "QUFB", MimeType: "audio/wav"},
				&VideoContent{Type: ContentTypeVideo, URL: "https://x/v.mp4"},
			},
			Timestamp: 1,
		},
		&AssistantMessage{
			Role: RoleAssistant,
			Content: []AssistantContent{
				&ThinkingContent{Type: ContentTypeThinking, Thinking: "hmm", ThinkingSignature: "reasoning_content"},
				&TextContent{Type: ContentTypeText, Text: "calling a tool"},
				&JSONContent{Type: ContentTypeJSON, SchemaName: "answer", Value: json.RawMessage(`{"id":9007199254740993}`)},
				&ToolCallContent{Type: ContentTypeToolCall, Id: "c1", Name: "read_file",
					Arguments: map[string]any{"path": "/tmp/x"}},
			},
			Api: "openai-completions", Provider: "deepseek", Model: "deepseek-chat",
			Usage:      Usage{Input: 10, Output: 5, TotalTokens: 15, Cost: Cost{Input: 40000, Total: 40000}},
			StopReason: StopReasonToolUse,
			Timestamp:  2,
		},
		&ToolResultMessage[any]{
			Role: RoleToolResult, ToolCallId: "c1", ToolName: "read_file",
			Content:   []ToolResultContent{&TextContent{Type: ContentTypeText, Text: "file body"}},
			Details:   &details,
			Timestamp: 3,
		},
	}

	encoded, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	decoded, err := UnmarshalMessages(encoded)
	if err != nil {
		t.Fatalf("UnmarshalMessages: %v", err)
	}
	if len(decoded) != len(original) {
		t.Fatalf("decoded %d messages, want %d", len(decoded), len(original))
	}

	// The stable equality check: decode → re-encode must reproduce the
	// original JSON byte-for-byte (semantically, via normalized maps).
	reencoded, err := json.Marshal(decoded)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	var a, b []map[string]any
	if err := json.Unmarshal(encoded, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(reencoded, &b); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("round trip drifted:\n first = %s\nsecond = %s", encoded, reencoded)
	}

	// Spot-check concrete types survived discrimination.
	user := decoded[0].(*UserMessage)
	if _, ok := user.Content[2].(*AudioContent); !ok {
		t.Errorf("user content[2] = %T, want *AudioContent", user.Content[2])
	}
	assistant := decoded[1].(*AssistantMessage)
	structured, ok := assistant.Content[2].(*JSONContent)
	if !ok || structured.SchemaName != "answer" || string(structured.Value) != `{"id":9007199254740993}` {
		t.Errorf("assistant content[2] = %#v", assistant.Content[2])
	}
	if tc, ok := assistant.Content[3].(*ToolCallContent); !ok || tc.Name != "read_file" {
		t.Errorf("assistant content[3] = %+v", assistant.Content[3])
	}
	if assistant.Usage.Cost.Input != 40000 {
		t.Errorf("cost lost: %+v", assistant.Usage)
	}
	tr := decoded[2].(*ToolResultMessage[json.RawMessage])
	if tr.ToolCallId != "c1" || tr.Details == nil {
		t.Errorf("tool result = %+v", tr)
	}
}

func TestUnmarshalMessageErrors(t *testing.T) {
	if _, err := UnmarshalMessage([]byte(`{"role":"martian"}`)); err == nil {
		t.Error("unknown role should fail")
	}
	if _, err := UnmarshalMessage([]byte(`{"role":"user","content":[{"type":"hologram"}]}`)); err == nil {
		t.Error("unknown content type should fail")
	}
	if _, err := UnmarshalMessages([]byte(`{`)); err == nil {
		t.Error("bad JSON should fail")
	}
}

func TestUnmarshalAssistantToolArgumentsPreservesLargeIntegers(t *testing.T) {
	original := &AssistantMessage{
		Role: RoleAssistant,
		Content: []AssistantContent{&ToolCallContent{
			Type: ContentTypeToolCall,
			Id:   "call_1",
			Name: "lookup",
			Arguments: map[string]any{
				"id": int64(9_007_199_254_740_993),
				"nested": map[string]any{
					"values": []any{uint64(9_007_199_254_740_995)},
				},
			},
		}},
	}

	wire, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal assistant message: %v", err)
	}
	decodedMessage, err := UnmarshalMessage(wire)
	if err != nil {
		t.Fatalf("UnmarshalMessage: %v", err)
	}
	call := decodedMessage.(*AssistantMessage).Content[0].(*ToolCallContent)
	if got, ok := call.Arguments["id"].(json.Number); !ok || got.String() != "9007199254740993" {
		t.Fatalf("id = %T(%v), want exact json.Number", call.Arguments["id"], call.Arguments["id"])
	}
	nested := call.Arguments["nested"].(map[string]any)
	value := nested["values"].([]any)[0]
	if got, ok := value.(json.Number); !ok || got.String() != "9007199254740995" {
		t.Fatalf("nested value = %T(%v), want exact json.Number", value, value)
	}

	reencoded, err := json.Marshal(decodedMessage)
	if err != nil {
		t.Fatalf("re-marshal assistant message: %v", err)
	}
	if !reflect.DeepEqual(wire, reencoded) {
		t.Fatalf("assistant message round trip drifted:\n first = %s\nsecond = %s", wire, reencoded)
	}
}
