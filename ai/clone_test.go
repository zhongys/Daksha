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

func TestCloneMessagesDetachesUserAndToolResultContent(t *testing.T) {
	user := &UserMessage{
		Role: RoleUser,
		Content: []UserContent{
			&TextContent{Type: ContentTypeText, Text: "original user"},
			&ImageContent{Type: ContentTypeImage, URL: "https://example.test/original.png"},
		},
	}
	toolResult := &ToolResultMessage[struct{ Trace string }]{
		Role:       RoleToolResult,
		ToolCallId: "call_1",
		Content: []ToolResultContent{
			&TextContent{Type: ContentTypeText, Text: "original result"},
			&ImageContent{Type: ContentTypeImage, URL: "https://example.test/result.png"},
		},
	}

	cloned := CloneMessages([]Message{user, toolResult})
	clonedUser := cloned[0].(*UserMessage)
	clonedResult, ok := cloned[1].(*ToolResultMessage[struct{ Trace string }])
	if !ok {
		t.Fatalf("tool result clone type = %T", cloned[1])
	}
	clonedUser.Content[0].(*TextContent).Text = "changed clone"
	clonedUser.Content[1].(*ImageContent).URL = "https://example.test/changed.png"
	clonedResult.Content[0].(*TextContent).Text = "changed result clone"
	clonedResult.Content[1].(*ImageContent).URL = "https://example.test/changed-result.png"

	if got := user.Content[0].(*TextContent).Text; got != "original user" {
		t.Fatalf("user content changed through clone: %q", got)
	}
	if got := user.Content[1].(*ImageContent).URL; got != "https://example.test/original.png" {
		t.Fatalf("user image changed through clone: %q", got)
	}
	if got := toolResult.Content[0].(*TextContent).Text; got != "original result" {
		t.Fatalf("tool result content changed through clone: %q", got)
	}
	if got := toolResult.Content[1].(*ImageContent).URL; got != "https://example.test/result.png" {
		t.Fatalf("tool result image changed through clone: %q", got)
	}
}

func TestCloneMessageDetachesToolResultDetails(t *testing.T) {
	details := map[string]any{
		"nested": map[string]any{"items": []string{"original"}},
	}
	message := &ToolResultMessage[map[string]any]{
		Role: RoleToolResult, ToolCallId: "call_1", Details: &details,
	}

	cloned := CloneMessage(message).(*ToolResultMessage[map[string]any])
	if cloned.Details == message.Details {
		t.Fatal("clone retained the original Details pointer")
	}
	(*cloned.Details)["nested"].(map[string]any)["items"].([]string)[0] = "changed clone"
	if got := details["nested"].(map[string]any)["items"].([]string)[0]; got != "original" {
		t.Fatalf("mutating cloned Details changed original to %q", got)
	}
	details["nested"].(map[string]any)["items"].([]string)[0] = "changed original"
	if got := (*cloned.Details)["nested"].(map[string]any)["items"].([]string)[0]; got != "changed clone" {
		t.Fatalf("mutating original Details changed clone to %q", got)
	}

	prompt, err := SnapshotPrompt(Prompt{Messages: []Message{message}})
	if err != nil {
		t.Fatalf("SnapshotPrompt: %v", err)
	}
	if got := prompt.Messages[0].(*ToolResultMessage[map[string]any]).Details; got != nil {
		t.Fatalf("prompt retained application-only Details: %#v", got)
	}
}

func TestSnapshotPromptDetachesMessagesAndToolSchemas(t *testing.T) {
	userText := &TextContent{Type: ContentTypeText, Text: "original prompt"}
	argumentItems := []string{"original argument"}
	arguments := map[string]any{
		"items": argumentItems,
		"large": int64(9_007_199_254_740_993),
	}
	required := []string{"answer"}
	property := map[string]any{"type": "string"}
	parameters := map[string]any{
		"type":       "object",
		"properties": map[string]any{"answer": property},
		"required":   required,
	}
	prompt := Prompt{
		Messages: []Message{
			&UserMessage{Role: RoleUser, Content: []UserContent{userText}},
			&AssistantMessage{Role: RoleAssistant, Content: []AssistantContent{
				&ToolCallContent{Type: ContentTypeToolCall, Id: "call_1", Name: "answer", Arguments: arguments},
			}},
		},
		Tools: []ToolDefinition{{Name: "answer", Parameters: parameters}},
	}

	snapshot, err := SnapshotPrompt(prompt)
	if err != nil {
		t.Fatalf("SnapshotPrompt: %v", err)
	}
	userText.Text = "changed original"
	argumentItems[0] = "changed original"
	property["type"] = "number"
	required[0] = "changed"

	if got := snapshot.Messages[0].(*UserMessage).Content[0].(*TextContent).Text; got != "original prompt" {
		t.Fatalf("snapshot user text = %q", got)
	}
	snapshotArgs := snapshot.Messages[1].(*AssistantMessage).Content[0].(*ToolCallContent).Arguments
	if got := snapshotArgs["items"].([]any)[0]; got != "original argument" {
		t.Fatalf("snapshot tool arguments = %#v", snapshotArgs)
	}
	if got, ok := snapshotArgs["large"].(json.Number); !ok || got.String() != "9007199254740993" {
		t.Fatalf("snapshot large argument = %T(%v)", snapshotArgs["large"], snapshotArgs["large"])
	}
	snapshotParameters := snapshot.Tools[0].Parameters
	if got := snapshotParameters["properties"].(map[string]any)["answer"].(map[string]any)["type"]; got != "string" {
		t.Fatalf("snapshot property type = %#v", got)
	}
	if got := snapshotParameters["required"].([]any)[0]; got != "answer" {
		t.Fatalf("snapshot required = %#v", snapshotParameters["required"])
	}
}

func TestSnapshotStreamOptionsDetachesEveryReferenceField(t *testing.T) {
	temperature := 0.25
	topP := 0.75
	maxTokens := int64(512)
	stops := []string{"original stop"}
	schemaProperty := map[string]any{"type": "string"}
	extraNested := map[string]any{"value": "original extra"}
	options := StreamOptions{
		Temperature:   &temperature,
		TopP:          &topP,
		MaxTokens:     &maxTokens,
		StopSequences: stops,
		OutputFormat: OutputFormat{
			Type: OutputFormatJSONSchema,
			JSONSchema: &JSONSchema{Name: "answer", Schema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"answer": schemaProperty},
			}},
		},
		Extra: map[string]any{"nested": extraNested},
	}

	snapshot, err := SnapshotStreamOptions(options)
	if err != nil {
		t.Fatalf("SnapshotStreamOptions: %v", err)
	}
	temperature = 1
	topP = 1
	maxTokens = 1
	stops[0] = "changed"
	schemaProperty["type"] = "number"
	extraNested["value"] = "changed"

	if *snapshot.Temperature != 0.25 || *snapshot.TopP != 0.75 || *snapshot.MaxTokens != 512 {
		t.Fatalf("snapshot scalar pointers = temperature %v, topP %v, maxTokens %v",
			*snapshot.Temperature, *snapshot.TopP, *snapshot.MaxTokens)
	}
	if snapshot.StopSequences[0] != "original stop" {
		t.Fatalf("snapshot stops = %#v", snapshot.StopSequences)
	}
	properties := snapshot.OutputFormat.JSONSchema.Schema["properties"].(map[string]any)
	if got := properties["answer"].(map[string]any)["type"]; got != "string" {
		t.Fatalf("snapshot schema type = %#v", got)
	}
	if got := snapshot.Extra["nested"].(map[string]any)["value"]; got != "original extra" {
		t.Fatalf("snapshot extra = %#v", snapshot.Extra)
	}
}

func TestCloneToolDefinitionPreservesConcreteSchemaTypesAndDetachesThem(t *testing.T) {
	required := []string{"value"}
	definition := ToolDefinition{Parameters: map[string]any{"required": required}}
	cloned := CloneToolDefinition(definition)
	required[0] = "changed original"

	got, ok := cloned.Parameters["required"].([]string)
	if !ok || len(got) != 1 || got[0] != "value" {
		t.Fatalf("cloned required = %T(%#v)", cloned.Parameters["required"], cloned.Parameters["required"])
	}
}

func TestSnapshotPromptRejectsCyclicToolArguments(t *testing.T) {
	arguments := map[string]any{}
	arguments["self"] = arguments
	_, err := SnapshotPrompt(Prompt{Messages: []Message{&AssistantMessage{
		Role: RoleAssistant,
		Content: []AssistantContent{&ToolCallContent{
			Type: ContentTypeToolCall, Id: "call_1", Name: "cycle", Arguments: arguments,
		}},
	}}})
	if err == nil {
		t.Fatal("SnapshotPrompt accepted cyclic tool arguments")
	}
}
