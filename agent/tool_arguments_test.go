package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/zhongys/Daksha/ai"
)

type rawOwnershipTool struct {
	definition ToolDefinition
	execute    func(context.Context, string, map[string]any, func(ToolUpdate)) (*ToolOutput, error)
}

func (t *rawOwnershipTool) Definition() ToolDefinition { return t.definition }

func (t *rawOwnershipTool) Execute(ctx context.Context, id string, args map[string]any,
	onUpdate func(ToolUpdate)) (*ToolOutput, error) {
	return t.execute(ctx, id, args, onUpdate)
}

func toolArguments(message ai.Message) (map[string]any, bool) {
	assistant, ok := message.(*ai.AssistantMessage)
	if !ok {
		return nil, false
	}
	for _, content := range assistant.Content {
		if call, ok := content.(*ai.ToolCallContent); ok && call != nil {
			return call.Arguments, true
		}
	}
	return nil, false
}

func mutateOwnedArguments(arguments map[string]any, owner string) {
	arguments["path"] = owner
	item := arguments["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	item["value"] = owner
	arguments["raw"].(json.RawMessage)[0] = '['
}

func TestToolArgumentsAreDetachedAcrossBoundaries(t *testing.T) {
	original := map[string]any{
		"path": "/safe",
		"nested": map[string]any{
			"items": []any{map[string]any{"value": "original"}},
		},
		"raw": json.RawMessage(`{"ok":true}`),
	}

	releaseHook := make(chan struct{})
	var hookPath, toolPath, toolNested string
	tool := &rawOwnershipTool{
		definition: ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "inspect"}},
		execute: func(_ context.Context, _ string, args map[string]any, _ func(ToolUpdate)) (*ToolOutput, error) {
			toolPath = args["path"].(string)
			item := args["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
			toolNested = item["value"].(string)

			// A raw tool owns this invocation copy. Mutating it must not rewrite
			// the model message retained in the transcript.
			args["path"] = "/mutated-by-tool"
			item["value"] = "mutated-by-tool"
			args["raw"].(json.RawMessage)[0] = '['
			return &ToolOutput{}, nil
		},
	}

	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("call_1", "inspect", original)),
		assistantText("done"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{tool},
		BeforeToolCall: func(_ context.Context, call ai.ToolCallContent) error {
			<-releaseHook
			hookPath = call.Arguments["path"].(string)
			call.Arguments["path"] = "/mutated-by-hook"
			call.Arguments["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)["value"] = "mutated-by-hook"
			return nil
		},
	})

	stream, err := a.PromptText(context.Background(), "go")
	if err != nil {
		t.Fatalf("PromptText: %v", err)
	}
	released := false
	mutatedBoundaries := map[string]bool{}
	for event := range stream.Events() {
		switch event := event.(type) {
		case MessageStartEvent:
			if arguments, ok := toolArguments(event.Message); ok {
				mutateOwnedArguments(arguments, "/mutated-by-message-start")
				mutatedBoundaries["message-start"] = true
			}
		case MessageUpdateEvent:
			if arguments, ok := toolArguments(event.Message); ok {
				mutateOwnedArguments(arguments, "/mutated-by-message-update")
				mutatedBoundaries["message-update"] = true
			}
			if delta, ok := event.Inner.(ai.TextDeltaEvent); ok {
				if arguments, ok := toolArguments(&delta.Partial); ok {
					mutateOwnedArguments(arguments, "/mutated-by-inner-event")
					mutatedBoundaries["inner-event"] = true
				}
			}
		case MessageEndEvent:
			if arguments, ok := toolArguments(event.Message); ok {
				mutateOwnedArguments(arguments, "/mutated-by-message-end")
				mutatedBoundaries["message-end"] = true
			}
		case ToolExecutionStartEvent:
			mutateOwnedArguments(event.Args, "/mutated-by-tool-start")
			mutatedBoundaries["tool-start"] = true
			if !released {
				close(releaseHook)
				released = true
			}
		case TurnEndEvent:
			if arguments, ok := toolArguments(event.Message); ok {
				mutateOwnedArguments(arguments, "/mutated-by-turn-end")
				mutatedBoundaries["turn-end"] = true
			}
		case AgentEndEvent:
			for _, message := range event.NewMessages {
				if arguments, ok := toolArguments(message); ok {
					mutateOwnedArguments(arguments, "/mutated-by-agent-end")
					mutatedBoundaries["agent-end"] = true
				}
			}
		}
	}
	result, err := stream.Result(context.Background())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if result.Err != nil {
		t.Fatalf("run error: %v", result.Err)
	}
	if !released {
		t.Fatal("tool start event was not published")
	}
	for _, boundary := range []string{
		"message-start", "message-update", "inner-event", "message-end",
		"tool-start", "turn-end", "agent-end",
	} {
		if !mutatedBoundaries[boundary] {
			t.Errorf("ownership boundary %q was not exercised", boundary)
		}
	}
	if hookPath != "/safe" || toolPath != "/safe" || toolNested != "original" {
		t.Fatalf("detached values: hook=%q tool=%q nested=%q", hookPath, toolPath, toolNested)
	}

	stored := result.NewMessages[1].(*ai.AssistantMessage).Content[0].(*ai.ToolCallContent).Arguments
	if stored["path"] != "/safe" {
		t.Fatalf("stored path = %v, want /safe", stored["path"])
	}
	storedItem := stored["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	if storedItem["value"] != "original" {
		t.Fatalf("stored nested value = %v, want original", storedItem["value"])
	}
	if got := string(stored["raw"].(json.RawMessage)); got != `{"ok":true}` {
		t.Fatalf("stored raw JSON = %s", got)
	}

	secondPromptArguments, ok := toolArguments(llm.promptAt(1).Messages[1])
	if !ok || secondPromptArguments["path"] != "/safe" {
		t.Fatalf("second-turn prompt arguments = %#v, want original values", secondPromptArguments)
	}

	// The run result is another ownership boundary: changing it after the run
	// must not rewrite the agent's retained transcript.
	mutateOwnedArguments(stored, "/mutated-by-result")
	historyArguments, ok := toolArguments(a.Messages()[1])
	if !ok || historyArguments["path"] != "/safe" {
		t.Fatalf("agent history arguments = %#v, want original values", historyArguments)
	}
	historyItem := historyArguments["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)
	if historyItem["value"] != "original" || string(historyArguments["raw"].(json.RawMessage)) != `{"ok":true}` {
		t.Fatalf("agent history nested arguments were mutated: %#v", historyArguments)
	}
}

func TestToolStateMismatchFailsWithoutAnotherTurn(t *testing.T) {
	t.Run("tool-use stop without calls", func(t *testing.T) {
		llm := &fakeLLM{script: []*ai.AssistantMessage{
			{Role: ai.RoleAssistant, StopReason: ai.StopReasonToolUse},
			assistantText("must not be reached"),
		}}
		a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

		stream, _ := a.PromptText(context.Background(), "go")
		_, result, err := drainRun(t, stream)
		if err != nil {
			t.Fatalf("Result: %v", err)
		}
		if result.Err == nil || !strings.Contains(result.Err.Error(), "without tool calls") {
			t.Fatalf("run error = %v, want tool-state mismatch", result.Err)
		}
		if llm.callCount() != 1 {
			t.Fatalf("LLM calls = %d, want 1", llm.callCount())
		}
	})

	t.Run("non-tool stop with calls", func(t *testing.T) {
		executed := false
		tool := &rawOwnershipTool{
			definition: ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "inspect"}},
			execute: func(context.Context, string, map[string]any, func(ToolUpdate)) (*ToolOutput, error) {
				executed = true
				return &ToolOutput{}, nil
			},
		}
		message := assistantToolCalls(toolCall("call_1", "inspect", map[string]any{}))
		message.StopReason = ai.StopReasonStop
		llm := &fakeLLM{script: []*ai.AssistantMessage{message, assistantText("must not be reached")}}
		a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m", Tools: []Tool{tool}})

		stream, _ := a.PromptText(context.Background(), "go")
		_, result, err := drainRun(t, stream)
		if err != nil {
			t.Fatalf("Result: %v", err)
		}
		if result.Err == nil || !strings.Contains(result.Err.Error(), "with stop reason") {
			t.Fatalf("run error = %v, want tool-state mismatch", result.Err)
		}
		if executed {
			t.Fatal("inconsistent tool call must not execute")
		}
		if llm.callCount() != 1 {
			t.Fatalf("LLM calls = %d, want 1", llm.callCount())
		}
	})

	t.Run("malformed successful tool calls", func(t *testing.T) {
		tests := []struct {
			name      string
			message   *ai.AssistantMessage
			errorPart string
		}{
			{
				name:      "empty id",
				message:   assistantToolCalls(toolCall("", "inspect", map[string]any{})),
				errorPart: "empty id",
			},
			{
				name:      "empty name",
				message:   assistantToolCalls(toolCall("call_1", "", map[string]any{})),
				errorPart: "empty name",
			},
			{
				name: "duplicate id",
				message: assistantToolCalls(
					toolCall("call_1", "inspect", map[string]any{}),
					toolCall("call_1", "inspect", map[string]any{}),
				),
				errorPart: "duplicate tool call id",
			},
			{
				name: "invalid content type",
				message: assistantToolCalls(&ai.ToolCallContent{
					Type: ai.ContentTypeText, Id: "call_1", Name: "inspect", Arguments: map[string]any{},
				}),
				errorPart: "invalid content type",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				llm := &fakeLLM{script: []*ai.AssistantMessage{
					tt.message, assistantText("must not be reached"),
				}}
				a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

				stream, _ := a.PromptText(context.Background(), "go")
				_, result, err := drainRun(t, stream)
				if err != nil {
					t.Fatalf("Result: %v", err)
				}
				if result.Err == nil || !strings.Contains(result.Err.Error(), tt.errorPart) {
					t.Fatalf("run error = %v, want %q", result.Err, tt.errorPart)
				}
				if llm.callCount() != 1 {
					t.Fatalf("LLM calls = %d, want 1", llm.callCount())
				}
			})
		}
	})

	t.Run("unknown stop reason", func(t *testing.T) {
		message := assistantText("done")
		message.StopReason = ai.StopReason("unexpected")
		llm := &fakeLLM{script: []*ai.AssistantMessage{message}}
		a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

		stream, _ := a.PromptText(context.Background(), "go")
		_, result, err := drainRun(t, stream)
		if err != nil {
			t.Fatalf("Result: %v", err)
		}
		if result.Err == nil || !strings.Contains(result.Err.Error(), "unknown stop reason") {
			t.Fatalf("run error = %v, want unknown stop reason", result.Err)
		}
	})
}

func TestErrorMessageMayRetainPartialToolCall(t *testing.T) {
	executed := false
	tool := &rawOwnershipTool{
		definition: ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "inspect"}},
		execute: func(context.Context, string, map[string]any, func(ToolUpdate)) (*ToolOutput, error) {
			executed = true
			return &ToolOutput{}, nil
		},
	}
	failed := assistantToolCalls(toolCall("call_1", "inspect", map[string]any{"partial": true}))
	failed.StopReason = ai.StopReasonError
	failed.ErrorMessage = "stream failed"
	llm := &fakeLLM{script: []*ai.AssistantMessage{failed}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m", Tools: []Tool{tool}})

	stream, _ := a.PromptText(context.Background(), "go")
	_, result, err := drainRun(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if result.Err != nil {
		t.Fatalf("model error must remain on Last, got run error %v", result.Err)
	}
	if executed {
		t.Fatal("partial tool call from an error message must not execute")
	}
}
