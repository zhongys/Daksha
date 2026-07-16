package agent

import (
	"encoding/json"
	"fmt"

	"github.com/zhongys/Daksha/ai"
)

// validatedToolCalls enforces the assistant-message invariant at the agent
// boundary. A successful tool-use turn must contain at least one concrete tool
// call, while successful non-tool turns must not smuggle tool calls past the
// execution gate. Error and aborted messages may retain partial tool-call
// content for diagnostics, but those calls are never executable.
func validatedToolCalls(message *ai.AssistantMessage) ([]*ai.ToolCallContent, error) {
	if message == nil {
		return nil, fmt.Errorf("agent: LLM completed with a nil assistant message")
	}

	var calls []*ai.ToolCallContent
	for _, content := range message.Content {
		call, ok := content.(*ai.ToolCallContent)
		if !ok {
			continue
		}
		if call == nil {
			return nil, fmt.Errorf("agent: assistant message contains a nil tool call")
		}
		calls = append(calls, cloneToolCallContent(call))
	}

	switch message.StopReason {
	case ai.StopReasonToolUse:
		if len(calls) == 0 {
			return nil, fmt.Errorf("agent: stop reason %q without tool calls", message.StopReason)
		}
		seenIDs := make(map[string]struct{}, len(calls))
		for i, call := range calls {
			if call.Type != ai.ContentTypeToolCall {
				return nil, fmt.Errorf(
					"agent: tool call %d has invalid content type %q", i, call.Type,
				)
			}
			if call.Id == "" {
				return nil, fmt.Errorf("agent: tool call %d has an empty id", i)
			}
			if call.Name == "" {
				return nil, fmt.Errorf("agent: tool call %q has an empty name", call.Id)
			}
			if _, duplicate := seenIDs[call.Id]; duplicate {
				return nil, fmt.Errorf("agent: duplicate tool call id %q", call.Id)
			}
			seenIDs[call.Id] = struct{}{}
		}
	case ai.StopReasonError, ai.StopReasonAborted:
		// A failed stream may retain partial tool calls for diagnostics. They are
		// intentionally not returned to the executor.
		return nil, nil
	case ai.StopReasonStop, ai.StopReasonLength:
		if len(calls) > 0 {
			return nil, fmt.Errorf(
				"agent: assistant message contains %d tool call(s) with stop reason %q",
				len(calls), message.StopReason,
			)
		}
	default:
		return nil, fmt.Errorf("agent: unknown stop reason %q", message.StopReason)
	}
	return calls, nil
}

func cloneToolCallContent(call *ai.ToolCallContent) *ai.ToolCallContent {
	if call == nil {
		return nil
	}
	cloned := *call
	cloned.Arguments = cloneToolArguments(call.Arguments)
	return &cloned
}

// cloneToolArguments detaches the JSON tree used for a tool call. Tool
// arguments produced by protocol adapters consist of maps, slices, immutable
// scalars and json.Number values; raw byte values are copied defensively too.
func cloneToolArguments(arguments map[string]any) map[string]any {
	if arguments == nil {
		return nil
	}
	cloned := make(map[string]any, len(arguments))
	for key, value := range arguments {
		cloned[key] = cloneToolArgumentValue(value)
	}
	return cloned
}

func cloneToolArgumentValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneToolArguments(value)
	case []any:
		cloned := make([]any, len(value))
		for i, item := range value {
			cloned[i] = cloneToolArgumentValue(item)
		}
		return cloned
	case json.RawMessage:
		return append(json.RawMessage(nil), value...)
	case []byte:
		return append([]byte(nil), value...)
	default:
		return value
	}
}
