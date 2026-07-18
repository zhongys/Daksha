package agent

import (
	"encoding/json"

	"github.com/zhongys/Daksha/ai"
)

func cloneMessage(message ai.Message) ai.Message {
	return ai.CloneMessage(message)
}

// cloneMessages detaches every protocol-visible content block. User and tool
// result messages are model input too, so retaining their pointers would let
// callers, callbacks or event consumers rewrite agent history.
func cloneMessages(messages []ai.Message) []ai.Message {
	return ai.CloneMessages(messages)
}

func cloneToolResultContents(contents []ai.ToolResultContent) []ai.ToolResultContent {
	message := &ai.ToolResultMessage[struct{}]{Content: contents}
	return ai.CloneMessage(message).(*ai.ToolResultMessage[struct{}]).Content
}

func cloneToolUpdate(update ToolUpdate) ToolUpdate {
	update.Content = cloneToolResultContents(update.Content)
	update.Details = cloneApplicationDetails(update.Details)
	return update
}

func cloneApplicationDetails(details any) any {
	value := details
	message := &ai.ToolResultMessage[any]{Details: &value}
	cloned := ai.CloneMessage(message).(*ai.ToolResultMessage[any])
	return *cloned.Details
}

func cloneRunOutput(output *RunOutput) *RunOutput {
	if output == nil {
		return nil
	}
	cloned := *output
	cloned.Value = cloneApplicationDetails(output.Value)
	return &cloned
}

// cloneAssistantEvent keeps the provider stream's event snapshot private from
// agent-level consumers. In particular, mutating MessageUpdateEvent.Inner must
// not rewrite the final tool arguments later returned by the provider stream.
func cloneAssistantEvent(event ai.AssistantMessageEvent) ai.AssistantMessageEvent {
	clone := func(message ai.AssistantMessage) ai.AssistantMessage {
		return *ai.CloneAssistantMessage(&message)
	}
	switch event := event.(type) {
	case ai.StartEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.TextStartEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.TextDeltaEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.TextEndEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.JSONStartEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.JSONDeltaEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.JSONEndEvent:
		event.Content = append(json.RawMessage(nil), event.Content...)
		event.Partial = clone(event.Partial)
		return event
	case ai.ThinkingStartEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.ThinkingDeltaEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.ThinkingEndEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.ToolCallStartEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.ToolCallDeltaEvent:
		event.Partial = clone(event.Partial)
		return event
	case ai.ToolCallEndEvent:
		event.ToolCall = *cloneToolCallContent(&event.ToolCall)
		event.Partial = clone(event.Partial)
		return event
	case ai.DoneEvent:
		event.Message = clone(event.Message)
		return event
	case ai.ErrorEvent:
		event.Error = clone(event.Error)
		return event
	default:
		return event
	}
}
