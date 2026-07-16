package agent

import (
	"encoding/json"

	"github.com/zhongys/Daksha/ai"
)

// cloneMessages detaches assistant messages, whose tool-call argument maps are
// the mutable values relevant to execution. Other message variants are kept as
// values in a new slice; the agent never derives executable input from them.
func cloneMessages(messages []ai.Message) []ai.Message {
	if messages == nil {
		return nil
	}
	cloned := make([]ai.Message, len(messages))
	for i, message := range messages {
		if assistant, ok := message.(*ai.AssistantMessage); ok {
			cloned[i] = ai.CloneAssistantMessage(assistant)
			continue
		}
		cloned[i] = message
	}
	return cloned
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
