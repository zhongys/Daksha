package ai

import "encoding/json"

// CloneAssistantMessage returns a detached copy of message.
//
// Assistant messages contain interface slices, pointer-backed content blocks,
// raw JSON byte slices, and JSON object trees. A plain struct copy would leave
// those mutable values shared with the caller. This helper is nil-safe and
// recursively copies every mutable value defined by AssistantMessage's content
// contract.
func CloneAssistantMessage(message *AssistantMessage) *AssistantMessage {
	if message == nil {
		return nil
	}

	cloned := *message
	if message.Content != nil {
		cloned.Content = make([]AssistantContent, len(message.Content))
		for i, content := range message.Content {
			switch content := content.(type) {
			case *TextContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				cloned.Content[i] = &copy
			case *JSONContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				copy.Value = cloneAssistantRawMessage(content.Value)
				cloned.Content[i] = &copy
			case *ThinkingContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				cloned.Content[i] = &copy
			case *ToolCallContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				copy.Arguments = cloneAssistantJSONMap(content.Arguments)
				cloned.Content[i] = &copy
			default:
				// AssistantContent is sealed to this package. Keeping the value is a
				// conservative fallback if a new immutable variant is added before
				// its dedicated copy case.
				cloned.Content[i] = content
			}
		}
	}
	cloned.Diagnostics.Details = cloneAssistantJSONMap(message.Diagnostics.Details)
	return &cloned
}

func cloneAssistantRawMessage(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

func cloneAssistantJSONMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	cloned := make(map[string]any, len(value))
	for key, item := range value {
		cloned[key] = cloneAssistantJSONValue(item)
	}
	return cloned
}

func cloneAssistantJSONValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneAssistantJSONMap(value)
	case []any:
		cloned := make([]any, len(value))
		for i, item := range value {
			cloned[i] = cloneAssistantJSONValue(item)
		}
		return cloned
	case json.RawMessage:
		return cloneAssistantRawMessage(value)
	case []byte:
		return append([]byte(nil), value...)
	default:
		return value
	}
}
