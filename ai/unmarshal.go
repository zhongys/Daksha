package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Message JSON deserialization. Marshaling interface-typed unions works out
// of the box, but reading them back needs role/type discrimination — this
// file is what lets the application layer restore persisted transcripts
// ([]byte from storage → []Message → agent.Config.Messages).

// UnmarshalMessage decodes one message by its role discriminator. Tool
// result messages are instantiated as ToolResultMessage[json.RawMessage],
// keeping app-specific Details opaque but round-trippable.
func UnmarshalMessage(data []byte) (Message, error) {
	var probe struct {
		Role Role `json:"role"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("ai: unmarshal message: %w", err)
	}
	switch probe.Role {
	case RoleUser:
		var m UserMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return &m, nil
	case RoleAssistant:
		var m AssistantMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return &m, nil
	case RoleToolResult:
		var m ToolResultMessage[json.RawMessage]
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return &m, nil
	default:
		return nil, fmt.Errorf("ai: unknown message role %q", probe.Role)
	}
}

// UnmarshalMessages decodes a JSON array of messages.
func UnmarshalMessages(data []byte) ([]Message, error) {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return nil, fmt.Errorf("ai: unmarshal messages: %w", err)
	}
	out := make([]Message, 0, len(raws))
	for i, raw := range raws {
		m, err := UnmarshalMessage(raw)
		if err != nil {
			return nil, fmt.Errorf("ai: message %d: %w", i, err)
		}
		out = append(out, m)
	}
	return out, nil
}

func (m *UserMessage) UnmarshalJSON(data []byte) error {
	var shadow struct {
		Role      Role              `json:"role"`
		Content   []json.RawMessage `json:"content"`
		Timestamp int64             `json:"timestamp"`
	}
	if err := json.Unmarshal(data, &shadow); err != nil {
		return err
	}
	m.Role = shadow.Role
	m.Timestamp = shadow.Timestamp
	m.Content = nil
	for _, raw := range shadow.Content {
		c, err := unmarshalUserContent(raw)
		if err != nil {
			return err
		}
		m.Content = append(m.Content, c)
	}
	return nil
}

func (m *AssistantMessage) UnmarshalJSON(data []byte) error {
	type plain AssistantMessage // strips methods to avoid recursion
	var shadow struct {
		plain
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &shadow); err != nil {
		return err
	}
	*m = AssistantMessage(shadow.plain)
	m.Content = nil
	for _, raw := range shadow.Content {
		c, err := unmarshalAssistantContent(raw)
		if err != nil {
			return err
		}
		m.Content = append(m.Content, c)
	}
	return nil
}

func (m *ToolResultMessage[T]) UnmarshalJSON(data []byte) error {
	var shadow struct {
		Role       Role              `json:"role"`
		ToolCallId string            `json:"toolCallId"`
		ToolName   string            `json:"toolName"`
		Content    []json.RawMessage `json:"content"`
		Details    *T                `json:"details"`
		IsError    bool              `json:"isError"`
		Timestamp  int64             `json:"timestamp"`
	}
	if err := json.Unmarshal(data, &shadow); err != nil {
		return err
	}
	m.Role = shadow.Role
	m.ToolCallId = shadow.ToolCallId
	m.ToolName = shadow.ToolName
	m.Details = shadow.Details
	m.IsError = shadow.IsError
	m.Timestamp = shadow.Timestamp
	m.Content = nil
	for _, raw := range shadow.Content {
		c, err := unmarshalToolResultContent(raw)
		if err != nil {
			return err
		}
		m.Content = append(m.Content, c)
	}
	return nil
}

func contentType(raw json.RawMessage) (ContentType, error) {
	var probe struct {
		Type ContentType `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return "", fmt.Errorf("ai: unmarshal content: %w", err)
	}
	return probe.Type, nil
}

func unmarshalUserContent(raw json.RawMessage) (UserContent, error) {
	t, err := contentType(raw)
	if err != nil {
		return nil, err
	}
	switch t {
	case ContentTypeText:
		return decodeContent[TextContent](raw)
	case ContentTypeImage:
		return decodeContent[ImageContent](raw)
	case ContentTypeAudio:
		return decodeContent[AudioContent](raw)
	case ContentTypeVideo:
		return decodeContent[VideoContent](raw)
	default:
		return nil, fmt.Errorf("ai: unknown user content type %q", t)
	}
}

func unmarshalAssistantContent(raw json.RawMessage) (AssistantContent, error) {
	t, err := contentType(raw)
	if err != nil {
		return nil, err
	}
	switch t {
	case ContentTypeText:
		return decodeContent[TextContent](raw)
	case ContentTypeThinking:
		return decodeContent[ThinkingContent](raw)
	case ContentTypeToolCall:
		return decodeToolCallContent(raw)
	default:
		return nil, fmt.Errorf("ai: unknown assistant content type %q", t)
	}
}

// decodeToolCallContent is deliberately specialized: Arguments is an
// interface-backed JSON tree whose numbers must remain exact. The default
// decoder would turn them into float64 when restoring a persisted transcript,
// corrupting integer tool arguments above 2^53. Other content decoders retain
// their existing concrete-number behavior.
func decodeToolCallContent(raw json.RawMessage) (*ToolCallContent, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var content ToolCallContent
	if err := decoder.Decode(&content); err != nil {
		return nil, err
	}
	return &content, nil
}

func unmarshalToolResultContent(raw json.RawMessage) (ToolResultContent, error) {
	t, err := contentType(raw)
	if err != nil {
		return nil, err
	}
	switch t {
	case ContentTypeText:
		return decodeContent[TextContent](raw)
	case ContentTypeImage:
		return decodeContent[ImageContent](raw)
	default:
		return nil, fmt.Errorf("ai: unknown tool result content type %q", t)
	}
}

func decodeContent[C any](raw json.RawMessage) (*C, error) {
	var c C
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	return &c, nil
}
