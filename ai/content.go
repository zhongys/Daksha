package ai

import (
	"encoding/json"
	"fmt"
)

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeJSON     ContentType = "json"
	ContentTypeImage    ContentType = "image"
	ContentTypeAudio    ContentType = "audio"
	ContentTypeVideo    ContentType = "video"
	ContentTypeThinking ContentType = "thinking"
	ContentTypeToolCall ContentType = "toolCall"
)

type Content interface {
	isContent()
	// GetType() ContentType
}

type UserContent interface {
	Content
	isUserContent()
}

type AssistantContent interface {
	Content
	isAssistantContent()
}
type ToolResultContent interface {
	Content
	isToolResultContent()
}

type TextContent struct {
	Type          ContentType `json:"type"`
	Text          string      `json:"text"`
	TextSignature string      `json:"textSignature,omitempty"`
}

func (*TextContent) isUserContent()       {}
func (*TextContent) isContent()           {}
func (*TextContent) isAssistantContent()  {}
func (*TextContent) isToolResultContent() {}

// JSONContent is a validated structured assistant response. During streaming,
// JSONStart/JSONDelta events carry an empty placeholder block; Value is filled
// only after the complete response passes final JSON validation.
type JSONContent struct {
	Type       ContentType     `json:"type"`
	SchemaName string          `json:"schemaName,omitempty"`
	Value      json.RawMessage `json:"value,omitempty"`
}

func (*JSONContent) isContent() {}

func (*JSONContent) isAssistantContent() {}

// Media content blocks carry their payload either inline (Data, base64, with
// MimeType) or by reference (URL) — exactly one of the two forms must be set.
// Adapters reject the empty and the double-set case, and fail loudly when the
// target endpoint cannot accept the given form; the ai layer never downloads
// or transcodes media.

type ImageContent struct {
	Type ContentType `json:"type"`
	// Data is the inline base64 payload; mutually exclusive with URL.
	Data string `json:"data,omitempty"`
	// MimeType is required when Data is set.
	MimeType string `json:"mimeType,omitempty"`
	// URL is the by-reference form, passed through to the provider verbatim.
	URL string `json:"url,omitempty"`
}

func (*ImageContent) isUserContent()       {}
func (*ImageContent) isContent()           {}
func (*ImageContent) isToolResultContent() {}
func (c *ImageContent) Validate() error {
	return ValidateMedia(ContentTypeImage, c.Data, c.MimeType, c.URL)
}

type AudioContent struct {
	Type ContentType `json:"type"`
	// Data is the inline base64 payload; mutually exclusive with URL.
	Data string `json:"data,omitempty"`
	// MimeType is required when Data is set.
	MimeType string `json:"mimeType,omitempty"`
	// URL is the by-reference form, passed through to the provider verbatim.
	URL string `json:"url,omitempty"`
}

func (*AudioContent) isUserContent() {}
func (*AudioContent) isContent()     {}
func (c *AudioContent) Validate() error {
	return ValidateMedia(ContentTypeAudio, c.Data, c.MimeType, c.URL)
}

type VideoContent struct {
	Type ContentType `json:"type"`
	// Data is the inline base64 payload; mutually exclusive with URL.
	Data string `json:"data,omitempty"`
	// MimeType is required when Data is set.
	MimeType string `json:"mimeType,omitempty"`
	// URL is the by-reference form, passed through to the provider verbatim.
	URL string `json:"url,omitempty"`
}

func (*VideoContent) isUserContent() {}
func (*VideoContent) isContent()     {}

func (c *VideoContent) Validate() error {
	return ValidateMedia(ContentTypeVideo, c.Data, c.MimeType, c.URL)
}

// ValidateMedia enforces the media union rule: exactly one of Data and URL,
// and a MimeType alongside Data.
func ValidateMedia(kind ContentType, data, mimeType, url string) error {
	switch {
	case data == "" && url == "":
		return fmt.Errorf("ai: %s content has neither data nor url", kind)
	case data != "" && url != "":
		return fmt.Errorf("ai: %s content has both data and url", kind)
	case data != "" && mimeType == "":
		return fmt.Errorf("ai: %s content has inline data without mimeType", kind)
	}
	return nil
}

type ThinkingContent struct {
	Type              ContentType `json:"type"`
	Thinking          string      `json:"thinking"`
	ThinkingSignature string      `json:"thinkingSignature,omitempty"`
	Redacted          bool        `json:"redacted,omitempty"`
}

func (*ThinkingContent) isAssistantContent() {}
func (*ThinkingContent) isContent()          {}

type ToolCallContent struct {
	Type ContentType `json:"type"`
	Id   string      `json:"id"`
	Name string      `json:"name"`
	// Arguments is a decoded JSON object. Numeric values from protocol
	// adapters and persisted transcripts are encoding/json.Number, including
	// values nested in maps and slices, so raw Tool implementations must not
	// assume float64. Typed tools JSON-decode these values into their declared
	// parameter types.
	Arguments        map[string]any `json:"arguments"`
	ThoughtSignature string         `json:"thoughtSignature,omitempty"`
}

func (*ToolCallContent) isAssistantContent() {}
func (*ToolCallContent) isContent()          {}
