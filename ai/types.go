package ai

import (
	"encoding/json"
	"fmt"
)

type Role string

const (
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
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

type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

// Cost is the derived price of one turn in nano-yuan (10⁻⁹ CNY).
// Tokens times per-token pricing is the source of truth for billing; Cost is
// computed from them with pure integer arithmetic and is always reproducible.
type Cost struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	CacheRead  int64 `json:"cacheRead"`
	CacheWrite int64 `json:"cacheWrite"`
	Total      int64 `json:"total"`
}

type Usage struct {
	Input        int64 `json:"input"`
	Output       int64 `json:"output"`
	CacheRead    int64 `json:"cacheRead"`
	CacheWrite   int64 `json:"cacheWrite"`
	CacheWrite1h int64 `json:"cacheWrite1h"`
	Reasoning    int64 `json:"reasoning,omitempty"`
	TotalTokens  int64 `json:"totalTokens"`
	Cost         Cost  `json:"cost"`
}

type UserContent interface {
	isUserContent()
}

type AssistantContent interface {
	isAssistantContent()
}
type ToolResultContent interface {
	isToolResultContent()
}

func (*TextContent) isUserContent()          {}
func (*TextContent) isAssistantContent()     {}
func (*TextContent) isToolResultContent()    {}
func (*JSONContent) isAssistantContent()     {}
func (*ImageContent) isUserContent()         {}
func (*ImageContent) isToolResultContent()   {}
func (*AudioContent) isUserContent()         {}
func (*VideoContent) isUserContent()         {}
func (*ThinkingContent) isAssistantContent() {}
func (*ToolCallContent) isAssistantContent() {}

type TextContent struct {
	Type          ContentType `json:"type"`
	Text          string      `json:"text"`
	TextSignature string      `json:"textSignature,omitempty"`
}

// JSONContent is a validated structured assistant response. During streaming,
// JSONStart/JSONDelta events carry an empty placeholder block; Value is filled
// only after the complete response passes final JSON validation.
type JSONContent struct {
	Type       ContentType     `json:"type"`
	SchemaName string          `json:"schemaName,omitempty"`
	Value      json.RawMessage `json:"value,omitempty"`
}

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

type AudioContent struct {
	Type ContentType `json:"type"`
	// Data is the inline base64 payload; mutually exclusive with URL.
	Data string `json:"data,omitempty"`
	// MimeType is required when Data is set.
	MimeType string `json:"mimeType,omitempty"`
	// URL is the by-reference form, passed through to the provider verbatim.
	URL string `json:"url,omitempty"`
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

func (c *ImageContent) Validate() error {
	return ValidateMedia(ContentTypeImage, c.Data, c.MimeType, c.URL)
}

func (c *AudioContent) Validate() error {
	return ValidateMedia(ContentTypeAudio, c.Data, c.MimeType, c.URL)
}

func (c *VideoContent) Validate() error {
	return ValidateMedia(ContentTypeVideo, c.Data, c.MimeType, c.URL)
}

type ThinkingContent struct {
	Type              ContentType `json:"type"`
	Thinking          string      `json:"thinking"`
	ThinkingSignature string      `json:"thinkingSignature,omitempty"`
	Redacted          bool        `json:"redacted,omitempty"`
}

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

type Message interface {
	isMessage()
}

func (m *UserMessage) isMessage()          {}
func (m *AssistantMessage) isMessage()     {}
func (m *ToolResultMessage[T]) isMessage() {}

type UserMessage struct {
	Role      Role          `json:"role"`
	Content   []UserContent `json:"content"`
	Timestamp int64         `json:"timestamp"`
}

type AssistantMessage struct {
	Role          Role                       `json:"role"`
	Content       []AssistantContent         `json:"content"`
	Api           string                     `json:"api"`
	Provider      string                     `json:"provider"`
	Model         string                     `json:"model"`
	ResponseModel string                     `json:"responseModel,omitempty"`
	ResponseId    string                     `json:"responseId,omitempty"`
	Diagnostics   AssistantMessageDiagnostic `json:"diagnostics,omitempty"`
	Usage         Usage                      `json:"usage"`
	StopReason    StopReason                 `json:"stopReason"`
	ErrorMessage  string                     `json:"errorMessage,omitempty"`
	Timestamp     int64                      `json:"timestamp"`
}

type ToolResultMessage[T any] struct {
	Role       Role                `json:"role"`
	ToolCallId string              `json:"toolCallId"`
	ToolName   string              `json:"toolName"`
	Content    []ToolResultContent `json:"content"`
	// Details is application metadata and is not sent to the model. Daksha
	// structurally snapshots ordinary pointer/map/slice/exported-struct data at
	// ownership boundaries. Opaque mutable state hidden behind unexported
	// fields, functions or channels remains caller-owned and must be immutable
	// after handoff.
	Details   *T    `json:"details,omitempty"`
	IsError   bool  `json:"isError"`
	Timestamp int64 `json:"timestamp"`
}

// ToolResult gives protocol adapters generic-free access to any
// ToolResultMessage[T] instantiation.
type ToolResult interface {
	Message
	ToolResultData() (toolCallID, toolName string, content []ToolResultContent, isError bool)
}

func (m *ToolResultMessage[T]) ToolResultData() (string, string, []ToolResultContent, bool) {
	return m.ToolCallId, m.ToolName, m.Content, m.IsError
}

type AssistantMessageEventType string

const (
	AssistantEventStart         AssistantMessageEventType = "start"
	AssistantEventTextStart     AssistantMessageEventType = "text_start"
	AssistantEventTextDelta     AssistantMessageEventType = "text_delta"
	AssistantEventTextEnd       AssistantMessageEventType = "text_end"
	AssistantEventJSONStart     AssistantMessageEventType = "json_start"
	AssistantEventJSONDelta     AssistantMessageEventType = "json_delta"
	AssistantEventJSONEnd       AssistantMessageEventType = "json_end"
	AssistantEventThinkingStart AssistantMessageEventType = "thinking_start"
	AssistantEventThinkingDelta AssistantMessageEventType = "thinking_delta"
	AssistantEventThinkingEnd   AssistantMessageEventType = "thinking_end"
	AssistantEventToolCallStart AssistantMessageEventType = "toolcall_start"
	AssistantEventToolCallDelta AssistantMessageEventType = "toolcall_delta"
	AssistantEventToolCallEnd   AssistantMessageEventType = "toolcall_end"
	AssistantEventDone          AssistantMessageEventType = "done"
	AssistantEventError         AssistantMessageEventType = "error"
)

type AssistantMessageEvent interface {
	EventType() AssistantMessageEventType
}

type StartEvent struct {
	Partial AssistantMessage
}

func (StartEvent) EventType() AssistantMessageEventType { return AssistantEventStart }

type TextStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

func (TextStartEvent) EventType() AssistantMessageEventType { return AssistantEventTextStart }

type TextDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

func (TextDeltaEvent) EventType() AssistantMessageEventType { return AssistantEventTextDelta }

type TextEndEvent struct {
	ContentIndex int
	Content      string
	Partial      AssistantMessage
}

func (TextEndEvent) EventType() AssistantMessageEventType { return AssistantEventTextEnd }

type JSONStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

func (JSONStartEvent) EventType() AssistantMessageEventType { return AssistantEventJSONStart }

// JSONDeltaEvent carries an unvalidated fragment. Consumers may concatenate
// Delta values for progressive display; only JSONEndEvent.Content is valid
// complete JSON.
type JSONDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

func (JSONDeltaEvent) EventType() AssistantMessageEventType { return AssistantEventJSONDelta }

type JSONEndEvent struct {
	ContentIndex int
	Content      json.RawMessage
	Partial      AssistantMessage
}

func (JSONEndEvent) EventType() AssistantMessageEventType { return AssistantEventJSONEnd }

type ThinkingStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

func (ThinkingStartEvent) EventType() AssistantMessageEventType { return AssistantEventThinkingStart }

type ThinkingDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

func (ThinkingDeltaEvent) EventType() AssistantMessageEventType { return AssistantEventThinkingDelta }

type ThinkingEndEvent struct {
	ContentIndex int
	Content      string
	Partial      AssistantMessage
}

func (ThinkingEndEvent) EventType() AssistantMessageEventType { return AssistantEventThinkingEnd }

type ToolCallStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

func (ToolCallStartEvent) EventType() AssistantMessageEventType { return AssistantEventToolCallStart }

type ToolCallDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

func (ToolCallDeltaEvent) EventType() AssistantMessageEventType { return AssistantEventToolCallDelta }

type ToolCallEndEvent struct {
	ContentIndex int
	ToolCall     ToolCallContent
	Partial      AssistantMessage
}

func (ToolCallEndEvent) EventType() AssistantMessageEventType { return AssistantEventToolCallEnd }

// DoneEvent is the sole terminal event for a successful turn. All content
// blocks emitted by the turn have ended before DoneEvent is published.
type DoneEvent struct {
	Reason  StopReason
	Message AssistantMessage
}

func (DoneEvent) EventType() AssistantMessageEventType { return AssistantEventDone }

// ErrorEvent is the sole terminal event for a failed turn. It implicitly
// aborts every content block that has emitted Start without a matching End;
// End events certify valid final content and therefore are not emitted for
// those blocks on an error path.
type ErrorEvent struct {
	Reason StopReason
	Error  AssistantMessage
}

func (ErrorEvent) EventType() AssistantMessageEventType { return AssistantEventError }

type DiagnosticErrorInfo struct {
	Name    string `json:"name,omitempty"`
	Message string `json:"message"`
	Stack   string `json:"stack,omitempty"`
	Code    string `json:"code,omitempty"`
}

type AssistantMessageDiagnostic struct {
	Type      string              `json:"type"`
	Timestamp int64               `json:"timestamp"`
	Error     DiagnosticErrorInfo `json:"error,omitempty"`
	Details   map[string]any      `json:"details,omitempty"`
}
