package ai

import (
	"encoding/json"
)

// ToolResult gives protocol adapters generic-free access to any
// ToolResultMessage[T] instantiation.
type ToolResult interface {
	Message
	ToolResultData() (toolCallID, toolName string, content []ToolResultContent, isError bool)
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
