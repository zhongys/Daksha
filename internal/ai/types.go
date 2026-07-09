package ai

type Role string

const (
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
)

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeImage    ContentType = "image"
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
func (*ImageContent) isUserContent()         {}
func (*ImageContent) isToolResultContent()   {}
func (*ThinkingContent) isAssistantContent() {}
func (*ToolCallContent) isAssistantContent() {}

type TextContent struct {
	Type          ContentType `json:"type"`
	Text          string      `json:"text"`
	TextSignature string      `json:"textSignature,omitempty"`
}

type ImageContent struct {
	Type     ContentType `json:"type"`
	Data     string      `json:"data"`
	MimeType string      `json:"mimeType"`
}

type ThinkingContent struct {
	Type              ContentType `json:"type"`
	Thinking          string      `json:"thinking"`
	ThinkingSignature string      `json:"thinkingSignature,omitempty"`
	Redacted          bool        `json:"redacted,omitempty"`
}

type ToolCallContent struct {
	Type             ContentType    `json:"type"`
	Id               string         `json:"id"`
	Name             string         `json:"name"`
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
	Details    *T                  `json:"details,omitempty"`
	IsError    bool                `json:"isError"`
	Timestamp  int64               `json:"timestamp"`
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

type DoneEvent struct {
	Reason  StopReason
	Message AssistantMessage
}

func (DoneEvent) EventType() AssistantMessageEventType { return AssistantEventDone }

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
