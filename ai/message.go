package ai

type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

type Role string

const (
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
)

type Message interface {
	isMessage()
	GetRole() Role
}

type AgentMessage = Message

type UserMessage struct {
	Role      Role          `json:"role"`
	Content   []UserContent `json:"content"`
	Timestamp int64         `json:"timestamp"`
}

func (m *UserMessage) isMessage() {}

func (m *UserMessage) GetRole() Role { return RoleUser }

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

func (m *AssistantMessage) isMessage() {}

func (m *AssistantMessage) GetRole() Role { return RoleAssistant }

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

func (m *ToolResultMessage[T]) isMessage() {}

func (m *ToolResultMessage[T]) GetRole() Role { return RoleToolResult }

func (m *ToolResultMessage[T]) ToolResultData() (string, string, []ToolResultContent, bool) {
	return m.ToolCallId, m.ToolName, m.Content, m.IsError
}
