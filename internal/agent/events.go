package agent

import "github.com/zhongys/Daksha/internal/ai"

type EventType string

const (
	EventAgentStart          EventType = "agent_start"
	EventAgentEnd            EventType = "agent_end"
	EventTurnStart           EventType = "turn_start"
	EventTurnEnd             EventType = "turn_end"
	EventMessageStart        EventType = "message_start"
	EventMessageUpdate       EventType = "message_update"
	EventMessageEnd          EventType = "message_end"
	EventToolExecutionStart  EventType = "tool_execution_start"
	EventToolExecutionUpdate EventType = "tool_execution_update"
	EventToolExecutionEnd    EventType = "tool_execution_end"
)

// Event is the agent-level event union, mirroring pi-agent-core. One run =
// one event stream; message_end events carry every finalized message in
// order and double as the application's incremental persistence hook.
type Event interface {
	EventType() EventType
}

type AgentStartEvent struct{}

func (AgentStartEvent) EventType() EventType { return EventAgentStart }

type AgentEndEvent struct {
	// NewMessages are the messages appended to the context by this run.
	NewMessages []ai.Message
}

func (AgentEndEvent) EventType() EventType { return EventAgentEnd }

type TurnStartEvent struct {
	// Turn counts from 1 within one run.
	Turn int
}

func (TurnStartEvent) EventType() EventType { return EventTurnStart }

type TurnEndEvent struct {
	Turn    int
	Message *ai.AssistantMessage
	// ToolResults are the toolResult messages of this turn, in assistant
	// source order.
	ToolResults []ai.Message
}

func (TurnEndEvent) EventType() EventType { return EventTurnEnd }

type MessageStartEvent struct {
	Message ai.Message
}

func (MessageStartEvent) EventType() EventType { return EventMessageStart }

// MessageUpdateEvent streams assistant deltas. Inner is the untranslated ai
// layer event — unwrap it for character-level rendering.
type MessageUpdateEvent struct {
	Message *ai.AssistantMessage
	Inner   ai.AssistantMessageEvent
}

func (MessageUpdateEvent) EventType() EventType { return EventMessageUpdate }

type MessageEndEvent struct {
	Message ai.Message
}

func (MessageEndEvent) EventType() EventType { return EventMessageEnd }

type ToolExecutionStartEvent struct {
	ToolCallID string
	ToolName   string
	Args       map[string]any
}

func (ToolExecutionStartEvent) EventType() EventType { return EventToolExecutionStart }

type ToolExecutionUpdateEvent struct {
	ToolCallID string
	ToolName   string
	Update     ToolUpdate
}

func (ToolExecutionUpdateEvent) EventType() EventType { return EventToolExecutionUpdate }

// ToolExecutionEndEvent fires as each tool finishes (completion order under
// parallel execution); the toolResult messages still land in the context in
// assistant source order.
type ToolExecutionEndEvent struct {
	ToolCallID string
	ToolName   string
	Result     ai.Message
	IsError    bool
}

func (ToolExecutionEndEvent) EventType() EventType { return EventToolExecutionEnd }

// RunResult is the final value of one run's event stream.
//
// Model-side failures (vendor errors, aborts) live on Last.StopReason per
// the ai layer contract. Err reports agent-level failures only: MaxTurns
// exceeded, TransformContext errors, or a broken stream. NewMessages is
// valid either way.
type RunResult struct {
	NewMessages []ai.Message
	Last        *ai.AssistantMessage
	Err         error
}
