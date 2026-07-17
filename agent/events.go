package agent

import "github.com/zhongys/Daksha/ai"

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
	// Output is present when a dedicated terminal tool ended the run.
	Output *RunOutput
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
	// Args is owned by this event and detached from history and execution.
	Args map[string]any
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

// RunOutput is the value accepted by a dedicated terminal tool. Value retains
// the concrete parameter type P supplied to NewTerminalTool[P]; ToolCallID and
// ToolName identify the model call that produced it. Event and RunResult
// outputs are independent structural snapshots.
type RunOutput struct {
	ToolCallID string
	ToolName   string
	Value      any
}

// RunOutputAs returns a terminal value with its original concrete type.
func RunOutputAs[T any](output *RunOutput) (T, bool) {
	var zero T
	if output == nil {
		return zero, false
	}
	value, ok := output.Value.(T)
	if !ok {
		return zero, false
	}
	return value, true
}

// RunResult is the final value of one run's event stream.
//
// Model-side failures (vendor errors, aborts) live on Last.StopReason per
// the ai layer contract. Err reports agent-level failures only: MaxTurns
// exceeded, TransformContext errors, invalid assistant/tool state, an
// ambiguous terminal-tool batch, or a broken stream. NewMessages is valid
// either way.
type RunResult struct {
	NewMessages []ai.Message
	Last        *ai.AssistantMessage
	// Output is present only when a dedicated terminal tool successfully
	// ended the run. Legacy ToolOutput.Terminate tools leave it nil.
	Output *RunOutput
	Err    error
}
