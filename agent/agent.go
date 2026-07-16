package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/zhongys/Daksha/ai"
)

// LLM is the slice of the ai layer the agent consumes — satisfied by
// *ai.Client and assembled in main. Defining it consumer-side keeps agent
// tests free of HTTP and leaves room for middleware (rate limiting, audit)
// between agent and client.
type LLM interface {
	Stream(ctx context.Context, provider, model string, prompt ai.Prompt,
		opts ai.StreamOptions) *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage]
}

// ErrBusy is returned by Prompt/Continue/Reset while a run is in progress.
// To interact with a running agent use Steer, FollowUp or Abort.
var ErrBusy = errors.New("agent: a run is already in progress")

// Config wires a new Agent. LLM, Provider and Model are required.
type Config struct {
	LLM      LLM
	Provider string
	Model    string

	SystemPrompt string
	Tools        []Tool
	// Options are the per-turn defaults (temperature, reasoning effort, …).
	// Mutable via SetOptions; changes apply from the next turn.
	Options ai.StreamOptions
	// ToolExecution is the batch execution mode; empty means sequential.
	// Any Sequential-wrapped tool in a batch forces the batch sequential.
	ToolExecution ExecutionMode
	// MaxTurns hard-caps turns per run; 0 means unlimited. Production
	// deployments should always set it — tool loops burn real money.
	MaxTurns int

	// TransformContext runs before each LLM call to prune, compact or
	// inject context. It receives a copy and returns the messages to send.
	TransformContext func(ctx context.Context, messages []ai.Message) ([]ai.Message, error)
	// BeforeToolCall runs after tool_execution_start; a non-nil error
	// blocks the call and is reported to the model as an IsError result.
	// The permission/audit choke point.
	BeforeToolCall func(ctx context.Context, call ai.ToolCallContent) error
	// ShouldStopAfterTurn is polled after each completed turn; returning
	// true ends the run gracefully before queue checks and the next LLM
	// call. Natural home for cost circuit breakers.
	ShouldStopAfterTurn func(ctx context.Context, last *ai.AssistantMessage, messages []ai.Message) bool

	// Messages is the initial context, e.g. a session restored from
	// storage via ai.UnmarshalMessages.
	Messages []ai.Message
}

// Agent is a stateful conversation loop: one instance per session. All
// methods are safe for concurrent use; at most one run executes at a time
// (single-flight — concurrent Prompt/Continue return ErrBusy).
type Agent struct {
	mu sync.RWMutex
	// toolStartMu makes Abort and the final pre-Execute cancellation check a
	// linearizable boundary. It is never held while a tool runs: handlers that
	// already claimed a start still receive cancellation and can unwind.
	toolStartMu sync.Mutex

	llm           LLM
	provider      string
	model         string
	systemPrompt  string
	tools         []Tool
	options       ai.StreamOptions
	toolExecution ExecutionMode
	maxTurns      int

	transformContext    func(ctx context.Context, messages []ai.Message) ([]ai.Message, error)
	beforeToolCall      func(ctx context.Context, call ai.ToolCallContent) error
	shouldStopAfterTurn func(ctx context.Context, last *ai.AssistantMessage, messages []ai.Message) bool

	messages []ai.Message
	steering []ai.Message
	followUp []ai.Message

	running bool
	cancel  context.CancelFunc
	// activeRunID identifies the run that owns running/cancel. A completed
	// run is retired before its result becomes observable; the identity keeps
	// its deferred cleanup from clearing a newer run started in that window.
	activeRunID uint64
	nextRunID   uint64
}

func New(cfg Config) (*Agent, error) {
	if cfg.LLM == nil {
		return nil, errors.New("agent: Config.LLM is required")
	}
	if cfg.Provider == "" || cfg.Model == "" {
		return nil, errors.New("agent: Config.Provider and Config.Model are required")
	}
	return &Agent{
		llm:                 cfg.LLM,
		provider:            cfg.Provider,
		model:               cfg.Model,
		systemPrompt:        cfg.SystemPrompt,
		tools:               append([]Tool(nil), cfg.Tools...),
		options:             cfg.Options,
		toolExecution:       cfg.ToolExecution,
		maxTurns:            cfg.MaxTurns,
		transformContext:    cfg.TransformContext,
		beforeToolCall:      cfg.BeforeToolCall,
		shouldStopAfterTurn: cfg.ShouldStopAfterTurn,
		messages:            append([]ai.Message(nil), cfg.Messages...),
	}, nil
}

// UserText builds a plain-text user message stamped now.
func UserText(text string) *ai.UserMessage {
	return &ai.UserMessage{
		Role:      ai.RoleUser,
		Content:   []ai.UserContent{&ai.TextContent{Type: ai.ContentTypeText, Text: text}},
		Timestamp: time.Now().UnixMilli(),
	}
}

// Prompt appends the user message and starts a run. The returned stream
// follows the ai layer contract: drain Events() until closed, then Result()
// yields the RunResult. The synchronous error covers caller mistakes only
// (busy, nil message); model and tool failures ride inside the stream.
func (a *Agent) Prompt(ctx context.Context, msg *ai.UserMessage,
) (*ai.EventStream[Event, *RunResult], error) {
	if msg == nil {
		return nil, errors.New("agent: nil message")
	}
	return a.beginRun(ctx, []ai.Message{msg}, false)
}

// PromptText is Prompt with a plain-text user message.
func (a *Agent) PromptText(ctx context.Context, text string,
) (*ai.EventStream[Event, *RunResult], error) {
	return a.Prompt(ctx, UserText(text))
}

// Continue resumes from the existing context without adding a message —
// the retry path after an error. The last message must be a user or
// toolResult message.
func (a *Agent) Continue(ctx context.Context) (*ai.EventStream[Event, *RunResult], error) {
	return a.beginRun(ctx, nil, true)
}

func (a *Agent) beginRun(ctx context.Context, initial []ai.Message, continuing bool,
) (*ai.EventStream[Event, *RunResult], error) {
	if ctx == nil {
		return nil, errors.New("agent: nil context")
	}

	// Construct the child context outside a.mu: a custom parent may run code
	// from AfterFunc while WithCancel wires cancellation propagation.
	runCtx, cancel := context.WithCancel(ctx)

	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		cancel()
		return nil, ErrBusy
	}
	if continuing {
		var last ai.Message
		if n := len(a.messages); n > 0 {
			last = a.messages[n-1]
		}
		if last == nil {
			a.mu.Unlock()
			cancel()
			return nil, errors.New("agent: cannot continue an empty context")
		}
		if _, isAssistant := last.(*ai.AssistantMessage); isAssistant {
			a.mu.Unlock()
			cancel()
			return nil, errors.New("agent: cannot continue after an assistant message")
		}
	}
	a.nextRunID++
	if a.nextRunID == 0 { // reserve zero for "no active run" after wraparound
		a.nextRunID++
	}
	runID := a.nextRunID
	a.running = true
	a.cancel = cancel
	a.activeRunID = runID
	a.mu.Unlock()

	stream, producer := ai.NewEventStream[Event, *RunResult](runCtx, 64)
	go a.run(runCtx, cancel, runID, producer, initial)
	return stream, nil
}

// retireRun makes a run inactive only if it still owns the active slot.
// It is deliberately idempotent: run calls it before publishing its result
// and again from its defer for early exits or panics.
func (a *Agent) retireRun(runID uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.activeRunID != runID {
		return
	}
	a.running = false
	a.cancel = nil
	a.activeRunID = 0
}

// Steer queues a user message for injection after the current tool batch
// completes; the model sees it on the next LLM call. Running tools are
// never interrupted. Safe to call anytime; queued messages are consumed by
// the active (or next) run.
func (a *Agent) Steer(msg *ai.UserMessage) error {
	if msg == nil {
		return errors.New("agent: nil message")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steering = append(a.steering, msg)
	return nil
}

// FollowUp queues a user message that starts another turn when the run
// would otherwise end.
func (a *Agent) FollowUp(msg *ai.UserMessage) error {
	if msg == nil {
		return errors.New("agent: nil message")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUp = append(a.followUp, msg)
	return nil
}

func (a *Agent) ClearSteering() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steering = nil
}

func (a *Agent) ClearFollowUp() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUp = nil
}

// Abort cancels the current run, if any, and establishes a hard tool-start
// barrier: after it returns, no tool which had not already claimed its start
// may enter Execute. Already-started tools receive the canceled context and
// are allowed to unwind normally.
func (a *Agent) Abort() {
	// Serialize cancellation with the final tool-start claim. Once Abort
	// returns, no not-yet-claimed handler can enter Execute.
	a.toolStartMu.Lock()
	a.mu.Lock()
	cancel := a.cancel
	a.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	a.toolStartMu.Unlock()
}

// claimToolStart is the last gate before invoking Tool.Execute. Returning
// true means the call was dispatched before any concurrent Abort boundary;
// the mutex is deliberately released before Execute so Abort can cancel an
// already-running handler instead of waiting for it.
func (a *Agent) claimToolStart(ctx context.Context) bool {
	a.toolStartMu.Lock()
	defer a.toolStartMu.Unlock()
	return ctx.Err() == nil
}

// Reset clears the context and both queues. Fails while a run is active.
func (a *Agent) Reset() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.running {
		return ErrBusy
	}
	a.messages = nil
	a.steering = nil
	a.followUp = nil
	return nil
}

// Messages returns a copy of the current context.
func (a *Agent) Messages() []ai.Message {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return append([]ai.Message(nil), a.messages...)
}

func (a *Agent) IsRunning() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.running
}

// Setters may be called anytime; a running run picks changes up on its next
// turn (per-turn snapshot semantics).

func (a *Agent) SetSystemPrompt(s string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.systemPrompt = s
}

func (a *Agent) SetModel(provider, model string) error {
	if provider == "" || model == "" {
		return fmt.Errorf("agent: provider and model must both be set")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.provider = provider
	a.model = model
	return nil
}

func (a *Agent) SetTools(tools []Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools = append([]Tool(nil), tools...)
}

func (a *Agent) SetOptions(opts ai.StreamOptions) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.options = opts
}
