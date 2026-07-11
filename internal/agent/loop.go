package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/zhongys/Daksha/internal/ai"
)

// runConfig is the per-turn snapshot of the agent's mutable configuration:
// setters take effect on the next turn, never mid-turn.
type runConfig struct {
	llm           LLM
	provider      string
	model         string
	systemPrompt  string
	tools         []Tool
	options       ai.StreamOptions
	toolExecution ExecutionMode
	maxTurns      int
}

func (a *Agent) snapshotConfig() runConfig {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return runConfig{
		llm:           a.llm,
		provider:      a.provider,
		model:         a.model,
		systemPrompt:  a.systemPrompt,
		tools:         append([]Tool(nil), a.tools...),
		options:       a.options,
		toolExecution: a.toolExecution,
		maxTurns:      a.maxTurns,
	}
}

func (a *Agent) appendMessage(m ai.Message) {
	a.mu.Lock()
	a.messages = append(a.messages, m)
	a.mu.Unlock()
}

func (a *Agent) drainSteering() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := a.steering
	a.steering = nil
	return out
}

func (a *Agent) drainFollowUp() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := a.followUp
	a.followUp = nil
	return out
}

// emitter serializes event publishing (parallel tool goroutines publish
// concurrently) and latches the first failure so the loop can bail out.
type emitter struct {
	ctx      context.Context
	producer *ai.Producer[Event, *RunResult]

	mu  sync.Mutex
	err error
}

func (e *emitter) publish(ev Event) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.err != nil {
		return false
	}
	e.err = e.producer.Publish(e.ctx, ev)
	return e.err == nil
}

// announce emits the start/end pair for a message that arrives whole
// (user, toolResult) — the application's incremental persistence hook.
func (e *emitter) announce(m ai.Message) bool {
	return e.publish(MessageStartEvent{Message: m}) && e.publish(MessageEndEvent{Message: m})
}

func (a *Agent) run(ctx context.Context, producer *ai.Producer[Event, *RunResult], initial []ai.Message) {
	defer func() {
		a.mu.Lock()
		a.running = false
		a.cancel = nil
		a.mu.Unlock()
	}()

	em := &emitter{ctx: ctx, producer: producer}
	var newMessages []ai.Message
	var last *ai.AssistantMessage
	var runErr error

	record := func(m ai.Message) {
		a.appendMessage(m)
		newMessages = append(newMessages, m)
	}

	em.publish(AgentStartEvent{})
	for _, m := range initial {
		record(m)
		em.announce(m)
	}

	cfg := a.snapshotConfig()

	for turn := 1; em.err == nil; turn++ {
		if cfg.maxTurns > 0 && turn > cfg.maxTurns {
			runErr = fmt.Errorf("agent: max turns (%d) exceeded", cfg.maxTurns)
			break
		}
		em.publish(TurnStartEvent{Turn: turn})

		cfg = a.snapshotConfig()
		msgs := a.Messages()
		if a.transformContext != nil {
			var err error
			if msgs, err = a.transformContext(ctx, msgs); err != nil {
				runErr = fmt.Errorf("agent: transform context: %w", err)
				break
			}
		}

		prompt := ai.Prompt{
			System:   cfg.systemPrompt,
			Messages: msgs,
			Tools:    definitions(cfg.tools),
		}
		stream := cfg.llm.Stream(ctx, cfg.provider, cfg.model, prompt, cfg.options)
		for ev := range stream.Events() {
			switch e := ev.(type) {
			case ai.StartEvent:
				partial := e.Partial
				em.publish(MessageStartEvent{Message: &partial})
			case ai.DoneEvent, ai.ErrorEvent:
				// The final message arrives via Result below.
			default:
				em.publish(MessageUpdateEvent{Message: partialOf(ev), Inner: ev})
			}
		}
		final, err := stream.Result(ctx)
		if err != nil {
			// Stream mechanics failed (context canceled before completion);
			// there is no final message to record.
			runErr = err
			break
		}
		record(final)
		em.publish(MessageEndEvent{Message: final})
		last = final

		var toolResults []ai.Message
		terminate := false
		if final.StopReason == ai.StopReasonToolUse {
			toolResults, terminate = a.executeTools(ctx, em, cfg, final)
			for _, r := range toolResults {
				newMessages = append(newMessages, r)
			}
		}
		em.publish(TurnEndEvent{Turn: turn, Message: final, ToolResults: toolResults})

		if final.StopReason == ai.StopReasonError || final.StopReason == ai.StopReasonAborted {
			break
		}
		if terminate {
			break
		}
		if a.shouldStopAfterTurn != nil && a.shouldStopAfterTurn(ctx, final, a.Messages()) {
			break
		}

		// Steering first: injected after the tool batch, seen next turn.
		if steer := a.drainSteering(); len(steer) > 0 {
			for _, m := range steer {
				record(m)
				em.announce(m)
			}
			continue
		}
		if final.StopReason == ai.StopReasonToolUse {
			continue
		}
		// Natural end: follow-ups get one more turn.
		if followUp := a.drainFollowUp(); len(followUp) > 0 {
			for _, m := range followUp {
				record(m)
				em.announce(m)
			}
			continue
		}
		break
	}

	if runErr == nil && em.err != nil {
		runErr = em.err
	}
	em.publish(AgentEndEvent{NewMessages: newMessages})
	_ = producer.Complete(ctx, &RunResult{NewMessages: newMessages, Last: last, Err: runErr})
}

// toolOutcome pairs one tool call with its finalized result.
type toolOutcome struct {
	call      *ai.ToolCallContent
	msg       *ai.ToolResultMessage[any]
	terminate bool
	// executed marks outcomes that ran (vs blocked/unknown), used only to
	// decide which entries still need running.
	pending bool
}

// executeTools runs the tool calls of one assistant message. Preflight
// (start events + BeforeToolCall) is always sequential in source order;
// execution honors the batch mode; tool_execution_end fires per completion;
// toolResult messages are recorded in assistant source order. Returns the
// recorded messages and whether every result asked to terminate.
func (a *Agent) executeTools(ctx context.Context, em *emitter, cfg runConfig,
	final *ai.AssistantMessage) ([]ai.Message, bool) {

	var calls []*ai.ToolCallContent
	for _, c := range final.Content {
		if tc, ok := c.(*ai.ToolCallContent); ok {
			calls = append(calls, tc)
		}
	}
	if len(calls) == 0 {
		return nil, false
	}

	byName := map[string]Tool{}
	for _, t := range cfg.tools {
		byName[t.Definition().Name] = t
	}

	sequential := cfg.toolExecution != ExecParallel
	for _, c := range calls {
		if t, ok := byName[c.Name]; ok && isSequentialOnly(t) {
			sequential = true
		}
	}

	// Preflight in source order.
	outcomes := make([]*toolOutcome, len(calls))
	for i, call := range calls {
		em.publish(ToolExecutionStartEvent{ToolCallID: call.Id, ToolName: call.Name, Args: call.Arguments})
		o := &toolOutcome{call: call}
		outcomes[i] = o

		tool, known := byName[call.Name]
		switch {
		case !known:
			o.msg = errorResult(call, fmt.Errorf("agent: unknown tool %q", call.Name))
		case a.beforeToolCall != nil:
			if err := a.beforeToolCall(ctx, *call); err != nil {
				o.msg = errorResult(call, fmt.Errorf("agent: tool call blocked: %w", err))
			}
		}
		if o.msg != nil {
			em.publish(ToolExecutionEndEvent{ToolCallID: call.Id, ToolName: call.Name, Result: o.msg, IsError: true})
			continue
		}
		o.pending = true
		_ = tool // executed below
	}

	execute := func(o *toolOutcome) {
		tool := byName[o.call.Name]
		onUpdate := func(u ToolUpdate) {
			em.publish(ToolExecutionUpdateEvent{ToolCallID: o.call.Id, ToolName: o.call.Name, Update: u})
		}
		out, err := tool.Execute(ctx, o.call.Id, o.call.Arguments, onUpdate)
		if err != nil {
			o.msg = errorResult(o.call, err)
		} else {
			o.msg = successResult(o.call, out)
			o.terminate = out != nil && out.Terminate
		}
		em.publish(ToolExecutionEndEvent{
			ToolCallID: o.call.Id, ToolName: o.call.Name, Result: o.msg, IsError: o.msg.IsError,
		})
	}

	if sequential {
		for _, o := range outcomes {
			if o.pending {
				execute(o)
			}
		}
	} else {
		var wg sync.WaitGroup
		for _, o := range outcomes {
			if !o.pending {
				continue
			}
			wg.Add(1)
			go func(o *toolOutcome) {
				defer wg.Done()
				execute(o)
			}(o)
		}
		wg.Wait()
	}

	// Record in assistant source order regardless of completion order.
	results := make([]ai.Message, 0, len(outcomes))
	terminate := true
	for _, o := range outcomes {
		a.appendMessage(o.msg)
		em.announce(o.msg)
		results = append(results, o.msg)
		if !o.terminate {
			terminate = false
		}
	}
	return results, terminate
}

func errorResult(call *ai.ToolCallContent, err error) *ai.ToolResultMessage[any] {
	return &ai.ToolResultMessage[any]{
		Role:       ai.RoleToolResult,
		ToolCallId: call.Id,
		ToolName:   call.Name,
		Content:    []ai.ToolResultContent{&ai.TextContent{Type: ai.ContentTypeText, Text: err.Error()}},
		IsError:    true,
		Timestamp:  time.Now().UnixMilli(),
	}
}

func successResult(call *ai.ToolCallContent, out *ToolOutput) *ai.ToolResultMessage[any] {
	msg := &ai.ToolResultMessage[any]{
		Role:       ai.RoleToolResult,
		ToolCallId: call.Id,
		ToolName:   call.Name,
		Timestamp:  time.Now().UnixMilli(),
	}
	if out != nil {
		msg.Content = out.Content
		if out.Details != nil {
			details := out.Details
			msg.Details = &details
		}
	}
	return msg
}

func definitions(tools []Tool) []ai.ToolDefinition {
	if len(tools) == 0 {
		return nil
	}
	defs := make([]ai.ToolDefinition, 0, len(tools))
	for _, t := range tools {
		defs = append(defs, t.Definition().ToolDefinition)
	}
	return defs
}

// partialOf extracts the partial assistant message carried by every ai
// layer streaming event.
func partialOf(ev ai.AssistantMessageEvent) *ai.AssistantMessage {
	var p ai.AssistantMessage
	switch e := ev.(type) {
	case ai.StartEvent:
		p = e.Partial
	case ai.TextStartEvent:
		p = e.Partial
	case ai.TextDeltaEvent:
		p = e.Partial
	case ai.TextEndEvent:
		p = e.Partial
	case ai.ThinkingStartEvent:
		p = e.Partial
	case ai.ThinkingDeltaEvent:
		p = e.Partial
	case ai.ThinkingEndEvent:
		p = e.Partial
	case ai.ToolCallStartEvent:
		p = e.Partial
	case ai.ToolCallDeltaEvent:
		p = e.Partial
	case ai.ToolCallEndEvent:
		p = e.Partial
	case ai.DoneEvent:
		p = e.Message
	case ai.ErrorEvent:
		p = e.Error
	default:
		return nil
	}
	return &p
}
