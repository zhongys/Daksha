package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/zhongys/Daksha/ai"
)

// runConfig is the per-turn snapshot of the agent's mutable configuration:
// setters take effect on the next turn, never mid-turn.
type runConfig struct {
	llm           LLM
	provider      string
	model         string
	systemPrompt  string
	tools         []Tool
	definitions   []ai.ToolDefinition
	options       ai.StreamOptions
	optionsErr    error
	toolExecution ExecutionMode
	maxTurns      int
}

func (a *Agent) snapshotConfig() (runConfig, error) {
	a.mu.RLock()
	config := runConfig{
		llm:           a.llm,
		provider:      a.provider,
		model:         a.model,
		systemPrompt:  a.systemPrompt,
		tools:         append([]Tool(nil), a.tools...),
		options:       a.options,
		optionsErr:    a.optionsErr,
		toolExecution: a.toolExecution,
		maxTurns:      a.maxTurns,
	}
	a.mu.RUnlock()
	if config.optionsErr != nil {
		return runConfig{}, config.optionsErr
	}

	// SnapshotStreamOptions may invoke a custom JSON marshaler. Never execute
	// caller code while holding the agent mutex: it may call a setter itself.
	options, err := ai.SnapshotStreamOptions(config.options)
	if err != nil {
		return runConfig{}, fmt.Errorf("agent: snapshot options: %w", err)
	}
	config.options = options
	return config, nil
}

func (a *Agent) appendMessage(m ai.Message) ai.Message {
	owned := cloneMessage(m)
	a.mu.Lock()
	a.messages = append(a.messages, owned)
	a.mu.Unlock()
	return owned
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
	return e.publish(MessageStartEvent{Message: cloneMessage(m)}) &&
		e.publish(MessageEndEvent{Message: cloneMessage(m)})
}

func (e *emitter) failure() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.err
}

func (a *Agent) run(ctx context.Context, cancel context.CancelFunc, runID uint64,
	producer *ai.Producer[Event, *RunResult], initial []ai.Message) {
	defer func() {
		// Always release the context's parent linkage. retireRun is guarded by
		// run identity so this cleanup cannot overwrite a newer run.
		a.retireRun(runID)
		cancel()
	}()

	em := &emitter{ctx: ctx, producer: producer}
	var newMessages []ai.Message
	var last *ai.AssistantMessage
	var runOutput *RunOutput
	var runErr error

	record := func(m ai.Message) {
		owned := a.appendMessage(m)
		newMessages = append(newMessages, owned)
	}

	em.publish(AgentStartEvent{})
	for _, m := range initial {
		record(m)
		em.announce(m)
	}

	cfg, err := a.snapshotConfig()
	if err != nil {
		runErr = err
	}

	for turn := 1; em.err == nil && runErr == nil; turn++ {
		if cfg.maxTurns > 0 && turn > cfg.maxTurns {
			runErr = fmt.Errorf("agent: max turns (%d) exceeded", cfg.maxTurns)
			break
		}
		em.publish(TurnStartEvent{Turn: turn})

		cfg, err = a.snapshotConfig()
		if err != nil {
			runErr = err
			break
		}
		msgs := a.Messages()
		if a.transformContext != nil {
			var err error
			if msgs, err = a.transformContext(ctx, msgs); err != nil {
				runErr = fmt.Errorf("agent: transform context: %w", err)
				break
			}
		}
		cfg.definitions, err = definitions(cfg.tools)
		if err != nil {
			runErr = err
			break
		}

		prompt, err := ai.SnapshotPrompt(ai.Prompt{
			System:   cfg.systemPrompt,
			Messages: msgs,
			Tools:    cfg.definitions,
		})
		if err != nil {
			runErr = fmt.Errorf("agent: snapshot prompt: %w", err)
			break
		}
		stream := cfg.llm.Stream(ctx, cfg.provider, cfg.model, prompt, cfg.options)
		for ev := range stream.Events() {
			switch e := ev.(type) {
			case ai.StartEvent:
				em.publish(MessageStartEvent{Message: ai.CloneAssistantMessage(&e.Partial)})
			case ai.DoneEvent, ai.ErrorEvent:
				// The final message arrives via Result below.
			default:
				inner := cloneAssistantEvent(ev)
				em.publish(MessageUpdateEvent{
					Message: ai.CloneAssistantMessage(partialOf(inner)), Inner: inner,
				})
			}
		}
		final, err := stream.Result(ctx)
		if err != nil {
			// Stream mechanics failed (context canceled before completion);
			// there is no final message to record.
			runErr = err
			break
		}
		final = ai.CloneAssistantMessage(final)
		// Freeze executable calls before publishing the final message. Event
		// consumers must not be able to change what the permission hook or tool
		// handler will receive.
		calls, toolErr := validatedToolCalls(final)
		if final == nil {
			runErr = toolErr
			break
		}
		record(final)
		em.publish(MessageEndEvent{Message: ai.CloneAssistantMessage(final)})
		last = final

		var toolResults []ai.Message
		terminate := false
		if toolErr == nil && final.StopReason == ai.StopReasonToolUse {
			toolResults, terminate, runOutput, toolErr = a.executeTools(ctx, em, cfg, calls)
			for _, r := range toolResults {
				newMessages = append(newMessages, r)
			}
		}
		em.publish(TurnEndEvent{
			Turn: turn, Message: ai.CloneAssistantMessage(final), ToolResults: cloneMessages(toolResults),
		})
		if toolErr != nil {
			runErr = toolErr
			break
		}

		if final.StopReason == ai.StopReasonError || final.StopReason == ai.StopReasonAborted {
			break
		}
		if terminate {
			break
		}
		if a.shouldStopAfterTurn != nil &&
			a.shouldStopAfterTurn(ctx, ai.CloneAssistantMessage(final), a.Messages()) {
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
	em.publish(AgentEndEvent{NewMessages: cloneMessages(newMessages), Output: cloneRunOutput(runOutput)})
	// Result() must imply that the agent is ready for its next run. Retire
	// before Complete makes the result visible; the run identity protects a
	// new run from the deferred cleanup above.
	a.retireRun(runID)
	_ = producer.Complete(ctx, &RunResult{
		NewMessages: cloneMessages(newMessages),
		Last:        ai.CloneAssistantMessage(last),
		Output:      cloneRunOutput(runOutput),
		Err:         runErr,
	})
}

// toolOutcome pairs one tool call with its finalized result.
type toolOutcome struct {
	call      *ai.ToolCallContent
	msg       *ai.ToolResultMessage[any]
	terminate bool
	output    *RunOutput
	// pending marks outcomes whose permission checks passed and whose handler
	// has not yet been finalized.
	pending bool
}

// executeTools runs the tool calls of one assistant message. Permission
// checks are always sequential in source order; execution honors the batch
// mode; tool_execution_end fires per completion; toolResult messages are
// recorded in assistant source order. Cancellation is a hard start barrier:
// a call which has not reached Execute is finalized as canceled, and neither
// its permission hook nor handler is invoked. Already-running handlers still
// get their canceled context and are allowed to unwind normally.
func (a *Agent) executeTools(ctx context.Context, em *emitter, cfg runConfig,
	calls []*ai.ToolCallContent) ([]ai.Message, bool, *RunOutput, error) {

	if len(calls) == 0 {
		return nil, false, nil, fmt.Errorf("agent: tool execution requested without tool calls")
	}

	byName := map[string]Tool{}
	for i, t := range cfg.tools {
		byName[cfg.definitions[i].Name] = t
	}

	sequential := cfg.toolExecution != ExecParallel
	for _, c := range calls {
		if t, ok := byName[c.Name]; ok && isSequentialOnly(t) {
			sequential = true
		}
	}

	outcomes := make([]*toolOutcome, len(calls))
	for i, call := range calls {
		outcomes[i] = &toolOutcome{call: call}
	}

	stopError := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return em.failure()
	}
	finalizeSkipped := func(o *toolOutcome, err error) {
		if o.msg != nil {
			return
		}
		if err == nil {
			err = context.Canceled
		}
		o.msg = errorResult(o.call, fmt.Errorf("agent: tool call canceled before execution: %w", err))
		o.pending = false
	}
	preflight := func(o *toolOutcome) bool {
		if err := stopError(); err != nil {
			finalizeSkipped(o, err)
			return false
		}
		if !em.publish(ToolExecutionStartEvent{
			ToolCallID: o.call.Id, ToolName: o.call.Name, Args: cloneToolArguments(o.call.Arguments),
		}) {
			finalizeSkipped(o, stopError())
			return false
		}

		if _, known := byName[o.call.Name]; !known {
			o.msg = errorResult(o.call, fmt.Errorf("agent: unknown tool %q", o.call.Name))
			em.publish(ToolExecutionEndEvent{
				ToolCallID: o.call.Id, ToolName: o.call.Name,
				Result: cloneMessage(o.msg), IsError: true,
			})
			return false
		}
		if a.beforeToolCall != nil {
			// Cancellation can land after the start event or while the hook is
			// running. In either case it wins over permission success/failure and
			// prevents the handler (and every later hook) from starting.
			if err := stopError(); err != nil {
				finalizeSkipped(o, err)
				return false
			}
			call := cloneToolCallContent(o.call)
			err := a.beforeToolCall(ctx, *call)
			if stopErr := stopError(); stopErr != nil {
				finalizeSkipped(o, stopErr)
				return false
			}
			if err != nil {
				o.msg = errorResult(o.call, fmt.Errorf("agent: tool call blocked: %w", err))
				em.publish(ToolExecutionEndEvent{
					ToolCallID: o.call.Id, ToolName: o.call.Name,
					Result: cloneMessage(o.msg), IsError: true,
				})
				return false
			}
		}
		o.pending = true
		return true
	}

	execute := func(o *toolOutcome) {
		if err := stopError(); err != nil {
			finalizeSkipped(o, err)
			return
		}
		tool := byName[o.call.Name]
		onUpdate := func(u ToolUpdate) {
			// Updates racing with Abort are discarded. The handler has already
			// started and may still return its final result while unwinding.
			if ctx.Err() != nil {
				return
			}
			em.publish(ToolExecutionUpdateEvent{
				ToolCallID: o.call.Id, ToolName: o.call.Name, Update: cloneToolUpdate(u),
			})
		}
		if !a.claimToolStart(ctx) {
			finalizeSkipped(o, ctx.Err())
			return
		}
		o.pending = false
		out, err := tool.Execute(ctx, o.call.Id, cloneToolArguments(o.call.Arguments), onUpdate)
		if err != nil {
			o.msg = errorResult(o.call, err)
		} else {
			o.msg = successResult(o.call, out)
			o.terminate = out != nil && out.Terminate
			if out != nil && out.terminal != nil {
				o.output = &RunOutput{
					ToolCallID: o.call.Id,
					ToolName:   o.call.Name,
					Value:      cloneApplicationDetails(out.terminal.value),
				}
			}
		}
		em.publish(ToolExecutionEndEvent{
			ToolCallID: o.call.Id, ToolName: o.call.Name,
			Result: cloneMessage(o.msg), IsError: o.msg.IsError,
		})
	}

	if sequential {
		for _, o := range outcomes {
			preflight(o)
			if o.pending {
				execute(o)
			}
		}
	} else {
		// Parallel calls are permission-checked serially. Once cancellation is
		// observed, remaining hooks are skipped; accepted calls still re-check
		// immediately inside their goroutine before entering Execute.
		for _, o := range outcomes {
			preflight(o)
		}
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

	// Any calls after the cancellation boundary were deliberately untouched
	// by preflight. Give each one an error result without starting a hook or a
	// handler, keeping the persisted protocol history structurally complete.
	for _, o := range outcomes {
		if o.msg == nil {
			finalizeSkipped(o, stopError())
		}
	}

	// Record in assistant source order regardless of completion order.
	results := make([]ai.Message, 0, len(outcomes))
	terminate := true
	var outputs []*RunOutput
	for _, o := range outcomes {
		owned := a.appendMessage(o.msg)
		em.announce(owned)
		results = append(results, owned)
		if !o.terminate {
			terminate = false
		}
		if o.output != nil {
			outputs = append(outputs, o.output)
		}
	}
	if !terminate {
		return results, false, nil, nil
	}
	switch len(outputs) {
	case 0:
		// Preserve the existing control-only Terminate behavior.
		return results, true, nil, nil
	case 1:
		return results, true, outputs[0], nil
	default:
		return results, true, nil, fmt.Errorf(
			"agent: terminal tool batch produced %d outputs; exactly one is allowed", len(outputs),
		)
	}
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
		msg.Content = cloneToolResultContents(out.Content)
		if out.Details != nil {
			details := cloneApplicationDetails(out.Details)
			msg.Details = &details
		}
	}
	return msg
}

func definitions(tools []Tool) ([]ai.ToolDefinition, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	defs := make([]ai.ToolDefinition, 0, len(tools))
	for _, t := range tools {
		if source, ok := t.(interface{ definitionSnapshotError() error }); ok {
			if err := source.definitionSnapshotError(); err != nil {
				return nil, err
			}
		}
		definition := t.Definition().ToolDefinition
		defs = append(defs, ai.CloneToolDefinition(definition))
	}
	return defs, nil
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
	case ai.JSONStartEvent:
		p = e.Partial
	case ai.JSONDeltaEvent:
		p = e.Partial
	case ai.JSONEndEvent:
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
