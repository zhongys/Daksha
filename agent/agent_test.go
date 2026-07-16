package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zhongys/Daksha/ai"
)

// fakeLLM replays a script of assistant messages, one per Stream call, and
// captures each prompt for assertions.
type fakeLLM struct {
	mu      sync.Mutex
	prompts []ai.Prompt
	script  []*ai.AssistantMessage
	block   chan struct{} // when set, responses wait for it (or ctx)
}

type jsonEventLLM struct {
	raw json.RawMessage
}

func (f *jsonEventLLM) Stream(ctx context.Context, _, _ string, _ ai.Prompt,
	_ ai.StreamOptions) *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage] {
	stream, producer := ai.NewEventStream[ai.AssistantMessageEvent, *ai.AssistantMessage](ctx, 8)
	go func() {
		placeholder := ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{&ai.JSONContent{
				Type: ai.ContentTypeJSON, SchemaName: "answer",
			}},
		}
		value := append(json.RawMessage(nil), f.raw...)
		final := &ai.AssistantMessage{
			Role: ai.RoleAssistant,
			Content: []ai.AssistantContent{&ai.JSONContent{
				Type: ai.ContentTypeJSON, SchemaName: "answer", Value: value,
			}},
			StopReason: ai.StopReasonStop,
		}
		_ = producer.Publish(ctx, ai.StartEvent{Partial: ai.AssistantMessage{Role: ai.RoleAssistant}})
		_ = producer.Publish(ctx, ai.JSONStartEvent{ContentIndex: 0, Partial: placeholder})
		_ = producer.Publish(ctx, ai.JSONDeltaEvent{ContentIndex: 0, Delta: string(f.raw), Partial: placeholder})
		_ = producer.Publish(ctx, ai.JSONEndEvent{
			ContentIndex: 0, Content: append(json.RawMessage(nil), f.raw...), Partial: *final,
		})
		_ = producer.Publish(ctx, ai.DoneEvent{Reason: ai.StopReasonStop, Message: *final})
		_ = producer.Complete(ctx, final)
	}()
	return stream
}

func (f *fakeLLM) Stream(ctx context.Context, provider, model string, prompt ai.Prompt,
	opts ai.StreamOptions) *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage] {

	f.mu.Lock()
	idx := len(f.prompts)
	f.prompts = append(f.prompts, prompt)
	var msg *ai.AssistantMessage
	if idx < len(f.script) {
		msg = f.script[idx]
	} else {
		msg = assistantText("out of script")
	}
	block := f.block
	f.mu.Unlock()

	stream, producer := ai.NewEventStream[ai.AssistantMessageEvent, *ai.AssistantMessage](ctx, 8)
	go func() {
		if block != nil {
			select {
			case <-block:
			case <-ctx.Done():
				return
			}
		}
		_ = producer.Publish(ctx, ai.StartEvent{Partial: *msg})
		_ = producer.Publish(ctx, ai.TextDeltaEvent{Delta: "x", Partial: *msg})
		_ = producer.Publish(ctx, ai.DoneEvent{Reason: msg.StopReason, Message: *msg})
		_ = producer.Complete(ctx, msg)
	}()
	return stream
}

func (f *fakeLLM) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.prompts)
}

func (f *fakeLLM) promptAt(i int) ai.Prompt {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.prompts[i]
}

func assistantText(text string) *ai.AssistantMessage {
	return &ai.AssistantMessage{
		Role:       ai.RoleAssistant,
		Content:    []ai.AssistantContent{&ai.TextContent{Type: ai.ContentTypeText, Text: text}},
		StopReason: ai.StopReasonStop,
	}
}

func assistantToolCalls(calls ...*ai.ToolCallContent) *ai.AssistantMessage {
	msg := &ai.AssistantMessage{Role: ai.RoleAssistant, StopReason: ai.StopReasonToolUse}
	for _, c := range calls {
		msg.Content = append(msg.Content, c)
	}
	return msg
}

func toolCall(id, name string, args map[string]any) *ai.ToolCallContent {
	return &ai.ToolCallContent{Type: ai.ContentTypeToolCall, Id: id, Name: name, Arguments: args}
}

type echoParams struct {
	Text string `json:"text"`
}

func echoTool() Tool {
	return NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "echo"}},
		func(ctx context.Context, id string, p echoParams, onUpdate func(ToolUpdate)) (*ToolOutput, error) {
			return &ToolOutput{
				Content: []ai.ToolResultContent{&ai.TextContent{Type: ai.ContentTypeText, Text: p.Text}},
			}, nil
		})
}

func drainRun(t *testing.T, stream *ai.EventStream[Event, *RunResult]) ([]Event, *RunResult, error) {
	t.Helper()
	var events []Event
	for ev := range stream.Events() {
		events = append(events, ev)
	}
	res, err := stream.Result(context.Background())
	return events, res, err
}

func types(events []Event) []EventType {
	out := make([]EventType, 0, len(events))
	for _, e := range events {
		out = append(out, e.EventType())
	}
	return out
}

func newAgent(t *testing.T, cfg Config) *Agent {
	t.Helper()
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func TestPromptTextOnlyTurn(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{assistantText("hello")}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m", SystemPrompt: "be nice"})

	stream, err := a.PromptText(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	events, res, err := drainRun(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res.Err != nil {
		t.Fatalf("RunResult.Err = %v", res.Err)
	}

	want := []EventType{
		EventAgentStart,
		EventMessageStart, EventMessageEnd, // user
		EventTurnStart,
		EventMessageStart, EventMessageUpdate, // assistant streaming
		EventMessageEnd,
		EventTurnEnd,
		EventAgentEnd,
	}
	if got := types(events); !equalTypes(got, want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	if len(res.NewMessages) != 2 {
		t.Fatalf("NewMessages = %d, want 2", len(res.NewMessages))
	}
	if res.Last == nil || res.Last.StopReason != ai.StopReasonStop {
		t.Fatalf("Last = %+v", res.Last)
	}
	if p := llm.promptAt(0); p.System != "be nice" || len(p.Messages) != 1 {
		t.Fatalf("prompt = %+v", p)
	}
	if a.IsRunning() {
		t.Fatal("agent should be idle")
	}
	if len(a.Messages()) != 2 {
		t.Fatalf("context = %d messages", len(a.Messages()))
	}
}

func TestAgentForwardsJSONEventsWithPartials(t *testing.T) {
	llm := &jsonEventLLM{raw: json.RawMessage(`{"answer":"ok"}`)}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

	stream, err := a.PromptText(context.Background(), "structured please")
	if err != nil {
		t.Fatalf("PromptText: %v", err)
	}
	events, res, err := drainRun(t, stream)
	if err != nil || res.Err != nil {
		t.Fatalf("err=%v runErr=%v", err, res.Err)
	}

	var updates []MessageUpdateEvent
	for _, event := range events {
		if update, ok := event.(MessageUpdateEvent); ok {
			updates = append(updates, update)
		}
	}
	if len(updates) != 3 {
		t.Fatalf("message updates = %d, want 3", len(updates))
	}
	want := []ai.AssistantMessageEventType{
		ai.AssistantEventJSONStart, ai.AssistantEventJSONDelta, ai.AssistantEventJSONEnd,
	}
	for i, update := range updates {
		if update.Message == nil || update.Inner.EventType() != want[i] {
			t.Fatalf("update[%d] = %#v, want %s with non-nil partial", i, update, want[i])
		}
	}
	structured, ok := res.Last.Content[0].(*ai.JSONContent)
	if !ok || string(structured.Value) != `{"answer":"ok"}` {
		t.Fatalf("final content = %#v", res.Last.Content)
	}
}

func equalTypes(got, want []EventType) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestToolLoop(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"text": "pong"})),
		assistantText("done"),
	}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m", Tools: []Tool{echoTool()}})

	stream, _ := a.PromptText(context.Background(), "ping")
	events, res, err := drainRun(t, stream)
	if err != nil || res.Err != nil {
		t.Fatalf("err=%v runErr=%v", err, res != nil)
	}

	if llm.callCount() != 2 {
		t.Fatalf("LLM calls = %d, want 2", llm.callCount())
	}
	// user, assistant(toolUse), toolResult, assistant(stop)
	if len(res.NewMessages) != 4 {
		t.Fatalf("NewMessages = %d, want 4", len(res.NewMessages))
	}
	tr, ok := res.NewMessages[2].(*ai.ToolResultMessage[any])
	if !ok || tr.IsError || tr.ToolCallId != "c1" {
		t.Fatalf("tool result = %+v", res.NewMessages[2])
	}
	if text := tr.Content[0].(*ai.TextContent).Text; text != "pong" {
		t.Fatalf("tool echoed %q", text)
	}
	// Second LLM call must see the tool result.
	second := llm.promptAt(1)
	if len(second.Messages) != 3 {
		t.Fatalf("second prompt has %d messages", len(second.Messages))
	}
	// Tool execution events present and ordered around the result message.
	var sawStart, sawEnd bool
	for _, e := range events {
		switch e.EventType() {
		case EventToolExecutionStart:
			sawStart = true
		case EventToolExecutionEnd:
			sawEnd = true
		}
	}
	if !sawStart || !sawEnd {
		t.Fatalf("missing tool execution events: %v", types(events))
	}
}

func TestNewToolDecodeErrorReturnsToModel(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"text": 123})), // wrong type
		assistantText("recovered"),
	}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m", Tools: []Tool{echoTool()}})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)

	tr := res.NewMessages[2].(*ai.ToolResultMessage[any])
	if !tr.IsError || !strings.Contains(tr.Content[0].(*ai.TextContent).Text, "invalid arguments") {
		t.Fatalf("decode failure should be an IsError result: %+v", tr)
	}
	if llm.callCount() != 2 {
		t.Fatal("loop should continue after a tool arg error")
	}
}

func TestUnknownToolIsError(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "nope", nil)),
		assistantText("ok"),
	}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	tr := res.NewMessages[2].(*ai.ToolResultMessage[any])
	if !tr.IsError || !strings.Contains(tr.Content[0].(*ai.TextContent).Text, "unknown tool") {
		t.Fatalf("tool result = %+v", tr)
	}
}

func TestParallelExecutionOrdering(t *testing.T) {
	bDone := make(chan struct{})
	releaseA := make(chan struct{})
	slowA := NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "a"}},
		func(ctx context.Context, id string, _ struct{}, _ func(ToolUpdate)) (*ToolOutput, error) {
			select {
			case <-bDone: // finish only after b — forces reverse completion order
			case <-time.After(5 * time.Second):
				return nil, errors.New("deadlock: batch did not run in parallel")
			}
			// Tool B closes bDone from a defer just before Execute returns. Wait
			// until B's end event is actually observed so scheduler timing cannot
			// let A publish its end event first.
			select {
			case <-releaseA:
			case <-time.After(5 * time.Second):
				return nil, errors.New("deadlock: b completion event was not consumed")
			}
			return &ToolOutput{}, nil
		})
	fastB := NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "b"}},
		func(ctx context.Context, id string, _ struct{}, _ func(ToolUpdate)) (*ToolOutput, error) {
			defer close(bDone)
			return &ToolOutput{}, nil
		})

	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("ca", "a", nil), toolCall("cb", "b", nil)),
		assistantText("done"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m",
		Tools: []Tool{slowA, fastB}, ToolExecution: ExecParallel,
	})

	stream, _ := a.PromptText(context.Background(), "go")
	var events []Event
	for event := range stream.Events() {
		events = append(events, event)
		if end, ok := event.(ToolExecutionEndEvent); ok && end.ToolName == "b" {
			close(releaseA)
		}
	}
	res, err := stream.Result(context.Background())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res.Err != nil {
		t.Fatalf("run failed: %v", res.Err)
	}

	// End events follow completion order: b before a.
	var endOrder []string
	for _, e := range events {
		if end, ok := e.(ToolExecutionEndEvent); ok {
			endOrder = append(endOrder, end.ToolName)
		}
	}
	if len(endOrder) != 2 || endOrder[0] != "b" || endOrder[1] != "a" {
		t.Fatalf("end order = %v, want [b a]", endOrder)
	}
	// Recorded messages follow assistant source order: a before b.
	first := res.NewMessages[2].(*ai.ToolResultMessage[any])
	second := res.NewMessages[3].(*ai.ToolResultMessage[any])
	if first.ToolName != "a" || second.ToolName != "b" {
		t.Fatalf("message order = %s, %s; want a, b", first.ToolName, second.ToolName)
	}
}

func TestSequentialWrapperForcesBatchSequential(t *testing.T) {
	var order []string
	var mu sync.Mutex
	mk := func(name string) Tool {
		return NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: name}},
			func(ctx context.Context, id string, _ struct{}, _ func(ToolUpdate)) (*ToolOutput, error) {
				mu.Lock()
				order = append(order, name)
				mu.Unlock()
				return &ToolOutput{}, nil
			})
	}
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "s", nil), toolCall("c2", "p", nil)),
		assistantText("done"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m",
		Tools: []Tool{Sequential(mk("s")), mk("p")}, ToolExecution: ExecParallel,
	})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("run failed: %v", res.Err)
	}
	if len(order) != 2 || order[0] != "s" || order[1] != "p" {
		t.Fatalf("execution order = %v, want [s p] (sequential source order)", order)
	}
}

func TestBeforeToolCallBlocks(t *testing.T) {
	executed := false
	tool := NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "danger"}},
		func(ctx context.Context, id string, _ struct{}, _ func(ToolUpdate)) (*ToolOutput, error) {
			executed = true
			return &ToolOutput{}, nil
		})
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "danger", nil)),
		assistantText("ok"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{tool},
		BeforeToolCall: func(ctx context.Context, call ai.ToolCallContent) error {
			return fmt.Errorf("tenant policy forbids %s", call.Name)
		},
	})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if executed {
		t.Fatal("blocked tool must not execute")
	}
	tr := res.NewMessages[2].(*ai.ToolResultMessage[any])
	if !tr.IsError || !strings.Contains(tr.Content[0].(*ai.TextContent).Text, "tenant policy") {
		t.Fatalf("tool result = %+v", tr)
	}
}

func TestShouldStopAfterTurn(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"text": "x"})),
		assistantText("never reached"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{echoTool()},
		ShouldStopAfterTurn: func(ctx context.Context, last *ai.AssistantMessage, msgs []ai.Message) bool {
			return true // e.g. cost circuit breaker tripped
		},
	})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("graceful stop is not an error: %v", res.Err)
	}
	if llm.callCount() != 1 {
		t.Fatalf("LLM calls = %d, want 1 (stopped after first turn)", llm.callCount())
	}
}

func TestMaxTurnsExceeded(t *testing.T) {
	loopCall := assistantToolCalls(toolCall("c1", "echo", map[string]any{"text": "again"}))
	llm := &fakeLLM{script: []*ai.AssistantMessage{loopCall, loopCall, loopCall, loopCall}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{echoTool()}, MaxTurns: 2,
	})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err == nil || !strings.Contains(res.Err.Error(), "max turns") {
		t.Fatalf("RunResult.Err = %v, want max turns error", res.Err)
	}
	if llm.callCount() != 2 {
		t.Fatalf("LLM calls = %d, want exactly MaxTurns", llm.callCount())
	}
}

func TestTerminateSkipsFollowUpCall(t *testing.T) {
	done := NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "finish"}},
		func(ctx context.Context, id string, _ struct{}, _ func(ToolUpdate)) (*ToolOutput, error) {
			return &ToolOutput{Terminate: true}, nil
		})
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "finish", nil)),
		assistantText("never reached"),
	}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m", Tools: []Tool{done}})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("terminate is not an error: %v", res.Err)
	}
	if llm.callCount() != 1 {
		t.Fatalf("LLM calls = %d, want 1", llm.callCount())
	}
	if res.Output != nil {
		t.Fatalf("legacy Terminate tool produced run output: %#v", res.Output)
	}
}

type finalAnswer struct {
	Answer string `json:"answer"`
	ID     int64  `json:"id"`
}

func terminalAnswerTool() Tool {
	return NewTerminalTool[finalAnswer](ToolDefinition{ToolDefinition: ai.ToolDefinition{
		Name:        "final_answer",
		Description: "Submit the final structured answer",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
				"id":     map[string]any{"type": "integer"},
			},
			"required": []string{"answer", "id"},
		},
	}})
}

func TestTerminalToolReturnsTypedRunOutput(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "final_answer", map[string]any{
			"answer": "done", "id": json.Number("9007199254740993"),
		})),
		assistantText("never reached"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{terminalAnswerTool()},
	})

	stream, _ := a.PromptText(context.Background(), "finish")
	events, res, err := drainRun(t, stream)
	if err != nil || res.Err != nil {
		t.Fatalf("err=%v runErr=%v", err, res.Err)
	}
	if llm.callCount() != 1 {
		t.Fatalf("LLM calls = %d, want 1", llm.callCount())
	}
	if res.Output == nil || res.Output.ToolCallID != "c1" || res.Output.ToolName != "final_answer" {
		t.Fatalf("RunResult.Output = %#v", res.Output)
	}
	value, ok := RunOutputAs[finalAnswer](res.Output)
	if !ok || value.Answer != "done" || value.ID != 9_007_199_254_740_993 {
		t.Fatalf("typed terminal value = %#v, ok=%v", value, ok)
	}
	result := res.NewMessages[2].(*ai.ToolResultMessage[any])
	if result.Details == nil {
		t.Fatal("terminal tool result did not persist details")
	}
	details, ok := (*result.Details).(finalAnswer)
	if !ok || details != value {
		t.Fatalf("tool result details = %#v, want %#v", result.Details, value)
	}
	var endOutput *RunOutput
	for _, event := range events {
		if end, ok := event.(AgentEndEvent); ok {
			endOutput = end.Output
		}
	}
	if endOutput != res.Output {
		t.Fatalf("AgentEnd output = %#v, result output = %#v", endOutput, res.Output)
	}
}

func TestTerminalToolOutputIgnoredWhenBatchContinues(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(
			toolCall("c1", "final_answer", map[string]any{"answer": "early", "id": 1}),
			toolCall("c2", "echo", map[string]any{"text": "work"}),
		),
		assistantText("done after tools"),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{terminalAnswerTool(), echoTool()},
	})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("mixed batch failed: %v", res.Err)
	}
	if llm.callCount() != 2 {
		t.Fatalf("LLM calls = %d, want follow-up after mixed batch", llm.callCount())
	}
	if res.Output != nil {
		t.Fatalf("non-terminating batch leaked terminal output: %#v", res.Output)
	}
}

func TestMultipleTerminalOutputsFailDeterministically(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(
			toolCall("c1", "final_answer", map[string]any{"answer": "one", "id": 1}),
			toolCall("c2", "final_answer", map[string]any{"answer": "two", "id": 2}),
		),
	}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m", Tools: []Tool{terminalAnswerTool()},
	})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err == nil || !strings.Contains(res.Err.Error(), "exactly one is allowed") {
		t.Fatalf("RunResult.Err = %v", res.Err)
	}
	if res.Output != nil {
		t.Fatalf("ambiguous terminal batch produced output: %#v", res.Output)
	}
	if llm.callCount() != 1 {
		t.Fatalf("LLM calls = %d, want 1", llm.callCount())
	}
}

func TestSteeringInjectedBetweenTurns(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantText("first"),
		assistantText("second"),
	}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})
	_ = a.Steer(UserText("actually, do this instead"))

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("run failed: %v", res.Err)
	}
	if llm.callCount() != 2 {
		t.Fatalf("LLM calls = %d, want 2 (steering forces another turn)", llm.callCount())
	}
	second := llm.promptAt(1)
	lastMsg := second.Messages[len(second.Messages)-1].(*ai.UserMessage)
	if text := lastMsg.Content[0].(*ai.TextContent).Text; !strings.Contains(text, "instead") {
		t.Fatalf("steering message missing from second prompt: %q", text)
	}
}

func TestFollowUpRunsAnotherTurn(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantText("first"),
		assistantText("second"),
	}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})
	_ = a.FollowUp(UserText("also summarize"))

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("run failed: %v", res.Err)
	}
	if llm.callCount() != 2 {
		t.Fatalf("LLM calls = %d, want 2", llm.callCount())
	}
	// user, assistant, followup-user, assistant
	if len(res.NewMessages) != 4 {
		t.Fatalf("NewMessages = %d, want 4", len(res.NewMessages))
	}
}

func TestErrBusyAndAbort(t *testing.T) {
	llm := &fakeLLM{block: make(chan struct{}), script: []*ai.AssistantMessage{assistantText("hi")}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

	stream, err := a.PromptText(context.Background(), "go")
	if err != nil {
		t.Fatalf("first Prompt: %v", err)
	}
	if _, err := a.PromptText(context.Background(), "again"); !errors.Is(err, ErrBusy) {
		t.Fatalf("second Prompt err = %v, want ErrBusy", err)
	}
	if err := a.Reset(); !errors.Is(err, ErrBusy) {
		t.Fatalf("Reset during run err = %v, want ErrBusy", err)
	}

	a.Abort()
	_, _, err = drainRun(t, stream)
	if err == nil {
		t.Fatal("aborted run should surface a stream error")
	}
	for a.IsRunning() {
		time.Sleep(time.Millisecond)
	}
	if _, err := a.PromptText(context.Background(), "next"); err != nil {
		t.Fatalf("agent must accept prompts after abort: %v", err)
	}
}

func TestContinueValidation(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{assistantText("hi")}}

	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})
	if _, err := a.Continue(context.Background()); err == nil {
		t.Fatal("Continue on empty context should fail")
	}

	a = newAgent(t, Config{LLM: llm, Provider: "p", Model: "m",
		Messages: []ai.Message{UserText("hello"), assistantText("hi")}})
	if _, err := a.Continue(context.Background()); err == nil {
		t.Fatal("Continue after assistant message should fail")
	}

	a = newAgent(t, Config{LLM: llm, Provider: "p", Model: "m",
		Messages: []ai.Message{UserText("hello")}})
	stream, err := a.Continue(context.Background())
	if err != nil {
		t.Fatalf("Continue: %v", err)
	}
	_, res, err := drainRun(t, stream)
	if err != nil || res.Err != nil {
		t.Fatalf("continue run failed: %v / %v", err, res.Err)
	}
	if len(res.NewMessages) != 1 { // only the assistant reply
		t.Fatalf("NewMessages = %d, want 1", len(res.NewMessages))
	}
}

func TestModelErrorEndsRun(t *testing.T) {
	failed := &ai.AssistantMessage{Role: ai.RoleAssistant, StopReason: ai.StopReasonError, ErrorMessage: "boom"}
	llm := &fakeLLM{script: []*ai.AssistantMessage{failed}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

	stream, _ := a.PromptText(context.Background(), "go")
	_, res, err := drainRun(t, stream)
	if err != nil {
		t.Fatalf("stream err = %v", err)
	}
	if res.Err != nil {
		t.Fatalf("model errors ride on the message, not RunResult.Err: %v", res.Err)
	}
	if res.Last.StopReason != ai.StopReasonError || res.Last.ErrorMessage != "boom" {
		t.Fatalf("Last = %+v", res.Last)
	}
	if llm.callCount() != 1 {
		t.Fatal("error must stop the loop")
	}
}

func TestTransformContext(t *testing.T) {
	llm := &fakeLLM{script: []*ai.AssistantMessage{assistantText("hi")}}
	a := newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m",
		Messages: []ai.Message{UserText("old1"), UserText("old2"), UserText("old3")},
		TransformContext: func(ctx context.Context, msgs []ai.Message) ([]ai.Message, error) {
			return msgs[len(msgs)-2:], nil // keep the last two
		},
	})

	stream, _ := a.PromptText(context.Background(), "new")
	_, res, _ := drainRun(t, stream)
	if res.Err != nil {
		t.Fatalf("run failed: %v", res.Err)
	}
	if got := len(llm.promptAt(0).Messages); got != 2 {
		t.Fatalf("LLM saw %d messages, want 2", got)
	}
	// The transform only shapes what the LLM sees; the context keeps everything.
	if got := len(a.Messages()); got != 5 {
		t.Fatalf("context = %d messages, want 5", got)
	}
}
