package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zhongys/Daksha/ai"
)

func waitForAgentIdle(t *testing.T, a *Agent) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for a.IsRunning() {
		if time.Now().After(deadline) {
			t.Fatal("agent did not finish unwinding after cancellation")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestCancellationStopsSequentialBatchBeforeNextTool(t *testing.T) {
	for _, test := range []struct {
		name  string
		abort bool
	}{
		{name: "Agent.Abort", abort: true},
		{name: "parent context", abort: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			firstEntered := make(chan struct{})
			firstReturned := make(chan struct{})
			var secondExecutions atomic.Int32

			first := NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "first"}},
				func(ctx context.Context, _ string, _ struct{}, onUpdate func(ToolUpdate)) (*ToolOutput, error) {
					close(firstEntered)
					<-ctx.Done()
					// A handler is allowed to finish unwinding, but progress emitted
					// after the hard-stop boundary must not escape the agent.
					onUpdate(ToolUpdate{Content: []ai.ToolResultContent{
						&ai.TextContent{Type: ai.ContentTypeText, Text: "too late"},
					}})
					close(firstReturned)
					return nil, ctx.Err()
				})
			second := NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "second"}},
				func(context.Context, string, struct{}, func(ToolUpdate)) (*ToolOutput, error) {
					secondExecutions.Add(1)
					return &ToolOutput{}, nil
				})

			var beforeMu sync.Mutex
			var before []string
			llm := &fakeLLM{script: []*ai.AssistantMessage{
				assistantToolCalls(toolCall("c1", "first", nil), toolCall("c2", "second", nil)),
				assistantText("must not be reached"),
			}}
			a := newAgent(t, Config{
				LLM: llm, Provider: "p", Model: "m", Tools: []Tool{first, second},
				ToolExecution: ExecSequential,
				BeforeToolCall: func(_ context.Context, call ai.ToolCallContent) error {
					beforeMu.Lock()
					before = append(before, call.Name)
					beforeMu.Unlock()
					return nil
				},
			})

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			stream, err := a.PromptText(ctx, "go")
			if err != nil {
				t.Fatalf("PromptText: %v", err)
			}
			select {
			case <-firstEntered:
			case <-time.After(2 * time.Second):
				t.Fatal("first tool did not start")
			}
			if test.abort {
				a.Abort()
			} else {
				cancel()
			}
			select {
			case <-firstReturned:
			case <-time.After(2 * time.Second):
				t.Fatal("running tool was not allowed to unwind")
			}

			var events []Event
			for event := range stream.Events() {
				events = append(events, event)
			}
			if _, err := stream.Result(context.Background()); !errors.Is(err, context.Canceled) {
				t.Fatalf("Result error = %v, want context.Canceled", err)
			}
			waitForAgentIdle(t, a)

			if got := secondExecutions.Load(); got != 0 {
				t.Fatalf("second tool executed %d times after cancellation", got)
			}
			beforeMu.Lock()
			gotBefore := append([]string(nil), before...)
			beforeMu.Unlock()
			if len(gotBefore) != 1 || gotBefore[0] != "first" {
				t.Fatalf("BeforeToolCall calls = %v, want only [first]", gotBefore)
			}
			for _, event := range events {
				if update, ok := event.(ToolExecutionUpdateEvent); ok && update.ToolName == "first" {
					t.Fatal("tool update published after cancellation")
				}
			}
			if llm.callCount() != 1 {
				t.Fatalf("LLM calls = %d, want 1", llm.callCount())
			}

			messages := a.Messages()
			if len(messages) != 4 {
				t.Fatalf("stored messages = %d, want user + assistant + two tool results", len(messages))
			}
			skipped := messages[3].(*ai.ToolResultMessage[any])
			if !skipped.IsError || !strings.Contains(
				skipped.Content[0].(*ai.TextContent).Text, "canceled before execution",
			) {
				t.Fatalf("skipped tool result = %+v", skipped)
			}
		})
	}
}

func TestAbortInsideBeforeToolCallStopsParallelBatch(t *testing.T) {
	var executions atomic.Int32
	mkTool := func(name string) Tool {
		return NewTool(ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: name}},
			func(context.Context, string, struct{}, func(ToolUpdate)) (*ToolOutput, error) {
				executions.Add(1)
				return &ToolOutput{}, nil
			})
	}

	llm := &fakeLLM{script: []*ai.AssistantMessage{
		assistantToolCalls(toolCall("c1", "first", nil), toolCall("c2", "second", nil)),
	}}
	var a *Agent
	var beforeMu sync.Mutex
	var before []string
	a = newAgent(t, Config{
		LLM: llm, Provider: "p", Model: "m",
		Tools: []Tool{mkTool("first"), mkTool("second")}, ToolExecution: ExecParallel,
		BeforeToolCall: func(_ context.Context, call ai.ToolCallContent) error {
			beforeMu.Lock()
			before = append(before, call.Name)
			beforeMu.Unlock()
			a.Abort()
			return nil
		},
	})

	stream, err := a.PromptText(context.Background(), "go")
	if err != nil {
		t.Fatalf("PromptText: %v", err)
	}
	for range stream.Events() {
	}
	if _, err := stream.Result(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Result error = %v, want context.Canceled", err)
	}
	waitForAgentIdle(t, a)

	if got := executions.Load(); got != 0 {
		t.Fatalf("handlers executed %d times after Abort in BeforeToolCall", got)
	}
	beforeMu.Lock()
	gotBefore := append([]string(nil), before...)
	beforeMu.Unlock()
	if len(gotBefore) != 1 || gotBefore[0] != "first" {
		t.Fatalf("BeforeToolCall calls = %v, want only [first]", gotBefore)
	}
}
