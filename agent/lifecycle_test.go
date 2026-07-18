package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zhongys/Daksha/ai"
)

// afterFuncContext exposes whether a derived context detached itself from
// its parent. context.WithCancel uses AfterFunc for cancellation propagation
// and invokes the returned stop function when its CancelFunc is called.
type afterFuncContext struct {
	done       chan struct{}
	registered chan struct{}
	stopped    chan struct{}
	register   sync.Once
	stop       sync.Once
}

func newAfterFuncContext() *afterFuncContext {
	return &afterFuncContext{
		done:       make(chan struct{}),
		registered: make(chan struct{}),
		stopped:    make(chan struct{}),
	}
}

func (*afterFuncContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *afterFuncContext) Done() <-chan struct{}     { return c.done }
func (*afterFuncContext) Err() error                  { return nil }
func (*afterFuncContext) Value(any) any               { return nil }

func (c *afterFuncContext) AfterFunc(func()) func() bool {
	c.register.Do(func() { close(c.registered) })
	return func() bool {
		called := false
		c.stop.Do(func() {
			called = true
			close(c.stopped)
		})
		return called
	}
}

func TestRunResultImpliesAgentIdle(t *testing.T) {
	// The old ordering completed the stream and only cleared running from a
	// goroutine defer. Repetition makes that scheduler race reliably visible
	// while keeping this test entirely at the public API boundary.
	for i := 0; i < 250; i++ {
		llm := &fakeLLM{script: []*ai.AssistantMessage{assistantText("done")}}
		a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

		stream, err := a.PromptText(context.Background(), "go")
		if err != nil {
			t.Fatalf("iteration %d: Prompt: %v", i, err)
		}
		// The event buffer holds this single-turn script, so Result can be
		// observed directly without introducing a drain-dependent delay.
		res, err := stream.Result(context.Background())
		if err != nil || res.Err != nil {
			t.Fatalf("iteration %d: Result errors = %v / %v", i, err, res.Err)
		}
		if a.IsRunning() {
			t.Fatalf("iteration %d: Result returned while agent was still running", i)
		}
	}
}

func TestStaleRunCleanupCannotRetireNewRun(t *testing.T) {
	newRunCanceled := make(chan struct{})
	var cancelOnce sync.Once
	a := &Agent{
		running:     true,
		activeRunID: 2,
		nextRunID:   2,
		cancel: func() {
			cancelOnce.Do(func() { close(newRunCanceled) })
		},
	}

	// Model the first run's deferred cleanup arriving after run 2 has
	// claimed the slot. It must not clear run 2's state or CancelFunc.
	a.retireRun(1)
	if !a.IsRunning() {
		t.Fatal("stale cleanup retired the newer run")
	}
	a.Abort()
	select {
	case <-newRunCanceled:
	case <-time.After(time.Second):
		t.Fatal("stale cleanup cleared the newer run's CancelFunc")
	}
}

func TestContinueAndResetAreAtomic(t *testing.T) {
	type continueResult struct {
		stream *ai.EventStream[Event, *RunResult]
		err    error
	}

	for i := 0; i < 1000; i++ {
		llm := &fakeLLM{
			block:  make(chan struct{}),
			script: []*ai.AssistantMessage{assistantText("done")},
		}
		a := newAgent(t, Config{
			LLM: llm, Provider: "p", Model: "m",
			Messages: []ai.Message{UserText("retry me")},
		})

		start := make(chan struct{})
		continued := make(chan continueResult, 1)
		reset := make(chan error, 1)
		go func() {
			<-start
			stream, err := a.Continue(context.Background())
			continued <- continueResult{stream: stream, err: err}
		}()
		go func() {
			<-start
			reset <- a.Reset()
		}()
		close(start)

		cr, resetErr := <-continued, <-reset
		switch {
		case cr.err == nil:
			// Continue claimed the single-flight slot while validating the
			// last message, so Reset must have observed that active run.
			if !errors.Is(resetErr, ErrBusy) {
				t.Fatalf("iteration %d: Continue and Reset both succeeded (Reset err %v)", i, resetErr)
			}
			a.Abort()
			for range cr.stream.Events() {
			}
			_, _ = cr.stream.Result(context.Background())
		case resetErr == nil:
			// Reset won the same lock and cleared the context before Continue
			// validated it. Continue must reject the now-empty context.
			if !strings.Contains(cr.err.Error(), "empty context") {
				t.Fatalf("iteration %d: Continue error = %v, want empty context", i, cr.err)
			}
		default:
			t.Fatalf("iteration %d: Continue err = %v, Reset err = %v", i, cr.err, resetErr)
		}
	}
}

func TestCompletedRunCancelsDerivedContext(t *testing.T) {
	parent := newAfterFuncContext()
	llm := &fakeLLM{script: []*ai.AssistantMessage{assistantText("done")}}
	a := newAgent(t, Config{LLM: llm, Provider: "p", Model: "m"})

	stream, err := a.PromptText(parent, "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	select {
	case <-parent.registered:
	case <-time.After(time.Second):
		t.Fatal("run context did not register cancellation with parent")
	}

	res, err := stream.Result(context.Background())
	if err != nil || res.Err != nil {
		t.Fatalf("Result errors = %v / %v", err, res.Err)
	}
	select {
	case <-parent.stopped:
	case <-time.After(time.Second):
		t.Fatal("completed run did not call its context CancelFunc")
	}
}
