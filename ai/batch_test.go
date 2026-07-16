package ai

import (
	"context"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCompleteBatchWorkerCountExtremeValues(t *testing.T) {
	tests := []struct {
		requests       int
		maxConcurrency int
		want           int
	}{
		{requests: 0, maxConcurrency: math.MaxInt, want: 0},
		{requests: 5, maxConcurrency: math.MaxInt, want: 5},
		{requests: math.MaxInt, maxConcurrency: 0, want: 1},
		{requests: math.MaxInt, maxConcurrency: 7, want: 7},
		{requests: math.MaxInt, maxConcurrency: math.MaxInt, want: maxCompleteBatchWorkers},
	}
	for _, tt := range tests {
		if got := completeBatchWorkerCount(tt.requests, tt.maxConcurrency); got != tt.want {
			t.Errorf("completeBatchWorkerCount(%d, %d) = %d, want %d",
				tt.requests, tt.maxConcurrency, got, tt.want)
		}
	}
}

// countingStreamer tracks concurrent executions and echoes the model id.
type countingStreamer struct {
	mu      sync.Mutex
	active  int
	peak    int
	total   atomic.Int64
	release chan struct{} // when set, streams block until closed
}

func (s *countingStreamer) Stream(ctx context.Context, provider Provider, model Model, prompt Prompt, opts StreamOptions,
) *EventStream[AssistantMessageEvent, *AssistantMessage] {
	stream, producer := NewEventStream[AssistantMessageEvent, *AssistantMessage](ctx, 2)
	go func() {
		s.mu.Lock()
		s.active++
		if s.active > s.peak {
			s.peak = s.active
		}
		s.mu.Unlock()
		if s.release != nil {
			<-s.release
		}
		s.total.Add(1)
		s.mu.Lock()
		s.active--
		s.mu.Unlock()

		msg := &AssistantMessage{Role: RoleAssistant, Model: model.ID, StopReason: StopReasonStop}
		_ = producer.Publish(ctx, DoneEvent{Reason: StopReasonStop, Message: *msg})
		_ = producer.Complete(ctx, msg)
	}()
	return stream
}

func batchClient(s Streamer) *Client {
	c := NewClient(map[string]Streamer{"openai-completions": s})
	_ = c.PutProvider(Provider{Name: "acme", APIKey: "sk"})
	for _, id := range []string{"m0", "m1", "m2", "m3", "m4", "m5"} {
		_ = c.PutModel(Model{Provider: "acme", ID: id})
	}
	return c
}

func TestCompleteBatchIndexAlignment(t *testing.T) {
	c := batchClient(&countingStreamer{})

	reqs := make([]BatchRequest, 6)
	for i := range reqs {
		reqs[i] = BatchRequest{Provider: "acme", Model: "m" + string(rune('0'+i))}
	}
	results := c.CompleteBatch(context.Background(), reqs, 3)

	if len(results) != len(reqs) {
		t.Fatalf("results = %d, want %d", len(results), len(reqs))
	}
	for i, r := range results {
		if r.Err != nil {
			t.Fatalf("results[%d].Err = %v", i, r.Err)
		}
		if want := reqs[i].Model; r.Message.Model != want {
			t.Errorf("results[%d].Model = %q, want %q", i, r.Message.Model, want)
		}
	}
}

func TestCompleteBatchBoundsConcurrency(t *testing.T) {
	s := &countingStreamer{release: make(chan struct{})}
	c := batchClient(s)

	reqs := make([]BatchRequest, 6)
	for i := range reqs {
		reqs[i] = BatchRequest{Provider: "acme", Model: "m0"}
	}

	done := make(chan []BatchResult)
	go func() { done <- c.CompleteBatch(context.Background(), reqs, 2) }()

	// Allow the first wave to start, then release everyone.
	for s.total.Load() == 0 {
		s.mu.Lock()
		started := s.active
		s.mu.Unlock()
		if started >= 2 {
			break
		}
	}
	close(s.release)
	<-done

	s.mu.Lock()
	peak := s.peak
	s.mu.Unlock()
	if peak > 2 {
		t.Fatalf("peak concurrency = %d, want <= 2", peak)
	}
	if got := s.total.Load(); got != 6 {
		t.Fatalf("completed = %d, want 6", got)
	}
}

func TestCompleteBatchIsolatesFailures(t *testing.T) {
	c := batchClient(&countingStreamer{})

	reqs := []BatchRequest{
		{Provider: "acme", Model: "m0"},
		{Provider: "acme", Model: "missing"}, // unknown model → error message
		{Provider: "acme", Model: "m1"},
	}
	results := c.CompleteBatch(context.Background(), reqs, 2)

	if results[0].Err != nil || results[2].Err != nil {
		t.Fatalf("healthy requests failed: %v / %v", results[0].Err, results[2].Err)
	}
	bad := results[1]
	if bad.Err != nil {
		t.Fatalf("lookup failures ride in the message, got Err %v", bad.Err)
	}
	if bad.Message.StopReason != StopReasonError || !strings.Contains(bad.Message.ErrorMessage, "unknown model") {
		t.Fatalf("results[1] = %+v", bad.Message)
	}
}

func TestCompleteBatchLargeInputUsesWorkerPool(t *testing.T) {
	streamer := &countingStreamer{}
	c := batchClient(streamer)
	const requestCount = 10_000
	reqs := make([]BatchRequest, requestCount)
	for i := range reqs {
		reqs[i] = BatchRequest{Provider: "acme", Model: "m0"}
	}

	results := c.CompleteBatch(context.Background(), reqs, 3)
	if len(results) != requestCount {
		t.Fatalf("results = %d, want %d", len(results), requestCount)
	}
	if got := streamer.total.Load(); got != requestCount {
		t.Fatalf("completed = %d, want %d", got, requestCount)
	}
	streamer.mu.Lock()
	peak := streamer.peak
	streamer.mu.Unlock()
	if peak > 3 {
		t.Fatalf("peak concurrency = %d, want <= 3", peak)
	}
}
