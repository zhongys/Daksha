package ai

import (
	"context"
	"sync"
)

// maxCompleteBatchWorkers is a defensive implementation ceiling, not a
// caller-visible concurrency promise. CompleteBatch guarantees an upper bound;
// using fewer workers than an extreme request preserves that contract while
// preventing unbounded goroutine creation.
const maxCompleteBatchWorkers = 1024

// BatchRequest is one independent turn in a CompleteBatch call.
type BatchRequest struct {
	Provider string
	Model    string
	Prompt   Prompt
	Options  StreamOptions
}

// BatchResult holds one request's outcome. Err reports stream-level
// failures (e.g. context cancellation); model-side failures arrive in the
// message itself — check StopReason, per the Complete contract.
type BatchResult struct {
	Message *AssistantMessage
	Err     error
}

// CompleteBatch runs independent requests concurrently with a hard
// concurrency bound and returns results index-aligned with reqs: results[i]
// belongs to reqs[i], which is the entire correlation mechanism. One
// request's failure never affects the others.
//
// This is the sanctioned entry point for fan-out workloads (e.g. splitting
// a large extraction across parallel model calls): the bound is the rate
// discipline, and future controls (global limits, retries, cost budgets)
// belong here. maxConcurrency values below 1 are treated as 1; values above
// maxCompleteBatchWorkers are safely capped.
func (c *Client) CompleteBatch(ctx context.Context, reqs []BatchRequest, maxConcurrency int,
) []BatchResult {
	results := make([]BatchResult, len(reqs))
	workerCount := completeBatchWorkerCount(len(reqs), maxConcurrency)
	if workerCount == 0 {
		return results
	}

	jobs := make(chan int)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for range workerCount {
		go func() {
			defer wg.Done()
			for i := range jobs {
				req := reqs[i]
				msg, err := c.Complete(ctx, req.Provider, req.Model, req.Prompt, req.Options)
				results[i] = BatchResult{Message: msg, Err: err}
			}
		}()
	}
	for i := range reqs {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return results
}

func completeBatchWorkerCount(requestCount, maxConcurrency int) int {
	if requestCount <= 0 {
		return 0
	}
	if maxConcurrency < 1 {
		maxConcurrency = 1
	}
	if maxConcurrency > requestCount {
		maxConcurrency = requestCount
	}
	if maxConcurrency > maxCompleteBatchWorkers {
		maxConcurrency = maxCompleteBatchWorkers
	}
	return maxConcurrency
}
