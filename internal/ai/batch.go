package ai

import (
	"context"
	"sync"
)

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
// belong here. maxConcurrency values below 1 are treated as 1.
func (c *Client) CompleteBatch(ctx context.Context, reqs []BatchRequest, maxConcurrency int,
) []BatchResult {
	if maxConcurrency < 1 {
		maxConcurrency = 1
	}
	results := make([]BatchResult, len(reqs))
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, req := range reqs {
		wg.Add(1)
		go func(i int, req BatchRequest) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			msg, err := c.Complete(ctx, req.Provider, req.Model, req.Prompt, req.Options)
			results[i] = BatchResult{Message: msg, Err: err}
		}(i, req)
	}
	wg.Wait()
	return results
}
