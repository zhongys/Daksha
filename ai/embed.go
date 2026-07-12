package ai

import (
	"context"
	"fmt"
)

// EmbedOptions carries tuning knobs for one embeddings request.
type EmbedOptions struct {
	// Dimensions truncates the output vectors where the model supports it.
	// Zero means the model default.
	Dimensions int64
	// Extra carries request-level fields merged into the request JSON.
	// Precedence: Provider.Extra < Model.Extra < EmbedOptions.Extra.
	Extra map[string]any
}

// EmbedResult is the outcome of one embeddings request. Vectors is parallel
// to the inputs slice: Vectors[i] embeds inputs[i].
type EmbedResult struct {
	Vectors  [][]float32
	Usage    Usage
	Provider string
	Model    string
	// ResponseModel is the model name the endpoint reported, when it differs
	// from the requested ID.
	ResponseModel string
}

// Embedder is the protocol-neutral entry point for one embeddings request.
//
// Implementations are stateless protocol translators, like Streamer. Unlike
// Stream there is no event stream to encode failures in, so transport and
// protocol errors return as plain errors.
type Embedder interface {
	Embed(ctx context.Context, provider Provider, model Model, inputs []string, opts EmbedOptions,
	) (*EmbedResult, error)
}

// PutEmbedder registers the embeddings adapter for a wire protocol, keyed
// like the Streamer route table (e.g. "openai-completions").
func (c *Client) PutEmbedder(api string, e Embedder) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.embedders[api] = e
}

// Embed runs one embeddings request against a registered provider and model.
// Configuration and the API key resolve at call time exactly like Stream.
func (c *Client) Embed(ctx context.Context, providerName, modelID string, inputs []string, opts EmbedOptions,
) (*EmbedResult, error) {
	provider, model, err := c.resolveEntry(providerName, modelID)
	if err != nil {
		return nil, err
	}

	c.mu.RLock()
	embedder, ok := c.embedders[provider.API]
	c.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("ai: no embedder registered for protocol %q", provider.API)
	}

	if len(inputs) == 0 {
		return &EmbedResult{Provider: providerName, Model: modelID}, nil
	}
	return embedder.Embed(ctx, provider, model, inputs, opts)
}
