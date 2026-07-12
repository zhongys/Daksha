package openaicompletions

import (
	"context"
	"fmt"

	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/openai"
)

// Embedder implements ai.Embedder over the OpenAI-compatible embeddings
// protocol. Like Streamer it is a stateless protocol translator: one instance
// serves every endpoint speaking this protocol, and each call carries the
// provider snapshot it should use.
type Embedder struct {
	embeddings openai.EmbeddingService
}

var _ ai.Embedder = (*Embedder)(nil)

// NewEmbedder builds the protocol adapter. Options carry infrastructure
// configuration only (shared http client, timeouts); per-endpoint business
// configuration travels with each call's provider.
func NewEmbedder(opts ...openai.RequestOption) *Embedder {
	return &Embedder{
		embeddings: openai.NewEmbeddingService(opts...),
	}
}

func (e *Embedder) Embed(ctx context.Context, provider ai.Provider, model ai.Model, inputs []string, opts ai.EmbedOptions,
) (*ai.EmbedResult, error) {
	params := openai.EmbeddingNewParams{
		Input:          inputs,
		Model:          model.ID,
		EncodingFormat: "float",
		Dimensions:     opts.Dimensions,
	}
	for _, extra := range []map[string]any{provider.Extra, model.Extra, opts.Extra} {
		for k, v := range extra {
			if params.ExtraFields == nil {
				params.ExtraFields = map[string]any{}
			}
			params.ExtraFields[k] = v
		}
	}

	var reqOpts []openai.RequestOption
	if provider.BaseURL != "" {
		reqOpts = append(reqOpts, openai.WithBaseURL(provider.BaseURL))
	}
	if provider.APIKey != "" {
		reqOpts = append(reqOpts, openai.WithAPIKey(provider.APIKey))
	}

	res, err := e.embeddings.New(ctx, params, reqOpts...)
	if err != nil {
		return nil, err
	}

	// Place vectors by the endpoint-reported index so the result is parallel
	// to inputs regardless of response order.
	vectors := make([][]float32, len(inputs))
	for _, d := range res.Data {
		if d.Index < 0 || d.Index >= int64(len(inputs)) {
			return nil, fmt.Errorf("openaicompletions: embedding index %d out of range for %d inputs", d.Index, len(inputs))
		}
		vectors[d.Index] = d.Embedding
	}
	for i, v := range vectors {
		if v == nil {
			return nil, fmt.Errorf("openaicompletions: endpoint returned no embedding for input %d", i)
		}
	}

	usage := ai.Usage{
		Input:       res.Usage.PromptTokens,
		TotalTokens: res.Usage.TotalTokens,
	}
	usage.Cost = ai.CalculateCost(model, usage)

	result := &ai.EmbedResult{
		Vectors:  vectors,
		Usage:    usage,
		Provider: provider.Name,
		Model:    model.ID,
	}
	if res.Model != "" && res.Model != model.ID {
		result.ResponseModel = res.Model
	}
	return result, nil
}
