package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
)

type EmbeddingService struct {
	Options []RequestOption
}

func NewEmbeddingService(opts ...RequestOption) (r EmbeddingService) {
	r = EmbeddingService{}
	r.Options = opts
	return
}

func (r *EmbeddingService) New(ctx context.Context, body EmbeddingNewParams, opts ...RequestOption) (res *CreateEmbeddingResponse, err error) {
	preClientOpts := []RequestOption{WithBearerAuthSecurity()}
	opts = slices.Concat(preClientOpts, r.Options, opts)
	path := "embeddings"
	err = ExecuteNewRequest(ctx, http.MethodPost, path, body, &res, opts...)
	return res, err
}

type EmbeddingNewParams struct {
	// Input is the list of texts to embed. The protocol also accepts a single
	// string; this client always sends the array form.
	Input []string `json:"input"`
	// Model is the wire model name, e.g. "text-embedding-3-small".
	Model string `json:"model"`
	// EncodingFormat selects the vector encoding: "float" or "base64".
	// This client only decodes the float form.
	EncodingFormat string `json:"encoding_format,omitzero"`
	// Dimensions truncates the output vectors where the model supports it.
	Dimensions int64 `json:"dimensions,omitzero"`
	// ExtraFields carries provider-specific request fields merged into the
	// top-level request JSON. A key present here overrides the standard field
	// of the same name.
	ExtraFields map[string]any `json:"-"`
}

func (r EmbeddingNewParams) MarshalJSON() ([]byte, error) {
	type shadow EmbeddingNewParams
	b, err := json.Marshal(shadow(r))
	if err != nil {
		return nil, err
	}
	if len(r.ExtraFields) == 0 {
		return b, nil
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	for k, v := range r.ExtraFields {
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("openai: marshal extra field %q: %w", k, err)
		}
		m[k] = raw
	}
	return json.Marshal(m)
}

type CreateEmbeddingResponse struct {
	// Data holds one embedding per input, ordered by Index.
	Data []Embedding `json:"data"`
	// The model used to generate the embeddings.
	Model string `json:"model"`
	// The object type, which is always "list".
	Object string `json:"object"`
	// Usage statistics for the request; embeddings only consume input tokens.
	Usage EmbeddingUsage `json:"usage"`
}

type Embedding struct {
	// The embedding vector in float form.
	Embedding []float32 `json:"embedding"`
	// Index of the corresponding input text.
	Index int64 `json:"index"`
	// The object type, which is always "embedding".
	Object string `json:"object"`
}

type EmbeddingUsage struct {
	PromptTokens int64 `json:"prompt_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}
