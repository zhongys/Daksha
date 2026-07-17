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
	r.Options = slices.Clone(opts)
	return
}

func (r *EmbeddingService) New(ctx context.Context, body EmbeddingNewParams, opts ...RequestOption) (res *CreateEmbeddingResponse, err error) {
	preClientOpts := []RequestOption{WithBearerAuthSecurity()}
	opts = slices.Concat(preClientOpts, r.Options, opts)
	path := "embeddings"
	err = ExecuteNewRequest(ctx, http.MethodPost, path, body, &res, opts...)
	if err != nil {
		return nil, err
	}
	if err := validateEmbeddingResponse(res, body); err != nil {
		return nil, err
	}
	return res, nil
}

func validateEmbeddingResponse(res *CreateEmbeddingResponse, request EmbeddingNewParams) error {
	if res == nil {
		return fmt.Errorf("openai: invalid embedding response: top-level value is null")
	}
	if res.Object != "" && res.Object != "list" {
		return fmt.Errorf("openai: invalid embedding response object %q", res.Object)
	}
	if len(res.Data) == 0 {
		return fmt.Errorf("openai: invalid embedding response: data is empty")
	}

	_, inputOverridden := request.ExtraFields["input"]
	if !inputOverridden && len(res.Data) != len(request.Input) {
		return fmt.Errorf(
			"openai: invalid embedding response: got %d embeddings for %d inputs",
			len(res.Data), len(request.Input),
		)
	}
	_, dimensionsOverridden := request.ExtraFields["dimensions"]

	seenIndexes := make(map[int64]struct{}, len(res.Data))
	vectorLength := -1
	for i, item := range res.Data {
		if item.Object != "" && item.Object != "embedding" {
			return fmt.Errorf("openai: invalid embedding item %d object %q", i, item.Object)
		}
		if item.Index < 0 || item.Index >= int64(len(res.Data)) {
			return fmt.Errorf("openai: invalid embedding item %d: index %d out of range", i, item.Index)
		}
		if _, exists := seenIndexes[item.Index]; exists {
			return fmt.Errorf("openai: invalid embedding response: duplicate index %d", item.Index)
		}
		seenIndexes[item.Index] = struct{}{}

		if len(item.Embedding) == 0 {
			return fmt.Errorf("openai: invalid embedding item %d: vector is empty", item.Index)
		}
		if vectorLength < 0 {
			vectorLength = len(item.Embedding)
		} else if len(item.Embedding) != vectorLength {
			return fmt.Errorf(
				"openai: invalid embedding item %d: vector length %d does not match %d",
				item.Index, len(item.Embedding), vectorLength,
			)
		}
		if request.Dimensions > 0 && !dimensionsOverridden && int64(len(item.Embedding)) != request.Dimensions {
			return fmt.Errorf(
				"openai: invalid embedding item %d: vector length %d does not match requested dimensions %d",
				item.Index, len(item.Embedding), request.Dimensions,
			)
		}
	}
	return nil
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
