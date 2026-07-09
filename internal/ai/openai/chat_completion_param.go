package openai

import (
	"encoding/json"
	"fmt"
)

type ChatCompletionNewParams struct {
	Messages []ChatCompletionMessageParamUnion `json:"messages,omitzero"`
	Model    string                            `json:"model,omitzero"`
	// Number between -2.0 and 2.0. Positive values penalize new tokens based on
	// their existing frequency in the text so far.
	FrequencyPenalty *float64 `json:"frequency_penalty,omitzero"`
	// Whether to return log probabilities of the output tokens.
	Logprobs            *bool  `json:"logprobs,omitzero"`
	MaxCompletionTokens *int64 `json:"max_completion_tokens,omitzero"`
	MaxTokens           *int64 `json:"max_tokens,omitzero"`
	// How many chat completion choices to generate for each input message. Keep
	// `n` as `1` to minimize costs.
	N *int64 `json:"n,omitzero"`
	// Number between -2.0 and 2.0. Positive values penalize new tokens based on
	// whether they appear in the text so far, increasing the model's likelihood
	// to talk about new topics.
	PresencePenalty   *float64         `json:"presence_penalty,omitzero"`
	Seed              *int64           `json:"seed,omitzero"`
	Temperature       *float64         `json:"temperature,omitzero"`
	TopLogprobs       *int64           `json:"top_logprobs,omitzero"`
	TopP              *float64         `json:"top_p,omitzero"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitzero"`
	User              *string          `json:"user,omitzero"`
	LogitBias         map[string]int64 `json:"logit_bias,omitzero"`
	// Constrains effort on reasoning for reasoning models.
	//
	// Any of "minimal", "low", "medium", "high".
	ReasoningEffort string                           `json:"reasoning_effort,omitzero"`
	Stop            ChatCompletionNewParamsStopUnion `json:"stop,omitzero"`
	// Options for streaming response. Only set this when you set `stream: true`.
	StreamOptions  ChatCompletionStreamOptionsParam           `json:"stream_options,omitzero"`
	Verbosity      ChatCompletionNewParamsVerbosity           `json:"verbosity,omitzero"`
	ResponseFormat ChatCompletionNewParamsResponseFormatUnion `json:"response_format,omitzero"`
	ToolChoice     ChatCompletionToolChoiceOptionUnionParam   `json:"tool_choice,omitzero"`
	Tools          []ChatCompletionToolUnionParam             `json:"tools,omitzero"`
	// ExtraFields carries provider-specific request fields merged into the
	// top-level request JSON, e.g. Qwen's enable_thinking. A key present here
	// overrides the standard field of the same name.
	ExtraFields map[string]any `json:"-"`
}

func (r ChatCompletionNewParams) MarshalJSON() ([]byte, error) {
	type shadow ChatCompletionNewParams
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

// ChatCompletionNewParamsResponseFormatUnion holds exactly one response
// format variant.
type ChatCompletionNewParamsResponseFormatUnion struct {
	OfText       *ResponseFormatTextParam       `json:"-"`
	OfJSONSchema *ResponseFormatJSONSchemaParam `json:"-"`
	OfJSONObject *ResponseFormatJSONObjectParam `json:"-"`
}

func (u ChatCompletionNewParamsResponseFormatUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfText != nil:
		return json.Marshal(u.OfText)
	case u.OfJSONObject != nil:
		return json.Marshal(u.OfJSONObject)
	case u.OfJSONSchema != nil:
		return json.Marshal(u.OfJSONSchema)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionNewParamsResponseFormatUnion")
	}
}

type ResponseFormatTextParam struct {
	Type string `json:"type"`
}

type ResponseFormatJSONObjectParam struct {
	Type string `json:"type"`
}

type ResponseFormatJSONSchemaParam struct {
	Type       string                                  `json:"type"`
	JSONSchema ResponseFormatJSONSchemaJSONSchemaParam `json:"json_schema"`
}

type ResponseFormatJSONSchemaJSONSchemaParam struct {
	Name        string         `json:"name"`
	Description *string        `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      *bool          `json:"strict,omitempty"`
}

func ResponseFormatText() ChatCompletionNewParamsResponseFormatUnion {
	return ChatCompletionNewParamsResponseFormatUnion{
		OfText: &ResponseFormatTextParam{Type: "text"},
	}
}

func ResponseFormatJSONObject() ChatCompletionNewParamsResponseFormatUnion {
	return ChatCompletionNewParamsResponseFormatUnion{
		OfJSONObject: &ResponseFormatJSONObjectParam{Type: "json_object"},
	}
}

func ResponseFormatJSONSchema(schema ResponseFormatJSONSchemaJSONSchemaParam) ChatCompletionNewParamsResponseFormatUnion {
	return ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &ResponseFormatJSONSchemaParam{
			Type:       "json_schema",
			JSONSchema: schema,
		},
	}
}

type ChatCompletionStreamOptionsParam struct {
	// If set, an additional chunk will be streamed before the `data: [DONE]`
	// message. The `usage` field on this chunk shows the token usage statistics
	// for the entire request, and the `choices` field will always be an empty
	// array.
	IncludeUsage *bool `json:"include_usage,omitzero"`
}

// ChatCompletionNewParamsStopUnion holds either a single stop sequence or a
// list of them.
type ChatCompletionNewParamsStopUnion struct {
	OfString      *string  `json:"-"`
	OfStringArray []string `json:"-"`
}

func (u ChatCompletionNewParamsStopUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfStringArray != nil:
		return json.Marshal(u.OfStringArray)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionNewParamsStopUnion")
	}
}

// Constrains the verbosity of the model's response. Currently supported
// values are `low`, `medium`, and `high`.
type ChatCompletionNewParamsVerbosity string

const (
	ChatCompletionNewParamsVerbosityLow    ChatCompletionNewParamsVerbosity = "low"
	ChatCompletionNewParamsVerbosityMedium ChatCompletionNewParamsVerbosity = "medium"
	ChatCompletionNewParamsVerbosityHigh   ChatCompletionNewParamsVerbosity = "high"
)
