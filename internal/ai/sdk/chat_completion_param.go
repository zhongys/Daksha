package sdk

import (
	"encoding/json"

	constant "github.com/zhongys/Daksha.git/internal/ai/sdk/shared"
)

type ChatCompletionNewParams struct {
	Messages []ChatCompletionMessageParamUnion `json:"messages,omitzero" api:"required"`
	Model    string                            `json:"model,omitzero" api:"required"`
	// Number between -2.0 and 2.0. Positive values penalize new tokens based on their

	FrequencyPenalty *float64 `json:"frequency_penalty,omitzero"`

	Logprobs            *bool  `json:"logprobs,omitzero"`
	MaxCompletionTokens *int64 `json:"max_completion_tokens,omitzero"`

	MaxTokens *int64 `json:"max_tokens,omitzero"`
	// How many chat completion choices to generate for each input message. Note that
	// you will be charged based on the number of generated tokens across all of the
	// choices. Keep `n` as `1` to minimize costs.
	N *int64 `json:"n,omitzero"`
	// Number between -2.0 and 2.0. Positive values penalize new tokens based on
	// whether they appear in the text so far, increasing the model's likelihood to
	// talk about new topics.
	PresencePenalty   *float64                 `json:"presence_penalty,omitzero"`
	Seed              *int64                   `json:"seed,omitzero"`
	Store             *bool                    `json:"store,omitzero"`
	Temperature       *float64                 `json:"temperature,omitzero"`
	TopLogprobs       *int64                   `json:"top_logprobs,omitzero"`
	TopP              *float64                 `json:"top_p,omitzero"`
	ParallelToolCalls *bool                    `json:"parallel_tool_calls,omitzero"`
	PromptCacheKey    *string                  `json:"prompt_cache_key,omitzero"`
	SafetyIdentifier  *string                  `json:"safety_identifier,omitzero"`
	User              *string                  `json:"user,omitzero"`
	Audio             ChatCompletionAudioParam `json:"audio,omitzero"`
	LogitBias         map[string]int64         `json:"logit_bias,omitzero"`
	Metadata          string                   `json:"metadata,omitzero"`
	Modalities        []string                 `json:"modalities,omitzero"`
	// Configuration for running moderation on the request input and generated output.
	Moderation           ChatCompletionNewParamsModeration           `json:"moderation,omitzero"`
	PromptCacheRetention ChatCompletionNewParamsPromptCacheRetention `json:"prompt_cache_retention,omitzero"`
	ReasoningEffort      string                                      `json:"reasoning_effort,omitzero"`
	ServiceTier          ChatCompletionNewParamsServiceTier          `json:"service_tier,omitzero"`
	Stop                 ChatCompletionNewParamsStopUnion            `json:"stop,omitzero"`
	// Options for streaming response. Only set this when you set `stream: true`.
	StreamOptions ChatCompletionStreamOptionsParam `json:"stream_options,omitzero"`
	Verbosity     ChatCompletionNewParamsVerbosity `json:"verbosity,omitzero"`
	// Static predicted output content, such as the content of a text file that is
	// being regenerated.
	Prediction       ChatCompletionPredictionContentParam       `json:"prediction,omitzero"`
	ResponseFormat   ChatCompletionNewParamsResponseFormatUnion `json:"response_format,omitzero"`
	ToolChoice       ChatCompletionToolChoiceOptionUnionParam   `json:"tool_choice,omitzero"`
	Tools            []ChatCompletionToolUnionParam             `json:"tools,omitzero"`
	WebSearchOptions ChatCompletionNewParamsWebSearchOptions    `json:"web_search_options,omitzero"`
}

// Configuration for running moderation on the request input and generated output.
//
// The property Model is required.
type ChatCompletionNewParamsModeration struct {
	// The moderation model to use for moderated completions, e.g.
	// 'omni-moderation-latest'.
	Model string `json:"model" api:"required"`
}

type ChatCompletionNewParamsResponseFormatUnion struct {
	OfText       *ResponseFormatTextParam       `json:",omitzero,inline"`
	OfJSONSchema *ResponseFormatJSONSchemaParam `json:",omitzero,inline"`
	OfJSONObject *ResponseFormatJSONObjectParam `json:",omitzero,inline"`
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
		return []byte("null"), nil
	}
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionNewParamsResponseFormatUnion) GetJSONSchema() *ResponseFormatJSONSchemaJSONSchemaParam {
	if u.OfJSONSchema == nil {
		return nil
	}
	return &u.OfJSONSchema.JSONSchema

}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionNewParamsResponseFormatUnion) GetType() *string {
	if u.OfText != nil {
		return &u.OfText.Type
	}
	if u.OfJSONObject != nil {
		return &u.OfJSONObject.Type
	}
	if u.OfJSONSchema != nil {
		return &u.OfJSONSchema.Type
	}
	return nil
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
		OfText: &ResponseFormatTextParam{Type: constant.ResponseFormatText},
	}
}

func ResponseFormatJSONObject() ChatCompletionNewParamsResponseFormatUnion {
	return ChatCompletionNewParamsResponseFormatUnion{
		OfJSONObject: &ResponseFormatJSONObjectParam{Type: constant.ResponseFormatJSONObject},
	}
}

func ResponseFormatJSONSchema(schema ResponseFormatJSONSchemaJSONSchemaParam) ChatCompletionNewParamsResponseFormatUnion {
	return ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &ResponseFormatJSONSchemaParam{
			Type:       constant.ResponseFormatJSONSchema,
			JSONSchema: schema,
		},
	}
}

// The properties Content, Type are required.
type ChatCompletionPredictionContentParam struct {
	// The content that should be matched when generating a model response. If
	// generated tokens would match this content, the entire model response can be
	// returned much more quickly.
	Content ChatCompletionPredictionContentContentUnionParam `json:"content,omitzero" api:"required"`
	// The type of the predicted content you want to provide. This type is currently
	// always `content`.
	//
	// This field can be elided, and will marshal its zero value as "content".
	Type string `json:"type" default:"content"`
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionPredictionContentContentUnionParam struct {
	OfString              *string                              `json:",omitzero,inline"`
	OfArrayOfContentParts []ChatCompletionContentPartTextParam `json:",omitzero,inline"`
}

func (u *ChatCompletionPredictionContentContentUnionParam) asAny() any {
	if u == nil {
		return nil
	}
	if u.OfString != nil {
		return u.OfString
	}
	if u.OfArrayOfContentParts != nil {
		return &u.OfArrayOfContentParts
	}
	return nil
}
func (u ChatCompletionPredictionContentContentUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return []byte("null"), nil
	}
}

type ChatCompletionStreamOptionsParam struct {
	// When true, stream obfuscation will be enabled. Stream obfuscation adds random
	// characters to an `obfuscation` field on streaming delta events to normalize
	// payload sizes as a mitigation to certain side-channel attacks. These obfuscation
	// fields are included by default, but add a small amount of overhead to the data
	// stream. You can set `include_obfuscation` to false to optimize for bandwidth if
	// you trust the network links between your application and the OpenAI API.
	IncludeObfuscation *bool `json:"include_obfuscation,omitzero"`
	// If set, an additional chunk will be streamed before the `data: [DONE]` message.
	// The `usage` field on this chunk shows the token usage statistics for the entire
	// request, and the `choices` field will always be an empty array.
	//
	// All other chunks will also include a `usage` field, but with a null value.
	// **NOTE:** If the stream is interrupted, you may not receive the final usage
	// chunk which contains the total token usage for the request.
	IncludeUsage *bool `json:"include_usage,omitzero"`
}

type ChatCompletionNewParamsServiceTier string

const (
	ChatCompletionNewParamsServiceTierAuto     ChatCompletionNewParamsServiceTier = "auto"
	ChatCompletionNewParamsServiceTierDefault  ChatCompletionNewParamsServiceTier = "default"
	ChatCompletionNewParamsServiceTierFlex     ChatCompletionNewParamsServiceTier = "flex"
	ChatCompletionNewParamsServiceTierScale    ChatCompletionNewParamsServiceTier = "scale"
	ChatCompletionNewParamsServiceTierPriority ChatCompletionNewParamsServiceTier = "priority"
)

// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionNewParamsStopUnion struct {
	OfString      *string  `json:",omitzero,inline"`
	OfStringArray []string `json:",omitzero,inline"`
}

func (u ChatCompletionNewParamsStopUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfStringArray != nil:
		return json.Marshal(u.OfStringArray)
	default:
		return []byte("null"), nil
	}
}

// responses. Currently supported values are `low`, `medium`, and `high`.
type ChatCompletionNewParamsVerbosity string

const (
	ChatCompletionNewParamsVerbosityLow    ChatCompletionNewParamsVerbosity = "low"
	ChatCompletionNewParamsVerbosityMedium ChatCompletionNewParamsVerbosity = "medium"
	ChatCompletionNewParamsVerbosityHigh   ChatCompletionNewParamsVerbosity = "high"
)

type ChatCompletionNewParamsPromptCacheRetention string

const (
	ChatCompletionNewParamsPromptCacheRetentionInMemory ChatCompletionNewParamsPromptCacheRetention = "in_memory"
	ChatCompletionNewParamsPromptCacheRetention24h      ChatCompletionNewParamsPromptCacheRetention = "24h"
)

// This tool searches the web for relevant results to use in a response. Learn more
// about the
// [web search tool](https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat).
type ChatCompletionNewParamsWebSearchOptions struct {
	// Approximate location parameters for the search.
	UserLocation ChatCompletionNewParamsWebSearchOptionsUserLocation `json:"user_location,omitzero"`
	// High level guidance for the amount of context window space to use for the
	// search. One of `low`, `medium`, or `high`. `medium` is the default.
	//
	// Any of "low", "medium", "high".
	SearchContextSize string `json:"search_context_size,omitzero"`
}
type ChatCompletionNewParamsWebSearchOptionsUserLocation struct {
	Type        string                                                          `json:"type"`
	Approximate *ChatCompletionNewParamsWebSearchOptionsUserLocationApproximate `json:"approximate,omitempty"`
}

type ChatCompletionNewParamsWebSearchOptionsUserLocationApproximate struct {
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	City     string `json:"city,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}
