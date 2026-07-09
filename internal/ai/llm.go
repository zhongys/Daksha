package ai

import "context"

// ThinkingFormat selects how a reasoning toggle is encoded in the request.
// OpenAI-compatible vendors each invented their own field for this.
type ThinkingFormat string

const (
	// reasoning_effort (OpenAI style)
	ThinkingFormatOpenAI ThinkingFormat = "openai"
	// thinking: {type: enabled|disabled} plus reasoning_effort
	ThinkingFormatDeepSeek ThinkingFormat = "deepseek"
	// top-level enable_thinking: bool (DashScope)
	ThinkingFormatQwen ThinkingFormat = "qwen"
	// reasoning: {effort: ...}
	ThinkingFormatOpenRouter ThinkingFormat = "openrouter"
	// thinking: {type: enabled|disabled} (GLM / z.ai)
	ThinkingFormatZai ThinkingFormat = "zai"
)

// OpenAICompat overrides quirk handling for OpenAI-compatible endpoints.
// Zero/nil fields are auto-detected from the provider's base URL.
type OpenAICompat struct {
	// Which request field carries the token limit:
	// "max_completion_tokens" (OpenAI) or "max_tokens" (Moonshot etc.).
	MaxTokensField string
	ThinkingFormat ThinkingFormat
	// Whether the endpoint accepts the "developer" role for instructions on
	// reasoning models; when false the system role is always used.
	SupportsDeveloperRole *bool
	// DeepSeek requires every replayed assistant message to carry a
	// reasoning_content field (empty string is fine) when reasoning is on.
	RequiresReasoningContentOnAssistantMessages *bool
}

// Model identifies a concrete model on a provider, plus its capabilities.
type Model struct {
	// Provider is the key of the owning Provider entry.
	Provider string
	// ID is the wire model name, e.g. "deepseek-reasoner".
	ID string
	// Reasoning marks models that can emit chain-of-thought; thinking format
	// parameters are only sent for these.
	Reasoning bool
	// Extra carries model-level request fields merged into every request.
	Extra map[string]any
	// Compat overrides endpoint quirk detection.
	Compat *OpenAICompat
}

type ToolDefinition struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object.
	Parameters map[string]any
}

// Prompt is everything the model sees for one turn.
type Prompt struct {
	System   string
	Messages []Message
	Tools    []ToolDefinition
}

type StreamOptions struct {
	Temperature   *float64
	TopP          *float64
	MaxTokens     *int64
	StopSequences []string
	// ToolChoice is "", "auto", "none" or "required".
	ToolChoice string
	// ReasoningEffort enables thinking on reasoning models: "minimal", "low",
	// "medium" or "high". Empty disables thinking where the vendor allows it.
	ReasoningEffort string
	// Extra carries request-level fields merged into the request JSON.
	// Precedence: Provider.Extra < Model.Extra < StreamOptions.Extra.
	Extra map[string]any
}

// Streamer is the protocol-neutral entry point for one LLM turn.
//
// Contract (mirrors pi-ai's StreamFunction): the call itself never fails.
// Request, transport and parse failures are encoded in the returned stream —
// an ErrorEvent followed by completion with a final AssistantMessage whose
// StopReason is StopReasonError or StopReasonAborted and whose ErrorMessage
// is set. Consumers must drain Events() until closed; Result() then returns
// the final message.
type Streamer interface {
	Stream(ctx context.Context, model Model, prompt Prompt, opts StreamOptions,
	) *EventStream[AssistantMessageEvent, *AssistantMessage]
}
