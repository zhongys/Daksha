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

// Model identifies a concrete model on a provider, plus its capabilities and
// pricing. Models are data, not code: the backing catalog (a database, for a
// backend) is loaded into a Client's registry at runtime.
type Model struct {
	// Provider is the key of the owning Provider entry.
	Provider string
	// ID is the wire model name, e.g. "deepseek-reasoner".
	ID string

	// Capability switches. Zero values are conservative: an unset capability
	// means the model does not have it.
	//
	// Reasoning marks models that can emit chain-of-thought; thinking format
	// parameters are only sent for these (a tuning knob — silently dropped
	// otherwise). The remaining switches gate semantic content and are
	// enforced loudly by Client.Stream before dispatch.
	Reasoning  bool
	ToolCall   bool
	ImageInput bool
	AudioInput bool
	VideoInput bool

	// Pricing is the per-token price in nano-yuan; zero means free/unknown
	// and yields a zero Cost.
	Pricing Pricing
	// ContextWindow and MaxOutputTokens inform the agent layer's budgeting.
	// They never alter requests: an unset StreamOptions.MaxTokens stays
	// unset and the vendor default applies.
	ContextWindow   int64
	MaxOutputTokens int64

	// Extra carries model-level request fields merged into every request.
	Extra map[string]any
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
// Implementations are stateless protocol translators: one instance per wire
// protocol serves every endpoint speaking it. The provider snapshot passed to
// each call carries all endpoint configuration (base URL, resolved API key,
// quirk overrides), so registry changes take effect on the next call while
// in-flight requests finish on the snapshot they started with.
//
// Contract (mirrors pi-ai's StreamFunction): the call itself never fails.
// Request, transport and parse failures are encoded in the returned stream —
// an ErrorEvent followed by completion with a final AssistantMessage whose
// StopReason is StopReasonError or StopReasonAborted and whose ErrorMessage
// is set. Consumers must drain Events() until closed; Result() then returns
// the final message.
//
// Adapters must accept this neutral StreamOptions as-is; inventing
// adapter-specific option types is forbidden (it is why pi needed a separate
// streamSimple layer, which this design deliberately makes impossible).
type Streamer interface {
	Stream(ctx context.Context, provider Provider, model Model, prompt Prompt, opts StreamOptions,
	) *EventStream[AssistantMessageEvent, *AssistantMessage]
}
