package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

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
	// "max_completion_tokens" (current OpenAI-compatible APIs) or the legacy
	// "max_tokens" field used by endpoints such as DeepSeek.
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
	// Client.PutModel snapshots values by their JSON representation; Get/List
	// return JSON tree types rather than caller-specific concrete Go types.
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

// OutputFormatType selects the semantic shape of the assistant's final text.
// The zero value leaves the provider default unchanged.
type OutputFormatType string

const (
	OutputFormatText       OutputFormatType = "text"
	OutputFormatJSONObject OutputFormatType = "json_object"
	OutputFormatJSONSchema OutputFormatType = "json_schema"
)

// JSONSchema describes a named JSON Schema response contract. Schema is kept
// protocol-neutral; adapters translate it to the target wire representation.
type JSONSchema struct {
	Name        string
	Description string
	Schema      map[string]any
	Strict      bool
}

// OutputFormat constrains the assistant's final response. JSON outputs arrive
// through JSONStart/JSONDelta/JSONEnd events; adapters must validate the
// complete value before publishing JSONEnd or a successful final result.
// JSONSchema conformance is enforced by the provider; local final validation
// guarantees JSON syntax only.
type OutputFormat struct {
	Type       OutputFormatType
	JSONSchema *JSONSchema
}

// Validate rejects ambiguous or incomplete output contracts before dispatch.
func (f OutputFormat) Validate() error {
	switch f.Type {
	case "", OutputFormatText, OutputFormatJSONObject:
		if f.JSONSchema != nil {
			return fmt.Errorf("ai: json schema is only valid with output format %q", OutputFormatJSONSchema)
		}
		return nil
	case OutputFormatJSONSchema:
		if f.JSONSchema == nil {
			return fmt.Errorf("ai: output format %q requires a json schema", OutputFormatJSONSchema)
		}
		if strings.TrimSpace(f.JSONSchema.Name) == "" {
			return fmt.Errorf("ai: output format %q requires a schema name", OutputFormatJSONSchema)
		}
		if len(f.JSONSchema.Schema) == 0 {
			return fmt.Errorf("ai: output format %q requires a non-empty schema", OutputFormatJSONSchema)
		}
		if _, err := json.Marshal(f.JSONSchema.Schema); err != nil {
			return fmt.Errorf("ai: output format %q has an invalid schema: %w", OutputFormatJSONSchema, err)
		}
		return nil
	default:
		return fmt.Errorf("ai: unknown output format %q", f.Type)
	}
}

// IsJSON reports whether a successful final response must contain valid JSON.
func (f OutputFormat) IsJSON() bool {
	return f.Type == OutputFormatJSONObject || f.Type == OutputFormatJSONSchema
}

type StreamOptions struct {
	Temperature   *float64
	TopP          *float64
	MaxTokens     *int64
	StopSequences []string
	// ToolChoice is "", "auto", "none" or "required".
	ToolChoice string
	// ReasoningEffort is provider-specific. OpenAI-style providers commonly use
	// "minimal", "low", "medium" or "high"; Kimi K3 currently accepts only
	// "max". Empty uses the provider default, or disables thinking where the
	// vendor accepts an explicit disable (deepseek/zai).
	ReasoningEffort string
	// OutputFormat is the protocol-neutral final-output contract. The zero value
	// leaves the provider default unchanged.
	OutputFormat OutputFormat
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
// While ctx remains live, request, transport and parse failures are encoded in
// the returned stream — a terminal ErrorEvent followed by completion with a
// final AssistantMessage whose StopReason is StopReasonError and whose
// ErrorMessage is set. ErrorEvent implicitly aborts any open content blocks.
// Consumers must drain Events() until closed; Result() then returns the final
// message. Canceling ctx is instead a stream-mechanics failure: Events closes
// and Result returns ctx.Err without requiring a terminal event or result.
// Implementations must finish reading or snapshotting every caller-owned
// mutable input before Stream returns. Callers may safely reuse or mutate the
// original Prompt and StreamOptions after that boundary.
//
// Adapters must accept this neutral StreamOptions as-is; inventing
// adapter-specific option types is forbidden (it is why pi needed a separate
// streamSimple layer, which this design deliberately makes impossible).
type Streamer interface {
	Stream(ctx context.Context, provider Provider, model Model, prompt Prompt, opts StreamOptions,
	) *EventStream[AssistantMessageEvent, *AssistantMessage]
}
