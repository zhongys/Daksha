package ai

// Provider describes one endpoint. Vendors that speak the same wire protocol
// differ only in configuration, so a provider is data, not code: adding a
// vendor means adding an entry to a Client's registry. Only a genuinely
// different wire protocol (e.g. Anthropic Messages) warrants a new adapter
// package next to ai/api/openaicompletions.
//
// Provider is trusted host-application configuration, not an untrusted wire
// DTO. Applications that accept dynamic provider configuration are responsible
// for authorization, URL and egress validation, credential isolation, and
// restricting Compat and Extra before calling Client.PutProvider.
type Provider struct {
	Name    string
	BaseURL string
	// API names the wire protocol this endpoint speaks and selects the
	// adapter in the Client's route table. Empty means "openai-completions".
	// A vendor exposing two protocols is two Provider entries.
	API string
	// APIKey is the key itself, filled by the application (database, secret
	// manager). Resolved on every Stream call, so rotation takes effect
	// immediately.
	APIKey string
	// APIKeyEnv names an environment variable used as a development-time
	// fallback when APIKey is empty.
	APIKeyEnv string
	// Compat overrides endpoint quirk detection. Unset fields fall back to
	// auto-detection from BaseURL.
	Compat *OpenAICompat
	// Extra carries provider-specific request fields merged into the
	// top-level request JSON, e.g. Qwen's enable_thinking. Client.PutProvider
	// snapshots values by their JSON representation; Get/List therefore return
	// JSON tree types rather than preserving caller-specific concrete Go types.
	Extra map[string]any
}
