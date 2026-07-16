package ai

// Provider describes one endpoint. Vendors that speak the same wire protocol
// differ only in configuration, so a provider is data, not code: adding a
// vendor means adding an entry to a Client's registry. Only a genuinely
// different wire protocol (e.g. Anthropic Messages) warrants a new adapter
// package next to ai/api/openaicompletions.
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

// SeedProviders returns starter entries for well-known endpoints. Pure data,
// no side effects: main decides whether to load them into a Client. The model
// catalog has no seed — its source of truth is the application's storage.
func SeedProviders() []Provider {
	return []Provider{
		{
			Name:      "openai",
			BaseURL:   "https://api.openai.com/v1/",
			APIKeyEnv: "OPENAI_API_KEY",
		},
		{
			Name:      "deepseek",
			BaseURL:   "https://api.deepseek.com/",
			APIKeyEnv: "DEEPSEEK_API_KEY",
		},
		{
			Name:      "qwen",
			BaseURL:   "https://dashscope.aliyuncs.com/compatible-mode/v1/",
			APIKeyEnv: "DASHSCOPE_API_KEY",
		},
	}
}
