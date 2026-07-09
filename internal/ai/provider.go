package ai

// Provider describes an OpenAI-compatible endpoint. Vendors that speak the
// same wire protocol differ only in configuration, so a provider is data,
// not code: adding a vendor means adding an entry here. Only a genuinely
// different wire protocol (e.g. Anthropic Messages) warrants a new protocol
// package next to ai/openai.
type Provider struct {
	Name    string
	BaseURL string
	// APIKeyEnv names the environment variable holding the API key.
	APIKeyEnv string
	// Extra carries provider-specific request fields merged into the
	// top-level request JSON, e.g. Qwen's enable_thinking.
	Extra map[string]any
}

var BuiltinProviders = map[string]Provider{
	"openai": {
		Name:      "openai",
		BaseURL:   "https://api.openai.com/v1/",
		APIKeyEnv: "OPENAI_API_KEY",
	},
	"deepseek": {
		Name:      "deepseek",
		BaseURL:   "https://api.deepseek.com/",
		APIKeyEnv: "DEEPSEEK_API_KEY",
	},
	"qwen": {
		Name:      "qwen",
		BaseURL:   "https://dashscope.aliyuncs.com/compatible-mode/v1/",
		APIKeyEnv: "DASHSCOPE_API_KEY",
	},
}
