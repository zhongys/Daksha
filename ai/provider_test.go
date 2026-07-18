package ai

import "testing"

func TestSeedProvidersIncludesIsolatedKimiK3Provider(t *testing.T) {
	var matches []Provider
	for _, provider := range SeedProviders() {
		if provider.Name == "kimi" {
			matches = append(matches, provider)
		}
	}
	if len(matches) != 1 {
		t.Fatalf("kimi providers = %d, want 1", len(matches))
	}

	provider := matches[0]
	if provider.BaseURL != "https://api.moonshot.cn/v1/" {
		t.Errorf("BaseURL = %q", provider.BaseURL)
	}
	if provider.APIKeyEnv != "MOONSHOT_API_KEY" {
		t.Errorf("APIKeyEnv = %q", provider.APIKeyEnv)
	}
	if provider.Compat == nil {
		t.Fatal("Compat is nil")
	}
	if provider.Compat.MaxTokensField != "max_completion_tokens" {
		t.Errorf("MaxTokensField = %q", provider.Compat.MaxTokensField)
	}
	if provider.Compat.ThinkingFormat != ThinkingFormatOpenAI {
		t.Errorf("ThinkingFormat = %q", provider.Compat.ThinkingFormat)
	}
}
