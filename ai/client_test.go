package ai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

type privateExtraFields struct {
	Values []string `json:"values"`
}

type privateEmbeddedExtra struct {
	privateExtraFields
}

type marshalOnlyExtra struct {
	values map[string]string
}

func (e marshalOnlyExtra) MarshalJSON() ([]byte, error) {
	return json.Marshal(e.values)
}

type failingExtra struct{}

func (failingExtra) MarshalJSON() ([]byte, error) {
	return nil, errors.New("cannot encode extra")
}

type fakeStreamer struct {
	gotProvider Provider
	gotModel    Model
}

func (f *fakeStreamer) Stream(ctx context.Context, provider Provider, model Model, prompt Prompt, opts StreamOptions,
) *EventStream[AssistantMessageEvent, *AssistantMessage] {
	f.gotProvider = provider
	f.gotModel = model
	stream, producer := NewEventStream[AssistantMessageEvent, *AssistantMessage](ctx, 2)
	go func() {
		msg := &AssistantMessage{Role: RoleAssistant, StopReason: StopReasonStop}
		_ = producer.Publish(ctx, DoneEvent{Reason: StopReasonStop, Message: *msg})
		_ = producer.Complete(ctx, msg)
	}()
	return stream
}

func newTestClient(fake *fakeStreamer) *Client {
	c := NewClient(map[string]Streamer{"openai-completions": fake})
	_ = c.PutProvider(Provider{Name: "acme", BaseURL: "https://acme.test/v1/", APIKey: "sk-test"})
	_ = c.PutModel(Model{Provider: "acme", ID: "m1", ToolCall: true, ImageInput: true})
	return c
}

func drain(t *testing.T, s *EventStream[AssistantMessageEvent, *AssistantMessage]) *AssistantMessage {
	t.Helper()
	for range s.Events() {
	}
	msg, err := s.Result(context.Background())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	return msg
}

func TestClientRegistryCRUD(t *testing.T) {
	c := NewClient(nil)

	if err := c.PutProvider(Provider{}); err == nil {
		t.Fatal("empty provider name should be rejected")
	}
	if err := c.PutModel(Model{Provider: "p"}); err == nil {
		t.Fatal("model without id should be rejected")
	}

	_ = c.PutProvider(Provider{Name: "b"})
	_ = c.PutProvider(Provider{Name: "a"})
	_ = c.PutModel(Model{Provider: "a", ID: "m2"})
	_ = c.PutModel(Model{Provider: "a", ID: "m1"})

	providers := c.ListProviders()
	if len(providers) != 2 || providers[0].Name != "a" || providers[1].Name != "b" {
		t.Fatalf("ListProviders = %+v", providers)
	}
	models := c.ListModels("a")
	if len(models) != 2 || models[0].ID != "m1" || models[1].ID != "m2" {
		t.Fatalf("ListModels = %+v", models)
	}

	c.DeleteProvider("a")
	if _, ok := c.GetProvider("a"); ok {
		t.Fatal("provider a should be gone")
	}
	if got := c.ListModels("a"); len(got) != 0 {
		t.Fatalf("models of deleted provider should be gone, got %+v", got)
	}
}

func TestClientRegistrySnapshotsDetachMutableConfiguration(t *testing.T) {
	developerRole := true
	reasoningContent := false
	providerPointer := 7
	provider := Provider{
		Name:   "acme",
		APIKey: "sk-test",
		Compat: &OpenAICompat{
			MaxTokensField:        "max_tokens",
			ThinkingFormat:        ThinkingFormatDeepSeek,
			SupportsDeveloperRole: &developerRole,
			RequiresReasoningContentOnAssistantMessages: &reasoningContent,
		},
		Extra: map[string]any{
			"top":     "provider-original",
			"nested":  map[string]any{"value": "provider-original"},
			"list":    []any{map[string]any{"value": "provider-original"}},
			"typed":   []string{"provider-original"},
			"pointer": &providerPointer,
		},
	}
	model := Model{
		Provider: "acme",
		ID:       "m1",
		Extra: map[string]any{
			"top":    "model-original",
			"nested": map[string]any{"value": "model-original"},
			"list":   []any{map[string]any{"value": "model-original"}},
			"typed":  []string{"model-original"},
		},
	}

	c := NewClient(nil)
	if err := c.PutProvider(provider); err != nil {
		t.Fatalf("PutProvider: %v", err)
	}
	if err := c.PutModel(model); err != nil {
		t.Fatalf("PutModel: %v", err)
	}

	mutateProviderConfig(&provider)
	mutateModelConfig(&model)

	gotProvider, ok := c.GetProvider("acme")
	if !ok {
		t.Fatal("provider acme should exist")
	}
	assertOriginalProviderConfig(t, gotProvider)
	gotModel, ok := c.GetModel("acme", "m1")
	if !ok {
		t.Fatal("model acme/m1 should exist")
	}
	assertOriginalModelConfig(t, gotModel)

	// Mutating Get results must not affect either the registry or later List
	// results.
	mutateProviderConfig(&gotProvider)
	mutateModelConfig(&gotModel)
	providers := c.ListProviders()
	if len(providers) != 1 {
		t.Fatalf("ListProviders returned %d entries, want 1", len(providers))
	}
	assertOriginalProviderConfig(t, providers[0])
	models := c.ListModels("acme")
	if len(models) != 1 {
		t.Fatalf("ListModels returned %d entries, want 1", len(models))
	}
	assertOriginalModelConfig(t, models[0])

	// List results are snapshots too.
	mutateProviderConfig(&providers[0])
	mutateModelConfig(&models[0])
	gotProvider, _ = c.GetProvider("acme")
	assertOriginalProviderConfig(t, gotProvider)
	gotModel, _ = c.GetModel("acme", "m1")
	assertOriginalModelConfig(t, gotModel)
}

func TestClientResolveReturnsDetachedSnapshots(t *testing.T) {
	developerRole := true
	c := NewClient(nil)
	if err := c.PutProvider(Provider{
		Name: "acme", APIKey: "sk-test",
		Compat: &OpenAICompat{SupportsDeveloperRole: &developerRole},
		Extra:  map[string]any{"nested": map[string]any{"value": "provider-original"}},
	}); err != nil {
		t.Fatalf("PutProvider: %v", err)
	}
	if err := c.PutModel(Model{
		Provider: "acme", ID: "m1",
		Extra: map[string]any{"nested": map[string]any{"value": "model-original"}},
	}); err != nil {
		t.Fatalf("PutModel: %v", err)
	}

	provider1, model1, err := c.resolveEntry("acme", "m1")
	if err != nil {
		t.Fatalf("first resolveEntry: %v", err)
	}
	provider2, model2, err := c.resolveEntry("acme", "m1")
	if err != nil {
		t.Fatalf("second resolveEntry: %v", err)
	}

	provider1.Extra["nested"].(map[string]any)["value"] = "changed"
	*provider1.Compat.SupportsDeveloperRole = false
	model1.Extra["nested"].(map[string]any)["value"] = "changed"

	if got := provider2.Extra["nested"].(map[string]any)["value"]; got != "provider-original" {
		t.Fatalf("second provider snapshot changed to %v", got)
	}
	if !*provider2.Compat.SupportsDeveloperRole {
		t.Fatal("second provider compat snapshot changed")
	}
	if got := model2.Extra["nested"].(map[string]any)["value"]; got != "model-original" {
		t.Fatalf("second model snapshot changed to %v", got)
	}

	storedProvider, _ := c.GetProvider("acme")
	if got := storedProvider.Extra["nested"].(map[string]any)["value"]; got != "provider-original" {
		t.Fatalf("resolved provider changed registry value to %v", got)
	}
	if !*storedProvider.Compat.SupportsDeveloperRole {
		t.Fatal("resolved provider changed registry compat")
	}
	storedModel, _ := c.GetModel("acme", "m1")
	if got := storedModel.Extra["nested"].(map[string]any)["value"]; got != "model-original" {
		t.Fatalf("resolved model changed registry value to %v", got)
	}
}

func TestClientExtraNormalizationDetachesWireVisiblePrivateState(t *testing.T) {
	embeddedValues := []string{"embedded-original"}
	customValues := map[string]string{"value": "custom-original"}
	const largeInteger int64 = 9_007_199_254_740_993

	c := NewClient(nil)
	if err := c.PutProvider(Provider{
		Name: "acme", APIKey: "sk-test",
		Extra: map[string]any{
			"embedded": privateEmbeddedExtra{privateExtraFields{Values: embeddedValues}},
			"custom":   marshalOnlyExtra{values: customValues},
			"large":    largeInteger,
		},
	}); err != nil {
		t.Fatalf("PutProvider: %v", err)
	}

	embeddedValues[0] = "mutated"
	customValues["value"] = "mutated"
	got, ok := c.GetProvider("acme")
	if !ok {
		t.Fatal("provider acme should exist")
	}
	embedded := got.Extra["embedded"].(map[string]any)["values"].([]any)
	if embedded[0] != "embedded-original" {
		t.Fatalf("embedded private state changed to %v", embedded[0])
	}
	custom := got.Extra["custom"].(map[string]any)
	if custom["value"] != "custom-original" {
		t.Fatalf("custom marshaler state changed to %v", custom["value"])
	}
	large, ok := got.Extra["large"].(json.Number)
	if !ok || large.String() != "9007199254740993" {
		t.Fatalf("large integer = %T(%v), want exact json.Number", got.Extra["large"], got.Extra["large"])
	}
	wire, err := json.Marshal(got.Extra)
	if err != nil || !strings.Contains(string(wire), `"large":9007199254740993`) {
		t.Fatalf("normalized wire JSON = %s, err = %v", wire, err)
	}

	// The normalized output is a snapshot too.
	embedded[0] = "changed snapshot"
	custom["value"] = "changed snapshot"
	again, _ := c.GetProvider("acme")
	if value := again.Extra["embedded"].(map[string]any)["values"].([]any)[0]; value != "embedded-original" {
		t.Fatalf("mutating embedded snapshot changed registry to %v", value)
	}
	if value := again.Extra["custom"].(map[string]any)["value"]; value != "custom-original" {
		t.Fatalf("mutating custom snapshot changed registry to %v", value)
	}
}

func TestClientRejectsUnencodableExtraWithoutReplacingEntry(t *testing.T) {
	c := NewClient(nil)
	if err := c.PutProvider(Provider{Name: "acme", APIKey: "original"}); err != nil {
		t.Fatalf("initial PutProvider: %v", err)
	}
	if err := c.PutProvider(Provider{
		Name: "acme", APIKey: "replacement", Extra: map[string]any{"bad": failingExtra{}},
	}); err == nil || !strings.Contains(err.Error(), "cannot encode extra") {
		t.Fatalf("invalid PutProvider error = %v", err)
	}
	got, ok := c.GetProvider("acme")
	if !ok || got.APIKey != "original" || got.Extra != nil {
		t.Fatalf("failed replacement changed registry: %+v", got)
	}
}

func TestClientProviderCompatInputCanChangeConcurrentlyAfterPut(t *testing.T) {
	developerRole := true
	reasoningContent := false
	provider := Provider{
		Name:   "acme",
		APIKey: "sk-test",
		Compat: &OpenAICompat{
			MaxTokensField:        "max_tokens",
			SupportsDeveloperRole: &developerRole,
			RequiresReasoningContentOnAssistantMessages: &reasoningContent,
		},
	}
	c := NewClient(nil)
	if err := c.PutProvider(provider); err != nil {
		t.Fatalf("PutProvider: %v", err)
	}
	if err := c.PutModel(Model{Provider: "acme", ID: "m1"}); err != nil {
		t.Fatalf("PutModel: %v", err)
	}

	const iterations = 2000
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for i := range iterations {
			provider.Compat.MaxTokensField = "changed"
			*provider.Compat.SupportsDeveloperRole = false
			*provider.Compat.RequiresReasoningContentOnAssistantMessages = true
			if i%2 == 0 {
				provider.Compat.MaxTokensField = "changed-again"
				*provider.Compat.SupportsDeveloperRole = true
				*provider.Compat.RequiresReasoningContentOnAssistantMessages = false
			}
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for range iterations {
			got, ok := c.GetProvider("acme")
			if !ok || got.Compat == nil {
				t.Errorf("GetProvider returned an incomplete provider: %+v", got)
				return
			}
			if got.Compat.MaxTokensField != "max_tokens" || !*got.Compat.SupportsDeveloperRole ||
				*got.Compat.RequiresReasoningContentOnAssistantMessages {
				t.Errorf("GetProvider observed caller mutation: %+v", got.Compat)
				return
			}
			listed := c.ListProviders()
			if len(listed) != 1 || listed[0].Compat == nil {
				t.Errorf("ListProviders returned incomplete data: %+v", listed)
				return
			}
			resolved, _, err := c.resolveEntry("acme", "m1")
			if err != nil || resolved.Compat == nil {
				t.Errorf("resolveEntry failed: provider=%+v err=%v", resolved, err)
				return
			}
			_ = resolved.Compat.MaxTokensField
			_ = *resolved.Compat.SupportsDeveloperRole
			_ = *resolved.Compat.RequiresReasoningContentOnAssistantMessages
		}
	}()
	close(start)
	wg.Wait()
}

func mutateProviderConfig(provider *Provider) {
	provider.Extra["top"] = "mutated"
	provider.Extra["nested"].(map[string]any)["value"] = "mutated"
	provider.Extra["list"].([]any)[0].(map[string]any)["value"] = "mutated"
	switch typed := provider.Extra["typed"].(type) {
	case []string:
		typed[0] = "mutated"
	case []any:
		typed[0] = "mutated"
	}
	if pointer, ok := provider.Extra["pointer"].(*int); ok {
		*pointer = 99
	} else {
		provider.Extra["pointer"] = "mutated"
	}
	provider.Compat.MaxTokensField = "mutated"
	provider.Compat.ThinkingFormat = ThinkingFormatQwen
	*provider.Compat.SupportsDeveloperRole = false
	*provider.Compat.RequiresReasoningContentOnAssistantMessages = true
}

func mutateModelConfig(model *Model) {
	model.Extra["top"] = "mutated"
	model.Extra["nested"].(map[string]any)["value"] = "mutated"
	model.Extra["list"].([]any)[0].(map[string]any)["value"] = "mutated"
	switch typed := model.Extra["typed"].(type) {
	case []string:
		typed[0] = "mutated"
	case []any:
		typed[0] = "mutated"
	}
}

func assertOriginalProviderConfig(t *testing.T, provider Provider) {
	t.Helper()
	if got := provider.Extra["top"]; got != "provider-original" {
		t.Fatalf("provider top extra = %v, want provider-original", got)
	}
	if got := provider.Extra["nested"].(map[string]any)["value"]; got != "provider-original" {
		t.Fatalf("provider nested extra = %v, want provider-original", got)
	}
	if got := provider.Extra["list"].([]any)[0].(map[string]any)["value"]; got != "provider-original" {
		t.Fatalf("provider list extra = %v, want provider-original", got)
	}
	if got := provider.Extra["typed"].([]any)[0]; got != "provider-original" {
		t.Fatalf("provider typed slice extra = %v, want provider-original", got)
	}
	if got, ok := provider.Extra["pointer"].(json.Number); !ok || got.String() != "7" {
		t.Fatalf("provider pointer extra = %v, want 7", got)
	}
	if provider.Compat == nil {
		t.Fatal("provider compat is nil")
	}
	if provider.Compat.MaxTokensField != "max_tokens" || provider.Compat.ThinkingFormat != ThinkingFormatDeepSeek {
		t.Fatalf("provider compat scalar fields = %+v", provider.Compat)
	}
	if provider.Compat.SupportsDeveloperRole == nil || !*provider.Compat.SupportsDeveloperRole {
		t.Fatalf("SupportsDeveloperRole = %v, want true", provider.Compat.SupportsDeveloperRole)
	}
	if provider.Compat.RequiresReasoningContentOnAssistantMessages == nil ||
		*provider.Compat.RequiresReasoningContentOnAssistantMessages {
		t.Fatalf("RequiresReasoningContentOnAssistantMessages = %v, want false",
			provider.Compat.RequiresReasoningContentOnAssistantMessages)
	}
}

func assertOriginalModelConfig(t *testing.T, model Model) {
	t.Helper()
	if got := model.Extra["top"]; got != "model-original" {
		t.Fatalf("model top extra = %v, want model-original", got)
	}
	if got := model.Extra["nested"].(map[string]any)["value"]; got != "model-original" {
		t.Fatalf("model nested extra = %v, want model-original", got)
	}
	if got := model.Extra["list"].([]any)[0].(map[string]any)["value"]; got != "model-original" {
		t.Fatalf("model list extra = %v, want model-original", got)
	}
	if got := model.Extra["typed"].([]any)[0]; got != "model-original" {
		t.Fatalf("model typed slice extra = %v, want model-original", got)
	}
}

func TestClientStreamUnknownLookups(t *testing.T) {
	c := newTestClient(&fakeStreamer{})

	for _, tc := range []struct {
		provider, model, want string
	}{
		{"nope", "m1", `unknown provider "nope"`},
		{"acme", "nope", `unknown model "nope"`},
	} {
		msg := drain(t, c.Stream(context.Background(), tc.provider, tc.model, Prompt{}, StreamOptions{}))
		if msg.StopReason != StopReasonError || !strings.Contains(msg.ErrorMessage, tc.want) {
			t.Fatalf("(%s,%s): got stop=%s err=%q, want %q", tc.provider, tc.model, msg.StopReason, msg.ErrorMessage, tc.want)
		}
	}
}

func TestClientStreamMissingKey(t *testing.T) {
	c := newTestClient(&fakeStreamer{})
	_ = c.PutProvider(Provider{Name: "nokey", APIKeyEnv: "DAKSHA_TEST_ABSENT_KEY"})
	_ = c.PutModel(Model{Provider: "nokey", ID: "m1"})

	msg := drain(t, c.Stream(context.Background(), "nokey", "m1", Prompt{}, StreamOptions{}))
	if msg.StopReason != StopReasonError || !strings.Contains(msg.ErrorMessage, "no API key") {
		t.Fatalf("got stop=%s err=%q", msg.StopReason, msg.ErrorMessage)
	}
}

func TestClientKeyEnvFallback(t *testing.T) {
	fake := &fakeStreamer{}
	c := newTestClient(fake)
	t.Setenv("DAKSHA_TEST_KEY", "sk-from-env")
	_ = c.PutProvider(Provider{Name: "envkey", APIKeyEnv: "DAKSHA_TEST_KEY"})
	_ = c.PutModel(Model{Provider: "envkey", ID: "m1"})

	msg := drain(t, c.Stream(context.Background(), "envkey", "m1", Prompt{}, StreamOptions{}))
	if msg.StopReason != StopReasonStop {
		t.Fatalf("stream failed: %s %q", msg.StopReason, msg.ErrorMessage)
	}
	if fake.gotProvider.APIKey != "sk-from-env" {
		t.Fatalf("adapter got APIKey %q, want env fallback", fake.gotProvider.APIKey)
	}
}

func TestClientCapabilityGate(t *testing.T) {
	c := newTestClient(&fakeStreamer{})
	_ = c.PutModel(Model{Provider: "acme", ID: "plain"}) // no capabilities

	tools := Prompt{Tools: []ToolDefinition{{Name: "t"}}}
	msg := drain(t, c.Stream(context.Background(), "acme", "plain", tools, StreamOptions{}))
	if msg.StopReason != StopReasonError || !strings.Contains(msg.ErrorMessage, "tool calls") {
		t.Fatalf("tools should be gated: %s %q", msg.StopReason, msg.ErrorMessage)
	}

	image := Prompt{Messages: []Message{&UserMessage{
		Role:    RoleUser,
		Content: []UserContent{&ImageContent{Type: ContentTypeImage, URL: "https://x.test/a.png"}},
	}}}
	msg = drain(t, c.Stream(context.Background(), "acme", "plain", image, StreamOptions{}))
	if msg.StopReason != StopReasonError || !strings.Contains(msg.ErrorMessage, "image input") {
		t.Fatalf("image should be gated: %s %q", msg.StopReason, msg.ErrorMessage)
	}

	audio := Prompt{Messages: []Message{&UserMessage{
		Role:    RoleUser,
		Content: []UserContent{&AudioContent{Type: ContentTypeAudio, URL: "https://x.test/a.mp3"}},
	}}}
	msg = drain(t, c.Stream(context.Background(), "acme", "plain", audio, StreamOptions{}))
	if msg.StopReason != StopReasonError || !strings.Contains(msg.ErrorMessage, "audio input") {
		t.Fatalf("audio should be gated: %s %q", msg.StopReason, msg.ErrorMessage)
	}
}

func TestClientDispatchSnapshot(t *testing.T) {
	fake := &fakeStreamer{}
	c := newTestClient(fake)

	msg, err := c.Complete(context.Background(), "acme", "m1", Prompt{}, StreamOptions{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if msg.StopReason != StopReasonStop {
		t.Fatalf("StopReason = %s", msg.StopReason)
	}
	if fake.gotProvider.Name != "acme" || fake.gotProvider.APIKey != "sk-test" {
		t.Fatalf("adapter got provider %+v", fake.gotProvider)
	}
	if fake.gotProvider.API != "openai-completions" {
		t.Fatalf("empty API should default, got %q", fake.gotProvider.API)
	}
	if fake.gotModel.ID != "m1" {
		t.Fatalf("adapter got model %+v", fake.gotModel)
	}
}

func TestClientNoAdapterForProtocol(t *testing.T) {
	c := NewClient(nil)
	_ = c.PutProvider(Provider{Name: "acme", APIKey: "sk", API: "anthropic-messages"})
	_ = c.PutModel(Model{Provider: "acme", ID: "m1"})

	msg := drain(t, c.Stream(context.Background(), "acme", "m1", Prompt{}, StreamOptions{}))
	if msg.StopReason != StopReasonError || !strings.Contains(msg.ErrorMessage, "no adapter") {
		t.Fatalf("got %s %q", msg.StopReason, msg.ErrorMessage)
	}
}
