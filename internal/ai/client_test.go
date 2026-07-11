package ai

import (
	"context"
	"strings"
	"testing"
)

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
