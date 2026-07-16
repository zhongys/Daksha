package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
	"time"
)

// Client is the application-facing entry point of the ai layer. It owns a
// concurrency-safe in-memory registry of providers and models plus a route
// table from protocol name to adapter, and dispatches each turn to the right
// adapter with a value snapshot of the configuration.
//
// The registry holds data only — persistence (database, config service) is
// the application's concern: load entries at startup, call the CRUD methods
// on changes. Adding, updating or deleting entries takes effect on the next
// Stream call; in-flight requests finish on the snapshot they started with.
type Client struct {
	mu        sync.RWMutex
	providers map[string]Provider
	models    map[string]map[string]Model // provider name -> model id

	adapters  map[string]Streamer
	embedders map[string]Embedder
}

// NewClient builds a Client with an explicit protocol route table, e.g.
//
//	ai.NewClient(map[string]ai.Streamer{
//		"openai-completions": openaicompletions.NewStreamer(),
//	})
//
// Adapters are stateless protocol translators, so one instance per protocol
// serves every provider speaking it.
func NewClient(adapters map[string]Streamer) *Client {
	c := &Client{
		providers: map[string]Provider{},
		models:    map[string]map[string]Model{},
		adapters:  map[string]Streamer{},
		embedders: map[string]Embedder{},
	}
	for name, s := range adapters {
		c.adapters[name] = s
	}
	return c
}

// PutProvider inserts or replaces a provider entry.
func (c *Client) PutProvider(p Provider) error {
	if p.Name == "" {
		return fmt.Errorf("ai: provider name is empty")
	}
	var err error
	p.Extra, err = normalizeExtra(p.Extra)
	if err != nil {
		return fmt.Errorf("ai: provider %q extra: %w", p.Name, err)
	}
	p.Compat = cloneOpenAICompat(p.Compat)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.providers[p.Name] = p
	return nil
}

// DeleteProvider removes a provider and its models. In-flight requests are
// unaffected; new calls against the name fail loudly.
func (c *Client) DeleteProvider(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.providers, name)
	delete(c.models, name)
}

// GetProvider returns a snapshot of a provider entry.
func (c *Client) GetProvider(name string) (Provider, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.providers[name]
	return cloneProvider(p), ok
}

// ListProviders returns snapshots of all provider entries, sorted by name.
func (c *Client) ListProviders() []Provider {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Provider, 0, len(c.providers))
	for _, p := range c.providers {
		out = append(out, cloneProvider(p))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// PutModel inserts or replaces a model entry, keyed by (Provider, ID). The
// provider does not have to exist yet; Stream checks it at call time.
func (c *Client) PutModel(m Model) error {
	if m.Provider == "" || m.ID == "" {
		return fmt.Errorf("ai: model provider and id must both be set")
	}
	var err error
	m.Extra, err = normalizeExtra(m.Extra)
	if err != nil {
		return fmt.Errorf("ai: model %q on provider %q extra: %w", m.ID, m.Provider, err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	byID := c.models[m.Provider]
	if byID == nil {
		byID = map[string]Model{}
		c.models[m.Provider] = byID
	}
	byID[m.ID] = m
	return nil
}

// DeleteModel removes a model entry.
func (c *Client) DeleteModel(providerName, modelID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.models[providerName], modelID)
}

// GetModel returns a snapshot of a model entry.
func (c *Client) GetModel(providerName, modelID string) (Model, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.models[providerName][modelID]
	return cloneModel(m), ok
}

// ListModels returns snapshots of a provider's models, sorted by id.
func (c *Client) ListModels(providerName string) []Model {
	c.mu.RLock()
	defer c.mu.RUnlock()
	byID := c.models[providerName]
	out := make([]Model, 0, len(byID))
	for _, m := range byID {
		out = append(out, cloneModel(m))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Stream runs one LLM turn against a registered provider and model.
//
// It resolves configuration snapshots and the API key at call time, enforces
// the model's content capability switches, and routes to the adapter named
// by the provider's protocol. Per the Streamer contract the call never
// fails: lookup, key and capability errors are delivered as an ErrorEvent
// plus a final message with StopReason error.
func (c *Client) Stream(ctx context.Context, providerName, modelID string, prompt Prompt, opts StreamOptions,
) *EventStream[AssistantMessageEvent, *AssistantMessage] {
	provider, model, adapter, err := c.resolve(providerName, modelID)
	if err == nil {
		err = checkCapabilities(model, prompt)
	}
	if err != nil {
		return failedStream(ctx, provider.API, providerName, modelID, err)
	}
	return adapter.Stream(ctx, provider, model, prompt, opts)
}

// Complete runs one turn and blocks until the final message. The returned
// error only reflects stream-level failures (e.g. context cancellation);
// model-side failures arrive in the message itself — check StopReason.
func (c *Client) Complete(ctx context.Context, providerName, modelID string, prompt Prompt, opts StreamOptions,
) (*AssistantMessage, error) {
	stream := c.Stream(ctx, providerName, modelID, prompt, opts)
	for range stream.Events() {
	}
	return stream.Result(ctx)
}

// resolve snapshots the provider and model, resolves the API key and picks
// the adapter. The returned provider is a value copy with APIKey filled in,
// so key rotation and registry edits never touch requests already running.
func (c *Client) resolve(providerName, modelID string) (Provider, Model, Streamer, error) {
	provider, model, err := c.resolveEntry(providerName, modelID)
	if err != nil {
		return provider, model, nil, err
	}
	adapter, ok := c.adapters[provider.API]
	if !ok {
		return provider, model, nil, fmt.Errorf("ai: no adapter registered for protocol %q", provider.API)
	}
	return provider, model, adapter, nil
}

// resolveEntry snapshots the provider and model and resolves the API key —
// the adapter-independent half of resolve, shared by Stream and Embed. The
// returned provider always has API defaulted and APIKey filled in.
func (c *Client) resolveEntry(providerName, modelID string) (Provider, Model, error) {
	c.mu.RLock()
	provider, okP := c.providers[providerName]
	model, okM := c.models[providerName][modelID]
	provider = cloneProvider(provider)
	model = cloneModel(model)
	c.mu.RUnlock()

	if !okP {
		return provider, model, fmt.Errorf("ai: unknown provider %q", providerName)
	}
	if !okM {
		return provider, model, fmt.Errorf("ai: unknown model %q on provider %q", modelID, providerName)
	}

	if provider.APIKey == "" && provider.APIKeyEnv != "" {
		provider.APIKey = os.Getenv(provider.APIKeyEnv)
	}
	if provider.APIKey == "" {
		return provider, model, fmt.Errorf("ai: no API key for provider %q (set APIKey or env %s)",
			providerName, provider.APIKeyEnv)
	}

	if provider.API == "" {
		provider.API = "openai-completions"
	}
	return provider, model, nil
}

// cloneProvider returns a detached provider value. Registry entries and
// request snapshots must never share mutable configuration with their caller.
func cloneProvider(provider Provider) Provider {
	provider.Extra = cloneExtra(provider.Extra)
	provider.Compat = cloneOpenAICompat(provider.Compat)
	return provider
}

// cloneModel returns a detached model value for the same reason as
// cloneProvider.
func cloneModel(model Model) Model {
	model.Extra = cloneExtra(model.Extra)
	return model
}

func cloneOpenAICompat(compat *OpenAICompat) *OpenAICompat {
	if compat == nil {
		return nil
	}
	cloned := *compat
	cloned.SupportsDeveloperRole = cloneBool(compat.SupportsDeveloperRole)
	cloned.RequiresReasoningContentOnAssistantMessages = cloneBool(compat.RequiresReasoningContentOnAssistantMessages)
	return &cloned
}

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

// normalizeExtra freezes arbitrary JSON-encodable input at registration time
// into the same map/slice/scalar tree that will be sent on the wire. UseNumber
// preserves large integers exactly. Concrete input types are intentionally not
// retained: Extra is request JSON, and wire-equivalent immutable snapshots are
// the registry contract.
func normalizeExtra(extra map[string]any) (map[string]any, error) {
	if extra == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(extra)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var normalized map[string]any
	if err := decoder.Decode(&normalized); err != nil {
		return nil, err
	}
	return normalized, nil
}

// cloneExtra copies a normalized JSON tree. Registry entries only reach this
// path after normalizeExtra succeeds, so Get/List/resolve cannot encounter an
// unreportable cloning error.
func cloneExtra(extra map[string]any) map[string]any {
	if extra == nil {
		return nil
	}
	cloned := make(map[string]any, len(extra))
	for key, value := range extra {
		cloned[key] = cloneExtraValue(value)
	}
	return cloned
}

func cloneExtraValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneExtra(value)
	case []any:
		cloned := make([]any, len(value))
		for i, item := range value {
			cloned[i] = cloneExtraValue(item)
		}
		return cloned
	default:
		return value
	}
}

// checkCapabilities enforces the model's content switches before dispatch.
// Tools and media are semantic content — silently dropping them would change
// the request's meaning — so a mismatch is a loud configuration error.
// Tuning knobs (ReasoningEffort on a non-reasoning model) are the adapter's
// business and degrade silently there.
func checkCapabilities(model Model, prompt Prompt) error {
	if len(prompt.Tools) > 0 && !model.ToolCall {
		return fmt.Errorf("ai: model %q does not support tool calls", model.ID)
	}
	for _, msg := range prompt.Messages {
		switch m := msg.(type) {
		case *UserMessage:
			for _, content := range m.Content {
				if err := checkMediaCapability(model, content); err != nil {
					return err
				}
			}
		case ToolResult:
			_, _, contents, _ := m.ToolResultData()
			for _, content := range contents {
				if err := checkMediaCapability(model, content); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func checkMediaCapability(model Model, content any) error {
	switch content.(type) {
	case *ImageContent:
		if !model.ImageInput {
			return fmt.Errorf("ai: model %q does not support image input", model.ID)
		}
	case *AudioContent:
		if !model.AudioInput {
			return fmt.Errorf("ai: model %q does not support audio input", model.ID)
		}
	case *VideoContent:
		if !model.VideoInput {
			return fmt.Errorf("ai: model %q does not support video input", model.ID)
		}
	}
	return nil
}

// failedStream delivers a pre-dispatch failure through the standard event
// stream contract: an ErrorEvent followed by a final error message.
func failedStream(ctx context.Context, api, providerName, modelID string, err error,
) *EventStream[AssistantMessageEvent, *AssistantMessage] {
	if api == "" {
		api = "openai-completions"
	}
	msg := &AssistantMessage{
		Role:         RoleAssistant,
		Api:          api,
		Provider:     providerName,
		Model:        modelID,
		StopReason:   StopReasonError,
		ErrorMessage: err.Error(),
		Timestamp:    time.Now().UnixMilli(),
	}
	stream, producer := NewEventStream[AssistantMessageEvent, *AssistantMessage](ctx, 2)
	go func() {
		_ = producer.Publish(ctx, ErrorEvent{Reason: StopReasonError, Error: *msg})
		_ = producer.Complete(ctx, msg)
	}()
	return stream
}
