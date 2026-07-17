package openaicompletions

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/zhongys/Daksha/ai"
)

type blockingJSONSnapshot struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (v *blockingJSONSnapshot) MarshalJSON() ([]byte, error) {
	v.once.Do(func() { close(v.entered) })
	<-v.release
	return []byte(`{"state":"captured"}`), nil
}

func TestStreamerSnapshotsAllWireInputsBeforeReturning(t *testing.T) {
	requests := make(chan []byte, 1)
	respond := make(chan struct{})
	var respondOnce sync.Once
	releaseResponse := func() { respondOnce.Do(func() { close(respond) }) }
	defer releaseResponse()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		requests <- body
		<-respond
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	probe := &blockingJSONSnapshot{entered: make(chan struct{}), release: make(chan struct{})}
	user := &ai.UserMessage{
		Role: ai.RoleUser,
		Content: []ai.UserContent{
			&ai.TextContent{Type: ai.ContentTypeText, Text: "before"},
		},
	}
	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "title": "before"},
		},
	}
	temperature := 0.25
	maxTokens := int64(64)
	stops := []string{"before"}
	providerExtra := map[string]any{"provider_value": map[string]any{"state": "before"}}
	modelExtra := map[string]any{"model_value": map[string]any{"state": "before"}}
	requestExtra := map[string]any{
		"request_value":  map[string]any{"state": "before"},
		"snapshot_probe": probe,
	}
	compat := &ai.OpenAICompat{MaxTokensField: "max_tokens"}
	prompt := ai.Prompt{
		Messages: []ai.Message{user},
		Tools:    []ai.ToolDefinition{{Name: "read", Parameters: parameters}},
	}
	opts := ai.StreamOptions{
		Temperature:   &temperature,
		MaxTokens:     &maxTokens,
		StopSequences: stops,
		Extra:         requestExtra,
	}

	returned := make(chan *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage], 1)
	go func() {
		returned <- NewStreamer().Stream(context.Background(), ai.Provider{
			Name: "p", BaseURL: server.URL + "/", APIKey: "k", Compat: compat, Extra: providerExtra,
		}, ai.Model{Provider: "p", ID: "m", ToolCall: true, Extra: modelExtra}, prompt, opts)
	}()

	select {
	case <-probe.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("request inputs were not serialized")
	}
	select {
	case <-returned:
		close(probe.release)
		t.Fatal("Stream returned before its caller-owned inputs were snapshotted")
	default:
	}
	close(probe.release)

	var stream *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage]
	select {
	case stream = <-returned:
	case <-time.After(3 * time.Second):
		t.Fatal("Stream did not return after snapshot serialization completed")
	}

	user.Content[0].(*ai.TextContent).Text = "after"
	parameters["properties"].(map[string]any)["path"].(map[string]any)["title"] = "after"
	temperature = 0.9
	maxTokens = 999
	stops[0] = "after"
	providerExtra["provider_value"].(map[string]any)["state"] = "after"
	modelExtra["model_value"].(map[string]any)["state"] = "after"
	requestExtra["request_value"].(map[string]any)["state"] = "after"
	compat.MaxTokensField = "max_completion_tokens"

	var raw []byte
	select {
	case raw = <-requests:
	case <-time.After(3 * time.Second):
		t.Fatal("server did not receive request")
	}
	releaseResponse()

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if got := body["temperature"]; got != 0.25 {
		t.Errorf("temperature = %#v, want 0.25", got)
	}
	if got := body["max_tokens"]; got != float64(64) {
		t.Errorf("max_tokens = %#v, want 64", got)
	}
	if _, exists := body["max_completion_tokens"]; exists {
		t.Error("request used Compat mutation made after Stream returned")
	}
	if got := body["stop"].([]any)[0]; got != "before" {
		t.Errorf("stop = %#v, want before", got)
	}
	messages := body["messages"].([]any)
	if got := messages[0].(map[string]any)["content"]; got != "before" {
		t.Errorf("message content = %#v, want before", got)
	}
	tools := body["tools"].([]any)
	function := tools[0].(map[string]any)["function"].(map[string]any)
	schema := function["parameters"].(map[string]any)
	if got := schema["properties"].(map[string]any)["path"].(map[string]any)["title"]; got != "before" {
		t.Errorf("tool schema title = %#v, want before", got)
	}
	for _, key := range []string{"provider_value", "model_value", "request_value"} {
		if got := body[key].(map[string]any)["state"]; got != "before" {
			t.Errorf("%s.state = %#v, want before", key, got)
		}
	}
	if got := body["snapshot_probe"].(map[string]any)["state"]; got != "captured" {
		t.Errorf("snapshot_probe.state = %#v, want captured", got)
	}

	events, message, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if message.StopReason != ai.StopReasonStop || len(events) == 0 {
		t.Fatalf("stream result = %#v, events = %v", message, eventTypes(events))
	}
}
