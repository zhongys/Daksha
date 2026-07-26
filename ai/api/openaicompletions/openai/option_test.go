package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

func TestWithJSONSetSnapshotsValueAtConstruction(t *testing.T) {
	value := map[string]any{"nested": map[string]any{"answer": "before"}}
	option, err := WithJSONSet("extra", value)
	if err != nil {
		t.Fatalf("WithJSONSet: %v", err)
	}
	value["nested"].(map[string]any)["answer"] = "after"

	cfg, err := NewRequestConfig(context.Background(), http.MethodPost, "/", map[string]any{}, nil, option)
	if err != nil {
		t.Fatalf("NewRequestConfig: %v", err)
	}
	body, err := io.ReadAll(cfg.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	got := decoded["extra"].(map[string]any)["nested"].(map[string]any)["answer"]
	if got != "before" {
		t.Fatalf("snapshotted value = %q, want before", got)
	}
}

func TestWithJSONSetReturnsSnapshotErrorImmediately(t *testing.T) {
	option, err := WithJSONSet("bad", func() {})
	if err == nil {
		t.Fatal("WithJSONSet succeeded for an unserializable value")
	}
	if option != nil {
		t.Fatalf("WithJSONSet option = %#v, want nil on error", option)
	}
}

func TestJSONBodyOptionsRejectNullBody(t *testing.T) {
	setOption, err := WithJSONSet("stream", true)
	if err != nil {
		t.Fatalf("WithJSONSet: %v", err)
	}
	for name, option := range map[string]RequestOption{
		"set": setOption,
		"del": WithJSONDel("stream"),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewRequestConfig(
				context.Background(), http.MethodPost, "/", json.RawMessage("null"), nil, option,
			); err == nil {
				t.Fatal("NewRequestConfig succeeded for a null JSON body")
			}
		})
	}
}

func TestWithRequestBodySnapshotsBytesAtConstruction(t *testing.T) {
	body := []byte(`{"value":"before"}`)
	option := WithRequestBody("application/json", body)
	copy(body, []byte(`{"value":"after!"}`))

	cfg, err := NewRequestConfig(context.Background(), http.MethodPost, "/", nil, nil, option)
	if err != nil {
		t.Fatalf("NewRequestConfig: %v", err)
	}
	got, err := io.ReadAll(cfg.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if !bytes.Equal(got, []byte(`{"value":"before"}`)) {
		t.Fatalf("body = %q, want frozen bytes", got)
	}
}

func TestClientConstructorsSnapshotOptionSlices(t *testing.T) {
	first := WithHeader("X-Choice", "first")
	second := WithHeader("X-Choice", "second")
	opts := []RequestOption{first}
	client := NewClient(opts...)
	opts[0] = second

	for name, stored := range map[string][]RequestOption{
		"client":      client.Options,
		"chat":        client.Chat.Options,
		"completions": client.Chat.Completions.Options,
		"embeddings":  client.Embeddings.Options,
	} {
		cfg, err := NewRequestConfig(context.Background(), http.MethodGet, "/", nil, nil, stored...)
		if err != nil {
			t.Fatalf("%s NewRequestConfig: %v", name, err)
		}
		if got := cfg.Request.Header.Get("X-Choice"); got != "first" {
			t.Errorf("%s option = %q, want first", name, got)
		}
	}
}
