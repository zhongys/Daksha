package openaicompletions

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zhongys/Daksha/ai"
)

// embedServer returns an httptest server that captures the request body and
// responds with the given embeddings payload.
func embedServer(t *testing.T, response string, gotBody *[]byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q, want /embeddings", r.URL.Path)
		}
		if gotBody != nil {
			b, _ := io.ReadAll(r.Body)
			*gotBody = b
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(response))
	}))
}

func TestEmbedderReordersByIndexAndBills(t *testing.T) {
	var gotBody []byte
	server := embedServer(t, `{
		"object": "list",
		"model": "text-embedding-v4",
		"data": [
			{"object": "embedding", "index": 1, "embedding": [0.4, 0.5]},
			{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
		],
		"usage": {"prompt_tokens": 7, "total_tokens": 7}
	}`, &gotBody)
	defer server.Close()

	provider := ai.Provider{Name: "test", BaseURL: server.URL + "/", APIKey: "k", Extra: map[string]any{"text_type": "document"}}
	model := ai.Model{Provider: "test", ID: "text-embedding-v4", Pricing: ai.Pricing{Input: 500}}

	res, err := NewEmbedder().Embed(t.Context(), provider, model, []string{"a", "b"}, ai.EmbedOptions{Dimensions: 2})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(res.Vectors) != 2 {
		t.Fatalf("vectors = %d, want 2", len(res.Vectors))
	}
	if res.Vectors[0][0] != 0.1 || res.Vectors[1][0] != 0.4 {
		t.Errorf("vectors not reordered by index: %v", res.Vectors)
	}
	if res.Usage.Input != 7 || res.Usage.Cost.Total != 7*500 {
		t.Errorf("usage = %+v, want input 7 cost 3500", res.Usage)
	}

	var req map[string]any
	if err := json.Unmarshal(gotBody, &req); err != nil {
		t.Fatalf("request body: %v", err)
	}
	if req["model"] != "text-embedding-v4" {
		t.Errorf("request model = %v", req["model"])
	}
	if req["encoding_format"] != "float" {
		t.Errorf("encoding_format = %v", req["encoding_format"])
	}
	if req["dimensions"] != float64(2) {
		t.Errorf("dimensions = %v", req["dimensions"])
	}
	if req["text_type"] != "document" {
		t.Errorf("provider extra not merged: %v", req["text_type"])
	}
}

func TestEmbedderMissingVectorFails(t *testing.T) {
	server := embedServer(t, `{
		"object": "list",
		"data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
		"usage": {"prompt_tokens": 3, "total_tokens": 3}
	}`, nil)
	defer server.Close()

	provider := ai.Provider{Name: "test", BaseURL: server.URL + "/", APIKey: "k"}
	model := ai.Model{Provider: "test", ID: "m"}

	_, err := NewEmbedder().Embed(t.Context(), provider, model, []string{"a", "b"}, ai.EmbedOptions{})
	if err == nil {
		t.Fatal("want error for missing embedding, got nil")
	}
}

func TestClientEmbedRoutesThroughRegistry(t *testing.T) {
	server := embedServer(t, `{
		"object": "list",
		"data": [{"object": "embedding", "index": 0, "embedding": [0.9]}],
		"usage": {"prompt_tokens": 2, "total_tokens": 2}
	}`, nil)
	defer server.Close()

	client := ai.NewClient(map[string]ai.Streamer{"openai-completions": NewStreamer()})
	client.PutEmbedder("openai-completions", NewEmbedder())
	if err := client.PutProvider(ai.Provider{Name: "test", BaseURL: server.URL + "/", APIKey: "k"}); err != nil {
		t.Fatal(err)
	}
	if err := client.PutModel(ai.Model{Provider: "test", ID: "emb"}); err != nil {
		t.Fatal(err)
	}

	res, err := client.Embed(t.Context(), "test", "emb", []string{"x"}, ai.EmbedOptions{})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(res.Vectors) != 1 || res.Vectors[0][0] != 0.9 {
		t.Errorf("vectors = %v", res.Vectors)
	}

	// Empty inputs short-circuit without a request.
	empty, err := client.Embed(t.Context(), "test", "emb", nil, ai.EmbedOptions{})
	if err != nil || len(empty.Vectors) != 0 {
		t.Errorf("empty inputs: res=%+v err=%v", empty, err)
	}

	// Unregistered protocol fails loudly.
	bare := ai.NewClient(map[string]ai.Streamer{})
	_ = bare.PutProvider(ai.Provider{Name: "p", BaseURL: "http://x/", APIKey: "k"})
	_ = bare.PutModel(ai.Model{Provider: "p", ID: "m"})
	if _, err := bare.Embed(t.Context(), "p", "m", []string{"x"}, ai.EmbedOptions{}); err == nil {
		t.Error("want no-embedder error, got nil")
	}
}
