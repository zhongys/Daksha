package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestStreamSkipsCommentOnlySSEBlocks(t *testing.T) {
	res := &http.Response{Body: io.NopCloser(strings.NewReader(
		": keepalive\n\n\n" +
			"data: {\"value\":1}\n\n" +
			"data: [DONE]\n\n",
	))}
	stream := NewStream[struct {
		Value int `json:"value"`
	}](NewDecoder(res), nil)
	defer stream.Close()

	if !stream.Next() {
		t.Fatalf("Next() = false, err = %v", stream.Err())
	}
	if got := stream.Current().Value; got != 1 {
		t.Fatalf("value = %d, want 1", got)
	}
	if stream.Next() {
		t.Fatal("Next() returned an event after [DONE]")
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

func TestStreamingTimeoutLivesUntilResponseBodyCloses(t *testing.T) {
	allowChunk := make(chan struct{})
	requestDone := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(requestDone)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()

		// Do not send the first chunk until NewStreaming (and therefore Execute)
		// has returned. With the old deferred cancel, the request context wins
		// this select and the client can never receive the chunk.
		select {
		case <-allowChunk:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, "data: {\"id\":\"chunk-1\",\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()

	service := NewChatCompletionService(
		WithBaseURL(server.URL+"/"),
		// Keep the natural deadline well outside the Close assertion below,
		// so only Close-triggered cancellation can make the test pass.
		WithRequestTimeout(10*time.Second),
	)
	stream := service.NewStreaming(context.Background(), ChatCompletionNewParams{
		Model:    "m",
		Messages: []ChatCompletionMessageParamUnion{UserMessage("hi")},
	})
	defer stream.Close()
	close(allowChunk)

	if !stream.Next() {
		t.Fatalf("Next() = false, err = %v", stream.Err())
	}
	if got := stream.Current().ID; got != "chunk-1" {
		t.Fatalf("chunk id = %q, want chunk-1", got)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	select {
	case <-requestDone:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("closing the stream did not cancel the request context")
	}
}
