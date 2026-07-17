package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"slices"
	"strings"
)

type ChatCompletionService struct {
	Options []RequestOption
}

func NewChatCompletionService(opts ...RequestOption) (r ChatCompletionService) {
	r = ChatCompletionService{}
	r.Options = slices.Clone(opts)
	return
}

func (r *ChatCompletionService) New(ctx context.Context, body ChatCompletionNewParams, opts ...RequestOption) (res *ChatCompletion, err error) {
	preClientOpts := []RequestOption{WithBearerAuthSecurity()}
	opts = slices.Concat(preClientOpts, r.Options, opts)
	path := "chat/completions"
	err = ExecuteNewRequest(ctx, http.MethodPost, path, body, &res, opts...)
	if err != nil {
		return nil, err
	}
	if err := validateChatCompletion(res); err != nil {
		return nil, err
	}
	return res, nil
}

func (r *ChatCompletionService) NewStreaming(ctx context.Context, body ChatCompletionNewParams, opts ...RequestOption) (stream *Stream[ChatCompletionChunk]) {
	return r.newStreaming(ctx, body, opts...)
}

// NewStreamingJSON starts a chat-completion stream from an already serialized
// request snapshot. It is useful to callers that must freeze mutable request
// fields before handing work to another goroutine.
func (r *ChatCompletionService) NewStreamingJSON(ctx context.Context, body json.RawMessage, opts ...RequestOption) *Stream[ChatCompletionChunk] {
	return r.newStreaming(ctx, body, opts...)
}

func (r *ChatCompletionService) newStreaming(ctx context.Context, body any, opts ...RequestOption) *Stream[ChatCompletionChunk] {
	var (
		raw *http.Response
		err error
	)
	preClientOpts := []RequestOption{WithBearerAuthSecurity()}
	opts = slices.Concat(preClientOpts, r.Options, opts)
	opts = append(opts, WithJSONSet("stream", true), WithHeader("Accept", "text/event-stream"))
	path := "chat/completions"
	err = ExecuteNewRequest(ctx, http.MethodPost, path, body, &raw, opts...)
	if err != nil {
		closeResponseBody(raw)
		return NewStream[ChatCompletionChunk](nil, err)
	}
	if err := validateStreamingResponse(raw); err != nil {
		closeResponseBody(raw)
		return NewStream[ChatCompletionChunk](nil, err)
	}
	return NewStream[ChatCompletionChunk](NewDecoder(raw), nil)
}

func validateChatCompletion(res *ChatCompletion) error {
	if res == nil {
		return fmt.Errorf("openai: invalid chat completion response: top-level value is null")
	}
	if res.Object != "" && res.Object != "chat.completion" {
		return fmt.Errorf("openai: invalid chat completion response object %q", res.Object)
	}
	if len(res.Choices) == 0 {
		return fmt.Errorf("openai: invalid chat completion response: choices is empty")
	}

	seenIndexes := make(map[int64]struct{}, len(res.Choices))
	for i, choice := range res.Choices {
		if choice.Index < 0 {
			return fmt.Errorf("openai: invalid chat completion choice %d: negative index %d", i, choice.Index)
		}
		if _, exists := seenIndexes[choice.Index]; exists {
			return fmt.Errorf("openai: invalid chat completion response: duplicate choice index %d", choice.Index)
		}
		seenIndexes[choice.Index] = struct{}{}
		if choice.FinishReason == "" {
			return fmt.Errorf("openai: invalid chat completion choice %d: finish_reason is empty", choice.Index)
		}
	}
	return nil
}

func validateStreamingResponse(res *http.Response) error {
	if res == nil {
		return fmt.Errorf("openai: streaming response is nil")
	}
	if res.Body == nil {
		return fmt.Errorf("openai: streaming response body is nil")
	}

	contentType := res.Header.Get("Content-Type")
	mediaType, _, parseErr := mime.ParseMediaType(contentType)
	if parseErr == nil && strings.EqualFold(mediaType, "text/event-stream") {
		return nil
	}

	// Some compatible gateways return a JSON error envelope with status 200.
	// Preserve that structured error even though the advertised media type is
	// invalid for a streaming response.
	contents, readErr := io.ReadAll(res.Body)
	if apiErr := errorFromEventData(contents); apiErr != nil {
		apiErr.StatusCode = res.StatusCode
		return apiErr
	}
	if readErr != nil {
		return fmt.Errorf("openai: expected streaming response content-type text/event-stream, got %q; reading response body: %w", contentType, readErr)
	}
	if parseErr != nil {
		return fmt.Errorf("openai: expected streaming response content-type text/event-stream, got %q: %w", contentType, parseErr)
	}
	return fmt.Errorf("openai: expected streaming response content-type text/event-stream, got %q", contentType)
}

func closeResponseBody(res *http.Response) {
	if res != nil && res.Body != nil {
		_ = res.Body.Close()
	}
}

type ChatCompletion struct {
	// A unique identifier for the chat completion.
	ID string `json:"id"`
	// A list of chat completion choices. Can be more than one if `n` is greater
	// than 1.
	Choices []ChatCompletionChoice `json:"choices"`
	// The Unix timestamp (in seconds) of when the chat completion was created.
	Created int64 `json:"created"`
	// The model used for the chat completion.
	Model string `json:"model"`
	// The object type, which is always `chat.completion`.
	Object string `json:"object"`
	// This fingerprint represents the backend configuration that the model runs with.
	SystemFingerprint string `json:"system_fingerprint"`
	// Usage statistics for the completion request.
	Usage CompletionUsage `json:"usage"`
	// Raw is the unmodified response JSON. Provider-specific fields that have
	// no explicit counterpart above can be unmarshaled from it.
	Raw json.RawMessage `json:"-"`
}

func (r *ChatCompletion) UnmarshalJSON(data []byte) error {
	type shadow ChatCompletion
	if err := json.Unmarshal(data, (*shadow)(r)); err != nil {
		return err
	}
	r.Raw = append([]byte(nil), data...)
	return nil
}

type ChatCompletionChoice struct {
	// The reason the model stopped generating tokens.
	//
	// Any of "stop", "length", "tool_calls", "content_filter",
	// "insufficient_system_resource".
	FinishReason string `json:"finish_reason"`
	// The index of the choice in the list of choices.
	Index int64 `json:"index"`
	// Log probability information for the choice.
	Logprobs ChatCompletionChoiceLogprobs `json:"logprobs"`
	// A chat completion message generated by the model.
	Message ChatCompletionMessage `json:"message"`
}

type ChatCompletionChoiceLogprobs struct {
	// A list of message content tokens with log probability information.
	Content []ChatCompletionTokenLogprob `json:"content"`
	// A list of message refusal tokens with log probability information.
	Refusal []ChatCompletionTokenLogprob `json:"refusal"`
}

type ChatCompletionTokenLogprob struct {
	// The token.
	Token string `json:"token"`
	// A list of integers representing the UTF-8 bytes representation of the token.
	// Can be `null` if there is no bytes representation for the token.
	Bytes []int64 `json:"bytes"`
	// The log probability of this token, if it is within the top 20 most likely
	// tokens. Otherwise, the value `-9999.0` is used to signify that the token is
	// very unlikely.
	Logprob float64 `json:"logprob"`
	// List of the most likely tokens and their log probability, at this token
	// position. The number of entries may be fewer than the requested `top_logprobs`.
	TopLogprobs []ChatCompletionTokenLogprobTopLogprob `json:"top_logprobs"`
}

type ChatCompletionTokenLogprobTopLogprob struct {
	// The token.
	Token string `json:"token"`
	// A list of integers representing the UTF-8 bytes representation of the token.
	// Can be `null` if there is no bytes representation for the token.
	Bytes []int64 `json:"bytes"`
	// The log probability of this token.
	Logprob float64 `json:"logprob"`
}
