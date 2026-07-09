package openaicompletions

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"time"

	"github.com/zhongys/Daksha/internal/ai"
	"github.com/zhongys/Daksha/internal/ai/openai"
)

const apiName = "openai-completions"

// Streamer implements ai.Streamer over the OpenAI-compatible completions
// protocol. One Streamer serves one provider endpoint.
type Streamer struct {
	provider    ai.Provider
	completions openai.ChatCompletionService
}

var _ ai.Streamer = (*Streamer)(nil)

// NewStreamer builds a Streamer for a provider. The API key is read from the
// provider's APIKeyEnv environment variable; pass openai.WithAPIKey to override.
func NewStreamer(provider ai.Provider, opts ...openai.RequestOption) *Streamer {
	var base []openai.RequestOption
	if provider.BaseURL != "" {
		base = append(base, openai.WithBaseURL(provider.BaseURL))
	}
	if provider.APIKeyEnv != "" {
		if key := os.Getenv(provider.APIKeyEnv); key != "" {
			base = append(base, openai.WithAPIKey(key))
		}
	}
	base = append(base, opts...)
	return &Streamer{
		provider:    provider,
		completions: openai.NewChatCompletionService(base...),
	}
}

// Stream never fails synchronously: request and transport errors are
// delivered as an ErrorEvent plus a final message with StopReason
// error/aborted, per the ai.Streamer contract.
func (s *Streamer) Stream(ctx context.Context, model ai.Model, prompt ai.Prompt, opts ai.StreamOptions,
) *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage] {
	stream, producer := ai.NewEventStream[ai.AssistantMessageEvent, *ai.AssistantMessage](ctx, 64)
	go s.run(ctx, producer, model, prompt, opts)
	return stream
}

func (s *Streamer) run(ctx context.Context, producer *ai.Producer[ai.AssistantMessageEvent, *ai.AssistantMessage],
	model ai.Model, prompt ai.Prompt, opts ai.StreamOptions) {

	msg := &ai.AssistantMessage{
		Role:       ai.RoleAssistant,
		Api:        apiName,
		Provider:   model.Provider,
		Model:      model.ID,
		StopReason: ai.StopReasonStop,
		Timestamp:  time.Now().UnixMilli(),
	}
	if msg.Provider == "" {
		msg.Provider = s.provider.Name
	}

	params, err := buildParams(s.provider, model, prompt, opts)
	if err != nil {
		fail(ctx, producer, msg, err)
		return
	}

	sse := s.completions.NewStreaming(ctx, params)
	defer sse.Close()
	if err := sse.Err(); err != nil {
		fail(ctx, producer, msg, err)
		return
	}

	if producer.Publish(ctx, ai.StartEvent{Partial: *msg}) != nil {
		return
	}

	acc := &accumulator{msg: msg, producer: producer}
	hasFinishReason := false

	for sse.Next() {
		chunk := sse.Current()

		// Every chunk of one completion carries the same id.
		if msg.ResponseId == "" {
			msg.ResponseId = chunk.ID
		}
		if msg.ResponseModel == "" && chunk.Model != "" && chunk.Model != model.ID {
			msg.ResponseModel = chunk.Model
		}
		if usageIsSet(chunk.Usage) {
			msg.Usage = convertUsage(chunk.Usage)
		}

		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		// Some providers (e.g. Moonshot) report usage on the choice.
		if !usageIsSet(chunk.Usage) && usageIsSet(choice.Usage) {
			msg.Usage = convertUsage(choice.Usage)
		}

		if choice.FinishReason != "" {
			reason, errMsg := mapStopReason(choice.FinishReason)
			msg.StopReason = reason
			if errMsg != "" {
				msg.ErrorMessage = errMsg
			}
			hasFinishReason = true
		}

		if err := acc.applyDelta(ctx, choice.Delta); err != nil {
			return // consumer gone or ctx canceled
		}
	}

	if err := sse.Err(); err != nil {
		fail(ctx, producer, msg, err)
		return
	}
	if err := acc.finishAll(ctx); err != nil {
		return
	}
	if ctx.Err() != nil {
		fail(ctx, producer, msg, ctx.Err())
		return
	}
	if msg.StopReason == ai.StopReasonError {
		fail(ctx, producer, msg, errors.New(msg.ErrorMessage))
		return
	}
	if !hasFinishReason {
		fail(ctx, producer, msg, errors.New("stream ended without finish_reason"))
		return
	}

	if producer.Publish(ctx, ai.DoneEvent{Reason: msg.StopReason, Message: *msg}) != nil {
		return
	}
	_ = producer.Complete(ctx, msg)
}

func fail(ctx context.Context, producer *ai.Producer[ai.AssistantMessageEvent, *ai.AssistantMessage],
	msg *ai.AssistantMessage, err error) {

	if ctx.Err() != nil {
		msg.StopReason = ai.StopReasonAborted
	} else {
		msg.StopReason = ai.StopReasonError
	}
	if msg.ErrorMessage == "" || msg.StopReason == ai.StopReasonAborted {
		msg.ErrorMessage = err.Error()
	}
	_ = producer.Publish(ctx, ai.ErrorEvent{Reason: msg.StopReason, Error: *msg})
	_ = producer.Complete(ctx, msg)
}

// accumulator reassembles content blocks from streamed deltas and emits the
// corresponding events. Mirrors pi-ai's block handling: text and thinking
// are single open blocks, tool calls are keyed by stream index with an id
// fallback for providers that omit the index, and all blocks close together
// when the stream ends.
type accumulator struct {
	msg      *ai.AssistantMessage
	producer *ai.Producer[ai.AssistantMessageEvent, *ai.AssistantMessage]

	text        *ai.TextContent
	textIdx     int
	thinking    *ai.ThinkingContent
	thinkingIdx int

	toolOrder   []*toolCallState
	toolByIndex map[int64]*toolCallState
	toolByID    map[string]*toolCallState
}

type toolCallState struct {
	block       *ai.ToolCallContent
	contentIdx  int
	partialArgs strings.Builder
}

func (a *accumulator) applyDelta(ctx context.Context, delta openai.ChatCompletionChunkChoiceDelta) error {
	if delta.Content != "" {
		if a.text == nil {
			a.text = &ai.TextContent{Type: ai.ContentTypeText}
			a.textIdx = len(a.msg.Content)
			a.msg.Content = append(a.msg.Content, a.text)
			if err := a.producer.Publish(ctx, ai.TextStartEvent{ContentIndex: a.textIdx, Partial: *a.msg}); err != nil {
				return err
			}
		}
		a.text.Text += delta.Content
		if err := a.producer.Publish(ctx, ai.TextDeltaEvent{
			ContentIndex: a.textIdx, Delta: delta.Content, Partial: *a.msg,
		}); err != nil {
			return err
		}
	}

	// Use the first non-empty reasoning field to avoid duplication; some
	// endpoints send the same content under two names.
	reasoningDelta, reasoningField := "", ""
	switch {
	case delta.ReasoningContent != "":
		reasoningDelta, reasoningField = delta.ReasoningContent, "reasoning_content"
	case delta.Reasoning != "":
		reasoningDelta, reasoningField = delta.Reasoning, "reasoning"
	case delta.ReasoningText != "":
		reasoningDelta, reasoningField = delta.ReasoningText, "reasoning_text"
	}
	if reasoningDelta != "" {
		if a.thinking == nil {
			a.thinking = &ai.ThinkingContent{Type: ai.ContentTypeThinking, ThinkingSignature: reasoningField}
			a.thinkingIdx = len(a.msg.Content)
			a.msg.Content = append(a.msg.Content, a.thinking)
			if err := a.producer.Publish(ctx, ai.ThinkingStartEvent{ContentIndex: a.thinkingIdx, Partial: *a.msg}); err != nil {
				return err
			}
		}
		a.thinking.Thinking += reasoningDelta
		if err := a.producer.Publish(ctx, ai.ThinkingDeltaEvent{
			ContentIndex: a.thinkingIdx, Delta: reasoningDelta, Partial: *a.msg,
		}); err != nil {
			return err
		}
	}

	for _, tc := range delta.ToolCalls {
		state, err := a.ensureToolCall(ctx, tc)
		if err != nil {
			return err
		}
		if state.block.Id == "" && tc.ID != "" {
			state.block.Id = tc.ID
			a.toolByID[tc.ID] = state
		}
		if state.block.Name == "" && tc.Function.Name != "" {
			state.block.Name = tc.Function.Name
		}

		if tc.Function.Arguments != "" {
			state.partialArgs.WriteString(tc.Function.Arguments)
			// Keep Arguments as the best parse of what has arrived so far.
			state.block.Arguments = parseStreamingJSON(state.partialArgs.String())
		}
		if err := a.producer.Publish(ctx, ai.ToolCallDeltaEvent{
			ContentIndex: state.contentIdx, Delta: tc.Function.Arguments, Partial: *a.msg,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (a *accumulator) ensureToolCall(ctx context.Context, tc openai.ChatCompletionChunkChoiceDeltaToolCall) (*toolCallState, error) {
	if a.toolByIndex == nil {
		a.toolByIndex = map[int64]*toolCallState{}
		a.toolByID = map[string]*toolCallState{}
	}

	if state, ok := a.toolByIndex[tc.Index]; ok {
		return state, nil
	}
	// Fallback for providers that omit the index on continuation deltas.
	if tc.ID != "" {
		if state, ok := a.toolByID[tc.ID]; ok {
			a.toolByIndex[tc.Index] = state
			return state, nil
		}
	}

	state := &toolCallState{
		block: &ai.ToolCallContent{
			Type:      ai.ContentTypeToolCall,
			Id:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: map[string]any{},
		},
		contentIdx: len(a.msg.Content),
	}
	a.msg.Content = append(a.msg.Content, state.block)
	a.toolOrder = append(a.toolOrder, state)
	a.toolByIndex[tc.Index] = state
	if tc.ID != "" {
		a.toolByID[tc.ID] = state
	}
	return state, a.producer.Publish(ctx, ai.ToolCallStartEvent{ContentIndex: state.contentIdx, Partial: *a.msg})
}

// finishAll closes every open block in content order, emitting the matching
// end events. Tool call arguments get one final parse of the full buffer.
func (a *accumulator) finishAll(ctx context.Context) error {
	if a.text != nil {
		if err := a.producer.Publish(ctx, ai.TextEndEvent{
			ContentIndex: a.textIdx, Content: a.text.Text, Partial: *a.msg,
		}); err != nil {
			return err
		}
	}
	if a.thinking != nil {
		if err := a.producer.Publish(ctx, ai.ThinkingEndEvent{
			ContentIndex: a.thinkingIdx, Content: a.thinking.Thinking, Partial: *a.msg,
		}); err != nil {
			return err
		}
	}
	for _, state := range a.toolOrder {
		state.block.Arguments = parseStreamingJSON(state.partialArgs.String())
		if err := a.producer.Publish(ctx, ai.ToolCallEndEvent{
			ContentIndex: state.contentIdx, ToolCall: *state.block, Partial: *a.msg,
		}); err != nil {
			return err
		}
	}
	return nil
}

func usageIsSet(u openai.CompletionUsage) bool {
	return u.PromptTokens != 0 || u.CompletionTokens != 0 || u.TotalTokens != 0
}

// convertUsage maps wire usage to neutral usage. Cached tokens are carved
// out of input because they are billed differently: input covers only
// uncached prompt tokens.
func convertUsage(u openai.CompletionUsage) ai.Usage {
	cacheRead := u.PromptTokensDetails.CachedTokens
	var cacheWrite int64
	if len(u.Raw) > 0 {
		var probe struct {
			PromptCacheHitTokens int64 `json:"prompt_cache_hit_tokens"`
			PromptTokensDetails  struct {
				CacheWriteTokens int64 `json:"cache_write_tokens"`
			} `json:"prompt_tokens_details"`
		}
		if json.Unmarshal(u.Raw, &probe) == nil {
			if cacheRead == 0 {
				cacheRead = probe.PromptCacheHitTokens
			}
			cacheWrite = probe.PromptTokensDetails.CacheWriteTokens
		}
	}

	input := u.PromptTokens - cacheRead - cacheWrite
	if input < 0 {
		input = 0
	}
	return ai.Usage{
		Input:       input,
		Output:      u.CompletionTokens,
		CacheRead:   cacheRead,
		CacheWrite:  cacheWrite,
		Reasoning:   u.CompletionTokensDetails.ReasoningTokens,
		TotalTokens: input + u.CompletionTokens + cacheRead + cacheWrite,
	}
}

func mapStopReason(reason string) (ai.StopReason, string) {
	switch reason {
	case "stop", "end":
		return ai.StopReasonStop, ""
	case "length":
		return ai.StopReasonLength, ""
	case "tool_calls", "function_call":
		return ai.StopReasonToolUse, ""
	default:
		// content_filter, network_error and anything unknown: never silently
		// treat an abnormal termination as a clean stop.
		return ai.StopReasonError, "provider finish_reason: " + reason
	}
}
