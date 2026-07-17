package openaicompletions

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/openai"
)

const apiName = "openai-completions"

// Streamer implements ai.Streamer over the OpenAI-compatible completions
// protocol. It is a stateless protocol translator: one instance serves every
// endpoint speaking this protocol, and each Stream call carries the provider
// snapshot (base URL, resolved API key, quirk overrides) it should use.
type Streamer struct {
	completions openai.ChatCompletionService
}

var _ ai.Streamer = (*Streamer)(nil)

// NewStreamer builds the protocol adapter. Options carry infrastructure
// configuration only (shared http client, timeouts); per-endpoint business
// configuration travels with each Stream call's provider.
func NewStreamer(opts ...openai.RequestOption) *Streamer {
	return &Streamer{
		completions: openai.NewChatCompletionService(opts...),
	}
}

// Stream never fails synchronously: while ctx remains live, request and
// transport errors are delivered as a terminal ErrorEvent plus a final message
// with StopReasonError. Canceling ctx is a stream-mechanics failure instead:
// Events closes and Result returns ctx.Err without a terminal event or result.
func (s *Streamer) Stream(ctx context.Context, provider ai.Provider, model ai.Model, prompt ai.Prompt, opts ai.StreamOptions,
) *ai.EventStream[ai.AssistantMessageEvent, *ai.AssistantMessage] {
	stream, producer := ai.NewEventStream[ai.AssistantMessageEvent, *ai.AssistantMessage](ctx, 64)

	// Everything below can observe caller-owned maps, slices, pointers or
	// custom JSON marshalers. Finish it before returning so the request
	// goroutine receives immutable wire bytes rather than mutable Go values.
	var setupErr error
	prompt, setupErr = ai.SnapshotPrompt(prompt)
	if setupErr == nil {
		opts, setupErr = ai.SnapshotStreamOptions(opts)
	}
	var requestBody json.RawMessage
	if setupErr == nil {
		var params openai.ChatCompletionNewParams
		params, setupErr = buildParams(provider, model, prompt, opts)
		if setupErr == nil {
			requestBody, setupErr = json.Marshal(params)
			if setupErr != nil {
				setupErr = fmt.Errorf("openai: snapshot request body: %w", setupErr)
			}
		}
	}
	if setupErr != nil {
		// Some encoding/json errors retain reflect.Values into the rejected
		// input. The asynchronous error path needs only an immutable message.
		setupErr = errors.New(setupErr.Error())
	}

	go s.run(ctx, producer, provider, model, opts, requestBody, setupErr)
	return stream
}

func (s *Streamer) run(ctx context.Context, producer *ai.Producer[ai.AssistantMessageEvent, *ai.AssistantMessage],
	provider ai.Provider, model ai.Model, opts ai.StreamOptions, requestBody json.RawMessage, setupErr error) {

	msg := &ai.AssistantMessage{
		Role:       ai.RoleAssistant,
		Api:        apiName,
		Provider:   model.Provider,
		Model:      model.ID,
		StopReason: ai.StopReasonStop,
		Timestamp:  time.Now().UnixMilli(),
	}
	if msg.Provider == "" {
		msg.Provider = provider.Name
	}
	if setupErr != nil {
		fail(ctx, producer, msg, setupErr)
		return
	}

	var reqOpts []openai.RequestOption
	if provider.BaseURL != "" {
		reqOpts = append(reqOpts, openai.WithBaseURL(provider.BaseURL))
	}
	if provider.APIKey != "" {
		reqOpts = append(reqOpts, openai.WithAPIKey(provider.APIKey))
	}

	sse := s.completions.NewStreamingJSON(ctx, requestBody, reqOpts...)
	defer sse.Close()
	if err := sse.Err(); err != nil {
		fail(ctx, producer, msg, err)
		return
	}

	if producer.Publish(ctx, ai.StartEvent{Partial: snapshotAssistantMessage(msg)}) != nil {
		return
	}

	acc := &accumulator{msg: msg, producer: producer, outputFormat: opts.OutputFormat}
	hasFinishReason := false
	finishReason := ""
	responseModel := ""
	choiceIndex := int64(0)
	hasChoiceIndex := false

	for sse.Next() {
		chunk := sse.Current()

		if chunk.Object != "" && chunk.Object != "chat.completion.chunk" {
			fail(ctx, producer, msg, fmt.Errorf(
				"openai: invalid streaming response object %q", chunk.Object,
			))
			return
		}
		// Non-empty completion identity fields must remain stable across usage
		// and content chunks. Compatible providers may omit them on some chunks.
		if msg.ResponseId != "" && chunk.ID != "" && chunk.ID != msg.ResponseId {
			fail(ctx, producer, msg, fmt.Errorf(
				"openai: streaming response id changed from %q to %q", msg.ResponseId, chunk.ID,
			))
			return
		}
		if msg.ResponseId == "" && chunk.ID != "" {
			msg.ResponseId = chunk.ID
		}
		if responseModel != "" && chunk.Model != "" && chunk.Model != responseModel {
			fail(ctx, producer, msg, fmt.Errorf(
				"openai: streaming response model changed from %q to %q", responseModel, chunk.Model,
			))
			return
		}
		if responseModel == "" && chunk.Model != "" {
			responseModel = chunk.Model
			if chunk.Model != model.ID {
				msg.ResponseModel = chunk.Model
			}
		}
		if usageIsSet(chunk.Usage) {
			msg.Usage = convertUsage(chunk.Usage, model)
		}

		if len(chunk.Choices) > 1 {
			fail(ctx, producer, msg, fmt.Errorf(
				"openai: streaming response returned %d choices; exactly one is supported", len(chunk.Choices),
			))
			return
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if hasChoiceIndex && choice.Index != choiceIndex {
			fail(ctx, producer, msg, fmt.Errorf(
				"openai: streaming choice index changed from %d to %d", choiceIndex, choice.Index,
			))
			return
		}
		if !hasChoiceIndex {
			choiceIndex = choice.Index
			hasChoiceIndex = true
		}

		// Some providers (e.g. Moonshot) report usage on the choice.
		if !usageIsSet(chunk.Usage) && usageIsSet(choice.Usage) {
			msg.Usage = convertUsage(choice.Usage, model)
		}

		if choice.FinishReason != "" {
			if finishReason != "" && choice.FinishReason != finishReason {
				fail(ctx, producer, msg, fmt.Errorf(
					"openai: conflicting finish_reason values %q and %q",
					finishReason, choice.FinishReason,
				))
				return
			}
			finishReason = choice.FinishReason
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
	if ctx.Err() != nil {
		return
	}
	if !hasFinishReason {
		fail(ctx, producer, msg, errors.New("stream ended without finish_reason"))
		return
	}
	if msg.StopReason == ai.StopReasonError {
		fail(ctx, producer, msg, errors.New(msg.ErrorMessage))
		return
	}
	if err := validateToolCallState(msg.StopReason, acc.toolOrder); err != nil {
		fail(ctx, producer, msg, err)
		return
	}
	if err := acc.finishAll(ctx); err != nil {
		if ctx.Err() == nil {
			fail(ctx, producer, msg, err)
		}
		return
	}
	if ctx.Err() != nil {
		return
	}

	if producer.Publish(ctx, ai.DoneEvent{Reason: msg.StopReason, Message: snapshotAssistantMessage(msg)}) != nil {
		return
	}
	_ = producer.Complete(ctx, msg)
}

func fail(ctx context.Context, producer *ai.Producer[ai.AssistantMessageEvent, *ai.AssistantMessage],
	msg *ai.AssistantMessage, err error) {

	if ctx.Err() != nil {
		// The EventStream itself owns cancellation. It closes Events and makes
		// Result return ctx.Err; publishing with the canceled context could only
		// race that shutdown and produce a nondeterministic terminal event.
		return
	}
	msg.StopReason = ai.StopReasonError
	msg.ErrorMessage = err.Error()
	recordErrorDiagnostics(msg, err)
	_ = producer.Publish(ctx, ai.ErrorEvent{Reason: msg.StopReason, Error: snapshotAssistantMessage(msg)})
	_ = producer.Complete(ctx, msg)
}

func recordErrorDiagnostics(msg *ai.AssistantMessage, err error) {
	diagnostics := &msg.Diagnostics
	diagnostics.Type = "error"
	diagnostics.Timestamp = time.Now().UnixMilli()
	diagnostics.Error = ai.DiagnosticErrorInfo{
		Name:    "Error",
		Message: err.Error(),
	}

	var streamErr *openai.StreamError
	if errors.As(err, &streamErr) {
		diagnostics.Error.Name = "StreamError"
		if diagnostics.Details == nil {
			diagnostics.Details = map[string]any{}
		}
		diagnostics.Details["eventType"] = streamErr.Event.Type
		diagnostics.Details["eventData"] = string(streamErr.Event.Data)
	}

	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) {
		return
	}
	if streamErr == nil {
		diagnostics.Error.Name = "APIError"
	}
	if apiErr.Message != "" {
		diagnostics.Error.Message = apiErr.Message
	}
	diagnostics.Error.Code = apiErr.Code
	if diagnostics.Details == nil {
		diagnostics.Details = map[string]any{}
	}
	if apiErr.StatusCode != 0 {
		diagnostics.Details["statusCode"] = apiErr.StatusCode
	}
	if apiErr.Type != "" {
		diagnostics.Details["apiErrorType"] = apiErr.Type
	}
	if apiErr.Param != "" {
		diagnostics.Details["param"] = apiErr.Param
	}
	if apiErr.Body != "" {
		diagnostics.Details["body"] = apiErr.Body
	}
}

// accumulator reassembles content blocks from streamed deltas and emits the
// corresponding events. Text, structured JSON and thinking are single open
// blocks; tool calls are keyed by stream index with an id fallback for
// providers that omit the index, and all blocks close together when the
// stream ends.
type accumulator struct {
	msg          *ai.AssistantMessage
	producer     *ai.Producer[ai.AssistantMessageEvent, *ai.AssistantMessage]
	outputFormat ai.OutputFormat

	text        *ai.TextContent
	textIdx     int
	jsonContent *ai.JSONContent
	jsonIdx     int
	jsonBuffer  strings.Builder
	thinking    *ai.ThinkingContent
	thinkingIdx int
	refusal     strings.Builder

	toolOrder   []*toolCallState
	toolByIndex map[int64]*toolCallState
	toolByID    map[string]*toolCallState
}

type toolCallState struct {
	block       *ai.ToolCallContent
	contentIdx  int
	partialArgs strings.Builder
}

// snapshotAssistantMessage returns an event-time snapshot. AssistantMessage
// contains interface slices, pointer-backed content blocks and JSON maps, so a
// plain struct copy would let later deltas rewrite already-published events.
func snapshotAssistantMessage(msg *ai.AssistantMessage) ai.AssistantMessage {
	if snapshot := ai.CloneAssistantMessage(msg); snapshot != nil {
		return *snapshot
	}
	return ai.AssistantMessage{}
}

func cloneRawMessage(src json.RawMessage) json.RawMessage {
	if src == nil {
		return nil
	}
	return append(json.RawMessage(nil), src...)
}

func cloneJSONMap(src map[string]any) map[string]any {
	if src == nil {
		return nil
	}
	dst := make(map[string]any, len(src))
	for key, value := range src {
		dst[key] = cloneJSONValue(value)
	}
	return dst
}

func cloneJSONValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneJSONMap(value)
	case []any:
		copy := make([]any, len(value))
		for i, item := range value {
			copy[i] = cloneJSONValue(item)
		}
		return copy
	default:
		return value
	}
}

func (a *accumulator) applyDelta(ctx context.Context, delta openai.ChatCompletionChunkChoiceDelta) error {
	if delta.Refusal != "" {
		a.refusal.WriteString(delta.Refusal)
		if a.msg.Diagnostics.Details == nil {
			a.msg.Diagnostics.Details = map[string]any{}
		}
		a.msg.Diagnostics.Details["refusal"] = a.refusal.String()
	}
	if delta.Content != "" {
		if a.outputFormat.IsJSON() {
			if a.jsonContent == nil {
				schemaName := ""
				if a.outputFormat.JSONSchema != nil {
					schemaName = a.outputFormat.JSONSchema.Name
				}
				a.jsonContent = &ai.JSONContent{Type: ai.ContentTypeJSON, SchemaName: schemaName}
				a.jsonIdx = len(a.msg.Content)
				a.msg.Content = append(a.msg.Content, a.jsonContent)
				if err := a.producer.Publish(ctx, ai.JSONStartEvent{
					ContentIndex: a.jsonIdx, Partial: snapshotAssistantMessage(a.msg),
				}); err != nil {
					return err
				}
			}
			a.jsonBuffer.WriteString(delta.Content)
			a.rememberRawJSON(a.jsonBuffer.String())
			if err := a.producer.Publish(ctx, ai.JSONDeltaEvent{
				ContentIndex: a.jsonIdx, Delta: delta.Content, Partial: snapshotAssistantMessage(a.msg),
			}); err != nil {
				return err
			}
		} else {
			if a.text == nil {
				a.text = &ai.TextContent{Type: ai.ContentTypeText}
				a.textIdx = len(a.msg.Content)
				a.msg.Content = append(a.msg.Content, a.text)
				if err := a.producer.Publish(ctx, ai.TextStartEvent{
					ContentIndex: a.textIdx, Partial: snapshotAssistantMessage(a.msg),
				}); err != nil {
					return err
				}
			}
			a.text.Text += delta.Content
			if err := a.producer.Publish(ctx, ai.TextDeltaEvent{
				ContentIndex: a.textIdx, Delta: delta.Content, Partial: snapshotAssistantMessage(a.msg),
			}); err != nil {
				return err
			}
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
			if err := a.producer.Publish(ctx, ai.ThinkingStartEvent{
				ContentIndex: a.thinkingIdx, Partial: snapshotAssistantMessage(a.msg),
			}); err != nil {
				return err
			}
		}
		a.thinking.Thinking += reasoningDelta
		if err := a.producer.Publish(ctx, ai.ThinkingDeltaEvent{
			ContentIndex: a.thinkingIdx, Delta: reasoningDelta, Partial: snapshotAssistantMessage(a.msg),
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
			ContentIndex: state.contentIdx, Delta: tc.Function.Arguments, Partial: snapshotAssistantMessage(a.msg),
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

	// An omitted index decodes to zero, which may already belong to another
	// tool. A supplied ID is therefore the authoritative continuation key.
	if tc.ID != "" {
		if state, ok := a.toolByID[tc.ID]; ok {
			return state, nil
		}
		if state, ok := a.toolByIndex[tc.Index]; ok {
			// Some providers reveal the ID after starting the call by index.
			if state.block.Id == "" || state.block.Id == tc.ID {
				return state, nil
			}
			// A new ID that conflicts with an occupied/defaulted index is a
			// distinct call. Do not replace the index mapping: later index-only
			// fragments must continue to reach the original call.
			return a.startToolCall(ctx, tc, false)
		}
	}
	if state, ok := a.toolByIndex[tc.Index]; ok {
		return state, nil
	}
	return a.startToolCall(ctx, tc, true)
}

func (a *accumulator) startToolCall(ctx context.Context,
	tc openai.ChatCompletionChunkChoiceDeltaToolCall, mapIndex bool,
) (*toolCallState, error) {
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
	if mapIndex {
		a.toolByIndex[tc.Index] = state
	}
	if tc.ID != "" {
		a.toolByID[tc.ID] = state
	}
	return state, a.producer.Publish(ctx, ai.ToolCallStartEvent{
		ContentIndex: state.contentIdx, Partial: snapshotAssistantMessage(a.msg),
	})
}

// finishAll validates completed tool arguments and structured final JSON before
// closing any open block. Partial JSON is for UI updates only: no terminal
// success event becomes observable until every final buffer satisfies its
// contract.
func (a *accumulator) finishAll(ctx context.Context) error {
	for _, state := range a.toolOrder {
		arguments, err := decodeJSONObject(state.partialArgs.String())
		if err != nil {
			return fmt.Errorf(
				"openai: invalid final JSON arguments for tool %q (call %q): %w",
				state.block.Name, state.block.Id, err,
			)
		}
		state.block.Arguments = arguments
	}
	jsonFinalized, err := a.finalizeJSON()
	if err != nil {
		return err
	}

	if a.text != nil {
		if err := a.producer.Publish(ctx, ai.TextEndEvent{
			ContentIndex: a.textIdx, Content: a.text.Text, Partial: snapshotAssistantMessage(a.msg),
		}); err != nil {
			return err
		}
	}
	if jsonFinalized {
		if err := a.producer.Publish(ctx, ai.JSONEndEvent{
			ContentIndex: a.jsonIdx,
			Content:      cloneRawMessage(a.jsonContent.Value),
			Partial:      snapshotAssistantMessage(a.msg),
		}); err != nil {
			return err
		}
	}
	if a.thinking != nil {
		if err := a.producer.Publish(ctx, ai.ThinkingEndEvent{
			ContentIndex: a.thinkingIdx, Content: a.thinking.Thinking, Partial: snapshotAssistantMessage(a.msg),
		}); err != nil {
			return err
		}
	}
	for _, state := range a.toolOrder {
		toolCall := *state.block
		toolCall.Arguments = cloneJSONMap(state.block.Arguments)
		if err := a.producer.Publish(ctx, ai.ToolCallEndEvent{
			ContentIndex: state.contentIdx, ToolCall: toolCall, Partial: snapshotAssistantMessage(a.msg),
		}); err != nil {
			return err
		}
	}
	return nil
}

// finalizeJSON publishes no terminal state itself. It validates and freezes a
// complete JSON block only when the provider supplied a usable finish reason.
// A tool-call-only turn is an intermediate agent step and may legitimately
// contain no JSON block even when the eventual answer is structured. If a
// structured tool-use turn does emit content, it must still be valid JSON so
// its JSONStart event can close with JSONEnd rather than dangling at Done.
func (a *accumulator) finalizeJSON() (bool, error) {
	if !a.outputFormat.IsJSON() {
		return false, nil
	}
	raw := a.jsonBuffer.String()
	if a.msg.StopReason == ai.StopReasonToolUse && a.jsonContent == nil {
		return false, nil
	}
	if refusal := a.refusal.String(); refusal != "" {
		return false, fmt.Errorf("openai: structured output refused: %s", refusal)
	}
	if a.jsonContent == nil || strings.TrimSpace(raw) == "" {
		return false, fmt.Errorf("openai: output format %q produced no JSON content", a.outputFormat.Type)
	}
	if a.outputFormat.Type == ai.OutputFormatJSONObject {
		var object map[string]json.RawMessage
		if err := json.Unmarshal([]byte(raw), &object); err != nil {
			a.rememberRawJSON(raw)
			return false, fmt.Errorf("openai: invalid final JSON object for output format %q: %w", a.outputFormat.Type, err)
		}
		if object == nil {
			a.rememberRawJSON(raw)
			return false, fmt.Errorf("openai: invalid final JSON object for output format %q: top-level value is null", a.outputFormat.Type)
		}
	} else {
		var value json.RawMessage
		if err := json.Unmarshal([]byte(raw), &value); err != nil {
			a.rememberRawJSON(raw)
			return false, fmt.Errorf("openai: invalid final JSON for output format %q: %w", a.outputFormat.Type, err)
		}
	}
	a.jsonContent.Value = json.RawMessage([]byte(raw))
	a.clearRawJSON()
	return true, nil
}

func (a *accumulator) rememberRawJSON(raw string) {
	if a.msg.Diagnostics.Details == nil {
		a.msg.Diagnostics.Details = map[string]any{}
	}
	a.msg.Diagnostics.Details["rawJSON"] = raw
}

func (a *accumulator) clearRawJSON() {
	delete(a.msg.Diagnostics.Details, "rawJSON")
	if len(a.msg.Diagnostics.Details) == 0 {
		a.msg.Diagnostics.Details = nil
	}
}

func usageIsSet(u openai.CompletionUsage) bool {
	return u.PromptTokens != 0 || u.CompletionTokens != 0 || u.TotalTokens != 0
}

// convertUsage maps wire usage to neutral usage and prices it against the
// model's per-token pricing. Cached tokens are carved out of input because
// they are billed differently: input covers only uncached prompt tokens.
func convertUsage(u openai.CompletionUsage, model ai.Model) ai.Usage {
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
	usage := ai.Usage{
		Input:       input,
		Output:      u.CompletionTokens,
		CacheRead:   cacheRead,
		CacheWrite:  cacheWrite,
		Reasoning:   u.CompletionTokensDetails.ReasoningTokens,
		TotalTokens: input + u.CompletionTokens + cacheRead + cacheWrite,
	}
	usage.Cost = ai.CalculateCost(model, usage)
	return usage
}

func mapStopReason(reason string) (ai.StopReason, string) {
	switch reason {
	case "stop", "end":
		return ai.StopReasonStop, ""
	case "length":
		return ai.StopReasonLength, ""
	case "tool_calls":
		return ai.StopReasonToolUse, ""
	case "function_call":
		return ai.StopReasonError,
			"provider used unsupported legacy function_call protocol; expected tool_calls"
	default:
		// content_filter, network_error and anything unknown: never silently
		// treat an abnormal termination as a clean stop.
		return ai.StopReasonError, "provider finish_reason: " + reason
	}
}

// validateToolCallState keeps the neutral message contract unambiguous: a
// successful tool-use finish must contain executable calls, and successful
// non-tool finishes must not contain calls that the agent would otherwise
// silently ignore. Error turns may retain partial calls for diagnostics.
func validateToolCallState(reason ai.StopReason, calls []*toolCallState) error {
	callCount := len(calls)
	switch reason {
	case ai.StopReasonToolUse:
		if callCount == 0 {
			return errors.New("openai: tool-use finish reason without tool calls")
		}
		seenIDs := make(map[string]struct{}, callCount)
		for i, state := range calls {
			call := state.block
			if call.Type != ai.ContentTypeToolCall {
				return fmt.Errorf("openai: tool call %d has invalid content type %q", i, call.Type)
			}
			if call.Id == "" {
				return fmt.Errorf("openai: tool call %d has an empty id", i)
			}
			if call.Name == "" {
				return fmt.Errorf("openai: tool call %q has an empty name", call.Id)
			}
			if _, duplicate := seenIDs[call.Id]; duplicate {
				return fmt.Errorf("openai: duplicate tool call id %q", call.Id)
			}
			seenIDs[call.Id] = struct{}{}
		}
	case ai.StopReasonStop, ai.StopReasonLength:
		if callCount > 0 {
			return fmt.Errorf(
				"openai: received %d tool call(s) with stop reason %q", callCount, reason,
			)
		}
	}
	return nil
}
