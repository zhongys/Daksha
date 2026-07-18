package openaicompletions

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/openai"
)

// resolvedCompat is the fully-resolved quirk configuration for one request:
// auto-detected from the provider's base URL, then overridden by
// provider.Compat. Quirks are endpoint properties, so they live on the
// Provider; a model needing different quirks is a second Provider entry.
type resolvedCompat struct {
	maxTokensField                              string
	thinkingFormat                              ai.ThinkingFormat
	supportsDeveloperRole                       bool
	requiresReasoningContentOnAssistantMessages bool
}

const kimiK3MaxCompletionTokens int64 = 1_048_576

func detectCompat(baseURL string) resolvedCompat {
	isDeepSeek := strings.Contains(baseURL, "deepseek.com")
	// Both Alibaba endpoints speak the qwen dialect: classic DashScope
	// (dashscope.aliyuncs.com/compatible-mode) and Model Studio (maas.aliyuncs.com).
	isQwen := strings.Contains(baseURL, "dashscope") || strings.Contains(baseURL, "maas.aliyuncs.com")
	isOpenRouter := strings.Contains(baseURL, "openrouter.ai")
	isZai := strings.Contains(baseURL, "bigmodel.cn") || strings.Contains(baseURL, "api.z.ai")
	isMoonshot := strings.Contains(baseURL, "api.moonshot.")

	c := resolvedCompat{
		maxTokensField:        "max_completion_tokens",
		thinkingFormat:        ai.ThinkingFormatOpenAI,
		supportsDeveloperRole: true,
	}
	switch {
	case isDeepSeek:
		c.thinkingFormat = ai.ThinkingFormatDeepSeek
		c.requiresReasoningContentOnAssistantMessages = true
	case isQwen:
		c.thinkingFormat = ai.ThinkingFormatQwen
	case isOpenRouter:
		c.thinkingFormat = ai.ThinkingFormatOpenRouter
	case isZai:
		c.thinkingFormat = ai.ThinkingFormatZai
	}
	if isDeepSeek || isZai || isMoonshot {
		c.maxTokensField = "max_tokens"
	}
	// DashScope accepts the standard max_completion_tokens nowadays; qwen
	// only keeps the developer-role quirk.
	if isDeepSeek || isQwen || isZai || isMoonshot {
		c.supportsDeveloperRole = false
	}
	return c
}

func resolveCompat(provider ai.Provider) resolvedCompat {
	c := detectCompat(provider.BaseURL)
	o := provider.Compat
	if o == nil {
		return c
	}
	if o.MaxTokensField != "" {
		c.maxTokensField = o.MaxTokensField
	}
	if o.ThinkingFormat != "" {
		c.thinkingFormat = o.ThinkingFormat
	}
	if o.SupportsDeveloperRole != nil {
		c.supportsDeveloperRole = *o.SupportsDeveloperRole
	}
	if o.RequiresReasoningContentOnAssistantMessages != nil {
		c.requiresReasoningContentOnAssistantMessages = *o.RequiresReasoningContentOnAssistantMessages
	}
	return c
}

func buildParams(provider ai.Provider, model ai.Model, prompt ai.Prompt, opts ai.StreamOptions) (openai.ChatCompletionNewParams, error) {
	compat := resolveCompat(provider)
	if err := validateKimiK3Options(provider, model, opts); err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	if err := opts.OutputFormat.Validate(); err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	messages, err := convertMessages(prompt, model, compat)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	params := openai.ChatCompletionNewParams{
		Model:    model.ID,
		Messages: messages,
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
		ExtraFields: map[string]any{},
	}

	params.Temperature = opts.Temperature
	params.TopP = opts.TopP
	if opts.MaxTokens != nil {
		if compat.maxTokensField == "max_tokens" {
			params.MaxTokens = opts.MaxTokens
		} else {
			params.MaxCompletionTokens = opts.MaxTokens
		}
	}
	if len(opts.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: opts.StopSequences}
	}
	if opts.ToolChoice != "" {
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String(opts.ToolChoice)}
	}
	if err := applyOutputFormat(&params, opts.OutputFormat); err != nil {
		return openai.ChatCompletionNewParams{}, err
	}

	for _, tool := range prompt.Tools {
		params.Tools = append(params.Tools, openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        tool.Name,
			Description: openai.String(tool.Description),
			Parameters:  tool.Parameters,
		}))
	}

	applyThinkingFormat(&params, model, compat, opts.ReasoningEffort)

	// Precedence: provider < model < request; explicit extras win over
	// everything, including the thinking-format fields above. response_format
	// remains available through Extra for compatibility only when the typed
	// OutputFormat is unset; mixing both forms is rejected.
	for _, extra := range []map[string]any{provider.Extra, model.Extra, opts.Extra} {
		for k, v := range extra {
			if k == "response_format" && opts.OutputFormat.Type != "" {
				return openai.ChatCompletionNewParams{}, fmt.Errorf(
					"openai: response_format cannot be set through Extra when OutputFormat is set",
				)
			}
			params.ExtraFields[k] = v
		}
	}
	return params, nil
}

// validateKimiK3Options keeps K3's fixed request contract local to the
// dedicated "kimi" provider. Other providers and even legacy Moonshot
// entries continue through the generic OpenAI-compatible path unchanged.
func validateKimiK3Options(provider ai.Provider, model ai.Model, opts ai.StreamOptions) error {
	if provider.Name != "kimi" || model.ID != "kimi-k3" {
		return nil
	}
	if !model.Reasoning {
		return fmt.Errorf("openai: kimi-k3 must be configured as a reasoning model")
	}
	if opts.ReasoningEffort != "" && opts.ReasoningEffort != "max" {
		return fmt.Errorf("openai: kimi-k3 reasoning effort must be %q, got %q", "max", opts.ReasoningEffort)
	}
	if opts.Temperature != nil && *opts.Temperature != 1.0 {
		return fmt.Errorf("openai: kimi-k3 temperature is fixed at 1.0, got %v", *opts.Temperature)
	}
	if opts.TopP != nil && *opts.TopP != 0.95 {
		return fmt.Errorf("openai: kimi-k3 top_p is fixed at 0.95, got %v", *opts.TopP)
	}
	if opts.MaxTokens != nil && (*opts.MaxTokens < 1 || *opts.MaxTokens > kimiK3MaxCompletionTokens) {
		return fmt.Errorf("openai: kimi-k3 max completion tokens must be between 1 and %d, got %d",
			kimiK3MaxCompletionTokens, *opts.MaxTokens)
	}
	switch opts.ToolChoice {
	case "", "auto", "none", "required":
	default:
		return fmt.Errorf("openai: kimi-k3 unsupported tool choice %q", opts.ToolChoice)
	}
	if len(opts.StopSequences) > 5 {
		return fmt.Errorf("openai: kimi-k3 supports at most 5 stop sequences, got %d", len(opts.StopSequences))
	}
	for i, stop := range opts.StopSequences {
		if len(stop) > 32 {
			return fmt.Errorf("openai: kimi-k3 stop sequence %d exceeds 32 bytes", i)
		}
	}

	for _, extra := range []struct {
		scope  string
		fields map[string]any
	}{
		{scope: "provider", fields: provider.Extra},
		{scope: "model", fields: model.Extra},
		{scope: "request", fields: opts.Extra},
	} {
		for key := range extra.fields {
			if isKimiK3ReservedExtraField(key) {
				return fmt.Errorf("openai: kimi-k3 %s extra field %q must use the typed option or be omitted",
					extra.scope, key)
			}
		}
	}
	return nil
}

func isKimiK3ReservedExtraField(key string) bool {
	switch key {
	case "thinking", "enable_thinking", "reasoning", "reasoning_effort",
		"max_tokens", "max_completion_tokens", "temperature", "top_p", "n",
		"presence_penalty", "frequency_penalty", "tool_choice":
		return true
	default:
		return false
	}
}

func applyOutputFormat(params *openai.ChatCompletionNewParams, format ai.OutputFormat) error {
	switch format.Type {
	case "":
		return nil
	case ai.OutputFormatText:
		params.ResponseFormat = openai.ResponseFormatText()
	case ai.OutputFormatJSONObject:
		params.ResponseFormat = openai.ResponseFormatJSONObject()
	case ai.OutputFormatJSONSchema:
		schema := format.JSONSchema
		if schema == nil {
			// buildParams validates first; keep this helper safe in isolation.
			return fmt.Errorf("openai: output format %q requires a json schema", format.Type)
		}
		wireSchema := openai.ResponseFormatJSONSchemaJSONSchemaParam{
			Name:   schema.Name,
			Schema: schema.Schema,
			Strict: openai.Bool(schema.Strict),
		}
		if schema.Description != "" {
			wireSchema.Description = openai.String(schema.Description)
		}
		params.ResponseFormat = openai.ResponseFormatJSONSchema(wireSchema)
	default:
		return fmt.Errorf("openai: unsupported output format %q", format.Type)
	}
	return nil
}

func applyThinkingFormat(params *openai.ChatCompletionNewParams, model ai.Model, compat resolvedCompat, effort string) {
	if !model.Reasoning {
		return
	}
	enabled := effort != ""

	switch compat.thinkingFormat {
	case ai.ThinkingFormatDeepSeek:
		if enabled {
			params.ExtraFields["thinking"] = map[string]any{"type": "enabled"}
			params.ReasoningEffort = effort
		} else {
			params.ExtraFields["thinking"] = map[string]any{"type": "disabled"}
		}
	case ai.ThinkingFormatQwen:
		// Omit the field entirely when disabled: DashScope's thinking-only
		// models reject enable_thinking=false outright (400 "restricted to
		// True"), and hybrid commercial models default to no thinking — so
		// absence still reads as "off" wherever off exists.
		if enabled {
			params.ExtraFields["enable_thinking"] = true
		}
	case ai.ThinkingFormatZai:
		if enabled {
			params.ExtraFields["thinking"] = map[string]any{"type": "enabled"}
		} else {
			params.ExtraFields["thinking"] = map[string]any{"type": "disabled"}
		}
	case ai.ThinkingFormatOpenRouter:
		if enabled {
			params.ExtraFields["reasoning"] = map[string]any{"effort": effort}
		}
	default: // ThinkingFormatOpenAI
		if enabled {
			params.ReasoningEffort = effort
		}
	}
}

func convertMessages(prompt ai.Prompt, model ai.Model, compat resolvedCompat) ([]openai.ChatCompletionMessageParamUnion, error) {
	var params []openai.ChatCompletionMessageParamUnion

	if prompt.System != "" {
		if model.Reasoning && compat.supportsDeveloperRole {
			params = append(params, openai.DeveloperMessage(prompt.System))
		} else {
			params = append(params, openai.SystemMessage(prompt.System))
		}
	}

	msgs := prompt.Messages
	for i := 0; i < len(msgs); i++ {
		switch m := msgs[i].(type) {
		case *ai.UserMessage:
			p, ok, err := convertUserMessage(m)
			if err != nil {
				return nil, err
			}
			if ok {
				params = append(params, p)
			}

		case *ai.AssistantMessage:
			if p, ok := convertAssistantMessage(m, model, compat); ok {
				params = append(params, p)
			}

		case ai.ToolResult:
			// Consume the whole run of consecutive tool results, collecting
			// their images into one trailing user message: the tool role
			// cannot carry images in this protocol.
			var images []openai.ChatCompletionContentPartUnionParam
			j := i
			for ; j < len(msgs); j++ {
				tr, ok := msgs[j].(ai.ToolResult)
				if !ok {
					break
				}
				toolMsg, imgs, err := convertToolResult(tr)
				if err != nil {
					return nil, err
				}
				params = append(params, toolMsg)
				images = append(images, imgs...)
			}
			i = j - 1
			if len(images) > 0 {
				parts := append(
					[]openai.ChatCompletionContentPartUnionParam{openai.TextContentPart("Attached image(s) from tool result:")},
					images...,
				)
				params = append(params, openai.UserMessage(parts))
			}

		default:
			return nil, fmt.Errorf("openai: unsupported message type %T", msgs[i])
		}
	}
	return params, nil
}

func convertUserMessage(m *ai.UserMessage) (openai.ChatCompletionMessageParamUnion, bool, error) {
	// Single text block goes out as a plain string, the most compatible form.
	if len(m.Content) == 1 {
		if t, ok := m.Content[0].(*ai.TextContent); ok {
			return openai.UserMessage(t.Text), true, nil
		}
	}

	var parts []openai.ChatCompletionContentPartUnionParam
	for _, c := range m.Content {
		switch v := c.(type) {
		case *ai.TextContent:
			parts = append(parts, openai.TextContentPart(v.Text))
		case *ai.ImageContent:
			ref, err := imageRef(v)
			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, false, err
			}
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: ref,
			}))
		case *ai.AudioContent:
			part, err := audioPart(v)
			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, false, err
			}
			parts = append(parts, part)
		case *ai.VideoContent:
			part, err := videoPart(v)
			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, false, err
			}
			parts = append(parts, part)
		}
	}
	if len(parts) == 0 {
		return openai.ChatCompletionMessageParamUnion{}, false, nil
	}
	return openai.UserMessage(parts), true, nil
}

func convertAssistantMessage(m *ai.AssistantMessage, model ai.Model, compat resolvedCompat) (openai.ChatCompletionMessageParamUnion, bool) {
	assistant := openai.ChatCompletionAssistantMessageParam{Role: "assistant"}

	// Assistant content always goes out as a plain string: sending content
	// part arrays is non-standard and makes some models mirror the block
	// structure literally in their output.
	var texts, thinkings []string
	thinkingSignature := ""
	var toolCalls []openai.ChatCompletionMessageToolCallUnionParam

	for _, c := range m.Content {
		switch v := c.(type) {
		case *ai.TextContent:
			if strings.TrimSpace(v.Text) != "" {
				texts = append(texts, v.Text)
			}
		case *ai.JSONContent:
			// OpenAI-compatible history still represents structured assistant
			// output as a content string on the wire.
			if len(v.Value) > 0 {
				texts = append(texts, string(v.Value))
			}
		case *ai.ThinkingContent:
			if strings.TrimSpace(v.Thinking) != "" {
				thinkings = append(thinkings, v.Thinking)
				if thinkingSignature == "" {
					thinkingSignature = v.ThinkingSignature
				}
			}
		case *ai.ToolCallContent:
			args, err := json.Marshal(v.Arguments)
			if err != nil {
				args = []byte("{}")
			}
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID:   v.Id,
					Type: "function",
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      v.Name,
						Arguments: string(args),
					},
				},
			})
		}
	}

	if text := strings.Join(texts, ""); text != "" {
		assistant.Content.OfString = openai.String(text)
	}
	// Replay thinking under the field name it originally arrived in
	// (recorded as the ThinkingSignature); endpoints like llama.cpp and
	// gpt-oss servers need it back for multi-turn reasoning.
	if len(thinkings) > 0 {
		if thinkingSignature == "" {
			thinkingSignature = "reasoning_content"
		}
		assistant.ExtraFields = map[string]any{
			thinkingSignature: strings.Join(thinkings, "\n"),
		}
	}
	assistant.ToolCalls = toolCalls

	if compat.requiresReasoningContentOnAssistantMessages && model.Reasoning {
		if assistant.ExtraFields == nil {
			assistant.ExtraFields = map[string]any{}
		}
		if _, ok := assistant.ExtraFields["reasoning_content"]; !ok {
			assistant.ExtraFields["reasoning_content"] = ""
		}
	}

	// Skip assistant turns with no content and no tool calls (e.g. aborted
	// responses); providers reject empty assistant messages.
	if assistant.Content.OfString == nil && len(toolCalls) == 0 {
		return openai.ChatCompletionMessageParamUnion{}, false
	}
	return openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant}, true
}

func convertToolResult(tr ai.ToolResult) (openai.ChatCompletionMessageParamUnion, []openai.ChatCompletionContentPartUnionParam, error) {
	toolCallID, _, content, _ := tr.ToolResultData()

	var texts []string
	var images []openai.ChatCompletionContentPartUnionParam
	for _, c := range content {
		switch v := c.(type) {
		case *ai.TextContent:
			texts = append(texts, v.Text)
		case *ai.ImageContent:
			ref, err := imageRef(v)
			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, nil, err
			}
			images = append(images, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: ref,
			}))
		}
	}

	text := strings.Join(texts, "\n")
	if text == "" {
		if len(images) > 0 {
			text = "(see attached image)"
		} else {
			text = "(no tool output)"
		}
	}
	return openai.ToolMessage(text, toolCallID), images, nil
}

// Media encoding: this layer never downloads or transcodes. URL forms are
// passed through verbatim and inline data goes out as base64; an endpoint
// that cannot accept the given form rejects the request, which surfaces
// loudly as an ErrorEvent.

func imageRef(img *ai.ImageContent) (string, error) {
	if err := img.Validate(); err != nil {
		return "", err
	}
	if img.URL != "" {
		return img.URL, nil
	}
	return fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Data), nil
}

func audioPart(a *ai.AudioContent) (openai.ChatCompletionContentPartUnionParam, error) {
	if err := a.Validate(); err != nil {
		return openai.ChatCompletionContentPartUnionParam{}, err
	}
	// The OpenAI standard takes base64 in input_audio.data; DashScope also
	// accepts a URL in the same slot.
	if a.URL != "" {
		return openai.InputAudioContentPart(openai.ChatCompletionContentPartInputAudioInputAudioParam{
			Data: a.URL,
		}), nil
	}
	return openai.InputAudioContentPart(openai.ChatCompletionContentPartInputAudioInputAudioParam{
		Data:   a.Data,
		Format: audioFormat(a.MimeType),
	}), nil
}

func audioFormat(mimeType string) string {
	switch mimeType {
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/wav", "audio/x-wav", "audio/wave":
		return "wav"
	default:
		return strings.TrimPrefix(mimeType, "audio/")
	}
}

func videoPart(v *ai.VideoContent) (openai.ChatCompletionContentPartUnionParam, error) {
	if err := v.Validate(); err != nil {
		return openai.ChatCompletionContentPartUnionParam{}, err
	}
	url := v.URL
	if url == "" {
		url = fmt.Sprintf("data:%s;base64,%s", v.MimeType, v.Data)
	}
	return openai.VideoContentPart(openai.ChatCompletionContentPartVideoVideoURLParam{URL: url}), nil
}
