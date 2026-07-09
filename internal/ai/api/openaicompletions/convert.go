package openaicompletions

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/zhongys/Daksha/internal/ai"
	"github.com/zhongys/Daksha/internal/ai/openai"
)

// resolvedCompat is the fully-resolved quirk configuration for one request:
// auto-detected from the provider's base URL, then overridden by model.Compat.
type resolvedCompat struct {
	maxTokensField                              string
	thinkingFormat                              ai.ThinkingFormat
	supportsDeveloperRole                       bool
	requiresReasoningContentOnAssistantMessages bool
}

func detectCompat(baseURL string) resolvedCompat {
	isDeepSeek := strings.Contains(baseURL, "deepseek.com")
	isQwen := strings.Contains(baseURL, "dashscope")
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
	if isDeepSeek || isQwen || isZai || isMoonshot {
		c.maxTokensField = "max_tokens"
		c.supportsDeveloperRole = false
	}
	return c
}

func resolveCompat(provider ai.Provider, model ai.Model) resolvedCompat {
	c := detectCompat(provider.BaseURL)
	o := model.Compat
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
	compat := resolveCompat(provider, model)

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

	for _, tool := range prompt.Tools {
		params.Tools = append(params.Tools, openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        tool.Name,
			Description: openai.String(tool.Description),
			Parameters:  tool.Parameters,
		}))
	}

	applyThinkingFormat(&params, model, compat, opts.ReasoningEffort)

	// Precedence: provider < model < request; explicit extras win over
	// everything, including the thinking-format fields above.
	for _, extra := range []map[string]any{provider.Extra, model.Extra, opts.Extra} {
		for k, v := range extra {
			params.ExtraFields[k] = v
		}
	}
	return params, nil
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
		params.ExtraFields["enable_thinking"] = enabled
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
			if p, ok := convertUserMessage(m); ok {
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
				toolMsg, imgs := convertToolResult(tr)
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

func convertUserMessage(m *ai.UserMessage) (openai.ChatCompletionMessageParamUnion, bool) {
	// Single text block goes out as a plain string, the most compatible form.
	if len(m.Content) == 1 {
		if t, ok := m.Content[0].(*ai.TextContent); ok {
			return openai.UserMessage(t.Text), true
		}
	}

	var parts []openai.ChatCompletionContentPartUnionParam
	for _, c := range m.Content {
		switch v := c.(type) {
		case *ai.TextContent:
			parts = append(parts, openai.TextContentPart(v.Text))
		case *ai.ImageContent:
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: imageURL(v),
			}))
		}
	}
	if len(parts) == 0 {
		return openai.ChatCompletionMessageParamUnion{}, false
	}
	return openai.UserMessage(parts), true
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

func convertToolResult(tr ai.ToolResult) (openai.ChatCompletionMessageParamUnion, []openai.ChatCompletionContentPartUnionParam) {
	toolCallID, _, content, _ := tr.ToolResultData()

	var texts []string
	var images []openai.ChatCompletionContentPartUnionParam
	for _, c := range content {
		switch v := c.(type) {
		case *ai.TextContent:
			texts = append(texts, v.Text)
		case *ai.ImageContent:
			images = append(images, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: imageURL(v),
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
	return openai.ToolMessage(text, toolCallID), images
}

func imageURL(img *ai.ImageContent) string {
	if strings.HasPrefix(img.Data, "http://") || strings.HasPrefix(img.Data, "https://") {
		return img.Data
	}
	return fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Data)
}
