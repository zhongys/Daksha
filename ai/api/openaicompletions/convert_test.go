package openaicompletions

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/openai"
)

func marshalParams(t *testing.T, p openai.ChatCompletionNewParams) string {
	t.Helper()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	return string(b)
}

func userText(text string) *ai.UserMessage {
	return &ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{
		&ai.TextContent{Type: ai.ContentTypeText, Text: text},
	}}
}

func TestBuildParamsThinkingFormats(t *testing.T) {
	prompt := ai.Prompt{Messages: []ai.Message{userText("hi")}}

	tests := []struct {
		name     string
		provider ai.Provider
		model    ai.Model
		opts     ai.StreamOptions
		want     []string
		notWant  []string
	}{
		{
			name:     "deepseek thinking enabled",
			provider: ai.Provider{Name: "deepseek", BaseURL: "https://api.deepseek.com/"},
			model:    ai.Model{Provider: "deepseek", ID: "deepseek-reasoner", Reasoning: true},
			opts:     ai.StreamOptions{ReasoningEffort: "high"},
			want:     []string{`"thinking":{"type":"enabled"}`, `"reasoning_effort":"high"`},
		},
		{
			name:     "deepseek thinking disabled",
			provider: ai.Provider{Name: "deepseek", BaseURL: "https://api.deepseek.com/"},
			model:    ai.Model{Provider: "deepseek", ID: "deepseek-chat", Reasoning: true},
			want:     []string{`"thinking":{"type":"disabled"}`},
			notWant:  []string{"reasoning_effort"},
		},
		{
			name:     "qwen enable_thinking",
			provider: ai.Provider{Name: "qwen", BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1/"},
			model:    ai.Model{Provider: "qwen", ID: "qwen-plus", Reasoning: true},
			opts:     ai.StreamOptions{ReasoningEffort: "medium"},
			want:     []string{`"enable_thinking":true`},
		},
		{
			name:     "openrouter reasoning effort",
			provider: ai.Provider{Name: "openrouter", BaseURL: "https://openrouter.ai/api/v1/"},
			model:    ai.Model{Provider: "openrouter", ID: "deepseek/deepseek-r1", Reasoning: true},
			opts:     ai.StreamOptions{ReasoningEffort: "low"},
			want:     []string{`"reasoning":{"effort":"low"}`},
		},
		{
			name:     "openai style reasoning_effort",
			provider: ai.Provider{Name: "openai", BaseURL: "https://api.openai.com/v1/"},
			model:    ai.Model{Provider: "openai", ID: "o4-mini", Reasoning: true},
			opts:     ai.StreamOptions{ReasoningEffort: "high"},
			want:     []string{`"reasoning_effort":"high"`},
		},
		{
			name:     "non-reasoning model gets no thinking fields",
			provider: ai.Provider{Name: "deepseek", BaseURL: "https://api.deepseek.com/"},
			model:    ai.Model{Provider: "deepseek", ID: "deepseek-chat"},
			opts:     ai.StreamOptions{ReasoningEffort: "high"},
			notWant:  []string{"thinking", "reasoning_effort", "enable_thinking"},
		},
		{
			name: "compat override wins over detection",
			provider: ai.Provider{
				Name: "custom", BaseURL: "https://my-vllm.internal/v1/",
				Compat: &ai.OpenAICompat{ThinkingFormat: ai.ThinkingFormatQwen},
			},
			model: ai.Model{Provider: "custom", ID: "m", Reasoning: true},
			opts:  ai.StreamOptions{ReasoningEffort: "low"},
			want:  []string{`"enable_thinking":true`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := buildParams(tt.provider, tt.model, prompt, tt.opts)
			if err != nil {
				t.Fatalf("buildParams: %v", err)
			}
			got := marshalParams(t, params)
			for _, w := range tt.want {
				if !strings.Contains(got, w) {
					t.Errorf("missing %q in %s", w, got)
				}
			}
			for _, nw := range tt.notWant {
				if strings.Contains(got, nw) {
					t.Errorf("unexpected %q in %s", nw, got)
				}
			}
		})
	}
}

func TestBuildParamsMaxTokensField(t *testing.T) {
	prompt := ai.Prompt{Messages: []ai.Message{userText("hi")}}
	opts := ai.StreamOptions{MaxTokens: openai.Int(500)}

	// OpenAI: max_completion_tokens.
	params, _ := buildParams(ai.Provider{BaseURL: "https://api.openai.com/v1/"}, ai.Model{ID: "gpt-5"}, prompt, opts)
	if got := marshalParams(t, params); !strings.Contains(got, `"max_completion_tokens":500`) {
		t.Errorf("openai: %s", got)
	}

	// DeepSeek: max_tokens.
	params, _ = buildParams(ai.Provider{BaseURL: "https://api.deepseek.com/"}, ai.Model{ID: "deepseek-chat"}, prompt, opts)
	got := marshalParams(t, params)
	if !strings.Contains(got, `"max_tokens":500`) || strings.Contains(got, "max_completion_tokens") {
		t.Errorf("deepseek: %s", got)
	}
}

func TestBuildParamsExtraPrecedence(t *testing.T) {
	prompt := ai.Prompt{Messages: []ai.Message{userText("hi")}}
	provider := ai.Provider{
		BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1/",
		Extra:   map[string]any{"enable_thinking": true, "provider_only": 1},
	}
	model := ai.Model{ID: "qwen-plus", Extra: map[string]any{"enable_thinking": false}}
	opts := ai.StreamOptions{Extra: map[string]any{"request_only": 2}}

	params, err := buildParams(provider, model, prompt, opts)
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	got := marshalParams(t, params)

	// Model.Extra overrides Provider.Extra; both non-conflicting keys survive.
	for _, w := range []string{`"enable_thinking":false`, `"provider_only":1`, `"request_only":2`} {
		if !strings.Contains(got, w) {
			t.Errorf("missing %q in %s", w, got)
		}
	}
}

func TestConvertMessagesSystemRole(t *testing.T) {
	prompt := ai.Prompt{System: "be brief", Messages: []ai.Message{userText("hi")}}

	// Reasoning model on an endpoint that supports it: developer role.
	msgs, err := convertMessages(prompt, ai.Model{Reasoning: true}, resolvedCompat{supportsDeveloperRole: true})
	if err != nil {
		t.Fatal(err)
	}
	if msgs[0].OfDeveloper == nil {
		t.Errorf("want developer role, got %+v", msgs[0])
	}

	// DeepSeek-style endpoint: system role even for reasoning models.
	msgs, _ = convertMessages(prompt, ai.Model{Reasoning: true}, resolvedCompat{supportsDeveloperRole: false})
	if msgs[0].OfSystem == nil {
		t.Errorf("want system role, got %+v", msgs[0])
	}
}

func TestConvertAssistantMessage(t *testing.T) {
	m := &ai.AssistantMessage{
		Role: ai.RoleAssistant,
		Content: []ai.AssistantContent{
			&ai.ThinkingContent{Type: ai.ContentTypeThinking, Thinking: "step 1", ThinkingSignature: "reasoning_content"},
			&ai.TextContent{Type: ai.ContentTypeText, Text: "answer"},
			&ai.ToolCallContent{
				Type: ai.ContentTypeToolCall, Id: "c1", Name: "read_file",
				Arguments: map[string]any{"path": "/tmp/x"},
			},
		},
	}

	p, ok := convertAssistantMessage(m, ai.Model{Reasoning: true}, resolvedCompat{})
	if !ok {
		t.Fatal("message dropped")
	}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)

	// Thinking replayed under its original field name; text as plain string;
	// tool call arguments as a JSON string.
	for _, w := range []string{
		`"reasoning_content":"step 1"`,
		`"content":"answer"`,
		`"id":"c1"`,
		`"name":"read_file"`,
		`"arguments":"{\"path\":\"/tmp/x\"}"`,
	} {
		if !strings.Contains(got, w) {
			t.Errorf("missing %q in %s", w, got)
		}
	}
}

func TestConvertAssistantMessageDeepSeekEmptyReasoning(t *testing.T) {
	m := &ai.AssistantMessage{
		Role:    ai.RoleAssistant,
		Content: []ai.AssistantContent{&ai.TextContent{Type: ai.ContentTypeText, Text: "plain answer"}},
	}
	compat := resolvedCompat{requiresReasoningContentOnAssistantMessages: true}

	p, ok := convertAssistantMessage(m, ai.Model{Reasoning: true}, compat)
	if !ok {
		t.Fatal("message dropped")
	}
	b, _ := json.Marshal(p)
	if !strings.Contains(string(b), `"reasoning_content":""`) {
		t.Errorf("missing empty reasoning_content: %s", b)
	}
}

func TestConvertAssistantMessageSkipsEmpty(t *testing.T) {
	m := &ai.AssistantMessage{Role: ai.RoleAssistant} // aborted turn: no content
	if _, ok := convertAssistantMessage(m, ai.Model{}, resolvedCompat{}); ok {
		t.Error("empty assistant message should be dropped")
	}
}

func TestConvertToolResultWithImages(t *testing.T) {
	prompt := ai.Prompt{Messages: []ai.Message{
		&ai.ToolResultMessage[struct{}]{
			Role: ai.RoleToolResult, ToolCallId: "c1", ToolName: "screenshot",
			Content: []ai.ToolResultContent{
				&ai.TextContent{Type: ai.ContentTypeText, Text: "took screenshot"},
				&ai.ImageContent{Type: ai.ContentTypeImage, Data: "AAAA", MimeType: "image/png"},
			},
		},
	}}

	msgs, err := convertMessages(prompt, ai.Model{}, resolvedCompat{})
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("want tool message + image user message, got %d messages", len(msgs))
	}
	if msgs[0].OfTool == nil || msgs[0].OfTool.ToolCallID != "c1" {
		t.Errorf("first message = %+v", msgs[0])
	}
	b, _ := json.Marshal(msgs[1])
	if !strings.Contains(string(b), `data:image/png;base64,AAAA`) {
		t.Errorf("image user message = %s", b)
	}
}

func TestConvertToolResultPlaceholders(t *testing.T) {
	// Image-only result gets a placeholder text; empty result too.
	tr := &ai.ToolResultMessage[struct{}]{
		Role: ai.RoleToolResult, ToolCallId: "c1",
		Content: []ai.ToolResultContent{
			&ai.ImageContent{Type: ai.ContentTypeImage, Data: "AAAA", MimeType: "image/png"},
		},
	}
	msg, imgs, err := convertToolResult(tr)
	if err != nil {
		t.Fatalf("convertToolResult: %v", err)
	}
	if got := *msg.OfTool.Content.OfString; got != "(see attached image)" {
		t.Errorf("placeholder = %q", got)
	}
	if len(imgs) != 1 {
		t.Errorf("images = %d", len(imgs))
	}

	empty := &ai.ToolResultMessage[struct{}]{Role: ai.RoleToolResult, ToolCallId: "c2"}
	msg, _, err = convertToolResult(empty)
	if err != nil {
		t.Fatalf("convertToolResult: %v", err)
	}
	if got := *msg.OfTool.Content.OfString; got != "(no tool output)" {
		t.Errorf("placeholder = %q", got)
	}
}

func TestConvertUserMessageImageURL(t *testing.T) {
	m := &ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{
		&ai.TextContent{Type: ai.ContentTypeText, Text: "look"},
		&ai.ImageContent{Type: ai.ContentTypeImage, URL: "https://x/1.png"},
	}}
	p, ok, err := convertUserMessage(m)
	if err != nil {
		t.Fatalf("convertUserMessage: %v", err)
	}
	if !ok {
		t.Fatal("dropped")
	}
	b, _ := json.Marshal(p)
	// The URL form passes through verbatim, not as a data URI.
	if !strings.Contains(string(b), `"url":"https://x/1.png"`) {
		t.Errorf("got %s", b)
	}
}

func TestConvertUserMessageMedia(t *testing.T) {
	m := &ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{
		&ai.AudioContent{Type: ai.ContentTypeAudio, Data: "QUFB", MimeType: "audio/wav"},
		&ai.VideoContent{Type: ai.ContentTypeVideo, URL: "https://x/v.mp4"},
	}}
	p, ok, err := convertUserMessage(m)
	if err != nil {
		t.Fatalf("convertUserMessage: %v", err)
	}
	if !ok {
		t.Fatal("dropped")
	}
	b, _ := json.Marshal(p)
	for _, want := range []string{
		`"type":"input_audio"`, `"data":"QUFB"`, `"format":"wav"`,
		`"type":"video_url"`, `"url":"https://x/v.mp4"`,
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("missing %q in %s", want, b)
		}
	}
}

func TestConvertUserMessageMediaUnionErrors(t *testing.T) {
	for _, c := range []ai.UserContent{
		&ai.ImageContent{Type: ai.ContentTypeImage},                                                        // neither form
		&ai.AudioContent{Type: ai.ContentTypeAudio, Data: "QUFB", MimeType: "audio/wav", URL: "https://x"}, // both forms
		&ai.VideoContent{Type: ai.ContentTypeVideo, Data: "QUFB"},                                          // data without mime
	} {
		m := &ai.UserMessage{Role: ai.RoleUser, Content: []ai.UserContent{c}}
		if _, _, err := convertUserMessage(m); err == nil {
			t.Errorf("%T should be rejected", c)
		}
	}
}
