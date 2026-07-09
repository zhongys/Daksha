package openai

import (
	"encoding/json"
	"fmt"
)

type FunctionDefinitionParam struct {
	Name        string         `json:"name"`
	Description *string        `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

func ChatCompletionFunctionTool(function FunctionDefinitionParam) ChatCompletionToolUnionParam {
	return ChatCompletionToolUnionParam{OfFunction: &ChatCompletionFunctionToolParam{
		Type:     "function",
		Function: function,
	}}
}

func ChatCompletionCustomTool(custom ChatCompletionCustomToolCustomParam) ChatCompletionToolUnionParam {
	return ChatCompletionToolUnionParam{OfCustom: &ChatCompletionCustomToolParam{
		Type:   "custom",
		Custom: custom,
	}}
}

type ChatCompletionFunctionToolParam struct {
	Function FunctionDefinitionParam `json:"function"`
	// The type of the tool. Defaults to "function" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionFunctionToolParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "function"
	}
	type shadow ChatCompletionFunctionToolParam
	return json.Marshal(shadow(r))
}

type ChatCompletionCustomToolParam struct {
	// Properties of the custom tool.
	Custom ChatCompletionCustomToolCustomParam `json:"custom"`
	// The type of the custom tool. Defaults to "custom" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionCustomToolParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "custom"
	}
	type shadow ChatCompletionCustomToolParam
	return json.Marshal(shadow(r))
}

// ChatCompletionToolUnionParam holds exactly one tool variant.
type ChatCompletionToolUnionParam struct {
	OfFunction *ChatCompletionFunctionToolParam `json:"-"`
	OfCustom   *ChatCompletionCustomToolParam   `json:"-"`
}

func (u ChatCompletionToolUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfFunction != nil:
		return json.Marshal(u.OfFunction)
	case u.OfCustom != nil:
		return json.Marshal(u.OfCustom)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionToolUnionParam")
	}
}

func ToolChoiceOptionAllowedTools(allowedTools ChatCompletionAllowedToolsParam) ChatCompletionToolChoiceOptionUnionParam {
	return ChatCompletionToolChoiceOptionUnionParam{OfAllowedTools: &ChatCompletionAllowedToolChoiceParam{
		Type:         "allowed_tools",
		AllowedTools: allowedTools,
	}}
}

func ToolChoiceOptionFunctionToolChoice(function ChatCompletionNamedToolChoiceFunctionParam) ChatCompletionToolChoiceOptionUnionParam {
	return ChatCompletionToolChoiceOptionUnionParam{OfFunctionToolChoice: &ChatCompletionNamedToolChoiceParam{
		Type:     "function",
		Function: function,
	}}
}

func ToolChoiceOptionCustomToolChoice(custom ChatCompletionNamedToolChoiceCustomCustomParam) ChatCompletionToolChoiceOptionUnionParam {
	return ChatCompletionToolChoiceOptionUnionParam{OfCustomToolChoice: &ChatCompletionNamedToolChoiceCustomParam{
		Type:   "custom",
		Custom: custom,
	}}
}

type ChatCompletionAllowedToolChoiceParam struct {
	// Constrains the tools available to the model to a pre-defined set.
	AllowedTools ChatCompletionAllowedToolsParam `json:"allowed_tools"`
	// Allowed tool configuration type. Defaults to "allowed_tools" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionAllowedToolChoiceParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "allowed_tools"
	}
	type shadow ChatCompletionAllowedToolChoiceParam
	return json.Marshal(shadow(r))
}

type ChatCompletionNamedToolChoiceParam struct {
	Function ChatCompletionNamedToolChoiceFunctionParam `json:"function"`
	// For function calling, the type is always `function`. Defaults to
	// "function" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionNamedToolChoiceParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "function"
	}
	type shadow ChatCompletionNamedToolChoiceParam
	return json.Marshal(shadow(r))
}

type ChatCompletionNamedToolChoiceFunctionParam struct {
	// The name of the function to call.
	Name string `json:"name"`
}

// Specifies a tool the model should use. Use to force the model to call a
// specific custom tool.
type ChatCompletionNamedToolChoiceCustomParam struct {
	Custom ChatCompletionNamedToolChoiceCustomCustomParam `json:"custom"`
	// For custom tool calling, the type is always `custom`. Defaults to
	// "custom" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionNamedToolChoiceCustomParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "custom"
	}
	type shadow ChatCompletionNamedToolChoiceCustomParam
	return json.Marshal(shadow(r))
}

type ChatCompletionNamedToolChoiceCustomCustomParam struct {
	// The name of the custom tool to call.
	Name string `json:"name"`
}

// ChatCompletionToolChoiceOptionUnionParam holds exactly one tool choice
// variant. OfAuto carries the string form: "none", "auto" or "required".
type ChatCompletionToolChoiceOptionUnionParam struct {
	OfAuto               *string                                   `json:"-"`
	OfAllowedTools       *ChatCompletionAllowedToolChoiceParam     `json:"-"`
	OfFunctionToolChoice *ChatCompletionNamedToolChoiceParam       `json:"-"`
	OfCustomToolChoice   *ChatCompletionNamedToolChoiceCustomParam `json:"-"`
}

func (u ChatCompletionToolChoiceOptionUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfAuto != nil:
		return json.Marshal(u.OfAuto)
	case u.OfAllowedTools != nil:
		return json.Marshal(u.OfAllowedTools)
	case u.OfFunctionToolChoice != nil:
		return json.Marshal(u.OfFunctionToolChoice)
	case u.OfCustomToolChoice != nil:
		return json.Marshal(u.OfCustomToolChoice)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionToolChoiceOptionUnionParam")
	}
}

// `none` means the model will not call any tool and instead generates a message.
// `auto` means the model can pick between generating a message or calling one or
// more tools. `required` means the model must call one or more tools.
type ChatCompletionToolChoiceOptionAuto string

const (
	ChatCompletionToolChoiceOptionAutoNone     ChatCompletionToolChoiceOptionAuto = "none"
	ChatCompletionToolChoiceOptionAutoAuto     ChatCompletionToolChoiceOptionAuto = "auto"
	ChatCompletionToolChoiceOptionAutoRequired ChatCompletionToolChoiceOptionAuto = "required"
)

type ChatCompletionToolMessageParam struct {
	// The contents of the tool message.
	Content ChatCompletionToolMessageParamContentUnion `json:"content,omitzero"`
	// Tool call that this message is responding to.
	ToolCallID string `json:"tool_call_id"`
	// The role of the messages author, in this case `tool`.
	// Defaults to "tool" when left empty.
	Role string `json:"role"`
}

func (r ChatCompletionToolMessageParam) MarshalJSON() ([]byte, error) {
	if r.Role == "" {
		r.Role = "tool"
	}
	type shadow ChatCompletionToolMessageParam
	return json.Marshal(shadow(r))
}

type ChatCompletionToolMessageParamContentUnion struct {
	OfString              *string                              `json:"-"`
	OfArrayOfContentParts []ChatCompletionContentPartTextParam `json:"-"`
}

func (u ChatCompletionToolMessageParamContentUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionToolMessageParamContentUnion")
	}
}

type ChatCompletionAllowedToolsParam struct {
	// Constrains the tools available to the model to a pre-defined set.
	//
	// `auto` allows the model to pick from among the allowed tools and generate a
	// message. `required` requires the model to call one or more of the allowed
	// tools.
	//
	// Any of "auto", "required".
	Mode ChatCompletionAllowedToolsMode `json:"mode"`
	// A list of tool definitions that the model should be allowed to call, e.g.
	// [{"type": "function", "function": {"name": "get_weather"}}].
	Tools []map[string]any `json:"tools"`
}

type ChatCompletionAllowedToolsMode string

const (
	ChatCompletionAllowedToolsModeAuto     ChatCompletionAllowedToolsMode = "auto"
	ChatCompletionAllowedToolsModeRequired ChatCompletionAllowedToolsMode = "required"
)

type ChatCompletionCustomToolCustomParam struct {
	// The name of the custom tool, used to identify it in tool calls.
	Name string `json:"name"`
	// Optional description of the custom tool, used to provide more context.
	Description *string `json:"description,omitzero"`
	// The input format for the custom tool. Default is unconstrained text.
	Format ChatCompletionCustomToolCustomFormatUnionParam `json:"format,omitzero"`
}

type ChatCompletionCustomToolCustomFormatUnionParam struct {
	OfText    *ChatCompletionCustomToolCustomFormatTextParam    `json:"-"`
	OfGrammar *ChatCompletionCustomToolCustomFormatGrammarParam `json:"-"`
}

func (u ChatCompletionCustomToolCustomFormatUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfText != nil:
		return json.Marshal(u.OfText)
	case u.OfGrammar != nil:
		return json.Marshal(u.OfGrammar)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionCustomToolCustomFormatUnionParam")
	}
}

type ChatCompletionCustomToolCustomFormatTextParam struct {
	// Unconstrained text format. Defaults to "text" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionCustomToolCustomFormatTextParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "text"
	}
	type shadow ChatCompletionCustomToolCustomFormatTextParam
	return json.Marshal(shadow(r))
}

type ChatCompletionCustomToolCustomFormatGrammarParam struct {
	// Your chosen grammar.
	Grammar ChatCompletionCustomToolCustomFormatGrammarGrammarParam `json:"grammar"`
	// Grammar format. Defaults to "grammar" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionCustomToolCustomFormatGrammarParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "grammar"
	}
	type shadow ChatCompletionCustomToolCustomFormatGrammarParam
	return json.Marshal(shadow(r))
}

type ChatCompletionCustomToolCustomFormatGrammarGrammarParam struct {
	// The grammar definition.
	Definition string `json:"definition"`
	// The syntax of the grammar definition. One of `lark` or `regex`.
	Syntax string `json:"syntax"`
}
