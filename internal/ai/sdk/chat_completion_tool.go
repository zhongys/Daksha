package sdk

import (
	"encoding/json"

	"github.com/zhongys/Daksha.git/internal/ai/sdk/shared"
)

func ChatCompletionFunctionTool(function shared.FunctionDefinitionParam) ChatCompletionToolUnionParam {
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
	Function shared.FunctionDefinitionParam `json:"function,omitzero" api:"required"`
	// The type of the tool. Currently, only `function` is supported.
	//
	// This field can be elided, and will marshal its zero value as "function".
	Type string `json:"type" default:"function"`
}

type ChatCompletionCustomToolParam struct {
	// Properties of the custom tool.
	Custom ChatCompletionCustomToolCustomParam `json:"custom,omitzero" api:"required"`
	// The type of the custom tool. Always `custom`.
	//
	// This field can be elided, and will marshal its zero value as "custom".
	Type string `json:"type" default:"custom"`
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionToolUnionParam struct {
	OfFunction *ChatCompletionFunctionToolParam `json:",omitzero,inline"`
	OfCustom   *ChatCompletionCustomToolParam   `json:",omitzero,inline"`
}

func (u ChatCompletionToolUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfFunction != nil:
		return json.Marshal(u.OfFunction)
	case u.OfCustom != nil:
		return json.Marshal(u.OfCustom)
	default:
		return []byte("null"), nil
	}
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolUnionParam) GetFunction() *shared.FunctionDefinitionParam {
	if u.OfFunction == nil {
		return nil
	}
	return &u.OfFunction.Function
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolUnionParam) GetCustom() *ChatCompletionCustomToolCustomParam {
	if u.OfCustom == nil {
		return nil
	}
	return &u.OfCustom.Custom
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolUnionParam) GetType() string {
	switch {
	case u.OfFunction != nil:
		return u.OfFunction.Type
	case u.OfCustom != nil:
		return u.OfCustom.Type
	default:
		return ""
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
	AllowedTools ChatCompletionAllowedToolsParam `json:"allowed_tools,omitzero" api:"required"`
	// Allowed tool configuration type. Always `allowed_tools`.
	//
	// This field can be elided, and will marshal its zero value as "allowed_tools".
	Type string `json:"type" default:"allowed_tools"`
}
type ChatCompletionNamedToolChoiceParam struct {
	Function ChatCompletionNamedToolChoiceFunctionParam `json:"function,omitzero" api:"required"`
	// For function calling, the type is always `function`.
	//
	// This field can be elided, and will marshal its zero value as "function".
	Type string `json:"type" default:"function"`
}

// The property Name is required.
type ChatCompletionNamedToolChoiceFunctionParam struct {
	// The name of the function to call.
	Name string `json:"name" api:"required"`
}

// Specifies a tool the model should use. Use to force the model to call a specific
// custom tool.
//
// The properties Custom, Type are required.
type ChatCompletionNamedToolChoiceCustomParam struct {
	Custom ChatCompletionNamedToolChoiceCustomCustomParam `json:"custom,omitzero" api:"required"`
	// For custom tool calling, the type is always `custom`.
	//
	// This field can be elided, and will marshal its zero value as "custom".
	Type string `json:"type" default:"custom"`
}

// The property Name is required.
type ChatCompletionNamedToolChoiceCustomCustomParam struct {
	// The name of the custom tool to call.
	Name string `json:"name" api:"required"`
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionToolChoiceOptionUnionParam struct {
	// Check if union is this variant with !param.IsOmitted(union.OfAuto)
	OfAuto               *string                                   `json:",omitzero,inline"`
	OfAllowedTools       *ChatCompletionAllowedToolChoiceParam     `json:",omitzero,inline"`
	OfFunctionToolChoice *ChatCompletionNamedToolChoiceParam       `json:",omitzero,inline"`
	OfCustomToolChoice   *ChatCompletionNamedToolChoiceCustomParam `json:",omitzero,inline"`
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
		return []byte("null"), nil
	}
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolChoiceOptionUnionParam) GetAllowedTools() *ChatCompletionAllowedToolsParam {
	if u.OfAllowedTools == nil {
		return nil
	}
	return &u.OfAllowedTools.AllowedTools
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolChoiceOptionUnionParam) GetFunction() *ChatCompletionNamedToolChoiceFunctionParam {
	if u.OfFunctionToolChoice == nil {
		return nil
	}
	return &u.OfFunctionToolChoice.Function
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolChoiceOptionUnionParam) GetCustom() *ChatCompletionNamedToolChoiceCustomCustomParam {
	if u.OfCustomToolChoice == nil {
		return nil
	}
	return &u.OfCustomToolChoice.Custom
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionToolChoiceOptionUnionParam) GetType() string {
	switch {
	case u.OfAllowedTools != nil:
		return u.OfAllowedTools.Type
	case u.OfFunctionToolChoice != nil:
		return u.OfFunctionToolChoice.Type
	case u.OfCustomToolChoice != nil:
		return u.OfCustomToolChoice.Type
	default:
		return ""
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

// The properties Content, Role, ToolCallID are required.
type ChatCompletionToolMessageParam struct {
	// The contents of the tool message.
	Content ChatCompletionToolMessageParamContentUnion `json:"content,omitzero" api:"required"`
	// Tool call that this message is responding to.
	ToolCallID string `json:"tool_call_id" api:"required"`
	// The role of the messages author, in this case `tool`.
	//
	// This field can be elided, and will marshal its zero value as "tool".
	Role string `json:"role" default:"tool"`
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionToolMessageParamContentUnion struct {
	OfString              *string                              `json:",omitzero,inline"`
	OfArrayOfContentParts []ChatCompletionContentPartTextParam `json:",omitzero,inline"`
}

func (u ChatCompletionToolMessageParamContentUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return []byte("null"), nil
	}
}

type ChatCompletionAllowedToolsParam struct {
	// Constrains the tools available to the model to a pre-defined set.
	//
	// `auto` allows the model to pick from among the allowed tools and generate a
	// message.
	//
	// `required` requires the model to call one or more of the allowed tools.
	//
	// Any of "auto", "required".
	Mode ChatCompletionAllowedToolsMode `json:"mode,omitzero" api:"required"`
	// A list of tool definitions that the model should be allowed to call.
	//
	// For the Chat Completions API, the list of tool definitions might look like:
	//
	// ```json
	// [
	//
	//	{ "type": "function", "function": { "name": "get_weather" } },
	//	{ "type": "function", "function": { "name": "get_time" } }
	//
	// ]
	// ```
	Tools []map[string]any `json:"tools,omitzero" api:"required"`
}

// Constrains the tools available to the model to a pre-defined set.
//
// `auto` allows the model to pick from among the allowed tools and generate a
// message.
//
// `required` requires the model to call one or more of the allowed tools.
type ChatCompletionAllowedToolsMode string

const (
	ChatCompletionAllowedToolsModeAuto     ChatCompletionAllowedToolsMode = "auto"
	ChatCompletionAllowedToolsModeRequired ChatCompletionAllowedToolsMode = "required"
)

type ChatCompletionCustomToolCustomParam struct {
	// The name of the custom tool, used to identify it in tool calls.
	Name string `json:"name" api:"required"`
	// Optional description of the custom tool, used to provide more context.
	Description *string `json:"description,omitzero"`
	// The input format for the custom tool. Default is unconstrained text.
	Format ChatCompletionCustomToolCustomFormatUnionParam `json:"format,omitzero"`
}

type ChatCompletionCustomToolCustomFormatUnionParam struct {
	OfText    *ChatCompletionCustomToolCustomFormatTextParam    `json:",omitzero,inline"`
	OfGrammar *ChatCompletionCustomToolCustomFormatGrammarParam `json:",omitzero,inline"`
}

func (u ChatCompletionCustomToolCustomFormatUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfText != nil:
		return json.Marshal(u.OfText)
	case u.OfGrammar != nil:
		return json.Marshal(u.OfGrammar)
	default:
		return []byte("null"), nil
	}
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionCustomToolCustomFormatUnionParam) GetGrammar() *ChatCompletionCustomToolCustomFormatGrammarGrammarParam {
	if u.OfGrammar == nil {
		return nil
	}
	return &u.OfGrammar.Grammar
}

func NewChatCompletionCustomToolCustomFormatTextParam() ChatCompletionCustomToolCustomFormatTextParam {
	return ChatCompletionCustomToolCustomFormatTextParam{
		Type: "text",
	}
}

type ChatCompletionCustomToolCustomFormatTextParam struct {
	// Unconstrained text format. Always `text`.
	Type string `json:"type" default:"text"`
}

type ChatCompletionCustomToolCustomFormatGrammarParam struct {
	// Your chosen grammar.
	Grammar ChatCompletionCustomToolCustomFormatGrammarGrammarParam `json:"grammar,omitzero" api:"required"`
	// Grammar format. Always `grammar`.
	//
	// This field can be elided, and will marshal its zero value as "grammar".
	Type string `json:"type" default:"grammar"`
}

type ChatCompletionCustomToolCustomFormatGrammarGrammarParam struct {
	// The grammar definition.
	Definition string `json:"definition" api:"required"`
	// The syntax of the grammar definition. One of `lark` or `regex`.
	//
	// Any of "lark", "regex".
	Syntax string `json:"syntax,omitzero" api:"required"`
}

type ChatCompletionFunctionCallOptionParam struct {
	// The name of the function to call.
	Name string `json:"name" api:"required"`
}

type ChatCompletionFunctionMessageParam struct {
	// The contents of the function message.
	Content *string `json:"content,omitzero" api:"required"`
	// The name of the function to call.
	Name string `json:"name" api:"required"`
	// The role of the messages author, in this case `function`.
	//
	// This field can be elided, and will marshal its zero value as "function".
	Role string `json:"role" default:"function"`
}
