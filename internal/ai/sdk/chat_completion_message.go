package sdk

import (
	"encoding/json"

	"github.com/zhongys/Daksha.git/internal/ai/sdk/option"
	constant "github.com/zhongys/Daksha.git/internal/ai/sdk/shared"
)

type ChatCompletionMessageService struct {
	Options []option.RequestOption
}

// NewChatCompletionMessageService generates a new service that applies the given
// options to each request. These options are applied after the parent client's
// options (if there is one), and before any request-specific options.
func NewChatCompletionMessageService(opts ...option.RequestOption) (r ChatCompletionMessageService) {
	r = ChatCompletionMessageService{}
	r.Options = opts
	return
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionMessageParamUnion struct {
	OfDeveloper *ChatCompletionDeveloperMessageParam `json:",omitzero,inline"`
	OfSystem    *ChatCompletionSystemMessageParam    `json:",omitzero,inline"`
	OfUser      *ChatCompletionUserMessageParam      `json:",omitzero,inline"`
	OfAssistant *ChatCompletionAssistantMessageParam `json:",omitzero,inline"`
	OfTool      *ChatCompletionToolMessageParam      `json:",omitzero,inline"`
	OfFunction  *ChatCompletionFunctionMessageParam  `json:",omitzero,inline"`
}

func (u ChatCompletionMessageParamUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfDeveloper != nil:
		return json.Marshal(u.OfDeveloper)
	case u.OfSystem != nil:
		return json.Marshal(u.OfSystem)
	case u.OfUser != nil:
		return json.Marshal(u.OfUser)
	case u.OfAssistant != nil:
		return json.Marshal(u.OfAssistant)
	case u.OfTool != nil:
		return json.Marshal(u.OfTool)
	case u.OfFunction != nil:
		return json.Marshal(u.OfFunction)
	default:
		return []byte("null"), nil
	}
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageParamUnion) GetAudio() *ChatCompletionAssistantMessageParamAudio {
	if u.OfAssistant == nil {
		return nil
	}
	return &u.OfAssistant.Audio
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageParamUnion) GetRefusal() *string {
	if u.OfAssistant == nil || u.OfAssistant.Refusal == nil {
		return nil
	}
	return u.OfAssistant.Refusal
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageParamUnion) GetToolCalls() []ChatCompletionMessageToolCallUnionParam {
	if u.OfAssistant == nil {
		return nil
	}
	return u.OfAssistant.ToolCalls
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageParamUnion) GetToolCallID() *string {
	if u.OfTool == nil {
		return nil
	}
	return &u.OfTool.ToolCallID
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageParamUnion) GetRole() *string {
	if u.OfDeveloper != nil {
		return &u.OfDeveloper.Role
	} else if u.OfSystem != nil {
		return &u.OfSystem.Role
	} else if u.OfUser != nil {
		return &u.OfUser.Role
	} else if u.OfAssistant != nil {
		return &u.OfAssistant.Role
	} else if u.OfTool != nil {
		return &u.OfTool.Role
	} else if u.OfFunction != nil {
		return &u.OfFunction.Role
	}
	return nil
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageParamUnion) GetName() *string {
	if u.OfDeveloper != nil && u.OfDeveloper.Name != nil {
		return u.OfDeveloper.Name
	} else if u.OfSystem != nil && u.OfSystem.Name != nil {
		return u.OfSystem.Name
	} else if u.OfUser != nil && u.OfUser.Name != nil {
		return u.OfUser.Name.Value
	} else if u.OfAssistant != nil && u.OfAssistant.Name != nil {
		return u.OfAssistant.Name
	} else if u.OfFunction != nil {
		return &u.OfFunction.Name
	}
	return nil
}

// Returns a subunion which exports methods to access subproperties
//
// Or use AsAny() to get the underlying value
func (u ChatCompletionMessageParamUnion) GetContent() (res chatCompletionMessageParamUnionContent) {
	if u.OfDeveloper != nil {
		res.any = u.OfDeveloper.Content
	} else if u.OfSystem != nil {
		res.any = u.OfSystem.Content
	} else if u.OfUser != nil {
		res.any = u.OfUser.Content
	} else if u.OfAssistant != nil {
		res.any = u.OfAssistant.Content
	} else if u.OfTool != nil {
		res.any = u.OfTool.Content
	} else if u.OfFunction != nil && u.OfFunction.Content != nil {
		res.any = &u.OfFunction.Content
	}
	return
}

type ChatCompletionDeveloperMessageParam struct {
	// The contents of the developer message.
	Content ChatCompletionDeveloperMessageParamContentUnion `json:"content,omitzero" api:"required"`
	// An optional name for the participant. Provides the model information to
	// differentiate between participants of the same role.
	Name *string `json:"name,omitzero"`
	// The role of the messages author, in this case `developer`.
	//
	// This field can be elided, and will marshal its zero value as "developer".
	Role string `json:"role" default:"developer"`
}

type ChatCompletionDeveloperMessageParamContentUnion struct {
	OfString              *string                              `json:",omitzero,inline"`
	OfArrayOfContentParts []ChatCompletionContentPartTextParam `json:",omitzero,inline"`
}

func (u ChatCompletionDeveloperMessageParamContentUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return []byte("null"), nil
	}
}

func DeveloperMessageContentString(content string) ChatCompletionDeveloperMessageParamContentUnion {
	return ChatCompletionDeveloperMessageParamContentUnion{
		OfString: &content,
	}
}

func DeveloperMessageContentParts(parts []ChatCompletionContentPartTextParam) ChatCompletionDeveloperMessageParamContentUnion {
	return ChatCompletionDeveloperMessageParamContentUnion{
		OfArrayOfContentParts: parts,
	}
}

type ChatCompletionSystemMessageParam struct {
	// The contents of the system message.
	Content ChatCompletionSystemMessageParamContentUnion `json:"content,omitzero" api:"required"`
	// An optional name for the participant. Provides the model information to
	// differentiate between participants of the same role.
	Name *string `json:"name,omitzero"`
	// The role of the messages author, in this case `system`.
	//
	// This field can be elided, and will marshal its zero value as "system".
	Role string `json:"role" default:"system"`
}

type ChatCompletionSystemMessageParamContentUnion struct {
	OfString              *string                              `json:",omitzero,inline"`
	OfArrayOfContentParts []ChatCompletionContentPartTextParam `json:",omitzero,inline"`
}

func (u ChatCompletionSystemMessageParamContentUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return []byte("null"), nil
	}
}

func SystemMessageContentString(content string) ChatCompletionSystemMessageParamContentUnion {
	return ChatCompletionSystemMessageParamContentUnion{
		OfString: &content,
	}
}

func SystemMessageContentParts(parts []ChatCompletionContentPartTextParam) ChatCompletionSystemMessageParamContentUnion {
	return ChatCompletionSystemMessageParamContentUnion{
		OfArrayOfContentParts: parts,
	}
}

type ChatCompletionUserMessageParam struct {
	// The contents of the user message.
	Content ChatCompletionUserMessageParamContentUnion `json:"content,omitzero" api:"required"`
	// An optional name for the participant. Provides the model information to
	// differentiate between participants of the same role.
	Name *string `json:"name,omitzero"`
	// The role of the messages author, in this case `user`.
	//
	// This field can be elided, and will marshal its zero value as "user".
	Role string `json:"role" default:"user"`
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionUserMessageParamContentUnion struct {
	OfString              *string                               `json:",omitzero,inline"`
	OfArrayOfContentParts []ChatCompletionContentPartUnionParam `json:",omitzero,inline"`
}

func (u ChatCompletionUserMessageParamContentUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return []byte("null"), nil
	}
}

type ChatCompletionAssistantMessageParam struct {
	// The refusal message by the assistant.
	Refusal *string `json:"refusal,omitzero"`
	// An optional name for the participant. Provides the model information to
	// differentiate between participants of the same role.
	Name *string `json:"name,omitzero"`
	// Data about a previous audio response from the model.
	// [Learn more](https://platform.openai.com/docs/guides/audio).
	Audio ChatCompletionAssistantMessageParamAudio `json:"audio,omitzero"`
	// The contents of the assistant message. Required unless `tool_calls` or
	// `function_call` is specified.
	Content ChatCompletionAssistantMessageParamContentUnion `json:"content,omitzero"`
	// Deprecated and replaced by `tool_calls`. The name and arguments of a function
	// that should be called, as generated by the model.
	// The tool calls generated by the model, such as function calls.
	ToolCalls []ChatCompletionMessageToolCallUnionParam `json:"tool_calls,omitzero"`
	// The role of the messages author, in this case `assistant`.
	//
	// This field can be elided, and will marshal its zero value as "assistant".
	Role string `json:"role" default:"assistant"`
}

type ChatCompletionAssistantMessageParamAudio struct {
	// Unique identifier for a previous audio response from the model.
	ID string `json:"id" api:"required"`
}

type ChatCompletionAssistantMessageParamContentUnion struct {
	OfString              *string                                                             `json:",omitzero,inline"`
	OfArrayOfContentParts []ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion `json:",omitzero,inline"`
}

func (u ChatCompletionAssistantMessageParamContentUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfString != nil:
		return json.Marshal(u.OfString)
	case u.OfArrayOfContentParts != nil:
		return json.Marshal(u.OfArrayOfContentParts)
	default:
		return []byte("null"), nil
	}
}

type ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion struct {
	OfText    *ChatCompletionContentPartTextParam    `json:",omitzero,inline"`
	OfRefusal *ChatCompletionContentPartRefusalParam `json:",omitzero,inline"`
}

func (u ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfText != nil:
		return json.Marshal(u.OfText)
	case u.OfRefusal != nil:
		return json.Marshal(u.OfRefusal)
	default:
		return []byte("null"), nil
	}
}

func (u *ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion) GetText() *string {
	if u == nil || u.OfText == nil {
		return nil
	}
	return &u.OfText.Text
}

// Returns a pointer to the underlying variant's property, if present.
func (u *ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion) GetRefusal() *string {
	if u == nil || u.OfRefusal == nil {
		return nil
	}
	return &u.OfRefusal.Refusal
}

// Returns a pointer to the underlying variant's property, if present.
func (u *ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion) GetType() *string {

	if u.OfText != nil {
		return &u.OfText.Type
	}

	if u.OfRefusal != nil {
		return &u.OfRefusal.Type
	}

	return nil
}

type ChatCompletionMessage struct {
	// The contents of the message.
	Content string `json:"content" api:"required"`
	// The refusal message generated by the model.
	Refusal string `json:"refusal" api:"required"`
	// The role of the author of this message.
	Role string `json:"role" default:"assistant"`
	// Annotations for the message, when applicable, as when using the
	// [web search tool](https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat).
	Annotations []ChatCompletionMessageAnnotation `json:"annotations"`
	// If the audio output modality is requested, this object contains data about the
	// audio response from the model.
	// [Learn more](https://platform.openai.com/docs/guides/audio).
	Audio ChatCompletionAudio `json:"audio" api:"nullable"`
	// Deprecated and replaced by `tool_calls`. The name and arguments of a function
	// that should be called, as generated by the model.
	//
	// Deprecated: deprecated
	FunctionCall ChatCompletionMessageFunctionCall `json:"function_call"`
	// The tool calls generated by the model, such as function calls.
	ToolCalls []ChatCompletionMessageToolCallUnion `json:"tool_calls"`
}

func (r ChatCompletionMessage) ToAssistantMessageParam() ChatCompletionAssistantMessageParam {
	var p ChatCompletionAssistantMessageParam

	// It is important to not rely on the JSON metadata property
	// here, it may be unset if the receiver was generated via a
	// [ChatCompletionAccumulator].
	//
	// Explicit null is intentionally elided from the response.
	if r.Content != "" {
		p.Content.OfString = String(r.Content)
	}
	if r.Refusal != "" {
		p.Refusal = String(r.Refusal)
	}

	p.Audio.ID = r.Audio.ID
	p.Role = r.Role
	p.FunctionCall.Arguments = r.FunctionCall.Arguments
	p.FunctionCall.Name = r.FunctionCall.Name

	if len(r.ToolCalls) > 0 {
		for _, v := range r.ToolCalls {
			u := ChatCompletionMessageToolCallUnionParam{}
			switch v.AsAny().(type) {
			case ChatCompletionMessageFunctionToolCall:
				u.OfFunction = &ChatCompletionMessageFunctionToolCallParam{
					ID: v.ID,
					Function: ChatCompletionMessageFunctionToolCallFunctionParam{
						Arguments: v.Function.Arguments,
						Name:      v.Function.Name,
					},
				}
			case ChatCompletionMessageCustomToolCall:
				u.OfCustom = &ChatCompletionMessageCustomToolCallParam{
					ID: v.ID,
					Custom: ChatCompletionMessageCustomToolCallCustomParam{
						Input: v.Custom.Input,
						Name:  v.Custom.Name,
					},
				}
			}

			p.ToolCalls = append(p.ToolCalls, u)
		}
	}
	return p
}

type ChatCompletionMessageAnnotation struct {
	// The type of the URL citation. Always `url_citation`.
	Type constant.URLCitation `json:"type" default:"url_citation"`
	// A URL citation when using web search.
	URLCitation ChatCompletionMessageAnnotationURLCitation `json:"url_citation" api:"required"`
	// JSON contains metadata for fields, check presence with [respjson.Field.Valid].
	JSON struct {
		Type        respjson.Field
		URLCitation respjson.Field
		ExtraFields map[string]respjson.Field
		raw         string
	} `json:"-"`
}

type ChatCompletionMessageAnnotationURLCitation struct {
	// The index of the last character of the URL citation in the message.
	EndIndex int64 `json:"end_index" api:"required"`
	// The index of the first character of the URL citation in the message.
	StartIndex int64 `json:"start_index" api:"required"`
	// The title of the web resource.
	Title string `json:"title" api:"required"`
	// The URL of the web resource.
	URL string `json:"url" api:"required" format:"uri"`
	// JSON contains metadata for fields, check presence with [respjson.Field.Valid].
	JSON struct {
		EndIndex    respjson.Field
		StartIndex  respjson.Field
		Title       respjson.Field
		URL         respjson.Field
		ExtraFields map[string]respjson.Field
		raw         string
	} `json:"-"`
}

// Returns the unmodified JSON received from the API
func (r ChatCompletionMessageAnnotationURLCitation) RawJSON() string { return r.JSON.raw }
func (r *ChatCompletionMessageAnnotationURLCitation) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// Deprecated and replaced by `tool_calls`. The name and arguments of a function
// that should be called, as generated by the model.
//
// Deprecated: deprecated
type ChatCompletionMessageFunctionCall struct {
	// The arguments to call the function with, as generated by the model in JSON
	// format. Note that the model does not always generate valid JSON, and may
	// hallucinate parameters not defined by your function schema. Validate the
	// arguments in your code before calling your function.
	Arguments string `json:"arguments" api:"required"`
	// The name of the function to call.
	Name string `json:"name" api:"required"`
	// JSON contains metadata for fields, check presence with [respjson.Field.Valid].
	JSON struct {
		Arguments   respjson.Field
		Name        respjson.Field
		ExtraFields map[string]respjson.Field
		raw         string
	} `json:"-"`
}

// Returns the unmodified JSON received from the API
func (r ChatCompletionMessageFunctionCall) RawJSON() string { return r.JSON.raw }
func (r *ChatCompletionMessageFunctionCall) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// A call to a custom tool created by the model.
type ChatCompletionMessageCustomToolCall struct {
	// The ID of the tool call.
	ID string `json:"id" api:"required"`
	// The custom tool that the model called.
	Custom ChatCompletionMessageCustomToolCallCustom `json:"custom" api:"required"`
	// The type of the tool. Always `custom`.
	Type constant.Custom `json:"type" default:"custom"`
	// JSON contains metadata for fields, check presence with [respjson.Field.Valid].
	JSON struct {
		ID          respjson.Field
		Custom      respjson.Field
		Type        respjson.Field
		ExtraFields map[string]respjson.Field
		raw         string
	} `json:"-"`
}

// Returns the unmodified JSON received from the API
func (r ChatCompletionMessageCustomToolCall) RawJSON() string { return r.JSON.raw }
func (r *ChatCompletionMessageCustomToolCall) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// ToParam converts this ChatCompletionMessageCustomToolCall to a
// ChatCompletionMessageCustomToolCallParam.
//
// Warning: the fields of the param type will not be present. ToParam should only
// be used at the last possible moment before sending a request. Test for this with
// ChatCompletionMessageCustomToolCallParam.Overrides()
func (r ChatCompletionMessageCustomToolCall) ToParam() ChatCompletionMessageCustomToolCallParam {
	return param.Override[ChatCompletionMessageCustomToolCallParam](json.RawMessage(r.RawJSON()))
}

// The custom tool that the model called.
type ChatCompletionMessageCustomToolCallCustom struct {
	// The input for the custom tool call generated by the model.
	Input string `json:"input" api:"required"`
	// The name of the custom tool to call.
	Name string `json:"name" api:"required"`
	// JSON contains metadata for fields, check presence with [respjson.Field.Valid].
	JSON struct {
		Input       respjson.Field
		Name        respjson.Field
		ExtraFields map[string]respjson.Field
		raw         string
	} `json:"-"`
}

// Returns the unmodified JSON received from the API
func (r ChatCompletionMessageCustomToolCallCustom) RawJSON() string { return r.JSON.raw }
func (r *ChatCompletionMessageCustomToolCallCustom) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// A call to a custom tool created by the model.
//
// The properties ID, Custom, Type are required.
type ChatCompletionMessageCustomToolCallParam struct {
	// The ID of the tool call.
	ID string `json:"id" api:"required"`
	// The custom tool that the model called.
	Custom ChatCompletionMessageCustomToolCallCustomParam `json:"custom,omitzero" api:"required"`
	// The type of the tool. Always `custom`.
	//
	// This field can be elided, and will marshal its zero value as "custom".
	Type constant.Custom `json:"type" default:"custom"`
	paramObj
}

func (r ChatCompletionMessageCustomToolCallParam) MarshalJSON() (data []byte, err error) {
	type shadow ChatCompletionMessageCustomToolCallParam
	return param.MarshalObject(r, (*shadow)(&r))
}
func (r *ChatCompletionMessageCustomToolCallParam) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// The custom tool that the model called.
//
// The properties Input, Name are required.
type ChatCompletionMessageCustomToolCallCustomParam struct {
	// The input for the custom tool call generated by the model.
	Input string `json:"input" api:"required"`
	// The name of the custom tool to call.
	Name string `json:"name" api:"required"`
	paramObj
}

func (r ChatCompletionMessageCustomToolCallCustomParam) MarshalJSON() (data []byte, err error) {
	type shadow ChatCompletionMessageCustomToolCallCustomParam
	return param.MarshalObject(r, (*shadow)(&r))
}
func (r *ChatCompletionMessageCustomToolCallCustomParam) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// A call to a function tool created by the model.
type ChatCompletionMessageFunctionToolCall struct {
	// The ID of the tool call.
	ID string `json:"id" api:"required"`
	// The function that the model called.
	Function ChatCompletionMessageFunctionToolCallFunction `json:"function" api:"required"`
	// The type of the tool. Currently, only `function` is supported.
	Type constant.Function `json:"type" default:"function"`
	// JSON contains metadata for fields, check presence with [respjson.Field.Valid].
	JSON struct {
		ID          respjson.Field
		Function    respjson.Field
		Type        respjson.Field
		ExtraFields map[string]respjson.Field
		raw         string
	} `json:"-"`
}

// Returns the unmodified JSON received from the API
func (r ChatCompletionMessageFunctionToolCall) RawJSON() string { return r.JSON.raw }
func (r *ChatCompletionMessageFunctionToolCall) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// ToParam converts this ChatCompletionMessageFunctionToolCall to a
// ChatCompletionMessageFunctionToolCallParam.
//
// Warning: the fields of the param type will not be present. ToParam should only
// be used at the last possible moment before sending a request. Test for this with
// ChatCompletionMessageFunctionToolCallParam.Overrides()
func (r ChatCompletionMessageFunctionToolCall) ToParam() ChatCompletionMessageFunctionToolCallParam {
	return param.Override[ChatCompletionMessageFunctionToolCallParam](json.RawMessage(r.RawJSON()))
}

// The function that the model called.
type ChatCompletionMessageFunctionToolCallFunction struct {
	// The arguments to call the function with, as generated by the model in JSON
	// format. Note that the model does not always generate valid JSON, and may
	// hallucinate parameters not defined by your function schema. Validate the
	// arguments in your code before calling your function.
	Arguments string `json:"arguments" api:"required"`
	// The name of the function to call.
	Name string `json:"name" api:"required"`
}

// A call to a function tool created by the model.
//
// The properties ID, Function, Type are required.
type ChatCompletionMessageFunctionToolCallParam struct {
	// The ID of the tool call.
	ID string `json:"id" api:"required"`
	// The function that the model called.
	Function ChatCompletionMessageFunctionToolCallFunctionParam `json:"function,omitzero" api:"required"`
	// The type of the tool. Currently, only `function` is supported.
	//
	// This field can be elided, and will marshal its zero value as "function".
	Type string `json:"type" default:"function"`
}

// The function that the model called.
//
// The properties Arguments, Name are required.
type ChatCompletionMessageFunctionToolCallFunctionParam struct {
	// The arguments to call the function with, as generated by the model in JSON
	// format. Note that the model does not always generate valid JSON, and may
	// hallucinate parameters not defined by your function schema. Validate the
	// arguments in your code before calling your function.
	Arguments string `json:"arguments" api:"required"`
	// The name of the function to call.
	Name string `json:"name" api:"required"`
}

func AssistantMessage[T string | []ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion](content T) ChatCompletionMessageParamUnion {
	var assistant ChatCompletionAssistantMessageParam
	switch v := any(content).(type) {
	case string:
		assistant.Content.OfString = param.NewOpt(v)
	case []ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
		assistant.Content.OfArrayOfContentParts = v
	}
	return ChatCompletionMessageParamUnion{OfAssistant: &assistant}
}

func DeveloperMessage[T string | []ChatCompletionContentPartTextParam](content T) ChatCompletionMessageParamUnion {
	var developer ChatCompletionDeveloperMessageParam
	switch v := any(content).(type) {
	case string:
		developer.Content.OfString = param.NewOpt(v)
	case []ChatCompletionContentPartTextParam:
		developer.Content.OfArrayOfContentParts = v
	}
	return ChatCompletionMessageParamUnion{OfDeveloper: &developer}
}

func SystemMessage[T string | []ChatCompletionContentPartTextParam](content T) ChatCompletionMessageParamUnion {
	var system ChatCompletionSystemMessageParam
	switch v := any(content).(type) {
	case string:
		system.Content.OfString = param.NewOpt(v)
	case []ChatCompletionContentPartTextParam:
		system.Content.OfArrayOfContentParts = v
	}
	return ChatCompletionMessageParamUnion{OfSystem: &system}
}

func UserMessage[T string | []ChatCompletionContentPartUnionParam](content T) ChatCompletionMessageParamUnion {
	var user ChatCompletionUserMessageParam
	switch v := any(content).(type) {
	case string:
		user.Content.OfString = param.NewOpt(v)
	case []ChatCompletionContentPartUnionParam:
		user.Content.OfArrayOfContentParts = v
	}
	return ChatCompletionMessageParamUnion{OfUser: &user}
}

func ChatCompletionMessageParamOfAssistant[
	T string | []ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion,
](content T) ChatCompletionMessageParamUnion {
	var assistant ChatCompletionAssistantMessageParam
	switch v := any(content).(type) {
	case string:
		assistant.Content.OfString = param.NewOpt(v)
	case []ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
		assistant.Content.OfArrayOfContentParts = v
	}
	return ChatCompletionMessageParamUnion{OfAssistant: &assistant}
}

func ToolMessage[T string | []ChatCompletionContentPartTextParam](content T, toolCallID string) ChatCompletionMessageParamUnion {
	var tool ChatCompletionToolMessageParam
	switch v := any(content).(type) {
	case string:
		tool.Content.OfString = param.NewOpt(v)
	case []ChatCompletionContentPartTextParam:
		tool.Content.OfArrayOfContentParts = v
	}
	tool.ToolCallID = toolCallID
	return ChatCompletionMessageParamUnion{OfTool: &tool}
}

func ChatCompletionMessageParamOfFunction(content string, name string) ChatCompletionMessageParamUnion {
	var function ChatCompletionFunctionMessageParam
	function.Content = param.NewOpt(content)
	function.Name = name
	return ChatCompletionMessageParamUnion{OfFunction: &function}
}

// Can have the runtime types [*string], [_[]ChatCompletionContentPartTextParam],
// [_[]ChatCompletionContentPartUnionParam],
// [\*[]ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion]
type chatCompletionMessageParamUnionContent struct{ any }

// Use the following switch statement to get the type of the union:
//
//	switch u.AsAny().(type) {
//	case *string:
//	case *[]openai.ChatCompletionContentPartTextParam:
//	case *[]openai.ChatCompletionContentPartUnionParam:
//	case *[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion:
//	default:
//	    fmt.Errorf("not present")
//	}
func (u chatCompletionMessageParamUnionContent) AsAny() any { return u.any }

// ChatCompletionMessageToolCallUnion contains all possible properties and values
// from [ChatCompletionMessageFunctionToolCall],
// [ChatCompletionMessageCustomToolCall].
//
// Use the [ChatCompletionMessageToolCallUnion.AsAny] method to switch on the
// variant.
//
// Use the methods beginning with 'As' to cast the union to one of its variants.
type ChatCompletionMessageToolCallUnion struct {
	ID string `json:"id"`
	// This field is from variant [ChatCompletionMessageFunctionToolCall].
	Function ChatCompletionMessageFunctionToolCallFunction `json:"function"`
	// Any of "function", "custom".
	Type string `json:"type"`
	// This field is from variant [ChatCompletionMessageCustomToolCall].
	Custom ChatCompletionMessageCustomToolCallCustom `json:"custom"`
	JSON   struct {
		ID       respjson.Field
		Function respjson.Field
		Type     respjson.Field
		Custom   respjson.Field
		raw      string
	} `json:"-"`
}

// anyChatCompletionMessageToolCall is implemented by each variant of
// [ChatCompletionMessageToolCallUnion] to add type safety for the return type of
// [ChatCompletionMessageToolCallUnion.AsAny]
type anyChatCompletionMessageToolCall interface {
	implChatCompletionMessageToolCallUnion()
}

func (ChatCompletionMessageFunctionToolCall) implChatCompletionMessageToolCallUnion() {}
func (ChatCompletionMessageCustomToolCall) implChatCompletionMessageToolCallUnion()   {}

// Use the following switch statement to find the correct variant
//
//	switch variant := ChatCompletionMessageToolCallUnion.AsAny().(type) {
//	case openai.ChatCompletionMessageFunctionToolCall:
//	case openai.ChatCompletionMessageCustomToolCall:
//	default:
//	  fmt.Errorf("no variant present")
//	}
func (u ChatCompletionMessageToolCallUnion) AsAny() anyChatCompletionMessageToolCall {
	switch u.Type {
	case "function":
		return u.AsFunction()
	case "custom":
		return u.AsCustom()
	}
	return nil
}

func (u ChatCompletionMessageToolCallUnion) AsFunction() (v ChatCompletionMessageFunctionToolCall) {
	apijson.UnmarshalRoot(json.RawMessage(u.JSON.raw), &v)
	return
}

func (u ChatCompletionMessageToolCallUnion) AsCustom() (v ChatCompletionMessageCustomToolCall) {
	apijson.UnmarshalRoot(json.RawMessage(u.JSON.raw), &v)
	return
}

// Returns the unmodified JSON received from the API
func (u ChatCompletionMessageToolCallUnion) RawJSON() string { return u.JSON.raw }

func (r *ChatCompletionMessageToolCallUnion) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, r)
}

// ToParam converts this ChatCompletionMessageToolCallUnion to a
// ChatCompletionMessageToolCallUnionParam.
//
// Warning: the fields of the param type will not be present. ToParam should only
// be used at the last possible moment before sending a request. Test for this with
// ChatCompletionMessageToolCallUnionParam.Overrides()
func (r ChatCompletionMessageToolCallUnion) ToParam() ChatCompletionMessageToolCallUnionParam {
	return param.Override[ChatCompletionMessageToolCallUnionParam](json.RawMessage(r.RawJSON()))
}

// Only one field can be non-zero.
//
// Use [param.IsOmitted] to confirm if a field is set.
type ChatCompletionMessageToolCallUnionParam struct {
	OfFunction *ChatCompletionMessageFunctionToolCallParam `json:",omitzero,inline"`
	OfCustom   *ChatCompletionMessageCustomToolCallParam   `json:",omitzero,inline"`
}

func (u ChatCompletionMessageToolCallUnionParam) MarshalJSON() ([]byte, error) {
	return param.MarshalUnion(u, u.OfFunction, u.OfCustom)
}
func (u *ChatCompletionMessageToolCallUnionParam) UnmarshalJSON(data []byte) error {
	return apijson.UnmarshalRoot(data, u)
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageToolCallUnionParam) GetFunction() *ChatCompletionMessageFunctionToolCallFunctionParam {
	if vt := u.OfFunction; vt != nil {
		return &vt.Function
	}
	return nil
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageToolCallUnionParam) GetCustom() *ChatCompletionMessageCustomToolCallCustomParam {
	if vt := u.OfCustom; vt != nil {
		return &vt.Custom
	}
	return nil
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageToolCallUnionParam) GetID() *string {
	if vt := u.OfFunction; vt != nil {
		return (*string)(&vt.ID)
	} else if vt := u.OfCustom; vt != nil {
		return (*string)(&vt.ID)
	}
	return nil
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionMessageToolCallUnionParam) GetType() *string {
	if vt := u.OfFunction; vt != nil {
		return (*string)(&vt.Type)
	} else if vt := u.OfCustom; vt != nil {
		return (*string)(&vt.Type)
	}
	return nil
}
