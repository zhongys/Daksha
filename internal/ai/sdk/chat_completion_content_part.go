package sdk

import (
	"encoding/json"

	constant "github.com/zhongys/Daksha.git/internal/ai/sdk/shared"
)

type ChatCompletionContentPartUnionParam struct {
	OfText       *ChatCompletionContentPartTextParam
	OfImageURL   *ChatCompletionContentPartImageParam
	OfInputAudio *ChatCompletionContentPartInputAudioParam
	OfFile       *ChatCompletionContentPartFileParam
}

func (u ChatCompletionContentPartUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfText != nil:
		return json.Marshal(u.OfText)
	case u.OfImageURL != nil:
		return json.Marshal(u.OfImageURL)
	case u.OfInputAudio != nil:
		return json.Marshal(u.OfInputAudio)
	case u.OfFile != nil:
		return json.Marshal(u.OfFile)
	default:
		return []byte("null"), nil
	}
}

func (u ChatCompletionContentPartUnionParam) GetText() *string {
	if u.OfText == nil {
		return nil
	}
	return &u.OfText.Text
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionContentPartUnionParam) GetImageURL() *ChatCompletionContentPartImageImageURLParam {
	if u.OfImageURL == nil {
		return nil
	}
	return &u.OfImageURL.ImageURL
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionContentPartUnionParam) GetInputAudio() *ChatCompletionContentPartInputAudioInputAudioParam {
	if u.OfInputAudio == nil {
		return nil
	}
	return &u.OfInputAudio.InputAudio
}

// Returns a pointer to the underlying variant's property, if present.
func (u ChatCompletionContentPartUnionParam) GetFile() *ChatCompletionContentPartFileFileParam {
	if u.OfFile == nil {
		return nil
	}
	return &u.OfFile.File

}

func TextContentPart(text string) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfText: &ChatCompletionContentPartTextParam{
			Type: constant.ContentTypeText,
			Text: text},
	}
}

func ImageContentPart(imageURL ChatCompletionContentPartImageImageURLParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfImageURL: &ChatCompletionContentPartImageParam{
			Type:     constant.ContentTypeImageURL,
			ImageURL: imageURL},
	}
}

func InputAudioContentPart(inputAudio ChatCompletionContentPartInputAudioInputAudioParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfInputAudio: &ChatCompletionContentPartInputAudioParam{
			Type:       constant.ContentTypeInputAudio,
			InputAudio: inputAudio},
	}
}

func FileContentPart(file ChatCompletionContentPartFileFileParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfFile: &ChatCompletionContentPartFileParam{
			Type: constant.ContentTypeInputAudio,
			File: file},
	}
}

type ChatCompletionContentPartTextParam struct {
	// The text content.
	Text string `json:"text" api:"required"`
	// The type of the content part.
	//
	// This field can be elided, and will marshal its zero value as "text".
	Type string `json:"type" default:"text"`
}

type ChatCompletionContentPartFileParam struct {
	File ChatCompletionContentPartFileFileParam `json:"file,omitzero" api:"required"`
	// The type of the content part. Always `file`.
	//
	// This field can be elided, and will marshal its zero value as "file".
	Type string `json:"type" default:"file"`
}

type ChatCompletionContentPartFileFileParam struct {
	// The base64 encoded file data, used when passing the file to the model as a
	// string.
	FileData *string `json:"file_data,omitzero"`
	// The ID of an uploaded file to use as input.
	FileID *string `json:"file_id,omitzero"`
	// The name of the file, used when passing the file to the model as a string.
	Filename *string `json:"filename,omitzero"`
}

type ChatCompletionContentPartImageParam struct {
	Type     string                                      `json:"type"`
	ImageURL ChatCompletionContentPartImageImageURLParam `json:"image_url"`
}

type ChatCompletionContentPartImageImageURLParam struct {
	// Either a URL of the image or the base64 encoded image data.
	URL string `json:"url"`
	// Specifies the detail level of the image. Learn more in the
	// [Vision guide](https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding).
	//
	// Any of "auto", "low", "high".
	Detail string `json:"detail,omitempty"`
}

type ChatCompletionContentPartInputAudioParam struct {
	InputAudio ChatCompletionContentPartInputAudioInputAudioParam `json:"input_audio,omitzero" api:"required"`
	// The type of the content part. Always `input_audio`.
	//
	// This field can be elided, and will marshal its zero value as "input_audio".
	Type string `json:"type" default:"input_audio"`
}

type ChatCompletionContentPartInputAudioInputAudioParam struct {
	// Base64 encoded audio data.
	Data string `json:"data" api:"required"`
	// The format of the encoded audio data. Currently supports "wav" and "mp3".
	//
	// Any of "wav", "mp3".
	Format string `json:"format,omitzero" api:"required"`
}

type ChatCompletionContentPartRefusalParam struct {
	// The refusal message generated by the model.
	Refusal string `json:"refusal" api:"required"`
	// The type of the content part.
	//
	// This field can be elided, and will marshal its zero value as "refusal".
	Type string `json:"type" default:"refusal"`
}
