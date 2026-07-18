package openai

import (
	"encoding/json"
	"fmt"
)

// ChatCompletionContentPartUnionParam holds exactly one user content part
// variant.
type ChatCompletionContentPartUnionParam struct {
	OfText       *ChatCompletionContentPartTextParam       `json:"-"`
	OfImageURL   *ChatCompletionContentPartImageParam      `json:"-"`
	OfInputAudio *ChatCompletionContentPartInputAudioParam `json:"-"`
	OfVideoURL   *ChatCompletionContentPartVideoParam      `json:"-"`
	OfFile       *ChatCompletionContentPartFileParam       `json:"-"`
}

func (u ChatCompletionContentPartUnionParam) MarshalJSON() ([]byte, error) {
	switch {
	case u.OfText != nil:
		return json.Marshal(u.OfText)
	case u.OfImageURL != nil:
		return json.Marshal(u.OfImageURL)
	case u.OfInputAudio != nil:
		return json.Marshal(u.OfInputAudio)
	case u.OfVideoURL != nil:
		return json.Marshal(u.OfVideoURL)
	case u.OfFile != nil:
		return json.Marshal(u.OfFile)
	default:
		return nil, fmt.Errorf("openai: empty union ChatCompletionContentPartUnionParam")
	}
}

func TextContentPart(text string) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfText: &ChatCompletionContentPartTextParam{
			Type: "text",
			Text: text,
		},
	}
}

func ImageContentPart(imageURL ChatCompletionContentPartImageImageURLParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfImageURL: &ChatCompletionContentPartImageParam{
			Type:     "image_url",
			ImageURL: imageURL,
		},
	}
}

func InputAudioContentPart(audio ChatCompletionContentPartInputAudioInputAudioParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfInputAudio: &ChatCompletionContentPartInputAudioParam{
			Type:       "input_audio",
			InputAudio: audio,
		},
	}
}

func VideoContentPart(videoURL ChatCompletionContentPartVideoVideoURLParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfVideoURL: &ChatCompletionContentPartVideoParam{
			Type:     "video_url",
			VideoURL: videoURL,
		},
	}
}

func FileContentPart(file ChatCompletionContentPartFileFileParam) ChatCompletionContentPartUnionParam {
	return ChatCompletionContentPartUnionParam{
		OfFile: &ChatCompletionContentPartFileParam{
			Type: "file",
			File: file,
		},
	}
}

type ChatCompletionContentPartTextParam struct {
	// The text content.
	Text string `json:"text"`
	// The type of the content part. Defaults to "text" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionContentPartTextParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "text"
	}
	type shadow ChatCompletionContentPartTextParam
	return json.Marshal(shadow(r))
}

type ChatCompletionContentPartImageParam struct {
	ImageURL ChatCompletionContentPartImageImageURLParam `json:"image_url"`
	// The type of the content part. Defaults to "image_url" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionContentPartImageParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "image_url"
	}
	type shadow ChatCompletionContentPartImageParam
	return json.Marshal(shadow(r))
}

type ChatCompletionContentPartImageImageURLParam struct {
	// Either a URL of the image or the base64 encoded image data.
	URL string `json:"url"`
	// Specifies the detail level of the image.
	//
	// Any of "auto", "low", "high".
	Detail string `json:"detail,omitempty"`
}

type ChatCompletionContentPartInputAudioParam struct {
	InputAudio ChatCompletionContentPartInputAudioInputAudioParam `json:"input_audio"`
	// The type of the content part. Defaults to "input_audio" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionContentPartInputAudioParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "input_audio"
	}
	type shadow ChatCompletionContentPartInputAudioParam
	return json.Marshal(shadow(r))
}

type ChatCompletionContentPartInputAudioInputAudioParam struct {
	// Base64 encoded audio data on the OpenAI standard; DashScope also
	// accepts a URL here.
	Data string `json:"data"`
	// The format of the encoded audio data, e.g. "wav" or "mp3". Optional on
	// endpoints that sniff the format from a URL.
	Format string `json:"format,omitempty"`
}

// ChatCompletionContentPartVideoParam is the "video_url" content part used by
// OpenAI-compatible endpoints with video understanding (DashScope, GLM); the
// OpenAI standard itself has no video part.
type ChatCompletionContentPartVideoParam struct {
	VideoURL ChatCompletionContentPartVideoVideoURLParam `json:"video_url"`
	// The type of the content part. Defaults to "video_url" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionContentPartVideoParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "video_url"
	}
	type shadow ChatCompletionContentPartVideoParam
	return json.Marshal(shadow(r))
}

type ChatCompletionContentPartVideoVideoURLParam struct {
	// Either a URL of the video or a base64 data URI, per endpoint support.
	URL string `json:"url"`
}

type ChatCompletionContentPartFileParam struct {
	File ChatCompletionContentPartFileFileParam `json:"file"`
	// The type of the content part. Defaults to "file" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionContentPartFileParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "file"
	}
	type shadow ChatCompletionContentPartFileParam
	return json.Marshal(shadow(r))
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

type ChatCompletionContentPartRefusalParam struct {
	// The refusal message generated by the model.
	Refusal string `json:"refusal"`
	// The type of the content part. Defaults to "refusal" when left empty.
	Type string `json:"type"`
}

func (r ChatCompletionContentPartRefusalParam) MarshalJSON() ([]byte, error) {
	if r.Type == "" {
		r.Type = "refusal"
	}
	type shadow ChatCompletionContentPartRefusalParam
	return json.Marshal(shadow(r))
}
