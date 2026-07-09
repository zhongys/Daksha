package sdk

import "github.com/zhongys/Daksha.git/internal/ai/sdk/option"

type ChatService struct {
	Options []option.RequestOption
	// Given a list of messages comprising a conversation, the model will return a
	// response.
	Completions ChatCompletionService
}

func NewChatService(opts ...option.RequestOption) (r ChatService) {
	r = ChatService{}
	r.Options = opts
	r.Completions = NewChatCompletionService(opts...)
	return
}
