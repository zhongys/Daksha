package openai

type ChatService struct {
	Options []RequestOption
	// Given a list of messages comprising a conversation, the model will
	// return a response.
	Completions ChatCompletionService
}

func NewChatService(opts ...RequestOption) (r ChatService) {
	r = ChatService{}
	r.Options = append([]RequestOption(nil), opts...)
	r.Completions = NewChatCompletionService(opts...)
	return
}
