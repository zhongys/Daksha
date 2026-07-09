package openai

// Client is the entry point for the OpenAI-compatible wire protocol.
// Vendor differences (base URL, API key, extra request fields) come in
// through RequestOptions built from an ai.Provider.
type Client struct {
	Options []RequestOption
	Chat    ChatService
}

func NewClient(opts ...RequestOption) Client {
	return Client{
		Options: opts,
		Chat:    NewChatService(opts...),
	}
}
