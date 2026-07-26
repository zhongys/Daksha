package openai

// Client is the entry point for the OpenAI-compatible wire protocol.
// This package is a self-contained SDK for the protocol (the equivalent of
// the official openai client): it knows nothing about the neutral ai layer.
// Vendor differences come in through RequestOptions.
type Client struct {
	Options    []RequestOption
	Chat       ChatService
	Embeddings EmbeddingService
}

func NewClient(opts ...RequestOption) Client {
	return Client{
		Options:    append([]RequestOption(nil), opts...),
		Chat:       NewChatService(opts...),
		Embeddings: NewEmbeddingService(opts...),
	}
}
