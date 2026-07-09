package sdk

type Client struct {
	Options []option.RequestOption
	// Given a prompt, the model will return one or more predicted completions, and can
	// also return the probabilities of alternative tokens at each position.
	Completions CompletionService
	Chat        ChatService
	// Get a vector representation of a given input that can be easily consumed by
	// machine learning models and algorithms.
	Embeddings EmbeddingService
	// Files are used to upload documents that can be used with features like
	// Assistants and Fine-tuning.
	Files FileService
	// Given a prompt and/or an input image, the model will generate a new image.
	Images ImageService
	Audio  AudioService
	// Given text and/or image inputs, classifies if those inputs are potentially
	// harmful.
	Moderations ModerationService
	// List and describe the various models available in the API.
	Models       ModelService
	FineTuning   FineTuningService
	Graders      GraderService
	VectorStores VectorStoreService
	Webhooks     webhooks.WebhookService
	Beta         BetaService
	// Create large batches of API requests to run asynchronously.
	Batches BatchService
	// Use Uploads to upload large files in multiple parts.
	Uploads   UploadService
	Admin     AdminService
	Responses responses.ResponseService
	Realtime  realtime.RealtimeService
	// Manage conversations and conversation items.
	Conversations conversations.ConversationService
	Containers    ContainerService
	Skills        SkillService
	Videos        VideoService
}
