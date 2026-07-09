package shared

type FunctionDefinitionParam struct {
	Name        string         `json:"name"`
	Description *string        `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}
