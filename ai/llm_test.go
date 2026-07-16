package ai

import (
	"strings"
	"testing"
)

func TestOutputFormatValidate(t *testing.T) {
	cyclic := map[string]any{}
	cyclic["self"] = cyclic

	validSchema := &JSONSchema{
		Name:   "answer",
		Schema: map[string]any{"type": "object"},
		Strict: true,
	}
	tests := []struct {
		name    string
		format  OutputFormat
		wantErr string
	}{
		{name: "provider default"},
		{name: "text", format: OutputFormat{Type: OutputFormatText}},
		{name: "json object", format: OutputFormat{Type: OutputFormatJSONObject}},
		{name: "json schema", format: OutputFormat{Type: OutputFormatJSONSchema, JSONSchema: validSchema}},
		{
			name:    "schema with text",
			format:  OutputFormat{Type: OutputFormatText, JSONSchema: validSchema},
			wantErr: "only valid",
		},
		{
			name:    "missing schema",
			format:  OutputFormat{Type: OutputFormatJSONSchema},
			wantErr: "requires a json schema",
		},
		{
			name: "missing schema name",
			format: OutputFormat{Type: OutputFormatJSONSchema, JSONSchema: &JSONSchema{
				Schema: map[string]any{"type": "object"},
			}},
			wantErr: "requires a schema name",
		},
		{
			name: "empty schema",
			format: OutputFormat{Type: OutputFormatJSONSchema, JSONSchema: &JSONSchema{
				Name: "answer",
			}},
			wantErr: "requires a non-empty schema",
		},
		{
			name: "schema is not JSON encodable",
			format: OutputFormat{Type: OutputFormatJSONSchema, JSONSchema: &JSONSchema{
				Name: "answer", Schema: cyclic,
			}},
			wantErr: "invalid schema",
		},
		{
			name:    "unknown type",
			format:  OutputFormat{Type: "yaml"},
			wantErr: "unknown output format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.format.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Validate error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestOutputFormatIsJSON(t *testing.T) {
	for _, tt := range []struct {
		format OutputFormat
		want   bool
	}{
		{format: OutputFormat{}, want: false},
		{format: OutputFormat{Type: OutputFormatText}, want: false},
		{format: OutputFormat{Type: OutputFormatJSONObject}, want: true},
		{format: OutputFormat{Type: OutputFormatJSONSchema}, want: true},
	} {
		if got := tt.format.IsJSON(); got != tt.want {
			t.Errorf("OutputFormat(%q).IsJSON() = %v, want %v", tt.format.Type, got, tt.want)
		}
	}
}
