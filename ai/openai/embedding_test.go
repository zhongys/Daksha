package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEmbeddingNewParamsMarshal(t *testing.T) {
	tests := []struct {
		name    string
		params  EmbeddingNewParams
		want    []string
		notWant []string
	}{
		{
			name: "minimal params omit unset fields",
			params: EmbeddingNewParams{
				Input: []string{"hello"},
				Model: "text-embedding-v4",
			},
			want: []string{
				`"input":["hello"]`,
				`"model":"text-embedding-v4"`,
			},
			notWant: []string{"encoding_format", "dimensions"},
		},
		{
			name: "optional fields serialize when set",
			params: EmbeddingNewParams{
				Input:          []string{"a", "b"},
				Model:          "m",
				EncodingFormat: "float",
				Dimensions:     1024,
			},
			want: []string{
				`"input":["a","b"]`,
				`"encoding_format":"float"`,
				`"dimensions":1024`,
			},
		},
		{
			name: "extra fields merge and override",
			params: EmbeddingNewParams{
				Input:       []string{"a"},
				Model:       "m",
				ExtraFields: map[string]any{"model": "override", "text_type": "query"},
			},
			want:    []string{`"model":"override"`, `"text_type":"query"`},
			notWant: []string{`"model":"m"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.params)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			s := string(b)
			for _, w := range tt.want {
				if !strings.Contains(s, w) {
					t.Errorf("want substring %q in %s", w, s)
				}
			}
			for _, nw := range tt.notWant {
				if strings.Contains(s, nw) {
					t.Errorf("unwanted substring %q in %s", nw, s)
				}
			}
		})
	}
}
