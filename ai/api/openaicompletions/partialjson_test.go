package openaicompletions

import (
	"reflect"
	"testing"
)

func TestParseStreamingJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  map[string]any
	}{
		{"empty", "", map[string]any{}},
		{"whitespace", "  \n", map[string]any{}},
		{"complete object", `{"a": 1, "b": "x"}`, map[string]any{"a": float64(1), "b": "x"}},
		{"open brace only", `{`, map[string]any{}},
		{"truncated mid-key", `{"pa`, map[string]any{}},
		{"truncated before colon", `{"path"`, map[string]any{}},
		{"truncated before value", `{"path": `, map[string]any{}},
		{"truncated mid-string value", `{"path": "/tmp/fi`, map[string]any{"path": "/tmp/fi"}},
		{"truncated mid-escape", `{"path": "a\`, map[string]any{"path": "a"}},
		{"truncated mid-unicode-escape", `{"path": "a\u00`, map[string]any{"path": "a"}},
		{"truncated number", `{"n": 12.`, map[string]any{"n": float64(12)}},
		{"truncated exponent", `{"n": 1e`, map[string]any{"n": float64(1)}},
		{"literal prefix true", `{"ok": tr`, map[string]any{"ok": true}},
		{"literal prefix false", `{"ok": fals`, map[string]any{"ok": false}},
		{"literal prefix null", `{"v": nu`, map[string]any{"v": nil}},
		{"truncated array", `{"xs": [1, 2, `, map[string]any{"xs": []any{float64(1), float64(2)}}},
		{"truncated nested object", `{"a": {"b": "c`, map[string]any{"a": map[string]any{"b": "c"}}},
		{"second pair truncated keeps first", `{"a": 1, "b`, map[string]any{"a": float64(1)}},
		{
			"raw newline inside string repaired",
			"{\"text\": \"line1\nline2\"}",
			map[string]any{"text": "line1\nline2"},
		},
		{
			"invalid escape repaired",
			`{"path": "C:\Users\x"}`,
			map[string]any{"path": `C:\Users\x`},
		},
		{"not an object", `[1, 2]`, map[string]any{}},
		{"garbage", `not json at all`, map[string]any{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseStreamingJSON(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseStreamingJSON(%q) = %#v, want %#v", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseStreamingJSONProgressive(t *testing.T) {
	// Simulates argument fragments arriving over a stream: every prefix must
	// parse without error and converge to the full object.
	full := `{"query": "golang \"json\" parsing", "limit": 10, "recursive": true}`
	for i := 1; i <= len(full); i++ {
		got := parseStreamingJSON(full[:i])
		if got == nil {
			t.Fatalf("nil result at prefix %d", i)
		}
	}
	final := parseStreamingJSON(full)
	want := map[string]any{"query": `golang "json" parsing`, "limit": float64(10), "recursive": true}
	if !reflect.DeepEqual(final, want) {
		t.Errorf("final = %#v, want %#v", final, want)
	}
}

func TestRepairJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"passthrough", `{"a":"b"}`, `{"a":"b"}`},
		{"raw newline", "\"a\nb\"", `"a\nb"`},
		{"raw tab", "\"a\tb\"", `"a\tb"`},
		{"invalid escape doubled", `"C:\x"`, `"C:\\x"`},
		{"valid escape kept", `"a\nb"`, `"a\nb"`},
		{"valid unicode kept", `"é"`, `"é"`},
		{"invalid unicode doubled", `"\uzz"`, `"\\uzz"`},
		{"trailing backslash", `"a\`, `"a\\`},
		{"outside string untouched", "{\n\t}", "{\n\t}"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairJSON(tt.input); got != tt.want {
				t.Errorf("repairJSON(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
