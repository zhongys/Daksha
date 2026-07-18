package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/zhongys/Daksha/ai"
)

func TestNewToolPreservesLargeIntegersInInterfaceParameters(t *testing.T) {
	const large = "9007199254740993"
	args := map[string]any{
		"id": json.Number(large),
		"nested": map[string]any{
			"values": []any{json.Number("9007199254740995")},
		},
	}

	t.Run("map parameters", func(t *testing.T) {
		tool := mustNewTool(t,
			ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "lookup"}},
			func(_ context.Context, _ string, params map[string]any, _ func(ToolUpdate)) (*ToolOutput, error) {
				assertToolJSONNumber(t, params["id"], large)
				nested := params["nested"].(map[string]any)
				assertToolJSONNumber(t, nested["values"].([]any)[0], "9007199254740995")
				return &ToolOutput{}, nil
			},
		)
		if _, err := tool.Execute(context.Background(), "call_1", args, nil); err != nil {
			t.Fatalf("Execute: %v", err)
		}
	})

	t.Run("interface fields", func(t *testing.T) {
		type params struct {
			ID     any            `json:"id"`
			Nested map[string]any `json:"nested"`
		}
		tool := mustNewTool(t,
			ToolDefinition{ToolDefinition: ai.ToolDefinition{Name: "lookup"}},
			func(_ context.Context, _ string, params params, _ func(ToolUpdate)) (*ToolOutput, error) {
				assertToolJSONNumber(t, params.ID, large)
				assertToolJSONNumber(t, params.Nested["values"].([]any)[0], "9007199254740995")
				return &ToolOutput{}, nil
			},
		)
		if _, err := tool.Execute(context.Background(), "call_1", args, nil); err != nil {
			t.Fatalf("Execute: %v", err)
		}
	})
}

func assertToolJSONNumber(t *testing.T, value any, want string) {
	t.Helper()
	number, ok := value.(json.Number)
	if !ok || number.String() != want {
		t.Fatalf("number = %T(%v), want json.Number(%s)", value, value, want)
	}
}
