package openaicompletions

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/zhongys/Daksha/ai"
)

func TestStreamerPreservesLargeIntegerToolArguments(t *testing.T) {
	server := sseServer(t, []string{
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"id\":9007199254740993,"}}]}}]}`,
		`{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"nested\":{\"value\":9007199254740995}}"}}]},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	})
	defer server.Close()

	stream := NewStreamer().Stream(context.Background(),
		ai.Provider{BaseURL: server.URL + "/", APIKey: "k"},
		ai.Model{ID: "m"},
		ai.Prompt{Messages: []ai.Message{userText("hi")}},
		ai.StreamOptions{},
	)

	events, msg, err := collectStream(t, stream)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	var firstDelta *ai.ToolCallDeltaEvent
	var end *ai.ToolCallEndEvent
	var done *ai.DoneEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallDeltaEvent:
			if firstDelta == nil {
				copy := event
				firstDelta = &copy
			}
		case ai.ToolCallEndEvent:
			copy := event
			end = &copy
		case ai.DoneEvent:
			copy := event
			done = &copy
		}
	}
	if firstDelta == nil || end == nil || done == nil {
		t.Fatalf("events = %v, want delta, end, and done", eventTypes(events))
	}

	partialCall := firstDelta.Partial.Content[firstDelta.ContentIndex].(*ai.ToolCallContent)
	assertJSONNumber(t, partialCall.Arguments["id"], "9007199254740993")

	want := map[string]any{
		"id": json.Number("9007199254740993"),
		"nested": map[string]any{
			"value": json.Number("9007199254740995"),
		},
	}
	finalCall := msg.Content[0].(*ai.ToolCallContent)
	if !reflect.DeepEqual(finalCall.Arguments, want) {
		t.Fatalf("final arguments = %#v, want %#v", finalCall.Arguments, want)
	}
	if !reflect.DeepEqual(end.ToolCall.Arguments, want) {
		t.Fatalf("tool-call end arguments = %#v, want %#v", end.ToolCall.Arguments, want)
	}
	doneCall := done.Message.Content[0].(*ai.ToolCallContent)
	if !reflect.DeepEqual(doneCall.Arguments, want) {
		t.Fatalf("done arguments = %#v, want %#v", doneCall.Arguments, want)
	}

	// The typed tool adapter performs this same JSON round-trip. Exact
	// json.Number values must therefore reach integer parameters unchanged.
	wire, err := json.Marshal(finalCall.Arguments)
	if err != nil {
		t.Fatalf("marshal final arguments: %v", err)
	}
	if got, want := string(wire), `{"id":9007199254740993,"nested":{"value":9007199254740995}}`; got != want {
		t.Fatalf("wire arguments = %s, want %s", got, want)
	}
	var params struct {
		ID     int64 `json:"id"`
		Nested struct {
			Value int64 `json:"value"`
		} `json:"nested"`
	}
	if err := json.Unmarshal(wire, &params); err != nil {
		t.Fatalf("decode typed tool parameters: %v", err)
	}
	if params.ID != 9_007_199_254_740_993 || params.Nested.Value != 9_007_199_254_740_995 {
		t.Fatalf("typed parameters = %+v", params)
	}
}

func assertJSONNumber(t *testing.T, value any, want string) {
	t.Helper()
	number, ok := value.(json.Number)
	if !ok || number.String() != want {
		t.Fatalf("number = %T(%v), want json.Number(%s)", value, value, want)
	}
}
