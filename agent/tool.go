package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/zhongys/Daksha/ai"
)

// ExecutionMode selects how the tool calls of one assistant message run.
// The zero value is sequential: parallel safety is a property a tool author
// must claim explicitly, never a default.
type ExecutionMode string

const (
	ExecSequential ExecutionMode = "sequential"
	ExecParallel   ExecutionMode = "parallel"
)

// ToolUpdate is a streamed progress snapshot from a running tool, surfaced
// as a tool_execution_update event.
type ToolUpdate struct {
	Content []ai.ToolResultContent
	Details any
}

// ToolOutput is a tool's successful result.
type ToolOutput struct {
	Content []ai.ToolResultContent
	// Details is app-specific structured data stored on the tool result
	// message; the LLM never sees it.
	Details any
	// Terminate hints that the loop should skip the automatic follow-up LLM
	// call. It only takes effect when every finalized result in the batch
	// sets it.
	Terminate bool
}

// Tool is one callable exposed to the model.
//
// Execute returns an error to signal tool failure — never encode failures
// into Content. The loop reports errors to the model as a toolResult with
// IsError set, so it can react. onUpdate may be nil.
type Tool interface {
	// Definition is what the LLM sees: name, description and the JSON
	// Schema of the arguments. Label (UI display name) rides alongside.
	Definition() ToolDefinition
	Execute(ctx context.Context, toolCallID string, args map[string]any,
		onUpdate func(ToolUpdate)) (*ToolOutput, error)
}

// ToolDefinition extends the wire-level definition with UI metadata.
type ToolDefinition struct {
	ai.ToolDefinition
	// Label is a human-readable display name; never sent to the model.
	Label string
}

// sequentialTool is the optional marker read by the loop: if any tool call
// in a batch targets a tool wrapped with Sequential, the whole batch runs
// sequentially regardless of the agent-level mode.
type sequentialTool struct{ Tool }

func (sequentialTool) sequentialOnly() bool { return true }

// Sequential marks a tool as unsafe for concurrent execution.
func Sequential(t Tool) Tool { return sequentialTool{t} }

func isSequentialOnly(t Tool) bool {
	_, ok := t.(interface{ sequentialOnly() bool })
	return ok
}

// NewTool adapts a typed handler into a Tool. Arguments are decoded into P
// via a JSON round-trip; a decode failure is returned as an error, which the
// loop reports to the model as an IsError toolResult so it can fix its
// arguments and retry — it never aborts the run.
func NewTool[P any](def ToolDefinition,
	fn func(ctx context.Context, toolCallID string, params P, onUpdate func(ToolUpdate)) (*ToolOutput, error),
) Tool {
	return &typedTool[P]{def: def, fn: fn}
}

type typedTool[P any] struct {
	def ToolDefinition
	fn  func(ctx context.Context, toolCallID string, params P, onUpdate func(ToolUpdate)) (*ToolOutput, error)
}

func (t *typedTool[P]) Definition() ToolDefinition { return t.def }

func (t *typedTool[P]) Execute(ctx context.Context, toolCallID string, args map[string]any,
	onUpdate func(ToolUpdate)) (*ToolOutput, error) {

	raw, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("agent: encode arguments for tool %q: %w", t.def.Name, err)
	}
	var params P
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, fmt.Errorf("agent: invalid arguments for tool %q: %w", t.def.Name, err)
	}
	return t.fn(ctx, toolCallID, params, onUpdate)
}
