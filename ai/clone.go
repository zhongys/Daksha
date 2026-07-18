package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
)

// CloneMessage returns a detached copy of a message. Protocol content is
// always fully detached. Tool-result Details are structurally cloned across
// pointers, maps, slices, arrays and exported struct fields; opaque state
// hidden behind unexported fields, functions or channels remains
// application-owned and must be treated as immutable after handoff.
func CloneMessage(message Message) Message {
	switch message := message.(type) {
	case *UserMessage:
		return cloneUserMessage(message)
	case *AssistantMessage:
		return CloneAssistantMessage(message)
	case interface{ cloneToolResultMessage() Message }:
		return message.cloneToolResultMessage()
	default:
		// Message is sealed to this package. Keep a conservative fallback for a
		// future immutable variant.
		return message
	}
}

// CloneMessages returns a new slice of detached message snapshots. It is
// nil-preserving and follows CloneMessage's metadata caveat.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	cloned := make([]Message, len(messages))
	for i, message := range messages {
		cloned[i] = CloneMessage(message)
	}
	return cloned
}

func cloneUserMessage(message *UserMessage) *UserMessage {
	if message == nil {
		return nil
	}
	cloned := *message
	if message.Content != nil {
		cloned.Content = make([]UserContent, len(message.Content))
		for i, content := range message.Content {
			cloned.Content[i] = cloneUserContent(content)
		}
	}
	return &cloned
}

func cloneUserContent(content UserContent) UserContent {
	switch content := content.(type) {
	case *TextContent:
		if content == nil {
			return content
		}
		cloned := *content
		return &cloned
	case *ImageContent:
		if content == nil {
			return content
		}
		cloned := *content
		return &cloned
	case *AudioContent:
		if content == nil {
			return content
		}
		cloned := *content
		return &cloned
	case *VideoContent:
		if content == nil {
			return content
		}
		cloned := *content
		return &cloned
	default:
		return content
	}
}

// The generic receiver preserves the concrete ToolResultMessage[T] type while
// still allowing CloneMessage to handle every instantiation through a small
// private interface.
func (message *ToolResultMessage[T]) cloneToolResultMessage() Message {
	if message == nil {
		return message
	}
	cloned := *message
	if message.Content != nil {
		cloned.Content = make([]ToolResultContent, len(message.Content))
		for i, content := range message.Content {
			cloned.Content[i] = cloneToolResultContent(content)
		}
	}
	if message.Details != nil {
		details := cloneApplicationValue(*message.Details)
		cloned.Details = &details
	}
	return &cloned
}

func (message *ToolResultMessage[T]) cloneToolResultForPrompt() Message {
	if message == nil {
		return message
	}
	cloned := *message
	if message.Content != nil {
		cloned.Content = make([]ToolResultContent, len(message.Content))
		for i, content := range message.Content {
			cloned.Content[i] = cloneToolResultContent(content)
		}
	}
	// Details is application metadata and deliberately never enters a model
	// prompt. Dropping it also avoids trying to copy opaque application state at
	// the wire-snapshot boundary.
	cloned.Details = nil
	return &cloned
}

func clonePromptMessage(message Message) Message {
	switch message := message.(type) {
	case *UserMessage:
		return cloneUserMessage(message)
	case *AssistantMessage:
		return CloneAssistantMessage(message)
	case interface{ cloneToolResultForPrompt() Message }:
		return message.cloneToolResultForPrompt()
	default:
		return message
	}
}

func cloneToolResultContent(content ToolResultContent) ToolResultContent {
	switch content := content.(type) {
	case *TextContent:
		if content == nil {
			return content
		}
		cloned := *content
		return &cloned
	case *ImageContent:
		if content == nil {
			return content
		}
		cloned := *content
		return &cloned
	default:
		return content
	}
}

// CloneToolDefinition detaches the parameter schema without changing its
// concrete JSON-compatible Go types. Request snapshots subsequently normalize
// the schema through encoding/json before it crosses an asynchronous boundary.
func CloneToolDefinition(definition ToolDefinition) ToolDefinition {
	definition.Parameters = cloneAssistantJSONMap(definition.Parameters)
	return definition
}

// SnapshotToolDefinition freezes a definition by its wire JSON
// representation. Unlike CloneToolDefinition, this invokes custom marshalers
// now and therefore cannot retain mutable state hidden in their private fields.
func SnapshotToolDefinition(definition ToolDefinition) (ToolDefinition, error) {
	parameters, err := snapshotJSONMap(definition.Parameters)
	if err != nil {
		return ToolDefinition{}, fmt.Errorf("ai: snapshot tool %q parameters: %w", definition.Name, err)
	}
	definition.Parameters = parameters
	return definition, nil
}

// SnapshotPrompt freezes all caller-owned values that can become visible on
// the request wire. JSON objects are normalized by their encoded form so
// custom marshalers and concrete map/slice types cannot retain hidden aliases;
// UseNumber preserves integer precision.
func SnapshotPrompt(prompt Prompt) (Prompt, error) {
	snapshot := prompt
	if prompt.Messages != nil {
		snapshot.Messages = make([]Message, len(prompt.Messages))
		for i, message := range prompt.Messages {
			snapshot.Messages[i] = clonePromptMessage(message)
		}
	}
	for messageIndex, message := range snapshot.Messages {
		assistant, ok := message.(*AssistantMessage)
		if !ok || assistant == nil {
			continue
		}
		for contentIndex, content := range assistant.Content {
			call, ok := content.(*ToolCallContent)
			if !ok || call == nil {
				continue
			}
			arguments, err := snapshotJSONMap(call.Arguments)
			if err != nil {
				return Prompt{}, fmt.Errorf(
					"ai: snapshot message %d tool call %d arguments: %w",
					messageIndex, contentIndex, err,
				)
			}
			call.Arguments = arguments
		}
	}

	if prompt.Tools != nil {
		snapshot.Tools = make([]ToolDefinition, len(prompt.Tools))
		for i, definition := range prompt.Tools {
			frozen, err := SnapshotToolDefinition(definition)
			if err != nil {
				return Prompt{}, err
			}
			snapshot.Tools[i] = frozen
		}
	}
	return snapshot, nil
}

// SnapshotStreamOptions freezes every reference-backed option before a
// request goroutine starts. It does not perform semantic option validation;
// protocol adapters remain responsible for that part of their contract.
func SnapshotStreamOptions(options StreamOptions) (StreamOptions, error) {
	snapshot := options
	if options.Temperature != nil {
		value := *options.Temperature
		snapshot.Temperature = &value
	}
	if options.TopP != nil {
		value := *options.TopP
		snapshot.TopP = &value
	}
	if options.MaxTokens != nil {
		value := *options.MaxTokens
		snapshot.MaxTokens = &value
	}
	if options.StopSequences != nil {
		snapshot.StopSequences = append([]string(nil), options.StopSequences...)
	}

	if options.OutputFormat.JSONSchema != nil {
		schema := *options.OutputFormat.JSONSchema
		clonedSchema, err := snapshotJSONMap(schema.Schema)
		if err != nil {
			return StreamOptions{}, fmt.Errorf("ai: snapshot output schema: %w", err)
		}
		schema.Schema = clonedSchema
		snapshot.OutputFormat.JSONSchema = &schema
	}

	extra, err := snapshotJSONMap(options.Extra)
	if err != nil {
		return StreamOptions{}, fmt.Errorf("ai: snapshot stream options extra: %w", err)
	}
	snapshot.Extra = extra
	return snapshot, nil
}

func snapshotJSONMap(value map[string]any) (map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var snapshot map[string]any
	if err := decoder.Decode(&snapshot); err != nil {
		return nil, err
	}
	return snapshot, nil
}

// CloneAssistantMessage returns a detached copy of message.
//
// Assistant messages contain interface slices, pointer-backed content blocks,
// raw JSON byte slices, and JSON object trees. A plain struct copy would leave
// those mutable values shared with the caller. This helper is nil-safe and
// recursively copies every mutable value defined by AssistantMessage's content
// contract.
func CloneAssistantMessage(message *AssistantMessage) *AssistantMessage {
	if message == nil {
		return nil
	}

	cloned := *message
	if message.Content != nil {
		cloned.Content = make([]AssistantContent, len(message.Content))
		for i, content := range message.Content {
			switch content := content.(type) {
			case *TextContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				cloned.Content[i] = &copy
			case *JSONContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				copy.Value = cloneAssistantRawMessage(content.Value)
				cloned.Content[i] = &copy
			case *ThinkingContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				cloned.Content[i] = &copy
			case *ToolCallContent:
				if content == nil {
					cloned.Content[i] = content
					continue
				}
				copy := *content
				copy.Arguments = cloneAssistantJSONMap(content.Arguments)
				cloned.Content[i] = &copy
			default:
				// AssistantContent is sealed to this package. Keeping the value is a
				// conservative fallback if a new immutable variant is added before
				// its dedicated copy case.
				cloned.Content[i] = content
			}
		}
	}
	cloned.Diagnostics.Details = cloneAssistantJSONMap(message.Diagnostics.Details)
	return &cloned
}

func cloneAssistantRawMessage(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

func cloneAssistantJSONMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	return cloneJSONCompatibleReflect(value).(map[string]any)
}

// cloneJSONCompatibleReflect handles caller-constructed JSON trees that use
// concrete map/slice types such as []string while preserving shared/cyclic
// topology. Cyclic input is outside the JSON contract and is rejected later
// by SnapshotPrompt/SnapshotStreamOptions.
func cloneJSONCompatibleReflect(value any) any {
	if value == nil {
		return nil
	}
	return cloneJSONReflectValue(reflect.ValueOf(value), map[jsonCloneVisit]reflect.Value{}).Interface()
}

func cloneApplicationValue[T any](value T) T {
	source := reflect.ValueOf(&value).Elem()
	cloned := cloneJSONReflectValue(source, map[jsonCloneVisit]reflect.Value{})
	result, ok := cloned.Interface().(T)
	if !ok {
		// A nil interface has no dynamic type to assert. Its original zero value
		// is already detached.
		return value
	}
	return result
}

type jsonCloneVisit struct {
	kind    reflect.Kind
	typeOf  reflect.Type
	pointer uintptr
	len     int
	cap     int
}

func cloneJSONReflectValue(value reflect.Value, visited map[jsonCloneVisit]reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}
	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := cloneJSONReflectValue(value.Elem(), visited)
		out := reflect.New(value.Type()).Elem()
		out.Set(cloned)
		return out
	case reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := jsonCloneVisit{kind: value.Kind(), typeOf: value.Type(), pointer: value.Pointer()}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		out := reflect.New(value.Type().Elem())
		visited[visit] = out
		out.Elem().Set(cloneJSONReflectValue(value.Elem(), visited))
		return out
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := jsonCloneVisit{kind: value.Kind(), typeOf: value.Type(), pointer: value.Pointer()}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		out := reflect.MakeMapWithSize(value.Type(), value.Len())
		visited[visit] = out
		iterator := value.MapRange()
		for iterator.Next() {
			out.SetMapIndex(iterator.Key(), cloneJSONReflectValue(iterator.Value(), visited))
		}
		return out
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := jsonCloneVisit{
			kind: value.Kind(), typeOf: value.Type(), pointer: value.Pointer(), len: value.Len(), cap: value.Cap(),
		}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		out := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		visited[visit] = out
		for i := 0; i < value.Len(); i++ {
			out.Index(i).Set(cloneJSONReflectValue(value.Index(i), visited))
		}
		return out
	case reflect.Array:
		out := reflect.New(value.Type()).Elem()
		for i := 0; i < value.Len(); i++ {
			out.Index(i).Set(cloneJSONReflectValue(value.Index(i), visited))
		}
		return out
	case reflect.Struct:
		// Start with a value copy so unexported immutable implementation fields
		// remain intact, then recursively detach exported JSON-visible fields.
		out := reflect.New(value.Type()).Elem()
		out.Set(value)
		for i := 0; i < value.NumField(); i++ {
			if out.Field(i).CanSet() && value.Type().Field(i).IsExported() {
				out.Field(i).Set(cloneJSONReflectValue(value.Field(i), visited))
			}
		}
		return out
	default:
		return value
	}
}
