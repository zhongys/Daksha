// Package agent implements a stateful, concurrency-safe agent loop on top of
// the provider-neutral ai package. One Agent represents one conversation
// session and permits at most one active run at a time.
//
// The package supports streamed lifecycle events, steering, follow-ups,
// cancellation, typed tools, permission hooks and terminal structured output.
package agent
