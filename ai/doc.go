// Package ai defines provider-neutral model, message, streaming, embedding,
// usage and cost primitives. Client owns an application-populated registry of
// providers and models and routes calls to protocol adapters.
//
// Provider and Model values are trusted host-application configuration. An
// application that derives them from untrusted input must enforce its own
// authorization, URL validation, credential isolation and request-field
// policy before registering them.
package ai
