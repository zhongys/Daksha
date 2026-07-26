package openai

// Ptr returns a pointer to v, for setting optional scalar params.
func Ptr[T any](v T) *T { return &v }

func String(v string) *string { return &v }

func Int(v int64) *int64 { return &v }

func Float(v float64) *float64 { return &v }

func Bool(v bool) *bool { return &v }
