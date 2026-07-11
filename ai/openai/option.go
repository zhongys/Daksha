package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// WithBaseURL returns a RequestOption that sets the BaseURL for the client.
//
// For security reasons, ensure that the base URL is trusted.
func WithBaseURL(base string) RequestOption {
	u, err := url.Parse(base)
	if err == nil && u.Path != "" && !strings.HasSuffix(u.Path, "/") {
		u.Path += "/"
	}

	return RequestOptionFunc(func(r *RequestConfig) error {
		if err != nil {
			return fmt.Errorf("requestoption: WithBaseURL failed to parse url %s", err)
		}

		r.BaseURL = u
		return nil
	})
}

// HTTPClient is primarily used to describe an [*http.Client], but also
// supports custom implementations.
//
// For bespoke implementations, prefer using an [*http.Client] with a
// custom transport. See [http.RoundTripper] for further information.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

// WithHTTPClient returns a RequestOption that changes the underlying http client used to make this
// request, which by default is [http.DefaultClient].
func WithHTTPClient(client HTTPClient) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		if client == nil {
			return fmt.Errorf("requestoption: custom http client cannot be nil")
		}

		if c, ok := client.(*http.Client); ok {
			// Prefer the native client if possible.
			r.HTTPClient = c
			r.CustomHTTPDoer = nil
		} else {
			r.CustomHTTPDoer = client
		}

		return nil
	})
}

// WithHeader returns a RequestOption that sets the header value to the associated key. It overwrites
// any value if there was one already present.
func WithHeader(key, value string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.SetHeader(key, value)
		return nil
	})
}

// WithHeaderAdd returns a RequestOption that adds the header value to the associated key. It appends
// onto any existing values.
func WithHeaderAdd(key, value string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.AddHeader(key, value)
		return nil
	})
}

// WithHeaderDel returns a RequestOption that deletes the header value(s) associated with the given key.
func WithHeaderDel(key string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.DelHeader(key)
		return nil
	})
}

// WithQuery returns a RequestOption that sets the query value to the associated key. It overwrites
// any value if there was one already present.
func WithQuery(key, value string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		query := r.Request.URL.Query()
		query.Set(key, value)
		r.Request.URL.RawQuery = query.Encode()
		return nil
	})
}

// WithQueryAdd returns a RequestOption that adds the query value to the associated key. It appends
// onto any existing values.
func WithQueryAdd(key, value string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		query := r.Request.URL.Query()
		query.Add(key, value)
		r.Request.URL.RawQuery = query.Encode()
		return nil
	})
}

// WithQueryDel returns a RequestOption that deletes the query value(s) associated with the key.
func WithQueryDel(key string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		query := r.Request.URL.Query()
		query.Del(key)
		r.Request.URL.RawQuery = query.Encode()
		return nil
	})
}

// WithJSONSet returns a RequestOption that sets the top-level key of the
// serialized JSON body to the given value. The body must be a JSON object.
func WithJSONSet(key string, value any) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) (err error) {
		m := map[string]json.RawMessage{}

		if r.Body != nil {
			buffer, ok := r.Body.(*bytes.Buffer)
			if !ok {
				return fmt.Errorf("cannot use WithJSONSet on a body that is not serialized as *bytes.Buffer")
			}
			if buffer.Len() > 0 {
				if err := json.Unmarshal(buffer.Bytes(), &m); err != nil {
					return err
				}
			}
		}

		raw, err := jsonMarshalNoEscape(value)
		if err != nil {
			return err
		}
		m[key] = raw

		b, err := jsonMarshalNoEscape(m)
		if err != nil {
			return err
		}
		r.Body = bytes.NewBuffer(b)
		return nil
	})
}

// WithJSONDel returns a RequestOption that deletes the top-level key from the
// serialized JSON body. The body must be a JSON object.
func WithJSONDel(key string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) (err error) {
		buffer, ok := r.Body.(*bytes.Buffer)
		if !ok {
			return fmt.Errorf("cannot use WithJSONDel on a body that is not serialized as *bytes.Buffer")
		}

		m := map[string]json.RawMessage{}
		if buffer.Len() > 0 {
			if err := json.Unmarshal(buffer.Bytes(), &m); err != nil {
				return err
			}
		}
		delete(m, key)

		b, err := jsonMarshalNoEscape(m)
		if err != nil {
			return err
		}
		r.Body = bytes.NewBuffer(b)
		return nil
	})
}

func jsonMarshalNoEscape(v any) ([]byte, error) {
	buf := new(bytes.Buffer)
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// WithResponseBodyInto returns a RequestOption that overwrites the deserialization target with
// the given destination. If provided, we don't deserialize into the default struct.
func WithResponseBodyInto(dst any) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.ResponseBodyInto = dst
		return nil
	})
}

// WithResponseInto returns a RequestOption that copies the [*http.Response] into the given address.
func WithResponseInto(dst **http.Response) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.ResponseInto = dst
		return nil
	})
}

// WithRequestBody returns a RequestOption that provides a custom serialized body with the given
// content type.
//
// body accepts an io.Reader or raw []bytes.
func WithRequestBody(contentType string, body any) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		if reader, ok := body.(io.Reader); ok {
			r.Body = reader
			return r.Apply(WithHeader("Content-Type", contentType))
		}

		if b, ok := body.([]byte); ok {
			r.Body = bytes.NewBuffer(b)
			return r.Apply(WithHeader("Content-Type", contentType))
		}

		return fmt.Errorf("body must be a byte slice or implement io.Reader")
	})
}

// WithRequestTimeout returns a RequestOption that sets the timeout for
// each request attempt. This should be smaller than the timeout defined in
// the context, which spans all retries.
func WithRequestTimeout(dur time.Duration) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.RequestTimeout = dur
		return nil
	})
}

// WithAPIKey returns a RequestOption that sets the client's API key, sent as
// a Bearer token.
func WithAPIKey(value string) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.SetAPIKey(value)
		return nil
	})
}
