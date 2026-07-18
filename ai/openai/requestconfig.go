package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func getDefaultHeaders() map[string]string {
	return map[string]string{
		"User-Agent": fmt.Sprintf("daksha/Go %s", "1.0"),
	}
}

func encodePathParam(value string) string {
	switch value {
	case ".":
		return "%2E"
	case "..":
		return "%2E%2E"
	}
	return url.PathEscape(value)
}

type RequestOption interface {
	Apply(*RequestConfig) error
}

type RequestOptionFunc func(*RequestConfig) error
type PreRequestOptionFunc func(*RequestConfig) error

func (s RequestOptionFunc) Apply(r *RequestConfig) error    { return s(r) }
func (s PreRequestOptionFunc) Apply(r *RequestConfig) error { return s(r) }

func NewRequestConfig(ctx context.Context, method string, u string, body any, dst any, opts ...RequestOption) (*RequestConfig, error) {
	var reader io.Reader

	contentType := "application/json"
	hasSerializationFunc := false

	switch body := body.(type) {
	case json.Marshaler:
		content, err := body.MarshalJSON()
		if err != nil {
			return nil, err
		}
		// A Marshaler is allowed to return storage it still owns. Freeze the
		// serialized request now so later caller mutations cannot change what is
		// sent when Execute eventually attaches the body to the request.
		reader = bytes.NewBuffer(bytes.Clone(content))
		hasSerializationFunc = true
	case []byte:
		// Treat raw request bytes as an input snapshot, not as shared mutable
		// storage retained until Execute.
		reader = bytes.NewBuffer(bytes.Clone(body))
		hasSerializationFunc = true
	case io.Reader:
		// Readers are inherently stateful and cannot be copied generically. The
		// caller retains ownership until Execute has consumed the request.
		reader = body
		hasSerializationFunc = true
	}

	// Fallback to json serialization if none of the serialization functions that we expect
	// to see is present.
	if body != nil && !hasSerializationFunc {
		buf := new(bytes.Buffer)
		enc := json.NewEncoder(buf)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(body); err != nil {
			return nil, err
		}
		reader = buf
	}

	req, err := http.NewRequestWithContext(ctx, method, u, nil)
	if err != nil {
		return nil, err
	}
	if reader != nil {
		req.Header.Set("Content-Type", contentType)
	}

	req.Header.Set("Accept", "application/json")
	for k, v := range getDefaultHeaders() {
		req.Header.Add(k, v)
	}

	cfg := RequestConfig{
		Context:    ctx,
		Request:    req,
		HTTPClient: http.DefaultClient,
		Body:       reader,
	}
	cfg.ResponseBodyInto = dst
	cfg.Security = Security{
		BearerAuth: true,
	}
	err = cfg.Apply(opts...)
	if err != nil {
		return nil, err
	}

	// This must run after `cfg.Apply(...)` above so we know which specific security scheme to add
	ApplySecurity(cfg)

	return &cfg, nil
}

type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

type RequestConfig struct {
	RequestTimeout time.Duration
	Context        context.Context
	Request        *http.Request
	BaseURL        *url.URL
	// DefaultBaseURL will be used if BaseURL is not explicitly overridden using
	// WithBaseURL.
	DefaultBaseURL     *url.URL
	CustomHTTPDoer     HTTPDoer
	HTTPClient         *http.Client
	APIKey             string
	authHeaderOverride bool
	// Configure which security scheme(s) should be enabled for this request
	Security Security
	// If ResponseBodyInto not nil, then we will attempt to deserialize into
	// ResponseBodyInto. If Destination is a []byte, then it will return the body as
	// is.
	ResponseBodyInto any
	// ResponseInto copies the *http.Response of the corresponding request into the
	// given address
	ResponseInto **http.Response
	Body         io.Reader
}

// cancelOnCloseBody keeps a request-scoped timeout alive while a caller owns
// a streaming response body, then releases its context resources with the
// body. A raw *http.Response must outlive Execute, so Execute cannot defer the
// timeout's cancel function in that case.
type cancelOnCloseBody struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b *cancelOnCloseBody) Close() error {
	// Cancel first so a custom response body whose Close waits on the request
	// context cannot deadlock waiting for the cancellation it is meant to
	// trigger.
	b.cancel()
	return b.ReadCloser.Close()
}

func (cfg *RequestConfig) Execute() (err error) {
	if cfg.BaseURL == nil {
		if cfg.DefaultBaseURL != nil {
			cfg.BaseURL = cfg.DefaultBaseURL
		} else {
			return fmt.Errorf("requestconfig: base url is not set")
		}
	}

	cfg.Request.URL, err = cfg.BaseURL.Parse(strings.TrimLeft(cfg.Request.URL.String(), "/"))
	if err != nil {
		return err
	}

	if cfg.Body != nil && cfg.Request.Body == nil {
		switch body := cfg.Body.(type) {
		case *bytes.Buffer:
			b := body.Bytes()
			cfg.Request.ContentLength = int64(body.Len())
			cfg.Request.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(b)), nil }
			cfg.Request.Body, _ = cfg.Request.GetBody()
		case *bytes.Reader:
			cfg.Request.ContentLength = int64(body.Len())
			cfg.Request.GetBody = func() (io.ReadCloser, error) {
				_, err := body.Seek(0, 0)
				return io.NopCloser(body), err
			}
			cfg.Request.Body, _ = cfg.Request.GetBody()
		default:
			if rc, ok := body.(io.ReadCloser); ok {
				cfg.Request.Body = rc
			} else {
				cfg.Request.Body = io.NopCloser(body)
			}
		}
	}

	handler := cfg.HTTPClient.Do
	if cfg.CustomHTTPDoer != nil {
		handler = cfg.CustomHTTPDoer.Do
	}

	ctx := cfg.Request.Context()
	var cancel context.CancelFunc
	if cfg.RequestTimeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, cfg.RequestTimeout)
		defer func() {
			if cancel != nil {
				cancel()
			}
		}()
	}
	req := cfg.Request.Clone(ctx)

	var res *http.Response
	res, err = handler(req)

	// Save *http.Response if it is requested to, even if there was an error making the request. This is
	// useful in cases where you might want to debug by inspecting the response. Note that if err != nil,
	// the response should be generally be empty, but there are edge cases.
	responseBodyInto, intoCustomResponseBody := cfg.ResponseBodyInto.(**http.Response)
	if cfg.ResponseInto != nil {
		*cfg.ResponseInto = res
	}
	if intoCustomResponseBody {
		*responseBodyInto = res
	}
	closeUnownedResponse := func() {
		if res == nil || res.Body == nil {
			return
		}
		if cfg.ResponseInto == nil && !intoCustomResponseBody {
			_ = res.Body.Close()
		}
	}

	if ctx != nil && ctx.Err() != nil {
		closeUnownedResponse()
		return ctx.Err()
	}

	// If there was a connection error in the final request or any other transport error,
	// return that early without trying to coerce into an APIError.
	if err != nil {
		closeUnownedResponse()
		return err
	}
	if res == nil {
		return errors.New("requestconfig: http client returned a nil response without an error")
	}
	if res.Body == nil {
		return errors.New("requestconfig: http client returned a response with a nil body")
	}

	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		contents, err := io.ReadAll(res.Body)
		_ = res.Body.Close()
		if err != nil {
			return err
		}

		// Re-populate the response body so that debugging utilities can
		// conveniently dump the response without issue.
		res.Body = io.NopCloser(bytes.NewBuffer(contents))

		return ParseAPIError(res.StatusCode, contents)
	}

	responseBodyHasOwner := intoCustomResponseBody || (cfg.ResponseBodyInto == nil && cfg.ResponseInto != nil)
	if responseBodyHasOwner && cancel != nil && res.Body != nil {
		res.Body = &cancelOnCloseBody{ReadCloser: res.Body, cancel: cancel}
		cancel = nil // ownership moves to res.Body.Close
	}
	if cfg.ResponseBodyInto == nil {
		if cfg.ResponseInto == nil && res.Body != nil {
			// No decoded destination and no raw-response owner: release the
			// connection here instead of returning an unreachable body.
			_ = res.Body.Close()
		}
		return nil
	}
	if intoCustomResponseBody {
		// We aren't reading the response body in this scope, but whoever is will need the
		// cancel func from the context to observe request timeouts.
		return nil
	}

	contents, err := io.ReadAll(res.Body)
	_ = res.Body.Close()
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	// If we are not json, return plaintext
	contentType := res.Header.Get("content-type")
	mediaType, _, mediaTypeErr := mime.ParseMediaType(contentType)
	isJSON := mediaTypeErr == nil && (strings.EqualFold(mediaType, "application/json") ||
		strings.HasSuffix(strings.ToLower(mediaType), "+json"))
	if !isJSON {
		switch dst := cfg.ResponseBodyInto.(type) {
		case *string:
			*dst = string(contents)
		case **string:
			tmp := string(contents)
			*dst = &tmp
		case *[]byte:
			*dst = contents
		default:
			return fmt.Errorf("expected destination type of 'string' or '[]byte' for responses with content-type '%s' that is not 'application/json'", contentType)
		}
		return nil
	}

	switch dst := cfg.ResponseBodyInto.(type) {
	// If the response happens to be a byte array, deserialize the body as-is.
	case *[]byte:
		*dst = contents
	default:
		trimmed := bytes.TrimSpace(contents)
		if len(trimmed) == 0 {
			return errors.New("error parsing response json: empty response body")
		}
		if bytes.Equal(trimmed, []byte("null")) {
			return errors.New("error parsing response json: top-level value is null")
		}
		if apiErr := errorFromEventData(contents); apiErr != nil {
			apiErr.StatusCode = res.StatusCode
			return apiErr
		}
		// Unmarshal, unlike a single Decoder.Decode call, requires the complete
		// body to contain exactly one JSON value and rejects a second value or
		// any other trailing non-whitespace data.
		if err = json.Unmarshal(contents, cfg.ResponseBodyInto); err != nil {
			return fmt.Errorf("error parsing response json: %w", err)
		}
	}

	return nil
}

func ExecuteNewRequest(ctx context.Context, method string, u string, body any, dst any, opts ...RequestOption) error {
	cfg, err := NewRequestConfig(ctx, method, u, body, dst, opts...)
	if err != nil {
		return err
	}
	return cfg.Execute()
}

func (cfg *RequestConfig) Clone(ctx context.Context) *RequestConfig {
	if cfg == nil {
		return nil
	}
	req := cfg.Request.Clone(ctx)
	var err error
	if req.Body != nil {
		if req.GetBody == nil {
			return nil
		}
		req.Body, err = req.GetBody()
	}
	if err != nil {
		return nil
	}
	return &RequestConfig{
		RequestTimeout:     cfg.RequestTimeout,
		Context:            ctx,
		Request:            req,
		BaseURL:            cfg.BaseURL,
		HTTPClient:         cfg.HTTPClient,
		APIKey:             cfg.APIKey,
		authHeaderOverride: cfg.authHeaderOverride,
	}
}

func (cfg *RequestConfig) SetHeader(key, value string) {
	cfg.Request.Header.Set(key, value)
	if strings.EqualFold(key, "Authorization") {
		cfg.authHeaderOverride = true
	}
}

func (cfg *RequestConfig) AddHeader(key, value string) {
	cfg.Request.Header.Add(key, value)
	if strings.EqualFold(key, "Authorization") {
		cfg.authHeaderOverride = true
	}
}

func (cfg *RequestConfig) DelHeader(key string) {
	cfg.Request.Header.Del(key)
	if strings.EqualFold(key, "Authorization") {
		cfg.authHeaderOverride = true
	}
}

func (cfg *RequestConfig) SetAPIKey(value string) {
	cfg.APIKey = value
	cfg.authHeaderOverride = false
}

func (cfg *RequestConfig) Apply(opts ...RequestOption) error {
	for _, opt := range opts {
		err := opt.Apply(cfg)
		if err != nil {
			return err
		}
	}
	return nil
}

func PreRequestOptions(opts ...RequestOption) (RequestConfig, error) {
	cfg := RequestConfig{}
	for _, opt := range opts {
		if opt, ok := opt.(PreRequestOptionFunc); ok {
			err := opt.Apply(&cfg)
			if err != nil {
				return cfg, err
			}
		}
	}
	return cfg, nil
}

func WithDefaultBaseURL(baseURL string) RequestOption {
	u, err := url.Parse(baseURL)
	return RequestOptionFunc(func(r *RequestConfig) error {
		if err != nil {
			return err
		}
		r.DefaultBaseURL = u
		return nil
	})
}

type Security struct {
	BearerAuth bool
}

func WithSecurity(security Security) RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.Security = security
		return nil
	})
}

func WithBearerAuthSecurity() RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.Security = Security{
			BearerAuth: true,
		}
		return nil
	})
}

func ApplySecurity(r RequestConfig) {
	if r.authHeaderOverride {
		return
	}

	if r.Security.BearerAuth && r.APIKey != "" {
		r.Request.Header.Set("authorization", fmt.Sprintf("Bearer %s", r.APIKey))
	}
}
