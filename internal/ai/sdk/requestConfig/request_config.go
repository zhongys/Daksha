package requestconfig

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/tidwall/gjson"

	"github.com/zhongys/Daksha.git/internal/ai/sdk/apierror"
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

func getNormalizedOS() string {
	switch runtime.GOOS {
	case "ios":
		return "iOS"
	case "android":
		return "Android"
	case "darwin":
		return "MacOS"
	case "windows":
		return "Windows"
	case "freebsd":
		return "FreeBSD"
	case "openbsd":
		return "OpenBSD"
	case "linux":
		return "Linux"
	default:
		return fmt.Sprintf("Other:%s", runtime.GOOS)
	}
}

func getNormalizedArchitecture() string {
	switch runtime.GOARCH {
	case "386":
		return "x32"
	case "amd64":
		return "x64"
	case "arm":
		return "arm"
	case "arm64":
		return "arm64"
	default:
		return fmt.Sprintf("other:%s", runtime.GOARCH)
	}
}

func getPlatformProperties() map[string]string {
	return map[string]string{
		"X-Stainless-Lang":            "go",
		"X-Stainless-Package-Version": "1.0",
		"X-Stainless-OS":              getNormalizedOS(),
		"X-Stainless-Arch":            getNormalizedArchitecture(),
		"X-Stainless-Runtime":         "go",
		"X-Stainless-Runtime-Version": runtime.Version(),
	}
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

	if body, ok := body.(json.Marshaler); ok {
		content, err := body.MarshalJSON()
		if err != nil {
			return nil, err
		}
		reader = bytes.NewBuffer(content)
		hasSerializationFunc = true
	}
	// if body, ok := body.(apiform.Marshaler); ok {
	// 	var (
	// 		content []byte
	// 		err     error
	// 	)
	// 	content, contentType, err = body.MarshalMultipart()
	// 	if err != nil {
	// 		return nil, err
	// 	}
	// 	reader = bytes.NewBuffer(content)
	// 	hasSerializationFunc = true
	// }

	if body, ok := body.([]byte); ok {
		reader = bytes.NewBuffer(body)
		hasSerializationFunc = true
	}
	if body, ok := body.(io.Reader); ok {
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
	req.Header.Set("X-Stainless-Retry-Count", "0")
	req.Header.Set("X-Stainless-Timeout", "0")
	for k, v := range getDefaultHeaders() {
		req.Header.Add(k, v)
	}

	for k, v := range getPlatformProperties() {
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

	// This must run after `cfg.Apply(...)` above in case the request timeout gets modified. We also only
	// apply our own logic for it if it's still "0" from above. If it's not, then it was deleted or modified
	// by the user and we should respect that.
	if req.Header.Get("X-Stainless-Timeout") == "0" {
		if cfg.RequestTimeout == time.Duration(0) {
			req.Header.Del("X-Stainless-Timeout")
		} else {
			req.Header.Set("X-Stainless-Timeout", strconv.Itoa(int(cfg.RequestTimeout.Seconds())))
		}
	}

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
	authPreference     authCredentialPreference
	// Configure which security scheme(s) should be enabled for this request
	Security Security
	// If ResponseBodyInto not nil, then we will attempt to deserialize into
	// ResponseBodyInto. If Destination is a []byte, then it will return the body as
	// is.
	ResponseBodyInto any
	// ResponseInto copies the \*http.Response of the corresponding request into the
	// given address
	ResponseInto **http.Response
	Body         io.Reader
}

func isBeforeContextDeadline(t time.Time, ctx context.Context) bool {
	d, ok := ctx.Deadline()
	if !ok {
		return true
	}
	return t.Before(d)
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

	var res *http.Response
	ctx := cfg.Request.Context()
	req := cfg.Request.Clone(ctx)

	res, err = handler(req)
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}

	// Save *http.Response if it is requested to, even if there was an error making the request. This is
	// useful in cases where you might want to debug by inspecting the response. Note that if err != nil,
	// the response should be generally be empty, but there are edge cases.
	if cfg.ResponseInto != nil {
		*cfg.ResponseInto = res
	}
	if responseBodyInto, ok := cfg.ResponseBodyInto.(**http.Response); ok {
		*responseBodyInto = res
	}

	// If there was a connection error in the final request or any other transport error,
	// return that early without trying to coerce into an APIError.
	if err != nil {
		return err
	}

	if res.StatusCode >= 400 {
		contents, err := io.ReadAll(res.Body)
		_ = res.Body.Close()
		if err != nil {
			return err
		}

		// If there is an APIError, re-populate the response body so that debugging
		// utilities can conveniently dump the response without issue.
		res.Body = io.NopCloser(bytes.NewBuffer(contents))

		// Load the contents into the error format if it is provided.
		aerr := apierror.Error{Request: cfg.Request, Response: res, StatusCode: res.StatusCode}
		unwrapped := gjson.GetBytes(contents, "error").Raw
		err = json.Unmarshal([]byte(unwrapped), &aerr)
		if err != nil {
			return err
		}
		return &aerr
	}

	_, intoCustomResponseBody := cfg.ResponseBodyInto.(**http.Response)
	if cfg.ResponseBodyInto == nil || intoCustomResponseBody {
		// We aren't reading the response body in this scope, but whoever is will need the
		// cancel func from the context to observe request timeouts.
		// Put the cancel function in the response body so it can be handled elsewhere.
		return nil
	}

	contents, err := io.ReadAll(res.Body)
	_ = res.Body.Close()
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	// If we are not json, return plaintext
	contentType := res.Header.Get("content-type")
	mediaType, _, _ := mime.ParseMediaType(contentType)
	isJSON := strings.Contains(mediaType, "application/json") || strings.HasSuffix(mediaType, "+json")
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
		err = json.NewDecoder(bytes.NewReader(contents)).Decode(cfg.ResponseBodyInto)
		if err != nil {
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
		req.Body, err = req.GetBody()
	}
	if err != nil {
		return nil
	}
	new := &RequestConfig{
		RequestTimeout:     cfg.RequestTimeout,
		Context:            ctx,
		Request:            req,
		BaseURL:            cfg.BaseURL,
		HTTPClient:         cfg.HTTPClient,
		APIKey:             cfg.APIKey,
		authHeaderOverride: cfg.authHeaderOverride,
		authPreference:     cfg.authPreference,
	}

	return new
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

type authCredentialPreference int

const (
	authCredentialPreferenceBearer authCredentialPreference = iota
)

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
func WithBearerAuthPreference() RequestOption {
	return RequestOptionFunc(func(r *RequestConfig) error {
		r.authPreference = authCredentialPreferenceBearer
		return nil
	})
}
func ApplySecurity(r RequestConfig) {
	if r.authHeaderOverride {
		return
	}

	if r.authPreference == authCredentialPreferenceBearer && r.Security.BearerAuth && r.APIKey != "" {
		r.Request.Header.Set("authorization", fmt.Sprintf("Bearer %s", r.APIKey))
		return
	}

	if r.Security.BearerAuth && r.APIKey != "" {
		r.Request.Header.Set("authorization", fmt.Sprintf("Bearer %s", r.APIKey))
	}
}
