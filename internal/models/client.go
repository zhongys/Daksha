package models

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	maxConcurrentStreams = 500
	streamQueueSize      = 16
	maxSSEEventSize      = 1024 * 1024
	maxRequestBodySize   = 1024 * 1024

	streamTotalTimeout = 5 * time.Minute
	streamWriteTimeout = 15 * time.Second
	streamAcquireWait  = 200 * time.Millisecond
)

type StreamLimiter struct {
	slots chan struct{}
}

func NewLimiter(maxConcurrent int) *StreamLimiter {
	if maxConcurrent <= 0 {
		return nil
	}

	return &StreamLimiter{
		slots: make(chan struct{}, maxConcurrent),
	}
}

func (l *StreamLimiter) Acquire(
	ctx context.Context,
) error {
	if l == nil {
		return nil
	}

	select {
	case l.slots <- struct{}{}:
		return nil

	case <-ctx.Done():
		return ctx.Err()
	}
}

func (l *StreamLimiter) Release() {
	if l != nil {
		<-l.slots
	}
}

type Config struct {
	APIKey               string
	BaseURL              string
	HTTPClient           *http.Client
	Headers              http.Header
	MaxConcurrentStreams int
}

type ChatCompletionCreateParams struct {
	Model             string                       `json:"model"`
	Messages          []ChatMessage                `json:"messages"`
	Stream            bool                         `json:"stream,omitempty"`
	StreamOptions     *ChatCompletionStreamOptions `json:"stream_options,omitempty"`
	Tools             []ChatTool                   `json:"tools,omitempty"`
	ToolChoice        any                          `json:"tool_choice,omitempty"`
	Temperature       *float64                     `json:"temperature,omitempty"`
	MaxTokens         *int                         `json:"max_tokens,omitempty"`
	MaxCompletion     *int                         `json:"max_completion_tokens,omitempty"`
	ReasoningEffort   string                       `json:"reasoning_effort,omitempty"`
	Store             *bool                        `json:"store,omitempty"`
	PromptCacheKey    string                       `json:"prompt_cache_key,omitempty"`
	PromptCacheRetain string                       `json:"prompt_cache_retention,omitempty"`
}

type StreamEvent struct {
	Raw []byte
}

type ModelClient struct {
	apiKey     string
	baseURL    *url.URL
	httpClient *http.Client
	headers    http.Header

	streamLimiter *StreamLimiter

	Chat *ChatService
}

func New(
	cfg Config,
) (*ModelClient, error) {
	base := cfg.BaseURL
	if base == "" {
		return nil, fmt.Errorf("base URL is not found")
	}

	parsed, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil {
		return nil, fmt.Errorf("parse base URL: %w", err)
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = defaultHTTPClient()
	}

	c := &ModelClient{
		apiKey:     cfg.APIKey,
		baseURL:    parsed,
		httpClient: httpClient,
		headers:    cloneHeader(cfg.Headers),
	}
	c.Chat = &ChatService{Completions: &CompletionsService{client: c}}
	return c, nil
}

type ChatService struct {
	Completions *CompletionsService
}

type CompletionsService struct {
	client *ModelClient
}

type RequestConfig struct {
	Timeout time.Duration
	Headers http.Header
}

type RequestOption func(*RequestConfig)

func WithTimeout(timeout time.Duration) RequestOption {
	return func(cfg *RequestConfig) {
		cfg.Timeout = timeout
	}
}

func WithHeader(key, value string) RequestOption {
	return func(cfg *RequestConfig) {
		if cfg.Headers == nil {
			cfg.Headers = make(http.Header)
		}
		cfg.Headers.Set(key, value)
	}
}

func WithHeaders(headers http.Header) RequestOption {
	return func(cfg *RequestConfig) {
		if cfg.Headers == nil {
			cfg.Headers = make(http.Header)
		}
		for key, values := range headers {
			for _, value := range values {
				cfg.Headers.Add(key, value)
			}
		}
	}
}

func (c *ModelClient) requestConfig(opts []RequestOption) RequestConfig {
	cfg := RequestConfig{
		Headers: make(http.Header),
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func (s *CompletionsService) CreateStreamWithResponse(
	ctx context.Context,
	params ChatCompletionCreateParams,
	opts ...RequestOption,
) (*StreamResponse[ChatCompletionChunk], error) {
	params.Stream = true
	resp, err := s.client.doJSON(ctx, http.MethodPost, "/chat/completions", params, s.client.requestConfig(opts))
	if err != nil {
		return nil, err
	}

	stream := NewSSEStream[ChatCompletionChunk](ctx, resp.Body)
	return &StreamResponse[ChatCompletionChunk]{
		Stream:     stream,
		Response:   resp,
		RequestID:  resp.Header.Get("x-request-id"),
		StatusCode: resp.StatusCode,
		Header:     resp.Header.Clone(),
	}, nil
}

type Stream[T any] struct {
	ctx    context.Context
	cancel context.CancelFunc
	body   io.ReadCloser

	closeOnce   sync.Once
	releaseOnce sync.Once
	release     func()

	chunks  chan T
	current T

	mu  sync.Mutex
	err error
}

func NewSSEStream[T any](ctx context.Context, body io.ReadCloser, release func()) *Stream[T] {
	ctx, cancel := context.WithCancel(ctx)
	s := &Stream[T]{
		ctx:     ctx,
		cancel:  cancel,
		body:    body,
		release: release,
		chunks:  make(chan T, 64),
	}
	go s.read()
	return s
}

func (s *Stream[T]) Next() bool {
	select {
	case item, ok := <-s.chunks:
		if !ok {
			return false
		}

		s.current = item
		return true

	case <-s.ctx.Done():
		return false
	}
}

func (s *Stream[T]) Current() T {
	return s.current
}

func (s *Stream[T]) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *Stream[T]) Close() error {
	s.cancel()
	return s.closeBody()
}

func (s *Stream[T]) read() {
	defer close(s.chunks)
	defer s.closeBody()

	err := ReadSSE(s.ctx, s.body, func(event SSEEvent) error {
		if event.Data == "" || strings.HasPrefix(event.Data, "[DONE]") {
			return nil
		}

		data := []byte(event.Data)
		if err := errorFromEventData(data); err != nil {
			return err
		}

		var item T
		if err := json.Unmarshal(data, &item); err != nil {
			return fmt.Errorf("decode SSE JSON: %w", err)
		}

		select {
		case s.chunks <- item:
			return nil
		case <-s.ctx.Done():
			return s.ctx.Err()
		}
	})

	if err != nil && !errors.Is(err, context.Canceled) {
		s.setErr(err)
	}
}

func (s *Stream[T]) closeBody() error {
	var err error
	s.closeOnce.Do(func() {
		err = s.body.Close()
		s.releaseOnce.Do(func() {
			if s.release != nil {
				s.release()
			}
		})
	})
	return err
}

func (s *Stream[T]) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
}

type SSEEvent struct {
	Event string
	Data  string
	Raw   []string
}

func ReadSSE(ctx context.Context, r io.Reader, onEvent func(SSEEvent) error) error {
	reader := bufio.NewReaderSize(r, 64*1024)

	var raw []string
	var data []string
	var eventName string

	flush := func() error {
		if len(raw) == 0 && len(data) == 0 && eventName == "" {
			return nil
		}
		event := SSEEvent{
			Event: eventName,
			Data:  strings.Join(data, "\n"),
			Raw:   append([]string(nil), raw...),
		}
		raw = raw[:0]
		data = data[:0]
		eventName = ""
		return onEvent(event)
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
		} else {
			raw = append(raw, line)
			if strings.HasPrefix(line, ":") {
				// comment or heartbeat
			} else if field, value, ok := strings.Cut(line, ":"); ok {
				value = strings.TrimPrefix(value, " ")
				switch field {
				case "event":
					eventName = value
				case "data":
					data = append(data, value)
				}
			}
		}

		if err == io.EOF {
			return flush()
		}
	}
}

func defaultHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).
		Clone()

	transport.ForceAttemptHTTP2 = true
	transport.MaxIdleConns = 200
	transport.MaxIdleConnsPerHost = 100
	transport.MaxConnsPerHost = 100
	transport.IdleConnTimeout = 90 * time.Second
	transport.ResponseHeaderTimeout = 30 * time.Second
	transport.TLSHandshakeTimeout = 10 * time.Second

	return &http.Client{
		Transport: transport,

		// SSE 整体超时由 Context 控制。
		Timeout: 0,
	}
}

func cloneHeader(in http.Header) http.Header {
	if in == nil {
		return make(http.Header)
	}
	return in.Clone()
}
