package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	requestconfig "github.com/zhongys/Daksha.git/internal/ai/sdk/requestConfig"
)

func main() {
	body := struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}{
		Model: "qwen3.7-max-2026-05-20",
		Input: "hello",
	}

	var response interface{}

	err := requestconfig.ExecuteNewRequest(
		context.Background(),
		http.MethodPost,
		"/chat/completions",
		body,
		&response,
		requestconfig.WithDefaultBaseURL(""),
		requestconfig.RequestOptionFunc(func(cfg *requestconfig.RequestConfig) error {
			cfg.SetAPIKey("")
			cfg.RequestTimeout = 30 * time.Second
			return nil
		}),
	)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%+v\n", response)
}
