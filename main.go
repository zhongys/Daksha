package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/zhongys/Daksha/internal/ai/openai"
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

	err := openai.ExecuteNewRequest(
		context.Background(),
		http.MethodPost,
		"/chat/completions",
		body,
		&response,
		openai.WithDefaultBaseURL(""),
		openai.RequestOptionFunc(func(cfg *openai.RequestConfig) error {
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
