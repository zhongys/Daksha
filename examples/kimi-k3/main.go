package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/zhongys/Daksha/agent"
	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/api/openaicompletions"
	"github.com/zhongys/Daksha/ai/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	if os.Getenv("MOONSHOT_API_KEY") == "" {
		return errors.New("MOONSHOT_API_KEY is not set")
	}

	client := ai.NewClient(map[string]ai.Streamer{
		"openai-completions": openaicompletions.NewStreamer(
			openai.WithRequestTimeout(2 * time.Minute),
		),
	})

	// Providers and models are trusted host-application configuration. A
	// production application can load them from its own database or config
	// service; the SDK intentionally does not own a vendor catalog.
	const providerName = "moonshot-production"
	if err := client.PutProvider(ai.Provider{
		Name:      providerName,
		BaseURL:   "https://api.moonshot.cn/v1/",
		APIKeyEnv: "MOONSHOT_API_KEY",
	}); err != nil {
		return err
	}
	if err := client.PutModel(ai.Model{
		Provider:        providerName,
		ID:              "kimi-k3",
		Reasoning:       true,
		ToolCall:        true,
		ImageInput:      true,
		VideoInput:      true,
		ContextWindow:   1_048_576,
		MaxOutputTokens: 1_048_576,
	}); err != nil {
		return err
	}

	runner, err := agent.New(agent.Config{
		LLM:          client,
		Provider:     providerName,
		Model:        "kimi-k3",
		SystemPrompt: "You are a concise assistant.",
		Options: ai.StreamOptions{
			ReasoningEffort: "max",
		},
		MaxTurns: 8,
	})
	if err != nil {
		return err
	}

	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(signalCtx, 5*time.Minute)
	defer cancel()

	stream, err := runner.PromptText(ctx, "用一句话介绍你自己。")
	if err != nil {
		return err
	}
	for event := range stream.Events() {
		if update, ok := event.(agent.MessageUpdateEvent); ok {
			if delta, ok := update.Inner.(ai.TextDeltaEvent); ok {
				fmt.Print(delta.Delta)
			}
		}
	}
	fmt.Println()

	result, err := stream.Result(context.Background())
	if err != nil {
		return fmt.Errorf("stream failed: %w", err)
	}
	if result == nil {
		return errors.New("agent run ended without a result")
	}
	if result.Err != nil {
		return fmt.Errorf("agent run failed: %w", result.Err)
	}
	if result.Last == nil {
		return errors.New("agent run ended without an assistant message")
	}
	if result.Last.StopReason == ai.StopReasonError || result.Last.StopReason == ai.StopReasonAborted {
		return fmt.Errorf("model failed: %s", result.Last.ErrorMessage)
	}

	fmt.Printf("tokens: in=%d out=%d\n",
		result.Last.Usage.Input, result.Last.Usage.Output)
	return nil
}
