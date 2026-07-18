package main

import (
	"context"
	"fmt"

	"github.com/zhongys/Daksha/agent"
	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/api/openaicompletions"
)

func main() {
	// ai layer assembly: one stateless adapter per wire protocol.
	client := ai.NewClient(map[string]ai.Streamer{
		"openai-completions": openaicompletions.NewStreamer(),
	})

	// Providers and models belong to the application. A production service
	// loads these values from its database or configuration service; this demo
	// injects one provider explicitly to show the same boundary.
	const providerName = "moonshot-production"
	if err := client.PutProvider(ai.Provider{
		Name:      providerName,
		BaseURL:   "https://api.moonshot.cn/v1/",
		APIKeyEnv: "MOONSHOT_API_KEY",
	}); err != nil {
		panic(err)
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
		panic(err)
	}

	// agent layer assembly: the application talks to the Agent only.
	a, err := agent.New(agent.Config{
		LLM:          client,
		Provider:     providerName,
		Model:        "kimi-k3",
		SystemPrompt: "You are a concise assistant.",
		MaxTurns:     8,
	})
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	stream, err := a.PromptText(ctx, "用一句话介绍你自己。")
	if err != nil {
		panic(err)
	}
	for ev := range stream.Events() {
		if update, ok := ev.(agent.MessageUpdateEvent); ok {
			if delta, ok := update.Inner.(ai.TextDeltaEvent); ok {
				fmt.Print(delta.Delta)
			}
		}
	}
	fmt.Println()

	res, err := stream.Result(ctx)
	if err != nil {
		panic(err)
	}
	if res.Err != nil {
		fmt.Printf("run failed: %v\n", res.Err)
		return
	}
	if res.Last.StopReason == ai.StopReasonError || res.Last.StopReason == ai.StopReasonAborted {
		fmt.Printf("turn failed: %s\n", res.Last.ErrorMessage)
		return
	}
	fmt.Printf("tokens: in=%d out=%d, cost: %d nano-yuan (¥%.6f)\n",
		res.Last.Usage.Input, res.Last.Usage.Output,
		res.Last.Usage.Cost.Total, float64(res.Last.Usage.Cost.Total)/1e9)
}
