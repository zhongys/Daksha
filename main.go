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
	for _, p := range ai.SeedProviders() {
		if err := client.PutProvider(p); err != nil {
			panic(err)
		}
	}
	// In production the model catalog is synced from the database. Pricing
	// is nano-yuan per token: ¥4/MTok in, ¥16/MTok out, ¥0.8/MTok cache read.
	if err := client.PutModel(ai.Model{
		Provider: "deepseek",
		ID:       "deepseek-chat",
		ToolCall: true,
		Pricing:  ai.Pricing{Input: 4000, Output: 16000, CacheRead: 800},
	}); err != nil {
		panic(err)
	}

	// agent layer assembly: the application talks to the Agent only.
	a, err := agent.New(agent.Config{
		LLM:          client,
		Provider:     "deepseek",
		Model:        "deepseek-chat",
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
