package main

import (
	"context"
	"fmt"

	"github.com/zhongys/Daksha/internal/ai"
	"github.com/zhongys/Daksha/internal/ai/api/openaicompletions"
)

func main() {
	// Assembly: one stateless adapter per wire protocol, injected explicitly.
	client := ai.NewClient(map[string]ai.Streamer{
		"openai-completions": openaicompletions.NewStreamer(),
	})

	// Seed well-known endpoints; keys resolve from env (APIKey field wins
	// when the application fills it from storage).
	for _, p := range ai.SeedProviders() {
		if err := client.PutProvider(p); err != nil {
			panic(err)
		}
	}

	// In production the model catalog lives in the database and is synced
	// into the registry; a dev entry is a few lines. Pricing is nano-yuan
	// per token: ¥4/MTok in, ¥16/MTok out, ¥0.8/MTok cache read.
	if err := client.PutModel(ai.Model{
		Provider: "deepseek",
		ID:       "deepseek-chat",
		ToolCall: true,
		Pricing:  ai.Pricing{Input: 4000, Output: 16000, CacheRead: 800},
	}); err != nil {
		panic(err)
	}

	prompt := ai.Prompt{
		Messages: []ai.Message{&ai.UserMessage{
			Role: ai.RoleUser,
			Content: []ai.UserContent{
				&ai.TextContent{Type: ai.ContentTypeText, Text: "用一句话介绍你自己。"},
			},
		}},
	}

	ctx := context.Background()
	stream := client.Stream(ctx, "deepseek", "deepseek-chat", prompt, ai.StreamOptions{})
	for ev := range stream.Events() {
		if delta, ok := ev.(ai.TextDeltaEvent); ok {
			fmt.Print(delta.Delta)
		}
	}
	fmt.Println()

	msg, err := stream.Result(ctx)
	if err != nil {
		panic(err)
	}
	if msg.StopReason == ai.StopReasonError || msg.StopReason == ai.StopReasonAborted {
		fmt.Printf("turn failed: %s\n", msg.ErrorMessage)
		return
	}
	fmt.Printf("tokens: in=%d out=%d, cost: %d nano-yuan (¥%.6f)\n",
		msg.Usage.Input, msg.Usage.Output, msg.Usage.Cost.Total, float64(msg.Usage.Cost.Total)/1e9)
}
