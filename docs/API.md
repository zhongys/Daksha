# Daksha API 指南

本文说明 Daksha 作为 Agent SDK 的公开契约。公开 GoDoc 以源码为准；行为或生命周期
契约发生变化时，必须同步更新本文和 CHANGELOG。

## 架构

```text
宿主应用
   │
   ▼
agent.Agent（每个会话一个）
   │
   ▼
ai.Client（Provider/Model 注册表与路由）
   │
   ▼
ai.Streamer / ai.Embedder（协议适配器）
   │
   ▼
OpenAI-compatible endpoint
```

`agent` 只依赖 `LLM` 接口，因此宿主可以在 Agent 与 `ai.Client` 之间加入限流、
审计、缓存或测试替身。`ai.Client` 的注册表只保存在内存中；数据库、配置服务和
密钥管理由宿主应用负责。

## Client、Provider 与 Model

使用协议名装配 Client：

```go
client := ai.NewClient(map[string]ai.Streamer{
	"openai-completions": openaicompletions.NewStreamer(
		openai.WithRequestTimeout(2 * time.Minute),
	),
})
client.PutEmbedder("openai-completions", openaicompletions.NewEmbedder(
	openai.WithRequestTimeout(30*time.Second),
))
```

Provider 名称是宿主定义的注册表键，不参与供应商识别。`Provider.API` 为空时默认
使用 `openai-completions`。适配器根据 `BaseURL` 自动识别常见端点差异；自定义代理
无法识别时可通过 `Provider.Compat` 显式覆盖。

Provider 和 Model 的 Put/Get/List 操作并发安全。注册时会把 `Extra` 规范化为独立
JSON 树；每次请求取得配置快照，因此配置更新只影响后续请求，不改变正在进行的流。

`APIKeyEnv` 只在 `APIKey` 为空时使用，并在每次请求解析，便于开发环境密钥轮换。
Provider 是可信宿主配置，不能直接绑定不可信网络请求。尤其注意：

- `GetProvider`/`ListProviders` 返回的快照可能包含 API Key。
- `Provider.Extra < Model.Extra < StreamOptions.Extra`，后者优先。
- Extra 与标准请求字段同名时可以覆盖标准字段，动态配置必须使用白名单。

Model 的能力开关会在发送前检查工具和媒体输入。`ContextWindow` 与
`MaxOutputTokens` 只是目录元数据；它们不会裁剪上下文或设置请求参数。宿主必须自行
执行 token/cost 预算，并通过 `StreamOptions.MaxTokens` 设置输出限制。

## Stream 与 Complete

`ai.Client.Stream` 返回模型事件流。配置、编码、HTTP 和解析错误不会作为同步 error
返回，而是产生终止 `ai.ErrorEvent`，最终 `AssistantMessage.StopReason` 为
`ai.StopReasonError`。context 取消属于流机制错误，由 `Result` 返回。

```go
stream := client.Stream(ctx, provider, model, prompt, options)
for event := range stream.Events() {
	// Handle Start/TextDelta/ThinkingDelta/ToolCallDelta/Done/Error.
}
message, err := stream.Result(context.Background())
```

必须消费 Events。Producer 在事件缓冲区填满后施加背压；只调用 `Result` 而不读取事件
可能阻塞。无需增量事件时使用 `Client.Complete`，它会在内部排空事件流。

## Prompt、消息与持久化

`ai.Prompt` 包含 system 指令、消息历史和工具定义。消息是以下联合类型：

- `*ai.UserMessage`
- `*ai.AssistantMessage`
- 任意实现 `ai.ToolResult` 的 `*ai.ToolResultMessage[T]`

内容类型包括文本、结构化 JSON、图片、音频、视频、思考和工具调用。媒体必须设置
Data 或 URL 之一；内联 Data 还必须设置 MIME type。

`Client.Stream`、Agent 配置和事件会对调用方拥有的 map/slice/pointer 做结构快照。
宿主可以在边界返回后复用原输入，但包含函数、channel 或未导出可变状态的对象仍由
调用方负责保持不可变。

持久化消息后可恢复：

```go
encoded, err := json.Marshal(runner.Messages())
if err != nil {
	return err
}
messages, err := ai.UnmarshalMessages(encoded)
if err != nil {
	return err
}
runner, err = agent.New(agent.Config{
	LLM: client, Provider: provider, Model: model, Messages: messages,
})
if err != nil {
	return err
}
```

Kimi K3 等模型要求多轮对话原样回传完整 assistant message。适配器会把流式
`reasoning_content` 保存为 `ThinkingContent` 并在后续请求中回放。

## StreamOptions 与结构化输出

`ai.StreamOptions` 提供 temperature、top-p、最大输出、停止词、工具选择、推理力度、
输出格式和请求级 Extra。供应商的合法值可能不同；协议适配器不会替宿主实现每个
模型的业务参数校验。

JSON Schema 输出示例：

```go
options := ai.StreamOptions{
	OutputFormat: ai.OutputFormat{
		Type: ai.OutputFormatJSONSchema,
		JSONSchema: &ai.JSONSchema{
			Name:   "person",
			Strict: true,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"required":             []string{"name", "age"},
				"additionalProperties": false,
			},
		},
	},
}
```

JSON streaming 使用 `JSONStartEvent`、`JSONDeltaEvent`、`JSONEndEvent`。只有 End
事件和最终 `JSONContent.Value` 保证是完整合法 JSON；JSON Schema 语义由供应商执行。

## Agent 状态与生命周期

一个 `agent.Agent` 表示一个会话，所有方法并发安全，但一次只允许一个活动 run。

| API | 契约 |
|---|---|
| `Prompt` / `PromptText` | 空闲时追加用户消息并异步启动 run |
| `Continue` | 历史以 user/toolResult 结尾时继续，常用于错误后重试 |
| `Steer` | 当前工具批完成后向下一轮注入用户消息 |
| `FollowUp` | run 本应自然结束时追加下一轮 |
| `Abort` | 取消活动 run；未开始的工具不会进入 Execute |
| `Reset` | 仅在空闲时清空历史和队列 |
| `Messages` | 返回会话历史快照 |
| setters | 可在运行时调用，从下一轮配置快照开始生效 |

并发 `Prompt` 或 `Continue` 返回 `agent.ErrBusy`。`MaxTurns=0` 表示无限制；生产宿主
应设置正数，并通过 `ShouldStopAfterTurn` 实现额外的成本或业务熔断。

## 工具

类型安全工具会把模型参数 JSON 解码到声明的 Go 类型：

```go
type WeatherArgs struct {
	City string `json:"city"`
}

weather, err := agent.NewTool[WeatherArgs](
	agent.ToolDefinition{
		ToolDefinition: ai.ToolDefinition{
			Name:        "get_weather",
			Description: "查询指定城市的天气",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required":             []string{"city"},
				"additionalProperties": false,
			},
		},
		Label: "天气查询",
	},
	func(ctx context.Context, _ string, params WeatherArgs, _ func(agent.ToolUpdate)) (*agent.ToolOutput, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return &agent.ToolOutput{
			Content: []ai.ToolResultContent{
				&ai.TextContent{
					Type: ai.ContentTypeText,
					Text: fmt.Sprintf("%s：晴，24°C", params.City),
				},
			},
		}, nil
	},
)
```

默认工具执行模式为串行。只有显式设置 `ExecParallel` 才会并行；非线程安全工具可用
`agent.Sequential(tool)` 强制整个批次串行。

工具 handler 的 error 会转换为 `IsError` 工具结果并交回模型，而不是自动终止 run。
handler 必须自行执行参数业务校验、幂等和授权。`BeforeToolCall` 是权限和审计入口；
有副作用的工具不应在缺少该策略时暴露给不可信 prompt。

`agent.NewTerminalTool[T]` 可把工具参数作为最终业务输出。成功后通过
`agent.RunOutputAs[T](result.Output)` 读取。

## 错误模型

| 层级 | 检查位置 | 示例 |
|---|---|---|
| 同步调用 | `Prompt`/`PromptText` error | nil context、Agent 正忙 |
| 流机制 | `stream.Result` error | context 取消或 deadline |
| Agent 循环 | `RunResult.Err` | MaxTurns、TransformContext、工具状态错误 |
| Provider/协议/模型 | `Last.StopReason` 与 `ErrorMessage` | HTTP、鉴权、解析、供应商错误 |
| 工具 handler | `ToolResultMessage.IsError` | 参数或业务执行错误 |

直接调用 `Client.Complete` 时，返回的 error 只表示流机制失败；仍必须检查
`AssistantMessage.StopReason`。

## Batch 与 Embeddings

`Client.CompleteBatch` 按输入顺序返回结果，并通过 `maxConcurrency` 限制并发。一个请求
失败不会取消其他请求；宿主仍应在更高层实现供应商速率、预算和重试策略。

`Client.Embed` 需要先通过 `PutEmbedder` 注册对应协议的 Embedder。返回向量与输入按
索引对齐。空输入直接返回空结果，不发送网络请求。

## Kimi K3

官方 Moonshot BaseURL 会自动启用 Kimi 兼容行为。Provider 名称可以任意命名。

- 模型 ID：`kimi-k3`
- 上下文：1,048,576 tokens
- `reasoning_effort`：当前仅 `max`，默认也是 `max`
- `max_completion_tokens`：默认 131072，最大 1048576
- 不使用 K2.x `thinking`
- 建议省略固定的 temperature、top-p、n 和 penalties

如果通过不含 Moonshot 域名的自定义代理访问 K3，应显式覆盖 endpoint compat：

```go
supportsDeveloperRole := false
provider.Compat = &ai.OpenAICompat{
	MaxTokensField:       "max_completion_tokens",
	ThinkingFormat:       ai.ThinkingFormatOpenAI,
	SupportsDeveloperRole: &supportsDeveloperRole,
}
```

模型参数仍由宿主配置层负责验证。SDK 的兼容层只处理 wire-level 差异，不根据 Provider
名称硬编码模型策略。

## API 稳定性

`v0.x.y` 阶段可能发生破坏性变更，但必须提升 minor 版本并提供 CHANGELOG 迁移说明。
`v1.0.0` 后遵循 Go module 与 SemVer 兼容承诺。`agent`、`ai`、
`ai/api/openaicompletions` 和 `ai/openai` 当前都属于公开包。
