# Daksha

[![CI](https://github.com/zhongys/Daksha/actions/workflows/ci.yml/badge.svg)](https://github.com/zhongys/Daksha/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/zhongys/Daksha/agent.svg)](https://pkg.go.dev/github.com/zhongys/Daksha/agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Daksha 是一个 Go Agent SDK。它提供与供应商无关的消息、流、工具和 Agent
循环，并通过协议适配器连接 OpenAI-compatible API。它是供宿主应用集成的
SDK，不是可直接部署的 HTTP 服务。

当前 API 处于 `v0` 阶段。在发布 `v1.0.0` 前，破坏性变更会通过次版本号和
CHANGELOG 明确说明。

## 功能

- 每个会话一个、并发安全的 Agent，支持流式事件、取消、转向和 follow-up。
- 类型安全工具、串行/并行执行、权限钩子和 terminal structured output。
- 应用持有的 Provider/Model 注册表，可在运行时更新并按请求快照。
- OpenAI-compatible Chat Completions 与 Embeddings 适配器。
- 文本、JSON、图片、音频、视频、思考内容和工具调用消息。
- 结构化输出、批量请求、token 使用量与成本计算。
- Kimi K3、DeepSeek、Qwen/DashScope、OpenRouter 和 z.ai 兼容处理。

## 要求与安装

- 最低 Go 版本：`1.26.5`

Go 1.26.5 包含 2026 年 7 月发布的 Go 标准库安全修复。

```bash
go get github.com/zhongys/Daksha/agent@latest
```

SDK 的主要导入路径是：

```go
import (
	"github.com/zhongys/Daksha/agent"
	"github.com/zhongys/Daksha/ai"
	"github.com/zhongys/Daksha/ai/api/openaicompletions"
	"github.com/zhongys/Daksha/ai/openai"
)
```

## Kimi K3 快速开始

设置密钥并运行完整示例：

```bash
export MOONSHOT_API_KEY="your-key"
go run ./examples/kimi-k3
```

PowerShell：

```powershell
$env:MOONSHOT_API_KEY = "your-key"
go run ./examples/kimi-k3
```

最小装配方式：

```go
client := ai.NewClient(map[string]ai.Streamer{
	"openai-completions": openaicompletions.NewStreamer(
		openai.WithRequestTimeout(2 * time.Minute),
	),
})

const providerName = "moonshot-production" // 只是注册表键，可自行命名
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
```

Kimi K3 当前始终推理，`reasoning_effort` 只支持 `max`，省略时也使用模型
默认值。不要传 K2.x 的 `thinking` 字段，也建议省略固定的 `temperature`、
`top_p` 等参数。

`ContextWindow` 和 `MaxOutputTokens` 是宿主应用可读取的目录元数据，不会自动
限制请求。需要显式限制输出时设置 `ai.StreamOptions.MaxTokens`；K3 当前
`max_completion_tokens` 默认 131072，最大 1048576。

## 消费流与处理错误

Agent run 是异步的。必须先持续消费 `Events()` 直到关闭，再读取 `Result()`；
事件通道具有背压，只等待结果而不消费事件可能阻塞生产端。

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

stream, err := runner.PromptText(ctx, "用一句话介绍你自己。")
if err != nil {
	return err // 调用错误，例如同一 Agent 正忙
}

for event := range stream.Events() {
	if update, ok := event.(agent.MessageUpdateEvent); ok {
		if delta, ok := update.Inner.(ai.TextDeltaEvent); ok {
			fmt.Print(delta.Delta)
		}
	}
}

result, err := stream.Result(context.Background())
if err != nil {
	return err // context 取消、超时或流机制失败
}
if result == nil {
	return errors.New("agent run ended without a result")
}
if result.Err != nil {
	return result.Err // Agent 循环、上下文转换或工具状态错误
}
if result.Last == nil {
	return errors.New("agent run ended without an assistant message")
}
if result.Last.StopReason == ai.StopReasonError ||
	result.Last.StopReason == ai.StopReasonAborted {
	return errors.New(result.Last.ErrorMessage) // Provider、协议或模型错误
}
```

同时配置两层时间预算：

- `context.WithTimeout` 限制整个 Agent run，包括多轮模型调用和工具执行。
- `openai.WithRequestTimeout` 限制每次 HTTP/SSE 请求，直到响应体关闭。

## Provider 信任契约

`ai.Provider` 是宿主应用的可信配置对象，不是面向不可信用户的 HTTP DTO。
Daksha 有意支持任意 OpenAI-compatible 网关、本地模型和代理，因此 SDK 不执行
域名白名单、租户鉴权或出口网络策略。

如果宿主应用允许用户动态配置 Provider，宿主必须在调用 `PutProvider` 前：

- 验证 BaseURL，并隔离内网和 metadata 出口。
- 隔离并加密 API Key，禁止用户任意指定 `APIKeyEnv`。
- 对 Provider、Model 和 Secret 执行租户授权。
- 限制 `Compat` 和 `Extra`；`Extra` 可以覆盖标准请求字段。

`GetProvider` 和 `ListProviders` 返回完整配置，可能包含 API Key。不要把结果直接
序列化为外部响应或写入日志。

## Agent 生命周期

一个 `Agent` 代表一个会话。它并发安全，但同一实例一次只运行一个 Prompt；并发
`Prompt`/`Continue` 返回 `agent.ErrBusy`。

- `Prompt` / `PromptText`：追加用户消息并启动 run。
- `Continue`：从以 user/toolResult 结尾的历史重试。
- `Steer`：当前工具批结束后注入下一轮。
- `FollowUp`：run 自然结束时追加一轮。
- `Abort`：取消当前 run，并阻止尚未开始的工具进入 `Execute`。
- `Reset`：Agent 空闲时清空历史和队列。
- `Messages`：返回可持久化的消息快照。

生产使用应始终设置 `MaxTurns`，并为有副作用的工具配置 `BeforeToolCall` 权限与
审计钩子。

## 包结构

| 包 | 用途 |
|---|---|
| `agent` | 会话状态、Agent 循环、工具和生命周期事件 |
| `ai` | Provider/Model、消息、流、结构化输出、Embedding、成本 |
| `ai/api/openaicompletions` | OpenAI-compatible 协议适配器 |
| `ai/openai` | 底层 HTTP、SSE 和 OpenAI-compatible wire types |

更多内容见 [API 指南](docs/API.md)。

## 开发与发布

```bash
go mod tidy -diff
go vet ./...
go test -shuffle=on -count=1 ./...
go test -race -shuffle=on -count=1 ./... # Linux + CGO
govulncheck ./...
```

CI 在 Go 1.26.5 上覆盖 Linux、Windows 和 macOS；race detector 在
Linux 上作为独立门禁运行。普通单元测试使用本地 HTTP/SSE 测试服务器，不需要
供应商密钥。完整 CI 还会每周定时运行，以发现新披露的可达漏洞。

版本遵循 SemVer。当前使用 `v0.x.y`，API 稳定后发布 `v1.0.0`。发布步骤见
[发布指南](docs/RELEASING.md)，用户可见变更见 [CHANGELOG](CHANGELOG.md)。

## License

Daksha 使用 [MIT License](LICENSE)。
