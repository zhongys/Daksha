# Changelog

本项目的用户可见变更记录在此文件中。格式参考 Keep a Changelog，版本遵循
Semantic Versioning。

## [Unreleased]

### Added

- Provider-neutral AI Client、消息与事件流抽象。
- Stateful Agent lifecycle、steering、follow-up、abort 和工具循环。
- Typed tools、parallel/sequential execution 和 terminal structured output。
- OpenAI-compatible Chat Completions 与 Embeddings adapter。
- Structured JSON output、batch completion、usage 和 nano-yuan cost accounting。
- Kimi K3 endpoint compatibility and model example。
- Linux/Windows/macOS CI、race detector 和 govulncheck 门禁。
- SDK、API、Provider 信任边界和发布文档。
- MIT License 与带不可变性验证的 GitHub Release workflow。

### Changed

- Provider 和 Model catalog 由宿主应用在运行时注册，SDK 不再持有 seed catalog。
- 最低 Go 版本提高到包含安全修复的 Go 1.26.5。

### Security

- 排除受 GO-2026-5856 影响的仓库工具链，并在 CI 和发布流程中执行 govulncheck。
