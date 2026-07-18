# 发布指南

Daksha 是纯 Go SDK。Go module tag 是正式发布物；GitHub Release 提供面向用户的发布
说明，不发布平台二进制包。

## 版本政策

版本遵循 Semantic Versioning：

- `v0.x.y`：API 尚未稳定。兼容修复提升 patch；功能和破坏性 API 变化至少提升 minor。
- `v1+`：兼容修复提升 patch，向后兼容功能提升 minor，破坏性变化提升 major。
- 预发布版本使用 `-alpha.1`、`-beta.1`、`-rc.1`。
- 安全修复使用 patch，并在 CHANGELOG 的 Security 小节说明。

`v0` 和 `v1` 使用模块路径 `github.com/zhongys/Daksha`。发布 `v2.0.0` 前必须把
module path 和内部 import 改为 `github.com/zhongys/Daksha/v2`。

已发布 tag 永不移动、覆盖或复用。错误版本应通过更高版本的 `retract` 指令撤回，
不能重新标记同一版本。

## 首次发布

建议首个候选版本为 `v0.1.0-rc.1`，验证完成后发布 `v0.1.0`。完整实现当前位于
`dev-agent-go`；必须先通过 PR 合并到受保护的 `main`，再从 `main` 发布。

首次运行发布 workflow 前，仓库管理员必须完成以下一次性配置：

1. 启用 GitHub Immutable Releases。该设置只保护启用后发布的新 Release。
2. 预先创建 `release` environment，配置 required reviewers，并把 deployment branch
   限制为受保护的 `main`。在该 environment 中添加 `RELEASE_SETTINGS_TOKEN` secret：
   使用仅授予本仓库 `Administration: read` 的 fine-grained token，用于在创建 tag 前
   查询 Immutable Releases 设置。仅在 YAML 中引用 environment 不会自动产生审批保护。
3. 为 `v*` 建立 tag rulesets：允许 GitHub Actions App 作为创建 tag 的 bypass actor；
   另以不提供日常 bypass 的规则禁止更新和删除 tag，保护 tag 创建到 Release 发布之间
   的窗口。
4. 启用受保护 `main`、必需 CI checks 和 Private vulnerability reporting。

## 准备发布

1. 确认目标提交已经合并到最新 `main`。
2. 根据 API 变化选择 SemVer。
3. 把 CHANGELOG 的 Unreleased 内容移入带日期的版本标题，例如：

   ```markdown
   ## [0.1.0] - 2026-07-18
   ```

4. 保留一个新的空 `## [Unreleased]` 区域。
5. 更新 README、API 文档、示例和迁移说明。
6. 确认本地检查通过：

   ```bash
   go mod tidy -diff
   go mod verify
   go vet ./...
   go test -shuffle=on -count=1 ./...
   go test -race -shuffle=on -count=1 ./...
   govulncheck ./...
   ```

7. 等待 `main` 的全部必需 CI checks 通过。

## 自动发布

在 GitHub Actions 中选择 `Release` workflow：

1. 将运行分支选择为 `main`。
2. 输入完整版本，例如 `v0.1.0`。
3. 核对上述仓库控制仍然启用，然后勾选 `release_controls_confirmed`。
4. workflow 验证：
   - 严格 SemVer 和 module major path。
   - 当前提交是最新 `origin/main`。
   - CHANGELOG 包含带日期的版本条目。
   - gofmt、module metadata、vet、test、race 和 govulncheck。
   - 同名 tag 不存在，或仅允许它已经指向同一提交以便恢复失败的 Release 创建。
5. `release` environment 获批后，workflow 先通过只读 token 确认 Immutable Releases
   已启用，再创建 annotated tag 和 GitHub Release，并确认 Release 非 draft、预发布
   状态正确、已不可变且 release attestation 验证通过。

发布 workflow 不会移动现有 tag。并发发布被串行化；如果 `main` 在验证后发生变化，
发布阶段会失败并要求重新开始。

## 发布后验证

替换下面的版本并在一个全新的临时模块中执行：

```bash
go list -m github.com/zhongys/Daksha@v0.1.0
go mod init example.com/daksha-smoke
go get github.com/zhongys/Daksha/agent@v0.1.0
go list github.com/zhongys/Daksha/agent github.com/zhongys/Daksha/ai
```

随后确认：

- GitHub Release 和自动生成的源码归档可用。
- `proxy.golang.org` 能解析版本。
- pkg.go.dev 已显示 `agent`、`ai` 和协议适配包的文档。
- README badge 与 CHANGELOG 链接正常。

仓库设置属于发布链的一部分；任何一项失效时都不要勾选
`release_controls_confirmed`，应先恢复控制再发布。
