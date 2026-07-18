# Security Policy

## Supported versions

在 `v1.0.0` 前，仅最新发布的 `v0` minor line 接收安全修复。`v1+` 的支持窗口会在
稳定版本发布时另行声明。宿主应用必须使用仍受 Go 官方支持且已安装最新安全补丁的
工具链；本仓库最低为 Go 1.26.5，并建议始终使用 Go 1.26 发布线的最新安全 patch。

## Reporting a vulnerability

请通过仓库的 [GitHub Security Advisory](https://github.com/zhongys/Daksha/security/advisories/new)
私密报告安全问题，不要在公开 issue 中披露
漏洞、凭证或可利用细节。报告应包含受影响版本、影响、复现步骤和建议缓解措施。

## SDK trust boundary

`ai.Provider`、`ai.Model`、工具定义和 StreamOptions 是可信宿主应用配置。Daksha 不
实现入站鉴权、租户隔离、供应商域名白名单或出口网络策略。把这些对象暴露给不可信
用户的宿主应用必须自行验证 BaseURL，隔离凭证，限制 Extra/Compat，并为有副作用的
工具执行授权和审计。
