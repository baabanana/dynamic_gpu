# 文档索引

项目介绍和快速上手见仓库根目录的 [README.md](../README.md)（English）/ [README.zh-CN.md](../README.zh-CN.md)（中文）。

## 使用

| 文档 | 用途 | 状态 |
|---|---|---|
| [usage.md](usage.md) | 使用说明：安装、配置、写 job.yaml、命令、数据、停止计费的保证 | 部分流程已在 Vast 上验证，见设计第 8 节 |
| [../skills/dgpu/SKILL.md](../skills/dgpu/SKILL.md) | Claude 用的 Skill：环境检查、job.yaml、标准流程、按结果处理、安全规则 | 草案（尚未完整地在真实平台验证） |
| [../examples/yolo/README.md](../examples/yolo/README.md) | YOLO 示例：coco8 上训练 YOLO26n、中断续跑、费用估算 | 草案（尚未在真实平台验证） |

## 设计

| 文档 | 用途 | 状态 |
|---|---|---|
| [dgpu-design.md](dgpu-design.md) | dgpu 调度工具的设计：任务约定、销毁保证、失败模式、验收标准、真实平台验证记录 | 设计已确认（2026-11-10）；4.12、4.13 为新增草案 |

## 开发过程记录（归档）

记录当时的计划和评审，内容反映写作时的状态，不随代码更新。

| 文档 | 用途 | 状态 |
|---|---|---|
| [history/dgpu-implementation-plan.md](history/dgpu-implementation-plan.md) | 分阶段并行开发方案：契约、工作包、探针、集成与评审 | 归档 |
| [history/dgpu-review-2026-11-10.md](history/dgpu-review-2026-11-10.md) | 对抗式评审问题清单：问题、核实方式、修复状态和对应测试 | 归档；问题已在本地修复 |

## 平台调研

2026-11-09 的调研笔记，价格和接口会变化，以平台官方文档为准。

| 文档 | 用途 | 状态 |
|---|---|---|
| [research/cloud-gpu-2026-11/vast_runpod.md](research/cloud-gpu-2026-11/vast_runpod.md) | Vast.ai 与 RunPod 的 API、价格、容器权限 | 调研笔记（未租用实测） |
| [research/cloud-gpu-2026-11/aws.md](research/cloud-gpu-2026-11/aws.md) | AWS Spot、SageMaker、Batch | 调研笔记（未租用实测） |
| [research/cloud-gpu-2026-11/autodl_china.md](research/cloud-gpu-2026-11/autodl_china.md) | AutoDL 与国内平台的 API、价格、到 R2 的网络 | 调研笔记（未租用实测） |
| [research/cloud-gpu-2026-11/orchestration.md](research/cloud-gpu-2026-11/orchestration.md) | SkyPilot、dstack、Modal 等编排方案 | 调研笔记（未运行） |
| [research/cloud-gpu-2026-11/r2_yolo_workflow.md](research/cloud-gpu-2026-11/r2_yolo_workflow.md) | R2 数据层、挂载与拷贝对比、YOLO 断点恢复 | 调研笔记（未实测） |

## 参与开发

| 文档 | 用途 |
|---|---|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | 开发环境、测试、提交 PR 的流程 |
| [../AGENTS.md](../AGENTS.md) | 完整的开发规则和项目红线（人和 AI 编程助手共用） |
| [../SECURITY.md](../SECURITY.md) | 报告安全问题；凭据和计费相关的威胁模型 |
| [../CHANGELOG.md](../CHANGELOG.md) | 版本变更记录 |
