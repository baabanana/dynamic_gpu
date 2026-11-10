# dgpu 实施方案：分阶段并行开发

状态：草案（待用户确认）
关联：[设计文档](dgpu-design.md)

## 1. 相对设计的变化

- 已确认：形态 CLI + Skill、只做单机（可多卡）、数据启动时拷到本地盘。
- 新增：AutoDL 后端，用弹性部署 API 的 `Job` 类型（用户有企业认证）。见设计 4.12。
- 新增：存储抽象成 S3 兼容的 profile，默认 R2；凭据改用预签名 URL 下发，不出现在平台控制台里。见设计 4.13。

## 2. 阶段

| 阶段 | 做什么 | 谁做 | 前置 | 需要你 |
|---|---|---|---|---|
| 0 骨架与契约 | 仓库结构、`AGENTS.md`、依赖、冻结的接口、共用的测试替身 | 我（主会话） | 本方案确认 | 否 |
| 1 并行开发 | 7 个工作包，每个 agent 一个独立 git worktree | 7 个 agent | 阶段 0 | 否 |
| P 云上探针 | 自毁、网络、FUSE、AutoDL 生命周期 | 我 | 你提供账户；与阶段 1 同时进行 | 账户；花钱的步骤先告诉你 |
| 2 集成与评审 | 按依赖合入集成分支，全量测试，对抗式评审 | 我 + 2 个评审 agent | 阶段 1 | 否 |
| 3 真实环境验收 | 设计第 6 节 E1–E5（Vast）、E1 和 E3（AutoDL） | 我 | 阶段 2 和 P | 花钱的步骤先告诉你 |
| 4 合入 | `feature/dgpu-mvp` → `dev` → `main` | 我 | 阶段 3 | **是** |

分支：每个工作包在 `wp/<包名>` 上开发，由我合入 `feature/dgpu-mvp`；`feature/dgpu-mvp` 合入 `dev` 和 `main` 前先征得你同意。

## 3. 阶段 0：先冻结的契约

几个 agent 能同时写代码，前提是接口先定死。阶段 0 只写这些文件：

| 文件 | 内容 |
|---|---|
| `src/dgpu/contracts.py` | `JobSpec`、`GpuRequest`、`Offer`、`LaunchSpec`、`InstanceRef`、`InstanceState` 数据类；`Backend`、`Store` 协议 |
| `src/dgpu/layout.py` | 存储路径约定（设计 4.4），只用标准库，runner 也要用 |
| `src/dgpu/states.py` | 任务状态、失败原因、事件名、runner 环境变量名，只用标准库 |
| `src/dgpu/testing.py` | `FakeBackend`、`MemoryStore`，所有工作包的测试共用 |
| `AGENTS.md`、`CLAUDE.md` | 开发规则：命令、文件归属、契约变更流程 |
| `pyproject.toml`、`Makefile` | uv、ruff、pytest；`make check` 一条命令跑完检查 |

接口草图：

```python
class Backend(Protocol):
    name: str                                              # "vast" | "autodl"
    def search(self, req: GpuRequest) -> list[Offer]: ...
    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef: ...
    def get(self, ref: InstanceRef) -> InstanceState: ...  # 统一阶段：pending/loading/running/exited/gone/unknown
    def destroy(self, ref: InstanceRef) -> None: ...       # 幂等，已不存在不报错
    def list(self, label_prefix: str) -> list[InstanceState]: ...

class Store(Protocol):
    def get_json(self, key: str) -> tuple[dict, str] | None: ...          # (内容, ETag)
    def put_json(self, key: str, obj: dict, *, if_match: str | None = None,
                 if_none_match: bool = False) -> str: ...                 # 条件不满足抛 PreconditionFailed
    def put_file(self, key: str, path: Path) -> None: ...
    def download_prefix(self, prefix: str, dest: Path) -> None: ...
    def list(self, prefix: str) -> list[str]: ...
    def presign_get(self, key: str, ttl_s: int) -> str: ...
    def delete(self, key: str) -> None: ...
```

## 4. 阶段 1：工作包

| 包 | 内容 | 负责的文件 | 模型 | 完成标准 |
|---|---|---|---|---|
| A 规格与打包 | 解析和校验 `job.yaml`（版本、固定镜像 tag、必填上限、按后端的镜像和价格）；读取配置；打包代码（git 文件列表或 `.dgpuignore`，排除敏感文件，200 MB 上限） | `spec.py`、`config.py`、`packaging.py` | Sonnet | 每种校验错误有测试 |
| B 存储 | S3 兼容 `Store`：JSON 读写、条件写、预签名、列举、整前缀下载；R2 的细节（`region=auto`、checksum 设为 `when_required`） | `store.py` | Sonnet | moto 测试，含两个写者竞争 |
| C Vast 后端 | 搜索、创建、状态、销毁、列举；单位和字段怪点；创建超时后按标签接管 | `backends/vast.py` | Sonnet | 模拟 HTTP，覆盖所有状态映射 |
| D AutoDL 后端 | 弹性部署 `Job`：库存、创建、状态、删除、列举、余额；元×1000 价格换算；部署名作标签 | `backends/autodl.py` | Sonnet | 同 C，假响应按官方文档字段构造 |
| E runner | bootstrap、runner、watchdog、bundle 构建；心跳和同步线程、日志上限、卡死检测、attempt 校验、收尾、自毁（Vast 调 API，AutoDL 结束进程） | `runner/`、`bundle.py` | Opus | 本地用假 rclone、假自毁跑通设计第 5 节中 runner 相关各行，含信号和超时 |
| F 编排与 CLI | `submit`、`wait`、`retry`、`cancel`、`gc`、`status`、`logs`、`pull`、`offers`、`doctor`；状态推导、条件写、换机、费用上限 | `orchestrator.py`、`cli.py` | Opus | 用测试替身跑通设计第 5 节中编排相关各行 |
| G Skill 与示例 | `SKILL.md`、YOLO 示例、使用文档 | `skills/dgpu/`、`examples/yolo/`、`docs/usage.md` | Sonnet | 文档里的命令与 CLI 一致；示例可选在本机 CPU 上冒烟 |

- **依赖**：A–G 都只依赖阶段 0 的契约，彼此不 import 对方未完成的实现。F 用 `FakeBackend` 和 `MemoryStore` 开发，阶段 2 再接真实实现。G 按设计 4.5 的命令表写，阶段 2 再核对。
- **模型**：全部工作包和评审都用 Opus（用户 2026-11-10 的决定），上表“模型”列里的 Sonnet 不再适用。

## 5. 并行开发规则（写进 `AGENTS.md`）

- 每个 agent 在独立 worktree 的 `wp/<包名>` 分支上工作，只改自己负责的文件。
- 不改 `contracts.py`、`layout.py`、`states.py`。需要改就在报告里提出，由主会话统一修改并通知相关工作包。
- 测试不访问真实云 API，不读真实凭据：HTTP 用 respx 模拟，S3 用 moto。
- 完成标准：`make check` 通过，设计第 5 节中归属本包的每一行至少有一个测试。
- 结束时提交到自己的分支，报告改了什么、测试结果、与设计不一致的地方、未解决的问题。

## 6. 阶段 P：探针

探针脚本由我写，只调用设计里列出的 API。

| 编号 | 验证什么 | 结果会改变什么 | 预计花费 |
|---|---|---|---|
| P1 | Vast 容器内用 `CONTAINER_API_KEY` 调 REST 销毁自身 | 不行就在 bundle 里带 `vastai` CLI | 约 $0.05 |
| P2 | 你的电脑到 R2 的读写 | 不通就要给 CLI 配代理 | 0 |
| P3 | Vast 实例与 R2 之间的速度 | 默认超时和磁盘估算 | 含在 P1 |
| P4 | Vast 上有没有 `/dev/fuse` | 只记录 | 含在 P1 |
| A1 | AutoDL `Job`：`cmd` 结束后的状态、计费是否停止、`DELETE` 的行为 | AutoDL 后端的状态映射和销毁逻辑 | 约 ¥2 |
| A2 | AutoDL 与 R2 之间的上下行速度，不同时段两次 | AutoDL 是否改用国内对象存储 | 含在 A1 |
| A3 | AutoDL 上有没有 `/dev/fuse` | 只记录 | 含在 A1 |
| A4 | 无库存、价格上限过低时创建接口的返回 | 换机重试逻辑 | 约 0 |
| A5 | `cmd` 长度上限；公共镜像能否运行 python3 bootstrap | bootstrap 的形式；是否需要你在 AutoDL 上保存私有镜像 | 含在 A1 |

合计预计不超过 $2 和 ¥20。探针结果与设计不符时，先改设计文档，再通知相关工作包。

## 7. 阶段 2：集成与评审

- **合并**：按 A、B → C、D → E → F → G 的顺序 rebase 合入 `feature/dgpu-mvp`，每合一个跑一次 `make check`。
- **评审**：两个 Opus agent 对照设计第 5 节做对抗式评审。
  - 一个看 runner 和 watchdog：信号、线程、收尾、自毁。
  - 一个看编排：attempt 条件写、重试幂等、`gc` 绝不碰无标签实例。
- **核实**：评审结论我逐条核实后再修改。

## 8. 风险与应对

| 风险 | 应对 |
|---|---|
| 开发中发现契约要改 | 只由主会话修改，改后通知相关 agent，在对应分支 rebase |
| 合并冲突 | 文件归属不重叠；共用的 `cli.py` 只归 F |
| agent 写出只为通过测试的代码 | 测试从设计第 5 节的失败模式出发；阶段 2 做对抗式评审 |
| AutoDL 文档没写的行为 | 先按文档实现，探针 A1、A4 后修正；探针前不在 AutoDL 上跑真实任务 |
| AutoDL 到 R2 跨境慢或不稳定 | 已预留存储 profile；不在实例内自建代理 |
| token 花费 | 7 个开发 agent + 2 个评审 agent；只有 E、F 和评审用 Opus |

## 9. 需要你提供

1. 确认本方案。
2. **Vast**：账户和 API key（建议用 `vastai create api-key` 建一个权限受限的 key），先充 $5 左右，不绑定自动扣款。
3. **R2**：一个 bucket，以及只授权这个 bucket 的 Object Read & Write token。
4. **AutoDL**：开发者 token（控制台 → 设置 → 开发者 Token）。如果公共镜像不能用（A5），需要你在 AutoDL 网站上保存一个带 python3 的私有镜像。
5. 密钥不要发到对话里，写进 `~/.config/dgpu/config.toml`，阶段 0 会生成模板。
