# dgpu

**一条命令在租来的云 GPU 上训练任意模型，用完自动销毁，不会忘了关机一直扣钱。**

[![CI](https://github.com/baabanana/dynamic_gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/baabanana/dynamic_gpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

[English](README.md) · 中文

`dgpu submit job.yaml` 在 [Vast.ai](https://vast.ai) 或 [AutoDL](https://www.autodl.com) 上租一台 GPU 机器，拷代码和数据，跑训练命令，把输出和 checkpoint 传回你自己的 [Cloudflare R2](https://developers.cloudflare.com/r2/) bucket，然后**销毁机器**。成功、失败、超时、取消、崩溃，结局都一样。

不用维护常驻的 GPU 服务器，dgpu 自己也不需要服务器。任务状态都在你的 bucket 里，换一台电脑也能查。

> [!WARNING]
> **这是会真金白银花钱的 alpha 软件。** Vast.ai 上已经用真实机器验证了一部分：机器自毁、超时、结果上传、训练进程降权（见[设计第 8 节](docs/dgpu-design.md#8-实现状态与验证记录)）。完整的成功训练、`retry`、`cancel`、`gc` 清理孤儿实例还没在 Vast 上验证。**AutoDL 后端和 YOLO 示例还完全没在真实平台上跑过。** 账户只充少量余额，不要开自动充值。见[限制](#限制)。

## 为什么做 dgpu

按小时租 GPU 很便宜，贵的是那台脚本崩了、电脑睡着了、Ctrl-C 按得不是时候之后还在跑的机器。dgpu 围绕一条规则设计：**任务的任何结局都以销毁实例结束**。创建任何东西之前，它先告诉你最坏要花多少钱。

| | |
|---|---|
| **不感知模型** | 任务就是一个 Docker 镜像加一条命令。训练代码从 `$DGPU_DATA` 读数据，往 `$DGPU_OUTPUT` 写结果，不需要接 SDK，代码里也不需要存储密钥。 |
| **多层销毁** | 机器上的 runner 在任务结束时销毁机器；runner 挂了，独立的看门狗接手；`submit` 期间 Ctrl-C、SIGTERM、SIGHUP 都会先清理；`dgpu gc` 按标签清理遗留机器；账户余额是最后一道防线。 |
| **先报最坏花费** | `submit` 创建机器前打印“每小时价格上限 × 最长计费时间”（启动 + `max_hours` + 收尾），并检查余额。 |
| **换机续跑** | `$DGPU_CKPT` 里的 checkpoint 定期同步到 R2，`dgpu retry` 换一台机器接着跑。 |
| **CLI 无状态** | 任务状态全在你的 bucket 里，本机不存数据库。配好同一个 bucket 的任何电脑看到的任务都一样。 |
| **机器上权限最小** | 平台 API key 不离开你的电脑。容器是 root 时（通常如此），训练以无特权用户运行，读不到存储令牌，也杀不掉负责销毁机器的进程。 |
| **两个平台，一份 job.yaml** | Vast.ai（美元，任意公开 Docker 镜像）和 AutoDL 弹性部署（人民币，平台镜像，需要企业认证），用 `--backend` 切换。 |
| **能配合 Claude Code** | 自带 [Skill](skills/dgpu/SKILL.md)：Claude 可以帮你检查环境、写 `job.yaml`、提交、等待、清理。 |

## 工作方式

```
你的电脑 / Claude                   你的 R2 bucket                   租来的 GPU 机器
dgpu submit ─代码、任务描述────────▶ dgpu/jobs/<任务ID>/
            ─搜机器、创建带标签的实例──────────────────────────────▶ bootstrap → runner + 看门狗
dgpu wait   ◀─状态、心跳─────────── status.json, heartbeat.json ◀── 定期同步
dgpu logs   ◀─日志────────────────── logs/, events/              ◀── 定期同步
dgpu pull   ◀─结果────────────────── output/, ckpt/              ◀── 最终同步，然后自毁
dgpu gc     ─销毁带 dgpu 标签的遗留实例───────────────────────────▶
```

机器上的 runner 只用 Python 标准库（3.8 及以上）和一个固定版本的 [rclone](https://rclone.org)。rclone 由 dgpu 在你的电脑上下载、校验 SHA-256 后打进任务包。数据在训练开始前拷到本地盘，不用 FUSE 挂载 R2：租来的容器大多没有特权，挂载也会让训练时逐个读小文件变慢。

## 环境要求

- **你的电脑**：macOS 或 Linux，Python ≥ 3.11，[uv](https://docs.astral.sh/uv/)。
- **存储**：一个 Cloudflare R2 bucket，和一个**只授权这个 bucket** 的 API 令牌（对象读和写）。其他 S3 兼容存储可以配置成 profile，但没有测试过。
- **算力**：[Vast.ai](https://vast.ai) API key，或 [AutoDL](https://www.autodl.com) 开发者 token（弹性部署 API 需要企业认证），有一个就行。
- **镜像**：Linux x86_64，里面有 `python3`（3.8 或更新）和 `bash`。Vast 上要写固定 tag，不能用 `latest`。

## 安装

```bash
git clone https://github.com/baabanana/dynamic_gpu.git
cd dynamic_gpu
uv tool install .        # 装成全局命令 dgpu
dgpu --help
```

不想安装的话，在仓库目录里用 `uv run dgpu ...`。

想让 Claude Code 帮你提交和管理任务，把 Skill 链接过去：

```bash
ln -s "$PWD/skills/dgpu" ~/.claude/skills/dgpu
```

## 快速上手

**1. 一次性配置。**

```bash
dgpu init        # 生成 ~/.config/dgpu/config.toml，权限 600
$EDITOR ~/.config/dgpu/config.toml
dgpu doctor      # 检查配置、R2 读写、平台 API、余额
```

```toml
[stores.r2]
endpoint = "https://<账户ID>.r2.cloudflarestorage.com"   # 只写到域名，不带 bucket
bucket = "my-dgpu"
access_key_id = "..."
secret_access_key = "..."

[backends.vast]
api_key = "..."
store = "r2"
```

密钥只放在配置文件或环境变量里（`VAST_API_KEY`、`AUTODL_TOKEN`、`R2_ACCESS_KEY_ID`、`R2_SECRET_ACCESS_KEY`），不要写进 `job.yaml`。

**2. 写任务描述。**

```yaml
# job.yaml
version: 1
name: my-train
backend: vast
image:
  vast: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime   # 必须是固定 tag
code: .
setup: python -m pip install -r requirements.txt
run: python train.py
data:
  - r2://my-dgpu/datasets/coco8      # 已经在 R2 上
  - ./datasets/small-set             # 本地目录：只传一次，按内容哈希去重
gpu:
  names: [RTX 3090, RTX 4090]
  count: 1
limits:
  max_hours: 4
  max_usd_per_hour: 0.40
env:
  EPOCHS: "100"
```

**3. 训练代码按约定读写。**

| 环境变量 | 含义 |
|---|---|
| `DGPU_DATA` | `data` 里的每一项在 `$DGPU_DATA/<路径最后一段>` |
| `DGPU_OUTPUT` | 结果写这里，大约每 2 分钟上传一次，结束时再传一次 |
| `DGPU_CKPT` | checkpoint 先写成 `xxx.tmp` 再 rename；`*.tmp` 不会上传 |
| `DGPU_RESUME` | 为 `1` 时说明 checkpoint 已恢复到 `$DGPU_CKPT`，应该接着训练 |
| `DGPU_JOB_ID`、`DGPU_ATTEMPT` | 任务 ID、第几次尝试 |

成功时退出码为 0。提交前先把这几个变量指向本地目录跑一遍，代码错误在本机发现不花钱。完整例子见 [`examples/yolo/train.py`](examples/yolo/train.py)。

**4. 运行。**

```bash
dgpu offers job.yaml            # 看有哪些机器、预估多少钱，不花钱
dgpu submit job.yaml            # 打印费用上限，租机器，preflight 通过后输出任务 ID
dgpu wait <任务ID>              # 等到结束，退出码表示结果
dgpu pull <任务ID> ./result     # 下载输出到 ./result/output/
dgpu gc --dry-run               # 确认没有遗留的机器在计费
```

## 命令

| 命令 | 作用 |
|---|---|
| `init` | 生成配置文件模板（权限 600），已存在时不覆盖 |
| `doctor` | 检查配置、存储读写、平台 API、余额 |
| `offers` | 列出候选机器和预估费用，不创建任何东西 |
| `submit` | 打包代码、创建机器，preflight 通过后返回任务 ID（`--wait`、`--retries`、`--yes`） |
| `wait` | 等任务结束；`--retries N` 在确认失联后换机重跑 |
| `status` / `list` / `logs` | 查看任务；`logs -f` 持续跟随 |
| `retry` | 销毁旧机器，换一台从 checkpoint 接着跑 |
| `cancel` | 先上传已有结果再销毁；`--force` 直接销毁 |
| `pull` | 下载输出（`--ckpt` 同时下载 checkpoint） |
| `gc` | 销毁遗留的 dgpu 机器。只碰标签形如 `dgpu-<任务ID>-a<次数>` 的机器 |

`dgpu wait` 的退出码：`0` 成功、`1` 失败、`3` 超时、`4` 已取消、`5` 失联、`6` 成功但清理失败（运行 `dgpu gc`）、`7` 连不上存储。完整说明见[使用说明第 6 节](docs/usage.md#6-命令一览)。

## 怎么保证停止计费

1. **正常结束**：runner 先上传结果，再销毁自己所在的机器。Vast 上用平台给这台机器单独发的 `CONTAINER_API_KEY`；AutoDL 上容器退出即释放。
2. **看门狗**：runner 崩溃或超过 `max_hours` 很久没结束时，独立的进程负责销毁。
3. **启动阶段**：机器起不来、实际价格高于上限、你中断了 `submit`，dgpu 都会销毁已经创建的机器。bootstrap 在 runner 起来之前放弃时，Vast 上的机器会自毁。
4. **`dgpu gc`**：电脑在提交中途关机、`kill -9`、自毁失败等情况留下的机器由它清理，可以用 cron 或 launchd 每 10 分钟跑一次。
5. **账户余额**：只充少量、不开自动充值。以上都失效时，损失也有上限。

在你的 bucket 里查不到对应任务的机器，`gc` 只报告、不销毁，要你确认（`--include-unknown`）。共用平台账户或用了多个 bucket 时，这一点很重要。

## 安全模型

- **平台 key**（Vast API key、AutoDL token）只留在你的电脑上，不会发到机器上，也不写进存储。
- **存储令牌会到机器上**：通过短时有效的预签名 URL 下发，不放在环境变量或启动命令里。社区机器的房东能读到磁盘，所以令牌只授权一个 bucket；数据敏感时打开 Vast 的 `datacenter_only = true`，或者不要放到租来的机器上。
- **训练代码默认无特权运行**。dgpu 没法降权时（例如容器本身不是 root），`dgpu status` 和日志会写明原因。`run_as_root: true` 让训练以 root 运行，这时它也能读到存储令牌，只对自己写的、可信的代码打开。
- 报告安全问题见 [SECURITY.md](SECURITY.md)。

## 限制

- 只支持单机（单机多卡可以），不支持多机分布式训练。
- 不支持可中断（spot）实例。目前只有 Vast.ai 和 AutoDL 两个后端；`Backend` 是协议，可以加新的。
- 本地数据目录里单个文件超过 5 GiB 时，要先用 rclone 传到 R2（dgpu 只做单段上传）。
- AutoDL：数据盘固定 50 GB；AutoDL 到 R2 走跨境网络，速度没有实测。
- 机器启动了但容器连 `python3` 都跑不起来，同时你的电脑关机、一直没跑 `gc`，这台机器会一直计费到余额耗尽。已知风险的完整列表见[设计第 9 节](docs/dgpu-design.md#9-后续风险)。

## 文档

- [使用说明](docs/usage.md)：安装、`job.yaml` 字段、数据、全部命令、常见问题
- [设计文档](docs/dgpu-design.md)：红线、状态机、销毁保证、失败模式、真实平台验证记录
- [YOLO 示例](examples/yolo/README.md)：在 coco8 上训练 YOLO26n，包括中断续跑
- [Claude Code Skill](skills/dgpu/SKILL.md)
- [文档索引](docs/README.md)

## 参与开发

欢迎贡献，issue 和 PR 用中文或英文都可以。不需要任何云账户：1000 多个测试全部离线运行，用内存和本地文件模拟存储和平台。先看 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
uv sync
make check     # ruff + pytest
```

## 许可证

[MIT](LICENSE)。dgpu **按原样提供，不做任何担保**，云账户上产生的费用由你自己负责。YOLO 示例会安装 [Ultralytics](https://github.com/ultralytics/ultralytics)，它是 AGPL-3.0 许可证，见[示例的许可证说明](examples/yolo/README.md#9-许可证)。
