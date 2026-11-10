# dgpu：一条命令在按需云 GPU 上训练任意模型，用完自动销毁

状态：设计已确认（2026-11-10）。4.12 AutoDL 后端、4.13 存储与凭据下发为新增草案，随[实施方案](dgpu-implementation-plan.md)一起确认
分支：`feature/dgpu-mvp`　关联：[平台调研笔记](research/cloud-gpu-2026-11/)、[实施方案](dgpu-implementation-plan.md)

## 1. 背景与问题

手上有小型 YOLO 模型，之后还会训练类似的模型。不想维护常驻 GPU 服务器，数据今后放 Cloudflare R2。需要一个可以直接调用的工具：提交训练任务，自动租 GPU，跑完把结果传回 R2，销毁机器停止计费。

调研结果（2026-11-09，详见关联笔记，未租用实例实测）：

| 方案 | 自动化能力 | 参考价格 | 结论 |
|---|---|---|---|
| Vast.ai | REST/CLI 完整；可按价格和可靠性搜机器；容器自带只能操作自身的 `CONTAINER_API_KEY` | 3090 按需约 $0.12/h，4090 约 $0.32/h（可靠性 ≥ 0.98 的最低价） | **第一个后端** |
| AutoDL | 普通实例无 API；Pro API 需实名，规格只有较贵的几种；弹性部署 API 需企业认证 | 4090 ¥1.88/h，3090 ¥1.32/h（首页标价，11 月 12 日起可能调整） | 用户有企业认证，用弹性部署作为**第二个后端**；到 R2 的网络待实测 |
| RunPod | 有 API，但无搜机器接口；Spot 已下线；REST v1 将于 2026-11-15 停用 | 社区 3090 $0.22/h | 不如 Vast |
| AWS EC2 Spot | 销毁机制最可靠 | g4dn.xlarge Spot 约 $0.25/h | 新账号 GPU 配额默认 0，门槛高 |
| SkyPilot | 多云比价和起停 | 取决于后端 | Vast 上禁用对象存储挂载，R2 只能拷入，输出要自己传；`--down` 在配置或同步失败时不销毁 |
| Modal | 无实例需要销毁，自带 R2 挂载 | T4 约 $0.59/h | 备选，部分细节未核实，大陆可达性未知 |

决定设计的三条证据：

- **FUSE 挂载 R2 大概率不可用。** Vast 文档写明实例是非特权容器，dstack 文档列出 Vast 和 RunPod 都不支持 privileged，RunPod 上有 `/dev/fuse: operation not permitted` 的报告；AutoDL 也是不支持 Docker 的普通容器。即使能挂载，小文件逐个经 FUSE 读取也会拖慢数据加载。
- **最大的风险是计费泄漏。** 按小时计费的机器没被销毁，费用就一直累积。
- **Ultralytics 写 `last.pt` 不是原子操作**（普通 `write_bytes`），后台同步可能传上半截文件，所以 checkpoint 需要明确的写入约定。

## 2. 目标与非目标

目标：

1. 用一个 `job.yaml` 描述任务（镜像、代码、命令、R2 数据、GPU、价格和时长上限），`dgpu submit` 之后无需人工干预。
2. 工具不感知模型：任务就是镜像加命令，路径通过环境变量传入。
3. 任务的任何结局都会销毁实例，单个任务的最坏费用有明确上限。
4. 中断后可以从 R2 上的 checkpoint 接着跑。
5. 人可以直接用 CLI，Claude 可以通过 Skill 调用（形态 CLI + Skill 已于 2026-11-10 确认）。
6. 支持 Vast 和 AutoDL 两个后端，同一份 `job.yaml` 用 `--backend` 切换（镜像和价格上限按后端分别填写）。

非目标：

- 多机分布式训练（单机可多卡）。
- FUSE 挂载 R2。改为启动时拷到本地盘，训练代码看到的同样是本地目录（用户 2026-11-09 已确认）。
- 可中断实例。Vast 的可中断实例被抢占前没有预警，小模型省下的钱有限。
- RunPod、AWS 等其他后端。保留 `Backend` 接口。
- 常驻调度服务、任务队列、多用户。
- MCP。需要时在同一核心上加一层。
- 超参搜索、实验管理、数据集打包工具（大量小文件可放 tar，在 `setup` 里解压）、存储之间的数据镜像命令。

## 3. 设计红线（不变量）

1. **不泄漏计费**：只要实例上的 runner 启动过，任务结束时实例就必须被销毁；未启动的情况由 `submit` 和 `gc` 兜底（见 4.8）。
2. **主 API key（Vast、AutoDL）永不进入实例。** Vast 容器内只用 `CONTAINER_API_KEY`，AutoDL 容器内不需要任何平台 key。
3. **`gc` 只碰带 dgpu 标签的实例**（Vast 标签 `dgpu:`，AutoDL 部署名前缀 `dgpu-`），不影响用户的其他实例。
4. **不静默丢数据**：上传失败、日志截断、重试、换机都写事件。
5. **单写者**：同一任务同时只有一个 attempt 向存储写入。
6. **存储是任务状态的唯一事实来源**，CLI 不依赖本地状态，换一台电脑也能查询。

## 4. 方案

### 4.1 总体流程

```
你的电脑 / Claude                   存储（默认 R2）                   GPU 实例（Vast / AutoDL）
dgpu submit ─打包代码、写 spec────▶ dgpu/jobs/<id>/
            ─搜机器、创建实例（带标签）──────────────────────────────▶ bootstrap
dgpu wait   ◀─读状态和心跳───────── status.json, heartbeat.json ◀─── runner + watchdog
dgpu logs   ◀─读日志─────────────── logs/, events/              ◀─── 周期同步
dgpu pull   ◀─下载结果───────────── output/, ckpt/              ◀─── 结束时最终同步，然后自毁
dgpu gc     ─按标签清理遗留实例──────────────────────────────────────▶
```

项目结构：`src/dgpu/`（CLI、`spec.py`、`store.py`、`backends/vast.py`、`backends/autodl.py`）、`src/dgpu/runner/`（只用标准库，随 bundle 下发）、`skills/dgpu/SKILL.md`。

### 4.2 任务规格（`job.yaml`）

```yaml
version: 1                       # 未知版本拒绝执行
name: yolo-demo
backend: vast                    # vast | autodl，可被 --backend 覆盖
image:
  vast: vastai/pytorch:<固定版本>  # 拒绝 latest 或无 tag
  autodl: <AutoDL 镜像 UUID>      # AutoDL 只能用平台上的镜像
code: .                          # 本地目录，见 4.5
setup: pip install ultralytics==8.4.208
run: python train.py
data:
  - r2://my-datasets/coco8       # 拷到 $DGPU_DATA/coco8
gpu:
  names: [RTX 3090, RTX 4090]
  count: 1
limits:                          # 必填，没有默认值；只需填所用后端的价格上限
  max_hours: 6
  max_usd_per_hour: 0.40         # Vast
  max_cny_per_hour: 2.00         # AutoDL
env:
  EPOCHS: "100"
```

镜像必须带 `python3`（≥ 3.8）和 `bash`，其余依赖由 bundle 提供。

### 4.3 训练代码的约定

| 环境变量 | 含义 |
|---|---|
| `DGPU_JOB_ID` / `DGPU_ATTEMPT` | 任务 ID、第几次尝试 |
| `DGPU_DATA` | 数据目录（已拷到本地） |
| `DGPU_OUTPUT` | 输出目录，周期同步到存储 |
| `DGPU_CKPT` | checkpoint 目录，周期同步到存储，重试时恢复 |
| `DGPU_RESUME` | `DGPU_CKPT` 里有恢复出来的文件时为 `1` |

- 退出码 0 为成功，其他为失败。
- **放进 `DGPU_CKPT` 的文件必须已经写完整**：先写 `*.tmp`，再 rename。runner 同步时排除 `*.tmp`。
- 存储凭据和 `CONTAINER_API_KEY` 从训练进程的环境中移除，训练代码不需要它们。

YOLO 配方（示例，未运行，随实现放入 `examples/yolo/`）：

```python
ckpt = Path(os.environ["DGPU_CKPT"]) / "last.pt"
def save_ckpt(trainer):
    tmp = ckpt.with_suffix(".pt.tmp"); shutil.copyfile(trainer.last, tmp); os.replace(tmp, ckpt)

model = YOLO(ckpt if os.environ.get("DGPU_RESUME") == "1" else "yolo26n.pt")
model.add_callback("on_model_save", save_ckpt)
model.train(data=f"{os.environ['DGPU_DATA']}/coco8/data.yaml", epochs=100, exist_ok=True,
            project=os.environ["DGPU_OUTPUT"], name="run", resume=os.environ.get("DGPU_RESUME") == "1")
```

### 4.4 存储布局与状态机

```
<bucket>/dgpu/tools/bundle-<sha256>.tar.gz     # runner + 固定版本的 rclone（linux/amd64），只传一次
<bucket>/dgpu/jobs/<job_id>/
  spec.json   code.tar.gz
  attempt.json      # 当前 attempt、实例 ID、cancel 标志；If-Match 条件写
  secrets-<n>.json  # 第 n 次尝试的存储凭据，见 4.13；第一次心跳后删除
  status.json       # 阶段、退出码、失败原因
  heartbeat.json    # 时间、attempt、阶段、GPU 利用率、最近日志时间、已同步字节
  events/<ts>-<source>-<rand>.json
  logs/attempt-<n>.log
  output/  ckpt/
```

状态：`PROVISIONING → STARTING → RUNNING → UPLOADING →` `SUCCEEDED | FAILED(原因) | CANCELLED | TIMED_OUT`。`LOST` 不写入，由 CLI 在心跳过期时推导。runner 启动前由 CLI 写状态，启动后由 runner 写。

事件名固定：`offer_rejected`、`create_failed`、`startup_timeout`、`runner_timeout`、`preflight_failed`、`data_sync_retry`、`ckpt_restored`、`sync_failed`、`log_truncated`、`stalled`、`deadline_hit`、`final_sync_failed`、`self_destroy_failed`、`attempt_fenced`、`gc_destroyed`、`instance_destroyed`。

### 4.5 CLI（`src/dgpu/cli.py`）

| 命令 | 作用 |
|---|---|
| `dgpu doctor` | 检查平台 API、存储读写、账户余额 |
| `dgpu offers job.yaml` | 列出候选机器和预估费用，不花钱 |
| `dgpu submit job.yaml [--backend vast\|autodl]` | 打包、创建实例，等到第一次心跳后返回 job ID |
| `dgpu wait <job> [--retries N]` | 前台等到结束，退出码反映结果；`LOST` 时自动重试 N 次 |
| `dgpu status` / `dgpu logs [-f]` / `dgpu list` | 查询 |
| `dgpu retry <job>` | 确认旧实例已销毁，attempt + 1，换机重跑并恢复 checkpoint |
| `dgpu cancel <job> [--force]` | 默认由 runner 收尾；`--force` 直接销毁 |
| `dgpu pull <job> [dir]` | 下载 `output/` |
| `dgpu gc [--dry-run]` | 清理两个平台上的遗留实例 |

代码打包：git 仓库用 `git ls-files -co --exclude-standard`，否则按 `.dgpuignore`。始终排除 `.git`、`.env*`、`*.pem`、`*.key`。超过 200 MB 报错，提示把数据放存储。

`submit` 前打印费用上限：`每小时价格上限 × max_hours + 流量估算`，并检查余额够覆盖上限。

### 4.6 Vast 后端（`src/dgpu/backends/vast.py`）

```python
class Backend(Protocol):
    def search(self, req: GpuRequest) -> list[Offer]: ...
    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef: ...
    def get(self, ref: InstanceRef) -> InstanceState: ...
    def destroy(self, ref: InstanceRef) -> None: ...     # 幂等
    def list(self, label_prefix: str) -> list[InstanceState]: ...
```

- 搜索：`POST /api/v0/bundles/`，默认 `verified`、`rentable`、`reliability ≥ 0.98`、`inet_down ≥ 500`、`disk_space ≥ 所需`、`duration ≥ max_hours`、流量单价 ≤ $0.01/GB。按“算力 + 磁盘 + 流量”的预估总价排序。
- 创建：`PUT /api/v0/asks/{id}/`，`runtype=ssh`、`cancel_unavail=true`、`label=dgpu:<job_id>:<attempt>`。实例 ID 取响应的 `new_contract`。
- 轮询 `actual_status`：`loading` 超 15 分钟，或进入 `exited`/`unknown`/`offline` 时销毁并换机。
- 已知怪点集中在这个文件里处理：`gpu_ram` 在 REST 中单位是 MB；`onstart` 上限 4048 字符。

创建请求超时时，先按标签查询，接管已有实例，避免重复创建。

### 4.7 容器内 runner（`src/dgpu/runner/`，只用 Python 标准库，兼容 3.8）

启动命令（Vast 的 `onstart`、AutoDL 的 `cmd`）只放 bootstrap：用 `python3` 通过预签名 URL 下载 bundle 和本次尝试的 `secrets-<n>.json`，校验 sha256，解压，启动 watchdog 和 runner。Vast 的 `onstart` 在每次容器启动时都会运行，所以 bootstrap 和 runner 必须幂等（`.ready` 标记）。

runner 的步骤：

1. **Preflight**：`nvidia-smi`、磁盘余量、能否读写存储。失败则销毁，由 `submit` 换机。
2. **拷代码和数据**：`rclone copy`，失败重试 3 次，完成后写 `.ready`。
3. **恢复** `ckpt/`，设置 `DGPU_RESUME`。
4. **`setup`，然后 `run`**：在独立进程组运行，输出由线程持续读取并写入日志（上限 200 MB，超出后继续读取但不再落盘，并写 `log_truncated`）。
5. **后台**：每 60 秒写心跳并读 `attempt.json`（发现 attempt 变大或 `cancel` 就停止）；每 120 秒同步 `output/`、`ckpt/`、日志。心跳和同步在不同线程，互不阻塞。
6. **卡死检测**：GPU 利用率为 0 且日志 60 分钟没有新内容，写 `stalled` 并终止。可配置，0 表示关闭。
7. **收尾**：最终同步（失败则重试，最长 20 分钟），写最终状态，自毁。

runner 以 `RCLONE_CONFIG_*` 环境变量把存储凭据交给 rclone，不传给训练进程。

### 4.8 销毁的多层保证

| 层 | 位置 | 触发 | 覆盖的情况 |
|---|---|---|---|
| 1 | runner | 所有退出路径 | 正常结束、失败、超时、取消 |
| 2 | watchdog（独立进程，用单调时钟） | runner 进程消失；或到达 `max_hours` + 收尾时间 + 10 分钟 | runner 崩溃、收尾卡住 |
| 3 | `submit` 前台 | 启动超时、心跳未出现、Ctrl-C | 镜像坏、runner 没跑起来 |
| 4 | `dgpu gc` | 任务已结束、超过截止时间、心跳过期 30 分钟、启动卡住 | 电脑中途关机、自毁 API 失败 |
| 5 | 账户设置 | 余额耗尽时实例被停止 | 以上全失效时的最终上限 |

Vast 上第 1、2 层的“销毁”是用 `CONTAINER_API_KEY` 调 API 销毁自身。AutoDL 上容器只在 `cmd` 运行期间存在，第 1、2 层的“销毁”就是结束 runner 进程树，CLI 在任务终态后再删除部署记录。

建议：用 launchd 每 10 分钟跑一次 `dgpu gc`（可选，电脑休眠时不运行）；两个平台都只充少量余额，不绑定自动扣款的卡。

### 4.9 Skill（`skills/dgpu/SKILL.md`）

放在本仓库，软链到 `~/myagent/skills/dgpu`，并链接到 `~/.claude/skills/dgpu`。内容包括：如何写 `job.yaml`、训练代码约定、YOLO 配方、先 `offers` 报价、`submit` 后在后台跑 `dgpu wait`（结束时自动通知）、结束后跑 `dgpu gc --dry-run` 确认无遗留。

### 4.10 配置

凭据放在 `~/.config/dgpu/config.toml`（权限 600）或环境变量：`VAST_API_KEY`、`AUTODL_TOKEN`、存储 profile 的 endpoint 与 key。存储 token 只授权所需的 bucket。

```toml
[stores.r2]
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
bucket = "<bucket>"
[backends.vast]
store = "r2"
[backends.autodl]
store = "r2"        # 探针 A2 之后可能改为国内对象存储 profile
```

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `vast.filters.reliability` | `0.98` | Vast 可靠性下限 |
| `vast.filters.min_inet_down_mbps` | `500` | 下载带宽下限 |
| `vast.filters.max_bandwidth_usd_per_gb` | `0.01` | 流量单价上限 |
| `vast.filters.datacenter_only` | `false` | 只用数据中心机器，敏感数据时打开 |
| `autodl.dc_list` | 不限 | AutoDL 地区，如 `westDC2` |
| `disk_gb` | `max(40, 2×数据量+20)` | Vast 创建后不可改；AutoDL 数据盘固定 50 GB |
| `startup_timeout_min` | `15` | 镜像拉取或调度超时 |
| `runner_timeout_min` | `5` | 实例运行后等第一次心跳 |
| `max_offers` | `5` | 启动失败最多换几台；连续 2 次 `runner_timeout` 就停止，提示检查镜像 |
| `sync_interval_s` | `120` | 周期同步 |
| `heartbeat_interval_s` | `60` | 心跳 |
| `stall_timeout_min` | `60` | 卡死判定，0 关闭 |
| `final_sync_timeout_min` | `20` | 最终同步重试时长 |
| `lost_after_min` | `10` | 心跳过期多久判 `LOST` |
| `gc_stale_min` | `30` | 心跳过期多久由 `gc` 销毁 |

### 4.11 兼容与迁移

新项目，无迁移。`job.yaml` 和存储中的 JSON 都带 `version`，遇到未知版本拒绝执行。

### 4.12 AutoDL 后端（`src/dgpu/backends/autodl.py`，新增草案）

用弹性部署 API 的 `Job` 类型（需企业认证，用户已有），Base URL `https://api.autodl.com`，开发者 token 放 `Authorization` 头：

- **搜索**：`POST /api/v1/dev/machine/region/gpu_stock` 查各地区空闲卡数，按 `gpu.names` 和 `autodl.dc_list` 过滤。文档提醒多卡任务即使显示有库存也可能调度不上。
- **创建**：`POST /api/v1/dev/deployment`，`deployment_type=Job`。`container_template` 填 `dc_list`、`gpu_name_set`、`gpu_num`、`cuda_v_from`、`price_from`/`price_to`（单位“元×1000”）、`image_uuid`、`cmd`（bootstrap）。部署名为 `dgpu-<job_id>-<attempt>`，作为标签使用。
- **状态**：`POST /api/v1/dev/deployment/container/list` 和容器事件列表，映射到统一的 `InstanceState`。
- **销毁**：`DELETE /api/v1/dev/deployment`（运行中会先停后删），幂等。
- **余额**：`POST /api/v1/dev/wallet/balance`（数值除以 1000 为元），供 `doctor` 和提交前检查。

与 Vast 的差异和限制：

- 镜像只能用 AutoDL 平台上的镜像（公共基础镜像或在网站上保存的私有镜像），不能直接用 Docker Hub 镜像。公共镜像较旧（如 CUDA 11.8、Python 3.8、PyTorch 2.0），所以 runner 保持兼容 Python 3.8。
- 数据盘固定 50 GB，提交前检查数据量。
- 价格单位是人民币，用 `max_cny_per_hour`。
- 没有环境变量字段，任务参数和凭据经 4.13 的预签名 URL 下发。
- 文档没写的行为要靠探针确认：无库存时创建是报错还是排队、状态字符串、计费粒度、`cmd` 长度上限（A1、A4、A5）。

### 4.13 存储 profile 与凭据下发（新增草案）

- **`Store` 是 S3 兼容端点的 profile**，默认 `r2`，每个后端可以指定自己的默认 profile。AutoDL 先用 R2；如果探针 A2 显示 AutoDL 到 R2 太慢或不稳定，就给 AutoDL 配同地域的国内对象存储（阿里云 OSS、腾讯云 COS 都兼容 S3），数据集需要先手动用 rclone 镜像过去。不在 AutoDL 实例内自建代理，AutoDL 协议禁止用专线或 VPN 开展跨境经营活动。
- **凭据不放进环境变量或启动命令**，因为 Vast 控制台和 AutoDL 部署列表都能看到它们。CLI 为每次尝试写一个 `secrets-<n>.json`，只通过有效期较短（启动超时 + 1 小时）的预签名 URL 下发，第一次心跳后由 CLI 删除。
- 这只能避免凭据出现在平台的控制台和 API 列表里。凭据和数据仍会落到租用机器的磁盘上，防不住房东，见第 9 节。

## 5. 失败模式与预期行为

| 故障或情况 | 预期行为 | 记录 |
|---|---|---|
| 没有符合条件的机器 | `submit` 失败，说明卡在哪个条件，不创建实例 | CLI 输出 |
| 机器被别人抢走、创建失败（含 AutoDL 无库存） | 换下一台或下一个地区，最多 `max_offers` 次 | `create_failed` |
| 创建请求超时，结果未知 | 按标签查询，接管或销毁，不重复创建 | `create_failed` |
| 镜像拉取卡住、AutoDL 部署迟迟不调度 | 超 15 分钟销毁或删除部署，换机 | `startup_timeout` |
| 镜像缺 python3、bootstrap 失败 | 等不到心跳，销毁；连续 2 次后停止并提示 | `runner_timeout` |
| 无 GPU、驱动异常 | 销毁，换机 | `preflight_failed` |
| 实例连不上存储 | 销毁，提示检查网络或换存储 profile | `preflight_failed` |
| 磁盘不足 | 提交前和下载前检查；训练中 ENOSPC 走失败收尾 | `preflight_failed` 或 `FAILED` |
| 下载数据失败 | 重试 3 次，之后 `FAILED(data_sync)`，销毁 | `data_sync_retry` |
| `setup` 失败、训练非 0 退出 | `FAILED`，上传日志，销毁，不自动重试（多半是代码问题） | `status.json` |
| 训练卡死 | 终止，`FAILED(stalled)`，销毁 | `stalled` |
| 超过 `max_hours` | SIGTERM，宽限 60 秒后 SIGKILL，收尾，`TIMED_OUT`，销毁 | `deadline_hit` |
| runner 崩溃 | watchdog 尽力同步一次，写 `FAILED(runner_crashed)`，销毁 | `status.json` |
| 运行中存储断连 | 继续训练，恢复后补传 | `sync_failed` |
| 最终上传失败 | 重试到时限；仍失败则写明未上传的文件，`FAILED(final_sync)`，销毁，本地数据随之丢失 | `final_sync_failed` |
| 宿主机宕机 | 心跳过期显示 `LOST`；`retry` 或 `wait --retries` 换机续跑 | `attempt_fenced`、`gc_destroyed` |
| 自毁 API 失败 | 每 30 秒重试，共 10 次；之后由 `gc` 清理 | `self_destroy_failed` |
| 启动途中电脑关机或断网 | 实例由 `gc` 清理 | `gc_destroyed` |
| 旧 attempt 没死透，新 attempt 已启动 | 重试前先确认旧实例销毁；旧 runner 读到更大的 attempt 后自行退出 | `attempt_fenced` |
| 两个 CLI 同时 `retry` | `attempt.json` 条件写，只有一个成功 | CLI 报错 |
| 账户余额耗尽 | 实例被平台停止，心跳过期显示 `LOST`；提交前已检查余额 | `status` 显示 |
| 日志过大 | 继续读取防止阻塞，不再落盘 | `log_truncated` |
| 重复 `submit` | 生成两个独立任务，属于预期行为 | 不同 job ID |

## 6. 验收标准

本地可验证（`FakeBackend` + 假 rclone + `moto` 模拟 S3，无需云账户）：

- `uv run pytest` 全部通过，第 5 节每行至少一个测试。
- `uv run ruff check` 通过。

必须在真实环境验证（预算约 $2 + ¥20）：

- **P1 自毁**：Vast 容器内用 `CONTAINER_API_KEY` 调 REST 销毁自身。文档只写了 CLI 路径，失败则改用 `vastai` CLI。
- **P2 电脑到 R2**：`dgpu doctor` 用真实账号 endpoint 读写。2026-11-09 的探测只确认 TCP 通，TLS 结果用假账号 ID 无法下结论。
- **P3 Vast 实例到 R2**：测下载和上传速度。
- **P4 探针**：Vast 上 `ls -l /dev/fuse`、`capsh --print`，记录结果。
- **A1 AutoDL Job 生命周期**：`cmd` 结束后的状态变化、计费是否停止、`DELETE` 的行为。
- **A2 AutoDL 到 R2**：上下行速度，不同时段各测一次，决定是否启用国内存储 profile。
- **A3**：AutoDL 上 `/dev/fuse`，记录结果。
- **A4**：无库存、价格上限过低时创建接口的返回。
- **A5**：`cmd` 长度上限；公共镜像能否运行 python3 bootstrap。
- **E1 端到端**：YOLO26n + coco8，结果出现在存储里，平台上查不到遗留实例。Vast、AutoDL 各一次。
- **E2 恢复**：运行中强制销毁实例，`dgpu retry` 从 checkpoint 继续，epoch 不归零。
- **E3 超时**：`max_hours: 0.05` + `sleep`，状态 `TIMED_OUT`，实例消失。Vast、AutoDL 各一次。
- **E4 坏镜像**：换机后干净失败，没有遗留。
- **E5 gc**：手动留一个带标签的孤儿实例，`gc` 清掉；同时存在的无标签实例不受影响。

## 7. 实施步骤

按[实施方案](dgpu-implementation-plan.md)分阶段进行：骨架与契约 → 并行开发与云上探针 → 集成与评审 → 真实环境验收 → 合入。

## 8. 实现状态与验证记录

- [x] 设计确认（2026-11-10；4.12、4.13 随实施方案确认）
- [ ] 阶段 0：骨架与契约
- [ ] 探针 P1–P4、A1–A5
- [ ] 工作包 A–G
- [ ] 集成与评审
- [ ] 真实环境验证（Vast E1–E5，AutoDL E1、E3）

## 9. 后续风险

- **凭据暴露给房东**：凭据和训练数据都会落到租用机器的磁盘上，社区机器的房东可以读到。v1 用只授权相关 bucket 的专用 token，敏感数据打开 `datacenter_only`。Cloudflare 可能提供带 TTL 的临时凭据，尚未核实，后续可替换。
- **遗留实例的残余风险**：实例启动了但 runner 没起来，同时电脑关机且一直没跑 `gc`，会一直计费到余额耗尽。需确认 Vast 是否有服务端定时销毁，有的话加一层。
- **AutoDL 跨境网络**：没有 AutoDL 到 R2 的实测，可能需要国内对象存储，届时数据要多维护一份。
- **AutoDL 文档空白**：无库存行为、状态字符串、计费粒度都要靠探针确认，探针前不在 AutoDL 上跑真实任务。
- **房东质量差异**：可靠性分数是 Vast 自己的统计，没有公开定义。
- **流量费**：R2 不收出口费，但 Vast 房东会收，已通过单价上限过滤。
- **拉镜像耗时**：自定义镜像可能要 10–60 分钟，默认用 Vast 预缓存的基础镜像。
- **依赖变动**：Ultralytics 发版频繁，配方里固定版本。
- **许可证**：Ultralytics 的官方立场是训练出的模型默认适用 AGPL-3.0，内部和商业使用需要企业许可（非法律意见）。如用于商业产品需要先确认。
