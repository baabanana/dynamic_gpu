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

1. **不泄漏计费**：只要实例上的 runner 启动过，任务结束时实例就必须被销毁；runner 没起来的情况由 bootstrap 的自毁（Vast）、`submit` 和 `gc` 兜底（见 4.8）。
2. **主 API key（Vast、AutoDL）永不进入实例。** Vast 容器内只用 `CONTAINER_API_KEY`，AutoDL 容器内不需要任何平台 key。
3. **`gc` 只碰带 dgpu 标签的实例**（标签 `dgpu-<job_id>-a<attempt>`，Vast 写在 label 字段，AutoDL 用作部署名），不影响用户的其他实例。
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
  - ./datasets/my-small-set      # 本地目录：submit 时先传到存储，再由实例拉取，见 4.5
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
- 存储凭据和 `CONTAINER_API_KEY` 从训练进程的环境中移除，训练代码不需要它们。训练命令用 `bash -lc` 运行（有的镜像靠 profile 配置 conda 等环境）；profile 执行之后 dgpu 再清除一次凭据类变量，把 profile 删掉的 PATH 条目补回末尾，重新导出 `DGPU_*`。
- **降权**（评审 B4，2026-11-10 用户确认）：容器里 runner 是 root 时（Vast、AutoDL 默认如此），`setup` 和 `run` 默认以无特权用户 `dgpu` 运行，`HOME` 是 `<工作目录>/home`。用户已存在就沿用，否则用 `useradd` 创建，没有 `useradd` 时取一个空闲的固定高位 uid/gid（不需要 passwd 条目）。这时训练进程读不到存储凭据（`launch.json`、`run/` 只有 root 能读；root 进程的 `/proc/<pid>/environ` 也读不到），也不能向 runner、watchdog、rclone 发信号；它能写的只有 `DGPU_OUTPUT`、`DGPU_CKPT`、`DGPU_DATA`、代码目录、`HOME` 和 `/tmp`。
  - `pip install x` 照常可用：site-packages 不可写（root 拥有的 conda 或系统 Python）时设 `PIP_USER=1`、`PYTHONUSERBASE=$HOME/.local`，包装到用户目录，`run` 里直接能 import。`conda install`、`apt-get` 需要 root。
  - **下列情况不降权，训练进程和 runner 同一用户，能读到凭据**：runner 本身不是 root；spec 的 `run_as_root` 为 true（`job.yaml` 的同名字段）；建不了用户、GPU 设备文件对其他用户不可读写、改不了目录属主或权限、换身份失败、这个用户跑不了 `python3`、Python 在 root 拥有且禁用用户 site 的 venv 里（pip 没有可写的安装位置）、容器里的 root 缺少降权依赖的能力（`capability_missing`，见下）。每种情况在 attempt 日志里记一行“训练进程没有降权（原因 …）”，写 `run_as_root` 事件（`data.reason` 是原因代码，`data.reason_text` 是一句话说明，`data.detail` 是细节），并写进心跳的 `identity`，`dgpu status` 显示为“训练进程：”一行（评审 F7）。
  - **降权前的能力探针**（评审 F8）：真的以训练用户起一个 `python3` 子进程，让它写一个只有自己可读（0600）的文件；runner 读这个文件（缺 `CAP_DAC_OVERRIDE` / `CAP_DAC_READ_SEARCH` 时读不了，也就同步不了训练写的结果），再给它发信号 0 和 SIGKILL（缺 `CAP_KILL` 时发不了，也就停不了训练）。任何一步不行就不降权，原因 `capability_missing`。
  - **属主只改一次**：`code/`、`data/`、`output/`、`ckpt/`、`home/` chown 完后在 `run/owner-<目录>.json` 记下 uid、gid；之后（容器重启）标记和目录本身的属主都对得上就不再逐个文件遍历。runner 以 root 往目录里写东西之前（下载代码、数据，恢复 checkpoint）删掉对应的标记；不降权时删掉全部标记。chown 每 256 个条目检查一次取消和截止，大数据集上也不会拖住停止。
  - 换身份本身（setuid 之后真的读不到凭据、发不了信号，GPU 可用，`useradd` 和 `pip --user` 在镜像里可用）只在本地用注入的系统调用测试过，要在真实实例上确认（第 9 节）。

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
<bucket>/dgpu/datasets/<内容哈希>/              # 从本地路径上传的数据，.complete 标记最后写
<bucket>/dgpu/jobs/<job_id>/
  spec.json   code.tar.gz
  request.json      # GPU 需求（型号、卡数、价格上限、时长、磁盘、镜像），retry 换机时重新搜索用
  attempt.json      # 当前 attempt、实例 ID、cancel 标志、截止时间、writer、suspect_labels；If-Match 条件写
  secrets-<n>.json  # 第 n 次尝试的存储凭据，见 4.13；preflight 通过（或这次尝试结束）后删除
  status.json       # 阶段、退出码、失败原因
  heartbeat.json    # 时间、attempt、阶段、GPU 利用率、最近日志时间、已同步字节、训练进程身份（identity）
  events/<ts>-<source>-<rand>.json
  logs/attempt-<n>.log
  logs/attempt-<n>.{runner,watchdog,bootstrap}.log   # 实例上各进程诊断日志的最后 256 KB
  output/  ckpt/
```

状态：`PROVISIONING → STARTING → RUNNING → UPLOADING →` `SUCCEEDED | FAILED(原因) | CANCELLED | TIMED_OUT`。`LOST` 不写入，由 CLI 在心跳过期时推导。runner 启动前由 CLI 写状态，启动后由 runner 写。runner 的最终状态是它自毁前最后一次写存储：日志和事件在它之前传完，CLI 看到终态就可以销毁实例。

`attempt.json` 的 `suspect_labels` 是创建结果未知、按标签查满窗口也没找到的标签（实例可能晚到，见 4.6），`status` 显示，`wait` 收尾和 `gc` 销毁晚到实例后去掉。`writer` 是每次写入换一个的随机标记。条件写其实成功、但响应丢失时，存储层重试会收到“条件不满足”；CLI 重新读取，存储里正是自己这次要写的文档（含 writer）就算成功，不会把 attempt 加两次。

失败原因：`no_offer`、`startup`、`runner_timeout`、`preflight`、`data_sync`、`setup`、`exit_code`、`stalled`、`runner_crashed`（runner 进程消失，watchdog 收尾）、`signalled`（runner 收到 SIGTERM/SIGINT，自己提前收尾）、`final_sync`、`fenced`。

事件名固定：`offer_rejected`、`create_failed`、`startup_timeout`、`runner_timeout`、`preflight_failed`、`data_sync_retry`、`ckpt_restored`、`sync_failed`、`log_truncated`、`stalled`、`deadline_hit`、`final_sync_failed`、`self_destroy_failed`、`attempt_fenced`、`gc_destroyed`、`instance_destroyed`、`dataset_uploaded`、`destroy_failed`、`relaunch`、`run_as_root`。其中 `relaunch` 由 CLI 在换机或重试把 attempt 加 1 时写，`attempt_fenced` 由 runner / watchdog 在发现更新的 attempt 时写；`destroy_failed` 是 CLI 或 `gc` 销毁实例失败、或销毁后没能确认实例消失（实例可能仍在计费）；`run_as_root` 是 runner 记下的“训练进程没有降权”（4.3）。

### 4.5 CLI（`src/dgpu/cli.py`）

| 命令 | 作用 |
|---|---|
| `dgpu doctor` | 检查平台 API、存储读写、账户余额 |
| `dgpu offers job.yaml` | 列出候选机器和预估费用，不花钱 |
| `dgpu submit job.yaml [--backend vast\|autodl] [--yes]` | 打包、创建实例，等到 runner 通过 preflight 后返回 job ID |
| `dgpu wait <job> [--retries N]` | 前台等到结束，退出码反映结果；确认失联（`LOST`）时自动重试 N 次，有未完成的取消请求时不重试 |
| `dgpu status` / `dgpu logs [-f]` / `dgpu list` | 查询 |
| `dgpu retry <job>` | 确认旧实例已销毁，attempt + 1，换机重跑并恢复 checkpoint；取消请求未完成时拒绝 |
| `dgpu cancel <job> [--force]` | 默认由 runner 收尾；`--force` 直接销毁 |
| `dgpu pull <job> [dir]` | 下载 `output/` |
| `dgpu gc [--dry-run] [--include-unknown [--yes]]` | 清理两个平台上的遗留实例；存储里查不到任务的实例默认只报告 |

代码打包：git 仓库用 `git ls-files -co --exclude-standard`，否则按 `.dgpuignore`。始终排除 `.git`、`.env*`、`*.pem`、`*.key`。超过 200 MB 报错，提示把数据放存储。代码在花钱和写存储之前先在本地打包；跳过的条目（指向代码目录之外或不存在的符号链接、设备和管道）逐个列出，其中坏掉或指向目录外的符号链接默认让 `submit` 报错（训练代码用到时会在实例上失败），`--yes` 放行。

本地数据路径（用户 2026-11-10 确认）：`data:` 里的本地目录在 `submit` 时上传到 `dgpu/datasets/<内容哈希>/`，实例再从那里拉取，拷到 `$DGPU_DATA/<目录名>`。内容哈希由相对路径、大小和文件 sha256 算出，已存在 `.complete` 标记时不重传。超过 5 GB 时先打印上传量并要求确认（`--yes` 跳过）。不做电脑到实例的 SSH 直传。位于代码目录里的本地数据目录不再打进代码包。本地目录只遍历一次（跟随符号链接，规则和上传相同）：大小估算、5 GB 确认、磁盘估算、AutoDL 50 GB 检查都用这份清单，上传时按同一份清单算哈希，清单之后有文件变了就拒绝上传。

单个文件最大 5 GiB：存储只做单段上传（R2 的分段上传 ETag 不可靠，条件写和完整性都依赖单段 ETag），本地数据里有超过 5 GiB 的文件时，`submit` 在上传任何东西之前报错，提示用户先用 rclone 传到 bucket，再用 `r2://` 引用。目录名（`$DGPU_DATA` 下的名字）只能用 `[A-Za-z0-9._-]`。`r2://` 引用的 bucket 不是存储 profile 的 bucket 时，`offers` 和 `submit` 提醒用户确认 token 能读它（实例用同一个 token），不拒绝。

`submit` 前打印费用上限：`每小时价格上限 ×（启动窗口 + 实例存活时限）`，实例存活时限就是 `runner/util.py` 的 `instance_lifetime_s`（`max_hours` + runner 最坏收尾时间 + watchdog 余量和收尾），默认配置下 2 小时的任务约按 3.6 小时算（按当前公式和默认值算出，见 4.8 的公式）；不含流量费。另外说明每台启动失败而被换掉的机器最多再花“价格 × 启动窗口”，每次重试再算一次上限。余额要够覆盖上限；平台给不出余额（如 Vast 的 key 没有读取账户信息的权限）时提示“未检查余额”后继续。

失联的判断：心跳“多久没更新”按存储的时钟算（`Store.get_json_meta`：对象的 `LastModified` 和这次响应的 `Date` 头；存储不给 `Date` 时写一个探针对象读它的 `LastModified`），不比较电脑和租来的实例的时钟。心跳过期超过 `lost_after_min` 时 `status` 显示 `LOST`；但心跳过期不等于机器挂了（runner 可能在最终上传，或暂时连不上存储而继续训练），所以 `wait` 先查平台：平台显示实例已退出、消失或状态未知，或者心跳停的时间超过 runner 最坏收尾时间（`teardown_worst_case_s`，按 4.8 的公式和当前默认值约 37 分钟），才确认失联（退出码 5；`--retries` 时先销毁 attempt ≤ 失联 attempt 的实例，不管换机成不成功，销毁失败就提示 `gc`，然后换机重试，评审 F5）；平台显示仍在运行时只报告“失联（实例仍在运行）”，不销毁任何东西，继续等。本 attempt 从没有过心跳（启动阶段超时）直接算失联。

取消优先：`attempt.json` 有取消标志、而当前 attempt 还没有终态时（取消还没完成），`wait --retries` 不自动重试，`retry` 拒绝并提示先 `cancel --force`；任务已结束（包括 `CANCELLED`）时，显式的 `retry` 清除取消标志。`retry` 在检查之后、推进 attempt 之前发现有人刚取消，同样放弃。

信号：`submit`、`retry`、`wait --retries`、`cancel --force` 收到 Ctrl-C、SIGTERM、SIGHUP 都走同一条中止路径：销毁本进程创建（或可能创建）的实例，写 `CANCELLED`，删启动文档；`cancel --force` 先把销毁做完。`attempt.json`、`status.json` 一建好就在中止路径之内（评审 F4），之后的信号或异常都写终态（`CANCELLED` 或 `FAILED`），不会留下永远 `PROVISIONING`、之后被当作 `LOST` 而被 `wait --retries` 开机器的任务。退出码 130（Ctrl-C）、143（SIGTERM）、129（SIGHUP）。清理期间再来的 SIGTERM / SIGHUP 不打断清理；SIGKILL 无法处理，留下的实例由 `gc` 清理。

`wait` 的退出码：0 成功、1 失败、3 超时、4 取消、5 确认失联、6 任务成功但结束后清理遗留实例失败（实例可能仍在计费，问题已大声输出并写了 `destroy_failed`）、7 连续 10 分钟读不到存储（瞬时错误按 5 秒起、最长 60 秒的间隔退避重试，任务本身不受影响但结果未知）、2 用法或环境错误。任务本身失败时，清理问题照样大声输出，退出码仍是任务的结果。`retry` 的新 attempt 在第一次心跳前就结束时，退出码同 `wait`。

并发安全：每个销毁决定都带着它依据的 attempt（“attempt N 已结束”“attempt N 要被替换”），只销毁标签里 attempt ≤ N 的实例；`wait` 看到终态后若 `attempt.json` 已经前进（并发的 `retry` 启动了新 attempt），跳过收尾。所以 `wait` 的收尾不会销毁并发 `retry` 刚开的机器，两个并发 `retry` 中输的一方也不会销毁赢的一方的机器。

### 4.6 Vast 后端（`src/dgpu/backends/vast.py`）

```python
class Backend(Protocol):
    def search(self, req: GpuRequest) -> list[Offer]: ...
    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef: ...
    def get(self, ref: InstanceRef) -> InstanceState: ...
    def destroy(self, ref: InstanceRef) -> None: ...     # 幂等
    def list(self, label_prefix: str) -> list[InstanceState]: ...
```

- 搜索：`POST /api/v0/bundles/`，默认 `verified`、`rentable`、`reliability ≥ 0.98`、`inet_down ≥ 500`、`disk_space ≥ 所需`、`duration ≥ max_hours`、流量单价 ≤ $0.01/GB。每小时价格 = 算力价（`dph_base`）+ 按 `disk_gb` 计的存储价，和 `max_usd_per_hour` 比较；按“算力 + 磁盘”的预估总价排序（流量用量未知，单价只展示）。
- 创建：`PUT /api/v0/asks/{id}/`，`runtype=ssh`、`cancel_unavail=true`、`label=dgpu-<job_id>-a<attempt>`。实例 ID 取响应的 `new_contract`。
- 查询和销毁：`GET` / `DELETE /api/v0/instances/{id}/`；创建时间取 `start_date`（gc 的年龄判断用）。
- 列出实例（gc、按标签查找）：`GET /api/v1/instances/`（分页，`next_token`），v1 返回 400/404/405 时退回 `GET /api/v0/instances/`。前缀必须以 `dgpu-` 开头，否则拒绝。
- 轮询 `actual_status`：`loading` 超 15 分钟，或进入 `exited`/`unknown`/`offline` 时销毁并换机。创建成功后 90 秒内查不到（404）按“等待调度”处理，不当作已销毁。
- 创建后回读：`get` 返回平台上存的 `label`，和创建时不一致（被截断或改写）就按实例 ID 销毁并停止提交，因为 `gc` 靠标签识别 dgpu 的实例；实例的实际价格（`dph_total`）高于上限时销毁、写 `offer_rejected`、换下一台。
- 销毁总是发 `DELETE`（平台查不到也发），之后用 `get` 和按标签 `list` 都确认实例消失才算完成；刚创建、平台还从没查到过的实例，查不到也观察到创建后 90 秒。
- 余额：`GET /api/v0/users/current/` 返回 401 且说明是权限不够或需要两步验证（2026-11-10 在真实 API 上看到：“Your key lacks proper privileges … Two Factor Authentication”）时，余额未知（`balance()` 返回 None，`balance_note` 写明原因，只警告一次）：`doctor` 报警告并用列出实例验证 key，`submit` 提示“未检查余额”后继续。key 无效仍然是硬错误。
- 已知怪点集中在这个文件里处理：`gpu_ram` 在 REST 中单位是 MB；`onstart` 上限 4048 字符。

创建的结果分三种（`BackendError.created`，评审 F3）：平台回答了并明确拒绝（4xx、业务错误码、限流），或者请求根本没发出（参数检查、连接没建立），是“确定没有创建”（`created=False`，`OfferUnavailable` 也是）：不查标签、不提示 `gc`；可换机的（无库存、被租走、限流）直接换下一台，attempt 不变；不可换机的（镜像名不对、key 无效）停止提交，把平台给的原因写进 `FAILED` 和输出。超时、断连、5xx、响应无法识别是“结果未知”：后端先按标签查一下（Vast 2、10 秒，AutoDL 2、5、10 秒），接管已有实例；仍没查到时，编排继续按标签查到 `create_lookup_s`（默认 120 秒，从调用 `create` 算起，后端查找的时间也算在内；时间点 0、2、5、10、20、30 秒，之后每 15 秒，用户 2026-11-10 决定）。窗口内出现就接管为这次 `create` 的结果，照常等启动，不销毁、不换机。查满窗口仍没有，才把标签记为可疑（`attempt.json` 的 `suspect_labels`），attempt + 1 换新标签，再换下一台。所以这种情况下 `submit` 最多多停顿约 2 分钟。

晚到的实例是旧 attempt：它的启动文档在这次尝试结束时已经删掉，bootstrap 下载不到（403/404 试 3 次就放弃），在 Vast 上用容器 key 自毁（4.7、4.8）；万一它的 runner 起来了，也会读到更大的 attempt 自行退出。`submit` / `retry` 返回前（成功或失败）和中止时，本次的可疑标签再查一次，找到就销毁；仍查不到就大声提示“标签为 X 的实例可能稍后出现，10 分钟后运行 `gc`”，写 `create_failed` 事件（`data.why=create_outcome_unknown`），标签留在 `attempt.json` 里：`status` 列出它们，`wait` 结束时的收尾（销毁 attempt ≤ 结束 attempt 的所有实例）和 `gc`（旧 attempt 一律销毁）在它出现后销毁它，并说明是晚到的实例。放弃这次启动（Ctrl-C、信号、出错）时 `create` 结果未知，先删掉启动文档、把标签记进 `suspect_labels`（等待中被 SIGKILL 也不留隐患），再按标签查到窗口结束（提示还要等多久，再按一次 Ctrl-C 跳过等待），找到就销毁并去掉标签；仍查不到就如实写“创建结果未知”，作废这个 attempt，提示 10 分钟后运行 `gc`，绝不声称“已销毁”。

### 4.7 容器内 runner（`src/dgpu/runner/`，只用 Python 标准库，兼容 3.8）

启动命令（Vast 的 `onstart`、AutoDL 的 `cmd`）只放 bootstrap：用 `python3` 通过预签名 URL 下载 bundle 和本次尝试的 `secrets-<n>.json`，校验 sha256，解压，在同一个进程里运行 boot，由它启动 watchdog 和 runner。Vast 的 `onstart` 在每次容器启动时都会运行，所以 bootstrap 和 runner 必须幂等（`.ready` 标记）。

bootstrap 或 boot 在 runner 起来之前放弃时（下载或校验 bundle 失败、启动文档拿不到、Python 不到 3.8 或 bundle 代码导入出错、工作目录不可用、runner 和 watchdog 都起不来），没有任何进程会收尾，所以它们自己销毁实例（评审 F1）：Vast 上用容器自己的 `CONTAINER_ID` / `CONTAINER_API_KEY`（本进程环境或 1 号进程的环境；boot 用 `runner/destroy.py`，还看 `/etc/environment`）调 `DELETE /api/v0/instances/<id>/`，有界重试（默认 10 次、每次 30 秒、间隔 30 秒），不需要存储凭据，每一步写进 `bootstrap.log` 和平台日志，key 不进日志；AutoDL 上退出即可，`cmd` 结束容器就释放。bundle 可能还没下载下来，所以 bootstrap 里是精简的内联版本（启动命令因此长了约 700 字符，载荷改用 base85；最长的任务 ID 和 bucket 名下约 3700 字符，低于 4000）。“已有 runner 在运行”（flock 被占用、`runner.alive` 被持有）的路径什么都不做，绝不自毁。容器 key 能不能这样删除实例要靠探针 P1 确认。

实例上的工作目录：环境变量 `DGPU_WORK_ROOT`；没有时，存在 AutoDL 的数据盘 `/root/autodl-tmp` 就用 `/root/autodl-tmp/dgpu`，否则 `/workspace/dgpu`。下面是 `code/`、`data/`、`output/`、`ckpt/`、`home/`（训练用户的 HOME）、`logs/`、`run/`（运行状态）。`run/`、`logs/`、`launch.json`、`bundle-*/` 只有 root 能读；降权时 `code/`、`data/`、`output/`、`ckpt/`、`home/` 归训练用户。bootstrap 只把 `flock` 的 EAGAIN 当作“已有 runner 在运行”，文件系统不支持 flock 时记一行继续，boot 和 watchdog 改看 `run/runner.pid`。

runner 的步骤：

1. **Preflight**：先写 `RUNNING` 和第一次心跳（`step=preflight`），再检查 `nvidia-smi`、磁盘余量、能否读回刚写的心跳。失败则写 `FAILED(preflight)` 并自毁，`submit` 换机；通过后立刻再写一次心跳（`step` 离开 `preflight`）。`submit` 看到心跳的 `step` 离开 `preflight` 才认为启动成功并返回。
2. **拷代码和数据**：`rclone copy`，失败重试 3 次，完成后写 `.ready`。`output/`、`ckpt/` 在启动时就建好。
3. **恢复** `ckpt/`，设置 `DGPU_RESUME`。下载下来的文件记为“已上传”（见第 7 步的同步记录）。
4. **准备训练用户**（4.3 的降权），然后 **`setup`，再 `run`**：在独立进程组运行，输出由线程持续读取并写入日志。日志上限 200 MB 只限制训练输出：超出后继续读取，只保留最后 1 MB（磁盘上两个轮换的尾部文件），收尾时附在日志末尾并写明丢弃了多少字节；runner 的说明（失败原因、退出码、最终状态）另有 1 MB 空间；写 `log_truncated`。日志始终只追加。
5. **后台**：每 60 秒写心跳并读 `attempt.json`（发现 attempt 变大或 `cancel` 就停止）；每 120 秒同步 `output/`、`ckpt/`、日志。周期同步写入之前，如果最近一次成功读 `attempt.json` 已超过 3 个心跳间隔（心跳线程卡住），先自己读一次，读不到这一轮就不写。心跳、同步、事件上传、GPU 利用率采样各在自己的线程，互不阻塞；写事件只进本地队列并唤醒上传线程，不等存储。
6. **卡死检测**（只用于 `run`）：GPU 利用率为 0 且日志 60 分钟没有新内容，写 `stalled` 并终止。利用率取读得出来的卡里最高的，某张卡报 N/A 不影响其他卡。可配置，0 表示关闭。
7. **收尾**：结论一确定就写进本地的 `run/final.json`（`synced=false`）。结束训练进程（SIGTERM，宽限 `term_grace_s` 后 SIGKILL）。本地 `output/`、`ckpt/` 里有文件时（只看磁盘：容器重启后已经超时或被取消时内存里的标志都没了；代码或数据下载失败时训练开始过但没有结果，评审 F2），先读一次 `attempt.json`，写 `UPLOADING`，做最终同步（失败则重试；`output/` 和 `ckpt/` 共用 `final_sync_timeout_min`，默认 20 分钟）。没有要传的东西的目录（不存在、是空的，或每个文件都和同步记录一致）不调用 rclone，直接算成功，存储连不上时也不会白白重试到时限。没传上去的文件按同步记录计算：每次成功同步一个目录，记下同步开始前的快照（路径、大小、mtime_ns），之后改过的文件（包括等大小改写）都算没上传；记录损坏时本地所有文件都算，目录列不出来时写“无法确定”，不会报 0。然后 `final.json` 改成 `synced=true`，上传日志（含 runner、watchdog、bootstrap 诊断日志的最后 256 KB）和事件，停心跳线程（它一直在读 `attempt.json`），必要时再确认一次 attempt，**最后**写最终状态，自毁。最终状态是自毁前最后一次写存储，CLI 看到终态就可以销毁实例而不会丢日志。收到 SIGTERM/SIGINT 时同样收尾，原因为 `signalled`。

同一 attempt 的容器重启（Vast 每次启动都执行 onstart）按 `run/final.json` 接着做：没有结论就重新运行（代码和数据按 `.ready` 不重下，从 ckpt 恢复）；结论已定但最终同步没做完，只做最终同步和后面的收尾，不重跑 `setup` 和训练；状态已写，只自毁。

子进程的等待都有上限：SIGKILL 之后最多等 `kill_wait_s`（默认 5 秒），进程仍不退出（如卡在驱动里的 `nvidia-smi`，D 状态）就放弃它。preflight 时 `nvidia-smi` 卡住算 GPU 故障（`FAILED(preflight)`，CLI 换机）；训练期间卡住则暂停 GPU 采样（利用率记为未知，不判卡死），取消和截止照常处理。

runner 以 `RCLONE_CONFIG_*` 环境变量把存储凭据交给 rclone，不传给训练进程。每个正在运行的 rclone 登记在 `run/rclone-<组件>/<pid>`，runner 被杀后 watchdog 按登记（核对命令行，防止 pid 复用）杀掉它们，免得它们在 watchdog 写状态之后才落盘。

### 4.8 销毁的多层保证

| 层 | 位置 | 触发 | 覆盖的情况 |
|---|---|---|---|
| 0 | bootstrap、boot | runner 起来之前放弃（下载、校验、启动文档、Python、导入、工作目录） | 镜像或网络有问题、创建结果未知而晚到的实例（启动文档已删）、submit 已经不在 |
| 1 | runner | 所有退出路径 | 正常结束、失败、超时、取消 |
| 2 | watchdog（独立进程，用单调时钟） | runner 进程消失；或到达硬截止 = `max_hours` + runner 最坏收尾时间 + 余量（默认 5 分钟） | runner 崩溃、收尾卡住 |
| 2' | AutoDL 前台的 boot | watchdog 意外退出（重启它，最多 5 次）；或到达 boot 自己的硬截止 = watchdog 应该收尾完的时间 + 余量 | watchdog 也失效、runner 又卡住 |
| 3 | `submit` / `retry` / `wait --retries` 前台 | 启动超时、心跳未出现、preflight 失败、标签被改写、价格高于上限、Ctrl-C / SIGTERM / SIGHUP | 镜像坏、runner 没跑起来、机器有问题、终端关闭、进程被系统停止 |
| 4 | `dgpu gc`，`dgpu wait` 结束时的收尾 | 任务已结束、更旧的 attempt（含创建结果未知、晚到的实例）、超过截止时间、心跳过期 30 分钟且平台显示实例已退出或状态未知（或心跳停满 runner 最坏收尾时间）、启动卡住；销毁后用详情和列表确认实例消失 | 电脑中途关机、submit 被 SIGKILL、自毁 API 失败 |
| 5 | 账户设置 | 余额耗尽时实例被停止 | 以上全失效时的最终上限 |

`gc` 只处理标签能解析、且任务 ID 是 `layout.new_job_id` 生成的形状（`<名字>-<12 位时间>-<4 位十六进制>`，`layout.is_generated_job_id`）的实例；手工起名的 `dgpu-x-a1` 不碰。存储里查不到对应任务的实例（可能属于另一份配置的 bucket，或共用平台账户的同事）只报告为“未知任务”（附年龄和标签），`gc --include-unknown` 确认后（`--yes` 跳过确认）才销毁。心跳过期按存储的时钟算（见 4.5）；平台显示实例仍在运行时报告“失联（实例仍在运行）”并保留。

runner 最坏收尾时间是各步骤时限之和：`term_grace_s` + 两次 `kill_wait_s`（SIGKILL 后等进程消失、回收）+ 停三个后台线程（同步、心跳、事件上传）+ `UPLOADING` 和最终同步前读 `attempt.json` + `final_sync_timeout_min` + 写日志、事件和最终状态的预算（`teardown_budget_s`，后一半留给最终状态）+ 自毁的全部重试（`runner/util.py` 的 `teardown_worst_case_s`；没上传的文件按本地同步记录计算，不访问存储）。watchdog 的硬截止总比它多出一个余量，所以 watchdog 不会杀掉还在时限内收尾的 runner。

watchdog 接手时（runner 消失，或到达硬截止时先 SIGKILL runner）：杀掉 runner 留下的 rclone 和训练进程，确认没有更新的 attempt（本地标记和存储里的 `attempt.json`），最终同步没做完时尽力同步一次（总时限 `watchdog_sync_timeout_s` 和 `final_sync_timeout_min` 取小；但这个 watchdog 启动时 runner 自己的截止已经过了——容器重启、停机时间长，runner 来不及在 watchdog 截止前做完自己的收尾——就给满 `final_sync_timeout_min`，失败重试到时限，和 runner 一样，评审 F6；这个上限计入 `watchdog_teardown_worst_case_s`），按同步记录算出没上传的文件，然后传日志和事件，最后写状态：runner 已经得出结论就用它的结论，否则崩溃写 `FAILED(runner_crashed)`、硬截止写 `TIMED_OUT`。每个存储操作都有时限，日志、事件、状态共用 `teardown_budget_s`，整个接手过程的上限是 `watchdog_teardown_worst_case_s`；自毁放在 `finally` 里。`attempt.json` 的 `deadline_at`（gc 用）= 创建时间 + 启动窗口（`startup_timeout_min + runner_timeout_min`）+ watchdog 硬截止 + watchdog 接手的上限 + 30 分钟，和 watchdog 用同一个公式（`instance_lifetime_s`）。

Vast 上第 0、1、2 层的“销毁”是用 `CONTAINER_API_KEY` 调 API 销毁自身。AutoDL 上容器只在 `cmd` 运行期间存在，第 1、2 层的“销毁”就是结束 runner 进程树，CLI 在任务终态后再删除部署记录。AutoDL 的 `cmd`（boot 留在前台）在最终状态已写进存储（或本 attempt 已被取代）时退出 0，不管任务结果如何，结果以 `status.json` 为准；一个状态都没记录时才非 0（探针 A1 确认 Job 会不会重启非 0 退出的容器之前，避免失败的任务被反复重启）。Vast 上 boot 启动后立即退出 0，不变。

建议：用 launchd 每 10 分钟跑一次 `dgpu gc`（可选，电脑休眠时不运行）；两个平台都只充少量余额，不绑定自动扣款的卡。

### 4.9 Skill（`skills/dgpu/SKILL.md`）

放在本仓库，软链到 `~/myagent/skills/dgpu`，并链接到 `~/.claude/skills/dgpu`。内容包括：如何写 `job.yaml`、训练代码约定、YOLO 配方、先 `offers` 报价、`submit` 后在后台跑 `dgpu wait`（结束时自动通知）、结束后跑 `dgpu gc --dry-run` 确认无遗留。

### 4.10 配置

凭据放在配置文件里，不写进 `job.yaml`。配置文件位置按优先级：`--config <路径>`、环境变量 `DGPU_CONFIG`、默认 `~/.config/dgpu/config.toml`。文件权限不是 600 时拒绝读取并提示。单个密钥也可以用环境变量覆盖：`VAST_API_KEY`、`AUTODL_TOKEN`、`R2_ACCESS_KEY_ID`、`R2_SECRET_ACCESS_KEY`。存储 token 只授权所需的 bucket。

```toml
[stores.r2]
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
bucket = "<bucket>"
access_key_id = "..."
secret_access_key = "..."
[backends.vast]
api_key = "..."
store = "r2"
[backends.autodl]
token = "..."
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
| `create_lookup_s` | `120` | 创建结果未知时按标签查找多久（秒，从调用 `create` 算起）再换下一台；0 表示只查一次 |
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
- **创建**：`POST /api/v1/dev/deployment`，`deployment_type=Job`。`container_template` 填 `dc_list`、`gpu_name_set`、`gpu_num`、`cuda_v_from`、`price_from`/`price_to`（单位“元×1000”）、`image_uuid`、`cmd`（bootstrap）。部署名为 `dgpu-<job_id>-a<attempt>`，作为标签使用。
- **价格口径**：`max_cny_per_hour` 是整个容器（全部卡）每小时的上限。文档没写清 `price_to` 是每卡还是每容器，目前假设每卡（`PRICE_IS_PER_GPU`），所以 `price_to` = 上限 ÷ 卡数，容器价格按“单价 × 卡数”换回整机价；探针确认后再改。库存接口不返回价格，offer 的价格填上限。
- **状态**：`POST /api/v1/dev/deployment/list`（部署记录，含 `created_at`，gc 的年龄判断用）和 `POST /api/v1/dev/deployment/container/list`（容器状态），映射到统一的 `InstanceState`。`list` 只看部署记录，只返回名称以 `dgpu-` 开头的部署。`get` 返回平台上存的部署名，编排据此确认标签没有被改写。
- **分页**：文档没写 `page_index` 从 0 还是 1 开始，不猜：第 1 页是空的而 `result_total` 不为 0 时改从 0 开始重新翻；翻完条数比 `result_total` 少时补读第 0 页（按 `uuid` 去重）。按 UUID 查单个部署同样处理。
- **工作目录**：bootstrap 发现 `/root/autodl-tmp`（AutoDL 的数据盘）就把工作目录放在 `/root/autodl-tmp/dgpu`，数据和 checkpoint 不占系统盘。
- **销毁**：`DELETE /api/v1/dev/deployment`（运行中会先停后删），幂等。部署列表里查不到时也照样发 `DELETE`（刚创建的部署可能还没出现在列表里），平台回答错误就按已删除处理；查到记录时仍然只删名称以 `dgpu-` 开头的部署。
- **余额**：`POST /api/v1/dev/wallet/balance`（数值除以 1000 为元），供 `doctor` 和提交前检查。

与 Vast 的差异和限制：

- 镜像只能用 AutoDL 平台上的镜像（公共基础镜像或在网站上保存的私有镜像），不能直接用 Docker Hub 镜像。公共镜像较旧（如 CUDA 11.8、Python 3.8、PyTorch 2.0），所以 runner 保持兼容 Python 3.8。
- 数据盘固定 50 GB，提交前检查数据量。
- 价格单位是人民币，用 `max_cny_per_hour`。
- 没有环境变量字段，任务参数和凭据经 4.13 的预签名 URL 下发。
- 文档没写的行为要靠探针确认：无库存时创建是报错还是排队、状态字符串、计费粒度、`cmd` 长度上限（A1、A4、A5）。

### 4.13 存储 profile 与凭据下发（新增草案）

- **`Store` 是 S3 兼容端点的 profile**，默认 `r2`，每个后端可以指定自己的默认 profile。AutoDL 先用 R2；如果探针 A2 显示 AutoDL 到 R2 太慢或不稳定，就给 AutoDL 配同地域的国内对象存储（阿里云 OSS、腾讯云 COS 都兼容 S3），数据集需要先手动用 rclone 镜像过去。不在 AutoDL 实例内自建代理，AutoDL 协议禁止用专线或 VPN 开展跨境经营活动。
- **凭据不放进环境变量或启动命令**，因为 Vast 控制台和 AutoDL 部署列表都能看到它们。CLI 为每次尝试写一个 `secrets-<n>.json`，只通过有效期较短（启动超时 + runner 超时 + 1 小时）的预签名 URL 下发，runner 通过 preflight 后（或这次尝试失败、结束时）由 CLI 删除。bootstrap 已经把它下载到实例的工作目录，容器重启时复用本地这份。
- 这只能避免凭据出现在平台的控制台和 API 列表里。凭据和数据仍会落到租用机器的磁盘上，防不住房东，见第 9 节。

## 5. 失败模式与预期行为

| 故障或情况 | 预期行为 | 记录 |
|---|---|---|
| 没有符合条件的机器 | `submit` 失败，说明卡在哪个条件，不创建实例 | CLI 输出 |
| 机器被别人抢走、创建失败（含 AutoDL 无库存） | 换下一台或下一个地区，最多 `max_offers` 次 | `create_failed` |
| 平台明确拒绝创建（镜像名不对、key 无效等 4xx，后端设 `created=False`） | 不按标签查找、不提示 `gc`；可换机的（无库存、限流）换下一台、attempt 不变，否则停止提交，把平台的原因写进 `FAILED` 和输出 | `create_failed`、`status.json` |
| 创建请求超时、断连、5xx，结果未知 | 按标签查到 `create_lookup_s`（默认 120 秒）：出现就接管，照常等启动；查不到才把标签记进 `suspect_labels`、attempt + 1（新标签）换下一台。返回前再查一次，找到就销毁，仍查不到就大声提示 10 分钟后 `gc`。晚到的实例拿不到启动文档，Vast 上 bootstrap 自毁；`wait` 收尾和 `gc` 也会销毁它，`status` 列出可疑标签 | `create_failed`、`relaunch`、`attempt.json` |
| 创建结果未知时放弃启动（Ctrl-C、信号、出错） | 按标签查到窗口结束（提示要等多久，再按一次 Ctrl-C 跳过）；查到就销毁，查不到就如实写“结果未知、标签为 X 的实例可能稍后出现”，记进 `suspect_labels`，作废这个 attempt，提示 10 分钟后跑 `gc`，不说“已销毁” | `relaunch`、`create_failed`、`status.json` |
| `attempt.json` / `status.json` 建好之后、第一次创建之前收到信号或出错 | 写终态 `CANCELLED`（信号）或 `FAILED`（出错），没有要销毁的；之后不会被当作 `LOST` 而被 `wait --retries` 开机器 | `status.json` |
| `submit` / `retry` / `wait --retries` 启动阶段收到 SIGTERM、SIGHUP（系统停止进程、终端关闭） | 和 Ctrl-C 一样销毁实例、写 `CANCELLED`、删启动文档，退出码 143 / 129；`cancel --force` 先把销毁做完。SIGKILL 由 `gc` 清理 | `status.json`、`instance_destroyed` |
| 刚创建的实例在平台上暂时查不到 | 创建后 90 秒内按“等待调度”处理；销毁时照样发请求，并观察到 90 秒 | — |
| 平台截断或改写了实例的标签 | 按实例 ID 销毁，`submit` 报错停止（`gc` 靠标签识别实例） | `create_failed` |
| 创建后实际价格高于上限 | 销毁，换下一台 | `offer_rejected` |
| 镜像拉取卡住、AutoDL 部署迟迟不调度 | 超 15 分钟销毁或删除部署，换机 | `startup_timeout` |
| 镜像缺 python3、bootstrap 失败 | 等不到心跳，销毁；连续 2 次后停止并提示 | `runner_timeout` |
| bootstrap / boot 在 runner 起来之前放弃（bundle 下载或校验失败、启动文档拿不到、Python 不到 3.8 或导入出错、工作目录不可用） | Vast 上用容器自己的 key 自毁（有界重试），AutoDL 上退出、容器释放；“已有 runner 在运行”时什么都不做 | `bootstrap.log`、平台日志 |
| 无 GPU、驱动异常 | 销毁，换机 | `preflight_failed` |
| 实例连不上存储 | 销毁，提示检查网络或换存储 profile | `preflight_failed` |
| 磁盘不足 | 提交前和下载前检查；训练中 ENOSPC 走失败收尾 | `preflight_failed` 或 `FAILED` |
| 下载数据失败 | 重试 3 次，之后 `FAILED(data_sync)`，销毁 | `data_sync_retry` |
| `setup` 失败、训练非 0 退出 | `FAILED`，上传日志，销毁，不自动重试（多半是代码问题） | `status.json` |
| 训练卡死 | 终止，`FAILED(stalled)`，销毁 | `stalled` |
| 超过 `max_hours` | SIGTERM，宽限 60 秒后 SIGKILL，收尾，`TIMED_OUT`，销毁 | `deadline_hit` |
| runner 崩溃 | watchdog 杀掉 runner 留下的 rclone，尽力同步一次，按同步记录列出没传上去的文件，写 `FAILED(runner_crashed)`（runner 已有结论时写它的结论），销毁 | `status.json` |
| runner 活着但超过硬截止还没收尾完 | watchdog 结束所有进程，同样尽力同步一次并列出没传上去的文件，写 `TIMED_OUT`（runner 已有结论时写它的结论），销毁 | `deadline_hit` |
| AutoDL 上 watchdog 意外退出 | boot 重启它（最多 5 次）；watchdog 也失效、runner 又卡住时，boot 到自己的硬截止结束一切，容器释放 | boot 的平台日志 |
| 同一 attempt 的容器重启（Vast） | 训练没结束：重新运行，代码和数据不重下，从 ckpt 恢复；结论已定但最终同步没做完：只做最终同步和收尾；重启时已超过 `max_hours` 或已取消：不重跑，上传本地结果后写 `TIMED_OUT` / `CANCELLED` | `status.json` |
| 容器停机超过了 runner 的截止（甚至 watchdog 的硬截止）后重启 | watchdog 到截止接手时给满 `final_sync_timeout_min`（和 runner 一样，失败重试到时限），不按自己较短的时限同步；上限计入实例存活时限 | `status.json`、`deadline_hit` |
| `nvidia-smi` 卡死 | preflight 时算 GPU 故障，销毁换机；训练期间暂停 GPU 采样（不判卡死），取消和截止照常处理 | `preflight_failed` / 日志 |
| 训练进程不能降权（runner 不是 root、`run_as_root`、建不了用户、能力探针失败等） | 和 runner 同一用户运行；attempt 日志一行、`dgpu status` 一行说明原因 | `run_as_root`（`data.reason`、`reason_text`） |
| 容器里的 root 不能给训练用户的进程发信号、读它的文件 | 降权前的探针发现，不降权（`capability_missing`） | `run_as_root` |
| 大数据集 chown 期间取消或到达截止 | chown 每 256 个条目检查一次，按取消 / 截止收尾；容器重启后属主已对的目录不再遍历 | `status.json` |
| 文件系统不支持 flock | 记一行，改用 `run/runner.pid` 判断 runner 是否在运行 | runner / watchdog 日志 |
| runner 收到 SIGTERM/SIGINT（容器被停止） | 停止训练、最终同步，写 `FAILED(signalled)`，销毁 | `status.json` |
| 运行中存储断连 | 继续训练，恢复后补传；CLI 看到心跳过期、平台显示实例仍在运行时只报告“失联（实例仍在运行）”，不销毁、不重试 | `sync_failed` |
| 最终上传失败 | 重试到时限；仍失败则写明未上传的文件（按同步记录，等大小改写也算；拿不准时写“无法确定”而不是 0），`FAILED(final_sync)`，销毁，本地数据随之丢失 | `final_sync_failed` |
| 代码或数据下载失败后收尾（包括存储对传输整个不可用） | 本地没有结果，不做最终同步，不写 `UPLOADING` 和 `final_sync_failed`；结果都已上传过时最终同步也不调用 rclone | `data_sync_retry` |
| 宿主机宕机 | 心跳过期（按存储的时钟）显示 `LOST`；平台也显示实例已退出、消失或状态未知（或心跳停满 runner 最坏收尾时间）后，`wait --retries` 才换机续跑（先销毁失联的实例，换机失败也一样，销毁失败提示 `gc`），`gc` 才销毁；`retry` 随时可以手动换机 | `relaunch`、`gc_destroyed`、`destroy_failed` |
| 电脑或实例的时钟不准 | 不影响：心跳过期用存储对象的 `LastModified` 和存储响应的 `Date` 计算 | — |
| 取消之后 runner 已经不在（取消没人执行） | `wait --retries` 不自动重试，`retry` 拒绝，提示 `cancel --force`；任务已结束后显式 `retry` 会清除取消标志 | CLI 输出 |
| 自毁 API 失败 | 每 30 秒重试，共 10 次；之后由 `gc` 清理 | `self_destroy_failed` |
| CLI 或 `gc` 销毁失败、销毁后实例仍在 | 大声报告实例 ID；`submit`、`retry` 停下不再创建新实例；`gc` 记为失败，下次再试 | `destroy_failed` |
| `wait` 在任务结束后清理实例失败（平台 API 出错、存储删不了启动文档） | 大声报告实例 ID 或标签并提示 `gc`；任务成功时退出码 6，失败时保持任务的退出码 | `destroy_failed` |
| `wait` 暂时读不到存储 | 退避重试，最多 10 分钟；之后退出码 7，说明任务本身不受影响 | CLI 输出 |
| 本地数据里有超过 5 GiB 的文件 | `submit` 在上传任何东西之前报错，提示先用 rclone 上传再写 `r2://` | CLI 输出 |
| 本地数据里有符号链接 | 大小估算、确认、磁盘、AutoDL 检查和上传用同一次遍历（跟随链接）的清单；清单之后数据变了就拒绝上传 | CLI 输出 |
| 代码目录里有坏掉或指向目录外的符号链接 | `submit` 列出跳过的条目并拒绝，`--yes` 放行 | CLI 输出 |
| 启动途中电脑关机或断网 | 实例由 `gc` 清理 | `gc_destroyed` |
| 旧 attempt 没死透，新 attempt 已启动 | 重试前先确认旧实例销毁；旧 runner 读到更大的 attempt 后自行退出 | `attempt_fenced` |
| 两个 CLI 同时 `retry` | `attempt.json` 条件写，只有一个成功；输的一方只销毁不新于它读到的 attempt 的实例，不碰赢家的 | CLI 报错 |
| `wait` 收尾时另一个 CLI 刚 `retry` 成功 | `attempt.json` 已前进，`wait` 跳过收尾，不销毁新 attempt 的实例 | — |
| 存储里查不到任务的 dgpu 实例（另一份配置的 bucket、同事的任务、手工起名） | `gc` 只报告“未知任务”（附年龄和标签），`--include-unknown` 确认后才销毁；任务 ID 不是 `new_job_id` 形状的标签不碰 | `gc` 输出 |
| 账户余额耗尽 | 实例被平台停止，心跳过期显示 `LOST`；提交前已检查余额 | `status` 显示 |
| Vast key 没有读取账户信息的权限（需要两步验证） | `doctor` 报警告并用列出实例验证 key；`submit` 提示“未检查余额”后继续；key 无效仍报错 | CLI 输出 |
| 日志过大 | 继续读取防止阻塞；训练输出只保留最后 1 MB，结束时附在日志末尾；失败原因、退出码、最终状态照常写进日志 | `log_truncated` |
| 重复 `submit` | 生成两个独立任务，属于预期行为 | 不同 job ID |

## 6. 验收标准

本地可验证（`FakeBackend` + 假 rclone + `moto` 模拟 S3，无需云账户）：

- `uv run pytest` 全部通过，第 5 节每行至少一个测试。
- `uv run ruff check` 通过。

必须在真实环境验证（预算约 $2 + ¥20）：

- **P1 自毁**：Vast 容器内用 `CONTAINER_API_KEY` 调 REST 销毁自身（runner、watchdog，以及 bootstrap 在 runner 起来之前放弃时）。文档只写了 CLI 路径，失败则改用 `vastai` CLI。
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
- [x] 阶段 0：骨架与契约（2026-11-10，`make check`：19 个测试通过；`layout.py`、`states.py` 在 Python 3.9 下可导入）
- [ ] 探针 P1–P4、A1–A5
- [x] 工作包 A–G（2026-11-10 合并）
- [ ] 集成与评审：接缝已修复；本地端到端测试 `tests/test_integration_e2e.py` 通过（真实的编排、bundle、bootstrap、runner、watchdog，存储和平台是本地替身，未接触真实平台）；第一轮评审（A、B）和第二轮复核（F1–F9）的问题已在本地修复并配回归测试，见[评审清单](dgpu-review-2026-11-10.md)，均未在真实平台验证
- [ ] 真实环境验证（Vast E1–E5，AutoDL E1、E3）

### 真实平台验证记录

**2026-11-10，Vast + R2，提交 `74cab6c`（配置有误，见下）**

- `dgpu doctor`、`dgpu offers`：通过，Vast 搜索返回 45 台符合条件的机器。
- 两次 `dgpu submit` 共创建 5 台实例（RTX 3060，约 $0.067/h），全部被销毁，账户接口确认实例数为 0，花费约 $0.01。
- 已验证：创建、标签、状态轮询（pending → loading → running）、bootstrap 和 runner 启动、runner 报告失败后 CLI 销毁并换机（attempt 递增）、收到 SIGTERM 后销毁实例并退出（两次）。
- 发现并修复：下载 rclone 被 downloads.rclone.org 拒绝（Python 默认 User-Agent，`74cab6c`）；配置里的 endpoint 末尾带了 bucket 名，读写落到 `<bucket>/<bucket>/...`，实例上的 rclone 找不到 `spec.json`，runner 每次 preflight 失败（`2778039`：拒绝带路径的 endpoint，`doctor` 增加列举检查）。
- 观察：未缓存镜像（6.2 GB）的机器从创建到 runner 启动约 7 分钟。
- 未验证：Ctrl-C（SIGINT）路径。测试脚本把 `submit` 放在后台运行，后台进程忽略 SIGINT，这一步没有测到。
- 遗留问题：SIGTERM 中止后有一个 `secrets-<n>.json` 没被删除（当时列举失败），需在正确配置下复查。

**2026-11-11，Vast + R2，提交 `2778039`，任务 `probe-timeout-261111062221-cfcb`**

命令 `dgpu submit`（`run` 写一个输出文件和一个 ckpt 文件后 `sleep 900`，`max_hours: 0.05`，上限 $0.20/h），不运行 `dgpu wait`，用脚本每 15 秒记录 `dgpu status` 和 Vast 实例列表。

| 时间（UTC） | 事件 |
|---|---|
| 06:22:21 | 提交；实例 60174144（RTX 3060，$0.067/h）创建 |
| 06:23:01 | runner 启动（镜像已缓存，创建到启动约 40 秒） |
| 06:23:11 | 代码解压完成，`run` 开始 |
| 06:25:58 | 到达 `max_hours`，SIGTERM 终止 `run` |
| 06:26:00 | 最终同步，写 `TIMED_OUT` |
| 06:26:22 之前 | 实例从 Vast 列表消失（runner 自毁），之后保持消失 |

- **P1 自毁：通过。** 容器内用 `CONTAINER_API_KEY` 调 REST 销毁自身成功，没有 `self_destroy_failed` 事件；`dgpu gc --dry-run` 无遗留。
- **E3 超时：通过。** 状态 `TIMED_OUT`，事件 `deadline_hit`。
- **结果上传：通过。** `output/result.txt`、`ckpt/last.txt`、`logs/attempt-1.log` 及三份诊断日志都在存储里，`dgpu pull` 取回 `result.txt`，`dgpu logs` 能读到训练输出。
- **降权：在 `vastai/pytorch:2.14.1-cu126-cuda-12.9-mini-py312-2026-10-23` 上生效。** `run` 以 uid 1002（用户 `dgpu`）运行，读不到 `launch.json`，`PIP_USER=1`；`nvidia-smi` 可用（驱动 535.154.05）。没有测 `pip install` 和杀进程。
- 花费：余额 9.9896 → 9.9827，约 $0.007。
- 观察：Vast 实例列表接口有一次瞬时返回空（06:23:49，实例实际在运行），15 秒后恢复。CLI 对“创建后短时间内查不到”的宽限是必要的；宽限期之外的瞬时空列表是否会被误判为实例消失，需要复查。
- 观察：心跳里 `已同步 0`，而输出文件实际已上传，同步字节数的统计有误。

仍未在真实平台验证：成功结束（`SUCCEEDED`）的任务、SIGINT、`cancel`、`retry` 续跑（E2）、坏镜像（E4）、`gc` 清理孤儿（E5）、`pip install` 在降权下可用、真实的 YOLO 训练（E1）、AutoDL 的全部内容。

## 9. 后续风险

- **凭据暴露给房东**：凭据和训练数据都会落到租用机器的磁盘上，社区机器的房东可以读到。v1 用只授权相关 bucket 的专用 token，敏感数据打开 `datacenter_only`。Cloudflare 可能提供带 TTL 的临时凭据，尚未核实，后续可替换。
- **训练进程读到凭据**：降权（4.3）只在 runner 是 root 且条件满足时生效；不降权时（有 `run_as_root` 事件）训练代码能读到存储凭据和 Vast 的容器 key。降权本身只在本地用注入的系统调用测试过，要在真实实例上确认：setuid 之后读不到 `launch.json` 和 root 进程的 `/proc/<pid>/environ`、不能向 runner 发信号；`/dev/nvidia*` 对新用户可用（标准 NVIDIA 容器里是 0666）；镜像里有 `useradd`；`pip --user` 在镜像的 Python 里可用。另外，镜像或平台如果把 `CONTAINER_API_KEY` 写进全局可读的 `/etc/environment`，降权也挡不住（这个 key 只能操作本实例）。Python 装在 root 拥有的 venv 里（禁用用户 site）时目前不降权，以后可以给训练用户叠一层可写的 venv。
- **被放弃的进程**：SIGKILL 无效的进程（卡在驱动里的 `nvidia-smi`、卡在内核里的 rclone）会被放弃而不是等待，它们一直留到实例销毁。
- **AutoDL 的 `cmd` 退出码**：最终状态已记录时一律退出 0，依据是“Job 可能重启非 0 退出的容器”的推测，探针 A1 之后再定。
- **遗留实例的残余风险**：实例启动了但 runner 没起来，同时电脑关机（或 `submit` 被 SIGKILL）且一直没跑 `gc`。Vast 上 bootstrap 现在会在放弃时用容器 key 自毁（4.7），只要容器能跑到 `python3` 就覆盖这种情况；前提是容器 key 能删除实例（探针 P1），以及 Vast 真的执行 `onstart`。容器根本跑不起来（镜像拉不下来、`python3` 不存在）时仍然只能靠 `submit` 和 `gc`，会一直计费到余额耗尽。需确认 Vast 是否有服务端定时销毁，有的话加一层。AutoDL 上 `cmd` 退出容器就释放，部署记录由 `wait` 或 `gc` 删除。
- **失联判断偏保守**：心跳过期而平台显示实例仍在运行时，`wait --retries` 和 `gc` 最多多等 runner 最坏收尾时间（由 4.8 的公式算出，当前默认值下约 37 分钟）才动手，这段时间可能在为一台卡死的机器付费；换来的是不会销毁正在最终上传或暂时连不上存储的实例。平台状态字符串（Vast 的 `offline`、AutoDL 的部署状态）和真实宕机的对应关系要靠探针确认。
- **存储时钟**：过期判断依赖存储响应的 `Date` 头（R2 和 S3 都有）；不给 `Date` 的存储每次读取多一次探针写入和删除。
- **未知任务的实例**：换了 bucket 或配置文件之后，旧 bucket 里任务留下的实例在 `gc` 看来是“未知任务”，只报告不销毁，需要用户确认后 `gc --include-unknown`。
- **创建结果未知**：dgpu 按标签查满 `create_lookup_s`（默认 2 分钟）才放弃，之后平台才让实例出现的情况应该很少。这样的实例拿不到已删除的启动文档：Vast 上 bootstrap 用容器 key 自毁（最多几分钟），AutoDL 上 `cmd` 退出；没有自毁成功时，在 `wait` 结束或用户跑 `gc` 之前会一直计费（不会训练）。提示里给出了标签和 `gc` 的时间，标签记在 `attempt.json` 里，`status` 会显示。
- **AutoDL 跨境网络**：没有 AutoDL 到 R2 的实测，可能需要国内对象存储，届时数据要多维护一份。
- **AutoDL 文档空白**：无库存行为、状态字符串、计费粒度、`cmd` 非 0 退出时是否重启容器、分页从 0 还是 1 开始都要靠探针确认，探针前不在 AutoDL 上跑真实任务。分页已按响应自检（见 4.12），但只覆盖“第 1 页空”和“条数不够”两种可观察的情况。
- **房东质量差异**：可靠性分数是 Vast 自己的统计，没有公开定义。
- **流量费**：R2 不收出口费，但 Vast 房东会收，已通过单价上限过滤。
- **拉镜像耗时**：自定义镜像可能要 10–60 分钟，默认用 Vast 预缓存的基础镜像。
- **依赖变动**：Ultralytics 发版频繁，配方里固定版本。
- **许可证**：Ultralytics 的官方立场是训练出的模型默认适用 AGPL-3.0，内部和商业使用需要企业许可（非法律意见）。如用于商业产品需要先确认。
