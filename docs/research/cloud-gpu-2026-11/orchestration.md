# 编排层 / Serverless GPU 工具调研：SkyPilot、dstack、Modal 及同类（面向「偶尔在租用 GPU 上训练小型 YOLO、数据在 Cloudflare R2」的单人用户）

> 调研日期：2026-11-09。凡标注日期的信息均为该日期的状态；"vendor" 表示来源为厂商自己的营销/文档，"third-party/anecdote" 表示第三方转述或个人经验。
> 方法说明：SkyPilot 的能力边界（各云不支持哪些特性）我直接读了 GitHub master 分支源码（2026-11-09 抓取，最新 release v0.14.2），这比文档摘要更可靠；其余来自官方文档页面与少量第三方文章。工具抓取的页面摘要由小模型生成，关键数字已尽量与源码/原文交叉核对。

---

## 1. SkyPilot：支持的后端、R2 原生支持与三种模式、FUSE 依赖、Managed Jobs、自动停机、选最便宜 GPU、安装成本、Vast/RunPod 已知限制

### Takeaway
SkyPilot（最新 v0.14.2，2026-10-28）原生支持 `r2://`（beta），但在 **RunPod 和 Vast.ai 上 MOUNT / MOUNT_CACHED 均被源码明确禁用，只有 COPY（只读预取）可用**，输出必须由 run 脚本自己上传到 R2；Managed Jobs 在没有远程 API server 的单人场景下会拉起一台约 $0.25/小时的 CPU 控制器 VM（空闲 10 分钟自动停）；`sky launch --down` 能在作业结束（成功或失败）后销毁集群，但「配置/传数据/setup 阶段出错时不会销毁」。SkyPilot 没有任何中国大陆云（阿里云/腾讯云/火山引擎等）的内置后端。

### Cited Findings

**版本与安装**
- 最新 release 为 v0.14.2，发布于 2026-10-28T18:26:56Z — [GitHub releases API](https://api.github.com/repos/skypilot-org/skypilot/releases/latest)（官方文档域名已从 docs.skypilot.co 301 到 docs.skypilot.ai）
- 安装：`pip install "skypilot[<extra>]"`（或 `uv tool install --with pip "skypilot[<extra>]"`）；每个云对应一个 extra；`skypilot[all]` 不含 Vast，需显式装 `vast` extra；安装后用 `sky check` 查看已启用的云。文档给出的 Python 范围：uv 路径 3.9–3.13，pip 路径写 3.7–3.13（3.7 明显是过时文字） — [SkyPilot installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)；原始 rst：[installation.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/getting-started/installation.rst)
- RunPod 配置：`pip install "runpod>=1.6.1"` + `runpod config`（填 API key）；Vast 配置：`pip install "vastai-sdk>=0.1.12"`，把 API key 写入 `~/.config/vastai/vast_api_key`；Lambda：API key 写入 `~/.lambda_cloud/lambda_keys` — [installation.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/getting-started/installation.rst)、[installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)

**支持的后端（文档 + 源码）**
- 文档列出：Kubernetes、Slurm、AWS、GCP、Azure、CoreWeave、Nebius、Together AI(经 K8s)；社区(community)徽章：RunPod、OCI、Lambda Cloud、Paperspace、Vast、Fluidstack、Cudo、Shadeform、IBM、SCP、vSphere、Prime Intellect、Seeweb、Verda — [installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)
- master 分支 `sky/clouds/` 目录共 27 个文件：aws、azure、cudo、do、fluidstack、gcp、hyperbolic、ibm、kubernetes、lambda_cloud、mithril、nebius、oci、paperspace、primeintellect、runpod、scp、seeweb、shadeform、slurm、ssh、vast、verda、vsphere、yotta（及 cloud.py 等基类）——**没有阿里云/腾讯云/火山引擎/华为云等中国云** — [GitHub API: sky/clouds](https://api.github.com/repos/skypilot-org/skypilot/contents/sky/clouds?ref=master)
- 存储专用后端：Cloudflare R2（`cloudflare` extra，beta）、VastData 对象存储（与 Vast.ai 无关）、Hugging Face（Buckets 读写，Hub repo 只读） — [installation docs](https://docs.skypilot.ai/en/latest/getting-started/installation.html)

**R2 原生支持与三种模式**
- R2 可作为源 `r2://<bucket>`，或 `store: r2` 让 SkyPilot 建桶；MOUNT 为默认模式。S3 兼容类存储在 x86_64 用 goofys、ARM64 用 rclone — [Storage docs](https://docs.skypilot.ai/en/latest/reference/storage.html)
- MOUNT：FUSE 挂载，读流式，close() 时整文件上传，close-to-open 一致性，不支持完整 POSIX（随机写、追加可能失败；不保留权限、不拷贝软链）；COPY：预取到本地盘，**只读，写入不会回传桶**，VM 磁盘必须大于桶内数据；MOUNT_CACHED：rclone VFS 本地缓存，异步按序上传，任务在缓存写入 flush 完成后才标记完成 — [Storage docs](https://docs.skypilot.ai/en/latest/reference/storage.html)
- `sky storage ls` 只列出 SkyPilot 自己创建的桶，外部桶不能经 SkyPilot 管理 — [Storage docs](https://docs.skypilot.ai/en/latest/reference/storage.html)
- R2 配置（官方安装文档，标注 beta）：`pip install "skypilot[cloudflare]"`；`AWS_SHARED_CREDENTIALS_FILE=~/.cloudflare/r2.credentials aws configure --profile r2`（region 填 `auto`，output 填 `json`）；把 Cloudflare Account ID 写入 `~/.cloudflare/accountid` — [installation.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/getting-started/installation.rst)
- 源码中存在 `R2CloudStorage`：对 `r2://` 源把 scheme 换成 `s3://`，用 `--endpoint-url`（由账号 ID 生成）和 `--profile r2`、`~/.cloudflare/r2.credentials` 调 aws s3 sync/cp 下载——即 COPY 路径本质上是远端执行 aws cli — [sky/cloud_stores.py](https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/cloud_stores.py)

**R2 MOUNT 是否在所有后端可用：不可以（源码证据）**
- `sky/clouds/runpod.py`：`STORAGE_MOUNTING` 被列为不支持，原文："Mounting object stores is not supported on RunPod. To read data from object stores on RunPod, use `mode: COPY` to copy the data to local disk." 同一字典还禁用：STOP、MULTI_NODE、CUSTOM_DISK_TIER、CUSTOM_NETWORK_TIER、HIGH_AVAILABILITY_CONTROLLERS、CUSTOM_MULTI_NETWORK、LOCAL_DISK — [runpod.py](https://github.com/skypilot-org/skypilot/blob/master/sky/clouds/runpod.py)（raw：[link](https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/clouds/runpod.py)）
- `sky/clouds/vast.py`：同样禁用 STORAGE_MOUNTING、MULTI_NODE、CUSTOM_DISK_TIER、CUSTOM_NETWORK_TIER、HIGH_AVAILABILITY_CONTROLLERS、CUSTOM_MULTI_NETWORK、LOCAL_DISK（注意：Vast 没有禁用 STOP，也没有禁用 SPOT_INSTANCE） — [vast.py](https://github.com/skypilot-org/skypilot/blob/master/sky/clouds/vast.py)
- 判定逻辑：`Task.get_required_cloud_features()` 仅当某个 storage mount 的 mode 属于 `MOUNTABLE_STORAGE_MODES = [MOUNT, MOUNT_CACHED]` 时才要求 `STORAGE_MOUNTING`，COPY 不触发该检查 — [sky/task.py](https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/task.py)、[sky/data/storage.py](https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/data/storage.py)。因此 RunPod/Vast 上 MOUNT 与 MOUNT_CACHED 都会在调度阶段被拒绝，COPY 可用。
- Vast 官方博客（vendor，2026-03-15）也列出 Vast 上 SkyPilot 不支持：多节点、对象存储挂载、启动后开端口、自定义磁盘/网络档位 — [Vast.ai article](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot)
- 佐证（anecdote）：GitHub issue #8592（2026-01-15，SkyPilot 0.11.1）报告 RunPod 容器里 JuiceFS FUSE 挂载失败（`/dev/fuse: operation not permitted`，加 `--privileged` 也不行），标记为 stale — [issue mirror](https://www.goodfirstissue.org/skypilot-org/skypilot/issues/8592)
- Lambda 的禁用清单（STOP、SPOT_INSTANCE、IMAGE_ID、CLONE_DISK_FROM_CLUSTER、CUSTOM_DISK_TIER、CUSTOM_NETWORK_TIER、HOST_CONTROLLERS、HIGH_AVAILABILITY_CONTROLLERS、CUSTOM_MULTI_NETWORK、LOCAL_DISK）**不含** STORAGE_MOUNTING — [lambda_cloud.py](https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/clouds/lambda_cloud.py)。也就是说 Lambda 在调度层面允许 MOUNT，但 Lambda 不能托管 jobs controller、不支持 spot。

**Managed Jobs（`sky jobs launch`）与控制器**
- 工作方式：SkyPilot 为每个 job 临时建集群，监控、故障恢复、结束后销毁；硬件故障/抢占默认恢复；用户代码报错只有在配置 `max_restarts_on_errors` 或 `recover_on_exit_codes` 时才重试；**恢复默认从头重跑**，要续训必须自己把 checkpoint 写到桶/卷 — [Managed jobs doc](https://docs.skypilot.ai/en/latest/examples/managed-jobs.html)；原始 rst：[managed-jobs.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/examples/managed-jobs.rst)
- 控制器：有**远程 API server** 时由 API server 直接管理 job（consolidation mode，默认），无需单独控制器 VM；**使用本地 API server（单人默认情形）时自动使用 "legacy" 远程 jobs controller**——一台小型按需 CPU VM 或 K8s pod，首次提交 job 时自动拉起，空闲 10 分钟自动 stop；文档给出成本约 $0.25/小时（运行时）、< $0.004/小时（停止时），默认约 4+ CPU / 16 GiB / 50 GB 盘 — [managed-jobs.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/examples/managed-jobs.rst)
- 文件上传：managed jobs 需要中间桶存放 workdir/本地 file_mounts；未配置时自动建临时桶并在结束后删除；可在 `~/.sky/config.yaml` 设 `jobs.bucket`，**支持 `r2://`**；对象存储不可用时回退为经控制器的两跳上传 — [managed-jobs.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/examples/managed-jobs.rst)
- 泄漏风险（官方警告）：控制器长期存活，应使用不过期的静态凭据；"If a credential expires, it could leave the controller with no way to clean up a job, leading to expensive cloud instance leaks." 销毁控制器只允许在无进行中 job 时进行 — [managed-jobs.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/examples/managed-jobs.rst)
- `sky jobs launch` 的 CLI 选项列表中**没有** `--down` / `-i` / `--retry-until-up`（teardown 由 managed job 生命周期负责）；有 `--use-spot`、`--job-recovery`、`--gpus`、`--priority` 等 — [CLI reference](https://docs.skypilot.ai/en/latest/reference/cli.html)

**autostop / autodown 能否「保证」销毁**
- `sky launch --down`：作业全部结束（成功或异常）后销毁集群；若同时设 `-i`，则空闲指定分钟后销毁。**"If provisioning, data syncing, or setup errors occur, the cluster is not torn down"（便于调试）**。`-i/--idle-minutes-to-autostop`：空闲 = 无运行/排队作业；默认 `--wait-for jobs_and_ssh` 还会等 SSH 连接结束；不设则永不自动停 — [CLI reference](https://docs.skypilot.ai/en/latest/reference/cli.html)
- RunPod、Lambda 均不支持 STOP（见上），所以这两个云上"autostop"只能以 autodown（`--down`）形式使用（推断，依据 STOP 不支持清单）。
- 官方没有"保证"措辞；`--retry-until-up` 在所有区域无容量时无限重试配置 — [CLI reference](https://docs.skypilot.ai/en/latest/reference/cli.html)

**是否自动找最便宜 GPU**
- 官方：优化器按成本排序候选（表中标出 CHOSEN）；`accelerators` 写成集合 `{...}` 时 "automatically find the cheapest available GPU"，写成列表则强制偏好顺序；容量/配额失败会先在同云各区域重试，再换"下一个最便宜的云"；固定 cloud/region/zone 则只在该范围内 failover；`any_of` / `ordered` 可表达候选资源 — [Auto-failover docs](https://docs.skypilot.ai/en/latest/examples/auto-failover.html)（该页示例只涉及 AWS/GCP/Azure/K8s，没有专门说明 Vast/RunPod）
- Vast 的价格口径：`accelerators_to_hourly_cost` 返回 0.0，"Vast includes accelerators in the hourly cost"，实例价格来自 Vast catalog — [vast.py](https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/clouds/vast.py)

**Vast / RunPod 上的已知坑（第三方/anecdote 为主）**
- 某开源项目 datarax 的云基准文档：skypilot 0.11.x + `vastai-sdk >= 1.0` 会报 `'VastAI' object has no attribute 'api_key_access'`，需 pin `vastai-sdk<0.3`；0.11.x + click>=8.3 报 `False backend is not supported`；改依赖后要 `sky api stop`；Vast 实例出现约需 30–120 秒（可到 2–6 分钟），默认启动超时 1800 秒；超时/回退竞态可能留下两台实例，需 `sky status --refresh` + `sky down`，仅在 Vast UI 可见的要手动终止；SkyPilot 可能挑到劣质主机池，可用 `--region` 固定 — [datarax cloud benchmarks (third-party, "early 2026")](https://datarax.readthedocs.io/en/latest/benchmarks/cloud/)。注意这些是 0.11.x 时代的问题，v0.14.2 是否已修复未验证。
- Vast spot：在 `resources` 设 `use_spot: true`，最高出价用 `~/.sky/config.yaml` 的 `create_instance_kwargs.bid_price`；要自动恢复需用 managed jobs — [Vast.ai article (vendor, 2026-03-15)](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot)
- Vast 示例 YAML（vendor）：`resources: {accelerators: H100:1, cloud: vast, image_id: docker:vllm/vllm-openai:latest, disk_size: 100}` — 同上

### Inferences
- 对「数据在 R2」的用户：在 RunPod/Vast 上 SkyPilot 的 R2 价值被削弱为"COPY 下载数据集 + 自己在 run 脚本里用 aws cli/rclone 把权重传回 R2"，与自己写脚本做的事情差不多；MOUNT 的便利只在 AWS/GCP/Azure/Lambda/Nebius/K8s 等 VM 类后端上才有（Lambda 上 FUSE 是否实际可用未验证）。
- 单人偶尔训练，若不想维护控制器：用 `sky launch --down`（客户端驱动、无控制器 VM）比 `sky jobs launch`（控制器 ≈ $0.25/h 且需要一个能托管 CPU 控制器的云）更轻；代价是没有自动抢占恢复。
- 因 `--down` 在 setup 失败时不销毁，且 Vast 有"双实例竞态"的第三方报告，使用时应把 `sky status -r` / `sky down -a` 与云控制台核查作为固定收尾动作，不能视为强保证。

### Gaps
- Managed jobs 的 legacy 控制器能否落在 RunPod/Vast（二者源码只禁用了 HIGH_AVAILABILITY_CONTROLLERS，未禁用 HOST_CONTROLLERS，但 Lambda 明确禁用）以及 Vast/RunPod 有无 CPU-only 小机型，本次未验证；通常用户需要 AWS/GCP 等账号托管控制器（未核实）。
- 自动停机由集群端守护进程还是本地客户端执行——本次没有找到官方文字依据（我的背景知识倾向于集群端，但未核实）。
- SkyPilot + R2 COPY 在 RunPod/Vast 默认镜像上是否能自动带上 `~/.cloudflare/r2.credentials` 并有 aws cli，未实测；也未找到 "R2 + RunPod/Vast" 的 GitHub issue。
- v0.14.x 上 Vast 集成的稳定性：没有找到 2026 年下半年的可靠报告。

---

## 2. dstack：后端、R2/存储、spot、自动终止、安装成本、Vast/RunPod 限制

### Takeaway
dstack（最新 0.22.7，2026-11-04）需要一个 `dstack server`（可跑在笔记本上或用托管的 dstack Sky）并先创建 fleet 再 `dstack apply`；Vast.ai 与 RunPod 在 dstack 里是「容器型」后端，**不支持 instance volumes、idle_duration 自动回收不适用**，文档没有任何 S3/R2 挂载功能，数据只能在 `commands` 里自己用 CLI 拉/传。支持 Lambda（VM 型）、没有任何中国大陆云。

### Cited Findings
- 最新 release 0.22.7，发布于 2026-11-04T10:52:05Z — [GitHub releases API](https://api.github.com/repos/dstackai/dstack/releases/latest)
- VM 型后端：AWS、Azure、GCP、Lambda、Nebius、Crusoe、Verda、AMD Developer Cloud、Digital Ocean、Hot Aisle、JarvisLabs、CloudRift、Vultr、Seeweb、OCI；容器型后端：Kubernetes、Slurm、Daytona、Lium、**RunPod（需 Read & Write API key，Community Cloud 需 `community_cloud: true` 开启）、Vast.ai（含 Community Cloud，`community_cloud: false` 可限制到 Server Cloud）**；另有 SSH fleets（自有机器，无需后端配置）— [dstack backends docs](https://dstack.ai/docs/concepts/backends/)。无阿里云/腾讯云/火山引擎等。
- 安装：`uv tool install "dstack[all]" -U` 或 `pip install "dstack[all]" -U`，然后 `dstack server`（启动时打印 admin token，默认 `http://127.0.0.1:3000/`）；服务器 "can run on your laptop or any environment with access to the cloud"；需要 Git 和 OpenSSH；配置文件 `~/.dstack/server/config.yml`；CLI 用 `dstack project add --name main --url http://127.0.0.1:3000 --token <admin-token>` 连接；另有 Docker 部署和托管的 dstack Sky — [dstack installation](https://dstack.ai/docs/installation/)
- 提交运行前必须先建 fleet：`dstack apply -f fleet.dstack.yml`（示例 `type: fleet, name: default, nodes: 0..2, idle_duration: 1h, gpu: 0..8`），然后 `dstack apply -f task.dstack.yml` — [dstack quickstart](https://dstack.ai/docs/quickstart/)
- Task 配置：`resources.gpu` 支持厂商/型号/显存/数量及范围（如 `H100:1..2`）；`dstack offer` 列出可用 offer；`spot_policy` 默认 `on-demand`（可选 spot/auto）；`max_duration` 默认 **off**（不限时）；`stop_duration` 默认 5m；`idle_duration` 默认 runs 5m / fleets 3d，**仅适用于 VM 型后端**；`retry` 默认 false（`on_events: no-capacity/interruption/error` + `duration`）；`utilization_policy` 在 GPU 利用率低于阈值持续一个 time_window（最小 5m）时终止运行；`max_price` 限制每小时价格 — [Task reference](https://dstack.ai/docs/reference/dstack.yml/task/)、[Tasks concept](https://dstack.ai/docs/concepts/tasks/)、[Protips](https://dstack.ai/docs/guides/protips/)
- `files` 单项限制 2MB（可由 `DSTACK_SERVER_CODE_UPLOAD_LIMIT` 调大）；`repos` 克隆 Git 仓库 — [Tasks concept](https://dstack.ai/docs/concepts/tasks/)
- Volumes：network volumes 支持 AWS、GCP、RunPod、Kubernetes、Daytona、Lium（Lambda 和 Vast.ai 不在列）；instance volumes 在除 Daytona、Lium、RunPod、Slurm、Vast.ai 之外的后端和 SSH fleet 可用；**文档没有 S3/R2 对象存储挂载**，只在 FAQ 里建议用对象存储跨后端共享数据 — [dstack volumes](https://dstack.ai/docs/concepts/volumes/)
- Idle 自动终止「不适用于」容器型后端（Kubernetes、Vast.ai、RunPod） — [Protips](https://dstack.ai/docs/guides/protips/)、[Task reference](https://dstack.ai/docs/reference/dstack.yml/task/)
- Vast.ai 的 dstack 指南称：停止服务会终止（terminate）Vast 实例从而停止计费 — [Vast docs: dstack + vLLM](https://docs.vast.ai/examples/serving-infrastructure/dstack-vllm)（仅看到搜索摘要，未打开全文）
- dstack 官方博客：dev environment 在用户离线后约 3 分钟内被自动检测并关停（针对 dev environment，不等同于 task/本地 server 掉线场景） — [dstack inactivity blog](https://dstack.ai/blog/inactivity-duration/)（仅看到搜索摘要）

### Inferences
- 要让训练数据/权重走 R2，在 dstack 里只能靠 `commands` 里 `aws s3 sync --endpoint-url ...` 或 `rclone`，R2 密钥通过 `env` 传入；没有 SkyPilot 那种一等公民的 `r2://` 抽象。
- 由于 `max_duration` 默认不限时，应在每个 YAML 显式设置 `max_duration` 和 `max_price`，作为防止训练卡死烧钱的保险；Vast/RunPod 上没有 idle_duration 回收，这点更重要。
- 单人偶尔用：必须长期记得先启 server、建 fleet，操作步骤比 SkyPilot 多一层；若不想本机跑 server，可考虑托管的 dstack Sky（价格/条款本次未核实）。

### Gaps
- dstack server 进程被杀/笔记本休眠时，已在 RunPod/Vast 上运行的实例是否会被孤立继续计费：没有找到官方说明或 issue。
- dstack Sky 的价格、是否有加价、能否从中国大陆访问：未找到。
- dstack 在 RunPod/Vast 上抢占（spot/interruptible）恢复的实际成熟度：未找到可靠报告。
- 容器型后端上的 `dstack apply` 实际端到端体验（冷启动时间、镜像拉取）：未找到。

---

## 3. dstack 与 SkyPilot 对比（面向本用例）

### Takeaway
没有找到独立的、公正的 2025–2026 两者实测对比；能确认的差异来自两者各自的文档：SkyPilot 无需常驻 server、有 `r2://` 存储抽象、跨 ~25 个云的成本优化与 managed-jobs 抢占恢复；dstack 需要 server+fleet，但在 Vast/RunPod 上走容器型后端并提供 `dstack offer`、`max_price`、`utilization_policy` 等成本护栏，无对象存储挂载。

### Cited Findings
- Lambda 官方博客（**vendor，Lambda 自己卖 GPU 云并推广自家 K8s，2026-08-04**）：SkyPilot 是"按可用性和成本把工作负载放到不同环境的作业启动器"，界面更简单、适合突发负载，但调度控制更少、增加一层调试；dstack 是"跨云/K8s/本地的 vendor-agnostic 控制面"，更新、社区更小、集成更少；结论是没有通用方案，SkyPilot 或 dstack 适合"可移植的作业执行" — [Lambda blog](https://lambda.com/blog/choosing-the-right-orchestration-layer-for-your-ai-use-cases)（无基准、无定价对比，且该文对 dstack 的论述最单薄）
- dstack README/文档把自己定位为统一控制面（"unified control plane for GPU provisioning and orchestration"），server 可本地/远程/托管 — 搜索摘要引用 [dstack README](https://cdn.jsdelivr.net/gh/dstackai/dstack@master/README.md)
- SkyPilot 自己的 "Comparing SkyPilot with other systems" 页面（vendor）本次抓取只拿到导航文字，没有取到对比正文 — [SkyPilot comparison doc](https://docs.skypilot.ai/en/latest/reference/comparison.html)
- 关键事实对比表（均来自上面第 1、2 节的文档/源码引用）：
  - 控制面：SkyPilot 本地 API server（单人默认），managed jobs 另需控制器 VM；dstack 必须有 server（本机或 Sky）+ fleet。
  - R2：SkyPilot 原生 `r2://`（beta），RunPod/Vast 仅 COPY；dstack 无。
  - Vast/RunPod 接入：SkyPilot 为社区后端，需 pin 依赖版本（第三方报告）；dstack 为内置容器型后端，instance volumes 不支持。
  - 自动终止：SkyPilot `--down`（setup 失败时不销毁）；dstack `max_duration`（默认关）、`utilization_policy`、`idle_duration`（Vast/RunPod 不适用）。
  - 中国云：两者都没有。

### Inferences
- 在"R2 数据 + Vast/RunPod + 一次一个单卡小任务"的场景，两者的实质功能重叠度很高，SkyPilot 在 R2 存储抽象和"一个 YAML + `sky launch --down`"的启动路径上更短；dstack 的优势（统一 server、团队/多用户、K8s/SSH fleet）对单人偶尔使用价值有限。
- 偏向 SkyPilot 的这个判断主要来自上述文档差异，没有独立实测佐证。

### Gaps
- 没有找到 2025–2026 年中立的 dstack vs SkyPilot 实测（启动时延、稳定性、Vast/RunPod 成功率）。
- 未打开第三方评测站（如 makerstack 的 dstack review）全文，其可信度亦未评估。

---

## 4. Modal（及 Beam 类 serverless GPU）：计费、最长时长、R2 挂载、冷启动、镜像构建、多小时训练、限制

### Takeaway
Modal 是这类里"零运维、不会漏关机"的最简单选项：按秒计费（T4 ≈ $0.59/h、A10 ≈ $1.10/h、A100-80GB ≈ $2.50/h），Starter 每月 $30 免费额度，单次调用最长 24 小时，且有官方 Ultralytics YOLO 示例和原生 R2 `CloudBucketMount`；缺点是函数默认可被抢占（GPU 函数无法设为不可抢占），需自己做 checkpoint/续训，R2 挂载基于 AWS Mountpoint，写入受限（仅顺序写、不能追加/随机写）。

### Cited Findings
- **价格（按秒，官网 2026-11 抓取）**：T4 $0.000164/s、L4 $0.000222/s、A10 $0.000306/s、L40S $0.000542/s、A100 40GB $0.000583/s、A100 80GB $0.000694/s、H100 $0.001097/s、H200 $0.001261/s、B200 $0.001736/s；CPU $0.0000131/核/s，内存 $0.00000222/GiB/s；Starter：$30/月免费额度、无月费、10 GPU 并发；Team：50 GPU 并发 + $100/月额度；页面还标注自 2026-09-01 起 Starter 的 Shared Endpoint token 用量从第一个请求起计费、不抵扣 $30 — [Modal pricing (vendor)](https://modal.com/pricing)。（换算：T4 ≈ $0.59/h，L4 ≈ $0.80/h，A10 ≈ $1.10/h，L40S ≈ $1.95/h，A100-40 ≈ $2.10/h，A100-80 ≈ $2.50/h，H100 ≈ $3.95/h；仅 GPU 部分，另计 CPU/内存。）Ultralytics 的 Modal 指南也给出近似值 T4 ≈ $0.59/h、A10G ≈ $1.10/h、A100 ≈ $2.10/h — [Ultralytics Modal quickstart](https://docs.ultralytics.com/guides/modal-quickstart)
- **最长运行时间**：函数 timeout 默认 300 秒，可设 1 秒–24 小时；`modal.Retries` 每次重试重新计时；`startup_timeout` 独立于执行 timeout — [Modal timeouts](https://modal.com/docs/guide/timeouts)
- **抢占**：所有函数默认可被抢占，Modal 会优雅终止并"在同一输入上重启"；`nonpreemptible=True`（client ≥ v1.2.3）使 CPU/内存价格 ×3，且 **"not supported for GPU Functions"**；官方建议长训练函数频繁 checkpoint — [Modal preemption](https://modal.com/docs/guide/preemption)
- **多小时训练**：官方 "long, resumable training" 示例：周期性把 checkpoint 存到 Volume、启动时检查最新 checkpoint；用 `.spawn(...).get()` 而非 `.remote`（`.remote` 创建的 function call 24 小时后过期），可用 `modal run --detach` 启动；链式重启可跑到 10 天 — 搜索摘要引用 [Modal long-training example](https://modal.com/docs/examples/long-training)（未打开全文）
- **R2 CloudBucketMount**：`modal.CloudBucketMount(bucket_name, bucket_endpoint_url=..., secret=..., read_only=..., key_prefix=...)`；`bucket_endpoint_url` "Required for Cloudflare R2 and Google Cloud Storage buckets"，R2 端点形如 `https://<ACCOUNT ID>.r2.cloudflarestorage.com`；Secret 需含 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`；API token 需对目标桶具备读/写/列权限（不要 admin、不要 IP 过滤） — [CloudBucketMount reference](https://modal.com/docs/reference/modal.CloudBucketMount)、[Cloud bucket mounts guide](https://modal.com/docs/guide/cloud-bucket-mounts)
- **挂载限制**（基于 AWS Mountpoint）：不能 append 打开、不支持随机写/seek+write、写文件必须 truncate 模式、不能重命名（S3 挂载）、桶流量不走 Proxy；典型报错 `PermissionError: [Errno 1] Operation not permitted`；变通：先写本地临时文件再 `shutil.move` 到挂载点；TensorBoard SummaryWriter 用 append，需先写临时目录再同步；针对大文件顺序读做了优化 — [Cloud bucket mounts guide](https://modal.com/docs/guide/cloud-bucket-mounts)
- **冷启动**：官方称 "Containers boot in about one second"（含全局作用域导入与 `modal.enter`）；空闲容器默认保留 60 秒（`scaledown_window` 2 秒–20 分钟）；页面没有给 GPU 冷启动数字 — [Modal cold start](https://modal.com/docs/guide/cold-start)
- **镜像构建**：Ultralytics 官方指南给出的 Image 写法：`modal.Image.debian_slim(python_version="3.11").apt_install("libgl1","libglib2.0-0").pip_install("ultralytics")`；本地 `modal run` 时首次自动构建 — [Ultralytics Modal quickstart](https://docs.ultralytics.com/guides/modal-quickstart)
- **YOLO 官方示例**：Ultralytics 指南：`pip install modal` → `modal token new`（"opens a browser window to authenticate"）→ `modal run train_yolo.py`；示例函数 `@app.function(image=image, gpu="T4", timeout=3600, volumes={"/data": volume})`，`YOLO("yolo26n.pt").train(data="coco8.yaml", epochs=3, imgsz=640, project="/data/runs")`，权重在 Volume 的 `/data/runs/.../weights/`；指南未讲如何把权重下载到本地 — [Ultralytics Modal quickstart](https://docs.ultralytics.com/guides/modal-quickstart)。Modal 官方 YOLOv10 微调示例：单张 A100、timeout 60 分钟、数据来自 Roboflow（需 Roboflow API key 的 Modal Secret）、Volume `example-yolo-finetune` 挂在 `/root/data`、"Training usually takes only a few minutes" — [Modal finetune_yolo example](https://modal.com/docs/examples/finetune_yolo)
- **Beam（同类，vendor 定价页，抓取不完整）**：Developer 计划含每月 $20 额度；RTX 4090 serverless $0.69/h；按需机器 4090 起价 $0.44/h、A100 80GB SXM4 起价 $1.36/h、H100 PCIe 起价 $1.83/h；serverless 按毫秒计费，VM/sandbox 按秒；T4/A10G 价格与最大 timeout 页面未列 — [Beam pricing](https://www.beam.cloud/pricing)

### Inferences
- 微型 YOLO（nano/small，几十分钟至 2 小时）在 Modal 上一次约 $0.6–1.5（T4/A10 + CPU/内存估算，我的推断），$30 免费额度大致可覆盖每月 ~50 个 T4 小时（$30 ÷ $0.59，未计 CPU/内存）；对"偶尔训练"而言，价差（相对 RunPod 4090 社区云 ~$0.34/h）基本可忽略，换来的是不用管机器回收。
- 超过 24 小时或必须不被抢占的训练在 Modal 不适合；小 YOLO 通常远低于此，但需要在训练脚本中开启 Ultralytics 的 `resume` 并把 `project` 指向 Volume 或 R2 挂载，才能应对抢占重启。
- R2 挂载更适合只读数据集；权重/日志建议先写本地容器盘或 Volume，训练结束后再顺序拷贝到 R2 挂载点（依据 Mountpoint 的写限制与官方变通）。

### Gaps
- Modal 对 GPU 函数的实际抢占频率（对 T4/A10 的真实中断率）无官方数字。
- GPU 冷启动（排队获取 GPU）的真实时延无官方数字，也没有可靠第三方实测。
- Modal 把权重从 Volume 取回本地的命令（我记得有 `modal volume get`，但本次没有在抓取页面中核实）。
- Beam 的最长任务时长、免费额度细节、Beam 与 R2 的集成：页面未提供/未验证。
- Modal 近期（2026 年）关于免费额度被滥用或政策变化的信息，未专门检索。

---

## 5. Lambda Cloud 与其他有简单 API 的按需 GPU 提供商：价格与可得性

### Takeaway
Lambda 的按需单卡价格对"小 YOLO"并不具竞争力（A10 24GB ≈ $1.29/h、A6000 ≈ $1.09/h、RTX 6000 24GB ≈ $0.69/h），且页面声明这些价格自 2026-11-16 起生效；RunPod 社区云（4090 ≈ $0.34/h、3090 ≈ $0.22/h）和 Vast 市场更便宜但可靠性/可得性更不确定。Lambda 按分钟计费，无出入站流量费，但在 SkyPilot 中不支持 spot、STOP 和托管控制器。

### Cited Findings
- **Lambda 价格（官网 lambda.com/pricing，抓取于 2026-11；页面写明 "Instance prices take effect November 16, 2026, at 00:00 UTC"，此前需登录查看当前价格）**，按 GPU/小时，1x 配置（列映射为抓取工具推断，需核实）：B200 $7.99、H100 SXM $4.89（PCIe $3.79）、GH200 $2.59、A100 40GB $1.99、A10 $1.29、A6000 $1.09、Quadro RTX 6000 $0.69；8x A100 80GB $3.19、8x V100 $0.79；"self-serve, first-come access"，无最低承诺 — [Lambda pricing (vendor)](https://lambda.com/pricing)（lambda.ai 已 301 到 lambda.com）
- 计费：按一分钟增量计费；实例启动并通过健康检查后开始计费，**terminate 时停止计费**（页面未说明 stop 是否停止计费）；Filesystem 按 GiB/月、小时增量计费，示例 $0.20/GiB/月；不收入站/出站流量费 — [Lambda billing docs](https://docs.lambda.com/public-cloud/billing/)
- 价格趋势（third-party，低可信）：多篇聚合站称 2026 年 Lambda 多次涨价（如 2026-06 H100 SXM $2.99→$3.99），但这些来自搜索摘要，我没有找到 Lambda 官方涨价公告，也没能确认 11-16 的新价是否高于旧价 — 搜索摘要引用 [activepieces Lambda pricing guide](https://www.activepieces.com/blog/lambda-labs-pricing-gpu-cost-per-hour-2026-guide)
- **RunPod（官网定价页抓取，社区云数字来自页面结构化数据/FAQ，需谨慎）**：RTX 4090 Secure $0.89/h、Community $0.34/h；RTX 3090 $0.50 / $0.22；A40 $0.59 / $0.35；L4 $0.59 / $0.44；L40S $1.09 / $0.79；RTX A5000 $0.27 / $0.16；A100 PCIe $1.79 / $1.19；H100 SXM $3.99 / $2.69；Pod 按秒计费、无承诺；存储：容器盘 $0.10/GB/月，卷盘运行时 $0.10、空闲 $0.20/GB/月 — [RunPod pricing (vendor)](https://www.runpod.io/pricing)
- **Vast.ai**：页面称 on-demand 保证不中断；Interruptible（spot）"at least 50% cheaper"，可被回收；按秒计费，无最低时长、不向上取整；GPU 实时价格卡片在抓取中未渲染，所以没有具体价格；页面未说明停止状态下存储是否继续计费 — [Vast.ai pricing (vendor)](https://vast.ai/pricing)
- Vast 与 SkyPilot：`use_spot` + `bid_price`（见第 1 节）。RunPod 与 dstack、SkyPilot 的接入见第 1、2 节。
- **Hugging Face Jobs**（见第 6 节）也是按分钟计费的云端作业，T4-small $0.40/h。

### Inferences
- 对于只需 12–24GB 显存的小 YOLO，Lambda 的最低档（A10 $1.29/h）比 RunPod 社区 4090（$0.34/h）、Modal T4（$0.59/h）、HF Jobs T4-small（$0.40/h）都贵；Lambda 的优势是 VM 型、可靠性口碑较好（未核实）、无流量费。若 Lambda 无货（未找到可得性数据），其"简单 API"优势并不转化为便宜的小 GPU。
- "直接调用某家 API 的脚本" 仅在你坚持用 RunPod/Vast 这类最便宜的市场时才有意义，且需要自己处理：创建实例、轮询就绪、SSH/启动命令、结束后销毁、失败兜底——这些正是 SkyPilot/dstack 帮你封装的部分。

### Gaps
- Lambda 2026 年单卡可得性（A10/A6000 经常缺货与否）：没有找到可靠数据，只有搜索摘要里的传闻级内容。
- Lambda 11-16 之前的旧价格与新价的差异：未获取（页面要求登录）。
- Vast 与 RunPod 的实时/最新价格与可靠性统计、Lambda 公共 API 的端点和限速等：未在本次一手核实。
- 其他提供商（如 Thunder Compute、Nebius、Hyperbolic、Prime Intellect 等 SkyPilot 已集成的新云）的价格与可得性：未检索。

---

## 6. 免费/近乎免费的基线：Colab、Kaggle、Hugging Face Jobs、Lightning AI

### Takeaway
Colab/Kaggle 能免费跑极小的 YOLO（nano、小数据集），但配额不公开、不稳定、难以自动化，不适合当作可靠流程；**Hugging Face Jobs** 是更像"云端 `docker run`"的低价选项（T4-small $0.40/h，按分钟计费，一条 CLI 提交，支持挂载 HF Bucket，需要预付/有余额）。

### Cited Findings
- **Colab（官方 FAQ）**：免费版 GPU "heavily restricted"，GPU/TPU 类型随时间变化；动态用量上限且 "Colab does not publish these limits"；免费版笔记本最长 12 小时（视可用性）；空闲会超时；Pro+ 可连续执行最长 24 小时（需足够 compute units）；后台执行仅在 Google AI 计划较高档提及 — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Colab 价格（third-party，来自搜索摘要，各来源不一致）：Pro $9.99/月 100 compute units；Pro+ $49.99/月（600 CU，另有来源写 500）；T4 每小时消耗 ~1.19–1.76 CU（估算不一） — 搜索摘要引用 [Hivenet Colab guide](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)；未核对 Google 官方定价页
- **Kaggle（third-party，GPUPerHour 2026-09-21，转述官方笔记本文档）**：CPU/GPU 笔记本会话最长 12 小时（TPU 9 小时）；Kaggle 文档没给每周 GPU 配额数字，用户普遍报告约 30 GPU 小时/周（账号设置里可见）；加速器选项随时间变化；硬件常见为 T4 或 P100 — [GPUPerHour free GPUs (2026-09-21; 作者所在公司是 GPU 租赁比价站，有联盟佣金，文章倾向引导付费租用)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)。我直接抓取 kaggle.com/docs/notebooks 时页面正文为空，未能一手确认。
- 同一文章（2026-09-21）：Lightning AI 免费计划 "1 free active Studio (4 hour restarts)" + 50GB 持久存储；Amazon SageMaker Studio Lab 已对新用户关闭；Colab Pro for Education 已不对新用户开放；HF ZeroGPU 每天 5 分钟共享 GPU，仅用于 Gradio demo/推理而非训练 — [GPUPerHour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- **Lightning AI（官方定价页）**：Free 计划 "Up to 30" credits 为一次性（FAQ 称约 50 个 interruptible GPU 小时），注册 5 + 加卡 25，未用 credits 12 个月过期；按量价格 T4 $0.55/h、L4 $0.79/h、A100 40GB $2.19/h、H100 $5.45/h（页面含多组价格未标注来源，取第一组）；免费版每 Studio 1 GPU、并发 2 GPU，A100/H100/H200 会话上限 4 小时，T4/L4/L40S 不限；是否支持 CLI/SDK 提交作业页面未说明 — [Lightning AI pricing (vendor)](https://lightning.ai/pricing)
- **Hugging Face Jobs（官方文档）**：按分钟计费，仅在 Starting/Running 计费，构建阶段不计费；需账户有正余额（PRO/Team/Enterprise 的月度额度可用于 Jobs）；GPU 价格：T4-small $0.40/h、T4-medium $0.60/h、1x L4 $0.80/h、1x L40S $1.80/h、A10G-small $1.00/h、A10G-large $1.50/h、A100-large $2.50/h、H200 $5.00/h、RTX PRO 6000 $2.75/h；10 分钟 a10g-small 约 $0.17；**默认超时 30 分钟**（`--timeout 3h`/`1d` 可改），失败自动重试用 `--attempts n`（重试会从头跑，要续训需把 checkpoint 写到挂载的 bucket）；提交示例 `hf jobs uv run --flavor t4-small --timeout 3h --with ultralytics train.py`；Volumes 可挂载 HF 模型/数据集（只读）、**Storage Buckets（默认读写）**或本地目录（会先同步到私有 bucket）；`-s KEY=VAL` 传加密 secrets；`hf auth login` 认证；Volume 挂载需 `huggingface_hub >= 1.8.0` — [HF Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)、[HF Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration)

### Inferences
- Colab/Kaggle：额度/可得性不公开、不稳定、需要浏览器会话；适合手动、一次性的小实验，不适合"可脚本化、可重复"的偶尔训练流程，更不适合依赖 R2 的自动化（我的推断；Kaggle 的 CLI/API 自动提交能力本次未检索）。
- HF Jobs 对本用例很合适：T4-small 约 $0.40/h，小 YOLO 一次几毛钱到一两美元；训练脚本里用 boto3/rclone 访问 R2（R2 端点是 S3 兼容的，Jobs 容器的出站网络权限文档未明说，推断可用）；作业结束/超时计费自动停止，漏关机风险低。
- 注意 HF 在中国大陆的可达性问题（见第 7 节）会直接影响 HF Jobs 的 CLI 与 hub 依赖。

### Gaps
- Kaggle 的官方周配额数字、Kaggle API 是否能无头提交 GPU 作业：未获官方一手确认。
- Colab 官方价格页/最新 CU 消耗率：未获一手确认。
- HF Jobs 容器是否有不受限制的出站网络、是否支持 R2 直连：文档未明说。
- HF Jobs 的 GPU 排队/容量情况：无数据。

---

## 7. 中国大陆网络可达性：控制面、CLI 是否需要代理

### Takeaway
我没有找到任何厂商或可靠第三方对「从中国大陆访问 SkyPilot/dstack/Modal/RunPod/Vast/Lambda 控制面或 API」的明确实测；能确认的结构性事实是：SkyPilot 的 API server、dstack 的 server 都可以在本机运行，不依赖厂商的中心化控制面，但 CLI 仍需直连各云厂商 API（RunPod/Vast/Lambda 等）以及 R2；Modal 和 HF Jobs 是 SaaS 控制面，必须直连其服务。Hugging Face 在大陆长期不稳定（2024-09 曾被封），R2 官方说明在大陆受限。

### Cited Findings
- SkyPilot 在"本地 API server"模式下运行（无需远程部署）；legacy jobs controller 在云上 VM 或 K8s 中，managed jobs 由其调度 — [managed-jobs.rst](https://raw.githubusercontent.com/skypilot-org/skypilot/master/docs/source/examples/managed-jobs.rst)
- dstack server "can run on your laptop or any environment with access to the cloud"；也有托管版 dstack Sky — [dstack installation](https://dstack.ai/docs/installation/)
- Modal：唯一涉及的来源是第三方聚合站 yangmao.ai，称在中国使用 Modal 需要代理，注册与运行都依赖稳定的国际网络；这是二手、无技术细节、可信度低的说法，Modal 官方没有关于大陆访问的任何声明 — [yangmao.ai Modal page](https://yangmao.ai/en/compute/modal/)
- RunPod：同一聚合站（页面更新时间 2026-06-24）称大陆使用 RunPod 可能需要代理或中转；RunPod 官方博客讨论的是其 Cloudflare 代理 vs 直连的取舍，没有谈到大陆网络；同样低可信 — [yangmao.ai RunPod page](https://yangmao.ai/zh/providers/runpod/)
- Vast.ai：没有找到大陆可达性的信息；官方 FAQ 称可用 $5 起充，也接受加密货币支付（Coinbase、Crypto.com），国内用户信用卡可能受限需自行确认 — 搜索摘要引用 [Vast.ai docs](https://docs.vast.ai/)
- Hugging Face：ChinaTalk 报道 2024-09 Hugging Face 曾被网信办封锁；Rest of World（2026）称大陆开发者虽认为国产平台下载更快，仍倾向通过 VPN 访问 Hugging Face；腾讯云开发者社区 2026-02 文章讨论国内镜像与缓存方案以解决下载慢/ConnectionError；当前准确状态未找到官方确认 — [ChinaTalk](https://www.chinatalk.media/p/hugging-face-blocked-self-castrating)、[Rest of World 2026](https://restofworld.org/2026/china-open-source-ai-hugging-face-modelscope-moark/)、[Tencent Cloud article](https://developer.cloud.tencent.com/article/2630676)
- Cloudflare R2：官方 China Network 文档称 R2 桶不能在中国大陆境内创建、大陆不支持自定义域名，但可经 Global Acceleration 延伸到大陆；经境外服务器的流量存在明显延迟/可靠性问题 — [Cloudflare China Network docs](https://developers.cloudflare.com/china-network/reference/available-products)；API 服务商 FAQ（third-party）称 Cloudflare 在大陆未被封，但高峰期国际出口拥塞，部分运营商路由不佳，质量不稳定 — [APIYI FAQ](https://docs.apiyi.com/en/faq/cdn-download-slow)
- 没有任何 SkyPilot/dstack 的内置后端是中国大陆云（见第 1、2 节）。SkyPilot 有 `kubernetes` 与 `ssh` 两种"自带机器"路径，dstack 有 SSH fleet 与 Kubernetes，理论上可接入国内自有/租用机器 — [SkyPilot clouds dir](https://api.github.com/repos/skypilot-org/skypilot/contents/sky/clouds?ref=master)、[dstack backends](https://dstack.ai/docs/concepts/backends/)

### Inferences
- 可以判断：SkyPilot 和 dstack 的"控制面"是本地进程，不存在一个必须访问的厂商中心服务，因此"控制面能否被访问"不是障碍；真正的瓶颈是你本机到 RunPod/Vast/Lambda/R2 的 API 与 SSH 连通性，以及大文件上传（数据从大陆上传到 R2）。我没有任何数据证明这些在大陆可直连，应当默认需要可靠的出境网络/代理，并在 shell 里设置 `HTTPS_PROXY`（该做法是常规推断，未核实 SkyPilot/dstack 是否完全遵守代理环境变量，特别是 SSH 连接需另行配置 ProxyCommand）。
- Modal 与 HF Jobs 是 SaaS 控制面，必须直连其 API；如果出境网络不稳，其"零运维"优势会被连接问题抵消；而租到的境外 GPU 节点拉取 R2 / PyPI / HF 模型的速度不受你大陆网络影响。
- SkyPilot/dstack 的远端 SSH 隧道（SkyPilot 通过 SSH 执行 setup/run）对跨境链路抖动敏感——这是对 GFW 环境下 SSH 长连接稳定性的一般性推断，本次没有直接来源。

### Gaps
- 无 Modal、RunPod、Vast、Lambda、SkyPilot、dstack 官方或可靠第三方的大陆实测可达性数据。
- 本机到 R2 `*.r2.cloudflarestorage.com` 在各大陆运营商上的实际速度/稳定性：只有定性描述，无实测数据。
- SkyPilot/dstack 是否支持/如何配置 HTTP(S)/SOCKS 代理：未在文档中检索。
- Colab/Kaggle 在大陆的可达性（它们依赖 Google 账号与服务）：未检索（常识上通常需要代理，但无来源）。

---

## 8. 实用结论：从零到「权重在 R2」的最少命令数、示例 YAML/脚本

### Takeaway
命令数（含一次性配置）：HF Jobs ≈ 3、Modal ≈ 4、SkyPilot ≈ 6–8（含 R2 凭据配置）、dstack ≈ 6–8（含 server、fleet）。对"偶尔训练小 YOLO、数据在 R2、用户在大陆"的单人用户，我的判断是：优先 **Modal 或 HF Jobs（无需管机器回收，价差可忽略）**，只有当你想固定使用 RunPod 社区 4090/Vast 这类最便宜的市场、且愿意写好 `--down` 兜底时才用 SkyPilot；dstack 对该场景附加价值最小。自己写直连 provider API 的脚本只有在只用一家提供商且愿意维护 provisioning/teardown 逻辑时才划算。（结论为推断，非实测。）

### Cited Findings
- 以下示例的**事实依据**见第 1–6 节引用；YAML/脚本是我基于这些文档化特性拼装的，**均未实测**，使用前需用 `--dryrun`/小任务验证。
- SkyPilot 的 `sky launch` 有 `--down` 选项和 `-y`；Task YAML 可含 `resources/file_mounts/setup/run`；`file_mounts` 的存储项可写 `source: r2://bucket/path` + `mode: COPY` — [CLI reference](https://docs.skypilot.ai/en/latest/reference/cli.html)、[Storage docs](https://docs.skypilot.ai/en/latest/reference/storage.html)
- Ultralytics 官方提供 Modal 快速入门（含训练函数）— [Ultralytics Modal quickstart](https://docs.ultralytics.com/guides/modal-quickstart)；Modal 官方 YOLO 微调示例 — [Modal finetune_yolo](https://modal.com/docs/examples/finetune_yolo)
- 没有找到官方 Ultralytics 的 SkyPilot 或 dstack 示例 YAML（搜索只找到 NVIDIA NeMo AutoModel 的 SkyPilot 启动器文档，与 YOLO 无关） — [NVIDIA NeMo AutoModel SkyPilot](https://docs.nvidia.com/nemo/automodel/job-launchers/skypilot)

### Inferences（拼装示例，未实测）

**A. SkyPilot（RunPod 或 Vast，R2 用 COPY 取数、脚本回传）**——命令：
```bash
uv tool install --with pip "skypilot[runpod,cloudflare]"      # 或 skypilot[vast,cloudflare]（Vast 注意 vastai-sdk 版本 pin）
runpod config                                                   # 填 RunPod API key
AWS_SHARED_CREDENTIALS_FILE=~/.cloudflare/r2.credentials aws configure --profile r2
mkdir -p ~/.cloudflare && echo <ACCOUNT_ID> > ~/.cloudflare/accountid
sky check                                                       # 确认 RunPod、Cloudflare 已 enabled
sky launch -c yolo --down -y yolo.yaml                          # 作业结束自动销毁；setup 失败时不会销毁，需 sky status -r / sky down -a 兜底
```
```yaml
# yolo.yaml（示意）
resources:
  accelerators: {RTX4090:1, RTX3090:1, A5000:1, L4:1}   # 集合 => 由优化器选最便宜；先用 `sky show-gpus` 确认名称
  disk_size: 100
file_mounts:
  /data:
    source: r2://my-bucket/datasets/mydata                # RunPod/Vast 只能 COPY（只读）
    mode: COPY
setup: |
  pip install ultralytics awscli
run: |
  yolo detect train data=/data/data.yaml model=yolo11n.pt epochs=100 project=/tmp/runs name=exp
  # 输出需自行上传（COPY 只读；MOUNT 在 RunPod/Vast 被禁）。凭据路径是否已同步到远端需验证：
  AWS_SHARED_CREDENTIALS_FILE=~/.cloudflare/r2.credentials aws s3 sync /tmp/runs/exp \
    s3://my-bucket/runs/exp --endpoint-url https://$(cat ~/.cloudflare/accountid).r2.cloudflarestorage.com --profile r2
```
（依据：RunPod/Vast 禁用 STORAGE_MOUNTING、COPY 不触发该检查、R2CloudStorage 用 aws cli + 凭据文件 + endpoint——见第 1 节。）

**B. dstack（RunPod/Vast，无存储抽象）**——命令：
```bash
uv tool install "dstack[all]" -U
dstack server                       # 另开终端常驻；先在 ~/.dstack/server/config.yml 写 runpod/vastai 的 API key
dstack project add --name main --url http://127.0.0.1:3000 --token <admin-token>
dstack apply -f fleet.dstack.yml    # 先建 fleet
dstack apply -f train.dstack.yml
```
```yaml
# train.dstack.yml（示意；env 透传与 image 字段按 dstack 文档核对）
type: task
name: yolo-train
python: "3.11"
env:
  - R2_ACCESS_KEY_ID
  - R2_SECRET_ACCESS_KEY
  - R2_ENDPOINT                      # https://<ACCOUNT_ID>.r2.cloudflarestorage.com
commands:
  - pip install ultralytics awscli
  - AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY aws s3 sync s3://my-bucket/datasets/mydata /data --endpoint-url $R2_ENDPOINT
  - yolo detect train data=/data/data.yaml model=yolo11n.pt epochs=100 project=/tmp/runs name=exp
  - AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY aws s3 sync /tmp/runs/exp s3://my-bucket/runs/exp --endpoint-url $R2_ENDPOINT
resources:
  gpu: 24GB
max_duration: 4h                      # 默认关闭，务必显式设置
max_price: 0.8
```
（依据：第 2 节——快速入门需 server + fleet；`max_duration` 默认 off；Vast/RunPod 无 idle 回收；无 R2 挂载。）

**C. Modal**——命令：
```bash
pip install modal
modal token new                                              # 浏览器认证
modal secret create r2-secret AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
modal run --detach train_yolo.py                              # 权重由脚本写到 R2 挂载点
```
```python
# train_yolo.py（示意；基于 Ultralytics 官方 Modal 指南 + CloudBucketMount 文档拼装；volumes 传入方式请对照 Modal 文档）
import modal
app = modal.App("yolo-train")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0").pip_install("ultralytics"))
r2 = modal.CloudBucketMount("my-bucket",
        bucket_endpoint_url="https://<ACCOUNT_ID>.r2.cloudflarestorage.com",
        secret=modal.Secret.from_name("r2-secret"))
@app.function(image=image, gpu="T4", timeout=4*3600, volumes={"/r2": r2})
def train():
    import shutil
    from ultralytics import YOLO
    YOLO("yolo11n.pt").train(data="/r2/datasets/mydata/data.yaml", epochs=100, project="/tmp/runs", name="exp")
    shutil.copytree("/tmp/runs/exp", "/r2/runs/exp")   # 顺序写入 R2 挂载点（Mountpoint 不支持随机写/追加）
@app.local_entrypoint()
def main(): train.remote()
```
（依据：第 4 节。）注意：数据集在 R2 的挂载读取多小文件的性能未验证；可在函数开头先把数据集 `shutil.copytree` 到本地盘再训练。

**D. Hugging Face Jobs**——命令：
```bash
pip install -U "huggingface_hub>=1.8"
hf auth login                                                 # 账户需有正余额
hf jobs uv run --flavor t4-small --timeout 3h --with ultralytics --with boto3 \
   -s R2_KEY_ID=... -s R2_SECRET=... train.py                 # train.py 内用 boto3 从 R2 拉数据、训练、再传回
```
（依据：第 6 节。）

**成本/价位速查（均为 vendor 页面，日期见各节）**：Modal T4 ≈ $0.59/h；HF Jobs T4-small $0.40/h；RunPod 社区 4090 $0.34/h、3090 $0.22/h、A5000 $0.16/h；Lambda A10 $1.29/h（2026-11-16 生效价）；Lightning T4 $0.55/h。

### Gaps
- 以上所有 YAML/脚本均未执行验证；dstack `env` 透传语法、Modal `volumes=` 传 `CloudBucketMount` 的写法、SkyPilot 远端是否带有 R2 凭据文件，需要用户或后续研究实测。
- 没有找到"一人偶尔训练、直连 RunPod/Vast API 的脚本 vs 用编排层"的公开对比；"直接脚本"一侧的结论完全是基于第 1–2 节功能差异的推断。
- 大陆网络条件下各条路径的真实成功率没有数据（见第 7 节）。
- Ultralytics 的 AGPL-3.0 许可对商用的影响（Modal 示例页提示需核对）不在本次范围内。
