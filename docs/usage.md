# dgpu 使用说明

dgpu 用一条命令在租来的云 GPU 上训练模型：租机器，拷代码和数据，跑训练，把结果传回 Cloudflare R2，然后销毁机器停止计费。支持 Vast.ai 和 AutoDL 两个平台，单机（可以多卡）。

**dgpu 还在开发中，下面的流程尚未在真实平台验证。** 命令的准确参数以 `dgpu --help` 和 `dgpu <命令> --help` 为准。设计细节见 [设计文档](dgpu-design.md)，日常使用只看本文就够了。

```
你的电脑                          R2（你的 bucket）               租来的 GPU 机器
dgpu submit ──代码、任务描述──▶  dgpu/jobs/<任务ID>/  ◀──拉代码和数据、跑训练、定期上传输出
dgpu wait / logs / pull ◀──状态、日志、输出──                   结束后销毁自己
```

任务状态都存在 R2 上，dgpu 不在本机保存状态。换一台电脑，配置好同一个 bucket 也能查。

## 1. 安装

需要 [uv](https://docs.astral.sh/uv/)。在 dgpu 仓库目录里：

```bash
uv tool install .          # 装成全局命令 dgpu
dgpu --help
```

代码更新后重新安装：`git pull && uv tool install --reinstall .`。

不想安装也可以，在仓库目录里用 `uv run dgpu ...`，比如 `uv run dgpu doctor`。

想让 Claude Code 帮你提交和管理任务，把 Skill 链接过去：

```bash
ln -s "$PWD/skills/dgpu" ~/.claude/skills/dgpu
```

## 2. 一次性准备

### 2.1 R2

1. 在 Cloudflare 控制台打开 R2，创建一个 bucket，例如 `my-dgpu`。
2. 在 R2 的 API 令牌页面创建令牌：权限选“对象读和写”，范围选“仅限指定 bucket”，只勾选这个 bucket。记下 Access Key ID 和 Secret Access Key。
3. 记下账户 ID，endpoint 是 `https://<账户ID>.r2.cloudflarestorage.com`。
4. 数据集也放在这个 bucket 里，例如 `datasets/coco8/`。令牌只能访问这一个 bucket。

dgpu 只使用 bucket 里的 `dgpu/` 目录，存放任务、日志、结果和从本地上传的数据集。

### 2.2 Vast.ai

1. 注册账户，充少量余额（例如 $5），**不要绑定自动充值**。
2. 在控制台的 API Keys 页面创建一个 key。

### 2.3 AutoDL（可选）

1. 需要企业认证：dgpu 用的是 AutoDL 的弹性部署 API，只对企业认证用户开放。
2. 在控制台“设置 → 开发者 Token”里拿到 token。
3. 充少量余额，不要开自动扣款。

AutoDL 到 R2 走跨境网络，速度尚未实测。

### 2.4 写配置文件

```bash
dgpu init                  # 生成 ~/.config/dgpu/config.toml，权限 600
```

用编辑器打开这个文件，把密钥填进去。内容大致如下，以 `dgpu init` 生成的模板为准：

```toml
[stores.r2]
endpoint = "https://<账户ID>.r2.cloudflarestorage.com"
bucket = "<bucket>"
access_key_id = "..."
secret_access_key = "..."

[backends.vast]
api_key = "..."
store = "r2"

[backends.autodl]          # 不用 AutoDL 就不填
token = "..."
store = "r2"
```

填完运行 `dgpu doctor`，它检查配置、R2 读写、平台 API 和余额。全部通过再提交任务。

配置文件的位置，按优先级：

1. 命令行 `dgpu --config <路径> <命令>`，例如 `dgpu --config ~/work/dgpu.toml doctor`；
2. 环境变量 `DGPU_CONFIG=<路径>`；
3. 默认 `~/.config/dgpu/config.toml`。

文件权限不是 600 时 dgpu 拒绝读取，用 `chmod 600 <路径>` 修好。单个密钥也可以用环境变量覆盖：`VAST_API_KEY`、`AUTODL_TOKEN`，以及 `R2_ACCESS_KEY_ID`、`R2_SECRET_ACCESS_KEY`（只作用于名为 `r2` 的存储，即 `[stores.r2]`）。不认识的配置项一律报错，拼错不会被静默忽略。

其余配置项都有默认值，需要时再加（`dgpu init` 生成的模板里有全部配置项，注释掉的就是默认值）：

```toml
startup_timeout_min = 15       # 顶层配置项，必须写在第一个 [表] 之前
lost_after_min = 10

[backends.vast.filters]        # Vast 的筛选条件；也可以直接写在 [backends.vast] 里，二选一
reliability = 0.98
min_inet_down_mbps = 500
max_bandwidth_usd_per_gb = 0.01
datacenter_only = false        # 数据敏感时打开，只用数据中心机器

[backends.autodl]
dc_list = ["westDC2"]          # 地区，默认不限
cuda_v_from = 118              # 118 即 CUDA 11.8

[runner]                       # 下发给机器上 dgpu 的参数；时间类可以写小数
sync_interval_s = 120
heartbeat_interval_s = 60
stall_timeout_min = 60         # 0 关闭卡死检测
final_sync_timeout_min = 20
```

目前所有命令共用一个存储：两个后端的 `store` 要指向同一个 `[stores.*]`，否则 `submit` 会报错。

密钥只放在配置文件或环境变量里：不提交到 git，不写进 `job.yaml`，也不要贴到和 Claude 的对话里。平台的主 key 永远不会被传到租来的机器上。

## 3. 第一次运行：YOLO 示例

仓库里的 [`examples/yolo/`](../examples/yolo/README.md) 是一个完整的例子：用 YOLO26n 在 8 张图的 coco8 上训练 30 轮，最坏花费 $0.40。按它的 README 准备好数据后：

```bash
cd examples/yolo
dgpu offers job.yaml
dgpu submit job.yaml
dgpu wait <任务ID>
dgpu pull <任务ID> ./result   # 输出下载到 ./result/output/
dgpu gc --dry-run
```

## 4. 写自己的任务

### 4.1 job.yaml

```yaml
version: 1
name: my-train
backend: vast
image:
  vast: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
  autodl: <AutoDL 镜像 UUID>
code: .
setup: python -m pip install -r requirements.txt
run: python train.py
data:
  - r2://my-dgpu/datasets/coco8
  - ./datasets/small-set
gpu:
  names: [RTX 3090, RTX 4090]
  count: 1
limits:
  max_hours: 4
  max_usd_per_hour: 0.40
  max_cny_per_hour: 2.50
env:
  EPOCHS: "100"
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `version` | 是 | 写 `1`。不认识的版本会被拒绝 |
| `name` | 是 | 任务名，任务 ID 会以它开头，例如 `my-train-261110083000-a1b2` |
| `backend` | 是 | `vast` 或 `autodl`。不写时提交必须带 `--backend`；写了也可以用 `--backend` 覆盖 |
| `image.vast` | 用 Vast 时 | 公开的 Docker 镜像，必须写固定 tag（或 `@sha256:` 摘要），不能用 `latest` 或不写 tag |
| `image.autodl` | 用 AutoDL 时 | AutoDL 平台上的镜像 UUID。公共镜像比较旧，需要新环境就在 AutoDL 网站上保存私有镜像 |
| `code` | 否 | 要打包上传的代码目录，相对 `job.yaml`，默认 `.`（`job.yaml` 所在目录） |
| `setup` | 否 | 训练前执行一次，例如装依赖 |
| `run` | 是 | 训练命令。退出码 0 表示成功 |
| `data` | 否 | 数据列表，见第 7 节 |
| `gpu.names` | 是 | 可以接受的 GPU 型号，任一即可 |
| `gpu.count` | 否 | 每台机器几张卡，默认 1 |
| `limits.max_hours` | 是 | 最长运行时间（可以是小数），到时间强制结束 |
| `limits.max_usd_per_hour` | 用 Vast 时 | 整台机器每小时的价格上限（全部卡加磁盘，不含流量费），美元 |
| `limits.max_cny_per_hour` | 用 AutoDL 时 | 整个容器每小时的价格上限（不是每卡），人民币 |
| `env` | 否 | 传给训练代码的环境变量，值写成字符串（`"100"`）。不要放密钥；以 `DGPU_`、`AWS_`、`R2_`、`RCLONE_CONFIG_`、`CONTAINER_API_KEY` 开头的名字会被拒绝 |

只用一个后端时，`image` 也可以直接写成一个字符串。不认识的字段、同一个字段写两次都会报错。

镜像里必须有 `python3`（3.8 或更新）和 `bash`。

代码怎么打包：代码目录在 git 仓库里时，打包 git 跟踪的和没被 `.gitignore` 忽略的文件；不在 git 仓库里时，按代码目录下的 `.dgpuignore` 排除。`.git`、`.env*`、`*.pem`、`*.key`、`__pycache__`、`.venv` 总是排除；`data` 里的本地目录如果在代码目录里，也不会再打进代码包（它们单独上传）。要打包的文件合计超过 200 MB（按未压缩大小）会报错：数据请放到 `data`，权重和输出不要放在代码目录里。

### 4.2 训练代码

dgpu 在机器上设置这些环境变量：

| 环境变量 | 含义 |
|---|---|
| `DGPU_DATA` | 数据目录，`data` 里的每一项在 `$DGPU_DATA/<目录名>` |
| `DGPU_OUTPUT` | 输出目录，大约每 2 分钟上传一次，结束时再上传一次 |
| `DGPU_CKPT` | checkpoint 目录，同样定期上传；重试时恢复到这里 |
| `DGPU_RESUME` | `DGPU_CKPT` 里有恢复出来的文件时为 `1` |
| `DGPU_JOB_ID` / `DGPU_ATTEMPT` | 任务 ID、第几次尝试 |

训练代码要做到：

1. 从 `$DGPU_DATA/<目录名>` 读数据，结果写进 `$DGPU_OUTPUT`。
2. checkpoint 写进 `$DGPU_CKPT`：先写成 `xxx.tmp`，写完再 rename 成正式文件名。dgpu 上传时跳过 `*.tmp`，这样存储里不会出现写了一半的文件。
3. `DGPU_RESUME=1` 时从 `$DGPU_CKPT` 加载，接着训练。
4. 成功时退出码 0。

训练进程拿不到 R2 的密钥，也不需要：上传下载都由 dgpu 完成。完整的例子见 [`examples/yolo/train.py`](../examples/yolo/train.py)。

提交前，最好先在本机用很小的数据跑一遍：把 `DGPU_DATA`、`DGPU_OUTPUT`、`DGPU_CKPT` 设成本地目录再运行训练脚本。代码错误在本机发现不花钱。

## 5. 常用流程

```bash
dgpu offers job.yaml          # 1. 看报价：有哪些机器、每小时多少钱。不花钱
dgpu submit job.yaml          # 2. 提交：打印费用上限，租机器，等机器跑起来后输出任务 ID
dgpu wait <任务ID>            # 3. 等结果：前台等到结束，退出码表示结果
dgpu pull <任务ID> ./result   # 4. 取结果：下载到 ./result/output/
dgpu gc --dry-run             # 5. 确认没有遗留的机器在计费
```

- `submit` 要等机器启动、拉完镜像、机器上的 dgpu 检查过 GPU、磁盘和存储（preflight）之后才返回，可能要十几分钟。检查不通过就自动销毁这台、换下一台。中途按 Ctrl-C 会销毁已经创建的机器。
- `submit --wait` 提交后直接接着等，等同于再运行 `dgpu wait`。
- `wait` 只是在等，按 Ctrl-C 退出不影响任务。之后随时可以再 `wait`，或用 `status`、`logs` 查看。

## 6. 命令一览

| 命令 | 作用 | 例子 |
|---|---|---|
| `init` | 生成配置文件模板（权限 600），已存在时不覆盖 | `dgpu init` |
| `doctor` | 检查配置、R2 读写、平台 API、余额；有一项不通过时退出码 1 | `dgpu doctor` |
| `offers` | 列出候选机器和预估费用，不花钱；`--limit` 显示几个（默认 10），`--json` 输出 JSON | `dgpu offers job.yaml --backend autodl` |
| `submit` | 打包代码、创建机器，preflight 通过后返回任务 ID；`--wait`（可配 `--retries`）接着等到结束 | `dgpu submit job.yaml --backend vast` |
| `wait` | 等任务结束；失联时可以自动重试 N 次 | `dgpu wait <任务ID> --retries 1` |
| `status` | 查看任务状态和失败原因；`--json` 输出 JSON | `dgpu status <任务ID>` |
| `logs` | 查看训练日志，`-f` 持续跟随；`--attempt N` 看第 N 次尝试 | `dgpu logs <任务ID> -f` |
| `list` | 列出所有任务，最新的在前；`--limit`（默认 30）、`--json` | `dgpu list` |
| `retry` | 换一台机器，从 checkpoint 接着跑 | `dgpu retry <任务ID>` |
| `cancel` | 取消任务：先上传已有结果再销毁；`--force` 直接销毁 | `dgpu cancel <任务ID> --force` |
| `pull` | 下载任务的输出（`$DGPU_OUTPUT` 的内容）到 `<目录>/output/`，不写目录时是 `./<任务ID>/output/`；`--ckpt` 同时下载 checkpoint 到 `<目录>/ckpt/` | `dgpu pull <任务ID> ./result --ckpt` |
| `gc` | 清理遗留的机器；`--dry-run` 只列出不销毁，`--json` 输出 JSON；有销毁失败的机器时退出码 1 | `dgpu gc --dry-run` |

所有命令都可以加 `--config <路径>` 指定配置文件，写在命令名前面：`dgpu --config <路径> list`。本地数据合计超过 5 GB（`local_data_confirm_gb`）时 `submit` 会先问是否上传，加 `--yes`（`-y`）跳过确认。用法错误、配置错误等可预期的错误打印一句话说明，退出码 2。

任务状态：

| 状态 | 意思 |
|---|---|
| `PROVISIONING` | 正在找机器、创建机器 |
| `STARTING` | 机器已创建，等 dgpu 在机器上启动 |
| `RUNNING` | 下载数据、执行 setup、训练中 |
| `UPLOADING` | 训练结束，正在上传最终结果 |
| `SUCCEEDED` | 成功 |
| `FAILED` | 失败，`dgpu status` 显示原因 |
| `CANCELLED` | 已取消 |
| `TIMED_OUT` | 超过 `max_hours` 被停止 |
| `LOST` | 超过 10 分钟没有心跳：机器宕机、断网或余额耗尽 |

`dgpu wait` 的退出码：

| 退出码 | 状态 |
|---|---|
| 0 | SUCCEEDED |
| 1 | FAILED |
| 2 | 命令用法错误 |
| 3 | TIMED_OUT |
| 4 | CANCELLED |
| 5 | LOST |

常见的失败原因：

| 原因 | 意思 | 怎么办 |
|---|---|---|
| `no_offer` | 没有符合条件的机器 | 提高价格上限或增加 GPU 型号 |
| `startup` | 机器一直没启动 | 换更小、更常见的镜像；AutoDL 换地区 |
| `runner_timeout` | 机器启动了，但 dgpu 没跑起来 | 镜像缺 `python3` 或 `bash`，换镜像 |
| `preflight` | 没有 GPU、磁盘不够或连不上 R2 | 看 `dgpu status` 的说明 |
| `data_sync` | 代码或数据下载失败 | 检查 `r2://` 路径和令牌权限 |
| `setup` | `setup` 命令失败 | `dgpu logs` 看报错，改好后重新 `submit` |
| `exit_code` | 训练命令非 0 退出 | `dgpu logs` 看报错，改好代码后重新 `submit`。`retry` 用的还是旧代码 |
| `stalled` | GPU 利用率为 0，且 60 分钟没有新日志 | 看日志最后几行，常见是数据加载卡住 |
| `runner_crashed` | 机器上的 dgpu 进程崩溃，看门狗收尾 | `dgpu retry <任务ID>` 从 checkpoint 接着跑 |
| `signalled` | 机器上的 dgpu 收到停止信号（容器被平台停止等），已提前收尾上传 | `dgpu retry <任务ID>` 从 checkpoint 接着跑 |
| `final_sync` | 最后一次上传失败 | `dgpu status` 列出没传上去的文件，这些文件已经丢失 |

## 7. 数据怎么到机器上

`data` 里每一项可以是 R2 路径，也可以是本地目录：

- **R2 路径** `r2://<bucket>/<路径>`：机器启动后把它拷到 `$DGPU_DATA/<路径最后一段>`。例如 `r2://my-dgpu/datasets/coco8` 到 `$DGPU_DATA/coco8`。机器用配置里那个存储的令牌读取：bucket 不是配置里的 bucket 时，`offers` 和 `submit` 会提醒你确认令牌能读它（不会拒绝），读不了的话任务以 `FAILED(data_sync)` 结束。
- **本地目录**，例如 `./datasets/small-set`：`submit` 时先上传到你 bucket 的 `dgpu/datasets/<内容哈希>/`，机器再从那里拉取。内容没变就不重传。合计超过 5 GB 时先告诉你上传量并要求确认。
- **单个文件最大 5 GiB**：dgpu 只做单段上传，本地目录里有超过 5 GiB 的文件时，`submit` 在上传任何东西之前就报错。这种数据先用 rclone 传到 bucket（如 `rclone copy ./big-set <remote>:<bucket>/datasets/big-set`），再在 `data` 里写成 `r2://<bucket>/datasets/big-set`。
- 目录名（路径最后一段，也就是 `$DGPU_DATA` 下的名字）只能用英文字母、数字、点、下划线和连字符，两项不能同名。

为什么不直接挂载 R2：租来的机器是没有特权的普通容器，大多不能用 FUSE 挂载；就算能挂，训练时一个个小文件走网络读取也会拖慢训练。所以 dgpu 在训练开始前把数据一次拷到本地盘，训练代码看到的就是普通目录。

其他要注意的：

- 很多小文件（比如几十万张图）先打成 tar 包放到 R2，在 `setup` 里解压，拷得更快。
- 磁盘：Vast 按数据量自动估算；AutoDL 数据盘固定 50 GB，数据要小于它。
- dgpu 不支持从你的电脑直接传数据到机器上，都经过 R2。

## 8. 怎么保证停止计费

按小时计费的机器只要没销毁，就一直花钱。dgpu 用几层办法兜底：

1. **正常结束**：不管成功、失败、超时还是取消，机器上的 dgpu 都先上传结果，再销毁机器。
2. **看门狗**：机器上另有一个独立进程。dgpu 的进程崩溃了，或者超过 `max_hours` 很久还没结束，它负责销毁。
3. **启动阶段**：`dgpu submit` 期间机器起不来，或者你按了 Ctrl-C，`submit` 会销毁它。
4. **`dgpu gc`**：电脑中途关机、机器自毁失败等情况留下的机器，由 `gc` 清理。
5. **账户余额**：两个平台都只充少量余额、不绑自动扣款。以上都失效时，最多花完余额。

Vast 上，机器用平台为这台机器单独发的 key 销毁自己。AutoDL 上，训练命令结束后容器就停止，dgpu 再删除部署记录。这两种销毁方式都尚未在真实平台验证。

`dgpu gc` 只处理名字（Vast 的 label、AutoDL 的部署名）形如 `dgpu-<任务ID>-a<次数>` 的机器，不碰你的其他机器。它清理这些情况：任务已经结束、超过了截止时间、心跳中断超过 30 分钟、启动卡住。

- `dgpu gc --dry-run`：只列出会被清理的机器，不销毁。建议每个任务结束后、任何报错后都跑一次。
- `dgpu gc`：真正销毁。
- 可以用 launchd 或 cron 每 10 分钟自动跑一次 `dgpu gc`（可选；电脑休眠时不会运行）。

单个任务的最坏花费约为“每小时价格上限 × `max_hours`”加上少量流量费。`submit` 会先打印这个上限，并检查余额够不够。

## 9. 常见问题

**训练代码要改多少？**
读数据和写输出改成用环境变量里的目录；要支持中断续跑，再按第 4.2 节保存和加载 checkpoint。不需要接 R2 的 SDK。

**机器中途挂了怎么办？**
状态会变成 `LOST`。运行 `dgpu retry <任务ID>`，dgpu 换一台机器，从 R2 上的 checkpoint 接着跑。也可以提交后用 `dgpu wait <任务ID> --retries 1`，失联时自动重试一次。前提是训练代码支持 `DGPU_RESUME`。

**超时了还想接着训练？**
`TIMED_OUT` 后，已上传的输出和 checkpoint 都在。先 `dgpu pull` 拿到已有结果，再 `dgpu retry <任务ID>` 接着跑。

**改了代码，可以 `retry` 吗？**
不行。`retry` 用的是提交时打包的代码。改了代码要重新 `submit`。

**换一台电脑能查任务吗？**
能。状态都在 R2 上，配置好同一个 bucket 就行。

**选 Vast 还是 AutoDL？**
默认 Vast：机器多，用美元付款，镜像可以用 Docker Hub 上的任意镜像。AutoDL 用人民币付款，但需要企业认证，镜像只能用平台上的，到 R2 是跨境网络（尚未实测，可能需要改用国内对象存储）。

**数据安全吗？**
租来的机器的房东能看到磁盘上的数据和 dgpu 用的 R2 令牌。所以令牌只授权一个 bucket。数据敏感时，在配置里打开 Vast 的“只用数据中心机器”（`datacenter_only`），或者不要放到租来的机器上。

**为什么镜像不能用 `latest`？**
`latest` 会悄悄变化，重试时可能换成另一个环境，checkpoint 就可能加载不了。写固定 tag 可以保证每次都一样。

**日志在哪？**
`dgpu logs <任务ID>`。R2 上的位置是 `dgpu/jobs/<任务ID>/logs/`。

**能多卡、多机吗？**
单机多卡可以，设置 `gpu.count`。多机分布式不支持。

**用 Ultralytics 训练的模型能商用吗？**
Ultralytics 的立场是需要企业许可，见 [`examples/yolo/README.md`](../examples/yolo/README.md) 的许可证一节。
