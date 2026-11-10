---
name: dgpu
description: 用 dgpu 在按需租用的云 GPU（Vast.ai、AutoDL）上训练任意模型，数据放 Cloudflare R2：检查配置、写 job.yaml、让训练代码按约定读数据和续跑、先报价再提交、后台等待结果、查日志和状态、取回结果、确认没有遗留的计费机器。用户要在云 GPU / Vast / AutoDL 上训练模型，提交、查看、取消、重试 dgpu 任务，或写 job.yaml 时使用。
---

# dgpu：在租来的云 GPU 上训练

`dgpu submit job.yaml` 会租一台 GPU 机器，拷代码和数据，跑训练，把输出和 checkpoint 传回 R2，然后销毁机器。成功、失败、超时都会销毁。任务状态存在 R2 上，换一台电脑也能查。

dgpu 还在开发中，尚未在真实平台验证。命令或参数和这里写的不一致时，以 `dgpu --help` 为准，并告诉用户。

## 1. 先检查环境

```bash
command -v dgpu      # 找不到：在 dgpu 仓库里用 uv run dgpu ...，或请用户 uv tool install .
dgpu doctor          # 检查配置、R2 读写、平台 API、余额
```

- 配置文件不存在：运行 `dgpu init` 生成模板（默认 `~/.config/dgpu/config.toml`，权限 600），然后请用户**自己**打开文件填写：R2 的 endpoint、bucket、access key；Vast API key；AutoDL token（只用一个平台就只填一个）。填完再跑 `dgpu doctor`。
- 配置文件在别处：`dgpu --config <路径> doctor`，或设置环境变量 `DGPU_CONFIG=<路径>`。
- **绝不**让用户把密钥贴到对话里；**绝不** `cat`、读取或打印配置文件；密钥不写进 job.yaml、`env`、命令行。
- `dgpu doctor` 全部通过之前，不提交任务。

## 2. 写 job.yaml

```yaml
version: 1                      # 固定写 1
name: my-train                  # 任务名，任务 ID 形如 my-train-261110083000-a1b2
backend: vast                   # vast 或 autodl；提交时 --backend 可以覆盖
image:
  vast: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime   # 必须是固定 tag，不能用 latest
  autodl: <AutoDL 镜像 UUID>     # 只能用 AutoDL 平台上的镜像
code: .                         # 要打包的代码目录（相对 job.yaml）
setup: python -m pip install -r requirements.txt       # 可选，训练前执行一次
run: python train.py            # 训练命令，退出码 0 表示成功
data:                           # 可选
  - r2://my-bucket/datasets/coco8   # 已在 R2 上：拷到 $DGPU_DATA/coco8
  - ./datasets/small-set            # 本地目录：提交时先传到 R2，再拷到 $DGPU_DATA/small-set
gpu:
  names: [RTX 3090, RTX 4090]   # 任一即可
  count: 1
limits:                         # 必填，没有默认值
  max_hours: 4                  # 到时间强制结束，状态 TIMED_OUT
  max_usd_per_hour: 0.40        # Vast，美元
  max_cny_per_hour: 2.50        # AutoDL，人民币
env:                            # 可选，传给训练代码；值写成字符串，不放密钥
  EPOCHS: "100"
```

- 必填：`version`、`name`、`backend`、`image`、`code`、`run`、`gpu`、`limits`。`image` 和价格上限只填要用的后端即可。
- 镜像里要有 `python3`（3.8 或更新）和 `bash`。Vast 用公开 Docker 镜像，写固定 tag。AutoDL 填镜像 UUID；公共镜像较旧（如 CUDA 11.8、Python 3.8），要新环境得请用户在 AutoDL 网站保存私有镜像。
- 代码打包：git 仓库里打包 git 跟踪的和没被忽略的文件，否则按 `.dgpuignore`；`.git`、`.env*`、`*.pem`、`*.key` 总是排除。超过 200 MB 会报错，所以数据、权重、输出不要放在代码目录里。
- `data` 每一项拷到 `$DGPU_DATA/<路径最后一段>`。大量小文件最好先打成 tar，在 `setup` 里解压。
- 写完先跑 `dgpu offers job.yaml`：它会检查 job.yaml，不花钱。

## 3. 训练代码要做的事

| 环境变量 | 含义 |
|---|---|
| `DGPU_DATA` | 数据目录，已拷到本地盘 |
| `DGPU_OUTPUT` | 输出目录，定期上传，结束后用 `dgpu pull` 取回 |
| `DGPU_CKPT` | checkpoint 目录，定期上传，重试时恢复 |
| `DGPU_RESUME` | `DGPU_CKPT` 里有恢复出来的文件时为 `1` |
| `DGPU_JOB_ID` / `DGPU_ATTEMPT` | 任务 ID、第几次尝试 |

1. 从 `$DGPU_DATA/<名字>` 读数据，结果写进 `$DGPU_OUTPUT`。
2. checkpoint 写进 `$DGPU_CKPT`，**先写 `xxx.tmp`，再 rename 成正式文件名**。上传时跳过 `*.tmp`，存储里只会有完整的文件。
3. `DGPU_RESUME=1` 时从 `$DGPU_CKPT` 加载并接着训练；加载失败时从头开始，并打印原因。
4. 成功退出码 0，失败非 0。训练进程拿不到存储密钥，也不需要。

```python
ckpt = Path(os.environ["DGPU_CKPT"]) / "last.pt"
start = 0
if os.environ.get("DGPU_RESUME") == "1" and ckpt.exists():
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
    start = state["epoch"] + 1
for epoch in range(start, epochs):
    train_one_epoch(...)
    tmp = ckpt.with_name("last.pt.tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, tmp)
    os.replace(tmp, ckpt)
torch.save(model.state_dict(), Path(os.environ["DGPU_OUTPUT"]) / "model.pt")
```

Ultralytics YOLO：直接用 dgpu 仓库的 `examples/yolo/`，`train.py` 已经处理好以上几条，改 `job.yaml` 里的数据和镜像即可。

提交前建议先在本机用很小的数据跑通训练脚本：把 `DGPU_DATA`、`DGPU_OUTPUT`、`DGPU_CKPT` 设成本地目录再运行。代码错误在本机发现不花钱。

## 4. 标准流程

1. **报价**：`dgpu offers job.yaml`（AutoDL 加 `--backend autodl`）。把 GPU 型号、每小时价格、预估总价告诉用户，并说明最坏花费约为“价格上限 × max_hours”。
2. **确认上限**：和用户确认 `max_hours` 和价格上限，用户同意后才提交。新代码先用小的 epoch 和 `max_hours` 跑一次。
3. **提交**：`dgpu submit job.yaml`。它要等机器启动、发出第一次心跳才返回任务 ID，可能要十几分钟，所以用 Bash 的 `run_in_background: true` 运行。不要随手打断：打断会销毁已创建的机器，任务也就结束了。还没返回时，`dgpu list` 能看到正在启动的任务。
   - 本地数据超过 5 GB 时 `submit` 会要求确认，后台运行没法回答。先用 `du -sh <目录>` 看大小告诉用户，用户同意后加 `--yes`。
4. **等结果**：`dgpu wait <job>`，同样用 `run_in_background: true`，结束时会收到通知，按第 5 节的退出码处理。不要在前台反复轮询。
5. **用户问进度时**：`dgpu status <job>`、`dgpu logs <job>`；`dgpu list` 列出所有任务。`dgpu logs <job> -f` 会一直跟随不退出，前台不要用。
6. **取结果**：`dgpu pull <job> <本地目录>`，下载 `$DGPU_OUTPUT` 里的内容。
7. **收尾**：`dgpu gc --dry-run`，确认没有遗留机器。列出了机器就告诉用户，用户同意后运行 `dgpu gc`。

用户要停止任务：`dgpu cancel <job>`，机器上的 dgpu 会先上传已有结果再销毁机器。它不响应时 `dgpu cancel <job> --force` 直接销毁，没上传的结果会丢失，先告诉用户。

## 5. 按结果处理

`dgpu wait` 的退出码：

| 退出码 | 状态 | 怎么做 |
|---|---|---|
| 0 | SUCCEEDED | `dgpu pull`，然后 `dgpu gc --dry-run` |
| 1 | FAILED | `dgpu status <job>` 看原因（下表），`dgpu logs <job>` 看最后几十行 |
| 2 | — | 命令用错了（参数不对、任务不存在），检查命令 |
| 3 | TIMED_OUT | 到了 `max_hours` 被停止。已上传的输出和 checkpoint 还在：先 `dgpu pull`。要接着训练就问用户，同意后 `dgpu retry <job>` |
| 4 | CANCELLED | 任务被取消，告诉用户 |
| 5 | LOST | 心跳中断：机器宕机、断网或余额耗尽。先 `dgpu gc --dry-run` 和 `dgpu doctor`（看余额），问用户后 `dgpu retry <job>` |

`retry` 会换一台机器，从 R2 上的 checkpoint 接着跑，用的是提交时打包的代码。用户事先同意自动重试时，可以用 `dgpu wait <job> --retries 1`，失联时自动重试一次。

FAILED 的原因：

| 原因 | 意思 | 怎么做 |
|---|---|---|
| `no_offer` | 没有符合条件的机器，或换机次数用完 | 告诉用户卡在哪个条件；放宽价格或加 GPU 型号要用户同意 |
| `startup` | 机器一直没启动（镜像太大、调度太慢） | 换更小、更常见的镜像；AutoDL 换地区 |
| `runner_timeout` | 机器启动了，但 dgpu 没跑起来 | 镜像多半缺 `python3` 或 `bash`，换镜像 |
| `preflight` | 没有 GPU、磁盘不够或连不上存储 | 看 message；AutoDL 连不上 R2 时告诉用户 |
| `data_sync` | 代码或数据下载失败 | 检查 `r2://` 路径是否存在、R2 token 能不能读这个 bucket |
| `setup` | `setup` 命令失败 | 看日志里的报错，改 job.yaml 后重新 `submit` |
| `exit_code` | 训练命令非 0 退出 | 看日志，修代码后重新 `submit`（`retry` 还是旧代码） |
| `stalled` | GPU 利用率为 0，且 60 分钟没有新日志 | 看日志最后几行，常见是数据加载卡住或死锁 |
| `runner_crashed` | dgpu 在机器上的进程崩溃了 | 问用户后 `dgpu retry <job>` |
| `final_sync` | 最后一次上传失败 | `dgpu status` 列出没传上去的文件，它们已随机器销毁，如实告诉用户 |
| `fenced` | 已经有更新的一次尝试，这一次自己退出了 | 看最新那次尝试的状态 |

**任何报错之后都运行一次 `dgpu gc --dry-run`。**

## 6. 安全规则

- 每个任务都要有 `max_hours` 和所用后端的价格上限。
- 不经用户同意，不提高 `max_hours`、价格上限、GPU 数量，不加 `--retries`，不 `retry`。这些都要花钱。
- 提交前一定先 `dgpu offers`，把价格告诉用户。
- 每个任务结束后、任何报错之后，运行 `dgpu gc --dry-run`。`dgpu gc` 只清理带 dgpu 标签的机器，不碰用户的其他机器。
- 不读、不打印配置文件，不在对话里收密钥。
- 租来的机器的房东能看到磁盘上的数据。数据敏感时先提醒用户。

选 Vast 还是 AutoDL：

| | Vast | AutoDL |
|---|---|---|
| 镜像 | 公开 Docker 镜像，固定 tag | AutoDL 平台上的镜像 UUID |
| 价格上限 | `max_usd_per_hour`（美元） | `max_cny_per_hour`（人民币） |
| 到 R2 的网络 | 机器多在境外，预计较快（尚未实测） | 跨境，速度未知（尚未实测） |
| 机器怎么销毁 | 机器上的 dgpu 调平台 API 销毁自己；失败时 `dgpu gc` 兜底 | 训练命令结束容器就停止，dgpu 再删除部署记录；`dgpu gc` 兜底 |
| 门槛 | API key，充少量美元 | 企业认证、开发者 token |
| 磁盘 | 按数据量估算 | 数据盘固定 50 GB |

默认用 Vast。用户要人民币付款，或数据已在国内时，考虑 AutoDL。

## 7. 排查

| 现象 | 处理 |
|---|---|
| `dgpu: command not found` | 在 dgpu 仓库里用 `uv run dgpu ...`，或请用户 `uv tool install .` |
| `doctor` 说配置文件权限不是 600 | `chmod 600 <配置文件路径>` |
| `doctor` 说 R2、Vast 或 AutoDL 认证失败 | 请用户检查配置里对应的项；不要读文件 |
| `offers` 没有结果 | 价格上限太低、GPU 型号太少或没有库存。告诉用户，由用户决定是否放宽 |
| 提交时说镜像 tag 不合法 | 写固定 tag，不用 `latest` |
| 代码包超过 200 MB | 把数据、权重、输出移出代码目录，或写进 `.gitignore` / `.dgpuignore` |
| 连续两次 `runner_timeout` | 镜像缺 `python3` 或 `bash`，换镜像 |
| 日志里有 `libGL.so.1` | OpenCV 缺系统库：`setup` 里装 `libgl1`，或改装 `opencv-python-headless` |
| 续跑后又从第 0 轮开始 | 训练代码没处理 `DGPU_RESUME`，或 checkpoint 没写进 `DGPU_CKPT` |
| AutoDL 上下载很慢或失败 | 跨境网络问题，告诉用户；可能要改用国内对象存储 |
