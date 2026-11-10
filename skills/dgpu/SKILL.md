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
- `dgpu doctor` 全部通过之前，不提交任务。`[警告]` 不算失败：比如 Vast 显示“余额无法查询（key 没有读取账户信息的权限）”时 key 是好的，只是查不到余额；告诉用户提交前要自己确认余额。

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
# run_as_root: true             # 可选，默认 false，见下
```

- 训练默认以无特权用户运行：`setup` 里 `pip install` 可以用，`apt install` 会失败。需要系统包时优先换一个装好的镜像；确实要 `apt install` 才写 `run_as_root: true`，并先告诉用户代价：训练代码能读到 R2 凭据、能杀掉负责销毁机器的进程。不经用户同意不打开它。

- 必填：`version`、`name`、`backend`（或提交时 `--backend`）、`image`、`run`、`gpu`、`limits`。`code` 默认 `.`。`image` 和价格上限只填要用的后端即可；价格上限是整台机器每小时的总价（全部卡加磁盘；AutoDL 是整个容器，不是每卡）。
- 镜像里要有 `python3`（3.8 或更新）和 `bash`。Vast 用公开 Docker 镜像，写固定 tag。AutoDL 填镜像 UUID；公共镜像较旧（如 CUDA 11.8、Python 3.8），要新环境得请用户在 AutoDL 网站保存私有镜像。
- 代码打包：git 仓库里打包 git 跟踪的和没被忽略的文件，否则按 `.dgpuignore`；`.git`、`.env*`、`*.pem`、`*.key` 总是排除，`data` 里在代码目录中的本地目录也不进代码包。超过 200 MB 会报错，所以权重、输出不要放在代码目录里。代码目录里有坏掉或指向目录外的符号链接时 `submit` 会列出并停下：把列出的链接告诉用户，由用户修好；用户确认不需要它们才加 `--yes`。
- `data` 每一项拷到 `$DGPU_DATA/<路径最后一段>`；这个名字只能用字母、数字、点、下划线、连字符。大量小文件最好先打成 tar，在 `setup` 里解压。
- 本地数据里单个文件超过 5 GiB 时 `submit` 直接报错：请用户先用 rclone 传到 bucket，再写成 `r2://<bucket>/<目录>`。`r2://` 的 bucket 不是配置里的 bucket 时 dgpu 会提醒，转告用户确认 R2 token 能读它。
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
4. 成功退出码 0，失败非 0。训练进程不需要存储密钥。
5. 训练通常以普通用户 `dgpu` 运行（读不到 R2 密钥）：`pip install` 照常可用（自动装到 `$HOME/.local`），`apt-get`、`conda install` 会失败——改用 pip 能装的包，或请用户决定在 `job.yaml` 里加 `run_as_root: true`（训练代码因此能读到 R2 密钥）。

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

1. **报价**：`dgpu offers job.yaml`（AutoDL 加 `--backend autodl`）。把 GPU 型号、每小时价格、预估总价告诉用户，并说明最坏花费是 `offers` 打印的“费用上限”：价格上限 ×（启动最多 20 分钟 + `max_hours` + 收尾时间），2 小时的任务约按 3.4 小时算；每台启动失败被换掉的机器、每次重试还会额外花钱。
2. **确认上限**：和用户确认 `max_hours` 和价格上限，用户同意后才提交。新代码先用小的 epoch 和 `max_hours` 跑一次。
3. **提交**：`dgpu submit job.yaml`。它要等机器启动、机器上的 dgpu 检查过 GPU、磁盘和存储才返回任务 ID（检查不过会自动换机），可能要十几分钟，所以用 Bash 的 `run_in_background: true` 运行。不要随手打断：打断（Ctrl-C，或结束后台任务发出的 SIGTERM）会销毁已创建的机器，任务也就结束了。**绝不** `kill -9` 它：那样没机会销毁机器，只能靠 `dgpu gc`。还没返回时，`dgpu list` 能看到正在启动的任务。
   - 本地数据超过 5 GB 时 `submit` 会要求确认，后台运行没法回答。先用 `du -sh <目录>` 看大小告诉用户，用户同意后加 `--yes`（它也会放行坏掉的符号链接，见第 2 节）。
   - 输出里出现“创建结果未知”时，把其中的机器名告诉用户，10 分钟后运行 `dgpu gc --dry-run` 确认，有就问用户后 `dgpu gc`。
4. **等结果**：`dgpu wait <job>`，同样用 `run_in_background: true`，结束时会收到通知，按第 5 节的退出码处理。不要在前台反复轮询。
5. **用户问进度时**：`dgpu status <job>`、`dgpu logs <job>`；`dgpu list` 列出所有任务。`dgpu logs <job> -f` 会一直跟随不退出，前台不要用。
6. **取结果**：`dgpu pull <job> <本地目录>`，把 `$DGPU_OUTPUT` 里的内容下载到 `<本地目录>/output/`（不写目录时是 `./<job>/output/`）；要 checkpoint 加 `--ckpt`。
7. **收尾**：`dgpu gc --dry-run`，确认没有遗留机器。列出了机器就告诉用户，用户同意后运行 `dgpu gc`。列出“未知任务”（R2 上查不到任务的 dgpu 机器）时，它们可能是用户另一份配置或同事的：告诉用户机器名和存在多久，**只有用户确认是自己的**才运行 `dgpu gc --include-unknown --yes`。

用户要停止任务：`dgpu cancel <job>`，机器上的 dgpu 会先上传已有结果再销毁机器。它不响应时 `dgpu cancel <job> --force` 直接销毁，没上传的结果会丢失，先告诉用户。取消之后、任务结束之前，`dgpu retry` 会拒绝，`wait --retries` 也不会重试：取消优先。

## 5. 按结果处理

`dgpu wait` 的退出码：

| 退出码 | 状态 | 怎么做 |
|---|---|---|
| 0 | SUCCEEDED | `dgpu pull`，然后 `dgpu gc --dry-run` |
| 1 | FAILED | `dgpu status <job>` 看原因（下表），`dgpu logs <job>` 看最后几十行 |
| 2 | — | 命令用错了（参数不对、任务不存在），检查命令 |
| 3 | TIMED_OUT | 到了 `max_hours` 被停止。已上传的输出和 checkpoint 还在：先 `dgpu pull`。要接着训练就问用户，同意后 `dgpu retry <job>` |
| 4 | CANCELLED | 任务被取消，告诉用户 |
| 5 | LOST | 确认失联：心跳中断，且平台显示机器已退出或状态未知（或心跳停了约 42 分钟以上）。先 `dgpu gc --dry-run` 和 `dgpu doctor`（看余额），问用户后 `dgpu retry <job>` |
| 6 | SUCCEEDED，但清理遗留机器失败 | 机器可能仍在计费。把以 `!!` 开头的输出告诉用户，运行 `dgpu gc --dry-run`，问用户后 `dgpu gc`；结果照常 `dgpu pull` |
| 7 | 不知道 | 连续 10 分钟连不上 R2。任务本身不受影响；告诉用户检查网络，恢复后再 `dgpu wait <job>` |

`wait` 期间提示“失联（实例仍在运行）”时不要做任何事：机器可能在做最后的上传或暂时连不上 R2，`wait` 会继续等，确认失联后才结束。

`retry` 会换一台机器，从 R2 上的 checkpoint 接着跑，用的是提交时打包的代码。用户事先同意自动重试时，可以用 `dgpu wait <job> --retries 1`，确认失联时自动重试一次。

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
| `signalled` | dgpu 在机器上的进程收到停止信号（容器被平台停止等），已提前收尾上传 | 问用户后 `dgpu retry <job>`，会从 checkpoint 接着跑 |
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
| `doctor` 警告 Vast 余额无法查询 | key 没有读取账户信息的权限（Vast 要求两步验证）。可以继续用；提交前请用户自己在 Vast 控制台看余额 |
| 提交时说代码目录里有坏掉或指向目录外的符号链接 | 把列出的链接告诉用户；修好或确认不需要后才加 `--yes` |
| `setup` 里 `apt install` 失败（Permission denied） | 训练默认不以 root 运行。换装好系统包的镜像；或经用户同意加 `run_as_root: true` |
| `offers` 没有结果 | 价格上限太低、GPU 型号太少或没有库存。告诉用户，由用户决定是否放宽 |
| 提交时说镜像 tag 不合法 | 写固定 tag，不用 `latest` |
| 代码包超过 200 MB | 把数据、权重、输出移出代码目录，或写进 `.gitignore` / `.dgpuignore` |
| 提交时说某个文件超过 5 GiB | 请用户用 rclone 把数据传到 bucket，`data` 里改写成 `r2://<bucket>/<目录>` |
| 连续两次 `runner_timeout` | 镜像缺 `python3` 或 `bash`，换镜像 |
| 日志里有 `libGL.so.1` | OpenCV 缺系统库：改装 `opencv-python-headless`；要在 `setup` 里 `apt-get install libgl1` 就得请用户同意加 `run_as_root: true` |
| `setup` 里 `apt-get` / `conda install` 报 Permission denied | 训练以普通用户运行，见第 3 节第 5 条 |
| dgpu 自己出错（日志里有“runner 内部错误”） | 看 R2 上 `logs/attempt-<n>.runner.log`（诊断日志的最后 256 KB） |
| 续跑后又从第 0 轮开始 | 训练代码没处理 `DGPU_RESUME`，或 checkpoint 没写进 `DGPU_CKPT` |
| AutoDL 上下载很慢或失败 | 跨境网络问题，告诉用户；可能要改用国内对象存储 |
