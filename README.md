# dgpu

**Train any model on rented cloud GPUs with one command, and never pay for a machine you forgot to shut down.**

[![CI](https://github.com/baabanana/dynamic_gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/baabanana/dynamic_gpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

English · [中文](README.zh-CN.md)

`dgpu submit job.yaml` rents a GPU machine on [Vast.ai](https://vast.ai) or [AutoDL](https://www.autodl.com), copies your code and data onto it, runs your training command, uploads outputs and checkpoints to your own [Cloudflare R2](https://developers.cloudflare.com/r2/) bucket, and then **destroys the machine**: on success, failure, timeout, cancel, or crash.

You don't keep a GPU server running, and dgpu doesn't need a server either. Job state lives in your bucket, so you can check a job from any computer.

> [!WARNING]
> **Alpha software that spends real money.** Parts of the Vast.ai path have been verified on real machines: self-destroy, timeout, result upload and dropping privileges for training (see [design §8](docs/dgpu-design.md#8-实现状态与验证记录)). A full successful training run, `retry`, `cancel` and `gc` of orphans have not been verified on Vast yet. **The AutoDL backend and the YOLO example have not been run on the real platforms at all.** Top up only a small balance and don't enable auto-recharge. See [Limitations](#limitations).

## Why dgpu

Hourly GPU rentals are cheap. The expensive part is the machine that is still running after your script crashed, your laptop went to sleep, or you pressed Ctrl-C at the wrong moment. dgpu is built around one rule: **every outcome of a job ends with the instance destroyed**. It also shows you the worst-case cost before anything is created.

| | |
|---|---|
| **Model-agnostic** | A job is a Docker image plus a shell command. Your code reads data from `$DGPU_DATA` and writes to `$DGPU_OUTPUT`. You don't need an SDK or any storage credentials in your code. |
| **Layered teardown** | The on-instance runner destroys the machine when the job ends. An independent watchdog takes over if the runner dies. `submit` cleans up on Ctrl-C, SIGTERM and SIGHUP. `dgpu gc` sweeps leftovers by label. Your account balance is the last line of defence. |
| **Cost ceiling up front** | Before creating anything, `submit` prints *price cap × maximum billable time* (startup + `max_hours` + teardown) and checks your balance. |
| **Resume on a new machine** | Checkpoints in `$DGPU_CKPT` are synced to R2. `dgpu retry` rents another machine and continues from them. |
| **Stateless CLI** | All job state lives in your bucket. dgpu keeps no local database, so any computer configured for the same bucket can see the same jobs. |
| **Least privilege on the instance** | Your platform API keys never leave your computer. When the container runs as root (the usual case), training runs as an unprivileged user, so it can't read the storage token or kill the process that destroys the machine. |
| **Two backends, one job file** | Vast.ai (USD, any public Docker image) and AutoDL elastic deployments (CNY, platform images, requires enterprise verification). Switch with `--backend`. |
| **Works with Claude Code** | A bundled [Skill](skills/dgpu/SKILL.md) lets Claude check your setup, write `job.yaml`, submit, wait, and clean up for you. |

## How it works

```
your computer / Claude            your R2 bucket                  rented GPU instance
dgpu submit ─ code, spec ───────▶ dgpu/jobs/<id>/
            ─ find offer, create labelled instance ─────────────▶ bootstrap → runner + watchdog
dgpu wait   ◀─ status, heartbeat ─ status.json, heartbeat.json ◀─ periodic sync
dgpu logs   ◀─ logs ────────────── logs/, events/               ◀─ periodic sync
dgpu pull   ◀─ results ─────────── output/, ckpt/               ◀─ final sync, then self-destroy
dgpu gc     ─ destroy leftover instances that carry a dgpu label ─▶
```

The runner on the instance uses only the Python standard library (3.8+) and a pinned [rclone](https://rclone.org) binary. dgpu downloads rclone on your computer, checks its SHA-256 and ships it in the job bundle. Data is copied to local disk before training starts; dgpu doesn't mount R2 with FUSE, because rented containers are usually unprivileged and reading many small files over the network slows training down.

## Requirements

- **Your computer:** macOS or Linux, Python ≥ 3.11, [uv](https://docs.astral.sh/uv/).
- **Storage:** a Cloudflare R2 bucket and an API token scoped to *that bucket only* (object read & write). Other S3-compatible stores can be configured as profiles but have not been tested.
- **Compute:** a [Vast.ai](https://vast.ai) API key, and/or an [AutoDL](https://www.autodl.com) developer token. AutoDL's elastic deployment API requires enterprise verification.
- **Image:** any Linux x86_64 image that has `python3` (3.8 or newer) and `bash`. On Vast, use a pinned tag, not `latest`.

## Install

```bash
git clone https://github.com/baabanana/dynamic_gpu.git
cd dynamic_gpu
uv tool install .        # installs the `dgpu` command
dgpu --help
```

Or run it without installing: `uv run dgpu ...` from inside the repository.

To let Claude Code drive dgpu, link the Skill:

```bash
ln -s "$PWD/skills/dgpu" ~/.claude/skills/dgpu
```

## Quick start

**1. Configure once.**

```bash
dgpu init        # writes ~/.config/dgpu/config.toml with mode 600
$EDITOR ~/.config/dgpu/config.toml
dgpu doctor      # checks config, R2 read/write, platform API and balance
```

```toml
[stores.r2]
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"   # domain only, no bucket path
bucket = "my-dgpu"
access_key_id = "..."
secret_access_key = "..."

[backends.vast]
api_key = "..."
store = "r2"
```

Keep keys in this file or in environment variables (`VAST_API_KEY`, `AUTODL_TOKEN`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`). Never put them in `job.yaml`.

**2. Describe the job.**

```yaml
# job.yaml
version: 1
name: my-train
backend: vast
image:
  vast: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime   # pinned tag required
code: .
setup: python -m pip install -r requirements.txt
run: python train.py
data:
  - r2://my-dgpu/datasets/coco8      # already in R2
  - ./datasets/small-set             # local: uploaded once, deduplicated by content hash
gpu:
  names: [RTX 3090, RTX 4090]
  count: 1
limits:
  max_hours: 4
  max_usd_per_hour: 0.40
env:
  EPOCHS: "100"
```

**3. Make your training script follow the contract.**

| Variable | Meaning |
|---|---|
| `DGPU_DATA` | Each `data` entry is at `$DGPU_DATA/<last path segment>` |
| `DGPU_OUTPUT` | Write results here. Uploaded about every 2 minutes and once more at the end |
| `DGPU_CKPT` | Write checkpoints here as `name.tmp`, then rename. `*.tmp` files are never uploaded |
| `DGPU_RESUME` | `1` when checkpoints were restored into `$DGPU_CKPT`, so you should resume |
| `DGPU_JOB_ID`, `DGPU_ATTEMPT` | Job ID and attempt number |

Exit with code 0 on success. Try it locally first by pointing these variables at local directories. A complete example is in [`examples/yolo/train.py`](examples/yolo/train.py).

**4. Run it.**

```bash
dgpu offers job.yaml            # matching machines and estimated cost; free
dgpu submit job.yaml            # prints the cost ceiling, rents, waits for preflight, prints the job ID
dgpu wait <job-id>              # blocks until the job ends; the exit code is the result
dgpu pull <job-id> ./result     # downloads outputs to ./result/output/
dgpu gc --dry-run               # confirms nothing is left running
```

## Commands

| Command | What it does |
|---|---|
| `init` | Write a config template (mode 600); never overwrites |
| `doctor` | Check config, storage read/write, platform API and balance |
| `offers` | List candidate machines and estimated cost without creating anything |
| `submit` | Package code, create an instance, return the job ID once preflight passes (`--wait`, `--retries`, `--yes`) |
| `wait` | Wait for the job to end; `--retries N` re-runs a confirmed-lost job on a new machine |
| `status` / `list` / `logs` | Inspect jobs; `logs -f` follows the log |
| `retry` | Destroy the old instance and continue from checkpoints on a new machine |
| `cancel` | Upload what exists, then destroy; `--force` destroys immediately |
| `pull` | Download outputs (`--ckpt` also downloads checkpoints) |
| `gc` | Destroy leftover dgpu instances. It only touches instances whose label has the form `dgpu-<job-id>-a<attempt>` |

`dgpu wait` exit codes: `0` succeeded, `1` failed, `3` timed out, `4` cancelled, `5` lost, `6` succeeded but cleanup failed (run `dgpu gc`), `7` storage unreachable. Full reference: [docs/usage.md §6](docs/usage.md#6-命令一览).

## How billing is stopped

1. **Normal end:** the runner uploads results, then destroys its own instance. On Vast it uses the per-instance `CONTAINER_API_KEY`; on AutoDL the container exits.
2. **Watchdog:** a separate process destroys the instance if the runner crashes or overruns `max_hours`.
3. **Startup:** if the instance never comes up, the price is above your cap, or you interrupt `submit`, dgpu destroys what it created. If the bootstrap script gives up before the runner starts, the Vast instance destroys itself.
4. **`dgpu gc`:** cleans up after a laptop that powered off mid-submit, a `kill -9`, or a failed self-destroy. You can run it every 10 minutes from cron or launchd.
5. **Your balance:** keep it small with no auto-recharge, so that if everything else fails, the loss is capped.

`gc` reports instances whose job it can't find in your bucket but never destroys them without confirmation (`--include-unknown`). This matters if you share a platform account or use several buckets.

## Security model

- **Platform keys** (Vast API key, AutoDL token) stay on your computer. They are never sent to an instance or written to storage.
- **The storage token does reach the instance**, delivered through a short-lived pre-signed URL and not through environment variables or the startup command. On community hosts, the host can read the instance's disk. Scope the token to one bucket. For sensitive data, set `datacenter_only = true` for Vast or don't use rented machines.
- **Training code runs unprivileged** by default. When dgpu can't drop privileges (for example, the container itself isn't root), `dgpu status` and the log say why. `run_as_root: true` gives training root, which also lets it read the storage token. Use that only for code you trust.
- See [SECURITY.md](SECURITY.md) for how to report a vulnerability.

## Limitations

- Single node only (multi-GPU on one machine is fine). No multi-node distributed training.
- No spot or interruptible instances. No backends other than Vast.ai and AutoDL yet; `Backend` is a protocol, so adding one is possible.
- A file over 5 GiB in a local data directory has to be uploaded with rclone first (dgpu uses single-part uploads).
- AutoDL: the data disk is fixed at 50 GB, and network speed from AutoDL to R2 (cross-border) hasn't been measured.
- If an instance starts but its container can't even run `python3`, and your computer is off and `gc` never runs, the instance bills until your balance runs out. See [design §9](docs/dgpu-design.md#9-后续风险) for the full list of known risks.

## Documentation

The detailed docs are currently in Chinese.

- [Usage guide](docs/usage.md): setup, `job.yaml` reference, data handling, every command, FAQ
- [Design](docs/dgpu-design.md): invariants, state machine, teardown guarantees, failure modes, real-platform test log
- [YOLO example](examples/yolo/README.md): YOLO26n on coco8, including resume
- [Claude Code Skill](skills/dgpu/SKILL.md)
- [Docs index](docs/README.md)

## Contributing

Contributions are welcome. Issues and PRs may be written in English or Chinese. You don't need any cloud account: the 1,000+ tests run offline against in-memory and file-backed fakes. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
uv sync
make check     # ruff + pytest
```

## License

[MIT](LICENSE). dgpu is provided **as is, without warranty**. You are responsible for charges on your cloud accounts. The YOLO example installs [Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0; see the [example's license note](examples/yolo/README.md#9-许可证).
