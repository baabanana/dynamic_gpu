# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). Until 1.0, minor versions may contain breaking changes to `job.yaml`, the config file, or the storage layout.

## [Unreleased]

First public version, planned as **0.1.0**.

### Added

- `dgpu` CLI: `init`, `doctor`, `offers`, `submit`, `wait`, `status`, `list`, `logs`, `retry`, `cancel`, `pull` and `gc`.
- `job.yaml` (version 1): image, code, `setup` / `run` commands, R2 or local data, GPU names and count, time and price limits, environment variables, and `run_as_root`.
- Backends: Vast.ai (on-demand instances) and AutoDL (elastic deployment `Job`).
- Storage: Cloudflare R2 through S3-compatible store profiles. Storage credentials are delivered to the instance through short-lived pre-signed URLs.
- Instance-side runner (Python 3.8+, stdlib only), watchdog and bootstrap. Data is copied in before training; outputs and checkpoints are synced periodically and once more at the end; the instance destroys itself.
- Layered teardown: runner self-destroy, an independent watchdog, cleanup on Ctrl-C, SIGTERM and SIGHUP during `submit`, bootstrap self-destroy on Vast, and label-based `gc`.
- Worst-case cost ceiling and balance check before submitting.
- Checkpoint resume on a new machine (`retry`, `wait --retries`).
- Training runs as an unprivileged user when the container is root.
- Claude Code Skill (`skills/dgpu/SKILL.md`) and a YOLO example (`examples/yolo/`).

### Verified on real platforms

Vast.ai + R2 (2026-11-10 / 11): instance creation and labels, status polling, bootstrap and runner start, replacing an instance after preflight failure, cleanup on SIGTERM, self-destroy with `CONTAINER_API_KEY`, `TIMED_OUT` handling, result upload, and privilege drop. Not yet verified: a successful end-to-end training run, SIGINT, `cancel`, `retry`, `gc` of orphans, and everything on AutoDL. See [design §8](docs/dgpu-design.md#8-实现状态与验证记录).

[Unreleased]: https://github.com/baabanana/dynamic_gpu/commits/dev
