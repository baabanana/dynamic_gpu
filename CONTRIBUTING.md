# Contributing to dgpu

Thanks for your interest! Issues and pull requests are welcome in **English or Chinese**.

> 中文贡献者：完整的开发规则见 [AGENTS.md](AGENTS.md)，本文是它的英文摘要，两者冲突时以 AGENTS.md 为准。

dgpu rents machines that cost real money, so the bar is "never leak billing, never leak credentials, never silently lose data". Most of this guide explains how to keep that true.

## Ways to contribute

- **Bug reports**, especially anything that left an instance running or exposed a credential. For security issues, follow [SECURITY.md](SECURITY.md) and don't open a public issue.
- **Real-platform test reports.** Many paths are only verified locally (see [design §8](docs/dgpu-design.md#8-实现状态与验证记录)). A report like "ran X on Vast/AutoDL at commit Y, here's what happened and what it cost" is very valuable. The AutoDL backend has never run on the real platform.
- **Docs**, including English translations of `docs/usage.md` and the design doc.
- **New backends** (RunPod, Lambda, AWS, …) behind the `Backend` protocol in `src/dgpu/contracts.py`. Please open an issue first to discuss the design.

For anything bigger than a small fix, **open an issue first**. This project is design-doc driven: behaviour changes are written into `docs/` before they are implemented.

## Development setup

You need Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). You **don't** need any cloud account.

```bash
git clone https://github.com/baabanana/dynamic_gpu.git
cd dynamic_gpu
uv sync
make check          # = uv run ruff check . && uv run pytest
```

The test suite has 1,000+ tests and takes a few minutes. It never touches the network or real credentials: HTTP is mocked with respx, S3 with moto, and platforms with the fakes in `src/dgpu/testing.py`. Time and sleep are injected.

Code shipped to instances must run on Python 3.8. Some tests run it under a real old interpreter if uv can find one. Install one to include them:

```bash
uv python install 3.8     # or 3.9; the tests pick up whichever `uv python find` returns
```

## Project layout

```
src/dgpu/
  contracts.py   dataclasses, Backend / Store / Bundler protocols, exceptions   (frozen interface)
  layout.py      storage paths, job IDs, instance labels                        (stdlib, Python 3.8)
  states.py      phases, failure reasons, event names, env vars, JSON formats   (stdlib, Python 3.8)
  testing.py     MemoryStore, FakeBackend, FakeBundler test doubles
  spec.py config.py packaging.py   job.yaml, config file, code packaging
  store.py                         S3-compatible store (R2)
  backends/vast.py backends/autodl.py
  bundle.py runner/                what runs on the instance (stdlib only, Python 3.8)
  orchestrator.py cli.py           the `dgpu` command
skills/dgpu/    Claude Code Skill
examples/yolo/  end-to-end example
tests/          tests/test_<module>*.py
```

## Rules that must not break

These are the project's invariants ([design §3](docs/dgpu-design.md#3-设计红线不变量)). A PR that weakens one won't be merged, however convenient it is.

1. **No billing leaks.** Every job outcome destroys the instance.
2. **Platform master keys (Vast, AutoDL) never reach an instance** and are never written to storage.
3. **Only touch dgpu-labelled instances.** Labels come from `layout.label()` and are recognised with `layout.parse_label()`.
4. **No silent data loss.** Failed uploads, truncated logs, retries and machine swaps are all recorded as events (`states.py`).
5. **Single writer.** `attempt.json` is only updated with conditional writes. A runner that sees a newer attempt exits.
6. **Storage is the only source of truth for job state.** The CLI keeps no local state.
7. **Credentials never appear** in logs, exception messages, startup commands or environment variables.

## Code guidelines

- **Instance-side code** (`runner/`, `layout.py`, `states.py`, `__init__.py`) uses only the standard library and stays Python 3.8 compatible. No `match`, no runtime `X | Y` or `list[int]`, and use `from __future__ import annotations` for type hints. `tests/test_runner_compat.py` checks this.
- **Modules talk through `contracts.py` protocols.** An implementation must behave like its test double in `testing.py`.
- **Expected errors** raise exceptions from `contracts.py`. The message says what happened, what it affects and what to do next.
- **Long-running behaviour matters.** Think about crashes, restarts, a full disk, an unreachable storage or platform API, duplicate runs and graceful shutdown. Always drain child-process pipes and release handles. Never hide a problem by swallowing an exception or turning off a check.
- User-facing messages and docs are currently in Chinese. Keep new CLI messages consistent with the existing ones. Code comments can be in either language.

## Tests

- Test **behaviour and failure paths**, not the implementation. Every row in the failure-mode table ([design §5](docs/dgpu-design.md#5-失败模式与预期行为)) has at least one test. Add one when you add a row.
- Tests must not use the network, real credentials or `~/.config/dgpu`.
- A bug fix comes with a regression test that fails without the fix.

## Docs

- If you change behaviour, data formats or how dgpu runs, update the design doc, `docs/usage.md` and `skills/dgpu/SKILL.md` in the same PR.
- New or moved docs go into the index in [`docs/README.md`](docs/README.md).
- **Be precise about verification.** Don't describe something as working on a real platform if it was only tested against fakes.

## Branches and pull requests

| Branch | Role |
|---|---|
| `dev` | Integration. **Target your PRs here.** |
| `main` | Releases. Fast-forwarded from `dev` only after `make check` passes and real-platform acceptance is done. |

1. Fork the repo and create a branch from `dev` (`fix/...`, `feature/...`, `docs/...`).
2. Keep each PR focused. Don't mix in unrelated refactors.
3. Run `make check`. CI runs it on Linux and macOS and also checks the instance-side code under Python 3.8.
4. Fill in the PR template. Say what changed, why, how you verified it, and what risks remain.

History stays linear (rebase or fast-forward, no merge commits). Commit messages follow `area: what changed`, e.g. `runner: keep writing heartbeats during the final sync`, in English or Chinese.

## Testing on real platforms (optional)

Real runs cost money and are never required for a PR. If you do run one:

- Use your **own** accounts with a small balance and **no auto-recharge**.
- Prefer cheap GPUs and `max_hours` values below 0.1 for lifecycle tests.
- Run `dgpu gc --dry-run` afterwards and check the platform console yourself.
- Report the date, commit, platform, what you verified, what you didn't, and the cost, in the format of the log in [design §8](docs/dgpu-design.md#8-实现状态与验证记录).
- Never paste credentials, account IDs or bucket endpoints into issues, PRs or logs.

## AI-assisted contributions

You're welcome to use AI coding assistants. [AGENTS.md](AGENTS.md) (loaded by Claude Code via `CLAUDE.md`, and read by Codex and similar tools) holds the rules they should follow. You remain responsible for the change: review it, run the tests, and make sure the PR description is accurate.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
