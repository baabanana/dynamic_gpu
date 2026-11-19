<!-- PRs target `dev`. English or Chinese are both fine. / PR 指向 dev，中英文都可以。 -->

## What and why / 改了什么、为什么

<!-- Link the issue if there is one: Fixes #123 -->

## How it was verified / 怎么验证的

- [ ] `make check` passes locally
- [ ] Added or updated tests for the behaviour and its failure paths
- [ ] Real platform run (optional): platform, commit, what was verified, cost:

## Checklist / 检查项

- [ ] Every job outcome still destroys the instance, and `gc` still only touches dgpu-labelled instances
- [ ] No credentials in logs, exception messages, startup commands or environment variables; platform keys never reach the instance
- [ ] Dropped, retried or degraded work is recorded as an event (`states.py`), not silent
- [ ] Instance-side code (`runner/`, `layout.py`, `states.py`, `__init__.py`) is stdlib-only and Python 3.8 compatible
- [ ] Docs updated if behaviour changed (`docs/dgpu-design.md`, `docs/usage.md`, `skills/dgpu/SKILL.md`), with nothing described as verified on a real platform unless it was

## Remaining risks / 遗留风险
