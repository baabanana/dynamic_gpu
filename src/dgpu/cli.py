"""dgpu 命令行（工作包 F）。

只做参数解析、输出格式和退出码，逻辑都在 orchestrator.py。其他工作包的实现只在
`build_deps()` 里 import（`init` 命令另有一处 import 配置模板函数），集成时修改都在这两处。

退出码：可预期的错误（DgpuError）为 2，不打印 traceback；`wait` 按
`states.EXIT_CODE_BY_PHASE` 返回，任务成功但清理遗留实例失败为 6，连续多分钟读不到
存储为 7；Ctrl-C 为 130，SIGHUP 为 129，SIGTERM 为 143（都在清理完之后）。
进度输出走 stderr，任务 ID、状态、日志、JSON 走 stdout，方便脚本和 Claude 解析。

信号：main() 把 SIGTERM、SIGHUP 变成 Terminated（KeyboardInterrupt 的子类），和 Ctrl-C 走同一条
中止路径：submit / retry / wait --retries 启动中的实例会被销毁，cancel --force 会先把销毁
做完。之后再来的 SIGTERM / SIGHUP 只记一笔，不打断清理。SIGKILL 无法处理，留下的实例由
`dgpu gc` 清理。
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from dgpu import __version__, states
from dgpu.contracts import (
    BACKEND_AUTODL,
    BACKEND_VAST,
    BACKENDS,
    Config,
    ConfigError,
    DgpuError,
    StoreProfile,
)
from dgpu.orchestrator import (
    GB,
    Deps,
    GcItem,
    JobView,
    Orchestrator,
    StoreUnreachable,
    Terminated,
    WaitOutcome,
    redact,
    secret_values,
)

EXIT_INTERRUPTED = 130  # Ctrl-C
EXIT_SIGHUP = 128 + signal.SIGHUP  # 129
EXIT_SIGTERM = 128 + signal.SIGTERM  # 143
TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


# ---------------------------------------------------------------- 装配真实实现


def build_deps(config_path: str | None) -> Deps:
    """读取配置，装配真实的存储、平台、bundle 和打包实现。

    其他工作包的模块只在这里 import（延迟 import，集成时的修改集中在这一处）。
    只为填了凭据的平台创建后端。
    """
    from dgpu.bundle import RunnerBundler
    from dgpu.config import load_config, resolve_config_path
    from dgpu.packaging import pack_code, resolve_data, scan_dataset
    from dgpu.spec import load_spec
    from dgpu.store import open_store

    path = resolve_config_path(Path(config_path) if config_path else None, os.environ)
    config = load_config(path)
    backends: dict[str, Any] = {}
    if config.vast.api_key:
        from dgpu.backends.vast import VastBackend

        backends[BACKEND_VAST] = VastBackend(config.vast)
    if config.autodl.token:
        from dgpu.backends.autodl import AutodlBackend

        backends[BACKEND_AUTODL] = AutodlBackend(config.autodl)
    store = open_store(primary_store_profile(config, list(backends)))
    return Deps(
        config=config,
        store=store,
        backends=backends,
        bundler=RunnerBundler(),
        pack_code=pack_code,
        resolve_data=resolve_data,
        scan_data=scan_dataset,
        clock=time.time,
        sleep=time.sleep,
        echo=_progress,
        load_spec=load_spec,
        config_path=Path(path),
    )


def primary_store_profile(config: Config, backend_names: list[str]) -> StoreProfile:
    """所有命令共用的存储：已配置平台用的那个 profile；都没配置时取第一个 [stores.*]。"""
    for name in backend_names or list(BACKENDS):
        try:
            return config.store_for(name)
        except ConfigError:
            if backend_names:
                raise
    if config.stores:
        return next(iter(config.stores.values()))
    raise ConfigError("配置里没有任何存储（[stores.<名字>]）。先运行 `dgpu init` 生成模板。")


# ---------------------------------------------------------------- 公共部分


def _progress(message: str) -> None:
    try:
        click.echo(message, err=True)
    except (OSError, ValueError):
        # 终端已经关了（SIGHUP 之后写 stderr 会 EIO）：输出丢了不影响清理，
        # 需要留档的都另外写进了存储的 events/
        pass


@contextlib.contextmanager
def termination_signals() -> Iterator[None]:
    """在这段代码里把 SIGTERM、SIGHUP 变成 Terminated（和 Ctrl-C 同一条中止路径）。

    第一次收到时抛 Terminated；清理期间再收到 SIGTERM / SIGHUP 只记一笔，不打断清理
    （再按 Ctrl-C 仍然可以强行打断）。退出时恢复原来的处理函数。只能在主线程里用。
    """
    fired: list[int] = []

    def handler(signum: int, frame: Any) -> None:
        if fired:
            _progress(f"已在清理中，忽略 {signal.Signals(signum).name}。")
            return
        fired.append(signum)
        raise Terminated(signum)

    previous: dict[int, Any] = {}
    for sig in TERMINATION_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # 不在主线程，或平台不支持这个信号
            pass
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


class _App:
    def __init__(self, config_path: str | None):
        self.config_path = config_path
        self._deps: Deps | None = None

    def deps(self) -> Deps:
        if self._deps is None:
            self._deps = replace(build_deps(self.config_path), echo=_progress)
        return self._deps

    def orchestrator(self) -> Orchestrator:
        return Orchestrator(self.deps())


class _Group(click.Group):
    """DgpuError → stderr 上的一句话和退出码 2（读不到存储的 wait 为 7）；
    Ctrl-C → 130，SIGHUP → 129，SIGTERM → 143。"""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except DgpuError as exc:
            message = str(exc)
            app = ctx.obj
            if isinstance(app, _App) and app._deps is not None:
                message = redact(message, secret_values(app._deps.config, app._deps.store))
            if isinstance(exc, StoreUnreachable):
                _progress(f"错误：连不上存储。{message}")
                ctx.exit(states.EXIT_STORE_UNREACHABLE)
            _progress(f"错误：{message}")
            ctx.exit(states.EXIT_USAGE)
        except KeyboardInterrupt as exc:
            if isinstance(exc, Terminated):
                _progress(f"收到 {exc.signame}，已清理并退出。")
                ctx.exit(exc.exit_code)
            _progress("已中断。")
            ctx.exit(EXIT_INTERRUPTED)


def _print_json(obj: Any) -> None:
    click.echo(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 120:
        return f"{seconds:.0f} 秒前"
    if seconds < 7200:
        return f"{seconds / 60:.0f} 分钟前"
    return f"{seconds / 3600:.1f} 小时前"


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "0"
    for unit, size in (("GB", GB), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def _exit_for_phase(ctx: click.Context, phase: str) -> None:
    ctx.exit(states.EXIT_CODE_BY_PHASE.get(phase, states.EXIT_FAILED))


def _finish_wait(ctx: click.Context, outcome: WaitOutcome) -> None:
    """输出 wait 的结果并按它退出。任务成功但清理遗留实例失败时退出码 6（不是 0）。"""
    click.echo(outcome.phase)
    if outcome.cleanup_problems:
        _progress(
            f"!! 任务结束（{outcome.phase}），但清理遗留实例时有 {len(outcome.cleanup_problems)} "
            "个问题（见上面以 !! 开头的行），实例可能仍在计费。请运行 `dgpu gc`，"
            "或到平台控制台检查。"
        )
        if outcome.phase == states.SUCCEEDED:
            ctx.exit(states.EXIT_CLEANUP_FAILED)
    _exit_for_phase(ctx, outcome.phase)


def _load_spec(deps: Deps, path: Path, backend: str | None):
    if deps.load_spec is None:
        raise DgpuError("内部错误：没有装配 job.yaml 解析器。")
    return deps.load_spec(path, backend_override=backend)


# ---------------------------------------------------------------- 命令


@click.group(cls=_Group, context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="配置文件路径（默认 $DGPU_CONFIG，或 ~/.config/dgpu/config.toml）。",
)
@click.version_option(__version__, prog_name="dgpu")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """在按需云 GPU（Vast.ai、AutoDL）上训练任意模型，用完自动销毁。"""
    ctx.obj = _App(config_path)


@cli.command()
@click.argument("job_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--backend", type=click.Choice(BACKENDS), help="覆盖 job.yaml 里的 backend。")
@click.option("--wait", "wait_", is_flag=True, help="提交后前台等到结束，退出码同 dgpu wait。")
@click.option(
    "--retries",
    type=click.IntRange(min=0),
    default=0,
    help="配合 --wait：确认失联（LOST）时自动重试的次数。",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="不再确认：本地数据较大时直接上传；代码目录里有坏掉或指向目录外的符号链接时照样提交。",
)
@click.pass_context
def submit(
    ctx: click.Context, job_file: Path, backend: str | None, wait_: bool, retries: int, yes: bool
) -> None:
    """打包、创建实例，等实例通过 preflight（GPU、磁盘、存储检查）后输出任务 ID。

    启动阶段收到 Ctrl-C、SIGTERM、SIGHUP 都会先销毁已创建的实例再退出。
    """
    app: _App = ctx.obj
    deps = app.deps()
    spec = _load_spec(deps, job_file, backend)
    threshold = deps.config.local_data_confirm_gb

    def confirm(gb: float) -> bool:
        return click.confirm(
            f"本地数据共 {gb:.1f} GB（超过 {threshold:g} GB），将上传到存储。继续吗？",
            default=False,
            err=True,
        )

    orch = app.orchestrator()
    result = orch.submit(spec, confirm_upload=None if yes else confirm, accept_skipped_links=yes)
    click.echo(result.job_id)
    if result.phase in states.TERMINAL_PHASES:
        _progress(
            f"任务在返回前已经结束：{result.phase}。用 `dgpu logs {result.job_id}` 查看日志。"
        )
        _exit_for_phase(ctx, result.phase)
    if wait_:
        _finish_wait(ctx, orch.wait_for(result.job_id, retries=retries))
    _progress(
        f"任务已在运行。查看：dgpu status {result.job_id}；等待结束：dgpu wait {result.job_id}"
    )


@cli.command()
@click.argument("job_id")
@click.option(
    "--retries",
    type=click.IntRange(min=0),
    default=0,
    help="确认失联（LOST）时自动换机重试的次数。有未完成的取消请求时不重试。",
)
@click.pass_context
def wait(ctx: click.Context, job_id: str, retries: int) -> None:
    """前台等到任务结束。

    退出码：0 成功、1 失败、3 超时、4 取消、5 确认失联（LOST）、6 任务成功但清理遗留实例
    失败（实例可能仍在计费，跑 dgpu gc）、7 连续 10 分钟读不到存储（任务结果未知）、
    2 用法或环境错误。心跳过期但平台上实例仍在运行时只提示“失联（实例仍在运行）”，继续等。
    """
    _finish_wait(ctx, ctx.obj.orchestrator().wait_for(job_id, retries=retries))


@cli.command()
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON。")
@click.pass_obj
def status(app: _App, job_id: str, as_json: bool) -> None:
    """查看任务状态（只读；心跳过期时显示 LOST）。"""
    view = app.orchestrator().status(job_id)
    if as_json:
        _print_json(view.to_dict())
        return
    click.echo(_format_view(view))


def _format_view(view: JobView) -> str:
    lines = [f"任务 {view.job_id}：{view.phase}"]
    if view.lost:
        lines[0] += f"（存储里的阶段是 {view.stored_phase or '-'}，心跳已过期）"
    if view.attempt is not None:
        where = f"{view.backend or '-'} 实例 {view.instance_id or '（尚未创建）'}"
        lines.append(f"  attempt {view.attempt}，{where}，标签 {view.label or '-'}")
    if view.heartbeat_at is not None:
        detail = f"  心跳：{_fmt_age(view.heartbeat_age_s)}，步骤 {view.step or '-'}"
        if view.gpu_util is not None:
            detail += f"，GPU {view.gpu_util:.0f}%"
        detail += f"，已同步 {_fmt_bytes(view.synced_bytes)}"
        lines.append(detail)
    if view.reason or view.exit_code is not None:
        lines.append(f"  原因：{view.reason or '-'}，退出码：{view.exit_code}")
    if view.message:
        lines.append(f"  说明：{view.message}")
    if view.not_uploaded:
        lines.append(f"  未上传的文件：{'、'.join(view.not_uploaded)}")
    identity = _format_identity(view.identity)
    if identity:
        lines.append(f"  训练进程：{identity}")
    if view.suspect_labels:
        lines.append(
            f"  !! 创建结果未知的实例标签：{'、'.join(view.suspect_labels)}。这样的实例如果晚到："
            "Vast 上拿不到启动文档会自毁，`dgpu wait` 结束时和 `dgpu gc` 也会销毁它；"
            "用 `dgpu gc --dry-run` 和平台控制台确认没有遗留。"
        )
    if view.cancel_requested and view.phase not in states.TERMINAL_PHASES:
        lines.append("  已请求取消，等待 runner 收尾")
    if view.lost:
        lines.append(
            "  心跳过期不一定是机器挂了：平台上实例仍在运行时，runner 可能在最终上传或暂时连不上"
            "存储。`dgpu wait` 会查平台状态，确认失联后才算 LOST。"
        )
    lines.append(f"  状态更新于 {_fmt_time(view.status_updated_at)}")
    return "\n".join(lines)


def _format_identity(identity: dict[str, Any] | None) -> str:
    """心跳里的训练进程身份 → 一行说明（评审 F7：降权没生效时原因要看得见）。"""
    if not identity:
        return ""
    if identity.get("dropped"):
        return f"以无特权用户 {identity.get('user') or '-'}（uid {identity.get('uid')}）运行"
    reason = identity.get("reason") or "-"
    detail = identity.get("detail") or ""
    return (
        f"没有降权，和 runner 同一用户运行，能读到存储凭据（原因 {reason}"
        + (f"：{detail}" if detail else "")
        + "）"
    )


@cli.command()
@click.argument("job_id")
@click.option("--follow", "-f", is_flag=True, help="持续输出新日志，直到任务结束。")
@click.option("--attempt", type=click.IntRange(min=1), default=None, help="看第几次尝试的日志。")
@click.pass_obj
def logs(app: _App, job_id: str, follow: bool, attempt: int | None) -> None:
    """输出任务日志（来自存储，最多滞后一个同步周期）。"""
    app.orchestrator().logs(job_id, follow=follow, attempt=attempt, out=click.echo)


@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON。")
@click.option("--limit", type=click.IntRange(min=1), default=30, show_default=True)
@click.pass_obj
def list_cmd(app: _App, as_json: bool, limit: int) -> None:
    """列出存储里的任务，最新的在前。"""
    jobs = app.orchestrator().list_jobs(limit=limit)
    if as_json:
        _print_json([job.to_dict() for job in jobs])
        return
    if not jobs:
        click.echo("（没有任务）")
        return
    for job in jobs:
        extra = f" {job.reason}" if job.reason else ""
        attempt = f"a{job.attempt}" if job.attempt else "-"
        click.echo(f"{job.job_id}  {job.phase}{extra}  {attempt}  {_fmt_time(job.updated_at)}")


@cli.command()
@click.argument("job_id")
@click.pass_context
def retry(ctx: click.Context, job_id: str) -> None:
    """确认旧实例已销毁，attempt + 1，换机重跑（runner 恢复 checkpoint）。

    有未完成的取消请求（dgpu cancel 之后任务还没结束）时拒绝：取消优先，要重来先
    `dgpu cancel --force`。任务已结束（包括 CANCELLED）时 retry 会清除取消标志。
    新的 attempt 在第一次心跳前就结束时，退出码同 dgpu wait（如 FAILED 为 1）。
    """
    result = ctx.obj.orchestrator().retry(job_id)
    click.echo(result.job_id)
    _progress(f"attempt {result.attempt} 已启动：{result.phase}")
    if result.phase in states.TERMINAL_PHASES:
        _progress(f"attempt {result.attempt} 在返回前已经结束。用 `dgpu logs {job_id}` 查看日志。")
        _exit_for_phase(ctx, result.phase)


@cli.command()
@click.argument("job_id")
@click.option("--force", is_flag=True, help="直接销毁实例（不做最终同步）并标记 CANCELLED。")
@click.pass_obj
def cancel(app: _App, job_id: str, force: bool) -> None:
    """取消任务：默认由 runner 收尾并上传结果；--force 直接销毁。

    取消请求未完成时 dgpu retry 和 dgpu wait --retries 都不会重新开机器。
    """
    view = app.orchestrator().cancel(job_id, force=force)
    click.echo(f"{view.job_id}：{view.phase}")


@cli.command()
@click.argument("job_id")
@click.argument("dest", required=False, type=click.Path(file_okay=False, path_type=Path))
@click.option("--ckpt", is_flag=True, help="同时下载 ckpt/。")
@click.pass_obj
def pull(app: _App, job_id: str, dest: Path | None, ckpt: bool) -> None:
    """下载 output/（默认到 ./<任务ID>/output）。"""
    result = app.orchestrator().pull(job_id, dest or Path(job_id), include_ckpt=ckpt)
    click.echo(f"已下载 {result.output_files} 个输出文件到 {result.dest / 'output'}")
    if ckpt:
        click.echo(f"已下载 {result.ckpt_files} 个 checkpoint 文件到 {result.dest / 'ckpt'}")


@cli.command()
@click.argument("job_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--backend", type=click.Choice(BACKENDS), help="覆盖 job.yaml 里的 backend。")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON。")
@click.option("--limit", type=click.IntRange(min=1), default=10, show_default=True)
@click.pass_obj
def offers(app: _App, job_file: Path, backend: str | None, as_json: bool, limit: int) -> None:
    """列出候选机器和预估费用，不创建实例、不花钱。"""
    deps = app.deps()
    spec = _load_spec(deps, job_file, backend)
    result = app.orchestrator().offers(spec)
    if as_json:
        data = result.to_dict()
        data["offers"] = data["offers"][:limit]
        _print_json(data)
        return
    click.echo(f"条件：{result.constraints}")
    click.echo(f"费用上限：{result.cost_ceiling:.2f} {result.currency}（不含流量费）")
    if not result.offers:
        click.echo("没有符合条件的机器。可以放宽 GPU 型号、提高价格上限或换 --backend。")
        return
    click.echo(
        f"共 {len(result.offers)} 个，按预估总价排序（前 {min(limit, len(result.offers))} 个）："
    )
    for offer in result.offers[:limit]:
        details = "，".join(f"{k}={v}" for k, v in sorted(offer.details.items()))
        click.echo(
            f"  {offer.id:<12} {offer.gpu_name} × {offer.gpu_count}  "
            f"{offer.price_per_hour:.3f} {offer.currency}/h  预估总价 {offer.est_total:.2f}"
            + (f"  {offer.region}" if offer.region else "")
            + (f"  {details}" if details else "")
        )


@cli.command()
@click.option("--dry-run", is_flag=True, help="只报告，不销毁。")
@click.option(
    "--include-unknown",
    is_flag=True,
    help="也销毁存储里查不到任务的 dgpu 实例（默认只报告，它们可能是别人的）。销毁前会确认。",
)
@click.option("--yes", "-y", is_flag=True, help="配合 --include-unknown：不再确认。")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON。")
@click.pass_context
def gc(ctx: click.Context, dry_run: bool, include_unknown: bool, yes: bool, as_json: bool) -> None:
    """清理两个平台上遗留的 dgpu 实例（只碰带 dgpu 标签的实例）。

    存储里查不到对应任务的实例（另一份配置、另一个 bucket，或共用平台账户的人的任务）
    只报告，不销毁；确认是自己的遗留实例后用 --include-unknown 销毁。
    心跳过期但平台显示实例仍在运行时保留（runner 可能在最终上传），过了 runner 最坏
    收尾时间才清理。
    """

    def confirm(items: list[GcItem]) -> bool:
        _progress("这些 dgpu 实例在存储里查不到对应的任务，可能属于别人：")
        for item in items:
            _progress(f"  {item.backend}/{item.instance_id}（{item.label}）：{item.reason}")
        return click.confirm(f"确定销毁这 {len(items)} 台实例吗？", default=False, err=True)

    report = ctx.obj.orchestrator().gc(
        dry_run=dry_run,
        include_unknown=include_unknown,
        confirm_unknown=None if yes else confirm,
    )
    if as_json:
        _print_json(report.to_dict())
    else:
        verb = "将会销毁" if dry_run else "已销毁"
        for item in report.destroyed:
            click.echo(f"{verb} {item.backend}/{item.instance_id}（{item.label}）：{item.reason}")
        for item in report.kept:
            click.echo(f"保留 {item.backend}/{item.instance_id}（{item.label}）：{item.reason}")
        for item in report.unknown:
            click.echo(
                f"未知任务 {item.backend}/{item.instance_id}（{item.label}）：{item.reason}"
            )
        for item in report.failed:
            click.echo(
                f"失败 {item.backend}/{item.instance_id or '-'}"
                f"（{item.label or '-'}）：{item.reason}"
            )
        click.echo(
            f"合计：{verb} {len(report.destroyed)}，保留 {len(report.kept)}，"
            f"未知任务 {len(report.unknown)}，失败 {len(report.failed)}"
        )
    if report.unknown:
        _progress(
            f"!! 有 {len(report.unknown)} 台 dgpu 实例在存储里查不到任务，dgpu 没有销毁它们，"
            "它们可能在计费。确认是自己的遗留实例后运行 `dgpu gc --include-unknown`。"
        )
    if report.failed:
        ctx.exit(states.EXIT_FAILED)


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """检查配置、存储读写和平台 API。全部通过时退出码为 0。"""
    app: _App = ctx.obj
    try:
        orch = app.orchestrator()
    except DgpuError as exc:
        click.echo(f"[失败] 配置：{exc}")
        ctx.exit(states.EXIT_FAILED)
    checks = orch.doctor()
    for check in checks:
        level = "失败" if not check.ok else ("警告" if check.warning else "OK")
        click.echo(f"[{level}] {check.name}：{check.detail}")
    if not all(check.ok for check in checks):
        ctx.exit(states.EXIT_FAILED)


@cli.command()
@click.pass_obj
def init(app: _App) -> None:
    """生成配置文件模板（权限 600）。已存在时不覆盖。"""
    from dgpu.config import resolve_config_path, write_config_template

    path = Path(resolve_config_path(Path(app.config_path) if app.config_path else None, os.environ))
    if path.exists():
        raise DgpuError(f"配置文件 {path} 已存在，没有覆盖。")
    write_config_template(path)
    click.echo(f"已生成配置模板：{path}。填好凭据后运行 `dgpu doctor` 检查。")


def main() -> None:
    """命令行入口。SIGTERM、SIGHUP 走和 Ctrl-C 相同的中止路径（见本文件开头）。"""
    with termination_signals():
        cli(prog_name="dgpu")


if __name__ == "__main__":  # pragma: no cover
    main()
