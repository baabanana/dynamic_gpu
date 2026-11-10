"""编排：submit、wait、retry、cancel、gc 等命令背后的逻辑（工作包 F）。

所有外部依赖（存储、平台、bundle、打包、时钟、输出）都通过 `Deps` 注入，本模块不 import
其他工作包的实现。必须守住的不变量（设计第 3 节）：

- 不泄漏计费：本进程创建（或可能创建）的实例在任何退出路径上都要销毁；销毁失败时
  大声报告实例 ID，并让调用方停下来，不再创建新实例。
- 只碰 dgpu 的实例：只处理 `layout.parse_label` 能解析的标签。
- 单写者：`attempt.json` 只用条件写更新；每次换机、重试都用新的 attempt 号，
  旧 runner 读到更大的 attempt 后自行退出。
- 平台主 key 不进入存储和启动命令；存储凭据只出现在 `secrets-<n>.json` 里，
  只通过预签名 URL 下发，runner 通过 preflight 后（或这次尝试结束时）删除。
- 不静默：重试、换机、销毁、gc 都写 `events/`。
- 存储是唯一事实来源：这里不保存任何本地状态。
"""

from __future__ import annotations

import codecs
import json
import math
import os
import re
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dgpu import layout, states
from dgpu.contracts import (
    BACKEND_AUTODL,
    BACKEND_VAST,
    CURRENCY_BY_BACKEND,
    INST_EXITED,
    INST_GONE,
    INST_RUNNING,
    INST_UNKNOWN,
    SINGLE_UPLOAD_LIMIT,
    Backend,
    BackendError,
    Bundler,
    BundleRef,
    Config,
    DataRef,
    DataSource,
    DgpuError,
    GpuRequest,
    InstanceRef,
    InstanceState,
    JobSpec,
    LaunchSpec,
    Offer,
    OfferUnavailable,
    PreconditionFailed,
    Store,
    build_launch_doc,
    build_remote_spec,
)
from dgpu.runner import util as runner_util  # 只用标准库；实例存活时限的公式和 watchdog 共用

# ---------------------------------------------------------------- 常量

POLL_STARTUP_S = 10.0  # 启动阶段轮询实例状态和心跳的间隔
POLL_WAIT_S = 30.0  # `dgpu wait` 轮询间隔
POLL_LOGS_S = 5.0  # `dgpu logs -f` 轮询间隔
DESTROY_CONFIRM_TIMEOUT_S = 300.0  # 销毁后等平台确认实例消失的最长时间
DESTROY_CONFIRM_POLL_S = 5.0
DESTROY_REISSUE_S = 60.0  # 确认期间每隔多久重发一次销毁
CAS_TRIES = 5  # attempt.json / status.json 读改写的最多次数
CAS_BACKOFF_S = 1.0
MIN_DISK_GB = 40
AUTODL_DATA_DISK_GB = 50  # AutoDL 数据盘固定大小（设计 4.12）
ORPHAN_MIN_AGE_S = 3600.0  # 存储里没有对应任务的实例，至少存在这么久才由 gc 销毁
DEADLINE_MARGIN_S = 1800.0  # 实例最晚自毁的时间之后 gc 再多等 30 分钟，避免和 watchdog 抢
MAX_CONSECUTIVE_RUNNER_TIMEOUTS = 2
PREFLIGHT_STEP = "preflight"  # runner 心跳里 preflight 期间的 step
LAUNCH_URL_EXTRA_S = 3600  # 预签名 URL 有效期 = 启动超时 + runner 超时 + 1 小时
DOCTOR_PREFIX = layout.ROOT + "/doctor/"
GB = 1024**3

RETRYABLE_PHASES = (states.LOST, states.FAILED, states.TIMED_OUT, states.CANCELLED)
_CLI_PHASES = (states.PROVISIONING, states.STARTING)  # runner 第一次心跳前由 CLI 写的阶段
_DEAD_INSTANCE_PHASES = (INST_EXITED, INST_UNKNOWN, INST_GONE)
_JOB_STAMP_RE = re.compile(r"-(\d{12})-[0-9a-f]{4}$")


# ---------------------------------------------------------------- 依赖注入


@dataclass
class Deps:
    """编排需要的一切。只有 cli.build_deps() 装配真实实现，测试装配替身。"""

    config: Config
    store: Store
    backends: dict[str, Backend]
    bundler: Bundler
    # 把代码目录打成 tar.gz 写到给定路径；exclude 是不打进包的绝对路径（代码目录里的本地数据）
    pack_code: Callable[..., Any]
    resolve_data: Callable[[Store, DataRef], DataSource]  # 本地数据先上传到存储
    clock: Callable[[], float]  # UTC epoch 秒；要和 runner 写的时间可比，所以是墙上时钟
    sleep: Callable[[float], None]
    echo: Callable[[str], None]  # 给人看的进度输出
    # 以下两项只有 CLI 用到
    load_spec: Callable[..., JobSpec] | None = None
    config_path: Path | None = None


# ---------------------------------------------------------------- 异常


class DestroyFailed(DgpuError):
    """销毁实例失败或无法确认已销毁。实例可能仍在计费。"""

    def __init__(self, ref: InstanceRef, message: str):
        super().__init__(message)
        self.ref = ref


class AttemptConflict(DgpuError):
    """attempt.json 的 attempt 已被别的进程推进，本进程被隔离（fenced）。"""

    def __init__(self, job_id: str, expected: int, actual: Any):
        super().__init__(
            f"任务 {job_id} 的 attempt 已经被另一个 dgpu 进程推进到 {actual}"
            f"（本进程以为是 {expected}），本进程停止操作。"
        )
        self.expected = expected
        self.actual = actual


class _CancelRequested(DgpuError):
    pass


# ---------------------------------------------------------------- 结果类型


@dataclass(frozen=True)
class SubmitResult:
    job_id: str
    attempt: int
    phase: str  # 返回时的阶段：通常是 RUNNING；runner 很快结束时是终态
    backend: str
    instance: InstanceRef | None
    offer: Offer | None
    cost_ceiling: float
    currency: str


@dataclass(frozen=True)
class JobView:
    """status.json + attempt.json + heartbeat.json 合成的任务视图。phase 可能是推导出的 LOST。"""

    job_id: str
    phase: str
    stored_phase: str | None
    attempt: int | None
    status_attempt: int | None
    backend: str | None
    instance_id: str | None
    label: str | None
    reason: str | None
    message: str | None
    exit_code: int | None
    not_uploaded: list[str]
    cancel_requested: bool
    deadline_at: float | None
    status_updated_at: float | None
    heartbeat_at: float | None
    heartbeat_age_s: float | None
    step: str | None
    gpu_util: float | None
    synced_bytes: int | None
    last_sync_at: float | None
    lost: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobSummary:
    job_id: str
    phase: str
    attempt: int | None
    reason: str | None
    updated_at: float | None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OffersResult:
    backend: str
    request: GpuRequest
    offers: list[Offer]
    currency: str
    cost_ceiling: float
    constraints: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "request": asdict(self.request),
            "currency": self.currency,
            "cost_ceiling": self.cost_ceiling,
            "constraints": self.constraints,
            "offers": [asdict(offer) for offer in self.offers],
        }


@dataclass(frozen=True)
class GcItem:
    backend: str
    instance_id: str
    label: str
    job_id: str | None
    attempt: int | None
    reason: str


@dataclass
class GcReport:
    dry_run: bool
    destroyed: list[GcItem] = field(default_factory=list)  # dry_run 时表示“将会销毁”
    kept: list[GcItem] = field(default_factory=list)
    failed: list[GcItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "destroyed": [asdict(item) for item in self.destroyed],
            "kept": [asdict(item) for item in self.kept],
            "failed": [asdict(item) for item in self.failed],
        }


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class PullResult:
    dest: Path
    phase: str
    output_files: int
    ckpt_files: int


@dataclass
class _Launch:
    """一次 submit / retry 的换机循环的上下文。ref / label 供异常清理使用。"""

    job_id: str
    backend_name: str
    backend: Backend
    request: GpuRequest
    offers: list[Offer]
    bundle: BundleRef
    attempt: int
    cost_ceiling: float
    currency: str
    ref: InstanceRef | None = None  # 当前存活、尚未销毁的实例
    label: str | None = None  # 当前 create 用的标签
    create_unknown: bool = False  # create 已调用但结果未知
    offer: Offer | None = None


@dataclass(frozen=True)
class _Outcome:
    kind: str  # started | ended | create_failed | startup_failed | runner_timeout | preflight
    reason: str | None = None
    instance_may_exist: bool = False
    detail: str = ""
    phase: str | None = None


@dataclass
class _JobDocs:
    exists: bool
    attempt: dict[str, Any] | None
    status: dict[str, Any] | None
    heartbeat: dict[str, Any] | None


# ---------------------------------------------------------------- 编排


class Orchestrator:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.config = deps.config
        self.store = deps.store
        self.backends = deps.backends
        self.bundler = deps.bundler
        self.clock = deps.clock
        self.sleep = deps.sleep
        self._secrets = secret_values(deps.config, deps.store)
        self.echo = lambda message: deps.echo(redact(message, self._secrets))

    # ============================================================ submit

    def submit(
        self,
        spec: JobSpec,
        *,
        confirm_upload: Callable[[float], bool] | None = None,
    ) -> SubmitResult:
        """提交任务：找机器、上传、创建实例，等到 runner 通过 preflight 后返回。

        confirm_upload：本地数据超过 config.local_data_confirm_gb 时调用，参数是 GB 数，
        返回 False 则放弃。None 表示不需要确认（--yes）。
        """
        backend_name = spec.backend
        backend = self._backend(backend_name)
        self._check_store_profile(backend_name)
        self._check_local_file_sizes(spec)
        self._warn_foreign_buckets(spec)

        # 1. 搜机器。没有就报错，此时没有创建任何东西，也没有写存储。
        request, local_bytes, has_remote = self._build_request(spec, backend_name)
        offers = backend.search(request)
        if not offers:
            raise DgpuError(self._no_offer_message(backend_name, request, has_remote))
        currency = CURRENCY_BY_BACKEND[backend_name]
        best = offers[0]
        self.echo(
            f"找到 {len(offers)} 台符合条件的机器（{backend_name}），最便宜的是 "
            f"{best.gpu_name} × {best.gpu_count}，{best.price_per_hour:.3f} {currency}/h"
        )

        # 2. 费用上限和余额。
        ceiling = self._print_ceiling(request, currency)
        self._check_balance(backend, ceiling, currency)
        local_gb = local_bytes / GB
        if confirm_upload is not None and local_gb > self.config.local_data_confirm_gb:
            if not confirm_upload(local_gb):
                raise DgpuError("已取消：没有上传数据，也没有创建实例。")

        # 3. 上传代码、数据、spec，写初始状态。
        job_id = layout.new_job_id(spec.name, now=self.clock())
        self.echo(f"任务 ID：{job_id}")
        self._upload_job_files(spec, job_id, backend_name, request)
        bundle = self.bundler.ensure_uploaded(self.store)
        now = self.clock()
        attempt_doc = states.new_attempt_doc(
            job_id,
            1,
            backend_name,
            layout.label(job_id, 1),
            deadline_at=self._deadline(request.max_hours),
            now=now,
        )
        self._create_doc(layout.attempt_key(job_id), attempt_doc)
        status_doc = states.new_status_doc(
            job_id, 1, states.PROVISIONING, states.SRC_CLI, message="正在创建实例", now=now
        )
        self._create_doc(layout.status_key(job_id), status_doc)

        # 4–6. 换机循环；任何异常都会销毁实例并写终态。
        ctx = _Launch(
            job_id=job_id,
            backend_name=backend_name,
            backend=backend,
            request=request,
            offers=offers,
            bundle=bundle,
            attempt=1,
            cost_ceiling=ceiling,
            currency=currency,
        )
        return self._launch(ctx)

    # ============================================================ 查询

    def status(self, job_id: str) -> JobView:
        """读 status.json、attempt.json、heartbeat.json 合成视图。只读，从不写存储。"""
        self._check_job_id(job_id)
        att = self._read_doc(layout.attempt_key(job_id), "attempt.json")
        st = self._read_doc(layout.status_key(job_id), "status.json")
        if att is None and st is None:
            if self.store.exists(layout.spec_key(job_id)):
                raise DgpuError(
                    f"任务 {job_id} 提交到一半就中断了：存储里有 spec.json，但没有状态。"
                    "这个任务没有创建过实例，可以重新 submit。"
                )
            raise DgpuError(
                f"找不到任务 {job_id}（存储 {self.store.profile.name}，"
                f"bucket {self.store.profile.bucket}，路径 {layout.job_prefix(job_id)}）。"
            )
        hb = self._read_doc(layout.heartbeat_key(job_id), "heartbeat.json")
        return self._view(
            job_id,
            att[0] if att else None,
            st[0] if st else None,
            hb[0] if hb else None,
        )

    def list_jobs(self, limit: int | None = None) -> list[JobSummary]:
        """列出存储里的任务，最新的在前。"""
        job_ids = self.store.list_dirs(layout.jobs_prefix())
        job_ids.sort(key=_job_sort_key, reverse=True)
        if limit is not None:
            job_ids = job_ids[:limit]
        result = []
        for job_id in job_ids:
            try:
                st = self._read_doc(layout.status_key(job_id), "status.json")
                if st is not None and st[0].get("phase") in states.TERMINAL_PHASES:
                    doc = st[0]
                    result.append(
                        JobSummary(
                            job_id=job_id,
                            phase=doc["phase"],
                            attempt=doc.get("attempt"),
                            reason=doc.get("reason"),
                            updated_at=doc.get("updated_at"),
                            message=doc.get("message"),
                        )
                    )
                    continue
                view = self.status(job_id)
                result.append(
                    JobSummary(
                        job_id=job_id,
                        phase=view.phase,
                        attempt=view.attempt,
                        reason=view.reason,
                        updated_at=view.status_updated_at,
                        message=view.message,
                    )
                )
            except DgpuError as exc:
                result.append(JobSummary(job_id, "?", None, None, None, str(exc)))
        return result

    def offers(self, spec: JobSpec) -> OffersResult:
        """搜索候选机器并给出预估费用。不创建任何东西，不写存储。"""
        backend_name = spec.backend
        backend = self._backend(backend_name)
        self._warn_foreign_buckets(spec)
        request, _, has_remote = self._build_request(spec, backend_name)
        found = backend.search(request)
        currency = CURRENCY_BY_BACKEND[backend_name]
        return OffersResult(
            backend=backend_name,
            request=request,
            offers=list(found),
            currency=currency,
            cost_ceiling=request.max_price_per_hour * request.max_hours,
            constraints=self._describe_constraints(backend_name, request, has_remote),
        )

    def logs(
        self,
        job_id: str,
        follow: bool = False,
        *,
        attempt: int | None = None,
        out: Callable[[str], None] | None = None,
    ) -> None:
        """输出某次尝试的日志（默认当前 attempt）。follow 时用范围读取增量输出，到终态为止。"""
        out = out or self.echo
        view = self.status(job_id)
        current = attempt or view.attempt or 1
        reader = _LogReader(self.store, layout.log_key(job_id, current), out)
        found = reader.pump()
        if not follow:
            reader.flush()
            if not found:
                self.echo(f"（attempt {current} 还没有日志）")
            return
        while True:
            reader.pump()
            view = self.status(job_id)
            if attempt is None and view.attempt and view.attempt != current:
                reader.pump()
                reader.flush()
                current = view.attempt
                self.echo(f"--- 任务换到了 attempt {current} ---")
                reader = _LogReader(self.store, layout.log_key(job_id, current), out)
                continue
            if view.phase in states.TERMINAL_PHASES or view.phase == states.LOST:
                reader.pump()
                reader.flush()
                self.echo(f"--- 任务 {job_id} 状态：{view.phase}，停止跟随 ---")
                return
            self.sleep(POLL_LOGS_S)

    def pull(self, job_id: str, dest: Path, *, include_ckpt: bool = False) -> PullResult:
        """下载 output/（可选 ckpt/）到 dest/output、dest/ckpt。"""
        view = self.status(job_id)
        if view.phase != states.SUCCEEDED:
            self.echo(
                f"警告：任务 {job_id} 的状态是 {view.phase}，不是 SUCCEEDED，输出可能不完整。"
            )
        if view.not_uploaded:
            self.echo(
                f"警告：最终同步有 {len(view.not_uploaded)} 个文件没有上传："
                + "、".join(view.not_uploaded[:10])
            )
        dest = Path(dest)
        n_out = self.store.download_prefix(layout.output_prefix(job_id), dest / "output")
        n_ckpt = 0
        if include_ckpt:
            n_ckpt = self.store.download_prefix(layout.ckpt_prefix(job_id), dest / "ckpt")
        if n_out == 0:
            self.echo(f"任务 {job_id} 的 output/ 里没有文件。")
        return PullResult(dest=dest, phase=view.phase, output_files=n_out, ckpt_files=n_ckpt)

    # ============================================================ wait / retry / cancel

    def wait(self, job_id: str, retries: int = 0) -> str:
        """等到终态或 LOST。LOST 且还有重试次数时自动 retry。返回最终阶段。"""
        last_shown: tuple | None = None
        while True:
            view = self.status(job_id)
            shown = (view.phase, view.attempt, view.step)
            if shown != last_shown:
                self.echo(_describe_progress(view))
                last_shown = shown
            if view.phase in states.TERMINAL_PHASES:
                self._cleanup_after_end(job_id)
                return view.phase
            if view.phase == states.LOST:
                if retries <= 0:
                    self.echo(
                        f"任务 {job_id} 心跳已过期（LOST）。可以用 `dgpu retry {job_id}` 换机续跑；"
                        "遗留实例会由 `dgpu gc` 清理。"
                    )
                    return states.LOST
                retries -= 1
                self.echo(f"任务心跳已过期（LOST），自动重试（还剩 {retries} 次）")
                try:
                    self.retry(job_id)
                except DgpuError as exc:
                    self.echo(f"自动重试失败：{exc}")
                    view = self.status(job_id)
                    if view.phase in states.TERMINAL_PHASES:
                        self._cleanup_after_end(job_id)
                        return view.phase
                    return states.LOST
                last_shown = None
                continue
            self.sleep(POLL_WAIT_S)

    def retry(self, job_id: str) -> SubmitResult:
        """换机重跑：先确认旧实例已销毁，再用条件写把 attempt + 1，然后走换机循环。"""
        view = self.status(job_id)
        if view.phase not in RETRYABLE_PHASES:
            raise DgpuError(
                f"任务 {job_id} 当前是 {view.phase}，只有 LOST、FAILED、TIMED_OUT、CANCELLED "
                "的任务可以重试。"
            )
        request, backend_name = self._read_request(job_id)
        backend = self._backend(backend_name)
        self._check_store_profile(backend_name)
        offers = backend.search(request)
        if not offers:
            raise DgpuError(self._no_offer_message(backend_name, request, False))
        currency = CURRENCY_BY_BACKEND[backend_name]
        ceiling = self._print_ceiling(request, currency)
        self._check_balance(backend, ceiling, currency)

        got = self._read_doc(layout.attempt_key(job_id), "attempt.json")
        if got is None:
            raise DgpuError(f"任务 {job_id} 没有 attempt.json，无法重试。")
        old_doc = got[0]
        old_attempt = int(old_doc["attempt"])
        # 旧实例必须确认销毁；失败就停在这里，不启动第二台。
        self._destroy_job_instances(
            job_id, backend, old_doc, why=f"重试前确认 attempt {old_attempt} 的实例已销毁"
        )
        bundle = self.bundler.ensure_uploaded(self.store)

        def bump(doc: dict[str, Any]) -> dict[str, Any]:
            new_attempt = int(doc["attempt"]) + 1
            doc.update(
                attempt=new_attempt,
                label=layout.label(job_id, new_attempt),
                instance_id=None,
                cancel=False,
                created_at=self.clock(),
                deadline_at=self._deadline(request.max_hours),
            )
            return doc

        try:
            new_doc = self._update_attempt(job_id, bump, expect_attempt=old_attempt)
        except AttemptConflict as exc:
            raise DgpuError(
                f"任务 {job_id} 正在被另一个 dgpu 进程重试（attempt 已经是 {exc.actual}），"
                "这次没有创建实例。"
            ) from None
        new_attempt = int(new_doc["attempt"])
        self._event(
            job_id,
            states.EV_RELAUNCH,
            new_attempt,
            f"dgpu retry：attempt {old_attempt} 作废，改用 attempt {new_attempt}",
            {"old_attempt": old_attempt, "new_attempt": new_attempt, "why": "retry"},
        )
        self.echo(f"重试任务 {job_id}：attempt {old_attempt} → {new_attempt}")
        self._write_cli_status(job_id, new_attempt, states.PROVISIONING, "重试：正在创建实例")
        ctx = _Launch(
            job_id=job_id,
            backend_name=backend_name,
            backend=backend,
            request=request,
            offers=offers,
            bundle=bundle,
            attempt=new_attempt,
            cost_ceiling=ceiling,
            currency=currency,
        )
        return self._launch(ctx)

    def cancel(self, job_id: str, force: bool = False) -> JobView:
        """在 attempt.json 里设置 cancel，由 runner 收尾；force 时直接销毁实例并写 CANCELLED。"""
        view = self.status(job_id)
        terminal = view.stored_phase in states.TERMINAL_PHASES and (
            view.status_attempt is None or view.status_attempt >= (view.attempt or 0)
        )
        if terminal and not force:
            self.echo(f"任务 {job_id} 已经结束（{view.stored_phase}），不需要取消。")
            return view
        if not terminal:

            def set_cancel(doc: dict[str, Any]) -> dict[str, Any]:
                doc["cancel"] = True
                return doc

            self._update_attempt(job_id, set_cancel)
            self.echo(f"已为任务 {job_id} 设置取消标志。")
        if not force:
            if view.phase == states.LOST:
                self.echo(
                    "注意：任务心跳已过期（LOST），runner 可能已经不在，没人会收尾。"
                    f"可以用 `dgpu cancel --force {job_id}` 直接销毁实例。"
                )
            elif view.phase in _CLI_PHASES:
                self.echo(
                    "runner 还没启动：正在运行的 `dgpu submit` 会看到取消标志并销毁实例；"
                    f"如果 submit 已经不在了，用 `dgpu cancel --force {job_id}`。"
                )
            else:
                interval = self.config.runner.heartbeat_interval_s
                self.echo(
                    f"runner 会在下一次心跳（约 {interval} 秒内）停止训练、上传结果并销毁实例。"
                )
            return self.status(job_id)

        got = self._read_doc(layout.attempt_key(job_id), "attempt.json")
        if got is None:
            raise DgpuError(f"任务 {job_id} 没有 attempt.json，找不到要销毁的实例。")
        att = got[0]
        backend = self._backend(att["backend"])
        self._destroy_job_instances(job_id, backend, att, why="dgpu cancel --force")
        attempt = int(att["attempt"])
        self._write_terminal_if_absent(
            job_id,
            attempt,
            states.CANCELLED,
            None,
            "dgpu cancel --force：实例已直接销毁，没有做最终同步",
        )
        self.store.delete(layout.secrets_key(job_id, attempt))
        return self.status(job_id)

    # ============================================================ gc / doctor

    def gc(self, dry_run: bool = False) -> GcReport:
        """清理遗留实例。只处理标签能被 layout.parse_label 解析的实例。"""
        report = GcReport(dry_run=dry_run)
        now = self.clock()
        cache: dict[str, _JobDocs] = {}
        for name, backend in self.backends.items():
            try:
                instances = backend.list(layout.LABEL_PREFIX)
            except Exception as exc:  # 一个平台失败不影响另一个
                report.failed.append(GcItem(name, "", "", None, None, f"列举实例失败：{exc}"))
                continue
            for inst in instances:
                label = inst.ref.label
                parsed = layout.parse_label(label)
                if parsed is None:
                    report.kept.append(
                        GcItem(name, inst.ref.id, label, None, None, "标签不是 dgpu 格式，不处理")
                    )
                    continue
                if inst.phase == INST_GONE:
                    continue
                job_id, attempt = parsed
                try:
                    if job_id not in cache:
                        cache[job_id] = self._load_job_docs(job_id)
                    docs = cache[job_id]
                    destroy, reason = self._gc_decide(docs, attempt, inst, now)
                except Exception as exc:
                    report.failed.append(
                        GcItem(
                            name,
                            inst.ref.id,
                            label,
                            job_id,
                            attempt,
                            f"读取任务状态失败，保留：{exc}",
                        )
                    )
                    continue
                item = GcItem(name, inst.ref.id, label, job_id, attempt, reason)
                if not destroy:
                    report.kept.append(item)
                    continue
                if dry_run:
                    report.destroyed.append(item)
                    continue
                try:
                    backend.destroy(inst.ref)
                    self._confirm_gone(backend, inst.ref)  # 平台受理了不等于已经停止计费
                except Exception as exc:
                    message = (
                        f"!! gc 销毁实例 {name}/{inst.ref.id}（标签 {label}）失败：{exc}。"
                        "它可能仍在计费，请稍后再跑 `dgpu gc` 或到平台控制台手动销毁。"
                    )
                    self.echo(message)
                    report.failed.append(GcItem(name, inst.ref.id, label, job_id, attempt, message))
                    if docs.exists:
                        self._event(
                            job_id,
                            states.EV_DESTROY_FAILED,
                            attempt,
                            message,
                            {"backend": name, "instance_id": inst.ref.id, "label": label},
                            source=states.SRC_GC,
                        )
                    continue
                report.destroyed.append(item)
                if docs.exists:
                    self._event(
                        job_id,
                        states.EV_GC_DESTROYED,
                        attempt,
                        f"gc 销毁实例 {inst.ref.id}：{reason}",
                        {"backend": name, "instance_id": inst.ref.id, "label": label},
                        source=states.SRC_GC,
                    )
        return report

    def doctor(self) -> list[CheckResult]:
        """检查配置、存储读写和各平台 API。每项独立，不因为一项失败而中断。"""
        checks = [
            CheckResult(
                "配置",
                True,
                f"已加载 {self.deps.config_path or ''}；存储 {self.store.profile.name}"
                f"（bucket {self.store.profile.bucket}）；平台："
                + ("、".join(self.backends) or "无"),
            )
        ]
        checks.append(self._check_store())
        if not self.backends:
            checks.append(
                CheckResult(
                    "平台",
                    False,
                    "配置里没有任何平台的凭据（[backends.vast] 或 [backends.autodl]）",
                )
            )
        for name, backend in self.backends.items():
            checks.append(self._check_backend(name, backend))
        return checks

    # ============================================================ 换机循环

    def _launch(self, ctx: _Launch) -> SubmitResult:
        cfg = self.config
        tries = 0
        runner_timeouts = 0
        needs_bump = False
        last_reason = states.REASON_NO_OFFER
        last_detail = ""
        limit = min(len(ctx.offers), cfg.max_offers)
        try:
            for offer in ctx.offers:
                if tries >= cfg.max_offers:
                    break
                tries += 1
                if needs_bump:
                    self._bump_attempt(ctx, why=last_detail)
                    needs_bump = False
                ctx.offer = offer
                self.echo(
                    f"[{tries}/{limit}] 尝试 offer {offer.id}："
                    f"{offer.gpu_name} × {offer.gpu_count}，"
                    f"{offer.price_per_hour:.3f} {offer.currency}/h（attempt {ctx.attempt}）"
                )
                outcome = self._try_offer(ctx, offer)
                # 本次尝试的启动文档（含存储凭据）不论成败都删掉。
                self.store.delete(layout.secrets_key(ctx.job_id, ctx.attempt))
                if outcome.kind in ("started", "ended"):
                    phase = outcome.phase or states.RUNNING
                    if phase in states.TERMINAL_PHASES and ctx.ref is not None:
                        self._destroy_ctx(ctx, ctx.ref, why=f"任务已结束（{phase}）")
                    if ctx.ref is not None:
                        self.echo(f"任务 {ctx.job_id} 已在实例 {ctx.ref.id} 上运行。")
                    return SubmitResult(
                        job_id=ctx.job_id,
                        attempt=ctx.attempt,
                        phase=phase,
                        backend=ctx.backend_name,
                        instance=ctx.ref,
                        offer=offer,
                        cost_ceiling=ctx.cost_ceiling,
                        currency=ctx.currency,
                    )
                needs_bump = outcome.instance_may_exist
                last_reason = outcome.reason or last_reason
                last_detail = outcome.detail
                if outcome.kind == "runner_timeout":
                    runner_timeouts += 1
                    if runner_timeouts >= MAX_CONSECUTIVE_RUNNER_TIMEOUTS:
                        message = (
                            f"连续 {runner_timeouts} 台机器上 runner 都没有发出心跳，"
                            "多半是镜像的问题"
                            f"（镜像 {ctx.request.image} 缺 python3 / bash，或 bootstrap 失败）。"
                            "请检查 job.yaml 的 image 后再提交。"
                        )
                        self._write_terminal_if_absent(
                            ctx.job_id,
                            ctx.attempt,
                            states.FAILED,
                            states.REASON_RUNNER_TIMEOUT,
                            message,
                        )
                        raise DgpuError(f"任务 {ctx.job_id}：{message}")
                elif outcome.kind == "preflight":
                    runner_timeouts = 0  # runner 跑起来了，镜像没问题
            message = f"试了 {tries} 台机器都没能启动任务"
            if last_detail:
                message += f"（最后一次：{last_detail}）"
            if tries >= cfg.max_offers and len(ctx.offers) > tries:
                message += f"；已达到 max_offers={cfg.max_offers}"
            self._write_terminal_if_absent(
                ctx.job_id, ctx.attempt, states.FAILED, last_reason, message
            )
            raise DgpuError(
                f"任务 {ctx.job_id}：{message}。可以稍后用 `dgpu retry {ctx.job_id}` 再试。"
            )
        except BaseException as exc:
            problems = self._abort_launch(ctx, exc)
            if problems and isinstance(exc, DgpuError):
                raise DgpuError(f"{exc}\n" + "\n".join(problems)) from exc
            raise

    def _try_offer(self, ctx: _Launch, offer: Offer) -> _Outcome:
        job_id, attempt = ctx.job_id, ctx.attempt
        launch = self._prepare_launch(ctx)
        ctx.label = launch.label
        ctx.ref = None
        ctx.create_unknown = True
        try:
            ref = ctx.backend.create(offer, launch)
        except OfferUnavailable as exc:
            ctx.create_unknown = False
            detail = f"offer {offer.id} 不可用：{exc}"
            self.echo(detail + "，换下一台")
            self._event(job_id, states.EV_CREATE_FAILED, attempt, detail, _offer_data(offer))
            return _Outcome("create_failed", states.REASON_NO_OFFER, False, detail)
        except BackendError as exc:
            if not exc.retryable:
                raise
            # 契约要求后端确认没创建才抛 retryable；这里再按标签查一次，宁可多一次 API 调用。
            found = self._lookup_label(ctx.backend, launch.label)
            ctx.create_unknown = False
            for stray in found:
                ctx.ref = stray
                self._destroy_ctx(ctx, stray, why="创建请求报错，但按标签查到了实例")
            detail = f"在 offer {offer.id} 上创建失败：{exc}"
            self.echo(detail + "，换下一台")
            self._event(
                job_id,
                states.EV_CREATE_FAILED,
                attempt,
                detail,
                {**_offer_data(offer), "found_by_label": [ref.id for ref in found]},
            )
            return _Outcome("create_failed", states.REASON_NO_OFFER, bool(found), detail)
        ctx.ref = ref
        ctx.create_unknown = False
        self.echo(f"已创建实例 {ref.id}（标签 {launch.label}），等待启动……")
        self._record_instance(ctx, ref)
        self._write_cli_status(
            job_id, attempt, states.STARTING, f"实例 {ref.id} 已创建，等待 runner 启动"
        )
        return self._await_runner(ctx, ref)

    def _await_runner(self, ctx: _Launch, ref: InstanceRef) -> _Outcome:
        """等 runner 启动并通过 preflight。

        runner 先写 RUNNING 和第一次心跳（step=preflight），再做 preflight；preflight 失败时
        写 FAILED(preflight) 并自毁，这时要换机。所以心跳的 step 离开 preflight 才算启动成功。
        """
        cfg = self.config
        job_id, attempt = ctx.job_id, ctx.attempt
        startup_deadline = self.clock() + cfg.startup_timeout_min * 60
        running_at: float | None = None
        preflight_since: float | None = None  # 第一次看到本 attempt 心跳的时间
        last_phase = None
        while True:
            ended = self._startup_status_outcome(ctx, ref)
            if ended is not None:
                return ended
            now = self.clock()
            hb = self._read_doc(layout.heartbeat_key(job_id), "heartbeat.json")
            if hb is not None and hb[0].get("attempt") == attempt:
                if hb[0].get("step") != PREFLIGHT_STEP:
                    self.echo("runner 已启动，preflight 通过。")
                    return _Outcome("started", phase=self._started_phase(job_id, attempt))
                if preflight_since is None:
                    preflight_since = now
                    self.echo("runner 已启动（收到第一次心跳），正在做 preflight……")
                elif now - preflight_since >= cfg.runner_timeout_min * 60:
                    self.echo(
                        f"runner 的 preflight 超过 {cfg.runner_timeout_min} 分钟还没结束，"
                        "不再等待；runner 仍在发心跳，"
                        "preflight 失败时任务会以 FAILED(preflight) 结束。"
                    )
                    return _Outcome("started", phase=self._started_phase(job_id, attempt))

            att = self._read_doc(layout.attempt_key(job_id), "attempt.json")
            if att is None:
                raise DgpuError(f"任务 {job_id} 的 attempt.json 不见了，停止启动。")
            if att[0].get("attempt") != attempt:
                raise AttemptConflict(job_id, attempt, att[0].get("attempt"))
            if att[0].get("cancel"):
                self.echo("任务已被取消（dgpu cancel），正在销毁实例。")
                self._destroy_ctx(ctx, ref, why="启动阶段被取消")
                self._write_terminal_if_absent(
                    job_id, attempt, states.CANCELLED, None, "启动阶段被取消，实例已销毁"
                )
                raise _CancelRequested(f"任务 {job_id} 在启动阶段被取消，实例 {ref.id} 已销毁。")

            state = ctx.backend.get(ref)
            now = self.clock()
            if state.phase != last_phase:
                self.echo(f"实例 {ref.id} 状态：{state.phase}（{state.raw_status or '-'}）")
                last_phase = state.phase
            if state.phase in _DEAD_INSTANCE_PHASES:
                # runner 可能在上面读过 status.json 之后才写最终状态并自毁：再看一次
                ended = self._startup_status_outcome(ctx, ref)
                if ended is not None:
                    return ended
                where = f"（平台状态 {state.raw_status or '-'}）"
                if preflight_since is not None:
                    detail = f"实例 {ref.id} 在 preflight 期间进入 {state.phase}{where}"
                    return self._preflight_failed(ctx, ref, detail)
                if running_at is None:
                    detail = f"实例 {ref.id} 在启动阶段进入 {state.phase}{where}"
                    return self._startup_failed(ctx, ref, detail)
                detail = f"实例 {ref.id} 运行后在第一次心跳前进入 {state.phase}，bootstrap 可能失败"
                return self._runner_timeout(ctx, ref, detail)
            if preflight_since is None:  # 还没有心跳
                if running_at is None:
                    if state.phase == INST_RUNNING:
                        running_at = now
                    elif now >= startup_deadline:
                        detail = (
                            f"实例 {ref.id} 在 {cfg.startup_timeout_min} 分钟内没有进入运行状态"
                            f"（平台状态 {state.raw_status or state.phase}）"
                        )
                        return self._startup_failed(ctx, ref, detail)
                elif now >= running_at + cfg.runner_timeout_min * 60:
                    detail = f"实例 {ref.id} 运行 {cfg.runner_timeout_min} 分钟后仍没有心跳"
                    return self._runner_timeout(ctx, ref, detail)
            self.sleep(POLL_STARTUP_S)

    def _startup_status_outcome(self, ctx: _Launch, ref: InstanceRef) -> _Outcome | None:
        """启动阶段 runner 已经写了本 attempt 的终态：preflight 失败就换机，否则任务已结束。"""
        st = self._read_doc(layout.status_key(ctx.job_id), "status.json")
        if st is None or st[0].get("attempt") != ctx.attempt:
            return None
        doc = st[0]
        if doc.get("phase") not in states.TERMINAL_PHASES:
            return None
        if doc.get("reason") == states.REASON_PREFLIGHT:
            detail = f"实例 {ref.id} preflight 失败：{doc.get('message') or '（无说明）'}"
            return self._preflight_failed(ctx, ref, detail)
        # runner 已经启动并结束（不是机器的问题），不换机。
        self.echo(f"runner 在启动阶段就结束了：{doc['phase']}（{doc.get('reason') or '-'}）")
        self._destroy_ctx(ctx, ref, why=f"任务已结束（{doc['phase']}）")
        return _Outcome("ended", phase=doc["phase"])

    def _started_phase(self, job_id: str, attempt: int) -> str:
        st = self._read_doc(layout.status_key(job_id), "status.json")
        if st is not None and st[0].get("attempt") == attempt:
            if st[0].get("phase") not in _CLI_PHASES:
                return st[0]["phase"]
        return states.RUNNING

    def _preflight_failed(self, ctx: _Launch, ref: InstanceRef, detail: str) -> _Outcome:
        self.echo(detail + "，换下一台")
        self._event(
            ctx.job_id, states.EV_PREFLIGHT_FAILED, ctx.attempt, detail, {"instance_id": ref.id}
        )
        self._destroy_ctx(ctx, ref, why="preflight 失败")
        return _Outcome("preflight", states.REASON_PREFLIGHT, True, detail)

    def _startup_failed(self, ctx: _Launch, ref: InstanceRef, detail: str) -> _Outcome:
        self.echo(detail + "，销毁并换下一台")
        self._event(
            ctx.job_id, states.EV_STARTUP_TIMEOUT, ctx.attempt, detail, {"instance_id": ref.id}
        )
        self._destroy_ctx(ctx, ref, why="启动失败")
        return _Outcome("startup_failed", states.REASON_STARTUP, True, detail)

    def _runner_timeout(self, ctx: _Launch, ref: InstanceRef, detail: str) -> _Outcome:
        self.echo(detail + "，销毁并换下一台")
        self._event(
            ctx.job_id, states.EV_RUNNER_TIMEOUT, ctx.attempt, detail, {"instance_id": ref.id}
        )
        self._destroy_ctx(ctx, ref, why="runner 没有心跳")
        return _Outcome("runner_timeout", states.REASON_RUNNER_TIMEOUT, True, detail)

    def _abort_launch(self, ctx: _Launch, exc: BaseException) -> list[str]:
        """异常路径：销毁（可能）创建的实例，写终态，删启动文档。返回需要大声报告的问题。"""
        problems: list[str] = []
        interrupted = isinstance(exc, KeyboardInterrupt)
        if interrupted:
            self.echo("收到中断，正在清理：销毁已创建的实例……")

        def report(step: str, cleanup_exc: BaseException) -> None:
            # 清理本身失败（含再次 Ctrl-C）：大声说，继续做剩下的步骤，不吞原异常
            target = ctx.ref.id if ctx.ref is not None else (ctx.label or "?")
            message = (
                f"!! 清理任务 {ctx.job_id} 时{step}出错：{cleanup_exc!r}。"
                f"实例 {target} 可能仍在计费，请运行 `dgpu gc` 或到平台控制台检查。"
            )
            self.echo(message)
            problems.append(message)

        # 1. 销毁已创建或可能已创建的实例
        try:
            refs: list[InstanceRef] = []
            if ctx.ref is not None:
                refs.append(ctx.ref)
            elif ctx.create_unknown and ctx.label:
                refs.extend(self._lookup_label(ctx.backend, ctx.label))
            for ref in refs:
                try:
                    self._destroy_ctx(
                        ctx, ref, why="submit 异常退出" if not interrupted else "Ctrl-C"
                    )
                except DestroyFailed as destroy_exc:
                    problems.append(str(destroy_exc))
        except BaseException as cleanup_exc:
            report("销毁实例", cleanup_exc)
        # 2. 写终态（runner 已写或已被别的进程接管时不写）
        if not isinstance(exc, AttemptConflict | _CancelRequested):
            if interrupted:
                phase, reason = states.CANCELLED, None
                message = "submit 被中断（Ctrl-C），启动中的实例已销毁"
            else:
                phase, reason = states.FAILED, states.REASON_STARTUP
                message = f"启动过程中出错：{exc}"
            if problems:
                message = message.replace("，启动中的实例已销毁", "")
                message += "；有实例可能没有销毁，见 `dgpu gc`"
            try:
                self._write_terminal_if_absent(ctx.job_id, ctx.attempt, phase, reason, message)
            except BaseException as cleanup_exc:
                report("写终态", cleanup_exc)
        # 3. 删掉含存储凭据的启动文档
        try:
            self.store.delete(layout.secrets_key(ctx.job_id, ctx.attempt))
        except BaseException as cleanup_exc:
            report("删除启动文档", cleanup_exc)
        return problems

    def _prepare_launch(self, ctx: _Launch) -> LaunchSpec:
        """写本次尝试的启动文档（含存储凭据），生成只含预签名 URL 的启动命令。"""
        job_id, attempt = ctx.job_id, ctx.attempt
        key = layout.secrets_key(job_id, attempt)
        doc = build_launch_doc(
            job_id=job_id, attempt=attempt, backend=ctx.backend_name, store=self.store.profile
        )
        self.store.put_json(key, doc)
        ttl = int(
            (self.config.startup_timeout_min + self.config.runner_timeout_min) * 60
            + LAUNCH_URL_EXTRA_S
        )
        bundle_url = self.store.presign_get(ctx.bundle.key, ttl)
        launch_url = self.store.presign_get(key, ttl)
        command = self.bundler.bootstrap_command(
            bundle_url=bundle_url, bundle_sha256=ctx.bundle.sha256, launch_url=launch_url
        )
        return LaunchSpec(
            job_id=job_id,
            attempt=attempt,
            image=ctx.request.image,
            bootstrap_cmd=command,
            disk_gb=ctx.request.disk_gb,
            max_price_per_hour=ctx.request.max_price_per_hour,
        )

    def _record_instance(self, ctx: _Launch, ref: InstanceRef) -> None:
        def record(doc: dict[str, Any]) -> dict[str, Any]:
            doc["instance_id"] = ref.id
            doc["label"] = ctx.label or ref.label
            return doc

        self._update_attempt(ctx.job_id, record, expect_attempt=ctx.attempt)

    def _bump_attempt(self, ctx: _Launch, why: str) -> None:
        """换机：attempt + 1（条件写），隔离可能半启动的旧 runner。"""
        old = ctx.attempt

        def bump(doc: dict[str, Any]) -> dict[str, Any]:
            if doc.get("cancel"):
                raise _CancelRequested(f"任务 {ctx.job_id} 已被取消，不再换机。")
            new_attempt = int(doc["attempt"]) + 1
            doc.update(
                attempt=new_attempt,
                label=layout.label(ctx.job_id, new_attempt),
                instance_id=None,
                created_at=self.clock(),
                deadline_at=self._deadline(ctx.request.max_hours),
            )
            return doc

        try:
            doc = self._update_attempt(ctx.job_id, bump, expect_attempt=old)
        except _CancelRequested:
            self._write_terminal_if_absent(
                ctx.job_id, old, states.CANCELLED, None, "换机前发现任务已被取消"
            )
            raise
        ctx.attempt = int(doc["attempt"])
        self._event(
            ctx.job_id,
            states.EV_RELAUNCH,
            ctx.attempt,
            f"换机：attempt {old} 作废，改用 attempt {ctx.attempt}（{why}）",
            {"old_attempt": old, "new_attempt": ctx.attempt, "why": why},
        )
        self._write_cli_status(
            ctx.job_id,
            ctx.attempt,
            states.PROVISIONING,
            f"换机：正在创建 attempt {ctx.attempt} 的实例",
        )

    # ============================================================ 销毁

    def _destroy(
        self,
        backend: Backend,
        ref: InstanceRef,
        *,
        job_id: str,
        attempt: int | None,
        why: str,
    ) -> None:
        """销毁并确认实例消失。失败时大声输出并抛 DestroyFailed。"""
        self.echo(f"正在销毁实例 {ref.backend}/{ref.id}（{why}）……")
        try:
            backend.destroy(ref)
            self._confirm_gone(backend, ref)
        except Exception as exc:
            message = (
                f"!! 销毁实例 {ref.backend}/{ref.id}（标签 {ref.label}）失败：{exc}。"
                "它可能仍在计费：请运行 `dgpu gc`，或到平台控制台手动销毁。"
            )
            self.echo(message)
            self._event(
                job_id,
                states.EV_DESTROY_FAILED,
                attempt,
                message,
                {"backend": ref.backend, "instance_id": ref.id, "label": ref.label, "why": why},
            )
            raise DestroyFailed(ref, message) from exc
        self._event(
            job_id,
            states.EV_INSTANCE_DESTROYED,
            attempt,
            f"已销毁实例 {ref.id}：{why}",
            {"backend": ref.backend, "instance_id": ref.id, "label": ref.label, "why": why},
        )

    def _destroy_ctx(self, ctx: _Launch, ref: InstanceRef, *, why: str) -> None:
        self._destroy(ctx.backend, ref, job_id=ctx.job_id, attempt=ctx.attempt, why=why)
        if ctx.ref is not None and ctx.ref.id == ref.id:
            ctx.ref = None

    def _confirm_gone(self, backend: Backend, ref: InstanceRef) -> None:
        start = self.clock()
        last_issue = start
        while True:
            state = backend.get(ref)
            if state.phase == INST_GONE:
                return
            now = self.clock()
            if now - start >= DESTROY_CONFIRM_TIMEOUT_S:
                raise DgpuError(
                    f"已请求销毁，但 {int(DESTROY_CONFIRM_TIMEOUT_S)} 秒后实例仍是 {state.phase}"
                    f"（{state.raw_status or '-'}）"
                )
            if now - last_issue >= DESTROY_REISSUE_S:
                backend.destroy(ref)
                last_issue = now
            self.sleep(DESTROY_CONFIRM_POLL_S)

    def _lookup_label(self, backend: Backend, label: str) -> list[InstanceRef]:
        """按精确标签查实例（list 按前缀匹配，a1 也是 a10 的前缀，所以再精确比较）。"""
        return [
            state.ref
            for state in backend.list(label)
            if state.ref.label == label and state.phase != INST_GONE
        ]

    def _destroy_job_instances(
        self, job_id: str, backend: Backend, att_doc: dict[str, Any], *, why: str
    ) -> None:
        """销毁这个任务在该平台上的所有实例（attempt.json 记录的和按标签查到的）。失败即抛。"""
        refs: dict[str, InstanceRef] = {}
        attempt = att_doc.get("attempt")
        if att_doc.get("instance_id"):
            label = att_doc.get("label") or layout.label(job_id, int(attempt or 1))
            ref = InstanceRef(backend=backend.name, id=str(att_doc["instance_id"]), label=label)
            refs[ref.id] = ref
        for state in backend.list(f"{layout.LABEL_PREFIX}{job_id}-a"):
            parsed = layout.parse_label(state.ref.label)
            if parsed is None or parsed[0] != job_id or state.phase == INST_GONE:
                continue
            refs.setdefault(state.ref.id, state.ref)
        for ref in refs.values():
            if backend.get(ref).phase == INST_GONE:
                continue
            parsed = layout.parse_label(ref.label)
            self._destroy(
                backend, ref, job_id=job_id, attempt=parsed[1] if parsed else attempt, why=why
            )

    def _cleanup_after_end(self, job_id: str) -> None:
        """任务到终态后确认实例已销毁（AutoDL 需要 CLI 删除部署记录），删掉残留的启动文档。"""
        got = self._read_doc(layout.attempt_key(job_id), "attempt.json")
        if got is None:
            return
        att = got[0]
        self.store.delete(layout.secrets_key(job_id, int(att["attempt"])))
        backend = self.backends.get(att.get("backend"))
        if backend is None:
            self.echo(
                f"注意：没有 {att.get('backend')} 平台的凭据，无法确认实例已销毁；"
                "请在有凭据的电脑上跑 `dgpu gc`。"
            )
            return
        try:
            self._destroy_job_instances(job_id, backend, att, why="任务已结束，清理实例")
        except DgpuError:
            pass  # _destroy 已经大声输出了实例 ID；任务结果照常返回，gc 会再试

    # ============================================================ attempt.json / status.json

    def _update_attempt(
        self,
        job_id: str,
        mutate: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        expect_attempt: int | None = None,
    ) -> dict[str, Any]:
        """attempt.json 的读改写，用 If-Match 条件写；冲突时重读重试。

        每次写入带一个新的 writer 标记。条件写其实成功了、但响应丢失时，存储层的重试会
        收到 PreconditionFailed；这时重新读取，存储里正是本次要写的文档（含 writer），
        就算成功，不再重复执行 mutate（比如重复把 attempt + 1）。
        """
        key = layout.attempt_key(job_id)
        for _ in range(CAS_TRIES):
            got = self._read_doc(key, "attempt.json")
            if got is None:
                raise DgpuError(f"任务 {job_id} 没有 attempt.json。")
            doc, etag = got
            if expect_attempt is not None and doc.get("attempt") != expect_attempt:
                raise AttemptConflict(job_id, expect_attempt, doc.get("attempt"))
            new_doc = mutate(dict(doc))
            new_doc["writer"] = states.new_writer_token()
            try:
                self.store.put_json(key, new_doc, if_match=etag)
                return new_doc
            except PreconditionFailed:
                if self._landed(key, new_doc):
                    return new_doc
                self.sleep(CAS_BACKOFF_S)
        raise DgpuError(
            f"任务 {job_id} 的 attempt.json 连续 {CAS_TRIES} 次条件写冲突，请稍后重试。"
        )

    def _landed(self, key: str, intended: dict[str, Any]) -> bool:
        """条件写报“条件不满足”之后：存储里是否正是本次要写的文档（之前的请求其实已生效）。"""
        got = self.store.get_json(key)
        return got is not None and got[0] == intended

    def _create_doc(self, key: str, doc: dict[str, Any]) -> None:
        """只在对象不存在时创建（If-None-Match）。响应丢失后重试报“已存在”时，按内容判断。"""
        try:
            self.store.put_json(key, doc, if_none_match=True)
        except PreconditionFailed:
            if not self._landed(key, doc):
                raise

    def _write_cli_status(self, job_id: str, attempt: int, phase: str, message: str) -> bool:
        """写 PROVISIONING / STARTING。runner 已接手（同一 attempt 的其他阶段）就不覆盖。"""
        key = layout.status_key(job_id)
        for _ in range(CAS_TRIES):
            got = self._read_doc(key, "status.json")
            if got is not None:
                doc, etag = got
                current = doc.get("attempt") or 0
                if current > attempt:
                    return False
                if current == attempt and doc.get("phase") not in _CLI_PHASES:
                    return False
            new_doc = states.new_status_doc(
                job_id,
                attempt,
                phase,
                states.SRC_CLI,
                message=redact(message, self._secrets),
                now=self.clock(),
            )
            try:
                if got is None:
                    self.store.put_json(key, new_doc, if_none_match=True)
                else:
                    self.store.put_json(key, new_doc, if_match=got[1])
                return True
            except PreconditionFailed:
                self.sleep(CAS_BACKOFF_S)
        raise DgpuError(f"任务 {job_id} 的 status.json 连续 {CAS_TRIES} 次条件写冲突。")

    def _write_terminal_if_absent(
        self, job_id: str, attempt: int, phase: str, reason: str | None, message: str
    ) -> bool:
        """写终态，除非这个 attempt 已有终态，或任务已被更新的 attempt 接管。"""
        key = layout.status_key(job_id)
        for _ in range(CAS_TRIES):
            att = self._read_doc(layout.attempt_key(job_id), "attempt.json")
            if att is not None and (att[0].get("attempt") or 0) > attempt:
                return False
            got = self._read_doc(key, "status.json")
            if got is not None:
                doc = got[0]
                current = doc.get("attempt") or 0
                if current > attempt:
                    return False
                if current == attempt and doc.get("phase") in states.TERMINAL_PHASES:
                    return False
            new_doc = states.new_status_doc(
                job_id,
                attempt,
                phase,
                states.SRC_CLI,
                reason=reason,
                message=redact(message, self._secrets),
                now=self.clock(),
            )
            try:
                if got is None:
                    self.store.put_json(key, new_doc, if_none_match=True)
                else:
                    self.store.put_json(key, new_doc, if_match=got[1])
                return True
            except PreconditionFailed:
                self.sleep(CAS_BACKOFF_S)
        raise DgpuError(f"任务 {job_id} 的 status.json 连续 {CAS_TRIES} 次条件写冲突。")

    # ============================================================ 内部工具

    def _backend(self, name: str | None) -> Backend:
        backend = self.backends.get(name or "")
        if backend is None:
            configured = "、".join(self.backends) or "无"
            raise DgpuError(
                f"后端 {name} 没有配置（已配置：{configured}）。"
                f"在配置文件的 [backends.{name}] 里填写凭据，或用 --backend 换一个。"
            )
        return backend

    def _warn_foreign_buckets(self, spec: JobSpec) -> None:
        """data 里引用了配置的 bucket 之外的 bucket：实例用同一个存储 token 去读，提醒权限。"""
        own = self.store.profile.bucket
        others = sorted(
            {
                ref.bucket
                for ref in spec.data
                if not ref.is_local and ref.bucket and ref.bucket != own
            }
        )
        for bucket in others:
            self.echo(
                f"注意：data 里引用了 bucket {bucket}，它不是存储 {self.store.profile.name} 配置的 "
                f"bucket {own}。实例用这个存储的 token 读取数据：token 必须有 bucket {bucket} 的"
                "读权限，否则下载数据会失败（FAILED(data_sync)）。"
            )

    def _check_local_file_sizes(self, spec: JobSpec) -> None:
        """本地数据里有超过单次上传上限的文件：上传任何东西之前就报错。"""
        too_big = _oversized_local_files(spec, SINGLE_UPLOAD_LIMIT)
        if not too_big:
            return
        name, path, size = too_big[0]
        more = f"（另有 {len(too_big) - 1} 个）" if len(too_big) > 1 else ""
        bucket = self.store.profile.bucket
        raise DgpuError(
            f"本地数据 {name} 里的文件 {path}（{size / GB:.1f} GiB）{more}超过存储单次上传的上限 "
            f"{SINGLE_UPLOAD_LIMIT / GB:.0f} GiB，dgpu 不会分段上传它。"
            "请先用 rclone 把这份数据传到 bucket"
            f"（如 rclone copy <数据目录> <remote>:{bucket}/datasets/{name}），"
            f"再在 job.yaml 的 data 里写成 r2://{bucket}/datasets/{name}。没有上传任何东西，也没有创建实例。"
        )

    def _check_store_profile(self, backend_name: str) -> None:
        wanted = self.config.store_for(backend_name).name
        if wanted != self.store.profile.name:
            raise DgpuError(
                f"后端 {backend_name} 配置的存储是 {wanted!r}，但 dgpu 的所有命令目前共用存储 "
                f"{self.store.profile.name!r}（多个存储 profile 还未支持）。"
                "请让两个后端用同一个存储。"
            )

    def _check_job_id(self, job_id: str) -> None:
        if not layout.is_valid_job_id(job_id):
            raise DgpuError(f"任务 ID {job_id!r} 格式不对：只能包含小写字母、数字和连字符。")

    def _read_doc(self, key: str, what: str) -> tuple[dict[str, Any], str] | None:
        got = self.store.get_json(key)
        if got is None:
            return None
        doc, etag = got
        try:
            states.check_version(doc, what)
        except ValueError as exc:
            raise DgpuError(f"{key}：{exc}。可能是更新版本的 dgpu 写的，请升级 dgpu。") from None
        return doc, etag

    def _read_request(self, job_id: str) -> tuple[GpuRequest, str]:
        got = self._read_doc(layout.request_key(job_id), "request.json")
        if got is None:
            raise DgpuError(f"任务 {job_id} 缺少 request.json（GPU 需求），无法重试。")
        doc = got[0]
        request = GpuRequest(
            names=tuple(doc["gpu_names"]),
            count=int(doc["gpu_count"]),
            max_price_per_hour=float(doc["max_price_per_hour"]),
            max_hours=float(doc["max_hours"]),
            disk_gb=int(doc["disk_gb"]),
            image=str(doc["image"]),
        )
        return request, str(doc["backend"])

    def _build_request(self, spec: JobSpec, backend_name: str) -> tuple[GpuRequest, int, bool]:
        max_price = spec.limits.max_price(backend_name)
        image = spec.image_for(backend_name)
        local_bytes = _local_data_bytes(spec)
        has_remote = any(not ref.is_local for ref in spec.data)
        if self.config.disk_gb is not None:
            disk_gb = int(self.config.disk_gb)
        else:
            disk_gb = max(MIN_DISK_GB, math.ceil(2 * local_bytes / GB + 20))
            if backend_name == BACKEND_AUTODL and disk_gb > AUTODL_DATA_DISK_GB:
                raise DgpuError(
                    f"AutoDL 的数据盘固定 {AUTODL_DATA_DISK_GB} GB，本地数据 "
                    f"{local_bytes / GB:.1f} GB 按 2×数据量+20 估算需要 {disk_gb} GB。"
                    "请减少数据量，或换 --backend vast；确认空间够用时可以在配置里设置 disk_gb。"
                )
        request = GpuRequest(
            names=tuple(spec.gpu.names),
            count=spec.gpu.count,
            max_price_per_hour=max_price,
            max_hours=spec.limits.max_hours,
            disk_gb=disk_gb,
            image=image,
        )
        return request, local_bytes, has_remote

    def _describe_constraints(self, backend_name: str, req: GpuRequest, has_remote: bool) -> str:
        currency = CURRENCY_BY_BACKEND[backend_name]
        parts = [
            f"GPU 为 {' / '.join(req.names)} 之一，至少 {req.count} 张",
            f"每小时价格 ≤ {req.max_price_per_hour:g} {currency}",
            f"可连续租用 ≥ {req.max_hours:g} 小时",
            f"磁盘 ≥ {req.disk_gb} GB" + ("（远程数据大小未知，未计入）" if has_remote else ""),
            f"镜像 {req.image}",
        ]
        if backend_name == BACKEND_VAST:
            vast = self.config.vast
            parts.append(
                f"Vast 筛选：可靠性 ≥ {vast.reliability:g}，"
                f"下载带宽 ≥ {vast.min_inet_down_mbps:g} Mbps，"
                f"流量单价 ≤ {vast.max_bandwidth_usd_per_gb:g} USD/GB"
                + ("，只用数据中心机器" if vast.datacenter_only else "")
            )
        elif backend_name == BACKEND_AUTODL:
            autodl = self.config.autodl
            regions = "、".join(autodl.dc_list) or "不限"
            parts.append(f"AutoDL 筛选：地区 {regions}，CUDA ≥ {autodl.cuda_v_from}")
        return "；".join(parts)

    def _no_offer_message(self, backend_name: str, req: GpuRequest, has_remote: bool) -> str:
        price_field = "max_usd_per_hour" if backend_name == BACKEND_VAST else "max_cny_per_hour"
        return (
            f"{backend_name} 上没有符合条件的机器。"
            f"条件：{self._describe_constraints(backend_name, req, has_remote)}。"
            f"可以放宽 gpu.names、提高 limits.{price_field}、减小 limits.max_hours，"
            "或用 --backend 换平台。没有创建实例，也没有写入存储。"
        )

    def _print_ceiling(self, req: GpuRequest, currency: str) -> float:
        ceiling = req.max_price_per_hour * req.max_hours
        self.echo(
            f"费用上限：{req.max_price_per_hour:g} {currency}/h × {req.max_hours:g} h = "
            f"{ceiling:.2f} {currency}（不含流量费）"
        )
        return ceiling

    def _check_balance(self, backend: Backend, ceiling: float, currency: str) -> None:
        balance = backend.balance()
        if balance is None:
            self.echo("平台没有提供余额，跳过余额检查。")
            return
        if balance.currency != currency:
            self.echo(
                f"注意：余额的货币是 {balance.currency}，费用上限是 {currency}，"
                "无法比较，跳过余额检查。"
            )
            return
        if balance.amount < ceiling:
            raise DgpuError(
                f"账户余额 {balance.amount:.2f} {currency} 不够覆盖本任务的费用上限 "
                f"{ceiling:.2f} {currency}。请充值，或降低 limits.max_hours / 价格上限。"
                "没有创建实例，也没有写入存储。"
            )
        self.echo(f"账户余额 {balance.amount:.2f} {currency}，够用。")

    def _upload_job_files(
        self, spec: JobSpec, job_id: str, backend_name: str, request: GpuRequest
    ) -> None:
        self.echo(f"打包代码 {spec.code} ……")
        # data 里的本地目录单独上传；它们在代码目录里时不能再打进代码包
        local_data = [Path(ref.local_path) for ref in spec.data if ref.local_path is not None]
        with tempfile.TemporaryDirectory(prefix="dgpu-") as tmp:
            archive = Path(tmp) / "code.tar.gz"
            self.deps.pack_code(Path(spec.code), archive, exclude=local_data)
            self.store.put_file(layout.code_key(job_id), archive)
        sources: list[DataSource] = []
        for ref in spec.data:
            if ref.is_local:
                self.echo(f"准备数据 {ref.name}（本地 {ref.local_path}）……")
            sources.append(self.deps.resolve_data(self.store, ref))
        remote_spec = build_remote_spec(
            spec, job_id=job_id, backend=backend_name, data=sources, runner=self.config.runner
        )
        self.store.put_json(layout.spec_key(job_id), remote_spec, if_none_match=True)
        self.store.put_json(
            layout.request_key(job_id),
            {
                "version": states.SCHEMA_VERSION,
                "job_id": job_id,
                "backend": backend_name,
                "gpu_names": list(request.names),
                "gpu_count": request.count,
                "max_price_per_hour": request.max_price_per_hour,
                "currency": CURRENCY_BY_BACKEND[backend_name],
                "max_hours": request.max_hours,
                "disk_gb": request.disk_gb,
                "image": request.image,
            },
            if_none_match=True,
        )

    def _deadline(self, max_hours: float) -> float:
        """超过这个时间仍存在的实例可以被 gc 销毁。

        启动窗口 + 实例最晚自己消失的时间（watchdog 硬截止 + watchdog 收尾，和 runner 用同一个
        公式、生产环境的默认调参）+ 30 分钟余量。
        """
        cfg = self.config
        settings = runner_util.runner_settings({"runner": cfg.runner.to_dict()})
        lifetime = runner_util.instance_lifetime_s(max_hours, settings, runner_util.DEFAULT_TUNING)
        startup = (cfg.startup_timeout_min + cfg.runner_timeout_min) * 60
        return self.clock() + startup + lifetime + DEADLINE_MARGIN_S

    def _event(
        self,
        job_id: str,
        name: str,
        attempt: int | None,
        message: str,
        data: dict[str, Any] | None = None,
        *,
        source: str = states.SRC_CLI,
    ) -> None:
        # 事件里常带平台返回的错误文本，写进存储前抹掉已知凭据（红线 2、7）
        message = redact(message, self._secrets)
        data = json.loads(redact(json.dumps(data or {}, ensure_ascii=False), self._secrets))
        ts = self.clock()
        doc = states.new_event_doc(name, source, job_id, attempt, message, data, now=ts)
        try:
            self.store.put_json(layout.event_key(job_id, source, ts=ts), doc)
        except Exception as exc:  # 事件写失败不能打断清理，但要让人看到
            self.echo(f"警告：写事件 {name} 失败（{exc}）：{message}")

    def _view(
        self,
        job_id: str,
        att: dict[str, Any] | None,
        st: dict[str, Any] | None,
        hb: dict[str, Any] | None,
    ) -> JobView:
        cfg = self.config
        now = self.clock()
        current = (att or {}).get("attempt") or (st or {}).get("attempt")
        status_attempt = (st or {}).get("attempt")
        if st is None or (status_attempt or 0) < (current or 0):
            base_phase = states.PROVISIONING  # 已换到新的 attempt，状态还没写
            base_time = (att or {}).get("created_at")
        else:
            base_phase = st.get("phase") or states.PROVISIONING
            base_time = st.get("updated_at")
        hb_current = hb if hb is not None and hb.get("attempt") == current else None
        if hb_current is not None and base_phase in _CLI_PHASES:
            base_phase = states.RUNNING  # runner 已发心跳，状态写入稍晚
        lost = False
        hb_age = now - hb_current["ts"] if hb_current and hb_current.get("ts") else None
        if base_phase in (states.RUNNING, states.UPLOADING):
            ref_time = hb_current.get("ts") if hb_current else base_time
            lost = ref_time is not None and now - ref_time > cfg.lost_after_min * 60
        elif base_phase in _CLI_PHASES:
            window = (cfg.startup_timeout_min + cfg.runner_timeout_min + cfg.lost_after_min) * 60
            lost = base_time is not None and now - base_time > window
        return JobView(
            job_id=job_id,
            phase=states.LOST if lost else base_phase,
            stored_phase=(st or {}).get("phase"),
            attempt=current,
            status_attempt=status_attempt,
            backend=(att or {}).get("backend"),
            instance_id=(att or {}).get("instance_id"),
            label=(att or {}).get("label"),
            reason=(st or {}).get("reason"),
            message=(st or {}).get("message"),
            exit_code=(st or {}).get("exit_code"),
            not_uploaded=list((st or {}).get("not_uploaded") or []),
            cancel_requested=bool((att or {}).get("cancel")),
            deadline_at=(att or {}).get("deadline_at"),
            status_updated_at=(st or {}).get("updated_at"),
            heartbeat_at=hb_current.get("ts") if hb_current else None,
            heartbeat_age_s=hb_age,
            step=hb_current.get("step") if hb_current else None,
            gpu_util=hb_current.get("gpu_util") if hb_current else None,
            synced_bytes=hb_current.get("synced_bytes") if hb_current else None,
            last_sync_at=hb_current.get("last_sync_at") if hb_current else None,
            lost=lost,
        )

    def _load_job_docs(self, job_id: str) -> _JobDocs:
        att = self._read_doc(layout.attempt_key(job_id), "attempt.json")
        st = self._read_doc(layout.status_key(job_id), "status.json")
        hb = self._read_doc(layout.heartbeat_key(job_id), "heartbeat.json")
        exists = att is not None or st is not None or self.store.exists(layout.spec_key(job_id))
        return _JobDocs(
            exists=exists,
            attempt=att[0] if att else None,
            status=st[0] if st else None,
            heartbeat=hb[0] if hb else None,
        )

    def _gc_decide(
        self, docs: _JobDocs, attempt: int, inst: InstanceState, now: float
    ) -> tuple[bool, str]:
        cfg = self.config
        inst_age = now - inst.created_at if inst.created_at is not None else None
        if not docs.exists:
            if inst_age is None:
                return False, "存储里没有这个任务，实例创建时间未知，保留"
            if inst_age > ORPHAN_MIN_AGE_S:
                return True, f"存储里没有这个任务，实例已存在 {inst_age / 60:.0f} 分钟"
            return False, "存储里没有这个任务，但实例创建不到 1 小时（可能正在别处提交），保留"
        att, st, hb = docs.attempt, docs.status, docs.heartbeat
        current = (att or {}).get("attempt") or (st or {}).get("attempt")
        if current is not None and attempt < current:
            return True, f"旧 attempt（当前是 attempt {current}）"
        if (
            st is not None
            and st.get("phase") in states.TERMINAL_PHASES
            and (st.get("attempt") or 0) >= attempt
        ):
            return True, f"任务已结束（{st['phase']}）"
        deadline = (att or {}).get("deadline_at")
        if deadline is not None and attempt == current and now > deadline:
            return True, f"超过截止时间 {(now - deadline) / 60:.0f} 分钟"
        hb_mine = hb if hb is not None and hb.get("attempt") == attempt else None
        hb_age = now - hb_mine["ts"] if hb_mine is not None and hb_mine.get("ts") else None
        if hb_age is not None and hb_age <= cfg.gc_stale_min * 60:
            return False, f"运行中（{hb_age:.0f} 秒前有心跳）"
        age = inst_age
        if age is None and att is not None and attempt == current and att.get("created_at"):
            age = now - att["created_at"]
        startup_window = (cfg.startup_timeout_min + cfg.runner_timeout_min) * 60
        if age is None:
            return False, "没有新心跳，但实例年龄未知，保留"
        if age <= startup_window:
            return False, "实例还在启动窗口内，保留"
        if hb_age is None:
            return True, f"实例已存在 {age / 60:.0f} 分钟仍没有心跳"
        return True, f"心跳已过期 {hb_age / 60:.0f} 分钟"

    def _check_store(self) -> CheckResult:
        name = f"存储 {self.store.profile.name}（bucket {self.store.profile.bucket}）"
        key = f"{DOCTOR_PREFIX}probe-{secrets.token_hex(4)}.json"
        probe = {"version": states.SCHEMA_VERSION, "probe": key, "ts": self.clock()}
        try:
            self.store.put_json(key, probe, if_none_match=True)
            got = self.store.get_json(key)
            if got is None or got[0].get("probe") != key:
                return CheckResult(name, False, "写入后读不回同样的内容")
            try:
                self.store.put_json(key, probe, if_none_match=True)
                conditional_ok = False
            except PreconditionFailed:
                conditional_ok = True
            self.store.delete(key)
            if self.store.exists(key):
                return CheckResult(name, False, "删除后对象仍然存在")
            if not conditional_ok:
                return CheckResult(
                    name,
                    False,
                    "条件写（If-None-Match）不生效，无法保证单写者；换一个支持条件写的存储",
                )
            return CheckResult(name, True, "读、写、条件写、删除都正常")
        except Exception as exc:
            try:
                self.store.delete(key)
            except Exception:  # 探针残留只是一个小对象，主错误已经报告
                pass
            return CheckResult(name, False, f"{type(exc).__name__}: {exc}")

    def _check_backend(self, name: str, backend: Backend) -> CheckResult:
        try:
            balance = backend.balance()
            if balance is not None:
                return CheckResult(
                    f"平台 {name}", True, f"可以访问，余额 {balance.amount:.2f} {balance.currency}"
                )
            probe = GpuRequest(
                names=("RTX 4090", "RTX 3090"),
                count=1,
                max_price_per_hour=1000.0,
                max_hours=1,
                disk_gb=MIN_DISK_GB,
                image="",
            )
            found = backend.search(probe)
            return CheckResult(
                f"平台 {name}", True, f"可以访问（平台不提供余额；搜索返回 {len(found)} 个 offer）"
            )
        except Exception as exc:
            return CheckResult(f"平台 {name}", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- 模块级工具


class _LogReader:
    """用范围读取增量拉日志，按完整的行输出；跨块的 UTF-8 字符不会被拆坏。"""

    def __init__(self, store: Store, key: str, out: Callable[[str], None]):
        self.store = store
        self.key = key
        self.out = out
        self.offset = 0
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def pump(self) -> bool:
        data = self.store.get_bytes(self.key, start=self.offset)
        if data is None:
            return False
        if data:
            self.offset += len(data)
            text = self.pending + self.decoder.decode(data)
            *lines, self.pending = text.split("\n")
            for line in lines:
                self.out(line)
        return True

    def flush(self) -> None:
        tail = self.pending + self.decoder.decode(b"", final=True)
        self.pending = ""
        if tail:
            self.out(tail)


def secret_values(config: Config, store: Store | None = None) -> tuple[str, ...]:
    """配置里的所有凭据，用于从输出和事件里抹掉。太短的值不算（避免误伤普通文本）。"""
    values = [config.vast.api_key, config.autodl.token]
    profiles = list(config.stores.values())
    if store is not None:
        profiles.append(store.profile)
    for profile in profiles:
        values += [profile.access_key_id, profile.secret_access_key]
    return tuple(sorted({v for v in values if v and len(v) >= 6}, key=len, reverse=True))


def redact(text: str, secrets: tuple[str, ...]) -> str:
    for value in secrets:
        if value in text:
            text = text.replace(value, "***")
    return text


def _offer_data(offer: Offer) -> dict[str, Any]:
    return {
        "offer_id": offer.id,
        "gpu_name": offer.gpu_name,
        "gpu_count": offer.gpu_count,
        "price_per_hour": offer.price_per_hour,
        "currency": offer.currency,
        "region": offer.region,
    }


def _local_data_bytes(spec: JobSpec) -> int:
    total = 0
    for ref in spec.data:
        if ref.local_path is None:
            continue
        path = Path(ref.local_path)
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            for root, _dirs, files in os.walk(path):
                for name in files:
                    full = os.path.join(root, name)
                    if not os.path.islink(full):
                        total += os.path.getsize(full)
    return total


def _oversized_local_files(spec: JobSpec, limit: int) -> list[tuple[str, str, int]]:
    """本地数据里大于 limit 的文件：(数据名, 路径, 大小)，从大到小。

    指向文件的符号链接按目标大小算；不进入指向目录的链接（上传前 packaging 还会再查一遍）。
    """
    found: list[tuple[str, str, int]] = []
    for ref in spec.data:
        if ref.local_path is None:
            continue
        for root, _dirs, files in os.walk(ref.local_path):
            for name in files:
                full = os.path.join(root, name)
                try:
                    size = os.stat(full).st_size
                except OSError:
                    continue  # 坏链接等：上传时 packaging 会报清楚的错
                if size > limit:
                    found.append((ref.name, full, size))
    return sorted(found, key=lambda item: item[2], reverse=True)


def _job_sort_key(job_id: str) -> tuple[str, str]:
    match = _JOB_STAMP_RE.search(job_id)
    return (match.group(1) if match else "", job_id)


def _describe_progress(view: JobView) -> str:
    text = f"任务 {view.job_id}：{view.phase}"
    if view.attempt:
        text += f"（attempt {view.attempt}）"
    if view.step:
        text += f"，步骤 {view.step}"
    if view.gpu_util is not None:
        text += f"，GPU {view.gpu_util:.0f}%"
    if view.reason:
        text += f"，原因 {view.reason}"
    if view.message and view.phase in states.TERMINAL_PHASES:
        text += f"：{view.message}"
    return text
