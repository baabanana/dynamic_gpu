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
import re
import secrets
import signal
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from dgpu import layout, states
from dgpu.contracts import (
    BACKEND_AUTODL,
    BACKEND_VAST,
    CURRENCY_BY_BACKEND,
    INST_EXITED,
    INST_GONE,
    INST_PENDING,
    INST_RUNNING,
    INST_UNKNOWN,
    SINGLE_UPLOAD_LIMIT,
    Backend,
    BackendError,
    Bundler,
    BundleRef,
    Config,
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
    StoredJson,
    StoreError,
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
DEADLINE_MARGIN_S = 1800.0  # 实例最晚自毁的时间之后 gc 再多等 30 分钟，避免和 watchdog 抢
MAX_CONSECUTIVE_RUNNER_TIMEOUTS = 2
PREFLIGHT_STEP = "preflight"  # runner 心跳里 preflight 期间的 step
LAUNCH_URL_EXTRA_S = 3600  # 预签名 URL 有效期 = 启动超时 + runner 超时 + 1 小时
DOCTOR_PREFIX = layout.ROOT + "/doctor/"
GB = 1024**3
# 创建请求的结果未知、又要放弃这次启动时（Ctrl-C、SIGTERM、出错），按标签重复查几次再下结论。
# 和后端自己确认 create 结果的间隔同一量级（Vast 2、10 秒，AutoDL 2、5、10 秒）。
CREATE_UNKNOWN_LOOKUP_DELAYS_S = (0.0, 2.0, 5.0, 10.0)
LATE_INSTANCE_GC_MIN = 10  # 结果未知时，提示用户多少分钟后跑 dgpu gc
# 刚创建的实例在平台上短时间内可能查不到（列表、详情接口有延迟）：这段时间里 GONE 当作还在调度，
# 销毁它时也多观察这么久，确认它没有“晚到”
CREATE_GONE_GRACE_S = 90.0
PRICE_TOLERANCE = 1e-6  # 实例实际价格和上限比较时的相对容差（浮点误差）
# dgpu wait 读存储失败时的退避：5 秒起翻倍，最长 60 秒一次，总共最多 10 分钟
STORE_RETRY_BASE_S = 5.0
STORE_RETRY_MAX_S = 60.0
STORE_OUTAGE_MAX_S = 600.0
MAX_SKIPPED_SHOWN = 20

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
    # 把代码目录打成 tar.gz 写到给定路径；exclude 是不打进包的绝对路径（代码目录里的本地数据）。
    # 返回值有 skipped（(相对路径, 原因)）和 skipped_links（坏掉或指向目录外的符号链接）时会被报告
    pack_code: Callable[..., Any]
    # resolve_data(store, ref, manifest=...)：本地数据先上传到存储（manifest 是 scan_data 的结果）
    resolve_data: Callable[..., DataSource]
    # 本地数据目录 → 文件清单（.total_bytes，.files[i].path / .size），跟随符号链接，和上传用
    # 同一次遍历：大小估算、确认、磁盘估算、AutoDL 50 GB 检查都用它，再原样交给 resolve_data
    scan_data: Callable[[Path], Any]
    clock: Callable[[], float]  # UTC epoch 秒，只用来计时和写时间戳；“多久没更新”用存储的时钟
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


class Terminated(KeyboardInterrupt):
    """进程收到 SIGTERM 或 SIGHUP。

    是 KeyboardInterrupt 的子类，所以走和 Ctrl-C 完全相同的中止路径：销毁本进程创建（或可能
    创建）的实例、写终态、删启动文档。cli.main() 安装信号处理把信号变成这个异常；
    CLI 清理完后以 128 + 信号值退出（SIGTERM 143，SIGHUP 129）。SIGKILL 无法处理，交给 gc。
    """

    def __init__(self, signum: int):
        self.signum = int(signum)
        try:
            self.signame = signal.Signals(self.signum).name
        except ValueError:
            self.signame = f"信号 {self.signum}"
        super().__init__(f"收到 {self.signame}")

    @property
    def exit_code(self) -> int:
        return 128 + self.signum


class StoreUnreachable(DgpuError):
    """dgpu wait 连续 STORE_OUTAGE_MAX_S 读不到存储。任务本身不受影响，结果未知。"""


def interrupt_name(exc: BaseException) -> str:
    """中断的来源，写进状态和输出：Ctrl-C、SIGTERM、SIGHUP。"""
    return exc.signame if isinstance(exc, Terminated) else "Ctrl-C"


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
    heartbeat_at: float | None  # runner 写进心跳的时间（实例的时钟，只用来显示）
    heartbeat_age_s: float | None  # 心跳多久没更新（存储的时钟：Date - LastModified）
    step: str | None
    gpu_util: float | None
    synced_bytes: int | None
    last_sync_at: float | None
    lost: bool
    # 本 attempt 最后一次有动静（心跳；没有心跳时是 status.json / attempt.json 的写入）到现在
    # 多久，用存储的时钟算。终态时为 None。
    silence_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WaitOutcome:
    """dgpu wait 的结果：最终阶段，以及任务结束后清理遗留实例时的问题（实例可能仍在计费）。"""

    phase: str
    cleanup_problems: tuple[str, ...] = ()


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
    # 存储里查不到对应任务的实例：可能属于另一份配置（另一个 bucket）或共用平台账户的同事，
    # 默认只报告；gc --include-unknown 才销毁（这时它们出现在 destroyed 里）
    unknown: list[GcItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "destroyed": [asdict(item) for item in self.destroyed],
            "kept": [asdict(item) for item in self.kept],
            "failed": [asdict(item) for item in self.failed],
            "unknown": [asdict(item) for item in self.unknown],
        }


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    warning: bool = False  # 通过，但有需要用户知道的问题（如余额查不到）


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
    created_at: float | None = None  # ref 创建成功的时间（本机时钟，只用来算刚创建的宽限期）
    seen_alive: bool = False  # 平台的 get 至少有一次查到过 ref（不是 GONE）


@dataclass(frozen=True)
class _Outcome:
    # started | ended | create_failed | startup_failed | runner_timeout | preflight | price
    kind: str
    reason: str | None = None
    instance_may_exist: bool = False
    detail: str = ""
    phase: str | None = None


@dataclass
class _JobDocs:
    exists: bool
    attempt: StoredJson | None
    status: StoredJson | None
    heartbeat: StoredJson | None


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
        self.echo = self._echo

    def _echo(self, message: str) -> None:
        """给人看的输出。输出失败（终端已关闭、SIGHUP 之后的 EIO）不能打断清理：
        需要留档的信息都另外写进了存储的 events/。"""
        try:
            self.deps.echo(redact(message, self._secrets))
        except Exception:  # noqa: BLE001 - 见上
            pass

    # ============================================================ submit

    def submit(
        self,
        spec: JobSpec,
        *,
        confirm_upload: Callable[[float], bool] | None = None,
        accept_skipped_links: bool = False,
    ) -> SubmitResult:
        """提交任务：找机器、上传、创建实例，等到 runner 通过 preflight 后返回。

        confirm_upload：本地数据超过 config.local_data_confirm_gb 时调用，参数是 GB 数，
        返回 False 则放弃。None 表示不需要确认（--yes）。
        accept_skipped_links：代码目录里有坏掉或指向目录外的符号链接时照样提交（--yes）；
        否则报错，什么都不上传。
        """
        backend_name = spec.backend
        backend = self._backend(backend_name)
        self._check_store_profile(backend_name)
        self._warn_foreign_buckets(spec)
        # 本地数据只遍历一次：单文件上限、大小确认、磁盘估算、AutoDL 检查和上传都用这份清单
        listings = self._scan_local_data(spec)
        self._check_local_file_sizes(listings)

        with tempfile.TemporaryDirectory(prefix="dgpu-") as tmp:
            # 0. 先在本地打包代码：有问题（坏链接、超过 200 MB）在花钱和写存储之前就报错
            archive = self._pack_code(spec, Path(tmp), accept_skipped_links)

            # 1. 搜机器。没有就报错，此时没有创建任何东西，也没有写存储。
            request, local_bytes, has_remote = self._build_request(spec, backend_name, listings)
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
            self._upload_job_files(spec, job_id, backend_name, request, archive, listings)
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
        att = self._read_meta(layout.attempt_key(job_id), "attempt.json")
        st = self._read_meta(layout.status_key(job_id), "status.json")
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
        hb = self._read_meta(layout.heartbeat_key(job_id), "heartbeat.json")
        return self._view(job_id, att, st, hb)

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
        listings = self._scan_local_data(spec)
        request, _, has_remote = self._build_request(spec, backend_name, listings)
        found = backend.search(request)
        currency = CURRENCY_BY_BACKEND[backend_name]
        return OffersResult(
            backend=backend_name,
            request=request,
            offers=list(found),
            currency=currency,
            cost_ceiling=request.max_price_per_hour * self._billed_hours(request.max_hours),
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
        """等到终态或确认失联（LOST），返回最终阶段。细节见 wait_for。"""
        return self.wait_for(job_id, retries=retries).phase

    def wait_for(self, job_id: str, retries: int = 0) -> WaitOutcome:
        """等到终态或确认失联。

        心跳过期（按存储的时钟）只是“失联”的迹象：平台上实例仍在运行时，runner 可能在最终
        上传或暂时连不上存储，这时只报告“失联（实例仍在运行）”，什么都不销毁，继续等。
        确认失联（平台显示实例已退出、消失或状态未知，或心跳停的时间超过 runner 最坏收尾时间）
        后：有重试次数、且没有未完成的取消请求时自动 retry，否则返回 LOST。
        读存储暂时失败时退避重试，最多 STORE_OUTAGE_MAX_S，之后抛 StoreUnreachable。
        """
        last_shown: tuple | None = None
        while True:
            view = self._status_patiently(job_id)
            verdict: tuple[bool, str] | None = None
            if view.phase == states.LOST:
                verdict = self._lost_verdict(view)
            if verdict is not None and not verdict[0]:
                shown: tuple = ("alive", view.attempt)
                text = (
                    f"任务 {job_id}：失联（实例仍在运行）。{verdict[1]}。"
                    "不销毁任何东西，继续等。"
                )
            else:
                shown = (view.phase, view.attempt, view.step)
                text = _describe_progress(view)
            if shown != last_shown:
                self.echo(text)
                last_shown = shown
            if view.phase in states.TERMINAL_PHASES:
                return WaitOutcome(view.phase, self._cleanup_after_end(job_id, view.attempt))
            if verdict is not None and verdict[0]:
                if self._cancel_pending(view):
                    self.echo(
                        f"任务 {job_id} 已确认失联（{verdict[1]}），但它有未完成的取消请求"
                        "（dgpu cancel），不会自动重试。runner 多半已经不在了：用 "
                        f"`dgpu cancel --force {job_id}` 销毁实例并把任务标记为 CANCELLED。"
                    )
                    return WaitOutcome(states.LOST)
                if retries <= 0:
                    self.echo(
                        f"任务 {job_id} 已确认失联（LOST：{verdict[1]}）。可以用 "
                        f"`dgpu retry {job_id}` 换机续跑；遗留实例会由 `dgpu gc` 清理。"
                    )
                    return WaitOutcome(states.LOST)
                retries -= 1
                self.echo(f"任务已确认失联（{verdict[1]}），自动重试（还剩 {retries} 次）")
                try:
                    self.retry(job_id)
                except DgpuError as exc:
                    self.echo(f"自动重试失败：{exc}")
                    view = self._status_patiently(job_id)
                    if view.phase in states.TERMINAL_PHASES:
                        problems = self._cleanup_after_end(job_id, view.attempt)
                        return WaitOutcome(view.phase, problems)
                    return WaitOutcome(states.LOST)
                last_shown = None
                continue
            self.sleep(POLL_WAIT_S)

    def _status_patiently(self, job_id: str) -> JobView:
        """status()，但存储暂时不可用（网络、5xx、限流）时退避重试，总共最多 STORE_OUTAGE_MAX_S。"""
        started: float | None = None
        delay = STORE_RETRY_BASE_S
        while True:
            try:
                return self.status(job_id)
            except StoreError as exc:
                if isinstance(exc, PreconditionFailed) or not exc.transient:
                    raise
                now = self.clock()
                if started is None:
                    started = now
                waited = now - started
                if waited >= STORE_OUTAGE_MAX_S:
                    raise StoreUnreachable(
                        f"连续 {waited / 60:.0f} 分钟读不到存储 {self.store.profile.name}"
                        f"（bucket {self.store.profile.bucket}）：{exc}。"
                        f"任务 {job_id} 本身不受影响（机器上的 runner 独立运行、自己销毁实例），"
                        f"但现在不知道它的结果。网络恢复后再运行 `dgpu wait {job_id}`。"
                    ) from exc
                self.echo(
                    f"读取任务状态失败（{exc}）。{delay:.0f} 秒后重试；"
                    f"最多再等 {(STORE_OUTAGE_MAX_S - waited) / 60:.0f} 分钟。"
                )
                self.sleep(delay)
                delay = min(delay * 2, STORE_RETRY_MAX_S)

    def _lost_verdict(self, view: JobView) -> tuple[bool, str]:
        """LOST 的任务是否确认失联：(True, 原因) 可以重试；(False, 说明) 实例可能还在干活。

        - 本 attempt 从没有心跳（启动阶段的 LOST）：runner 没起来，确认失联。
        - 心跳停的时间超过 runner 最坏收尾时间：runner 早该自己结束了，确认失联。
        - 否则看平台：实例已退出、消失或状态未知才算确认失联。
        """
        silence = view.silence_s or 0.0
        quiet = f"心跳已停 {silence / 60:.0f} 分钟"
        runner_never_started = view.heartbeat_age_s is None and (
            view.stored_phase in (None, *_CLI_PHASES)
            or (view.status_attempt or 0) < (view.attempt or 0)
        )
        if runner_never_started:
            return True, f"启动阶段 {silence / 60:.0f} 分钟没有进展，runner 没有发出心跳"
        teardown = self._teardown_worst_case_s()
        if silence > teardown:
            return True, f"{quiet}，超过 runner 最坏收尾时间 {teardown / 60:.0f} 分钟"
        phase, detail = self._probe_instance(view)
        if phase in _DEAD_INSTANCE_PHASES:
            return True, f"{quiet}，平台显示{detail}"
        if phase is None:
            return False, f"{quiet}，查不到平台上的实例状态（{detail}）"
        return False, (
            f"{quiet}，但平台显示{detail}：runner 可能在最终上传，或暂时连不上存储；"
            f"心跳停满 {teardown / 60:.0f} 分钟或实例退出后才算失联"
        )

    def _probe_instance(self, view: JobView) -> tuple[str | None, str]:
        """平台上当前 attempt 的实例状态：(INST_*, 说明)；查不了时 (None, 原因)。"""
        backend = self.backends.get(view.backend or "")
        if backend is None:
            return None, f"没有 {view.backend} 平台的凭据"
        label = view.label or (
            layout.label(view.job_id, view.attempt) if view.attempt else None
        )
        try:
            if view.instance_id:
                ref = InstanceRef(backend=backend.name, id=str(view.instance_id), label=label or "")
                state = backend.get(ref)
            elif label:
                found = [s for s in backend.list(label) if s.ref.label == label]
                if not found:
                    return INST_GONE, f"没有标签为 {label} 的实例"
                state = found[0]
            else:
                return None, "attempt.json 里没有实例 ID 和标签"
        except Exception as exc:  # noqa: BLE001 - 平台查询失败时保守地当作“不知道”
            return None, f"查询失败：{exc}"
        return state.phase, f"实例 {state.ref.id} 是 {state.phase}（{state.raw_status or '-'}）"

    def _cancel_pending(self, view: JobView) -> bool:
        """有取消请求，而当前 attempt 还没有终态：取消还没完成，不能自动或手动重试。"""
        return view.cancel_requested and not _terminal_for_current(view)

    def retry(self, job_id: str) -> SubmitResult:
        """换机重跑：先确认旧实例已销毁，再用条件写把 attempt + 1，然后走换机循环。

        有未完成的取消请求（取消标志已设置、当前 attempt 还没有终态）时拒绝：取消优先。
        任务已经结束（包括 CANCELLED）时，显式的 retry 会清除取消标志。
        """
        view = self.status(job_id)
        if view.phase not in RETRYABLE_PHASES:
            raise DgpuError(
                f"任务 {job_id} 当前是 {view.phase}，只有 LOST、FAILED、TIMED_OUT、CANCELLED "
                "的任务可以重试。"
            )
        if self._cancel_pending(view):
            raise DgpuError(
                f"任务 {job_id} 有未完成的取消请求（dgpu cancel 设置了取消标志，任务还没结束），"
                "不重试。确实要放弃这次运行重新开始：先 "
                f"`dgpu cancel --force {job_id}`（销毁实例并标记 CANCELLED），"
                f"再 `dgpu retry {job_id}`。"
            )
        cancel_seen = view.cancel_requested
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
        # 旧实例必须确认销毁；失败就停在这里，不启动第二台。只销毁不新于 old_attempt 的实例：
        # 并发的另一个 retry 可能已经推进到下一个 attempt 并创建了实例，那不是这里该销毁的。
        self._destroy_job_instances(
            job_id,
            backend,
            old_doc,
            why=f"重试前确认 attempt {old_attempt} 的实例已销毁",
            max_attempt=old_attempt,
        )
        bundle = self.bundler.ensure_uploaded(self.store)

        def bump(doc: dict[str, Any]) -> dict[str, Any]:
            if doc.get("cancel") and not cancel_seen:
                # 检查之后才有人 dgpu cancel：取消优先，这次不重试
                raise _CancelRequested(f"任务 {job_id} 刚被取消（dgpu cancel），这次没有重试。")
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
        attempt = int(att["attempt"])

        def destroy() -> None:
            self._destroy_job_instances(
                job_id, backend, att, why="dgpu cancel --force", max_attempt=attempt
            )

        try:
            destroy()
        except KeyboardInterrupt as exc:
            # 用户要的就是销毁：Ctrl-C / SIGTERM / SIGHUP 时先把销毁做完再退出
            # （同类信号不会再打断；再按一次 Ctrl-C 可以放弃）
            self.echo(f"收到中断（{interrupt_name(exc)}）：先把实例销毁做完再退出……")
            destroy()
            self._finish_force_cancel(job_id, attempt)
            raise
        self._finish_force_cancel(job_id, attempt)
        return self.status(job_id)

    def _finish_force_cancel(self, job_id: str, attempt: int) -> None:
        self._write_terminal_if_absent(
            job_id,
            attempt,
            states.CANCELLED,
            None,
            "dgpu cancel --force：实例已直接销毁，没有做最终同步",
        )
        self.store.delete(layout.secrets_key(job_id, attempt))

    # ============================================================ gc / doctor

    def gc(
        self,
        dry_run: bool = False,
        *,
        include_unknown: bool = False,
        confirm_unknown: Callable[[list[GcItem]], bool] | None = None,
    ) -> GcReport:
        """清理遗留实例。

        只处理标签能被 layout.parse_label 解析、且任务 ID 是 layout.new_job_id 生成的形状的实例。
        存储里查不到任务的实例（可能属于另一份配置或共用平台账户的同事）默认只报告，放在
        report.unknown；include_unknown 时才销毁，销毁前调用 confirm_unknown（None 表示
        不再确认，即 --yes）。
        """
        report = GcReport(dry_run=dry_run)
        now = self.clock()
        cache: dict[str, _JobDocs] = {}
        unknown: list[tuple[Backend, InstanceState, GcItem]] = []
        for name, backend in self.backends.items():
            try:
                instances = backend.list(layout.LABEL_PREFIX)
            except Exception as exc:  # 一个平台失败不影响另一个
                report.failed.append(GcItem(name, "", "", None, None, f"列举实例失败：{exc}"))
                continue
            for inst in instances:
                label = inst.ref.label
                parsed = layout.parse_label(label)
                if parsed is None or not layout.is_generated_job_id(parsed[0]):
                    report.kept.append(
                        GcItem(
                            name,
                            inst.ref.id,
                            label,
                            None,
                            None,
                            "标签不是 dgpu 生成的格式（dgpu-<名字>-<时间>-<随机>-a<次数>），不处理",
                        )
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
                if not docs.exists:
                    unknown.append((backend, inst, item))
                    continue
                if not destroy:
                    report.kept.append(item)
                    continue
                self._gc_destroy(report, backend, inst, item, job_known=True)
        if unknown:
            items = [item for _, _, item in unknown]
            if not include_unknown:
                report.unknown.extend(items)
            elif dry_run or confirm_unknown is None or confirm_unknown(items):
                for backend, inst, item in unknown:
                    self._gc_destroy(report, backend, inst, item, job_known=False)
            else:
                report.unknown.extend(items)
        return report

    def _gc_destroy(
        self,
        report: GcReport,
        backend: Backend,
        inst: InstanceState,
        item: GcItem,
        *,
        job_known: bool,
    ) -> None:
        if report.dry_run:
            report.destroyed.append(item)
            return
        name, label, job_id, attempt = item.backend, item.label, item.job_id, item.attempt
        try:
            backend.destroy(inst.ref)
            self._confirm_gone(backend, inst.ref)  # 平台受理了不等于已经停止计费
        except Exception as exc:
            message = (
                f"!! gc 销毁实例 {name}/{inst.ref.id}（标签 {label}）失败：{exc}。"
                "它可能仍在计费，请稍后再跑 `dgpu gc` 或到平台控制台手动销毁。"
            )
            self.echo(message)
            report.failed.append(replace(item, reason=message))
            if job_known and job_id:
                self._event(
                    job_id,
                    states.EV_DESTROY_FAILED,
                    attempt,
                    message,
                    {"backend": name, "instance_id": inst.ref.id, "label": label},
                    source=states.SRC_GC,
                )
            return
        report.destroyed.append(item)
        if job_known and job_id:  # 存储里没有这个任务时不凭空建目录
            self._event(
                job_id,
                states.EV_GC_DESTROYED,
                attempt,
                f"gc 销毁实例 {inst.ref.id}：{item.reason}",
                {"backend": name, "instance_id": inst.ref.id, "label": label},
                source=states.SRC_GC,
            )

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
            # 不是平台明确的拒绝：请求可能已经执行，实例可能稍后才出现在列表里。下一次创建
            # 换新的 attempt 和标签，晚到的实例就是旧 attempt：它的 runner 被隔离，gc 会销毁它。
            return _Outcome("create_failed", states.REASON_NO_OFFER, True, detail)
        ctx.ref = ref
        ctx.created_at = self.clock()
        ctx.seen_alive = False
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
        created = self.clock()
        startup_deadline = created + cfg.startup_timeout_min * 60
        running_at: float | None = None
        preflight_since: float | None = None  # 第一次看到本 attempt 心跳的时间
        last_phase = None
        price_checked = False
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
            if state.phase != INST_GONE:
                ctx.seen_alive = True
            elif now - created < CREATE_GONE_GRACE_S:
                # 刚创建的实例，平台的详情 / 列表接口可能还查不到：按“等待调度”处理
                state = replace(
                    state,
                    phase=INST_PENDING,
                    raw_status=f"刚创建，平台暂时查不到（{state.raw_status or '-'}）",
                )
            if state.phase != INST_GONE:
                self._check_platform_label(ctx, ref, state)
                if not price_checked and state.price_per_hour is not None:
                    price_checked = True
                    if state.price_per_hour > ctx.request.max_price_per_hour * (
                        1 + PRICE_TOLERANCE
                    ):
                        return self._price_rejected(ctx, ref, state.price_per_hour)
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

    def _check_platform_label(self, ctx: _Launch, ref: InstanceRef, state: InstanceState) -> None:
        """平台回报的标签必须和创建时的一字不差：gc 和按标签查找都靠它识别 dgpu 的实例。"""
        actual = state.ref.label
        if not actual or actual == ref.label:
            return
        message = (
            f"!! 平台把实例 {ref.backend}/{ref.id} 的标签改成了 {actual!r}"
            f"（创建时是 {ref.label!r}，"
            "可能是长度或字符被截断、改写）。dgpu gc 和按标签查找都靠这个标签识别 dgpu 的实例，"
            "标签不对就可能漏掉遗留实例，所以停止提交，并按实例 ID 销毁它。请把这个问题报告给 "
            "dgpu 的维护者（附上任务名），并用 `dgpu gc --dry-run` 和平台控制台确认没有遗留。"
        )
        self.echo(message)
        self._event(
            ctx.job_id,
            states.EV_CREATE_FAILED,
            ctx.attempt,
            message,
            {"instance_id": ref.id, "expected_label": ref.label, "actual_label": actual},
        )
        raise DgpuError(message)  # _abort_launch 按 ctx.ref（实例 ID）销毁

    def _price_rejected(self, ctx: _Launch, ref: InstanceRef, price: float) -> _Outcome:
        """实例的实际价格高于上限（搜索到创建之间涨价，或平台的价格口径和假设不同）。"""
        ceiling = ctx.request.max_price_per_hour
        detail = (
            f"实例 {ref.id} 的实际价格 {price:.4f} {ctx.currency}/h 高于上限 "
            f"{ceiling:g} {ctx.currency}/h（搜索时 offer {ctx.offer.id if ctx.offer else '?'} "
            f"报价 {ctx.offer.price_per_hour if ctx.offer else '?'}）"
        )
        self.echo(detail + "，销毁并换下一台")
        self._event(
            ctx.job_id,
            states.EV_OFFER_REJECTED,
            ctx.attempt,
            detail,
            {
                "instance_id": ref.id,
                "price_per_hour": price,
                "max_price_per_hour": ceiling,
                **(_offer_data(ctx.offer) if ctx.offer else {}),
            },
        )
        self._destroy_ctx(ctx, ref, why="实际价格高于上限")
        return _Outcome("price", states.REASON_NO_OFFER, True, detail)

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
        """异常路径：销毁（可能）创建的实例，写终态，删启动文档。返回需要大声报告的问题。

        Ctrl-C、SIGTERM、SIGHUP（Terminated）都是 KeyboardInterrupt，走同一条路径。
        create 的结果未知时，按标签重复查几次；仍然查不到就如实说“结果未知”，绝不说
        “已销毁”，并把这个 attempt 作废：晚到的实例的 runner 会被隔离，gc 把它当作旧 attempt
        销毁。
        """
        problems: list[str] = []
        interrupted = isinstance(exc, KeyboardInterrupt)
        how = interrupt_name(exc) if interrupted else ""
        if interrupted:
            self.echo(f"收到中断（{how}），正在清理：销毁已创建的实例……")
        launch_attempt = ctx.attempt  # 启动文档属于这个 attempt

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
        destroyed: list[str] = []
        unknown_label: str | None = None
        why = f"启动被中断（{how}）" if interrupted else "启动异常退出"
        try:
            refs: list[InstanceRef] = []
            if ctx.ref is not None:
                refs.append(ctx.ref)
            elif ctx.create_unknown and ctx.label:
                unknown_label = ctx.label  # 查到之前都算结果未知
                refs.extend(self._lookup_label_patiently(ctx.backend, ctx.label))
                if refs:
                    unknown_label = None
            for ref in refs:
                try:
                    self._destroy_ctx(ctx, ref, why=why)
                    destroyed.append(ref.id)
                except DestroyFailed as destroy_exc:
                    problems.append(str(destroy_exc))
        except BaseException as cleanup_exc:
            report("销毁实例", cleanup_exc)
        fenced_by_other = isinstance(exc, AttemptConflict | _CancelRequested)
        if unknown_label is not None:
            message = (
                f"!! 创建实例的请求结果未知：按标签 {unknown_label} 查了 "
                f"{len(CREATE_UNKNOWN_LOOKUP_DELAYS_S)} 次都没找到实例，"
                "但它可能稍后才出现并开始计费。"
                f"请在 {LATE_INSTANCE_GC_MIN} 分钟后运行 `dgpu gc`（它会销毁这个任务留下的实例），"
                "并到平台控制台确认。"
            )
            self.echo(message)
            problems.append(message)
            if not fenced_by_other:
                try:
                    self._fence_unknown_attempt(ctx, unknown_label)
                except BaseException as cleanup_exc:
                    report("作废 attempt", cleanup_exc)
        # 2. 写终态（runner 已写或已被别的进程接管时不写）
        if not fenced_by_other:
            if interrupted:
                phase, reason = states.CANCELLED, None
                message = f"启动被中断（{how}）"
            else:
                phase, reason = states.FAILED, states.REASON_STARTUP
                message = f"启动过程中出错：{exc}"
            if destroyed:
                message += f"；已销毁实例 {'、'.join(destroyed)}"
            if unknown_label is not None:
                message += (
                    f"；创建实例的结果未知，标签为 {unknown_label} 的实例可能稍后出现并计费，"
                    f"请在 {LATE_INSTANCE_GC_MIN} 分钟后运行 `dgpu gc`（它会销毁这个实例）"
                )
            elif problems:
                message += "；有实例可能没有销毁，见 `dgpu gc`"
            try:
                self._write_terminal_if_absent(ctx.job_id, ctx.attempt, phase, reason, message)
            except BaseException as cleanup_exc:
                report("写终态", cleanup_exc)
        # 3. 删掉含存储凭据的启动文档
        try:
            self.store.delete(layout.secrets_key(ctx.job_id, launch_attempt))
        except BaseException as cleanup_exc:
            report("删除启动文档", cleanup_exc)
        return problems

    def _lookup_label_patiently(self, backend: Backend, label: str) -> list[InstanceRef]:
        """按标签查几次（刚创建的实例可能稍后才出现在列表里）。查询本身出错也继续查。"""
        errors: list[str] = []
        for delay in CREATE_UNKNOWN_LOOKUP_DELAYS_S:
            if delay:
                self.sleep(delay)
            try:
                found = self._lookup_label(backend, label)
            except Exception as exc:  # noqa: BLE001 - 平台暂时不可用：记下来，下一次再查
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if found:
                return found
        if errors:
            self.echo(f"按标签 {label} 查询实例时出错 {len(errors)} 次，最后一次：{errors[-1]}")
        return []

    def _fence_unknown_attempt(self, ctx: _Launch, label: str) -> None:
        """create 结果未知、要放弃这次启动：把 attempt + 1，作废可能晚到的实例。

        晚到实例的 runner 启动时读到更大的 attempt 就自行退出（不写状态），gc 把它当作旧
        attempt 销毁。终态随后写给新的 attempt。
        """
        old = ctx.attempt

        def bump(doc: dict[str, Any]) -> dict[str, Any]:
            new_attempt = int(doc["attempt"]) + 1
            doc.update(
                attempt=new_attempt,
                label=layout.label(ctx.job_id, new_attempt),
                instance_id=None,
                created_at=self.clock(),
            )
            return doc

        try:
            doc = self._update_attempt(ctx.job_id, bump, expect_attempt=old)
        except AttemptConflict:
            return  # 别的进程已经推进了 attempt：旧 attempt 同样被作废
        ctx.attempt = int(doc["attempt"])
        self._event(
            ctx.job_id,
            states.EV_RELAUNCH,
            ctx.attempt,
            f"attempt {old} 的创建结果未知（标签 {label}），作废它，改记为 attempt {ctx.attempt}；"
            "晚到的实例会被隔离，由 gc 销毁",
            {
                "old_attempt": old,
                "new_attempt": ctx.attempt,
                "why": "create_unknown",
                "label": label,
            },
        )

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
        already_gone: bool = False,
        settle_until: float | None = None,
    ) -> None:
        """销毁并确认实例消失。失败时大声输出并抛 DestroyFailed。

        already_gone：平台已经查不到它。照样发一次销毁（刚创建的实例可能只是还没出现在
        查询结果里），但不算作“销毁了一台实例”，不写 instance_destroyed。
        settle_until：刚创建的实例，查不到也要观察到这个时间，确认它没有晚到。
        """
        if not already_gone:
            self.echo(f"正在销毁实例 {ref.backend}/{ref.id}（{why}）……")
        try:
            backend.destroy(ref)
            self._confirm_gone(backend, ref, settle_until=settle_until)
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
        if already_gone:
            return
        self._event(
            job_id,
            states.EV_INSTANCE_DESTROYED,
            attempt,
            f"已销毁实例 {ref.id}：{why}",
            {"backend": ref.backend, "instance_id": ref.id, "label": ref.label, "why": why},
        )

    def _destroy_ctx(self, ctx: _Launch, ref: InstanceRef, *, why: str) -> None:
        settle = None
        mine = ctx.ref is not None and ctx.ref.id == ref.id
        if mine and not ctx.seen_alive and ctx.created_at is not None:
            # 平台还从没查到过这台刚创建的实例：查不到不等于没有，观察到宽限期结束
            settle = ctx.created_at + CREATE_GONE_GRACE_S
        self._destroy(
            ctx.backend, ref, job_id=ctx.job_id, attempt=ctx.attempt, why=why, settle_until=settle
        )
        if ctx.ref is not None and ctx.ref.id == ref.id:
            ctx.ref = None

    def _confirm_gone(
        self, backend: Backend, ref: InstanceRef, *, settle_until: float | None = None
    ) -> None:
        """销毁后确认实例消失：详情（get）和列表（list）都查不到才算。

        刚创建的实例（settle_until 之前）查不到也继续观察，期间照常重发销毁。
        """
        start = self.clock()
        last_issue = start
        while True:
            state = backend.get(ref)
            if state.phase == INST_GONE:
                listed = self._listed_alive(backend, ref)
                if listed is None:
                    now = self.clock()
                    if settle_until is None or now >= settle_until:
                        return
                    state = replace(state, raw_status="刚创建，观察它是否晚到")
                else:
                    state = listed
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

    def _listed_alive(self, backend: Backend, ref: InstanceRef) -> InstanceState | None:
        """按标签列出实例时，这个 ID 还在（且不是 GONE）就返回它的状态。"""
        if not ref.label.startswith(layout.LABEL_PREFIX):
            return None  # list 只接受 dgpu- 前缀；非 dgpu 标签的实例 dgpu 本来就不碰
        for state in backend.list(ref.label):
            if state.ref.id == ref.id and state.phase != INST_GONE:
                return state
        return None

    def _lookup_label(self, backend: Backend, label: str) -> list[InstanceRef]:
        """按精确标签查实例（list 按前缀匹配，a1 也是 a10 的前缀，所以再精确比较）。"""
        return [
            state.ref
            for state in backend.list(label)
            if state.ref.label == label and state.phase != INST_GONE
        ]

    def _destroy_job_instances(
        self,
        job_id: str,
        backend: Backend,
        att_doc: dict[str, Any],
        *,
        why: str,
        max_attempt: int,
    ) -> None:
        """销毁这个任务在该平台上 attempt ≤ max_attempt 的实例。

        包括 attempt.json 记录的和按标签查到的。

        max_attempt 是这次销毁所依据的判断（“attempt N 已结束”“attempt N 要被替换”）里的 N。
        更新的 attempt 的实例属于并发的另一个 dgpu 进程（比如刚完成的 retry），不归这里销毁。
        失败即抛。
        """
        refs: dict[str, InstanceRef] = {}
        attempt = int(att_doc.get("attempt") or 1)
        if att_doc.get("instance_id") and attempt <= max_attempt:
            label = att_doc.get("label") or layout.label(job_id, attempt)
            ref = InstanceRef(backend=backend.name, id=str(att_doc["instance_id"]), label=label)
            refs[ref.id] = ref
        for state in backend.list(f"{layout.LABEL_PREFIX}{job_id}-a"):
            parsed = layout.parse_label(state.ref.label)
            if parsed is None or parsed[0] != job_id or state.phase == INST_GONE:
                continue
            if parsed[1] > max_attempt:
                self.echo(
                    f"跳过实例 {state.ref.id}（标签 {state.ref.label}）：它属于更新的 attempt "
                    f"{parsed[1]}，这次只处理 attempt ≤ {max_attempt} 的实例。"
                )
                continue
            refs.setdefault(state.ref.id, state.ref)
        for ref in refs.values():
            # 平台说已经没有了也照样发一次销毁（刚创建的实例可能只是还没出现在查询结果里）
            gone = backend.get(ref).phase == INST_GONE
            parsed = layout.parse_label(ref.label)
            self._destroy(
                backend,
                ref,
                job_id=job_id,
                attempt=parsed[1] if parsed else attempt,
                why=why,
                already_gone=gone,
            )

    def _cleanup_after_end(self, job_id: str, seen_attempt: int | None) -> tuple[str, ...]:
        """任务到终态后确认实例已销毁（AutoDL 需要 CLI 删除部署记录），删掉残留的启动文档。

        seen_attempt 是看到终态的那个 attempt：attempt.json 已经前进（并发的 retry 启动了新
        attempt）时什么都不做，新 attempt 的实例不归这里管。返回需要大声报告的问题（实例
        可能仍在计费）；问题已经输出，并写了 destroy_failed 事件。
        """
        problems: list[str] = []

        def problem(message: str, data: dict[str, Any]) -> None:
            self.echo(message)
            problems.append(message)
            self._event(job_id, states.EV_DESTROY_FAILED, seen_attempt, message, data)

        try:
            got = self._read_doc(layout.attempt_key(job_id), "attempt.json")
        except DgpuError as exc:
            problem(
                f"!! 任务 {job_id} 已结束，但读不到 attempt.json，无法确认实例已销毁：{exc}。"
                "实例可能仍在计费：请稍后运行 `dgpu gc`。",
                {"why": "cleanup_read_attempt"},
            )
            return tuple(problems)
        if got is None:
            return ()
        att = got[0]
        current = int(att.get("attempt") or 0)
        if seen_attempt is not None and current > seen_attempt:
            self.echo(
                f"任务 {job_id} 已经被重试到 attempt {current}，attempt {seen_attempt} 的实例"
                "交给那次重试（它先确认旧实例已销毁）和 `dgpu gc` 处理。"
            )
            return ()
        upto = seen_attempt if seen_attempt is not None else current
        label = att.get("label") or layout.label(job_id, upto or 1)
        try:
            self.store.delete(layout.secrets_key(job_id, upto))
        except Exception as exc:  # noqa: BLE001 - 大声报告，继续清理实例
            message = (
                f"!! 删除任务 {job_id} attempt {upto} 的启动文档（含存储凭据）失败：{exc}。"
                "请稍后再运行 `dgpu wait` 或手动删除。"
            )
            self.echo(message)
            problems.append(message)
        backend = self.backends.get(att.get("backend"))
        if backend is None:
            problem(
                f"!! 没有 {att.get('backend')} 平台的凭据，无法确认任务 {job_id} 的实例"
                f"（标签 {label}）已销毁，它可能仍在计费：请在有凭据的电脑上运行 `dgpu gc`。",
                {"backend": att.get("backend"), "label": label, "why": "no_credentials"},
            )
            return tuple(problems)
        try:
            self._destroy_job_instances(
                job_id, backend, att, why="任务已结束，清理实例", max_attempt=upto
            )
        except DestroyFailed as exc:
            problems.append(str(exc))  # _destroy 已经大声输出并写了 destroy_failed
        except Exception as exc:  # noqa: BLE001 - 平台或存储出错：大声报告，任务结果照常返回
            problem(
                f"!! 任务 {job_id} 已结束，但清理实例时出错：{exc}。标签为 {label} 的实例"
                "（实例 ID " + str(att.get("instance_id") or "未知") + "）可能仍在计费："
                "请稍后运行 `dgpu gc`，或到平台控制台检查。",
                {
                    "backend": backend.name,
                    "instance_id": att.get("instance_id"),
                    "label": label,
                    "why": "cleanup_error",
                },
            )
        return tuple(problems)

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

    def _scan_local_data(self, spec: JobSpec) -> dict[str, Any]:
        """data 里每个本地目录的文件清单（数据名 → scan_data 的结果）。只遍历这一次。"""
        listings: dict[str, Any] = {}
        for ref in spec.data:
            if ref.local_path is None:
                continue
            listings[ref.name] = self.deps.scan_data(Path(ref.local_path))
        return listings

    def _check_local_file_sizes(self, listings: dict[str, Any]) -> None:
        """本地数据里有超过单次上传上限的文件：上传任何东西之前就报错。"""
        too_big = sorted(
            (
                (name, str(Path(listing.root) / item.path), item.size)
                for name, listing in listings.items()
                for item in listing.files
                if item.size > SINGLE_UPLOAD_LIMIT
            ),
            key=lambda entry: entry[2],
            reverse=True,
        )
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
        self._check_version(key, doc, what)
        return doc, etag

    def _read_meta(self, key: str, what: str) -> StoredJson | None:
        """读 JSON 文档和它的存储时间（LastModified、存储的当前时间），用来算多久没更新。"""
        got = self.store.get_json_meta(key)
        if got is None:
            return None
        self._check_version(key, got.doc, what)
        return got

    @staticmethod
    def _check_version(key: str, doc: dict[str, Any], what: str) -> None:
        try:
            states.check_version(doc, what)
        except ValueError as exc:
            raise DgpuError(f"{key}：{exc}。可能是更新版本的 dgpu 写的，请升级 dgpu。") from None

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

    def _build_request(
        self, spec: JobSpec, backend_name: str, listings: dict[str, Any]
    ) -> tuple[GpuRequest, int, bool]:
        max_price = spec.limits.max_price(backend_name)
        image = spec.image_for(backend_name)
        # 和上传同一份清单（跟随符号链接），估算的就是实际要传、要拷到实例上的量
        local_bytes = sum(int(listing.total_bytes) for listing in listings.values())
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

    def _runner_settings(self) -> dict[str, float]:
        return runner_util.runner_settings({"runner": self.config.runner.to_dict()})

    def _teardown_worst_case_s(self) -> float:
        """runner 从决定停止到自毁结束最多用多久（和 runner、watchdog 同一个公式）。"""
        settings = self._runner_settings()
        return runner_util.teardown_worst_case_s(settings, runner_util.DEFAULT_TUNING)

    def _startup_window_s(self) -> float:
        return (self.config.startup_timeout_min + self.config.runner_timeout_min) * 60.0

    def _billed_hours(self, max_hours: float) -> float:
        """一台机器最坏计费多久：启动窗口 + 实例存活时限（max_hours + 收尾 + watchdog 余量）。"""
        lifetime = runner_util.instance_lifetime_s(
            max_hours, self._runner_settings(), runner_util.DEFAULT_TUNING
        )
        return (self._startup_window_s() + lifetime) / 3600.0

    def _print_ceiling(self, req: GpuRequest, currency: str) -> float:
        price = req.max_price_per_hour
        hours = self._billed_hours(req.max_hours)
        startup_h = self._startup_window_s() / 3600.0
        teardown_h = hours - startup_h - req.max_hours
        ceiling = price * hours
        self.echo(
            f"费用上限：{price:g} {currency}/h × {hours:.2f} h = {ceiling:.2f} {currency}"
            f"（不含流量费；{hours:.2f} h = 启动最多 {startup_h:.2f} h + max_hours "
            f"{req.max_hours:g} h + 收尾最多 {teardown_h:.2f} h，到时间实例一定被销毁）"
        )
        self.echo(
            f"另外：每台启动失败而被换掉的机器最多再花 {price * startup_h:.2f} {currency}"
            f"（最多换 {self.config.max_offers} 台）；每次 retry（包括 wait --retries）"
            "都按上面的上限再算一次。"
        )
        return ceiling

    def _check_balance(self, backend: Backend, ceiling: float, currency: str) -> None:
        balance = backend.balance()
        if balance is None:
            note = getattr(backend, "balance_note", "") or "平台没有提供余额"
            self.echo(
                f"注意：未检查余额（{note}）。请自己确认账户余额够覆盖费用上限 "
                f"{ceiling:.2f} {currency}。"
            )
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

    def _pack_code(self, spec: JobSpec, tmp: Path, accept_skipped_links: bool) -> Path:
        """把代码目录打包到 tmp/code.tar.gz。跳过的条目逐个列出；坏链接默认拒绝。"""
        self.echo(f"打包代码 {spec.code} ……")
        # data 里的本地目录单独上传；它们在代码目录里时不能再打进代码包
        local_data = [Path(ref.local_path) for ref in spec.data if ref.local_path is not None]
        archive = tmp / "code.tar.gz"
        result = self.deps.pack_code(Path(spec.code), archive, exclude=local_data)
        skipped = tuple(getattr(result, "skipped", ()) or ())
        bad_links = tuple(getattr(result, "skipped_links", ()) or ())
        if skipped:
            self.echo(f"注意：代码包跳过了 {len(skipped)} 个条目，它们不会出现在实例上：")
            for rel, reason in skipped[:MAX_SKIPPED_SHOWN]:
                self.echo(f"  {rel}：{reason}")
            if len(skipped) > MAX_SKIPPED_SHOWN:
                self.echo(f"  ……另有 {len(skipped) - MAX_SKIPPED_SHOWN} 个")
        if bad_links and not accept_skipped_links:
            shown = "、".join(bad_links[:5]) + ("等" if len(bad_links) > 5 else "")
            raise DgpuError(
                f"代码目录里有 {len(bad_links)} 个坏掉或指向代码目录之外的符号链接（{shown}），"
                "它们没有打进代码包，训练代码用到它们时会在实例上失败。修好或删掉这些链接；"
                "确认不需要它们时加 --yes 提交。没有上传任何东西，也没有创建实例。"
            )
        return archive

    def _upload_job_files(
        self,
        spec: JobSpec,
        job_id: str,
        backend_name: str,
        request: GpuRequest,
        archive: Path,
        listings: dict[str, Any],
    ) -> None:
        self.store.put_file(layout.code_key(job_id), archive)
        sources: list[DataSource] = []
        for ref in spec.data:
            if ref.is_local:
                self.echo(f"准备数据 {ref.name}（本地 {ref.local_path}）……")
                source = self.deps.resolve_data(self.store, ref, manifest=listings.get(ref.name))
            else:
                source = self.deps.resolve_data(self.store, ref)
            sources.append(source)
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
        att_meta: StoredJson | None,
        st_meta: StoredJson | None,
        hb_meta: StoredJson | None,
    ) -> JobView:
        """合成任务视图。“多久没动静”一律用存储的时钟算（对象的 LastModified 和读取时
        存储的当前时间），不比较本机和租来的实例的时钟。"""
        cfg = self.config
        att = att_meta.doc if att_meta else None
        st = st_meta.doc if st_meta else None
        hb = hb_meta.doc if hb_meta else None
        current = (att or {}).get("attempt") or (st or {}).get("attempt")
        status_attempt = (st or {}).get("attempt")
        if st is None or (status_attempt or 0) < (current or 0):
            base_phase = states.PROVISIONING  # 已换到新的 attempt，状态还没写
            base_meta = att_meta  # 换 attempt 的时间就是 attempt.json 的写入时间
        else:
            base_phase = st.get("phase") or states.PROVISIONING
            base_meta = st_meta
        hb_current = hb if hb is not None and hb.get("attempt") == current else None
        if hb_current is not None and base_phase in _CLI_PHASES:
            base_phase = states.RUNNING  # runner 已发心跳，状态写入稍晚
        lost = False
        hb_age = hb_meta.age_s if hb_current is not None and hb_meta is not None else None
        silence: float | None = None
        if base_phase in (states.RUNNING, states.UPLOADING):
            silence = hb_age if hb_age is not None else (base_meta.age_s if base_meta else None)
            lost = silence is not None and silence > cfg.lost_after_min * 60
        elif base_phase in _CLI_PHASES:
            silence = base_meta.age_s if base_meta else None
            window = self._startup_window_s() + cfg.lost_after_min * 60
            lost = silence is not None and silence > window
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
            silence_s=silence,
        )

    def _load_job_docs(self, job_id: str) -> _JobDocs:
        att = self._read_meta(layout.attempt_key(job_id), "attempt.json")
        st = self._read_meta(layout.status_key(job_id), "status.json")
        hb = self._read_meta(layout.heartbeat_key(job_id), "heartbeat.json")
        exists = att is not None or st is not None or self.store.exists(layout.spec_key(job_id))
        return _JobDocs(exists=exists, attempt=att, status=st, heartbeat=hb)

    def _gc_decide(
        self, docs: _JobDocs, attempt: int, inst: InstanceState, now: float
    ) -> tuple[bool, str]:
        """gc 是否销毁这台实例：(是否销毁, 原因)。

        心跳过期用存储的时钟算。心跳过期只说明可能失联：平台显示实例已退出或状态未知，或者
        心跳停的时间超过 runner 最坏收尾时间，才销毁；平台显示仍在运行时报告“失联（实例仍在
        运行）”并保留（runner 可能在最终上传或暂时连不上存储）。
        本 attempt 从没有过心跳的实例按启动窗口判断（实例创建时间来自平台）。
        """
        cfg = self.config
        inst_age = now - inst.created_at if inst.created_at is not None else None
        if not docs.exists:
            age = f"实例已存在 {inst_age / 60:.0f} 分钟" if inst_age is not None else "实例年龄未知"
            return False, (
                f"存储 {self.store.profile.name}（bucket {self.store.profile.bucket}）里查不到这个"
                f"任务，可能属于另一份配置或共用平台账户的人；{age}。不自动销毁，确认是自己的遗留"
                "实例后用 `dgpu gc --include-unknown` 销毁"
            )
        att = docs.attempt.doc if docs.attempt else None
        st = docs.status.doc if docs.status else None
        hb = docs.heartbeat.doc if docs.heartbeat else None
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
        hb_mine = hb is not None and hb.get("attempt") == attempt
        if hb_mine and docs.heartbeat is not None:
            silence = docs.heartbeat.age_s
            if silence <= cfg.gc_stale_min * 60:
                return False, f"运行中（{silence:.0f} 秒前有心跳）"
            quiet = f"心跳已停 {silence / 60:.0f} 分钟"
            if inst.phase in (INST_EXITED, INST_UNKNOWN):
                return True, f"{quiet}，平台显示实例 {inst.phase}（{inst.raw_status or '-'}）"
            teardown = self._teardown_worst_case_s()
            if silence > teardown:
                return True, f"{quiet}，超过 runner 最坏收尾时间 {teardown / 60:.0f} 分钟"
            return False, (
                f"失联（实例仍在运行）：{quiet}，平台显示 {inst.phase}；runner 可能在最终上传或"
                f"暂时连不上存储。心跳停满 {teardown / 60:.0f} 分钟或实例退出后才清理"
            )
        age = inst_age
        if age is None and docs.attempt is not None and attempt == current:
            age = docs.attempt.age_s  # 平台没给创建时间：用换到这个 attempt 的时间（存储时钟）
        if age is None:
            return False, "没有心跳，但实例年龄未知，保留"
        if age <= self._startup_window_s():
            return False, "实例还在启动窗口内，保留"
        return True, f"实例已存在 {age / 60:.0f} 分钟仍没有心跳"

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
        title = f"平台 {name}"
        try:
            balance = backend.balance()
            if balance is not None:
                return CheckResult(
                    title, True, f"可以访问，余额 {balance.amount:.2f} {balance.currency}"
                )
            # 余额查不到：用一个只读、需要认证的调用确认 key 本身有效
            listed = backend.list(layout.LABEL_PREFIX)
        except Exception as exc:
            return CheckResult(title, False, f"{type(exc).__name__}: {exc}")
        verified = f"key 有效（列出实例成功，当前有 {len(listed)} 台 dgpu 实例）"
        note = getattr(backend, "balance_note", "")
        if note:
            return CheckResult(
                title,
                True,
                f"{verified}；余额无法查询（{note}），提交前不会检查余额，请自己留意",
                warning=True,
            )
        return CheckResult(title, True, f"{verified}；平台不提供余额")


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


def _terminal_for_current(view: JobView) -> bool:
    """当前 attempt 已经有终态（status.json 是这个 attempt 写的终态）。"""
    return view.stored_phase in states.TERMINAL_PHASES and (view.status_attempt or 0) >= (
        view.attempt or 0
    )


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
