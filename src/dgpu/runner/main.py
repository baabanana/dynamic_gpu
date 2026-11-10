"""容器内 runner 的主流程（设计 4.7）。

    python3 -m dgpu.runner.main --work-root <root> [--launch <root>/launch.json]

步骤：读启动文档和 spec → 检查 attempt（是否已有更新的 attempt 或已取消）→ 写 RUNNING
和第一次心跳 → preflight → 下载代码和数据 → 恢复 checkpoint → 准备训练用户（降权，见
privsep.py）→ setup → run → 收尾（最终同步、日志、最终状态）→ 自毁。

收尾和自毁在所有退出路径上执行：正常结束、预期内失败、未处理异常、SIGTERM/SIGINT。
心跳线程、周期同步线程、事件上传线程、GPU 采样线程互相独立，任何一个卡住都不影响主循环
处理取消和截止时间。发现更新的 attempt 之后不再写 output/、ckpt/、status.json、
heartbeat.json（事件和本 attempt 的日志照常写）。

结论一确定就写进 run/final.json（synced=false），最终同步完成后改成 synced=true，最终状态
写进存储后 status_written=true。同一 attempt 的容器重启（Vast 每次启动都执行 onstart）按这个
文件接着做没做完的步骤，不重跑 setup 和训练。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from dgpu import layout, states
from dgpu.runner import archive, gpu, privsep, procs, syncstate
from dgpu.runner.destroy import DestroyResult, SelfDestroyer
from dgpu.runner.events import Events, flush_spool
from dgpu.runner.logsink import LogSink, Pump
from dgpu.runner.rclone import Rclone, RcloneError
from dgpu.runner.util import (
    INTERNAL_ENV,
    KILL_WAIT_S,
    RCLONE_ENV,
    WORK_ROOT_ENV,
    Clock,
    Paths,
    backoff,
    diag,
    exc_text,
    guess_backend,
    load_launch,
    load_tuning,
    read_json,
    runner_settings,
    scrubbed,
    short_op_s,
    write_json_atomic,
)

COMPONENT = "runner"
BACKENDS = ("vast", "autodl")
SAFE_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
# 同步的目录：(名字, 是否跳过 *.tmp)
SYNC_TARGETS = (("output", False), ("ckpt", True))
PATH0_ENV = "__DGPU_PATH0"  # 训练命令的登录 shell 执行 profile 之前的 PATH


class Outcome:
    """runner 的结论：最终阶段、失败原因、退出码、说明，以及是否因新 attempt 而退出。"""

    def __init__(
        self,
        phase: str,
        reason: Optional[str] = None,
        exit_code: Optional[int] = None,
        message: Optional[str] = None,
        fenced: bool = False,
        not_uploaded: Optional[List[str]] = None,
    ):
        self.phase = phase
        self.reason = reason
        self.exit_code = exit_code
        self.message = message
        self.fenced = fenced
        self.not_uploaded = list(not_uploaded or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "reason": self.reason,
            "exit_code": self.exit_code,
            "message": self.message,
            "fenced": self.fenced,
            "not_uploaded": self.not_uploaded,
        }

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "Outcome":
        return cls(
            doc.get("phase") or states.FAILED,
            doc.get("reason"),
            doc.get("exit_code"),
            doc.get("message"),
            bool(doc.get("fenced")),
            doc.get("not_uploaded") or [],
        )

    def apply_not_uploaded(self, missing: List[str], count: int) -> None:
        """最终同步后还有没传上去的文件：成功的训练改成 FAILED(final_sync)，其他结论补一句说明。"""
        if not missing:
            return
        if self.phase == states.SUCCEEDED:
            self.phase = states.FAILED
            self.reason = states.REASON_FINAL_SYNC
            self.message = "训练成功，但最终上传失败，%d 个文件没有传上去" % count
        else:
            self.message = "%s；另有 %d 个文件没有传上去" % (self.message or self.phase, count)
        self.not_uploaded = missing


class StepFailed(Exception):
    """预期内的失败或停止，带着结论。"""

    def __init__(self, outcome: Outcome):
        super().__init__(outcome.message or outcome.phase)
        self.outcome = outcome


def default_rclone_binary(environ: Dict[str, str]) -> str:
    """rclone 的位置：DGPU_RCLONE，否则 bundle 里的 bin/rclone，否则 PATH 上的 rclone。"""
    if environ.get(RCLONE_ENV):
        return environ[RCLONE_ENV]
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bundled = os.path.join(pkg_root, "bin", "rclone")
    if os.path.isfile(bundled):
        return bundled
    return "rclone"


def write_status_doc(
    rclone: Rclone,
    job_id: str,
    attempt: int,
    outcome: Outcome,
    source: str,
    timeout: float,
    guard: Optional[Callable[[], bool]] = None,
) -> None:
    doc = states.new_status_doc(
        job_id,
        attempt,
        outcome.phase,
        source,
        reason=outcome.reason,
        exit_code=outcome.exit_code,
        message=outcome.message,
        not_uploaded=outcome.not_uploaded,
    )
    data = json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")
    rclone.rcat(layout.status_key(job_id), data, timeout=timeout, tag="status", guard=guard)


def local_files(root: str, skip_tmp: bool) -> Dict[str, int]:
    """root 下所有文件的 相对路径 → 大小。"""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if skip_tmp and name.endswith(".tmp"):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            found[os.path.relpath(path, root).replace(os.sep, "/")] = size
    return found


def target_dirs(paths: Paths, job_id: str) -> List[Tuple[str, str, str, bool]]:
    """(名字, 本地目录, 存储前缀, 是否跳过 *.tmp)。"""
    return [
        ("output", paths.output, layout.output_prefix(job_id), False),
        ("ckpt", paths.ckpt, layout.ckpt_prefix(job_id), True),
    ]


def sync_dir(
    rclone: Rclone,
    paths: Paths,
    name: str,
    root: str,
    prefix: str,
    skip_tmp: bool,
    timeout: float,
    tag: str,
    guard: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, Optional[str]]:
    """同步一个目录，成功后把同步前的快照记为已上传。返回 (成功, 错误)。

    本地目录不存在时什么都不用传，算成功。被 guard 拦下（已被更新的 attempt 取代、正在停止）
    时返回 (False, None)。
    """
    if not os.path.isdir(root):
        return True, None
    try:
        before = syncstate.snapshot(root, skip_tmp)
    except syncstate.ListError as exc:
        return False, "cannot list %s: %s" % (root, exc)
    try:
        rclone.copy_up(
            root,
            prefix,
            exclude=["*.tmp"] if skip_tmp else [],
            timeout=timeout,
            tag=tag,
            guard=guard,
        )
    except RcloneError as exc:
        if exc.blocked:
            return False, None
        return False, str(exc)
    try:
        syncstate.record_success(paths.run, name, before)
    except OSError as exc:
        diag("sync", "cannot save the sync record of %s: %s" % (name, exc))
    return True, None


def not_uploaded(paths: Paths, job_id: str, limit: int) -> Tuple[List[str], int]:
    """按同步记录算出没有按当前内容上传的文件：(最多 limit 项的列表, 总数)。"""
    missing = []  # type: List[str]
    for name, root, _prefix, skip_tmp in target_dirs(paths, job_id):
        missing += syncstate.pending(root, skip_tmp, paths.run, name)
    return syncstate.limit_list(missing, max(1, limit)), len(missing)


def upload_diag_logs(
    rclone: Rclone,
    paths: Paths,
    job_id: str,
    attempt: int,
    tail_bytes: int,
    timeout: Callable[[], float],
    tag: str = "diag",
) -> List[str]:
    """上传 runner.log、watchdog.log、bootstrap.log 的最后 tail_bytes 字节。返回失败的说明。"""
    errors = []
    try:
        os.makedirs(paths.diag_upload, mode=0o700, exist_ok=True)
    except OSError as exc:
        return ["cannot create %s: %s" % (paths.diag_upload, exc)]
    for component in layout.DIAG_COMPONENTS:
        source = paths.diag_log(component)
        if not os.path.isfile(source):
            continue
        copy = os.path.join(paths.diag_upload, component + ".log")
        try:
            size = os.path.getsize(source)
            with open(source, "rb") as handle:
                if size > tail_bytes:
                    handle.seek(size - tail_bytes)
                data = handle.read(tail_bytes)
            with open(copy, "wb") as out:
                if size > tail_bytes:
                    out.write(
                        (
                            "[dgpu] %s.log 共 %d 字节，这里只有最后 %d 字节\n"
                            % (component, size, len(data))
                        ).encode("utf-8")
                    )
                out.write(data)
        except OSError as exc:
            errors.append("%s: %s" % (component, exc))
            continue
        try:
            rclone.copyto_up(
                copy, layout.diag_log_key(job_id, attempt, component), timeout=timeout(), tag=tag
            )
        except RcloneError as exc:
            errors.append("%s: %s" % (component, exc))
    return errors


def shell_script(code_dir: str, command: str, dgpu_env: Dict[str, str]) -> str:
    """训练命令外面包的 bash 脚本（用 `bash -lc` 运行）。

    登录 shell 会先执行镜像的 profile（/etc/profile 等，conda 镜像可能靠它设置环境）。profile
    执行完之后：再次移除凭据类变量（profile 可能从别处重新导出），把 profile 删掉的 PATH
    条目补回末尾（profile 可以加入或调整顺序，但不能让镜像的 PATH 丢东西），重新导出 dgpu 的
    变量，最后进入代码目录执行命令。
    """
    patterns = "|".join(_case_insensitive(prefix) + "*" for prefix in states.SCRUBBED_ENV_PREFIXES)
    exports = " ".join(
        "%s=%s" % (key, shlex.quote(value)) for key, value in sorted(dgpu_env.items())
    )
    lines = [
        "__dgpu_names=$(compgen -e 2>/dev/null) || __dgpu_names=$(env | sed 's/=.*//')",
        "for __dgpu_v in $__dgpu_names; do",
        '  case "$__dgpu_v" in %s) unset "$__dgpu_v" 2>/dev/null;; esac' % patterns,
        "done",
        "__dgpu_ifs=$IFS; IFS=:; set -f",
        "for __dgpu_d in ${%s-}; do" % PATH0_ENV,
        '  case ":$PATH:" in *":$__dgpu_d:"*) ;; *) PATH="${PATH:+$PATH:}$__dgpu_d";; esac',
        "done",
        "set +f; IFS=$__dgpu_ifs",
        "export PATH",
        "unset __dgpu_names __dgpu_v __dgpu_d __dgpu_ifs %s" % PATH0_ENV,
        "export %s" % exports,
        "cd %s && %s" % (shlex.quote(code_dir), command),
    ]
    return "\n".join(lines)


def _case_insensitive(prefix: str) -> str:
    """bash case 模式：字母两种大小写都匹配（兼容没有 ${v^^} 的 bash 3）。"""
    out = []
    for char in prefix:
        if char.isalpha():
            out.append("[%s%s]" % (char.upper(), char.lower()))
        else:
            out.append(char)
    return "".join(out)


class Runner:
    def __init__(
        self,
        paths: Paths,
        launch_path: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
        clock: Optional[Clock] = None,
        environ: Optional[Dict[str, str]] = None,
        rclone_binary: Optional[str] = None,
        destroyer: Optional[SelfDestroyer] = None,
        gpu_query: Optional[gpu.QueryFn] = None,
        privsep_system: Optional[privsep.System] = None,
    ):
        self.paths = paths
        self.launch_path = launch_path or paths.launch
        self.environ = dict(os.environ if environ is None else environ)
        self.tuning = tuning if tuning is not None else load_tuning(self.environ)
        self.clock = clock or Clock()
        self.rclone_binary = rclone_binary or default_rclone_binary(self.environ)
        self._destroyer = destroyer
        self._gpu_query = gpu_query or (lambda timeout: gpu.query(timeout))
        self._privsep_system = privsep_system
        self.t_start = self.clock.monotonic()

        self.launch = None  # type: Optional[Dict[str, Any]]
        self.job_id = ""
        self.attempt = 0
        self.backend = ""
        self.rclone = None  # type: Optional[Rclone]
        self.events = None  # type: Optional[Events]
        self.spec = None  # type: Optional[Dict[str, Any]]
        self.settings = runner_settings({})
        self.deadline = None  # type: Optional[float]
        self.max_hours = 0.0
        self.sink = None  # type: Optional[LogSink]
        self.identity = privsep.inactive(privsep.REASON_NOT_ROOT, "尚未决定")

        self.outcome = None  # type: Optional[Outcome]
        self.previous_final = None  # type: Optional[Dict[str, Any]]
        self.training_started = False
        self.resume = False
        self.phase = states.RUNNING
        self.step = "start"

        self.fenced = threading.Event()
        self.cancel = threading.Event()
        self.signal_name = None  # type: Optional[str]
        self.wake = threading.Event()
        self._flag_lock = threading.Lock()
        self._stop_events_sent = set()  # type: set
        self._attempt_checked_at = None  # type: Optional[float]

        self.gpu_util = None  # type: Optional[float]
        self.sampler = None  # type: Optional[gpu.Sampler]
        self.phase_start = None  # type: Optional[float]
        self.child = None
        self.synced_bytes = 0
        self.last_sync_at = None  # type: Optional[float]

        self._hb_thread = None  # type: Optional[threading.Thread]
        self._hb_stop = threading.Event()
        self._sync_thread = None  # type: Optional[threading.Thread]
        self._sync_stop = threading.Event()
        self._sync_fail_count = 0
        self._sync_fail_last_event = None  # type: Optional[float]
        self._budget_end = 0.0

    # ================================================================ 入口

    def run(self) -> int:
        try:
            self._run_job()
        except StepFailed as exc:
            if self.outcome is None:
                self.outcome = exc.outcome
        except BaseException as exc:  # 任何异常都要走收尾和自毁
            diag(COMPONENT, "unhandled error: %s\n%s" % (exc_text(exc), traceback.format_exc()))
            self._note(
                "runner 内部错误：%s（traceback 见 logs/attempt-%d.runner.log）"
                % (exc_text(exc), self.attempt)
            )
            if self.outcome is None:
                stop = self._stop_reason()
                if stop is not None:
                    self.outcome = self._outcome_for_stop(stop)
                else:
                    self.outcome = Outcome(
                        states.FAILED,
                        states.REASON_RUNNER_CRASHED,
                        message="runner 内部错误：%s" % exc_text(exc),
                    )
        return self._teardown()

    def request_stop(self, name: str) -> None:
        """信号处理函数调用：记下信号，唤醒主循环。只设标志，不做其他事。"""
        if self.signal_name is None:
            self.signal_name = name
        self.wake.set()

    # ================================================================ 主流程

    def _run_job(self) -> None:
        self._load_launch()
        self.previous_final = self._read_final_marker()
        if self.previous_final is not None:
            # 同一 attempt 已经得出结论（Vast 容器重启后 onstart 又执行了一次）：不重跑训练，
            # 只接着做没做完的收尾（最终同步、日志、最终状态）
            self.outcome = Outcome.from_dict(self.previous_final)
            self._load_local_spec()
            pending = not self.previous_final.get("synced", True)
            self._note(
                "本 attempt 已经得出结论 %s%s，只做剩下的收尾"
                % (self.outcome.phase, "（最终同步没有做完）" if pending else "")
            )
            if pending and not os.path.exists(self.paths.fenced):
                self.phase = states.UPLOADING
                self.step = "final_sync"
                self._start_heartbeat_thread()
            return
        self._load_spec()
        self._fence_check_at_start()
        self._announce_running()
        self._checkpoint()
        self._preflight()
        self._checkpoint()
        self._report_preflight_passed()
        self.training_started = True
        self._fetch_code_and_data()
        self._checkpoint()
        self._restore_ckpt()
        self._checkpoint()
        self._prepare_identity()
        self._checkpoint()
        self._start_sync_thread()
        setup = self.spec.get("setup") if self.spec else None
        if setup:
            rc = self._run_command("setup", setup)
            if rc != 0:
                raise StepFailed(
                    Outcome(states.FAILED, states.REASON_SETUP, rc, "setup 命令退出码 %d" % rc)
                )
        rc = self._run_command("run", self.spec["run"])
        if rc != 0:
            raise StepFailed(
                Outcome(states.FAILED, states.REASON_EXIT_CODE, rc, "run 命令退出码 %d" % rc)
            )
        self.outcome = Outcome(states.SUCCEEDED, exit_code=0, message="训练命令正常结束")

    # ---------------------------------------------------------------- 启动文档、存储、spec

    def _load_launch(self) -> None:
        self.paths.ensure()
        try:
            self.launch = load_launch(self.launch_path)
        except (ValueError, TypeError) as exc:
            self.backend = guess_backend(self.environ)
            raise StepFailed(
                Outcome(states.FAILED, states.REASON_PREFLIGHT, message="启动文档无效：%s" % exc)
            ) from None
        self.job_id = str(self.launch["job_id"])
        self.attempt = int(self.launch["attempt"])
        self.backend = str(self.launch["backend"])
        if not layout.is_valid_job_id(self.job_id):
            raise StepFailed(
                Outcome(states.FAILED, states.REASON_PREFLIGHT, message="启动文档里的任务 ID 无效")
            )
        self.rclone = Rclone(
            self.rclone_binary,
            self.launch["store"],
            self.paths.rclone_conf,
            base_env=self.environ,
            small_timeout=self.tuning["small_op_timeout_s"],
            poll_s=self.tuning["rclone_poll_s"],
            clock=self.clock,
            pid_dir=self.paths.rclone_pids(COMPONENT),
        )
        self.events = Events(
            self.rclone,
            self.job_id,
            self.attempt,
            states.SRC_RUNNER,
            self.paths.event_spool(states.SRC_RUNNER),
            self.paths.events_tmp,
        )
        self.events.start_flusher(self.tuning["events_flush_interval_s"])
        if self.backend not in BACKENDS:
            raise StepFailed(
                Outcome(
                    states.FAILED,
                    states.REASON_PREFLIGHT,
                    message="启动文档里的后端 %r 未知" % self.backend,
                )
            )

    def _load_spec(self) -> None:
        def read() -> Optional[bytes]:
            return self.rclone.cat(layout.spec_key(self.job_id), interrupt=self._stop_requested)

        try:
            data = self._retry_startup("读取 spec.json", read)
        except RcloneError as exc:
            raise self._storage_failure("读不到 spec.json，实例可能连不上存储：%s" % exc) from None
        try:
            if data is None:
                raise ValueError("spec.json does not exist")
            spec = json.loads(data.decode("utf-8"))
            self._accept_spec(spec)
        except (KeyError, TypeError, ValueError) as exc:
            raise StepFailed(
                Outcome(states.FAILED, states.REASON_PREFLIGHT, message="spec.json 无效：%s" % exc)
            ) from None
        write_json_atomic(self.paths.spec_copy, spec)
        self.deadline = self.t_start + self._remaining_budget_s()
        self._open_sink()
        self.sink.note("runner 启动：任务 %s，第 %d 次尝试" % (self.job_id, self.attempt))

    def _accept_spec(self, spec: Any) -> None:
        if not isinstance(spec, dict):
            raise ValueError("spec.json is not an object")
        states.check_version(spec, "spec.json")
        if spec.get("job_id") != self.job_id:
            raise ValueError("spec.json belongs to another job")
        if not isinstance(spec.get("run"), str) or not spec["run"].strip():
            raise ValueError("spec.json has no run command")
        if not isinstance(spec.get("data") or [], list):
            raise ValueError("spec.json data is not a list")
        if not isinstance(spec.get("env") or {}, dict):
            raise ValueError("spec.json env is not an object")
        self.max_hours = float(spec["max_hours"])
        self.spec = spec
        self.settings = runner_settings(spec)

    def _load_local_spec(self) -> None:
        """重启后只做收尾：用上次运行留在本地的 spec 副本（不依赖存储），读不到就用默认参数。"""
        doc = read_json(self.paths.spec_copy)
        if doc is not None:
            try:
                self._accept_spec(doc)
            except (KeyError, TypeError, ValueError) as exc:
                diag(COMPONENT, "local spec copy unusable: %s" % exc)
        self._open_sink()

    def _open_sink(self) -> None:
        if self.sink is not None:
            return
        self.sink = LogSink(
            self.paths.attempt_log(self.attempt),
            int(self.settings["max_log_mb"] * 1024 * 1024),
            self.clock,
            on_truncate=self._on_log_truncated,
            tail_bytes=int(self.tuning["log_tail_kb"] * 1024),
            note_reserve=int(self.tuning["log_note_reserve_kb"] * 1024),
        )

    def _remaining_budget_s(self) -> float:
        """max_hours 从第一次启动算起（容器重启不重新计时）。"""
        budget = self.max_hours * 3600.0
        started = read_json(self.paths.started)
        if started and started.get("attempt") == self.attempt:
            try:
                elapsed = max(0.0, self.clock.time() - float(started["wall"]))
                budget -= elapsed
            except (KeyError, TypeError, ValueError):
                pass
        return budget

    def _retry_startup(self, what: str, fn: Callable[[], Any]) -> Any:
        attempts = max(1, int(self.tuning["startup_attempts"]))
        last = None
        for index in range(attempts):
            self._checkpoint()
            try:
                return fn()
            except RcloneError as exc:
                last = exc
                diag(COMPONENT, "%s failed (%d/%d): %s" % (what, index + 1, attempts, exc))
            if index + 1 < attempts:
                self._wait(self._backoff(index))
        self._checkpoint()
        raise last

    # ---------------------------------------------------------------- attempt 与停止

    def _read_attempt(
        self, timeout: Optional[float] = None, tag: str = "attempt"
    ) -> Optional[Dict[str, Any]]:
        data = self.rclone.cat(layout.attempt_key(self.job_id), timeout=timeout, tag=tag)
        if data is None:
            return None
        doc = json.loads(data.decode("utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("attempt.json is not an object")
        states.check_version(doc, "attempt.json")
        return doc

    def _check_attempt(self, timeout: Optional[float] = None, tag: str = "attempt") -> bool:
        """读一次 attempt.json 并处理（更新的 attempt、取消）。返回是否读成功。"""
        try:
            doc = self._read_attempt(timeout=timeout, tag=tag)
        except (RcloneError, ValueError) as exc:
            diag(COMPONENT, "cannot read attempt.json: %s" % exc)
            return False
        self._attempt_checked_at = self.clock.monotonic()
        if doc is not None:
            self._apply_attempt(doc)
        return True

    def _attempt_fresh(self) -> bool:
        """最近一次成功读 attempt.json 是否足够新（周期同步写入之前用）。"""
        if self._attempt_checked_at is None:
            return False
        limit = max(1.0, 3.0 * self.settings["heartbeat_interval_s"])
        return self.clock.monotonic() - self._attempt_checked_at <= limit

    def _apply_attempt(self, doc: Dict[str, Any]) -> None:
        try:
            current = int(doc.get("attempt"))
        except (TypeError, ValueError):
            diag(COMPONENT, "attempt.json has no valid attempt number; ignored")
            return
        if current > self.attempt:
            self._mark_fenced(current)
        elif current == self.attempt and doc.get("cancel") is True:
            if not self.cancel.is_set():
                self.cancel.set()
                self._note("收到取消请求")
                self.wake.set()
        elif current < self.attempt:
            diag(COMPONENT, "attempt.json says attempt %d < ours (%d)" % (current, self.attempt))

    def _mark_fenced(self, newer: int) -> None:
        with self._flag_lock:
            if self.fenced.is_set():
                return
            self.fenced.set()
        # 先置标志再中止：之后任何受保护的写入都会被 guard 拦下
        if self.rclone is not None:
            self.rclone.abort()
        try:
            write_json_atomic(self.paths.fenced, {"attempt": self.attempt, "newer_attempt": newer})
        except OSError as exc:
            diag(COMPONENT, "cannot write fenced marker: %s" % exc)
        self._note("发现更新的 attempt %d，本 attempt 停止且不再写入结果和状态" % newer)
        if self.events is not None:
            self.events.emit(
                states.EV_ATTEMPT_FENCED,
                "attempt %d 发现更新的 attempt %d，停止训练，不再写 output/ckpt/status/heartbeat"
                % (self.attempt, newer),
                {"newer_attempt": newer},
            )
        self.wake.set()

    def _may_write(self) -> bool:
        return not self.fenced.is_set()

    def _stop_reason(self, include_stall: bool = False) -> Optional[str]:
        if self.fenced.is_set():
            return "fenced"
        if self.cancel.is_set():
            return "cancel"
        if self.signal_name is not None:
            return "signal"
        if self.deadline is not None and self.clock.monotonic() >= self.deadline:
            return "deadline"
        if include_stall and self._stalled():
            return "stalled"
        return None

    def _stop_requested(self) -> bool:
        return self._stop_reason() is not None

    def _outcome_for_stop(self, reason: str) -> Outcome:
        if reason == "fenced":
            return Outcome(
                states.FAILED,
                states.REASON_FENCED,
                message="发现更新的 attempt，本 attempt 退出",
                fenced=True,
            )
        if reason == "cancel":
            return Outcome(states.CANCELLED, message="已按请求取消")
        if reason == "signal":
            return Outcome(
                states.FAILED,
                states.REASON_SIGNALLED,
                message="runner 收到 %s，提前收尾" % (self.signal_name or "signal"),
            )
        if reason == "deadline":
            return Outcome(states.TIMED_OUT, message="超过 max_hours=%g 小时" % self.max_hours)
        if reason == "stalled":
            return Outcome(
                states.FAILED,
                states.REASON_STALLED,
                message="GPU 利用率为 0 且 %g 分钟没有新日志" % self.settings["stall_timeout_min"],
            )
        return Outcome(
            states.FAILED, states.REASON_RUNNER_CRASHED, message="unknown stop: %s" % reason
        )

    def _record_stop(self, reason: str) -> None:
        """停止原因第一次出现时写事件（deadline_hit、stalled）。事件只进本地队列，不等存储。"""
        if reason in self._stop_events_sent or self.events is None:
            return
        self._stop_events_sent.add(reason)
        if reason == "deadline":
            self._note("到达 max_hours，终止训练")
            self.events.emit(
                states.EV_DEADLINE_HIT,
                "到达 max_hours=%g 小时，step=%s" % (self.max_hours, self.step),
                {"max_hours": self.max_hours, "step": self.step},
            )
        elif reason == "stalled":
            self._note("卡死检测触发，终止训练")
            self.events.emit(
                states.EV_STALLED,
                "GPU 利用率为 0 且 %g 分钟没有新日志" % self.settings["stall_timeout_min"],
                {"stall_timeout_min": self.settings["stall_timeout_min"], "step": self.step},
            )

    def _checkpoint(self) -> None:
        reason = self._stop_reason()
        if reason is not None:
            self._record_stop(reason)
            raise StepFailed(self._outcome_for_stop(reason))

    def _wait(self, seconds: float) -> None:
        if self.clock.sleep(seconds, self.wake):
            self.wake.clear()

    def _backoff(self, index: int) -> float:
        return backoff(self.tuning["retry_backoff_s"], index, self.tuning["retry_backoff_max_s"])

    def _fence_check_at_start(self) -> None:
        try:
            doc = self._retry_startup("读取 attempt.json", lambda: self._read_attempt())
        except RcloneError as exc:
            raise self._storage_failure(
                "读不到 attempt.json，实例可能连不上存储：%s" % exc
            ) from None
        except ValueError as exc:
            raise StepFailed(
                Outcome(
                    states.FAILED, states.REASON_PREFLIGHT, message="attempt.json 无效：%s" % exc
                )
            ) from None
        self._attempt_checked_at = self.clock.monotonic()
        if doc is None:
            diag(COMPONENT, "attempt.json does not exist; skipping the start-up attempt check")
            return
        self._apply_attempt(doc)
        if self.fenced.is_set():
            raise StepFailed(self._outcome_for_stop("fenced"))
        if self.cancel.is_set():
            raise StepFailed(Outcome(states.CANCELLED, message="启动前已被取消"))

    # ---------------------------------------------------------------- 状态与心跳

    def _put_status(self, outcome: Outcome, timeout: Optional[float] = None) -> None:
        write_status_doc(
            self.rclone,
            self.job_id,
            self.attempt,
            outcome,
            states.SRC_RUNNER,
            self.tuning["small_op_timeout_s"] if timeout is None else timeout,
            guard=self._may_write,
        )

    def _put_heartbeat(self) -> float:
        doc = states.new_heartbeat_doc(
            self.job_id,
            self.attempt,
            self.phase,
            self.step,
            gpu_util=self.gpu_util,
            last_log_at=self.sink.last_output_wall if self.sink else None,
            synced_bytes=self.synced_bytes,
            last_sync_at=self.last_sync_at,
        )
        data = json.dumps(doc, sort_keys=True).encode("utf-8")
        self.rclone.rcat(
            layout.heartbeat_key(self.job_id), data, tag="heartbeat", guard=self._may_write
        )
        return doc["ts"]

    def _storage_failure(self, message: str) -> StepFailed:
        """启动阶段存储不可用：记 preflight_failed（存储恢复前只留在本地队列）。"""
        self._note(message)
        if self.events is not None:
            self.events.emit(states.EV_PREFLIGHT_FAILED, message, {"problems": [message]})
        return StepFailed(Outcome(states.FAILED, states.REASON_PREFLIGHT, message=message))

    def _announce_running(self) -> None:
        self.phase = states.RUNNING
        self.step = "preflight"
        try:
            self._retry_startup(
                "写 RUNNING 状态",
                lambda: self._put_status(Outcome(states.RUNNING, message="runner 已启动")),
            )
            self._retry_startup("写第一次心跳", self._put_heartbeat)
        except RcloneError as exc:
            if exc.blocked:
                raise StepFailed(self._outcome_for_stop("fenced")) from None
            raise self._storage_failure("写不进存储：%s" % exc) from None
        self._start_heartbeat_thread()

    def _start_heartbeat_thread(self) -> None:
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, name="heartbeat", daemon=True
        )
        self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        interval = max(0.01, self.settings["heartbeat_interval_s"])
        while not self._hb_stop.wait(interval):
            try:
                self._beat()
            except Exception as exc:
                diag(COMPONENT, "heartbeat error: %s" % exc_text(exc))

    def _beat(self) -> None:
        self._check_attempt()
        if self.fenced.is_set() or self._hb_stop.is_set():
            return
        try:
            self._put_heartbeat()
        except RcloneError as exc:
            if not exc.blocked:
                self._note_sync_failure("heartbeat", str(exc))

    # ---------------------------------------------------------------- preflight

    def _preflight(self) -> None:
        self.step = "preflight"
        problems = []
        # nvidia-smi 卡住时查询有上限（超时 + SIGKILL 后的等待），算作 GPU 故障，CLI 会换机
        gpus, error = self._gpu_query(self.tuning["nvidia_smi_timeout_s"])
        if not gpus:
            problems.append("GPU 不可用：%s" % error)
        else:
            names = ", ".join(str(item.get("name")) for item in gpus)
            self._note("GPU：%s" % names)
        need, unknown = self._disk_need()
        try:
            free = shutil.disk_usage(self.paths.root).free
        except OSError as exc:
            free = None
            problems.append("读不到磁盘余量：%s" % exc)
        if free is not None and free < need:
            problems.append(
                "磁盘余量不足：可用 %.1f GB，至少需要 %.1f GB" % (free / 1e9, need / 1e9)
            )
        if unknown:
            self._note("没能估算这些数据的大小，磁盘检查不含它们：%s" % ", ".join(unknown))
        try:
            data = self.rclone.cat(
                layout.heartbeat_key(self.job_id), interrupt=self._stop_requested
            )
            doc = json.loads(data.decode("utf-8")) if data else None
            if not isinstance(doc, dict) or doc.get("job_id") != self.job_id:
                problems.append("存储读写不一致：读不回刚写的心跳")
        except (RcloneError, ValueError) as exc:
            self._checkpoint()
            problems.append("读不回刚写的心跳：%s" % exc)
        if problems:
            self._checkpoint()
            message = "；".join(problems)
            self._note("preflight 失败：%s" % message)
            self.events.emit(states.EV_PREFLIGHT_FAILED, message, {"problems": problems})
            raise StepFailed(Outcome(states.FAILED, states.REASON_PREFLIGHT, message=message))

    def _report_preflight_passed(self) -> None:
        """preflight 通过后立刻写一次心跳（step 已离开 preflight）。

        CLI 看到 step 离开 preflight 才认为实例可用（preflight 失败要换机），不用等下一个
        心跳周期。写失败只记录，心跳线程随后还会写。
        """
        self.step = "fetch"
        try:
            self._put_heartbeat()
        except RcloneError as exc:
            if not exc.blocked:
                diag(COMPONENT, "heartbeat after preflight failed: %s" % exc)

    def _disk_need(self) -> Tuple[int, List[str]]:
        need = int(self.tuning["min_free_margin_mb"] * 1024 * 1024)
        unknown = []
        items = [
            (source.get("name"), source.get("bucket"), source.get("prefix"))
            for source in self._sources()
        ]
        for name, bucket, prefix in items:
            if os.path.exists(self.paths.ready_marker("data-%s" % name)):
                continue
            try:
                need += self.rclone.size(prefix, bucket=bucket, interrupt=self._stop_requested)[1]
            except RcloneError:
                self._checkpoint()
                unknown.append(str(name))
        try:
            need += self.rclone.size(
                layout.ckpt_prefix(self.job_id), interrupt=self._stop_requested
            )[1]
        except RcloneError:
            self._checkpoint()
            unknown.append("ckpt")
        return need, unknown

    def _sources(self) -> List[Dict[str, Any]]:
        sources = []
        for item in (self.spec or {}).get("data") or []:
            if isinstance(item, dict):
                sources.append(item)
        return sources

    # ---------------------------------------------------------------- 代码、数据、checkpoint

    def _data_retry(self, item: str, fn: Callable[[], Any]) -> Any:
        attempts = 1 + max(0, int(self.settings["data_sync_retries"]))
        last = None
        for index in range(attempts):
            self._checkpoint()
            try:
                return fn()
            except archive.UnsafeArchive as exc:
                raise StepFailed(
                    Outcome(
                        states.FAILED, states.REASON_DATA_SYNC, message="代码包不安全：%s" % exc
                    )
                ) from None
            except (RcloneError, OSError, ValueError) as exc:
                last = exc
            self._checkpoint()
            if index + 1 < attempts:
                self._note(
                    "下载 %s 失败（第 %d 次），稍后重试：%s" % (item, index + 1, exc_text(last))
                )
                self.events.emit(
                    states.EV_DATA_SYNC_RETRY,
                    "下载 %s 失败，第 %d 次重试" % (item, index + 1),
                    {"item": item, "retry": index + 1, "of": attempts - 1, "error": exc_text(last)},
                )
                self._wait(self._backoff(index))
        message = "下载 %s 失败（已重试 %d 次）：%s" % (item, attempts - 1, exc_text(last))
        self._note(message)
        raise StepFailed(Outcome(states.FAILED, states.REASON_DATA_SYNC, message=message))

    def _fetch_code_and_data(self) -> None:
        self.step = "fetch"
        transfer = self.tuning["transfer_timeout_s"]
        if os.path.exists(self.paths.ready_marker("code")) and os.path.isdir(self.paths.code):
            self._note("代码已在本地（.ready），跳过下载")
        else:

            def fetch_code() -> None:
                self.rclone.copyto_down(
                    layout.code_key(self.job_id),
                    self.paths.code_archive,
                    timeout=transfer,
                    interrupt=self._stop_requested,
                )
                tmp = self.paths.code + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                count = archive.safe_extract(self.paths.code_archive, tmp)
                shutil.rmtree(self.paths.code, ignore_errors=True)
                os.rename(tmp, self.paths.code)
                self._note("代码已解压（%d 个条目）" % count)

            self._data_retry("code", fetch_code)
            _touch(self.paths.ready_marker("code"))
        os.makedirs(self.paths.data, exist_ok=True)
        os.makedirs(self.paths.output, exist_ok=True)
        os.makedirs(self.paths.ckpt, exist_ok=True)
        for source in self._sources():
            name = str(source.get("name") or "")
            if not name or name in (".", "..") or not set(name) <= SAFE_NAME_CHARS:
                raise StepFailed(
                    Outcome(
                        states.FAILED, states.REASON_DATA_SYNC, message="数据名不合法：%r" % name
                    )
                )
            marker = self.paths.ready_marker("data-%s" % name)
            dest = os.path.join(self.paths.data, name)
            if os.path.exists(marker) and os.path.isdir(dest):
                self._note("数据 %s 已在本地（.ready），跳过下载" % name)
                continue
            bucket, prefix = source.get("bucket"), source.get("prefix")
            if not bucket or not prefix:
                raise StepFailed(
                    Outcome(
                        states.FAILED,
                        states.REASON_DATA_SYNC,
                        message="数据 %s 缺少 bucket 或 prefix" % name,
                    )
                )
            self._note("下载数据 %s" % name)

            def fetch_data(prefix: str = prefix, dest: str = dest, bucket: str = bucket) -> None:
                self.rclone.copy_down(
                    prefix, dest, bucket=bucket, timeout=transfer, interrupt=self._stop_requested
                )

            self._data_retry(name, fetch_data)
            _touch(marker)
            self._note("数据 %s 下载完成" % name)

    def _restore_ckpt(self) -> None:
        self.step = "restore"
        prefix = layout.ckpt_prefix(self.job_id)
        listing = self._data_retry(
            "ckpt", lambda: self.rclone.lsjson(prefix, interrupt=self._stop_requested)
        )
        remote = [item for item in listing if not str(item.get("Path", "")).endswith(".tmp")]
        if remote:
            before = self._snapshot_or_empty(self.paths.ckpt, True)
            self._data_retry(
                "ckpt",
                lambda: self.rclone.copy_down(
                    prefix,
                    self.paths.ckpt,
                    exclude=["*.tmp"],
                    update=True,
                    timeout=self.tuning["transfer_timeout_s"],
                    interrupt=self._stop_requested,
                ),
            )
            # 下载下来的文件就是存储里的版本：记为已上传，最终同步失败时不会被误报
            after = self._snapshot_or_empty(self.paths.ckpt, True)
            try:
                syncstate.record_downloaded(self.paths.run, "ckpt", before, after)
            except OSError as exc:
                diag(COMPONENT, "cannot save the ckpt sync record: %s" % exc)
            total = sum(int(item.get("Size") or 0) for item in remote)
            self._note("从存储恢复了 %d 个 checkpoint 文件" % len(remote))
            self.events.emit(
                states.EV_CKPT_RESTORED,
                "恢复了 %d 个 checkpoint 文件（%d 字节）" % (len(remote), total),
                {"files": len(remote), "bytes": total},
            )
        self.resume = bool(local_files(self.paths.ckpt, skip_tmp=True))
        if self.resume:
            self._note("DGPU_RESUME=1")

    @staticmethod
    def _snapshot_or_empty(root: str, skip_tmp: bool) -> syncstate.Snapshot:
        try:
            return syncstate.snapshot(root, skip_tmp)
        except syncstate.ListError:
            return {}

    # ---------------------------------------------------------------- 训练进程

    def _base_child_env(self) -> Dict[str, str]:
        env = {key: value for key, value in self.environ.items() if not scrubbed(key)}
        for key in INTERNAL_ENV:
            env.pop(key, None)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.update(self._dgpu_env())
        return env

    def _dgpu_env(self) -> Dict[str, str]:
        return {
            states.ENV_JOB_ID: self.job_id,
            states.ENV_ATTEMPT: str(self.attempt),
            states.ENV_DATA: self.paths.data,
            states.ENV_OUTPUT: self.paths.output,
            states.ENV_CKPT: self.paths.ckpt,
            states.ENV_RESUME: "1" if self.resume else "0",
        }

    def _child_env(self) -> Dict[str, str]:
        env = self._base_child_env()
        dropped = []
        for key, value in (self.spec.get("env") or {}).items():
            if scrubbed(str(key)):
                dropped.append(str(key))
            else:
                env[str(key)] = str(value)
        if dropped:
            self._note(
                "job.yaml 的 env 里这些变量匹配凭据前缀，没有传给训练进程：%s"
                % ", ".join(sorted(dropped))
            )
        env.update(self.identity.env)
        env.update(self._dgpu_env())
        env[PATH0_ENV] = env.get("PATH", "")
        return env

    def _prepare_identity(self) -> None:
        """决定训练进程以什么用户运行，准备目录（见 privsep.py）。"""
        self.step = "prepare"
        self.identity = privsep.prepare(
            self.paths,
            run_as_root=bool((self.spec or {}).get("run_as_root")),
            base_env=self._base_child_env(),
            system=self._privsep_system,
            probe_timeout=self.tuning["privsep_probe_timeout_s"],
            note=self._note,
        )
        try:
            write_json_atomic(self.paths.privsep, self.identity.to_dict())
        except OSError as exc:
            diag(COMPONENT, "cannot write privsep marker: %s" % exc)
        self._note(self.identity.describe())
        if not self.identity.active:
            self.events.emit(
                states.EV_RUN_AS_ROOT,
                "训练进程没有降权：%s" % self.identity.detail,
                {"reason": self.identity.reason, "detail": self.identity.detail[:500]},
            )

    def _run_command(self, step: str, command: str) -> int:
        self._checkpoint()
        self.step = step
        self.phase_start = self.clock.monotonic()
        self._note(
            "开始 %s：%s" % (step, command if len(command) <= 300 else command[:300] + "...")
        )
        script = shell_script(self.paths.code, command, self._dgpu_env())
        proc = procs.start_session(
            ["bash", "-lc", script],
            self.paths.code,
            self._child_env(),
            preexec_fn=self.identity.preexec(),
        )
        self.child = proc
        try:
            write_json_atomic(self.paths.child, {"pid": proc.pid, "pgid": proc.pid, "step": step})
        except OSError as exc:
            diag(COMPONENT, "cannot write child marker: %s" % exc)
        pump = Pump(proc.stdout, self.sink)
        pump.start()
        self.sampler = gpu.Sampler(
            self._gpu_query,
            self.tuning["nvidia_smi_timeout_s"],
            min(self.tuning["gpu_sample_max_s"], max(0.01, self.settings["heartbeat_interval_s"])),
            self.clock,
            note=self._note,
        )
        self.sampler.start()
        stop = None
        rc = None  # type: Optional[int]
        try:
            while proc.poll() is None:
                self.gpu_util = self.sampler.snapshot()[0]
                # 卡死检测只用于 run：setup 里安静的下载、解压不算卡死
                stop = self._stop_reason(include_stall=(step == "run"))
                if stop is not None:
                    break
                self._wait(self.tuning["tick_s"])
            if stop is not None:
                self._record_stop(stop)
                result = procs.terminate(
                    proc.pid, proc.pid, self.settings["term_grace_s"], self.clock, reap=proc.poll
                )
                self._note("%s 已终止（%s，%s）" % (step, stop, result))
            elif procs.group_alive(proc.pid):
                result = procs.terminate(
                    proc.pid, proc.pid, self.tuning["lingering_grace_s"], self.clock, reap=proc.poll
                )
                self._note("%s 结束后清理了残留的子进程（%s）" % (step, result))
            rc = procs.wait_bounded(proc, KILL_WAIT_S, close_pipes=False)
        finally:
            if proc.poll() is None:
                procs.terminate(
                    proc.pid, proc.pid, self.settings["term_grace_s"], self.clock, reap=proc.poll
                )
                rc = procs.wait_bounded(proc, KILL_WAIT_S, close_pipes=False)
            self.sampler.stop()
            pump.join(self.tuning["thread_join_s"])
            if pump.is_alive():
                diag(COMPONENT, "log pump of %s still running after the process exited" % step)
            self.child = None
            _unlink(self.paths.child)
        if rc is None:
            self._note("%s 的进程在 SIGKILL 之后仍未退出，放弃等待" % step)
            rc = -int(signal.SIGKILL)
        self._note("%s 结束，退出码 %d" % (step, rc))
        if stop is not None:
            raise StepFailed(self._outcome_for_stop(stop))
        return rc

    def _stalled(self) -> bool:
        limit = self.settings["stall_timeout_min"] * 60.0
        if limit <= 0 or self.step != "run" or self.phase_start is None or self.sampler is None:
            return False
        util, last_busy, sampled = self.sampler.snapshot()
        if not sampled or util is None or util > 0:
            return False  # 利用率未知或不为 0：不判卡死
        marks = [self.phase_start, last_busy]
        if self.sink is not None:
            marks.append(self.sink.last_output_mono)
        latest = max(mark for mark in marks if mark is not None)
        return self.clock.monotonic() - latest >= limit

    def _on_log_truncated(self, max_bytes: int) -> None:
        # 在读日志的线程里调用：只写本地事件队列，由后台线程上传，不能卡住读取
        if self.events is not None:
            self.events.emit(
                states.EV_LOG_TRUNCATED,
                "日志超过 %d MB，之后的训练输出只保留最后一段，附在日志末尾"
                % (max_bytes // (1024 * 1024)),
                {"max_bytes": max_bytes, "tail_bytes": int(self.tuning["log_tail_kb"] * 1024)},
            )

    # ---------------------------------------------------------------- 同步

    def _start_sync_thread(self) -> None:
        self._sync_thread = threading.Thread(target=self._sync_loop, name="sync", daemon=True)
        self._sync_thread.start()

    def _periodic_guard(self) -> bool:
        return self._may_write() and not self._sync_stop.is_set()

    def _sync_loop(self) -> None:
        interval = max(0.01, self.settings["sync_interval_s"])
        while not self._sync_stop.wait(interval):
            if self.fenced.is_set():
                return
            # 写 output/、ckpt/ 之前确认本 attempt 仍是当前的：心跳线程卡住时这里自己读一次
            if not self._attempt_fresh():
                ok = self._check_attempt(tag="sync")
                if self.fenced.is_set() or self._sync_stop.is_set():
                    return
                if not ok:
                    self._note_sync_failure("attempt", "读不到 attempt.json，这一轮不写入")
                    continue
            try:
                errors = self._sync_targets(
                    self.tuning["periodic_sync_timeout_s"], "sync", stop=self._sync_stop
                )
            except Exception as exc:
                errors = {"sync": exc_text(exc)}
            if self._sync_stop.is_set():
                return
            if errors:
                for what, error in sorted(errors.items()):
                    self._note_sync_failure(what, error)
            elif self._sync_fail_count:
                self._note("同步恢复（此前失败 %d 次）" % self._sync_fail_count)
                self._sync_fail_count = 0
                self._sync_fail_last_event = None

    def _sync_targets(
        self,
        timeout: float,
        tag: str,
        include_log: bool = True,
        end: Optional[float] = None,
        stop: Optional[threading.Event] = None,
    ) -> Dict[str, str]:
        """同步 output/、ckpt/（不含 *.tmp）和日志。返回 {目标: 错误}。

        end（单调时钟）：给了就只把剩下的时间给每一步，时间用完的步骤不再开始（记为失败），
        整轮不超过 end。最终同步用它，所以总时长受 final_sync_timeout_min 约束。
        stop：周期同步的停止标志，每个目标开始前检查，并作为写入的 guard（收尾时周期同步
        不会再开始新的上传，和最终同步抢带宽）。
        """
        errors = {}
        guard = self._periodic_guard if stop is not None else self._may_write
        for name, root, prefix, skip_tmp in target_dirs(self.paths, self.job_id):
            if stop is not None and stop.is_set():
                return errors
            if self.fenced.is_set():
                continue
            limit = timeout
            if end is not None:
                limit = end - self.clock.monotonic()
                if limit <= 0:
                    errors[name] = "final sync time limit reached before this step"
                    continue
            ok, error = sync_dir(
                self.rclone, self.paths, name, root, prefix, skip_tmp, limit, tag, guard
            )
            if error is not None:
                errors[name] = error
        if include_log and self.sink is not None and not (stop is not None and stop.is_set()):
            try:
                self._upload_log(timeout, tag, guard=(lambda: not stop.is_set()) if stop else None)
            except RcloneError as exc:
                if not exc.blocked:
                    errors["log"] = str(exc)
        self.synced_bytes = syncstate.total_bytes(self.paths.run, ("output", "ckpt"))
        if not errors.get("output") and not errors.get("ckpt") and not self.fenced.is_set():
            self.last_sync_at = self.clock.time()
        return errors

    def _upload_log(
        self, timeout: float, tag: str, guard: Optional[Callable[[], bool]] = None
    ) -> None:
        path = self.paths.attempt_log(self.attempt)
        if os.path.exists(path):
            self.rclone.copyto_up(
                path,
                layout.log_key(self.job_id, self.attempt),
                timeout=timeout,
                tag=tag,
                guard=guard,
            )

    def _note_sync_failure(self, what: str, error: str) -> None:
        """周期同步或心跳失败：训练继续，事件限速（第一次立即写，之后按间隔汇总）。"""
        self._sync_fail_count += 1
        now = self.clock.monotonic()
        interval = self.tuning["sync_failed_event_interval_s"]
        if self._sync_fail_last_event is not None and now - self._sync_fail_last_event < interval:
            return
        self._sync_fail_last_event = now
        self._note("同步 %s 失败，训练继续：%s" % (what, error))
        if self.events is not None:
            self.events.emit(
                states.EV_SYNC_FAILED,
                "同步 %s 失败，训练继续，恢复后补传" % what,
                {"what": what, "error": error[:500], "failures": self._sync_fail_count},
            )

    def _stop_sync_thread(self) -> None:
        self._sync_stop.set()
        if self._sync_thread is not None and self._sync_thread.is_alive():
            if self.rclone is not None:
                self.rclone.abort("sync")
            self._sync_thread.join(self.tuning["thread_join_s"])
            if self._sync_thread.is_alive():
                diag(COMPONENT, "periodic sync thread did not stop in time")

    def _stop_heartbeat_thread(self) -> None:
        self._hb_stop.set()
        if self._hb_thread is not None and self._hb_thread.is_alive():
            if self.rclone is not None:
                self.rclone.abort("heartbeat")
                self.rclone.abort("attempt")
            self._hb_thread.join(self.tuning["thread_join_s"])

    def _final_sync(self) -> Tuple[bool, List[str], int, str]:
        """最终同步：output/ 和 ckpt/ 共用 final_sync_timeout_min，失败就重试到时限。

        返回 (成功, 没传上去的文件, 没传上去的总数, 最后一次的错误)。
        """
        limit = max(0.0, self.settings["final_sync_timeout_min"] * 60.0)
        end = self.clock.monotonic() + limit
        tries = 0
        last = ""
        while True:
            tries += 1
            remaining = end - self.clock.monotonic()
            errors = self._sync_targets(remaining, "final", include_log=False, end=end)
            failed = {key: value for key, value in errors.items() if key in ("output", "ckpt")}
            if not failed and not self.fenced.is_set():
                if tries > 1:
                    self._note("最终同步在第 %d 次尝试成功" % tries)
                missing, count = self._not_uploaded()
                return not missing, missing, count, "files changed during the final sync"
            last = "; ".join("%s: %s" % item for item in sorted(failed.items()))
            diag(COMPONENT, "final sync attempt %d failed: %s" % (tries, last))
            remaining = end - self.clock.monotonic()
            if remaining <= 0 or self.fenced.is_set():
                break
            self.clock.sleep(min(remaining, self.tuning["final_sync_backoff_s"]))
        missing, count = self._not_uploaded()
        return False, missing, count, last

    def _not_uploaded(self) -> Tuple[List[str], int]:
        return not_uploaded(self.paths, self.job_id, int(self.tuning["max_not_uploaded"]))

    def _local_results_exist(self) -> bool:
        for _name, root, _prefix, skip_tmp in target_dirs(self.paths, self.job_id):
            try:
                if syncstate.snapshot(root, skip_tmp):
                    return True
            except syncstate.ListError:
                return True
        return False

    def _needs_final_sync(self) -> bool:
        """训练开始过、上次运行的结论还没同步完，或者本地有结果文件（容器重启后已经超时或
        被取消时，内存里的标志都没了，看磁盘）。"""
        return (
            self.training_started or self.previous_final is not None or self._local_results_exist()
        )

    # ================================================================ 收尾

    def _teardown(self) -> int:
        try:
            self._teardown_storage()
        except BaseException as exc:  # 收尾出错也必须自毁
            diag(COMPONENT, "teardown error: %s\n%s" % (exc_text(exc), traceback.format_exc()))
        finally:
            self._self_destroy()
        outcome = self.outcome
        if outcome is None or outcome.fenced or self.fenced.is_set():
            return states.EXIT_FAILED
        if outcome.phase == states.SUCCEEDED:
            return states.EXIT_SUCCEEDED
        return states.EXIT_CODE_BY_PHASE.get(outcome.phase, states.EXIT_FAILED)

    def _stop_child(self) -> None:
        child = self.child
        if child is not None and child.poll() is None:
            procs.terminate(
                child.pid, child.pid, self.settings["term_grace_s"], self.clock, reap=child.poll
            )
            procs.wait_bounded(child, KILL_WAIT_S, close_pipes=False)
        if self.sampler is not None:
            self.sampler.stop()

    def _teardown_storage(self) -> None:
        if self.outcome is None:
            self.outcome = Outcome(
                states.FAILED, states.REASON_RUNNER_CRASHED, message="runner 没有得出结论就结束了"
            )
        self.step = "final_sync"
        self._stop_child()
        self._stop_sync_thread()
        if self.rclone is None:
            # 启动文档无效：没有存储可写，只能自毁
            self._write_final_marker(synced=True, status_written=False)
            return
        prev = self.previous_final
        if prev is not None:
            if os.path.exists(self.paths.fenced):
                self.fenced.set()
            if prev.get("status_written") or self.fenced.is_set():
                diag(COMPONENT, "status already written (or fenced); only self-destroy is left")
                self._stop_heartbeat_thread()
                self.events.stop_flusher(self.tuning["thread_join_s"])
                return
        else:
            # 结论一确定就落盘：之后容器重启只接着收尾，不重跑 setup 和训练
            self._write_final_marker(synced=False, status_written=False)
        if self.sink is not None:
            self.sink.flush_tail()
        outcome = self.outcome
        synced = prev is not None and prev.get("synced", True)
        if not synced and not self.fenced.is_set() and self._needs_final_sync():
            # 最终同步之前确认本 attempt 仍是当前的（心跳线程最后一次读取可能已经过去一会儿）
            self._check_attempt(timeout=short_op_s(self.tuning), tag="final-attempt")
        if not synced and not self.fenced.is_set() and self._needs_final_sync():
            self.phase = states.UPLOADING
            try:
                self._put_status(
                    Outcome(states.UPLOADING, message="正在上传结果"),
                    timeout=short_op_s(self.tuning),
                )
            except RcloneError as exc:
                diag(COMPONENT, "cannot write UPLOADING status: %s" % exc)
            self._note("最终同步 output/ 和 ckpt/")
            ok, missing, count, error = self._final_sync()
            if not ok and not self.fenced.is_set():
                self._note("最终同步失败，%d 个文件没有传上去：%s" % (count, error))
                self.events.emit(
                    states.EV_FINAL_SYNC_FAILED,
                    "最终同步在 %g 分钟内没有成功，%d 个文件没有传上去"
                    % (self.settings["final_sync_timeout_min"], count),
                    {"not_uploaded": missing[:100], "count": count, "error": error[:500]},
                )
                outcome.apply_not_uploaded(missing, count)
        if self.fenced.is_set() and not outcome.fenced:
            outcome.fenced = True
        self.outcome = outcome
        # 本地先记下结论：之后 runner 死掉，watchdog 照这个结论补写状态
        self._write_final_marker(synced=True, status_written=False)
        fenced = self.fenced.is_set()
        if fenced:
            self._note("本 attempt 已被更新的 attempt 取代，不写最终状态")
        else:
            self._note(
                "最终状态 %s%s"
                % (outcome.phase, "（%s）" % outcome.reason if outcome.reason else "")
            )
        # 收尾预算分两半：前一半给日志和事件，后一半留给最终状态，日志卡住也不会挤掉状态。
        # 最终状态是自毁前最后一次写存储：CLI 看到终态就会销毁实例，日志和事件必须在它之前传完；
        # 心跳线程一直运行到这里（继续读 attempt.json），写最终状态之前才停。
        half = self.tuning["teardown_budget_s"] / 2.0
        self._budget_end = self.clock.monotonic() + half
        self._upload_final_logs()
        self.events.stop_flusher(self.tuning["thread_join_s"])
        self.events.flush(block=True, timeout=self._budget_timeout())
        self._stop_heartbeat_thread()
        if fenced:
            return
        self._budget_end = self.clock.monotonic() + half
        if not self._attempt_fresh():
            self._check_attempt(timeout=self._budget_timeout(), tag="final-attempt")
        if self.fenced.is_set():
            outcome.fenced = True
            self._note("写最终状态前发现更新的 attempt，不写最终状态")
            self._write_final_marker(synced=True, status_written=False)
            return
        written = self._write_final_status(outcome)
        self._write_final_marker(synced=True, status_written=written)
        if not written:
            diag(COMPONENT, "final status %s was not written" % outcome.phase)

    def _budget_left(self) -> float:
        return self._budget_end - self.clock.monotonic()

    def _budget_timeout(self) -> float:
        """收尾阶段单次存储操作的时限：不超过剩余预算，至少 1 秒。"""
        return max(1.0, min(self.tuning["small_op_timeout_s"], self._budget_left()))

    def _write_final_status(self, outcome: Outcome) -> bool:
        attempts = max(1, int(self.tuning["status_attempts"]))
        for index in range(attempts):
            try:
                self._put_status(outcome, timeout=self._budget_timeout())
                return True
            except RcloneError as exc:
                if exc.blocked:
                    return False
                diag(
                    COMPONENT, "final status write failed (%d/%d): %s" % (index + 1, attempts, exc)
                )
            wait = min(self._backoff(index), self._budget_left())
            if index + 1 >= attempts or wait <= 0:
                break
            self.clock.sleep(wait)
        return False

    def _upload_final_logs(self) -> None:
        """传 attempt 日志（重试几次）和诊断日志的尾部，都在前一半预算之内。"""
        if self.sink is not None:
            attempts = 3
            last = None
            for index in range(attempts):
                try:
                    self._upload_log(self._budget_timeout(), "final")
                    last = None
                    break
                except RcloneError as exc:
                    last = exc
                wait = min(self._backoff(index), self._budget_left())
                if index + 1 >= attempts or wait <= 0:
                    break
                self.clock.sleep(wait)
            if last is not None:
                self.events.emit(
                    states.EV_SYNC_FAILED,
                    "最终日志没有传上去",
                    {"what": "log", "error": str(last)[:500], "final": True},
                    flush=False,
                )
        if self._budget_left() <= 0:
            diag(COMPONENT, "no budget left for the diagnostic logs")
            return
        errors = upload_diag_logs(
            self.rclone,
            self.paths,
            self.job_id,
            self.attempt,
            int(self.tuning["diag_log_tail_kb"] * 1024),
            self._budget_timeout,
            tag="final",
        )
        if errors:
            self.events.emit(
                states.EV_SYNC_FAILED,
                "诊断日志没有全部传上去",
                {"what": "diag_logs", "error": "; ".join(errors)[:500], "final": True},
                flush=False,
            )

    def _write_final_marker(self, synced: bool, status_written: bool) -> None:
        doc = self.outcome.to_dict() if self.outcome is not None else {}
        doc.update({"attempt": self.attempt, "synced": synced, "status_written": status_written})
        try:
            write_json_atomic(self.paths.final, doc)
        except OSError as exc:
            diag(COMPONENT, "cannot write final marker: %s" % exc)

    def _read_final_marker(self) -> Optional[Dict[str, Any]]:
        doc = read_json(self.paths.final)
        if doc is None or doc.get("attempt") != self.attempt or not doc.get("phase"):
            return None
        return doc

    def _destroyer_for(self) -> SelfDestroyer:
        if self._destroyer is not None:
            return self._destroyer
        return SelfDestroyer(
            self.backend or guess_backend(self.environ),
            attempts=self.tuning["destroy_attempts"],
            interval_s=self.tuning["destroy_interval_s"],
            api_base=self.tuning["vast_api_base"],
            timeout_s=self.tuning["destroy_http_timeout_s"],
            component=COMPONENT,
        )

    def _self_destroy(self) -> None:
        self.step = "destroy"
        if self.events is not None:
            self.events.stop_flusher(self.tuning["thread_join_s"])
        if self.sink is not None:
            self.sink.close()
        try:
            result = self._destroyer_for().destroy()
        except Exception as exc:
            diag(COMPONENT, "self-destroy raised: %s" % exc_text(exc))
            result = DestroyResult(False, 0, None, exc_text(exc))
        record = result.to_dict()
        record.update({"attempt": self.attempt, "by": states.SRC_RUNNER})
        try:
            write_json_atomic(self.paths.destroyed, record)
        except OSError as exc:
            diag(COMPONENT, "cannot write destroyed marker: %s" % exc)
        if not result.ok:
            diag(COMPONENT, "self-destroy failed: %s" % result.error)
            if self.events is not None:
                self.events.emit(
                    states.EV_SELF_DESTROY_FAILED,
                    "自毁失败（尝试 %d 次），实例需要由 dgpu gc 清理" % result.attempts,
                    {"attempts": result.attempts, "error": (result.error or "")[:500]},
                    flush=False,
                )
                flush_spool(
                    self.rclone,
                    self.job_id,
                    self.events.spool_dir,
                    timeout=self.tuning["small_op_timeout_s"],
                )

    # ================================================================ 杂项

    def _note(self, text: str) -> None:
        diag(COMPONENT, text)
        if self.sink is not None:
            self.sink.note(text)


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("ok\n")


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _hold_fds(spec: Optional[str]) -> None:
    """boot 传下来的锁的文件描述符：保持打开，但不再让孙进程继承。"""
    for item in (spec or "").split(","):
        if item.strip().isdigit():
            try:
                os.set_inheritable(int(item), False)
            except OSError:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dgpu.runner.main")
    parser.add_argument(
        "--work-root", default=os.environ.get(WORK_ROOT_ENV) or states.DEFAULT_WORK_ROOT
    )
    parser.add_argument("--launch", default=None)
    parser.add_argument("--hold-fds", default=None)
    args = parser.parse_args(argv)
    _hold_fds(args.hold_fds)
    runner = Runner(Paths(args.work_root), launch_path=args.launch)

    def on_signal(signum: int, _frame: Any) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        runner.request_stop(name)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    code = runner.run()
    diag(COMPONENT, "exit code %d" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
