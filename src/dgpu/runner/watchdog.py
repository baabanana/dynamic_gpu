"""watchdog：runner 旁边的独立进程（设计 4.8 第 2 层）。

    python3 -m dgpu.runner.watchdog --work-root <root> [--launch <root>/launch.json]

1. runner 进程消失而没有走完收尾：结束残留的训练进程和 runner 启动的 rclone，按 runner 留下
   的结论（run/final.json）接着收尾；没有结论就是崩溃，写 FAILED(runner_crashed)。最终同步
   没做完时尽力同步一次，按同步记录算出没传上去的文件。
2. 到达硬截止时间（max_hours + runner 最坏的收尾时间 + 余量，见
   util.watchdog_hard_deadline_s；从第一次启动算起，用单调时钟计时）：不管 runner 处于
   什么状态，结束所有进程，同样尽力同步一次、算出没传上去的文件，写 TIMED_OUT（或 runner
   已经得出的结论），自毁。

两条路径上都先确认没有更新的 attempt（本地标记和存储里的 attempt.json），被取代时不写
状态。runner 是否存活看 run/runner.alive 上的 flock：boot 在启动 runner 之前锁住它，只有
runner 进程持有这个文件描述符，进程一退出（包括被 SIGKILL、变成僵尸）内核就释放锁；文件
系统不支持 flock 时退回 pid 文件。

watchdog 的每个存储操作都有时限：尽力同步有总时限（util.watchdog_sync_s），之后的日志、
事件、状态共用 teardown_budget_s（后一半留给状态）。整个接手过程的上限是
util.watchdog_teardown_worst_case_s。自毁在 finally 里，接手过程出任何错都会执行。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from dgpu import layout, states
from dgpu.runner import logsink, procs
from dgpu.runner.destroy import SelfDestroyer
from dgpu.runner.events import Events, flush_spool
from dgpu.runner.main import (
    Outcome,
    not_uploaded,
    sync_dir,
    target_dirs,
    upload_diag_logs,
    write_status_doc,
)
from dgpu.runner.main import default_rclone_binary as _default_rclone
from dgpu.runner.rclone import Rclone, RcloneError
from dgpu.runner.util import (
    WORK_ROOT_ENV,
    Clock,
    Paths,
    diag,
    exc_text,
    guess_backend,
    load_launch,
    load_tuning,
    read_json,
    runner_settings,
    short_op_s,
    watchdog_hard_deadline_s,
    watchdog_kill_grace_s,
    watchdog_sync_s,
    write_json_atomic,
)

COMPONENT = "watchdog"


class Watchdog:
    def __init__(
        self,
        paths: Paths,
        launch_path: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
        clock: Optional[Clock] = None,
        environ: Optional[Dict[str, str]] = None,
        rclone_binary: Optional[str] = None,
        destroyer: Optional[SelfDestroyer] = None,
    ):
        self.paths = paths
        self.launch_path = launch_path or paths.launch
        self.environ = dict(os.environ if environ is None else environ)
        self.tuning = tuning if tuning is not None else load_tuning(self.environ)
        self.clock = clock or Clock()
        self.rclone_binary = rclone_binary or _default_rclone(self.environ)
        self._destroyer = destroyer
        self.t_start = self.clock.monotonic()
        self.launch = None  # type: Optional[Dict[str, Any]]
        self.job_id = ""
        self.attempt = 0
        self.backend = guess_backend(self.environ)
        self.rclone = None  # type: Optional[Rclone]
        self.events = None  # type: Optional[Events]
        self.spec = None  # type: Optional[Dict[str, Any]]
        self.settings = runner_settings({})
        self.hard_deadline = None  # type: Optional[float]
        self._alive_fd = None  # type: Optional[int]
        self._flock_broken = False
        self._budget_end = 0.0
        self.finished = False

    # ------------------------------------------------------------ 入口

    def run(self) -> int:
        try:
            self._setup()
            self._loop()
        except BaseException as exc:  # watchdog 自己出错也要自毁
            diag(COMPONENT, "watchdog error: %s\n%s" % (exc_text(exc), traceback.format_exc()))
            if not self.finished:
                try:
                    self._kill_everything()
                finally:
                    self._self_destroy()
        return 0

    def _setup(self) -> None:
        self.paths.ensure()
        try:
            self.launch = load_launch(self.launch_path)
            self.job_id = str(self.launch["job_id"])
            self.attempt = int(self.launch["attempt"])
            self.backend = str(self.launch["backend"])
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
                states.SRC_WATCHDOG,
                self.paths.event_spool(states.SRC_WATCHDOG),
                self.paths.events_tmp,
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            diag(
                COMPONENT, "launch document unusable (%s); will only guard the runner process" % exc
            )
        self._alive_fd = os.open(self.paths.runner_alive, os.O_RDWR | os.O_CREAT, 0o600)

    def _loop(self) -> None:
        poll = max(0.01, self.tuning["watchdog_poll_s"])
        while True:
            if self._runner_gone():
                self._on_runner_exit()
                return
            if self.spec is None:
                self._load_spec()
            now = self.clock.monotonic()
            if self.spec is None and now - self.t_start > self.tuning["watchdog_spec_wait_s"]:
                self._on_hard_deadline(
                    "runner 在 %.0f 秒内没有读到 spec.json" % (now - self.t_start)
                )
                return
            if self.hard_deadline is not None and now >= self.hard_deadline:
                self._on_hard_deadline("到达硬截止时间（max_hours + runner 最坏收尾时间 + 余量）")
                return
            self.clock.sleep(poll)

    # ------------------------------------------------------------ 观察

    def _runner_gone(self) -> bool:
        if not self._flock_broken:
            try:
                fcntl.flock(self._alive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False  # 锁还被 runner 持有
            except OSError as exc:
                # 文件系统不支持 flock（ENOLCK、EINVAL 等）：只记一次，之后看 pid 文件
                self._flock_broken = True
                diag(COMPONENT, "flock failed (%s); falling back to the pid file" % exc)
            else:
                return True  # 拿到了锁：持有它的 runner 已经不在
        pid = _read_pid(self.paths.runner_pid)
        return pid is not None and not procs.pid_alive(pid)

    def _load_spec(self) -> None:
        spec = read_json(self.paths.spec_copy)
        if spec is None or spec.get("job_id") != self.job_id:
            return
        try:
            max_hours = float(spec["max_hours"])
        except (KeyError, TypeError, ValueError):
            return
        self.spec = spec
        self.settings = runner_settings(spec)
        budget = watchdog_hard_deadline_s(max_hours, self.settings, self.tuning)
        started = read_json(self.paths.started)
        if started and started.get("attempt") == self.attempt:
            try:
                budget -= max(0.0, self.clock.time() - float(started["wall"]))
            except (KeyError, TypeError, ValueError):
                pass
        else:
            budget -= self.clock.monotonic() - self.t_start
        self.hard_deadline = self.clock.monotonic() + budget
        diag(COMPONENT, "hard deadline in %.0f s" % budget)

    def _fenced(self) -> bool:
        doc = read_json(self.paths.fenced)
        return doc is not None and doc.get("attempt") == self.attempt

    def _final(self) -> Optional[Dict[str, Any]]:
        doc = read_json(self.paths.final)
        if doc is None or doc.get("attempt") != self.attempt or not doc.get("phase"):
            return None
        return doc

    def _destroyed(self) -> Optional[Dict[str, Any]]:
        doc = read_json(self.paths.destroyed)
        if doc is None or doc.get("attempt") != self.attempt:
            return None
        return doc

    # ------------------------------------------------------------ 处理

    def _on_runner_exit(self) -> None:
        self.finished = True
        destroyed = self._destroyed()
        if destroyed is not None:
            diag(COMPONENT, "runner finished its self-destroy step (ok=%s)" % destroyed.get("ok"))
            self._kill_leftovers()
            return
        try:  # 这里出什么错都要自毁
            self._kill_leftovers()
            final = self._final()
            if final is None:
                diag(COMPONENT, "runner exited without reaching an outcome; taking over")
                self._take_over(
                    Outcome(
                        states.FAILED,
                        states.REASON_RUNNER_CRASHED,
                        message="runner 进程意外退出，watchdog 已尽力同步一次",
                    ),
                    synced=False,
                )
            elif final.get("status_written") or final.get("fenced"):
                diag(COMPONENT, "runner exited after writing its final status, before self-destroy")
            else:
                # runner 得出结论之后、写最终状态之前退出：按它的结论接着收尾
                diag(COMPONENT, "runner exited after reaching its outcome; finishing it up")
                self._take_over(Outcome.from_dict(final), synced=bool(final.get("synced", True)))
        finally:
            self._self_destroy()

    def _on_hard_deadline(self, reason: str) -> None:
        self.finished = True
        diag(COMPONENT, "hard deadline: %s" % reason)
        try:  # 这里出什么错都要自毁
            self._kill_everything()
            if self.events is not None:
                self.events.emit(
                    states.EV_DEADLINE_HIT,
                    "watchdog：%s，结束所有进程并自毁" % reason,
                    {"by": states.SRC_WATCHDOG},
                    flush=False,
                )
            final = self._final()
            if final is None:
                self._take_over(
                    Outcome(
                        states.TIMED_OUT,
                        message="watchdog 硬截止：%s；runner 没有按时收尾" % reason,
                    ),
                    synced=False,
                )
            elif final.get("status_written") or final.get("fenced"):
                self._budget_end = self.clock.monotonic() + self.tuning["teardown_budget_s"]
                self._flush_events()
            else:
                self._take_over(Outcome.from_dict(final), synced=bool(final.get("synced", True)))
        finally:
            self._self_destroy()

    def _take_over(self, outcome: Outcome, synced: bool) -> None:
        """替 runner 收尾：确认没被取代 → 尽力同步一次 → 日志、事件 → 最终状态。"""
        fenced = self._fenced() or self._check_attempt_fenced()
        outcome.fenced = outcome.fenced or fenced
        written = False
        if self.rclone is not None and not fenced:
            if not synced:
                missing, count = self._best_effort_sync()
                outcome.apply_not_uploaded(missing, count)
            half = self.tuning["teardown_budget_s"] / 2.0
            self._budget_end = self.clock.monotonic() + half
            # 日志和事件先传，最终状态最后写（CLI 看到终态就会销毁实例）
            self._upload_logs()
            self._flush_events()
            self._budget_end = self.clock.monotonic() + half
            written = self._write_status(outcome)
        doc = outcome.to_dict()
        doc.update(
            {
                "attempt": self.attempt,
                "synced": True,
                "status_written": written,
                "by": states.SRC_WATCHDOG,
            }
        )
        self._save_final(doc)

    def _kill_training(self, grace: float) -> None:
        child = read_json(self.paths.child)
        if not child:
            return
        try:
            pgid = int(child.get("pgid") or 0)
            pid = int(child.get("pid") or 0)
        except (TypeError, ValueError):
            return
        if pgid > 0 and (procs.group_alive(pgid) or procs.pid_alive(pid)):
            result = procs.terminate(pgid, pid, grace, self.clock)
            diag(COMPONENT, "training process group %d: %s" % (pgid, result))

    def _kill_runner_rclones(self) -> None:
        """runner 被杀后它启动的 rclone 还活着：杀掉，免得它们在 watchdog 写状态之后才落盘。"""
        killed = procs.kill_registered(self.paths.rclone_pids("runner"))
        if killed:
            diag(COMPONENT, "killed rclone processes left by the runner: %s" % killed)

    def _kill_leftovers(self) -> None:
        self._kill_runner_rclones()
        self._kill_training(grace=watchdog_kill_grace_s(self.settings))

    def _kill_everything(self) -> None:
        pid = _read_pid(self.paths.runner_pid)
        if pid is not None and procs.pid_alive(pid):
            procs.kill_pid(pid, signal.SIGKILL)
            diag(COMPONENT, "killed runner %d" % pid)
        self._kill_leftovers()

    def _check_attempt_fenced(self) -> bool:
        if self.rclone is None:
            return False
        try:
            data = self.rclone.cat(
                layout.attempt_key(self.job_id), timeout=short_op_s(self.tuning), tag="attempt"
            )
        except RcloneError as exc:
            diag(COMPONENT, "cannot read attempt.json: %s" % exc)
            return False
        if not data:
            return False
        try:
            doc = json.loads(data.decode("utf-8"))
            states.check_version(doc, "attempt.json")
            newer = int(doc.get("attempt")) > self.attempt
        except (ValueError, TypeError, AttributeError):
            return False
        if newer:
            try:
                write_json_atomic(
                    self.paths.fenced,
                    {"attempt": self.attempt, "newer_attempt": doc.get("attempt")},
                )
            except OSError:
                pass
            if self.events is not None:
                self.events.emit(
                    states.EV_ATTEMPT_FENCED,
                    "watchdog：已有更新的 attempt %s，不写状态" % doc.get("attempt"),
                    {"newer_attempt": doc.get("attempt")},
                    flush=False,
                )
        return newer

    def _best_effort_sync(self) -> Tuple[List[str], int]:
        """一次有总时限的同步；返回按同步记录算出的没传上去的文件和总数。"""
        end = self.clock.monotonic() + watchdog_sync_s(self.settings, self.tuning)
        for name, root, prefix, skip_tmp in target_dirs(self.paths, self.job_id):
            limit = max(1.0, end - self.clock.monotonic())
            ok, error = sync_dir(
                self.rclone, self.paths, name, root, prefix, skip_tmp, limit, COMPONENT
            )
            if not ok:
                diag(COMPONENT, "sync of %s failed: %s" % (name, error))
        return not_uploaded(self.paths, self.job_id, int(self.tuning["max_not_uploaded"]))

    def _budget_timeout(self) -> float:
        left = self._budget_end - self.clock.monotonic()
        return max(1.0, min(self.tuning["small_op_timeout_s"], left))

    def _upload_logs(self) -> None:
        if self.rclone is None:
            return
        path = self.paths.attempt_log(self.attempt)
        if os.path.exists(path):
            # runner 死前截断的训练输出尾部还在磁盘上：先接到日志末尾
            logsink.append_tail(path, "\n[dgpu] 训练输出超过上限，下面是最后保留的部分：\n")
            try:
                with open(path, "ab") as handle:
                    handle.write(
                        ("[dgpu] watchdog 接手收尾（runner 没有走完收尾）\n").encode("utf-8")
                    )
            except OSError:
                pass
            try:
                self.rclone.copyto_up(
                    path,
                    layout.log_key(self.job_id, self.attempt),
                    timeout=self._budget_timeout(),
                    tag=COMPONENT,
                )
            except RcloneError as exc:
                diag(COMPONENT, "log upload failed: %s" % exc)
        if self._budget_end - self.clock.monotonic() <= 0:
            return
        errors = upload_diag_logs(
            self.rclone,
            self.paths,
            self.job_id,
            self.attempt,
            int(self.tuning["diag_log_tail_kb"] * 1024),
            self._budget_timeout,
            tag=COMPONENT,
        )
        for error in errors:
            diag(COMPONENT, "diagnostic log upload failed: %s" % error)

    def _write_status(self, outcome: Outcome) -> bool:
        if self.rclone is None:
            return False
        for index in range(2):
            if index and self._budget_end - self.clock.monotonic() <= 0:
                break
            try:
                write_status_doc(
                    self.rclone,
                    self.job_id,
                    self.attempt,
                    outcome,
                    states.SRC_WATCHDOG,
                    self._budget_timeout(),
                )
                return True
            except RcloneError as exc:
                diag(COMPONENT, "status write failed: %s" % exc)
        return False

    def _save_final(self, doc: Dict[str, Any]) -> None:
        try:
            write_json_atomic(self.paths.final, doc)
        except OSError as exc:
            diag(COMPONENT, "cannot write final marker: %s" % exc)

    def _flush_events(self) -> None:
        if self.rclone is None:
            return
        for source in (states.SRC_RUNNER, states.SRC_WATCHDOG):
            if self._budget_end - self.clock.monotonic() <= 0:
                diag(COMPONENT, "no budget left to upload %s events" % source)
                return
            flush_spool(
                self.rclone,
                self.job_id,
                self.paths.event_spool(source),
                timeout=self._budget_timeout(),
            )

    def _self_destroy(self) -> None:
        destroyer = self._destroyer or SelfDestroyer(
            self.backend,
            attempts=self.tuning["destroy_attempts"],
            interval_s=self.tuning["destroy_interval_s"],
            api_base=self.tuning["vast_api_base"],
            timeout_s=self.tuning["destroy_http_timeout_s"],
            component=COMPONENT,
        )
        try:
            result = destroyer.destroy()
        except Exception as exc:
            diag(COMPONENT, "self-destroy raised: %s" % exc_text(exc))
            return
        record = result.to_dict()
        record.update({"attempt": self.attempt, "by": states.SRC_WATCHDOG})
        try:
            write_json_atomic(self.paths.destroyed, record)
        except OSError:
            pass
        if not result.ok and self.events is not None:
            self.events.emit(
                states.EV_SELF_DESTROY_FAILED,
                "watchdog 自毁失败（尝试 %d 次），实例需要由 dgpu gc 清理" % result.attempts,
                {"attempts": result.attempts, "error": (result.error or "")[:500]},
                flush=False,
            )
            self._budget_end = self.clock.monotonic() + short_op_s(self.tuning)
            self._flush_events()


def _read_pid(path: str) -> Optional[int]:
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dgpu.runner.watchdog")
    parser.add_argument(
        "--work-root", default=os.environ.get(WORK_ROOT_ENV) or states.DEFAULT_WORK_ROOT
    )
    parser.add_argument("--launch", default=None)
    parser.add_argument("--hold-fds", default=None)
    args = parser.parse_args(argv)
    for item in (args.hold_fds or "").split(","):
        if item.strip().isdigit():
            try:
                os.set_inheritable(int(item), False)
            except OSError:
                pass
    # watchdog 是最后一道防线：不响应 SIGTERM / SIGINT / SIGHUP，只能被 SIGKILL 结束
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    return Watchdog(Paths(args.work_root), launch_path=args.launch).run()


if __name__ == "__main__":
    sys.exit(main())
