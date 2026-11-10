"""boot：bootstrap 下载、校验、解压 bundle 之后 exec 的 Python 入口。

    python3 -m dgpu.runner.boot --work-root <root> [--lock-fd N] [--mode auto|detach|wait]

- 幂等：持有 <root>/boot.lock 的 flock 才继续。bootstrap 已经拿到锁时用 --lock-fd 传进来；
  锁被别人持有（同一个启动文档的 runner 还活着）就什么都不做，返回 0。只有 EAGAIN
  （BlockingIOError）表示锁被占用；文件系统不支持 flock 时记一行，改看 run/runner.pid 里的
  进程是否还活着。锁随后交给 watchdog 和 runner 持有，直到它们都退出。
- 先启动 watchdog，再启动 runner，两者都在新会话里运行，输出写到 logs/watchdog.log、
  logs/runner.log。
- Vast（detach）：启动后立即返回 0，onstart 结束不影响它们。
- AutoDL（wait）：弹性部署的容器只在 cmd 运行期间存在，所以留在前台看着它们：
  * watchdog 意外退出（runner 还没自毁）就重启它，最多 boot_watchdog_respawns 次；
  * 自己也有硬截止（util.boot_hard_deadline_s，从第一次启动算起：watchdog 应该已经收尾完的
    时间再加余量），到了就结束 runner、watchdog、训练进程、rclone，然后退出——watchdog
    也失效、runner 又卡住时，容器照样会被释放；
  * 收到 SIGTERM/SIGINT 转发给 runner；
  * 退出码：最终状态已经写进存储（或本 attempt 已被更新的 attempt 取代）时为 0，不管任务
    结果如何（结果以 status.json 为准）；只有一个状态都没能记录时非 0。
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from typing import Any, List, Optional, Tuple

from dgpu import states
from dgpu.runner import procs
from dgpu.runner.util import (
    KILL_WAIT_S,
    WORK_ROOT_ENV,
    Paths,
    boot_hard_deadline_s,
    boot_spec_wait_s,
    diag,
    guess_backend,
    load_launch,
    load_tuning,
    read_json,
    runner_settings,
    write_json_atomic,
)

COMPONENT = "boot"
RUNNER_MODULE = "dgpu.runner.main"
WATCHDOG_MODULE = "dgpu.runner.watchdog"


def acquire_lock(path: str) -> Tuple[Optional[int], bool]:
    """非阻塞地拿 flock，返回 (文件描述符, 是否真的锁上)。

    被别人持有返回 (None, False)。文件系统不支持 flock（ENOLCK、EINVAL 等）时记一行，
    返回 (fd, False)：调用方要用别的办法确认没有别的 runner。
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, False
    except OSError as exc:
        diag(COMPONENT, "flock on %s failed (%s); continuing without the lock" % (path, exc))
        return fd, False
    return fd, True


def package_root() -> str:
    """bundle 解压目录（含 dgpu/ 的那一层），作为子进程的工作目录和 sys.path 入口。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spawn(
    python: str, module: str, paths: Paths, log_name: str, fds: List[int], launch: str
) -> subprocess.Popen:
    out = open(os.path.join(paths.logs, log_name), "ab")
    try:
        try:
            os.chmod(os.path.join(paths.logs, log_name), 0o600)
        except OSError:
            pass
        env = dict(os.environ)
        env[WORK_ROOT_ENV] = paths.root
        # 以 -m 启动时当前目录在 sys.path 最前面，所以工作目录设为 bundle 根目录即可，
        # 不改 PYTHONPATH（训练进程会继承 runner 的环境）
        root = package_root()
        argv = [
            python,
            "-m",
            module,
            "--work-root",
            paths.root,
            "--launch",
            launch,
            "--hold-fds",
            ",".join(str(fd) for fd in fds),
        ]
        return subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=tuple(fds),
        )
    finally:
        out.close()


def _write_pid(path: str, pid: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write("%d\n" % pid)
    os.replace(tmp, path)


def _read_pid(path: str) -> Optional[int]:
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def runner_running(paths: Paths) -> bool:
    """没有 flock 时的兜底：run/runner.pid 里的进程还活着，且确实是 runner。"""
    pid = _read_pid(paths.runner_pid)
    if pid is None or not procs.pid_alive(pid):
        return False
    line = procs.cmdline(pid)
    return line is None or RUNNER_MODULE in line


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dgpu.runner.boot")
    parser.add_argument(
        "--work-root", default=os.environ.get(WORK_ROOT_ENV) or states.DEFAULT_WORK_ROOT
    )
    parser.add_argument("--launch", default=None)
    parser.add_argument("--lock-fd", type=int, default=None)
    parser.add_argument("--mode", choices=("auto", "detach", "wait"), default="auto")
    args = parser.parse_args(argv)

    paths = Paths(args.work_root)
    paths.ensure()
    launch_path = os.path.abspath(args.launch or paths.launch)
    tuning = load_tuning()

    if args.lock_fd is not None:
        lock_fd = args.lock_fd  # bootstrap 已经锁住（或确认了文件系统不支持 flock）
    else:
        lock_fd, _locked = acquire_lock(paths.boot_lock)
        if lock_fd is None:
            diag(COMPONENT, "a runner for this work root is already active; nothing to do")
            return 0

    attempt = None
    try:
        launch = load_launch(launch_path)
        backend = str(launch["backend"])
        attempt = int(launch["attempt"])
    except (ValueError, TypeError, KeyError) as exc:
        backend = guess_backend()
        diag(COMPONENT, "launch document unusable (%s); assuming backend %s" % (exc, backend))

    mode = args.mode
    if mode == "auto":
        mode = "wait" if backend == "autodl" else "detach"

    started = read_json(paths.started)
    if attempt is not None and (started is None or started.get("attempt") != attempt):
        write_json_atomic(paths.started, {"attempt": attempt, "wall": time.time()})

    alive_fd, alive_locked = acquire_lock(paths.runner_alive)
    if alive_fd is None:
        diag(COMPONENT, "a runner process still holds runner.alive; nothing to do")
        return 0
    if not alive_locked and runner_running(paths):
        diag(COMPONENT, "flock unsupported and the runner in runner.pid is alive; nothing to do")
        os.close(alive_fd)
        return 0
    # 容器重启后这些文件来自上一次的进程：pid 可能已被别的进程复用，先删掉免得误杀；
    # 自毁记录也删掉，新一轮进程总要重新执行自毁（自毁是幂等的）
    for stale in (paths.child, paths.runner_pid, paths.watchdog_pid, paths.destroyed):
        try:
            os.unlink(stale)
        except OSError:
            pass

    python = sys.executable or "python3"
    watchdog = None
    try:
        watchdog = _spawn(python, WATCHDOG_MODULE, paths, "watchdog.log", [lock_fd], launch_path)
        _write_pid(paths.watchdog_pid, watchdog.pid)
    except OSError as exc:
        # 没有 watchdog 也要启动 runner：自毁的第一层在 runner 里
        diag(COMPONENT, "cannot start the watchdog: %s" % exc)
    try:
        runner = _spawn(
            python, RUNNER_MODULE, paths, "runner.log", [lock_fd, alive_fd], launch_path
        )
    except OSError as exc:
        # watchdog 会看到 runner.alive 被释放，按 runner 崩溃处理并自毁
        diag(COMPONENT, "cannot start the runner: %s" % exc)
        os.close(alive_fd)
        os.close(lock_fd)
        return 1
    _write_pid(paths.runner_pid, runner.pid)
    # runner.alive 只由 runner 持有（boot 自己留着的话 watchdog 永远看不到 runner 退出）
    os.close(alive_fd)
    diag(
        COMPONENT,
        "started watchdog %s and runner %d (%s mode)"
        % (watchdog.pid if watchdog else "-", runner.pid, mode),
    )
    if mode == "detach":
        os.close(lock_fd)
        return 0
    try:
        return Supervisor(paths, tuning, python, launch_path, lock_fd, attempt).run(
            runner, watchdog
        )
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


class Supervisor:
    """AutoDL 前台模式：看着 runner 和 watchdog，必要时重启 watchdog，到自己的硬截止就结束一切。"""

    def __init__(
        self,
        paths: Paths,
        tuning: dict,
        python: str,
        launch_path: str,
        lock_fd: int,
        attempt: Optional[int],
        clock: Any = time,
        poll_s: float = 0.2,
    ):
        self.paths = paths
        self.tuning = tuning
        self.python = python
        self.launch_path = launch_path
        self.lock_fd = lock_fd
        self.attempt = attempt
        self.clock = clock
        self.poll_s = poll_s
        self.respawns = 0
        self.deadline = None  # type: Optional[float]
        self.spec_wait_end = clock.monotonic() + boot_spec_wait_s(tuning)
        self.killed_at_deadline = False

    def run(self, runner: subprocess.Popen, watchdog: Optional[subprocess.Popen]) -> int:
        def forward(signum: int, _frame: Any) -> None:
            if runner.poll() is None:
                try:
                    os.kill(runner.pid, signum)
                except OSError:
                    pass

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        runner_done_at = None
        while True:
            runner_rc = runner.poll()
            watchdog_rc = watchdog.poll() if watchdog is not None else 0
            now = self.clock.monotonic()
            if self.deadline is None:
                self.deadline = self._compute_deadline()
            limit = self.deadline if self.deadline is not None else self.spec_wait_end
            if now >= limit:
                diag(COMPONENT, "boot hard deadline reached; killing everything")
                self._kill_all(runner, watchdog)
                self.killed_at_deadline = True
                break
            if watchdog_rc is not None and self._needs_watchdog(runner_rc):
                watchdog = self._respawn_watchdog(watchdog)
                watchdog_rc = watchdog.poll() if watchdog is not None else 0
            if runner_rc is not None and watchdog_rc is not None:
                break
            if runner_rc is not None and runner_done_at is None:
                runner_done_at = now
            if (
                runner_done_at is not None
                and watchdog is not None
                and watchdog_rc is None
                and now - runner_done_at > self.tuning["boot_watchdog_wait_s"]
            ):
                diag(COMPONENT, "watchdog did not exit in time; killing it")
                procs.finish(watchdog)
                break
            self.clock.sleep(self.poll_s)
        return self._exit_code(runner)

    def _needs_watchdog(self, runner_rc: Optional[int]) -> bool:
        """watchdog 不在了：runner 还活着，或者 runner 已经退出但没人走完自毁。"""
        if self.respawns >= int(self.tuning["boot_watchdog_respawns"]):
            return False
        destroyed = read_json(self.paths.destroyed)
        if destroyed is not None and destroyed.get("attempt") == self.attempt:
            return False
        return True

    def _respawn_watchdog(self, old: Optional[subprocess.Popen]) -> Optional[subprocess.Popen]:
        self.respawns += 1
        diag(
            COMPONENT,
            "watchdog exited (rc %s) before the job finished; restarting it (%d/%d)"
            % (
                old.returncode if old is not None else "-",
                self.respawns,
                int(self.tuning["boot_watchdog_respawns"]),
            ),
        )
        try:
            new = _spawn(
                self.python,
                WATCHDOG_MODULE,
                self.paths,
                "watchdog.log",
                [self.lock_fd],
                self.launch_path,
            )
        except OSError as exc:
            diag(COMPONENT, "cannot restart the watchdog: %s" % exc)
            return None
        _write_pid(self.paths.watchdog_pid, new.pid)
        return new

    def _compute_deadline(self) -> Optional[float]:
        """从 runner 写下的 spec 副本和第一次启动的时间算出 boot 的硬截止（单调时钟）。"""
        spec = read_json(self.paths.spec_copy)
        if spec is None:
            return None
        try:
            max_hours = float(spec["max_hours"])
        except (KeyError, TypeError, ValueError):
            return None
        total = boot_hard_deadline_s(max_hours, runner_settings(spec), self.tuning)
        started = read_json(self.paths.started)
        if started and started.get("attempt") == self.attempt:
            try:
                total -= max(0.0, time.time() - float(started["wall"]))
            except (KeyError, TypeError, ValueError):
                pass
        diag(COMPONENT, "boot hard deadline in %.0f s" % total)
        return self.clock.monotonic() + total

    def _kill_all(self, runner: subprocess.Popen, watchdog: Optional[subprocess.Popen]) -> None:
        for proc in (runner, watchdog):
            if proc is not None and proc.poll() is None:
                procs.kill_group_of(proc)
        child = read_json(self.paths.child) or {}
        try:
            pgid = int(child.get("pgid") or 0)
        except (TypeError, ValueError):
            pgid = 0
        if pgid > 0:
            procs.terminate(pgid, pgid, 0.0)
        for component in ("runner", "watchdog"):
            procs.kill_registered(self.paths.rclone_pids(component))
        for proc in (runner, watchdog):
            if proc is not None:
                procs.wait_bounded(proc, KILL_WAIT_S)

    def _exit_code(self, runner: subprocess.Popen) -> int:
        final = read_json(self.paths.final)
        recorded = (
            final is not None
            and self.attempt is not None
            and final.get("attempt") == self.attempt
            and bool(final.get("status_written") or final.get("fenced"))
        )
        code = 0 if recorded else states.EXIT_FAILED
        diag(
            COMPONENT,
            "runner exited with %s; final status %s; exit %d"
            % (runner.returncode, "recorded" if recorded else "NOT recorded", code),
        )
        return code


if __name__ == "__main__":
    sys.exit(main())
