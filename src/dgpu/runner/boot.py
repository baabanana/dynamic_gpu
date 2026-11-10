"""boot：bootstrap 下载、校验、解压 bundle 之后 exec 的 Python 入口。

    python3 -m dgpu.runner.boot --work-root <root> [--lock-fd N] [--mode auto|detach|wait]

- 幂等：持有 <root>/boot.lock 的 flock 才继续。bootstrap 已经拿到锁时用 --lock-fd 传进来；
  锁被别人持有（同一个启动文档的 runner 还活着）就什么都不做，返回 0。锁随后交给 watchdog
  和 runner 持有，直到它们都退出。
- 先启动 watchdog，再启动 runner，两者都在新会话里运行，输出写到 logs/watchdog.log、
  logs/runner.log。
- Vast（detach）：启动后立即返回，onstart 结束不影响它们。
- AutoDL（wait）：弹性部署的容器只在 cmd 运行期间存在，所以留在前台，等 runner 和
  watchdog 都退出后才返回（返回 runner 的退出码）。收到 SIGTERM/SIGINT 转发给 runner。
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from typing import Any, List, Optional

from dgpu import states
from dgpu.runner.util import (
    WORK_ROOT_ENV,
    Paths,
    diag,
    guess_backend,
    load_launch,
    load_tuning,
    read_json,
    write_json_atomic,
)

COMPONENT = "boot"


def acquire_lock(path: str) -> Optional[int]:
    """非阻塞地拿 flock；被别人持有返回 None。"""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def package_root() -> str:
    """bundle 解压目录（含 dgpu/ 的那一层），作为子进程的工作目录和 sys.path 入口。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spawn(
    python: str, module: str, paths: Paths, log_name: str, fds: List[int], launch: str
) -> subprocess.Popen:
    out = open(os.path.join(paths.logs, log_name), "ab")
    try:
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
        lock_fd = args.lock_fd  # bootstrap 已经锁住
    else:
        lock_fd = acquire_lock(paths.boot_lock)
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

    alive_fd = acquire_lock(paths.runner_alive)
    if alive_fd is None:
        diag(COMPONENT, "a runner process still holds runner.alive; nothing to do")
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
        watchdog = _spawn(
            python, "dgpu.runner.watchdog", paths, "watchdog.log", [lock_fd], launch_path
        )
        _write_pid(paths.watchdog_pid, watchdog.pid)
    except OSError as exc:
        # 没有 watchdog 也要启动 runner：自毁的第一层在 runner 里
        diag(COMPONENT, "cannot start the watchdog: %s" % exc)
    try:
        runner = _spawn(
            python, "dgpu.runner.main", paths, "runner.log", [lock_fd, alive_fd], launch_path
        )
    except OSError as exc:
        # watchdog 会看到 runner.alive 被释放，按 runner 崩溃处理并自毁
        diag(COMPONENT, "cannot start the runner: %s" % exc)
        os.close(alive_fd)
        os.close(lock_fd)
        return 1
    _write_pid(paths.runner_pid, runner.pid)
    # 锁只由 runner（runner.alive、boot.lock）和 watchdog（boot.lock）持有
    os.close(alive_fd)
    os.close(lock_fd)
    diag(
        COMPONENT,
        "started watchdog %s and runner %d (%s mode)"
        % (watchdog.pid if watchdog else "-", runner.pid, mode),
    )
    if mode == "detach":
        return 0
    return _wait(runner, watchdog, tuning["boot_watchdog_wait_s"])


def _wait(
    runner: subprocess.Popen, watchdog: Optional[subprocess.Popen], watchdog_wait_s: float
) -> int:
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
        if runner_rc is not None and watchdog_rc is not None:
            break
        if runner_rc is not None and runner_done_at is None:
            runner_done_at = time.monotonic()
        if (
            runner_done_at is not None
            and watchdog_rc is None
            and time.monotonic() - runner_done_at > watchdog_wait_s
        ):
            diag(COMPONENT, "watchdog did not exit in time; killing it")
            watchdog.kill()
            watchdog.wait()
            break
        time.sleep(0.2)
    code = runner.returncode if runner.returncode is not None else states.EXIT_FAILED
    if code < 0:
        code = states.EXIT_FAILED
    diag(COMPONENT, "runner exited with %d; watchdog exited" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
