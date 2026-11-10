"""进程组：启动训练命令、判断进程存活、按 SIGTERM → 宽限 → SIGKILL 结束整棵进程树。

训练命令在独立的会话（进程组）里运行，组号等于子进程 pid。结束时既向整个进程组发信号，
也向发信号前快照到的所有后代发信号，覆盖自己 setsid 脱离进程组的孙进程。
Linux 上用 /proc，其他系统（测试用的 macOS）退回到 ps。
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from dgpu.runner.util import Clock


def start_session(argv: List[str], cwd: str, env: Dict[str, str]) -> subprocess.Popen:
    """在新会话里启动命令，stdout 和 stderr 合并到一个管道（由调用方持续读取）。"""
    return subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )


def _proc_state(pid: int) -> Optional[str]:
    """Linux：/proc/<pid>/stat 的状态字母；读不到返回 None。"""
    try:
        with open("/proc/%d/stat" % pid, "rb") as handle:
            data = handle.read().decode("ascii", "replace")
    except OSError:
        return None
    try:
        return data.rsplit(")", 1)[1].split()[0]
    except IndexError:
        return None


def _ps_state(pid: int) -> Optional[str]:
    """ps 报告的状态字母；进程不存在返回 ""，ps 不可用返回 None。"""
    try:
        out = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            close_fds=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.decode("ascii", "replace").strip()
    return text[:1]


def pid_alive(pid: int) -> bool:
    """进程是否存在且不是僵尸。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if os.path.isdir("/proc/self"):
        state = _proc_state(pid)
        if state is None:
            return False
        return state not in ("Z", "X")
    state = _ps_state(pid)
    if state is None:
        return True  # 无法确认时按存活处理
    return state not in ("", "Z")


def group_alive(pgid: int) -> bool:
    """进程组里是否还有不是僵尸的进程。"""
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    members = [pid for pid, _ppid, group in list_processes() if group == pgid]
    if not members:
        # 列不出进程表时保守地认为还活着
        return not _process_table_available()
    return any(pid_alive(pid) for pid in members)


def _process_table_available() -> bool:
    return os.path.isdir("/proc/self") or bool(_ps_table())


def list_processes() -> List[Tuple[int, int, int]]:
    """所有进程的 (pid, ppid, pgid)。"""
    if os.path.isdir("/proc/self"):
        rows = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open("/proc/%s/stat" % name, "rb") as handle:
                    data = handle.read().decode("ascii", "replace")
                fields = data.rsplit(")", 1)[1].split()
                rows.append((int(name), int(fields[1]), int(fields[2])))
            except (OSError, IndexError, ValueError):
                continue
        return rows
    return _ps_table()


def _ps_table() -> List[Tuple[int, int, int]]:
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,pgid="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            close_fds=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.decode("ascii", "replace").splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return rows


def descendants(roots: Iterable[int], pgid: Optional[int] = None) -> Set[int]:
    """roots 的所有后代，加上进程组 pgid 的所有成员。"""
    table = list_processes()
    children = {}  # type: Dict[int, List[int]]
    for pid, ppid, _group in table:
        children.setdefault(ppid, []).append(pid)
    found = set()  # type: Set[int]
    stack = [root for root in roots if root > 0]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    if pgid is not None and pgid > 0:
        found.update(pid for pid, _ppid, group in table if group == pgid)
    found.discard(os.getpid())
    return found


def _send(pgid: int, pids: Iterable[int], sig: int) -> None:
    if pgid > 0 and pgid != os.getpgrp():  # 永远不向自己所在的进程组发信号
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in pids:
        if pid <= 0 or pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate(
    pgid: int,
    leader: int,
    grace_s: float,
    clock: Optional[Clock] = None,
    reap: Optional[Callable[[], object]] = None,
    poll_s: float = 0.05,
) -> str:
    """结束进程组 pgid（组长 leader）及其全部后代。

    先 SIGTERM，宽限 grace_s 秒后对还活着的发 SIGKILL。reap 用来回收自己的子进程
    （否则组长会作为僵尸一直“存在”）。返回 "exited"（本来就没有进程）、"terminated"
    （宽限内退出）或 "killed"。
    """
    clock = clock or Clock()
    targets = descendants([leader], pgid) | {leader}

    def alive() -> bool:
        if reap is not None:
            reap()
        if group_alive(pgid):
            return True
        return any(pid_alive(pid) for pid in targets)

    if not alive():
        return "exited"
    _send(pgid, targets, signal.SIGTERM)
    end = clock.monotonic() + max(0.0, grace_s)
    while clock.monotonic() < end:
        if not alive():
            return "terminated"
        clock.sleep(poll_s)
    if not alive():
        return "terminated"
    targets |= descendants(list(targets), pgid)
    _send(pgid, targets, signal.SIGKILL)
    end = clock.monotonic() + 5.0
    while clock.monotonic() < end and alive():
        clock.sleep(poll_s)
    return "killed"


def kill_pid(pid: int, sig: int = signal.SIGKILL) -> None:
    if pid > 0 and pid != os.getpid():
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
