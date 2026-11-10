"""进程组：启动训练命令、判断进程存活、按 SIGTERM → 宽限 → SIGKILL 结束整棵进程树。

训练命令在独立的会话（进程组）里运行，组号等于子进程 pid。结束时既向整个进程组发信号，
也向发信号前快照到的所有后代发信号，覆盖自己 setsid 脱离进程组的孙进程。
Linux 上用 /proc，其他系统（测试用的 macOS）退回到 ps。

SIGKILL 之后的等待都有上限（KILL_WAIT_S）：进程卡在不可中断的内核调用里（D 状态，
比如驱动出问题时的 nvidia-smi）时 SIGKILL 不起作用，无限等待会让调用方一起卡死。
等不到就放弃这个进程（记下来，之后顺手回收），调用方按失败处理。
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from dgpu.runner.util import KILL_WAIT_S, Clock

# Linux 的进程表；测试替换成伪造的目录树
PROC_ROOT = "/proc"

# 放弃等待的子进程（SIGKILL 之后仍未退出）。保留引用，之后再回收，避免僵尸进程
_ABANDONED = []  # type: List[subprocess.Popen]
_ABANDONED_LOCK = threading.Lock()


def _has_proc() -> bool:
    return os.path.isdir(os.path.join(PROC_ROOT, "self"))


def start_session(
    argv: List[str],
    cwd: str,
    env: Dict[str, str],
    preexec_fn: Optional[Callable[[], None]] = None,
) -> subprocess.Popen:
    """在新会话里启动命令，stdout 和 stderr 合并到一个管道（由调用方持续读取）。

    preexec_fn：在子进程里 exec 之前执行（降权，见 privsep.py）。
    """
    return subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        preexec_fn=preexec_fn,
    )


# ---------------------------------------------------------------- 有上限的子进程调用


class Completed:
    """run_bounded 的结果。rc 为 None 表示进程被放弃（SIGKILL 之后仍没有退出）。"""

    __slots__ = ("rc", "out", "err", "timed_out", "abandoned")

    def __init__(self, rc: Optional[int], out: bytes, err: bytes, timed_out: bool, abandoned: bool):
        self.rc = rc
        self.out = out
        self.err = err
        self.timed_out = timed_out
        self.abandoned = abandoned


def kill_group_of(proc: subprocess.Popen) -> None:
    """向进程组（组号 = pid，start_new_session）发 SIGKILL，失败时只杀进程本身。"""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


def abandon(proc: subprocess.Popen, close_pipes: bool = True) -> None:
    """不再等这个进程：关掉管道（close_pipes），留着引用以便之后回收。"""
    for stream in (proc.stdin, proc.stdout, proc.stderr) if close_pipes else ():
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    with _ABANDONED_LOCK:
        _ABANDONED.append(proc)


def reap_abandoned() -> int:
    """回收已经退出的被放弃进程，返回仍未退出的数量。"""
    with _ABANDONED_LOCK:
        alive = []
        for proc in _ABANDONED:
            try:
                if proc.poll() is None:
                    alive.append(proc)
            except OSError:
                pass
        _ABANDONED[:] = alive
        return len(alive)


def finish(
    proc: subprocess.Popen,
    kill_wait: float = KILL_WAIT_S,
    kill: Optional[Callable[[subprocess.Popen], None]] = None,
    close_pipes: bool = True,
) -> Optional[int]:
    """杀掉进程组并最多等 kill_wait 秒；等不到就放弃，返回 None。"""
    if proc.poll() is not None:
        return proc.returncode
    (kill or kill_group_of)(proc)
    return wait_bounded(proc, kill_wait, close_pipes)


def wait_bounded(
    proc: subprocess.Popen, timeout: float = KILL_WAIT_S, close_pipes: bool = True
) -> Optional[int]:
    """等进程退出，最多 timeout 秒；等不到就放弃它，返回 None。"""
    try:
        return proc.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        abandon(proc, close_pipes)
        return None


def run_bounded(
    argv: List[str],
    timeout: float,
    kill_wait: float = KILL_WAIT_S,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    preexec_fn: Optional[Callable[[], None]] = None,
    kill: Optional[Callable[[subprocess.Popen], None]] = None,
) -> Completed:
    """运行命令并收集输出，总时长不超过 timeout + kill_wait。

    超时后 SIGKILL 整个进程组，再最多等 kill_wait 秒；进程仍不退出（D 状态）就放弃它，
    结果的 rc 为 None。启动失败抛 OSError（以及 preexec_fn 出错时的 SubprocessError）。
    kill：测试注入，模拟 SIGKILL 不起作用。
    """
    reap_abandoned()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
        preexec_fn=preexec_fn,
    )
    try:
        out, err = proc.communicate(timeout=max(0.0, timeout))
        return Completed(proc.returncode, out or b"", err or b"", False, False)
    except subprocess.TimeoutExpired:
        pass
    (kill or kill_group_of)(proc)
    try:
        out, err = proc.communicate(timeout=max(0.0, kill_wait))
        return Completed(proc.returncode, out or b"", err or b"", True, False)
    except subprocess.TimeoutExpired:
        abandon(proc)
        return Completed(None, b"", b"", True, True)


# ---------------------------------------------------------------- 进程表


def _proc_state(pid: int) -> Optional[str]:
    """Linux：/proc/<pid>/stat 的状态字母；读不到返回 None。"""
    try:
        with open(os.path.join(PROC_ROOT, str(pid), "stat"), "rb") as handle:
            data = handle.read().decode("ascii", "replace")
    except OSError:
        return None
    try:
        return data.rsplit(")", 1)[1].split()[0]
    except IndexError:
        return None


def _ps(args: List[str], timeout: float) -> Optional[bytes]:
    try:
        result = run_bounded(["ps"] + args, timeout=timeout, kill_wait=2.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.rc is None:
        return None
    return result.out


def _ps_state(pid: int) -> Optional[str]:
    """ps 报告的状态字母；进程不存在返回 ""，ps 不可用返回 None。"""
    out = _ps(["-o", "stat=", "-p", str(pid)], 5.0)
    if out is None:
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
    if _has_proc():
        state = _proc_state(pid)
        if state is None:
            return False
        return state not in ("Z", "X")
    state = _ps_state(pid)
    if state is None:
        return True  # 无法确认时按存活处理
    return state not in ("", "Z")


def cmdline(pid: int) -> Optional[str]:
    """进程的命令行（参数用空格连接）；读不到返回 None。用来在杀进程前确认 pid 没被复用。"""
    if pid <= 0:
        return None
    if _has_proc():
        try:
            with open(os.path.join(PROC_ROOT, str(pid), "cmdline"), "rb") as handle:
                data = handle.read()
        except OSError:
            return None
        return data.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    out = _ps(["-ww", "-o", "command=", "-p", str(pid)], 5.0)
    if out is None:
        return None
    return out.decode("utf-8", "replace").strip() or None


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
    return _has_proc() or bool(_ps_table())


def list_processes() -> List[Tuple[int, int, int]]:
    """所有进程的 (pid, ppid, pgid)。"""
    if _has_proc():
        rows = []
        try:
            names = os.listdir(PROC_ROOT)
        except OSError:
            return []
        for name in names:
            if not name.isdigit():
                continue
            try:
                with open(os.path.join(PROC_ROOT, name, "stat"), "rb") as handle:
                    data = handle.read().decode("ascii", "replace")
                # 第二个字段是带括号的进程名，里面可能有空格和括号：从最后一个 ")" 之后开始数
                fields = data.rsplit(")", 1)[1].split()
                rows.append((int(name), int(fields[1]), int(fields[2])))
            except (OSError, IndexError, ValueError):
                continue
        return rows
    return _ps_table()


def _ps_table() -> List[Tuple[int, int, int]]:
    out = _ps(["-A", "-o", "pid=,ppid=,pgid="], 10.0)
    if out is None:
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
    kill_wait: float = KILL_WAIT_S,
) -> str:
    """结束进程组 pgid（组长 leader）及其全部后代。

    先 SIGTERM，宽限 grace_s 秒后对还活着的发 SIGKILL，再最多等 kill_wait 秒。reap 用来
    回收自己的子进程（否则组长会作为僵尸一直“存在”）。返回 "exited"（本来就没有进程）、
    "terminated"（宽限内退出）、"killed"，或 "stuck"（SIGKILL 之后仍有进程，已放弃等待）。
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
    end = clock.monotonic() + max(0.0, kill_wait)
    while clock.monotonic() < end:
        if not alive():
            return "killed"
        clock.sleep(poll_s)
    return "stuck" if alive() else "killed"


def kill_pid(pid: int, sig: int = signal.SIGKILL) -> None:
    if pid > 0 and pid != os.getpid():
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------- 登记的子进程（rclone）


def register(directory: str, pid: int, tag: str) -> None:
    """在 directory 下登记一个子进程（文件名是 pid，内容是用来核对命令行的标记）。"""
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        path = os.path.join(directory, str(pid))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(tag)
    except OSError:
        pass


def unregister(directory: str, pid: int) -> None:
    try:
        os.unlink(os.path.join(directory, str(pid)))
    except OSError:
        pass


def kill_registered(directory: str) -> List[int]:
    """杀掉 directory 下登记的、仍在运行的子进程（连同进程组），返回杀掉的 pid。

    登记的进程已经退出、pid 被别的进程复用时不杀：只有命令行里包含登记时写下的标记
    （rclone 可执行文件的路径）才算同一个进程。
    """
    killed = []
    try:
        names = os.listdir(directory)
    except OSError:
        return killed
    for name in names:
        path = os.path.join(directory, name)
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(path, encoding="utf-8") as handle:
                tag = handle.read().strip()
        except OSError:
            tag = ""
        if pid_alive(pid):
            line = cmdline(pid)
            if tag and line is not None and tag in line:
                _send(pid, [pid], signal.SIGKILL)
                killed.append(pid)
        try:
            os.unlink(path)
        except OSError:
            pass
    return killed
