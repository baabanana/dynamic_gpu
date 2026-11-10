"""训练进程降权（设计 4.7）。

runner 是 root 时，setup 和 run 默认以无特权用户运行：

- 用户：已有的 dgpu 用户；没有就用 useradd 创建；没有 useradd 或创建失败，就取一个没人用的
  固定高位 uid/gid（不需要 passwd 条目，HOME、USER、LOGNAME 由环境变量给出）。
- 目录：code/、data/、output/、ckpt/ 和 HOME（<工作目录>/home）归这个用户；run/、logs/、
  launch.json、boot.lock、bundle-*/ 只有 root 能读（0700 / 0600），工作目录本身 0711。
  训练进程读不到存储凭据（launch.json、root 进程的 /proc/<pid>/environ），也不能向 runner、
  watchdog、rclone 发信号（内核不允许非 root 进程给别的用户的进程发信号）。runner 仍以 root
  读 output/、ckpt/ 做同步，也能结束训练进程组。
- 换身份在 preexec_fn 里做（setgroups([]) → setgid → setuid），Python 3.8 的 subprocess 没有
  user= 参数。
- pip：先以这个用户试运行一次 python3。site-packages 可写就什么都不做；用户 site 可用
  （系统 Python、conda 等）就设 PIP_USER=1、PYTHONUSERBASE=<HOME>/.local，`pip install x`
  装到用户目录，run 里可以直接 import；两者都不行（root 拥有的 venv）就不降权。

不降权的情况都记一行说明和一个 run_as_root 事件（凭据对训练进程可读）：spec 里
run_as_root 为 true、runner 不是 root、建不了用户、GPU 设备文件对其他用户不可读写、
改不了目录属主或权限、换身份失败、这个用户跑不了 python3、pip 没有可写的安装位置。

系统调用都经过 System，测试替换它；真正的换身份只有在真实实例（root）上才会发生。
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import signal
import stat
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from dgpu.runner import procs
from dgpu.runner.util import Paths, diag, read_json, write_json_atomic

USER_NAME = "dgpu"
FALLBACK_ID = 47100  # 没有 useradd 时用的 uid/gid（从这里往上找一个没人用的）
FALLBACK_SCAN = 200

REASON_REQUESTED = "requested"  # spec 的 run_as_root 为 true
REASON_NOT_ROOT = "not_root"  # runner 自己不是 root，没法换成别的用户
REASON_NO_USER = "no_user"
REASON_GPU_DEVICES = "gpu_devices"
REASON_LAYOUT = "layout"  # 改不了目录属主或权限
REASON_SWITCH = "switch_failed"  # setgid / setuid 失败（如用户命名空间里没有映射这个 uid）
REASON_PYTHON = "python_unusable"
REASON_PIP = "pip_not_writable"
# 容器里的 root 缺少降权依赖的能力：给训练用户的进程发信号（CAP_KILL）、读它的文件
# （CAP_DAC_OVERRIDE / CAP_DAC_READ_SEARCH）。降权之后 runner 就停不了训练、传不了结果
REASON_CAPS = "capability_missing"

# 每种不降权的原因的一句话说明，写进日志、run_as_root 事件的 data.reason_text 和心跳
REASON_TEXT = {
    REASON_REQUESTED: "job.yaml 的 run_as_root 为 true",
    REASON_NOT_ROOT: "runner 不是 root，没法换成别的用户",
    REASON_NO_USER: "建不了无特权用户",
    REASON_GPU_DEVICES: "GPU 设备文件对其他用户不可读写",
    REASON_LAYOUT: "改不了目录属主或权限",
    REASON_SWITCH: "换成无特权用户失败",
    REASON_PYTHON: "无特权用户运行不了 python3",
    REASON_PIP: "pip 没有可写的安装位置（root 拥有、禁用用户 site 的 venv）",
    REASON_CAPS: "容器里的 root 不能给训练用户的进程发信号或读它的文件",
}

CHOWN_CHECK_EVERY = 256  # chown_tree 每处理这么多个条目调用一次 check（取消、截止）

# 能力探针：以训练用户写一个只有自己可读的文件，然后一直睡着，等 runner 读文件、发信号
CAPS_PROBE_SCRIPT = (
    "import os, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)\n"
    "os.write(fd, b'dgpu-probe')\n"
    "os.close(fd)\n"
    "time.sleep(float(sys.argv[2]))\n"
)
CAPS_PROBE_CONTENT = b"dgpu-probe"

PROBE_SCRIPT = (
    "import json, os, site, sys, sysconfig\n"
    "purelib = sysconfig.get_paths().get('purelib') or ''\n"
    "try:\n"
    "    user_site = site.getusersitepackages()\n"
    "except Exception:\n"
    "    user_site = ''\n"
    "print(json.dumps({'version': sys.version.split()[0], 'prefix': sys.prefix,\n"
    "    'venv': sys.prefix != getattr(sys, 'base_prefix', sys.prefix),\n"
    "    'user_site_enabled': bool(site.ENABLE_USER_SITE), 'user_site': user_site,\n"
    "    'purelib': purelib, 'writable': bool(purelib) and os.access(purelib, os.W_OK),\n"
    "    'uid': os.getuid()}))\n"
)


class PrivsepError(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def switch_user(uid: int, gid: int) -> None:
    """在子进程里（preexec_fn）换成 uid/gid：先丢掉附加组，再换组，最后换用户。"""
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)


class System:
    """降权用到的系统调用。测试替换其中的函数。"""

    def __init__(self) -> None:
        import grp
        import pwd

        self.geteuid = os.geteuid  # type: Callable[[], int]
        self.getpwnam = pwd.getpwnam  # type: Callable[[str], Any]
        self.getpwuid = pwd.getpwuid  # type: Callable[[int], Any]
        self.getgrgid = grp.getgrgid  # type: Callable[[int], Any]
        self.which = shutil.which  # type: Callable[[str], Optional[str]]
        self.run = procs.run_bounded  # type: Callable[..., procs.Completed]
        self.lchown = os.lchown  # type: Callable[[str, int, int], None]
        self.chmod = os.chmod  # type: Callable[[str, int], None]
        self.stat = os.stat  # type: Callable[[str], os.stat_result]
        self.devices = lambda: sorted(glob.glob("/dev/nvidia*"))  # type: Callable[[], List[str]]
        self.switch = switch_user  # type: Callable[[int, int], None]
        self.spawn = _spawn  # type: Callable[..., subprocess.Popen]
        self.kill = os.kill  # type: Callable[[int, int], None]
        self.read_file = _read_head  # type: Callable[[str], bytes]


def _spawn(
    argv: List[str], env: Dict[str, str], cwd: str, preexec_fn: Callable[[], None]
) -> subprocess.Popen:
    return subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        preexec_fn=preexec_fn,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _read_head(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(64)


class Identity:
    """训练进程的身份。active 为 False 时和 runner 同一用户（同样的权限，能读到凭据）。"""

    def __init__(
        self,
        active: bool,
        uid: Optional[int] = None,
        gid: Optional[int] = None,
        user: str = "",
        how: str = "",
        env: Optional[Dict[str, str]] = None,
        reason: str = "",
        detail: str = "",
        switch: Optional[Callable[[int, int], None]] = None,
        python: Optional[Dict[str, Any]] = None,
    ):
        self.active = active
        self.uid = uid
        self.gid = gid
        self.user = user
        self.how = how
        self.env = dict(env or {})
        self.reason = reason
        self.detail = detail
        self._switch = switch or switch_user
        self.python = python or {}

    def preexec(self) -> Optional[Callable[[], None]]:
        if not self.active:
            return None
        uid, gid, switch = self.uid, self.gid, self._switch

        def drop() -> None:
            switch(uid, gid)

        return drop

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "uid": self.uid,
            "gid": self.gid,
            "user": self.user,
            "how": self.how,
            "reason": self.reason,
            "detail": self.detail,
            "env": sorted(self.env),
            "python": self.python,
        }

    @property
    def reason_text(self) -> str:
        return REASON_TEXT.get(self.reason, self.reason)

    def describe(self) -> str:
        """一行说明，写进 attempt 日志（评审 F7：不降权时原因一眼能看到）。"""
        if self.active:
            pip = (
                "PIP_USER=1，pip 装到 $HOME/.local" if self.env.get("PIP_USER") else "pip 照常安装"
            )
            return "训练进程以无特权用户 %s（uid %s）运行，读不到存储凭据；%s" % (
                self.user,
                self.uid,
                pip,
            )
        return "训练进程没有降权（原因 %s：%s；%s），和 runner 同一用户运行，能读到存储凭据" % (
            self.reason,
            self.reason_text,
            self.detail,
        )

    def summary(self) -> Dict[str, Any]:
        """写进心跳的简短记录（dgpu status 显示）。"""
        return {
            "dropped": self.active,
            "user": self.user or None,
            "uid": self.uid,
            "reason": self.reason or None,
            "reason_text": self.reason_text if not self.active else None,
            "detail": (self.detail or "")[:300] or None,
        }


def inactive(reason: str, detail: str) -> Identity:
    return Identity(False, reason=reason, detail=detail)


# ---------------------------------------------------------------- 用户


def _taken(system: System, number: int) -> bool:
    for lookup in (system.getpwuid, system.getgrgid):
        try:
            lookup(number)
            return True
        except KeyError:
            continue
    return False


def resolve_user(system: System, home: str) -> Tuple[int, int, str, str]:
    """(uid, gid, 用户名, 来源)。来源：existing、useradd、numeric。"""
    try:
        entry = system.getpwnam(USER_NAME)
    except KeyError:
        entry = None
    if entry is not None:
        if int(entry.pw_uid) == 0 or int(entry.pw_gid) == 0:
            raise PrivsepError(REASON_NO_USER, "用户 %s 的 uid 或 gid 是 0" % USER_NAME)
        return int(entry.pw_uid), int(entry.pw_gid), USER_NAME, "existing"
    useradd = system.which("useradd")
    problem = "没有 useradd"
    if useradd:
        argv = [
            useradd,
            "--no-create-home",
            "--home-dir",
            home,
            "--shell",
            "/bin/bash",
            "--user-group",
            USER_NAME,
        ]
        try:
            result = system.run(argv, 30.0)
            if result.rc == 0:
                entry = system.getpwnam(USER_NAME)
                if int(entry.pw_uid) != 0 and int(entry.pw_gid) != 0:
                    return int(entry.pw_uid), int(entry.pw_gid), USER_NAME, "useradd"
                problem = "useradd 建出的用户 uid 是 0"
            else:
                problem = "useradd 退出码 %s：%s" % (
                    result.rc,
                    (result.err or b"").decode("utf-8", "replace").strip()[:200],
                )
        except (OSError, subprocess.SubprocessError, KeyError) as exc:
            problem = "useradd 失败（%s）" % type(exc).__name__
    for number in range(FALLBACK_ID, FALLBACK_ID + FALLBACK_SCAN):
        if not _taken(system, number):
            diag("privsep", "%s; using uid/gid %d without a passwd entry" % (problem, number))
            return number, number, USER_NAME, "numeric"
    raise PrivsepError(REASON_NO_USER, "%s，也找不到空闲的 uid/gid" % problem)


# ---------------------------------------------------------------- 设备和目录


def gpu_device_problem(system: System) -> Optional[str]:
    """GPU 设备文件（/dev/nvidia*）对其他用户不可读写时返回说明。没有设备文件时不检查。"""
    bad = []
    for path in system.devices():
        try:
            st = system.stat(path)
        except OSError:
            continue
        if not stat.S_ISCHR(st.st_mode):
            continue
        if st.st_mode & 0o006 != 0o006:
            bad.append("%s（%o）" % (path, st.st_mode & 0o777))
    if bad:
        return "GPU 设备文件对其他用户不可读写：%s" % "、".join(bad)
    return None


def _ancestors(path: str) -> List[str]:
    found = []
    current = os.path.dirname(os.path.abspath(path))
    while True:
        found.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return found


def chown_tree(
    system: System,
    top: str,
    uid: int,
    gid: int,
    check: Optional[Callable[[], None]] = None,
) -> int:
    """把 top 及其下所有文件、目录、链接（不跟随）的属主改成 uid:gid，返回处理的条目数。

    check：每处理 CHOWN_CHECK_EVERY 个条目调用一次，可以抛异常中止（runner 传入取消、截止的
    检查，大数据集上 chown 不会拖住停止；评审 F8）。
    """
    if not os.path.lexists(top):
        return 0
    count = 1
    system.lchown(top, uid, gid)
    if os.path.islink(top) or not os.path.isdir(top):
        return count
    errors = []  # type: List[OSError]
    for dirpath, dirnames, filenames in os.walk(top, onerror=errors.append):
        for name in dirnames + filenames:
            system.lchown(os.path.join(dirpath, name), uid, gid)
            count += 1
            if check is not None and count % CHOWN_CHECK_EVERY == 0:
                check()
    if errors:
        raise errors[0]
    return count


# ---------------------------------------------------------------- 属主标记
# chown 过的目录在 run/ 里留一个标记（uid、gid）。之后（容器重启、重新准备）标记和目录本身的
# 属主都对得上就不再遍历：大数据集只走一遍（评审 F8）。runner 以 root 往目录里写东西（下载
# 代码、数据，恢复 checkpoint）之前删掉对应的标记；不降权时删掉全部标记。

OWNED_TOPS = ("code", "data", "output", "ckpt", "home")


def _owner_marker(paths: Paths, name: str) -> str:
    return os.path.join(paths.run, "owner-%s.json" % name)


def forget_ownership(paths: Paths, *names: str) -> None:
    """root 要往这些目录里写东西了（或不再降权）：删掉属主标记，下次准备时重新 chown。"""
    for name in names or OWNED_TOPS:
        try:
            os.unlink(_owner_marker(paths, name))
        except OSError:
            pass


def _ownership_current(system: System, paths: Paths, name: str, uid: int, gid: int) -> bool:
    marker = read_json(_owner_marker(paths, name))
    if not marker or marker.get("uid") != uid or marker.get("gid") != gid:
        return False
    try:
        st = system.stat(getattr(paths, name))
    except OSError:
        return False
    return int(st.st_uid) == uid and int(st.st_gid) == gid


def lay_out(
    paths: Paths,
    uid: int,
    gid: int,
    system: System,
    note: Callable[[str], None],
    check: Optional[Callable[[], None]] = None,
) -> None:
    """目录属主和权限：训练用的目录归训练用户，运行状态和凭据只有 root 能读。"""
    for ancestor in _ancestors(paths.root):
        mode = system.stat(ancestor).st_mode
        if not mode & 0o001:
            system.chmod(ancestor, (mode & 0o7777) | 0o001)
            note("给 %s 加上其他用户的 x 权限（训练用户要能进入工作目录）" % ancestor)
    system.chmod(paths.root, 0o711)
    for path, mode in ((paths.run, 0o700), (paths.logs, 0o700)):
        if os.path.isdir(path):
            system.chmod(path, mode)
    for path in (paths.launch, paths.boot_lock):
        if os.path.exists(path):
            system.chmod(path, 0o600)
    for path in sorted(glob.glob(os.path.join(paths.root, "bundle-*"))):
        if os.path.isdir(path):
            system.chmod(path, 0o700)
    if os.path.isdir(paths.logs):
        for name in os.listdir(paths.logs):
            full = os.path.join(paths.logs, name)
            if os.path.isfile(full) and not os.path.islink(full):
                system.chmod(full, 0o600)
    os.makedirs(paths.home, exist_ok=True)
    system.chmod(paths.home, 0o700)
    for name in OWNED_TOPS:
        top = getattr(paths, name)
        os.makedirs(top, exist_ok=True)
        if _ownership_current(system, paths, name, uid, gid):
            note("%s/ 的属主已经是训练用户（uid %d），不再逐个文件 chown" % (name, uid))
            continue
        count = chown_tree(system, top, uid, gid, check)
        write_json_atomic(_owner_marker(paths, name), {"uid": uid, "gid": gid, "count": count})


def identity_env(paths: Paths, user: str, base_path: str) -> Dict[str, str]:
    userbase = os.path.join(paths.home, ".local")
    path = os.path.join(userbase, "bin")
    if base_path:
        path += os.pathsep + base_path
    return {
        "HOME": paths.home,
        "USER": user,
        "LOGNAME": user,
        "PYTHONUSERBASE": userbase,
        "PATH": path,
    }


# ---------------------------------------------------------------- 试运行 python3


def probe_python(
    system: System,
    paths: Paths,
    env: Dict[str, str],
    uid: int,
    gid: int,
    timeout: float,
) -> Dict[str, Any]:
    """以训练用户运行一次 python3，返回它看到的环境。失败抛 PrivsepError。"""

    def drop() -> None:
        system.switch(uid, gid)

    try:
        result = system.run(
            ["python3", "-c", PROBE_SCRIPT], timeout, env=env, cwd=paths.code, preexec_fn=drop
        )
    except subprocess.SubprocessError as exc:
        raise PrivsepError(REASON_SWITCH, "换成 uid %d 失败（%s）" % (uid, exc)) from None
    except OSError as exc:
        raise PrivsepError(
            REASON_PYTHON, "训练用户运行不了 python3（%s: %s）" % (type(exc).__name__, exc)
        ) from None
    if result.rc is None or result.timed_out:
        raise PrivsepError(REASON_PYTHON, "训练用户运行 python3 超过 %.0f 秒没有结束" % timeout)
    if result.rc != 0:
        detail = (result.err or b"").decode("utf-8", "replace").strip().splitlines()
        raise PrivsepError(
            REASON_PYTHON,
            "训练用户运行 python3 失败（退出码 %s）：%s" % (result.rc, (detail or [""])[-1][:200]),
        )
    try:
        info = json.loads((result.out or b"").decode("utf-8", "replace").strip().splitlines()[-1])
        if not isinstance(info, dict):
            raise ValueError("not an object")
    except (ValueError, IndexError) as exc:
        raise PrivsepError(REASON_PYTHON, "读不懂 python3 的试运行结果（%s）" % exc) from None
    return info


# ---------------------------------------------------------------- 能力探针


def probe_capabilities(
    system: System,
    paths: Paths,
    env: Dict[str, str],
    uid: int,
    gid: int,
    timeout: float,
    kill_wait: float = 5.0,
) -> None:
    """真的以训练用户起一个进程，确认降权之后 runner（root）还管得住它（评审 F8）：

    - 能读训练用户写的、只有它自己可读的文件（同步 output/、ckpt/ 要读；缺 CAP_DAC_OVERRIDE
      和 CAP_DAC_READ_SEARCH 时读不了）；
    - 能给训练用户的进程发信号 0 和 SIGKILL（停止训练要用；缺 CAP_KILL 时发不了）。

    做不到就抛 PrivsepError(REASON_CAPS)，调用方退回不降权。探针进程最多睡 timeout 秒，
    runner 杀不掉它时它也会自己退出。
    """
    probe = os.path.join(paths.home, ".dgpu-capprobe")
    try:
        os.unlink(probe)
    except OSError:
        pass

    def drop() -> None:
        system.switch(uid, gid)

    try:
        proc = system.spawn(
            ["python3", "-c", CAPS_PROBE_SCRIPT, probe, "%g" % max(1.0, timeout)],
            env,
            paths.home,
            drop,
        )
    except subprocess.SubprocessError as exc:
        raise PrivsepError(REASON_SWITCH, "换成 uid %d 失败（%s）" % (uid, exc)) from None
    except OSError as exc:
        raise PrivsepError(
            REASON_PYTHON, "训练用户运行不了 python3（%s: %s）" % (type(exc).__name__, exc)
        ) from None
    try:
        end = time.monotonic() + timeout
        while not os.path.exists(probe):
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
                raise PrivsepError(
                    REASON_CAPS,
                    "训练用户的探针进程退出了（退出码 %s）：%s"
                    % (proc.returncode, (err.strip().splitlines() or [""])[-1][:200]),
                )
            if time.monotonic() >= end:
                raise PrivsepError(
                    REASON_CAPS, "训练用户的探针进程 %.0f 秒内没有写出文件" % timeout
                )
            time.sleep(0.02)
        content = b""
        for _ in range(50):  # 文件刚建好、内容还没写完
            try:
                content = system.read_file(probe)
            except PermissionError as exc:
                raise PrivsepError(
                    REASON_CAPS,
                    "runner（root）读不了训练用户的文件（%s），没法同步它写的结果"
                    "（容器可能没有 CAP_DAC_OVERRIDE）" % exc,
                ) from None
            if content == CAPS_PROBE_CONTENT:
                break
            time.sleep(0.02)
        for signum in (0, signal.SIGKILL):
            try:
                system.kill(proc.pid, signum)
            except PermissionError as exc:
                raise PrivsepError(
                    REASON_CAPS,
                    "runner（root）不能给训练用户的进程发信号 %d（%s），没法停止训练"
                    "（容器可能没有 CAP_KILL）" % (signum, exc),
                ) from None
        if procs.wait_bounded(proc, kill_wait) is None:
            raise PrivsepError(REASON_CAPS, "SIGKILL 之后训练用户的探针进程仍在运行")
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
            procs.wait_bounded(proc, kill_wait)
        if proc.stderr is not None:
            proc.stderr.close()
        try:
            os.unlink(probe)
        except OSError:
            pass


# ---------------------------------------------------------------- 入口


def prepare(
    paths: Paths,
    run_as_root: bool,
    base_env: Dict[str, str],
    system: Optional[System] = None,
    probe_timeout: float = 60.0,
    note: Optional[Callable[[str], None]] = None,
    check: Optional[Callable[[], None]] = None,
    kill_wait: float = 5.0,
) -> Identity:
    """决定训练进程的身份，并把目录准备好。

    不会因为降权的问题抛异常：任何问题都退回不降权并说明原因（这时删掉属主标记，下次重新
    chown）。check 由 runner 传入（取消、截止），它抛的异常照常往外传，中止准备。
    """
    system = system or System()
    note = note or (lambda text: diag("privsep", text))
    identity = _prepare(paths, run_as_root, base_env, system, probe_timeout, note, check, kill_wait)
    if not identity.active:
        forget_ownership(paths)
    return identity


def _prepare(
    paths: Paths,
    run_as_root: bool,
    base_env: Dict[str, str],
    system: System,
    probe_timeout: float,
    note: Callable[[str], None],
    check: Optional[Callable[[], None]],
    kill_wait: float,
) -> Identity:
    if run_as_root:
        return inactive(REASON_REQUESTED, "spec 里 run_as_root 为 true")
    try:
        euid = system.geteuid()
    except OSError as exc:
        return inactive(REASON_NOT_ROOT, "读不到 runner 的 uid（%s）" % exc)
    if euid != 0:
        return inactive(REASON_NOT_ROOT, "runner 不是 root（uid %d），没法换成别的用户" % euid)
    try:
        uid, gid, user, how = resolve_user(system, paths.home)
        problem = gpu_device_problem(system)
        if problem:
            raise PrivsepError(REASON_GPU_DEVICES, problem)
        try:
            lay_out(paths, uid, gid, system, note, check)
        except OSError as exc:
            raise PrivsepError(
                REASON_LAYOUT, "改不了目录属主或权限（%s: %s）" % (type(exc).__name__, exc)
            ) from None
        env = dict(base_env)
        env.update(identity_env(paths, user, base_env.get("PATH", "")))
        info = probe_python(system, paths, env, uid, gid, probe_timeout)
        if info.get("uid") is not None and int(info["uid"]) != uid:
            raise PrivsepError(
                REASON_SWITCH, "试运行的 python3 的 uid 是 %s，不是 %d" % (info.get("uid"), uid)
            )
        extra = identity_env(paths, user, base_env.get("PATH", ""))
        if info.get("writable"):
            pass  # site-packages 可写：pip 照常安装
        elif info.get("user_site_enabled"):
            extra["PIP_USER"] = "1"
            user_site = str(info.get("user_site") or "")
            if user_site and user_site.startswith(paths.home + os.sep):
                # 提前建好用户 site 目录：python 只在它存在时把它加进 sys.path
                os.makedirs(user_site, exist_ok=True)
                chown_tree(system, os.path.join(paths.home, ".local"), uid, gid)
        else:
            raise PrivsepError(
                REASON_PIP,
                "python3（%s）的 site-packages 不可写，用户 site 也被禁用（root 拥有的 venv？），"
                "训练用户没法 pip install" % info.get("prefix"),
            )
        probe_capabilities(system, paths, env, uid, gid, probe_timeout, kill_wait)
    except PrivsepError as exc:
        return inactive(exc.reason, str(exc))
    except OSError as exc:
        return inactive(REASON_LAYOUT, "%s: %s" % (type(exc).__name__, exc))
    return Identity(
        True,
        uid=uid,
        gid=gid,
        user=user,
        how=how,
        env=extra,
        switch=system.switch,
        python=info,
    )
