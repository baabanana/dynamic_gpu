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
import stat
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

from dgpu.runner import procs
from dgpu.runner.util import Paths, diag

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

    def describe(self) -> str:
        if self.active:
            pip = (
                "PIP_USER=1，pip 装到 $HOME/.local" if self.env.get("PIP_USER") else "pip 照常安装"
            )
            return "训练进程以无特权用户 %s（uid %s）运行，读不到存储凭据；%s" % (
                self.user,
                self.uid,
                pip,
            )
        return "训练进程和 runner 同一用户运行（%s：%s），能读到存储凭据" % (
            self.reason,
            self.detail,
        )


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


def chown_tree(system: System, top: str, uid: int, gid: int) -> int:
    """把 top 及其下所有文件、目录、链接（不跟随）的属主改成 uid:gid，返回处理的条目数。"""
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
    if errors:
        raise errors[0]
    return count


def lay_out(paths: Paths, uid: int, gid: int, system: System, note: Callable[[str], None]) -> None:
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
    for top in (paths.code, paths.data, paths.output, paths.ckpt, paths.home):
        os.makedirs(top, exist_ok=True)
        chown_tree(system, top, uid, gid)


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


# ---------------------------------------------------------------- 入口


def prepare(
    paths: Paths,
    run_as_root: bool,
    base_env: Dict[str, str],
    system: Optional[System] = None,
    probe_timeout: float = 60.0,
    note: Optional[Callable[[str], None]] = None,
) -> Identity:
    """决定训练进程的身份，并把目录准备好。不会抛异常：任何问题都退回不降权并说明原因。"""
    system = system or System()
    note = note or (lambda text: diag("privsep", text))
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
            lay_out(paths, uid, gid, system, note)
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
