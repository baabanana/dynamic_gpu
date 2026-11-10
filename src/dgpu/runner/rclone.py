"""rclone 子进程封装：runner 对存储的所有访问都经过这里。

- 存储凭据只放进 rclone 子进程的环境变量（RCLONE_CONFIG_DGPU_*），不进命令行参数、
  不进 runner 自己的环境，也就不会传给训练进程。
- 每次调用都有时限，并且可以被中断（取消、发现新 attempt、超时），rclone 卡住不会
  拖住调用方以外的线程。
- 写入可以带 guard：guard 返回 False 时不再启动新的写入；abort() 会杀掉正在进行的调用。
  两者在同一把锁下进行，所以“发现新 attempt 之后不再写”没有竞争窗口。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dgpu.runner import procs
from dgpu.runner.util import Clock

REMOTE = "dgpu"
ENV_PREFIX = "RCLONE_CONFIG_DGPU_"

# rclone 的退出码：3 目录不存在，4 文件不存在
NOT_FOUND_CODES = (3, 4)

# 所有调用都带的参数。--local-no-check-updated：训练进程边写边传时不把“文件在上传中
# 变化”当作错误，下一轮同步会补传新内容；最终同步发生在训练进程退出之后。
COMMON_FLAGS = (
    "-q",
    "--contimeout",
    "30s",
    "--timeout",
    "300s",
    "--low-level-retries",
    "5",
    "--local-no-check-updated",
)


class RcloneError(Exception):
    """rclone 调用失败。消息只含命令名、退出码和 stderr 摘要，不含凭据。"""

    def __init__(
        self,
        op: str,
        message: str,
        rc: Optional[int] = None,
        timed_out: bool = False,
        aborted: bool = False,
        blocked: bool = False,
    ):
        super().__init__("rclone %s: %s" % (op, message))
        self.op = op
        self.rc = rc
        self.timed_out = timed_out
        self.aborted = aborted
        self.blocked = blocked

    @property
    def not_found(self) -> bool:
        text = str(self).lower()
        return self.rc in NOT_FOUND_CODES or "not found" in text


class Result:
    __slots__ = ("rc", "out", "err", "timed_out", "aborted")

    def __init__(self, rc: int, out: bytes, err: bytes, timed_out: bool, aborted: bool):
        self.rc = rc
        self.out = out
        self.err = err
        self.timed_out = timed_out
        self.aborted = aborted


def store_env(store: Dict[str, Any]) -> Dict[str, str]:
    """启动文档的 store 段 → rclone 的 S3 remote 配置（环境变量形式）。"""
    return {
        ENV_PREFIX + "TYPE": "s3",
        ENV_PREFIX + "PROVIDER": str(store.get("provider") or "Cloudflare"),
        ENV_PREFIX + "ENDPOINT": str(store["endpoint"]),
        ENV_PREFIX + "REGION": str(store.get("region") or "auto"),
        ENV_PREFIX + "ACCESS_KEY_ID": str(store["access_key_id"]),
        ENV_PREFIX + "SECRET_ACCESS_KEY": str(store["secret_access_key"]),
        ENV_PREFIX + "ENV_AUTH": "false",
        # 只授权单个 bucket 的 token 没有建 bucket 的权限
        ENV_PREFIX + "NO_CHECK_BUCKET": "true",
    }


def _tail(data: bytes, limit: int = 400) -> str:
    text = data.decode("utf-8", "replace").strip()
    if len(text) > limit:
        text = "..." + text[-limit:]
    return text.replace("\n", " | ")


class Rclone:
    def __init__(
        self,
        binary: str,
        store: Dict[str, Any],
        config_path: str,
        base_env: Optional[Dict[str, str]] = None,
        small_timeout: float = 120.0,
        poll_s: float = 0.2,
        clock: Optional[Clock] = None,
        pid_dir: Optional[str] = None,
    ):
        """pid_dir：在这里登记正在运行的 rclone（文件名是 pid）。runner 被 SIGKILL 时它启动的
        rclone 还活着，watchdog 按登记把它们杀掉，免得它们在 watchdog 写最终状态之后才落盘。"""
        self.binary = binary
        self.pid_dir = pid_dir
        self.bucket = str(store["bucket"])
        self.small_timeout = small_timeout
        self._poll_s = max(0.01, poll_s)
        self._clock = clock or Clock()
        env = dict(os.environ if base_env is None else base_env)
        for key in list(env):
            if key.startswith(ENV_PREFIX) or key == "RCLONE_CONFIG":
                del env[key]
        env.update(store_env(store))
        env["RCLONE_CONFIG"] = config_path
        self._env = env
        if not os.path.exists(config_path):
            fd = os.open(config_path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(fd)
        self._lock = threading.Lock()
        self._procs = {}  # type: Dict[subprocess.Popen, Optional[str]]
        self._aborted = set()  # type: set

    # ------------------------------------------------------------ 路径

    def path(self, key: str, bucket: Optional[str] = None) -> str:
        key = key.strip("/")
        return "%s:%s/%s" % (REMOTE, bucket or self.bucket, key)

    # ------------------------------------------------------------ 进程

    def run(
        self,
        args: Iterable[str],
        timeout: Optional[float] = None,
        data: Optional[bytes] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
        guard: Optional[Callable[[], bool]] = None,
    ) -> Result:
        args = list(args)
        op = args[0] if args else "?"
        cmd = [self.binary] + args + list(COMMON_FLAGS)
        with self._lock:
            if guard is not None and not guard():
                raise RcloneError(op, "write blocked (attempt fenced)", blocked=True)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._env,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RcloneError(op, "cannot start rclone (%s)" % type(exc).__name__) from None
            self._procs[proc] = tag
            if self.pid_dir is not None:
                procs.register(self.pid_dir, proc.pid, self.binary)
        timed_out = False
        out, err = b"", b""
        try:
            deadline = None if timeout is None else self._clock.monotonic() + timeout
            first = True
            while True:
                step = self._poll_s
                if deadline is not None:
                    step = max(0.0, min(step, deadline - self._clock.monotonic()))
                try:
                    if first:
                        first = False
                        out, err = proc.communicate(data, timeout=step)
                    else:
                        out, err = proc.communicate(timeout=step)
                    break
                except subprocess.TimeoutExpired:
                    expired = deadline is not None and self._clock.monotonic() >= deadline
                    stop = interrupt is not None and interrupt()
                    if expired or stop or proc in self._aborted:
                        timed_out = expired
                        _kill_group(proc)
                        # SIGKILL 之后的等待也有上限：进程卡在内核里时放弃它
                        try:
                            out, err = proc.communicate(timeout=procs.KILL_WAIT_S)
                        except subprocess.TimeoutExpired:
                            procs.abandon(proc)
                            out, err = b"", b"rclone ignored SIGKILL; abandoned"
                        break
        finally:
            with self._lock:
                self._procs.pop(proc, None)
                aborted = proc in self._aborted
                self._aborted.discard(proc)
            if proc.poll() is None:
                procs.finish(proc, kill=_kill_group)
            if self.pid_dir is not None and proc.poll() is not None:
                procs.unregister(self.pid_dir, proc.pid)
        rc = proc.returncode if proc.returncode is not None else -9
        return Result(rc, out or b"", err or b"", timed_out, aborted)

    def abort(self, tag: Optional[str] = None) -> int:
        """杀掉正在进行的调用（tag 为 None 时全部）。返回杀掉的数量。"""
        count = 0
        with self._lock:
            for proc, proc_tag in list(self._procs.items()):
                if proc.returncode is not None:
                    continue  # 已回收，pid 可能被复用
                if tag is None or proc_tag == tag:
                    self._aborted.add(proc)
                    _kill_group(proc)
                    count += 1
        return count

    def _check(self, op: str, result: Result) -> Result:
        if result.aborted:
            raise RcloneError(op, "aborted", rc=result.rc, aborted=True)
        if result.timed_out:
            raise RcloneError(op, "timed out", rc=result.rc, timed_out=True)
        if result.rc != 0:
            detail = _tail(result.err) or "no error output"
            raise RcloneError(op, "exit code %s: %s" % (result.rc, detail), rc=result.rc)
        return result

    # ------------------------------------------------------------ 操作

    def cat(
        self,
        key: str,
        bucket: Optional[str] = None,
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
    ) -> Optional[bytes]:
        """读一个对象；不存在返回 None。

        S3 上对不存在的 key 执行 cat 可能是成功但无输出（rclone 把它当作空目录），也可能
        报 not found；两种都按“不存在”处理。我们读的 JSON 文档都不会是空的。
        """
        result = self.run(
            ["cat", self.path(key, bucket)],
            timeout=self.small_timeout if timeout is None else timeout,
            interrupt=interrupt,
            tag=tag,
        )
        if result.rc != 0 and not (result.aborted or result.timed_out):
            error = RcloneError("cat", _tail(result.err), rc=result.rc)
            if error.not_found:
                return None
        self._check("cat", result)
        return result.out if result.out else None

    def rcat(
        self,
        key: str,
        data: bytes,
        timeout: Optional[float] = None,
        tag: Optional[str] = None,
        guard: Optional[Callable[[], bool]] = None,
    ) -> None:
        result = self.run(
            ["rcat", self.path(key)],
            timeout=self.small_timeout if timeout is None else timeout,
            data=data,
            tag=tag,
            guard=guard,
        )
        self._check("rcat", result)

    def copy_up(
        self,
        local_dir: str,
        prefix: str,
        exclude: Iterable[str] = (),
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
        guard: Optional[Callable[[], bool]] = None,
    ) -> None:
        args = ["copy", local_dir, self.path(prefix)]
        for pattern in exclude:
            args += ["--exclude", pattern]
        self._check(
            "copy", self.run(args, timeout=timeout, interrupt=interrupt, tag=tag, guard=guard)
        )

    def copy_down(
        self,
        prefix: str,
        local_dir: str,
        bucket: Optional[str] = None,
        exclude: Iterable[str] = (),
        update: bool = False,
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
    ) -> None:
        args = ["copy", self.path(prefix, bucket), local_dir]
        for pattern in exclude:
            args += ["--exclude", pattern]
        if update:
            args.append("--update")
        self._check("copy", self.run(args, timeout=timeout, interrupt=interrupt, tag=tag))

    def copyto_up(
        self,
        local_file: str,
        key: str,
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
        guard: Optional[Callable[[], bool]] = None,
    ) -> None:
        args = ["copyto", local_file, self.path(key)]
        self._check(
            "copyto", self.run(args, timeout=timeout, interrupt=interrupt, tag=tag, guard=guard)
        )

    def copyto_down(
        self,
        key: str,
        local_file: str,
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
    ) -> None:
        args = ["copyto", self.path(key), local_file]
        self._check("copyto", self.run(args, timeout=timeout, interrupt=interrupt, tag=tag))

    def move_up(
        self,
        local_dir: str,
        prefix: str,
        timeout: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> None:
        args = ["move", local_dir, self.path(prefix)]
        self._check("move", self.run(args, timeout=timeout, tag=tag))

    def lsjson(
        self,
        prefix: str,
        bucket: Optional[str] = None,
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """递归列出前缀下的文件；前缀不存在返回空列表。"""
        args = ["lsjson", "-R", "--files-only", self.path(prefix, bucket)]
        result = self.run(
            args,
            timeout=self.small_timeout if timeout is None else timeout,
            interrupt=interrupt,
            tag=tag,
        )
        if result.rc in NOT_FOUND_CODES and not (result.aborted or result.timed_out):
            return []
        self._check("lsjson", result)
        try:
            items = json.loads(result.out.decode("utf-8") or "[]")
        except ValueError:
            raise RcloneError("lsjson", "output is not JSON") from None
        return [item for item in items if isinstance(item, dict) and not item.get("IsDir")]

    def size(
        self,
        prefix: str,
        bucket: Optional[str] = None,
        timeout: Optional[float] = None,
        interrupt: Optional[Callable[[], bool]] = None,
    ) -> Tuple[int, int]:
        """前缀下的 (文件数, 字节数)；前缀不存在返回 (0, 0)。"""
        args = ["size", "--json", self.path(prefix, bucket)]
        result = self.run(
            args, timeout=self.small_timeout if timeout is None else timeout, interrupt=interrupt
        )
        if result.rc in NOT_FOUND_CODES and not (result.aborted or result.timed_out):
            return 0, 0
        self._check("size", result)
        try:
            info = json.loads(result.out.decode("utf-8"))
            return int(info.get("count", 0)), int(info.get("bytes", 0))
        except (ValueError, AttributeError):
            raise RcloneError("size", "output is not JSON") from None


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass
