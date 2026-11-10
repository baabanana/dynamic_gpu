"""attempt 日志：训练进程的输出和 runner 的进度行写进同一个文件。

读取线程（Pump）持续读管道，训练进程永远不会因为管道写满而卡住。超过上限后继续
读取但不再落盘，只在截断的那一刻写一行说明并回调一次（runner 用它写 log_truncated）。
磁盘写满等写入错误同样不影响读取。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from dgpu.runner.util import Clock, diag

READ_CHUNK = 65536


class LogSink:
    def __init__(
        self,
        path: str,
        max_bytes: int,
        clock: Optional[Clock] = None,
        on_truncate: Optional[Callable[[int], None]] = None,
    ):
        self.path = path
        self.max_bytes = max(0, int(max_bytes))
        self._clock = clock or Clock()
        self._on_truncate = on_truncate
        self._lock = threading.Lock()
        self._file = open(path, "ab")
        # 同一 attempt 重启（Vast 容器重启）时接着已有内容计数
        self._written = self._file.tell()
        self.truncated = self._written >= self.max_bytes > 0
        self.dropped_bytes = 0
        self.last_output_mono = None  # type: Optional[float]
        self.last_output_wall = None  # type: Optional[float]
        self._write_failed = False

    def child_output(self, data: bytes) -> None:
        """训练进程的输出。更新“最近有输出”的时间，用于卡死检测和心跳。"""
        if not data:
            return
        self.last_output_mono = self._clock.monotonic()
        self.last_output_wall = self._clock.time()
        self._write(data)

    def note(self, text: str) -> None:
        """runner 的进度行。不算作训练进程的输出。"""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock.time()))
        self._write(("[dgpu %s] %s\n" % (stamp, text)).encode("utf-8", "replace"))

    def _write(self, data: bytes) -> None:
        fire = None
        with self._lock:
            if self._file is None:
                return
            if self.truncated:
                self.dropped_bytes += len(data)
                return
            room = self.max_bytes - self._written
            if len(data) <= room:
                self._raw_write(data)
                self._written += len(data)
                return
            self._raw_write(data[:room])
            self._written += room
            self.dropped_bytes += len(data) - room
            self.truncated = True
            marker = (
                "\n[dgpu] 日志超过 %d MB，之后的输出不再写入文件（训练进程的输出仍在被读取）\n"
                % (self.max_bytes // (1024 * 1024))
            )
            self._raw_write(marker.encode("utf-8"))
            fire = self._on_truncate
        if fire is not None:
            try:
                fire(self.max_bytes)
            except Exception as exc:
                diag("log", "truncate callback failed: %s" % exc)

    def _raw_write(self, data: bytes) -> None:
        try:
            self._file.write(data)
            self._file.flush()
        except (OSError, ValueError) as exc:
            if not self._write_failed:
                self._write_failed = True
                diag("log", "cannot write %s: %s; output is still drained" % (self.path, exc))

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None


class Pump(threading.Thread):
    """持续读取一个管道直到 EOF，交给 LogSink。"""

    def __init__(self, stream, sink: LogSink):
        super().__init__(name="log-pump", daemon=True)
        self._stream = stream
        self._sink = sink

    def run(self) -> None:
        fd = self._stream.fileno()
        try:
            while True:
                try:
                    chunk = os.read(fd, READ_CHUNK)
                except InterruptedError:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    self._sink.child_output(chunk)
                except Exception as exc:  # 记录失败也不能停止读取，否则训练进程会卡在写管道上
                    diag("log", "log sink error ignored: %s" % exc)
        finally:
            try:
                self._stream.close()
            except OSError:
                pass
