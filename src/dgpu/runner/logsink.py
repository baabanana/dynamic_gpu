"""attempt 日志：训练进程的输出和 runner 的进度行写进同一个文件。

读取线程（Pump）持续读管道，训练进程永远不会因为管道写满而卡住。磁盘写满等写入错误
同样不影响读取。

日志上限（max_log_mb）只限制训练进程的输出。超过之后：

- 训练输出继续读取，但只保留最后 tail_bytes 字节：写进两个轮流使用的尾部文件
  （<日志>.tail.0 / .tail.1，各占一半，写满一个就清空另一个接着写），磁盘占用有上限，
  runner 崩溃时 watchdog 也能从磁盘上拿到；
- runner 的进度行（失败原因、退出码、最终状态等）另有 note_reserve 字节的空间，照常写；
- 收尾时（flush_tail）把尾部追加到日志末尾，前面写明丢弃了多少字节。日志始终只追加，
  `dgpu logs -f` 的增量读取不受影响。

截断的那一刻写一行说明并回调一次（runner 用它写 log_truncated）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional

from dgpu.runner.util import Clock, diag

READ_CHUNK = 65536
TAIL_SUFFIXES = (".tail.0", ".tail.1")


def tail_paths(log_path: str) -> List[str]:
    return [log_path + suffix for suffix in TAIL_SUFFIXES]


def append_tail(log_path: str, header: str = "") -> int:
    """把尾部文件按时间顺序追加到日志末尾，然后删掉它们；返回追加的训练输出字节数。

    runner 收尾时和 watchdog 接手时都调用；尾部文件不存在时什么都不做。
    """
    parts = []
    for path in tail_paths(log_path):
        try:
            st = os.stat(path)
        except OSError:
            continue
        parts.append((st.st_mtime_ns, path))
    if not parts:
        return 0
    parts.sort()
    total = 0
    try:
        with open(log_path, "ab") as out:
            if header:
                out.write(header.encode("utf-8", "replace"))
            for _mtime, path in parts:
                try:
                    with open(path, "rb") as handle:
                        data = handle.read()
                except OSError:
                    continue
                out.write(data)
                total += len(data)
            out.write(
                ("\n[dgpu] （以上是被截断的训练输出的最后 %d 字节）\n" % total).encode("utf-8")
            )
    except OSError as exc:
        diag("log", "cannot append the log tail to %s: %s" % (log_path, exc))
        return 0
    for _mtime, path in parts:
        try:
            os.unlink(path)
        except OSError:
            pass
    return total


class LogSink:
    def __init__(
        self,
        path: str,
        max_bytes: int,
        clock: Optional[Clock] = None,
        on_truncate: Optional[Callable[[int], None]] = None,
        tail_bytes: int = 1024 * 1024,
        note_reserve: int = 1024 * 1024,
    ):
        self.path = path
        self.max_bytes = max(0, int(max_bytes))
        self.tail_bytes = max(0, int(tail_bytes))
        self.note_reserve = max(0, int(note_reserve))
        self._clock = clock or Clock()
        self._on_truncate = on_truncate
        self._lock = threading.Lock()
        # 上一次运行（同一 attempt 的容器重启）留下的尾部先接到日志末尾，不和这次的混在一起
        append_tail(path, "\n[dgpu] 上一次运行被截断的训练输出的结尾：\n")
        self._file = open(path, "ab")
        # 同一 attempt 重启（Vast 容器重启）时接着已有内容计数
        self._written = self._file.tell()
        self.truncated = self._written >= self.max_bytes > 0
        self.dropped_bytes = 0
        self.last_output_mono = None  # type: Optional[float]
        self.last_output_wall = None  # type: Optional[float]
        self._write_failed = False
        self._notes_dropped = False
        self._tail_index = 0
        self._tail_file = None  # type: Optional[object]
        self._tail_size = 0

    def child_output(self, data: bytes) -> None:
        """训练进程的输出。更新“最近有输出”的时间，用于卡死检测和心跳。"""
        if not data:
            return
        self.last_output_mono = self._clock.monotonic()
        self.last_output_wall = self._clock.time()
        self._write(data)

    def note(self, text: str) -> None:
        """runner 的进度行。不算作训练进程的输出，不受训练输出上限的限制（另有预留空间）。"""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock.time()))
        data = ("[dgpu %s] %s\n" % (stamp, text)).encode("utf-8", "replace")
        with self._lock:
            if self._file is None:
                return
            if self._written + len(data) > self.max_bytes + self.note_reserve:
                if not self._notes_dropped:
                    self._notes_dropped = True
                    diag("log", "note reserve of %s exhausted; dropping notes" % self.path)
                return
            self._raw_write(data)
            self._written += len(data)

    def _write(self, data: bytes) -> None:
        fire = None
        with self._lock:
            if self._file is None:
                return
            if self.truncated:
                self.dropped_bytes += len(data)
                self._tail_write(data)
                return
            room = self.max_bytes - self._written
            if len(data) <= room:
                self._raw_write(data)
                self._written += len(data)
                return
            room = max(0, room)
            self._raw_write(data[:room])
            self._written += room
            self.dropped_bytes += len(data) - room
            self.truncated = True
            marker = (
                "\n[dgpu] 日志超过 %d MB，之后的训练输出只保留最后 %d KB，结束时附在日志末尾"
                "（训练进程的输出仍在被读取）\n"
                % (self.max_bytes // (1024 * 1024), self.tail_bytes // 1024)
            )
            encoded = marker.encode("utf-8")
            self._raw_write(encoded)
            self._written += len(encoded)
            self._tail_write(data[room:])
            fire = self._on_truncate
        if fire is not None:
            try:
                fire(self.max_bytes)
            except Exception as exc:
                diag("log", "truncate callback failed: %s" % exc)

    def _tail_write(self, data: bytes) -> None:
        """写进当前的尾部文件；写满一半就换另一个（清空后重写）。"""
        half = self.tail_bytes // 2
        if half <= 0 or not data:
            return
        if len(data) > half:
            data = data[-half:]
        try:
            if self._tail_file is None or self._tail_size + len(data) > half:
                if self._tail_file is not None:
                    self._tail_file.close()
                    self._tail_index = 1 - self._tail_index
                path = tail_paths(self.path)[self._tail_index]
                self._tail_file = open(path, "wb")
                self._tail_size = 0
            self._tail_file.write(data)
            self._tail_file.flush()
            self._tail_size += len(data)
        except (OSError, ValueError) as exc:
            if not self._write_failed:
                self._write_failed = True
                diag("log", "cannot write the log tail of %s: %s" % (self.path, exc))

    def flush_tail(self) -> int:
        """把截断后保留的训练输出尾部追加到日志末尾（收尾时调用，可以重复调用）。"""
        with self._lock:
            if self._tail_file is not None:
                try:
                    self._tail_file.close()
                except OSError:
                    pass
                self._tail_file = None
                self._tail_size = 0
            if self._file is not None:
                try:
                    self._file.flush()
                except (OSError, ValueError):
                    pass
            if not self.truncated:
                return 0
            header = "\n[dgpu] 训练输出超过上限，共丢弃 %d 字节；下面是最后保留的部分：\n" % (
                self.dropped_bytes
            )
            added = append_tail(self.path, header)
            self._written += added
            return added

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
            for handle in (self._file, self._tail_file):
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
            self._file = None
            self._tail_file = None


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
