"""事件：先原子地写进本地队列目录，再用 rclone move 上传（上传成功才删除本地文件）。

存储断开期间产生的事件不会丢，恢复后随下一次 flush 补传。runner 和 watchdog 各用
自己的队列目录，避免两个 rclone move 同时处理同一批文件。

emit 本身从不等存储：启动了后台上传线程（start_flusher）时只唤醒它；没有启动时只做一次
非阻塞的尝试。所以在主循环停止训练之前、在读日志的线程里（日志截断回调）写事件都不会
被卡住的存储拖住。
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from dgpu import layout, states
from dgpu.runner.rclone import Rclone, RcloneError
from dgpu.runner.util import diag


class Events:
    def __init__(
        self,
        rclone: Optional[Rclone],
        job_id: str,
        attempt: Optional[int],
        source: str,
        spool_dir: str,
        tmp_dir: str,
    ):
        self.rclone = rclone
        self.job_id = job_id
        self.attempt = attempt
        self.source = source
        self.spool_dir = spool_dir
        self.tmp_dir = tmp_dir
        os.makedirs(spool_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._flusher = None  # type: Optional[threading.Thread]

    # ------------------------------------------------------------ 后台上传

    def start_flusher(self, interval: float = 30.0) -> None:
        """后台线程：被 emit 唤醒或每 interval 秒上传一次队列。"""
        if self._flusher is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                self._wake.wait(max(0.01, interval))
                self._wake.clear()
                if self._stop.is_set():
                    return
                try:
                    self.flush(block=False)
                except Exception as exc:  # 上传失败只记录，事件留在本地
                    diag(self.source, "event flusher error: %s" % exc)

        self._flusher = threading.Thread(target=loop, name="events", daemon=True)
        self._flusher.start()

    def stop_flusher(self, join_s: float = 10.0) -> None:
        """停止后台上传：中止正在进行的上传（事件留在本地队列），最多等 join_s 秒。"""
        thread = self._flusher
        if thread is None:
            return
        self._stop.set()
        self._wake.set()
        if self.rclone is not None:
            self.rclone.abort("events")
        thread.join(max(0.0, join_s))
        if thread.is_alive():
            diag(self.source, "event flusher did not stop in time")
        self._flusher = None

    def emit(
        self,
        name: str,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
        flush: bool = True,
    ) -> None:
        doc = states.new_event_doc(name, self.source, self.job_id, self.attempt, message, data)
        key = layout.event_key(self.job_id, self.source, ts=doc["ts"])
        filename = key.rsplit("/", 1)[1]
        tmp = os.path.join(self.tmp_dir, filename)
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, os.path.join(self.spool_dir, filename))
        except OSError as exc:
            diag(self.source, "cannot spool event %s: %s" % (name, exc))
        diag(self.source, "event %s: %s" % (name, message))
        if flush:
            if self._flusher is not None:
                self._wake.set()
            else:
                self.flush(block=False)

    def pending(self) -> int:
        try:
            return len(os.listdir(self.spool_dir))
        except OSError:
            return 0

    def flush(self, block: bool = False, timeout: Optional[float] = None) -> bool:
        """上传队列里的事件。block=False 时如果另一个线程正在上传就直接返回；
        block=True 时等锁最多 timeout 秒。"""
        return flush_spool(self.rclone, self.job_id, self.spool_dir, self._lock, block, timeout)


def flush_spool(
    rclone: Optional[Rclone],
    job_id: str,
    spool_dir: str,
    lock: Optional[threading.Lock] = None,
    block: bool = True,
    timeout: Optional[float] = None,
) -> bool:
    if rclone is None:
        return False
    if lock is not None:
        if not block:
            acquired = lock.acquire(False)
        else:
            acquired = lock.acquire(True, -1 if timeout is None else max(0.0, timeout))
        if not acquired:
            return False
    try:
        try:
            if not os.listdir(spool_dir):
                return True
        except OSError:
            return True
        try:
            limit = rclone.small_timeout if timeout is None else timeout
            rclone.move_up(spool_dir, layout.events_prefix(job_id), timeout=limit, tag="events")
            return True
        except RcloneError as exc:
            diag("events", "event upload failed, kept locally: %s" % exc)
            return False
    finally:
        if lock is not None:
            lock.release()
