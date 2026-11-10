"""事件：先原子地写进本地队列目录，再用 rclone move 上传（上传成功才删除本地文件）。

存储断开期间产生的事件不会丢，恢复后随下一次 flush 补传。runner 和 watchdog 各用
自己的队列目录，避免两个 rclone move 同时处理同一批文件。
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
            self.flush(block=False)

    def pending(self) -> int:
        try:
            return len(os.listdir(self.spool_dir))
        except OSError:
            return 0

    def flush(self, block: bool = False, timeout: Optional[float] = None) -> bool:
        """上传队列里的事件。block=False 时如果另一个线程正在上传就直接返回。"""
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
    if lock is not None and not lock.acquire(block):
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
