"""通过 nvidia-smi 查询 GPU。测试把假的 nvidia-smi 放在 PATH 上，或直接替换 query。

nvidia-smi 在驱动出问题时可能卡在内核里（D 状态），SIGKILL 也结束不了。所以：

- query 的总时长有上限（超时 + SIGKILL 后最多等 procs.KILL_WAIT_S），等不到就放弃那个进程；
- 训练期间的利用率采样放在单独的线程里（Sampler），主循环只读结果，处理取消、截止时间
  不会被一次卡住的查询拖住；有被放弃的 nvidia-smi 还没退出时暂停采样（利用率记为未知），
  不再一个接一个地堆积卡死的进程。
"""

from __future__ import annotations

import subprocess
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from dgpu.runner import procs
from dgpu.runner.util import Clock, diag

QUERY_ARGS = (
    "--query-gpu=index,name,utilization.gpu,memory.total",
    "--format=csv,noheader,nounits",
)

GpuList = List[Dict[str, Any]]
QueryFn = Callable[[float], Tuple[Optional[GpuList], str]]

HUNG_PREFIX = "nvidia-smi did not answer"

# 被放弃的 nvidia-smi（SIGKILL 之后仍未退出）
_HUNG = []  # type: List[subprocess.Popen]
_HUNG_LOCK = threading.Lock()


def hung_count() -> int:
    """仍未退出的被放弃的 nvidia-smi 数量（顺手回收已经退出的）。"""
    with _HUNG_LOCK:
        _HUNG[:] = [proc for proc in _HUNG if proc.poll() is None]
        return len(_HUNG)


def query(
    timeout: float = 15.0,
    binary: str = "nvidia-smi",
    kill: Optional[Callable[[subprocess.Popen], None]] = None,
    kill_wait: float = procs.KILL_WAIT_S,
) -> Tuple[Optional[GpuList], str]:
    """返回 (GPU 列表, 错误说明)。失败时列表为 None。总时长不超过 timeout + kill_wait。"""
    hung = []  # type: List[subprocess.Popen]

    def kill_and_track(proc: subprocess.Popen) -> None:
        hung.append(proc)
        (kill or procs.kill_group_of)(proc)

    try:
        result = procs.run_bounded(
            [binary] + list(QUERY_ARGS), timeout, kill_wait=kill_wait, kill=kill_and_track
        )
    except FileNotFoundError:
        return None, "nvidia-smi not found"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "cannot run nvidia-smi (%s)" % type(exc).__name__
    if result.timed_out:
        if result.abandoned:
            with _HUNG_LOCK:
                _HUNG.extend(hung)
            return None, "%s within %.0f s and ignored SIGKILL (stuck in the kernel?)" % (
                HUNG_PREFIX,
                timeout,
            )
        return None, "%s within %.0f s" % (HUNG_PREFIX, timeout)
    if result.rc != 0:
        detail = (result.out + result.err).decode("utf-8", "replace").strip()[:300]
        return None, "nvidia-smi exit code %s: %s" % (result.rc, detail)
    gpus = parse(result.out.decode("utf-8", "replace"))
    if not gpus:
        return None, "nvidia-smi lists no GPU"
    return gpus, ""


def parse(text: str) -> GpuList:
    gpus = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "util": _number(parts[2]),
                "memory_mb": _number(parts[3]),
            }
        )
    return gpus


def max_utilization(gpus: Optional[GpuList]) -> Optional[float]:
    """读得出利用率的卡里最高的利用率；一张都读不出来时返回 None（未知）。

    某张卡报 [N/A]（如 MIG 设备）不影响其他卡：只看读得出来的那些。
    """
    if not gpus:
        return None
    values = [gpu.get("util") for gpu in gpus if gpu.get("util") is not None]
    if not values:
        return None
    return max(values)


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None  # 如 "[N/A]"


class Sampler:
    """后台线程：每 interval 秒查询一次 GPU 利用率。

    util：最近一次的最高利用率（None 表示未知）；last_busy：最近一次利用率大于 0 的单调时间；
    sampled：这一轮（start 之后）至少成功采样过一次。查询卡住（超时）之后，在被放弃的
    nvidia-smi 退出之前不再查询，利用率记为未知（卡死检测因此不会触发，也不会误判）。
    """

    def __init__(
        self,
        query_fn: QueryFn,
        timeout: float,
        interval: float,
        clock: Optional[Clock] = None,
        note: Optional[Callable[[str], None]] = None,
        hung: Callable[[], int] = hung_count,
    ):
        self._query = query_fn
        self._timeout = timeout
        self._interval = max(0.01, interval)
        self._clock = clock or Clock()
        self._note = note or (lambda text: diag("gpu", text))
        self._hung = hung
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None  # type: Optional[threading.Thread]
        self.util = None  # type: Optional[float]
        self.last_busy = None  # type: Optional[float]
        self.sampled = False
        self.available = True

    def start(self) -> None:
        with self._lock:
            self.util = None
            self.last_busy = None
            self.sampled = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self, join_s: float = 0.0) -> None:
        """停止采样；不等正在进行的查询（它自己有上限）。"""
        self._stop.set()
        thread = self._thread
        if thread is not None and join_s > 0:
            thread.join(join_s)

    def snapshot(self) -> Tuple[Optional[float], Optional[float], bool]:
        with self._lock:
            return self.util, self.last_busy, self.sampled

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception as exc:  # 采样出错不影响训练
                diag("gpu", "GPU sampling error: %s" % exc)
            if self._stop.wait(self._interval):
                return

    def sample_once(self) -> None:
        if self._hung() > 0:
            if self.available:
                self.available = False
                self._note("GPU 利用率采样暂停：之前的 nvidia-smi 卡住了还没退出，利用率记为未知")
            with self._lock:
                self.util = None
            return
        gpus, error = self._query(self._timeout)
        now = self._clock.monotonic()
        util = max_utilization(gpus)
        if error.startswith(HUNG_PREFIX):
            if self.available:
                self.available = False
                self._note("GPU 利用率采样暂停：%s" % error)
        elif not self.available and gpus is not None:
            self.available = True
            self._note("GPU 利用率采样恢复")
        with self._lock:
            self.util = util
            if gpus is not None:
                self.sampled = True
            if util is not None and util > 0:
                self.last_busy = now
