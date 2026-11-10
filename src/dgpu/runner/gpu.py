"""通过 nvidia-smi 查询 GPU。测试把假的 nvidia-smi 放在 PATH 上，或直接替换 query。"""

from __future__ import annotations

import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

QUERY_ARGS = (
    "--query-gpu=index,name,utilization.gpu,memory.total",
    "--format=csv,noheader,nounits",
)

GpuList = List[Dict[str, Any]]
QueryFn = Callable[[float], Tuple[Optional[GpuList], str]]


def query(timeout: float = 15.0, binary: str = "nvidia-smi") -> Tuple[Optional[GpuList], str]:
    """返回 (GPU 列表, 错误说明)。失败时列表为 None。"""
    try:
        proc = subprocess.run(
            [binary] + list(QUERY_ARGS),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            close_fds=True,
        )
    except FileNotFoundError:
        return None, "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi did not answer within %.0f s" % timeout
    except OSError as exc:
        return None, "cannot run nvidia-smi (%s)" % type(exc).__name__
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).decode("utf-8", "replace").strip()[:300]
        return None, "nvidia-smi exit code %d: %s" % (proc.returncode, detail)
    gpus = parse(proc.stdout.decode("utf-8", "replace"))
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
    """所有卡里最高的利用率；任何一张卡读不出来就返回 None（未知）。"""
    if not gpus:
        return None
    values = [gpu.get("util") for gpu in gpus]
    if any(value is None for value in values):
        return None
    return max(values)


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None  # 如 "[N/A]"
