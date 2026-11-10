"""任务状态、失败原因、事件名、环境变量名，以及存储中各 JSON 文档的格式。

冻结的契约：CLI 和容器内 runner 都依赖本文件。
只能用标准库，并保持兼容 Python 3.8（随 bundle 下发到实例）。
时间一律用 UTC 的 epoch 秒（float）。
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1

# ---------------------------------------------------------------- 任务阶段

PROVISIONING = "PROVISIONING"  # CLI 正在找机器、创建实例
STARTING = "STARTING"  # 实例已创建，runner 还没发出第一次心跳
RUNNING = "RUNNING"  # runner 已启动（含下载数据、setup、训练）
UPLOADING = "UPLOADING"  # 训练进程已结束，正在做最终同步
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
TIMED_OUT = "TIMED_OUT"
LOST = "LOST"  # 不写入存储；CLI 在心跳过期时推导

ACTIVE_PHASES = (PROVISIONING, STARTING, RUNNING, UPLOADING)
TERMINAL_PHASES = (SUCCEEDED, FAILED, CANCELLED, TIMED_OUT)

# status.json 由谁写：runner 发出第一次心跳之前由 CLI 写（PROVISIONING、STARTING，
# 以及启动阶段的 FAILED）；之后只由 runner / watchdog 写。

# ---------------------------------------------------------------- 失败原因（status.reason）

REASON_NO_OFFER = "no_offer"  # 没有符合条件的机器，或换机次数用尽
REASON_STARTUP = "startup"  # 实例一直没有进入运行状态
REASON_RUNNER_TIMEOUT = "runner_timeout"  # 实例运行了但没有心跳
REASON_PREFLIGHT = "preflight"  # 无 GPU、磁盘不足、连不上存储
REASON_DATA_SYNC = "data_sync"  # 代码或数据下载失败
REASON_SETUP = "setup"  # setup 命令非 0 退出
REASON_EXIT_CODE = "exit_code"  # run 命令非 0 退出
REASON_STALLED = "stalled"  # 卡死检测触发
REASON_RUNNER_CRASHED = "runner_crashed"  # runner 进程消失，由 watchdog 收尾
REASON_SIGNALLED = "signalled"  # runner 收到 SIGTERM / SIGINT（容器被停止等），提前收尾
REASON_FINAL_SYNC = "final_sync"  # 最终上传在时限内没有成功
REASON_FENCED = "fenced"  # 发现更新的 attempt，本 attempt 自行退出

# ---------------------------------------------------------------- 事件名（固定，可按名字查询）

EV_OFFER_REJECTED = "offer_rejected"
EV_CREATE_FAILED = "create_failed"
EV_STARTUP_TIMEOUT = "startup_timeout"
EV_RUNNER_TIMEOUT = "runner_timeout"
EV_PREFLIGHT_FAILED = "preflight_failed"
EV_DATA_SYNC_RETRY = "data_sync_retry"
EV_CKPT_RESTORED = "ckpt_restored"
EV_SYNC_FAILED = "sync_failed"
EV_LOG_TRUNCATED = "log_truncated"
EV_STALLED = "stalled"
EV_DEADLINE_HIT = "deadline_hit"
EV_FINAL_SYNC_FAILED = "final_sync_failed"
EV_SELF_DESTROY_FAILED = "self_destroy_failed"
EV_ATTEMPT_FENCED = "attempt_fenced"
EV_GC_DESTROYED = "gc_destroyed"
EV_INSTANCE_DESTROYED = "instance_destroyed"
EV_DATASET_UPLOADED = "dataset_uploaded"
EV_DESTROY_FAILED = "destroy_failed"  # CLI 一侧销毁实例失败或无法确认，实例可能仍在计费
EV_RELAUNCH = "relaunch"  # CLI 换机或重试：attempt + 1（runner 一侧发现后写 attempt_fenced）
# 训练进程和 runner 同一用户运行（没有降权）：凭据对训练进程可读。data.reason 说明原因
EV_RUN_AS_ROOT = "run_as_root"

EVENT_NAMES = (
    EV_OFFER_REJECTED,
    EV_CREATE_FAILED,
    EV_STARTUP_TIMEOUT,
    EV_RUNNER_TIMEOUT,
    EV_PREFLIGHT_FAILED,
    EV_DATA_SYNC_RETRY,
    EV_CKPT_RESTORED,
    EV_SYNC_FAILED,
    EV_LOG_TRUNCATED,
    EV_STALLED,
    EV_DEADLINE_HIT,
    EV_FINAL_SYNC_FAILED,
    EV_SELF_DESTROY_FAILED,
    EV_ATTEMPT_FENCED,
    EV_GC_DESTROYED,
    EV_INSTANCE_DESTROYED,
    EV_DATASET_UPLOADED,
    EV_DESTROY_FAILED,
    EV_RELAUNCH,
    EV_RUN_AS_ROOT,
)

# 事件来源（写进事件文件名和内容）
SRC_CLI = "cli"
SRC_RUNNER = "runner"
SRC_WATCHDOG = "watchdog"
SRC_GC = "gc"

# ---------------------------------------------------------------- 训练进程的环境变量

ENV_JOB_ID = "DGPU_JOB_ID"
ENV_ATTEMPT = "DGPU_ATTEMPT"
ENV_DATA = "DGPU_DATA"
ENV_OUTPUT = "DGPU_OUTPUT"
ENV_CKPT = "DGPU_CKPT"
ENV_RESUME = "DGPU_RESUME"

# 实例内的工作目录：<WORK_ROOT>/code、data、output、ckpt、logs
DEFAULT_WORK_ROOT = "/workspace/dgpu"

# runner 启动训练进程前从环境中移除的变量（前缀匹配）
SCRUBBED_ENV_PREFIXES = ("RCLONE_CONFIG_", "CONTAINER_API_KEY", "AWS_", "R2_")

# ---------------------------------------------------------------- `dgpu wait` 的退出码

EXIT_SUCCEEDED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_TIMED_OUT = 3
EXIT_CANCELLED = 4
EXIT_LOST = 5

# 不对应某个阶段的退出码（只有 CLI 用）
EXIT_CLEANUP_FAILED = 6  # 任务 SUCCEEDED，但结束后清理遗留实例失败（实例可能仍在计费）
EXIT_STORE_UNREACHABLE = 7  # wait 连续多分钟读不到存储，任务结果未知（任务本身不受影响）

EXIT_CODE_BY_PHASE = {
    SUCCEEDED: EXIT_SUCCEEDED,
    FAILED: EXIT_FAILED,
    TIMED_OUT: EXIT_TIMED_OUT,
    CANCELLED: EXIT_CANCELLED,
    LOST: EXIT_LOST,
}

# ---------------------------------------------------------------- JSON 文档构造函数
# 读取方遇到 version != SCHEMA_VERSION 时必须拒绝，不猜测。


def now() -> float:
    return time.time()


def _now(value: Optional[float]) -> float:
    return time.time() if value is None else value


def new_writer_token() -> str:
    """attempt.json 的 writer 字段：每次写入换一个随机值。

    条件写成功但响应丢失时，重试会收到“条件不满足”；重新读取后，存储里的文档带着自己
    这次写入的 writer 和内容，就说明那次写入其实已经生效。
    """
    return secrets.token_hex(8)


def new_attempt_doc(
    job_id: str,
    attempt: int,
    backend: str,
    label: str,
    instance_id: Optional[str] = None,
    deadline_at: Optional[float] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """attempt.json：当前有效的 attempt。只由 CLI 用条件写更新。

    runner 每次心跳读取它：attempt 比自己大，或 cancel 为 True，就停止并收尾。
    now：时间戳（默认当前时间），调用方有自己的时钟时传入。
    """
    return {
        "version": SCHEMA_VERSION,
        "job_id": job_id,
        "attempt": attempt,
        "backend": backend,
        "label": label,
        "instance_id": instance_id,
        "cancel": False,
        "created_at": _now(now),
        "deadline_at": deadline_at,  # 超过这个时间仍存在的实例可以被 gc 销毁
        "writer": new_writer_token(),  # 本次写入的随机标记，见 new_writer_token
    }


def new_status_doc(
    job_id: str,
    attempt: int,
    phase: str,
    source: str,
    reason: Optional[str] = None,
    exit_code: Optional[int] = None,
    message: Optional[str] = None,
    not_uploaded: Optional[list] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """status.json：任务阶段。"""
    return {
        "version": SCHEMA_VERSION,
        "job_id": job_id,
        "attempt": attempt,
        "phase": phase,
        "reason": reason,
        "exit_code": exit_code,  # run 或 setup 命令的退出码
        "message": message,  # 给人看的一句话
        "not_uploaded": not_uploaded or [],  # final_sync 失败时没传上去的文件
        "source": source,
        "updated_at": _now(now),
    }


def new_heartbeat_doc(
    job_id: str,
    attempt: int,
    phase: str,
    step: str,
    gpu_util: Optional[float] = None,
    last_log_at: Optional[float] = None,
    synced_bytes: int = 0,
    last_sync_at: Optional[float] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """heartbeat.json：runner 每个心跳周期覆盖写一次。

    step 是 runner 内部的细分步骤：preflight、fetch、restore、setup、run、final_sync。
    """
    return {
        "version": SCHEMA_VERSION,
        "job_id": job_id,
        "attempt": attempt,
        "phase": phase,
        "step": step,
        "ts": _now(now),
        "gpu_util": gpu_util,
        "last_log_at": last_log_at,
        "synced_bytes": synced_bytes,
        "last_sync_at": last_sync_at,
    }


def new_event_doc(
    name: str,
    source: str,
    job_id: str,
    attempt: Optional[int] = None,
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """events/ 下的一条事件。name 必须在 EVENT_NAMES 里。"""
    if name not in EVENT_NAMES:
        raise ValueError("unknown event name: %s" % name)
    return {
        "version": SCHEMA_VERSION,
        "name": name,
        "source": source,
        "job_id": job_id,
        "attempt": attempt,
        "ts": _now(now),
        "message": message,
        "data": data or {},
    }


def check_version(doc: Dict[str, Any], what: str) -> Dict[str, Any]:
    """校验文档版本，未知版本抛 ValueError。"""
    version = doc.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            "%s has unsupported version %r (expected %d)" % (what, version, SCHEMA_VERSION)
        )
    return doc
