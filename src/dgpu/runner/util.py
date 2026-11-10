"""runner 各进程共用的部件：时钟、调参、工作目录布局、JSON 文件、诊断输出。

只用标准库，兼容 Python 3.8。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

from dgpu import states

# 环境变量：工作目录、rclone 路径、调参（JSON），由 bootstrap / 测试设置
WORK_ROOT_ENV = "DGPU_WORK_ROOT"
RCLONE_ENV = "DGPU_RCLONE"
TUNING_ENV = "DGPU_RUNNER_TUNING"
INTERNAL_ENV = (WORK_ROOT_ENV, RCLONE_ENV, TUNING_ENV)

# 不在 spec.json 里的内部参数。生产环境用默认值；测试通过 DGPU_RUNNER_TUNING 缩短。
DEFAULT_TUNING = {
    "tick_s": 0.5,  # 主循环检查子进程、截止时间、取消标志的间隔
    "rclone_poll_s": 0.2,  # rclone 子进程检查中断标志的间隔
    "retry_backoff_s": 10.0,  # 启动阶段和数据下载重试的退避基数（指数增长）
    "retry_backoff_max_s": 120.0,
    "startup_attempts": 6,  # 读 spec / attempt、写 RUNNING 状态和第一次心跳的尝试次数
    "status_attempts": 5,  # 写最终状态的尝试次数
    "small_op_timeout_s": 120.0,  # cat / rcat / lsjson / size 的单次时限
    "transfer_timeout_s": 6 * 3600.0,  # 下载一份代码或数据的单次时限
    "periodic_sync_timeout_s": 3600.0,  # 周期同步一个目录的单次时限
    "final_sync_backoff_s": 15.0,  # 最终同步失败后的重试间隔
    "sync_failed_event_interval_s": 600.0,  # 连续失败时 sync_failed 事件的最小间隔
    "gpu_sample_max_s": 30.0,  # GPU 利用率采样间隔上限（另受心跳间隔约束）
    "nvidia_smi_timeout_s": 15.0,
    "min_free_margin_mb": 1024.0,  # 磁盘检查：数据和 checkpoint 之外至少留的余量
    "lingering_grace_s": 5.0,  # 训练进程退出后清理残留子进程的 SIGTERM 宽限
    "thread_join_s": 10.0,  # 收尾时等后台线程停下的时间
    "teardown_budget_s": 300.0,  # 最终同步之后写状态、传日志、传事件的总时限（不含自毁）
    "destroy_attempts": 10,
    "destroy_interval_s": 30.0,
    "destroy_http_timeout_s": 30.0,
    "vast_api_base": "https://console.vast.ai",
    "watchdog_poll_s": 2.0,
    "watchdog_margin_s": 600.0,  # 硬截止 = max_hours + final_sync_timeout + 这个余量
    "watchdog_spec_wait_s": 1800.0,  # runner 一直没读到 spec 时，watchdog 最多等多久
    "watchdog_sync_timeout_s": 600.0,  # runner 崩溃后 watchdog 尽力同步一次的时限
    "boot_watchdog_wait_s": 1800.0,  # AutoDL 前台模式：runner 退出后最多等 watchdog 多久
    "max_not_uploaded": 1000,  # status.not_uploaded 最多列多少个文件
}

# spec.json 的 runner 段缺字段时的默认值（与 contracts.RunnerSettings 一致；
# runner 不能 import contracts.py）。这里都按浮点数读，测试可以用小数。
RUNNER_DEFAULTS = {
    "sync_interval_s": 120.0,
    "heartbeat_interval_s": 60.0,
    "stall_timeout_min": 60.0,
    "final_sync_timeout_min": 20.0,
    "term_grace_s": 60.0,
    "max_log_mb": 200.0,
    "data_sync_retries": 3.0,
}


class Clock:
    """可替换的时钟。所有等待都可被 threading.Event 打断。"""

    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float, stop: Optional[threading.Event] = None) -> bool:
        """等待 seconds 秒；stop 被置位时提前返回 True。"""
        if stop is None:
            if seconds > 0:
                time.sleep(seconds)
            return False
        if seconds <= 0:
            return stop.is_set()
        return stop.wait(seconds)


def diag(component: str, message: str) -> None:
    """写一行诊断信息到 stderr（boot 把它重定向到 logs/<组件>.log）。"""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        sys.stderr.write("%s [%s] %s\n" % (stamp, component, message))
        sys.stderr.flush()
    except Exception:
        pass


def load_tuning(environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """默认调参，叠加 DGPU_RUNNER_TUNING（JSON）。格式错误或未知键只记录，不中断。"""
    tuning = dict(DEFAULT_TUNING)
    env = os.environ if environ is None else environ
    raw = env.get(TUNING_ENV)
    if not raw:
        return tuning
    try:
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise ValueError("not a JSON object")
    except ValueError as exc:
        diag("tuning", "ignoring %s: %s" % (TUNING_ENV, exc))
        return tuning
    for key, value in overrides.items():
        if key not in DEFAULT_TUNING:
            diag("tuning", "ignoring unknown key %r" % key)
            continue
        default = DEFAULT_TUNING[key]
        try:
            if isinstance(default, bool):
                tuning[key] = bool(value)
            elif isinstance(default, int):
                tuning[key] = int(value)
            elif isinstance(default, float):
                tuning[key] = float(value)
            else:
                tuning[key] = str(value)
        except (TypeError, ValueError):
            diag("tuning", "ignoring bad value for %r" % key)
    return tuning


def runner_settings(spec: Dict[str, Any]) -> Dict[str, float]:
    """spec.json 的 runner 段，缺的字段用默认值，全部转成 float。"""
    raw = spec.get("runner") or {}
    settings = {}
    for key, default in RUNNER_DEFAULTS.items():
        try:
            settings[key] = float(raw.get(key, default))
        except (TypeError, ValueError):
            settings[key] = default
    return settings


def backoff(base: float, attempt: int, cap: float) -> float:
    """第 attempt 次（从 0 开始）重试前的等待时间。"""
    return min(cap, base * (2 ** min(attempt, 16)))


class Paths:
    """实例上工作目录的布局（DGPU_WORK_ROOT，默认 states.DEFAULT_WORK_ROOT）。

    <root>/
      boot.lock              bootstrap、runner、watchdog 共同持有的锁，保证同时只有一套在跑
      launch.json            启动文档（含存储凭据，0600）
      bundle.tgz             下载的 bundle；bundle-<sha256>/ 是解压结果（dgpu/、bin/rclone）
      code/                  训练代码，setup 和 run 的工作目录
      data/<name>/           DGPU_DATA
      output/                DGPU_OUTPUT
      ckpt/                  DGPU_CKPT
      logs/attempt-<n>.log   训练输出和 runner 的进度行，同步到存储的 logs/attempt-<n>.log
      logs/{bootstrap,runner,watchdog}.log   各进程的诊断输出（boot 只写 stderr，即平台日志）
      run/                   本次尝试的运行状态，见下面的属性
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        join = os.path.join
        self.launch = join(self.root, "launch.json")
        self.boot_lock = join(self.root, "boot.lock")
        self.code = join(self.root, "code")
        self.data = join(self.root, "data")
        self.output = join(self.root, "output")
        self.ckpt = join(self.root, "ckpt")
        self.logs = join(self.root, "logs")
        self.run = join(self.root, "run")
        self.rclone_conf = join(self.run, "rclone.conf")  # 空文件：凭据走环境变量
        self.spec_copy = join(self.run, "spec.json")  # runner 读到的 spec，供 watchdog 使用
        self.started = join(self.run, "started.json")  # 第一次启动的时间（容器重启后沿用）
        self.runner_pid = join(self.run, "runner.pid")
        self.watchdog_pid = join(self.run, "watchdog.pid")
        self.runner_alive = join(self.run, "runner.alive")  # runner 存活期间持有的 flock
        self.child = join(self.run, "child.json")  # 正在运行的 setup/run 的进程组
        self.fenced = join(self.run, "fenced")  # 发现更新的 attempt 后写入
        self.final = join(self.run, "final.json")  # runner 的最终结论和状态是否已写入存储
        self.destroyed = join(self.run, "destroyed.json")  # 自毁步骤的结果
        self.ready = join(self.run, "ready")  # 代码和每份数据下载完成的 .ready 标记
        self.code_archive = join(self.run, "code.tar.gz")
        self.events_tmp = join(self.run, "events.tmp")

    def attempt_log(self, attempt: int) -> str:
        return os.path.join(self.logs, "attempt-%d.log" % attempt)

    def event_spool(self, source: str) -> str:
        return os.path.join(self.run, "events-%s" % source)

    def ready_marker(self, item: str) -> str:
        return os.path.join(self.ready, item + ".ready")

    def ensure(self) -> None:
        for path in (self.logs, self.run, self.ready, self.events_tmp):
            os.makedirs(path, exist_ok=True)


def write_json_atomic(path: str, obj: Any, mode: int = 0o644) -> None:
    tmp = "%s.tmp-%d-%d" % (path, os.getpid(), threading.get_ident())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    """读 JSON 文件；不存在或损坏返回 None。"""
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def load_launch(path: str) -> Dict[str, Any]:
    """读取并校验启动文档（contracts.build_launch_doc 的格式）。不合法抛 ValueError。"""
    doc = read_json(path)
    if doc is None:
        raise ValueError("launch document %s is missing or not valid JSON" % path)
    states.check_version(doc, "launch document")
    for key in ("job_id", "attempt", "backend", "store"):
        if key not in doc:
            raise ValueError("launch document has no %r" % key)
    store = doc["store"]
    if not isinstance(store, dict):
        raise ValueError("launch document store is not an object")
    for key in ("endpoint", "bucket", "access_key_id", "secret_access_key"):
        if not store.get(key):
            raise ValueError("launch document store has no %r" % key)
    doc["attempt"] = int(doc["attempt"])
    return doc


def guess_backend(environ: Optional[Dict[str, str]] = None) -> str:
    """启动文档读不出来时的兜底：有 Vast 的容器变量就当 Vast，否则按 AutoDL（结束进程即释放）。"""
    env = os.environ if environ is None else environ
    if env.get("CONTAINER_ID") or env.get("CONTAINER_API_KEY"):
        return "vast"
    return "autodl"


def exc_text(exc: BaseException, limit: int = 300) -> str:
    text = "%s: %s" % (type(exc).__name__, exc)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def scrubbed(name: str) -> bool:
    """是否属于要从训练进程环境中移除的变量（states.SCRUBBED_ENV_PREFIXES，大小写不敏感）。"""
    upper = name.upper()
    return any(upper.startswith(prefix) for prefix in states.SCRUBBED_ENV_PREFIXES)
