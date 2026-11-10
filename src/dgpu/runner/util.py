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
    # 硬截止 = max_hours + runner 最坏收尾时间（teardown_worst_case_s）+ 这个余量；余量覆盖
    # 公式里没算的零碎等待（SIGKILL 后等进程消失、每次存储操作至少 1 秒等）
    "watchdog_margin_s": 300.0,
    "watchdog_spec_wait_s": 1800.0,  # runner 一直没读到 spec 时，watchdog 最多等多久
    "watchdog_sync_timeout_s": 600.0,  # runner 崩溃或硬截止后 watchdog 尽力同步一次的时限
    "boot_watchdog_wait_s": 1800.0,  # AutoDL 前台模式：runner 退出后最多等 watchdog 多久
    "boot_watchdog_respawns": 5,  # AutoDL 前台模式：watchdog 意外退出时 boot 最多重启它几次
    "max_not_uploaded": 1000,  # status.not_uploaded 最多列多少个文件
    "log_tail_kb": 1024.0,  # 日志超过上限后保留的训练输出尾部
    "log_note_reserve_kb": 1024.0,  # 日志超过上限后仍留给 runner 进度行的空间
    "diag_log_tail_kb": 256.0,  # 上传 runner.log、watchdog.log、bootstrap.log 的最后多少 KB
    "events_flush_interval_s": 30.0,  # 后台上传事件的最长间隔（写事件时立即唤醒）
    "privsep_probe_timeout_s": 60.0,  # 以无特权用户试运行 python3 的时限（见 privsep.py）
    "kill_wait_s": 5.0,  # 结束进程时 SIGKILL 之后最多等多久（之后放弃，见 procs.py）
}

# SIGKILL 之后最多等进程消失多久（进程卡在内核里时放弃等待，见 procs.py）
KILL_WAIT_S = 5.0

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


# ---------------------------------------------------------------- 时限（runner、watchdog、CLI）


def destroy_worst_case_s(tuning: Dict[str, Any]) -> float:
    """自毁最多用多久：每次 HTTP 请求的时限，加上两次之间的间隔。"""
    attempts = max(1, int(tuning["destroy_attempts"]))
    return (
        attempts * tuning["destroy_http_timeout_s"] + (attempts - 1) * tuning["destroy_interval_s"]
    )


def short_op_s(tuning: Dict[str, Any]) -> float:
    """收尾时单个“小”存储操作（UPLOADING 状态、读 attempt.json）的时限。"""
    return min(30.0, tuning["small_op_timeout_s"])


def teardown_worst_case_s(settings: Dict[str, float], tuning: Dict[str, Any]) -> float:
    """runner 从决定停止（如到达 max_hours）到自毁结束最多用多久：各步骤时限之和。

    结束训练进程（term_grace_s，SIGKILL 后最多等 kill_wait_s，再最多等 kill_wait_s 回收）、
    停后台线程（同步、心跳、事件上传）、写 UPLOADING 和最终同步前读 attempt.json、最终同步
    （final_sync_timeout，output/ 和 ckpt/ 共用）、传日志事件和写最终状态（teardown_budget_s）、
    自毁。没上传的文件按本地的同步记录计算，不访问存储。
    """
    return (
        settings["term_grace_s"]
        + 2 * tuning["kill_wait_s"]
        + 3 * tuning["thread_join_s"]
        + 2 * short_op_s(tuning)
        + settings["final_sync_timeout_min"] * 60.0
        + tuning["teardown_budget_s"]
        + destroy_worst_case_s(tuning)
    )


def watchdog_hard_deadline_s(
    max_hours: float, settings: Dict[str, float], tuning: Dict[str, Any]
) -> float:
    """watchdog 的硬截止（从第一次启动算起）：总比 runner 最坏的收尾多出 watchdog_margin_s。"""
    return (
        max_hours * 3600.0 + teardown_worst_case_s(settings, tuning) + tuning["watchdog_margin_s"]
    )


def watchdog_kill_grace_s(settings: Dict[str, float]) -> float:
    """watchdog 结束训练进程时给的 SIGTERM 宽限。"""
    return min(settings["term_grace_s"], 10.0)


def watchdog_sync_s(
    settings: Dict[str, float], tuning: Dict[str, Any], restart: bool = False
) -> float:
    """watchdog 尽力同步 output/、ckpt/ 的总时限（两个目录共用）。

    平时 runner 已经用过自己的最终同步时限，watchdog 只再试一次，取 watchdog_sync_timeout_s
    和 final_sync_timeout_min 的小者。restart：容器重启时 runner 自己的截止（max_hours）已经
    过了，runner 来不及在 watchdog 截止之前做完收尾，这时 watchdog 接手给和 runner 一样的
    最终同步时限 final_sync_timeout_min，失败就重试到时限（评审 F6）。这是 watchdog 同步的
    上限，计入 watchdog_teardown_worst_case_s 和实例存活时限。
    """
    final = settings["final_sync_timeout_min"] * 60.0
    if restart:
        return max(1.0, final)
    return max(1.0, min(tuning["watchdog_sync_timeout_s"], final))


def watchdog_teardown_worst_case_s(settings: Dict[str, float], tuning: Dict[str, Any]) -> float:
    """watchdog 接手后（runner 消失或到达硬截止）最多用多久。

    结束 runner、训练进程和 rclone（宽限 + SIGKILL 后的等待）、读 attempt.json、同步（按两种
    情况里较长的那个：容器重启后接手时给满 final_sync_timeout_min）、传日志和事件并写状态
    （teardown_budget_s）、自毁。每一步都有时限。
    """
    return (
        watchdog_kill_grace_s(settings)
        + 2 * tuning["kill_wait_s"]
        + short_op_s(tuning)
        + watchdog_sync_s(settings, tuning, restart=True)
        + 2.0  # 每个目录的复制至少给 1 秒
        + tuning["teardown_budget_s"]
        + destroy_worst_case_s(tuning)
    )


def instance_lifetime_s(
    max_hours: float, settings: Dict[str, float], tuning: Dict[str, Any]
) -> float:
    """runner 启动之后，实例最晚在多久之内自己消失（watchdog 硬截止 + watchdog 收尾）。"""
    return watchdog_hard_deadline_s(max_hours, settings, tuning) + watchdog_teardown_worst_case_s(
        settings, tuning
    )


def boot_hard_deadline_s(
    max_hours: float, settings: Dict[str, float], tuning: Dict[str, Any]
) -> float:
    """AutoDL 前台的 boot 自己的硬截止（从第一次启动算起）：watchdog 应该已经收尾完的时间
    再加一个余量。到了就结束一切并退出，容器随之释放（watchdog 也死掉时的最后一层）。"""
    return instance_lifetime_s(max_hours, settings, tuning) + tuning["watchdog_margin_s"]


def boot_spec_wait_s(tuning: Dict[str, Any]) -> float:
    """runner 一直没写出 run/spec.json 时 boot 最多等多久：watchdog 等 spec 的时间加它的收尾。"""
    settings = runner_settings({})
    return (
        tuning["watchdog_spec_wait_s"]
        + watchdog_teardown_worst_case_s(settings, tuning)
        + tuning["watchdog_margin_s"]
    )


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
      home/                  无特权的训练用户的 HOME（见 privsep.py）
      logs/attempt-<n>.log   训练输出和 runner 的进度行，同步到存储的 logs/attempt-<n>.log
      logs/{bootstrap,runner,watchdog}.log   各进程的诊断输出，尾部随日志上传
                             （logs/attempt-<n>.<组件>.log；boot 只写 stderr，即平台日志）
      run/                   本次尝试的运行状态，见下面的属性

    run/、logs/、launch.json、bundle-*/ 只有 runner 的用户（root）能读；训练进程降权运行时
    读不到凭据，也碰不到 runner 和 watchdog 的状态。
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
        self.home = join(self.root, "home")
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
        self.diag_upload = join(self.run, "diag")  # 上传前截取的诊断日志尾部
        self.privsep = join(self.run, "privsep.json")  # 训练进程以什么身份运行

    def attempt_log(self, attempt: int) -> str:
        return os.path.join(self.logs, "attempt-%d.log" % attempt)

    def event_spool(self, source: str) -> str:
        return os.path.join(self.run, "events-%s" % source)

    def rclone_pids(self, component: str) -> str:
        """component（runner / watchdog）正在运行的 rclone 的登记目录。"""
        return os.path.join(self.run, "rclone-%s" % component)

    def diag_log(self, component: str) -> str:
        return os.path.join(self.logs, "%s.log" % component)

    def ready_marker(self, item: str) -> str:
        return os.path.join(self.ready, item + ".ready")

    def ensure(self) -> None:
        for path in (self.logs, self.run, self.ready, self.events_tmp):
            os.makedirs(path, exist_ok=True)
        # 运行状态和诊断日志只给 runner 的用户（root）看
        for path in (self.run, self.logs):
            try:
                os.chmod(path, 0o700)
            except OSError:
                pass
        # 训练的目录提前建好：代码下载失败时最终同步也不会对着不存在的目录重试
        for path in (self.output, self.ckpt, self.data):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                pass


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
