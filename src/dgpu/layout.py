"""存储中的路径约定、任务 ID 和实例标签。

冻结的契约：CLI 和容器内 runner 都依赖本文件。
只能用标准库，并保持兼容 Python 3.8（随 bundle 下发到实例）。
所有 key 都是相对 bucket 的路径，不带开头的斜杠。
"""

from __future__ import annotations

import re
import secrets
import time
from typing import Optional, Tuple

ROOT = "dgpu"

_JOB_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LABEL_RE = re.compile(r"^dgpu-(?P<job_id>[a-z0-9]+(?:-[a-z0-9]+)*)-a(?P<attempt>\d+)$")
_NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+")
# new_job_id 生成的形状：<名字>-<yymmddHHMMSS>-<4 位小写十六进制>
_GENERATED_JOB_ID_RE = re.compile(
    r"^(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)-(?P<stamp>\d{12})-(?P<rand>[0-9a-f]{4})$"
)

LABEL_PREFIX = "dgpu-"
MAX_NAME_LEN = 24


# ---------------------------------------------------------------- 任务 ID 与标签


def new_job_id(name: str, now: Optional[float] = None) -> str:
    """生成任务 ID：<名字>-<UTC 时间 yymmddHHMMSS>-<4 位随机十六进制>。

    只含小写字母、数字和连字符，可以直接用在存储路径和平台标签里。
    """
    slug = _NAME_CLEAN_RE.sub("-", name.lower()).strip("-")[:MAX_NAME_LEN].strip("-") or "job"
    stamp = time.strftime("%y%m%d%H%M%S", time.gmtime(now if now is not None else time.time()))
    return "%s-%s-%s" % (slug, stamp, secrets.token_hex(2))


def is_valid_job_id(job_id: str) -> bool:
    return bool(_JOB_ID_RE.match(job_id))


def is_generated_job_id(job_id: Optional[str]) -> bool:
    """job_id 是否正是 new_job_id 生成的形状（比 is_valid_job_id 严格）。

    <名字>-<UTC 时间 yymmddHHMMSS>-<4 位十六进制>，名字不超过 MAX_NAME_LEN 个字符，
    时间的月、日、时、分、秒在合法范围内。gc 只处理这种任务 ID 的标签：手工起名的
    dgpu-xxx-a1 之类的实例不碰。
    """
    if not job_id:
        return False
    match = _GENERATED_JOB_ID_RE.match(job_id)
    if not match or len(match.group("slug")) > MAX_NAME_LEN:
        return False
    stamp = match.group("stamp")
    month, day = int(stamp[2:4]), int(stamp[4:6])
    hour, minute, second = int(stamp[6:8]), int(stamp[8:10]), int(stamp[10:12])
    return 1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60 and second < 61


def label(job_id: str, attempt: int) -> str:
    """实例标签，两个平台通用：Vast 的 label 字段、AutoDL 的部署名。

    gc 只处理标签以 LABEL_PREFIX 开头且能被 parse_label 解析的实例。
    """
    if not is_valid_job_id(job_id):
        raise ValueError("invalid job id: %r" % job_id)
    if attempt < 1:
        raise ValueError("attempt starts at 1")
    return "dgpu-%s-a%d" % (job_id, attempt)


def parse_label(text: Optional[str]) -> Optional[Tuple[str, int]]:
    """解析标签，返回 (job_id, attempt)；不是 dgpu 的标签返回 None。"""
    if not text:
        return None
    match = _LABEL_RE.match(text)
    if not match:
        return None
    return match.group("job_id"), int(match.group("attempt"))


# ---------------------------------------------------------------- 路径


def tools_bundle_key(sha256: str) -> str:
    return "%s/tools/bundle-%s.tar.gz" % (ROOT, sha256)


def dataset_prefix(content_hash: str) -> str:
    """从本地路径上传的数据集目录（以 / 结尾）。"""
    return "%s/datasets/%s/" % (ROOT, content_hash)


def dataset_complete_key(content_hash: str) -> str:
    """数据集上传完成标记，所有文件传完后最后写。"""
    return "%s/datasets/%s.complete" % (ROOT, content_hash)


def jobs_prefix() -> str:
    return "%s/jobs/" % ROOT


def job_prefix(job_id: str) -> str:
    return "%s/jobs/%s/" % (ROOT, job_id)


def spec_key(job_id: str) -> str:
    return job_prefix(job_id) + "spec.json"


def code_key(job_id: str) -> str:
    return job_prefix(job_id) + "code.tar.gz"


def attempt_key(job_id: str) -> str:
    return job_prefix(job_id) + "attempt.json"


def request_key(job_id: str) -> str:
    """GPU 需求（型号、卡数、价格上限、时长、磁盘、镜像），CLI 在 retry 换机时重新搜索用。"""
    return job_prefix(job_id) + "request.json"


def secrets_key(job_id: str, attempt: int) -> str:
    """本次尝试的启动文档（含存储凭据），只通过预签名 URL 下发。"""
    return job_prefix(job_id) + "secrets-%d.json" % attempt


def status_key(job_id: str) -> str:
    return job_prefix(job_id) + "status.json"


def heartbeat_key(job_id: str) -> str:
    return job_prefix(job_id) + "heartbeat.json"


def events_prefix(job_id: str) -> str:
    return job_prefix(job_id) + "events/"


def event_key(job_id: str, source: str, ts: Optional[float] = None) -> str:
    """事件文件名按时间排序：<毫秒时间戳>-<来源>-<随机>.json。"""
    millis = int((ts if ts is not None else time.time()) * 1000)
    return "%s%013d-%s-%s.json" % (events_prefix(job_id), millis, source, secrets.token_hex(3))


def logs_prefix(job_id: str) -> str:
    return job_prefix(job_id) + "logs/"


def log_key(job_id: str, attempt: int) -> str:
    return "%sattempt-%d.log" % (logs_prefix(job_id), attempt)


# 实例上各进程的诊断日志（尾部），和 attempt 日志放在一起，排查 runner 自身的问题用
DIAG_COMPONENTS = ("runner", "watchdog", "bootstrap")


def diag_log_key(job_id: str, attempt: int, component: str) -> str:
    """logs/attempt-<n>.<组件>.log：runner、watchdog、bootstrap 的诊断输出的最后一段。"""
    if component not in DIAG_COMPONENTS:
        raise ValueError("unknown component: %r" % component)
    return "%sattempt-%d.%s.log" % (logs_prefix(job_id), attempt, component)


def output_prefix(job_id: str) -> str:
    return job_prefix(job_id) + "output/"


def ckpt_prefix(job_id: str) -> str:
    return job_prefix(job_id) + "ckpt/"
