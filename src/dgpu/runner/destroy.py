"""实例自毁（设计 4.8 第 1、2 层）。

- Vast：用容器里的 CONTAINER_ID 和只能操作本实例的 CONTAINER_API_KEY 调
  DELETE /api/v0/instances/<CONTAINER_ID>/。永远不用用户的主 key（它不在实例里）。
  每 30 秒重试一次，共 10 次。
- AutoDL：没有可调的接口。弹性部署的容器只在 cmd 运行期间存在，所有进程退出即释放，
  所以这里什么都不调，由调用方结束进程。

用容器 key 调 REST 删除实例在文档里没有写明（文档只写了 vastai CLI），要靠探针 P1 验证。
如果不行，在 SelfDestroyer._vast_methods 里追加 vastai CLI 方式即可，调用方不用改。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from dgpu.runner.util import diag

VAST_API_BASE = "https://console.vast.ai"
BACKEND_VAST = "vast"
BACKEND_AUTODL = "autodl"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

HttpFn = Callable[[str, str, Dict[str, str], float], Tuple[int, bytes]]


class DestroyError(Exception):
    pass


class DestroyResult:
    def __init__(self, ok: bool, attempts: int, method: Optional[str], error: Optional[str]):
        self.ok = ok
        self.attempts = attempts
        self.method = method
        self.error = error

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "attempts": self.attempts,
            "method": self.method,
            "error": self.error,
        }


def _parse_environ_bytes(data: bytes) -> Dict[str, str]:
    env = {}
    for item in data.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def _parse_env_file(text: str) -> Dict[str, str]:
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if sep:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            env[key.strip()] = value
    return env


def read_container_credentials(
    environ: Optional[Dict[str, str]] = None,
    proc_environ: str = "/proc/1/environ",
    etc_environment: str = "/etc/environment",
) -> Tuple[Optional[str], Optional[str]]:
    """(CONTAINER_ID, CONTAINER_API_KEY)。

    依次看本进程环境、容器 1 号进程的环境（docker 注入的变量一定在那里）、
    /etc/environment（Vast 文档建议 SSH 会话从这里导出）。
    """
    sources = [dict(os.environ if environ is None else environ)]
    try:
        with open(proc_environ, "rb") as handle:
            sources.append(_parse_environ_bytes(handle.read()))
    except OSError:
        pass
    try:
        with open(etc_environment, encoding="utf-8", errors="replace") as handle:
            sources.append(_parse_env_file(handle.read()))
    except OSError:
        pass
    for env in sources:
        cid = env.get("CONTAINER_ID")
        key = env.get("CONTAINER_API_KEY")
        if cid and key:
            return cid.strip(), key.strip()
    return None, None


def urllib_http(
    method: str, url: str, headers: Dict[str, str], timeout: float
) -> Tuple[int, bytes]:
    """发 HTTP 请求，返回 (状态码, 响应体前 64 KB)。网络错误抛异常。

    只使用环境变量里的代理设置，不读 macOS 的系统代理。
    """
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies_environment())
    )
    request = urllib.request.Request(url, method=method, headers=headers)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(65536)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(65536)
        except Exception:
            body = b""
        return exc.code, body


class SelfDestroyer:
    def __init__(
        self,
        backend: str,
        http: Optional[HttpFn] = None,
        credentials: Optional[Callable[[], Tuple[Optional[str], Optional[str]]]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        attempts: int = 10,
        interval_s: float = 30.0,
        api_base: str = VAST_API_BASE,
        timeout_s: float = 30.0,
        component: str = "destroy",
    ):
        self.backend = backend
        self._http = http or urllib_http
        self._credentials = credentials or read_container_credentials
        self._sleep = sleep or time.sleep
        self.attempts = max(1, int(attempts))
        self.interval_s = interval_s
        self.api_base = api_base.rstrip("/")
        self.timeout_s = timeout_s
        self.component = component

    def destroy(self) -> DestroyResult:
        if self.backend == BACKEND_AUTODL:
            # 没有接口可调：调用方结束所有进程后，AutoDL 释放容器
            return DestroyResult(True, 0, "process_exit", None)
        if self.backend != BACKEND_VAST:
            return DestroyResult(False, 0, None, "unknown backend %r" % self.backend)
        last_error = None
        for attempt in range(1, self.attempts + 1):
            for name, method in self._vast_methods():
                try:
                    method()
                    diag(self.component, "self-destroy accepted (%s, attempt %d)" % (name, attempt))
                    return DestroyResult(True, attempt, name, None)
                except Exception as exc:
                    last_error = "%s: %s" % (name, exc)
                    diag(
                        self.component, "self-destroy attempt %d failed: %s" % (attempt, last_error)
                    )
            if attempt < self.attempts:
                self._sleep(self.interval_s)
        return DestroyResult(False, self.attempts, None, last_error)

    def _vast_methods(self) -> List[Tuple[str, Callable[[], None]]]:
        # 探针 P1 之后如需改用 CLI：追加 ("vastai_cli", self._vast_cli)，按顺序尝试。
        return [("vast_rest", self._vast_rest)]

    def _vast_rest(self) -> None:
        cid, key = self._credentials()
        if not cid or not key:
            raise DestroyError("CONTAINER_ID / CONTAINER_API_KEY not found in the container")
        if not _ID_RE.match(cid):
            raise DestroyError("CONTAINER_ID has an unexpected format")
        url = "%s/api/v0/instances/%s/" % (self.api_base, cid)
        headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
        try:
            status, body = self._http("DELETE", url, headers, self.timeout_s)
        except Exception as exc:
            raise DestroyError("request failed (%s)" % type(exc).__name__) from None
        text = body.decode("utf-8", "replace").replace(key, "***")[:300]
        if 200 <= status < 300:
            try:
                doc = json.loads(text) if text.strip() else {}
            except ValueError:
                doc = {}
            if isinstance(doc, dict) and doc.get("success") is False:
                raise DestroyError("HTTP %d but success=false: %s" % (status, text))
            return
        # 404 不当作成功：实例真的没了的话这个容器也不会还在运行；更可能是路径或权限不对
        raise DestroyError("HTTP %d: %s" % (status, text))
