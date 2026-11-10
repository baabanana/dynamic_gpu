"""S3 兼容存储（Cloudflare R2 等）上的 Store 实现（工作包 B）。

行为与 dgpu.testing.MemoryStore 一致，另外针对 R2 做了这些处理：

- region 用 profile.region（R2 是 ``auto``），SigV4、path-style 寻址；
  新版 boto3 默认附加的 CRC 校验改为 ``when_required``，避免 S3 兼容存储拒收。
- 所有上传都是单段 PutObject，并带 Content-MD5 让服务端校验内容。R2 的分段上传
  ETag 不可靠，条件写和完整性都依赖单段 ETag。单段上限 5 GiB，超过直接报错。
- 条件写用 If-Match / If-None-Match，412 和 409 都映射为 PreconditionFailed。
  条件写如果经过了重试，前一次请求可能已经生效，异常信息里会写明，调用方应重新读取。
- R2 同一个 key 每秒最多写 1 次，超出返回 429；botocore 不重试的 429 在这里按退避重试。
- R2 的 ListObjects 结果不保证有序，list / list_dirs 自己排序。
- 重试耗尽后的错误统一抛 StoreError，信息里有操作、key 和排查方向，不含凭据。
"""

from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TypeVar

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    HTTPClientError,
    IncompleteReadError,
    NoCredentialsError,
    PartialCredentialsError,
)
from botocore.exceptions import (
    ConnectionError as BotoConnectionError,
)

from dgpu.contracts import (
    SINGLE_UPLOAD_LIMIT,
    ConfigError,
    PreconditionFailed,
    StoredJson,
    StoreError,
    StoreProfile,
)

__all__ = [
    "MAX_PRESIGN_TTL_S",
    "SINGLE_PART_LIMIT",
    "S3Store",
    "make_client",
    "open_store",
]

SINGLE_PART_LIMIT = SINGLE_UPLOAD_LIMIT  # S3 / R2 单段 PutObject 的上限（5 GiB）
MAX_PRESIGN_TTL_S = 7 * 24 * 3600  # SigV4 预签名 URL 最长 7 天

DEFAULT_MAX_ATTEMPTS = 5  # 每个请求最多尝试的次数（含第一次）
DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_READ_TIMEOUT_S = 60.0

_CHUNK = 1024 * 1024
_MAX_BACKOFF_S = 20.0
# 存储不返回 Date 头时，get_json_meta 写这里的探针对象来读存储的当前时间
_CLOCK_PROBE_PREFIX = "dgpu/clock-probe/"

# 响应体读到一半断开：botocore 不会重试，这里重新发起 GET
_STREAM_ERRORS = (HTTPClientError, IncompleteReadError)

_MISSING_CODES = frozenset({"NoSuchKey", "NotFound", "404"})
_PRECONDITION_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
_AUTH_CODES = frozenset(
    {
        "AccessDenied",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "InvalidToken",
        "ExpiredToken",
        "Unauthorized",
        "401",
        "403",
    }
)
_THROTTLE_CODES = frozenset({"SlowDown", "TooManyRequests", "ServiceUnavailable", "Throttling"})

_T = TypeVar("_T")


def make_client(
    profile: StoreProfile,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
    read_timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> BaseClient:
    """按 profile 创建 boto3 S3 客户端。凭据只取自 profile，不读环境变量里的密钥。"""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if not profile.endpoint or not profile.endpoint.startswith(("https://", "http://")):
        raise ConfigError(
            f"存储 profile {profile.name!r} 的 endpoint 必须是 http(s):// 开头的 URL，"
            f"当前是 {profile.endpoint!r}。R2 的格式是 https://<账户 ID>.r2.cloudflarestorage.com"
        )
    if not profile.bucket:
        raise ConfigError(f"存储 profile {profile.name!r} 没有填写 bucket")
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"mode": "standard", "total_max_attempts": max_attempts},
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    try:
        session = boto3.session.Session()
        return session.client(
            "s3",
            endpoint_url=profile.endpoint,
            region_name=profile.region or "auto",
            aws_access_key_id=profile.access_key_id,
            aws_secret_access_key=profile.secret_access_key,
            config=config,
        )
    except (BotoCoreError, ValueError) as exc:
        raise ConfigError(
            f"无法为存储 profile {profile.name!r} 创建 S3 客户端："
            f"{type(exc).__name__}: {_redact(str(exc), profile)}。"
            "检查 endpoint 和 region；如果设置了 AWS_PROFILE 等环境变量，确认它们指向存在的配置"
        ) from None


def open_store(profile: StoreProfile) -> S3Store:
    """打开 profile 对应的 bucket。不发网络请求。"""
    return S3Store(profile)


class S3Store:
    """绑定到 profile.bucket 的 S3 兼容存储，实现 contracts.Store。"""

    def __init__(
        self,
        profile: StoreProfile,
        *,
        client: BaseClient | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout: float = DEFAULT_READ_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.profile = profile
        self._bucket = profile.bucket
        self._max_attempts = max(1, max_attempts)
        self._sleep = sleep
        self._client = (
            client
            if client is not None
            else make_client(
                profile,
                max_attempts=max_attempts,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        )

    def __repr__(self) -> str:
        return f"S3Store({self.profile!r})"

    # ------------------------------------------------------------ JSON

    def get_json(self, key: str) -> tuple[dict[str, Any], str] | None:
        got = self._get_object("get_json", key)
        if got is None:
            return None
        meta, data = got
        return self._decode_json(key, meta, data)

    def get_json_meta(self, key: str) -> StoredJson | None:
        """get_json，另外返回对象的 LastModified 和存储服务端的当前时间。

        服务端时间取这次响应的 Date 头（R2 和 S3 都会给）；没有 Date 头的存储写一个探针
        对象，读它的 LastModified，再删掉。两种办法都只用存储的时钟。
        """
        got = self._get_object("get_json", key)
        if got is None:
            return None
        meta, data = got
        doc, etag = self._decode_json(key, meta, data)
        modified = meta.get("LastModified")
        if not isinstance(modified, datetime):
            raise StoreError(
                f"读取 {self._where(key)} 的响应里没有 LastModified，无法判断它多久没更新。"
                "这个存储可能不完全兼容 S3；换一个存储 profile"
            )
        store_time = _response_date(meta)
        if store_time is None:
            store_time = self._probe_store_time()
        return StoredJson(
            doc=doc, etag=etag, last_modified=modified.timestamp(), store_time=store_time
        )

    def _probe_store_time(self) -> float:
        """没有 Date 头时：写一个小对象，用它的 LastModified 作为存储的当前时间。"""
        key = f"{_CLOCK_PROBE_PREFIX}{secrets.token_hex(6)}"
        self.put_bytes(key, b"")
        try:
            try:
                head = self._call(lambda: self._client.head_object(Bucket=self._bucket, Key=key))
            except (ClientError, BotoCoreError) as exc:
                raise self._failure("probe_time", key, exc) from exc
        finally:
            try:
                self.delete(key)
            except StoreError:
                pass  # 探针只是一个空对象，主结果更重要；下次 gc 不会碰它
        modified = head.get("LastModified")
        if not isinstance(modified, datetime):
            raise StoreError(
                f"存储 {self.profile.name} 的响应既没有 Date 头，也没有 LastModified，"
                "无法得到存储的当前时间。换一个存储 profile"
            )
        return modified.timestamp()

    def _decode_json(self, key: str, meta: dict[str, Any], data: bytes) -> tuple[dict, str]:
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise StoreError(
                f"{self._where(key)} 不是合法的 UTF-8 JSON（{exc}）。"
                "这个对象可能被别的程序覆盖或只写了一半；检查它的内容，必要时删除后重新生成"
            ) from None
        if not isinstance(obj, dict):
            raise StoreError(
                f"{self._where(key)} 的 JSON 顶层是 {type(obj).__name__}，应为对象。"
                "检查是否被别的程序覆盖"
            )
        return obj, _etag_of(meta, data)

    def put_json(
        self,
        key: str,
        obj: dict[str, Any],
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
        params: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": data,
            "ContentMD5": _b64md5(data),
            "ContentType": "application/json; charset=utf-8",
        }
        if if_match is not None:
            params["IfMatch"] = _quote_etag(if_match)
        if if_none_match:
            params["IfNoneMatch"] = "*"
        try:
            resp = self._call(lambda: self._client.put_object(**params))
        except ClientError as exc:
            conditional = if_match is not None or if_none_match
            if conditional and _is_precondition_failure(exc, if_match is not None):
                raise self._precondition_failed(key, exc, if_match, if_none_match) from None
            raise self._failure("put_json", key, exc) from exc
        except BotoCoreError as exc:
            raise self._failure("put_json", key, exc) from exc
        return _etag_of(resp, data)

    # ------------------------------------------------------------ 字节与文件

    def get_bytes(self, key: str, *, start: int = 0) -> bytes | None:
        if start < 0:
            raise ValueError("start must be >= 0")
        got = self._get_object("get_bytes", key, start=start)
        if got is None:
            return None
        meta, data = got
        if start and not meta.get("ContentRange"):
            # 服务端忽略了 Range、返回了整个对象
            data = data[start:]
        return data

    def put_bytes(self, key: str, data: bytes) -> None:
        data = bytes(data)
        if len(data) > SINGLE_PART_LIMIT:
            raise self._too_large(key, len(data))
        params = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": data,
            "ContentMD5": _b64md5(data),
        }
        try:
            self._call(lambda: self._client.put_object(**params))
        except (ClientError, BotoCoreError) as exc:
            raise self._failure("put_bytes", key, exc) from exc

    def put_file(self, key: str, path: Path) -> None:
        path = Path(path)
        size = path.stat().st_size
        if size > SINGLE_PART_LIMIT:
            raise self._too_large(key, size, path)
        with open(path, "rb") as handle:
            md5 = _b64md5_stream(handle)

            def upload() -> Any:
                handle.seek(0)
                return self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=handle,
                    ContentLength=size,
                    ContentMD5=md5,
                )

            try:
                self._call(upload)
            except ClientError as exc:
                if _error_code(exc) == "BadDigest":
                    raise StoreError(
                        f"上传 {path} 到 {self._where(key)} 时内容校验失败（BadDigest）："
                        "文件可能在上传过程中被修改。等文件写完后重试"
                    ) from exc
                raise self._failure("put_file", key, exc) from exc
            except BotoCoreError as exc:
                raise self._failure("put_file", key, exc) from exc

    # ------------------------------------------------------------ 查询

    def exists(self, key: str) -> bool:
        try:
            self._call(lambda: self._client.head_object(Bucket=self._bucket, Key=key))
        except ClientError as exc:
            if _is_missing(exc):
                return False
            raise self._failure("exists", key, exc) from exc
        except BotoCoreError as exc:
            raise self._failure("exists", key, exc) from exc
        return True

    def list(self, prefix: str) -> list[str]:
        keys: set[str] = set()
        for page in self._pages("list", prefix):
            for item in page.get("Contents", ()):
                keys.add(item["Key"])
        return sorted(keys)

    def list_dirs(self, prefix: str) -> list[str]:
        names: set[str] = set()
        for page in self._pages("list_dirs", prefix, Delimiter="/"):
            for item in page.get("CommonPrefixes", ()):
                child = item["Prefix"][len(prefix) :]
                names.add(child[:-1] if child.endswith("/") else child)
        return sorted(names)

    def download_prefix(self, prefix: str, dest: Path) -> int:
        dest = Path(dest)
        root = dest.resolve()
        # 先检查全部 key 再下载，避免恶意或异常的 key 造成部分写入
        files: list[tuple[str, Path]] = []
        dirs: list[Path] = []
        claimed: dict[Path, str] = {}
        for key in self.list(prefix):
            target = self._local_target(prefix, key, root)
            if target is None:
                continue
            if not key.endswith("/") and target in claimed:
                raise StoreError(
                    f"{self._where(claimed[target])} 和 {self._where(key)} 会下载到同一个本地文件 "
                    f"{target}，后一个会覆盖前一个。没有写入任何文件。给其中一个改名后重试"
                )
            claimed[target] = key
            if key.endswith("/"):
                dirs.append(target)  # 目录占位对象（如控制台建的“文件夹”）
            else:
                files.append((key, target))
        root.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        for key, target in files:
            self._download_file(key, target)
        return len(files)

    def presign_get(self, key: str, ttl_s: int) -> str:
        ttl_s = int(ttl_s)
        if not 1 <= ttl_s <= MAX_PRESIGN_TTL_S:
            raise ValueError(f"ttl_s must be between 1 and {MAX_PRESIGN_TTL_S}, got {ttl_s}")
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=ttl_s,
            )
        except BotoCoreError as exc:
            raise self._failure("presign_get", key, exc) from exc

    def delete(self, key: str) -> None:
        try:
            self._call(lambda: self._client.delete_object(Bucket=self._bucket, Key=key))
        except ClientError as exc:
            if _is_missing(exc):
                return
            raise self._failure("delete", key, exc) from exc
        except BotoCoreError as exc:
            raise self._failure("delete", key, exc) from exc

    # ------------------------------------------------------------ 内部

    def _call(self, fn: Callable[[], _T]) -> _T:
        """执行一次 API 调用。botocore 已经重试过瞬时错误，这里只补上它不重试的 429。"""
        attempt = 1
        while True:
            try:
                return fn()
            except ClientError as exc:
                if (
                    _status(exc) == 429
                    and _retry_attempts(exc) == 0
                    and attempt < self._max_attempts
                ):
                    self._backoff(attempt)
                    attempt += 1
                    continue
                raise

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(_MAX_BACKOFF_S, 2.0 ** (attempt - 1)))

    def _get_object(
        self, op: str, key: str, *, start: int = 0
    ) -> tuple[dict[str, Any], bytes] | None:
        """GET 并读完响应体。不存在返回 None；Range 超出末尾返回 ({}, b"")。"""
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if start:
            params["Range"] = f"bytes={start}-"
        attempt = 1
        while True:
            try:
                resp = self._call(lambda: self._client.get_object(**params))
            except ClientError as exc:
                if _is_missing(exc):
                    return None
                if start and (_status(exc) == 416 or _error_code(exc) == "InvalidRange"):
                    return {}, b""
                raise self._failure(op, key, exc) from exc
            except BotoCoreError as exc:
                raise self._failure(op, key, exc) from exc
            body = resp["Body"]
            try:
                return resp, body.read()
            except _STREAM_ERRORS as exc:
                if attempt >= self._max_attempts:
                    raise self._failure(op, key, exc) from exc
                self._backoff(attempt)
                attempt += 1
            finally:
                body.close()

    def _pages(self, op: str, prefix: str, **extra: Any) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix, **extra}
        pages: list[dict[str, Any]] = []
        while True:
            try:
                page = self._call(lambda: self._client.list_objects_v2(**params))
            except (ClientError, BotoCoreError) as exc:
                raise self._failure(op, prefix, exc) from exc
            pages.append(page)
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated") or not token:
                return pages
            params["ContinuationToken"] = token

    def _local_target(self, prefix: str, key: str, root: Path) -> Path | None:
        rel = key[len(prefix) :].lstrip("/")
        if not rel:
            if key.endswith("/"):
                return None  # 前缀本身的目录占位对象
            rel = PurePosixPath(key).name  # 前缀正好是一个对象
        parts = PurePosixPath(rel).parts
        target = root.joinpath(*parts).resolve() if parts else root
        backslash = os.sep == "\\" and "\\" in rel  # Windows 上反斜杠也是分隔符
        if ".." in parts or backslash or target == root or not target.is_relative_to(root):
            raise StoreError(
                f"拒绝下载 {self._where(key)}：它的相对路径 {rel!r} 会写到目标目录 {root} 之外。"
                "没有写入任何文件。检查这个前缀下的对象是谁写的，删除或改名后重试"
            )
        return target

    def _download_file(self, key: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            try:
                resp = self._call(lambda: self._client.get_object(Bucket=self._bucket, Key=key))
            except ClientError as exc:
                if _is_missing(exc):
                    raise StoreError(
                        f"下载 {self._where(key)} 时对象已不存在（列出之后被删除）。"
                        "确认没有其他进程在改这个前缀后重试"
                    ) from exc
                raise self._failure("download_prefix", key, exc) from exc
            except BotoCoreError as exc:
                raise self._failure("download_prefix", key, exc) from exc
            body = resp["Body"]
            tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.part")
            try:
                with open(tmp, "wb") as out:
                    for chunk in body.iter_chunks(_CHUNK):
                        out.write(chunk)
                os.replace(tmp, target)
                return
            except _STREAM_ERRORS as exc:
                if attempt >= self._max_attempts:
                    raise self._failure("download_prefix", key, exc) from exc
                self._backoff(attempt)
                attempt += 1
            finally:
                body.close()
                tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------ 错误信息

    def _where(self, key: str) -> str:
        return f"{self._bucket}/{key}"

    def _too_large(self, key: str, size: int, path: Path | None = None) -> StoreError:
        what = f"{path}（{size} 字节）" if path is not None else f"{size} 字节的数据"
        return StoreError(
            f"无法上传 {what} 到 {self._where(key)}：超过单段上传上限 5 GiB。"
            "R2 的分段上传 ETag 不可靠，dgpu 只做单段上传。把文件拆小，"
            "或者先用 rclone 把数据传到 bucket，再在 job.yaml 的 data 里写成 "
            "r2://<bucket>/<目录> 引用"
        )

    def _precondition_failed(
        self, key: str, exc: ClientError, if_match: str | None, if_none_match: bool
    ) -> PreconditionFailed:
        if if_none_match and if_match is None:
            what = f"{self._where(key)} 已存在"
        elif _is_missing(exc):
            what = f"{self._where(key)} 不存在，无法按 ETag {if_match} 条件更新"
        else:
            what = f"{self._where(key)} 已被其他写者修改（期望 ETag {if_match}）"
        message = f"条件写失败：{what}。重新读取后再决定是否继续"
        retries = _retry_attempts(exc)
        if retries:
            message += (
                f"。注意：这次写入重试过 {retries} 次，之前的某次请求可能已经生效，"
                "重新读取时要考虑这个可能"
            )
        return PreconditionFailed(message)

    def _failure(self, op: str, key: str, exc: BaseException) -> StoreError:
        if isinstance(exc, ClientError):
            code = _error_code(exc)
            status = _status(exc)
            detail = f"{code or '未知错误'}（HTTP {status}）"
            text = exc.response.get("Error", {}).get("Message")
            if text:
                detail += f" {text}"
            retries = _retry_attempts(exc)
            if retries:
                detail += f"，已重试 {retries} 次"
            hint = self._hint(code, status)
            # 限流和服务端错误稍后可能恢复；认证、bucket 不存在等 4xx 重试没有意义
            transient = code in _THROTTLE_CODES or status in (429,) or (status or 0) >= 500
        else:
            detail = f"{type(exc).__name__}: {exc}"
            hint = self._hint_for_exception(exc)
            transient = not isinstance(exc, NoCredentialsError | PartialCredentialsError)
        message = (
            f"存储操作 {op} 失败：{self._where(key)}"
            f"（profile {self.profile.name}，endpoint {self.profile.endpoint}）：{detail}。{hint}"
        )
        return StoreError(_redact(message, self.profile), transient=transient)

    def _hint(self, code: str, status: int | None) -> str:
        name = self.profile.name
        if code in _AUTH_CODES or status in (401, 403):
            return (
                f"检查存储 profile {name!r} 的 access_key_id / secret_access_key 是否正确，"
                f"以及令牌是否有 bucket {self._bucket!r} 的读写权限"
            )
        if code == "NoSuchBucket":
            return (
                f"bucket {self._bucket!r} 不存在：检查 bucket 名，以及 endpoint 里的账户 ID "
                "是否属于这个 bucket"
            )
        if code in _THROTTLE_CODES or status in (429, 503):
            return "存储限流或暂时不可用（R2 同一 key 每秒最多写 1 次），稍后重试"
        if status is not None and status >= 500:
            return "存储服务端错误，已按重试策略重试；稍后再试，持续出现时查看服务状态页"
        return "检查存储 profile 的配置和这个 key 是否正确后重试"

    def _hint_for_exception(self, exc: BaseException) -> str:
        if isinstance(exc, NoCredentialsError | PartialCredentialsError):
            return f"存储 profile {self.profile.name!r} 缺少 access_key_id 或 secret_access_key"
        if isinstance(exc, BotoConnectionError | HTTPClientError | IncompleteReadError):
            return (
                f"网络连接失败，已按重试策略重试。检查本机能否访问 {self.profile.endpoint}"
                "（代理、防火墙、DNS），以及 endpoint 是否写对"
            )
        return "检查存储 profile 的配置后重试"


# ---------------------------------------------------------------- 工具函数


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code") or "")


def _status(exc: ClientError) -> int | None:
    return exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")


def _retry_attempts(exc: ClientError) -> int:
    return int(exc.response.get("ResponseMetadata", {}).get("RetryAttempts") or 0)


def _is_missing(exc: ClientError) -> bool:
    code = _error_code(exc)
    if code == "NoSuchBucket":
        return False
    return code in _MISSING_CODES or (_status(exc) == 404 and not code)


def _is_precondition_failure(exc: ClientError, has_if_match: bool) -> bool:
    if _error_code(exc) in _PRECONDITION_CODES or _status(exc) in (409, 412):
        return True
    # If-Match 指向不存在的对象时，S3 返回 404 NoSuchKey；与 MemoryStore 一致，算条件不满足
    return has_if_match and _is_missing(exc)


def _response_date(meta: dict[str, Any]) -> float | None:
    """响应的 Date 头（服务端生成响应的时间），epoch 秒；没有或无法解析时返回 None。"""
    headers = meta.get("ResponseMetadata", {}).get("HTTPHeaders") or {}
    value = headers.get("date") or headers.get("Date")
    if not value:
        return None
    try:
        moment = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if moment is None or moment.tzinfo is None:  # RFC 7231 的 Date 一律是 GMT
        return None
    return moment.timestamp()


def _quote_etag(etag: str) -> str:
    return f'"{_normalize_etag(etag)}"'


def _normalize_etag(etag: str) -> str:
    return etag.strip().strip('"')


def _etag_of(meta: dict[str, Any], data: bytes) -> str:
    """响应里的 ETag（去掉引号）；服务端没给时退回内容 md5（单段上传的 ETag 就是它）。"""
    etag = meta.get("ETag")
    return _normalize_etag(etag) if etag else hashlib.md5(data).hexdigest()


def _b64md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


def _b64md5_stream(handle: BinaryIO) -> str:
    digest = hashlib.md5()
    for chunk in iter(lambda: handle.read(_CHUNK), b""):
        digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _redact(text: str, profile: StoreProfile) -> str:
    for secret in (profile.secret_access_key, profile.access_key_id):
        if secret:
            text = text.replace(secret, "***")
    return text
