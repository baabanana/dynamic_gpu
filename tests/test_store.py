"""工作包 B：S3Store 的行为测试。

前半部分对 MemoryStore 和 S3Store（moto）跑同一组断言，证明替身和真实实现一致；
后半部分是只有 S3 才有的行为：客户端配置、请求头、分页、错误映射、路径逃逸。
moto 不校验的东西（Content-MD5、409 冲突、429 限流）用 botocore Stubber 或
请求钩子检查请求本身。测试不联网：socket.connect 被禁用。
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket

import boto3
import botocore.endpoint
import pytest
from botocore.exceptions import (
    ConnectionClosedError,
    EndpointConnectionError,
    ResponseStreamingError,
)
from botocore.stub import ANY, Stubber
from moto import mock_aws

from dgpu.contracts import ConfigError, PreconditionFailed, Store, StoreError, StoreProfile
from dgpu.store import SINGLE_PART_LIMIT, S3Store, make_client, open_store
from dgpu.testing import MemoryStore

# moto 能识别的 endpoint；region 用 R2 的 auto
S3_PROFILE = StoreProfile(
    name="r2",
    endpoint="https://s3.amazonaws.com",
    bucket="dgpu-test",
    access_key_id="AKIDDGPUTEST00000001",
    secret_access_key="sEcReT/never-print-me+0001",
)
SECRETS = (S3_PROFILE.access_key_id, S3_PROFILE.secret_access_key)
KEY = "dgpu/jobs/j1/attempt.json"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """不读用户的 AWS 配置，也不许真的联网。"""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))

    def refuse(*args, **kwargs):
        raise OSError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    # botocore 自己的重试退避不真的睡
    monkeypatch.setattr(botocore.endpoint.time, "sleep", lambda seconds: None)


@pytest.fixture
def moto_bucket():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=S3_PROFILE.bucket)
        yield


@pytest.fixture
def s3(moto_bucket) -> S3Store:
    return open_store(S3_PROFILE)


@pytest.fixture(params=["memory", "s3"])
def store(request):
    if request.param == "memory":
        return MemoryStore()
    request.getfixturevalue("moto_bucket")
    return open_store(S3_PROFILE)


@pytest.fixture
def second_writer(store):
    """另一个 CLI（另一台电脑）：MemoryStore 共享同一份数据，S3Store 用独立的客户端。"""
    if isinstance(store, MemoryStore):
        return store
    return open_store(S3_PROFILE)


def stubbed(**kwargs) -> tuple[S3Store, Stubber, list[float]]:
    """用 Stubber 编排响应的 S3Store；不发请求。"""
    client = make_client(S3_PROFILE)
    stubber = Stubber(client)
    stubber.activate()
    sleeps: list[float] = []
    return S3Store(S3_PROFILE, client=client, sleep=sleeps.append, **kwargs), stubber, sleeps


def headers_of(request) -> dict[str, str]:
    return {k: v.decode() if isinstance(v, bytes) else v for k, v in request.headers.items()}


def assert_no_secrets(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


# ================================================================ 与 MemoryStore 一致的行为


def test_implements_store_protocol(store):
    assert isinstance(store, Store)


def test_json_round_trip_keeps_non_ascii_as_utf8(store):
    obj = {"名字": "训练-🙂", "n": 3, "nested": {"列表": [1, "二"]}}
    etag = store.put_json(KEY, obj)

    assert '"' not in etag
    assert store.get_json(KEY) == (obj, etag)
    raw = store.get_bytes(KEY)
    assert "训练-🙂".encode() in raw
    assert b"\\u" not in raw


def test_if_none_match_creates_only_once(store, second_writer):
    first = store.put_json(KEY, {"attempt": 1}, if_none_match=True)

    with pytest.raises(PreconditionFailed):
        second_writer.put_json(KEY, {"attempt": 2}, if_none_match=True)
    assert store.get_json(KEY) == ({"attempt": 1}, first)


def test_if_match_etag_chain(store):
    e1 = store.put_json(KEY, {"attempt": 1})
    e2 = store.put_json(KEY, {"attempt": 2}, if_match=e1)

    with pytest.raises(PreconditionFailed):
        store.put_json(KEY, {"attempt": 99}, if_match=e1)  # 过期的 ETag
    assert store.get_json(KEY) == ({"attempt": 2}, e2)
    e3 = store.put_json(KEY, {"attempt": 3}, if_match=e2)
    assert store.get_json(KEY) == ({"attempt": 3}, e3)


def test_if_match_on_missing_key_fails(store):
    with pytest.raises(PreconditionFailed):
        store.put_json(KEY, {"attempt": 1}, if_match="0" * 32)
    assert store.get_json(KEY) is None


def test_two_writers_racing_with_if_match_exactly_one_wins(store, second_writer):
    store.put_json(KEY, {"attempt": 1})
    _, seen_a = store.get_json(KEY)
    _, seen_b = second_writer.get_json(KEY)
    assert seen_a == seen_b

    winners = []
    for name, writer, seen in (("a", store, seen_a), ("b", second_writer, seen_b)):
        try:
            writer.put_json(KEY, {"attempt": 2, "by": name}, if_match=seen)
            winners.append(name)
        except PreconditionFailed:
            pass

    assert winners == ["a"]
    assert store.get_json(KEY)[0] == {"attempt": 2, "by": "a"}


def test_two_writers_racing_to_create_exactly_one_wins(store, second_writer):
    winners = []
    for name, writer in (("a", store), ("b", second_writer)):
        try:
            writer.put_json(KEY, {"by": name}, if_none_match=True)
            winners.append(name)
        except PreconditionFailed:
            pass

    assert winners == ["a"]


def test_get_bytes_from_offset(store):
    key = "dgpu/jobs/j1/logs/attempt-1.log"
    store.put_bytes(key, b"0123456789")

    assert store.get_bytes(key) == b"0123456789"
    assert store.get_bytes(key, start=4) == b"456789"
    assert store.get_bytes(key, start=9) == b"9"
    assert store.get_bytes(key, start=10) == b""  # 正好在末尾
    assert store.get_bytes(key, start=500) == b""  # 超出末尾


def test_empty_object(store):
    store.put_bytes("empty", b"")

    assert store.exists("empty")
    assert store.get_bytes("empty") == b""
    assert store.get_bytes("empty", start=1) == b""


def test_missing_keys(store, tmp_path):
    assert store.get_json("nope.json") is None
    assert store.get_bytes("nope.log") is None
    assert store.get_bytes("nope.log", start=10) is None
    assert store.exists("nope") is False
    assert store.list("nope/") == []
    assert store.list_dirs("nope/") == []
    assert store.download_prefix("nope/", tmp_path / "out") == 0
    store.delete("nope")  # 不报错


def test_put_file_and_put_bytes_round_trip(store, tmp_path):
    payload = bytes(range(256)) * 4000
    path = tmp_path / "code.tar.gz"
    path.write_bytes(payload)

    store.put_file("dgpu/jobs/j1/code.tar.gz", path)
    store.put_bytes("dgpu/jobs/j1/blob", payload[:10])

    assert store.get_bytes("dgpu/jobs/j1/code.tar.gz") == payload
    assert store.get_bytes("dgpu/jobs/j1/blob") == payload[:10]


def test_put_overwrites(store):
    store.put_bytes("k", b"one")
    store.put_bytes("k", b"two")
    assert store.get_bytes("k") == b"two"


def test_delete_is_idempotent(store):
    store.put_bytes("k", b"x")
    store.delete("k")
    store.delete("k")
    assert not store.exists("k")
    assert store.get_bytes("k") is None


def test_list_is_recursive_sorted_and_respects_prefix(store):
    keys = [
        "dgpu/jobs/j1/status.json",
        "dgpu/jobs/j1/events/2-cli-b.json",
        "dgpu/jobs/j1/events/1-cli-a.json",
        "dgpu/jobs/j1/output/deep/x/y.txt",
    ]
    for key in keys:
        store.put_bytes(key, b"x")
    store.put_bytes("dgpu/jobs/j10/status.json", b"x")  # 相邻前缀，不应出现

    assert store.list("dgpu/jobs/j1/") == sorted(keys)
    assert store.list("dgpu/jobs/j1/events/") == [
        "dgpu/jobs/j1/events/1-cli-a.json",
        "dgpu/jobs/j1/events/2-cli-b.json",
    ]


def test_list_dirs_returns_immediate_children(store):
    for key in (
        "dgpu/jobs/b/spec.json",
        "dgpu/jobs/a/events/1.json",
        "dgpu/jobs/a/status.json",
        "dgpu/jobs/top.json",
    ):
        store.put_bytes(key, b"x")

    assert store.list_dirs("dgpu/jobs/") == ["a", "b"]
    assert store.list_dirs("dgpu/jobs/a/") == ["events"]
    assert store.list_dirs("dgpu/jobs/a/events/") == []


def test_download_prefix_preserves_relative_paths(store, tmp_path):
    prefix = "dgpu/jobs/j1/output/"
    files = {
        "results.csv": b"epoch,loss\n1,0.5\n",
        "weights/best.pt": b"\x00\x01best",
        "weights/sub/last.pt": b"\x00\x02last",
    }
    for rel, data in files.items():
        store.put_bytes(prefix + rel, data)
    store.put_bytes("dgpu/jobs/j1/ckpt/other.pt", b"not me")
    dest = tmp_path / "pulled"

    assert store.download_prefix(prefix, dest) == 3
    for rel, data in files.items():
        assert (dest / rel).read_bytes() == data
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == sorted(
        files
    )
    # 再拉一次：覆盖，结果相同
    assert store.download_prefix(prefix, dest) == 3
    assert (dest / "weights/best.pt").read_bytes() == b"\x00\x01best"


# ================================================================ 客户端配置


def test_open_store_configures_client_for_r2():
    store = open_store(S3_PROFILE)
    client = store._client
    config = client.meta.config

    assert client.meta.endpoint_url == S3_PROFILE.endpoint
    assert client.meta.region_name == "auto"
    assert config.signature_version == "s3v4"
    assert config.s3["addressing_style"] == "path"
    assert config.retries == {"mode": "standard", "total_max_attempts": 5}
    assert config.connect_timeout == 10
    assert config.read_timeout == 60
    assert config.request_checksum_calculation == "when_required"
    assert config.response_checksum_validation == "when_required"
    # path-style：bucket 在路径里，不在主机名里
    url = store.presign_get("dgpu/x.json", 60)
    assert url.startswith(f"{S3_PROFILE.endpoint}/{S3_PROFILE.bucket}/dgpu/x.json?")


def test_client_uses_profile_credentials_not_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDFROMENVIRONMENT")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    creds = open_store(S3_PROFILE)._client._request_signer._credentials

    assert creds.access_key == S3_PROFILE.access_key_id
    assert creds.secret_key == S3_PROFILE.secret_access_key


@pytest.mark.parametrize("endpoint", ["", "r2.example.com", "ftp://x"])
def test_bad_endpoint_is_a_config_error(endpoint):
    profile = StoreProfile(
        name="r2", endpoint=endpoint, bucket="b", access_key_id="a", secret_access_key="s"
    )
    with pytest.raises(ConfigError, match="endpoint"):
        open_store(profile)


def test_repr_has_no_credentials():
    text = repr(open_store(S3_PROFILE))
    assert "dgpu-test" in text
    assert_no_secrets(text)


# ================================================================ 条件写：请求头和错误映射


def test_conditional_headers_and_content_md5_are_sent(s3):
    sent: list[dict[str, str]] = []
    s3._client.meta.events.register(
        "before-send.s3.PutObject", lambda request, **_: sent.append(headers_of(request))
    )

    e1 = s3.put_json(KEY, {"attempt": 1}, if_none_match=True)
    s3.put_json(KEY, {"attempt": 2}, if_match=e1)
    s3.put_json(KEY, {"attempt": 3})

    created, updated, plain = sent
    assert created["If-None-Match"] == "*"
    assert "If-Match" not in created
    assert updated["If-Match"] == f'"{e1}"'
    assert "If-None-Match" not in updated
    assert "If-Match" not in plain and "If-None-Match" not in plain
    body = s3.get_bytes(KEY)
    assert plain["Content-MD5"] == base64.b64encode(hashlib.md5(body).digest()).decode()
    assert plain["Content-Type"].startswith("application/json")


def test_quoted_etag_from_caller_is_not_double_quoted(s3):
    etag = s3.put_json(KEY, {"attempt": 1})
    s3.put_json(KEY, {"attempt": 2}, if_match=f'"{etag}"')
    assert s3.get_json(KEY)[0] == {"attempt": 2}


@pytest.mark.parametrize(
    ("code", "status"), [("PreconditionFailed", 412), ("ConditionalRequestConflict", 409)]
)
def test_conditional_errors_map_to_precondition_failed(code, status):
    store, stubber, _ = stubbed()
    expected = {
        "Bucket": S3_PROFILE.bucket,
        "Key": KEY,
        "Body": ANY,
        "ContentMD5": ANY,
        "ContentType": ANY,
        "IfMatch": '"abc123"',
    }
    stubber.add_client_error(
        "put_object", service_error_code=code, http_status_code=status, expected_params=expected
    )
    stubber.add_client_error(
        "put_object",
        service_error_code=code,
        http_status_code=status,
        expected_params={
            **{k: v for k, v in expected.items() if k != "IfMatch"},
            "IfNoneMatch": "*",
        },
    )

    with pytest.raises(PreconditionFailed, match="条件写失败"):
        store.put_json(KEY, {"a": 1}, if_match="abc123")
    with pytest.raises(PreconditionFailed, match="已存在"):
        store.put_json(KEY, {"a": 1}, if_none_match=True)
    stubber.assert_no_pending_responses()


def test_unconditional_put_conflict_is_a_store_error_not_precondition():
    store, stubber, _ = stubbed()
    stubber.add_client_error("put_object", service_error_code="InternalError", http_status_code=500)

    with pytest.raises(StoreError) as info:
        store.put_json(KEY, {"a": 1})
    assert not isinstance(info.value, PreconditionFailed)


def test_conditional_write_retried_after_lost_response_warns_it_may_have_landed(s3):
    """第一次请求已经写进去但响应丢了；botocore 重试时条件不再满足。"""
    e0 = s3.put_json(KEY, {"attempt": 1})
    other = open_store(S3_PROFILE)
    calls = []

    def lose_first_response(request, **_):
        calls.append(1)
        if len(calls) == 1:
            other.put_json(KEY, {"attempt": 2})  # 模拟这次请求已在服务端生效
            raise ConnectionClosedError(endpoint_url=S3_PROFILE.endpoint)

    s3._client.meta.events.register("before-send.s3.PutObject", lose_first_response)

    with pytest.raises(PreconditionFailed, match="可能已经生效"):
        s3.put_json(KEY, {"attempt": 2}, if_match=e0)
    assert len(calls) == 2


# ================================================================ 单段上传


def test_put_file_uses_a_single_put_object(s3, tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(os.urandom(9 * 1024 * 1024))  # 超过 boto3 默认的 8 MiB 分段阈值
    operations: list[str] = []
    headers: list[dict[str, str]] = []
    s3._client.meta.events.register(
        "before-call.s3", lambda model, **_: operations.append(model.name)
    )
    s3._client.meta.events.register(
        "before-send.s3.PutObject", lambda request, **_: headers.append(headers_of(request))
    )

    s3.put_file("dgpu/datasets/h/big.bin", path)

    assert operations == ["PutObject"]
    expected_md5 = base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
    assert headers[0]["Content-MD5"] == expected_md5
    assert s3.get_bytes("dgpu/datasets/h/big.bin") == path.read_bytes()


def test_put_file_over_single_part_limit_is_refused_without_uploading(tmp_path):
    store, stubber, _ = stubbed()  # 没有编排任何响应：一旦发请求就会报错
    path = tmp_path / "huge.bin"
    with open(path, "wb") as handle:
        handle.truncate(SINGLE_PART_LIMIT + 1)  # 稀疏文件，不占磁盘

    with pytest.raises(StoreError, match="5 GiB"):
        store.put_file("dgpu/datasets/h/huge.bin", path)
    stubber.assert_no_pending_responses()


def test_put_file_retry_after_429_resends_the_whole_file(tmp_path):
    store, stubber, sleeps = stubbed()
    path = tmp_path / "f.bin"
    path.write_bytes(b"payload")
    bodies: list[bytes] = []

    def capture(params, **_):
        bodies.append(params["Body"].read())
        params["Body"].seek(0)

    store._client.meta.events.register("before-parameter-build.s3.PutObject", capture)
    stubber.add_client_error(
        "put_object", service_error_code="TooManyRequests", http_status_code=429
    )
    stubber.add_response("put_object", {"ETag": '"x"'})

    store.put_file("k", path)

    assert bodies == [b"payload", b"payload"]
    assert sleeps == [1.0]


# ================================================================ 限流、断连和其他错误


def test_429_is_retried_with_backoff_then_gives_up():
    store, stubber, sleeps = stubbed(max_attempts=3)
    for _ in range(3):
        stubber.add_client_error(
            "put_object", service_error_code="TooManyRequests", http_status_code=429
        )

    with pytest.raises(StoreError, match="限流") as info:
        store.put_bytes(KEY, b"x")
    assert sleeps == [1.0, 2.0]
    assert "put_bytes" in str(info.value) and KEY in str(info.value)


def test_access_denied_is_a_store_error_with_hint_and_no_credentials():
    store, stubber, _ = stubbed()
    leaky = f"Access denied for {S3_PROFILE.access_key_id} / {S3_PROFILE.secret_access_key}"
    stubber.add_client_error(
        "get_object", service_error_code="AccessDenied", service_message=leaky, http_status_code=403
    )

    with pytest.raises(StoreError) as info:
        store.get_json(KEY)
    message = str(info.value)
    assert "get_json" in message and KEY in message
    assert "access_key_id" in message  # 告诉用户查什么
    assert_no_secrets(message)


def test_missing_bucket_is_an_error_not_a_missing_key(moto_bucket):
    profile = StoreProfile(
        name="r2",
        endpoint=S3_PROFILE.endpoint,
        bucket="no-such-bucket",
        access_key_id=S3_PROFILE.access_key_id,
        secret_access_key=S3_PROFILE.secret_access_key,
    )
    store = open_store(profile)

    with pytest.raises(StoreError, match="NoSuchBucket"):
        store.get_json(KEY)
    with pytest.raises(StoreError, match="NoSuchBucket"):
        store.list("dgpu/")


@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s.put_bytes(KEY, b"x"),
        lambda s: s.get_json(KEY),
        lambda s: s.exists(KEY),  # 断网不能被当成“不存在”
        lambda s: s.list("dgpu/"),
        lambda s: s.delete(KEY),
    ],
)
def test_connection_failure_after_retries_is_a_store_error(operation):
    store = S3Store(S3_PROFILE, max_attempts=3, sleep=lambda s: None)
    attempts = []

    def unreachable(request, **_):
        attempts.append(request.url)
        raise EndpointConnectionError(endpoint_url=S3_PROFILE.endpoint)

    store._client.meta.events.register("before-send.s3", unreachable)

    with pytest.raises(StoreError) as info:
        operation(store)
    assert len(attempts) == 3
    message = str(info.value)
    assert S3_PROFILE.endpoint in message and "网络" in message
    assert_no_secrets(message)


def test_server_error_is_a_store_error():
    store, stubber, _ = stubbed()
    stubber.add_client_error(
        "head_object", service_error_code="InternalError", http_status_code=500
    )
    with pytest.raises(StoreError, match="exists"):
        store.exists(KEY)


def test_corrupt_json_is_a_store_error(s3):
    s3.put_bytes(KEY, b"{half a json")
    with pytest.raises(StoreError, match="JSON"):
        s3.get_json(KEY)
    s3.put_bytes(KEY, b"[1, 2]")
    with pytest.raises(StoreError, match="JSON"):
        s3.get_json(KEY)


# ================================================================ 读到一半断开


class _BrokenBody:
    def read(self, amt=None):
        raise ResponseStreamingError(error="connection reset by peer")

    def iter_chunks(self, chunk_size=1024):
        yield b"partial-"
        raise ResponseStreamingError(error="connection reset by peer")

    def close(self):
        pass


class _FlakyGets:
    """包装真实客户端，让前 failures 次 GetObject 的响应体读到一半断开。"""

    def __init__(self, client, failures):
        self._client = client
        self.failures = failures

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_object(self, **kwargs):
        resp = self._client.get_object(**kwargs)
        if self.failures > 0:
            self.failures -= 1
            resp["Body"].close()
            resp["Body"] = _BrokenBody()
        return resp


def test_broken_stream_is_retried(s3, tmp_path):
    s3.put_bytes("p/a.bin", b"full content")
    s3.put_json(KEY, {"ok": True})
    sleeps: list[float] = []
    flaky = _FlakyGets(s3._client, failures=2)
    store = S3Store(S3_PROFILE, client=flaky, sleep=sleeps.append)

    assert store.download_prefix("p/", tmp_path / "out") == 1
    assert (tmp_path / "out" / "a.bin").read_bytes() == b"full content"
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["a.bin"]  # 没有残留的 .part
    flaky.failures = 1
    assert store.get_json(KEY)[0] == {"ok": True}
    assert sleeps == [1.0, 2.0, 1.0]


def test_broken_stream_gives_up_without_leaving_partial_files(s3, tmp_path):
    s3.put_bytes("p/a.bin", b"full content")
    store = S3Store(S3_PROFILE, client=_FlakyGets(s3._client, failures=99), sleep=lambda s: None)

    with pytest.raises(StoreError, match="download_prefix"):
        store.download_prefix("p/", tmp_path / "out")
    assert list((tmp_path / "out").rglob("*")) == []


# ================================================================ 分页与排序


def test_list_and_list_dirs_paginate_past_1000(s3):
    raw = s3._client
    names = [f"d{i:04d}" for i in range(1005)]
    for name in names:
        raw.put_object(Bucket=S3_PROFILE.bucket, Key=f"dgpu/many/{name}/f.txt", Body=b"x")
    pages: list[str] = []
    raw.meta.events.register("before-call.s3.ListObjectsV2", lambda **_: pages.append("page"))

    keys = s3.list("dgpu/many/")
    assert keys == [f"dgpu/many/{name}/f.txt" for name in names]
    assert len(pages) >= 2

    pages.clear()
    assert s3.list_dirs("dgpu/many/") == names
    assert len(pages) >= 2


def test_unsorted_listing_is_sorted_and_continuation_is_followed():
    """R2 的 ListObjects 不保证有序。"""
    store, stubber, _ = stubbed()
    stubber.add_response(
        "list_objects_v2",
        {
            "IsTruncated": True,
            "NextContinuationToken": "t1",
            "Contents": [{"Key": "p/c"}, {"Key": "p/a"}],
        },
        {"Bucket": S3_PROFILE.bucket, "Prefix": "p/"},
    )
    stubber.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "Contents": [{"Key": "p/b"}, {"Key": "p/0"}]},
        {"Bucket": S3_PROFILE.bucket, "Prefix": "p/", "ContinuationToken": "t1"},
    )
    stubber.add_response(
        "list_objects_v2",
        {"IsTruncated": False, "CommonPrefixes": [{"Prefix": "p/z/"}, {"Prefix": "p/m/"}]},
        {"Bucket": S3_PROFILE.bucket, "Prefix": "p/", "Delimiter": "/"},
    )

    assert store.list("p/") == ["p/0", "p/a", "p/b", "p/c"]
    assert store.list_dirs("p/") == ["m", "z"]
    stubber.assert_no_pending_responses()


# ================================================================ download_prefix 的路径安全


@pytest.mark.parametrize("bad", ["../evil.txt", "a/../../evil.txt", "a/../.."])
def test_download_prefix_refuses_keys_escaping_dest(s3, tmp_path, bad):
    prefix = "dgpu/jobs/j1/output/"
    s3.put_bytes(prefix + "aaa-first.txt", b"ok")  # 排在前面，也不能被写下来
    s3.put_bytes(prefix + bad, b"evil")
    dest = tmp_path / "work" / "dest"

    with pytest.raises(StoreError, match="拒绝下载"):
        s3.download_prefix(prefix, dest)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_download_prefix_refuses_to_follow_symlinks_out_of_dest(s3, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "link").symlink_to(outside, target_is_directory=True)
    s3.put_bytes("p/link/evil.txt", b"evil")

    with pytest.raises(StoreError, match="拒绝下载"):
        s3.download_prefix("p/", dest)
    assert list(outside.iterdir()) == []


def test_download_prefix_directory_markers_and_single_object(s3, tmp_path):
    s3.put_bytes("p/", b"")  # 控制台建的“文件夹”
    s3.put_bytes("p/empty-dir/", b"")
    s3.put_bytes("p/f.txt", b"f")

    assert s3.download_prefix("p/", tmp_path / "a") == 1
    assert (tmp_path / "a" / "empty-dir").is_dir()
    assert (tmp_path / "a" / "f.txt").read_bytes() == b"f"
    # 前缀不以 / 结尾也能用；前缀正好是一个对象时按文件名落盘
    assert s3.download_prefix("p", tmp_path / "b") == 1
    assert (tmp_path / "b" / "f.txt").read_bytes() == b"f"
    assert s3.download_prefix("p/f.txt", tmp_path / "c") == 1
    assert (tmp_path / "c" / "f.txt").read_bytes() == b"f"


# ================================================================ 预签名


def test_presigned_url_is_fetchable_with_plain_http_get(s3):
    import requests  # moto 的依赖；mock_aws 会拦截 requests 的请求

    s3.put_bytes("dgpu/jobs/j1/secrets-1.json", "启动文档".encode())
    url = s3.presign_get("dgpu/jobs/j1/secrets-1.json", 3600)

    assert "X-Amz-Expires=3600" in url
    assert "X-Amz-Signature=" in url
    assert S3_PROFILE.secret_access_key not in url
    response = requests.get(url, timeout=5)
    assert response.status_code == 200
    assert response.content == "启动文档".encode()


@pytest.mark.parametrize("ttl", [0, -1, 7 * 24 * 3600 + 1])
def test_presign_rejects_out_of_range_ttl(ttl):
    with pytest.raises(ValueError):
        open_store(S3_PROFILE).presign_get("k", ttl)


def test_get_bytes_rejects_negative_start(s3):
    with pytest.raises(ValueError):
        s3.get_bytes("k", start=-1)


def test_download_prefix_refuses_keys_that_collide_locally(s3, tmp_path):
    s3.put_bytes("p/a.txt", b"one")
    s3.put_bytes("p//a.txt", b"two")  # 不同的 key，落到同一个本地文件

    with pytest.raises(StoreError, match="同一个本地文件"):
        s3.download_prefix("p/", tmp_path / "out")
    assert not (tmp_path / "out" / "a.txt").exists()
