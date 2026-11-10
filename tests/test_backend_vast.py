"""VastBackend 的测试：请求内容、状态映射、客户端过滤、重试与失败路径。HTTP 全部由 respx 模拟。"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from dgpu import layout
from dgpu.backends import vast
from dgpu.backends.vast import VastBackend
from dgpu.contracts import (
    INST_EXITED,
    INST_GONE,
    INST_LOADING,
    INST_PENDING,
    INST_RUNNING,
    INST_UNKNOWN,
    Backend,
    BackendError,
    Balance,
    GpuRequest,
    InstanceRef,
    LaunchSpec,
    OfferUnavailable,
    VastSettings,
)
from dgpu.testing import make_offer

BASE = "https://vast.test"
KEY = "vast-SECRET-key-0123456789abcdef"
JOB = "yolo-261110120000-ab12"
LABEL = layout.label(JOB, 1)

REQ = GpuRequest(
    names=("RTX 3090", "RTX 4090"),
    count=1,
    max_price_per_hour=0.5,
    max_hours=2.0,
    disk_gb=40,
    image="ultralytics/ultralytics:8.3.0",
)
LAUNCH = LaunchSpec(
    job_id=JOB,
    attempt=1,
    image="ultralytics/ultralytics:8.3.0",
    bootstrap_cmd="echo bootstrap",
    disk_gb=40,
    max_price_per_hour=0.5,
)
OFFER = make_offer("123")
REF = InstanceRef(backend="vast", id="9001", label=LABEL)


@pytest.fixture
def api():
    # assert_all_mocked：任何没有模拟的请求都会失败，测试不可能访问网络
    with respx.mock(base_url=BASE, assert_all_called=False, assert_all_mocked=True) as router:
        yield router


@pytest.fixture
def sleeps():
    return []


def make_backend(sleeps=None, **settings) -> VastBackend:
    recorded = sleeps if sleeps is not None else []
    return VastBackend(
        VastSettings(api_key=KEY, base_url=BASE, **settings),
        client=httpx.Client(),
        sleep=recorded.append,
    )


def offer_row(offer_id=1, **overrides):
    row = {
        "id": offer_id,
        "ask_contract_id": offer_id,
        "machine_id": 1000 + offer_id,
        "host_id": 2000 + offer_id,
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.20,
        "dph_total": 0.21,
        "storage_cost": 0.18,  # $/GB/月：40 GB → 0.18 * 40 / 720 = 0.01 $/h
        "inet_down": 800.0,
        "inet_up": 600.0,
        "inet_down_cost": 0.002,
        "inet_up_cost": 0.003,
        "reliability": 0.995,
        "reliability2": 0.995,
        "disk_space": 200.0,
        "duration": 30 * 86400.0,
        "verification": "verified",
        "rentable": True,
        "rented": False,
        "external": None,
        "is_bid": False,
        "hosting_type": 0,
        "geolocation": "Ireland, IE",
        "cuda_max_good": 12.4,
    }
    row.update(overrides)
    return row


def instance_row(instance_id=9001, label=LABEL, status="running", **overrides):
    row = {
        "id": instance_id,
        "label": label,
        "actual_status": status,
        "dph_total": 0.21,
        "start_date": 1762770000.5,
    }
    row.update(overrides)
    return row


def body_of(call) -> dict:
    return json.loads(call.request.content)


# ---------------------------------------------------------------- 协议


def test_implements_backend_protocol_and_repr_hides_key():
    backend = make_backend()
    assert isinstance(backend, Backend)
    assert backend.name == "vast"
    assert KEY not in repr(backend)


# ---------------------------------------------------------------- search


def test_search_sends_all_filters_and_bearer_key(api):
    route = api.post("/api/v0/bundles/").respond(json={"offers": []})
    assert make_backend().search(REQ) == []

    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    query = json.loads(request.content)
    assert query["verified"] == {"eq": True}
    assert query["rentable"] == {"eq": True}
    assert query["rented"] == {"eq": False}
    assert query["external"] == {"eq": False}
    assert query["num_gpus"] == {"eq": 1}
    assert query["gpu_name"] == {"in": ["RTX 3090", "RTX 4090"]}
    assert query["reliability"] == {"gte": 0.98}
    assert query["inet_down"] == {"gte": 500.0}
    assert query["disk_space"] == {"gte": 40}
    # 秒：max_hours 2 小时 + 1 小时余量
    assert query["duration"] == {"gte": (2.0 + 1.0) * 3600}
    assert query["inet_down_cost"] == {"lte": 0.01}
    assert query["inet_up_cost"] == {"lte": 0.01}
    assert query["type"] == "ondemand"
    assert query["allocated_storage"] == 40
    assert query["order"] == [["dph_total", "asc"]]
    assert query["limit"] == vast.SEARCH_LIMIT
    assert "datacenter" not in query


def test_search_uses_configured_thresholds_and_datacenter_filter(api):
    route = api.post("/api/v0/bundles/").respond(
        json={
            "offers": [
                offer_row(1, hosting_type=0),
                offer_row(2, hosting_type=1),
                offer_row(3, hosting_type=None, datacenter=True),
            ]
        }
    )
    backend = make_backend(
        reliability=0.99,
        min_inet_down_mbps=1000.0,
        max_bandwidth_usd_per_gb=0.005,
        datacenter_only=True,
    )
    offers = backend.search(REQ)

    query = body_of(route.calls.last)
    assert query["datacenter"] == {"eq": True}
    assert query["reliability"] == {"gte": 0.99}
    assert query["inet_down"] == {"gte": 1000.0}
    assert query["inet_up_cost"] == {"lte": 0.005}
    # inet_down 800 < 1000，全部被客户端过滤掉
    assert offers == []

    api.post("/api/v0/bundles/").respond(
        json={
            "offers": [
                offer_row(1, hosting_type=0, inet_down=2000.0),
                offer_row(2, hosting_type=1, inet_down=2000.0),
                offer_row(3, hosting_type=None, datacenter=True, inet_down=2000.0),
            ]
        }
    )
    assert sorted(offer.id for offer in backend.search(REQ)) == ["2", "3"]


def test_search_normalizes_underscored_gpu_names(api):
    route = api.post("/api/v0/bundles/").respond(json={"offers": [offer_row(1)]})
    req = GpuRequest(
        names=("RTX_3090",), count=1, max_price_per_hour=1, max_hours=1, disk_gb=40, image="x"
    )
    offers = make_backend().search(req)
    assert body_of(route.calls.last)["gpu_name"] == {"in": ["RTX 3090"]}
    assert [offer.id for offer in offers] == ["1"]


BAD_ROWS = {
    "other gpu": {"gpu_name": "RTX 3060"},
    "two gpus": {"num_gpus": 2},
    "unverified": {"verification": "unverified"},
    "verified flag false": {"verified": False},
    "not rentable": {"rentable": False},
    "already rented": {"rented": True},
    "external": {"external": True},
    "interruptible": {"is_bid": True},
    "low reliability": {"reliability": 0.90},
    "no reliability": {"reliability": None, "reliability2": None},
    "slow download": {"inet_down": 100.0},
    "small disk": {"disk_space": 30.0},
    "short max duration": {"duration": 2.5 * 3600},
    "missing duration": {"duration": None},
    "expensive download traffic": {"inet_down_cost": 0.05},
    "expensive upload traffic": {"inet_up_cost": 0.05},
    "missing traffic price": {"inet_up_cost": None},
    "gpu price over limit": {"dph_base": 0.6},
    "disk pushes price over limit": {"dph_base": 0.495},
    "missing price": {"dph_base": None, "dph_total": None},
    "missing storage price": {"storage_cost": None},
    "missing id": {"id": None},
}


@pytest.mark.parametrize("overrides", BAD_ROWS.values(), ids=BAD_ROWS.keys())
def test_search_refilters_client_side(api, overrides):
    bad = offer_row(2, **overrides)
    api.post("/api/v0/bundles/").respond(json={"offers": [offer_row(1), bad, "garbage"]})
    assert [offer.id for offer in make_backend().search(REQ)] == ["1"]


def test_search_falls_back_to_reliability2(api):
    api.post("/api/v0/bundles/").respond(
        json={"offers": [offer_row(1, reliability=None, reliability2=0.99)]}
    )
    [offer] = make_backend().search(REQ)
    assert offer.details["reliability"] == 0.99


def test_search_prices_gpu_plus_disk_and_sorts_by_est_total(api):
    api.post("/api/v0/bundles/").respond(
        json={
            "offers": [
                # 算力最便宜，但磁盘很贵：0.10 + 3.6*40/720 = 0.30
                offer_row(1, dph_base=0.10, storage_cost=3.6),
                # 0.20 + 0.01 = 0.21
                offer_row(2, dph_base=0.20, storage_cost=0.18, gpu_name="RTX 4090"),
                # 0.15 + 0.01 = 0.16
                offer_row(3, dph_base=0.15, storage_cost=0.18),
                # 没有 dph_base 时用 dph_total（偏保守）：0.25 + 0.01 = 0.26
                offer_row(4, dph_base=None, dph_total=0.25),
                offer_row(3, dph_base=0.15, storage_cost=0.18),  # 重复的 offer
            ]
        }
    )
    offers = make_backend().search(REQ)

    assert [offer.id for offer in offers] == ["3", "2", "4", "1"]
    expected_hourly = {"3": 0.16, "2": 0.21, "4": 0.26, "1": 0.30}
    for offer in offers:
        assert offer.backend == "vast"
        assert offer.currency == "USD"
        assert offer.gpu_count == 1
        assert offer.price_per_hour == pytest.approx(expected_hourly[offer.id])
        assert offer.est_total == pytest.approx(expected_hourly[offer.id] * REQ.max_hours)

    best = offers[0]
    assert best.gpu_name == "RTX 3090"
    assert best.region == "Ireland, IE"
    details = best.details
    assert details["reliability"] == 0.995
    assert details["inet_down_mbps"] == 800.0
    assert details["inet_down_usd_per_gb"] == 0.002
    assert details["inet_up_usd_per_gb"] == 0.003
    assert details["geolocation"] == "Ireland, IE"
    assert details["verified"] is True
    assert details["datacenter"] is False
    assert details["machine_id"] == 1003
    assert details["host_id"] == 2003
    assert details["gpu_usd_per_hour"] == pytest.approx(0.15)
    assert details["disk_usd_per_hour"] == pytest.approx(0.01)
    assert details["gpu_ram_gb"] == pytest.approx(24.576)  # REST 的 gpu_ram 是 MB
    assert details["max_duration_days"] == pytest.approx(30.0)


def test_search_accepts_single_object_offers_shape(api):
    # 文档的响应示例把 offers 写成了单个对象
    api.post("/api/v0/bundles/").respond(json={"offers": offer_row(7)})
    assert [offer.id for offer in make_backend().search(REQ)] == ["7"]


def test_search_malformed_response_is_retryable_error(api):
    api.post("/api/v0/bundles/").respond(json={"unexpected": True})
    with pytest.raises(BackendError) as info:
        make_backend().search(REQ)
    assert info.value.retryable


def test_search_bad_request_is_not_retried(api, sleeps):
    route = api.post("/api/v0/bundles/").respond(
        400, json={"error": "invalid_request", "msg": "Invalid json body"}
    )
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).search(REQ)
    assert "Invalid json body" in str(info.value)
    assert not info.value.retryable
    assert route.call_count == 1
    assert sleeps == []


def test_search_retries_429_and_5xx_with_backoff(api, sleeps):
    route = api.post("/api/v0/bundles/")
    route.side_effect = [
        httpx.Response(429, json={"detail": "API requests too frequent"}),
        httpx.Response(502, text="bad gateway"),
        httpx.ConnectError("connection refused"),
        httpx.Response(200, json={"offers": [offer_row(1)]}),
    ]
    offers = make_backend(sleeps).search(REQ)
    assert [offer.id for offer in offers] == ["1"]
    assert route.call_count == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_search_gives_up_after_bounded_retries(api, sleeps):
    route = api.post("/api/v0/bundles/").respond(503, text="unavailable")
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).search(REQ)
    assert info.value.retryable
    assert route.call_count == vast.ATTEMPTS
    assert len(sleeps) == vast.ATTEMPTS - 1


def test_retry_after_header_is_honored_and_long_waits_give_up(api, sleeps):
    route = api.post("/api/v0/bundles/")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(200, json={"offers": []}),
    ]
    make_backend(sleeps).search(REQ)
    assert sleeps == [7.0]

    route.side_effect = [httpx.Response(429, headers={"Retry-After": "3600"})]
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).search(REQ)
    assert info.value.retryable
    assert sleeps == [7.0]


def test_search_quota_exceeded_is_not_retried(api, sleeps):
    route = api.post("/api/v0/bundles/").respond(429, json={"error": "search_quota_exceeded"})
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).search(REQ)
    assert "配额" in str(info.value)
    assert route.call_count == 1
    assert sleeps == []


# ---------------------------------------------------------------- create


def test_create_sends_launch_spec_and_returns_new_contract(api):
    route = api.put("/api/v0/asks/123/").respond(json={"success": True, "new_contract": 9001})
    ref = make_backend().create(OFFER, LAUNCH)

    assert ref == InstanceRef(backend="vast", id="9001", label=LABEL)
    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {
        "image": LAUNCH.image,
        "disk": 40,
        "onstart": "echo bootstrap",
        "runtype": "ssh",
        "label": LABEL,
        "cancel_unavail": True,
        "target_state": "running",
    }
    # 凭据不进实例：不发送 env，请求体里也没有 key
    assert KEY.encode() not in request.content


def test_create_rejects_too_long_onstart_before_calling_api(api):
    route = api.put("/api/v0/asks/123/").respond(json={"success": True, "new_contract": 1})
    too_long = LaunchSpec(
        job_id=JOB,
        attempt=1,
        image="img",
        bootstrap_cmd="x" * (vast.ONSTART_MAX_CHARS + 1),
        disk_gb=40,
        max_price_per_hour=0.5,
    )
    with pytest.raises(BackendError) as info:
        make_backend().create(OFFER, too_long)
    assert not isinstance(info.value, OfferUnavailable)
    assert not route.called

    at_limit = LaunchSpec(
        job_id=JOB,
        attempt=1,
        image="img",
        bootstrap_cmd="x" * vast.ONSTART_MAX_CHARS,
        disk_gb=40,
        max_price_per_hour=0.5,
    )
    assert make_backend().create(OFFER, at_limit).id == "1"


UNAVAILABLE = {
    "success false": (200, {"success": False, "msg": "offer taken"}),
    "404 no_such_ask": (
        404,
        {"success": False, "error": "invalid_args", "msg": "no_such_ask Instance type by id 123"},
    ),
    "410 gone": (410, {"success": False, "error": "no_such_ask", "msg": "no longer available"}),
    "400 no_such_ask": (400, {"success": False, "error": "no_such_ask", "msg": "gone"}),
    "400 not rentable": (400, {"success": False, "msg": "Offer 123 is not rentable"}),
}


@pytest.mark.parametrize("status, payload", UNAVAILABLE.values(), ids=UNAVAILABLE.keys())
def test_create_offer_unavailable(api, status, payload):
    api.put("/api/v0/asks/123/").respond(status, json=payload)
    list_route = api.get("/api/v1/instances/").respond(json={"instances": []})
    with pytest.raises(OfferUnavailable):
        make_backend().create(OFFER, LAUNCH)
    assert not list_route.called  # 结果是确定的，不需要按标签查


def test_create_bad_request_is_a_plain_error(api):
    api.put("/api/v0/asks/123/").respond(
        400, json={"success": False, "error": "invalid_args", "msg": "bad image"}
    )
    with pytest.raises(BackendError) as info:
        make_backend().create(OFFER, LAUNCH)
    assert not isinstance(info.value, OfferUnavailable)
    assert not info.value.retryable
    assert "bad image" in str(info.value)


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection reset"),
        httpx.Response(504, text="gateway timeout"),
        httpx.Response(200, json={"success": True}),  # 成功但没有 new_contract
    ],
    ids=["timeout", "connect error", "5xx", "no new_contract"],
)
def test_create_unknown_outcome_adopts_instance_found_by_label(api, sleeps, outcome):
    put = api.put("/api/v0/asks/123/")
    put.side_effect = [outcome]
    api.get("/api/v1/instances/").respond(
        json={
            "instances": [
                instance_row(1, label="my-own-notebook"),
                instance_row(2, label=None),
                instance_row(3, label=LABEL + "0"),  # -a10：前缀相同但不是同一个标签
                instance_row(4242, label=LABEL, status=None),
            ],
            "next_token": None,
        }
    )
    ref = make_backend(sleeps).create(OFFER, LAUNCH)
    assert ref == InstanceRef(backend="vast", id="4242", label=LABEL)
    assert put.call_count == 1  # 不盲目重试 create
    assert sleeps == [vast.CREATE_CONFIRM_DELAYS_S[0]]


def test_create_unknown_outcome_found_on_second_lookup(api, sleeps):
    api.put("/api/v0/asks/123/").side_effect = [httpx.ReadTimeout("timed out")]
    api.get("/api/v1/instances/").side_effect = [
        httpx.Response(200, json={"instances": []}),
        httpx.Response(200, json={"instances": [instance_row(77)]}),
    ]
    assert make_backend(sleeps).create(OFFER, LAUNCH).id == "77"
    assert sleeps == list(vast.CREATE_CONFIRM_DELAYS_S)


def test_create_unknown_outcome_not_found_is_retryable(api, sleeps):
    put = api.put("/api/v0/asks/123/")
    put.side_effect = [httpx.ReadTimeout("timed out")]
    lookups = api.get("/api/v1/instances/").respond(
        json={"instances": [instance_row(5, label=layout.label(JOB, 2))]}
    )
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).create(OFFER, LAUNCH)
    assert info.value.retryable
    assert not isinstance(info.value, OfferUnavailable)
    assert LABEL in str(info.value)
    assert put.call_count == 1
    assert lookups.call_count == len(vast.CREATE_CONFIRM_DELAYS_S)


def test_create_unknown_outcome_with_failing_lookup_is_not_retryable(api, sleeps):
    api.put("/api/v0/asks/123/").side_effect = [httpx.ReadTimeout("timed out")]
    api.get("/api/v1/instances/").respond(503)
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).create(OFFER, LAUNCH)
    # 无法确认是否已创建：不能让调用方换机再建一个，提示用 gc 清理
    assert not info.value.retryable
    assert "gc" in str(info.value)


def test_create_retries_rate_limited_request(api, sleeps):
    put = api.put("/api/v0/asks/123/")
    put.side_effect = [
        httpx.Response(429, json={"detail": "API requests too frequent endpoint threshold=4.5"}),
        httpx.Response(200, json={"success": True, "new_contract": 31}),
    ]
    assert make_backend(sleeps).create(OFFER, LAUNCH).id == "31"
    assert put.call_count == 2
    assert sleeps == [1.0]


def test_create_rejects_offer_of_other_backend(api):
    with pytest.raises(BackendError):
        make_backend().create(make_offer("1", backend="autodl"), LAUNCH)


# ---------------------------------------------------------------- get

STATUSES = [
    (None, INST_PENDING, ""),
    ("", INST_PENDING, ""),
    ("loading", INST_LOADING, "loading"),
    ("created", INST_LOADING, "created"),
    ("running", INST_RUNNING, "running"),
    ("exited", INST_EXITED, "exited"),
    ("stopped", INST_EXITED, "stopped"),
    ("offline", INST_UNKNOWN, "offline"),
    ("unknown", INST_UNKNOWN, "unknown"),
    ("frobnicating", INST_UNKNOWN, "frobnicating"),
]


@pytest.mark.parametrize("status, phase, raw", STATUSES)
def test_get_maps_actual_status(api, status, phase, raw):
    route = api.get("/api/v0/instances/9001/").respond(
        json={"instances": instance_row(status=status)}
    )
    state = make_backend().get(REF)
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"
    assert state.ref == REF
    assert state.phase == phase
    assert state.raw_status == raw
    assert state.price_per_hour == 0.21
    assert state.created_at == 1762770000.5


def test_get_accepts_bare_instance_object(api):
    api.get("/api/v0/instances/9001/").respond(json=instance_row(status="loading"))
    assert make_backend().get(REF).phase == INST_LOADING


@pytest.mark.parametrize(
    "status, payload",
    [
        (404, {"success": False, "error": "not_found", "msg": "Instance not found"}),
        (200, {"instances": None}),
        (200, {"instances": {}}),
        (200, {"instances": []}),
    ],
    ids=["404", "null", "empty object", "empty list"],
)
def test_get_missing_instance_is_gone(api, status, payload):
    api.get("/api/v0/instances/9001/").respond(status, json=payload)
    state = make_backend().get(REF)
    assert state.phase == INST_GONE
    assert state.ref == REF


def test_get_retries_transient_errors(api, sleeps):
    route = api.get("/api/v0/instances/9001/")
    route.side_effect = [
        httpx.ReadTimeout("timed out"),
        httpx.Response(500),
        httpx.Response(200, json={"instances": instance_row()}),
    ]
    assert make_backend(sleeps).get(REF).phase == INST_RUNNING
    assert sleeps == [1.0, 2.0]


def test_get_unrecognized_payload_is_an_error_not_gone(api):
    api.get("/api/v0/instances/9001/").respond(json={"success": False, "msg": "?"})
    with pytest.raises(BackendError):
        make_backend().get(REF)


def test_get_rejects_non_numeric_id(api):
    with pytest.raises(BackendError):
        make_backend().get(InstanceRef(backend="vast", id="../users/current", label=LABEL))


# ---------------------------------------------------------------- destroy


def test_destroy_sends_delete(api):
    route = api.delete("/api/v0/instances/9001/").respond(json={"success": True})
    assert make_backend().destroy(REF) is None
    request = route.calls.last.request
    assert request.method == "DELETE"
    assert request.headers["Authorization"] == f"Bearer {KEY}"


def test_destroy_is_idempotent(api):
    route = api.delete("/api/v0/instances/9001/")
    route.side_effect = [
        httpx.Response(200, json={"success": True}),
        httpx.Response(404, json={"success": False, "error": "not_found", "msg": "not found"}),
        httpx.Response(200, json={"success": False, "msg": "Instance not found"}),
    ]
    backend = make_backend()
    backend.destroy(REF)
    backend.destroy(REF)
    backend.destroy(REF)
    assert route.call_count == 3


def test_destroy_retries_transient_failures(api, sleeps):
    route = api.delete("/api/v0/instances/9001/")
    route.side_effect = [
        httpx.ConnectError("connection refused"),
        httpx.Response(429),
        httpx.Response(502),
        httpx.Response(200, json={"success": True}),
    ]
    make_backend(sleeps).destroy(REF)
    assert route.call_count == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_destroy_persistent_failure_warns_about_billing(api, sleeps):
    route = api.delete("/api/v0/instances/9001/").respond(503)
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).destroy(REF)
    assert route.call_count == vast.DESTROY_ATTEMPTS
    assert len(sleeps) == vast.DESTROY_ATTEMPTS - 1
    assert max(sleeps) <= vast.BACKOFF_MAX_S
    assert "计费" in str(info.value)
    assert "9001" in str(info.value)
    assert info.value.retryable


def test_destroy_refused_by_platform_is_an_error(api):
    api.delete("/api/v0/instances/9001/").respond(json={"success": False, "msg": "busy"})
    with pytest.raises(BackendError) as info:
        make_backend().destroy(REF)
    assert "计费" in str(info.value)


# ---------------------------------------------------------------- list


def test_list_returns_only_matching_labels(api):
    other_job = layout.label("other-261110120000-cd34", 1)
    api.get("/api/v1/instances/").respond(
        json={
            "instances": [
                instance_row(1, label=None),
                instance_row(2, label=""),
                instance_row(3, label="my-jupyter"),
                instance_row(4, label="DGPU-" + JOB + "-a1"),
                instance_row(5, label="xdgpu-" + JOB + "-a1"),
                instance_row(6, label=LABEL, status="exited"),
                instance_row(7, label=other_job),
                {"label": layout.label(JOB, 3)},  # 没有 id，跳过
            ],
            "next_token": None,
        }
    )
    backend = make_backend()

    states = backend.list()
    assert sorted((s.ref.id, s.ref.label) for s in states) == [("6", LABEL), ("7", other_job)]
    exited = next(s for s in states if s.ref.id == "6")
    assert exited.phase == INST_EXITED
    assert exited.ref.backend == "vast"
    # gc 的年龄判断要用创建时间（start_date，epoch 秒）
    assert all(s.created_at == 1762770000.5 for s in states)

    assert [s.ref.id for s in backend.list(f"dgpu-{JOB}-")] == ["6"]


@pytest.mark.parametrize("prefix", ["", "my-", "dgp"])
def test_list_refuses_non_dgpu_prefix(api, prefix):
    with pytest.raises(ValueError):
        make_backend().list(prefix)


def test_list_follows_pagination(api):
    route = api.get("/api/v1/instances/")
    route.side_effect = [
        httpx.Response(200, json={"instances": [instance_row(1)], "next_token": "tok-1"}),
        httpx.Response(200, json={"instances": [instance_row(2)], "next_token": "tok-2"}),
        httpx.Response(200, json={"instances": [instance_row(3)], "next_token": None}),
    ]
    states = make_backend().list()
    assert [s.ref.id for s in states] == ["1", "2", "3"]

    first, second, third = (call.request.url.params for call in route.calls)
    assert first["limit"] == "25"
    assert json.loads(first["order_by"]) == [{"col": "id", "dir": "asc"}]
    assert "after_token" not in first
    assert second["after_token"] == "tok-1"
    assert third["after_token"] == "tok-2"


def test_list_repeated_page_token_is_an_error(api):
    api.get("/api/v1/instances/").respond(json={"instances": [], "next_token": "same"})
    with pytest.raises(BackendError) as info:
        make_backend().list()
    assert info.value.retryable


def test_list_falls_back_to_v0_endpoint(api):
    api.get("/api/v1/instances/").respond(404, json={"error": "not_found", "msg": "no route"})
    api.get("/api/v0/instances/").respond(
        json={"instances_found": 2, "instances": [instance_row(8), instance_row(9, label=None)]}
    )
    assert [s.ref.id for s in make_backend().list()] == ["8"]


def test_list_retries_then_fails_loudly(api, sleeps):
    api.get("/api/v1/instances/").respond(500)
    with pytest.raises(BackendError) as info:
        make_backend(sleeps).list()
    assert info.value.retryable
    assert len(sleeps) == vast.ATTEMPTS - 1


# ---------------------------------------------------------------- balance


def test_balance_reads_credit(api):
    route = api.get("/api/v0/users/current/").respond(
        json={"id": 1, "email": "x@example.com", "credit": 25.5, "balance": -3.0}
    )
    assert make_backend().balance() == Balance(25.5, "USD")
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"


def test_balance_without_credit_field_is_none(api):
    api.get("/api/v0/users/current/").respond(json={"id": 1, "balance": 10.0})
    assert make_backend().balance() is None


# 2026-11-10 在真实 API 上看到的原文（key 有效，但没有读取账户信息的权限）
NO_PRIVILEGE_MSG = (
    "Your key lacks proper privileges to perform this action as it requires you to have "
    "logged in using Two Factor Authentication"
)


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "error": "auth_error", "msg": NO_PRIVILEGE_MSG},
        {"success": False, "msg": NO_PRIVILEGE_MSG},
        {"detail": NO_PRIVILEGE_MSG},
    ],
)
def test_balance_without_account_permission_is_unknown_not_an_invalid_key(
    api, sleeps, caplog, payload
):
    route = api.get("/api/v0/users/current/").respond(401, json=payload)
    backend = make_backend(sleeps)
    caplog.set_level("WARNING")
    assert backend.balance() is None
    assert backend.balance() is None
    assert backend.balance_note == "key 没有读取账户信息的权限"
    assert route.call_count == 2 and sleeps == []  # 不重试
    warnings = [r for r in caplog.records if "cannot read account info" in r.getMessage()]
    assert len(warnings) == 1  # 只警告一次
    assert KEY not in caplog.text


def test_balance_with_an_invalid_key_is_still_a_hard_failure(api, sleeps):
    api.get("/api/v0/users/current/").respond(
        401, json={"success": False, "error": "auth_error", "msg": "Invalid user key"}
    )
    backend = make_backend(sleeps)
    with pytest.raises(BackendError, match="API key"):
        backend.balance()
    assert backend.balance_note == ""


def test_get_reports_the_label_the_platform_actually_stored(api):
    """平台截断或改写了 label：get 返回平台上的 label，编排据此发现并按 ID 销毁。"""
    api.get("/api/v0/instances/9001/").respond(
        json={"instances": instance_row(label=LABEL[:16], status="loading")}
    )
    state = make_backend().get(REF)
    assert state.ref.id == "9001" and state.ref.label == LABEL[:16]
    api.get("/api/v0/instances/9001/").respond(
        json={"instances": instance_row(label=None, status="loading")}
    )
    assert make_backend().get(REF).ref.label == LABEL  # 平台没给 label 时保留创建时的


# ---------------------------------------------------------------- 认证与凭据


def _call_search(backend):
    return backend.search(REQ)


def _call_create(backend):
    return backend.create(OFFER, LAUNCH)


def _call_get(backend):
    return backend.get(REF)


def _call_destroy(backend):
    return backend.destroy(REF)


def _call_list(backend):
    return backend.list()


def _call_balance(backend):
    return backend.balance()


OPERATIONS = {
    "search": ("POST", "/api/v0/bundles/", _call_search),
    "create": ("PUT", "/api/v0/asks/123/", _call_create),
    "get": ("GET", "/api/v0/instances/9001/", _call_get),
    "destroy": ("DELETE", "/api/v0/instances/9001/", _call_destroy),
    "list": ("GET", "/api/v1/instances/", _call_list),
    "balance": ("GET", "/api/v0/users/current/", _call_balance),
}

AUTH_FAILURES = {
    "401": (401, {"success": False, "error": "auth_error", "msg": "Invalid user key"}),
    "403": (403, {"success": False, "error": "forbidden", "msg": "scope missing"}),
    # 搜索接口对无效 key 返回 404；不能把它当作“实例已不存在”或“销毁成功”
    "404 invalid key": (404, {"success": False, "msg": "Invalid user key"}),
}


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
@pytest.mark.parametrize("status, payload", AUTH_FAILURES.values(), ids=AUTH_FAILURES.keys())
def test_auth_failures_ask_to_check_key_and_never_leak_it(api, sleeps, operation, status, payload):
    method, path, call = operation
    route = api.route(method=method, path=path).respond(status, json=payload)
    with pytest.raises(BackendError) as info:
        call(make_backend(sleeps))
    message = str(info.value)
    assert "API key" in message
    assert KEY not in message
    assert KEY not in repr(info.value)
    assert route.call_count == 1  # 认证错误不重试
    assert sleeps == []


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
def test_key_echoed_by_server_is_redacted(api, sleeps, operation):
    method, path, call = operation
    payload = {"success": False, "error": "invalid_args", "msg": f"bad request for {KEY}"}
    api.route(method=method, path=path).respond(400, json=payload)
    api.get("/api/v0/instances/").respond(400, json=payload)  # list 在 v1 返回 400 时退回 v0
    with pytest.raises(BackendError) as info:
        call(make_backend(sleeps))
    assert KEY not in str(info.value)
    assert "***" in str(info.value)


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
def test_key_in_transport_errors_is_redacted(api, sleeps, operation):
    method, path, call = operation
    api.route(method=method, path=path).side_effect = httpx.ConnectError(f"failed {KEY}")
    api.get("/api/v1/instances/").side_effect = httpx.ConnectError(f"failed {KEY}")
    with pytest.raises(BackendError) as info:
        call(make_backend(sleeps))
    assert KEY not in str(info.value)
    assert info.value.__cause__ is None or KEY not in str(info.value.__cause__)


def test_key_never_logged(api, sleeps, caplog):
    caplog.set_level("DEBUG")
    api.post("/api/v0/bundles/").side_effect = httpx.ConnectError(f"failed {KEY}")
    api.put("/api/v0/asks/123/").side_effect = httpx.ReadTimeout(f"timeout {KEY}")
    api.get("/api/v1/instances/").respond(json={"instances": []})
    backend = make_backend(sleeps)
    with pytest.raises(BackendError):
        backend.search(REQ)
    with pytest.raises(BackendError):
        backend.create(OFFER, LAUNCH)
    assert caplog.records  # 重试和结果未知都留下了日志
    assert KEY not in caplog.text


def test_no_authorization_header_without_key(api):
    route = api.post("/api/v0/bundles/").respond(json={"offers": []})
    backend = VastBackend(VastSettings(base_url=BASE), client=httpx.Client(), sleep=lambda s: None)
    backend.search(REQ)
    assert "Authorization" not in route.calls.last.request.headers
