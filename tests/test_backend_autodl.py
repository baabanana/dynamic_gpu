"""AutoDL 后端（工作包 D）的测试。

HTTP 全部由 respx 模拟；假平台的请求和响应字段取自官方文档
https://www.autodl.com/docs/esd_api_doc/ 的样例。sleep 注入，测试不真正等待。
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from typing import Any

import httpx
import pytest
import respx

from dgpu import layout
from dgpu.backends import autodl
from dgpu.backends.autodl import AutodlBackend, normalize_gpu_name
from dgpu.contracts import (
    INST_EXITED,
    INST_GONE,
    INST_LOADING,
    INST_PENDING,
    INST_RUNNING,
    INST_UNKNOWN,
    AutodlSettings,
    Backend,
    BackendError,
    ConfigError,
    DgpuError,
    GpuRequest,
    InstanceRef,
    LaunchSpec,
    Offer,
    OfferUnavailable,
)

BASE = "https://api.autodl.com"
TOKEN = "dev-token-SECRET-6f1c2b9e8d7a"
JOB_ID = "yolo-261110120000-ab12"

# ---------------------------------------------------------------- 假平台


def ok(data: Any) -> httpx.Response:
    return httpx.Response(
        200, json={"code": "Success", "msg": "", "data": data, "request_id": "req-ok"}
    )


def api_error(code: str, msg: str, *, status: int = 200, request_id: str = "req-err"):
    return httpx.Response(
        status, json={"code": code, "msg": msg, "data": None, "request_id": request_id}
    )


def deployment(
    uuid: str,
    name: str,
    status: str = "running",
    *,
    starting: int = 0,
    running: int = 0,
    finished: int = 0,
    created_at: str = "2026-11-10T12:00:00+08:00",
) -> dict[str, Any]:
    """部署列表里的一条记录，字段同文档“获取部署列表”的响应样例。"""
    return {
        "id": 214,
        "uid": 58,
        "uuid": uuid,
        "name": name,
        "deployment_type": "Job",
        "status": status,
        "replica_num": 1,
        "parallelism_num": 1,
        "reuse_container": False,
        "service_port_protocol": "http",
        "starting_num": starting,
        "running_num": running,
        "finished_num": finished,
        "image_uuid": "image-db8346e037",
        "template": {"dc_list": ["westDC2"], "gpu_name_set": ["RTX 4090"], "gpu_num": 1},
        "price_estimates": 0,
        "created_at": created_at,
        "updated_at": created_at,
        "stopped_at": None,
    }


def container(
    deployment_uuid: str,
    status: str,
    *,
    suffix: str = "c1",
    price: float = 1881,
    gpu_num: int = 1,
    created_at: str = "2026-11-10T12:01:00+08:00",
) -> dict[str, Any]:
    """容器列表里的一条记录，字段同文档“查询容器”的响应样例。"""
    return {
        "id": 195,
        "uuid": f"{deployment_uuid}-f94411a60c-{suffix}",
        "data_center": "westDC2",
        "machine_id": "f94411a60c",
        "deployment_uuid": deployment_uuid,
        "status": status,
        "gpu_name": "RTX 4090",
        "gpu_num": gpu_num,
        "cpu_num": 16,
        "memory_size": 2147483648,
        "image_uuid": "image-db8346e037",
        "price": price,
        "info": {"ssh_command": "ssh -p 21305 root@region-1.autodl.com", "root_password": "x"},
        "started_at": created_at,
        "stopped_at": None,
        "created_at": created_at,
        "updated_at": created_at,
    }


class FakeAutodl:
    """按文档行为模拟的弹性部署 API。可以排队注入失败。"""

    def __init__(self, router: respx.MockRouter):
        self.deployments: list[dict[str, Any]] = []
        self.containers: dict[str, list[dict[str, Any]]] = {}
        self.stock: dict[str, list[dict[str, Any]]] = {}
        self.balance_assets = 123456
        self.calls: list[tuple[str, str, Any]] = []
        self.requests: list[httpx.Request] = []
        # 每个端点的失败队列：元素是 httpx.Response 或异常；空了就走正常逻辑
        self.fail: dict[str, list[Any]] = {}
        # create 时先建好部署再返回队列里的失败（模拟“请求超时但其实已创建”）
        self.create_then_fail: list[Any] = []
        # 每个端点在正常处理前调用的钩子，模拟平台侧的并发变化
        self.before: dict[str, Any] = {}
        self.honor_uuid_filter = True
        self.page_size_cap: int | None = None
        self._ids = (f"{n:015x}" for n in itertools.count(0xA000))
        router.post("/api/v1/dev/deployment/list").mock(
            side_effect=self._handle("list", self._list)
        )
        router.post("/api/v1/dev/deployment/container/list").mock(
            side_effect=self._handle("containers", self._containers)
        )
        router.post("/api/v1/dev/deployment").mock(side_effect=self._handle("create", self._create))
        router.delete("/api/v1/dev/deployment").mock(
            side_effect=self._handle("delete", self._delete)
        )
        router.post("/api/v1/dev/machine/region/gpu_stock").mock(
            side_effect=self._handle("stock", self._stock)
        )
        router.post("/api/v1/dev/wallet/balance").mock(
            side_effect=self._handle("balance", self._balance)
        )

    # ---- 辅助

    def add(self, record: dict[str, Any], containers: list[dict[str, Any]] = ()) -> dict:
        self.deployments.append(record)
        self.containers[record["uuid"]] = list(containers)
        return record

    def bodies(self, endpoint: str) -> list[Any]:
        return [body for name, _, body in self.calls if name == endpoint]

    def count(self, endpoint: str) -> int:
        return len(self.bodies(endpoint))

    def _handle(self, endpoint: str, handler):
        def side_effect(request: httpx.Request):
            body = json.loads(request.content) if request.content else None
            self.calls.append((endpoint, request.method, body))
            self.requests.append(request)
            if endpoint in self.before:
                self.before[endpoint]()
            queue = self.fail.get(endpoint)
            if queue:
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return handler(body)

        return side_effect

    # ---- 端点

    def _page(self, items: list[dict[str, Any]], body: dict[str, Any]) -> httpx.Response:
        size = body["page_size"]
        if self.page_size_cap:
            size = min(size, self.page_size_cap)
        page = body["page_index"]
        start = (page - 1) * size
        return ok(
            {
                "list": items[start : start + size],
                "page_index": page,
                "page_size": size,
                "offset": start,
                "max_page": math.ceil(len(items) / size) if items else 0,
                "result_total": len(items),
                "page": page,
            }
        )

    def _list(self, body):
        items = self.deployments
        if body.get("deployment_uuid") and self.honor_uuid_filter:
            items = [d for d in items if d["uuid"] == body["deployment_uuid"]]
        if body.get("name"):
            items = [d for d in items if d["name"] == body["name"]]
        return self._page(items, body)

    def _containers(self, body):
        return self._page(self.containers.get(body["deployment_uuid"], []), body)

    def _create(self, body):
        uuid = next(self._ids)
        self.add(deployment(uuid, body["name"], "running"))
        if self.create_then_fail:
            item = self.create_then_fail.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return ok({"deployment_uuid": uuid})

    def _delete(self, body):
        uuid = body["deployment_uuid"]
        before = len(self.deployments)
        self.deployments = [d for d in self.deployments if d["uuid"] != uuid]
        if len(self.deployments) == before:
            return api_error("NotFound", "部署不存在")  # 假设的错误；后端不依赖它
        self.containers.pop(uuid, None)
        return ok(None)

    def _stock(self, body):
        return ok(self.stock.get(body["region_sign"], []))

    def _balance(self, body):
        return ok({"assets": self.balance_assets, "accumulate": 1000, "voucher_balance": 0})


@pytest.fixture
def router():
    with respx.mock(base_url=BASE, assert_all_called=False, assert_all_mocked=True) as mock:
        yield mock


@pytest.fixture
def platform(router) -> FakeAutodl:
    return FakeAutodl(router)


@pytest.fixture
def sleeps() -> list[float]:
    return []


def make_backend(sleeps: list[float], **settings: Any) -> AutodlBackend:
    settings.setdefault("token", TOKEN)
    return AutodlBackend(AutodlSettings(**settings), client=httpx.Client(), sleep=sleeps.append)


@pytest.fixture
def backend(platform, sleeps) -> AutodlBackend:
    return make_backend(sleeps, dc_list=("westDC2", "westDC3"))


def request(names=("RTX 4090",), count=1, price=2.5, hours=4.0) -> GpuRequest:
    return GpuRequest(
        names=tuple(names),
        count=count,
        max_price_per_hour=price,
        max_hours=hours,
        disk_gb=40,
        image="image-db8346e037",
    )


def launch(attempt: int = 1, price: float = 2.5, cmd: str = "echo bootstrap 00000000"):
    return LaunchSpec(
        job_id=JOB_ID,
        attempt=attempt,
        image="image-db8346e037",
        bootstrap_cmd=cmd,
        disk_gb=40,
        max_price_per_hour=price,
    )


def offer(region="westDC2", gpu="RTX 4090", count=1) -> Offer:
    return Offer(
        backend="autodl",
        id=f"{region}/{gpu}",
        gpu_name=gpu,
        gpu_count=count,
        price_per_hour=2.5,
        currency="CNY",
        est_total=10.0,
        region=region,
    )


def ref(uuid: str, attempt: int = 1) -> InstanceRef:
    return InstanceRef(backend="autodl", id=uuid, label=layout.label(JOB_ID, attempt))


LABEL = layout.label(JOB_ID, 1)


# ---------------------------------------------------------------- 协议与鉴权


def test_backend_satisfies_protocol_and_repr_hides_token(backend):
    assert isinstance(backend, Backend)
    assert backend.name == "autodl"
    assert TOKEN not in repr(backend)


def test_every_request_sends_raw_token_in_authorization_header(backend, platform):
    platform.stock["westDC2"] = [{"RTX 4090": {"idle_gpu_num": 3, "total_gpu_num": 9}}]
    backend.search(request())
    backend.balance()
    ref_ = backend.create(offer(), launch())
    backend.get(ref_)
    backend.list()
    backend.destroy(ref_)
    assert platform.requests
    for sent in platform.requests:
        assert sent.headers["Authorization"] == TOKEN  # 文档：headers = {"Authorization": token}
        assert sent.url.host == "api.autodl.com"
    posted = [r for r in platform.requests if r.content]
    assert all(r.headers["Content-Type"] == "application/json" for r in posted)


def test_default_client_uses_settings_base_url(platform):
    backend = AutodlBackend(AutodlSettings(token=TOKEN, base_url=BASE + "/"))
    try:
        assert backend.balance().amount == pytest.approx(123.456)
    finally:
        backend.close()
    assert str(platform.requests[0].url) == BASE + "/api/v1/dev/wallet/balance"


def test_missing_token_is_config_error_without_any_request(platform, sleeps):
    backend = make_backend(sleeps, token="")
    with pytest.raises(ConfigError, match="AUTODL_TOKEN"):
        backend.balance()
    assert platform.calls == []


def test_envelope_error_with_http_200_includes_code_msg_and_request_id(backend, platform):
    platform.fail["balance"] = [api_error("AuthFailed", "token 无效", request_id="rid-42")]
    with pytest.raises(BackendError) as info:
        backend.balance()
    text = str(info.value)
    assert "AuthFailed" in text and "token 无效" in text and "rid-42" in text
    assert info.value.retryable is False


def test_http_401_hints_at_token_and_enterprise_verification(backend, platform):
    platform.fail["balance"] = [api_error("Unauthorized", "unauthorized", status=401)]
    with pytest.raises(BackendError, match="企业认证"):
        backend.balance()


def test_unparseable_success_response_is_an_error(backend, platform):
    platform.fail["balance"] = [httpx.Response(200, text="<html>gateway</html>")]
    with pytest.raises(BackendError, match="无法识别"):
        backend.balance()


# ---------------------------------------------------------------- 瞬时错误重试


def test_idempotent_call_retries_429_5xx_and_timeouts_then_succeeds(backend, platform, sleeps):
    platform.fail["balance"] = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(502),
        httpx.ReadTimeout("timed out"),
    ]
    assert backend.balance().amount == pytest.approx(123.456)
    assert platform.count("balance") == 4
    assert sleeps == [7.0, 2.0, 4.0]  # Retry-After 生效，其余按退避表


def test_idempotent_call_gives_up_after_bounded_retries(backend, platform, sleeps):
    platform.fail["balance"] = [httpx.Response(503)] * 10
    with pytest.raises(BackendError) as info:
        backend.balance()
    assert info.value.retryable is True
    assert platform.count("balance") == 1 + len(autodl.TRANSIENT_RETRY_DELAYS_S)
    assert sleeps == list(autodl.TRANSIENT_RETRY_DELAYS_S)


def test_retry_after_is_capped(backend, platform, sleeps):
    platform.fail["balance"] = [httpx.Response(429, headers={"Retry-After": "3600"})]
    backend.balance()
    assert sleeps == [autodl.MAX_RETRY_AFTER_S]


# ---------------------------------------------------------------- search


def test_search_queries_each_configured_region_with_documented_filters(backend, platform):
    backend.search(request(price=2.5))
    bodies = platform.bodies("stock")
    assert [b["region_sign"] for b in bodies] == ["westDC2", "westDC3"]
    assert bodies[0] == {
        "region_sign": "westDC2",
        "cuda_v_from": 118,
        "cuda_v_to": autodl.CUDA_V_TO,
        "price_from": autodl.PRICE_FROM_MILLI,
        "price_to": 2500,  # 元 × 1000
    }


def test_search_without_dc_list_queries_all_documented_regions(platform, sleeps):
    make_backend(sleeps).search(request())
    regions = [b["region_sign"] for b in platform.bodies("stock")]
    assert regions == list(autodl.DEFAULT_REGIONS)
    assert {"westDC2", "westDC3", "beijingDC1", "yangzhouDC1", "neimengDC3"} <= set(regions)


def test_search_builds_one_offer_per_region_and_gpu(backend, platform):
    # 文档样例格式：data 是 [{型号: {idle_gpu_num, total_gpu_num}}, ...]
    platform.stock["westDC2"] = [
        {"RTX 4090": {"idle_gpu_num": 215, "total_gpu_num": 2285}},
        {"RTX 3080 Ti": {"idle_gpu_num": 20, "total_gpu_num": 392}},
        {"RTX A4000": {"idle_gpu_num": 6, "total_gpu_num": 24}},
    ]
    platform.stock["westDC3"] = [{"RTX 4090": {"idle_gpu_num": 900, "total_gpu_num": 1000}}]
    offers = backend.search(request(names=("RTX 4090",), price=2.5, hours=4))
    assert [(o.region, o.gpu_name) for o in offers] == [
        ("westDC3", "RTX 4090"),  # est_total 相同，按空闲卡数降序
        ("westDC2", "RTX 4090"),
    ]
    first = offers[0]
    assert first.backend == "autodl"
    assert first.id == "westDC3/RTX 4090"
    assert first.currency == "CNY"
    assert first.gpu_count == 1
    assert first.price_per_hour == 2.5
    assert first.est_total == pytest.approx(10.0)
    assert first.details["idle_gpu_num"] == 900
    assert first.details["total_gpu_num"] == 1000
    assert first.details["price_is_upper_bound"] is True
    assert "multi_gpu_warning" not in first.details


def test_search_skips_gpus_with_too_few_idle_cards_and_warns_for_multi_gpu(backend, platform):
    platform.stock["westDC2"] = [{"RTX 4090": {"idle_gpu_num": 1, "total_gpu_num": 8}}]
    platform.stock["westDC3"] = [{"RTX 4090": {"idle_gpu_num": 4, "total_gpu_num": 8}}]
    offers = backend.search(request(count=2, price=6.0))
    assert [o.region for o in offers] == ["westDC3"]
    assert offers[0].gpu_count == 2
    assert "调度不上" in offers[0].details["multi_gpu_warning"]
    # 多卡时价格上限按卡分摊（假设：AutoDL 价格是每卡基准价）
    assert platform.bodies("stock")[0]["price_to"] == 3000


def test_search_accepts_stock_under_a_list_key(backend, platform):
    platform.stock["westDC2"] = {"list": [{"RTX 4090": {"idle_gpu_num": 2, "total_gpu_num": 3}}]}
    assert [o.id for o in backend.search(request())] == ["westDC2/RTX 4090"]


def test_search_matches_gpu_names_tolerantly(backend, platform):
    platform.stock["westDC2"] = [
        {"RTX 3090": {"idle_gpu_num": 5, "total_gpu_num": 10}},
        {"RTX 4090D": {"idle_gpu_num": 5, "total_gpu_num": 10}},
        {"RTX 3080 Ti": {"idle_gpu_num": 5, "total_gpu_num": 10}},
    ]
    offers = backend.search(request(names=("3090", "rtx-4090", "RTX 3080")))
    assert [o.gpu_name for o in offers] == ["RTX 3090"]  # 4090 ≠ 4090D，3080 ≠ 3080 Ti
    assert offers[0].id == "westDC2/RTX 3090"  # id 里用平台的原名


@pytest.mark.parametrize(
    ("user", "platform_name", "match"),
    [
        ("3090", "RTX 3090", True),
        ("RTX 3090", "RTX 3090", True),
        ("rtx3090", "RTX 3090", True),
        ("NVIDIA GeForce RTX 4090", "RTX 4090", True),
        ("A4000", "RTX A4000", True),
        ("3080ti", "RTX 3080 Ti", True),
        ("RTX 4090", "RTX 4090D", False),
        ("RTX 3080", "RTX 3080 Ti", False),
        ("V100", "Tesla V100-SXM2-32GB", False),
        ("Tesla V100-SXM2-32GB", "Tesla V100-SXM2-32GB", True),
    ],
)
def test_gpu_name_rule(user, platform_name, match):
    assert (normalize_gpu_name(user) == normalize_gpu_name(platform_name)) is match


def test_search_survives_one_failing_region(backend, platform, caplog):
    platform.fail["stock"] = [api_error("RegionError", "地区不可用")]
    platform.stock["westDC3"] = [{"RTX 4090": {"idle_gpu_num": 2, "total_gpu_num": 3}}]
    with caplog.at_level(logging.WARNING):
        offers = backend.search(request())
    assert [o.region for o in offers] == ["westDC3"]
    assert "westDC2" in caplog.text


def test_search_fails_when_every_region_fails(backend, platform):
    platform.fail["stock"] = [api_error("Boom", "x"), api_error("Boom", "y")]
    with pytest.raises(BackendError, match="所有地区"):
        backend.search(request())


def test_search_with_no_matching_names_returns_empty(backend, platform):
    platform.stock["westDC2"] = [{"RTX 4090": {"idle_gpu_num": 2, "total_gpu_num": 3}}]
    assert backend.search(request(names=("H800",))) == []


# ---------------------------------------------------------------- create


def test_create_sends_job_deployment_with_documented_fields(backend, platform):
    created = backend.create(offer("westDC3", "RTX 4090"), launch(price=1.88))
    (body,) = platform.bodies("create")
    assert body == {
        "name": LABEL,
        "deployment_type": "Job",
        "replica_num": 1,
        "parallelism_num": 1,
        "reuse_container": False,
        "container_template": {
            "dc_list": ["westDC3"],
            "gpu_name_set": ["RTX 4090"],
            "gpu_num": 1,
            "cuda_v_from": 118,
            "cuda_v_to": autodl.CUDA_V_TO,
            "cpu_num_from": 1,
            "cpu_num_to": 100,
            "memory_size_from": 1,
            "memory_size_to": 256,
            "price_from": autodl.PRICE_FROM_MILLI,
            "price_to": 1880,  # round(1.88 × 1000)
            "image_uuid": "image-db8346e037",
            "cmd": "echo bootstrap 00000000",
        },
    }
    assert created == InstanceRef(backend="autodl", id=platform.deployments[0]["uuid"], label=LABEL)


@pytest.mark.parametrize(("price", "milli"), [(0.1, 100), (2.5, 2500), (1.2345, 1234), (9, 9000)])
def test_price_is_converted_to_yuan_times_1000(backend, platform, price, milli):
    backend.create(offer(), launch(price=price))
    assert platform.bodies("create")[0]["container_template"]["price_to"] == milli


def test_multi_gpu_create_scales_cpu_memory_and_splits_price(backend, platform):
    backend.create(offer(count=4), launch(price=8.0))
    template = platform.bodies("create")[0]["container_template"]
    assert template["gpu_num"] == 4
    assert template["cpu_num_to"] == 400
    assert template["memory_size_to"] == 1024
    assert template["price_to"] == 2000  # 每卡 2 元，整机不超过 8 元


def test_create_uses_configured_cuda_version(platform, sleeps):
    make_backend(sleeps, cuda_v_from=121).create(offer(), launch())
    assert platform.bodies("create")[0]["container_template"]["cuda_v_from"] == 121


def test_create_refuses_a_command_containing_the_token(backend, platform):
    with pytest.raises(BackendError) as info:
        backend.create(offer(), launch(cmd=f"export T={TOKEN}; run"))
    assert TOKEN not in str(info.value)
    assert platform.count("create") == 0


def test_create_rejects_offers_from_other_backends(backend, platform):
    foreign = Offer("vast", "12345", "RTX 4090", 1, 0.3, "USD", 1.2)
    with pytest.raises(BackendError, match="不是 AutoDL"):
        backend.create(foreign, launch())
    assert platform.count("create") == 0


@pytest.mark.parametrize(
    ("code", "msg"),
    [
        ("NoStock", "库存不足"),
        ("Fail", "当前地区暂无可用GPU资源"),
        ("ResourceInsufficient", "no available machine in price range"),
    ],
)
def test_create_no_stock_is_offer_unavailable(backend, platform, code, msg):
    platform.fail["create"] = [api_error(code, msg, request_id="rid-stock")]
    with pytest.raises(OfferUnavailable) as info:
        backend.create(offer(), launch())
    assert info.value.retryable is True
    assert "rid-stock" in str(info.value)
    assert platform.count("create") == 1


def test_create_other_api_errors_are_not_offer_unavailable(backend, platform):
    platform.fail["create"] = [api_error("Fail", "余额不足，资源不足时无法创建")]
    with pytest.raises(BackendError) as info:
        backend.create(offer(), launch())
    assert not isinstance(info.value, OfferUnavailable)
    assert info.value.retryable is False
    assert platform.count("list") == 0  # 明确的拒绝不需要按名称查


def test_create_is_never_retried_blindly(backend, platform):
    platform.fail["create"] = [httpx.Response(503)]
    with pytest.raises(BackendError):
        backend.create(offer(), launch())
    assert platform.count("create") == 1


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("timed out"), httpx.Response(502), httpx.Response(200, text="oops")],
    ids=["timeout", "5xx", "garbled"],
)
def test_create_outcome_unknown_then_found_by_name(backend, platform, sleeps, failure):
    platform.create_then_fail = [failure]
    created = backend.create(offer(), launch())
    assert created.label == LABEL
    assert created.id == platform.deployments[0]["uuid"]
    assert platform.count("create") == 1
    assert sleeps[0] == autodl.CREATE_RECOVERY_DELAYS_S[0]


def test_create_timeout_then_not_found_is_retryable(backend, platform, sleeps):
    # 已有一个 a10 的部署：名称以 a1 开头但不是同一个标签，不能被当成刚建好的
    platform.add(deployment("other-a10", layout.label(JOB_ID, 10)))
    platform.fail["create"] = [httpx.ReadTimeout("timed out")]
    with pytest.raises(BackendError) as info:
        backend.create(offer(), launch(attempt=1))
    assert not isinstance(info.value, OfferUnavailable)
    assert info.value.retryable is True
    assert "确认平台上没有" in str(info.value)
    assert platform.count("create") == 1
    assert platform.count("list") == len(autodl.CREATE_RECOVERY_DELAYS_S)
    assert sleeps == list(autodl.CREATE_RECOVERY_DELAYS_S)


def test_create_timeout_appearing_late_is_still_found(backend, platform):
    platform.fail["create"] = [httpx.ReadTimeout("timed out")]

    def appear_on_second_lookup():  # 平台处理得慢：第一次按名称查找时还没有
        if platform.count("list") == 2:
            platform.add(deployment("late-uuid", LABEL))

    platform.before["list"] = appear_on_second_lookup
    created = backend.create(offer(), launch())
    assert created.id == "late-uuid"
    assert platform.count("list") == 2


def test_create_unknown_outcome_with_failing_lookup_is_not_retryable(backend, platform):
    platform.fail["create"] = [httpx.ReadTimeout("timed out")]
    platform.fail["list"] = [httpx.Response(503)] * 100
    with pytest.raises(BackendError) as info:
        backend.create(offer(), launch())
    assert info.value.retryable is False
    assert "dgpu gc" in str(info.value)


def test_create_connect_error_with_failing_lookup_is_retryable(backend, platform):
    platform.fail["create"] = [httpx.ConnectError("connection refused")]
    platform.fail["list"] = [httpx.ConnectError("connection refused")] * 100
    with pytest.raises(BackendError) as info:
        backend.create(offer(), launch())
    assert info.value.retryable is True  # 连接没建立，请求不可能被受理


def test_create_duplicate_name_error_adopts_the_existing_deployment(backend, platform):
    platform.add(deployment("existing-uuid", LABEL))
    platform.fail["create"] = [api_error("Fail", "部署名称已存在")]
    assert backend.create(offer(), launch()).id == "existing-uuid"


def test_create_success_without_uuid_is_resolved_by_name(backend, platform):
    platform.create_then_fail = [ok({})]
    created = backend.create(offer(), launch())
    assert created.id == platform.deployments[0]["uuid"]


# ---------------------------------------------------------------- get


@pytest.mark.parametrize(
    ("status", "phase"),
    [
        ("creating", INST_LOADING),
        ("created", INST_LOADING),
        ("starting", INST_LOADING),
        ("oss_merged", INST_LOADING),
        ("running", INST_RUNNING),
        ("shutting_down", INST_EXITED),
        ("shutdown", INST_EXITED),
        ("rebooting", INST_UNKNOWN),
        ("", INST_UNKNOWN),
    ],
)
def test_get_maps_every_documented_container_status(backend, platform, status, phase):
    platform.add(deployment("d1", LABEL), [container("d1", status)])
    state = backend.get(ref("d1"))
    assert state.phase == phase
    assert f"container={status}" in state.raw_status
    assert state.ref == ref("d1")


def test_get_mapping_table_covers_the_documented_lifecycle():
    documented = ["creating", "created", "starting", "oss_merged", "running"]
    documented += ["shutting_down", "shutdown"]
    assert sorted(autodl.CONTAINER_STATUS_PHASES) == sorted(documented)


def test_get_requests_deployment_by_uuid_and_its_containers(backend, platform):
    platform.add(deployment("d1", LABEL), [container("d1", "running")])
    backend.get(ref("d1"))
    assert platform.bodies("list")[0]["deployment_uuid"] == "d1"
    assert platform.bodies("containers")[0]["deployment_uuid"] == "d1"
    assert platform.bodies("containers")[0]["page_index"] == 1


@pytest.mark.parametrize(
    ("record", "phase"),
    [
        (deployment("d1", LABEL, "running"), INST_PENDING),  # 已受理，还没有容器
        (deployment("d1", LABEL, "running", finished=1), INST_EXITED),
        (deployment("d1", LABEL, "running", starting=1), INST_LOADING),
        (deployment("d1", LABEL, "running", running=1), INST_RUNNING),
        (deployment("d1", LABEL, "stopped", finished=1), INST_EXITED),
        (deployment("d1", LABEL, "stopped"), INST_EXITED),
        (deployment("d1", LABEL, "paused"), INST_UNKNOWN),
    ],
)
def test_get_without_containers_uses_the_deployment_record(backend, platform, record, phase):
    platform.add(record)
    state = backend.get(ref("d1"))
    assert state.phase == phase
    assert f"deployment={record['status']}" in state.raw_status


def test_get_unknown_deployment_status_keeps_raw_status(backend, platform):
    platform.add(deployment("d1", LABEL, "migrating"))
    state = backend.get(ref("d1"))
    assert state.phase == INST_UNKNOWN
    assert state.raw_status == "deployment=migrating"


def test_get_missing_deployment_is_gone(backend, platform):
    platform.add(deployment("other", layout.label(JOB_ID, 2)))
    state = backend.get(ref("d1"))
    assert state.phase == INST_GONE
    assert platform.count("containers") == 0


def test_get_does_not_report_gone_when_the_uuid_filter_is_ignored(backend, platform):
    platform.honor_uuid_filter = False
    platform.page_size_cap = 2
    for n in range(5):
        platform.add(deployment(f"x{n}", layout.label(JOB_ID, n + 2)))
    platform.add(deployment("d1", LABEL, "stopped"))
    assert backend.get(ref("d1")).phase == INST_EXITED


def test_get_prefers_the_most_alive_container_and_reports_price(backend, platform):
    platform.add(
        deployment("d1", LABEL),
        [
            container("d1", "shutdown", suffix="old", created_at="2026-11-10T12:00:00+08:00"),
            container("d1", "running", suffix="new", price=1881, gpu_num=2),
        ],
    )
    state = backend.get(ref("d1"))
    assert state.phase == INST_RUNNING
    assert state.price_per_hour == pytest.approx(1.881 * 2)
    assert state.created_at == pytest.approx(1794283200.0)  # 2026-11-10T12:00:00+08:00


def test_get_counters_override_a_stale_container_list(backend, platform):
    platform.add(deployment("d1", LABEL, starting=1), [container("d1", "shutdown")])
    assert backend.get(ref("d1")).phase == INST_LOADING


# ---------------------------------------------------------------- destroy


def test_destroy_sends_delete_with_deployment_uuid(backend, platform):
    platform.add(deployment("d1", LABEL, running=1), [container("d1", "running")])
    backend.destroy(ref("d1"))
    assert platform.bodies("delete") == [{"deployment_uuid": "d1"}]
    assert platform.deployments == []


def test_destroy_is_idempotent(backend, platform):
    platform.add(deployment("d1", LABEL))
    backend.destroy(ref("d1"))
    backend.destroy(ref("d1"))
    backend.destroy(ref("never-existed"))
    assert platform.count("delete") == 1


def test_destroy_treats_not_found_error_after_race_as_success(backend, platform, sleeps):
    # 查询时还在，DELETE 时已被别人（如 gc）删掉，平台返回“不存在”的错误
    platform.add(deployment("d1", LABEL))
    platform.before["delete"] = platform.deployments.clear
    backend.destroy(ref("d1"))
    assert platform.count("delete") == 1
    assert platform.count("list") == 2  # 再查一次确认已不存在
    assert sleeps == [autodl.DESTROY_RETRY_DELAYS_S[0]]


def test_destroy_retries_transient_failures(backend, platform, sleeps):
    platform.add(deployment("d1", LABEL))
    platform.fail["delete"] = [httpx.Response(500), httpx.ReadTimeout("timed out")]
    backend.destroy(ref("d1"))
    assert platform.deployments == []
    assert platform.count("delete") == 3
    assert sleeps == [1.0, 2.0]


def test_destroy_retries_envelope_errors_with_backoff(backend, platform, sleeps):
    platform.add(deployment("d1", LABEL))
    platform.fail["delete"] = [api_error("Busy", "部署正在停止中，请稍后")]
    backend.destroy(ref("d1"))
    assert platform.deployments == []
    assert sleeps == [autodl.DESTROY_RETRY_DELAYS_S[0]]


def test_destroy_gives_up_after_bounded_attempts(backend, platform, sleeps):
    platform.add(deployment("d1", LABEL))
    platform.fail["delete"] = [api_error("Busy", "系统繁忙", request_id="rid-del")] * 100
    with pytest.raises(BackendError) as info:
        backend.destroy(ref("d1"))
    attempts = 1 + len(autodl.DESTROY_RETRY_DELAYS_S)
    assert platform.count("delete") == attempts
    assert sleeps == list(autodl.DESTROY_RETRY_DELAYS_S)
    assert "rid-del" in str(info.value) and "dgpu gc" in str(info.value)
    assert info.value.retryable is True


def test_destroy_refuses_deployments_without_dgpu_name(backend, platform):
    platform.add(deployment("theirs", "my-inference-service"))
    with pytest.raises(BackendError, match="拒绝删除"):
        backend.destroy(InstanceRef(backend="autodl", id="theirs", label=LABEL))
    assert platform.count("delete") == 0
    assert len(platform.deployments) == 1


# ---------------------------------------------------------------- list


def test_list_returns_only_dgpu_deployments_across_pages(backend, platform):
    platform.page_size_cap = 3  # 平台每页最多 3 条，强制翻页
    platform.add(deployment("f1", "my-inference-service"))
    platform.add(deployment("f2", "dgpu"))
    platform.add(deployment("f3", "DGPU-upper-a1"))
    platform.add(deployment("f4", "api自动创建"))
    for n in range(1, 6):
        platform.add(deployment(f"d{n}", layout.label(JOB_ID, n), running=1))
    platform.add(deployment("e1", layout.label("other-261110120000-ffff", 1), "stopped"))

    states = backend.list()
    assert sorted(s.ref.id for s in states) == ["d1", "d2", "d3", "d4", "d5", "e1"]
    assert all(s.ref.label.startswith(layout.LABEL_PREFIX) for s in states)
    assert {s.phase for s in states if s.ref.id.startswith("d")} == {INST_RUNNING}
    assert next(s for s in states if s.ref.id == "e1").phase == INST_EXITED
    # gc 的年龄判断要用创建时间：部署记录的 created_at（文档样例带 +08:00）
    assert all(s.created_at == pytest.approx(1794283200.0) for s in states)
    pages = [b["page_index"] for b in platform.bodies("list")]
    assert pages == [1, 2, 3, 4]
    assert platform.count("containers") == 0  # list 只看部署记录


def test_list_filters_by_label_prefix(backend, platform):
    platform.add(deployment("d1", layout.label(JOB_ID, 1)))
    platform.add(deployment("x1", layout.label("other-261110120000-ffff", 1)))
    states = backend.list(f"dgpu-{JOB_ID}-")
    assert [s.ref.id for s in states] == ["d1"]


def test_list_never_returns_foreign_deployments_even_with_empty_prefix(backend, platform):
    platform.add(deployment("d1", LABEL))
    platform.add(deployment("x1", "someone-elses-work"))
    assert [s.ref.id for s in backend.list("")] == ["d1"]


def test_list_dedupes_and_stops_without_max_page(backend, platform, router):
    page = {"list": [deployment("d1", LABEL), deployment("d1", LABEL)], "page_index": 1}
    platform.fail["list"] = [ok(page)]
    assert [s.ref.id for s in backend.list()] == ["d1"]
    assert platform.count("list") == 1


def test_list_refuses_to_page_forever(backend, platform, monkeypatch):
    monkeypatch.setattr(autodl, "MAX_PAGES", 3)
    platform.page_size_cap = 1
    for n in range(1, 10):
        platform.add(deployment(f"d{n}", layout.label(JOB_ID, n)))
    with pytest.raises(BackendError, match="分页"):
        backend.list()


# ---------------------------------------------------------------- balance


def test_balance_converts_thousandths_to_yuan(backend, platform):
    platform.balance_assets = 15230
    balance = backend.balance()
    assert balance.amount == pytest.approx(15.23)
    assert balance.currency == "CNY"
    (sent,) = platform.requests
    assert sent.method == "POST" and sent.url.path == "/api/v1/dev/wallet/balance"


def test_balance_without_assets_is_an_error(backend, platform):
    platform.fail["balance"] = [ok({"accumulate": 1})]
    with pytest.raises(BackendError, match="assets"):
        backend.balance()


# ---------------------------------------------------------------- token 不外泄


def _token_leak_scenarios(platform: FakeAutodl):
    echo = api_error("AuthFailed", f"invalid token {TOKEN}", request_id=f"rid-{TOKEN}")
    echo_401 = api_error("AuthFailed", f"bad Authorization: {TOKEN}", status=401)
    echo_5xx = httpx.Response(500, json={"code": "Err", "msg": TOKEN, "request_id": TOKEN})
    return [
        ("balance", lambda b: b.balance(), {"balance": [echo]}),
        ("balance-401", lambda b: b.balance(), {"balance": [echo_401]}),
        ("balance-5xx", lambda b: b.balance(), {"balance": [echo_5xx] * 10}),
        ("timeout", lambda b: b.balance(), {"balance": [httpx.ReadTimeout(TOKEN)] * 10}),
        ("search", lambda b: b.search(request()), {"stock": [echo, echo]}),
        ("create", lambda b: b.create(offer(), launch()), {"create": [echo]}),
        (
            "create-unknown",
            lambda b: b.create(offer(), launch()),
            {"create": [echo_5xx], "list": [echo_5xx] * 100},
        ),
        ("get", lambda b: b.get(ref("d1")), {"list": [echo]}),
        ("list", lambda b: b.list(), {"list": [echo_401]}),
        ("destroy", lambda b: b.destroy(ref("d1")), {"list": [echo] * 100}),
    ]


@pytest.mark.parametrize("index", range(10))
def test_token_never_appears_in_raised_errors_or_logs(platform, sleeps, caplog, index):
    name, call, failures = _token_leak_scenarios(platform)[index]
    platform.fail.update({key: list(value) for key, value in failures.items()})
    backend = make_backend(sleeps, dc_list=("westDC2", "westDC3"))
    with caplog.at_level(logging.DEBUG), pytest.raises(DgpuError) as info:
        call(backend)
    assert TOKEN not in str(info.value), name
    assert TOKEN not in repr(info.value), name
    assert TOKEN not in caplog.text, name
