"""Vast.ai 后端：通过 REST API 搜索机器、创建、查询、列出、销毁实例，查询余额。

Vast 的所有怪点（字段名、单位、长度上限、响应形状）都集中在本文件，并注明出处。
出处缩写：
- [API-search]  https://docs.vast.ai/api-reference/search/search-offers.md
- [API-create]  https://docs.vast.ai/api-reference/instances/create-instance.md
- [API-show]    https://docs.vast.ai/api-reference/instances/show-instance.md
- [API-list]    https://docs.vast.ai/api-reference/instances/show-instances.md
- [API-destroy] https://docs.vast.ai/api-reference/instances/destroy-instance.md
- [API-user]    https://docs.vast.ai/api-reference/accounts/show-user.md
- [docs-llms]   https://docs.vast.ai/llms-full.txt（hello world、API key 指南）
- [CLI]         https://github.com/vast-ai/vast-cli 的 vast.py（官方 CLI，2026-11 的 master）
以上都只是文档和源码，没有对真实 API 验证过；未定的假设在注释里标为“假设”。

凭据：API key 只放在每个请求的 Authorization 头里，不进入异常信息、日志和 repr；
所有对外的消息都经过 _redact 过滤一次，防止服务器把 key 回显出来。
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx

from dgpu import layout
from dgpu.contracts import (
    BACKEND_VAST,
    CURRENCY_BY_BACKEND,
    INST_EXITED,
    INST_GONE,
    INST_LOADING,
    INST_PENDING,
    INST_RUNNING,
    INST_UNKNOWN,
    BackendError,
    Balance,
    GpuRequest,
    InstanceRef,
    InstanceState,
    LaunchSpec,
    Offer,
    OfferUnavailable,
    VastSettings,
)

log = logging.getLogger(__name__)

CURRENCY = CURRENCY_BY_BACKEND[BACKEND_VAST]

# ---------------------------------------------------------------- Vast 的常量与怪点

# onstart 字段上限 4048 字符 [API-search 与 API-create 页头的 Key Quirks]。
# CLI 快速入门写的是 16KB，以 REST 文档的更严格值为准。
ONSTART_MAX_CHARS = 4048

# 搜索的机器类型。[API-search] 的枚举是 ondemand|bid|reserved；[docs-llms] 的
# hello world 也用 "ondemand"；[CLI] 发的是 "on-demand"。调研笔记里 "ondemand" 实测有结果。
OFFER_TYPE = "ondemand"

# 一次搜索最多取多少个 offer（按 dph_total 升序）。调研时单次返回约 64 个。
SEARCH_LIMIT = 64

# duration 过滤的单位是秒 [API-search]：“Minimum required rental duration in seconds”；
# [CLI] 的 offers_mult 也把天数乘以 86400。响应里的 duration 同样是秒（[CLI] 显示时除以 86400）。
# 在 max_hours 之外多要 1 小时余量，覆盖启动、最终同步和 watchdog 的收尾时间（设计 4.8）。
DURATION_MARGIN_S = 3600.0

# storage_cost 的单位是 $/GB/月 [API-search]、[CLI]。假设：一个月按 30 天 = 720 小时折算，
# 比 730 小时略贵，偏保守。
HOURS_PER_MONTH = 720.0

# gpu_ram 在 REST 里单位是 MB（CLI 里是 GB）[API-search Key Quirks]；[CLI] 显示时除以 1000。
GPU_RAM_MB_PER_GB = 1000.0

# 被视为“数据中心”机器的 hosting_type 取值。假设：调研笔记实测数据中心 offer 的
# hosting_type=1；metrics 接口用 "secure_cloud" 字符串。datacenter 本身是“虚拟列”，
# 只能用于过滤，响应里不一定有（[docs-llms] Queryable Columns）。
DATACENTER_HOSTING_TYPES = (1, "1", "secure_cloud", "datacenter")

# create 响应里表示 offer 已不可用的错误码和消息片段 [API-create] 的 404/410 示例。
UNAVAILABLE_ERRORS = ("no_such_ask",)
UNAVAILABLE_HINTS = (
    "no_such_ask",
    "not available",
    "no longer available",
    "unavailable",
    "not rentable",
    "already rented",
)

# 表示 API key 有问题的错误码和消息片段。[API-search] 对无效 key 返回 404
# {"msg": "Invalid user key"}，[API-show] 的 401 示例是 error=auth_error，所以不能只看状态码：
# 否则无效 key 下 get 会误报 GONE、destroy 会误报成功，实例继续计费。
AUTH_ERRORS = ("auth_error", "unauthorized", "invalid_api_key", "invalid_user_key")
AUTH_HINTS = ("invalid user key", "invalid api key", "api key")

# 429 且 error=search_quota_exceeded 是当日搜索配额用完，到 UTC 午夜才恢复，重试没有意义 [CLI]。
SEARCH_QUOTA_ERROR = "search_quota_exceeded"

# 实例列表：[CLI] 说 v0 的 /instances 列表已弃用，改为分页的 /api/v1/instances/
# （每页最多 25 条，用 next_token 翻页）[API-list]。v1 不可用（首页 400/404/405）时退回 v0，
# v0 的响应是 {"instances": [...]}（[docs-llms] quickstart）。
LIST_PAGE_SIZE = 25
LIST_MAX_PAGES = 400  # 1 万个实例；防止 next_token 出错时死循环

# ---------------------------------------------------------------- 重试参数

ATTEMPTS = 4  # 幂等调用：1 次 + 3 次重试
DESTROY_ATTEMPTS = 6  # 销毁失败意味着可能继续计费，多试几次
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 30.0
MAX_RETRY_AFTER_S = 60.0  # Retry-After 超过 60 秒就不等了，与 [CLI] 一致
# create 结果未知时按标签查询的等待时间：先等 2 秒查一次，再等 10 秒查一次，
# 给平台一点时间让刚创建的实例出现在列表里。
CREATE_CONFIRM_DELAYS_S = (2.0, 10.0)

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
CREATE_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# actual_status → InstanceState.phase [API-show]、[docs-llms] hello world：
# null（调度中）→ loading（拉镜像）→ running；exited、unknown、offline 不会再变成 running。
_PHASE_BY_STATUS = {
    "loading": INST_LOADING,
    "created": INST_LOADING,
    "running": INST_RUNNING,
    "exited": INST_EXITED,
    "stopped": INST_EXITED,
    "offline": INST_UNKNOWN,
    "unknown": INST_UNKNOWN,
}


def map_status(actual_status: Any) -> str:
    """把 Vast 的 actual_status 映射成 INST_*。null 或空串是 pending，不认识的是 unknown。"""
    if actual_status is None:
        return INST_PENDING
    text = str(actual_status).strip().lower()
    if text in ("", "null", "none"):
        return INST_PENDING
    return _PHASE_BY_STATUS.get(text, INST_UNKNOWN)


class _OutcomeUnknown(Exception):
    """非幂等请求（create）超时、断连或 5xx：平台可能已经执行了，也可能没有。"""


class VastBackend:
    """contracts.Backend 的 Vast.ai 实现。"""

    name = BACKEND_VAST

    def __init__(
        self,
        settings: VastSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=TIMEOUT)
        self._sleep = sleep

    def __repr__(self) -> str:  # 不把 key 打进日志
        return f"VastBackend(base_url={self._base_url!r})"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VastBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ search

    def search(self, req: GpuRequest) -> list[Offer]:
        names = _api_gpu_names(req.names)
        if not names:
            return []
        resp = self._request(
            "POST", "/api/v0/bundles/", what="搜索机器", json_body=self._search_query(req, names)
        )
        if not resp.is_success:
            raise self._error(
                f"Vast 搜索机器失败（{self._describe(resp)}）。没有创建任何实例；"
                "检查 job.yaml 的 gpu、limits 和配置里的 vast 过滤条件。"
            )
        data = _json(resp)
        rows = data.get("offers") if isinstance(data, dict) else None
        if isinstance(rows, dict):  # [API-search] 的响应示例把 offers 写成了单个对象
            rows = [rows]
        if not isinstance(rows, list):
            raise self._error(
                "Vast 搜索机器的响应里没有 offers 列表，可能是 API 变了。"
                "没有创建任何实例；稍后重试。",
                retryable=True,
            )

        wanted = {_norm_gpu_name(name) for name in names}
        offers: list[Offer] = []
        seen: set[str] = set()
        rejected: Counter[str] = Counter()
        for row in rows:
            result = self._offer_from(row, req, wanted) if isinstance(row, dict) else "malformed"
            if isinstance(result, str):
                rejected[result] += 1
            elif result.id not in seen:
                seen.add(result.id)
                offers.append(result)
        if rejected:
            # 服务器漏掉的过滤条件在这里补上；记下来，方便排查“为什么没有机器”。
            log.info(
                "vast search: %d offers returned, %d kept; dropped client-side: %s",
                len(rows),
                len(offers),
                dict(rejected),
            )
        offers.sort(key=lambda offer: (offer.est_total, offer.price_per_hour, offer.id))
        return offers

    def _search_query(self, req: GpuRequest, names: list[str]) -> dict[str, Any]:
        s = self._settings
        query: dict[str, Any] = {
            # 与 [CLI] search offers 的默认条件一致：verified、未被外部使用、可租、未被自己租用
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": req.count},
            # gpu_name 用空格，如 "RTX 3090"（[CLI] 把下划线替换成空格后再发送）
            "gpu_name": {"in": names},
            # REST 字段就叫 reliability（[API-search]；[CLI] 里映射到 reliability2 的别名已注释掉）
            "reliability": {"gte": s.reliability},
            # 假设：inet_down 单位是 Mbit/s（[CLI] 写 Mb/s，[API-search] 写 MB/s，调研实测像 Mbps）
            "inet_down": {"gte": s.min_inet_down_mbps},
            "disk_space": {"gte": req.disk_gb},  # GB
            "duration": {"gte": _min_duration_s(req)},  # 秒
            "inet_down_cost": {"lte": s.max_bandwidth_usd_per_gb},  # $/GB
            "inet_up_cost": {"lte": s.max_bandwidth_usd_per_gb},  # $/GB
            "type": OFFER_TYPE,
            # 让服务器按我们要的磁盘大小计算 dph_total（[API-search]，默认 8 GB）
            "allocated_storage": req.disk_gb,
            "order": [["dph_total", "asc"]],
            "limit": SEARCH_LIMIT,
        }
        if s.datacenter_only:
            query["datacenter"] = {"eq": True}
        return query

    def _offer_from(self, row: dict[str, Any], req: GpuRequest, wanted: set[str]) -> Offer | str:
        """把一行 offer 变成 Offer；不满足条件时返回原因（字符串）。不信任服务器做过的过滤。"""
        s = self._settings
        offer_id = row.get("id")
        if isinstance(offer_id, bool) or not isinstance(offer_id, (int, str)) or offer_id == "":
            return "missing id"
        gpu_name = row.get("gpu_name")
        if not isinstance(gpu_name, str) or _norm_gpu_name(gpu_name) not in wanted:
            return "gpu_name"
        if _num(row, "num_gpus") != req.count:
            return "num_gpus"
        if not _is_verified(row):
            return "verified"
        if row.get("rentable") is False:
            return "rentable"
        if row.get("rented"):  # [CLI] 也在客户端补做这个过滤，None 视为未租用
            return "rented"
        if row.get("external") is True:
            return "external"
        if row.get("is_bid") is True:
            return "interruptible"
        if self._settings.datacenter_only and not _is_datacenter(row):
            return "datacenter"

        reliability = _num(row, "reliability")
        if reliability is None:
            reliability = _num(row, "reliability2")
        checks = (
            ("reliability", reliability, s.reliability, ">="),
            ("inet_down", _num(row, "inet_down"), s.min_inet_down_mbps, ">="),
            ("disk_space", _num(row, "disk_space"), req.disk_gb, ">="),
            ("duration", _num(row, "duration"), _min_duration_s(req), ">="),
            ("inet_down_cost", _num(row, "inet_down_cost"), s.max_bandwidth_usd_per_gb, "<="),
            ("inet_up_cost", _num(row, "inet_up_cost"), s.max_bandwidth_usd_per_gb, "<="),
        )
        for field_name, value, limit, op in checks:
            if value is None:
                return f"missing {field_name}"
            if (op == ">=" and value < limit) or (op == "<=" and value > limit):
                return field_name

        # 价格：dph_base 是不含存储的算力价（$/h），dph_total 含按 allocated_storage 计的存储费。
        # 没有 dph_base 时用 dph_total 作为算力价，偏保守。磁盘价按 storage_cost（$/GB/月）自己算。
        gpu_price = _num(row, "dph_base")
        if gpu_price is None:
            gpu_price = _num(row, "dph_total")
        storage_cost = _num(row, "storage_cost")
        if gpu_price is None or gpu_price < 0:
            return "missing price"
        if storage_cost is None or storage_cost < 0:
            return "missing storage_cost"
        disk_price = storage_cost * req.disk_gb / HOURS_PER_MONTH
        price = gpu_price + disk_price
        if price > req.max_price_per_hour:
            return "price"

        # est_total 不含流量费：流量用量此时未知，单价放在 details 里展示。
        details: dict[str, Any] = {
            "reliability": reliability,
            "inet_down_mbps": _num(row, "inet_down"),
            "inet_up_mbps": _num(row, "inet_up"),
            "inet_down_usd_per_gb": _num(row, "inet_down_cost"),
            "inet_up_usd_per_gb": _num(row, "inet_up_cost"),
            "geolocation": row.get("geolocation") or "",
            "verified": True,
            "datacenter": _is_datacenter(row),
            "machine_id": row.get("machine_id"),
            "host_id": row.get("host_id"),
            "gpu_usd_per_hour": gpu_price,
            "disk_usd_per_hour": disk_price,
            "disk_gb": req.disk_gb,
            "storage_usd_per_gb_month": storage_cost,
            "listed_dph_total": _num(row, "dph_total"),
            "max_duration_days": (_num(row, "duration") or 0.0) / 86400.0,
        }
        gpu_ram_mb = _num(row, "gpu_ram")
        if gpu_ram_mb is not None:
            details["gpu_ram_gb"] = gpu_ram_mb / GPU_RAM_MB_PER_GB
        cuda = _num(row, "cuda_max_good")
        if cuda is not None:
            details["cuda_max_good"] = cuda
        return Offer(
            backend=self.name,
            id=str(offer_id),
            gpu_name=gpu_name,
            gpu_count=req.count,
            price_per_hour=price,
            currency=CURRENCY,
            est_total=price * req.max_hours,
            region=str(row.get("geolocation") or ""),
            details=details,
        )

    # ------------------------------------------------------------ create

    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef:
        if offer.backend != self.name:
            raise BackendError(f"offer {offer.id} 属于 {offer.backend} 后端，不能在 Vast 上创建")
        onstart = launch.bootstrap_cmd
        if len(onstart) > ONSTART_MAX_CHARS:
            raise BackendError(
                f"启动命令有 {len(onstart)} 个字符，超过 Vast onstart 的上限 {ONSTART_MAX_CHARS}。"
                "没有创建实例；这是 bootstrap 命令生成的问题，需要缩短启动命令。"
            )
        # 不发送 env：凭据只通过预签名 URL 下发（设计 4.13），平台主 key 永不进入实例。
        body = {
            "image": launch.image,
            "disk": launch.disk_gb,  # GB [API-create]
            "onstart": onstart,
            "runtype": "ssh",  # ssh 模式下镜像的 ENTRYPOINT 被替换，onstart 每次容器启动都执行
            "label": launch.label,
            # 不能立即启动时报错，而不是留下一个停止状态的实例 [API-create]
            "cancel_unavail": True,
            "target_state": "running",
        }
        path = f"/api/v0/asks/{_path_id(offer.id, 'offer')}/"
        try:
            resp = self._request(
                "PUT",
                path,
                what=f"创建实例（offer {offer.id}）",
                json_body=body,
                timeout=CREATE_TIMEOUT,
                idempotent=False,
            )
        except _OutcomeUnknown as exc:
            return self._recover_create(launch, reason=str(exc))

        data = _json(resp)
        if resp.is_success:
            if isinstance(data, dict) and data.get("success") is False:
                raise OfferUnavailable(
                    self._redact(
                        f"Vast 没有接受 offer {offer.id}（{self._describe(resp)}），换下一台机器。"
                    )
                )
            contract = data.get("new_contract") if isinstance(data, dict) else None
            # 实例 ID 是响应的 new_contract，不是 id [API-create Key Quirks]
            if (
                isinstance(contract, bool)
                or not isinstance(contract, (int, str))
                or not str(contract).strip()
            ):
                return self._recover_create(launch, reason="响应里没有 new_contract")
            return InstanceRef(backend=self.name, id=str(contract).strip(), label=launch.label)

        if resp.status_code in (404, 410) or _says_unavailable(data):
            raise OfferUnavailable(
                self._redact(f"offer {offer.id} 已不可用（{self._describe(resp)}），换下一台机器。")
            )
        raise self._error(
            f"Vast 拒绝创建实例（offer {offer.id}，{self._describe(resp)}）。没有创建实例；"
            "检查镜像名和磁盘大小。"
        )

    def _recover_create(self, launch: LaunchSpec, *, reason: str) -> InstanceRef:
        """create 结果未知：按标签查询。找到就接管，确认没有才允许调用方重试。"""
        label = launch.label
        log.warning(
            "vast create outcome unknown (%s); looking up instance by label %s",
            self._redact(reason),
            label,
        )
        confirmed_absent = False
        last_error = ""
        for delay in CREATE_CONFIRM_DELAYS_S:
            self._sleep(delay)
            try:
                states = self.list(label)
            except BackendError as exc:
                confirmed_absent = False
                last_error = str(exc)
                continue
            # 精确匹配：前缀查询下 -a1 也会匹配到 -a10
            matches = [state for state in states if state.ref.label == label]
            if matches:
                matches.sort(key=lambda state: _id_sort_key(state.ref.id))
                if len(matches) > 1:
                    log.warning(
                        "vast: %d instances carry label %s; adopting %s, gc will handle the rest",
                        len(matches),
                        label,
                        matches[0].ref.id,
                    )
                return matches[0].ref
            confirmed_absent = True
        if confirmed_absent:
            raise BackendError(
                self._redact(
                    f"Vast 创建请求结果未知（{reason}），按标签 {label} 查询确认没有创建实例。"
                    "可以重试或换下一台机器。"
                ),
                retryable=True,
            )
        raise BackendError(
            self._redact(
                f"Vast 创建请求结果未知（{reason}），按标签 {label} 查询也失败了（{last_error}）。"
                f"标签为 {label} 的实例可能已经创建并在计费；稍后运行 dgpu gc 清理，"
                "或在 https://cloud.vast.ai/instances/ 手动检查。"
            ),
            retryable=False,
        )

    # ------------------------------------------------------------ get

    def get(self, ref: InstanceRef) -> InstanceState:
        path = f"/api/v0/instances/{_path_id(ref.id, 'instance')}/"
        resp = self._request("GET", path, what=f"查询实例 {ref.id}")
        if resp.status_code == 404:
            return InstanceState(ref=ref, phase=INST_GONE, raw_status="not found")
        if not resp.is_success:
            raise self._error(f"Vast 查询实例 {ref.id} 失败（{self._describe(resp)}）。")
        data = _json(resp)
        found, inst = _unwrap_instance(data, ref.id)
        if not found:
            return InstanceState(ref=ref, phase=INST_GONE, raw_status="not found")
        if inst is None:
            raise self._error(
                f"Vast 查询实例 {ref.id} 的响应无法识别（{self._describe(resp)}），稍后重试。",
                retryable=True,
            )
        return _state_from_row(inst, ref)

    # ------------------------------------------------------------ destroy

    def destroy(self, ref: InstanceRef) -> None:
        path = f"/api/v0/instances/{_path_id(ref.id, 'instance')}/"
        warning = (
            f"实例 {ref.id}（{ref.label}）可能仍在计费：稍后运行 dgpu gc，"
            "或在 https://cloud.vast.ai/instances/ 手动销毁。"
        )
        try:
            # [CLI] 的 destroy 也带一个空 JSON body
            resp = self._request(
                "DELETE", path, what=f"销毁实例 {ref.id}", json_body={}, attempts=DESTROY_ATTEMPTS
            )
        except BackendError as exc:
            raise BackendError(f"{exc} {warning}", retryable=exc.retryable) from None
        if resp.status_code == 404:  # 已经不存在，幂等
            return
        data = _json(resp)
        if resp.is_success:
            if isinstance(data, dict) and data.get("success") is False:
                if _says_not_found(data):
                    return
                raise self._error(
                    f"Vast 没有受理销毁实例 {ref.id}（{self._describe(resp)}）。{warning}",
                    retryable=True,
                )
            return
        raise self._error(f"Vast 销毁实例 {ref.id} 失败（{self._describe(resp)}）。{warning}")

    # ------------------------------------------------------------ list

    def list(self, label_prefix: str = layout.LABEL_PREFIX) -> list[InstanceState]:
        # 只碰 dgpu 的实例（红线 3）：前缀必须以 dgpu- 开头，空前缀会匹配所有带标签的实例。
        if not label_prefix or not label_prefix.startswith(layout.LABEL_PREFIX):
            raise ValueError(
                f"label_prefix must start with {layout.LABEL_PREFIX!r}, got {label_prefix!r}"
            )
        states = []
        for row in self._fetch_instances():
            label = row.get("label")
            if not isinstance(label, str) or not label.startswith(label_prefix):
                continue  # 没有标签或别人的实例，绝不返回
            instance_id = row.get("id")
            if isinstance(instance_id, bool) or not isinstance(instance_id, (int, str)):
                log.warning("vast list: skipping instance with label %s but no id", label)
                continue
            ref = InstanceRef(backend=self.name, id=str(instance_id), label=label)
            states.append(_state_from_row(row, ref))
        return states

    def _fetch_instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        params = {
            "limit": str(LIST_PAGE_SIZE),
            "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
        }
        seen_tokens: set[str] = set()
        for page in range(LIST_MAX_PAGES):
            resp = self._request("GET", "/api/v1/instances/", what="列出实例", params=params)
            if page == 0 and resp.status_code in (400, 404, 405):
                log.warning(
                    "vast list: /api/v1/instances/ answered %d, falling back to /api/v0/instances/",
                    resp.status_code,
                )
                return self._fetch_instances_v0()
            data = self._instances_page(resp)
            rows.extend(row for row in data["instances"] if isinstance(row, dict))
            token = data.get("next_token")
            if not token:
                return rows
            token = str(token)
            if token in seen_tokens:
                raise self._error(
                    "Vast 实例列表的分页游标重复，列表可能不完整，稍后重试。", retryable=True
                )
            seen_tokens.add(token)
            params["after_token"] = token
        raise self._error(
            f"Vast 实例列表超过 {LIST_MAX_PAGES} 页，列表可能不完整，稍后重试。", retryable=True
        )

    def _fetch_instances_v0(self) -> list[dict[str, Any]]:
        resp = self._request("GET", "/api/v0/instances/", what="列出实例")
        data = self._instances_page(resp)
        return [row for row in data["instances"] if isinstance(row, dict)]

    def _instances_page(self, resp: httpx.Response) -> dict[str, Any]:
        if not resp.is_success:
            raise self._error(
                f"Vast 列出实例失败（{self._describe(resp)}），稍后重试。", retryable=True
            )
        data = _json(resp)
        if not isinstance(data, dict) or not isinstance(data.get("instances"), list):
            raise self._error(
                "Vast 实例列表的响应里没有 instances 列表，可能是 API 变了，稍后重试。",
                retryable=True,
            )
        return data

    # ------------------------------------------------------------ balance

    def balance(self) -> Balance | None:
        resp = self._request("GET", "/api/v0/users/current/", what="查询余额")
        if not resp.is_success:
            raise self._error(f"Vast 查询余额失败（{self._describe(resp)}）。", retryable=True)
        data = _json(resp)
        if not isinstance(data, dict):
            raise self._error("Vast 账户信息的响应无法识别，稍后重试。", retryable=True)
        # 账户余额字段是 credit（[docs-llms] API key 指南的示例响应；[CLI] 的 user_fields）。
        # [API-user] 的 schema 只列了 balance；两者含义不同，没有 credit 时不拿 balance 顶替。
        credit = _num(data, "credit")
        if credit is None:
            return None
        return Balance(credit, CURRENCY)

    # ------------------------------------------------------------ HTTP（重试只在这里处理）

    def _request(
        self,
        method: str,
        path: str,
        *,
        what: str,
        json_body: Any = None,
        params: dict[str, str] | None = None,
        attempts: int = ATTEMPTS,
        timeout: httpx.Timeout = TIMEOUT,
        idempotent: bool = True,
    ) -> httpx.Response:
        """发请求并统一处理暂时性错误，返回其余所有响应（含 4xx）由调用方解释。

        - 401/403，或响应体表明 key 无效：抛 BackendError，提示检查 API key，不重试。
        - 429：被限流的请求没有被执行，任何调用都可以等待后重试（与 [CLI] 一致）；
          当日搜索配额用完不重试。
        - 5xx、超时、连接错误：幂等调用按指数退避重试；非幂等调用（create）结果未知，
          抛 _OutcomeUnknown，由调用方按标签确认。
        """
        url = self._base_url + path
        last = ""
        for attempt in range(1, attempts + 1):
            delay = _backoff(attempt)
            try:
                resp = self._client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=self._headers(),
                    timeout=timeout,
                )
            except httpx.RequestError as exc:
                last = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                if not idempotent:
                    raise _OutcomeUnknown(self._redact(last)) from None
            else:
                if _is_auth_failure(resp):
                    raise self._auth_error(what, resp)
                if resp.status_code == 429:
                    last = self._describe(resp)
                    if _json_field(resp, "error") == SEARCH_QUOTA_ERROR:
                        raise self._error(
                            f"Vast {what}失败：今日搜索配额已用完（{last}），UTC 午夜后恢复。"
                        )
                    retry_after = _retry_after(resp)
                    if retry_after is not None:
                        if retry_after > MAX_RETRY_AFTER_S:
                            raise self._error(
                                f"Vast {what}被限流，要求等待 {retry_after:.0f} 秒"
                                f"（{last}），稍后重试。",
                                retryable=True,
                            )
                        delay = retry_after
                elif resp.status_code >= 500:
                    last = self._describe(resp)
                    if not idempotent:
                        raise _OutcomeUnknown(last)
                else:
                    return resp
            if attempt < attempts:
                log.warning(
                    "vast %s %s: %s; retry %d/%d in %.1fs",
                    method,
                    path,
                    self._redact(last),
                    attempt,
                    attempts - 1,
                    delay,
                )
                self._sleep(delay)
        raise self._error(
            f"Vast {what}失败：重试 {attempts} 次后仍然出错（{last}）。", retryable=True
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "dgpu"}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        return headers

    def _redact(self, text: str) -> str:
        key = self._settings.api_key
        if key and key in text:
            text = text.replace(key, "***")
        return text

    def _error(self, message: str, *, retryable: bool = False) -> BackendError:
        return BackendError(self._redact(message), retryable=retryable)

    def _auth_error(self, what: str, resp: httpx.Response) -> BackendError:
        missing = "" if self._settings.api_key else "当前没有配置 API key。"
        return self._error(
            f"Vast {what}被拒绝（{self._describe(resp)}）：API key 无效、过期或权限不足。{missing}"
            "检查配置文件 [backends.vast] 的 api_key 或环境变量 VAST_API_KEY。"
        )

    def _describe(self, resp: httpx.Response) -> str:
        data = _json(resp)
        detail = ""
        if isinstance(data, dict):
            parts = [str(data[k]) for k in ("error", "msg", "detail") if data.get(k)]
            detail = " / ".join(parts)
        elif resp.content:
            detail = " ".join(resp.text.split())
        detail = detail[:300]
        return self._redact(f"HTTP {resp.status_code}" + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------- 辅助函数


def _backoff(attempt: int) -> float:
    return min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S)


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        value = float(resp.headers["Retry-After"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _json_field(resp: httpx.Response, name: str) -> Any:
    data = _json(resp)
    return data.get(name) if isinstance(data, dict) else None


def _is_auth_failure(resp: httpx.Response) -> bool:
    if resp.status_code in (401, 403):
        return True
    if resp.status_code < 400:
        return False
    data = _json(resp)
    if not isinstance(data, dict):
        return False
    error = str(data.get("error") or "").lower()
    message = str(data.get("msg") or data.get("detail") or "").lower()
    return error in AUTH_ERRORS or any(hint in message for hint in AUTH_HINTS)


def _says_unavailable(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if str(data.get("error") or "").lower() in UNAVAILABLE_ERRORS:
        return True
    message = str(data.get("msg") or "").lower()
    return any(hint in message for hint in UNAVAILABLE_HINTS)


def _says_not_found(data: dict[str, Any]) -> bool:
    error = str(data.get("error") or "").lower()
    message = str(data.get("msg") or "").lower()
    return error in ("not_found", "no_such_instance") or "not found" in message


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _norm_gpu_name(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).casefold()


def _api_gpu_names(names: tuple[str, ...]) -> list[str]:
    """job.yaml 里的 GPU 名转成 API 用的写法：下划线换成空格（[CLI] parse_query）。"""
    result: list[str] = []
    for name in names:
        clean = " ".join(name.replace("_", " ").split())
        if clean and clean not in result:
            result.append(clean)
    return result


def _min_duration_s(req: GpuRequest) -> float:
    return req.max_hours * 3600.0 + DURATION_MARGIN_S


def _is_verified(row: dict[str, Any]) -> bool:
    # verified 是只用于过滤的虚拟列；响应里通常是 verification="verified"（[API-search] 示例）
    if isinstance(row.get("verified"), bool):
        return row["verified"]
    return row.get("verification") == "verified"


def _is_datacenter(row: dict[str, Any]) -> bool:
    value = row.get("datacenter")
    if isinstance(value, bool):
        return value
    return row.get("hosting_type") in DATACENTER_HOSTING_TYPES


def _path_id(value: str, kind: str) -> str:
    text = str(value).strip()
    if not text.isdigit():
        raise BackendError(f"无效的 Vast {kind} ID：{value!r}（应为数字）")
    return text


def _id_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (2**63, value)


def _unwrap_instance(data: Any, instance_id: str) -> tuple[bool, dict[str, Any] | None]:
    """从 show instance 的响应里取出实例。

    返回 (found, row)：found=False 表示平台说不存在；found=True 且 row=None 表示响应无法识别。
    支持 {"instances": {...}}（[API-show]、[CLI]）、{"instances": [...]} 和裸对象三种形状；
    {"instances": null}、空对象、空列表都表示不存在（[CLI] 把 null 当作 “no longer exists”）。
    """
    if not isinstance(data, dict):
        return True, None
    if "instances" in data:
        inner = data["instances"]
        if not inner:
            return False, None
        if isinstance(inner, dict):
            return True, inner
        if isinstance(inner, list):
            rows = [row for row in inner if isinstance(row, dict)]
            for row in rows:
                if str(row.get("id")) == instance_id:
                    return True, row
            if len(rows) == 1 and rows[0].get("id") is None:
                return True, rows[0]
            return False, None
        return True, None
    if "id" in data or "actual_status" in data:
        return True, data
    return True, None


def _state_from_row(row: dict[str, Any], ref: InstanceRef) -> InstanceState:
    raw = row.get("actual_status")
    return InstanceState(
        ref=ref,
        phase=map_status(raw),
        raw_status="" if raw is None else str(raw),
        price_per_hour=_num(row, "dph_total"),
        created_at=_num(row, "start_date"),  # epoch 秒 [API-show]
    )
