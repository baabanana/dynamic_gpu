"""AutoDL 后端：弹性部署 API 的 Job 类型（设计 4.12）。

一个 dgpu 实例 = 一个 AutoDL 部署（deployment_type=Job，副本 1，并行 1），部署名就是
dgpu 标签 ``dgpu-<job_id>-a<attempt>``，实例 ID 是部署 UUID。容器只在 ``cmd``
（bootstrap）运行期间存在，cmd 结束容器就关机；部署记录要由 CLI 删除。

字段名、端点、状态字符串取自官方文档（2026-11-10 抓取）：
https://www.autodl.com/docs/esd_api_doc/ 、https://www.autodl.com/docs/elastic_deploy/ 、
https://www.autodl.com/docs/common_api/ 。文档没写、这里只能假设的行为都集中在下面的
常量表里，并标了“假设”，等真实环境探针（设计第 6 节 A1、A4、A5）确认后再改。

所有 HTTP 调用都经过 ``AutodlBackend._request``：鉴权头、响应信封
``{code, msg, data, request_id}`` 的校验、429/5xx/超时的有限重试都只在那里处理。
开发者 token 只出现在请求头里，不进异常信息、日志和 repr。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from dgpu import layout
from dgpu.contracts import (
    BACKEND_AUTODL,
    INST_EXITED,
    INST_GONE,
    INST_LOADING,
    INST_PENDING,
    INST_RUNNING,
    INST_UNKNOWN,
    AutodlSettings,
    BackendError,
    Balance,
    ConfigError,
    GpuRequest,
    InstanceRef,
    InstanceState,
    LaunchSpec,
    Offer,
    OfferUnavailable,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 状态映射
# 来自官方文档，未经真实环境验证（探针 A1）。
# 容器状态：文档没有给出枚举，只在“查询容器事件”的响应样例里按生命周期顺序出现
#   creating → created → starting → oss_merged → running → shutting_down → shutdown。
# 部署状态：文档只在“获取部署列表”的筛选参数里写了 running（部署中）和 stopped（已停止）。
# 不在表里的字符串一律映射为 INST_UNKNOWN，原文保留在 InstanceState.raw_status。
CONTAINER_STATUS_PHASES: dict[str, str] = {
    "creating": INST_LOADING,
    "created": INST_LOADING,
    "starting": INST_LOADING,
    "oss_merged": INST_LOADING,
    "running": INST_RUNNING,
    "shutting_down": INST_EXITED,  # cmd 已结束，容器正在关机
    "shutdown": INST_EXITED,  # 容器已关机，部署记录还在
}
# 查不到容器时按部署记录判断。部署记录不存在 → INST_GONE。
# 部署记录里的计数优先：running_num > 0 → INST_RUNNING；starting_num > 0 → INST_LOADING；
# 状态为 running 但 finished_num 已达到副本数 → INST_EXITED（Job 已结束、记录状态还没变）。
DEPLOYMENT_STATUS_PHASES: dict[str, str] = {
    "running": INST_PENDING,  # 部署中：已受理，还没有调度出容器
    "stopped": INST_EXITED,  # 已停止：不会再调度容器，记录还在
}
# 同一部署有多个容器（或容器与计数不一致）时，取最“活”的那个。
_PHASE_PRIORITY = {
    INST_RUNNING: 0,
    INST_LOADING: 1,
    INST_UNKNOWN: 2,
    INST_PENDING: 3,
    INST_EXITED: 4,
}

# ---------------------------------------------------------------- 端点与平台常量

PATH_DEPLOYMENT = "/api/v1/dev/deployment"  # POST 创建，DELETE 删除
PATH_DEPLOYMENT_LIST = "/api/v1/dev/deployment/list"
PATH_CONTAINER_LIST = "/api/v1/dev/deployment/container/list"
PATH_GPU_STOCK = "/api/v1/dev/machine/region/gpu_stock"
PATH_WALLET_BALANCE = "/api/v1/dev/wallet/balance"

SUCCESS_CODE = "Success"  # 文档：响应代码，成功时为 Success

# 文档附录“创建部署时的 dc_list 或 region_sign 参数值”。autodl.dc_list 为空时查询全部。
DEFAULT_REGIONS: tuple[str, ...] = (
    "westDC2",  # 西北企业区（推荐）
    "westDC3",  # 西北 B 区
    "beijingDC1",  # 北京 A 区
    "beijingDC2",  # 北京 B 区
    "beijingDC4",  # L20 专区（原北京 C 区）
    "beijingDC3",  # V100 专区（原华南 A 区）
    "neimengDC1",  # 内蒙 A 区
    "foshanDC1",  # 佛山区
    "chongqingDC1",  # 重庆 A 区
    "yangzhouDC1",  # 3090 专区
    "neimengDC3",  # 内蒙 B 区
)

DATA_DISK_GB = 50  # 文档：弹性部署容器数据盘固定 50 GB，不能扩容

# 价格单位：元 × 1000（文档：0.1 元填 100）。下限取文档样例里的 10（0.01 元/时）。
PRICE_FROM_MILLI = 10
# 假设：price_from/price_to 和容器的 price 是“每卡”的基准价（与 AutoDL 市场按卡标价、
# 库存按单卡统计一致）。所以多卡时把整机每小时上限除以卡数，保证总价不超过上限。
# 文档没写清楚；探针确认是“每容器”价格后改成 False。
PRICE_IS_PER_GPU = True

# cuda_v_to 是必填项，文档样例最高写到 128。这里放开上限，避免排除驱动更新的机器。
# 假设平台按整数区间比较，不校验具体版本号；如果创建被拒绝，改成真实存在的版本号（如 130）。
CUDA_V_TO = 999

# CPU、内存范围是必填项。文档样例为 1–100 核、1–256 GB（单卡）；调度单元的 CPU 和内存
# 随卡数等比例放大（弹性部署概述“算力调度单元”），所以上限按卡数放大，保持宽松。
CPU_NUM_PER_GPU_MAX = 100
MEMORY_GB_PER_GPU_MAX = 256

# 文档没有错误码表。下面的关键词是假设（探针 A4 确认后改成错误码），用来识别
# “无库存 / 价格范围内没有机器” → OfferUnavailable，和“部署名重复” → 按名称接管。
_NO_STOCK_MARKERS = (
    "库存",
    "资源不足",
    "无可用",
    "没有可用",
    "暂无可用",
    "无空闲",
    "没有空闲",
    "no stock",
    "out of stock",
    "no available",
    "no machine",
    "no idle",
    "insufficient resource",
    "insufficient gpu",
    "not enough gpu",
)
_BALANCE_MARKERS = ("余额", "欠费", "balance", "arrear")  # 余额不足不是换机能解决的
_DUPLICATE_NAME_MARKERS = ("已存在", "重复", "already exist", "duplicate")

# ---------------------------------------------------------------- 重试与分页

TRANSIENT_RETRY_DELAYS_S = (1.0, 2.0, 4.0)  # 幂等调用遇到 429/5xx/超时：最多再试 3 次
MAX_RETRY_AFTER_S = 30.0
CREATE_RECOVERY_DELAYS_S = (2.0, 5.0, 10.0)  # 创建结果未知时，按名称查找的间隔
DESTROY_RETRY_DELAYS_S = (2.0, 5.0, 10.0)  # 删除失败后再试 3 次
PAGE_SIZE = 50
MAX_PAGES = 200  # 超过就报错，不静默截断列表

_CHINA_TZ = timezone(timedelta(hours=8))
_VENDOR_PREFIXES = ("nvidia", "geforce", "tesla", "rtx", "gtx")


# ---------------------------------------------------------------- 异常


class AutodlApiError(BackendError):
    """平台返回了非 Success 的 code（HTTP 200 也算）。"""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        msg: str,
        request_id: str,
        http_status: int,
        retryable: bool = False,
    ):
        super().__init__(message, retryable=retryable)
        self.code = code
        self.msg = msg
        self.request_id = request_id
        self.http_status = http_status


class _TransientError(BackendError):
    """429、5xx、超时或网络错误。sent=False 表示请求确定没有发出（连接没建立）。"""

    def __init__(self, message: str, *, sent: bool, retry_after: float | None = None):
        super().__init__(message, retryable=True)
        self.sent = sent
        self.retry_after = retry_after


class _MalformedResponse(BackendError):
    """HTTP 2xx 但响应无法解析。"""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


# ---------------------------------------------------------------- 后端


class AutodlBackend:
    """AutoDL 弹性部署（Job）后端，实现 contracts.Backend。"""

    name = BACKEND_AUTODL

    def __init__(
        self,
        settings: AutodlSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        self._sleep = sleep

    def __repr__(self) -> str:  # 不含 token
        return f"AutodlBackend(base_url={self._base_url!r}, dc_list={self._settings.dc_list!r})"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------ search

    def search(self, req: GpuRequest) -> list[Offer]:
        """按地区查空闲卡数，每个（地区，GPU 型号）一个 offer。

        GPU 名称的匹配规则见 normalize_gpu_name：两边规范化后相等才算匹配。
        库存接口不返回价格，price_per_hour 填价格上限（details.price_is_upper_bound）。
        """
        wanted = {normalize_gpu_name(name) for name in req.names if name.strip()}
        if not wanted or req.count < 1:
            return []
        price_to = self._price_to_milli(req.max_price_per_hour, req.count)
        if price_to < 1:
            return []
        filters = {
            "cuda_v_from": self._settings.cuda_v_from,
            "cuda_v_to": CUDA_V_TO,
            "price_from": min(PRICE_FROM_MILLI, price_to),
            "price_to": price_to,
        }
        regions = self._settings.dc_list or DEFAULT_REGIONS
        offers: list[Offer] = []
        failures: list[tuple[str, BackendError]] = []
        for region in regions:
            try:
                data = self._request("POST", PATH_GPU_STOCK, {"region_sign": region, **filters})
                stock = _parse_stock(data)
            except BackendError as exc:
                failures.append((region, exc))
                continue
            for gpu_name, idle, total in stock:
                if normalize_gpu_name(gpu_name) not in wanted or idle < req.count:
                    continue
                offers.append(self._make_offer(region, gpu_name, idle, total, req))
        if failures and len(failures) == len(regions):
            region, first = failures[0]
            raise BackendError(
                f"AutoDL 所有地区（{len(regions)} 个）的库存查询都失败了，"
                f"第一个是 {region}：{first}。没有拿到任何报价，稍后重试",
                retryable=all(exc.retryable for _, exc in failures),
            )
        for region, exc in failures:
            log.warning("AutoDL 地区 %s 的库存查询失败，跳过该地区：%s", region, exc)
        offers.sort(
            key=lambda o: (o.est_total, -o.details["idle_gpu_num"], o.region, o.gpu_name)
        )
        return offers

    def _make_offer(
        self, region: str, gpu_name: str, idle: int, total: int, req: GpuRequest
    ) -> Offer:
        price = req.max_price_per_hour
        details: dict[str, Any] = {
            "idle_gpu_num": idle,
            "total_gpu_num": total,
            "price_is_upper_bound": True,
            "price_note": "AutoDL 库存接口不返回价格；这里是价格上限（创建时的 price_to），"
            "实际单价在容器调度后才知道",
            "data_disk_gb": DATA_DISK_GB,
        }
        if req.count > 1:
            details["multi_gpu_warning"] = (
                "库存按单卡统计，空闲卡可能分散在不同主机上，"
                f"需要 {req.count} 卡的容器即使显示有库存也可能调度不上"
            )
        return Offer(
            backend=self.name,
            id=f"{region}/{gpu_name}",
            gpu_name=gpu_name,
            gpu_count=req.count,
            price_per_hour=price,
            currency="CNY",
            est_total=price * req.max_hours,
            region=region,
            details=details,
        )

    # ------------------------------------------------------------ create

    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef:
        region, gpu_name = _parse_offer_id(offer)
        gpu_num = max(1, offer.gpu_count)
        token = self._settings.token
        if token and any(token in text for text in (launch.bootstrap_cmd, launch.image)):
            raise BackendError(
                "拒绝创建 AutoDL 部署：启动命令或镜像字段里出现了 AutoDL token。"
                "部署列表会显示 cmd，平台 key 不能进入实例（设计红线 2），检查 bootstrap 生成逻辑"
            )
        price_to = self._price_to_milli(launch.max_price_per_hour, gpu_num)
        if price_to < 1:
            raise BackendError(
                f"每小时价格上限 {launch.max_price_per_hour} 元太低，换算成 AutoDL 的"
                "“元×1000”后为 0，不可能调度到机器。调高 limits.max_cny_per_hour"
            )
        label = launch.label
        body = {
            "name": label,
            "deployment_type": "Job",
            "replica_num": 1,
            "parallelism_num": 1,
            # 不复用已停止的容器：复用可能带着上一个任务留在磁盘上的文件（含启动文档）
            "reuse_container": False,
            "container_template": {
                "dc_list": [region],
                "gpu_name_set": [gpu_name],
                "gpu_num": gpu_num,
                "cuda_v_from": self._settings.cuda_v_from,
                "cuda_v_to": CUDA_V_TO,
                "cpu_num_from": 1,
                "cpu_num_to": CPU_NUM_PER_GPU_MAX * gpu_num,
                "memory_size_from": 1,
                "memory_size_to": MEMORY_GB_PER_GPU_MAX * gpu_num,
                "price_from": min(PRICE_FROM_MILLI, price_to),
                "price_to": price_to,
                "image_uuid": launch.image,
                "cmd": launch.bootstrap_cmd,
            },
        }
        try:
            data = self._request("POST", PATH_DEPLOYMENT, body, idempotent=False)
        except _TransientError as exc:
            return self._recover_create(launch, exc, sent=exc.sent)
        except _MalformedResponse as exc:
            return self._recover_create(launch, exc, sent=True)
        except AutodlApiError as exc:
            if _is_no_stock(exc):
                raise OfferUnavailable(
                    f"AutoDL 地区 {region} 的 {gpu_name}×{gpu_num} 在价格上限内没有可用机器"
                    f"（code={exc.code!r} msg={exc.msg!r} request_id={exc.request_id!r}）。"
                    "换下一个地区或 GPU 型号"
                ) from None
            if _matches(exc, _DUPLICATE_NAME_MARKERS):  # 之前的请求其实已经建好了
                return self._recover_create(launch, exc, sent=True, fallback=exc)
            raise
        uuid = data.get("deployment_uuid") if isinstance(data, dict) else None
        if not isinstance(uuid, str) or not uuid:
            missing = _MalformedResponse("创建部署的响应里没有 deployment_uuid")
            return self._recover_create(launch, missing, sent=True)
        return InstanceRef(backend=self.name, id=uuid, label=label)

    def _recover_create(
        self,
        launch: LaunchSpec,
        cause: BackendError,
        *,
        sent: bool,
        fallback: BackendError | None = None,
    ) -> InstanceRef:
        """创建结果未知：按部署名（精确匹配）查找。找到就接管，确认没有才允许重试。"""
        label = launch.label
        last_lookup_ok = False
        lookup_error: BackendError | None = None
        for delay in CREATE_RECOVERY_DELAYS_S:
            self._sleep(delay)
            try:
                matches = [state for state in self.list(label) if state.ref.label == label]
            except BackendError as exc:
                last_lookup_ok, lookup_error = False, exc
                continue
            last_lookup_ok = True
            if matches:
                matches.sort(key=lambda s: (s.created_at or 0.0, s.ref.id))
                if len(matches) > 1:
                    log.warning(
                        "AutoDL 上有 %d 个同名部署 %s，接管最早的 %s；其余的由 dgpu gc 清理",
                        len(matches),
                        label,
                        matches[0].ref.id,
                    )
                log.warning("创建部署 %s 的请求失败（%s），但部署已存在，接管它", label, cause)
                return matches[0].ref
        if fallback is not None and last_lookup_ok:
            raise fallback
        if last_lookup_ok:
            raise BackendError(
                f"创建 AutoDL 部署 {label} 失败（{cause}）；已按名称确认平台上没有这个部署，"
                "可以重试或换下一个地区",
                retryable=True,
            )
        if not sent:
            raise BackendError(
                f"创建 AutoDL 部署 {label} 的请求没有发出（{cause}），平台上不会有这个部署；"
                "检查网络后重试",
                retryable=True,
            )
        raise BackendError(
            f"创建 AutoDL 部署 {label} 的结果未知（{cause}），按名称查询也失败了"
            f"（{lookup_error}）。平台上可能已有这个部署并在计费：网络恢复后运行 dgpu gc 清理",
            retryable=False,
        )

    # ------------------------------------------------------------ get

    def get(self, ref: InstanceRef) -> InstanceState:
        if not ref.id:
            return InstanceState(ref=ref, phase=INST_GONE, raw_status="not_found")
        record = self._find_deployment(ref.id)
        if record is None:
            return InstanceState(ref=ref, phase=INST_GONE, raw_status="not_found")
        containers = [
            item
            for item in self._paginate(PATH_CONTAINER_LIST, {"deployment_uuid": ref.id})
            if item.get("deployment_uuid") in (None, "", ref.id)
        ]
        return _state_from_record(record, containers, ref=ref)

    # ------------------------------------------------------------ destroy

    def destroy(self, ref: InstanceRef) -> None:
        """删除部署（运行中的部署平台会先停后删）。已不存在算成功。

        部署列表里查不到时也照样发 DELETE：刚创建的部署可能还没出现在列表里。这时没法核对
        名称（红线 3 的检查只在查到记录时做；ref 本来就来自 dgpu 自己创建或按 dgpu 标签列出
        的部署），平台回答“不存在”之类的错误就算已删除。
        """
        if not ref.id:
            return
        last_error: BackendError | None = None
        attempts = 1 + len(DESTROY_RETRY_DELAYS_S)
        for attempt in range(attempts):
            if attempt:
                self._sleep(DESTROY_RETRY_DELAYS_S[attempt - 1])
            try:
                record = self._find_deployment(ref.id)
            except BackendError as exc:
                last_error = exc
                continue
            if record is not None:
                name = str(record.get("name") or "")
                if not name.startswith(layout.LABEL_PREFIX):
                    raise BackendError(
                        f"拒绝删除 AutoDL 部署 {ref.id}：它的名称 {name!r} 不是 dgpu 标签，"
                        "不是 dgpu 创建的部署（设计红线 3）。检查传入的实例 ID"
                    )
            try:
                self._request("DELETE", PATH_DEPLOYMENT, {"deployment_uuid": ref.id})
                return
            except AutodlApiError as exc:
                if record is None:
                    # 列表里没有、平台也拒绝删除：按“已经不存在”处理
                    log.info("AutoDL 部署 %s 查不到，DELETE 返回 %s，按已删除处理", ref.id, exc)
                    return
                last_error = exc
                log.warning(
                    "删除 AutoDL 部署 %s（%s）失败（第 %d/%d 次）：%s",
                    ref.id,
                    ref.label,
                    attempt + 1,
                    attempts,
                    exc,
                )
            except BackendError as exc:
                last_error = exc
                log.warning(
                    "删除 AutoDL 部署 %s（%s）失败（第 %d/%d 次）：%s",
                    ref.id,
                    ref.label,
                    attempt + 1,
                    attempts,
                    exc,
                )
        raise BackendError(
            f"删除 AutoDL 部署 {ref.id}（{ref.label}）失败，已尝试 {attempts} 次：{last_error}。"
            "部署可能仍在计费：稍后运行 dgpu gc，或在 AutoDL 控制台手动删除",
            retryable=True,
        )

    # ------------------------------------------------------------ list / balance

    def list(self, label_prefix: str = layout.LABEL_PREFIX) -> list[InstanceState]:
        """列出名称以 label_prefix 开头的部署。

        无论 label_prefix 是什么，都只返回名称以 dgpu 标签前缀开头的部署（设计红线 3）。
        状态只按部署记录判断（不逐个查容器）；需要细节时用 get。
        """
        states: list[InstanceState] = []
        seen: set[str] = set()
        for record in self._paginate(PATH_DEPLOYMENT_LIST, {}):
            uuid = record.get("uuid")
            name = record.get("name")
            if not isinstance(uuid, str) or not uuid or not isinstance(name, str):
                continue
            if not (name.startswith(layout.LABEL_PREFIX) and name.startswith(label_prefix)):
                continue
            if uuid in seen:
                continue
            seen.add(uuid)
            states.append(_state_from_record(record, None))
        return states

    def balance(self) -> Balance | None:
        data = self._request("POST", PATH_WALLET_BALANCE)
        assets = data.get("assets") if isinstance(data, dict) else None
        if isinstance(assets, bool) or not isinstance(assets, int | float):
            raise BackendError("AutoDL 余额接口的响应里没有数值字段 data.assets", retryable=True)
        return Balance(assets / 1000, "CNY")

    # ------------------------------------------------------------ 内部

    def _price_to_milli(self, max_price_per_hour: float, gpu_num: int) -> int:
        per_unit = max_price_per_hour / gpu_num if PRICE_IS_PER_GPU else max_price_per_hour
        return round(per_unit * 1000)

    def _find_deployment(self, uuid: str) -> dict[str, Any] | None:
        body = {"page_index": 1, "page_size": 10, "deployment_uuid": uuid}
        data = self._request("POST", PATH_DEPLOYMENT_LIST, body)
        batch = _page_items(data)
        total = data.get("result_total")
        if not batch and _is_int(total) and total > 0:  # 分页从 0 开始：第 1 页是空的
            data = self._request("POST", PATH_DEPLOYMENT_LIST, {**body, "page_index": 0})
            batch = _page_items(data)
        for item in batch:
            if item.get("uuid") == uuid:
                return item
        if not batch:
            return None
        # 返回了别的部署：平台没有按 UUID 筛选。翻完全部记录，避免把存在的部署误判为已销毁。
        for item in self._paginate(PATH_DEPLOYMENT_LIST, {}):
            if item.get("uuid") == uuid:
                return item
        return None

    def _paginate(self, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        """翻完整个列表。文档没写 page_index 从 0 还是从 1 开始，所以不猜，而是看响应：

        - 第 1 页是空的、但 result_total 不为 0：分页从 0 开始，改从第 0 页重新翻；
        - 翻完后条数比 result_total 少：可能从 0 开始、第 1 页其实是第二页，补读第 0 页
          （按记录去重，平台把第 0 页当作第 1 页时也不会重复）。
        """
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(batch: list[dict[str, Any]]) -> None:
            for item in batch:
                key = _item_key(item)
                if key not in seen:
                    seen.add(key)
                    items.append(item)

        page, fetched, zero_based = 1, 0, False
        total: Any = None
        while True:
            data = self._request("POST", path, {**body, "page_index": page, "page_size": PAGE_SIZE})
            batch = _page_items(data)
            fetched += 1
            max_page = data.get("max_page")
            total = data.get("result_total")
            if page == 1 and fetched == 1 and not batch and _is_int(total) and total > 0:
                log.warning(
                    "AutoDL %s 的第 1 页是空的，但 result_total=%d：按从 0 开始的分页重新读取",
                    path,
                    total,
                )
                page, fetched, zero_based = 0, 0, True
                continue
            add(batch)
            if not batch:
                break
            if _is_int(max_page) and max_page > 0:
                if fetched >= max_page:
                    break
            elif _is_int(total):
                if len(items) >= total:
                    break
            elif len(batch) < PAGE_SIZE:
                break
            page += 1
            if fetched >= MAX_PAGES:
                raise BackendError(
                    f"AutoDL {path} 的分页超过 {MAX_PAGES} 页，列表可能不完整，停止查询。"
                    "在控制台清理旧部署后重试"
                )
        if not zero_based and _is_int(total) and 0 < len(items) < total:
            data = self._request("POST", path, {**body, "page_index": 0, "page_size": PAGE_SIZE})
            add(_page_items(data))
            if len(items) < total:
                log.warning(
                    "AutoDL %s 翻完只拿到 %d 条，result_total=%d（列表在翻页期间可能有变化）",
                    path,
                    len(items),
                    total,
                )
        return items

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        idempotent: bool = True,
    ) -> Any:
        """发请求、校验信封，返回 data。唯一处理鉴权和瞬时错误重试的地方。

        幂等调用遇到 429/5xx/超时/网络错误时按 TRANSIENT_RETRY_DELAYS_S 重试；
        非幂等调用（create）不重试，直接抛 _TransientError，由调用方按名称确认结果。
        """
        token = self._settings.token
        if not token:
            raise ConfigError(
                "没有配置 AutoDL 开发者 token：在配置文件 [backends.autodl] 里填 token，"
                "或设置环境变量 AUTODL_TOKEN（控制台 → 设置 → 开发者 Token）"
            )
        headers = {"Authorization": token}
        if body is not None:
            headers["Content-Type"] = "application/json"
        url = self._base_url + path
        attempts = 1 + (len(TRANSIENT_RETRY_DELAYS_S) if idempotent else 0)
        failure: _TransientError | None = None
        for attempt in range(attempts):
            try:
                response = self._client.request(method, url, json=body, headers=headers)
            except httpx.TransportError as exc:
                sent = not isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout)
                failure = _TransientError(
                    self._redact(f"AutoDL {method} {path} 网络错误：{type(exc).__name__} {exc}"),
                    sent=sent,
                )
            else:
                status = response.status_code
                if status != 429 and status < 500:
                    return self._unwrap(method, path, response)
                failure = _TransientError(
                    self._redact(
                        f"AutoDL {method} {path} 暂时不可用：HTTP {status}"
                        f"{_envelope_summary(response)}"
                    ),
                    sent=True,
                    retry_after=_retry_after(response),
                )
            if not idempotent:
                raise failure
            if attempt + 1 < attempts:
                delay = TRANSIENT_RETRY_DELAYS_S[attempt]
                if failure.retry_after is not None:
                    delay = min(max(delay, failure.retry_after), MAX_RETRY_AFTER_S)
                log.warning(
                    "%s；%.0f 秒后重试（第 %d/%d 次）", failure, delay, attempt + 1, attempts - 1
                )
                self._sleep(delay)
        assert failure is not None
        raise _TransientError(
            f"{failure}（已重试 {attempts - 1} 次）。AutoDL API 暂时不可用，稍后重试",
            sent=failure.sent,
        )

    def _unwrap(self, method: str, path: str, response: httpx.Response) -> Any:
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict) or "code" not in payload:
            if response.is_success:
                raise _MalformedResponse(
                    f"AutoDL {method} {path} 返回了无法识别的响应（HTTP {status}，没有 code 字段）"
                )
            raise BackendError(
                f"AutoDL {method} {path} 失败：HTTP {status}{_auth_hint(status)}"
            )
        if payload.get("code") == SUCCESS_CODE and response.is_success:
            return payload.get("data")
        code = self._redact(str(payload.get("code") or ""))
        msg = self._redact(str(payload.get("msg") or ""))
        request_id = self._redact(str(payload.get("request_id") or ""))
        raise AutodlApiError(
            f"AutoDL {method} {path} 返回错误：code={code!r} msg={msg!r} "
            f"request_id={request_id!r}（HTTP {status}）{_auth_hint(status)}",
            code=code,
            msg=msg,
            request_id=request_id,
            http_status=status,
        )

    def _redact(self, text: str) -> str:
        token = self._settings.token
        return text.replace(token, "***") if token else text


# ---------------------------------------------------------------- 纯函数


def normalize_gpu_name(name: str) -> str:
    """GPU 型号的匹配键：小写，去掉非字母数字，再去掉开头的厂商/系列词。

    去掉的前缀：nvidia、geforce、tesla、rtx、gtx（可叠加）。两边的键相等才算匹配，不做前缀或
    模糊匹配，所以 "3090"、"RTX 3090"、"rtx-3090" 互相匹配，但 "RTX 4090" 不匹配
    "RTX 4090D"，"RTX 3080" 不匹配 "RTX 3080 Ti"，"V100" 不匹配 "Tesla V100-SXM2-32GB"
    （带显存或接口后缀的型号要写全名）。
    """
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    changed = True
    while changed:
        changed = False
        for prefix in _VENDOR_PREFIXES:
            if key.startswith(prefix) and len(key) > len(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _parse_offer_id(offer: Offer) -> tuple[str, str]:
    region, sep, gpu_name = offer.id.partition("/")
    if offer.backend != BACKEND_AUTODL or not sep or not region or not gpu_name:
        raise BackendError(
            f"offer {offer.id!r}（后端 {offer.backend!r}）不是 AutoDL search 生成的，"
            "id 应为 <地区码>/<GPU 型号>"
        )
    return region, gpu_name


def _parse_stock(data: Any) -> list[tuple[str, int, int]]:
    """库存响应：文档样例是 [{"RTX 4090": {"idle_gpu_num": .., "total_gpu_num": ..}}, ...]。

    也接受 {"list": [...]}（文档表格写的是 data -> list）和单个 {型号: 库存} 字典。
    """
    if isinstance(data, dict) and isinstance(data.get("list"), list):
        data = data["list"]
    entries: list[tuple[Any, Any]] = []
    if data is None:
        return []
    if isinstance(data, dict):
        entries = list(data.items())
    elif isinstance(data, list):
        for element in data:
            if isinstance(element, dict):
                entries.extend(element.items())
    else:
        raise _MalformedResponse("AutoDL 库存接口的 data 既不是列表也不是对象")
    result = []
    for gpu_name, counts in entries:
        if not isinstance(gpu_name, str) or not isinstance(counts, dict):
            continue
        result.append(
            (gpu_name, _as_int(counts.get("idle_gpu_num")), _as_int(counts.get("total_gpu_num")))
        )
    return result


def _item_key(item: dict[str, Any]) -> str:
    """列表记录的去重键：部署和容器都有 uuid；没有时用整条记录。"""
    uuid = item.get("uuid")
    if isinstance(uuid, str) and uuid:
        return "uuid:" + uuid
    return "json:" + json.dumps(item, sort_keys=True, default=str)


def _page_items(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise _MalformedResponse("AutoDL 列表接口的 data 不是对象")
    batch = data.get("list")
    if batch is None:
        return []
    if not isinstance(batch, list):
        raise _MalformedResponse("AutoDL 列表接口的 data.list 不是列表")
    return [item for item in batch if isinstance(item, dict)]


def _state_from_record(
    record: dict[str, Any],
    containers: list[dict[str, Any]] | None,
    *,
    ref: InstanceRef | None = None,
) -> InstanceState:
    """把部署记录和（可选的）容器列表合成一个 InstanceState，规则见文件开头的映射表。

    ref 的标签取平台上的部署名（调用方据此确认标签没有被平台改写）。
    """
    if ref is None:
        ref = InstanceRef(
            backend=BACKEND_AUTODL, id=str(record.get("uuid")), label=str(record.get("name"))
        )
    else:
        name = record.get("name")
        if isinstance(name, str) and name and name != ref.label:
            ref = InstanceRef(backend=ref.backend, id=ref.id, label=name)
    dep_status = str(record.get("status") or "")
    candidates: list[tuple[str, str, dict[str, Any] | None]] = []
    for container in containers or []:
        status = str(container.get("status") or "")
        phase = CONTAINER_STATUS_PHASES.get(status, INST_UNKNOWN)
        candidates.append((phase, f"container={status}", container))
    if _as_int(record.get("running_num")) > 0:
        candidates.append((INST_RUNNING, "running_num>0", None))
    elif _as_int(record.get("starting_num")) > 0:
        candidates.append((INST_LOADING, "starting_num>0", None))

    chosen: dict[str, Any] | None = None
    if candidates:
        phase, detail, chosen = min(
            candidates,
            key=lambda c: (
                _PHASE_PRIORITY[c[0]],
                -(_parse_time(c[2].get("created_at")) or 0.0) if c[2] else 0.0,
            ),
        )
    else:
        phase = DEPLOYMENT_STATUS_PHASES.get(dep_status, INST_UNKNOWN)
        detail = ""
        replicas = max(1, _as_int(record.get("replica_num")))
        if phase == INST_PENDING and _as_int(record.get("finished_num")) >= replicas:
            phase, detail = INST_EXITED, "finished_num>=replica_num"
    raw_status = f"deployment={dep_status}" + (f" {detail}" if detail else "")
    return InstanceState(
        ref=ref,
        phase=phase,
        raw_status=raw_status,
        price_per_hour=_container_price(chosen),
        created_at=_parse_time(record.get("created_at")),
    )


def _container_price(container: dict[str, Any] | None) -> float | None:
    if not container:
        return None
    price = container.get("price")  # 文档：基准价格，单位 元*1000
    if isinstance(price, bool) or not isinstance(price, int | float) or price <= 0:
        return None
    gpus = max(1, _as_int(container.get("gpu_num"))) if PRICE_IS_PER_GPU else 1
    return price * gpus / 1000


def _is_no_stock(error: AutodlApiError) -> bool:
    return _matches(error, _NO_STOCK_MARKERS) and not _matches(error, _BALANCE_MARKERS)


def _matches(error: AutodlApiError, markers: tuple[str, ...]) -> bool:
    text = f"{error.code} {error.msg}".lower()
    return any(marker in text for marker in markers)


def _envelope_summary(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return (
        f" code={payload.get('code')!r} msg={payload.get('msg')!r} "
        f"request_id={payload.get('request_id')!r}"
    )


def _auth_hint(status: int) -> str:
    if status in (401, 403):
        return "。检查 AutoDL 开发者 token 是否正确，账号是否已完成企业认证（弹性部署 API 需要）"
    return ""


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:  # 文档样例都带 +08:00；没带时区按北京时间
        moment = moment.replace(tzinfo=_CHINA_TZ)
    return moment.timestamp()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
