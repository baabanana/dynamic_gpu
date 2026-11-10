"""模块之间的冻结契约：数据类、协议、异常，以及下发给实例的两份文档的格式。

各工作包只依赖本文件、layout.py、states.py 和 testing.py，不互相 import 实现。
需要改动契约时不要自己改，向主会话提出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from dgpu import layout, states

BACKEND_VAST = "vast"
BACKEND_AUTODL = "autodl"
BACKENDS = (BACKEND_VAST, BACKEND_AUTODL)

CURRENCY_BY_BACKEND = {BACKEND_VAST: "USD", BACKEND_AUTODL: "CNY"}

DEFAULT_CONFIG_PATH = Path("~/.config/dgpu/config.toml")
ENV_CONFIG_PATH = "DGPU_CONFIG"

# ---------------------------------------------------------------- 异常


class DgpuError(Exception):
    """所有可预期错误的基类。CLI 捕获后打印 message，不打印 traceback。"""


class SpecError(DgpuError):
    """job.yaml 不合法。message 要说明哪个字段、为什么、怎么改。"""


class ConfigError(DgpuError):
    """配置文件缺失、权限不对或内容不合法。"""


class StoreError(DgpuError):
    """存储访问失败（重试耗尽后）。"""


class PreconditionFailed(StoreError):
    """条件写不满足：对象已被别人修改或已存在。"""


class BackendError(DgpuError):
    """平台 API 调用失败。retryable 表示换一台机器或稍后重试可能成功。"""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class OfferUnavailable(BackendError):
    """选中的机器已被租走、无库存或创建被拒绝。换下一个 offer。"""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


# ---------------------------------------------------------------- 任务规格（job.yaml）


@dataclass(frozen=True)
class GpuSpec:
    names: tuple[str, ...]  # 如 ("RTX 3090", "RTX 4090")，任一即可
    count: int = 1


@dataclass(frozen=True)
class Limits:
    max_hours: float
    max_usd_per_hour: float | None = None  # Vast
    max_cny_per_hour: float | None = None  # AutoDL

    def max_price(self, backend: str) -> float:
        value = self.max_usd_per_hour if backend == BACKEND_VAST else self.max_cny_per_hour
        if value is None:
            raise SpecError(f"limits 缺少 {backend} 后端的价格上限")
        return value


@dataclass(frozen=True)
class DataRef:
    """job.yaml 里 data 的一项。远程：bucket + prefix；本地：local_path。"""

    name: str  # 实例上的目录名：$DGPU_DATA/<name>
    bucket: str | None = None
    prefix: str | None = None
    local_path: Path | None = None

    @property
    def is_local(self) -> bool:
        return self.local_path is not None


@dataclass(frozen=True)
class DataSource:
    """已经在存储里的数据，实例可以直接拉取。"""

    name: str
    bucket: str
    prefix: str  # 目录前缀，以 / 结尾

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "bucket": self.bucket, "prefix": self.prefix}


@dataclass(frozen=True)
class JobSpec:
    version: int
    name: str
    backend: str
    images: dict[str, str]  # 后端名 → 镜像（Vast：带固定 tag 的 Docker 镜像；AutoDL：镜像 UUID）
    code: Path  # 本地代码目录（绝对路径）
    run: str
    limits: Limits
    gpu: GpuSpec
    setup: str | None = None
    data: tuple[DataRef, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def image_for(self, backend: str) -> str:
        try:
            return self.images[backend]
        except KeyError:
            raise SpecError(f"image 缺少 {backend} 后端的镜像") from None


# ---------------------------------------------------------------- 配置（config.toml）


@dataclass(frozen=True)
class StoreProfile:
    """一个 S3 兼容存储。"""

    name: str
    endpoint: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str = "auto"
    provider: str = "Cloudflare"  # rclone 的 s3 provider 名

    def __repr__(self) -> str:  # 不把密钥打进日志
        return (
            f"StoreProfile(name={self.name!r}, endpoint={self.endpoint!r}, "
            f"bucket={self.bucket!r})"
        )


@dataclass(frozen=True)
class VastSettings:
    api_key: str = field(repr=False, default="")
    store: str = "r2"
    reliability: float = 0.98
    min_inet_down_mbps: float = 500.0
    max_bandwidth_usd_per_gb: float = 0.01
    datacenter_only: bool = False
    base_url: str = "https://console.vast.ai"


@dataclass(frozen=True)
class AutodlSettings:
    token: str = field(repr=False, default="")
    store: str = "r2"
    dc_list: tuple[str, ...] = ()  # 空表示不限地区
    cuda_v_from: int = 118  # 118 即 CUDA 11.8
    base_url: str = "https://api.autodl.com"


@dataclass(frozen=True)
class RunnerSettings:
    """下发给 runner 的参数，写进 spec.json 的 runner 段。

    时间类参数（TIME_FIELDS）可以是小数，runner 一律按浮点数读取；另外两个是整数。
    """

    TIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "sync_interval_s",
        "heartbeat_interval_s",
        "stall_timeout_min",
        "final_sync_timeout_min",
        "term_grace_s",
    )

    sync_interval_s: float = 120
    heartbeat_interval_s: float = 60
    stall_timeout_min: float = 60  # 0 关闭卡死检测
    final_sync_timeout_min: float = 20
    term_grace_s: float = 60  # SIGTERM 后多久 SIGKILL
    max_log_mb: int = 200
    data_sync_retries: int = 3

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Config:
    stores: dict[str, StoreProfile]
    vast: VastSettings = field(default_factory=VastSettings)
    autodl: AutodlSettings = field(default_factory=AutodlSettings)
    runner: RunnerSettings = field(default_factory=RunnerSettings)
    disk_gb: int | None = None  # None：按 max(40, 2×数据量+20) 估算
    startup_timeout_min: int = 15
    runner_timeout_min: int = 5
    max_offers: int = 5
    lost_after_min: int = 10
    gc_stale_min: int = 30
    local_data_confirm_gb: float = 5.0

    def store_for(self, backend: str) -> StoreProfile:
        name = self.vast.store if backend == BACKEND_VAST else self.autodl.store
        try:
            return self.stores[name]
        except KeyError:
            raise ConfigError(f"配置里没有名为 {name!r} 的存储（[stores.{name}]）") from None


# ---------------------------------------------------------------- 后端


@dataclass(frozen=True)
class GpuRequest:
    names: tuple[str, ...]
    count: int
    max_price_per_hour: float  # 后端自己的货币
    max_hours: float
    disk_gb: int
    image: str


@dataclass(frozen=True)
class Offer:
    backend: str
    id: str  # Vast：offer id；AutoDL：地区码
    gpu_name: str
    gpu_count: int
    price_per_hour: float
    currency: str
    est_total: float  # 按 max_hours 估算的总价（算力 + 磁盘 + 流量），用于排序和展示
    region: str = ""
    details: dict[str, Any] = field(default_factory=dict)  # 展示用：可靠性、带宽、库存等


@dataclass(frozen=True)
class LaunchSpec:
    """创建实例所需的全部信息。不含任何凭据。"""

    job_id: str
    attempt: int
    image: str
    bootstrap_cmd: str  # 实例启动后执行的 shell 命令，由 Bundler.bootstrap_command 生成
    disk_gb: int
    max_price_per_hour: float

    @property
    def label(self) -> str:
        return layout.label(self.job_id, self.attempt)


@dataclass(frozen=True)
class InstanceRef:
    backend: str
    id: str
    label: str


# InstanceState.phase 的取值，各后端把平台状态映射到这几种
INST_PENDING = "pending"  # 已受理，等待调度
INST_LOADING = "loading"  # 正在拉镜像或启动容器
INST_RUNNING = "running"
INST_EXITED = "exited"  # 容器已退出但实例或部署记录还在（可能仍计费）
INST_GONE = "gone"  # 已销毁或不存在
INST_UNKNOWN = "unknown"  # 平台查不到心跳或状态无法识别

INSTANCE_PHASES = (
    INST_PENDING,
    INST_LOADING,
    INST_RUNNING,
    INST_EXITED,
    INST_GONE,
    INST_UNKNOWN,
)


@dataclass(frozen=True)
class InstanceState:
    ref: InstanceRef
    phase: str
    raw_status: str = ""  # 平台原始状态字符串，排查用
    price_per_hour: float | None = None
    created_at: float | None = None  # epoch 秒


@dataclass(frozen=True)
class Balance:
    amount: float
    currency: str


@runtime_checkable
class Backend(Protocol):
    """一个 GPU 平台。实现不得读取或写入存储，只和平台 API 打交道。"""

    name: str

    def search(self, req: GpuRequest) -> list[Offer]:
        """返回满足条件的 offer，按 est_total 升序。没有则返回空列表。"""

    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef:
        """创建实例并打上 launch.label。

        机器不可用抛 OfferUnavailable。请求超时等结果未知的情况，实现必须先按
        label 查询：已创建就返回它，确认没创建才抛 BackendError(retryable=True)。
        """

    def get(self, ref: InstanceRef) -> InstanceState:
        """查询实例。不存在时返回 phase=INST_GONE，不抛异常。"""

    def destroy(self, ref: InstanceRef) -> None:
        """销毁实例并停止计费。幂等：已不存在不报错。返回前须确认平台已受理。"""

    def list(self, label_prefix: str = layout.LABEL_PREFIX) -> list[InstanceState]:
        """列出标签以 label_prefix 开头的实例。绝不返回其他实例。"""

    def balance(self) -> Balance | None:
        """账户余额；平台不提供时返回 None。"""


# ---------------------------------------------------------------- 存储

# 单个对象的上传上限：S3 / R2 单段 PutObject 最多 5 GiB。R2 的分段上传 ETag 不可靠，
# dgpu 只做单段上传，更大的文件要用户自己用 rclone 传到 bucket，再用 r2:// 引用。
SINGLE_UPLOAD_LIMIT = 5 * 1024**3


@runtime_checkable
class Store(Protocol):
    """绑定到一个 bucket 的 S3 兼容存储。key 的约定见 layout.py。

    put_bytes / put_file 的单个对象不超过 SINGLE_UPLOAD_LIMIT，超过时抛 StoreError。
    """

    profile: StoreProfile

    def get_json(self, key: str) -> tuple[dict[str, Any], str] | None:
        """返回 (内容, ETag)；不存在返回 None。"""

    def put_json(
        self,
        key: str,
        obj: dict[str, Any],
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        """写 JSON，返回新 ETag。

        if_match：仅当当前 ETag 等于它时写入；if_none_match：仅当对象不存在时写入。
        条件不满足抛 PreconditionFailed。
        """

    def get_bytes(self, key: str, *, start: int = 0) -> bytes | None:
        """从 start 字节读到末尾；不存在返回 None；start 超出末尾返回 b""。"""

    def put_bytes(self, key: str, data: bytes) -> None: ...

    def put_file(self, key: str, path: Path) -> None: ...

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str) -> list[str]:
        """返回前缀下所有 key（递归），按字典序。"""

    def list_dirs(self, prefix: str) -> list[str]:
        """返回前缀下一层的“子目录”名（不带斜杠），按字典序。"""

    def download_prefix(self, prefix: str, dest: Path) -> int:
        """把前缀下的对象下载到 dest（保留相对路径），返回文件数。"""

    def presign_get(self, key: str, ttl_s: int) -> str:
        """返回可直接 HTTP GET 的预签名 URL。"""

    def delete(self, key: str) -> None:
        """删除对象。不存在不报错。"""


# ---------------------------------------------------------------- bundle


@dataclass(frozen=True)
class BundleRef:
    key: str
    sha256: str


@runtime_checkable
class Bundler(Protocol):
    """runner bundle 的构建与启动命令生成（工作包 E 实现）。"""

    def ensure_uploaded(self, store: Store) -> BundleRef:
        """构建 bundle（runner + rclone），不在存储里就上传。按内容哈希去重。"""

    def bootstrap_command(self, *, bundle_url: str, bundle_sha256: str, launch_url: str) -> str:
        """生成实例启动命令：下载并校验 bundle、下载启动文档、拉起 watchdog 和 runner。

        只依赖镜像里的 bash 和 python3（≥ 3.8）。必须幂等（Vast 每次容器启动都会执行）。
        长度不超过 4000 字符（Vast onstart 上限 4048）。
        """


# ---------------------------------------------------------------- 下发给实例的两份文档
# runner 只认这里定义的字段。读取方遇到未知 version 必须拒绝。


def build_remote_spec(
    spec: JobSpec,
    *,
    job_id: str,
    backend: str,
    data: list[DataSource],
    runner: RunnerSettings,
) -> dict[str, Any]:
    """spec.json：任务内容，不含凭据。存放在 layout.spec_key(job_id)。"""
    return {
        "version": states.SCHEMA_VERSION,
        "job_id": job_id,
        "name": spec.name,
        "backend": backend,
        "image": spec.image_for(backend),
        "setup": spec.setup,
        "run": spec.run,
        "env": dict(spec.env),
        "data": [source.to_dict() for source in data],
        "max_hours": spec.limits.max_hours,
        "runner": runner.to_dict(),
    }


def build_launch_doc(*, job_id: str, attempt: int, backend: str, store: StoreProfile) -> dict:
    """secrets-<n>.json：本次尝试的启动文档，含存储凭据。

    存放在 layout.secrets_key(job_id, attempt)，只通过短期预签名 URL 下发，
    CLI 在第一次心跳后删除。runner 用它配置 rclone，然后读取 spec.json 和 code.tar.gz。
    """
    return {
        "version": states.SCHEMA_VERSION,
        "job_id": job_id,
        "attempt": attempt,
        "backend": backend,
        "store": {
            "provider": store.provider,
            "endpoint": store.endpoint,
            "region": store.region,
            "bucket": store.bucket,
            "access_key_id": store.access_key_id,
            "secret_access_key": store.secret_access_key,
        },
    }
