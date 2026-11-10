"""读取 dgpu 配置文件（设计 4.10），得到 contracts.Config。

文件位置的优先级：--config 参数、环境变量 DGPU_CONFIG、默认 ~/.config/dgpu/config.toml。
文件权限必须只有本人可读写（600）。不认识的 key 一律报错，避免拼写错误被静默忽略。
错误信息只写字段名和类型，绝不包含密钥的值。

TOML 布局：

    disk_gb = 60                    # 顶层标量，覆盖 Config 的同名字段（要写在第一个 [表] 之前）
    [stores.<名字>]                 # S3 兼容存储：endpoint、bucket、access_key_id、
                                    # secret_access_key，可选 region、provider
    [backends.vast]                 # api_key、store、base_url，以及搜索过滤条件
                                    # reliability、min_inet_down_mbps、
                                    # max_bandwidth_usd_per_gb、datacenter_only
    [backends.vast.filters]         # 过滤条件也可以写在这个子表里（与上面二选一）
    [backends.autodl]               # token、store、dc_list、cuda_v_from、base_url
    [runner]                        # RunnerSettings 的字段（时间类可以写小数）

环境变量覆盖单个密钥：VAST_API_KEY、AUTODL_TOKEN；R2_ACCESS_KEY_ID、R2_SECRET_ACCESS_KEY
作用于名为 r2 的存储。空字符串视为未设置。
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dgpu.contracts import (
    BACKEND_AUTODL,
    BACKEND_VAST,
    DEFAULT_CONFIG_PATH,
    ENV_CONFIG_PATH,
    AutodlSettings,
    Config,
    ConfigError,
    RunnerSettings,
    StoreProfile,
    VastSettings,
)

ENV_VAST_API_KEY = "VAST_API_KEY"
ENV_AUTODL_TOKEN = "AUTODL_TOKEN"
ENV_R2_ACCESS_KEY_ID = "R2_ACCESS_KEY_ID"
ENV_R2_SECRET_ACCESS_KEY = "R2_SECRET_ACCESS_KEY"
ENV_OVERRIDE_STORE = "r2"  # R2_* 环境变量作用的存储名

VAST_FILTER_KEYS = (
    "reliability",
    "min_inet_down_mbps",
    "max_bandwidth_usd_per_gb",
    "datacenter_only",
)

# 顶层标量：key → (类型, 最小值)
_TOP_SCALARS: dict[str, tuple[type, float]] = {
    "disk_gb": (int, 1),
    "startup_timeout_min": (int, 1),
    "runner_timeout_min": (int, 1),
    "max_offers": (int, 1),
    "create_lookup_s": (int, 0),
    "lost_after_min": (int, 1),
    "gc_stale_min": (int, 1),
    "local_data_confirm_gb": (float, 0),
}
_TOP_TABLES = ("stores", "backends", "runner")

# RunnerSettings 各字段的最小值；未列出的为 0
_RUNNER_MIN = {
    "sync_interval_s": 1,
    "heartbeat_interval_s": 1,
    "stall_timeout_min": 0,  # 0 关闭卡死检测
    "final_sync_timeout_min": 1,
    "term_grace_s": 0,
    "max_log_mb": 1,
    "data_sync_retries": 0,
}

_PLACEHOLDER_RE = re.compile(r"<[^<>\n]+>")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_MISSING: Any = object()


# ---------------------------------------------------------------- 入口


def resolve_config_path(cli_path: Path | None, env: Mapping[str, str] = os.environ) -> Path:
    """配置文件路径：--config 参数，其次环境变量 DGPU_CONFIG，最后默认位置。"""
    if cli_path is not None:
        return Path(cli_path).expanduser()
    from_env = env.get(ENV_CONFIG_PATH, "").strip()
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_CONFIG_PATH.expanduser()


def load_config(path: Path, env: Mapping[str, str] = os.environ) -> Config:
    """读取并校验配置文件，再应用环境变量里的密钥覆盖。"""
    path = Path(path).expanduser()
    try:
        info = path.stat()
    except FileNotFoundError:
        raise ConfigError(
            f"找不到配置文件 {path}。运行 `dgpu init` 在这里生成模板（权限 600），"
            "再填入存储和平台凭据；也可以用 --config 或环境变量 "
            f"{ENV_CONFIG_PATH} 指定其他位置。"
        ) from None
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}：{exc.strerror}。检查路径和权限。") from None
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError(f"配置文件 {path} 不是普通文件。--config 要指向 config.toml 文件。")
    if os.name == "posix" and info.st_mode & 0o077:
        raise ConfigError(
            f"配置文件 {path} 的权限是 {stat.S_IMODE(info.st_mode):03o}，同组或其他用户可以读取，"
            f"里面的密钥可能泄漏。运行 chmod 600 {path} 后重试。"
        )
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"配置文件 {path} 不是合法的 TOML：{exc}。"
            "按提示的行列改正；格式可参考 `dgpu init` 生成的模板。"
        ) from None
    except UnicodeDecodeError:
        raise ConfigError(f"配置文件 {path} 不是 UTF-8 编码的文本。用 UTF-8 重新保存。") from None
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}：{exc.strerror}。检查文件权限。") from None

    try:
        return parse_config(data, env)
    except ConfigError as exc:
        raise ConfigError(f"配置文件 {path} 有误：{exc}。") from None


def parse_config(data: dict[str, Any], env: Mapping[str, str] = os.environ) -> Config:
    """校验已解析的 TOML 文档。"""
    root = _Table(data, "")

    stores = _parse_stores(root.table("stores"), env)
    backends = root.table("backends")
    vast = _parse_vast(backends.table(BACKEND_VAST) if backends else None, env, stores)
    autodl = _parse_autodl(backends.table(BACKEND_AUTODL) if backends else None, env, stores)
    if backends:
        backends.finish(hint=f"[backends] 下只支持 {BACKEND_VAST}、{BACKEND_AUTODL}")
    runner = _parse_runner(root.table("runner"))

    defaults = Config(stores={})
    scalars: dict[str, Any] = {}
    for key, (kind, minimum) in _TOP_SCALARS.items():
        default = getattr(defaults, key)
        if kind is int:
            scalars[key] = root.integer(key, default=default, minimum=int(minimum))
        else:
            scalars[key] = root.number(key, default=default, minimum=minimum)
    root.finish()

    return Config(stores=stores, vast=vast, autodl=autodl, runner=runner, **scalars)


# ---------------------------------------------------------------- 各节


def _parse_stores(table: _Table | None, env: Mapping[str, str]) -> dict[str, StoreProfile]:
    stores: dict[str, StoreProfile] = {}
    if table is None:
        return stores
    defaults = StoreProfile(name="", endpoint="", bucket="", access_key_id="", secret_access_key="")
    for name in list(table.data):
        section = table.table(name)
        assert section is not None
        endpoint = section.string("endpoint")
        if not endpoint.startswith(("https://", "http://")):
            raise ConfigError(
                f"{section.field('endpoint')} 应该以 https:// 开头。"
                "R2 写成 https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
            )
        bucket = section.string("bucket")
        if not _BUCKET_RE.match(bucket):
            raise ConfigError(
                f"{section.field('bucket')} 不是合法的 bucket 名。"
                "bucket 名是 3–63 个小写字母、数字、点或连字符，只写名字，不带 r2:// 或斜杠"
            )
        access_key_id = section.secret("access_key_id")
        secret_access_key = section.secret("secret_access_key")
        if name == ENV_OVERRIDE_STORE:
            access_key_id = _env(env, ENV_R2_ACCESS_KEY_ID) or access_key_id
            secret_access_key = _env(env, ENV_R2_SECRET_ACCESS_KEY) or secret_access_key
        for key, value, env_name in (
            ("access_key_id", access_key_id, ENV_R2_ACCESS_KEY_ID),
            ("secret_access_key", secret_access_key, ENV_R2_SECRET_ACCESS_KEY),
        ):
            via_env = f"（也可以用环境变量 {env_name} 提供）" if name == ENV_OVERRIDE_STORE else ""
            if not value:
                raise ConfigError(f"{section.field(key)} 没有填写。填入存储的访问密钥{via_env}")
            _check_placeholder(section.field(key), value)
        region = section.string("region", default=defaults.region)
        provider = section.string("provider", default=defaults.provider)
        section.finish()
        stores[name] = StoreProfile(
            name=name,
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            provider=provider,
        )
    return stores


def _parse_vast(
    table: _Table | None, env: Mapping[str, str], stores: Mapping[str, StoreProfile]
) -> VastSettings:
    defaults = VastSettings()
    api_key = _env(env, ENV_VAST_API_KEY)
    if table is None:
        return dataclasses.replace(defaults, api_key=api_key or defaults.api_key)
    api_key = api_key or table.secret("api_key")
    table.mark_used("api_key")
    if api_key:
        _check_placeholder(table.field("api_key"), api_key)
    store = _store_ref(table, stores, defaults.store)
    base_url = table.string("base_url", default=defaults.base_url)

    filters = table.table("filters")
    values: dict[str, Any] = {}
    for key in VAST_FILTER_KEYS:
        source = table
        if filters is not None and filters.has(key):
            if table.has(key):
                raise ConfigError(
                    f"{table.field(key)} 和 {filters.field(key)} 重复。只保留其中一个"
                )
            source = filters
        default = getattr(defaults, key)
        if key == "datacenter_only":
            values[key] = source.boolean(key, default=default)
        elif key == "reliability":
            values[key] = source.number(key, default=default, minimum=0, maximum=1)
        else:
            values[key] = source.number(key, default=default, minimum=0)
    if filters is not None:
        filters.finish()
    table.finish()
    return VastSettings(api_key=api_key, store=store, base_url=base_url, **values)


def _parse_autodl(
    table: _Table | None, env: Mapping[str, str], stores: Mapping[str, StoreProfile]
) -> AutodlSettings:
    defaults = AutodlSettings()
    token = _env(env, ENV_AUTODL_TOKEN)
    if table is None:
        return dataclasses.replace(defaults, token=token or defaults.token)
    token = token or table.secret("token")
    table.mark_used("token")
    if token:
        _check_placeholder(table.field("token"), token)
    store = _store_ref(table, stores, defaults.store)
    dc_list = table.strings("dc_list", default=defaults.dc_list)
    cuda_v_from = table.integer("cuda_v_from", default=defaults.cuda_v_from, minimum=0)
    base_url = table.string("base_url", default=defaults.base_url)
    table.finish()
    return AutodlSettings(
        token=token, store=store, dc_list=dc_list, cuda_v_from=cuda_v_from, base_url=base_url
    )


def _parse_runner(table: _Table | None) -> RunnerSettings:
    defaults = RunnerSettings()
    if table is None:
        return defaults
    values: dict[str, Any] = {}
    for f in dataclasses.fields(RunnerSettings):
        default = getattr(defaults, f.name)
        minimum = _RUNNER_MIN.get(f.name, 0)
        if f.name in RunnerSettings.TIME_FIELDS:  # 时间可以写小数，如 term_grace_s = 7.5
            value = table.number(f.name, default=default, minimum=minimum)
            values[f.name] = int(value) if float(value).is_integer() else value
        else:
            values[f.name] = table.integer(f.name, default=default, minimum=minimum)
    table.finish()
    return RunnerSettings(**values)


def _store_ref(table: _Table, stores: Mapping[str, StoreProfile], default: str) -> str:
    explicit = table.has("store")
    name = table.string("store", default=default)
    if explicit and name not in stores:
        known = "、".join(stores) or "（没有）"
        raise ConfigError(
            f"{table.field('store')} 指向不存在的存储 {name!r}。已配置的存储：{known}；"
            f"加上 [stores.{name}] 一节，或者改成已有的名字"
        )
    return name


def _env(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _check_placeholder(field: str, value: str) -> None:
    if _PLACEHOLDER_RE.search(value):
        raise ConfigError(
            f"{field} 还是模板里的 <...> 占位符。换成真实值；不用的后端可以把整节删掉"
        )


# ---------------------------------------------------------------- 带类型检查的读取


class _Table:
    """一个 TOML 表。读过的 key 记下来，finish() 时剩下的就是不认识的 key。"""

    def __init__(self, data: Any, where: str):
        if not isinstance(data, dict):
            raise ConfigError(f"{where} 应该是一个表（[{where}]），现在是{_kind(data)}")
        self.data: dict[str, Any] = data
        self.where = where
        self.used: set[str] = set()

    def field(self, key: str) -> str:
        return f"{self.where}.{key}" if self.where else key

    def has(self, key: str) -> bool:
        return key in self.data

    def mark_used(self, key: str) -> None:
        self.used.add(key)

    def _get(self, key: str) -> Any:
        self.used.add(key)
        return self.data.get(key, _MISSING)

    def table(self, key: str) -> _Table | None:
        value = self._get(key)
        if value is _MISSING:
            return None
        if not isinstance(value, dict):
            raise ConfigError(
                f"{self.field(key)} 应该是一个表（[{self.field(key)}]），现在是{_kind(value)}"
            )
        return _Table(value, self.field(key))

    def string(self, key: str, *, default: Any = _MISSING) -> str:
        value = self._get(key)
        if value is _MISSING:
            if default is _MISSING:
                raise ConfigError(f"{self.field(key)} 没有填写。这是必填项")
            return default
        if not isinstance(value, str):
            raise ConfigError(
                f"{self.field(key)} 应该是字符串（用引号括起来），现在是{_kind(value)}"
            )
        value = value.strip()
        if not value:
            raise ConfigError(f"{self.field(key)} 不能为空。填上内容，或者删掉这一行使用默认值")
        _check_placeholder(self.field(key), value)
        return value

    def secret(self, key: str) -> str:
        """读取密钥；缺省为空字符串，由调用方在合并环境变量后检查。不做占位符检查。"""
        value = self._get(key)
        if value is _MISSING:
            return ""
        if not isinstance(value, str):
            raise ConfigError(
                f"{self.field(key)} 应该是字符串（用引号括起来），现在是{_kind(value)}"
            )
        return value.strip()

    def strings(self, key: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
        value = self._get(key)
        if value is _MISSING:
            return default
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ConfigError(f'{self.field(key)} 应该是字符串列表，如 ["westDC2"]；空列表表示不限')
        for item in value:
            _check_placeholder(self.field(key), item)
        return tuple(item.strip() for item in value)

    def integer(self, key: str, *, default: Any, minimum: int) -> Any:
        value = self._get(key)
        if value is _MISSING:
            return default
        if type(value) is not int:
            raise ConfigError(f"{self.field(key)} 应该是整数，现在是{_kind(value)}")
        if value < minimum:
            raise ConfigError(f"{self.field(key)} 不能小于 {minimum}，现在是 {value}")
        return value

    def number(
        self, key: str, *, default: Any, minimum: float, maximum: float | None = None
    ) -> Any:
        value = self._get(key)
        if value is _MISSING:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self.field(key)} 应该是数字，现在是{_kind(value)}")
        if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
            if maximum is None:  # 与 integer() 的说法一致
                raise ConfigError(f"{self.field(key)} 不能小于 {minimum}，现在是 {value}")
            raise ConfigError(
                f"{self.field(key)} 应该在 {minimum} 到 {maximum} 之间，现在是 {value}"
            )
        return float(value)

    def boolean(self, key: str, *, default: bool) -> bool:
        value = self._get(key)
        if value is _MISSING:
            return default
        if not isinstance(value, bool):
            raise ConfigError(
                f"{self.field(key)} 应该是 true 或 false（不加引号），现在是{_kind(value)}"
            )
        return value

    def finish(self, *, hint: str = "") -> None:
        unknown = sorted(set(self.data) - self.used)
        if not unknown:
            return
        key = unknown[0]
        allowed = "、".join(sorted(self.used)) or "（无）"
        message = f"{self.field(key)} 是不认识的配置项。检查拼写"
        if hint:
            message += f"；{hint}"
        elif self.where and key in _TOP_SCALARS:
            message += f"；{key} 是顶层配置项，要写在文件开头、第一个 [表] 之前"
        else:
            message += f"；这里允许的配置项：{allowed}"
        raise ConfigError(message)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "布尔值"
    if isinstance(value, int):
        return "整数"
    if isinstance(value, float):
        return "小数"
    if isinstance(value, str):
        return "字符串"
    if isinstance(value, list):
        return "列表"
    if isinstance(value, dict):
        return "表"
    return type(value).__name__


# ---------------------------------------------------------------- 模板


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return repr(value)


def config_template() -> str:
    """dgpu init 写出的配置模板。默认值取自 contracts 里的数据类。"""
    config = Config(stores={})
    vast = VastSettings()
    autodl = AutodlSettings()
    runner = RunnerSettings()
    top = "\n".join(
        f"# {key} = 60   # 默认不设，按 max(40, 2×数据量+20) 估算"
        if key == "disk_gb"
        else f"# {key} = {_toml(getattr(config, key))}"
        for key in _TOP_SCALARS
    )
    runner_lines = "\n".join(
        f"# {f.name} = {_toml(getattr(runner, f.name))}" for f in dataclasses.fields(runner)
    )
    vast_filters = "\n".join(f"# {key} = {_toml(getattr(vast, key))}" for key in VAST_FILTER_KEYS)
    return f"""\
# dgpu 配置文件。
# - 权限必须是 600（只有本人可读写）：chmod 600 <本文件>。不要提交到 git，不要发给别人。
# - 把所有 <...> 占位符换成真实值；不用的后端可以把整节删掉。
# - 单个密钥也可以用环境变量覆盖：{ENV_VAST_API_KEY}、{ENV_AUTODL_TOKEN}、
#   {ENV_R2_ACCESS_KEY_ID}、{ENV_R2_SECRET_ACCESS_KEY}（作用于 [stores.{ENV_OVERRIDE_STORE}]）。
# - 注释掉的行是默认值，需要时去掉开头的 # 再修改。

# ---- 顶层配置项（必须写在第一个 [表] 之前）
{top}

# ---- 存储：S3 兼容端点，默认 Cloudflare R2
# R2 的 API token 只授权 dgpu 用的那个 bucket（Object Read & Write）。
[stores.{ENV_OVERRIDE_STORE}]
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
bucket = "<BUCKET>"
access_key_id = "<R2_ACCESS_KEY_ID>"
secret_access_key = "<R2_SECRET_ACCESS_KEY>"
# region = "auto"
# provider = "Cloudflare"   # rclone 的 s3 provider 名，如 Alibaba、TencentCOS

# ---- Vast.ai（主 key 只留在本机，永不进入实例）
[backends.{BACKEND_VAST}]
api_key = "<VAST_API_KEY>"
store = "{vast.store}"
# 搜索机器的过滤条件：
{vast_filters}

# ---- AutoDL 弹性部署（需要企业认证）
[backends.{BACKEND_AUTODL}]
token = "<AUTODL_TOKEN>"
store = "{autodl.store}"
# dc_list = ["westDC2"]   # 地区，默认不限
# cuda_v_from = {autodl.cuda_v_from}       # 118 即 CUDA 11.8

# ---- 下发给实例上 runner 的参数
# [runner]
{runner_lines}
"""


def write_config_template(path: Path) -> None:
    """写出带注释和占位符的配置模板，权限 600。文件已存在时拒绝覆盖。"""
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise ConfigError(f"配置文件 {path} 已经存在，不会覆盖。要重新生成，先把它改名或删除。")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ConfigError(
            f"配置文件 {path} 已经存在，不会覆盖。要重新生成，先把它改名或删除。"
        ) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o600)  # 不受 umask 影响
            handle.write(config_template())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
