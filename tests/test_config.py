"""配置文件：路径优先级、权限检查、环境变量覆盖、未知 key、错误信息不含密钥、模板。

所有测试都用 tmp_path 里的文件和显式传入的 env，不读真实的 ~/.config/dgpu。
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

import pytest

from dgpu.config import (
    load_config,
    resolve_config_path,
    write_config_template,
)
from dgpu.contracts import (
    DEFAULT_CONFIG_PATH,
    ENV_CONFIG_PATH,
    AutodlSettings,
    Config,
    ConfigError,
    RunnerSettings,
    StoreProfile,
    VastSettings,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="文件权限检查只在 POSIX 上进行")

SECRET = "S3CR3T-do-not-print"

FULL = f"""
disk_gb = 80
max_offers = 3
local_data_confirm_gb = 2.5

[stores.r2]
endpoint = "https://acc.r2.cloudflarestorage.com"
bucket = "my-bucket"
access_key_id = "r2-id-from-file"
secret_access_key = "{SECRET}"

[stores.oss]
endpoint = "https://oss-cn-beijing.aliyuncs.com"
bucket = "cn-bucket-1250000000"
access_key_id = "oss-id"
secret_access_key = "oss-secret"
region = "cn-beijing"
provider = "Alibaba"

[backends.vast]
api_key = "vast-key-from-file"
store = "r2"
reliability = 0.99
datacenter_only = true

[backends.vast.filters]
min_inet_down_mbps = 800

[backends.autodl]
token = "autodl-token-from-file"
store = "oss"
dc_list = ["westDC2", "beijingDC1"]
cuda_v_from = 121

[runner]
sync_interval_s = 60
stall_timeout_min = 0
"""

MINIMAL = """
[stores.r2]
endpoint = "https://acc.r2.cloudflarestorage.com"
bucket = "my-bucket"
access_key_id = "id"
secret_access_key = "secret"
"""


def _write(tmp_path: Path, text: str, mode: int = 0o600, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)
    return path


def _error(path: Path, *needles: str, env: dict[str, str] | None = None) -> str:
    with pytest.raises(ConfigError) as info:
        load_config(path, env=env or {})
    message = str(info.value)
    for needle in needles:
        assert needle in message, f"{needle!r} not in {message!r}"
    assert re.search(r"[一-鿿]", message), message
    return message


# ---------------------------------------------------------------- 路径


def test_explicit_path_wins_over_env_and_default(tmp_path):
    env = {ENV_CONFIG_PATH: str(tmp_path / "from-env.toml")}
    assert resolve_config_path(tmp_path / "cli.toml", env) == tmp_path / "cli.toml"


def test_env_path_wins_over_default(tmp_path):
    env = {ENV_CONFIG_PATH: str(tmp_path / "from-env.toml")}
    assert resolve_config_path(None, env) == tmp_path / "from-env.toml"


def test_default_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / ".config" / "dgpu" / "config.toml"
    assert resolve_config_path(None, {}) == expected
    assert resolve_config_path(None, {ENV_CONFIG_PATH: "  "}) == expected
    assert DEFAULT_CONFIG_PATH.expanduser() == expected


def test_tilde_in_explicit_and_env_paths_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_config_path(Path("~/a.toml"), {}) == tmp_path / "a.toml"
    assert resolve_config_path(None, {ENV_CONFIG_PATH: "~/b.toml"}) == tmp_path / "b.toml"


# ---------------------------------------------------------------- 文件本身的问题


def test_missing_file_explains_how_to_create_it(tmp_path):
    path = tmp_path / "nope" / "config.toml"
    _error(path, str(path), "dgpu init", ENV_CONFIG_PATH)


def test_directory_is_not_a_config_file(tmp_path):
    _error(tmp_path, "不是普通文件")


@posix_only
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o666, 0o610, 0o601])
def test_group_or_other_access_is_refused(tmp_path, mode):
    path = _write(tmp_path, FULL, mode=mode)
    message = _error(path, "chmod 600", str(path), f"{mode & 0o777:03o}")
    assert SECRET not in message


@posix_only
@pytest.mark.parametrize("mode", [0o600, 0o400, 0o700])
def test_owner_only_modes_are_accepted(tmp_path, mode):
    path = _write(tmp_path, MINIMAL, mode=mode)
    assert load_config(path, env={}).stores["r2"].bucket == "my-bucket"


def test_invalid_toml(tmp_path):
    _error(_write(tmp_path, "[stores.r2\nendpoint = 1\n"), "TOML", "line")


def test_non_utf8_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_bytes("disk_gb = 1 # 中文\n".encode("gbk"))
    os.chmod(path, 0o600)
    _error(path, "UTF-8")


# ---------------------------------------------------------------- 内容


def test_full_config_is_parsed(tmp_path):
    config = load_config(_write(tmp_path, FULL), env={})

    assert config.stores["r2"] == StoreProfile(
        name="r2",
        endpoint="https://acc.r2.cloudflarestorage.com",
        bucket="my-bucket",
        access_key_id="r2-id-from-file",
        secret_access_key=SECRET,
    )
    oss = config.stores["oss"]
    assert (oss.region, oss.provider, oss.bucket) == (
        "cn-beijing",
        "Alibaba",
        "cn-bucket-1250000000",
    )

    assert config.vast == VastSettings(
        api_key="vast-key-from-file",
        store="r2",
        reliability=0.99,
        min_inet_down_mbps=800.0,
        datacenter_only=True,
    )
    assert config.autodl == AutodlSettings(
        token="autodl-token-from-file",
        store="oss",
        dc_list=("westDC2", "beijingDC1"),
        cuda_v_from=121,
    )
    assert config.runner == RunnerSettings(sync_interval_s=60, stall_timeout_min=0)
    assert (config.disk_gb, config.max_offers, config.local_data_confirm_gb) == (80, 3, 2.5)
    assert config.store_for("autodl") is oss
    assert SECRET not in repr(config)


def test_minimal_config_uses_defaults(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL), env={})
    defaults = Config(stores={})
    assert config.vast == VastSettings()
    assert config.autodl == AutodlSettings()
    assert config.runner == RunnerSettings()
    for f in dataclasses.fields(Config):
        if f.name not in ("stores", "vast", "autodl", "runner"):
            assert getattr(config, f.name) == getattr(defaults, f.name)
    assert config.stores["r2"].region == "auto"
    assert config.stores["r2"].provider == "Cloudflare"


def test_empty_file_gives_defaults_without_stores(tmp_path):
    config = load_config(_write(tmp_path, ""), env={})
    assert config.stores == {}
    with pytest.raises(ConfigError):
        config.store_for("vast")


def test_every_top_level_config_field_can_be_set(tmp_path):
    scalars = [
        f.name
        for f in dataclasses.fields(Config)
        if f.name not in ("stores", "vast", "autodl", "runner")
    ]
    text = "".join(f"{name} = 7\n" for name in scalars)
    config = load_config(_write(tmp_path, text), env={})
    for name in scalars:
        assert getattr(config, name) == 7, name


def test_every_runner_setting_can_be_set(tmp_path):
    names = [f.name for f in dataclasses.fields(RunnerSettings)]
    text = "[runner]\n" + "".join(f"{name} = 7\n" for name in names)
    config = load_config(_write(tmp_path, text), env={})
    assert config.runner == RunnerSettings(**{name: 7 for name in names})
    assert config.runner.to_dict() == {name: 7 for name in names}


def test_runner_time_settings_accept_fractions(tmp_path):
    text = "[runner]\nsync_interval_s = 1.5\nterm_grace_s = 7.5\nfinal_sync_timeout_min = 2.5\n"
    config = load_config(_write(tmp_path, text), env={})
    assert config.runner.sync_interval_s == 1.5
    assert config.runner.term_grace_s == 7.5
    assert config.runner.final_sync_timeout_min == 2.5
    assert config.runner.to_dict()["term_grace_s"] == 7.5
    assert set(RunnerSettings.TIME_FIELDS) <= set(config.runner.to_dict())
    assert "max_log_mb" not in RunnerSettings.TIME_FIELDS


def test_vast_filters_may_be_flat_or_in_subtable(tmp_path):
    flat = "[backends.vast]\nreliability = 0.9\nmax_bandwidth_usd_per_gb = 0.02\n"
    nested = "[backends.vast.filters]\nreliability = 0.9\nmax_bandwidth_usd_per_gb = 0.02\n"
    for text in (flat, nested):
        vast = load_config(_write(tmp_path, text), env={}).vast
        assert (vast.reliability, vast.max_bandwidth_usd_per_gb) == (0.9, 0.02)


# ---------------------------------------------------------------- 环境变量覆盖


def test_env_overrides_secrets(tmp_path):
    env = {
        "VAST_API_KEY": "vast-from-env",
        "AUTODL_TOKEN": "autodl-from-env",
        "R2_ACCESS_KEY_ID": "r2-id-from-env",
        "R2_SECRET_ACCESS_KEY": "r2-secret-from-env",
    }
    config = load_config(_write(tmp_path, FULL), env=env)
    assert config.vast.api_key == "vast-from-env"
    assert config.autodl.token == "autodl-from-env"
    assert config.stores["r2"].access_key_id == "r2-id-from-env"
    assert config.stores["r2"].secret_access_key == "r2-secret-from-env"
    # R2_* 只作用于名为 r2 的存储
    assert config.stores["oss"].access_key_id == "oss-id"
    assert config.stores["oss"].secret_access_key == "oss-secret"


def test_empty_env_values_do_not_override(tmp_path):
    env = {"VAST_API_KEY": "", "R2_SECRET_ACCESS_KEY": "  "}
    config = load_config(_write(tmp_path, FULL), env=env)
    assert config.vast.api_key == "vast-key-from-file"
    assert config.stores["r2"].secret_access_key == SECRET


def test_secrets_may_come_only_from_env(tmp_path):
    text = '[stores.r2]\nendpoint = "https://a.r2.cloudflarestorage.com"\nbucket = "bkt"\n'
    env = {
        "R2_ACCESS_KEY_ID": "id",
        "R2_SECRET_ACCESS_KEY": "secret",
        "VAST_API_KEY": "vk",
        "AUTODL_TOKEN": "at",
    }
    config = load_config(_write(tmp_path, text), env=env)
    assert config.stores["r2"].secret_access_key == "secret"
    assert config.vast.api_key == "vk"
    assert config.autodl.token == "at"


def test_env_secret_replaces_template_placeholder(tmp_path):
    text = MINIMAL.replace('"secret"', '"<R2_SECRET_ACCESS_KEY>"')
    _error(_write(tmp_path, text), "stores.r2.secret_access_key", "占位符")
    config = load_config(_write(tmp_path, text), env={"R2_SECRET_ACCESS_KEY": "real"})
    assert config.stores["r2"].secret_access_key == "real"


def test_missing_store_secret_mentions_env_var_for_r2_only(tmp_path):
    text = '[stores.r2]\nendpoint = "https://a.example"\nbucket = "bkt"\n'
    _error(_write(tmp_path, text), "stores.r2.access_key_id", "R2_ACCESS_KEY_ID")
    text = '[stores.oss]\nendpoint = "https://a.example"\nbucket = "bkt"\naccess_key_id = "x"\n'
    message = _error(_write(tmp_path, text), "stores.oss.secret_access_key", "没有填写")
    assert "R2_" not in message


# ---------------------------------------------------------------- 校验错误

INVALID = [
    # 未知 key：拼写错误不能被静默忽略
    ("dsik_gb = 1\n", ["dsik_gb", "不认识"]),
    (MINIMAL + 'acess_key_id = "x"\n', ["stores.r2.acess_key_id", "不认识"]),
    ("[backends.vast]\nrelability = 0.9\n", ["backends.vast.relability", "不认识"]),
    ("[backends.vast.filters]\nfoo = 1\n", ["backends.vast.filters.foo", "不认识"]),
    ('[backends.autodl]\ndclist = ["a"]\n', ["backends.autodl.dclist", "不认识"]),
    ('[backends.runpod]\napi_key = "x"\n', ["backends.runpod", "vast"]),
    ("[runner]\nsync_interval = 10\n", ["runner.sync_interval", "不认识"]),
    ("[runner]\ndisk_gb = 10\n", ["runner.disk_gb", "顶层"]),
    ("[tools]\nx = 1\n", ["tools", "不认识"]),
    # 结构
    ('stores = "r2"\n', ["stores", "表"]),
    ('[stores]\nr2 = "x"\n', ["stores.r2", "表"]),
    ("backends = 1\n", ["backends", "表"]),
    ("runner = []\n", ["runner", "表"]),
    # 存储
    (
        '[stores.r2]\nbucket = "bkt"\naccess_key_id = "a"\nsecret_access_key = "b"\n',
        ["stores.r2.endpoint", "必填"],
    ),
    (
        '[stores.r2]\nendpoint = "https://a"\naccess_key_id = "a"\nsecret_access_key = "b"\n',
        ["stores.r2.bucket", "必填"],
    ),
    (
        MINIMAL.replace("https://acc.r2.cloudflarestorage.com", "acc.r2.cloudflarestorage.com"),
        ["stores.r2.endpoint", "https://"],
    ),
    (MINIMAL.replace('"my-bucket"', '"r2://my-bucket"'), ["stores.r2.bucket", "bucket 名"]),
    (MINIMAL.replace('"my-bucket"', '"My_Bucket"'), ["stores.r2.bucket"]),
    (MINIMAL.replace('"my-bucket"', '""'), ["stores.r2.bucket", "不能为空"]),
    (MINIMAL.replace('"my-bucket"', "5"), ["stores.r2.bucket", "字符串"]),
    (MINIMAL + "region = 1\n", ["stores.r2.region", "字符串"]),
    # 后端
    ('[backends.vast]\nstore = "nope"\n', ["backends.vast.store", "nope", "[stores.nope]"]),
    (MINIMAL + '[backends.autodl]\nstore = "oss"\n', ["backends.autodl.store", "oss", "r2"]),
    ("[backends.vast]\nreliability = 1.5\n", ["backends.vast.reliability", "0 到 1"]),
    ('[backends.vast]\nreliability = "high"\n', ["backends.vast.reliability", "数字"]),
    ("[backends.vast]\nmin_inet_down_mbps = -1\n", ["backends.vast.min_inet_down_mbps"]),
    (
        "[backends.vast]\nmax_bandwidth_usd_per_gb = nan\n",
        ["backends.vast.max_bandwidth_usd_per_gb"],
    ),
    ('[backends.vast]\ndatacenter_only = "yes"\n', ["backends.vast.datacenter_only", "true"]),
    ("[backends.vast]\ndatacenter_only = 1\n", ["backends.vast.datacenter_only"]),
    (
        "[backends.vast]\nreliability = 0.9\n[backends.vast.filters]\nreliability = 0.95\n",
        ["backends.vast.reliability", "backends.vast.filters.reliability", "重复"],
    ),
    ('[backends.vast]\nbase_url = ""\n', ["backends.vast.base_url", "不能为空"]),
    ('[backends.autodl]\ndc_list = "westDC2"\n', ["backends.autodl.dc_list", "列表"]),
    ('[backends.autodl]\ndc_list = ["westDC2", 3]\n', ["backends.autodl.dc_list", "列表"]),
    ('[backends.autodl]\ncuda_v_from = "11.8"\n', ["backends.autodl.cuda_v_from", "整数"]),
    # runner 和顶层标量
    ("[runner]\nsync_interval_s = 0\n", ["runner.sync_interval_s", "不能小于 1"]),
    ('[runner]\nheartbeat_interval_s = "60"\n', ["runner.heartbeat_interval_s", "数字"]),
    ("[runner]\nheartbeat_interval_s = 0.5\n", ["runner.heartbeat_interval_s", "不能小于 1"]),
    ("[runner]\nmax_log_mb = 1.5\n", ["runner.max_log_mb", "整数"]),
    ("[runner]\nstall_timeout_min = -1\n", ["runner.stall_timeout_min", "不能小于 0"]),
    ("[runner]\nmax_log_mb = true\n", ["runner.max_log_mb", "整数"]),
    ('max_offers = "3"\n', ["max_offers", "整数"]),
    ("max_offers = 0\n", ["max_offers", "不能小于 1"]),
    ("disk_gb = 40.5\n", ["disk_gb", "整数"]),
    ("startup_timeout_min = 0\n", ["startup_timeout_min"]),
    ("local_data_confirm_gb = -1\n", ["local_data_confirm_gb"]),
    ("local_data_confirm_gb = inf\n", ["local_data_confirm_gb"]),
]


@pytest.mark.parametrize(("text", "needles"), INVALID, ids=[n[0] for _, n in INVALID])
def test_invalid_config_is_rejected(tmp_path, text, needles):
    path = _write(tmp_path, text)
    message = _error(path, *needles)
    assert str(path) in message


SECRET_LEAK_CASES = [
    # 语法错误出现在含密钥的那一行
    f'[backends.vast]\napi_key = "{SECRET}\n',
    # 密钥类型不对
    "[backends.vast]\napi_key = 9876543210\n",
    '[backends.autodl]\ntoken = ["' + SECRET + '"]\n',
    MINIMAL.replace('"secret"', "9876543210"),
    # 密钥还是占位符
    f'[backends.vast]\napi_key = "<{SECRET}>"\n',
    MINIMAL.replace('"secret"', f'"<{SECRET}>"'),
    # 密钥旁边有拼写错误
    f'[backends.vast]\napi_key = "{SECRET}"\napikey = "{SECRET}"\n',
    # 同一节里别的字段不合法
    MINIMAL.replace('"https://acc.r2.cloudflarestorage.com"', '"x"').replace(
        '"secret"', f'"{SECRET}"'
    ),
]


@pytest.mark.parametrize("text", SECRET_LEAK_CASES)
def test_errors_never_contain_secret_values(tmp_path, text):
    message = _error(_write(tmp_path, text))
    assert SECRET not in message
    assert "9876543210" not in message


# ---------------------------------------------------------------- 模板


@posix_only
def test_template_is_private_and_needs_filling(tmp_path):
    path = tmp_path / "new" / "dgpu" / "config.toml"
    old_umask = os.umask(0)  # 即使 umask 宽松，文件也必须是 600
    try:
        write_config_template(path)
    finally:
        os.umask(old_umask)
    assert path.stat().st_mode & 0o777 == 0o600

    # 原样加载：只因为占位符失败，而不是因为格式或未知 key
    _error(path, "占位符")


def test_filled_template_loads_with_defaults(tmp_path):
    path = tmp_path / "config.toml"
    write_config_template(path)
    text = path.read_text(encoding="utf-8")
    for placeholder, value in {
        "<ACCOUNT_ID>": "acc",
        "<BUCKET>": "my-bucket",
        "<R2_ACCESS_KEY_ID>": "id",
        "<R2_SECRET_ACCESS_KEY>": "secret",
        "<VAST_API_KEY>": "vk",
        "<AUTODL_TOKEN>": "at",
    }.items():
        assert placeholder in text
        text = text.replace(placeholder, value)
    path.write_text(text, encoding="utf-8")

    config = load_config(path, env={})
    assert config.stores["r2"].endpoint == "https://acc.r2.cloudflarestorage.com"
    assert config.vast == VastSettings(api_key="vk")
    assert config.autodl == AutodlSettings(token="at")
    assert config.runner == RunnerSettings()


def test_template_defaults_become_active_when_uncommented(tmp_path):
    path = tmp_path / "config.toml"
    write_config_template(path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<[^<>]+>", "x", text).replace('"x"', '"xyz"', 1)
    text = text.replace("https://x.r2", "https://acc.r2").replace('bucket = "x"', 'bucket = "bkt"')
    # 把所有被注释掉的 key = value 行（runner 表头除外）都打开，结果应该等于默认值
    uncommented = re.sub(r"^# ([a-z_]+ = )", r"\1", text, flags=re.M).replace(
        "# [runner]", "[runner]"
    )
    path.write_text(uncommented, encoding="utf-8")
    config = load_config(path, env={})
    assert config.runner == RunnerSettings()
    assert config.vast.reliability == VastSettings().reliability
    assert config.autodl.dc_list == ("westDC2",)
    assert config.disk_gb == 60


def test_template_refuses_to_overwrite(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("mine\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="已经存在"):
        write_config_template(path)
    assert path.read_text(encoding="utf-8") == "mine\n"


def test_template_refuses_to_follow_a_symlink(tmp_path):
    target = tmp_path / "elsewhere.toml"
    link = tmp_path / "config.toml"
    link.symlink_to(target)
    with pytest.raises(ConfigError, match="已经存在"):
        write_config_template(link)
    assert not target.exists()
