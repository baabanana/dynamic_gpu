"""dgpu 命令行：退出码、输出、错误处理，以及 build_deps 的装配。

build_deps 被替换成 World 的替身；最后两个测试用假模块验证 build_deps / init
import 的外部名字（集成后这些模块由其他工作包提供）。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from dgpu import cli as cli_mod
from dgpu import layout, states
from dgpu.contracts import (
    BACKEND_AUTODL,
    BACKEND_VAST,
    AutodlSettings,
    BackendError,
    ConfigError,
    DataRef,
    DataSource,
    VastSettings,
)
from dgpu.orchestrator import Deps
from dgpu.testing import TEST_PROFILE, MemoryStore
from test_orchestrator_support import (
    MAIN_VAST_KEY,
    T0,
    Plan,
    World,
    make_config,
    make_spec,
    seed_job,
)

JOB = "demo-260101000000-abcd"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """World + 指向它的 build_deps，以及一个存在的 job.yaml。"""
    world = World(tmp_path)
    job_file = tmp_path / "job.yaml"
    job_file.write_text("version: 1\n")
    state = {"data": ()}

    def load_spec(path, backend_override=None):
        assert Path(path) == job_file
        return make_spec(tmp_path, backend=backend_override or "vast", data=state["data"])

    def fake_build_deps(config_path):
        return world.deps(load_spec=load_spec)

    monkeypatch.setattr(cli_mod, "build_deps", fake_build_deps)
    return types.SimpleNamespace(world=world, job_file=job_file, state=state, tmp=tmp_path)


def run(*args: str, input: str | None = None):
    return CliRunner().invoke(cli_mod.cli, list(args), input=input)


# ---------------------------------------------------------------- submit


def test_submit_prints_job_id_on_stdout(env):
    result = run("submit", str(env.job_file))
    assert result.exit_code == 0, result.output
    job_id = result.stdout.strip()
    assert layout.is_valid_job_id(job_id)
    assert "费用上限" in result.stderr
    assert env.world.doc(layout.status_key(job_id))["phase"] == states.RUNNING


def test_submit_error_exits_2_without_traceback(env):
    env.world.backend.offers = []
    result = run("submit", str(env.job_file))
    assert result.exit_code == states.EXIT_USAGE
    assert result.stderr.startswith("错误：") and "没有符合条件的机器" in result.stderr
    assert "Traceback" not in result.output
    assert result.stdout == ""


def test_submit_backend_override(env):
    result = run("submit", str(env.job_file), "--backend", "autodl")
    assert result.exit_code == 2 and "后端 autodl 没有配置" in result.stderr


def test_submit_large_local_data_asks_and_respects_no(env):
    data_dir = env.tmp / "ds"
    data_dir.mkdir()
    (data_dir / "a.bin").write_bytes(b"x" * 10)
    env.state["data"] = (DataRef(name="ds", local_path=data_dir),)
    env.world.config = make_config(local_data_confirm_gb=0.0)

    result = run("submit", str(env.job_file), input="n\n")
    assert result.exit_code == 2 and "已取消" in result.stderr
    assert "继续吗" in result.output
    assert env.world.store.objects == {} and env.world.backend.creates() == []

    result = run("submit", str(env.job_file), "--yes")
    assert result.exit_code == 0, result.output
    assert "继续吗" not in result.output
    assert env.world.data_uploads == ["ds"]


def test_submit_wait_returns_job_exit_code(env):
    env.world.plan_queue = [Plan(run_s=300, final_phase=states.FAILED, final_reason="exit_code")]
    result = run("submit", str(env.job_file), "--wait")
    assert result.exit_code == states.EXIT_FAILED
    lines = result.stdout.split()
    assert layout.is_valid_job_id(lines[0]) and lines[-1] == states.FAILED


# ---------------------------------------------------------------- wait / status


@pytest.mark.parametrize(
    ("phase", "hb_ago", "code"),
    [
        (states.SUCCEEDED, 0, 0),
        (states.FAILED, 0, 1),
        (states.TIMED_OUT, 0, 3),
        (states.CANCELLED, 0, 4),
        (states.RUNNING, 3600, 5),  # 心跳过期 → LOST
    ],
)
def test_wait_exit_codes(env, phase, hb_ago, code):
    env.world.now = T0 + 3600
    seed_job(env.world.store, JOB, phase=phase, updated_at=T0 + 3600, hb_at=T0 + 3600 - hb_ago)
    result = run("wait", JOB)
    assert result.exit_code == code
    assert result.exit_code == states.EXIT_CODE_BY_PHASE[result.stdout.strip()]


def test_wait_unknown_job_exits_2(env):
    result = run("wait", JOB)
    assert result.exit_code == 2 and "找不到任务" in result.stderr


def test_ctrl_c_exits_130(env):
    env.world.now = T0 + 60
    seed_job(env.world.store, JOB, phase=states.RUNNING, hb_at=T0 + 60)

    def interrupt(world):
        raise KeyboardInterrupt

    env.world.on_sleep.append(interrupt)
    result = run("wait", JOB)
    assert result.exit_code == 130 and "已中断" in result.stderr


def test_status_json_and_text(env):
    env.world.now = T0 + 3600
    seed_job(env.world.store, JOB, phase=states.RUNNING, hb_at=T0, instance_id="42")
    result = run("status", JOB, "--json")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["phase"] == states.LOST and data["stored_phase"] == states.RUNNING
    assert data["instance_id"] == "42" and data["lost"] is True

    result = run("status", JOB)
    assert result.exit_code == 0
    assert f"任务 {JOB}：LOST" in result.stdout and "心跳已过期" in result.stdout


# ---------------------------------------------------------------- 其他命令


def test_list_json(env):
    seed_job(env.world.store, JOB, phase=states.SUCCEEDED)
    result = run("list", "--json")
    assert result.exit_code == 0
    assert [job["job_id"] for job in json.loads(result.stdout)] == [JOB]
    result = run("list")
    assert JOB in result.stdout and "SUCCEEDED" in result.stdout


def test_offers_json_and_text(env):
    result = run("offers", str(env.job_file), "--json", "--limit", "2")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert [offer["id"] for offer in data["offers"]] == ["o1", "o2"]
    assert data["currency"] == "USD" and data["cost_ceiling"] == pytest.approx(1.0)
    result = run("offers", str(env.job_file))
    assert "o3" in result.stdout and "费用上限" in result.stdout
    assert env.world.store.objects == {} and env.world.backend.creates() == []


def test_gc_json_dry_run(env):
    env.world.now = T0 + 7200
    seed_job(env.world.store, JOB, phase=states.FAILED)
    env.world.backend.add_instance("1", layout.label(JOB, 1), created_at=T0)
    env.world.backend.add_foreign("f1")
    result = run("gc", "--dry-run", "--json")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["dry_run"] is True and [i["instance_id"] for i in data["destroyed"]] == ["1"]
    assert env.world.backend.destroyed == []

    result = run("gc")
    assert result.exit_code == 0 and "已销毁 vast/1" in result.stdout
    assert env.world.backend.destroyed == ["1"]


def test_gc_failure_exits_1(env):
    env.world.now = T0 + 7200
    seed_job(env.world.store, JOB, phase=states.FAILED)
    env.world.backend.add_instance("1", layout.label(JOB, 1), created_at=T0)
    env.world.backend.destroy_errors["1"] = RuntimeError("boom")
    result = run("gc")
    assert result.exit_code == 1 and "失败 vast/1" in result.stdout


def test_cancel_retry_logs_pull(env):
    submitted = run("submit", str(env.job_file))
    job_id = submitted.stdout.strip()

    result = run("cancel", job_id, "--force")
    assert result.exit_code == 0 and "CANCELLED" in result.stdout
    assert env.world.alive() == []

    result = run("retry", job_id)
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == job_id
    assert env.world.doc(layout.attempt_key(job_id))["attempt"] == 2

    env.world.store.put_bytes(layout.log_key(job_id, 2), b"epoch 1\nepoch 2\n")
    result = run("logs", job_id)
    assert result.stdout == "epoch 1\nepoch 2\n"

    env.world.store.put_bytes(layout.output_prefix(job_id) + "best.pt", b"w")
    result = run("pull", job_id, str(env.tmp / "dl"))
    assert result.exit_code == 0
    assert (env.tmp / "dl/output/best.pt").read_bytes() == b"w"


def test_doctor_exit_codes(env, monkeypatch):
    result = run("doctor")
    assert result.exit_code == 0 and result.stdout.count("[OK]") == 3

    env.world.backend.balance = lambda: (_ for _ in ()).throw(RuntimeError("down"))
    result = run("doctor")
    assert result.exit_code == 1 and "[失败] 平台 vast" in result.stdout

    def no_config(config_path):
        raise ConfigError("配置文件权限不是 600")

    monkeypatch.setattr(cli_mod, "build_deps", no_config)
    result = run("doctor")
    assert result.exit_code == 1 and "[失败] 配置：配置文件权限不是 600" in result.stdout


def test_config_error_on_regular_command_exits_2(monkeypatch):
    def no_config(config_path):
        raise ConfigError(f"找不到配置文件 {config_path}")

    monkeypatch.setattr(cli_mod, "build_deps", no_config)
    result = run("--config", "/nowhere.toml", "status", JOB)
    assert result.exit_code == 2 and "/nowhere.toml" in result.stderr


# ---------------------------------------------------------------- 外部名字的装配


def test_init_writes_template_and_refuses_to_overwrite(tmp_path, monkeypatch):
    written = []
    fake = types.ModuleType("dgpu.config")
    fake.resolve_config_path = lambda cli_path, env: cli_path or tmp_path / "default.toml"
    fake.write_config_template = lambda path: (written.append(path), Path(path).write_text("x"))
    monkeypatch.setitem(sys.modules, "dgpu.config", fake)

    target = tmp_path / "cfg.toml"
    result = run("--config", str(target), "init")
    assert result.exit_code == 0 and written == [target]
    result = run("--config", str(target), "init")
    assert result.exit_code == 2 and "已存在" in result.stderr


def test_build_deps_wires_expected_external_names(tmp_path, monkeypatch):
    config = make_config(
        vast=VastSettings(api_key="k"), autodl=AutodlSettings(token="")
    )  # 只配置了 Vast
    seen = {}

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    class FakeVast:
        name = BACKEND_VAST

        def __init__(self, settings):
            seen["vast_settings"] = settings

    class FakeAutodl:
        name = BACKEND_AUTODL

        def __init__(self, settings):
            raise AssertionError("AutoDL 没有凭据，不应创建")

    def resolve_config_path(cli_path, env):
        seen["cli_path"] = cli_path
        return tmp_path / "config.toml"

    def open_store(profile):
        seen["profile"] = profile
        return MemoryStore(profile)

    module("dgpu.config", resolve_config_path=resolve_config_path, load_config=lambda p: config)
    module("dgpu.spec", load_spec=lambda path, backend_override=None: "spec")
    module(
        "dgpu.packaging",
        pack_code=lambda code_dir, dest: None,
        upload_dataset=lambda store, ref: DataSource(ref.name, "uploaded", "p/"),
        resolve_remote=lambda ref: DataSource(ref.name, "remote", "p/"),
    )
    module("dgpu.store", open_store=open_store)
    module("dgpu.backends.vast", VastBackend=FakeVast)
    module("dgpu.backends.autodl", AutodlBackend=FakeAutodl)
    module("dgpu.bundle", RunnerBundler=lambda: "bundler")

    deps = cli_mod.build_deps(str(tmp_path / "given.toml"))
    assert isinstance(deps, Deps)
    assert seen["cli_path"] == tmp_path / "given.toml"
    assert seen["vast_settings"] is config.vast
    assert seen["profile"] is TEST_PROFILE
    assert set(deps.backends) == {BACKEND_VAST}
    assert deps.bundler == "bundler"
    assert deps.config_path == tmp_path / "config.toml"
    assert deps.load_spec("x") == "spec"
    local = DataRef(name="a", local_path=tmp_path)
    remote = DataRef(name="b", bucket="bk", prefix="p/")
    assert deps.resolve_data(deps.store, local).bucket == "uploaded"
    assert deps.resolve_data(deps.store, remote).bucket == "remote"


def test_error_message_never_prints_main_key(env):
    def leaky_search(request):
        raise BackendError(f"bad key {MAIN_VAST_KEY}")

    env.world.backend.search = leaky_search
    result = run("submit", str(env.job_file))
    assert result.exit_code == 2
    assert MAIN_VAST_KEY not in result.output and "bad key ***" in result.stderr
