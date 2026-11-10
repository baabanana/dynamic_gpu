"""dgpu 命令行：退出码、输出、错误处理，以及 build_deps 的装配。

build_deps 被替换成 World 的替身；最后两个测试用假模块验证 build_deps / init
import 的外部名字（集成后这些模块由其他工作包提供）。
"""

from __future__ import annotations

import json
import signal
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
    StoreError,
    VastSettings,
)
from dgpu.orchestrator import Deps, Terminated
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
    last = result.stderr.strip().splitlines()[-1]
    assert last.startswith("错误：") and "没有符合条件的机器" in last
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
    # 价格上限 × (启动窗口 + max_hours + 收尾)，比 0.5 × 2 h 大
    assert data["currency"] == "USD" and 1.0 < data["cost_ceiling"] < 2.0
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

    def resolve_data(store, ref):
        return DataSource(ref.name, "uploaded" if ref.is_local else "remote", "p/")

    module(
        "dgpu.packaging",
        pack_code=lambda code_dir, dest, exclude=(): None,
        resolve_data=resolve_data,
        scan_dataset=lambda path: ("listing", path),
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
    assert deps.scan_data(tmp_path) == ("listing", tmp_path)


def test_error_message_never_prints_main_key(env):
    def leaky_search(request):
        raise BackendError(f"bad key {MAIN_VAST_KEY}")

    env.world.backend.search = leaky_search
    result = run("submit", str(env.job_file))
    assert result.exit_code == 2
    assert MAIN_VAST_KEY not in result.output and "bad key ***" in result.stderr


# ---------------------------------------------------------------- 评审 A1：SIGTERM / SIGHUP


def test_termination_signals_become_terminated_once_and_handlers_are_restored():
    before = {sig: signal.getsignal(sig) for sig in cli_mod.TERMINATION_SIGNALS}
    with cli_mod.termination_signals():
        with pytest.raises(Terminated) as info:
            signal.raise_signal(signal.SIGTERM)
        assert info.value.exit_code == 143 and info.value.signame == "SIGTERM"
        signal.raise_signal(signal.SIGHUP)  # 清理期间再来的信号只记一笔，不打断
    assert {sig: signal.getsignal(sig) for sig in cli_mod.TERMINATION_SIGNALS} == before
    with cli_mod.termination_signals():
        with pytest.raises(Terminated) as info:
            signal.raise_signal(signal.SIGHUP)
        assert info.value.exit_code == 129


@pytest.mark.parametrize(("signum", "code"), [(signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_submit_destroys_the_instance_on_sigterm_or_sighup(env, signum, code):
    env.world.plan_queue = [Plan(boot="stuck")]

    def send_signal(world):
        if world.sleeps == 3:
            signal.raise_signal(signum)

    env.world.on_sleep.append(send_signal)
    with cli_mod.termination_signals():
        result = run("submit", str(env.job_file))
    assert result.exit_code == code, result.output
    assert env.world.backend.creates() == ["o1"]
    assert env.world.alive() == []  # 实例已销毁，不会一直计费
    (job_id,) = env.world.store.list_dirs(layout.jobs_prefix())
    status = env.world.doc(layout.status_key(job_id))
    assert status["phase"] == states.CANCELLED
    assert signal.Signals(signum).name in status["message"]
    assert f"收到 {signal.Signals(signum).name}，已清理并退出" in result.stderr


def test_cancel_force_on_sigterm_still_destroys(env):
    submitted = run("submit", str(env.job_file))
    job_id = submitted.stdout.strip()
    real_destroy = env.world.backend.destroy
    calls = []

    def destroy_with_signal(ref):
        calls.append(ref.id)
        if len(calls) == 1:
            signal.raise_signal(signal.SIGTERM)
        real_destroy(ref)

    env.world.backend.destroy = destroy_with_signal
    with cli_mod.termination_signals():
        result = run("cancel", job_id, "--force")
    assert result.exit_code == 143
    assert env.world.alive() == []
    assert env.world.doc(layout.status_key(job_id))["phase"] == states.CANCELLED


def test_progress_output_never_breaks_cleanup_when_the_terminal_is_gone(monkeypatch):
    def broken_echo(*args, **kwargs):
        raise OSError(5, "Input/output error")  # SIGHUP 之后写 stderr

    monkeypatch.setattr(cli_mod.click, "echo", broken_echo)
    cli_mod._progress("清理中……")  # 不抛异常


# ---------------------------------------------------------------- 评审 A7、A12：退出码


def test_wait_exit_code_6_when_the_job_succeeded_but_cleanup_failed(env):
    env.world.plan_queue = [Plan(run_s=300, self_destroy=False)]
    job_id = run("submit", str(env.job_file)).stdout.strip()
    env.world.backend.destroy_errors["1"] = BackendError("api unavailable")
    result = run("wait", job_id)
    assert result.exit_code == states.EXIT_CLEANUP_FAILED == 6
    assert result.stdout.strip() == states.SUCCEEDED
    assert "dgpu gc" in result.stderr and "vast/1" in result.stderr


def test_wait_keeps_the_job_exit_code_when_a_failed_job_also_failed_cleanup(env):
    env.world.plan_queue = [
        Plan(run_s=300, self_destroy=False, final_phase=states.FAILED, final_reason="exit_code")
    ]
    job_id = run("submit", str(env.job_file)).stdout.strip()
    env.world.backend.destroy_errors["1"] = BackendError("api unavailable")
    result = run("wait", job_id)
    assert result.exit_code == states.EXIT_FAILED
    assert "dgpu gc" in result.stderr


def test_wait_exit_code_7_when_storage_stays_unreachable(env):
    env.world.now = T0 + 60
    seed_job(env.world.store, JOB, phase=states.RUNNING, hb_at=T0 + 60, updated_at=T0 + 60)

    def unreachable(key):
        raise StoreError("read timeout", transient=True)

    env.world.store.get_json_meta = unreachable
    result = run("wait", JOB)
    assert result.exit_code == states.EXIT_STORE_UNREACHABLE == 7
    assert "连不上存储" in result.stderr and "不受影响" in result.stderr


def test_wait_permanent_storage_error_is_still_exit_2(env):
    seed_job(env.world.store, JOB, phase=states.RUNNING)

    def denied(key):
        raise StoreError("AccessDenied (HTTP 403)")

    env.world.store.get_json_meta = denied
    result = run("wait", JOB)
    assert result.exit_code == states.EXIT_USAGE


def test_retry_exits_with_the_phase_when_the_new_attempt_ends_before_running(env):
    seed_job(env.world.store, JOB, phase=states.FAILED, reason="exit_code", hb_at=None)
    env.world.plan_queue = [
        Plan(runner="ends", final_phase=states.FAILED, final_reason="data_sync")
    ]
    result = run("retry", JOB)
    assert result.exit_code == states.EXIT_FAILED, result.output
    assert result.stdout.strip() == JOB and "FAILED" in result.stderr


# ---------------------------------------------------------------- 评审 A3、A4、A10：新参数和帮助


def test_retry_with_a_pending_cancel_is_refused_and_help_explains_the_rule(env):
    env.world.now = T0 + 3600
    seed_job(env.world.store, JOB, phase=states.RUNNING, hb_at=T0, updated_at=T0)
    assert run("cancel", JOB).exit_code == 0
    result = run("retry", JOB)
    assert result.exit_code == 2 and "未完成的取消请求" in result.stderr
    assert env.world.backend.creates() == []
    help_text = run("retry", "--help").stdout
    assert "取消请求" in help_text and "cancel --force" in help_text


def test_gc_reports_unknown_jobs_and_include_unknown_asks_first(env):
    env.world.now = T0 + 7200
    env.world.backend.add_instance("9", layout.label(JOB, 1), created_at=T0)
    result = run("gc")
    assert result.exit_code == 0 and env.world.backend.destroyed == []
    assert "未知任务 vast/9" in result.stdout and "--include-unknown" in result.stderr

    result = run("gc", "--include-unknown", input="n\n")
    assert env.world.backend.destroyed == [] and "确定销毁" in result.stderr
    data = json.loads(run("gc", "--dry-run", "--json").stdout)
    assert [item["instance_id"] for item in data["unknown"]] == ["9"]

    result = run("gc", "--include-unknown", input="y\n")
    assert result.exit_code == 0 and env.world.backend.destroyed == ["9"]


def test_gc_include_unknown_with_yes_does_not_ask(env):
    env.world.now = T0 + 7200
    env.world.backend.add_instance("9", layout.label(JOB, 1), created_at=T0)
    result = run("gc", "--include-unknown", "--yes")
    assert result.exit_code == 0 and env.world.backend.destroyed == ["9"]
    assert "确定销毁" not in result.output


def test_submit_with_a_broken_symlink_needs_yes(env):
    from dgpu.packaging import PackResult

    env.world.pack_result = PackResult(
        path=env.tmp / "x.tar.gz",
        file_count=1,
        total_bytes=1,
        archive_bytes=1,
        sha256="0" * 64,
        skipped=(("weights", "符号链接指向代码目录之外：/data/w"),),
        skipped_links=("weights",),
    )
    result = run("submit", str(env.job_file))
    assert result.exit_code == 2 and "weights" in result.stderr and "--yes" in result.stderr
    assert env.world.backend.creates() == []
    result = run("submit", str(env.job_file), "--yes")
    assert result.exit_code == 0, result.output


def test_doctor_prints_a_warning_for_a_key_without_account_permission(env):
    env.world.backend.balance_value = None
    env.world.backend.balance_note = "key 没有读取账户信息的权限"
    result = run("doctor")
    assert result.exit_code == 0
    assert "[警告] 平台 vast" in result.stdout
    assert "余额无法查询（key 没有读取账户信息的权限）" in result.stdout


def test_submit_proceeds_without_balance_and_says_so(env):
    env.world.backend.balance_value = None
    env.world.backend.balance_note = "key 没有读取账户信息的权限"
    result = run("submit", str(env.job_file))
    assert result.exit_code == 0
    assert "未检查余额（key 没有读取账户信息的权限）" in result.stderr
