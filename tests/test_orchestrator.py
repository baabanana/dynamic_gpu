"""submit 的行为和失败路径（设计第 5 节编排相关各行）。

runner 和平台都由 test_orchestrator_support.World 模拟，时间由注入的 sleep 推进。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dgpu import layout, orchestrator, states
from dgpu.contracts import (
    BackendError,
    Balance,
    DataRef,
    DgpuError,
)
from dgpu.orchestrator import AttemptConflict
from dgpu.runner import util as runner_util
from dgpu.testing import TEST_PROFILE, MemoryStore, make_offer
from test_orchestrator_support import (
    MAIN_AUTODL_TOKEN,
    MAIN_VAST_KEY,
    T0,
    Plan,
    World,
    make_config,
    make_spec,
)


def only_job_id(world: World) -> str:
    (job_id,) = world.store.list_dirs(layout.jobs_prefix())
    return job_id


def billed_hours(world: World, max_hours: float) -> float:
    """一台机器最坏计费的小时数：启动窗口 + 实例存活时限（和 watchdog 同一个公式）。"""
    cfg = world.config
    settings = runner_util.runner_settings({"runner": cfg.runner.to_dict()})
    lifetime = runner_util.instance_lifetime_s(max_hours, settings, runner_util.DEFAULT_TUNING)
    return ((cfg.startup_timeout_min + cfg.runner_timeout_min) * 60 + lifetime) / 3600


def secrets_left(world: World) -> list[str]:
    return [key for key in world.store.objects if "/secrets-" in key]


# ---------------------------------------------------------------- 正常路径


def test_submit_happy_path(tmp_path):
    world = World(tmp_path)
    result = world.orchestrator().submit(make_spec(tmp_path))
    job_id = result.job_id

    assert layout.is_valid_job_id(job_id)
    assert result.phase == states.RUNNING
    assert result.attempt == 1
    assert result.instance is not None and world.alive() == [result.instance.id]
    assert world.backend.creates() == ["o1"]  # 最便宜的那台

    attempt = world.doc(layout.attempt_key(job_id))
    assert attempt["attempt"] == 1
    assert attempt["instance_id"] == result.instance.id
    assert attempt["label"] == layout.label(job_id, 1)
    assert attempt["deadline_at"] > T0 + 2 * 3600
    assert world.doc(layout.status_key(job_id))["phase"] == states.RUNNING
    spec_doc = world.doc(layout.spec_key(job_id))
    assert spec_doc["run"] == "python train.py" and spec_doc["job_id"] == job_id
    assert world.store.exists(layout.code_key(job_id))

    launch = world.backend.launches[result.instance.id]
    assert launch.label == layout.label(job_id, 1)
    assert launch.disk_gb == 40
    assert launch.image == "pytorch/pytorch:2.4.0-cuda12.1"
    # 预签名 URL 的有效期 = 启动超时 + runner 超时 + 1 小时
    command = world.bundler.commands[0]
    assert layout.secrets_key(job_id, 1) in command["launch_url"]
    assert command["launch_url"].endswith(f"ttl={(15 + 5) * 60 + 3600}")
    hours = billed_hours(world, 2.0)
    assert f"费用上限：0.5 USD/h × {hours:.2f} h = {0.5 * hours:.2f} USD" in world.text()
    assert not world.events(job_id, states.EV_CREATE_FAILED)


def test_launch_doc_readable_at_boot_and_deleted_after_first_heartbeat(tmp_path):
    world = World(tmp_path)
    result = world.orchestrator().submit(make_spec(tmp_path))
    sim = world.sims[result.instance.id]
    assert sim.saw_launch_doc is True  # runner 启动时能读到启动文档
    assert secrets_left(world) == []  # 第一次心跳后就删了


def test_offers_creates_nothing(tmp_path):
    world = World(tmp_path)
    result = world.orchestrator().offers(make_spec(tmp_path))
    assert [offer.id for offer in result.offers] == ["o1", "o2", "o3"]
    assert result.cost_ceiling == pytest.approx(0.5 * billed_hours(world, 2.0))
    assert "RTX 3090" in result.constraints
    assert world.store.objects == {}
    assert world.backend.creates() == []


# ---------------------------------------------------------------- 提交前的检查


def test_no_offers_explains_constraints_and_writes_nothing(tmp_path):
    world = World(tmp_path, offers=[make_offer("o1", gpu_name="A100")])
    with pytest.raises(DgpuError) as info:
        world.orchestrator().submit(make_spec(tmp_path, price=0.25))
    message = str(info.value)
    assert "没有符合条件的机器" in message
    assert "RTX 3090" in message and "0.25 USD" in message and "可靠性 ≥ 0.98" in message
    assert "没有创建实例" in message
    assert world.store.objects == {}
    assert world.backend.creates() == []


def test_insufficient_balance_stops_before_anything_is_created(tmp_path):
    world = World(tmp_path)
    world.backend.balance_value = Balance(1.5, "USD")  # 上限是 0.5 × 约 3.4 h ≈ 1.72
    with pytest.raises(DgpuError, match="余额 1.50 USD 不够"):
        world.orchestrator().submit(make_spec(tmp_path))
    assert world.store.objects == {}
    assert world.backend.creates() == []


def test_balance_in_other_currency_is_not_compared(tmp_path):
    world = World(tmp_path)
    world.backend.balance_value = Balance(0.01, "CNY")
    world.orchestrator().submit(make_spec(tmp_path))
    assert "无法比较" in world.text()


def test_large_local_data_needs_confirmation(tmp_path):
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    (data_dir / "a.bin").write_bytes(b"x" * 100)
    world = World(tmp_path, config=make_config(local_data_confirm_gb=0.0))
    spec = make_spec(tmp_path, data=(DataRef(name="ds", local_path=data_dir),))
    asked = []

    def decline(gb: float) -> bool:
        asked.append(gb)
        return False

    with pytest.raises(DgpuError, match="已取消"):
        world.orchestrator().submit(spec, confirm_upload=decline)
    assert len(asked) == 1
    assert world.store.objects == {} and world.data_uploads == []
    assert world.backend.creates() == []

    world.orchestrator().submit(spec, confirm_upload=lambda gb: True)
    assert world.data_uploads == ["ds"]


def test_disk_size_follows_local_data(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "GB", 1000)  # 让 15000 字节算作 15 GB
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    (data_dir / "a.bin").write_bytes(b"x" * 15000)
    spec = make_spec(tmp_path, data=(DataRef(name="ds", local_path=data_dir),))
    world = World(tmp_path)
    assert world.orchestrator().offers(spec).request.disk_gb == 50  # max(40, 2×15+20)

    world = World(tmp_path, config=make_config(disk_gb=120))
    assert world.orchestrator().offers(spec).request.disk_gb == 120

    world = World(tmp_path, backend_name="autodl")
    autodl_spec = make_spec(
        tmp_path, backend="autodl", data=(DataRef(name="ds", local_path=data_dir),)
    )
    (data_dir / "b.bin").write_bytes(b"x" * 5000)  # 20 GB → 需要 60 GB > 50
    with pytest.raises(DgpuError, match="AutoDL 的数据盘固定 50 GB"):
        world.orchestrator().submit(autodl_spec)
    assert world.store.objects == {}


def test_unconfigured_backend_is_a_clear_error(tmp_path):
    world = World(tmp_path)
    with pytest.raises(DgpuError, match="后端 autodl 没有配置"):
        world.orchestrator().submit(make_spec(tmp_path, backend="autodl"))


def test_backend_with_a_different_store_profile_is_refused(tmp_path):
    other = TEST_PROFILE.__class__(**{**TEST_PROFILE.__dict__, "name": "oss"})
    config = make_config(stores={"r2": TEST_PROFILE, "oss": other})
    config = config.__class__(
        **{**config.__dict__, "vast": config.vast.__class__(api_key="k", store="oss")}
    )
    world = World(tmp_path, config=config)
    with pytest.raises(DgpuError, match="多个存储 profile 还未支持"):
        world.orchestrator().submit(make_spec(tmp_path))
    assert world.store.objects == {}


# ---------------------------------------------------------------- 换机


def test_offer_unavailable_then_next_offer_succeeds(tmp_path):
    world = World(tmp_path)
    world.backend.unavailable_offer_ids = {"o1"}
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.creates() == ["o1", "o2"]
    assert result.offer.id == "o2"
    assert result.attempt == 1  # 没有创建过实例，不需要换 attempt
    events = world.events(result.job_id, states.EV_CREATE_FAILED)
    assert len(events) == 1 and events[0]["data"]["offer_id"] == "o1"
    assert world.alive() == [result.instance.id]


def test_max_offers_exhausted_fails_cleanly(tmp_path):
    world = World(tmp_path, config=make_config(max_offers=2))
    world.backend.unavailable_offer_ids = {"o1", "o2", "o3"}
    with pytest.raises(DgpuError, match="max_offers=2"):
        world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.creates() == ["o1", "o2"]
    job_id = only_job_id(world)
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.FAILED and status["reason"] == states.REASON_NO_OFFER
    assert len(world.events(job_id, states.EV_CREATE_FAILED)) == 2
    assert world.alive() == []
    assert secrets_left(world) == []


def test_startup_timeout_destroys_and_moves_on_with_new_attempt(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck"), Plan()])
    result = world.orchestrator().submit(make_spec(tmp_path))
    job_id = result.job_id

    assert world.backend.creates() == ["o1", "o2"]
    assert world.backend.destroyed == ["1"]
    assert world.alive() == ["2"]
    assert world.now - T0 >= 15 * 60
    assert result.attempt == 2
    assert world.backend.launches["2"].label == layout.label(job_id, 2)
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 2
    assert world.doc(layout.status_key(job_id))["attempt"] == 2
    assert len(world.events(job_id, states.EV_STARTUP_TIMEOUT)) == 1
    relaunch = world.events(job_id, states.EV_RELAUNCH)
    assert [(e["data"]["old_attempt"], e["data"]["new_attempt"]) for e in relaunch] == [(1, 2)]
    assert world.events(job_id, states.EV_ATTEMPT_FENCED) == []  # 这个名字留给 runner
    destroyed = world.events(job_id, states.EV_INSTANCE_DESTROYED)
    assert [e["data"]["instance_id"] for e in destroyed] == ["1"]
    assert secrets_left(world) == []


def test_instance_exiting_during_startup_is_destroyed_and_replaced(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="exits"), Plan()])
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.destroyed == ["1"]  # exited 的实例也可能在计费，要显式销毁
    assert result.instance.id == "2"
    (event,) = world.events(result.job_id, states.EV_STARTUP_TIMEOUT)
    assert "exited" in event["message"]
    assert world.now - T0 < 15 * 60  # 不用等满启动超时


@pytest.mark.parametrize("second", ["none", "crash"])
def test_two_runner_timeouts_stop_with_image_hint(tmp_path, second):
    world = World(tmp_path, plans=[Plan(runner="none"), Plan(runner=second), Plan()])
    with pytest.raises(DgpuError, match="镜像"):
        world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.creates() == ["o1", "o2"]  # 第三台没有试
    assert world.alive() == []
    job_id = only_job_id(world)
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.FAILED and status["reason"] == states.REASON_RUNNER_TIMEOUT
    assert len(world.events(job_id, states.EV_RUNNER_TIMEOUT)) == 2
    assert secrets_left(world) == []


def test_preflight_failure_relaunches_on_next_offer(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="preflight"), Plan()])
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert result.attempt == 2 and result.instance.id == "2"
    assert world.alive() == ["2"]
    (event,) = world.events(result.job_id, states.EV_PREFLIGHT_FAILED)
    assert event["source"] == states.SRC_CLI and "nvidia-smi" in event["message"]
    status = world.doc(layout.status_key(result.job_id))
    assert status["phase"] == states.RUNNING and status["attempt"] == 2


# ---------------------------------------------------------------- 异常路径一定销毁


def test_keyboard_interrupt_mid_startup_destroys_instance(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck")])

    def interrupt_once(w: World) -> None:
        if w.sleeps == 3:
            raise KeyboardInterrupt

    world.on_sleep.append(interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.creates() == ["o1"]
    assert world.alive() == []
    job_id = only_job_id(world)
    assert world.doc(layout.status_key(job_id))["phase"] == states.CANCELLED
    assert world.events(job_id, states.EV_INSTANCE_DESTROYED)
    assert secrets_left(world) == []


def test_unknown_create_outcome_is_resolved_by_label(tmp_path):
    world = World(tmp_path)
    world.backend.raise_after_create["o1"] = TimeoutError("read timed out")
    with pytest.raises(TimeoutError):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert ("list", layout.label(job_id, 1)) in world.backend.calls
    assert world.backend.destroyed == ["1"]
    assert world.alive() == []
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.FAILED and "read timed out" in status["message"]


def test_retryable_create_error_that_did_create_is_cleaned_and_next_offer_used(tmp_path):
    world = World(tmp_path)
    world.backend.raise_after_create["o1"] = BackendError("gateway timeout", retryable=True)
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.destroyed == ["1"]
    assert result.instance.id == "2" and result.attempt == 2  # 可能有实例存在过，换新 attempt
    (event,) = world.events(result.job_id, states.EV_CREATE_FAILED)
    assert event["data"]["found_by_label"] == ["1"]


def test_label_lookup_is_exact_not_prefix(tmp_path):
    world = World(tmp_path)
    orch = world.orchestrator()
    world.backend.add_instance("99", "dgpu-x-1-a10", created_at=T0)
    world.backend.add_instance("98", "dgpu-x-1-a1", created_at=T0)
    assert [ref.id for ref in orch._lookup_label(world.backend, "dgpu-x-1-a1")] == ["98"]


def test_destroy_failure_is_loud_and_stops_creating(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck"), Plan()])
    world.backend.destroy_errors["1"] = BackendError("api unavailable")
    with pytest.raises(DgpuError) as info:
        world.orchestrator().submit(make_spec(tmp_path))
    assert "vast/1" in str(info.value) and "dgpu gc" in str(info.value)
    assert world.backend.creates() == ["o1"]  # 没确认销毁就不再开第二台
    assert any(line.startswith("!!") and "vast/1" in line for line in world.messages)


def test_destroy_that_never_takes_effect_is_reported(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck"), Plan()])
    world.backend.destroy = lambda ref: world.backend.calls.append(("destroy", ref.id))
    with pytest.raises(DgpuError, match="仍是 loading"):
        world.orchestrator().submit(make_spec(tmp_path))
    assert world.backend.creates() == ["o1"]


def test_cancel_during_startup_stops_submit(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck")])
    canceller = world.orchestrator()

    def cancel_later(w: World) -> None:
        if w.sleeps == 2:
            canceller.cancel(only_job_id(w))

    world.on_sleep.append(cancel_later)
    with pytest.raises(DgpuError, match="启动阶段被取消"):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert world.alive() == []
    assert world.doc(layout.status_key(job_id))["phase"] == states.CANCELLED
    assert world.backend.creates() == ["o1"]


def test_submit_fenced_by_another_process_destroys_own_instance(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck")])

    def bump_elsewhere(w: World) -> None:
        if w.sleeps == 2:
            job_id = only_job_id(w)
            doc, etag = w.store.get_json(layout.attempt_key(job_id))
            w.store.put_json(layout.attempt_key(job_id), {**doc, "attempt": 2}, if_match=etag)

    world.on_sleep.append(bump_elsewhere)
    with pytest.raises(AttemptConflict):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert world.alive() == []
    assert world.now - T0 < 60  # 下一次轮询就发现，不用等到启动超时
    # 被隔离的进程不能替新 attempt 写终态
    assert world.doc(layout.status_key(job_id))["phase"] == states.STARTING


def test_interrupt_after_being_fenced_does_not_write_terminal_status(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck")])

    def bump_then_interrupt(w: World) -> None:
        if w.sleeps == 2:
            job_id = only_job_id(w)
            doc, etag = w.store.get_json(layout.attempt_key(job_id))
            w.store.put_json(layout.attempt_key(job_id), {**doc, "attempt": 2}, if_match=etag)
            raise KeyboardInterrupt

    world.on_sleep.append(bump_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert world.alive() == []  # 自己的实例照样销毁
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.STARTING and status["attempt"] == 1  # 但不替别人写终态


# ---------------------------------------------------------------- 凭据


class RecordingStore(MemoryStore):
    """记录写入存储的每一个字节，检查凭据从没被写进去。"""

    def __init__(self) -> None:
        super().__init__()
        self.written: list[str] = []

    def put_json(self, key, obj, *, if_match=None, if_none_match=False):
        self.written.append(key + json.dumps(obj, ensure_ascii=False))
        return super().put_json(key, obj, if_match=if_match, if_none_match=if_none_match)

    def put_bytes(self, key, data):
        self.written.append(key + data.decode("utf-8", "replace"))
        super().put_bytes(key, data)

    def put_file(self, key, path):
        self.written.append(key + Path(path).read_bytes().decode("utf-8", "replace"))
        super().put_file(key, path)


def test_main_api_keys_never_reach_storage_or_bootstrap(tmp_path):
    store = RecordingStore()
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    (data_dir / "a.bin").write_bytes(b"data")
    world = World(tmp_path, store=store, plans=[Plan(boot="stuck"), Plan(runner="preflight")])
    spec = make_spec(tmp_path, data=(DataRef(name="ds", local_path=data_dir),))
    result = world.orchestrator().submit(spec)
    world.orchestrator().cancel(result.job_id, force=True)

    launches = [
        f"{launch.image} {launch.bootstrap_cmd} {launch.label}"
        for launch in world.backend.launches.values()
    ]
    bootstrap_inputs = [json.dumps(command) for command in world.bundler.commands]
    everything = store.written + launches + bootstrap_inputs + world.messages
    for secret in (MAIN_VAST_KEY, MAIN_AUTODL_TOKEN):
        assert not any(secret in text for text in everything)
    # 存储凭据只出现在 secrets-<n>.json 里，且不进启动命令、不留在存储里、不打印
    store_secret = TEST_PROFILE.secret_access_key
    assert all("/secrets-" in text for text in store.written if store_secret in text)
    assert not any(store_secret in text for text in launches + bootstrap_inputs + world.messages)
    assert not any(store_secret.encode() in data for data in store.objects.values())


def test_backend_error_text_with_a_key_is_redacted_everywhere_we_write(tmp_path):
    world = World(tmp_path)
    leaky = f"401 unauthorized for Authorization: Bearer {MAIN_VAST_KEY}"
    world.backend.raise_after_create["o1"] = BackendError(leaky, retryable=True)
    world.backend.raise_after_create["o2"] = BackendError(leaky)  # 不可重试：直接失败
    with pytest.raises(BackendError):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert world.events(job_id, states.EV_CREATE_FAILED)[0]["message"].count("***") == 1
    assert "***" in world.doc(layout.status_key(job_id))["message"]
    assert not any(MAIN_VAST_KEY.encode() in data for data in world.store.objects.values())
    assert not any(MAIN_VAST_KEY in line for line in world.messages)
    assert world.alive() == []


def test_cleanup_continues_when_platform_is_down(tmp_path):
    world = World(tmp_path)
    world.backend.raise_after_create["o1"] = TimeoutError("read timed out")

    def list_down(label_prefix=layout.LABEL_PREFIX):
        raise BackendError("platform API down")

    world.backend.list = list_down
    with pytest.raises(TimeoutError):  # 原异常照常抛出，清理失败另外大声报告
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    status = world.doc(layout.status_key(job_id))
    # 查不了标签就不知道有没有创建：如实说“结果未知”，不说“已销毁”，提示 gc
    assert status["phase"] == states.FAILED and "结果未知" in status["message"]
    assert "已销毁" not in status["message"] and "dgpu gc" in status["message"]
    assert secrets_left(world) == []  # 平台不可用也要删掉凭据
    assert any(line.startswith("!!") for line in world.messages)
