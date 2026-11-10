"""集成时补上的接缝：编排一侧（集成清单第 1–6、9、10 项）。

runner 和平台由 test_orchestrator_support.World 模拟；真实进程之间的配合见
test_integration_e2e.py。
"""

from __future__ import annotations

import pytest

from dgpu import layout, orchestrator, states
from dgpu.contracts import SINGLE_UPLOAD_LIMIT, BackendError, DataRef, DgpuError, PreconditionFailed
from dgpu.runner import util as runner_util
from dgpu.testing import TEST_PROFILE, MemoryStore
from test_orchestrator_support import T0, Plan, World, make_spec, seed_job

JOB = "demo-260101000000-abcd"


# ---------------------------------------------------------------- 第 1 项：代码包不含本地数据


def test_local_data_dirs_are_excluded_from_the_code_archive(tmp_path):
    world = World(tmp_path)
    code = make_spec(tmp_path).code
    inside = code / "datasets" / "set1"
    inside.mkdir(parents=True)
    (inside / "a.txt").write_text("a")
    outside = tmp_path / "other"
    outside.mkdir()
    (outside / "b.txt").write_text("b")
    spec = make_spec(
        tmp_path,
        data=(
            DataRef("set1", local_path=inside),
            DataRef("other", local_path=outside),
            DataRef("remote", bucket=TEST_PROFILE.bucket, prefix="sets/remote/"),
        ),
    )
    world.orchestrator().submit(spec)
    assert world.pack_excludes == [[inside, outside]]


# ---------------------------------------------------------------- 第 2 项：别的 bucket


def test_data_in_another_bucket_warns_on_offers_and_submit_but_is_allowed(tmp_path):
    world = World(tmp_path)
    spec = make_spec(
        tmp_path,
        data=(
            DataRef("coco", bucket="someone-elses", prefix="sets/coco/"),
            DataRef("own", bucket=TEST_PROFILE.bucket, prefix="sets/own/"),
        ),
    )
    orch = world.orchestrator()
    orch.offers(spec)
    (warning,) = [line for line in world.messages if "读权限" in line]
    assert "someone-elses" in warning and TEST_PROFILE.bucket in warning
    world.messages.clear()
    result = orch.submit(spec)
    assert result.phase == states.RUNNING
    assert [line for line in world.messages if "读权限" in line] == [warning]


def test_data_in_the_configured_bucket_does_not_warn(tmp_path):
    world = World(tmp_path)
    spec = make_spec(tmp_path, data=(DataRef("own", bucket=TEST_PROFILE.bucket, prefix="s/own/"),))
    world.orchestrator().offers(spec)
    assert not any("读权限" in line for line in world.messages)


# ---------------------------------------------------------------- 第 3 项：5 GiB 单文件上限


def test_local_file_over_the_upload_limit_fails_before_anything_is_written(tmp_path):
    world = World(tmp_path)
    data = tmp_path / "bigset"
    data.mkdir()
    (data / "small.txt").write_text("ok")
    with open(data / "huge.bin", "wb") as handle:
        handle.truncate(SINGLE_UPLOAD_LIMIT + 1)  # 稀疏文件，不占磁盘
    spec = make_spec(tmp_path, data=(DataRef("bigset", local_path=data),))
    with pytest.raises(DgpuError) as info:
        world.orchestrator().submit(spec)
    message = str(info.value)
    assert "huge.bin" in message and "rclone" in message
    assert f"r2://{TEST_PROFILE.bucket}/datasets/bigset" in message
    assert world.store.objects == {}  # 什么都没上传
    assert world.backend.calls == []  # 也没有调用平台
    assert world.data_uploads == []


# ---------------------------------------------------------------- 第 4 项：条件写的响应丢失


class LostResponseStore(MemoryStore):
    """条件写真的生效了，但响应丢了：存储层重试时看到的是自己刚写的版本，报 PreconditionFailed。"""

    def __init__(self) -> None:
        super().__init__()
        self.armed: dict[str, int] = {}  # key 后缀 → 还要“丢”几次响应
        self.conditional_puts: list[str] = []

    def put_json(self, key, obj, *, if_match=None, if_none_match=False):
        etag = super().put_json(key, obj, if_match=if_match, if_none_match=if_none_match)
        if if_match is not None or if_none_match:
            self.conditional_puts.append(key)
            for suffix, left in self.armed.items():
                if key.endswith(suffix) and left > 0:
                    self.armed[suffix] = left - 1
                    raise PreconditionFailed(f"{key} was modified (retried after a lost response)")
        return etag


def test_submit_survives_a_lost_response_when_creating_attempt_and_status(tmp_path):
    store = LostResponseStore()
    store.armed = {"attempt.json": 1, "status.json": 1}
    world = World(tmp_path, store=store)
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert result.attempt == 1 and result.phase == states.RUNNING
    assert store.armed == {"attempt.json": 0, "status.json": 0}  # 两次都“丢”过


def test_cancel_after_a_lost_response_is_success_and_not_repeated(tmp_path):
    store = LostResponseStore()
    world = World(tmp_path, store=store)
    seed_job(store, JOB)
    before = world.doc(layout.attempt_key(JOB))
    store.armed = {"attempt.json": 1}
    world.orchestrator().cancel(JOB)
    after = world.doc(layout.attempt_key(JOB))
    assert after["cancel"] is True
    assert after["writer"] != before["writer"]  # 每次写入换一个 writer
    assert store.conditional_puts.count(layout.attempt_key(JOB)) == 1  # 没有再写一次


def test_retry_after_a_lost_response_bumps_the_attempt_only_once(tmp_path):
    store = LostResponseStore()
    world = World(tmp_path, store=store, plans=[Plan()])
    seed_job(store, JOB, hb_at=T0 - 3600, updated_at=T0 - 3600)  # 心跳过期：LOST
    orch = world.orchestrator()
    assert orch.status(JOB).phase == states.LOST
    store.armed = {"attempt.json": 1}
    result = orch.retry(JOB)
    assert result.attempt == 2
    assert world.doc(layout.attempt_key(JOB))["attempt"] == 2
    assert [launch.attempt for launch in world.backend.launches.values()] == [2]
    assert len(world.events(JOB, states.EV_RELAUNCH)) == 1


def test_a_real_conflict_is_still_a_conflict(tmp_path):
    """有人真的改了 attempt.json：不能因为“可能是自己写的”就当作成功。"""
    store = LostResponseStore()
    world = World(tmp_path, store=store)
    seed_job(store, JOB)
    orch = world.orchestrator()
    original = store.put_json
    raced = []

    def put_json(key, obj, *, if_match=None, if_none_match=False):
        if key == layout.attempt_key(JOB) and if_match is not None and not raced:
            raced.append(True)  # 另一个进程抢先改了 deadline_at
            doc, etag = store.get_json(key)
            original(key, {**doc, "deadline_at": 1.0}, if_match=etag)
        return original(key, obj, if_match=if_match, if_none_match=if_none_match)

    store.put_json = put_json
    orch.cancel(JOB)
    doc = world.doc(layout.attempt_key(JOB))
    assert doc["cancel"] is True and doc["deadline_at"] == 1.0  # 重读后在新版本上再改


# ---------------------------------------------------------------- 第 5 项：request.json


def test_request_doc_lives_at_layout_request_key(tmp_path):
    world = World(tmp_path)
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert layout.request_key(result.job_id) == layout.job_prefix(result.job_id) + "request.json"
    request = world.doc(layout.request_key(result.job_id))
    assert request["gpu_names"] == ["RTX 3090", "RTX 4090"] and request["backend"] == "vast"


# ---------------------------------------------------------------- 第 6 项：事件名、时间戳


def test_timestamps_come_from_the_injected_clock(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck"), Plan()])  # 换一次机，留下几条事件
    result = world.orchestrator().submit(make_spec(tmp_path))
    attempt = world.doc(layout.attempt_key(result.job_id))
    assert T0 < attempt["created_at"] <= world.now  # 换机时由注入的时钟写入
    events = world.events(result.job_id)
    assert len(events) >= 3 and all(T0 <= event["ts"] <= world.now for event in events)
    status = world.doc(layout.status_key(result.job_id))
    assert T0 <= status["updated_at"] <= world.now


def test_destroy_failure_is_recorded_as_destroy_failed(tmp_path):
    world = World(tmp_path, plans=[Plan(boot="stuck"), Plan()])
    world.backend.destroy_errors["1"] = BackendError("api unavailable")
    with pytest.raises(DgpuError):
        world.orchestrator().submit(make_spec(tmp_path))
    (job_id,) = world.store.list_dirs(layout.jobs_prefix())
    events = world.events(job_id, states.EV_DESTROY_FAILED)  # 换机时一次，异常清理时再试一次
    assert events and all(e["data"]["instance_id"] == "1" for e in events)
    assert all(e["source"] == states.SRC_CLI and "可能仍在计费" in e["message"] for e in events)


# ---------------------------------------------------------------- 第 9 项：deadline_at


def test_deadline_at_covers_the_watchdogs_hard_deadline_and_its_teardown(tmp_path):
    world = World(tmp_path)
    result = world.orchestrator().submit(make_spec(tmp_path, max_hours=2.0))
    doc = world.doc(layout.attempt_key(result.job_id))
    cfg = world.config
    settings = runner_util.runner_settings({"runner": cfg.runner.to_dict()})
    tuning = runner_util.DEFAULT_TUNING
    startup = (cfg.startup_timeout_min + cfg.runner_timeout_min) * 60
    lifetime = runner_util.instance_lifetime_s(2.0, settings, tuning)
    assert doc["deadline_at"] == pytest.approx(
        doc["created_at"] + startup + lifetime + orchestrator.DEADLINE_MARGIN_S
    )
    hard = runner_util.watchdog_hard_deadline_s(2.0, settings, tuning)
    worst = runner_util.teardown_worst_case_s(settings, tuning)
    assert doc["deadline_at"] - doc["created_at"] > startup + hard > startup + 2 * 3600 + worst


# ---------------------------------------------------------------- 第 10 项：gc 确认实例已消失


def test_gc_reports_an_unconfirmed_destroy_as_failed(tmp_path):
    world = World(tmp_path)
    now = T0 + 10 * 3600
    world.now = now
    seed_job(world.store, JOB, phase=states.FAILED)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=now - 3600)
    # 平台受理了销毁请求，但实例一直还在
    world.backend.destroy = lambda ref: world.backend.calls.append(("destroy", ref.id))
    report = world.orchestrator().gc()
    assert report.destroyed == []
    assert [item.instance_id for item in report.failed] == ["1"]
    assert "仍是 running" in report.failed[0].reason
    (event,) = world.events(JOB, states.EV_DESTROY_FAILED)
    assert event["source"] == states.SRC_GC
    assert world.events(JOB, states.EV_GC_DESTROYED) == []
    # 等待有上限
    limit = orchestrator.DESTROY_CONFIRM_TIMEOUT_S + orchestrator.DESTROY_CONFIRM_POLL_S
    assert world.now - now <= limit


def test_gc_success_is_only_reported_after_the_instance_is_gone(tmp_path):
    world = World(tmp_path)
    now = T0 + 10 * 3600
    world.now = now
    seed_job(world.store, JOB, phase=states.FAILED)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=now - 3600)
    real_destroy = world.backend.destroy
    pending: list = []

    def slow_destroy(ref):  # 受理后过一会儿才真正消失
        world.backend.calls.append(("destroy", ref.id))
        if not pending:
            pending.append(ref)
            world.on_sleep.append(lambda w: real_destroy(ref) if w.now - now > 30 else None)

    world.backend.destroy = slow_destroy
    report = world.orchestrator().gc()
    assert [item.instance_id for item in report.destroyed] == ["1"]
    assert world.backend.instances == {}
    assert world.now - now > 30
