"""gc 的每条销毁规则、绝不碰外人的实例，以及 doctor。"""

from __future__ import annotations

import pytest

from dgpu import layout, states
from dgpu.contracts import BackendError, StoreError
from dgpu.testing import MemoryStore
from test_orchestrator_support import T0, SimBackend, World, seed_job

JOB = "demo-260101000000-abcd"
MIN = 60.0
NOW = T0 + 10 * 3600


@pytest.fixture
def world(tmp_path) -> World:
    w = World(tmp_path)
    w.now = NOW
    return w


def gc_events(world: World, job_id: str = JOB):
    return world.events(job_id, states.EV_GC_DESTROYED)


def test_gc_destroys_instance_of_finished_job(world):
    seed_job(world.store, JOB, phase=states.SUCCEEDED, hb_at=NOW - 60 * MIN)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    report = world.orchestrator().gc()
    assert world.backend.destroyed == ["1"]
    assert [item.instance_id for item in report.destroyed] == ["1"]
    assert "SUCCEEDED" in report.destroyed[0].reason
    (event,) = gc_events(world)
    assert event["source"] == states.SRC_GC and event["attempt"] == 1
    assert event["data"]["instance_id"] == "1"


def test_gc_destroys_older_attempt_and_keeps_current(world):
    seed_job(world.store, JOB, attempt=2, phase=states.RUNNING, hb_at=NOW - 30)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 7200)
    world.backend.add_instance("2", layout.label(JOB, 2), created_at=NOW - 3600)
    report = world.orchestrator().gc()
    assert world.backend.destroyed == ["1"]
    assert "旧 attempt" in report.destroyed[0].reason
    assert [item.instance_id for item in report.kept] == ["2"]


def test_gc_destroys_after_deadline(world):
    seed_job(world.store, JOB, hb_at=NOW - 30, deadline_at=NOW - 1)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    report = world.orchestrator().gc()
    assert world.backend.destroyed == ["1"]
    assert "截止时间" in report.destroyed[0].reason


@pytest.mark.parametrize(
    ("hb_ago", "inst_age", "destroy"),
    [
        (31 * MIN, 120 * MIN, True),  # 心跳过期 30 分钟以上
        (29 * MIN, 120 * MIN, False),  # 心跳还新
        (None, 25 * MIN, True),  # 从没有心跳，且已超过启动 + runner 超时（20 分钟）
        (None, 15 * MIN, False),  # 还在启动窗口内，可能正在启动
    ],
)
def test_gc_heartbeat_rule_respects_startup_window(world, hb_ago, inst_age, destroy):
    seed_job(
        world.store,
        JOB,
        phase=states.STARTING if hb_ago is None else states.RUNNING,
        hb_at=None if hb_ago is None else NOW - hb_ago,
        attempt_created_at=NOW - inst_age,
    )
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - inst_age)
    report = world.orchestrator().gc()
    assert (world.backend.destroyed == ["1"]) is destroy
    assert len(report.destroyed if destroy else report.kept) == 1


def test_gc_uses_attempt_time_when_platform_has_no_creation_time(world):
    seed_job(world.store, JOB, phase=states.STARTING, hb_at=None, attempt_created_at=NOW - 5 * MIN)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=None)
    world.orchestrator().gc()
    assert world.backend.destroyed == []


@pytest.mark.parametrize(
    ("created_ago", "destroy"), [(2 * 3600, True), (10 * MIN, False), (None, False)]
)
def test_gc_job_missing_from_storage(world, created_ago, destroy):
    created_at = None if created_ago is None else NOW - created_ago
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=created_at)
    report = world.orchestrator().gc()
    assert (world.backend.destroyed == ["1"]) is destroy
    assert "存储里没有这个任务" in (report.destroyed or report.kept)[0].reason
    assert world.store.objects == {}  # 不为不存在的任务凭空建目录


def test_gc_keeps_new_attempt_whose_status_still_shows_old_failure(world):
    # retry 已把 attempt 推到 2 并创建了实例，status.json 还是 attempt 1 的 FAILED
    seed_job(
        world.store,
        JOB,
        attempt=2,
        status_attempt=1,
        phase=states.FAILED,
        hb_at=None,
        attempt_created_at=NOW - 5 * MIN,
    )
    world.backend.add_instance("2", layout.label(JOB, 2), created_at=NOW - 5 * MIN)
    world.orchestrator().gc()
    assert world.backend.destroyed == []


def test_gc_never_touches_foreign_or_unparseable_instances(world):
    seed_job(world.store, JOB, phase=states.SUCCEEDED)
    world.backend.add_foreign("f1")
    world.backend.add_foreign("f2", label="dgpu")
    for iid, label in [("x1", "dgpu-Not_A_Valid-a1"), ("x2", "dgpu-abc"), ("x3", "dgpu-abc-a")]:
        world.backend.add_instance(iid, label, created_at=NOW - 30 * 3600)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    report = world.orchestrator().gc()
    assert world.backend.destroyed == ["1"]
    assert {"f1", "f2"} & {arg for name, arg in world.backend.calls if name == "destroy"} == set()
    kept = {item.instance_id: item.reason for item in report.kept}
    assert set(kept) == {"x1", "x2", "x3"}
    assert all("不处理" in reason for reason in kept.values())


def test_gc_dry_run_destroys_nothing(world):
    seed_job(world.store, JOB, phase=states.FAILED)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    world.backend.add_instance("2", layout.label("orphan-260101000000-ffff", 1), created_at=T0)
    report = world.orchestrator().gc(dry_run=True)
    assert report.dry_run
    assert {item.instance_id for item in report.destroyed} == {"1", "2"}
    assert world.backend.destroyed == []
    assert not any(name == "destroy" for name, _ in world.backend.calls)
    assert gc_events(world) == []


def test_gc_reports_destroy_failure_and_continues(world):
    seed_job(world.store, JOB, phase=states.FAILED)
    other = "demo-260101000001-abcd"
    seed_job(world.store, other, phase=states.CANCELLED)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    world.backend.add_instance("2", layout.label(other, 1), created_at=NOW - 3600)
    world.backend.destroy_errors["1"] = BackendError("boom")
    report = world.orchestrator().gc()
    assert [item.instance_id for item in report.failed] == ["1"]
    assert "可能仍在计费" in report.failed[0].reason
    assert world.backend.destroyed == ["2"]
    assert any(line.startswith("!!") for line in world.messages)


def test_gc_handles_each_backend_independently(world):
    broken = SimBackend(world, "autodl")

    def fail_list(label_prefix=layout.LABEL_PREFIX):
        raise BackendError("autodl API down")

    broken.list = fail_list
    seed_job(world.store, JOB, phase=states.SUCCEEDED)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    orch = world.orchestrator(backends={"autodl": broken, "vast": world.backend})
    report = orch.gc()
    assert world.backend.destroyed == ["1"]
    assert report.failed[0].backend == "autodl" and "列举实例失败" in report.failed[0].reason


def test_gc_keeps_instance_when_job_state_cannot_be_read(world):
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 30 * 3600)

    def broken_get(key):
        raise StoreError("R2 unreachable")

    world.store.get_json = broken_get
    report = world.orchestrator().gc()
    assert world.backend.destroyed == []
    assert "读取任务状态失败" in report.failed[0].reason


# ---------------------------------------------------------------- doctor


def test_doctor_all_ok(world):
    checks = world.orchestrator().doctor()
    assert [check.ok for check in checks] == [True, True, True]
    assert not any(key.startswith("dgpu/doctor/") for key in world.store.objects)  # 探针已删


def test_doctor_reports_each_failure_without_stopping(world):
    def broken_put(*args, **kwargs):
        raise StoreError("403 Forbidden")

    world.store.put_json = broken_put
    world.backend.balance = lambda: (_ for _ in ()).throw(BackendError("401 bad key"))
    checks = world.orchestrator().doctor()
    assert [check.ok for check in checks] == [True, False, False]
    assert "403" in checks[1].detail and "401" in checks[2].detail


def test_doctor_detects_store_without_conditional_writes(world):
    class NoConditionalStore(MemoryStore):
        def put_json(self, key, obj, *, if_match=None, if_none_match=False):
            return super().put_json(key, obj)

    store = NoConditionalStore()
    checks = world.orchestrator(store=store).doctor()
    assert checks[1].ok is False and "条件写" in checks[1].detail
    assert store.objects == {}


def test_doctor_uses_search_when_platform_has_no_balance(world):
    world.backend.balance_value = None
    checks = world.orchestrator().doctor()
    assert checks[2].ok and "搜索返回" in checks[2].detail
    assert ("search", "") in world.backend.calls


def test_doctor_without_backends(world):
    checks = world.orchestrator(backends={}).doctor()
    assert checks[-1].ok is False and "没有任何平台" in checks[-1].detail
