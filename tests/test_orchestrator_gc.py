"""gc 的每条销毁规则、绝不碰外人的实例，以及 doctor。"""

from __future__ import annotations

import pytest

from dgpu import layout, states
from dgpu.contracts import (
    INST_EXITED,
    INST_RUNNING,
    INST_UNKNOWN,
    BackendError,
    RunnerSettings,
    StoreError,
)
from dgpu.runner import util as runner_util
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


def teardown_min() -> float:
    """runner 最坏收尾时间（分钟），默认配置下约 42 分钟。"""
    settings = runner_util.runner_settings({"runner": RunnerSettings().to_dict()})
    return runner_util.teardown_worst_case_s(settings, runner_util.DEFAULT_TUNING) / 60


@pytest.mark.parametrize(
    ("hb_ago", "inst_age", "platform", "destroy"),
    [
        (29 * MIN, 120 * MIN, INST_RUNNING, False),  # 心跳还新
        # 心跳过期 30 分钟以上，但平台显示仍在运行：可能在最终上传或和存储断连，保留
        (31 * MIN, 120 * MIN, INST_RUNNING, False),
        # 心跳过期，平台也显示实例退出或状态未知：销毁
        (31 * MIN, 120 * MIN, INST_EXITED, True),
        (31 * MIN, 120 * MIN, INST_UNKNOWN, True),
        # 心跳停的时间超过 runner 最坏收尾时间：平台显示 running 也销毁
        ("teardown+1", 200 * MIN, INST_RUNNING, True),
        (None, 25 * MIN, INST_RUNNING, True),  # 从没有心跳，且已超过启动窗口（20 分钟）
        (None, 15 * MIN, INST_RUNNING, False),  # 还在启动窗口内，可能正在启动
    ],
)
def test_gc_heartbeat_rule(world, hb_ago, inst_age, platform, destroy):
    if hb_ago == "teardown+1":
        hb_ago = (teardown_min() + 1) * MIN
    seed_job(
        world.store,
        JOB,
        phase=states.STARTING if hb_ago is None else states.RUNNING,
        hb_at=None if hb_ago is None else NOW - hb_ago,
        attempt_created_at=NOW - inst_age,
    )
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - inst_age, phase=platform)
    report = world.orchestrator().gc()
    assert (world.backend.destroyed == ["1"]) is destroy
    assert len(report.destroyed if destroy else report.kept) == 1
    if hb_ago == 31 * MIN and not destroy:
        assert "失联（实例仍在运行）" in report.kept[0].reason


def test_gc_staleness_uses_storage_time_not_the_instance_clock(world):
    """实例的时钟慢了两小时：心跳内容里的 ts 很旧，但存储刚刚收到它。不能算过期。"""
    seed_job(world.store, JOB, hb_at=NOW - 2 * 3600)
    world.store.modified[layout.heartbeat_key(JOB)] = NOW - 30  # 存储 30 秒前收到这次心跳
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3 * 3600)
    report = world.orchestrator().gc()
    assert world.backend.destroyed == []
    assert "运行中" in report.kept[0].reason


def test_gc_staleness_ignores_a_fast_laptop_clock(world):
    """本机时钟快了两小时，存储的时间正常：心跳是新的，gc 不能销毁。"""
    seed_job(world.store, JOB, hb_at=NOW - 30)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    orch = world.orchestrator(clock=lambda: NOW + 2 * 3600)
    orch.gc()
    assert world.backend.destroyed == []


def test_gc_uses_attempt_time_when_platform_has_no_creation_time(world):
    seed_job(world.store, JOB, phase=states.STARTING, hb_at=None, attempt_created_at=NOW - 5 * MIN)
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=None)
    world.orchestrator().gc()
    assert world.backend.destroyed == []


@pytest.mark.parametrize("created_ago", [30 * 3600, 10 * MIN, None])
def test_gc_never_auto_destroys_instances_of_unknown_jobs(world, created_ago):
    """存储里查不到任务：可能是另一份配置（另一个 bucket）或同事的任务，只报告，不销毁。"""
    created_at = None if created_ago is None else NOW - created_ago
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=created_at)
    report = world.orchestrator().gc()
    assert world.backend.destroyed == []
    assert not any(name == "destroy" for name, _ in world.backend.calls)
    (item,) = report.unknown
    assert item.instance_id == "1" and item.label == layout.label(JOB, 1)
    assert "查不到这个任务" in item.reason and "--include-unknown" in item.reason
    if created_ago == 30 * 3600:
        assert "1800 分钟" in item.reason  # 报告实例年龄
    assert report.destroyed == [] and report.kept == []
    assert world.store.objects == {}  # 不为不存在的任务凭空建目录


def test_gc_include_unknown_destroys_after_confirmation(world):
    world.backend.add_instance("1", layout.label(JOB, 1), created_at=NOW - 3600)
    asked: list[list[str]] = []

    def decline(items):
        asked.append([item.instance_id for item in items])
        return False

    report = world.orchestrator().gc(include_unknown=True, confirm_unknown=decline)
    assert asked == [["1"]] and world.backend.destroyed == []
    assert [item.instance_id for item in report.unknown] == ["1"]

    dry = world.orchestrator().gc(dry_run=True, include_unknown=True, confirm_unknown=decline)
    assert [item.instance_id for item in dry.destroyed] == ["1"] and len(asked) == 1

    report = world.orchestrator().gc(include_unknown=True, confirm_unknown=lambda items: True)
    assert world.backend.destroyed == ["1"]
    assert [item.instance_id for item in report.destroyed] == ["1"]
    assert world.store.objects == {}

    world.backend.add_instance("2", layout.label(JOB, 2), created_at=NOW - 3600)
    world.orchestrator().gc(include_unknown=True, confirm_unknown=None)  # --yes
    assert world.backend.destroyed == ["1", "2"]


@pytest.mark.parametrize(
    "label",
    [
        "dgpu-x-a1",  # 手工起名
        "dgpu-my-job-a1",
        "dgpu-demo-260101000000-zzzz-a1",  # 随机部分不是十六进制
        "dgpu-demo-261301000000-abcd-a1",  # 13 月
        "dgpu-" + "n" * 30 + "-260101000000-abcd-a1",  # 名字超过 new_job_id 的截断长度
    ],
)
def test_gc_only_handles_labels_with_generated_job_ids(world, label):
    world.backend.add_instance("1", label, created_at=NOW - 30 * 3600, phase=INST_EXITED)
    report = world.orchestrator().gc(include_unknown=True, confirm_unknown=None)
    assert world.backend.destroyed == []
    (item,) = report.kept
    assert "不处理" in item.reason


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
    assert {item.instance_id for item in report.destroyed} == {"1"}
    assert {item.instance_id for item in report.unknown} == {"2"}  # 查不到任务：只报告
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


def test_doctor_verifies_the_key_when_platform_has_no_balance(world):
    world.backend.balance_value = None
    checks = world.orchestrator().doctor()
    assert checks[2].ok and not checks[2].warning and "key 有效" in checks[2].detail
    assert ("list", layout.LABEL_PREFIX) in world.backend.calls


def test_doctor_warns_when_the_key_cannot_read_the_balance(world):
    """Vast 的 key 没有读取账户信息的权限：余额查不到是警告，不是失败；key 仍要验证。"""
    world.backend.balance_value = None
    world.backend.balance_note = "key 没有读取账户信息的权限"
    check = world.orchestrator().doctor()[2]
    assert check.ok and check.warning
    assert "余额无法查询（key 没有读取账户信息的权限）" in check.detail
    assert ("list", layout.LABEL_PREFIX) in world.backend.calls

    def bad_key(label_prefix=layout.LABEL_PREFIX):
        raise BackendError("Vast 列出实例被拒绝（HTTP 401）：API key 无效")

    world.backend.list = bad_key
    check = world.orchestrator().doctor()[2]
    assert not check.ok and "API key 无效" in check.detail


def test_doctor_without_backends(world):
    checks = world.orchestrator(backends={}).doctor()
    assert checks[-1].ok is False and "没有任何平台" in checks[-1].detail
