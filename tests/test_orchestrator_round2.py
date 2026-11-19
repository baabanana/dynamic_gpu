"""第二轮复核（docs/history/dgpu-review-2026-11-10.md 第 E 节）编排一侧的回归测试：F1、F3、F4、F5。

runner 和平台由 test_orchestrator_support.World 模拟，时间由注入的 sleep 推进；平台“晚一点才
看得见”的实例用 test_orchestrator_review.LaggyBackend。每个测试在修复之前都会失败。
"""

from __future__ import annotations

import signal

import pytest

from dgpu import layout, states
from dgpu.cli import _format_view
from dgpu.contracts import INST_EXITED, BackendError, DgpuError
from dgpu.orchestrator import Terminated
from test_orchestrator_review import LaggyBackend, lookups_in, use_backend
from test_orchestrator_support import T0, Plan, World, make_config, make_spec, seed_job

MIN = 60.0
SUSPECT_FIELD = "suspect_labels"  # attempt.json 里创建结果未知的标签


def only_job_id(world: World) -> str:
    (job_id,) = world.store.list_dirs(layout.jobs_prefix())
    return job_id


def unclear(world: World, offer: str, appears_after: float) -> LaggyBackend:
    """offer 上的 create 报“结果未知”（可重试），实例其实建好了，appears_after 秒后才查得到。"""
    backend = use_backend(world, LaggyBackend(world, hidden={offer: appears_after}))
    backend.raise_after_create[offer] = BackendError("gateway timeout", retryable=True)
    return backend


def suspects(world: World, job_id: str) -> list[str]:
    return list(world.doc(layout.attempt_key(job_id)).get(SUSPECT_FIELD) or [])


# ================================================================ F1：结果未知的 create


@pytest.mark.parametrize("appears_after", [20.0, 60.0, 110.0])
def test_f1_instance_appearing_within_the_lookup_window_is_adopted(tmp_path, appears_after):
    world = World(tmp_path, plans=[Plan(), Plan()])
    backend = unclear(world, "o1", appears_after)
    result = world.orchestrator().submit(make_spec(tmp_path))
    # 接管晚到的实例作为这次 create 的结果：不销毁、不换机、不开第二台
    assert backend.creates() == ["o1"] and backend.destroyed == []
    assert result.instance.id == "1" and result.attempt == 1 and result.phase == states.RUNNING
    assert world.events(result.job_id, states.EV_RELAUNCH) == []
    assert suspects(world, result.job_id) == []
    (event,) = world.events(result.job_id, states.EV_CREATE_FAILED)
    assert event["data"]["adopted_instance"] == "1"
    assert "接管" in world.text()


def test_f1_lookup_window_is_configurable_and_counts_from_the_create_call(tmp_path):
    config = make_config(create_lookup_s=0)
    world = World(tmp_path, plans=[Plan(runner="none"), Plan()], config=config)
    backend = unclear(world, "o1", 20.0)
    result = world.orchestrator().submit(make_spec(tmp_path))
    label = layout.label(result.job_id, 1)
    assert [c for c in backend.calls if c == ("list", label)][:1] == [("list", label)]
    assert result.attempt == 2 and backend.creates() == ["o1", "o2"]  # 窗口为 0：只查一次


def test_f1_instance_appearing_after_the_window_is_found_by_the_recheck(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="none"), Plan()])
    backend = unclear(world, "o1", 150.0)
    result = world.orchestrator().submit(make_spec(tmp_path))
    job_id = result.job_id
    late = layout.label(job_id, 1)
    lookups = [c for c in backend.calls if c == ("list", late)]
    assert len(lookups) >= lookups_in(world.config.create_lookup_s)  # 查满了窗口才换机
    assert backend.creates() == ["o1", "o2"] and result.attempt == 2
    # submit 返回前再查一次：晚到的实例出现了，销毁，不再列为可疑
    assert backend.destroyed == ["1"] and world.alive() == ["2"]
    assert suspects(world, job_id) == []
    assert any("晚到了，已销毁" in line for line in world.messages)


def test_f1_instance_still_missing_is_announced_persisted_and_collected_later(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="none"), Plan(run_s=30 * MIN)])
    backend = unclear(world, "o1", 600.0)
    result = world.orchestrator().submit(make_spec(tmp_path))
    job_id = result.job_id
    late = layout.label(job_id, 1)
    # 大声提示：标签、gc、10 分钟
    loud = [line for line in world.messages if line.startswith("!!") and late in line]
    assert loud and "dgpu gc" in loud[0] and "10 分钟后" in loud[0]
    assert suspects(world, job_id) == [late]
    unknown = [
        e
        for e in world.events(job_id, states.EV_CREATE_FAILED)
        if e["data"].get("why") == "create_outcome_unknown"
    ]
    assert unknown and unknown[0]["data"]["suspect_labels"] == [late]
    # dgpu status 显示可疑标签
    view = world.orchestrator().status(job_id)
    assert view.suspect_labels == [late]
    assert late in _format_view(view)
    # 实例晚到了；dgpu wait 结束时的收尾销毁它，并说一声
    world.messages.clear()
    assert world.orchestrator().wait(job_id) == states.SUCCEEDED
    assert "1" not in world.alive() and "1" in backend.destroyed
    assert any("已销毁晚到的实例" in line and late in line for line in world.messages)
    assert suspects(world, job_id) == []


def test_f1_gc_destroys_a_late_instance_and_says_so(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="none"), Plan()])
    backend = unclear(world, "o1", 900.0)
    job_id = world.orchestrator().submit(make_spec(tmp_path)).job_id
    late = layout.label(job_id, 1)
    assert suspects(world, job_id) == [late]
    world.now += 15 * MIN
    report = world.orchestrator().gc()
    (item,) = [item for item in report.destroyed if item.label == late]
    assert "晚到" in item.reason
    assert "1" in backend.destroyed and suspects(world, job_id) == []


def test_f1_signal_right_after_an_unclear_create_still_reports_the_late_instance(tmp_path):
    """评审复现：o1 结果未知，换下一台之前收到 SIGTERM。可疑标签要再查、提示 gc、记下来。"""
    world = World(tmp_path, plans=[Plan(runner="none"), Plan()])
    unclear(world, "o1", 300.0)
    real_delete = world.store.delete
    fired: list[int] = []

    def delete(key):
        real_delete(key)
        if "/secrets-1" in key and not fired:
            fired.append(1)
            raise Terminated(signal.SIGTERM)

    world.store.delete = delete
    with pytest.raises(Terminated):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    late = layout.label(job_id, 1)
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.CANCELLED
    assert late in status["message"] and "dgpu gc" in status["message"]
    assert any(line.startswith("!!") and late in line for line in world.messages)
    assert suspects(world, job_id) == [late]
    world.now += 400
    world.orchestrator().gc()
    assert world.alive() == []


def test_f1_abort_waits_the_window_and_a_second_ctrl_c_skips_it(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="none")])
    backend = use_backend(world, LaggyBackend(world, hidden={"o1": 600.0}))
    backend.raise_after_create["o1"] = KeyboardInterrupt()  # create 还没返回就 Ctrl-C
    pressed: list[int] = []
    during_wait: dict = {}

    def second_ctrl_c(w: World) -> None:
        if not pressed and any("再按一次 Ctrl-C" in line for line in w.messages):
            pressed.append(w.sleeps)
            # 等待期间（比如被 SIGKILL）：启动文档已经删了，标签已经记下了
            job = only_job_id(w)
            during_wait["secrets"] = [k for k in w.store.objects if "/secrets-" in k]
            during_wait["suspects"] = suspects(w, job)
            raise KeyboardInterrupt()

    world.on_sleep.append(second_ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    late = layout.label(job_id, 1)
    assert during_wait == {"secrets": [], "suspects": [late]}
    lookups = [c for c in backend.calls if c == ("list", late)]
    assert pressed and len(lookups) < lookups_in(world.config.create_lookup_s)  # 没等满
    assert any("跳过等待" in line for line in world.messages)
    loud = [line for line in world.messages if line.startswith("!!") and late in line]
    assert loud and "dgpu gc" in loud[0]
    assert suspects(world, job_id) == [late]
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.CANCELLED and "dgpu gc" in status["message"]
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 2  # 作废可能晚到的 attempt
    assert not any("/secrets-" in key for key in world.store.objects)


def test_f1_abort_finds_an_instance_that_appears_within_the_window(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="none")])
    backend = use_backend(world, LaggyBackend(world, hidden={"o1": 100.0}))
    backend.raise_after_create["o1"] = Terminated(signal.SIGTERM)
    with pytest.raises(Terminated):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert backend.destroyed == ["1"] and world.alive() == []
    status = world.doc(layout.status_key(job_id))
    assert "已销毁实例 1" in status["message"] and "结果未知" not in status["message"]
    assert suspects(world, job_id) == []


# ================================================================ F3：平台明确拒绝


def test_f3_definite_rejection_is_reported_without_lookups_or_gc_hint(tmp_path):
    world = World(tmp_path)

    def reject(offer, launch):
        world.backend.calls.append(("create", offer.id))
        raise BackendError(
            "Vast 拒绝创建实例（offer o1，HTTP 400: invalid_args / bad image）", created=False
        )

    world.backend.create = reject
    sleeps_before_abort: list[int] = []
    with pytest.raises(DgpuError) as info:
        world.orchestrator().submit(make_spec(tmp_path))
    sleeps_before_abort.append(world.sleeps)
    job_id = only_job_id(world)
    assert "bad image" in str(info.value) and "dgpu gc" not in str(info.value)
    assert not [c for c in world.backend.calls if c[0] == "list"]  # 不按标签查找
    assert world.sleeps == 0  # 不等 17 秒、也不等 2 分钟
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.FAILED and "bad image" in status["message"]
    assert "结果未知" not in status["message"] and "dgpu gc" not in status["message"]
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 1
    assert suspects(world, job_id) == []
    assert not any("dgpu gc" in line for line in world.messages)


def test_f3_retryable_rejection_moves_to_the_next_offer_without_a_new_attempt(tmp_path):
    world = World(tmp_path)
    real_create = world.backend.create

    def create(offer, launch):
        if offer.id == "o1":
            world.backend.calls.append(("create", offer.id))
            raise BackendError("rate limited", retryable=True, created=False)
        return real_create(offer, launch)

    world.backend.create = create
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert result.offer.id == "o2" and result.attempt == 1
    assert world.events(result.job_id, states.EV_RELAUNCH) == []
    assert not [c for c in world.backend.calls if c[0] == "list" and c[1].endswith("-a1")]


# ================================================================ F4：建好状态之后、换机循环之前


@pytest.mark.parametrize(
    ("raised", "phase"),
    [(Terminated(signal.SIGTERM), states.CANCELLED), (OSError("disk full"), states.FAILED)],
    ids=["sigterm", "error"],
)
def test_f4_interruption_right_after_the_status_is_created_writes_a_terminal_status(
    tmp_path, raised, phase
):
    world = World(tmp_path, plans=[Plan(run_s=60)])
    real = world.store.put_json
    fired: list[int] = []

    def put_json(key, obj, **kw):
        out = real(key, obj, **kw)
        if key.endswith("/status.json") and not fired:
            fired.append(1)
            raise raised
        return out

    world.store.put_json = put_json
    with pytest.raises(type(raised)):
        world.orchestrator().submit(make_spec(tmp_path))
    world.store.put_json = real
    job_id = only_job_id(world)
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == phase and world.backend.creates() == []
    # 一小时后也不会变成 LOST，wait --retries 不会为用户中断的任务开机器
    world.now += 3600
    assert world.orchestrator().status(job_id).phase == phase
    assert world.orchestrator().wait(job_id, retries=1) == phase
    assert world.backend.creates() == []


def test_f4_retry_interrupted_before_its_first_create_is_cancelled(tmp_path):
    world = World(
        tmp_path,
        plans=[Plan(run_s=60, final_phase=states.FAILED, final_reason=states.REASON_EXIT_CODE)],
    )
    job_id = world.orchestrator().submit(make_spec(tmp_path)).job_id
    world.sleep(5 * MIN)
    real = world.store.put_json

    def put_json(key, obj, **kw):
        if key.endswith("/status.json") and obj.get("phase") == states.PROVISIONING:
            raise Terminated(signal.SIGHUP)
        return real(key, obj, **kw)

    world.store.put_json = put_json
    with pytest.raises(Terminated):
        world.orchestrator().retry(job_id)
    world.store.put_json = real
    status = world.doc(layout.status_key(job_id))
    assert (status["attempt"], status["phase"]) == (2, states.CANCELLED)
    assert world.backend.creates() == ["o1"]


# ================================================================ F5：确认失联后换机失败


def test_f5_lost_instance_is_destroyed_even_when_the_relaunch_fails(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=2 * MIN, die_platform_phase=INST_EXITED)])
    job_id = world.orchestrator().submit(make_spec(tmp_path)).job_id
    world.backend.offers = []  # 换机时没有机器
    assert world.orchestrator().wait(job_id, retries=1) == states.LOST
    assert world.alive() == []  # 已退出的实例（仍按磁盘计费）被销毁
    assert "1" in world.backend.destroyed


def test_f5_failing_to_destroy_the_lost_instance_prints_a_gc_hint(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=2 * MIN, die_platform_phase=INST_EXITED)])
    job_id = world.orchestrator().submit(make_spec(tmp_path)).job_id
    world.backend.offers = []
    world.backend.destroy_errors["1"] = BackendError("platform API down")
    assert world.orchestrator().wait(job_id, retries=1) == states.LOST
    loud = [line for line in world.messages if line.startswith("!!") and "dgpu gc" in line]
    assert loud
    assert world.events(job_id, states.EV_DESTROY_FAILED)


# ================================================================ F7：降权没生效的原因看得见


@pytest.mark.parametrize("dropped", [False, True])
def test_f7_status_shows_whether_the_training_runs_as_root_and_why(tmp_path, dropped):
    world = World(tmp_path)
    job = "demo-260101000000-abcd"
    seed_job(world.store, job, phase=states.RUNNING, hb_at=T0)
    key = layout.heartbeat_key(job)
    hb = world.doc(key)
    hb["identity"] = (
        {"dropped": True, "user": "dgpu", "uid": 1001, "reason": None, "detail": None}
        if dropped
        else {
            "dropped": False,
            "user": None,
            "uid": None,
            "reason": "pip_not_writable",
            "reason_text": "pip 没有可写的安装位置",
            "detail": "python3（/venv/main）的 site-packages 不可写",
        }
    )
    world.store.put_json(key, hb)
    view = world.orchestrator().status(job)
    text = _format_view(view)
    if dropped:
        assert "以无特权用户 dgpu（uid 1001）运行" in text
    else:
        assert "没有降权" in text and "pip_not_writable" in text and "/venv/main" in text
        assert view.to_dict()["identity"]["reason"] == "pip_not_writable"
