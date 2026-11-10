"""提交之后：status 推导、wait、retry、cancel、logs、pull、list。"""

from __future__ import annotations

import copy

import pytest

from dgpu import layout, states
from dgpu.contracts import BackendError, DgpuError
from test_orchestrator_support import (
    T0,
    HookedStore,
    Plan,
    World,
    make_spec,
    seed_job,
)

JOB = "demo-260101000000-abcd"
MIN = 60.0


def submit(world: World, tmp_path) -> str:
    return world.orchestrator().submit(make_spec(tmp_path)).job_id


def advance_until(world: World, predicate, *, step: float = 30.0, limit: float = 6 * 3600):
    start = world.now
    while not predicate():
        assert world.now - start < limit, "simulated time limit reached"
        world.sleep(step)


# ---------------------------------------------------------------- status 与 LOST 推导


@pytest.mark.parametrize(
    ("phase", "updated_ago", "hb_ago", "expected"),
    [
        (states.RUNNING, 1 * MIN, 5 * MIN, states.RUNNING),
        (states.RUNNING, 1 * MIN, 11 * MIN, states.LOST),
        (states.UPLOADING, 1 * MIN, 11 * MIN, states.LOST),
        (states.STARTING, 20 * MIN, None, states.STARTING),  # 还在启动窗口内（15 + 5 + 10）
        (states.STARTING, 31 * MIN, None, states.LOST),
        (states.PROVISIONING, 31 * MIN, None, states.LOST),
        (states.SUCCEEDED, 1 * MIN, 600 * MIN, states.SUCCEEDED),
        (states.FAILED, 1 * MIN, None, states.FAILED),
    ],
)
def test_status_derives_lost_from_stale_heartbeat(tmp_path, phase, updated_ago, hb_ago, expected):
    world = World(tmp_path)
    world.now = T0 + 1000 * MIN
    seed_job(
        world.store,
        JOB,
        phase=phase,
        updated_at=world.now - updated_ago,
        hb_at=None if hb_ago is None else world.now - hb_ago,
    )
    before = copy.deepcopy(world.store.objects)
    view = world.orchestrator().status(JOB)
    assert view.phase == expected
    assert view.lost == (expected == states.LOST)
    assert world.store.objects == before  # status 从不写存储


def test_status_ignores_heartbeat_of_an_older_attempt(tmp_path):
    world = World(tmp_path)
    world.now = T0 + 5 * MIN
    seed_job(world.store, JOB, attempt=2, phase=states.STARTING, hb_attempt=1, hb_at=world.now)
    view = world.orchestrator().status(JOB)
    assert view.phase == states.STARTING and view.heartbeat_at is None


def test_status_after_attempt_bump_shows_provisioning(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, attempt=2, status_attempt=1, phase=states.FAILED, hb_at=None)
    view = world.orchestrator().status(JOB)
    assert view.phase == states.PROVISIONING and view.attempt == 2


def test_status_of_unknown_or_invalid_job(tmp_path):
    world = World(tmp_path)
    with pytest.raises(DgpuError, match="找不到任务"):
        world.orchestrator().status(JOB)
    with pytest.raises(DgpuError, match="格式不对"):
        world.orchestrator().status("../etc")
    world.store.put_json(layout.spec_key(JOB), {"version": 1})
    with pytest.raises(DgpuError, match="提交到一半"):
        world.orchestrator().status(JOB)


def test_status_rejects_unknown_document_version(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB)
    doc, _ = world.store.get_json(layout.status_key(JOB))
    world.store.put_json(layout.status_key(JOB), {**doc, "version": 99})
    with pytest.raises(DgpuError, match="升级 dgpu"):
        world.orchestrator().status(JOB)


# ---------------------------------------------------------------- wait


def test_wait_returns_final_phase_and_removes_leftover_instance(tmp_path):
    # AutoDL 式：runner 结束后实例停在 exited，CLI 负责删除部署记录
    world = World(tmp_path, plans=[Plan(run_s=10 * MIN, self_destroy=False)])
    job_id = submit(world, tmp_path)
    assert world.orchestrator().wait(job_id) == states.SUCCEEDED
    assert world.alive() == []
    (event,) = world.events(job_id, states.EV_INSTANCE_DESTROYED)
    assert event["data"]["instance_id"] == "1"


@pytest.mark.parametrize(
    "final", [(states.FAILED, states.REASON_EXIT_CODE), (states.TIMED_OUT, None)]
)
def test_wait_reports_failures(tmp_path, final):
    world = World(
        tmp_path, plans=[Plan(run_s=5 * MIN, final_phase=final[0], final_reason=final[1])]
    )
    job_id = submit(world, tmp_path)
    assert world.orchestrator().wait(job_id) == final[0]


def test_wait_without_retries_returns_lost(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=2 * MIN)])
    job_id = submit(world, tmp_path)
    assert world.orchestrator().wait(job_id) == states.LOST
    assert "dgpu retry" in world.text()
    assert world.alive() == ["1"]  # 可能只是和存储断连，交给 retry / gc，不擅自销毁


def test_wait_retries_relaunch_with_next_attempt(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=2 * MIN), Plan(run_s=10 * MIN)])
    job_id = submit(world, tmp_path)
    assert world.orchestrator().wait(job_id, retries=1) == states.SUCCEEDED

    assert world.backend.destroyed[0] == "1"  # 先确认旧实例销毁，再开新的
    assert world.backend.creates() == ["o1", "o1"]
    assert world.backend.launches["2"].attempt == 2
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 2
    fenced = world.events(job_id, states.EV_ATTEMPT_FENCED)
    assert [(e["data"]["old_attempt"], e["data"]["new_attempt"]) for e in fenced] == [(1, 2)]
    destroyed = [e["data"]["instance_id"] for e in world.events(job_id, "instance_destroyed")]
    assert destroyed[0] == "1"


def test_wait_retries_exhausted_returns_lost(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=60), Plan(die_after_s=60)])
    job_id = submit(world, tmp_path)
    assert world.orchestrator().wait(job_id, retries=1) == states.LOST
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 2


# ---------------------------------------------------------------- retry


def make_lost_job(world: World, tmp_path) -> str:
    job_id = submit(world, tmp_path)
    orch = world.orchestrator()
    advance_until(world, lambda: orch.status(job_id).phase == states.LOST)
    return job_id


def test_retry_refuses_when_old_instance_cannot_be_destroyed(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=60)])
    job_id = make_lost_job(world, tmp_path)
    world.backend.destroy_errors["1"] = BackendError("platform API down")
    with pytest.raises(DgpuError) as info:
        world.orchestrator().retry(job_id)
    assert "vast/1" in str(info.value)
    assert world.backend.creates() == ["o1"]  # 没有启动第二台
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 1


def test_retry_refuses_when_destroy_is_not_confirmed(tmp_path):
    world = World(tmp_path, plans=[Plan(die_after_s=60)])
    job_id = make_lost_job(world, tmp_path)
    world.backend.destroy = lambda ref: world.backend.calls.append(("destroy", ref.id))
    with pytest.raises(DgpuError, match="仍是 running"):
        world.orchestrator().retry(job_id)
    assert world.backend.creates() == ["o1"]
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 1


def test_retry_destroys_stray_instances_found_by_label(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.FAILED, hb_at=None)  # attempt.json 没记实例 ID
    world.backend.add_instance("77", layout.label(JOB, 1), created_at=T0)
    result = world.orchestrator().retry(JOB)
    assert "77" in world.backend.destroyed
    assert result.attempt == 2


@pytest.mark.parametrize("phase", [states.RUNNING, states.SUCCEEDED, states.STARTING])
def test_retry_refuses_active_or_succeeded_jobs(tmp_path, phase):
    world = World(tmp_path)
    world.now = T0 + MIN
    seed_job(world.store, JOB, phase=phase)
    with pytest.raises(DgpuError, match="只有 LOST、FAILED"):
        world.orchestrator().retry(JOB)
    assert world.backend.creates() == []


def test_two_concurrent_retries_only_one_creates_an_instance(tmp_path):
    store = HookedStore()
    world = World(tmp_path, store=store)
    seed_job(store, JOB, phase=states.FAILED, reason=states.REASON_EXIT_CODE, hb_at=None)
    first, second = world.orchestrator(), world.orchestrator()
    raced = []

    def race(key, obj, if_match):
        # 第一个 retry 正要条件写 attempt 2 的瞬间，第二个 retry 完整跑完
        if key == layout.attempt_key(JOB) and if_match and obj.get("attempt") == 2 and not raced:
            raced.append(True)
            store.before_put = None
            second.retry(JOB)

    store.before_put = race
    with pytest.raises(DgpuError, match="正在被另一个 dgpu 进程重试"):
        first.retry(JOB)
    assert raced
    assert world.backend.creates() == ["o1"]  # 只有赢家创建了实例
    assert world.doc(layout.attempt_key(JOB))["attempt"] == 2
    assert world.alive() == ["1"]


# ---------------------------------------------------------------- cancel


def test_cancel_lets_runner_wind_down(tmp_path):
    world = World(tmp_path)
    job_id = submit(world, tmp_path)
    view = world.orchestrator().cancel(job_id)
    assert view.cancel_requested
    assert world.alive() == ["1"]  # 不直接销毁，由 runner 收尾
    assert world.orchestrator().wait(job_id) == states.CANCELLED
    assert world.alive() == []


def test_cancel_force_destroys_and_marks_cancelled(tmp_path):
    world = World(tmp_path, plans=[Plan(runner="ok")])
    job_id = submit(world, tmp_path)
    view = world.orchestrator().cancel(job_id, force=True)
    assert view.phase == states.CANCELLED
    assert world.alive() == []
    assert world.doc(layout.status_key(job_id))["source"] == states.SRC_CLI
    assert world.events(job_id, states.EV_INSTANCE_DESTROYED)


def test_cancel_retries_conditional_write_on_conflict(tmp_path):
    store = HookedStore()
    world = World(tmp_path, store=store)
    seed_job(store, JOB, instance_id="5")
    conflicts = []

    def concurrent_writer(key, obj, if_match):
        if key == layout.attempt_key(JOB) and obj.get("cancel") and not conflicts:
            conflicts.append(True)
            doc, etag = store.get_json(key)
            store.before_put = None
            store.put_json(key, {**doc, "deadline_at": 123.0}, if_match=etag)
            store.before_put = concurrent_writer

    store.before_put = concurrent_writer
    world.orchestrator().cancel(JOB)
    doc = world.doc(layout.attempt_key(JOB))
    assert conflicts
    assert doc["cancel"] is True and doc["deadline_at"] == 123.0  # 两次写都没丢


def test_cancel_finished_job_is_a_noop(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.SUCCEEDED)
    before = copy.deepcopy(world.store.objects)
    world.orchestrator().cancel(JOB)
    assert world.store.objects == before
    assert "已经结束" in world.text()


def test_cancel_lost_job_hints_force(tmp_path):
    world = World(tmp_path)
    world.now = T0 + 60 * MIN
    seed_job(world.store, JOB, phase=states.RUNNING, hb_at=T0)
    world.orchestrator().cancel(JOB)
    assert "--force" in world.text()


# ---------------------------------------------------------------- logs / pull / list


def test_logs_follow_reads_incrementally_until_terminal(tmp_path):
    store = HookedStore()
    world = World(tmp_path, store=store, plans=[Plan(log=True, run_s=60)])
    job_id = submit(world, tmp_path)
    lines: list[str] = []
    world.orchestrator().logs(job_id, follow=True, out=lines.append)

    expected = [f"第 {n} 行 epoch" for n in range(1, len(lines) + 1)]
    assert lines == expected and len(lines) >= 5  # 完整、有序、不重复
    key = layout.log_key(job_id, 1)
    starts = [start for read_key, start in store.byte_reads if read_key == key]
    assert len([s for s in starts if s > 0]) >= 3  # 用范围读取，而不是每次读整个文件
    assert starts == sorted(starts)
    assert "停止跟随" in world.text()


def test_logs_keep_multibyte_characters_split_across_reads(tmp_path):
    world = World(tmp_path)
    world.now = T0 + MIN
    seed_job(world.store, JOB, hb_at=world.now)
    key = layout.log_key(JOB, 1)
    payload = "你好，世界\n".encode()
    world.store.put_bytes(key, payload[:4])  # 在“好”字中间截断

    def finish(w: World) -> None:
        w.store.put_bytes(key, payload)
        doc, etag = w.store.get_json(layout.status_key(JOB))
        w.store.put_json(layout.status_key(JOB), {**doc, "phase": states.SUCCEEDED})

    world.on_sleep.append(finish)
    lines: list[str] = []
    world.orchestrator().logs(JOB, follow=True, out=lines.append)
    assert lines == ["你好，世界"]


def test_logs_without_follow(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.SUCCEEDED)
    lines: list[str] = []
    world.orchestrator().logs(JOB, out=lines.append)
    assert lines == [] and "还没有日志" in world.text()
    world.store.put_bytes(layout.log_key(JOB, 1), b"a\nb\nno newline")
    world.orchestrator().logs(JOB, out=lines.append)
    assert lines == ["a", "b", "no newline"]


def test_pull_downloads_output_and_optional_ckpt(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.SUCCEEDED)
    world.store.put_bytes(layout.output_prefix(JOB) + "run/weights/best.pt", b"best")
    world.store.put_bytes(layout.ckpt_prefix(JOB) + "last.pt", b"last")

    result = world.orchestrator().pull(JOB, tmp_path / "out")
    assert result.output_files == 1 and result.ckpt_files == 0
    assert (tmp_path / "out/output/run/weights/best.pt").read_bytes() == b"best"
    assert not (tmp_path / "out/ckpt").exists()
    assert "警告" not in world.text()

    result = world.orchestrator().pull(JOB, tmp_path / "out2", include_ckpt=True)
    assert (tmp_path / "out2/ckpt/last.pt").read_bytes() == b"last"


def test_pull_warns_when_job_did_not_succeed(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.FAILED, reason=states.REASON_FINAL_SYNC)
    doc, _ = world.store.get_json(layout.status_key(JOB))
    world.store.put_json(layout.status_key(JOB), {**doc, "not_uploaded": ["output/big.bin"]})
    world.orchestrator().pull(JOB, tmp_path / "out")
    assert "不是 SUCCEEDED" in world.text()
    assert "output/big.bin" in world.text()
    assert "没有文件" in world.text()


def test_list_jobs_newest_first(tmp_path):
    world = World(tmp_path)
    world.now = T0 + MIN
    old, mid, new = (f"demo-2601010{n}0000-abcd" for n in (1, 2, 3))
    seed_job(world.store, old, phase=states.SUCCEEDED)
    seed_job(world.store, new, phase=states.RUNNING, hb_at=world.now)
    seed_job(world.store, mid, phase=states.FAILED)
    jobs = world.orchestrator().list_jobs()
    assert [job.job_id for job in jobs] == [new, mid, old]
    assert [job.phase for job in jobs] == [states.RUNNING, states.FAILED, states.SUCCEEDED]
    assert [job.job_id for job in world.orchestrator().list_jobs(limit=1)] == [new]
