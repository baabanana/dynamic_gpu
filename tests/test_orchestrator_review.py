"""评审（docs/dgpu-review-2026-11-10.md）A 节每一条的回归测试，以及“需要真实平台确认的怀疑”
里能在本地防住的部分。每个测试在修复之前都会失败。

runner 和平台由 test_orchestrator_support.World 模拟，时间由注入的 sleep 推进。
CLI 一层（信号、退出码、参数）见 test_cli.py；真实进程的 SIGTERM 见 test_integration_e2e.py。
"""

from __future__ import annotations

import math
import signal
from dataclasses import replace

import pytest

from dgpu import layout, orchestrator, states
from dgpu.contracts import (
    INST_GONE,
    INST_UNKNOWN,
    BackendError,
    DataRef,
    DgpuError,
    InstanceRef,
    InstanceState,
    StoreError,
)
from dgpu.orchestrator import (
    CREATE_UNKNOWN_LOOKUP_DELAYS_S,
    StoreUnreachable,
    Terminated,
)
from dgpu.packaging import PackResult
from dgpu.runner import util as runner_util
from dgpu.testing import MemoryStore
from test_orchestrator_support import (
    T0,
    HookedStore,
    Plan,
    SimBackend,
    World,
    make_config,
    make_spec,
    seed_job,
)

JOB = "demo-260101000000-abcd"
MIN = 60.0


def submit(world: World, tmp_path, **kwargs) -> str:
    return world.orchestrator().submit(make_spec(tmp_path), **kwargs).job_id


def only_job_id(world: World) -> str:
    (job_id,) = world.store.list_dirs(layout.jobs_prefix())
    return job_id


def teardown_s(world: World) -> float:
    settings = runner_util.runner_settings({"runner": world.config.runner.to_dict()})
    return runner_util.teardown_worst_case_s(settings, runner_util.DEFAULT_TUNING)


class LaggyBackend(SimBackend):
    """平台接口有延迟：在某些 offer 上新建的实例，要过 hidden[offer] 秒才查得到（get 和 list）。"""

    def __init__(self, world: World, hidden: dict[str, float], **kwargs):
        super().__init__(world, world.backend.name, list(world.backend.offers), **kwargs)
        self.hidden = hidden

    def _visible(self, instance_id: str) -> bool:
        state = self.instances.get(instance_id)
        if state is None:
            return False
        delay = self.hidden.get(self.offer_of.get(instance_id, ""), 0.0)
        return state.created_at is None or self.world.now - state.created_at >= delay

    def get(self, ref: InstanceRef) -> InstanceState:
        self.calls.append(("get", ref.id))
        if not self._visible(ref.id):
            return InstanceState(ref=ref, phase=INST_GONE, raw_status="not found")
        return self.instances[ref.id]

    def list(self, label_prefix: str = layout.LABEL_PREFIX) -> list[InstanceState]:
        return [state for state in super().list(label_prefix) if self._visible(state.ref.id)]


def use_backend(world: World, backend: SimBackend) -> SimBackend:
    world.backend = backend
    return backend


# ================================================================ A1：SIGTERM / SIGHUP


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a1_termination_signal_during_startup_takes_the_ctrl_c_path(tmp_path, signum):
    world = World(tmp_path, plans=[Plan(boot="stuck")])

    def terminate(w: World) -> None:
        if w.sleeps == 3:
            raise Terminated(signum)

    world.on_sleep.append(terminate)
    with pytest.raises(Terminated) as info:
        world.orchestrator().submit(make_spec(tmp_path))
    assert info.value.exit_code == 128 + signum
    assert world.backend.creates() == ["o1"] and world.alive() == []
    job_id = only_job_id(world)
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.CANCELLED
    assert signal.Signals(signum).name in status["message"] and "已销毁实例 1" in status["message"]
    assert not any("/secrets-" in key for key in world.store.objects)


def test_a1_cancel_force_finishes_the_destroy_when_terminated(tmp_path):
    world = World(tmp_path)
    job_id = submit(world, tmp_path)
    real_destroy = world.backend.destroy
    calls = []

    def destroy_interrupted_once(ref):
        calls.append(ref.id)
        if len(calls) == 1:
            raise Terminated(signal.SIGTERM)
        real_destroy(ref)

    world.backend.destroy = destroy_interrupted_once
    with pytest.raises(Terminated):
        world.orchestrator().cancel(job_id, force=True)
    assert world.alive() == []  # 收到 SIGTERM 也先把销毁做完
    status = world.doc(layout.status_key(job_id))
    assert (status["phase"], status["source"]) == (states.CANCELLED, states.SRC_CLI)


# ================================================================ A2：create 结果未知


@pytest.mark.parametrize(
    "interrupt", [KeyboardInterrupt(), Terminated(signal.SIGTERM)], ids=["ctrl-c", "sigterm"]
)
def test_a2_unknown_create_outcome_is_reported_honestly_and_the_late_instance_is_collected(
    tmp_path, interrupt
):
    world = World(tmp_path, plans=[Plan(runner="none")])  # 晚到的实例上 runner 没起来
    backend = use_backend(world, LaggyBackend(world, hidden={"o1": 600.0}))
    backend.raise_after_create["o1"] = interrupt  # 创建请求还没返回就被打断
    with pytest.raises(KeyboardInterrupt):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    label = layout.label(job_id, 1)

    # 按标签查了好几次（有间隔），而不是查一次就下结论
    lookups = [call for call in backend.calls if call == ("list", label)]
    assert len(lookups) == len(CREATE_UNKNOWN_LOOKUP_DELAYS_S)
    status = world.doc(layout.status_key(job_id))
    assert status["phase"] == states.CANCELLED
    assert "结果未知" in status["message"] and label in status["message"]
    assert "已销毁" not in status["message"]  # 绝不声称已销毁
    assert "分钟后运行 `dgpu gc`" in status["message"]
    loud = [line for line in world.messages if line.startswith("!!") and label in line]
    assert loud and "dgpu gc" in loud[0]
    # attempt 1 被作废：晚到实例的 runner 会被隔离，gc 把它当旧 attempt
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 2
    assert status["attempt"] == 2
    (relaunch,) = world.events(job_id, states.EV_RELAUNCH)
    assert relaunch["data"]["why"] == "create_unknown"

    world.now += 700  # 实例终于出现在平台的列表里
    report = world.orchestrator().gc()
    assert world.backend.destroyed == ["1"]
    assert "旧 attempt" in report.destroyed[0].reason


def test_a2_instance_found_by_a_later_lookup_is_destroyed(tmp_path):
    world = World(tmp_path)
    backend = use_backend(world, LaggyBackend(world, hidden={"o1": 4.0}))
    backend.raise_after_create["o1"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        world.orchestrator().submit(make_spec(tmp_path))
    job_id = only_job_id(world)
    assert world.backend.destroyed == ["1"] and world.alive() == []
    status = world.doc(layout.status_key(job_id))
    assert "已销毁实例 1" in status["message"] and "结果未知" not in status["message"]
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 1


# ================================================================ A3：取消优先


def test_a3_wait_does_not_auto_retry_a_job_with_a_pending_cancel(tmp_path):
    world = World(
        tmp_path, plans=[Plan(die_after_s=2 * MIN, die_platform_phase=INST_UNKNOWN), Plan()]
    )
    job_id = submit(world, tmp_path)
    world.sleep(5 * MIN)  # 宿主机宕机
    world.orchestrator().cancel(job_id)  # 取消标志设置了，但 runner 已经看不到
    assert world.orchestrator().wait(job_id, retries=1) == states.LOST
    assert world.backend.creates() == ["o1"]  # 没有重新开机器
    attempt = world.doc(layout.attempt_key(job_id))
    assert attempt["attempt"] == 1 and attempt["cancel"] is True
    assert "cancel --force" in world.text()


def test_a3_retry_refuses_while_a_cancel_is_pending(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.RUNNING, hb_at=T0 - 3600, updated_at=T0 - 3600)
    orch = world.orchestrator()
    orch.cancel(JOB)
    assert orch.status(JOB).phase == states.LOST
    with pytest.raises(DgpuError, match="未完成的取消请求"):
        orch.retry(JOB)
    assert world.backend.calls == []  # 没有销毁、没有搜索、没有创建
    assert world.doc(layout.attempt_key(JOB))["cancel"] is True


def test_a3_explicit_retry_of_a_cancelled_job_clears_the_flag(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.CANCELLED, hb_at=None)
    doc, etag = world.store.get_json(layout.attempt_key(JOB))
    world.store.put_json(layout.attempt_key(JOB), {**doc, "cancel": True}, if_match=etag)
    result = world.orchestrator().retry(JOB)
    assert result.attempt == 2
    assert world.doc(layout.attempt_key(JOB))["cancel"] is False


def test_a3_retry_gives_up_if_a_cancel_arrives_after_its_check(tmp_path):
    store = HookedStore()
    world = World(tmp_path, store=store)
    seed_job(store, JOB, phase=states.FAILED, reason=states.REASON_EXIT_CODE, hb_at=None)

    def cancel_concurrently(key, obj, if_match):
        if key == layout.attempt_key(JOB) and obj.get("attempt") == 2:
            store.before_put = None
            doc, etag = store.get_json(key)
            store.put_json(key, {**doc, "cancel": True}, if_match=etag)

    store.before_put = cancel_concurrently
    with pytest.raises(DgpuError, match="刚被取消"):
        world.orchestrator().retry(JOB)
    assert world.backend.creates() == []
    doc = world.doc(layout.attempt_key(JOB))
    assert doc["attempt"] == 1 and doc["cancel"] is True


# ======================================================= A5：结果不明的 create 之后换 attempt


def test_a5_unclear_create_bumps_the_attempt_so_a_late_instance_is_fenced(tmp_path):
    # o1：平台其实创建了实例，但列表要 10 分钟后才显示；后端按标签没查到，报“可重试”
    world = World(tmp_path, plans=[Plan(runner="none"), Plan()])
    backend = use_backend(world, LaggyBackend(world, hidden={"o1": 600.0}))
    backend.raise_after_create["o1"] = BackendError("gateway timeout", retryable=True)
    result = world.orchestrator().submit(make_spec(tmp_path))
    job_id = result.job_id
    assert result.offer.id == "o2" and result.attempt == 2
    assert backend.launches["2"].label == layout.label(job_id, 2)  # 新标签，不和晚到的撞
    (relaunch,) = world.events(job_id, states.EV_RELAUNCH)
    assert (relaunch["data"]["old_attempt"], relaunch["data"]["new_attempt"]) == (1, 2)

    world.now += 700  # 晚到的 attempt 1 实例出现了
    report = world.orchestrator().gc()
    assert world.backend.destroyed == ["1"]
    assert "旧 attempt" in report.destroyed[0].reason
    assert "2" in world.alive()


def test_a5_clean_rejection_keeps_the_attempt(tmp_path):
    world = World(tmp_path)
    world.backend.unavailable_offer_ids = {"o1"}
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert result.attempt == 1 and world.events(result.job_id, states.EV_RELAUNCH) == []


# ================================================================ A6：销毁带上依据的 attempt


def test_a6_wait_cleanup_does_not_destroy_the_instance_of_a_concurrent_retry(tmp_path):
    world = World(
        tmp_path,
        plans=[
            Plan(
                run_s=5 * MIN,
                final_phase=states.FAILED,
                final_reason=states.REASON_EXIT_CODE,
                self_destroy=False,  # AutoDL 式：结束后实例停在 exited，等 CLI 删除
            ),
            Plan(),
        ],
    )
    job_id = submit(world, tmp_path)
    waiter, racer = world.orchestrator(), world.orchestrator()
    real_status = waiter.status
    raced: list[bool] = []

    def status_then_race(job: str):
        view = real_status(job)
        if view.phase == states.FAILED and not raced:
            raced.append(True)
            racer.retry(job)  # wait 看到 FAILED 之后、收尾之前，另一个进程 retry 成功
        return view

    waiter.status = status_then_race
    assert waiter.wait(job_id) == states.FAILED
    assert raced
    assert world.alive() == ["2"]  # retry 的新实例还在
    assert world.doc(layout.attempt_key(job_id))["attempt"] == 2


def test_a6_losing_retry_does_not_destroy_the_winners_instance(tmp_path):
    world = World(tmp_path)
    seed_job(world.store, JOB, phase=states.FAILED, reason=states.REASON_EXIT_CODE, hb_at=None)
    winner, loser = world.orchestrator(), world.orchestrator()
    real_list = world.backend.list
    armed = [True]

    def list_after_winner_finished(label_prefix=layout.LABEL_PREFIX):
        if armed[0] and label_prefix == f"{layout.LABEL_PREFIX}{JOB}-a":
            armed[0] = False
            winner.retry(JOB)  # 输家读完 attempt.json、列出实例之前，赢家完整跑完
        return real_list(label_prefix)

    world.backend.list = list_after_winner_finished
    with pytest.raises(DgpuError, match="另一个 dgpu 进程"):
        loser.retry(JOB)
    assert world.alive() == ["1"]  # 赢家的实例没有被输家销毁
    assert world.backend.destroyed == []
    assert world.doc(layout.attempt_key(JOB))["attempt"] == 2


# ================================================================ A7：收尾清理失败要大声


def test_a7_cleanup_failure_after_success_is_loud_recorded_and_returned(tmp_path):
    world = World(tmp_path, plans=[Plan(run_s=5 * MIN, self_destroy=False)])
    job_id = submit(world, tmp_path)

    def list_down(label_prefix=layout.LABEL_PREFIX):
        raise BackendError("platform API down")

    world.backend.list = list_down
    outcome = world.orchestrator().wait_for(job_id)
    assert outcome.phase == states.SUCCEEDED and outcome.cleanup_problems
    label = layout.label(job_id, 1)
    loud = [line for line in world.messages if line.startswith("!!")]
    assert loud and label in loud[-1] and "dgpu gc" in loud[-1] and "platform API down" in loud[-1]
    (event,) = world.events(job_id, states.EV_DESTROY_FAILED)
    assert event["data"]["label"] == label and event["data"]["instance_id"] == "1"


def test_a7_failing_to_delete_the_launch_doc_does_not_crash_wait(tmp_path):
    class NoDeleteStore(MemoryStore):
        def delete(self, key):
            if "/secrets-" in key:
                raise StoreError("R2 unreachable", transient=True)
            super().delete(key)

    world = World(tmp_path, store=NoDeleteStore())
    seed_job(world.store, JOB, phase=states.SUCCEEDED)  # 已结束的任务，存储删不了东西
    outcome = world.orchestrator().wait_for(JOB)  # 修复前：StoreError 直接抛出（退出码 2）
    assert outcome.phase == states.SUCCEEDED
    assert any("启动文档" in problem for problem in outcome.cleanup_problems)
    assert any(line.startswith("!!") and "启动文档" in line for line in world.messages)


def test_a7_destroy_failure_during_cleanup_is_reported_as_a_problem(tmp_path):
    world = World(tmp_path, plans=[Plan(run_s=5 * MIN, self_destroy=False)])
    job_id = submit(world, tmp_path)
    world.backend.destroy_errors["1"] = BackendError("api unavailable")
    outcome = world.orchestrator().wait_for(job_id)
    assert outcome.phase == states.SUCCEEDED
    assert any("vast/1" in problem for problem in outcome.cleanup_problems)
    assert world.events(job_id, states.EV_DESTROY_FAILED)


# ================================================================ A8：本地数据只遍历一次


def test_a8_size_estimate_follows_symlinks_like_the_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "GB", 1000)  # 让 1000 字节算作 1 GB
    outside = tmp_path / "elsewhere"
    (outside / "dir").mkdir(parents=True)
    (outside / "big.bin").write_bytes(b"x" * 20000)  # 20 GB
    (outside / "dir" / "more.bin").write_bytes(b"x" * 10000)  # 10 GB
    data = tmp_path / "ds"
    data.mkdir()
    (data / "small.bin").write_bytes(b"x" * 1000)
    (data / "big.bin").symlink_to(outside / "big.bin")  # 指向文件的链接
    (data / "linked").symlink_to(outside / "dir")  # 指向目录的链接
    spec = make_spec(tmp_path, data=(DataRef("ds", local_path=data),))

    # 磁盘按跟随链接后的 31 GB 估算
    request = World(tmp_path).orchestrator().offers(spec).request
    assert request.disk_gb == math.ceil(2 * 31 + 20)

    # 大小确认用同一个数
    world = World(tmp_path, config=make_config(local_data_confirm_gb=30.0))
    asked: list[float] = []
    with pytest.raises(DgpuError, match="已取消"):
        world.orchestrator().submit(spec, confirm_upload=lambda gb: asked.append(gb) or False)
    assert asked == [pytest.approx(31.0)]

    # AutoDL 的 50 GB 检查也用它
    world = World(tmp_path, backend_name="autodl")
    autodl_spec = make_spec(tmp_path, backend="autodl", data=(DataRef("ds", local_path=data),))
    with pytest.raises(DgpuError, match="AutoDL 的数据盘固定 50 GB"):
        world.orchestrator().submit(autodl_spec)
    assert world.store.objects == {}

    # 上传复用这份清单，不再遍历
    world = World(tmp_path)
    world.orchestrator().submit(spec)
    manifest = world.data_manifests["ds"]
    assert {item.path for item in manifest.files} == {"small.bin", "big.bin", "linked/more.bin"}
    assert manifest.total_bytes == 31000


# ================================================================ A9：按存储时间判断失联


def test_a9_staleness_uses_storage_time_not_the_instance_or_laptop_clock(tmp_path):
    world = World(tmp_path)
    world.now = T0 + 3600
    # 实例的时钟慢了 2 小时：心跳内容里的 ts 很旧，但存储 30 秒前刚收到它
    seed_job(world.store, JOB, hb_at=world.now - 2 * 3600, updated_at=world.now - 3600)
    world.store.modified[layout.heartbeat_key(JOB)] = world.now - 30
    view = world.orchestrator().status(JOB)
    assert view.phase == states.RUNNING and view.heartbeat_age_s == pytest.approx(30)
    # 本机时钟快了 2 小时：不影响判断
    assert world.orchestrator(clock=lambda: world.now + 2 * 3600).status(JOB).phase == (
        states.RUNNING
    )
    # 存储里的心跳确实 11 分钟没更新：LOST
    world.store.modified[layout.heartbeat_key(JOB)] = world.now - 11 * MIN
    assert world.orchestrator().status(JOB).phase == states.LOST


def test_a9_wait_retries_does_not_destroy_an_instance_the_platform_still_runs(tmp_path):
    # runner 和存储断连（平台显示仍在运行）：心跳过期后只报告失联，等满 runner 最坏收尾时间
    world = World(tmp_path, plans=[Plan(die_after_s=2 * MIN), Plan(run_s=10 * MIN)])
    job_id = submit(world, tmp_path)
    silent_for: list[float] = []  # 第一次销毁时，存储里的心跳已经多久没更新
    real_destroy = world.backend.destroy

    def record(ref):
        silent_for.append(world.now - world.store.modified[layout.heartbeat_key(job_id)])
        real_destroy(ref)

    world.backend.destroy = record
    assert world.orchestrator().wait(job_id, retries=1) == states.SUCCEEDED
    assert "失联（实例仍在运行）" in world.text()
    assert silent_for and silent_for[0] > teardown_s(world)  # 修复前：心跳停 10 分钟就销毁


@pytest.mark.parametrize("platform_phase", [INST_UNKNOWN, "exited"])
def test_a9_wait_retries_promptly_when_the_platform_reports_the_host_down(
    tmp_path, platform_phase
):
    world = World(
        tmp_path,
        plans=[Plan(die_after_s=2 * MIN, die_platform_phase=platform_phase), Plan(run_s=10 * MIN)],
    )
    job_id = submit(world, tmp_path)
    died_at = world.sims["1"].first_hb + 2 * MIN
    destroyed_at: list[float] = []
    real_destroy = world.backend.destroy

    def record(ref):
        destroyed_at.append(world.now)
        real_destroy(ref)

    world.backend.destroy = record
    assert world.orchestrator().wait(job_id, retries=1) == states.SUCCEEDED
    lost_after = world.config.lost_after_min * MIN
    assert lost_after <= destroyed_at[0] - died_at < lost_after + 2 * MIN


# ================================================================ A10：代码包跳过的文件


def pack_result_with(skipped, links) -> PackResult:
    return PackResult(
        path=None,  # type: ignore[arg-type]
        file_count=1,
        total_bytes=1,
        archive_bytes=1,
        sha256="0" * 64,
        skipped=tuple(skipped),
        skipped_links=tuple(links),
    )


def test_a10_broken_or_outside_symlinks_stop_the_submit_unless_yes(tmp_path):
    world = World(tmp_path)
    world.pack_result = pack_result_with(
        [
            ("weights", "符号链接指向代码目录之外：/data/weights"),
            ("old.txt", "符号链接指向不存在的文件：gone.txt"),
            ("fifo", "不是普通文件（设备、管道或套接字）"),
        ],
        ["weights", "old.txt"],
    )
    with pytest.raises(DgpuError) as info:
        world.orchestrator().submit(make_spec(tmp_path))
    message = str(info.value)
    assert "weights" in message and "old.txt" in message and "--yes" in message
    assert world.store.objects == {} and world.backend.calls == []  # 什么都没上传、没调平台
    assert "weights：符号链接指向代码目录之外" in world.text()  # 跳过的每一项都列出来

    world.messages.clear()
    result = world.orchestrator().submit(make_spec(tmp_path), accept_skipped_links=True)
    assert result.phase == states.RUNNING
    assert "代码包跳过了 3 个条目" in world.text() and "fifo：不是普通文件" in world.text()


def test_a10_other_skipped_entries_are_printed_but_do_not_stop_the_submit(tmp_path):
    world = World(tmp_path)
    world.pack_result = pack_result_with([("sock", "不是普通文件（设备、管道或套接字）")], [])
    world.orchestrator().submit(make_spec(tmp_path))
    assert "sock：不是普通文件" in world.text()


# ================================================================ A11：费用上限


def test_a11_ceiling_covers_startup_and_teardown_and_mentions_extra_machines(tmp_path):
    world = World(tmp_path)
    world.orchestrator().submit(make_spec(tmp_path, max_hours=2.0, price=0.5))
    settings = runner_util.runner_settings({"runner": world.config.runner.to_dict()})
    lifetime = runner_util.instance_lifetime_s(2.0, settings, runner_util.DEFAULT_TUNING)
    hours = ((15 + 5) * 60 + lifetime) / 3600
    assert hours > 2.0 + (15 + 5) / 60  # 比“价格 × max_hours”多出启动和收尾
    assert f"× {hours:.2f} h = {0.5 * hours:.2f} USD" in world.text()
    assert "每台启动失败而被换掉的机器最多再花" in world.text()
    assert "每次 retry" in world.text()


# ================================================================ A12：wait 遇到存储瞬时错误


class FlakyStore(MemoryStore):
    """get_json_meta 前 fail 次抛错（模拟 R2 暂时连不上）。"""

    def __init__(self) -> None:
        super().__init__()
        self.fail = 0
        self.transient = True

    def get_json_meta(self, key):
        if self.fail > 0:
            self.fail -= 1
            raise StoreError("R2: read timeout", transient=self.transient)
        return super().get_json_meta(key)


def test_a12_wait_rides_out_a_short_storage_outage(tmp_path):
    store = FlakyStore()
    world = World(tmp_path, store=store, plans=[Plan(run_s=10 * MIN)])
    job_id = submit(world, tmp_path)
    store.fail = 4
    assert world.orchestrator().wait(job_id) == states.SUCCEEDED
    assert "读取任务状态失败" in world.text()


def test_a12_wait_gives_up_after_a_long_storage_outage_with_a_distinct_error(tmp_path):
    store = FlakyStore()
    world = World(tmp_path, store=store, plans=[Plan()])
    job_id = submit(world, tmp_path)
    store.fail = 10**6
    start = world.now
    with pytest.raises(StoreUnreachable, match="读不到存储"):
        world.orchestrator().wait(job_id)
    limit = orchestrator.STORE_OUTAGE_MAX_S
    assert limit <= world.now - start < limit + 120


def test_a12_wait_does_not_retry_a_permanent_storage_error(tmp_path):
    store = FlakyStore()
    world = World(tmp_path, store=store, plans=[Plan()])
    job_id = submit(world, tmp_path)
    store.fail, store.transient = 10**6, False
    start = world.now
    with pytest.raises(StoreError) as info:
        world.orchestrator().wait(job_id)
    assert not isinstance(info.value, StoreUnreachable) and world.now == start


# ======================================================= 怀疑 (a)：刚创建的实例查不到、销毁的确认


def test_suspicion_a_instance_not_yet_visible_after_create_is_not_treated_as_gone(tmp_path):
    world = World(tmp_path)
    use_backend(world, LaggyBackend(world, hidden={"o1": 30.0}))
    result = world.orchestrator().submit(make_spec(tmp_path))
    assert result.offer.id == "o1" and result.attempt == 1  # 没有因为“查不到”而换机
    assert world.events(result.job_id, states.EV_STARTUP_TIMEOUT) == []


def test_suspicion_a_destroy_is_sent_even_when_the_platform_says_gone(tmp_path):
    world = World(tmp_path)
    backend = use_backend(world, LaggyBackend(world, hidden={"": 10**6}))
    seed_job(world.store, JOB, phase=states.FAILED, hb_at=None, instance_id="5")
    backend.add_instance("5", layout.label(JOB, 1), created_at=world.now)  # 平台查不到它
    world.orchestrator().retry(JOB)
    assert ("destroy", "5") in backend.calls
    # 平台本来就查不到：不算作“销毁了一台实例”
    destroyed = [e["data"]["instance_id"] for e in world.events(JOB, "instance_destroyed")]
    assert "5" not in destroyed


class GhostBackend(SimBackend):
    """销毁后 get 立刻说没有了，但列表里还一直看得到（平台没有真正停掉）。"""

    def __init__(self, world: World):
        super().__init__(world, world.backend.name, list(world.backend.offers))
        self.ghosts: dict[str, InstanceState] = {}

    def destroy(self, ref):
        state = self.instances.get(ref.id)
        super().destroy(ref)
        if state is not None:
            self.ghosts[ref.id] = state

    def list(self, label_prefix=layout.LABEL_PREFIX):
        listed = super().list(label_prefix)
        return listed + [s for s in self.ghosts.values() if s.ref.label.startswith(label_prefix)]


def test_suspicion_a_destroy_is_confirmed_by_the_list_as_well(tmp_path):
    world = World(tmp_path)
    backend = use_backend(world, GhostBackend(world))
    seed_job(world.store, JOB, phase=states.FAILED, hb_at=None, instance_id="5")
    backend.add_instance("5", layout.label(JOB, 1), created_at=world.now)
    with pytest.raises(DgpuError, match="vast/5"):
        world.orchestrator().retry(JOB)
    assert backend.creates() == []  # 没确认旧实例消失，不开新机器
    assert world.events(JOB, states.EV_DESTROY_FAILED)


# ================================================================ 怀疑 (b)：标签被平台改写


class RelabelingBackend(SimBackend):
    """平台把标签截断成 20 个字符。"""

    def __init__(self, world: World):
        super().__init__(world, world.backend.name, list(world.backend.offers))

    def get(self, ref):
        state = super().get(ref)
        if state.phase == INST_GONE:
            return state
        return replace(state, ref=replace(state.ref, label=state.ref.label[:20]))


def test_suspicion_b_altered_label_fails_loudly_and_destroys_by_id(tmp_path):
    world = World(tmp_path)
    backend = use_backend(world, RelabelingBackend(world))
    with pytest.raises(DgpuError, match="标签") as info:
        world.orchestrator().submit(make_spec(tmp_path))
    assert backend.creates() == ["o1"]  # 不再试别的机器：是平台的行为，不是机器的问题
    assert backend.destroyed == ["1"] and world.alive() == []
    job_id = only_job_id(world)
    assert layout.label(job_id, 1)[:20] in str(info.value)
    (event,) = world.events(job_id, states.EV_CREATE_FAILED)
    assert event["data"]["actual_label"] == layout.label(job_id, 1)[:20]


# ================================================================ 怀疑 (e)：创建后价格高于上限


class PriceyBackend(SimBackend):
    """在某些 offer 上创建出来的实例，实际价格和搜索时不同。"""

    def __init__(self, world: World, prices: dict[str, float]):
        super().__init__(world, world.backend.name, list(world.backend.offers))
        self.prices = prices

    def create(self, offer, launch):
        ref = super().create(offer, launch)
        if offer.id in self.prices:
            self.instances[ref.id] = replace(
                self.instances[ref.id], price_per_hour=self.prices[offer.id]
            )
        return ref


def test_suspicion_e_instance_priced_above_the_ceiling_is_destroyed_and_replaced(tmp_path):
    world = World(tmp_path)
    backend = use_backend(world, PriceyBackend(world, {"o1": 0.9}))  # 上限 0.5
    result = world.orchestrator().submit(make_spec(tmp_path, price=0.5))
    assert result.offer.id == "o2" and result.attempt == 2
    assert backend.destroyed == ["1"]
    (event,) = world.events(result.job_id, states.EV_OFFER_REJECTED)
    assert event["data"]["price_per_hour"] == 0.9 and event["data"]["max_price_per_hour"] == 0.5


def test_suspicion_e_price_within_the_ceiling_is_accepted(tmp_path):
    world = World(tmp_path)
    use_backend(world, PriceyBackend(world, {"o1": 0.5}))
    result = world.orchestrator().submit(make_spec(tmp_path, price=0.5))
    assert result.offer.id == "o1"
