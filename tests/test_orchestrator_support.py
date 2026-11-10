"""编排测试共用的模拟世界：假时钟、假平台、假 runner。

假 runner 由注入的 sleep 驱动：编排每 sleep 一次，世界前进相应秒数，
每台实例按自己的 Plan 推进（拉镜像 → 运行 → 心跳 → 结束 / 失联），
并把心跳、状态、日志写进 MemoryStore，就像真的 runner 一样。

本文件没有测试，只放其他 test_orchestrator*.py 和 test_cli.py 共用的工具。
"""

from __future__ import annotations

import tarfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dgpu import layout, states
from dgpu.contracts import (
    INST_EXITED,
    INST_RUNNING,
    AutodlSettings,
    Config,
    DataRef,
    DataSource,
    GpuSpec,
    InstanceRef,
    InstanceState,
    JobSpec,
    LaunchSpec,
    Limits,
    Offer,
    Store,
    VastSettings,
)
from dgpu.orchestrator import Deps, Orchestrator
from dgpu.packaging import scan_dataset
from dgpu.testing import TEST_PROFILE, FakeBackend, FakeBundler, MemoryStore, make_offer

MAIN_VAST_KEY = "VAST-MAIN-KEY-must-never-leak-7f3a"
MAIN_AUTODL_TOKEN = "AUTODL-MAIN-TOKEN-must-never-leak-91c2"
T0 = 1_800_000_000.0
MAX_SLEEPS = 20_000


def make_config(**overrides: Any) -> Config:
    values: dict[str, Any] = {
        "stores": {"r2": TEST_PROFILE},
        "vast": VastSettings(api_key=MAIN_VAST_KEY),
        "autodl": AutodlSettings(token=MAIN_AUTODL_TOKEN),
    }
    values.update(overrides)
    return Config(**values)


def make_spec(
    tmp_path: Path,
    *,
    name: str = "demo",
    backend: str = "vast",
    data: tuple[DataRef, ...] = (),
    max_hours: float = 2.0,
    price: float = 0.5,
    gpu_names: tuple[str, ...] = ("RTX 3090", "RTX 4090"),
) -> JobSpec:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "train.py").write_text("print('train')\n")
    return JobSpec(
        version=1,
        name=name,
        backend=backend,
        images={"vast": "pytorch/pytorch:2.4.0-cuda12.1", "autodl": "image-uuid-1"},
        code=code,
        run="python train.py",
        limits=Limits(max_hours=max_hours, max_usd_per_hour=price, max_cny_per_hour=price * 7),
        gpu=GpuSpec(names=gpu_names),
        setup="pip install -r requirements.txt",
        data=data,
        env={"EPOCHS": "3"},
    )


@dataclass
class Plan:
    """一台实例的剧本。"""

    boot: str = "ok"  # ok | stuck（一直 loading）| exits（启动时退出）
    boot_s: float = 60.0  # loading → running 用时
    # ok | none（不发心跳）| preflight（preflight 失败并自毁）| crash（运行后、心跳前退出）
    # | ends（preflight 后立刻以 final_phase 结束并自毁）
    runner: str = "ok"
    hb_delay_s: float = 30.0  # running → 第一次心跳
    run_s: float | None = None  # 第一次心跳后多久结束；None 表示一直跑
    final_phase: str = states.SUCCEEDED
    final_reason: str | None = None
    die_after_s: float | None = None  # 第一次心跳后多久宿主机失联（不再心跳，实例仍在）
    # 失联时平台显示的状态：None 表示平台仍显示 running（比如只是和存储断连）；
    # "unknown" / "exited" 表示平台也知道它挂了（宿主机宕机）
    die_platform_phase: str | None = None
    self_destroy: bool = True  # 结束时自毁（Vast）；False 则停在 exited（AutoDL 的部署记录）
    log: bool = False  # 运行时每个 tick 追加一行日志


@dataclass
class Sim:
    iid: str
    offer_id: str
    plan: Plan
    created: float
    attempt: int
    job_id: str
    running_at: float | None = None
    first_hb: float | None = None
    preflight_done: bool = False
    done: bool = False
    saw_launch_doc: bool | None = None
    log_lines: int = 0


class SimBackend(FakeBackend):
    """FakeBackend + 创建时间 + 可编排的创建异常和销毁失败。"""

    def __init__(self, world: World, name: str = "vast", offers: list[Offer] | None = None):
        super().__init__(name, offers)
        self.world = world
        self.raise_after_create: dict[str, BaseException] = {}  # offer id → 创建后抛的异常
        self.destroy_errors: dict[str, BaseException] = {}  # instance id → destroy 抛的异常
        self.offer_of: dict[str, str] = {}

    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef:
        ref = super().create(offer, launch)
        self.instances[ref.id] = replace(self.instances[ref.id], created_at=self.world.now)
        self.offer_of[ref.id] = offer.id
        self.world.on_created(ref, offer, launch)
        error = self.raise_after_create.pop(offer.id, None)
        if error is not None:
            raise error
        return ref

    def destroy(self, ref: InstanceRef) -> None:
        error = self.destroy_errors.get(ref.id)
        if error is not None:
            self.calls.append(("destroy", ref.id))
            raise error
        super().destroy(ref)

    def creates(self) -> list[str]:
        return [arg for name, arg in self.calls if name == "create"]

    def add_instance(
        self, instance_id: str, label: str, *, created_at: float | None, phase: str = INST_RUNNING
    ) -> None:
        ref = InstanceRef(backend=self.name, id=instance_id, label=label)
        self.instances[instance_id] = InstanceState(
            ref=ref, phase=phase, raw_status=phase, created_at=created_at
        )


class World:
    def __init__(
        self,
        tmp_path: Path,
        *,
        offers: list[Offer] | None = None,
        plans: list[Plan] | None = None,
        config: Config | None = None,
        store: MemoryStore | None = None,
        backend_name: str = "vast",
    ):
        self.tmp_path = tmp_path
        self.now = T0
        self.store = store or MemoryStore()
        self.store.clock = self.clock  # 存储的时钟就是模拟世界的时钟（LastModified、Date）
        if offers is None:
            offers = [
                make_offer("o1", backend=backend_name, price=0.2),
                make_offer("o2", backend=backend_name, price=0.3),
                make_offer("o3", backend=backend_name, price=0.4),
            ]
        self.backend = SimBackend(self, backend_name, offers)
        self.bundler = FakeBundler()
        self.config = config or make_config()
        self.plan_queue: list[Plan] = list(plans or [])
        self.default_plan = Plan()
        self.sims: dict[str, Sim] = {}
        self.messages: list[str] = []
        self.sleeps = 0
        self.on_sleep: list[Callable[[World], None]] = []
        self.data_uploads: list[str] = []
        self.data_manifests: dict[str, Any] = {}  # resolve_data 收到的清单（数据名 → 清单）
        self.pack_excludes: list[list[Path]] = []
        self.pack_result: Any = None  # pack_code 的返回值（测试可以换成带 skipped 的结果）

    # ---- 注入点

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        if self.sleeps > MAX_SLEEPS:  # 编排里的循环必须有界；失控时让测试失败而不是挂住
            raise AssertionError(f"orchestrator slept more than {MAX_SLEEPS} times")
        self.now += seconds
        for hook in list(self.on_sleep):
            hook(self)
        self.tick()

    def echo(self, message: str) -> None:
        self.messages.append(message)

    def pack_code(self, code_dir: Path, dest: Path, *, exclude: Any = ()) -> Any:
        self.pack_excludes.append([Path(path) for path in exclude])
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(code_dir, arcname=".")
        return self.pack_result

    def resolve_data(self, store: Store, ref: DataRef, *, manifest: Any = None) -> DataSource:
        if ref.is_local:
            prefix = layout.dataset_prefix("hash" + ref.name)
            store.put_bytes(prefix + "data.bin", b"dataset")
            store.put_bytes(layout.dataset_complete_key("hash" + ref.name), b"")
            self.data_uploads.append(ref.name)
            self.data_manifests[ref.name] = manifest
            return DataSource(ref.name, store.profile.bucket, prefix)
        return DataSource(ref.name, ref.bucket or "", ref.prefix or "")

    def deps(self, **overrides: Any) -> Deps:
        values: dict[str, Any] = {
            "config": self.config,
            "store": self.store,
            "backends": {self.backend.name: self.backend},
            "bundler": self.bundler,
            "pack_code": self.pack_code,
            "resolve_data": self.resolve_data,
            "scan_data": scan_dataset,  # 真实的遍历：符号链接、大小都和上传一致
            "clock": self.clock,
            "sleep": self.sleep,
            "echo": self.echo,
        }
        values.update(overrides)
        return Deps(**values)

    def orchestrator(self, **overrides: Any) -> Orchestrator:
        return Orchestrator(self.deps(**overrides))

    # ---- 查询辅助

    def text(self) -> str:
        return "\n".join(self.messages)

    def events(self, job_id: str, name: str | None = None) -> list[dict[str, Any]]:
        keys = self.store.list(layout.events_prefix(job_id))
        docs = [self.store.get_json(key)[0] for key in keys]
        return [doc for doc in docs if name is None or doc["name"] == name]

    def doc(self, key: str) -> dict[str, Any] | None:
        got = self.store.get_json(key)
        return got[0] if got else None

    def alive(self) -> list[str]:
        return sorted(self.backend.instances)

    # ---- 模拟平台和 runner

    def on_created(self, ref: InstanceRef, offer: Offer, launch: LaunchSpec) -> None:
        plan = self.plan_queue.pop(0) if self.plan_queue else self.default_plan
        self.sims[ref.id] = Sim(
            iid=ref.id,
            offer_id=offer.id,
            plan=plan,
            created=self.now,
            attempt=launch.attempt,
            job_id=launch.job_id,
        )

    def tick(self) -> None:
        for sim in list(self.sims.values()):
            self._step(sim)

    def _status(self, sim: Sim, phase: str, reason: str | None = None, message: str = "") -> None:
        doc = states.new_status_doc(
            sim.job_id, sim.attempt, phase, states.SRC_RUNNER, reason=reason, message=message
        )
        doc["updated_at"] = self.now
        self.store.put_json(layout.status_key(sim.job_id), doc)

    def _heartbeat(self, sim: Sim, step: str = "run") -> None:
        doc = states.new_heartbeat_doc(
            sim.job_id, sim.attempt, states.RUNNING, step, gpu_util=87.0, synced_bytes=1024
        )
        doc["ts"] = self.now
        self.store.put_json(layout.heartbeat_key(sim.job_id), doc)

    def _self_destroy(self, sim: Sim) -> None:
        sim.done = True
        if sim.plan.self_destroy:
            self.backend.instances.pop(sim.iid, None)
        else:
            self.backend.set_phase(sim.iid, INST_EXITED, "exited")

    def _step(self, sim: Sim) -> None:
        backend = self.backend
        if sim.done or sim.iid not in backend.instances:
            return
        plan = sim.plan
        if sim.running_at is None:
            if plan.boot == "stuck" or self.now - sim.created < plan.boot_s:
                return
            if plan.boot == "exits":
                backend.set_phase(sim.iid, INST_EXITED, "exited")
                sim.done = True
                return
            backend.set_phase(sim.iid, INST_RUNNING, "running")
            sim.running_at = self.now
            return
        since_running = self.now - sim.running_at
        if plan.runner == "none":
            return
        if plan.runner == "crash":
            if since_running >= plan.hb_delay_s:
                backend.set_phase(sim.iid, INST_EXITED, "exited")
                sim.done = True
            return
        attempt_doc = self.doc(layout.attempt_key(sim.job_id)) or {}
        fenced = (attempt_doc.get("attempt") or 0) > sim.attempt
        if sim.first_hb is None:
            if since_running < plan.hb_delay_s:
                return
            sim.saw_launch_doc = self.store.exists(layout.secrets_key(sim.job_id, sim.attempt))
            if fenced:
                self._self_destroy(sim)
                return
            # 和真实 runner 一样：先写 RUNNING 和第一次心跳（step=preflight），再做 preflight
            sim.first_hb = self.now
            self._status(sim, states.RUNNING)
            self._heartbeat(sim, "preflight")
            return
        if not sim.preflight_done:
            sim.preflight_done = True
            if plan.runner == "preflight":
                self._status(sim, states.FAILED, states.REASON_PREFLIGHT, "nvidia-smi failed")
                self._self_destroy(sim)
                return
            if plan.runner == "ends":  # preflight 后立刻以 final_phase 结束（如下载数据失败）
                self._status(sim, plan.final_phase, plan.final_reason, "ended right away")
                self._self_destroy(sim)
                return
            self._heartbeat(sim, "fetch")  # preflight 通过后立刻写一次心跳
            return
        since_hb = self.now - sim.first_hb
        if plan.die_after_s is not None and since_hb >= plan.die_after_s:
            # 失联：实例还在计费，但不再心跳，也不再响应取消和隔离
            if plan.die_platform_phase and backend.instances[sim.iid].phase != (
                plan.die_platform_phase
            ):
                backend.set_phase(sim.iid, plan.die_platform_phase)
            return
        if fenced:
            self._self_destroy(sim)
            return
        if attempt_doc.get("cancel"):
            self._status(sim, states.CANCELLED, message="cancelled by user")
            self._self_destroy(sim)
            return
        self._heartbeat(sim)
        if plan.log:
            sim.log_lines += 1
            key = layout.log_key(sim.job_id, sim.attempt)
            old = self.store.objects.get(key, b"")  # 不经 get_bytes，免得混进 CLI 的读取记录
            self.store.put_bytes(key, old + f"第 {sim.log_lines} 行 epoch\n".encode())
        if plan.run_s is not None and since_hb >= plan.run_s:
            self._status(sim, plan.final_phase, plan.final_reason)
            self._self_destroy(sim)


class HookedStore(MemoryStore):
    """MemoryStore + put_json 前的钩子（用来制造并发写竞争）和 get_bytes 调用记录。"""

    def __init__(self) -> None:
        super().__init__()
        self.before_put: Callable[[str, dict, str | None], None] | None = None
        self.byte_reads: list[tuple[str, int]] = []

    def put_json(self, key, obj, *, if_match=None, if_none_match=False):
        hook = self.before_put
        if hook is not None:
            hook(key, obj, if_match)
        return super().put_json(key, obj, if_match=if_match, if_none_match=if_none_match)

    def get_bytes(self, key, *, start=0):
        self.byte_reads.append((key, start))
        return super().get_bytes(key, start=start)


def seed_job(
    store: MemoryStore,
    job_id: str,
    *,
    attempt: int = 1,
    phase: str = states.RUNNING,
    status_attempt: int | None = None,
    updated_at: float = T0,
    hb_at: float | None = T0,
    hb_attempt: int | None = None,
    instance_id: str | None = None,
    backend: str = "vast",
    deadline_at: float | None = None,
    attempt_created_at: float = T0,
    reason: str | None = None,
    with_request: bool = True,
) -> None:
    """直接往存储里写一个任务的状态文档（gc、status、wait 的测试用）。

    每个文档在存储里的修改时间（LastModified）就是它自己的时间字段：attempt.json 是
    attempt_created_at，status.json 是 updated_at，heartbeat.json 是 hb_at。
    """
    att = states.new_attempt_doc(
        job_id, attempt, backend, layout.label(job_id, attempt), instance_id, deadline_at
    )
    att["created_at"] = attempt_created_at
    store.put_json(layout.attempt_key(job_id), att)
    store.modified[layout.attempt_key(job_id)] = attempt_created_at
    st = states.new_status_doc(
        job_id, status_attempt or attempt, phase, states.SRC_RUNNER, reason=reason
    )
    st["updated_at"] = updated_at
    store.put_json(layout.status_key(job_id), st)
    store.modified[layout.status_key(job_id)] = updated_at
    if hb_at is not None:
        hb = states.new_heartbeat_doc(job_id, hb_attempt or attempt, states.RUNNING, "run")
        hb["ts"] = hb_at
        store.put_json(layout.heartbeat_key(job_id), hb)
        store.modified[layout.heartbeat_key(job_id)] = hb_at
    store.put_json(layout.spec_key(job_id), {"version": 1, "job_id": job_id})
    if with_request:
        store.put_json(
            layout.request_key(job_id),
            {
                "version": 1,
                "job_id": job_id,
                "backend": backend,
                "gpu_names": ["RTX 3090"],
                "gpu_count": 1,
                "max_price_per_hour": 0.5,
                "currency": "USD",
                "max_hours": 2.0,
                "disk_gb": 40,
                "image": "pytorch/pytorch:2.4.0-cuda12.1",
            },
        )
