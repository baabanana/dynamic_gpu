"""集成测试：真实的 Orchestrator 驱动真实的 bundle、bootstrap、boot、runner、watchdog。

只把云换成本地的替身：

- 存储：DirStore，把 `<remote>/<bucket>/<key>` 当作对象，和 runner 用的假 rclone
  （tests/test_runner_fakes.py）读写同一个目录；预签名 URL 是 file:// 地址，由 bootstrap
  的 urllib 直接下载。条件写和假 rclone 的写入共用一把锁，和 S3 一样是原子的。
- 平台：ProcBackend（FakeBackend 的子类），create 时在子进程里执行真实的
  `launch.bootstrap_cmd`（工作目录在 tmp 下，PATH 上的 python3 尽量是 3.8/3.9），
  destroy 时结束这台“实例”的整棵进程树。runner 自毁调用的 Vast API 由本地的
  DestroyStub 代替，它把实例标成已消失。
- spec、配置、打包、bundle、编排都用真实实现（job.yaml 由 spec.load_spec 读取）。

每个场景结束后检查：`gc --dry-run` 没有要销毁的实例，这个任务没有留下任何进程。
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from dgpu import bundle, layout, packaging, states
from dgpu.bundle import RunnerBundler
from dgpu.config import parse_config
from dgpu.contracts import (
    INST_RUNNING,
    InstanceRef,
    LaunchSpec,
    Offer,
    PreconditionFailed,
    RunnerSettings,
    StoreError,
    StoreProfile,
)
from dgpu.orchestrator import Deps, Orchestrator
from dgpu.runner import procs
from dgpu.spec import load_spec
from dgpu.testing import FakeBackend, make_offer
from test_runner_harness import FAST_TUNING, TESTS_DIR, WRAPPER, old_python

MAIN_VAST_KEY = "VAST-MAIN-KEY-integration-must-never-leak"
STORE_SECRET = "store-secret-integration-0001"
SCENARIO_BUDGET_S = 90.0  # 单个场景的上限：编排卡住时让测试失败，而不是挂住

FAST_RUNNER = RunnerSettings(
    sync_interval_s=0.3,
    heartbeat_interval_s=0.2,
    stall_timeout_min=0,
    final_sync_timeout_min=0.05,
    term_grace_s=1.0,
    max_log_mb=5,
    data_sync_retries=1,
)

TRAIN_PY = r"""
import json, os, sys, time
from pathlib import Path

out = Path(os.environ["DGPU_OUTPUT"])
ckpt = Path(os.environ["DGPU_CKPT"])
data = Path(os.environ["DGPU_DATA"])
mode = os.environ.get("MODE", "ok")
attempt = os.environ["DGPU_ATTEMPT"]
resume = os.environ.get("DGPU_RESUME")
print("train start attempt=%s resume=%s mode=%s" % (attempt, resume, mode), flush=True)
state = ckpt / "state.json"
info = {
    "attempt": attempt,
    "resume": resume,
    "ckpt": json.loads(state.read_text()) if state.exists() else None,
    "data": sorted(
        p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()
    ) if data.is_dir() else [],
    "secrets_in_env": sorted(
        k for k in os.environ if k.startswith(("RCLONE_CONFIG", "CONTAINER_API_KEY", "AWS_"))
    ),
}
(out / ("info-%s.json" % attempt)).write_text(json.dumps(info))
if mode == "fail":
    print("boom: failing on purpose", flush=True)
    sys.exit(3)
if mode == "hang" or (mode == "resume" and resume != "1"):
    tmp = ckpt / "state.json.tmp"
    tmp.write_text(json.dumps({"epoch": 1, "attempt": attempt}))
    os.replace(str(tmp), str(state))
    print("checkpoint written", flush=True)
    while True:
        print("tick", flush=True)
        time.sleep(0.2)
(out / "model.txt").write_text("weights from attempt %s" % attempt)
(out / "sub").mkdir(exist_ok=True)
(out / "sub" / "metrics.csv").write_text("epoch,loss\n1,0.1\n")
print("train done", flush=True)
"""


# ================================================================ 存储：本地目录


class DirStore:
    """contracts.Store 的本地目录实现，和假 rclone 共用 `<root>/<bucket>/`。"""

    TMP_MARK = ".dirstore-tmp-"
    _HIDDEN_RE = re.compile(r"\.fake-\d+$")  # 假 rclone 写到一半的临时文件

    def __init__(self, root: Path, profile: StoreProfile):
        self.profile = profile
        self.root = Path(root)
        self.base = self.root / profile.bucket
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".control" / "store.lock"
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.puts: list[str] = []

    # ---- 内部

    def _path(self, key: str) -> Path:
        parts = key.split("/")
        if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise StoreError(f"bad key {key!r}")
        return self.base.joinpath(*parts)

    @contextlib.contextmanager
    def _locked(self):
        with open(self._lock_path, "a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield

    def _write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}{self.TMP_MARK}{os.getpid()}-{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self.puts.append(key)

    def _read(self, key: str) -> bytes | None:
        try:
            return self._path(key).read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            return None

    @staticmethod
    def _etag(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    # ---- Store

    def get_json(self, key: str) -> tuple[dict[str, Any], str] | None:
        data = self._read(key)
        if data is None:
            return None
        return json.loads(data), self._etag(data)

    def put_json(self, key, obj, *, if_match=None, if_none_match=False) -> str:
        data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
        with self._locked():
            current = self._read(key)
            if if_none_match and current is not None:
                raise PreconditionFailed(f"{key} already exists")
            if if_match is not None and (current is None or self._etag(current) != if_match):
                raise PreconditionFailed(f"{key} was modified")
            self._write(key, data)
        return self._etag(data)

    def get_bytes(self, key: str, *, start: int = 0) -> bytes | None:
        data = self._read(key)
        return None if data is None else data[start:]

    def put_bytes(self, key: str, data: bytes) -> None:
        with self._locked():
            self._write(key, bytes(data))

    def put_file(self, key: str, path: Path) -> None:
        self.put_bytes(key, Path(path).read_bytes())

    def exists(self, key: str) -> bool:
        return self._read(key) is not None

    def list(self, prefix: str) -> list[str]:
        keys = []
        for dirpath, _dirs, files in os.walk(self.base):
            for name in files:
                if self.TMP_MARK in name or self._HIDDEN_RE.search(name):
                    continue
                rel = Path(dirpath, name).relative_to(self.base).as_posix()
                if rel.startswith(prefix):
                    keys.append(rel)
        return sorted(keys)

    def list_dirs(self, prefix: str) -> list[str]:
        names = set()
        for key in self.list(prefix):
            head, sep, _ = key[len(prefix) :].partition("/")
            if sep:
                names.add(head)
        return sorted(names)

    def download_prefix(self, prefix: str, dest: Path) -> int:
        count = 0
        for key in self.list(prefix):
            rel = key[len(prefix) :].lstrip("/") or PurePosixPath(key).name
            target = Path(dest).joinpath(*PurePosixPath(rel).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._read(key) or b"")
            count += 1
        return count

    def presign_get(self, key: str, ttl_s: int) -> str:
        return self._path(key).as_uri()

    def delete(self, key: str) -> None:
        with self._locked():
            with contextlib.suppress(FileNotFoundError):
                self._path(key).unlink()

    # ---- 测试辅助

    def text_of_everything(self) -> str:
        chunks = []
        for key in self.list(""):
            data = self._read(key) or b""
            if not key.endswith((".tar.gz", ".tgz")):
                chunks.append(data.decode("utf-8", "replace"))
        return "\n".join(chunks)


# ================================================================ 平台：子进程里的“实例”


class DestroyStub:
    """代替 console.vast.ai：runner / watchdog 用容器 key 删除自己时，把实例标成已消失。"""

    def __init__(self, on_destroy):
        self.calls: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_DELETE(self):  # noqa: N802 - http.server 的命名
                match = re.fullmatch(r"/api/v0/instances/([^/]+)/", self.path)
                auth = self.headers.get("Authorization") or ""
                ok = bool(match) and auth == f"Bearer {container_key(match.group(1))}"
                outer.calls.append({"path": self.path, "ok": ok, "ts": time.time()})
                if ok:
                    on_destroy(match.group(1))
                body = json.dumps({"success": ok}).encode()
                self.send_response(200 if ok else 403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def container_key(instance_id: str) -> str:
    return f"container-key-{instance_id}"


class ProcBackend(FakeBackend):
    """FakeBackend，但 create 真的在子进程里执行启动命令，destroy 结束整棵进程树。"""

    def __init__(self, cloud: Cloud, offers: list[Offer]):
        super().__init__("vast", offers)
        self.cloud = cloud
        self.lock = threading.RLock()  # 自毁桩在另一个线程里改 instances
        self.boot_procs: dict[str, subprocess.Popen] = {}
        self.work_roots: dict[str, Path] = {}
        self.self_destroyed: list[str] = []
        self.killed: list[str] = []
        self.status_at_kill: dict[str, dict[str, Any] | None] = {}  # 被杀时存储里的 status.json
        self.no_gpu: set[str] = set()  # 这些 offer 上创建的实例没有 GPU（preflight 失败）
        self.zombies: set[str] = set()  # destroy 只改平台记录、不结束进程的实例（平台“说谎”）
        self.commands: list[str] = []
        self.envs: list[dict[str, str]] = []

    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef:
        with self.lock:
            ref = super().create(offer, launch)
            state = self.instances[ref.id]
            self.instances[ref.id] = dataclasses.replace(
                state, phase=INST_RUNNING, raw_status="running", created_at=time.time()
            )
        work = self.cloud.tmp / "instances" / ref.id
        (work / "root").mkdir(parents=True)
        env = self.cloud.instance_env(ref.id, work / "root", no_gpu=offer.id in self.no_gpu)
        self.commands.append(launch.bootstrap_cmd)
        self.envs.append(env)
        out = open(work / "bootstrap.out", "wb")  # noqa: SIM115 - 交给子进程
        try:
            self.boot_procs[ref.id] = subprocess.Popen(
                ["bash", "-c", launch.bootstrap_cmd],
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            out.close()
        self.work_roots[ref.id] = work / "root"
        return ref

    def get(self, ref):
        with self.lock:
            return super().get(ref)

    def list(self, label_prefix: str = layout.LABEL_PREFIX):
        with self.lock:
            return super().list(label_prefix)

    def destroy(self, ref: InstanceRef) -> None:
        if ref.id not in self.zombies:
            self.kill_tree(ref.id)
        with self.lock:
            super().destroy(ref)

    # ---- 测试辅助

    def self_destroy(self, instance_id: str) -> None:
        """自毁桩收到 DELETE：平台受理，实例消失。容器里的进程自己退出。"""
        with self.lock:
            if self.instances.pop(instance_id, None) is not None:
                self.self_destroyed.append(instance_id)

    def crash_host(self, instance_id: str) -> None:
        """宿主机宕机：进程全没了，平台上的实例记录还在（仍在计费）。"""
        self.kill_tree(instance_id)

    def pids(self, instance_id: str) -> set[int]:
        root = self.work_roots.get(instance_id)
        found: set[int] = set()
        if root is None:
            return found
        boot = self.boot_procs.get(instance_id)
        if boot is not None and boot.poll() is None:
            found.add(boot.pid)
        for name in ("runner.pid", "watchdog.pid"):
            with contextlib.suppress(OSError, ValueError):
                found.add(int((root / "run" / name).read_text().strip()))
        groups = set(found)
        with contextlib.suppress(OSError, ValueError):
            child = json.loads((root / "run" / "child.json").read_text())
            groups.add(int(child["pgid"]))
        found |= procs.descendants(list(found | groups))
        for pid, _ppid, pgid in procs.list_processes():
            if pgid in groups:
                found.add(pid)
        return {pid for pid in found if pid > 0 and pid != os.getpid()}

    def kill_tree(self, instance_id: str) -> None:
        if instance_id not in self.work_roots:
            return
        self.killed.append(instance_id)
        got = self.cloud.store.get_json(layout.status_key(self.launches[instance_id].job_id))
        self.status_at_kill[instance_id] = got[0] if got else None
        for _ in range(3):  # 进程树可能在发信号期间长出新进程
            pids = self.pids(instance_id)
            if not pids:
                break
            for pid in pids:
                procs.kill_pid(pid, signal.SIGKILL)
            time.sleep(0.05)
        boot = self.boot_procs.get(instance_id)
        if boot is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                boot.wait(5)

    def reap(self) -> None:
        for proc in self.boot_procs.values():
            if proc.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(1)


# ================================================================ 时钟


class SkewClock:
    """墙上时钟加一个可调的偏移，用来“快进”编排看到的时间（判 LOST）。"""

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


# ================================================================ 整个本地云


class Cloud:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.remote = tmp / "remote"
        self.bin = tmp / "bin"
        self.nogpu_bin = tmp / "bin-nogpu"
        self.project = tmp / "project"
        for path in (self.remote / ".control", self.bin, self.nogpu_bin, self.project):
            path.mkdir(parents=True, exist_ok=True)
        self.python = old_python() or sys.executable
        self._write_tools()
        self.started = time.monotonic()

        self.config = dataclasses.replace(
            parse_config(
                {
                    "startup_timeout_min": 1,
                    "runner_timeout_min": 1,
                    "lost_after_min": 1,
                    "gc_stale_min": 2,
                    "max_offers": 3,
                    "stores": {
                        "r2": {
                            "endpoint": "https://r2.test.invalid",
                            "bucket": "test-bucket",
                            "access_key_id": "store-key-integration",
                            "secret_access_key": STORE_SECRET,
                        }
                    },
                    "backends": {"vast": {"api_key": MAIN_VAST_KEY}},
                },
                env={},
            ),
            runner=FAST_RUNNER,
        )
        self.store = DirStore(self.remote, self.config.stores["r2"])
        self.backend = ProcBackend(self, [make_offer("o1", price=0.2), make_offer("o2", price=0.3)])
        self.stub = DestroyStub(self.backend.self_destroy)
        self.bundler = self._bundler()
        self.clock = SkewClock()
        self.messages: list[str] = []
        self.orch = Orchestrator(
            Deps(
                config=self.config,
                store=self.store,
                backends={"vast": self.backend},
                bundler=self.bundler,
                pack_code=packaging.pack_code,
                resolve_data=packaging.resolve_data,
                clock=self.clock,
                sleep=self.sleep,
                echo=self.messages.append,
                load_spec=load_spec,
            )
        )

    # ---- 准备

    def _write_tools(self) -> None:
        for name, func in (("rclone", "fake_rclone_main"), ("nvidia-smi", "fake_nvidia_smi_main")):
            path = self.bin / name
            path.write_text(WRAPPER.format(python=sys.executable, tests=str(TESTS_DIR), func=func))
            path.chmod(0o755)
        for folder in (self.bin, self.nogpu_bin):
            wrapper = folder / "python3"
            wrapper.write_text(f'#!/bin/sh\nexec "{self.python}" "$@"\n')
            wrapper.chmod(0o755)
        broken = self.nogpu_bin / "nvidia-smi"
        broken.write_text(
            "#!/bin/sh\necho 'NVIDIA-SMI has failed: no devices were found'\nexit 9\n"
        )
        broken.chmod(0o755)

    def _bundler(self) -> RunnerBundler:
        """真实的 RunnerBundler；bundle 里的 rclone 换成假 rclone（不下载真的）。"""
        fake = (self.bin / "rclone").read_bytes()
        cache = self.tmp / "cache"
        target = cache / f"rclone-{bundle.RCLONE_VERSION}-linux-amd64" / "rclone"
        target.parent.mkdir(parents=True)
        target.write_bytes(fake)
        target.chmod(0o755)

        def no_fetch(url: str) -> bytes:
            raise AssertionError(f"tests must not download rclone: {url}")

        return RunnerBundler(
            cache_dir=cache, fetch=no_fetch, rclone_binary_sha256=hashlib.sha256(fake).hexdigest()
        )

    def instance_env(self, instance_id: str, root: Path, *, no_gpu: bool) -> dict[str, str]:
        drop = ("RCLONE_", "CONTAINER_", "AWS_", "R2_", "DGPU_", "VAST_", "PYTHON")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(drop) and "PROXY" not in key.upper()
        }
        tools = self.nogpu_bin if no_gpu else self.bin
        env.update(
            {
                "PATH": f"{tools}{os.pathsep}{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "HOME": str(root.parent),
                "FAKE_RCLONE_ROOT": str(self.remote),
                "DGPU_WORK_ROOT": str(root),
                "DGPU_RUNNER_TUNING": json.dumps(dict(FAST_TUNING, vast_api_base=self.stub.url)),
                "DGPU_BOOT_BACKOFF_S": "0.05",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "CONTAINER_ID": instance_id,
                "CONTAINER_API_KEY": container_key(instance_id),
            }
        )
        return env

    def sleep(self, seconds: float) -> None:
        if time.monotonic() - self.started > SCENARIO_BUDGET_S:
            raise AssertionError("scenario exceeded its time budget\n" + self.diagnostics())
        time.sleep(min(seconds, 0.05))

    def write_job(
        self,
        name: str,
        *,
        mode: str = "ok",
        max_hours: float = 0.05,
        data: list[str] | None = None,
    ) -> Path:
        (self.project / "train.py").write_text(TRAIN_PY)
        doc: dict[str, Any] = {
            "version": 1,
            "name": name,
            "backend": "vast",
            "image": {"vast": "local/e2e:1"},
            "code": ".",
            "run": f"'{self.python}' train.py",
            "gpu": {"names": ["RTX 3090"], "count": 1},
            "limits": {"max_hours": max_hours, "max_usd_per_hour": 1.0},
            "env": {"MODE": mode},
        }
        if data:
            doc["data"] = data
        path = self.project / f"{name}.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        return path

    def submit(self, name: str, **kwargs: Any):
        spec = load_spec(self.write_job(name, **kwargs))
        return self.orch.submit(spec, confirm_upload=None)

    # ---- 查询

    def wait_for(self, predicate, what: str, timeout: float = 30.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}\n{self.diagnostics()}")

    def heartbeat(self, job_id: str) -> dict[str, Any] | None:
        got = self.store.get_json(layout.heartbeat_key(job_id))
        return got[0] if got else None

    def status_doc(self, job_id: str) -> dict[str, Any]:
        got = self.store.get_json(layout.status_key(job_id))
        assert got is not None
        return got[0]

    def events(self, job_id: str) -> list[dict[str, Any]]:
        docs = []
        for key in self.store.list(layout.events_prefix(job_id)):
            got = self.store.get_json(key)
            if got is not None:
                docs.append(got[0])
        return docs

    def event_names(self, job_id: str) -> list[str]:
        return [event["name"] for event in self.events(job_id)]

    def info(self, job_id: str, attempt: int) -> dict[str, Any]:
        data = self.store.get_bytes(layout.output_prefix(job_id) + f"info-{attempt}.json")
        assert data is not None, self.diagnostics()
        return json.loads(data)

    def job_processes(self) -> list[str]:
        """命令行里带着本测试 tmp 目录的进程（实例上的一切都在这个目录下运行）。"""
        out = subprocess.run(
            ["ps", "-axww", "-o", "pid=,stat=,command="], capture_output=True, text=True
        ).stdout
        marker = str(self.tmp)
        real = os.path.realpath(self.tmp)
        rows = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3 or parts[1].startswith("Z") or int(parts[0]) == os.getpid():
                continue
            if marker in parts[2] or real in parts[2]:
                rows.append(line.strip())
        return rows

    def diagnostics(self) -> str:
        parts = ["--- orchestrator ---", "\n".join(self.messages[-40:])]
        for iid, root in self.backend.work_roots.items():
            for name in ("runner.log", "watchdog.log", "bootstrap.log"):
                path = root / "logs" / name
                if path.exists():
                    parts.append(f"--- {iid} {name} ---\n{path.read_text()[-3000:]}")
            out = root.parent / "bootstrap.out"
            if out.exists() and out.read_text().strip():
                parts.append(f"--- {iid} bootstrap.out ---\n{out.read_text()[-2000:]}")
        return "\n".join(parts)

    # ---- 场景结束后的检查

    def assert_clean(self, *, crashed: tuple[str, ...] = ()) -> None:
        """没有遗留实例（gc --dry-run 什么都不销毁），没有遗留进程，主 key 没有泄漏。

        除了模拟宕机的实例（crashed），CLI 只会销毁 runner 已经写完最终状态的实例
        （此后 runner 只剩自毁，日志和事件都已经传完）。
        """
        report = self.orch.gc(dry_run=True)
        assert report.destroyed == [] and report.failed == [], report
        assert self.backend.instances == {}, self.backend.instances
        self.wait_for(lambda: not self.job_processes(), "every job process to exit", 20)
        for iid in self.backend.killed:
            if iid not in crashed:
                status = self.backend.status_at_kill.get(iid) or {}
                attempt = self.backend.launches[iid].attempt
                assert status.get("attempt") == attempt, (iid, status, self.diagnostics())
                assert status.get("phase") in states.TERMINAL_PHASES, (iid, status)
                assert status.get("source") != states.SRC_CLI, (iid, status)
        assert MAIN_VAST_KEY not in self.store.text_of_everything()
        for command, env in zip(self.backend.commands, self.backend.envs, strict=True):
            assert MAIN_VAST_KEY not in command and STORE_SECRET not in command
            assert MAIN_VAST_KEY not in json.dumps(env) and STORE_SECRET not in json.dumps(env)

    def cleanup(self) -> None:
        for iid in list(self.backend.work_roots):
            self.backend.kill_tree(iid)
        self.backend.reap()
        self.stub.stop()


@pytest.fixture
def cloud(tmp_path):
    # macOS 的 tmp 路径很长，Unix socket 等不受影响；这里只要求路径里没有空格
    c = Cloud(tmp_path)
    yield c
    c.cleanup()


def _code_archive_names(cloud: Cloud, job_id: str) -> list[str]:
    data = cloud.store.get_bytes(layout.code_key(job_id))
    assert data is not None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return sorted(tar.getnames())


# ================================================================ 场景


def test_success_pulls_outputs_uploads_local_data_once_and_leaves_nothing(cloud, tmp_path):
    data_dir = cloud.project / "datasets" / "mydata"
    (data_dir / "sub").mkdir(parents=True)
    (data_dir / "a.txt").write_text("alpha")
    (data_dir / "sub" / "b.txt").write_text("beta")

    result = cloud.submit("e2e-ok", data=["./datasets/mydata"])
    assert result.attempt == 1 and result.phase in (states.RUNNING, states.SUCCEEDED)
    job_id = result.job_id
    assert cloud.orch.wait(job_id) == states.SUCCEEDED, cloud.diagnostics()

    status = cloud.status_doc(job_id)
    assert (status["phase"], status["exit_code"], status["source"]) == (
        states.SUCCEEDED,
        0,
        states.SRC_RUNNER,
    )
    pulled = cloud.orch.pull(job_id, tmp_path / "pulled")
    files = sorted(
        p.relative_to(tmp_path / "pulled" / "output").as_posix()
        for p in (tmp_path / "pulled" / "output").rglob("*")
        if p.is_file()
    )
    assert files == ["info-1.json", "model.txt", "sub/metrics.csv"]
    assert pulled.output_files == 3
    info = cloud.info(job_id, 1)
    assert info["data"] == ["mydata/a.txt", "mydata/sub/b.txt"]
    assert info["resume"] == "0" and info["secrets_in_env"] == []
    # 本地数据在代码目录里：只作为数据集上传，不进代码包（第 1 项）
    assert not any(name.startswith("datasets") for name in _code_archive_names(cloud, job_id))
    assert "train.py" in _code_archive_names(cloud, job_id)
    dataset_puts = [key for key in cloud.store.puts if key.startswith("dgpu/datasets/")]
    assert len(dataset_puts) == 3  # 两个文件 + 完成标记
    # 实例已自毁；启动文档已删除
    assert cloud.backend.self_destroyed == [result.instance.id]
    assert not cloud.store.exists(layout.secrets_key(job_id, 1))
    log = "\n".join(_collect_logs(cloud, job_id))
    assert "train done" in log and "最终状态 SUCCEEDED" in log
    cloud.assert_clean()

    # 第二个任务用同一份数据：不重传
    second = cloud.submit("e2e-ok2", data=["./datasets/mydata"])
    assert cloud.orch.wait(second.job_id) == states.SUCCEEDED, cloud.diagnostics()
    assert [k for k in cloud.store.puts if k.startswith("dgpu/datasets/")] == dataset_puts
    assert cloud.info(second.job_id, 1)["data"] == ["mydata/a.txt", "mydata/sub/b.txt"]
    cloud.assert_clean()


def _collect_logs(cloud: Cloud, job_id: str, attempt: int | None = None) -> list[str]:
    lines: list[str] = []
    cloud.orch.logs(job_id, attempt=attempt, out=lines.append)
    return lines


def test_nonzero_exit_is_failed_with_exit_code_and_log(cloud):
    result = cloud.submit("e2e-fail", mode="fail")
    assert cloud.orch.wait(result.job_id) == states.FAILED, cloud.diagnostics()
    view = cloud.orch.status(result.job_id)
    assert (view.reason, view.exit_code) == (states.REASON_EXIT_CODE, 3)
    log = "\n".join(_collect_logs(cloud, result.job_id))
    assert "boom: failing on purpose" in log
    assert "run 结束，退出码 3" in log
    # 日志在最终状态之前传完：CLI 一看到终态就可以销毁实例，不会丢掉日志的结尾
    assert "最终状态 FAILED（exit_code）" in log
    cloud.assert_clean()


def test_tiny_max_hours_times_out(cloud):
    result = cloud.submit("e2e-timeout", mode="hang", max_hours=0.001)
    followed: list[str] = []
    cloud.orch.logs(result.job_id, follow=True, out=followed.append)  # logs -f 跟到终态为止
    assert "tick" in followed and any("最终状态 TIMED_OUT" in line for line in followed)
    assert cloud.orch.wait(result.job_id) == states.TIMED_OUT, cloud.diagnostics()
    assert states.EV_DEADLINE_HIT in cloud.event_names(result.job_id)
    # 超时前写的 checkpoint 和输出在最终同步里上传了
    assert cloud.store.exists(layout.ckpt_prefix(result.job_id) + "state.json")
    assert cloud.info(result.job_id, 1)["attempt"] == "1"
    cloud.assert_clean()


def test_lost_instance_is_retried_and_resumes_from_checkpoint(cloud):
    result = cloud.submit("e2e-lost", mode="resume")
    job_id = result.job_id
    first = result.instance.id
    cloud.wait_for(
        lambda: cloud.store.exists(layout.ckpt_prefix(job_id) + "state.json"),
        "attempt 1 to sync its checkpoint",
    )
    cloud.backend.crash_host(first)
    assert first in cloud.backend.instances  # 平台上还在（计费）

    # 心跳过期：快进编排看到的时间
    cloud.clock.offset = (cloud.config.lost_after_min + 1) * 60
    assert cloud.orch.status(job_id).phase == states.LOST
    retried = cloud.orch.retry(job_id)
    cloud.clock.offset = 0.0
    assert retried.attempt == 2
    assert first not in cloud.backend.instances  # retry 先确认旧实例已销毁

    assert cloud.orch.wait(job_id) == states.SUCCEEDED, cloud.diagnostics()
    info = cloud.info(job_id, 2)
    assert info["resume"] == "1"
    assert info["ckpt"] == {"epoch": 1, "attempt": "1"}
    names = cloud.event_names(job_id)
    assert states.EV_CKPT_RESTORED in names
    assert states.EV_RELAUNCH in names
    assert "attempt_fenced" not in [e["name"] for e in cloud.events(job_id) if e["source"] == "cli"]
    assert "train done" in "\n".join(_collect_logs(cloud, job_id, attempt=2))
    assert "checkpoint written" in "\n".join(_collect_logs(cloud, job_id, attempt=1))
    cloud.assert_clean(crashed=(first,))


def test_cancel_mid_run_is_cancelled_after_final_sync(cloud):
    result = cloud.submit("e2e-cancel", mode="hang")
    job_id = result.job_id
    cloud.wait_for(
        lambda: (cloud.heartbeat(job_id) or {}).get("step") == "run", "the run step to start"
    )
    cloud.orch.cancel(job_id)
    assert cloud.orch.wait(job_id) == states.CANCELLED, cloud.diagnostics()
    assert cloud.status_doc(job_id)["source"] == states.SRC_RUNNER
    assert cloud.info(job_id, 1)["attempt"] == "1"  # 取消时做了最终同步
    assert cloud.store.exists(layout.ckpt_prefix(job_id) + "state.json")
    cloud.assert_clean()


def test_preflight_failure_switches_to_the_next_machine(cloud):
    cloud.backend.no_gpu.add("o1")
    result = cloud.submit("e2e-preflight")
    job_id = result.job_id
    assert result.offer.id == "o2" and result.attempt == 2, cloud.diagnostics()
    assert cloud.orch.wait(job_id) == states.SUCCEEDED, cloud.diagnostics()
    events = cloud.events(job_id)
    preflight = [e for e in events if e["name"] == states.EV_PREFLIGHT_FAILED]
    assert {e["source"] for e in preflight} >= {states.SRC_CLI, states.SRC_RUNNER}
    assert states.EV_RELAUNCH in [e["name"] for e in events]
    cloud.assert_clean()


def test_cancel_force_destroys_without_final_sync(cloud):
    result = cloud.submit("e2e-force", mode="hang")
    job_id = result.job_id
    cloud.wait_for(
        lambda: (cloud.heartbeat(job_id) or {}).get("step") == "run", "the run step to start"
    )
    if old_python() is not None:  # runner 本身也跑在 3.8/3.9 上
        runner_pid = (
            cloud.backend.work_roots[result.instance.id] / "run" / "runner.pid"
        ).read_text()
        command = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", runner_pid.strip()],
            capture_output=True,
            text=True,
        ).stdout
        assert command.startswith(cloud.python), command
    view = cloud.orch.cancel(job_id, force=True)
    assert view.phase == states.CANCELLED
    status = cloud.status_doc(job_id)
    assert (status["phase"], status["source"]) == (states.CANCELLED, states.SRC_CLI)
    assert cloud.orch.wait(job_id) == states.CANCELLED
    assert not cloud.store.exists(layout.secrets_key(job_id, 1))
    cloud.assert_clean(crashed=(result.instance.id,))


def test_zombie_old_attempt_is_fenced_by_the_retry(cloud):
    """平台说旧实例已销毁，但它的 runner 还活着：retry 把 attempt 推进到 2 之后，旧 runner
    读到更大的 attempt 就停下，不再写 status / heartbeat / output / ckpt，新 attempt 正常完成。"""
    result = cloud.submit("e2e-fence", mode="resume")
    job_id = result.job_id
    first = result.instance.id
    cloud.wait_for(
        lambda: cloud.store.exists(layout.ckpt_prefix(job_id) + "state.json"),
        "attempt 1 to sync its checkpoint",
    )
    cloud.backend.zombies.add(first)
    cloud.clock.offset = (cloud.config.lost_after_min + 1) * 60  # 看起来失联了
    retried = cloud.orch.retry(job_id)
    cloud.clock.offset = 0.0
    assert retried.attempt == 2 and first not in cloud.backend.instances
    assert cloud.orch.wait(job_id) == states.SUCCEEDED, cloud.diagnostics()

    root = cloud.backend.work_roots[first]
    cloud.wait_for(lambda: (root / "run" / "destroyed.json").exists(), "the old runner to stop")
    fenced = [e for e in cloud.events(job_id) if e["name"] == states.EV_ATTEMPT_FENCED]
    assert fenced and all(e["source"] == states.SRC_RUNNER and e["attempt"] == 1 for e in fenced)
    status = cloud.status_doc(job_id)
    assert (status["attempt"], status["phase"]) == (2, states.SUCCEEDED)
    assert cloud.heartbeat(job_id)["attempt"] == 2
    assert cloud.info(job_id, 2)["ckpt"] == {"epoch": 1, "attempt": "1"}
    assert "发现更新的 attempt 2" in (root / "logs" / "attempt-1.log").read_text()
    cloud.assert_clean()
