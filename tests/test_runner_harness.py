"""Harness for end-to-end runner tests (no tests here).

Runs the real entry points (`python -m dgpu.runner.boot`, which starts the watchdog and the
runner) against the fake rclone / nvidia-smi from test_runner_fakes.py and a local HTTP
server standing in for the Vast API. All intervals are shrunk through spec.json (runner
settings) and DGPU_RUNNER_TUNING so a whole job takes about a second.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dgpu import layout, states
from dgpu.contracts import (
    DataSource,
    GpuSpec,
    JobSpec,
    Limits,
    RunnerSettings,
    build_launch_doc,
    build_remote_spec,
)
from dgpu.runner import procs
from dgpu.testing import TEST_PROFILE

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CONTAINER_ID = "4242"
CONTAINER_KEY = "container-key-not-a-real-secret"

FAST_TUNING = {
    "tick_s": 0.05,
    "rclone_poll_s": 0.02,
    "retry_backoff_s": 0.05,
    "retry_backoff_max_s": 0.2,
    "startup_attempts": 3,
    "status_attempts": 3,
    "small_op_timeout_s": 10.0,
    "final_sync_backoff_s": 0.1,
    "sync_failed_event_interval_s": 60.0,
    "nvidia_smi_timeout_s": 5.0,
    "min_free_margin_mb": 1.0,
    "lingering_grace_s": 0.5,
    "thread_join_s": 5.0,
    "teardown_budget_s": 20.0,
    "destroy_attempts": 3,
    "destroy_interval_s": 0.05,
    "destroy_http_timeout_s": 5.0,
    "watchdog_poll_s": 0.1,
    "watchdog_margin_s": 2.0,
    "watchdog_spec_wait_s": 30.0,
    "watchdog_sync_timeout_s": 5.0,
    "boot_watchdog_wait_s": 30.0,
}

FAST_RUNNER = {
    "sync_interval_s": 0.3,
    "heartbeat_interval_s": 0.2,
    "stall_timeout_min": 0,
    "final_sync_timeout_min": 0.05,
    "term_grace_s": 1.0,
    "max_log_mb": 5,
    "data_sync_retries": 2,
}

WRAPPER = """#!{python} -S
import sys
sys.path.insert(0, {tests!r})
import test_runner_fakes
sys.exit(test_runner_fakes.{func}(sys.argv[1:]))
"""


def old_python() -> str | None:
    """A Python 3.8 or 3.9 interpreter found by `uv python find`, if any (cached)."""
    if "python" not in _OLD_PYTHON:
        _OLD_PYTHON["python"] = _find_old_python()
    return _OLD_PYTHON["python"]


_OLD_PYTHON: dict[str, str | None] = {}


def _find_old_python() -> str | None:
    for version in ("3.8", "3.9"):
        try:
            found = subprocess.run(
                ["uv", "python", "find", version], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return None
        path = found.stdout.strip()
        if found.returncode == 0 and path:
            check = subprocess.run(
                [path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True,
                text=True,
            )
            if check.stdout.strip() == version:
                return path
    return None


class DestroyServer:
    """Stands in for console.vast.ai: records DELETE /api/v0/instances/<id>/ calls."""

    def __init__(self):
        self.calls = []
        self.fail_first = 0
        self.always_fail = False
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_DELETE(self):  # noqa: N802 - http.server naming
                with outer.lock:
                    outer.calls.append(
                        {
                            "ts": time.time(),
                            "path": self.path,
                            "auth": self.headers.get("Authorization"),
                        }
                    )
                    failing = outer.always_fail or len(outer.calls) <= outer.fail_first
                body = json.dumps({"success": not failing}).encode()
                self.send_response(500 if failing else 200)
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
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        backend: str = "vast",
        job_id: str = "job-test-1",
        attempt: int = 1,
        python: str | None = None,
    ):
        self.tmp = tmp_path
        self.work = tmp_path / "work"
        self.remote = tmp_path / "remote"
        self.control = self.remote / ".control"
        self.bin = tmp_path / "bin"
        self.home = tmp_path / "home"
        self.bucket = TEST_PROFILE.bucket
        self.job_id = job_id
        self.attempt = attempt
        self.backend = backend
        self.python = python or sys.executable
        for path in (self.work, self.control, self.bin, self.home):
            path.mkdir(parents=True, exist_ok=True)
        self._write_wrapper("rclone", "fake_rclone_main")
        self._write_wrapper("nvidia-smi", "fake_nvidia_smi_main")
        self.server = DestroyServer()
        self.tuning = dict(FAST_TUNING, vast_api_base=self.server.url)
        self.extra_env: dict[str, str] = {}
        self.boot_proc: subprocess.Popen | None = None
        self._extra_pids: list[int] = []

    def _write_wrapper(self, name: str, func: str) -> None:
        path = self.bin / name
        path.write_text(WRAPPER.format(python=sys.executable, tests=str(TESTS_DIR), func=func))
        path.chmod(0o755)

    # ------------------------------------------------------------ remote objects

    def remote_path(self, key: str) -> Path:
        return self.remote / self.bucket / key

    def put(self, key: str, data: bytes | str) -> None:
        path = self.remote_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".harness-tmp")
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes | None:
        path = self.remote_path(key)
        return path.read_bytes() if path.is_file() else None

    def get_json(self, key: str) -> dict | None:
        data = self.get(key)
        return json.loads(data) if data else None

    def status(self) -> dict | None:
        return self.get_json(layout.status_key(self.job_id))

    def heartbeat(self) -> dict | None:
        return self.get_json(layout.heartbeat_key(self.job_id))

    def events(self) -> list[dict]:
        folder = self.remote_path(layout.events_prefix(self.job_id))
        if not folder.is_dir():
            return []
        return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]

    def event_names(self) -> list[str]:
        return [event["name"] for event in self.events()]

    def log_text(self) -> str:
        data = self.get(layout.log_key(self.job_id, self.attempt))
        return data.decode("utf-8", "replace") if data else ""

    def remote_files(self, prefix: str) -> list[str]:
        root = self.remote_path(prefix)
        if not root.is_dir():
            return []
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    def calls(self) -> list[dict]:
        path = self.control / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def status_history(self) -> list[str]:
        key = layout.status_key(self.job_id)
        phases = []
        for call in self.calls():
            if call["op"] == "rcat" and call["args"][0].endswith(key) and call["rc"] == 0:
                phases.append(json.loads(call["payload"])["phase"])
        return phases

    def violations(self) -> list[dict]:
        path = self.control / "violations.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------ job setup

    def prepare(
        self,
        run: str = "true",
        setup: str | None = None,
        data: tuple[tuple[str, str], ...] = (),
        env: dict[str, str] | None = None,
        max_hours: float = 0.01,
        code: dict[str, str] | None = None,
        attempt_doc: dict | None = None,
        **runner: float,
    ) -> dict:
        """Write spec.json, code.tar.gz, attempt.json and the local launch.json."""
        job = JobSpec(
            version=1,
            name="test",
            backend=self.backend,
            images={self.backend: "img:1"},
            code=Path("/unused"),
            run=run,
            setup=setup,
            limits=Limits(max_hours=max_hours, max_usd_per_hour=1.0, max_cny_per_hour=1.0),
            gpu=GpuSpec(names=("RTX 3090",)),
            env=dict(env or {}),
        )
        sources = [DataSource(name, self.bucket, prefix) for name, prefix in data]
        spec = build_remote_spec(
            job, job_id=self.job_id, backend=self.backend, data=sources, runner=RunnerSettings()
        )
        spec["runner"].update(FAST_RUNNER)
        spec["runner"].update(runner)
        self.put(layout.spec_key(self.job_id), json.dumps(spec))
        self.write_code(code or {"train.py": "print('hello')\n"})
        doc = attempt_doc or states.new_attempt_doc(
            self.job_id, self.attempt, self.backend, layout.label(self.job_id, self.attempt)
        )
        self.put(layout.attempt_key(self.job_id), json.dumps(doc))
        self.write_launch()
        return spec

    def write_code(self, files: dict[str, str]) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, text in sorted(files.items()):
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        self.put(layout.code_key(self.job_id), buf.getvalue())

    def write_launch(self) -> None:
        doc = build_launch_doc(
            job_id=self.job_id, attempt=self.attempt, backend=self.backend, store=TEST_PROFILE
        )
        path = self.work / "launch.json"
        path.write_text(json.dumps(doc))
        path.chmod(0o600)

    def set_attempt(self, attempt: int | None = None, cancel: bool = False) -> None:
        doc = states.new_attempt_doc(
            self.job_id,
            attempt or self.attempt,
            self.backend,
            layout.label(self.job_id, attempt or self.attempt),
        )
        doc["cancel"] = cancel
        self.put(layout.attempt_key(self.job_id), json.dumps(doc))

    # ------------------------------------------------------------ fault injection

    def add_rule(
        self,
        op: str | None,
        match: str,
        mode: str = "fail",
        times: int = -1,
        delay: float = 0.0,
        rc: int = 1,
    ) -> None:
        path = self.control / "rules.json"
        rules = json.loads(path.read_text()) if path.exists() else []
        rules.append(
            {"op": op, "match": match, "mode": mode, "times": times, "delay": delay, "rc": rc}
        )
        path.write_text(json.dumps(rules))

    def clear_rules(self) -> None:
        (self.control / "rules.json").unlink(missing_ok=True)

    def set_offline(self, offline: bool) -> None:
        flag = self.control / "offline"
        if offline:
            flag.write_text("1")
        else:
            flag.unlink(missing_ok=True)

    def freeze(self, prefixes: list[str]) -> None:
        (self.control / "frozen").write_text("\n".join(prefixes) + "\n")

    def set_gpu(self, util: float | None = None, missing: bool = False) -> None:
        if util is not None:
            (self.control / "gpu_util").write_text(str(util))
        flag = self.control / "no_gpu"
        if missing:
            flag.write_text("1")
        else:
            flag.unlink(missing_ok=True)

    # ------------------------------------------------------------ running

    def env(self) -> dict[str, str]:
        drop = ("RCLONE_", "CONTAINER_", "AWS_", "R2_", "DGPU_", "VAST_", "PYTHON")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(drop) and "PROXY" not in key.upper()
        }
        env.update(
            {
                "PATH": "{}{}{}".format(self.bin, os.pathsep, os.environ.get("PATH", "")),
                "HOME": str(self.home),
                "FAKE_RCLONE_ROOT": str(self.remote),
                "DGPU_RCLONE": str(self.bin / "rclone"),
                "DGPU_RUNNER_TUNING": json.dumps(self.tuning),
                "DGPU_WORK_ROOT": str(self.work),
                "PYTHONPATH": str(SRC_DIR),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        if self.backend == "vast":
            env["CONTAINER_ID"] = CONTAINER_ID
            env["CONTAINER_API_KEY"] = CONTAINER_KEY
        env.update(self.extra_env)
        return env

    def start(self, mode: str = "auto") -> subprocess.Popen:
        self.boot_proc = subprocess.Popen(
            [self.python, "-m", "dgpu.runner.boot", "--work-root", str(self.work), "--mode", mode],
            cwd=self.tmp,
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return self.boot_proc

    def run_to_end(self, timeout: float = 30.0, mode: str = "auto") -> None:
        self.start(mode)
        self.wait_done(timeout)

    def pid(self, name: str) -> int | None:
        path = self.work / "run" / (f"{name}.pid")
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def child(self) -> dict | None:
        path = self.work / "run" / "child.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def boot_output(self) -> str:
        if self.boot_proc is None or self.boot_proc.stdout is None:
            return ""
        return self.boot_proc.stdout.read().decode("utf-8", "replace")

    def wait_done(self, timeout: float = 30.0) -> None:
        """Wait until boot, runner and watchdog have all exited."""
        end = time.monotonic() + timeout
        if self.boot_proc is not None:
            try:
                self.boot_proc.wait(max(0.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise AssertionError("boot did not exit\n" + self.diagnostics()) from None
        for name in ("runner", "watchdog"):
            while True:
                pid = self.pid(name)
                if pid is None or not procs.pid_alive(pid):
                    break
                if time.monotonic() > end:
                    raise AssertionError(f"{name} still running\n{self.diagnostics()}")
                time.sleep(0.05)

    def wait_for(self, predicate, timeout: float = 15.0, what: str = "condition"):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = predicate()
            if value:
                return value
            time.sleep(0.03)
        raise AssertionError(f"timed out waiting for {what}\n{self.diagnostics()}")

    def wait_running(self, timeout: float = 15.0) -> None:
        """Wait until the training command (step run) is running."""
        self.wait_for(
            lambda: (self.child() or {}).get("step") == "run", timeout, "the run step to start"
        )

    def diagnostics(self) -> str:
        parts = []
        for name in ("runner", "watchdog"):
            path = self.work / "logs" / (f"{name}.log")
            if path.exists():
                parts.append(f"--- {name}.log ---\n{path.read_text()[-4000:]}")
        log = self.work / "logs" / f"attempt-{self.attempt}.log"
        if log.exists():
            parts.append("--- attempt log ---\n" + log.read_text(errors="replace")[-3000:])
        parts.append(f"--- events --- {self.event_names()}")
        parts.append(f"--- status --- {self.status()}")
        return "\n".join(parts)

    def track_pid(self, pid: int) -> None:
        self._extra_pids.append(pid)

    def cleanup(self) -> None:
        pids = [self.pid("runner"), self.pid("watchdog")] + self._extra_pids
        child = self.child()
        for pid in pids:
            if pid:
                procs.kill_pid(pid, signal.SIGKILL)
        if child and child.get("pgid"):
            try:
                os.killpg(int(child["pgid"]), signal.SIGKILL)
            except OSError:
                pass
        if self.boot_proc is not None and self.boot_proc.poll() is None:
            self.boot_proc.kill()
            self.boot_proc.wait()
        self.server.stop()
