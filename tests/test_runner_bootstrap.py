"""Bootstrap end-to-end: the real command from RunnerBundler, a real bundle, file:// URLs.

The bundle carries the fake rclone in place of the real binary; the runner must find it
inside the extracted bundle (DGPU_RCLONE and PYTHONPATH are not set), which proves the
runner really runs from the bundle. `python3` on PATH is Python 3.8/3.9 when available.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from dgpu import bundle, layout, states
from dgpu.bundle import RunnerBundler
from dgpu.runner import procs
from test_runner_harness import CONTAINER_ID, CONTAINER_KEY, Harness, old_python


def _no_fetch(url):
    raise AssertionError(f"tests must not download rclone: {url}")


class Boot:
    def __init__(self, tmp_path, backend="vast", run="echo started; sleep 300"):
        self.h = Harness(tmp_path, backend=backend)
        h = self.h
        fake = (h.bin / "rclone").read_bytes()
        cache = tmp_path / "cache"
        target = cache / (f"rclone-{bundle.RCLONE_VERSION}-linux-amd64") / "rclone"
        target.parent.mkdir(parents=True)
        target.write_bytes(fake)
        target.chmod(0o755)
        self.bundler = RunnerBundler(
            cache_dir=cache,
            fetch=_no_fetch,
            rclone_binary_sha256=hashlib.sha256(fake).hexdigest(),
        )
        self.bundle_path, self.sha = self.bundler.build()
        h.prepare(run=run)
        self.launch_src = tmp_path / "launch-src.json"
        os.replace(h.work / "launch.json", self.launch_src)
        python = old_python() or sys.executable
        wrapper = h.bin / "python3"
        wrapper.write_text(f'#!/bin/sh\nexec "{python}" "$@"\n')
        wrapper.chmod(0o755)
        self.env = h.env()
        for key in ("DGPU_RCLONE", "PYTHONPATH"):
            self.env.pop(key)

    def command(self, sha=None):
        return self.bundler.bootstrap_command(
            bundle_url=self.bundle_path.as_uri(),
            bundle_sha256=sha or self.sha,
            launch_url=self.launch_src.as_uri(),
        )

    def run(self, command, timeout=60):
        return subprocess.run(
            ["bash", "-c", command],
            cwd=self.h.tmp,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@pytest.fixture
def boot(tmp_path):
    b = Boot(tmp_path)
    yield b
    b.h.cleanup()


def test_bootstrap_starts_the_runner_once_and_a_second_run_does_nothing(boot):
    h = boot.h
    first = boot.run(boot.command())
    assert first.returncode == 0, first.stderr
    h.wait_running()
    runner = h.pid("runner")
    bundle_dir = h.work / (f"bundle-{boot.sha}")
    assert (bundle_dir / "bin" / "rclone").exists()
    assert not (h.work / "bundle.tgz").exists()
    assert (h.work / "launch.json").stat().st_mode & 0o777 == 0o600
    extracted_at = bundle_dir.stat().st_mtime_ns

    second = boot.run(boot.command(), timeout=30)
    assert second.returncode == 0
    assert "already active" in second.stderr
    assert h.server.calls == [], "the 'already active' path must never self-destroy"
    assert h.pid("runner") == runner and procs.pid_alive(runner)
    assert bundle_dir.stat().st_mtime_ns == extracted_at

    used = {os.path.realpath(call["argv0"]) for call in h.calls()}
    assert used == {os.path.realpath(bundle_dir / "bin" / "rclone")}, "runs from the bundle"

    h.set_attempt(cancel=True)
    h.wait_done()
    assert h.status()["phase"] == states.CANCELLED, h.diagnostics()
    assert len(h.server.calls) == 1


def test_bootstrap_refuses_a_bundle_with_the_wrong_checksum(boot):
    h = boot.h
    boot.env["DGPU_BOOT_BACKOFF_S"] = "0.001"
    result = boot.run(boot.command(sha="0" * 64))
    assert result.returncode == 1
    assert "giving up" in result.stderr
    assert not list(h.work.glob("bundle-*"))
    assert h.pid("runner") is None
    fd = os.open(h.work / "boot.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the lock was released
    finally:
        os.close(fd)
    # F1: no runner will ever start here, so the bootstrap destroys the instance itself
    _assert_self_destroyed(h, result)


def test_bootstrap_on_autodl_stays_in_the_foreground(tmp_path):
    b = Boot(tmp_path, backend="autodl", run="sleep 0.5; echo done > $DGPU_OUTPUT/done")
    try:
        result = b.run(b.command())
        assert result.returncode == 0, b.h.diagnostics()
        # when cmd returns the container is gone: everything must already be finished
        assert b.h.status()["phase"] == states.SUCCEEDED
        for name in ("runner", "watchdog"):
            assert not procs.pid_alive(b.h.pid(name))
        assert b.h.remote_files(layout.output_prefix(b.h.job_id)) == ["done"]
        assert b.h.server.calls == []
    finally:
        b.h.cleanup()


# ---------------------------------------------------------------- F1: give up => self-destroy


def _assert_self_destroyed(h, result):
    """Exactly one DELETE of this container with its own key; the key never reaches the logs."""
    assert result.returncode == 1
    assert len(h.server.calls) == 1, h.server.calls
    (call,) = h.server.calls
    assert call["path"] == f"/api/v0/instances/{CONTAINER_ID}/"
    assert call["auth"] == f"Bearer {CONTAINER_KEY}"
    assert "self-destroy of instance" in result.stderr and "accepted" in result.stderr
    log = (h.work / "logs" / "bootstrap.log").read_text()
    assert "giving up before the runner started" in log
    for text in (result.stderr, log):
        assert CONTAINER_KEY not in text
    assert h.pid("runner") is None


class _Files(ThreadingHTTPServer):
    """Serves a directory over HTTP; missing files are 404 (like a deleted presigned object)."""

    def __init__(self, root):
        handler = functools.partial(_QuietHandler, directory=str(root))
        super().__init__(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def url(self, name):
        return f"http://127.0.0.1:{self.server_address[1]}/{name}"

    def stop(self):
        self.shutdown()
        self.server_close()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_bootstrap_self_destroys_when_the_launch_document_is_gone(boot, tmp_path):
    """The late instance of an unclear create: its launch document was deleted by the CLI. The
    bundle still downloads, the launch document answers 404: give up after 3 tries and destroy
    the instance with the container key (no storage credentials needed)."""
    h = boot.h
    served = tmp_path / "served"
    served.mkdir()
    shutil.copy(boot.bundle_path, served / "bundle.tar.gz")
    files = _Files(served)
    try:
        boot.env["DGPU_BOOT_BACKOFF_S"] = "0.001"
        command = boot.bundler.bootstrap_command(
            bundle_url=files.url("bundle.tar.gz"),
            bundle_sha256=boot.sha,
            launch_url=files.url("secrets-1.json"),  # deleted
        )
        result = boot.run(command)
    finally:
        files.stop()
    _assert_self_destroyed(h, result)
    assert "HTTP Error 404" in result.stderr and "try 3/10" in result.stderr
    assert "try 4/10" not in result.stderr  # a deleted object does not come back: stop early
    assert (h.work / f"bundle-{boot.sha}").is_dir()


def test_bootstrap_self_destroys_when_the_bundle_code_cannot_be_imported(tmp_path):
    """A Python error in the shipped code (e.g. a too old python3) is caught by the bootstrap."""
    package = tmp_path / "pkg" / "dgpu"
    shutil.copytree(Path(bundle.__file__).parent, package, ignore=shutil.ignore_patterns("*.pyc"))
    (package / "runner" / "main.py").write_text("def broken(:\n")
    b = Boot(tmp_path / "job")
    try:
        b.bundler.package_dir = package
        b.bundle_path, b.sha = b.bundler.build()
        result = b.run(b.command())
        _assert_self_destroyed(b.h, result)
        assert "SyntaxError" in result.stderr
    finally:
        b.h.cleanup()


def test_bootstrap_on_autodl_gives_up_by_exiting(tmp_path):
    b = Boot(tmp_path, backend="autodl")
    try:
        b.env["DGPU_BOOT_BACKOFF_S"] = "0.001"
        result = b.run(b.command(sha="0" * 64))
        assert result.returncode == 1
        assert "no Vast container key" in result.stderr
        assert b.h.server.calls == []
    finally:
        b.h.cleanup()


def test_bootstrap_self_destroy_retries_a_failing_api_and_stays_bounded(boot):
    h = boot.h
    h.server.always_fail = True
    tuning = json.loads(boot.env["DGPU_RUNNER_TUNING"])
    tuning.update(destroy_attempts=3, destroy_interval_s=0.01)
    boot.env["DGPU_RUNNER_TUNING"] = json.dumps(tuning)
    boot.env["DGPU_BOOT_BACKOFF_S"] = "0.001"
    result = boot.run(boot.command(sha="0" * 64))
    assert result.returncode == 1
    assert len(h.server.calls) == 3
    assert "self-destroy try 3 failed" in result.stderr
    assert CONTAINER_KEY not in result.stderr


@pytest.mark.parametrize("backend", ["vast", "autodl"])
def test_boot_gives_up_before_the_runner_starts(tmp_path, backend):
    """boot itself (bundle already there) fails before starting the runner: the work root is not
    usable. Nobody else would clean up, so it self-destroys on Vast and just exits on AutoDL."""
    h = Harness(tmp_path, backend=backend)
    try:
        not_a_dir = tmp_path / "file"
        not_a_dir.write_text("x")
        result = subprocess.run(
            [sys.executable, "-m", "dgpu.runner.boot", "--work-root", str(not_a_dir)],
            cwd=tmp_path,
            env=h.env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 1
        assert "giving up before the runner started" in result.stderr
        if backend == "vast":
            assert [c["path"] for c in h.server.calls] == [f"/api/v0/instances/{CONTAINER_ID}/"]
            assert "self-destroy accepted" in result.stderr
        else:
            assert h.server.calls == []
        assert CONTAINER_KEY not in result.stderr
    finally:
        h.cleanup()
