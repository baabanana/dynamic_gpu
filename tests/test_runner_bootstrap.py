"""Bootstrap end-to-end: the real command from RunnerBundler, a real bundle, file:// URLs.

The bundle carries the fake rclone in place of the real binary; the runner must find it
inside the extracted bundle (DGPU_RCLONE and PYTHONPATH are not set), which proves the
runner really runs from the bundle. `python3` on PATH is Python 3.8/3.9 when available.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys

import pytest

from dgpu import bundle, layout, states
from dgpu.bundle import RunnerBundler
from dgpu.runner import procs
from test_runner_harness import Harness, old_python


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
    assert h.server.calls == []


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
