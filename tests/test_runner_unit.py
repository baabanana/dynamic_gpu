"""Unit tests for the runner's building blocks (no network, injectable clocks)."""

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

import pytest

from dgpu import layout, states
from dgpu.runner import archive, gpu, procs
from dgpu.runner.destroy import (
    DestroyResult,
    SelfDestroyer,
    read_container_credentials,
)
from dgpu.runner.logsink import LogSink, Pump
from dgpu.runner.main import Runner
from dgpu.runner.rclone import Rclone, RcloneError
from dgpu.runner.util import Paths, load_tuning, runner_settings, scrubbed
from dgpu.testing import TEST_PROFILE
from test_runner_harness import Harness

STORE = {
    "provider": TEST_PROFILE.provider,
    "endpoint": TEST_PROFILE.endpoint,
    "region": TEST_PROFILE.region,
    "bucket": TEST_PROFILE.bucket,
    "access_key_id": TEST_PROFILE.access_key_id,
    "secret_access_key": TEST_PROFILE.secret_access_key,
}


# ---------------------------------------------------------------- safe extraction


def _tar(path, entries):
    with tarfile.open(path, "w:gz") as tar:
        for info, data in entries:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)


def _file(name, data=b"x", mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    return info, data


def _link(name, target, kind=tarfile.SYMTYPE):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info, None


def test_safe_extract_accepts_normal_archive_and_strips_setuid(tmp_path):
    archive_path = tmp_path / "code.tar.gz"
    _tar(
        archive_path,
        [
            _file("./train.py", b"print(1)\n"),
            _file("pkg/mod.py"),
            _file("run.sh", b"#!/bin/sh\n", mode=0o4755),
            _link("latest.py", "train.py"),
        ],
    )
    dest = tmp_path / "out"
    assert archive.safe_extract(str(archive_path), str(dest)) == 4
    assert (dest / "train.py").read_bytes() == b"print(1)\n"
    assert (dest / "pkg" / "mod.py").exists()
    assert os.readlink(dest / "latest.py") == "train.py"
    mode = (dest / "run.sh").stat().st_mode
    assert mode & 0o4000 == 0 and mode & 0o100


@pytest.mark.parametrize(
    "entry",
    [
        _file("/etc/evil"),
        _file("../evil"),
        _file("a/../../evil"),
        _link("l", "/etc/passwd"),
        _link("l", "../../outside"),
        _link("h", "../x", kind=tarfile.LNKTYPE),
    ],
    ids=["absolute", "dotdot", "nested-dotdot", "abs-symlink", "escaping-symlink", "hardlink"],
)
def test_safe_extract_rejects_unsafe_entries(tmp_path, entry):
    archive_path = tmp_path / "bad.tar.gz"
    _tar(archive_path, [_file("ok.txt"), entry])
    with pytest.raises(archive.UnsafeArchive):
        archive.safe_extract(str(archive_path), str(tmp_path / "out"))
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "out" / "ok.txt").exists(), "nothing is written on rejection"


def test_safe_extract_rejects_device_files(tmp_path):
    info = tarfile.TarInfo("fifo")
    info.type = tarfile.FIFOTYPE
    archive_path = tmp_path / "bad.tar.gz"
    _tar(archive_path, [(info, None)])
    with pytest.raises(archive.UnsafeArchive):
        archive.safe_extract(str(archive_path), str(tmp_path / "out"))


# ---------------------------------------------------------------- log sink and pump


def test_log_sink_truncates_once_and_tracks_child_output(tmp_path):
    fired = []
    sink = LogSink(str(tmp_path / "a.log"), 100, on_truncate=fired.append)
    sink.note("runner line")
    assert sink.last_output_mono is None, "runner notes are not training output"
    sink.child_output(b"x" * 60)
    first = sink.last_output_mono
    assert first is not None
    sink.child_output(b"y" * 60)
    sink.child_output(b"z" * 60)
    sink.close()
    data = (tmp_path / "a.log").read_bytes()
    assert data.count(b"x") == 60 and b"z" not in data
    assert fired == [100]
    assert sink.truncated and sink.dropped_bytes >= 60


def test_pump_drains_a_chatty_child_beyond_the_cap(tmp_path):
    sink = LogSink(str(tmp_path / "a.log"), 64 * 1024)
    code = "import sys\nfor _ in range(4096): sys.stdout.write('y' * 1023 + '\\n')\n"
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    pump = Pump(proc.stdout, sink)
    pump.start()
    assert proc.wait(timeout=20) == 0
    pump.join(5)
    assert not pump.is_alive()
    sink.close()
    assert (tmp_path / "a.log").stat().st_size < 70 * 1024
    assert sink.dropped_bytes > 3 * 1024 * 1024


def test_log_sink_write_errors_never_raise(tmp_path):
    sink = LogSink(str(tmp_path / "a.log"), 1000)
    sink._file.close()  # simulate a broken file (e.g. ENOSPC)
    sink.child_output(b"still drained")
    assert sink.last_output_mono is not None


# ---------------------------------------------------------------- self-destroy


class FakeHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, method, url, headers, timeout):
        self.requests.append((method, url, dict(headers)))
        response = self.responses.pop(0) if self.responses else (200, b'{"success": true}')
        if isinstance(response, Exception):
            raise response
        return response


def _destroyer(http, sleeps, backend="vast", creds=("777", "ckey-123")):
    return SelfDestroyer(
        backend,
        http=http,
        credentials=lambda: creds,
        sleep=sleeps.append,
        attempts=10,
        interval_s=30.0,
        api_base="https://console.vast.ai/",
    )


def test_vast_destroy_uses_container_key_and_retries_until_accepted():
    http = FakeHttp([(500, b"oops"), OSError("timeout"), (200, b'{"success": true}')])
    sleeps = []
    result = _destroyer(http, sleeps).destroy()
    assert result.ok and result.attempts == 3 and result.method == "vast_rest"
    assert sleeps == [30.0, 30.0]
    method, url, headers = http.requests[0]
    assert method == "DELETE"
    assert url == "https://console.vast.ai/api/v0/instances/777/"
    assert headers["Authorization"] == "Bearer ckey-123"


def test_vast_destroy_gives_up_after_ten_attempts():
    http = FakeHttp([(503, b"down")] * 10)
    sleeps = []
    result = _destroyer(http, sleeps).destroy()
    assert not result.ok and result.attempts == 10
    assert len(http.requests) == 10 and sleeps == [30.0] * 9
    assert "503" in result.error


@pytest.mark.parametrize(
    "response",
    [(404, b"not found"), (200, b'{"success": false, "msg": "nope"}'), (401, b"bad key ckey-123")],
    ids=["404", "success-false", "unauthorized"],
)
def test_vast_destroy_does_not_mistake_errors_for_success(response):
    http = FakeHttp([response] * 10)
    result = _destroyer(http, []).destroy()
    assert not result.ok
    assert "ckey-123" not in result.error, "the container key is redacted"


def test_vast_destroy_without_container_credentials_fails_without_calling():
    http = FakeHttp([])
    result = _destroyer(http, [], creds=(None, None)).destroy()
    assert not result.ok and http.requests == []


def test_autodl_destroy_is_process_exit_and_calls_nothing():
    http = FakeHttp([])
    result = _destroyer(http, [], backend="autodl").destroy()
    assert result.ok and result.method == "process_exit" and http.requests == []
    assert not _destroyer(http, [], backend="other").destroy().ok


def test_container_credentials_fallbacks(tmp_path):
    proc_env = tmp_path / "environ"
    proc_env.write_bytes(b"PATH=/bin\0CONTAINER_ID=11\0CONTAINER_API_KEY=k11\0")
    etc_env = tmp_path / "environment"
    etc_env.write_text("CONTAINER_ID=\"22\"\nexport CONTAINER_API_KEY='k22'\n")
    missing = str(tmp_path / "missing")
    assert read_container_credentials(
        {"CONTAINER_ID": "1", "CONTAINER_API_KEY": "k"}, missing, missing
    ) == ("1", "k")
    assert read_container_credentials({}, str(proc_env), str(etc_env)) == ("11", "k11")
    assert read_container_credentials({}, missing, str(etc_env)) == ("22", "k22")
    assert read_container_credentials({}, missing, missing) == (None, None)


# ---------------------------------------------------------------- rclone wrapper (fake rclone)


@pytest.fixture
def fake(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.cleanup()


def _rclone(h: Harness, **kwargs) -> Rclone:
    return Rclone(
        str(h.bin / "rclone"),
        STORE,
        str(h.work / "rclone.conf"),
        base_env=h.env(),
        poll_s=0.02,
        **kwargs,
    )


def test_rclone_passes_credentials_only_through_the_environment(fake):
    rclone = _rclone(fake)
    rclone.rcat("dgpu/x.json", b'{"a": 1}')
    assert rclone.cat("dgpu/x.json") == b'{"a": 1}'
    calls = fake.calls()
    assert [c["op"] for c in calls] == ["rcat", "cat"]
    assert all(c["has_creds"] and not c["secret_in_argv"] for c in calls)
    assert TEST_PROFILE.secret_access_key not in json.dumps([c["args"] for c in calls])
    assert "RCLONE_CONFIG_DGPU_SECRET_ACCESS_KEY" not in os.environ


def test_rclone_missing_objects(fake):
    rclone = _rclone(fake)
    assert rclone.cat("dgpu/missing.json") is None  # S3 style: empty output, exit 0
    fake.add_rule("cat", "other.json", rc=4)
    assert rclone.cat("dgpu/other.json") is None  # exit 4: file not found
    assert rclone.lsjson("dgpu/nothing/") == []
    assert rclone.size("dgpu/nothing/") == (0, 0)
    fake.add_rule("cat", "broken.json")
    with pytest.raises(RcloneError):
        rclone.cat("dgpu/broken.json")


def test_rclone_timeout_and_interrupt_kill_a_hung_call(fake):
    rclone = _rclone(fake)
    fake.add_rule("cat", "slow.json", mode="hang")
    started = time.monotonic()
    with pytest.raises(RcloneError) as err:
        rclone.cat("dgpu/slow.json", timeout=0.5)
    assert err.value.timed_out and time.monotonic() - started < 5
    flag = threading.Event()
    threading.Timer(0.3, flag.set).start()
    with pytest.raises(RcloneError) as err:
        rclone.cat("dgpu/slow.json", timeout=30, interrupt=flag.is_set)
    assert not err.value.timed_out


def test_rclone_abort_kills_in_flight_and_guard_blocks_new_writes(fake):
    rclone = _rclone(fake)
    fake.add_rule("rcat", "slow.json", mode="hang")
    errors = []

    def write():
        try:
            rclone.rcat("dgpu/slow.json", b"{}", timeout=30)
        except RcloneError as exc:
            errors.append(exc)

    worker = threading.Thread(target=write)
    worker.start()
    time.sleep(0.5)
    assert rclone.abort() == 1
    worker.join(10)
    assert errors and errors[0].aborted
    with pytest.raises(RcloneError) as err:
        rclone.rcat("dgpu/x.json", b"{}", guard=lambda: False)
    assert err.value.blocked
    assert [c for c in fake.calls() if c["args"] and c["args"][0].endswith("x.json")] == []


def test_fake_rclone_records_writes_to_frozen_prefixes(fake):
    """Sanity check of the fencing test's instrument."""
    rclone = _rclone(fake)
    fake.freeze([fake.bucket + "/dgpu/frozen/"])
    rclone.rcat("dgpu/frozen/status.json", b"{}")
    rclone.rcat("dgpu/free/status.json", b"{}")
    assert [v["key"] for v in fake.violations()] == [fake.bucket + "/dgpu/frozen/status.json"]


# ---------------------------------------------------------------- small pieces


def test_tuning_overrides_are_validated():
    tuning = load_tuning(
        {"DGPU_RUNNER_TUNING": json.dumps({"tick_s": "0.1", "destroy_attempts": 2, "bogus": 1})}
    )
    assert tuning["tick_s"] == 0.1 and tuning["destroy_attempts"] == 2 and "bogus" not in tuning
    assert load_tuning({"DGPU_RUNNER_TUNING": "{not json"})["tick_s"] == 0.5
    assert load_tuning({})["destroy_interval_s"] == 30.0


def test_runner_settings_default_and_accept_fractions():
    settings = runner_settings({"runner": {"final_sync_timeout_min": 0.05, "max_log_mb": "x"}})
    assert settings["final_sync_timeout_min"] == 0.05
    assert settings["max_log_mb"] == 200.0 and settings["heartbeat_interval_s"] == 60.0


def test_scrubbed_prefixes():
    for name in ("RCLONE_CONFIG_DGPU_TYPE", "CONTAINER_API_KEY", "AWS_SECRET", "R2_KEY", "aws_x"):
        assert scrubbed(name)
    for name in ("PATH", "CONTAINER_ID", "DGPU_DATA", "MY_R2"):
        assert not scrubbed(name)


def test_gpu_parse_and_unknown_utilization():
    gpus = gpu.parse("0, RTX 3090, 37, 24576\n1, RTX 3090, [N/A], 24576\n")
    assert [g["util"] for g in gpus] == [37.0, None]
    assert gpu.max_utilization(gpus) is None
    assert gpu.max_utilization(gpus[:1]) == 37.0
    assert gpu.max_utilization([]) is None
    assert gpu.query(binary="/nonexistent/nvidia-smi")[0] is None


def test_terminate_escalates_to_sigkill_for_the_whole_tree():
    script = (
        "import os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "grandchild = 'import os,time; os.setsid(); time.sleep(60)'\n"
        "kid = subprocess.Popen([sys.executable, '-c', grandchild])\n"
        "print(kid.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = procs.start_session([sys.executable, "-c", script], os.getcwd(), dict(os.environ))
    grandchild = int(proc.stdout.readline())
    assert procs.terminate(proc.pid, proc.pid, 0.3, reap=proc.poll) == "killed"
    proc.wait(5)
    end = time.monotonic() + 5
    while procs.pid_alive(grandchild) and time.monotonic() < end:
        time.sleep(0.05)
    assert not procs.pid_alive(grandchild)
    assert procs.terminate(proc.pid, proc.pid, 0.3, reap=proc.poll) == "exited"


def test_terminate_lets_a_cooperative_process_exit_on_sigterm():
    proc = procs.start_session(["sleep", "30"], os.getcwd(), dict(os.environ))
    assert procs.terminate(proc.pid, proc.pid, 5.0, reap=proc.poll) == "terminated"
    assert proc.wait(5) == -signal.SIGTERM


# ---------------------------------------------------------------- runner in-process


class CountingDestroyer:
    def __init__(self):
        self.calls = 0

    def destroy(self):
        self.calls += 1
        return DestroyResult(True, 1, "fake", None)


def _in_process_runner(h: Harness, destroyer) -> Runner:
    tuning = load_tuning({"DGPU_RUNNER_TUNING": json.dumps(h.tuning)})
    return Runner(
        Paths(str(h.work)),
        tuning=tuning,
        environ=h.env(),
        destroyer=destroyer,
        gpu_query=lambda timeout: ([{"index": "0", "name": "fake", "util": 50.0}], ""),
    )


def test_unhandled_exception_still_finishes_and_destroys(fake):
    fake.prepare(run="true")
    destroyer = CountingDestroyer()
    runner = _in_process_runner(fake, destroyer)

    def boom():
        raise RuntimeError("bug in restore")

    runner._restore_ckpt = boom
    assert runner.run() == states.EXIT_FAILED
    status = fake.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_RUNNER_CRASHED)
    assert "bug in restore" in status["message"]
    assert destroyer.calls == 1
    assert fake.get(layout.log_key(fake.job_id, 1)) is not None


def test_error_inside_teardown_still_destroys(fake):
    fake.prepare(run="true")
    destroyer = CountingDestroyer()
    runner = _in_process_runner(fake, destroyer)

    def boom():
        raise RuntimeError("teardown bug")

    runner._final_sync = boom
    runner.run()
    assert destroyer.calls == 1


def test_invalid_launch_document_still_destroys(fake):
    fake.prepare(run="true")
    (fake.work / "launch.json").write_text(json.dumps({"version": 99}))
    destroyer = CountingDestroyer()
    runner = _in_process_runner(fake, destroyer)
    assert runner.run() == states.EXIT_FAILED
    assert destroyer.calls == 1
    assert runner.outcome.reason == states.REASON_PREFLIGHT
