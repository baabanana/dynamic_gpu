"""Regression tests for the review of 2026-11-10, section B (docs/dgpu-review-2026-11-10.md).

Each test names the review item it covers. They run the real boot / runner / watchdog
processes (or the classes in-process) against the fake rclone and nvidia-smi from
test_runner_fakes.py, like test_runner_e2e.py.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from dgpu import layout, states
from dgpu.runner import boot as boot_mod
from dgpu.runner import gpu, procs, syncstate, util
from dgpu.runner.destroy import DestroyResult
from dgpu.runner.events import Events
from dgpu.runner.main import Runner
from dgpu.runner.rclone import Rclone
from dgpu.runner.util import Paths, load_tuning, runner_settings
from dgpu.runner.watchdog import Watchdog
from test_runner_harness import SRC_DIR, Harness

PY = sys.executable


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.cleanup()


def _events(h: Harness, name: str) -> list[dict]:
    return [event for event in h.events() if event["name"] == name]


def _wait_dead(pids, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        alive = [pid for pid in pids if procs.pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if procs.pid_alive(pid)]


class CountingDestroyer:
    def __init__(self):
        self.calls = 0

    def destroy(self):
        self.calls += 1
        return DestroyResult(True, 1, "fake", None)


def _tuning(h: Harness) -> dict:
    return load_tuning({"DGPU_RUNNER_TUNING": json.dumps(h.tuning)})


def _runner(h: Harness, destroyer=None, **kwargs) -> Runner:
    return Runner(
        Paths(str(h.work)),
        tuning=_tuning(h),
        environ=h.env(),
        destroyer=destroyer or CountingDestroyer(),
        gpu_query=kwargs.pop(
            "gpu_query", lambda timeout: ([{"index": "0", "name": "fake", "util": 50.0}], "")
        ),
        **kwargs,
    )


def _watchdog(h: Harness, destroyer) -> Watchdog:
    return Watchdog(
        Paths(str(h.work)),
        tuning=_tuning(h),
        environ=h.env(),
        rclone_binary=str(h.bin / "rclone"),
        destroyer=destroyer,
    )


def _rclone_pids(h: Harness, needle: str) -> list[int]:
    """Running fake-rclone processes whose command line mentions needle (found via ps, so
    the check does not depend on the runner's own registry)."""
    out = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True)
    found = []
    for line in out.stdout.splitlines():
        pid, _, command = line.strip().partition(" ")
        if str(h.bin / "rclone") in command and needle in command and procs.pid_alive(int(pid)):
            found.append(int(pid))
    return found


# ================================================================ B1


def test_b1_same_size_rewrite_after_upload_is_reported_not_uploaded(harness):
    """B1: a checkpoint rewritten with the same size after it was uploaded, then the final
    upload fails: the file is reported in not_uploaded (it used to report 0 files)."""
    run = (
        "printf AAAA > $DGPU_CKPT/last.pt; "
        "while [ ! -f $DGPU_OUTPUT/go ]; do sleep 0.05; done; "
        "sleep 1.1; printf BBBB > $DGPU_CKPT/l.tmp; mv $DGPU_CKPT/l.tmp $DGPU_CKPT/last.pt"
    )
    harness.prepare(run=run, final_sync_timeout_min=0.02)
    harness.start()
    remote = layout.ckpt_prefix(harness.job_id) + "last.pt"
    harness.wait_for(lambda: harness.get(remote) == b"AAAA", what="the first upload")
    harness.add_rule("copy", "/ckpt", mode="fail")
    (harness.work / "output" / "go").write_text("1")
    harness.wait_done()
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_FINAL_SYNC), (
        harness.diagnostics()
    )
    assert status["not_uploaded"] == ["ckpt/last.pt"]
    assert harness.get(remote) == b"AAAA", "storage really has the stale version"
    assert "1 个文件" in status["message"]


def test_b1_sync_record_rules(tmp_path):
    """B1 unit: same size but newer mtime is pending; a corrupt record means every local file
    is pending (never 0); an unlistable directory reports 'unknown' instead of nothing."""
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "a.pt").write_bytes(b"1234")
    (root / "b.pt").write_bytes(b"xy")
    (root / "c.pt.tmp").write_bytes(b"partial")
    before = syncstate.snapshot(str(root), True)
    assert sorted(before) == ["a.pt", "b.pt"]
    syncstate.record_success(str(run), "ckpt", before)
    assert syncstate.pending(str(root), True, str(run), "ckpt") == []
    stat = os.stat(root / "a.pt")
    (root / "a.pt").write_bytes(b"5678")
    os.utime(root / "a.pt", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert syncstate.pending(str(root), True, str(run), "ckpt") == ["ckpt/a.pt"]
    (run / "synced-ckpt.json").write_text("{broken")
    assert syncstate.pending(str(root), True, str(run), "ckpt") == ["ckpt/a.pt", "ckpt/b.pt"]
    if os.geteuid() != 0:
        sub = root / "locked"
        sub.mkdir()
        sub.chmod(0)
        try:
            pending = syncstate.pending(str(root), True, str(run), "ckpt")
        finally:
            sub.chmod(0o700)
        assert len(pending) == 1 and "无法列出" in pending[0]


# ================================================================ B2 / B3


def _started_long_ago(h: Harness) -> None:
    """started.json from an hour ago: the watchdog's hard deadline has already passed."""
    run_dir = h.work / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = {"attempt": h.attempt, "wall": time.time() - 3600}
    (run_dir / "started.json").write_text(json.dumps(doc))


def _fake_runner(h: Harness) -> subprocess.Popen:
    """A live process that holds runner.alive like the runner does, and keeps 'uploading'."""
    code = textwrap.dedent(
        """
        import fcntl, os, sys, time
        fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        print("locked", flush=True)
        while True:
            time.sleep(0.1)  # alive, not stopped: e.g. a slow upload in its teardown
        """
    )
    run_dir = h.work / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [PY, "-c", code, str(run_dir / "runner.alive")],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert proc.stdout.readline().strip() == b"locked"
    (run_dir / "runner.pid").write_text(f"{proc.pid}\n")
    h.track_pid(proc.pid)
    return proc


def _fake_registered_rclone(h: Harness) -> subprocess.Popen:
    """A fake rclone call left running by the runner (registered like Rclone does)."""
    proc = subprocess.Popen(
        [str(h.bin / "rclone"), "copy", str(h.work / "output"), "dgpu:x/y"],
        env=dict(
            h.env(),
            RCLONE_CONFIG_DGPU_TYPE="s3",
            RCLONE_CONFIG_DGPU_ACCESS_KEY_ID="k",
            RCLONE_CONFIG_DGPU_SECRET_ACCESS_KEY="s",
            RCLONE_CONFIG_DGPU_ENDPOINT="e",
        ),
        start_new_session=True,
    )
    h.track_pid(proc.pid)
    procs.register(str(h.work / "run" / "rclone-runner"), proc.pid, str(h.bin / "rclone"))
    return proc


def test_b2_b3_b12_watchdog_hard_deadline_with_live_slow_runner_and_hung_storage(harness):
    """B2/B3/B12 and test gaps: the runner is alive but slow (not SIGSTOPped), storage hangs.
    At the hard deadline the watchdog kills the runner and the rclone it left running, makes
    one bounded sync attempt, reports the files that did not make it (not an empty list),
    and finishes within util.watchdog_teardown_worst_case_s, then self-destroys."""
    harness.tuning.update(
        small_op_timeout_s=1.0,
        teardown_budget_s=2.0,
        watchdog_sync_timeout_s=1.0,
        thread_join_s=0.2,
        destroy_attempts=1,
        destroy_http_timeout_s=1.0,
        watchdog_margin_s=0.2,
    )
    spec = harness.prepare(run="true", max_hours=0.0001, final_sync_timeout_min=0.001)
    (harness.work / "run").mkdir(parents=True, exist_ok=True)
    (harness.work / "run" / "spec.json").write_text(json.dumps(spec))
    (harness.work / "output").mkdir(exist_ok=True)
    (harness.work / "output" / "model.pt").write_text("weights")
    (harness.work / "ckpt").mkdir(exist_ok=True)
    (harness.work / "ckpt" / "last.pt").write_text("ckpt")
    _started_long_ago(harness)
    runner = _fake_runner(harness)
    rclone = _fake_registered_rclone(harness)
    harness.add_rule("copy", "", mode="hang")
    harness.add_rule("copyto", "", mode="hang")
    harness.add_rule("move", "", mode="hang")
    destroyer = CountingDestroyer()
    watchdog = _watchdog(harness, destroyer)
    started = time.monotonic()
    watchdog.run()
    elapsed = time.monotonic() - started
    worst = util.watchdog_teardown_worst_case_s(runner_settings(spec), _tuning(harness))
    assert elapsed <= worst + 3.0, (elapsed, worst)
    assert runner.wait(5) == -signal.SIGKILL
    assert rclone.wait(5) == -signal.SIGKILL, "the runner's rclone was killed (B12)"
    status = harness.status()  # status writes are not hung: rcat has no rule
    assert (status["phase"], status["source"]) == (states.TIMED_OUT, states.SRC_WATCHDOG)
    assert sorted(status["not_uploaded"]) == ["ckpt/last.pt", "output/model.pt"]
    assert "2 个文件" in status["message"]
    assert destroyer.calls == 1
    final = json.loads((harness.work / "run" / "final.json").read_text())
    assert final["status_written"] is True and final["by"] == states.SRC_WATCHDOG


def test_b2_watchdog_hard_deadline_uploads_results_when_storage_works(harness):
    """B2: at the hard deadline the watchdog's one bounded sync uploads output/ and ckpt/."""
    spec = harness.prepare(run="true", max_hours=0.0001, final_sync_timeout_min=0.05)
    (harness.work / "run").mkdir(parents=True, exist_ok=True)
    (harness.work / "run" / "spec.json").write_text(json.dumps(spec))
    (harness.work / "output").mkdir(exist_ok=True)
    (harness.work / "output" / "model.pt").write_text("weights")
    harness.tuning.update(
        watchdog_margin_s=0.1,
        teardown_budget_s=2.0,
        small_op_timeout_s=2.0,
        thread_join_s=0.1,
        destroy_attempts=1,
        destroy_http_timeout_s=1.0,
    )
    _started_long_ago(harness)
    _fake_runner(harness)
    watchdog = _watchdog(harness, CountingDestroyer())
    watchdog.run()
    status = harness.status()
    assert (status["phase"], status["not_uploaded"]) == (states.TIMED_OUT, []), status
    assert harness.remote_files(layout.output_prefix(harness.job_id)) == ["model.pt"]


def test_b2_runner_slow_in_teardown_within_its_limits_is_not_killed(harness):
    """B2: the runner is alive and still uploading after max_hours (final sync keeps failing
    until its limit). The watchdog's hard deadline covers the runner's worst case, so the
    runner itself writes the outcome (TIMED_OUT with not_uploaded), not the watchdog."""
    harness.tuning.update(watchdog_margin_s=0.5)
    run = "echo w > $DGPU_OUTPUT/model.pt; sleep 300"
    harness.prepare(run=run, max_hours=0.0003, final_sync_timeout_min=0.05)
    harness.start()
    harness.wait_running()
    harness.wait_for(
        lambda: harness.remote_files(layout.output_prefix(harness.job_id)), what="first upload"
    )
    (harness.work / "output" / "model.pt").write_text("newer weights")
    harness.add_rule("copy", "/output", mode="fail")
    harness.wait_done(timeout=60)
    status = harness.status()
    assert (status["phase"], status["source"]) == (states.TIMED_OUT, states.SRC_RUNNER), (
        harness.diagnostics()
    )
    assert status["not_uploaded"] == ["output/model.pt"]
    assert len(harness.server.calls) == 1
    assert not [e for e in harness.events() if e["source"] == states.SRC_WATCHDOG]


# ================================================================ B4 (privilege separation)
# Unit tests live in test_runner_privsep.py; here: the fallback is visible on a real run.


def test_b4_not_root_falls_back_with_one_line_and_an_event(harness):
    harness.prepare(run="id -u > $DGPU_OUTPUT/uid")
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    event = _events(harness, states.EV_RUN_AS_ROOT)
    assert len(event) == 1 and event[0]["data"]["reason"] == "not_root"
    log = harness.log_text()
    assert log.count("训练进程没有降权（原因 not_root") == 1
    uid = harness.get(layout.output_prefix(harness.job_id) + "uid").decode().strip()
    assert uid == str(os.getuid())


# ================================================================ B5 / B8 (restarts)


def _restart_job(harness, run: str, **runner) -> None:
    harness.prepare(run=run, **runner)
    harness.start()
    harness.wait_running()


def test_b5_restart_after_the_deadline_uploads_local_results(harness):
    """B5: the container restarts after max_hours has passed. Results written before the
    restart but never uploaded are uploaded in the final sync; training does not rerun."""
    run = "echo x >> $DGPU_OUTPUT/runs; echo c > $DGPU_CKPT/c.pt; sleep 300"
    harness.add_rule("copy", "/output", mode="fail")
    harness.add_rule("copy", "/ckpt", mode="fail")
    _restart_job(harness, run, max_hours=0.0008)  # ~3 s
    harness.wait_for(lambda: (harness.work / "ckpt" / "c.pt").exists(), what="ckpt written")
    harness.kill_all()
    harness.clear_rules()
    time.sleep(3.2)  # the restart happens after the deadline
    harness.run_to_end()
    status = harness.status()
    assert status["phase"] == states.TIMED_OUT, harness.diagnostics()
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["c.pt"]
    assert harness.remote_files(layout.output_prefix(harness.job_id)) == ["runs"]
    assert (harness.work / "output" / "runs").read_text().split() == ["x"]


def test_b5_restart_after_cancel_uploads_local_results(harness):
    """B5: cancelled while the container was down: on restart nothing reruns, but the local
    checkpoint is uploaded before CANCELLED is written."""
    run = "echo x >> $DGPU_OUTPUT/runs; echo c > $DGPU_CKPT/c.pt; sleep 300"
    harness.add_rule("copy", "/ckpt", mode="fail")
    _restart_job(harness, run)
    harness.wait_for(lambda: (harness.work / "ckpt" / "c.pt").exists(), what="ckpt written")
    harness.kill_all()
    harness.clear_rules()
    harness.set_attempt(cancel=True)
    harness.run_to_end()
    status = harness.status()
    assert status["phase"] == states.CANCELLED, harness.diagnostics()
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["c.pt"]
    assert (harness.work / "output" / "runs").read_text().split() == ["x"]
    assert harness.status_history()[-2:] == [states.UPLOADING, states.CANCELLED]


def test_b8_restart_during_final_sync_does_not_rerun_setup_or_training(harness):
    """B8: the container restarts while the final sync is in progress. The outcome was
    persisted when it was decided, so the restart only finishes the final sync."""
    harness.add_rule("copy", "/output", mode="hang")
    harness.prepare(
        setup="echo s >> $DGPU_OUTPUT/setups",
        run="echo r >> $DGPU_OUTPUT/runs; echo c > $DGPU_CKPT/c.pt",
        final_sync_timeout_min=1.0,
    )
    harness.start()
    harness.wait_for(
        lambda: (harness.status() or {}).get("phase") == states.UPLOADING, what="UPLOADING"
    )
    final = json.loads((harness.work / "run" / "final.json").read_text())
    assert (final["phase"], final["synced"]) == (states.SUCCEEDED, False)
    harness.kill_all()
    harness.clear_rules()
    harness.add_rule("copy", "/output", mode="delay", delay=1.0)
    restarted = time.time()
    harness.start()
    beat = harness.wait_for(
        lambda: (hb := harness.heartbeat()) and hb["ts"] > restarted and hb,
        what="a heartbeat during the resumed final sync",
    )
    assert (beat["phase"], beat["step"]) == (states.UPLOADING, "final_sync")
    harness.wait_done()
    status = harness.status()
    assert status["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert (harness.work / "output" / "setups").read_text().split() == ["s"]
    assert (harness.work / "output" / "runs").read_text().split() == ["r"]
    assert sorted(harness.remote_files(layout.output_prefix(harness.job_id))) == ["runs", "setups"]
    assert len(harness.server.calls) == 1


# ================================================================ B6 (hung nvidia-smi)

HANG_SCRIPT = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"


def test_b6_bounded_wait_gives_up_on_a_process_that_survives_sigkill(tmp_path):
    """B6: a process that does not die after SIGKILL (D state; simulated by a kill that does
    nothing) no longer blocks the caller: run_bounded returns after timeout + kill_wait."""
    script = tmp_path / "hang.py"
    script.write_text(HANG_SCRIPT)
    started = time.monotonic()
    result = procs.run_bounded([PY, str(script)], 0.3, kill_wait=0.3, kill=lambda proc: None)
    elapsed = time.monotonic() - started
    assert result.rc is None and result.timed_out and result.abandoned
    assert elapsed < 3.0
    assert procs.reap_abandoned() >= 1
    for proc in list(procs._ABANDONED):
        procs.kill_group_of(proc)
        proc.wait(5)
    assert procs.reap_abandoned() == 0


def test_b6_gpu_query_times_out_and_marks_stuck_processes(tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text(f"#!{PY}\n" + HANG_SCRIPT)
    fake.chmod(0o755)
    started = time.monotonic()
    gpus, error = gpu.query(0.3, binary=str(fake))
    assert gpus is None and error.startswith(gpu.HUNG_PREFIX)
    assert time.monotonic() - started < 3.0
    gpus, error = gpu.query(0.3, binary=str(fake), kill=lambda proc: None, kill_wait=0.2)
    assert gpus is None and "ignored SIGKILL" in error
    assert gpu.hung_count() == 1
    for proc in list(gpu._HUNG):
        procs.kill_group_of(proc)
        proc.wait(5)
    assert gpu.hung_count() == 0


def test_b6_sampler_pauses_while_a_query_is_stuck_and_never_blocks_the_caller():
    calls = []
    release = threading.Event()

    def stuck_query(timeout):
        calls.append(timeout)
        release.wait(5)
        return None, gpu.HUNG_PREFIX + " within 1 s"

    hung = {"n": 0}
    notes = []
    sampler = gpu.Sampler(stuck_query, 1.0, 0.05, note=notes.append, hung=lambda: hung["n"])
    sampler.start()
    started = time.monotonic()
    assert sampler.snapshot() == (None, None, False)  # returns at once while the query hangs
    assert time.monotonic() - started < 0.1
    hung["n"] = 1
    release.set()
    time.sleep(0.3)
    sampler.stop(1.0)
    assert len(calls) == 1, "no new query while a stuck nvidia-smi is still around"
    assert any("暂停" in note for note in notes)
    assert sampler.snapshot()[0] is None


def test_b6_hung_nvidia_smi_at_preflight_fails_preflight(harness):
    harness.tuning["nvidia_smi_timeout_s"] = 0.5
    harness.set_gpu(hang=True)
    harness.prepare(run="touch $DGPU_OUTPUT/ran")
    started = time.monotonic()
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_PREFLIGHT)
    assert "did not answer" in _events(harness, states.EV_PREFLIGHT_FAILED)[0]["message"]
    assert time.monotonic() - started < 20
    assert not (harness.work / "output" / "ran").exists()


def test_b6_hung_nvidia_smi_during_training_does_not_delay_cancel(harness):
    """B6: nvidia-smi starts hanging during training. Cancellation is still handled at once
    (the sampler runs in its own thread), not after nvidia_smi_timeout_s."""
    harness.tuning["nvidia_smi_timeout_s"] = 20.0
    harness.prepare(run="echo started; sleep 300")
    harness.start()
    harness.wait_running()
    harness.set_gpu(hang=True)
    time.sleep(0.5)  # let a sample start and hang
    started = time.monotonic()
    harness.set_attempt(cancel=True)
    harness.wait_for(
        lambda: (harness.status() or {}).get("phase") == states.CANCELLED, 30, "CANCELLED"
    )
    assert time.monotonic() - started < 10, "cancel waited for the hung nvidia-smi"
    harness.set_gpu(hang=False)
    harness.wait_done(30)


# ================================================================ B7


def test_b7_code_download_failure_finishes_fast_without_final_sync_failed(harness):
    """B7: the code download fails. The final sync must not retry for final_sync_timeout_min
    against directories that were never created, and no final_sync_failed is recorded."""
    harness.add_rule("copyto", "code.tar.gz", mode="fail")
    harness.prepare(run="true", final_sync_timeout_min=0.5)  # 30 s if it were retried
    started = time.monotonic()
    harness.run_to_end(timeout=60)
    elapsed = time.monotonic() - started
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_DATA_SYNC)
    assert not _events(harness, states.EV_FINAL_SYNC_FAILED)
    assert elapsed < 15, elapsed


# ================================================================ B9


def test_b9_failure_reason_and_output_tail_survive_log_truncation(harness):
    """B9: the log cap is hit, then the job fails. The uploaded log still has the end of the
    training output, the exit code and the final status line."""
    harness.tuning.update(log_tail_kb=8, log_note_reserve_kb=16)
    code = {
        "chatty.py": "import sys\nfor i in range(4 * 1024):\n    print('x' * 500)\n"
        "print('LAST-LINE-MARKER', flush=True)\nsys.exit(7)\n"
    }
    harness.prepare(code=code, run=f"{PY} chatty.py", max_log_mb=1)
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"], status["exit_code"]) == (
        states.FAILED,
        states.REASON_EXIT_CODE,
        7,
    )
    log = harness.log_text()
    assert "LAST-LINE-MARKER" in log
    assert "run 结束，退出码 7" in log
    assert "最终状态 FAILED（exit_code）" in log
    assert log.index("LAST-LINE-MARKER") < log.index("最终状态 FAILED")
    assert "共丢弃" in log


def test_b9_diagnostic_logs_are_uploaded_next_to_the_attempt_log(harness):
    harness.prepare(run="true")
    harness.run_to_end()
    runner_log = harness.get(layout.diag_log_key(harness.job_id, 1, "runner"))
    watchdog_log = harness.get(layout.diag_log_key(harness.job_id, 1, "watchdog"))
    assert runner_log is not None and b"[runner]" in runner_log
    assert watchdog_log is not None and b"hard deadline in" in watchdog_log
    assert harness.status()["phase"] == states.SUCCEEDED


def test_b9_runner_traceback_reaches_storage(harness, monkeypatch):
    """B9: an unhandled runner error: its traceback (runner.log) is uploaded."""
    harness.prepare(run="true")
    runner = _runner(harness)

    def boom():
        raise RuntimeError("bug in restore")

    runner._restore_ckpt = boom
    (harness.work / "logs").mkdir(parents=True, exist_ok=True)
    with open(harness.work / "logs" / "runner.log", "w") as handle:
        monkeypatch.setattr(sys, "stderr", handle)
        runner.run()
    uploaded = harness.get(layout.diag_log_key(harness.job_id, 1, "runner")).decode()
    assert "Traceback" in uploaded and "bug in restore" in uploaded
    assert "runner 内部错误" in harness.log_text()


def test_b9_log_tail_survives_a_runner_crash(tmp_path):
    """B9: the kept tail lives on disk, so the watchdog can fold it in after a crash."""
    from dgpu.runner import logsink

    path = tmp_path / "attempt-1.log"
    sink = logsink.LogSink(str(path), 100, tail_bytes=40, note_reserve=200)
    sink.child_output(b"a" * 100)
    for index in range(10):
        sink.child_output(b"line-%02d\n" % index)
    sink.note("exit code 3")
    # crash: no flush_tail; the watchdog appends what is on disk
    added = logsink.append_tail(str(path), "\n-- tail --\n")
    data = path.read_bytes()
    assert added <= 40 and b"line-09" in data and b"exit code 3" in data
    assert b"line-00" not in data, "only the last part is kept"
    assert not list(tmp_path.glob("*.tail.*"))


# ================================================================ B10 (AutoDL boot)


def test_b10_autodl_boot_restarts_a_dead_watchdog(tmp_path):
    """B10: AutoDL foreground: the watchdog dies while the runner trains. boot restarts it;
    when the runner then dies, the new watchdog takes over and records the outcome."""
    h = Harness(tmp_path, backend="autodl")
    try:
        h.prepare(run="echo c > $DGPU_CKPT/c.pt; sleep 300")
        proc = h.start()
        h.wait_running()
        first = h.pid("watchdog")
        os.kill(first, signal.SIGKILL)
        h.wait_for(lambda: h.pid("watchdog") not in (None, first), what="a new watchdog")
        second = h.pid("watchdog")
        h.wait_for(
            lambda: (
                b"hard deadline in" in (h.work / "logs" / "watchdog.log").read_bytes()
                and procs.pid_alive(second)
            ),
            what="the new watchdog to run",
        )
        os.kill(h.pid("runner"), signal.SIGKILL)
        assert proc.wait(60) == 0, h.diagnostics()
        status = h.status()
        assert (status["phase"], status["reason"], status["source"]) == (
            states.FAILED,
            states.REASON_RUNNER_CRASHED,
            states.SRC_WATCHDOG,
        )
        assert h.remote_files(layout.ckpt_prefix(h.job_id)) == ["c.pt"]
    finally:
        h.cleanup()


class JumpClock:
    """Monotonic time that jumps forward 60 s per call; sleep does nothing."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        self.now += 60.0
        return self.now

    def sleep(self, _seconds):
        time.sleep(0.01)


def test_b10_boot_kills_everything_at_its_own_deadline(tmp_path):
    """B10: watchdog gone and runner stuck: boot reaches its own hard deadline, kills the runner,
    the watchdog, the training group and registered rclones, and exits non-zero (no status)."""
    h = Harness(tmp_path, backend="autodl")
    try:
        spec = h.prepare(run="true", max_hours=0.01)
        paths = Paths(str(h.work))
        paths.ensure()
        (h.work / "run" / "spec.json").write_text(json.dumps(spec))
        (h.work / "run" / "started.json").write_text(
            json.dumps({"attempt": 1, "wall": time.time()})
        )
        runner = subprocess.Popen(["sleep", "300"], start_new_session=True)
        watchdog = subprocess.Popen(["sleep", "300"], start_new_session=True)
        training = subprocess.Popen(["sleep", "300"], start_new_session=True)
        (h.work / "run" / "child.json").write_text(
            json.dumps({"pid": training.pid, "pgid": training.pid})
        )
        rclone = _fake_registered_rclone(h)
        for p in (runner, watchdog, training):
            h.track_pid(p.pid)
        os.kill(runner.pid, signal.SIGSTOP)
        tuning = dict(util.DEFAULT_TUNING, boot_watchdog_respawns=0)
        sup = boot_mod.Supervisor(
            paths, tuning, PY, str(h.work / "launch.json"), -1, 1, clock=JumpClock(), poll_s=0
        )
        assert sup.run(runner, watchdog) == states.EXIT_FAILED
        assert sup.killed_at_deadline
        assert _wait_dead([runner.pid, watchdog.pid, training.pid, rclone.pid]) == []
    finally:
        h.cleanup()


# ================================================================ B11


def test_b11_periodic_sync_stops_between_targets(harness):
    """B11: stopping the periodic sync while it uploads output/ must not let it go on to
    ckpt/ (competing with the final sync)."""
    harness.prepare(run="true", sync_interval_s=0.05, heartbeat_interval_s=60)
    runner = _runner(harness)
    runner._load_launch()
    runner._load_spec()
    for name, file in (("output", "o.txt"), ("ckpt", "c.pt")):
        (harness.work / name).mkdir(exist_ok=True)
        (harness.work / name / file).write_text(name)
    harness.add_rule("copy", "/output", mode="hang")
    runner._attempt_checked_at = runner.clock.monotonic() + 3600  # attempt check is fresh
    runner._start_sync_thread()
    harness.wait_for(lambda: _rclone_pids(harness, "/output"), what="the output copy to start")
    runner._stop_sync_thread()
    time.sleep(0.5)
    copies = [c for c in harness.calls() if c["op"] == "copy" and "/ckpt" in " ".join(c["args"])]
    assert copies == [], "the periodic sync went on to ckpt/ after it was told to stop"
    assert not runner._sync_thread.is_alive()
    runner.events.stop_flusher(1.0)


# ================================================================ B12


def test_b12_watchdog_kills_rclone_left_running_by_a_killed_runner(harness):
    harness.prepare(run="echo o > $DGPU_OUTPUT/o.txt; sleep 300")
    harness.start()
    harness.wait_running()
    harness.add_rule("copy", "/output", mode="hang")
    harness.wait_for(lambda: _rclone_pids(harness, "/output"), what="a hung periodic sync")
    left = _rclone_pids(harness, "/output")
    for pid in left:
        harness.track_pid(pid)
    os.kill(harness.pid("runner"), signal.SIGKILL)
    harness.wait_done()
    assert _wait_dead(left) == [], "rclone started by the runner is still running"
    status = harness.status()
    assert (status["reason"], status["source"]) == (
        states.REASON_RUNNER_CRASHED,
        states.SRC_WATCHDOG,
    )


# ================================================================ unreproduced items


def test_flock_errors_other_than_eagain_do_not_mean_busy(tmp_path, monkeypatch):
    """flock failing with ENOLCK (filesystem without flock) is not 'another runner is active'."""
    path = str(tmp_path / "boot.lock")

    def no_flock(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(boot_mod.fcntl, "flock", no_flock)
    fd, locked = boot_mod.acquire_lock(path)
    assert fd is not None and not locked
    os.close(fd)

    def busy(fd, op):
        raise BlockingIOError(errno.EAGAIN, "busy")

    monkeypatch.setattr(boot_mod.fcntl, "flock", busy)
    assert boot_mod.acquire_lock(path) == (None, False)


SITECUSTOMIZE = """
import errno, fcntl
def _flock(fd, op):
    raise OSError(errno.ENOLCK, "No locks available")
fcntl.flock = _flock
"""


def test_flock_unsupported_job_still_runs_and_second_boot_does_nothing(harness, tmp_path):
    """Without flock (every flock call fails with ENOLCK) the job runs; a second boot while
    the runner is alive sees it through runner.pid and does nothing."""
    site = tmp_path / "noflock"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    harness.extra_env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{site}"
    harness.prepare(run="echo c > $DGPU_CKPT/c.pt; sleep 300")
    harness.start()
    harness.wait_running()
    runner = harness.pid("runner")
    second = subprocess.run(
        [PY, "-m", "dgpu.runner.boot", "--work-root", str(harness.work)],
        env=harness.env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 0
    assert "nothing to do" in second.stderr
    assert harness.pid("runner") == runner
    harness.set_attempt(cancel=True)
    harness.wait_done()
    assert harness.status()["phase"] == states.CANCELLED, harness.diagnostics()
    assert b"flock failed" in (harness.work / "logs" / "watchdog.log").read_bytes()


def test_fencing_periodic_sync_rechecks_a_stale_attempt_before_writing(harness):
    """Fencing window: the heartbeat thread is stuck (its last attempt.json read is old). The
    periodic sync reads attempt.json itself and writes nothing once a newer attempt exists."""
    harness.prepare(run="true", sync_interval_s=0.05, heartbeat_interval_s=0.1)
    runner = _runner(harness)
    runner._load_launch()
    runner._load_spec()
    (harness.work / "output").mkdir(exist_ok=True)
    (harness.work / "output" / "o.txt").write_text("o")
    runner._attempt_checked_at = runner.clock.monotonic() - 3600
    harness.set_attempt(attempt=2)
    runner._start_sync_thread()
    harness.wait_for(runner.fenced.is_set, what="the sync thread to notice attempt 2")
    runner._stop_sync_thread()
    assert harness.remote_files(layout.output_prefix(harness.job_id)) == []
    runner.events.stop_flusher(1.0)


def test_fencing_watchdog_hard_deadline_checks_attempt_json(harness):
    """Fencing window: at the hard deadline the watchdog also reads attempt.json (not only the
    local marker) and does not write a status for a superseded attempt."""
    spec = harness.prepare(run="true", max_hours=0.0001, final_sync_timeout_min=0.01)
    (harness.work / "run").mkdir(parents=True, exist_ok=True)
    (harness.work / "run" / "spec.json").write_text(json.dumps(spec))
    harness.set_attempt(attempt=2)
    harness.tuning.update(
        watchdog_margin_s=0.1,
        teardown_budget_s=1.0,
        thread_join_s=0.1,
        destroy_attempts=1,
        small_op_timeout_s=2.0,
    )
    _started_long_ago(harness)
    _fake_runner(harness)
    destroyer = CountingDestroyer()
    _watchdog(harness, destroyer).run()
    assert harness.status() is None
    assert destroyer.calls == 1
    assert (harness.work / "run" / "fenced").exists()


def test_stall_detection_ignores_a_quiet_setup(harness):
    """Stall detection applies to `run` only: a silent setup with an idle GPU is not stalled."""
    harness.set_gpu(util=0)
    harness.prepare(setup="sleep 2", run="echo ok", stall_timeout_min=0.01)
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert not _events(harness, states.EV_STALLED)


def test_stall_detection_with_one_gpu_reporting_na(harness):
    """One GPU reporting [N/A] must not disable stall detection for the other (idle) one."""
    harness.set_gpu(lines="0, RTX 3090, 0, 24576\n1, RTX 3090, [N/A], 24576\n")
    harness.prepare(run="echo started; sleep 300", stall_timeout_min=0.01)
    harness.run_to_end(timeout=60)
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_STALLED), (
        harness.diagnostics()
    )


def test_stopping_is_not_delayed_by_a_hung_event_upload(harness):
    """Events: the deadline_hit event written right before the training is stopped, and the
    log_truncated event written from the log-reading thread, must not wait for storage."""
    harness.prepare(run="true")
    runner = _runner(harness)
    runner._load_launch()
    runner._load_spec()
    harness.add_rule("move", "", mode="hang")
    started = time.monotonic()
    runner._record_stop("deadline")
    runner._on_log_truncated(1024 * 1024)
    assert time.monotonic() - started < 1.0
    names = sorted(
        json.loads(p.read_text())["name"]
        for p in (harness.work / "run" / "events-runner").glob("*.json")
    )
    assert names == [states.EV_DEADLINE_HIT, states.EV_LOG_TRUNCATED]
    if runner.events is not None and hasattr(runner.events, "stop_flusher"):
        runner.events.stop_flusher(2.0)


def test_events_emit_never_waits_for_storage(harness):
    """Events: emitting (before stopping training, inside the log-truncation callback) only
    spools locally and wakes the uploader; a hung storage cannot block the caller."""
    harness.add_rule("move", "", mode="hang")
    rclone = Rclone(
        str(harness.bin / "rclone"),
        {
            "endpoint": "https://x.invalid",
            "bucket": harness.bucket,
            "access_key_id": "k",
            "secret_access_key": "s",
        },
        str(harness.work / "rclone.conf"),
        base_env=harness.env(),
        poll_s=0.02,
        small_timeout=30.0,
    )
    events = Events(
        rclone,
        harness.job_id,
        1,
        states.SRC_RUNNER,
        str(harness.work / "spool"),
        str(harness.work / "tmp"),
    )
    events.start_flusher(30.0)
    started = time.monotonic()
    events.emit(states.EV_DEADLINE_HIT, "deadline")
    events.emit(states.EV_LOG_TRUNCATED, "big log")
    assert time.monotonic() - started < 0.5
    stop_started = time.monotonic()
    events.stop_flusher(5.0)
    assert time.monotonic() - stop_started < 5.0
    assert events.pending() == 2, "events stay spooled when the upload is aborted"


def test_login_profile_cannot_reexport_secrets_or_drop_path(harness):
    """`bash -lc`: a login profile that re-exports credentials, resets PATH and overrides a
    dgpu variable is neutralised before the training command runs."""
    profile = (
        "export AWS_SECRET_ACCESS_KEY=leaked-aws\n"
        "export RCLONE_CONFIG_DGPU_SECRET_ACCESS_KEY=leaked-rclone\n"
        "export container_api_key=leaked-lower\n"
        "export PATH=/usr/bin:/bin\n"
        "export DGPU_OUTPUT=/tmp/wrong-output\n"
        "export FROM_PROFILE=yes\n"
    )
    (harness.home / ".bash_profile").write_text(profile)
    harness.prepare(run="env > $DGPU_OUTPUT/env.txt; command -v nvidia-smi > $DGPU_OUTPUT/which")
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    env_txt = harness.get(layout.output_prefix(harness.job_id) + "env.txt").decode()
    assert "leaked" not in env_txt
    assert "FROM_PROFILE=yes" in env_txt, "the profile did run"
    assert f"DGPU_OUTPUT={harness.work / 'output'}" in env_txt
    which = harness.get(layout.output_prefix(harness.job_id) + "which").decode().strip()
    assert which == str(harness.bin / "nvidia-smi"), "PATH entries dropped by the profile are back"
    assert "__DGPU_PATH0" not in env_txt


def test_procs_parses_a_linux_proc_tree(tmp_path, monkeypatch):
    """Test gap: the /proc branch of procs.py (never executed on macOS)."""
    root = tmp_path / "proc"
    (root / "self").mkdir(parents=True)
    rows = {
        100: "100 (python3) S 1 100 100 0",
        101: "101 (weird ) (name) R 100 100 100 0",
        102: "102 (zombie) Z 100 102 102 0",
    }
    for pid, line in rows.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / "stat").write_text(line)
        (root / str(pid) / "cmdline").write_bytes(b"python3\0-m\0dgpu.runner.main\0")
    (root / "notapid").mkdir()
    monkeypatch.setattr(procs, "PROC_ROOT", str(root))
    assert sorted(procs.list_processes()) == [(100, 1, 100), (101, 100, 100), (102, 100, 102)]
    assert procs._proc_state(101) == "R" and procs._proc_state(102) == "Z"
    assert procs._proc_state(999) is None
    assert procs.cmdline(100) == "python3 -m dgpu.runner.main"
    assert procs.descendants([100]) == {101, 102}
    assert procs.descendants([], pgid=102) == {102}


def test_bootstrap_https_failure_is_explained_and_redacted(tmp_path):
    """Bootstrap diagnostics: an HTTPS/TLS failure is logged with its reason (not just the
    exception type), URLs are redacted, the free disk space is logged when giving up."""
    import socket

    from test_runner_bootstrap import Boot

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    port = server.getsockname()[1]
    stop = threading.Event()

    def serve():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            try:
                conn.recv(1024)
                conn.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    b = Boot(tmp_path)
    try:
        b.env["DGPU_BOOT_BACKOFF_S"] = "0.001"
        url = f"https://127.0.0.1:{port}/bundle?X-Amz-Signature=deadbeef"
        command = b.bundler.bootstrap_command(
            bundle_url=url, bundle_sha256=b.sha, launch_url=b.launch_src.as_uri()
        )
        result = b.run(command)
        assert result.returncode == 1
        log = (b.h.work / "logs" / "bootstrap.log").read_text()
        assert "SSL" in log or "ssl" in log, log
        assert "deadbeef" not in log and "https://" not in log
        assert "MB free" in log and "giving up on bundle.tgz" in log
        assert "python 3." in log
        assert (b.h.work / "logs" / "bootstrap.log").stat().st_mode & 0o077 == 0
        assert b.h.pid("runner") is None
    finally:
        stop.set()
        server.close()
        b.h.cleanup()


def test_bootstrap_flock_unsupported_continues(tmp_path):
    """Bootstrap: flock failing with ENOLCK is logged and the bootstrap goes on."""
    from test_runner_bootstrap import Boot

    b = Boot(tmp_path, backend="autodl", run="echo done > $DGPU_OUTPUT/done")
    try:
        site = tmp_path / "noflock"
        site.mkdir()
        (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
        b.env["PYTHONPATH"] = str(site)
        result = b.run(b.command())
        assert result.returncode == 0, b.h.diagnostics() + result.stderr
        assert "flock unsupported" in result.stderr
        assert b.h.status()["phase"] == states.SUCCEEDED
        uploaded = b.h.get(layout.diag_log_key(b.h.job_id, 1, "bootstrap"))
        assert uploaded is not None and b"bundle ready" in uploaded
    finally:
        b.h.cleanup()
