"""End-to-end runner tests: the real boot / runner / watchdog processes against fakes.

Each test names the row of the failure-mode table (docs/dgpu-design.md section 5) it covers.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from dgpu import layout, states
from dgpu.runner import procs
from dgpu.testing import TEST_PROFILE
from test_runner_harness import CONTAINER_ID, CONTAINER_KEY, Harness, old_python

PY = sys.executable


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.cleanup()


def _events(h: Harness, name: str) -> list[dict]:
    return [event for event in h.events() if event["name"] == name]


def _event(h: Harness, name: str) -> dict:
    found = _events(h, name)
    assert found, f"no {name} event; got {h.event_names()}\n{h.diagnostics()}"
    return found[0]


def _wait_dead(pids, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        alive = [pid for pid in pids if procs.pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if procs.pid_alive(pid)]


def _first_heartbeat(h: Harness) -> dict:
    return h.wait_for(h.heartbeat, what="the first heartbeat")


# ---------------------------------------------------------------- normal end


def test_success_uploads_results_then_destroys_once_and_last(harness):
    """Normal end: status sequence, heartbeat, outputs and ckpt uploaded (*.tmp excluded),
    log uploaded, self-destroy called exactly once after every storage write."""
    run = (
        "echo training; env | sort > $DGPU_OUTPUT/env.txt; "
        "mkdir -p $DGPU_OUTPUT/run1 && echo w > $DGPU_OUTPUT/run1/best.pt; "
        "echo ck > $DGPU_CKPT/last.pt; echo partial > $DGPU_CKPT/last.pt.tmp"
    )
    harness.prepare(run=run, setup="echo setting-up", env={"EPOCHS": "3"})
    harness.run_to_end()

    status = harness.status()
    assert status["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert status["exit_code"] == 0 and status["source"] == states.SRC_RUNNER
    assert harness.status_history() == [states.RUNNING, states.UPLOADING, states.SUCCEEDED]
    heartbeat = harness.heartbeat()
    assert heartbeat["attempt"] == 1 and heartbeat["job_id"] == harness.job_id

    assert harness.remote_files(layout.output_prefix(harness.job_id)) == ["env.txt", "run1/best.pt"]
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["last.pt"]
    log = harness.log_text()
    assert "setting-up" in log and "training" in log and "SUCCEEDED" in log

    env_txt = harness.get(layout.output_prefix(harness.job_id) + "env.txt").decode()
    assert f"DGPU_JOB_ID={harness.job_id}" in env_txt
    assert "DGPU_ATTEMPT=1" in env_txt and "DGPU_RESUME=0" in env_txt and "EPOCHS=3" in env_txt
    assert "DGPU_OUTPUT=%s" % (harness.work / "output") in env_txt
    assert "DGPU_DATA=%s" % (harness.work / "data") in env_txt

    calls = harness.server.calls
    assert len(calls) == 1
    assert calls[0]["path"] == f"/api/v0/instances/{CONTAINER_ID}/"
    assert calls[0]["auth"] == f"Bearer {CONTAINER_KEY}"
    assert calls[0]["ts"] >= max(call["ts_end"] for call in harness.calls())

    rclone_calls = harness.calls()
    assert all(call["has_creds"] for call in rclone_calls)
    assert not any(call["secret_in_argv"] for call in rclone_calls)
    assert json.loads((harness.work / "run" / "destroyed.json").read_text())["ok"] is True
    assert (harness.work / "launch.json").stat().st_mode & 0o077 == 0


def test_autodl_boot_stays_in_foreground_until_everything_exited(tmp_path):
    """AutoDL: the container lives only while cmd runs, so boot must wait for the runner and
    the watchdog; 'destroy' is just exiting (no HTTP call)."""
    h = Harness(tmp_path, backend="autodl")
    try:
        h.prepare(run="sleep 0.5; echo done > $DGPU_OUTPUT/done")
        proc = h.start()  # auto mode -> wait on AutoDL
        assert proc.wait(30) == states.EXIT_SUCCEEDED, h.diagnostics()
        for name in ("runner", "watchdog"):
            assert not procs.pid_alive(h.pid(name)), name
        assert h.status()["phase"] == states.SUCCEEDED
        assert h.remote_files(layout.output_prefix(h.job_id)) == ["done"]
        assert h.server.calls == []
        destroyed = json.loads((h.work / "run" / "destroyed.json").read_text())
        assert destroyed["ok"] is True and destroyed["method"] == "process_exit"
    finally:
        h.cleanup()


# ---------------------------------------------------------------- setup / run failures


def test_setup_failure_skips_run_uploads_log_and_destroys(harness):
    """Row: setup fails -> FAILED(setup), log uploaded, destroyed, no automatic retry."""
    harness.prepare(setup="echo broken-setup; exit 3", run="touch $DGPU_OUTPUT/ran")
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"], status["exit_code"]) == (
        states.FAILED,
        states.REASON_SETUP,
        3,
    ), harness.diagnostics()
    assert harness.remote_files(layout.output_prefix(harness.job_id)) == []
    assert "broken-setup" in harness.log_text()
    assert len(harness.server.calls) == 1


def test_nonzero_exit_on_autodl_returns_failure_code(tmp_path):
    """Row: training exits non-zero -> FAILED(exit_code), log uploaded; boot (AutoDL cmd)
    exits non-zero only after everything stopped."""
    h = Harness(tmp_path, backend="autodl")
    try:
        h.prepare(run="echo oops >&2; echo partial > $DGPU_OUTPUT/p.txt; exit 7")
        assert h.start().wait(30) == states.EXIT_FAILED, h.diagnostics()
        status = h.status()
        assert (status["phase"], status["reason"], status["exit_code"]) == (
            states.FAILED,
            states.REASON_EXIT_CODE,
            7,
        )
        assert "oops" in h.log_text()
        assert h.remote_files(layout.output_prefix(h.job_id)) == ["p.txt"]
        assert not procs.pid_alive(h.pid("watchdog"))
    finally:
        h.cleanup()


# ---------------------------------------------------------------- limits


def test_deadline_sends_sigterm_then_sigkill_to_the_whole_tree(harness):
    """Row: max_hours exceeded -> SIGTERM, grace, SIGKILL (child ignores TERM and has
    grandchildren, one in its own session), final sync, TIMED_OUT, destroyed."""
    ignore_term = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
    )
    own_session = "import os,time; os.setsid(); time.sleep(300)"
    run = (
        "trap 'echo got TERM' TERM; "
        "sleep 300 & echo $! >> $DGPU_OUTPUT/pids; "
        f'{PY} -c "{ignore_term}" & echo $! >> $DGPU_OUTPUT/pids; '
        f'{PY} -c "{own_session}" & echo $! >> $DGPU_OUTPUT/pids; '
        "echo $$ >> $DGPU_OUTPUT/pids; echo started; "
        "while true; do sleep 0.1; done"
    )
    harness.prepare(run=run, max_hours=0.0006, term_grace_s=1.0)
    started = time.monotonic()
    harness.run_to_end()
    elapsed = time.monotonic() - started

    status = harness.status()
    assert status["phase"] == states.TIMED_OUT, harness.diagnostics()
    assert harness.status_history()[-2:] == [states.UPLOADING, states.TIMED_OUT]
    _event(harness, states.EV_DEADLINE_HIT)
    log = harness.log_text()
    assert "got TERM" in log, "SIGTERM must come first"
    assert "killed" in log, "the TERM-ignoring child needed SIGKILL"
    pids = [int(p) for p in (harness.work / "output" / "pids").read_text().split()]
    assert len(pids) == 4
    assert _wait_dead(pids) == []
    assert "pids" in harness.remote_files(layout.output_prefix(harness.job_id))
    assert len(harness.server.calls) == 1
    assert elapsed < 20


def test_stall_detection_kills_idle_silent_job(harness):
    """Row: training hangs -> GPU idle and no log output for stall_timeout -> FAILED(stalled)."""
    harness.set_gpu(util=0)
    harness.prepare(run="echo started; sleep 300", stall_timeout_min=0.01)
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_STALLED), (
        harness.diagnostics()
    )
    _event(harness, states.EV_STALLED)
    assert not _events(harness, states.EV_DEADLINE_HIT)
    assert len(harness.server.calls) == 1


@pytest.mark.parametrize(
    ("util", "run"),
    [
        ("0", "for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do echo tick $i; sleep 0.1; done"),
        ("50", "sleep 1.5"),
    ],
    ids=["idle-gpu-but-logging", "silent-but-gpu-busy"],
)
def test_stall_needs_both_idle_gpu_and_silence(harness, util, run):
    harness.set_gpu(util=float(util))
    harness.prepare(run=run, stall_timeout_min=0.01)
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert not _events(harness, states.EV_STALLED)


# ---------------------------------------------------------------- cancel and fencing


def test_cancel_flag_stops_training_and_uploads(harness):
    harness.prepare(run="echo ck > $DGPU_CKPT/c.pt; echo started; sleep 300")
    harness.start()
    harness.wait_running()
    _first_heartbeat(harness)
    harness.set_attempt(cancel=True)
    harness.wait_done()
    assert harness.status()["phase"] == states.CANCELLED, harness.diagnostics()
    assert harness.status_history()[-2:] == [states.UPLOADING, states.CANCELLED]
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["c.pt"]
    assert len(harness.server.calls) == 1


def test_cancelled_before_start_does_not_run(harness):
    doc = states.new_attempt_doc(harness.job_id, 1, "vast", layout.label(harness.job_id, 1))
    doc["cancel"] = True
    harness.prepare(run="touch $DGPU_OUTPUT/ran", attempt_doc=doc)
    harness.run_to_end()
    assert harness.status()["phase"] == states.CANCELLED
    assert states.RUNNING not in harness.status_history()
    assert not (harness.work / "output" / "ran").exists()
    assert len(harness.server.calls) == 1


def test_newer_attempt_before_start_does_not_run_or_write_status(harness):
    """Row: old attempt not dead yet when a newer one exists -> exits without touching status."""
    doc = states.new_attempt_doc(harness.job_id, 2, "vast", layout.label(harness.job_id, 2))
    harness.prepare(run="touch $DGPU_OUTPUT/ran", attempt_doc=doc)
    harness.run_to_end()
    assert harness.status() is None
    assert harness.heartbeat() is None
    assert _event(harness, states.EV_ATTEMPT_FENCED)["data"]["newer_attempt"] == 2
    assert not (harness.work / "output" / "ran").exists()
    assert len(harness.server.calls) == 1


def test_fenced_by_newer_attempt_stops_and_never_writes_again(harness):
    """Row: newer attempt started while this one runs -> attempt_fenced, kill training, no
    writes to output/ckpt/status/heartbeat after noticing, FAILED(fenced) never written."""
    run = (
        "i=0; while true; do i=$((i+1)); echo $i > $DGPU_OUTPUT/o$i.txt; "
        "echo $i > $DGPU_CKPT/c.pt; sleep 0.05; done"
    )
    harness.prepare(run=run)
    harness.start()
    harness.wait_running()
    _first_heartbeat(harness)
    # let the periodic sync upload something first, so writes are in flight when fenced
    harness.wait_for(
        lambda: harness.remote_files(layout.output_prefix(harness.job_id)), what="an upload"
    )
    sentinel = states.new_status_doc(harness.job_id, 2, states.RUNNING, states.SRC_RUNNER)
    harness.put(layout.status_key(harness.job_id), json.dumps(sentinel))
    harness.set_attempt(attempt=2)
    harness.wait_for(lambda: _events(harness, states.EV_ATTEMPT_FENCED), what="attempt_fenced")
    prefix = f"{harness.bucket}/"
    harness.freeze(
        [
            prefix + layout.status_key(harness.job_id),
            prefix + layout.heartbeat_key(harness.job_id),
            prefix + layout.output_prefix(harness.job_id),
            prefix + layout.ckpt_prefix(harness.job_id),
        ]
    )
    harness.wait_done()
    assert harness.violations() == []
    assert harness.status() == sentinel
    assert states.FAILED not in harness.status_history()
    assert states.UPLOADING not in harness.status_history()
    assert (harness.work / "run" / "fenced").exists()
    assert len(harness.server.calls) == 1
    assert harness.log_text(), "the per-attempt log is still uploaded"


# ---------------------------------------------------------------- data, checkpoints, storage


def test_data_sync_retries_then_fails(harness):
    """Row: data download fails -> data_sync_retry per retry, then FAILED(data_sync), destroy."""
    harness.put("datasets/coco8/a.txt", "x")
    harness.prepare(data=(("coco8", "datasets/coco8/"),), run="touch $DGPU_OUTPUT/ran")
    harness.add_rule("copy", "datasets/coco8", mode="fail")
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_DATA_SYNC)
    retries = _events(harness, states.EV_DATA_SYNC_RETRY)
    assert [event["data"]["retry"] for event in retries] == [1, 2]
    copies = [c for c in harness.calls() if c["op"] == "copy" and "datasets/coco8" in c["args"][0]]
    assert len(copies) == 3
    assert not (harness.work / "output" / "ran").exists()
    assert len(harness.server.calls) == 1


def test_data_sync_retry_then_success(harness):
    harness.put("datasets/coco8/sub/a.txt", "x")
    harness.prepare(data=(("coco8", "datasets/coco8/"),), run="test -f $DGPU_DATA/coco8/sub/a.txt")
    harness.add_rule("copy", "datasets/coco8", mode="fail", times=1)
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert len(_events(harness, states.EV_DATA_SYNC_RETRY)) == 1


def test_ckpt_restore_sets_resume(harness):
    harness.put(layout.ckpt_prefix(harness.job_id) + "last.pt", "weights")
    harness.put(layout.ckpt_prefix(harness.job_id) + "half.pt.tmp", "partial")
    run = (
        'test "$DGPU_RESUME" = 1 && test "$(cat $DGPU_CKPT/last.pt)" = weights '
        "&& test ! -e $DGPU_CKPT/half.pt.tmp"
    )
    harness.prepare(run=run)
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert _event(harness, states.EV_CKPT_RESTORED)["data"]["files"] == 1


def test_storage_outage_during_training_keeps_training_and_catches_up(harness):
    """Row: storage disconnects while training -> training continues, sync_failed, catch up."""
    run = "for i in $(seq 1 25); do echo $i > $DGPU_OUTPUT/f$i.txt; sleep 0.1; done; echo finished"
    harness.prepare(run=run)
    harness.start()
    harness.wait_running()
    _first_heartbeat(harness)
    harness.set_offline(True)
    time.sleep(1.0)
    harness.set_offline(False)
    harness.wait_done()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert "finished" in harness.log_text()
    assert len(harness.remote_files(layout.output_prefix(harness.job_id))) == 25
    failed = _event(harness, states.EV_SYNC_FAILED)
    assert failed["data"]["what"] in ("output", "ckpt", "log", "heartbeat")
    assert len(_events(harness, states.EV_SYNC_FAILED)) <= 2, "sync_failed is rate-limited"


def test_final_sync_failure_reports_not_uploaded_and_still_destroys(harness):
    """Row: final upload keeps failing -> retry until the limit, FAILED(final_sync) with the
    files that did not make it, final_sync_failed, destroyed."""
    harness.prepare(
        run="echo r > $DGPU_OUTPUT/result.txt; echo c > $DGPU_CKPT/c.pt",
        final_sync_timeout_min=0.02,
    )
    harness.add_rule("copy", "/output", mode="fail")
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_FINAL_SYNC)
    assert status["not_uploaded"] == ["output/result.txt"]
    assert _event(harness, states.EV_FINAL_SYNC_FAILED)["data"]["count"] == 1
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["c.pt"]
    output_copies = [
        c for c in harness.calls() if c["op"] == "copy" and "/output" in " ".join(c["args"])
    ]
    assert len(output_copies) >= 2, "the final sync was retried"
    assert len(harness.server.calls) == 1


def test_storage_unreachable_at_start_still_destroys(harness):
    """Row: instance cannot reach storage -> nothing to run, instance destroyed anyway."""
    harness.prepare()
    harness.set_offline(True)
    harness.run_to_end()
    assert harness.status() is None
    assert len(harness.server.calls) == 1
    final = json.loads((harness.work / "run" / "final.json").read_text())
    assert (final["phase"], final["reason"]) == (states.FAILED, states.REASON_PREFLIGHT)
    spooled = list((harness.work / "run" / "events-runner").glob("*.json"))
    assert [json.loads(p.read_text())["name"] for p in spooled] == [states.EV_PREFLIGHT_FAILED]


@pytest.mark.parametrize("problem", ["no-gpu", "disk"])
def test_preflight_failure(harness, problem):
    """Rows: no GPU / driver broken, and not enough disk -> FAILED(preflight), destroyed."""
    if problem == "no-gpu":
        harness.set_gpu(missing=True)
    else:
        harness.tuning["min_free_margin_mb"] = 1e12
    harness.prepare(run="touch $DGPU_OUTPUT/ran")
    harness.run_to_end()
    status = harness.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_PREFLIGHT)
    event = _event(harness, states.EV_PREFLIGHT_FAILED)
    assert ("GPU" if problem == "no-gpu" else "磁盘") in event["message"]
    assert not (harness.work / "output" / "ran").exists()
    assert len(harness.server.calls) == 1


# ---------------------------------------------------------------- logs and environment


def test_log_truncation_does_not_block_a_chatty_child(harness):
    """Row: huge log -> keep draining, stop writing, log_truncated once; child never blocks."""
    code = {
        "chatty.py": "import sys\nline = 'x' * 1023 + '\\n'\n"
        "for _ in range(8 * 1024):\n    sys.stdout.write(line)\n"
    }
    harness.prepare(code=code, run=f"{PY} chatty.py && echo done > $DGPU_OUTPUT/done", max_log_mb=1)
    started = time.monotonic()
    harness.run_to_end()
    assert time.monotonic() - started < 20
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    assert len(_events(harness, states.EV_LOG_TRUNCATED)) == 1
    local = harness.work / "logs" / "attempt-1.log"
    assert 1024 * 1024 <= local.stat().st_size <= 1024 * 1024 + 4096
    assert len(harness.log_text().encode()) <= 1024 * 1024 + 4096


def test_child_environment_is_scrubbed(harness):
    harness.extra_env = {
        "AWS_SECRET_ACCESS_KEY": "aws-secret-value",
        "R2_SECRET_ACCESS_KEY": "r2-secret-value",
        "RCLONE_CONFIG_OTHER_PASS": "other-pass",
        "KEEP_ME": "kept",
    }
    harness.prepare(run="env > $DGPU_OUTPUT/env.txt", env={"AWS_REGION": "eu", "MY_VAR": "1"})
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED, harness.diagnostics()
    env_txt = harness.get(layout.output_prefix(harness.job_id) + "env.txt").decode()
    names = {line.split("=", 1)[0] for line in env_txt.splitlines() if "=" in line}
    for prefix in states.SCRUBBED_ENV_PREFIXES:
        assert not [name for name in names if name.startswith(prefix)], prefix
    for secret in (
        TEST_PROFILE.secret_access_key,
        TEST_PROFILE.access_key_id,
        CONTAINER_KEY,
        "aws-secret-value",
        "r2-secret-value",
    ):
        assert secret not in env_txt
    assert "KEEP_ME=kept" in env_txt and "MY_VAR=1" in env_txt
    assert "AWS_REGION" in harness.log_text(), "dropping a job env var is reported"
    assert all(call["has_creds"] for call in harness.calls())


# ---------------------------------------------------------------- runner death and watchdog


def test_runner_killed_watchdog_syncs_reports_and_destroys(harness):
    """Row: runner crashes -> watchdog syncs once, FAILED(runner_crashed), destroyed."""
    harness.prepare(run="echo c > $DGPU_CKPT/c.pt; echo started; sleep 300")
    harness.start()
    harness.wait_running()
    _first_heartbeat(harness)
    child = harness.child()
    os.kill(harness.pid("runner"), signal.SIGKILL)
    harness.wait_done()
    status = harness.status()
    assert (status["phase"], status["reason"], status["source"]) == (
        states.FAILED,
        states.REASON_RUNNER_CRASHED,
        states.SRC_WATCHDOG,
    ), harness.diagnostics()
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["c.pt"]
    assert _wait_dead([child["pid"]]) == []
    assert len(harness.server.calls) == 1


def test_runner_sigterm_tears_down_cleanly(harness):
    harness.prepare(run="echo c > $DGPU_CKPT/c.pt; echo started; sleep 300")
    harness.start()
    harness.wait_running()
    _first_heartbeat(harness)
    child = harness.child()
    os.kill(harness.pid("runner"), signal.SIGTERM)
    harness.wait_done()
    status = harness.status()
    assert (status["phase"], status["reason"], status["source"]) == (
        states.FAILED,
        states.REASON_RUNNER_CRASHED,
        states.SRC_RUNNER,
    ), harness.diagnostics()
    assert "SIGTERM" in status["message"]
    assert harness.status_history()[-2] == states.UPLOADING
    assert harness.remote_files(layout.ckpt_prefix(harness.job_id)) == ["c.pt"]
    assert _wait_dead([child["pid"]]) == []
    assert len(harness.server.calls) == 1


def test_watchdog_hard_deadline_kills_a_stuck_runner(harness):
    """Layer 2: runner stuck (SIGSTOP) -> at max_hours + final sync + margin the watchdog kills
    everything, records the deadline and destroys."""
    harness.tuning["watchdog_margin_s"] = 0.5
    harness.prepare(run="sleep 300", max_hours=0.0006, final_sync_timeout_min=0.005)
    harness.start()
    harness.wait_running()
    runner = harness.pid("runner")
    child = harness.child()
    os.kill(runner, signal.SIGSTOP)
    harness.wait_done(timeout=30)
    assert _wait_dead([runner, child["pid"]]) == []
    deadline = _event(harness, states.EV_DEADLINE_HIT)
    assert deadline["source"] == states.SRC_WATCHDOG
    status = harness.status()
    assert (status["phase"], status["source"]) == (states.TIMED_OUT, states.SRC_WATCHDOG)
    assert len(harness.server.calls) == 1


def test_self_destroy_retries_then_records_failure(harness):
    """Row: self-destroy API fails -> retry (10 x 30 s in production), self_destroy_failed,
    left for gc; the watchdog does not start another round."""
    harness.server.always_fail = True
    harness.prepare()
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED
    assert len(harness.server.calls) == harness.tuning["destroy_attempts"] == 3
    event = _event(harness, states.EV_SELF_DESTROY_FAILED)
    assert event["data"]["attempts"] == 3 and event["source"] == states.SRC_RUNNER
    assert json.loads((harness.work / "run" / "destroyed.json").read_text())["ok"] is False


# ---------------------------------------------------------------- restarts (Vast onstart reruns)


def test_restart_of_same_attempt_reuses_ready_data_and_resumes(harness):
    harness.put("datasets/d/a.txt", "x")
    run = (
        "n=$(cat $DGPU_OUTPUT/count 2>/dev/null || echo 0); echo $((n+1)) > $DGPU_OUTPUT/count; "
        "echo $DGPU_RESUME >> $DGPU_OUTPUT/resume; echo c > $DGPU_CKPT/c.pt; sleep 300"
    )
    harness.prepare(data=(("d", "datasets/d/"),), run=run)
    harness.start()
    harness.wait_running()
    harness.wait_for(lambda: (harness.work / "ckpt" / "c.pt").exists(), what="ckpt written")
    # container restart: every process dies at once
    for pid in (harness.pid("watchdog"), harness.pid("runner")):
        os.kill(pid, signal.SIGKILL)
    os.killpg(harness.child()["pgid"], signal.SIGKILL)
    _wait_dead([harness.pid("watchdog"), harness.pid("runner")])

    harness.start()
    harness.wait_for(
        lambda: (harness.work / "output" / "count").read_text().strip() == "2", what="second run"
    )
    harness.set_attempt(cancel=True)
    harness.wait_done()
    assert harness.status()["phase"] == states.CANCELLED, harness.diagnostics()
    data_copies = [c for c in harness.calls() if c["op"] == "copy" and "datasets/d" in c["args"][0]]
    assert len(data_copies) == 1
    code_fetches = [
        c for c in harness.calls() if c["op"] == "copyto" and "code.tar.gz" in c["args"][0]
    ]
    assert len(code_fetches) == 1
    assert (harness.work / "output" / "resume").read_text().split() == ["0", "1"]


def test_restart_after_finish_does_not_rerun_training(harness):
    harness.prepare(run="echo x >> $DGPU_OUTPUT/runs")
    harness.run_to_end()
    assert harness.status()["phase"] == states.SUCCEEDED
    history = harness.status_history()
    harness.run_to_end()
    assert (harness.work / "output" / "runs").read_text().split() == ["x"]
    assert harness.status_history() == history
    assert len(harness.server.calls) == 2, "destroy is retried; it is idempotent on the platform"


# ---------------------------------------------------------------- oldest supported Python


def test_success_path_on_python_38_or_39(tmp_path):
    """The runner ships to images with Python 3.8; run the whole flow on the oldest local one."""
    python = old_python()
    if python is None:
        pytest.skip("no Python 3.8 or 3.9 found by `uv python find`")
    h = Harness(tmp_path, python=python)
    try:
        h.put("datasets/d/a.txt", "x")
        h.prepare(
            data=(("d", "datasets/d/"),),
            setup="echo setup",
            run="test -f $DGPU_DATA/d/a.txt && echo ok > $DGPU_OUTPUT/ok && "
            "echo c > $DGPU_CKPT/c.pt",
        )
        h.run_to_end()
        assert h.status()["phase"] == states.SUCCEEDED, h.diagnostics()
        assert h.remote_files(layout.output_prefix(h.job_id)) == ["ok"]
        assert len(h.server.calls) == 1
        version = subprocess.run([python, "--version"], capture_output=True, text=True).stdout
        assert version.startswith(("Python 3.8", "Python 3.9"))
    finally:
        h.cleanup()
