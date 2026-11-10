"""Second-round review (docs/dgpu-review-2026-11-10.md, section E), instance side: F2 and F6.

Real runner / watchdog processes (or in-process instances) against the fake rclone of
test_runner_fakes.py. Each test fails without its fix.
"""

from __future__ import annotations

import json
import time

import pytest

from dgpu import layout, states
from dgpu.runner import privsep, syncstate, util
from dgpu.runner.main import sync_dir
from dgpu.runner.rclone import RcloneError
from dgpu.runner.util import Paths, runner_settings
from test_runner_harness import Harness
from test_runner_review import (
    CountingDestroyer,
    _events,
    _fake_runner,
    _started_long_ago,
    _watchdog,
)


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.cleanup()


def _copies_of(h: Harness, folder: str) -> list[dict]:
    return [c for c in h.calls() if c["op"] == "copy" and any(folder in a for a in c["args"])]


# ================================================================ F2


def test_f2_storage_down_for_transfers_finishes_fast_without_final_sync_failed(harness):
    """Reviewer's reproduction: the code download fails because every transfer fails (storage
    down for copies). There are no local results, so there is no final sync at all: no retries
    of empty output/ and ckpt/ until final_sync_timeout_min, no spurious final_sync_failed."""
    h = harness
    h.prepare(run="true", final_sync_timeout_min=0.1)  # 6 s here, 20 min in production
    h.add_rule("copyto", "code.tar.gz", mode="fail")
    h.add_rule("copy", "", mode="fail")
    started = time.monotonic()
    h.run_to_end(60)
    elapsed = time.monotonic() - started
    status = h.status()
    assert (status["phase"], status["reason"]) == (states.FAILED, states.REASON_DATA_SYNC)
    assert not _events(h, states.EV_FINAL_SYNC_FAILED)
    assert _copies_of(h, "/output") == [] and _copies_of(h, "/ckpt") == []
    assert states.UPLOADING not in h.status_history()
    assert elapsed < 6, elapsed


def test_f2_sync_dir_with_nothing_new_to_upload_does_not_call_rclone(tmp_path):
    paths = Paths(str(tmp_path / "work"))
    paths.ensure()

    class DownStorage:
        calls = 0

        def copy_up(self, *args, **kwargs):
            DownStorage.calls += 1
            raise RcloneError("copy", "storage unreachable")

    rclone = DownStorage()
    args = ("output", paths.output, "dgpu/jobs/j/output/", False, 5.0, "final")
    # empty directory: nothing to upload
    assert sync_dir(rclone, paths, *args) == (True, None)
    # everything already uploaded (the sync record matches): still nothing to upload
    model = tmp_path / "work" / "output" / "model.pt"
    model.write_text("weights")
    syncstate.record_success(paths.run, "output", syncstate.snapshot(paths.output, False))
    assert sync_dir(rclone, paths, *args) == (True, None)
    assert DownStorage.calls == 0
    # a changed file has to be uploaded: rclone is called and its failure is reported
    model.write_text("newer weights!")
    ok, error = sync_dir(rclone, paths, *args)
    assert not ok and "unreachable" in error and DownStorage.calls == 1


def test_f2_final_sync_after_everything_was_uploaded_survives_a_storage_outage(harness):
    """Results already uploaded by the periodic sync; storage then goes down for transfers.
    The final sync has nothing new to upload, so the job still SUCCEEDS without retrying."""
    h = harness
    h.prepare(
        run="echo w > $DGPU_OUTPUT/model.pt; echo c > $DGPU_CKPT/c.pt; sleep 2",
        final_sync_timeout_min=0.1,
    )
    h.start()
    h.wait_for(
        lambda: h.remote_files(layout.output_prefix(h.job_id))
        and h.remote_files(layout.ckpt_prefix(h.job_id)),
        what="the periodic sync to upload both folders",
    )
    h.add_rule("copy", "", mode="fail")
    h.wait_done(60)
    status = h.status()
    assert status["phase"] == states.SUCCEEDED, h.diagnostics()
    assert status["not_uploaded"] == []
    assert not _events(h, states.EV_FINAL_SYNC_FAILED)


# ================================================================ F6


def test_f6_watchdog_after_a_long_restart_gives_the_final_sync_the_runners_budget(harness):
    """The container was down past the watchdog's hard deadline. On restart the watchdog takes
    over at once; its sync must get the runner's final sync budget (final_sync_timeout_min),
    not the shorter watchdog_sync_timeout_s, so a slow but working upload completes."""
    h = harness
    spec = h.prepare(run="true", max_hours=0.0001, final_sync_timeout_min=0.1)  # 6 s
    (h.work / "run").mkdir(parents=True, exist_ok=True)
    (h.work / "run" / "spec.json").write_text(json.dumps(spec))
    (h.work / "output").mkdir(exist_ok=True)
    (h.work / "output" / "model.pt").write_text("large results")
    h.tuning.update(
        watchdog_sync_timeout_s=1.0,  # the watchdog's own (shorter) cap
        watchdog_margin_s=0.1,
        teardown_budget_s=4.0,
        small_op_timeout_s=2.0,
        thread_join_s=0.1,
        destroy_attempts=1,
        destroy_http_timeout_s=1.0,
    )
    h.add_rule("copy", "/output", mode="delay", delay=2.5)  # slow, but it works
    _started_long_ago(h)
    _fake_runner(h)
    watchdog = _watchdog(h, CountingDestroyer())
    watchdog.run()
    status = h.status()
    assert status["phase"] == states.TIMED_OUT, status
    assert status["not_uploaded"] == [], status
    assert h.remote_files(layout.output_prefix(h.job_id)) == ["model.pt"]


def test_f6_lifetime_formula_includes_the_restart_takeover_sync():
    settings = runner_settings({})
    tuning = util.DEFAULT_TUNING
    restart = util.watchdog_sync_s(settings, tuning, restart=True)
    assert restart == settings["final_sync_timeout_min"] * 60.0  # the runner's budget
    assert util.watchdog_sync_s(settings, tuning) <= restart
    worst = util.watchdog_teardown_worst_case_s(settings, tuning)
    assert worst >= restart + tuning["teardown_budget_s"] + util.destroy_worst_case_s(tuning)
    lifetime = util.instance_lifetime_s(2.0, settings, tuning)
    assert lifetime == util.watchdog_hard_deadline_s(2.0, settings, tuning) + worst


# ================================================================ F7 / F8: visible fallback


def test_f7_fallback_reason_reaches_the_heartbeat_the_event_and_one_log_line(harness):
    """The tests do not run as root, so the training is not dropped (reason not_root). The
    reason must be visible: in the heartbeat (dgpu status), in the run_as_root event with a
    human-readable reason_text, and as exactly one line in the attempt log."""
    h = harness
    h.prepare(run="true")
    h.run_to_end()
    assert h.status()["phase"] == states.SUCCEEDED, h.diagnostics()
    identity = h.heartbeat().get("identity")
    assert identity and identity["dropped"] is False and identity["reason"] == "not_root"
    assert identity["reason_text"] == privsep.REASON_TEXT["not_root"]
    (event,) = _events(h, states.EV_RUN_AS_ROOT)
    assert event["data"]["reason_text"] == privsep.REASON_TEXT["not_root"]
    assert h.log_text().count("训练进程没有降权") == 1
