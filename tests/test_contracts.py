"""阶段 0 契约的测试：标签、路径、文档格式，以及测试替身是否满足协议。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dgpu import layout, states
from dgpu.contracts import (
    INST_GONE,
    Backend,
    Bundler,
    DataSource,
    GpuRequest,
    GpuSpec,
    InstanceRef,
    JobSpec,
    LaunchSpec,
    Limits,
    OfferUnavailable,
    PreconditionFailed,
    RunnerSettings,
    SpecError,
    Store,
    build_launch_doc,
    build_remote_spec,
)
from dgpu.testing import TEST_PROFILE, FakeBackend, FakeBundler, MemoryStore, make_offer


def test_job_id_is_valid_and_label_round_trips():
    job_id = layout.new_job_id("YOLO demo_v2!", now=0)
    assert layout.is_valid_job_id(job_id)
    assert job_id.startswith("yolo-demo-v2-700101000000-")
    assert layout.parse_label(layout.label(job_id, 3)) == (job_id, 3)


@pytest.mark.parametrize("text", [None, "", "my-training", "dgpu-", "dgpu-abc", "dgpu-abc-a", "x"])
def test_parse_label_rejects_foreign_labels(text):
    assert layout.parse_label(text) is None


def test_label_rejects_bad_input():
    with pytest.raises(ValueError):
        layout.label("Bad_ID", 1)
    with pytest.raises(ValueError):
        layout.label("ok", 0)


def test_keys_live_under_job_prefix():
    prefix = layout.job_prefix("j1")
    for key in (
        layout.spec_key("j1"),
        layout.code_key("j1"),
        layout.attempt_key("j1"),
        layout.secrets_key("j1", 2),
        layout.status_key("j1"),
        layout.heartbeat_key("j1"),
        layout.event_key("j1", states.SRC_CLI),
        layout.log_key("j1", 1),
        layout.output_prefix("j1"),
        layout.ckpt_prefix("j1"),
    ):
        assert key.startswith(prefix)
    assert not layout.dataset_complete_key("h").startswith(layout.dataset_prefix("h"))


def test_event_keys_sort_by_time():
    assert layout.event_key("j", "cli", ts=1.0) < layout.event_key("j", "cli", ts=2.0)


def test_event_doc_rejects_unknown_name():
    with pytest.raises(ValueError):
        states.new_event_doc("made_up", states.SRC_CLI, "j1")


def test_check_version_rejects_unknown_version():
    doc = states.new_status_doc("j1", 1, states.RUNNING, states.SRC_RUNNER)
    assert states.check_version(doc, "status") is doc
    with pytest.raises(ValueError):
        states.check_version({"version": 99}, "status")


def _spec() -> JobSpec:
    return JobSpec(
        version=1,
        name="demo",
        backend="vast",
        images={"vast": "vastai/pytorch:1.0"},
        code=Path("/tmp/code"),
        run="python train.py",
        limits=Limits(max_hours=2, max_usd_per_hour=0.3),
        gpu=GpuSpec(names=("RTX 3090",)),
        env={"EPOCHS": "3"},
    )


def test_remote_spec_has_no_credentials_and_launch_doc_has_them():
    spec = _spec()
    remote = build_remote_spec(
        spec,
        job_id="j1",
        backend="vast",
        data=[DataSource("coco8", "bkt", "coco8/")],
        runner=RunnerSettings(),
    )
    assert remote["image"] == "vastai/pytorch:1.0"
    assert remote["data"] == [{"name": "coco8", "bucket": "bkt", "prefix": "coco8/"}]
    assert TEST_PROFILE.secret_access_key not in repr(remote)

    launch = build_launch_doc(job_id="j1", attempt=1, backend="vast", store=TEST_PROFILE)
    assert launch["store"]["secret_access_key"] == TEST_PROFILE.secret_access_key
    assert TEST_PROFILE.secret_access_key not in repr(TEST_PROFILE)


def test_spec_reports_missing_backend_fields():
    spec = _spec()
    with pytest.raises(SpecError):
        spec.image_for("autodl")
    with pytest.raises(SpecError):
        spec.limits.max_price("autodl")


def test_fakes_satisfy_protocols():
    assert isinstance(MemoryStore(), Store)
    assert isinstance(FakeBackend(), Backend)
    assert isinstance(FakeBundler(), Bundler)


def test_memory_store_conditional_writes():
    store = MemoryStore()
    etag = store.put_json("k", {"a": 1}, if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put_json("k", {"a": 2}, if_none_match=True)
    new_etag = store.put_json("k", {"a": 2}, if_match=etag)
    with pytest.raises(PreconditionFailed):
        store.put_json("k", {"a": 3}, if_match=etag)
    assert store.get_json("k") == ({"a": 2}, new_etag)
    assert store.get_json("missing") is None


def test_memory_store_listing_and_ranges(tmp_path):
    store = MemoryStore()
    store.put_bytes("p/a/1.txt", b"hello")
    store.put_bytes("p/b/2.txt", b"x")
    store.put_bytes("p/top.txt", b"y")
    assert store.list_dirs("p/") == ["a", "b"]
    assert store.get_bytes("p/a/1.txt", start=2) == b"llo"
    assert store.get_bytes("p/a/1.txt", start=99) == b""
    assert store.download_prefix("p/", tmp_path) == 3
    assert (tmp_path / "a" / "1.txt").read_bytes() == b"hello"
    store.delete("p/top.txt")
    store.delete("p/top.txt")


def test_fake_backend_lifecycle_and_label_filter():
    backend = FakeBackend(offers=[make_offer("o1", price=0.3), make_offer("o2", price=0.1)])
    backend.unavailable_offer_ids.add("o2")
    backend.add_foreign("f1")
    req = GpuRequest(("RTX 3090",), 1, 0.5, 1.0, 40, "img")
    offers = backend.search(req)
    assert [offer.id for offer in offers] == ["o2", "o1"]

    launch = LaunchSpec("j1", 1, "img", "echo hi", 40, 0.5)
    with pytest.raises(OfferUnavailable):
        backend.create(offers[0], launch)
    ref = backend.create(offers[1], launch)
    assert [state.ref for state in backend.list()] == [ref]

    backend.destroy(ref)
    backend.destroy(ref)
    assert backend.destroyed == [ref.id]
    assert backend.get(ref).phase == INST_GONE
    with pytest.raises(AssertionError):
        backend.destroy(InstanceRef("vast", "f1", "someone-elses-work"))
