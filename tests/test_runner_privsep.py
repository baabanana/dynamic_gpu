"""B4: privilege separation of the training process (runner/privsep.py).

The real uid/gid switch needs root, which the tests do not have. Every system call goes
through privsep.System, so these tests drive the decision logic, the directory ownership and
modes, the pip set-up and every fallback with a fake one; the switch itself is replaced by a
function that records its arguments from inside the forked child (preexec_fn).

Not covered until a real instance: that setgroups/setgid/setuid really drop root, that the
training then cannot read launch.json or /proc/<runner>/environ or signal the runner, that
useradd works in the image, that /dev/nvidia* are usable by the new user, and that
`pip install` with PIP_USER=1 works in the image's Python.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dgpu import layout, states
from dgpu.runner import privsep, procs
from dgpu.runner.util import Paths, load_tuning
from test_runner_harness import Harness

PY = sys.executable


class FakeSystem(privsep.System):
    """privsep.System with recorded side effects. chmod / lchown are only applied (chmod) or
    only recorded (lchown) inside `sandbox`, so nothing outside the test directory changes."""

    def __init__(
        self,
        sandbox,
        euid=0,
        users=None,
        useradd=True,
        useradd_rc=0,
        devices=(),
        device_mode=stat.S_IFCHR | 0o666,
        probe=None,
        switch=None,
    ):
        super().__init__()
        self.sandbox = str(sandbox)
        self.users = dict(users or {})
        self.groups = {gid for _uid, gid in self.users.values()}
        self.useradd_rc = useradd_rc
        self.useradd_calls = []
        self.chowns = []
        self.chmods = []
        self.probe = probe
        self.geteuid = lambda: euid
        self.getpwnam = self._getpwnam
        self.getpwuid = self._getpwuid
        self.getgrgid = self._getgrgid
        self.which = lambda name: "/usr/sbin/useradd" if useradd and name == "useradd" else None
        self.run = self._run
        self.lchown = lambda path, uid, gid: self.chowns.append((path, uid, gid))
        self.chmod = self._chmod
        self.devices = lambda: list(devices)
        self._device_mode = device_mode
        self.stat = self._stat
        self.switch = switch or (lambda uid, gid: None)

    def _getpwnam(self, name):
        if name not in self.users:
            raise KeyError(name)
        uid, gid = self.users[name]
        return SimpleNamespace(pw_uid=uid, pw_gid=gid)

    def _getpwuid(self, uid):
        for _name, (u, g) in self.users.items():
            if u == uid:
                return SimpleNamespace(pw_uid=u, pw_gid=g)
        raise KeyError(uid)

    def _getgrgid(self, gid):
        if gid in self.groups:
            return SimpleNamespace(gr_gid=gid)
        raise KeyError(gid)

    def _stat(self, path):
        if path.startswith("/dev/nvidia"):
            return SimpleNamespace(st_mode=self._device_mode)
        return os.stat(path)

    def _chmod(self, path, mode):
        self.chmods.append((path, mode))
        if path.startswith(self.sandbox):
            os.chmod(path, mode)

    def _run(self, argv, timeout, **kwargs):
        if argv[0].endswith("useradd"):
            self.useradd_calls.append(argv)
            if self.useradd_rc == 0:
                self.users[privsep.USER_NAME] = (1777, 1777)
                self.groups.add(1777)
            return procs.Completed(self.useradd_rc, b"", b"useradd: failure", False, False)
        if self.probe is not None:
            if isinstance(self.probe, Exception):
                raise self.probe
            return self.probe
        return procs.run_bounded(argv, timeout, **kwargs)


def _probe(**fields):
    info = {
        "version": "3.10.0",
        "prefix": "/opt/conda",
        "venv": False,
        "user_site_enabled": True,
        "user_site": "",
        "purelib": "/opt/conda/lib/python3.10/site-packages",
        "writable": False,
        "uid": fields.pop("uid", 1500),
    }
    info.update(fields)
    return procs.Completed(0, json.dumps(info).encode() + b"\n", b"", False, False)


@pytest.fixture
def layout_root(tmp_path):
    """A work root inside a parent directory that other users cannot traverse (like /root)."""
    parent = tmp_path / "private"
    parent.mkdir()
    parent.chmod(0o700)
    paths = Paths(str(parent / "dgpu"))
    paths.ensure()
    for folder in (paths.code, paths.data, paths.output, paths.ckpt):
        os.makedirs(folder, exist_ok=True)
    (open(os.path.join(paths.code, "train.py"), "w")).close()
    os.makedirs(os.path.join(paths.data, "coco8", "images"))
    (open(os.path.join(paths.data, "coco8", "images", "a.jpg"), "w")).close()
    os.symlink("/etc/passwd", os.path.join(paths.code, "link"))
    with open(paths.launch, "w") as handle:
        handle.write("{}")
    os.chmod(paths.launch, 0o644)
    os.makedirs(os.path.join(paths.root, "bundle-" + "ab" * 32, "dgpu"))
    with open(paths.diag_log("runner"), "w") as handle:
        handle.write("x")
    os.chmod(paths.diag_log("runner"), 0o644)
    return tmp_path, paths


def test_not_root_falls_back_without_touching_anything(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(sandbox, euid=1000)
    identity = privsep.prepare(paths, False, dict(os.environ), system=system)
    assert not identity.active and identity.reason == privsep.REASON_NOT_ROOT
    assert identity.preexec() is None and identity.env == {}
    assert system.chowns == [] and system.chmods == []
    assert "uid 1000" in identity.describe()


def test_run_as_root_requested_is_respected(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(sandbox, users={"dgpu": (1500, 1500)})
    identity = privsep.prepare(paths, True, dict(os.environ), system=system)
    assert not identity.active and identity.reason == privsep.REASON_REQUESTED
    assert system.chowns == []


def test_root_with_existing_user_sets_ownership_modes_and_env(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(sandbox, users={"dgpu": (1500, 1500)}, probe=_probe(writable=True))
    base = {"PATH": "/opt/conda/bin:/usr/bin", "KEEP": "1"}
    identity = privsep.prepare(paths, False, base, system=system)
    assert identity.active, identity.detail
    assert (identity.uid, identity.gid, identity.how) == (1500, 1500, "existing")
    owned = {path for path, uid, gid in system.chowns if (uid, gid) == (1500, 1500)}
    for top in (paths.code, paths.data, paths.output, paths.ckpt, paths.home):
        assert top in owned
    assert os.path.join(paths.data, "coco8", "images", "a.jpg") in owned
    assert os.path.join(paths.code, "link") in owned  # lchown: the link, not /etc/passwd
    for private in (paths.run, paths.logs, paths.launch, paths.boot_lock, paths.root):
        assert private not in owned
    assert not any(path.startswith(paths.run + os.sep) for path in owned)
    mode = lambda path: stat.S_IMODE(os.stat(path).st_mode)  # noqa: E731
    assert mode(paths.root) == 0o711
    assert mode(paths.run) == 0o700 and mode(paths.logs) == 0o700
    assert mode(paths.launch) == 0o600 and mode(paths.diag_log("runner")) == 0o600
    assert mode(os.path.join(paths.root, "bundle-" + "ab" * 32)) == 0o700
    assert mode(paths.home) == 0o700
    parent = os.path.dirname(paths.root)
    assert mode(parent) & 0o001, "the training user can traverse into the work root"
    assert identity.env["HOME"] == paths.home and identity.env["USER"] == "dgpu"
    assert identity.env["PYTHONUSERBASE"] == os.path.join(paths.home, ".local")
    assert identity.env["PATH"].split(":")[0] == os.path.join(paths.home, ".local", "bin")
    assert identity.env["PATH"].endswith("/opt/conda/bin:/usr/bin")
    assert "PIP_USER" not in identity.env, "site-packages is writable: pip works as is"


def test_root_owned_conda_uses_user_site_for_pip(layout_root):
    sandbox, paths = layout_root
    user_site = os.path.join(paths.home, ".local", "lib", "python3.10", "site-packages")
    system = FakeSystem(
        sandbox, users={"dgpu": (1500, 1500)}, probe=_probe(writable=False, user_site=user_site)
    )
    identity = privsep.prepare(paths, False, {"PATH": "/usr/bin"}, system=system)
    assert identity.active and identity.env["PIP_USER"] == "1"
    assert os.path.isdir(user_site), "created up front so python puts it on sys.path"
    assert (user_site, 1500, 1500) in system.chowns


def test_root_owned_venv_without_user_site_falls_back(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(
        sandbox,
        users={"dgpu": (1500, 1500)},
        probe=_probe(venv=True, user_site_enabled=False, writable=False, prefix="/venv/main"),
    )
    identity = privsep.prepare(paths, False, {}, system=system)
    assert not identity.active and identity.reason == privsep.REASON_PIP
    assert "/venv/main" in identity.detail


def test_useradd_creates_the_user_when_missing(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(sandbox, probe=_probe(writable=True, uid=1777))
    identity = privsep.prepare(paths, False, {}, system=system)
    assert identity.active and (identity.uid, identity.how) == (1777, "useradd")
    (argv,) = system.useradd_calls
    assert argv[-1] == "dgpu" and "--home-dir" in argv and paths.home in argv
    assert "--no-create-home" in argv


@pytest.mark.parametrize("useradd", [False, True])
def test_numeric_ids_without_useradd_or_when_it_fails(layout_root, useradd):
    sandbox, paths = layout_root
    system = FakeSystem(
        sandbox,
        users={"other": (privsep.FALLBACK_ID, privsep.FALLBACK_ID)},
        useradd=useradd,
        useradd_rc=1,
        probe=_probe(writable=True, uid=privsep.FALLBACK_ID + 1),
    )
    identity = privsep.prepare(paths, False, {}, system=system)
    assert identity.active and identity.how == "numeric"
    assert identity.uid == identity.gid == privsep.FALLBACK_ID + 1, "taken ids are skipped"
    assert identity.env["USER"] == "dgpu" and identity.env["HOME"] == paths.home


def test_gpu_devices_not_usable_by_other_users_fall_back(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(
        sandbox,
        users={"dgpu": (1500, 1500)},
        devices=["/dev/nvidia0", "/dev/nvidiactl"],
        device_mode=stat.S_IFCHR | 0o660,
        probe=_probe(writable=True),
    )
    identity = privsep.prepare(paths, False, {}, system=system)
    assert not identity.active and identity.reason == privsep.REASON_GPU_DEVICES
    assert "/dev/nvidia0" in identity.detail
    assert system.chowns == [], "decided before touching the layout"


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        (OSError(2, "No such file or directory: 'python3'"), privsep.REASON_PYTHON),
        (subprocess.SubprocessError("Exception occurred in preexec_fn."), privsep.REASON_SWITCH),
        (
            procs.Completed(1, b"", b"python3: Permission denied\n", False, False),
            privsep.REASON_PYTHON,
        ),
        (procs.Completed(None, b"", b"", True, True), privsep.REASON_PYTHON),
        (_probe(writable=True, uid=0), privsep.REASON_SWITCH),
    ],
    ids=["no-python", "switch-failed", "permission-denied", "hung", "still-root"],
)
def test_probe_failures_fall_back(layout_root, probe, reason):
    sandbox, paths = layout_root
    system = FakeSystem(sandbox, users={"dgpu": (1500, 1500)}, probe=probe)
    identity = privsep.prepare(paths, False, {}, system=system)
    assert not identity.active and identity.reason == reason, identity.detail


def test_layout_errors_fall_back(layout_root):
    sandbox, paths = layout_root
    system = FakeSystem(sandbox, users={"dgpu": (1500, 1500)}, probe=_probe(writable=True))

    def refuse(path, uid, gid):
        raise PermissionError(1, "Operation not permitted", path)

    system.lchown = refuse
    identity = privsep.prepare(paths, False, {}, system=system)
    assert not identity.active and identity.reason == privsep.REASON_LAYOUT


def test_real_switch_without_root_is_a_detectable_failure(tmp_path):
    """Not root: the real setgroups/setgid/setuid fail in the child, Popen reports it, and
    prepare() falls back (the 'no permission' case)."""
    result = subprocess.run(
        [PY, "-c", "from dgpu.runner.privsep import switch_user; switch_user(47100, 47100)"],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(privsep.__file__).rsplit("/dgpu/", 1)[0]),
    )
    if os.geteuid() == 0:
        pytest.skip("running as root")
    assert result.returncode != 0 and "PermissionError" in result.stderr
    paths = Paths(str(tmp_path / "w"))
    paths.ensure()
    system = FakeSystem(tmp_path, users={"dgpu": (47100, 47100)}, switch=privsep.switch_user)
    identity = privsep.prepare(paths, False, dict(os.environ), system=system)
    assert not identity.active and identity.reason == privsep.REASON_SWITCH, identity.detail


def test_preexec_calls_the_switch_with_the_identity():
    calls = []
    identity = privsep.Identity(True, uid=5, gid=6, switch=lambda u, g: calls.append((u, g)))
    identity.preexec()()
    assert calls == [(5, 6)]
    assert privsep.inactive("x", "y").preexec() is None


# ---------------------------------------------------------------- the runner uses it


class CountingDestroyer:
    def destroy(self):
        from dgpu.runner.destroy import DestroyResult

        return DestroyResult(True, 1, "fake", None)


def test_runner_runs_setup_and_run_through_the_identity(tmp_path):
    """In-process runner as 'root' (fake system): setup and run get the training user's
    environment and go through the uid/gid switch; the decision is recorded."""
    h = Harness(tmp_path)
    try:
        marker = tmp_path / "switched"
        user_site = str(h.work / "home" / ".local" / "lib" / "python3.10" / "site-packages")

        def switch(uid, gid):  # runs in the forked child
            with open(marker, "a") as handle:
                handle.write(f"{uid}:{gid}\n")

        system = FakeSystem(
            tmp_path,
            users={"dgpu": (1500, 1500)},
            probe=_probe(writable=False, user_site=user_site),
            switch=switch,
        )
        h.prepare(
            setup="echo setup-as-$USER > $DGPU_OUTPUT/setup.txt",
            run="env > $DGPU_OUTPUT/env.txt",
        )
        from dgpu.runner.main import Runner

        runner = Runner(
            Paths(str(h.work)),
            tuning=load_tuning({"DGPU_RUNNER_TUNING": json.dumps(h.tuning)}),
            environ=h.env(),
            destroyer=CountingDestroyer(),
            gpu_query=lambda timeout: ([{"index": "0", "name": "fake", "util": 50.0}], ""),
            privsep_system=system,
        )
        assert runner.run() == states.EXIT_SUCCEEDED
        assert marker.read_text().split() == ["1500:1500", "1500:1500"]
        env_txt = h.get(layout.output_prefix(h.job_id) + "env.txt").decode()
        assert f"HOME={h.work / 'home'}" in env_txt and "USER=dgpu" in env_txt
        assert "PIP_USER=1" in env_txt
        assert f"PYTHONUSERBASE={h.work / 'home' / '.local'}" in env_txt
        assert h.get(layout.output_prefix(h.job_id) + "setup.txt") == b"setup-as-dgpu\n"
        record = json.loads((h.work / "run" / "privsep.json").read_text())
        assert record["active"] is True and record["uid"] == 1500
        assert not [e for e in h.events() if e["name"] == states.EV_RUN_AS_ROOT]
        assert stat.S_IMODE(os.stat(h.work / "launch.json").st_mode) == 0o600
        assert stat.S_IMODE(os.stat(h.work / "run").st_mode) == 0o700
        assert "无特权用户 dgpu" in h.log_text()
    finally:
        h.cleanup()


def test_remote_spec_carries_run_as_root():
    from pathlib import Path

    from dgpu.contracts import GpuSpec, JobSpec, Limits, RunnerSettings, build_remote_spec

    def spec(**kw):
        return JobSpec(
            version=1,
            name="n",
            backend="vast",
            images={"vast": "img:1"},
            code=Path("/x"),
            run="true",
            limits=Limits(max_hours=1, max_usd_per_hour=1),
            gpu=GpuSpec(names=("RTX 3090",)),
            **kw,
        )

    remote = build_remote_spec(spec(), job_id="j", backend="vast", data=[], runner=RunnerSettings())
    assert remote["run_as_root"] is False
    remote = build_remote_spec(
        spec(run_as_root=True), job_id="j", backend="vast", data=[], runner=RunnerSettings()
    )
    assert remote["run_as_root"] is True
