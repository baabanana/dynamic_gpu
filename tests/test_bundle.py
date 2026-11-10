"""Tests for dgpu.bundle: deterministic bundle, pinned rclone, bootstrap command."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest

import dgpu
from dgpu import bundle, layout
from dgpu.bundle import BundleError, RunnerBundler
from dgpu.contracts import Bundler
from dgpu.testing import TEST_PROFILE, MemoryStore

FAKE_RCLONE = b"#!/bin/sh\necho fake rclone\n"
FAKE_VERSION = "9.9.9"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip(binary: bytes = FAKE_RCLONE, version: str = FAKE_VERSION) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(f"rclone-v{version}-linux-amd64/README.txt", "readme")
        archive.writestr(f"rclone-v{version}-linux-amd64/rclone", binary)
    return buf.getvalue()


class Fetcher:
    def __init__(self, data: bytes):
        self.data = data
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.data


def _bundler(tmp_path: Path, fetch=None, **kwargs) -> RunnerBundler:
    zip_bytes = _zip()
    options = {
        "cache_dir": tmp_path / "cache",
        "fetch": fetch or Fetcher(zip_bytes),
        "rclone_version": FAKE_VERSION,
        "rclone_zip_sha256": _sha(zip_bytes),
        "rclone_binary_sha256": _sha(FAKE_RCLONE),
    }
    options.update(kwargs)
    return RunnerBundler(**options)


def _members(path: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(path) as tar:
        return {member.name: member for member in tar.getmembers()}


def test_bundler_satisfies_the_protocol(tmp_path):
    assert isinstance(_bundler(tmp_path), Bundler)


def test_bundle_is_deterministic_and_contains_only_runner_files(tmp_path):
    first_path, first_sha = _bundler(tmp_path / "a").build()
    second_path, second_sha = _bundler(tmp_path / "b").build()
    assert first_sha == second_sha == _sha(first_path.read_bytes())
    assert first_path.read_bytes() == second_path.read_bytes()

    package = Path(dgpu.__file__).parent
    runner_files = {
        "dgpu/runner/" + p.relative_to(package / "runner").as_posix()
        for p in (package / "runner").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    expected_files = {"dgpu/__init__.py", "dgpu/layout.py", "dgpu/states.py", "bin/rclone"}
    expected_files |= runner_files
    members = _members(first_path)
    files = {name for name, info in members.items() if info.isfile()}
    dirs = {name for name, info in members.items() if info.isdir()}
    assert files == expected_files
    assert dirs == {"bin", "dgpu", "dgpu/runner"}
    assert "dgpu/contracts.py" not in files and "dgpu/testing.py" not in files
    assert "dgpu/runner/main.py" in files and "dgpu/runner/boot.py" in files
    for name, info in members.items():
        assert info.mtime == 0 and info.uid == 0 and info.gid == 0 and not info.uname
        if name == "bin/rclone":
            assert info.mode == 0o755
        elif info.isfile():
            assert info.mode == 0o644
    with tarfile.open(first_path) as tar:
        assert tar.extractfile("bin/rclone").read() == FAKE_RCLONE
        assert tar.extractfile("dgpu/states.py").read() == (package / "states.py").read_bytes()
    with gzip.open(first_path) as handle:
        handle.read()
    assert first_path.read_bytes()[4:8] == b"\0\0\0\0", "gzip header mtime is zero"


def test_bundle_contains_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "env-secret-should-not-leak")
    path, _sha256 = _bundler(tmp_path).build()
    raw = gzip.decompress(path.read_bytes())
    for secret in (
        b"env-secret-should-not-leak",
        TEST_PROFILE.secret_access_key.encode(),
        TEST_PROFILE.access_key_id.encode(),
    ):
        assert secret not in raw


def test_build_reuses_the_cached_bundle(tmp_path):
    bundler = _bundler(tmp_path)
    path, sha = bundler.build()
    mtime = path.stat().st_mtime_ns
    again_path, again_sha = bundler.build()
    assert (again_path, again_sha) == (path, sha)
    assert again_path.stat().st_mtime_ns == mtime


def test_ensure_uploaded_uploads_once(tmp_path):
    store = MemoryStore()
    puts = []
    original = store.put_file
    store.put_file = lambda key, p: (puts.append(key), original(key, p))[1]
    bundler = _bundler(tmp_path)
    ref = bundler.ensure_uploaded(store)
    assert ref.key == layout.tools_bundle_key(ref.sha256)
    assert _sha(store.objects[ref.key]) == ref.sha256
    assert bundler.ensure_uploaded(store) == ref
    assert puts == [ref.key]


def test_rclone_is_downloaded_once_verified_and_cached(tmp_path):
    fetch = Fetcher(_zip())
    bundler = _bundler(tmp_path, fetch=fetch)
    path = bundler.rclone_binary()
    assert path == tmp_path / "cache" / f"rclone-{FAKE_VERSION}-linux-amd64" / "rclone"
    assert path.read_bytes() == FAKE_RCLONE and path.stat().st_mode & 0o777 == 0o755
    assert fetch.urls == [
        f"https://downloads.rclone.org/v{FAKE_VERSION}/rclone-v{FAKE_VERSION}-linux-amd64.zip"
    ]
    bundler.rclone_binary()
    assert len(fetch.urls) == 1
    path.write_bytes(b"tampered")
    assert bundler.rclone_binary().read_bytes() == FAKE_RCLONE
    assert len(fetch.urls) == 2


def test_rclone_with_wrong_zip_hash_is_rejected(tmp_path):
    fetch = Fetcher(_zip(binary=b"evil"))
    bundler = _bundler(tmp_path, fetch=fetch)
    with pytest.raises(BundleError, match="sha256"):
        bundler.rclone_binary()
    assert not bundler.rclone_cache_path.exists()


def test_rclone_with_wrong_binary_hash_is_rejected(tmp_path):
    zip_bytes = _zip()
    bundler = _bundler(tmp_path, fetch=Fetcher(zip_bytes), rclone_binary_sha256="0" * 64)
    with pytest.raises(BundleError):
        bundler.rclone_binary()
    assert not bundler.rclone_cache_path.exists()


@pytest.mark.parametrize("field", ["rclone_zip_sha256", "rclone_binary_sha256"])
@pytest.mark.parametrize("value", [bundle.UNPINNED, "", "abc"])
def test_unpinned_rclone_hash_refuses_to_build(tmp_path, field, value):
    fetch = Fetcher(_zip())
    bundler = _bundler(tmp_path, fetch=fetch, **{field: value})
    with pytest.raises(BundleError, match="没有固定"):
        bundler.build()
    assert fetch.urls == []


def test_real_rclone_pins_are_set():
    for value in (bundle.RCLONE_ZIP_SHA256, bundle.RCLONE_BINARY_SHA256):
        assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert re.fullmatch(r"\d+\.\d+\.\d+", bundle.RCLONE_VERSION)
    assert bundle.DEFAULT_CACHE_DIR == Path.home() / ".cache" / "dgpu"


def test_download_failure_explains_the_manual_fallback(tmp_path):
    def broken(url):
        raise OSError("network down")

    bundler = _bundler(tmp_path, fetch=broken)
    with pytest.raises(BundleError, match="手动下载"):
        bundler.rclone_binary()


# ---------------------------------------------------------------- bootstrap command

SHA = "ab" * 32


def _presigned(key: str) -> str:
    return (
        "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com/my-training-bucket/"
        + key
        + "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0123456789abcdef0123456789abcdef"
        "%2F20261110%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261110T101010Z&X-Amz-Expires=4500"
        "&X-Amz-SignedHeaders=host&X-Amz-Signature=" + "f" * 64
    )


def test_bootstrap_command_fits_vast_onstart_with_realistic_urls():
    job_id = "a-rather-long-job-name-x-261110101010-abcd"
    command = RunnerBundler().bootstrap_command(
        bundle_url=_presigned(layout.tools_bundle_key(SHA)),
        bundle_sha256=SHA,
        launch_url=_presigned(layout.secrets_key(job_id, 12)),
    )
    assert len(command) <= bundle.MAX_BOOTSTRAP_LEN
    assert subprocess.run(["bash", "-n", "-c", command]).returncode == 0
    assert TEST_PROFILE.secret_access_key not in command


def test_bootstrap_command_rejects_bad_input():
    bundler = RunnerBundler()
    url = "https://example.invalid/x"
    with pytest.raises(BundleError):
        bundler.bootstrap_command(bundle_url=url, bundle_sha256="nothex", launch_url=url)
    with pytest.raises(BundleError):
        bundler.bootstrap_command(bundle_url="ftp://x", bundle_sha256=SHA, launch_url=url)
    with pytest.raises(BundleError):
        bundler.bootstrap_command(bundle_url=url + "\nrm -rf /", bundle_sha256=SHA, launch_url=url)
    with pytest.raises(BundleError, match="4000"):
        bundler.bootstrap_command(
            bundle_url=url + "?" + "a" * 3000, bundle_sha256=SHA, launch_url=url + "?" + "b" * 1000
        )


def test_bootstrap_command_quotes_hostile_urls(tmp_path):
    """Whatever the URL contains, python3 receives it as one exact argument."""
    capture = tmp_path / "argv.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    dump = f"open({str(capture)!r}, 'w').write(json.dumps(sys.argv[1:]))"
    (fake_bin / "python3").write_text(f"#!{sys.executable}\nimport json, sys\n{dump}\n")
    (fake_bin / "python3").chmod(0o755)
    hostile = "https://x.invalid/a'b\"c$(touch pwned)`id`;&|<>*?~ %0A{d}"
    launch = "https://x.invalid/l?sig=a&b=$HOME"
    command = RunnerBundler().bootstrap_command(
        bundle_url=hostile, bundle_sha256=SHA, launch_url=launch
    )
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=env)
    assert result.returncode == 0
    argv = json.loads(capture.read_text())
    assert argv[0] == "-c" and argv[1].startswith("import base64,zlib;")
    assert argv[2:] == [hostile, SHA, launch]
    assert not (tmp_path / "pwned").exists()


def test_bootstrap_loader_decodes_to_the_documented_source():
    command = RunnerBundler().bootstrap_command(
        bundle_url="https://x.invalid/b", bundle_sha256=SHA, launch_url="https://x.invalid/l"
    )
    loader = shlex.split(command)[3]
    encoded = re.search(r'b85decode\("([^"\\]+)"\)', loader).group(1)
    assert zlib.decompress(base64.b85decode(encoded)).decode() == bundle.BOOTSTRAP_SOURCE
    compile(bundle.BOOTSTRAP_SOURCE, "<bootstrap>", "exec")


def test_bootstrap_command_fits_with_the_longest_job_id_and_bucket_name():
    """F1 的自毁逻辑让启动命令变长：最长的任务 ID（42 字符）、最长的 bucket 名（63 字符）、
    两位数的 attempt 下也不超过 4000，并留出余量。"""
    job_id = layout.new_job_id("x" * 60, now=0)
    assert len(job_id) == 42
    bucket = "b" * 63

    def presigned(key: str) -> str:
        return (
            "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com/"
            + bucket
            + "/"
            + key
            + "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0123456789abcdef0123456789abcdef"
            "%2F20261110%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261110T101010Z"
            "&X-Amz-Expires=4500&X-Amz-SignedHeaders=host&X-Amz-Signature=" + "f" * 64
        )

    command = RunnerBundler().bootstrap_command(
        bundle_url=presigned(layout.tools_bundle_key(SHA)),
        bundle_sha256=SHA,
        launch_url=presigned(layout.secrets_key(job_id, 99)),
    )
    assert len(command) <= bundle.MAX_BOOTSTRAP_LEN - 200, len(command)
