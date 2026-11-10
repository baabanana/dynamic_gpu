"""代码打包和数据集上传：文件列表、排除规则、确定性 tar.gz、上传幂等和中断重传。"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from dgpu import layout
from dgpu.contracts import DataRef, DataSource, SpecError, StoreError
from dgpu.packaging import (
    FIXED_MTIME,
    hash_dataset,
    list_code_files,
    pack_code,
    resolve_remote,
    upload_dataset,
)
from dgpu.testing import MemoryStore

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")


@pytest.fixture(autouse=True)
def _isolate_git(tmp_path, monkeypatch):
    """tmp_path 之外的 git 仓库不影响测试；不读用户的 git 配置。"""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _tree(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode()
        path.write_bytes(content)
    return root


def _names(paths: list[Path]) -> list[str]:
    return [path.as_posix() for path in paths]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _members(archive: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(archive, "r:gz") as tar:
        return {member.name: member for member in tar.getmembers()}


SECRETS_AND_JUNK = {
    ".env": "TOKEN=1",
    ".env.local": "TOKEN=2",
    ".envrc": "export TOKEN=3",
    "server.pem": "-----BEGIN",
    "deploy/id.key": "key",
    "sub/.env": "TOKEN=4",
    ".git/config": "[core]",
    "__pycache__/train.cpython-311.pyc": b"\x00",
    "sub/__pycache__/util.cpython-311.pyc": b"\x00",
    ".venv/bin/python": "#!",
    ".env.d/settings": "x",
}
CODE = {
    "train.py": "print('train')\n",
    "sub/util.py": "X = 1\n",
    "README.md": "# demo\n",
    "keys.txt": "not a secret file name\n",
    "environment.yml": "name: demo\n",
}


# ---------------------------------------------------------------- 文件列表（非 git）


def test_walk_excludes_secrets_and_junk_at_any_depth(tmp_path):
    root = _tree(tmp_path / "code", {**CODE, **SECRETS_AND_JUNK})
    assert _names(list_code_files(root)) == sorted(CODE)


def test_dgpuignore_supported_patterns(tmp_path):
    root = _tree(
        tmp_path / "code",
        {
            "train.py": "",
            "weights/best.ckpt": "",
            "deep/a/b/last.ckpt": "",
            "runs/exp1/log.txt": "",
            "sub/runs/x.txt": "",
            "notes/runs": "a file named runs, not a directory",
            "build/out.bin": "",
            "src/build/keep.py": "",
            "docs/manual.pdf": "",
            "docs/guide.md": "",
            "other/docs/manual.pdf": "",
            "data[1].csv": "",
            ".dgpuignore": ("# 注释\n\n*.ckpt\nruns/\n/build\ndocs/*.pdf\ndata[[]1].csv\n"),
        },
    )
    assert _names(list_code_files(root)) == [
        ".dgpuignore",
        "docs/guide.md",
        "notes/runs",
        "other/docs/manual.pdf",
        "src/build/keep.py",
        "train.py",
    ]


def test_dgpuignore_negation_is_rejected_not_ignored(tmp_path):
    root = _tree(tmp_path / "code", {"a.py": "", ".dgpuignore": "*.log\n!keep.log\n"})
    with pytest.raises(SpecError, match="第 2 行.*不支持 !"):
        list_code_files(root)


def test_extra_excludes_drop_data_dirs_inside_code(tmp_path):
    root = _tree(
        tmp_path / "code", {"train.py": "", "datasets/set1/a.jpg": "", "datasets/b.txt": ""}
    )
    files = list_code_files(root, exclude=[root / "datasets" / "set1", tmp_path / "outside"])
    assert _names(files) == ["datasets/b.txt", "train.py"]


def test_missing_code_dir(tmp_path):
    with pytest.raises(SpecError, match="不存在"):
        list_code_files(tmp_path / "nope")


# ---------------------------------------------------------------- 文件列表（git）


@needs_git
def test_git_lists_tracked_and_untracked_but_not_ignored(tmp_path):
    repo = _tree(
        tmp_path / "repo",
        {
            **CODE,
            ".gitignore": "data/\n*.log\n",
            "data/big.bin": b"\x00" * 10,
            "train.log": "log",
            "gone.py": "deleted later",
            ".env": "TOKEN=1",
        },
    )
    _git(repo, "init", "-q")
    _git(repo, "add", "train.py", "gone.py", ".gitignore")
    _git(repo, "add", "-f", ".env")  # 误提交的 .env 也必须被排除
    (repo / "gone.py").unlink()  # 已跟踪但在工作区里删掉了

    assert _names(list_code_files(repo)) == sorted([*CODE, ".gitignore"])


@needs_git
def test_git_listing_is_scoped_to_code_subdirectory(tmp_path):
    repo = _tree(
        tmp_path / "repo",
        {"top.py": "", "proj/train.py": "", "proj/lib/m.py": "", "proj/x.log": ""},
    )
    (repo / ".gitignore").write_text("*.log\n")
    _git(repo, "init", "-q")
    assert _names(list_code_files(repo / "proj")) == ["lib/m.py", "train.py"]


@needs_git
def test_git_mode_also_honours_dgpuignore(tmp_path):
    repo = _tree(
        tmp_path / "repo", {"train.py": "", "notebooks/a.ipynb": "", ".dgpuignore": "notebooks/\n"}
    )
    _git(repo, "init", "-q")
    assert _names(list_code_files(repo)) == [".dgpuignore", "train.py"]


@needs_git
def test_git_nested_repository_is_included(tmp_path):
    repo = _tree(
        tmp_path / "repo", {"train.py": "", "vendor/lib/core.py": "", "vendor/lib/.env": "x"}
    )
    _git(repo, "init", "-q")
    _git(repo / "vendor" / "lib", "init", "-q")
    assert _names(list_code_files(repo)) == ["train.py", "vendor/lib/core.py"]


@needs_git
def test_code_dir_entirely_gitignored_is_reported(tmp_path):
    repo = _tree(tmp_path / "repo", {"ignored/train.py": "", ".gitignore": "ignored/\n"})
    _git(repo, "init", "-q")
    with pytest.raises(SpecError, match="没有要打包的文件"):
        pack_code(repo / "ignored", tmp_path / "code.tar.gz")


# ---------------------------------------------------------------- 打包


def test_pack_is_deterministic(tmp_path):
    first = _tree(tmp_path / "a" / "code", {**CODE, "run.sh": "#!/bin/sh\n"})
    os.chmod(first / "run.sh", 0o755)
    result_1 = pack_code(first, tmp_path / "1.tar.gz")

    # 同样的内容，不同的位置、修改时间和创建顺序
    second = tmp_path / "b" / "code"
    for rel in reversed(sorted([*CODE, "run.sh"])):
        _tree(second, {rel: (first / rel).read_bytes()})
    os.chmod(second / "run.sh", 0o700)
    for path in second.rglob("*"):
        os.utime(path, (1_700_000_000, 1_700_000_000))
    result_2 = pack_code(second, tmp_path / "2.tar.gz")

    assert result_1.sha256 == result_2.sha256
    assert (tmp_path / "1.tar.gz").read_bytes() == (tmp_path / "2.tar.gz").read_bytes()

    (second / "train.py").write_text("print('changed')\n")
    assert pack_code(second, tmp_path / "3.tar.gz").sha256 != result_1.sha256


def test_pack_is_deterministic_for_unicode_and_long_names(tmp_path):
    files = {"数据/说明.md": "中文\n", "/".join(["deep" * 10] * 5) + "/m.py": "x = 1\n"}
    a = pack_code(_tree(tmp_path / "a", files), tmp_path / "a.tar.gz")
    b = pack_code(_tree(tmp_path / "b", files), tmp_path / "b.tar.gz")
    assert a.sha256 == b.sha256
    assert set(_members(a.path)) == set(files)


def test_pack_metadata_is_normalised_and_content_round_trips(tmp_path):
    root = _tree(tmp_path / "code", {**CODE, "run.sh": "#!/bin/sh\necho hi\n"})
    os.chmod(root / "run.sh", 0o750)
    os.chmod(root / "train.py", 0o600)
    result = pack_code(root, tmp_path / "out" / "code.tar.gz")

    assert result.path == tmp_path / "out" / "code.tar.gz"
    assert result.file_count == len(CODE) + 1
    assert result.total_bytes == sum(len(v) for v in CODE.values()) + len("#!/bin/sh\necho hi\n")
    assert result.archive_bytes == result.path.stat().st_size
    assert result.skipped == ()

    members = _members(result.path)
    assert list(members) == sorted(members)
    for member in members.values():
        assert member.mtime == FIXED_MTIME
        assert (member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "")
    assert members["run.sh"].mode == 0o755
    assert members["train.py"].mode == 0o644

    with tarfile.open(result.path, "r:gz") as tar:
        for rel, content in CODE.items():
            assert tar.extractfile(rel).read() == content.encode()


def test_pack_never_contains_secret_files(tmp_path):
    root = _tree(tmp_path / "code", {**CODE, **SECRETS_AND_JUNK})
    names = set(_members(pack_code(root, tmp_path / "c.tar.gz").path))
    assert names == set(CODE)


def test_pack_over_limit_suggests_storage(tmp_path):
    root = _tree(
        tmp_path / "code", {"train.py": "x", "weights/model.pt": b"\x00" * (2 * 1024 * 1024)}
    )
    dest = tmp_path / "code.tar.gz"
    with pytest.raises(SpecError) as info:
        pack_code(root, dest, max_mb=1)
    message = str(info.value)
    assert "超过上限 1 MB" in message
    assert "data" in message and "存储" in message
    assert "weights/model.pt" in message
    assert not dest.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_symlinks_inside_are_kept_relative_and_outside_are_skipped(tmp_path):
    outside = _tree(tmp_path / "outside", {"secret.txt": "do not ship"})
    root = _tree(tmp_path / "code", {"train.py": "x", "configs/base.yaml": "lr: 1\n"})
    (root / "configs" / "current.yaml").symlink_to("base.yaml")
    (root / "abs_link.yaml").symlink_to(root / "configs" / "base.yaml")  # 绝对路径但在目录内
    (root / "cfgdir").symlink_to(root / "configs", target_is_directory=True)
    (root / "leak.txt").symlink_to(outside / "secret.txt")
    (root / "leakdir").symlink_to(outside, target_is_directory=True)
    (root / "dangling").symlink_to(root / "no-such-file")

    result = pack_code(root, tmp_path / "c.tar.gz")
    skipped = dict(result.skipped)
    assert set(skipped) == {"leak.txt", "leakdir", "dangling"}
    assert "之外" in skipped["leak.txt"] and "之外" in skipped["leakdir"]
    assert "不存在" in skipped["dangling"]

    members = _members(result.path)
    assert members["configs/current.yaml"].issym()
    assert members["configs/current.yaml"].linkname == "base.yaml"
    assert members["abs_link.yaml"].linkname == "configs/base.yaml"
    assert members["cfgdir"].linkname == "configs"
    assert "leak.txt" not in members

    out = tmp_path / "extracted"
    with tarfile.open(result.path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):  # Python ≥ 3.11.4：用最严格的解包过滤
            tar.extractall(out, filter="data")
        else:
            tar.extractall(out)
    assert (out / "abs_link.yaml").read_text() == "lr: 1\n"
    assert (out / "cfgdir" / "base.yaml").read_text() == "lr: 1\n"
    assert b"do not ship" not in io.BytesIO(result.path.read_bytes()).getvalue()


def test_pack_into_code_dir_does_not_include_itself(tmp_path):
    root = _tree(tmp_path / "code", CODE)
    dest = root / "build" / "code.tar.gz"
    first = pack_code(root, dest)
    second = pack_code(root, dest)
    assert second.sha256 == first.sha256
    assert "build/code.tar.gz" not in _members(dest)
    assert [p.name for p in dest.parent.iterdir()] == ["code.tar.gz"]  # 没有残留的临时文件


def test_pack_empty_dir_is_an_error(tmp_path):
    root = _tree(tmp_path / "code", {".env": "x"})
    with pytest.raises(SpecError, match="没有要打包的文件"):
        pack_code(root, tmp_path / "c.tar.gz")


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="需要非 root 的 POSIX 用户")
def test_unreadable_file_fails_cleanly(tmp_path):
    root = _tree(tmp_path / "code", {"a.py": "", "b.py": "secret"})
    os.chmod(root / "b.py", 0)
    try:
        with pytest.raises(SpecError, match="无法读取.*b.py"):
            pack_code(root, tmp_path / "out" / "c.tar.gz")
    finally:
        os.chmod(root / "b.py", 0o644)
    assert list((tmp_path / "out").iterdir()) == []


# ---------------------------------------------------------------- 数据集哈希

DATA = {
    "images/0001.jpg": b"\xff\xd8 one",
    "images/0002.jpg": b"\xff\xd8 two",
    "labels.csv": "a,b\n",
}


def test_dataset_hash_depends_only_on_content(tmp_path):
    a = _tree(tmp_path / "a" / "set", DATA)
    b = _tree(tmp_path / "b" / "other-name", {k: DATA[k] for k in reversed(list(DATA))})
    for path in b.rglob("*"):
        os.utime(path, (1_600_000_000, 1_600_000_000))
    manifest_a, manifest_b = hash_dataset(a), hash_dataset(b)

    assert manifest_a.content_hash == manifest_b.content_hash
    assert [f.path for f in manifest_a.files] == sorted(DATA)
    assert manifest_a.total_bytes == sum(len(v) for v in DATA.values())
    assert manifest_a.root == a

    (b / "labels.csv").write_text("a,c\n")  # 同样大小，不同内容
    assert hash_dataset(b).content_hash != manifest_a.content_hash
    (b / "labels.csv").write_text("a,b\n")
    (b / "images" / "0002.jpg").rename(b / "images" / "0003.jpg")
    assert hash_dataset(b).content_hash != manifest_a.content_hash


def test_dataset_excludes_secrets_and_finder_files_and_reports_them(tmp_path):
    root = _tree(
        tmp_path / "set", {**DATA, ".DS_Store": "x", "images/.DS_Store": "y", ".env": "T=1"}
    )
    _tree(root, {".git/HEAD": "ref"})
    manifest = hash_dataset(root)
    assert [f.path for f in manifest.files] == sorted(DATA)
    assert manifest.excluded == (".DS_Store", ".env", ".git/", "images/.DS_Store")
    assert manifest.content_hash == hash_dataset(_tree(tmp_path / "clean", DATA)).content_hash


def test_dataset_follows_symlinks(tmp_path):
    store = _tree(tmp_path / "store", {"big.bin": b"payload", "shared/x.txt": "x"})
    root = _tree(tmp_path / "set", {"a.txt": "a"})
    (root / "big.bin").symlink_to(store / "big.bin")
    (root / "shared").symlink_to(store / "shared", target_is_directory=True)
    manifest = hash_dataset(root)
    assert {f.path: f.size for f in manifest.files} == {"a.txt": 1, "big.bin": 7, "shared/x.txt": 1}


@pytest.mark.parametrize(
    ("setup", "needle"),
    [
        (lambda root: None, "没有文件"),
        (lambda root: (root / "broken").symlink_to(root / "missing"), "不存在"),
        (lambda root: (root / "loop").symlink_to(root, target_is_directory=True), "循环"),
        (lambda root: os.mkfifo(root / "pipe"), "不是普通文件"),
    ],
    ids=["empty", "dangling", "cycle", "fifo"],
)
def test_bad_dataset_dirs_are_rejected(tmp_path, setup, needle):
    root = tmp_path / "set"
    root.mkdir()
    if needle != "没有文件":
        (root / "ok.txt").write_text("ok")
    setup(root)
    with pytest.raises(SpecError, match=needle):
        hash_dataset(root)


def test_symlink_cycle_through_sibling_dirs_is_detected(tmp_path):
    root = _tree(tmp_path / "set", {"a/x.txt": "x", "b/y.txt": "y"})
    (root / "a" / "to_b").symlink_to(root / "b", target_is_directory=True)
    (root / "b" / "to_a").symlink_to(root / "a", target_is_directory=True)
    with pytest.raises(SpecError, match="循环"):
        hash_dataset(root)


def test_two_links_to_same_dir_are_not_a_cycle(tmp_path):
    shared = _tree(tmp_path / "shared", {"x.txt": "x"})
    root = _tree(tmp_path / "set", {"a.txt": "a"})
    (root / "one").symlink_to(shared, target_is_directory=True)
    (root / "two").symlink_to(shared, target_is_directory=True)
    assert [f.path for f in hash_dataset(root).files] == ["a.txt", "one/x.txt", "two/x.txt"]


# ---------------------------------------------------------------- 数据集上传


class RecordingStore(MemoryStore):
    """记录写入顺序；可以在第 fail_after 次 put_file 之后失败，模拟中途断网。"""

    def __init__(self, fail_after: int | None = None):
        super().__init__()
        self.writes: list[str] = []
        self.fail_after = fail_after

    def put_file(self, key, path):
        if self.fail_after is not None and len(self.writes) >= self.fail_after:
            raise StoreError("connection reset")
        super().put_file(key, path)
        self.writes.append(key)

    def put_json(self, key, obj, **kwargs):
        etag = super().put_json(key, obj, **kwargs)
        self.writes.append(key)
        return etag


def _local_ref(root: Path, name: str = "set") -> DataRef:
    return DataRef(name=name, local_path=root)


def test_upload_writes_files_then_marker_last(tmp_path):
    root = _tree(tmp_path / "set", DATA)
    store = RecordingStore()
    calls: list[tuple[str, int, int]] = []

    source = upload_dataset(store, _local_ref(root, "coco"), progress=lambda *a: calls.append(a))

    manifest = hash_dataset(root)
    prefix = layout.dataset_prefix(manifest.content_hash)
    marker = layout.dataset_complete_key(manifest.content_hash)
    assert source == DataSource(name="coco", bucket=store.profile.bucket, prefix=prefix)
    assert store.writes == [prefix + rel for rel in sorted(DATA)] + [marker]
    for rel, content in DATA.items():
        data = content.encode() if isinstance(content, str) else content
        assert store.objects[prefix + rel] == data

    doc, _ = store.get_json(marker)
    assert doc["content_hash"] == manifest.content_hash
    assert doc["file_count"] == len(DATA)
    assert [entry[0] for entry in doc["files"]] == sorted(DATA)

    assert [c[0] for c in calls] == sorted(DATA)
    assert [c[1] for c in calls] == sorted(c[1] for c in calls)
    assert calls[-1][1:] == (manifest.total_bytes, manifest.total_bytes)


def test_second_upload_is_skipped(tmp_path):
    root = _tree(tmp_path / "set", DATA)
    store = RecordingStore()
    first = upload_dataset(store, _local_ref(root))
    store.writes.clear()

    calls = []
    second = upload_dataset(store, _local_ref(root), progress=lambda *a: calls.append(a))
    assert second == first
    assert store.writes == [] and calls == []

    # 别处同样内容的数据也直接复用
    copy = _tree(tmp_path / "elsewhere" / "set", DATA)
    assert upload_dataset(store, _local_ref(copy)) == first
    assert store.writes == []


def test_interrupted_upload_has_no_marker_and_is_redone(tmp_path):
    root = _tree(tmp_path / "set", DATA)
    manifest = hash_dataset(root)
    marker = layout.dataset_complete_key(manifest.content_hash)

    store = RecordingStore(fail_after=1)
    with pytest.raises(StoreError):
        upload_dataset(store, _local_ref(root))
    assert not store.exists(marker)
    assert len(store.list(layout.dataset_prefix(manifest.content_hash))) == 1

    store.fail_after = None
    store.writes.clear()
    upload_dataset(store, _local_ref(root))
    prefix = layout.dataset_prefix(manifest.content_hash)
    assert store.writes == [prefix + rel for rel in sorted(DATA)] + [marker]


def test_data_changed_during_upload_gets_no_marker(tmp_path):
    root = _tree(tmp_path / "set", DATA)
    store = MemoryStore()

    def modify_after_first(path, done, total):
        if path == "images/0001.jpg":
            (root / "images" / "0001.jpg").write_bytes(b"changed while uploading")

    with pytest.raises(SpecError, match="上传过程中被修改"):
        upload_dataset(store, _local_ref(root), progress=modify_after_first)
    assert not any(key.endswith(".complete") for key in store.objects)


def test_concurrent_upload_that_finished_first_is_fine(tmp_path):
    root = _tree(tmp_path / "set", DATA)
    store = MemoryStore()
    marker = layout.dataset_complete_key(hash_dataset(root).content_hash)

    def other_process_finishes(path, done, total):
        if done == total:
            store.put_json(marker, {"by": "someone else"})

    upload_dataset(store, _local_ref(root), progress=other_process_finishes)
    assert store.get_json(marker)[0] == {"by": "someone else"}


def test_precomputed_manifest_is_reused(tmp_path, monkeypatch):
    root = _tree(tmp_path / "set", DATA)
    manifest = hash_dataset(root)
    import dgpu.packaging as packaging

    monkeypatch.setattr(packaging, "hash_dataset", lambda path: pytest.fail("hashed twice"))
    source = upload_dataset(MemoryStore(), _local_ref(root), manifest=manifest)
    assert source.prefix == layout.dataset_prefix(manifest.content_hash)

    other = _tree(tmp_path / "other", DATA)
    with pytest.raises(ValueError):
        upload_dataset(MemoryStore(), _local_ref(other), manifest=manifest)


def test_upload_rejects_remote_ref(tmp_path):
    with pytest.raises(ValueError):
        upload_dataset(MemoryStore(), DataRef(name="coco8", bucket="bkt", prefix="coco8/"))


# ---------------------------------------------------------------- 远程数据


def test_resolve_remote():
    assert resolve_remote(DataRef(name="coco8", bucket="bkt", prefix="sets/coco8/")) == DataSource(
        name="coco8", bucket="bkt", prefix="sets/coco8/"
    )
    assert resolve_remote(DataRef(name="c", bucket="bkt", prefix="c")).prefix == "c/"
    with pytest.raises(ValueError):
        resolve_remote(DataRef(name="local", local_path=Path("/tmp/x")))
