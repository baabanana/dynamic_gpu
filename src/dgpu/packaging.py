"""代码打包和本地数据集上传（设计 4.5）。

代码文件列表
    代码目录在 git 工作区里时用 `git ls-files -co --exclude-standard`（已跟踪 + 未跟踪且
    未被 .gitignore 忽略的文件，只列代码目录之下的部分，嵌套的仓库和子模块递归列出）；
    否则遍历目录。两种情况都再按代码目录根下的 .dgpuignore 过滤，并且始终排除
    ALWAYS_EXCLUDED 里的名字（任意层级的文件或目录）。

.dgpuignore 支持 gitignore 的一个子集（fnmatch 匹配，区分大小写）：
    - 空行和以 # 开头的行忽略；
    - 结尾的 / 表示只匹配目录；目录被排除时其下所有文件一起排除；
    - 不含 /（结尾的 / 除外）的模式匹配任意层级的文件名或目录名，如 *.ckpt、runs/；
    - 含 / 的模式从代码目录根开始匹配整个相对路径，开头的 / 可写可不写，如 /build、docs/*.pdf；
    - 通配符 *、?、[abc]；* 也会匹配 /，没有专门的 **；
    - 不支持 ! 取反和反斜杠转义：写了 ! 开头的行会报错，而不是静默忽略。

代码包
    确定性的 tar.gz：条目按路径排序，mtime、uid、gid、属主名固定，权限归一为 644/755，
    gzip 头不含文件名和时间，所以同一份文件内容得到同一个 sha256。代码目录内的符号链接
    保留为相对链接；指向目录外或不存在的链接跳过，并在 PackResult.skipped 里列出。

数据集
    scan_dataset 只遍历一次目录，得到文件清单和大小（submit 用它估算、确认、算磁盘）；
    这份清单原样交给 upload_dataset，按它算哈希和上传，不再遍历，估算的和上传的一致。
    本地数据目录按内容寻址：哈希由排序后的相对路径、大小和每个文件的 sha256 算出。
    上传到 layout.dataset_prefix(哈希)，所有文件传完后最后写 layout.dataset_complete_key(哈希)。
    已有完成标记就整体跳过；中断的上传没有标记，下次全部重传。数据目录里的符号链接
    会被跟随（上传目标文件的内容）。
"""

from __future__ import annotations

import contextlib
import fnmatch
import gzip
import hashlib
import json
import os
import posixpath
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from dgpu import layout, states
from dgpu.contracts import (
    SINGLE_UPLOAD_LIMIT,
    DataRef,
    DataSource,
    PreconditionFailed,
    SpecError,
    Store,
)

# 任意层级的文件名或目录名匹配到就排除，代码包和数据集都适用
ALWAYS_EXCLUDED = (".git", ".env*", "*.pem", "*.key", "__pycache__", ".venv")
# 数据集额外排除：macOS 的 Finder 会随手改写 .DS_Store，会让内容哈希变化而整体重传
DATASET_EXCLUDED = ALWAYS_EXCLUDED + (".DS_Store",)

IGNORE_FILE = ".dgpuignore"
DEFAULT_MAX_MB = 200
FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z，代码包里所有条目的修改时间
DATASET_HASH_SALT = b"dgpu-dataset-v1\n"

_GIT_ENV_DROP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX")
_CHUNK = 1024 * 1024

ProgressFn = Callable[[str, int, int], None]
"""上传进度回调：(刚上传完的相对路径, 已上传字节数, 总字节数)，每个文件传完调用一次。"""


# ---------------------------------------------------------------- 结果类型


@dataclass(frozen=True)
class PackResult:
    path: Path  # 写出的 tar.gz
    file_count: int  # 包里的条目数（普通文件 + 符号链接）
    total_bytes: int  # 普通文件未压缩的总大小，max_mb 上限按它计算
    archive_bytes: int  # tar.gz 的大小
    sha256: str  # tar.gz 的 sha256
    skipped: tuple[tuple[str, str], ...] = ()  # (相对路径, 原因)：没有打进包的条目
    # skipped 里坏掉（指向不存在的文件）或指向代码目录之外的符号链接的相对路径：
    # 训练代码用到它们会在实例上失败，submit 默认拒绝（--yes 放行）
    skipped_links: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetFile:
    path: str  # 相对数据目录的路径，/ 分隔
    size: int
    sha256: str  # scan_dataset 的清单里为空串（还没读内容）
    mtime_ns: int  # 只用来检测上传过程中的修改，不参与内容哈希


@dataclass(frozen=True)
class DatasetManifest:
    root: Path  # 数据目录（绝对路径）
    content_hash: str  # scan_dataset 的清单里为空串（还没算哈希）
    total_bytes: int
    files: tuple[DatasetFile, ...]  # 按 path 排序
    excluded: tuple[str, ...] = ()  # 被 DATASET_EXCLUDED 排除的相对路径（目录以 / 结尾）

    @property
    def hashed(self) -> bool:
        return bool(self.content_hash)


# ---------------------------------------------------------------- 代码文件列表


@dataclass(frozen=True)
class _Rule:
    pattern: str
    dir_only: bool
    anchored: bool


def list_code_files(code_dir: Path, *, exclude: Iterable[Path] = ()) -> list[Path]:
    """返回要打包的文件，路径相对 code_dir，按路径排序。

    exclude：额外排除的绝对路径（文件或目录），如位于代码目录里的本地数据目录。
    """
    root = _check_dir(code_dir, "代码目录")
    rules = _load_ignore_rules(root)
    extra = _relative_excludes(root, exclude)
    candidates = _git_files(root, set())
    if candidates is None:
        candidates = _walk_code(root, rules)
    keep = {
        rel
        for rel in candidates
        if rel and not _is_excluded(rel.split("/"), False, rules) and not _under_any(rel, extra)
    }
    return [Path(rel) for rel in sorted(keep)]


def _check_dir(path: Path, what: str) -> Path:
    root = Path(os.path.abspath(Path(path).expanduser()))
    if not root.is_dir():
        raise SpecError(f"{what} {root} 不存在或不是目录。检查 job.yaml 里的路径。")
    return root


def _load_ignore_rules(root: Path) -> list[_Rule]:
    path = root / IGNORE_FILE
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise SpecError(f"{path} 不是 UTF-8 编码的文本。用 UTF-8 重新保存。") from None
    rules: list[_Rule] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            raise SpecError(
                f"{path} 第 {lineno} 行：.dgpuignore 不支持 ! 取反。"
                "改成只写要排除的模式，或者把需要的文件移出被排除的目录。"
            )
        dir_only = line.endswith("/")
        pattern = line.rstrip("/")
        anchored = "/" in pattern
        pattern = pattern.lstrip("/")
        if not pattern:
            continue
        rules.append(_Rule(pattern=pattern, dir_only=dir_only, anchored=anchored))
    return rules


def _is_excluded(parts: list[str], is_dir: bool, rules: list[_Rule]) -> bool:
    """parts 是相对路径的各段；任何一层目录被排除，其下的文件也被排除。"""
    for index, name in enumerate(parts):
        if _matches_any(name, ALWAYS_EXCLUDED):
            return True
        part_is_dir = is_dir or index < len(parts) - 1
        sub = "/".join(parts[: index + 1])
        for rule in rules:
            if rule.dir_only and not part_is_dir:
                continue
            if fnmatch.fnmatchcase(sub if rule.anchored else name, rule.pattern):
                return True
    return False


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _relative_excludes(root: Path, paths: Iterable[Path]) -> list[str]:
    result = []
    for path in paths:
        rel = os.path.relpath(os.path.abspath(path), root)
        if rel == ".":
            raise SpecError(f"不能把整个代码目录 {root} 排除在代码包之外。")
        if Path(rel).parts[0] != "..":  # 代码目录之外的路径本来就不在包里
            result.append(Path(rel).as_posix())
    return result


def _under_any(rel: str, prefixes: list[str]) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROP}


def _git_files(root: Path, visited: set[str]) -> list[str] | None:
    """git 工作区里的文件列表（相对 root）；root 不在 git 工作区里返回 None。"""
    real = os.path.realpath(root)
    if real in visited:
        return []
    visited.add(real)
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            capture_output=True,
            text=True,
            env=_git_env(),
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return None
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            env=_git_env(),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise SpecError(f"在 {root} 运行 git ls-files 超时。检查仓库状态后重试。") from None
    if listing.returncode != 0:
        detail = listing.stderr.decode(errors="replace").strip()
        raise SpecError(
            f"在 {root} 运行 git ls-files 失败：{detail}。修复 git 仓库，"
            "或者把代码放到 git 仓库之外（改用 .dgpuignore）。"
        )
    files: list[str] = []
    for raw in listing.stdout.split(b"\0"):
        rel = os.fsdecode(raw).rstrip("/")
        if not rel or rel == ".":
            continue
        full = root / rel
        if full.is_symlink():
            files.append(rel)
        elif full.is_dir():  # 嵌套的仓库或子模块
            nested = _git_files(full, visited)
            if nested is None:
                nested = _walk_code(full, [])
            files.extend(f"{rel}/{sub}" for sub in nested)
        elif full.exists():
            files.append(rel)
        # 否则：已跟踪但在工作区里删掉了，不打包
    return files


def _walk_code(root: Path, rules: list[_Rule]) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        prefix = [] if rel_dir == "." else list(Path(rel_dir).parts)
        kept = []
        for name in dirnames:
            parts = [*prefix, name]
            if os.path.islink(os.path.join(dirpath, name)):
                files.append("/".join(parts))  # 指向目录的链接按单个条目处理，不进入
            elif not _is_excluded(parts, True, rules):
                kept.append(name)
        dirnames[:] = kept
        files.extend("/".join([*prefix, name]) for name in filenames)
    return files


# ---------------------------------------------------------------- 代码包


def pack_code(
    code_dir: Path,
    dest: Path,
    *,
    max_mb: int = DEFAULT_MAX_MB,
    exclude: Iterable[Path] = (),
) -> PackResult:
    """把代码目录打成确定性的 tar.gz，写到 dest（先写临时文件再改名）。

    未压缩的文件总大小超过 max_mb 时抛 SpecError。exclude 同 list_code_files。
    """
    root = _check_dir(code_dir, "代码目录")
    dest = Path(os.path.abspath(dest))
    real_root = os.path.realpath(root)
    rel_files = list_code_files(root, exclude=[*exclude, dest])

    regular: list[tuple[str, Path]] = []
    links: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []
    bad_links: list[str] = []
    sizes: dict[str, int] = {}
    for rel_path in rel_files:
        rel = rel_path.as_posix()
        full = root / rel_path
        try:
            info = os.lstat(full)
        except FileNotFoundError:
            skipped.append((rel, "列出文件后被删除"))
            continue
        if stat.S_ISLNK(info.st_mode):
            target = os.path.realpath(full)
            if not _within(target, real_root):
                skipped.append((rel, f"符号链接指向代码目录之外：{os.readlink(full)}"))
                bad_links.append(rel)
            elif not os.path.exists(target):
                skipped.append((rel, f"符号链接指向不存在的文件：{os.readlink(full)}"))
                bad_links.append(rel)
            else:
                target_rel = os.path.relpath(target, real_root)
                link_dir = posixpath.dirname(rel) or "."
                links[rel] = Path(os.path.relpath(target_rel, link_dir)).as_posix()
        elif stat.S_ISREG(info.st_mode):
            regular.append((rel, full))
            sizes[rel] = info.st_size
        else:
            skipped.append((rel, "不是普通文件（设备、管道或套接字）"))

    if not regular and not links:
        raise SpecError(
            f"代码目录 {root} 里没有要打包的文件"
            "（可能都被 .gitignore、.dgpuignore 或默认排除规则排除了）。"
            "检查 job.yaml 的 code 路径。"
        )
    total = sum(sizes.values())
    limit = max_mb * 1024 * 1024
    if total > limit:
        biggest = sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:5]
        listing = "；".join(f"{rel}（{size / 1024 / 1024:.1f} MB）" for rel, size in biggest)
        raise SpecError(
            f"代码目录 {root} 要打包的文件共 {total / 1024 / 1024:.1f} MB，超过上限 {max_mb} MB。"
            "代码包只应包含代码：数据集、模型权重等大文件放到存储里，用 job.yaml 的 data 引用"
            "（r2://<bucket>/<目录>，或本地目录，会单独上传），并写进 .gitignore 或 .dgpuignore。"
            f"最大的几个文件：{listing}"
        )

    entries = sorted([(rel, "file") for rel, _ in regular] + [(rel, "link") for rel in links])
    paths = dict(regular)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    written = 0
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for rel, kind in entries:
                        if kind == "link":
                            tar.addfile(_tarinfo(rel, tarfile.SYMTYPE, 0o777, linkname=links[rel]))
                        else:
                            written += _add_file(tar, rel, paths[rel])
        os.replace(tmp_name, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise

    return PackResult(
        path=dest,
        file_count=len(entries),
        total_bytes=written,
        archive_bytes=dest.stat().st_size,
        sha256=_sha256_file(dest),
        skipped=tuple(skipped),
        skipped_links=tuple(bad_links),
    )


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _tarinfo(
    name: str, kind: bytes, mode: int, *, size: int = 0, linkname: str = ""
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode
    info.size = size
    info.linkname = linkname
    info.mtime = FIXED_MTIME
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _add_file(tar: tarfile.TarFile, rel: str, path: Path) -> int:
    changed = (
        f"文件 {path} 在打包过程中被修改。"
        "等写入它的程序结束后重新提交，或者把它排除在代码目录之外。"
    )
    handle = _open_for_read(path, "代码文件")
    with handle:
        info = os.fstat(handle.fileno())
        mode = 0o755 if info.st_mode & 0o111 else 0o644
        try:
            tar.addfile(_tarinfo(rel, tarfile.REGTYPE, mode, size=info.st_size), handle)
        except OSError as exc:
            if "unexpected end of data" in str(exc):  # 读的时候文件变短了
                raise SpecError(changed) from None
            raise  # 写包失败（如磁盘满）原样抛出
        if handle.read(1):  # 读的时候文件变长了
            raise SpecError(changed)
    return info.st_size


def _open_for_read(path: Path, what: str):
    try:
        return open(path, "rb")  # noqa: SIM115 - 调用方负责关闭
    except OSError as exc:
        raise SpecError(
            f"无法读取{what} {path}：{exc.strerror or exc}。检查文件权限，或者把它移出目录。"
        ) from None


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return _sha256_stream(handle)


# ---------------------------------------------------------------- 数据集


def scan_dataset(path: Path) -> DatasetManifest:
    """遍历本地数据目录，得到文件清单和大小，不读文件内容（content_hash 为空）。

    和上传用的是同一次遍历、同样的规则（跟随符号链接，排除 DATASET_EXCLUDED）：submit 用
    它估算大小、确认、算磁盘，再把它交给 upload_dataset，上传时按这份清单算哈希，不再遍历。
    目录为空或含坏链接时抛 SpecError；超过单次上传上限的文件不在这里报错（调用方检查）。
    """
    root = _check_dir(path, "数据目录")
    found, excluded = _walk_dataset(root)
    if not found:
        raise SpecError(
            f"数据目录 {root} 里没有文件（或者都被默认规则排除了）。检查 job.yaml 的 data 路径。"
        )
    files: list[DatasetFile] = []
    for rel in sorted(found):
        info = os.stat(root / rel)
        files.append(DatasetFile(path=rel, size=info.st_size, sha256="", mtime_ns=info.st_mtime_ns))
    return DatasetManifest(
        root=root,
        content_hash="",
        total_bytes=sum(item.size for item in files),
        files=tuple(files),
        excluded=tuple(sorted(excluded)),
    )


def hash_dataset(
    path: Path,
    *,
    max_file_bytes: int = SINGLE_UPLOAD_LIMIT,
    listing: DatasetManifest | None = None,
) -> DatasetManifest:
    """计算本地数据目录的内容哈希和文件清单。

    listing 是事先 scan_dataset 得到的清单：给了就按它读文件，不再遍历目录；文件在两次之间
    变了（大小或修改时间不同）时抛 SpecError，保证估算的和上传的是同一份数据。
    目录为空、含坏链接，或有文件超过 max_file_bytes（存储单次上传的上限）时抛 SpecError；
    大小在读文件算哈希之前检查。
    """
    root = _check_dir(path, "数据目录")
    if listing is None:
        listing = scan_dataset(root)
    elif Path(listing.root) != root:
        raise ValueError(f"清单属于 {listing.root}，不是 {root}")
    _check_file_sizes(root, [(item.path, item.size) for item in listing.files], max_file_bytes)
    files: list[DatasetFile] = []
    for item in listing.files:
        with _open_for_read(root / item.path, "数据文件") as handle:
            info = os.fstat(handle.fileno())
            if info.st_size != item.size or info.st_mtime_ns != item.mtime_ns:
                raise SpecError(
                    f"数据文件 {root / item.path} 在提交过程中被修改（估算大小之后又变了）。"
                    "等数据不再变化后重新提交。没有上传这份数据。"
                )
            sha256 = _sha256_stream(handle)
        files.append(
            DatasetFile(path=item.path, size=item.size, sha256=sha256, mtime_ns=item.mtime_ns)
        )
    content = hashlib.sha256(DATASET_HASH_SALT)
    for item in files:
        content.update(json.dumps([item.path, item.size, item.sha256]).encode() + b"\n")
    return DatasetManifest(
        root=root,
        content_hash=content.hexdigest(),
        total_bytes=sum(item.size for item in files),
        files=tuple(files),
        excluded=listing.excluded,
    )


def _walk_dataset(root: Path) -> tuple[list[str], list[str]]:
    files: list[str] = []
    excluded: list[str] = []
    top = os.fspath(root)
    # 每个目录从根到它经过的真实路径；跟随符号链接时回到其中任何一个就是循环
    chains: dict[str, frozenset[str]] = {top: frozenset({os.path.realpath(top)})}
    for dirpath, dirnames, filenames in os.walk(top, followlinks=True):
        rel_dir = os.path.relpath(dirpath, top)
        prefix = "" if rel_dir == "." else Path(rel_dir).as_posix() + "/"
        chain = chains.pop(dirpath)
        kept = []
        for name in sorted(dirnames):
            full = os.path.join(dirpath, name)
            if _matches_any(name, DATASET_EXCLUDED):
                excluded.append(f"{prefix}{name}/")
                continue
            target = os.path.realpath(full)
            if target in chain:
                raise SpecError(
                    f"数据目录里的符号链接 {full} 指向它自己的上级目录，会无限循环。"
                    "删掉或改掉这个链接。"
                )
            chains[full] = chain | {target}
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            full = os.path.join(dirpath, name)
            if _matches_any(name, DATASET_EXCLUDED):
                excluded.append(prefix + name)
            elif not os.path.exists(full):
                raise SpecError(
                    f"数据目录里的符号链接 {full} 指向不存在的文件。修复或删掉这个链接后重试。"
                )
            elif not os.path.isfile(full):
                raise SpecError(f"数据目录里的 {full} 不是普通文件。把它移出数据目录后重试。")
            else:
                files.append(prefix + name)
    return files, excluded


def upload_dataset(
    store: Store,
    ref: DataRef,
    *,
    progress: ProgressFn | None = None,
    manifest: DatasetManifest | None = None,
) -> DataSource:
    """把本地数据目录上传到存储，返回实例可以拉取的 DataSource。幂等。

    已有完成标记时直接返回，不重传；否则上传全部文件，最后写完成标记。
    manifest 可以传入事先算好的 hash_dataset 结果（避免重复计算哈希），或 scan_dataset 的
    清单（按清单算哈希，不再遍历目录）。
    存储失败时 StoreError 原样抛出，此时没有完成标记，下次会重传。
    """
    if not ref.is_local or ref.local_path is None:
        raise ValueError(f"data {ref.name!r} 不是本地目录，用 resolve_remote")
    root = Path(os.path.abspath(ref.local_path))
    if manifest is not None and manifest.root != root:
        raise ValueError(f"manifest 属于 {manifest.root}，不是 {root}")
    if manifest is None:
        manifest = hash_dataset(root)
    elif not manifest.hashed:
        manifest = hash_dataset(root, listing=manifest)
    _check_file_sizes(
        root, [(item.path, item.size) for item in manifest.files], SINGLE_UPLOAD_LIMIT
    )

    prefix = layout.dataset_prefix(manifest.content_hash)
    marker = layout.dataset_complete_key(manifest.content_hash)
    source = DataSource(name=ref.name, bucket=store.profile.bucket, prefix=prefix)
    if store.exists(marker):
        return source

    done = 0
    for item in manifest.files:
        store.put_file(prefix + item.path, root / item.path)
        done += item.size
        if progress is not None:
            progress(item.path, done, manifest.total_bytes)

    _check_unchanged(manifest)
    try:
        store.put_json(marker, _marker_doc(manifest), if_none_match=True)
    except PreconditionFailed:
        pass  # 另一个进程同时上传了同样的内容并先写了标记，结果相同
    return source


def _check_file_sizes(root: Path, files: list[tuple[str, int]], limit: int) -> None:
    """有文件超过存储单次上传的上限时，在上传任何东西之前报错，指给用户 rclone + r2:// 的做法。"""
    too_big = sorted((item for item in files if item[1] > limit), key=lambda i: i[1], reverse=True)
    if not too_big:
        return
    rel, size = too_big[0]
    more = f"（另有 {len(too_big) - 1} 个文件也超过）" if len(too_big) > 1 else ""
    gib = 1024**3
    raise SpecError(
        f"数据目录 {root} 里的 {rel}（{size / gib:.1f} GiB）{more}超过存储单次上传的上限 "
        f"{limit / gib:g} GiB，dgpu 不会分段上传它，所以没有上传任何文件。请先用 rclone 把整个"
        f"目录传到 bucket（如 rclone copy {root} <remote>:<bucket>/datasets/{root.name}），"
        f"再在 job.yaml 的 data 里写成 r2://<bucket>/datasets/{root.name}。"
    )


def _check_unchanged(manifest: DatasetManifest) -> None:
    for item in manifest.files:
        path = manifest.root / item.path
        try:
            info = os.stat(path)
        except FileNotFoundError:
            info = None
        if info is None or info.st_size != item.size or info.st_mtime_ns != item.mtime_ns:
            raise SpecError(
                f"数据文件 {path} 在上传过程中被修改或删除，存储里的内容和哈希可能不一致，"
                "所以没有写完成标记。等数据不再变化后重新提交（会重新上传）。"
            )


def _marker_doc(manifest: DatasetManifest) -> dict:
    return {
        "version": states.SCHEMA_VERSION,
        "content_hash": manifest.content_hash,
        "file_count": len(manifest.files),
        "total_bytes": manifest.total_bytes,
        "files": [[item.path, item.size, item.sha256] for item in manifest.files],
        "uploaded_at": states.now(),
    }


def resolve_remote(ref: DataRef) -> DataSource:
    """已经在存储里的数据（r2://bucket/prefix）直接转换成 DataSource。"""
    if ref.is_local or not ref.bucket or not ref.prefix:
        raise ValueError(f"data {ref.name!r} 不是存储里的数据，用 upload_dataset")
    prefix = ref.prefix if ref.prefix.endswith("/") else ref.prefix + "/"
    return DataSource(name=ref.name, bucket=ref.bucket, prefix=prefix)


def resolve_data(
    store: Store, ref: DataRef, *, manifest: DatasetManifest | None = None
) -> DataSource:
    """data 的一项 → 实例可以拉取的 DataSource：本地目录先上传到存储，存储里的直接转换。

    manifest：本地目录事先 scan_dataset / hash_dataset 得到的清单，上传时复用，不再遍历。
    """
    if ref.is_local:
        return upload_dataset(store, ref, manifest=manifest)
    return resolve_remote(ref)
