"""runner bundle：构建、上传，以及实例的启动命令（实现 contracts.Bundler）。

bundle 是一个确定性的 tar.gz，只含实例上需要的东西：

    dgpu/__init__.py  dgpu/layout.py  dgpu/states.py  dgpu/runner/*.py   （只用标准库）
    bin/rclone                                                          （固定版本，linux/amd64）

同样的输入在同一台电脑上总是得到同样的字节和 sha256，存储里按 sha256 去重
（layout.tools_bundle_key）。rclone 从本地缓存取，缓存里没有时下载官方发布包，并按
本文件里固定的 SHA-256 校验压缩包和解出的可执行文件，校验不过就拒绝继续。

启动命令（Vast 的 onstart、AutoDL 的 cmd）只依赖镜像里的 bash 和 python3（≥ 3.8），
长度不超过 4000 字符；内容见 BOOTSTRAP_SOURCE。
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import os
import re
import shlex
import tarfile
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path

import dgpu
from dgpu import layout
from dgpu.contracts import BundleRef, DgpuError, Store

# ---------------------------------------------------------------- 固定版本的 rclone

RCLONE_VERSION = "1.75.2"
# 来自 https://downloads.rclone.org/v1.75.2/SHA256SUMS。2026-11-10 下载核对：该文件的 PGP
# 签名由 rclone 发布密钥 FBF737ECE9F8AB18604BD2AC93935E02FF3B54FA 签发且有效，压缩包实测
# 哈希一致；可执行文件的哈希是从这个压缩包里解出后计算的。
RCLONE_ZIP_SHA256 = "349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702"
RCLONE_BINARY_SHA256 = "ed7ab1f17876bb006c3df3949a3044959671afd9112362dacaa5dd1647735bc1"
RCLONE_URL_TEMPLATE = "https://downloads.rclone.org/v{version}/rclone-v{version}-linux-amd64.zip"
# 升级 rclone 时：哈希未核实前把常量设成 UNPINNED，代码会拒绝构建而不是跳过校验
UNPINNED = "UNPINNED"

BUNDLE_FORMAT = 1  # bundle 内容布局变化时加 1（进入缓存键）
MAX_BOOTSTRAP_LEN = 4000  # Vast onstart 上限 4048
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dgpu"
FILE_MODE = 0o644
EXEC_MODE = 0o755
DIR_MODE = 0o755

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FetchFn = Callable[[str], bytes]


class BundleError(DgpuError):
    """bundle 构建或启动命令生成失败。"""


def default_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:  # 固定的 https 地址
        return response.read()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runner_package_files(package_dir: Path) -> list[tuple[str, Path]]:
    """bundle 里的 Python 文件：(包内路径, 本地路径)，按路径排序。"""
    files = [
        ("dgpu/__init__.py", package_dir / "__init__.py"),
        ("dgpu/layout.py", package_dir / "layout.py"),
        ("dgpu/states.py", package_dir / "states.py"),
    ]
    runner_dir = package_dir / "runner"
    for path in sorted(runner_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(("dgpu/runner/" + path.relative_to(runner_dir).as_posix(), path))
    return sorted(files)


def _tarinfo(name: str, kind: bytes, mode: int, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


class RunnerBundler:
    """contracts.Bundler 的实现。"""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        fetch: FetchFn | None = None,
        rclone_version: str = RCLONE_VERSION,
        rclone_zip_sha256: str = RCLONE_ZIP_SHA256,
        rclone_binary_sha256: str = RCLONE_BINARY_SHA256,
        package_dir: Path | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.fetch = fetch or default_fetch
        self.rclone_version = rclone_version
        self.rclone_zip_sha256 = rclone_zip_sha256
        self.rclone_binary_sha256 = rclone_binary_sha256
        self.package_dir = Path(package_dir or Path(dgpu.__file__).parent)

    # ------------------------------------------------------------ rclone

    @property
    def rclone_cache_path(self) -> Path:
        return self.cache_dir / f"rclone-{self.rclone_version}-linux-amd64" / "rclone"

    def rclone_binary(self) -> Path:
        """缓存里的 rclone（linux/amd64）。缺失或损坏时下载官方压缩包并校验。"""
        for name, value in (
            ("RCLONE_ZIP_SHA256", self.rclone_zip_sha256),
            ("RCLONE_BINARY_SHA256", self.rclone_binary_sha256),
        ):
            if not value or value == UNPINNED or not _SHA256_RE.match(value):
                raise BundleError(
                    f"rclone {self.rclone_version} 的 {name} 没有固定，拒绝在不校验的情况下打包。"
                    "请从 downloads.rclone.org 的 SHA256SUMS（核对 PGP 签名）填入 bundle.py。"
                )
        path = self.rclone_cache_path
        if path.is_file():
            if _sha256_file(path) == self.rclone_binary_sha256:
                return path
            path.unlink()  # 损坏或被替换：重新下载
        url = RCLONE_URL_TEMPLATE.format(version=self.rclone_version)
        try:
            data = self.fetch(url)
        except Exception as exc:
            raise BundleError(
                f"下载 rclone 失败（{url}）：{exc}。可以手动下载这个压缩包，把其中的 rclone 放到 "
                f"{path}，然后重试。"
            ) from exc
        actual = _sha256_bytes(data)
        if actual != self.rclone_zip_sha256:
            raise BundleError(
                f"下载的 rclone 压缩包 sha256 不符（期望 {self.rclone_zip_sha256}，"
                f"实际 {actual}），已丢弃。可能是下载被篡改或不完整，请重试或检查网络代理。"
            )
        member = f"rclone-v{self.rclone_version}-linux-amd64/rclone"
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                binary = archive.read(member)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise BundleError(f"rclone 压缩包里没有 {member}：{exc}") from exc
        actual = _sha256_bytes(binary)
        if actual != self.rclone_binary_sha256:
            raise BundleError(
                f"rclone 可执行文件 sha256 不符（期望 {self.rclone_binary_sha256}，实际 {actual}）"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(binary)
        tmp.chmod(EXEC_MODE)
        os.replace(tmp, path)
        return path

    # ------------------------------------------------------------ bundle

    def _inputs(self) -> tuple[list[tuple[str, Path, int]], str]:
        rclone = self.rclone_binary()
        entries = [(name, path, FILE_MODE) for name, path in runner_package_files(self.package_dir)]
        entries.append(("bin/rclone", rclone, EXEC_MODE))
        digest = hashlib.sha256(f"dgpu-bundle-format-{BUNDLE_FORMAT}\n".encode())
        for name, path, mode in entries:
            content = self.rclone_binary_sha256 if name == "bin/rclone" else _sha256_file(path)
            digest.update(f"{name}\0{mode:o}\0{content}\n".encode())
        return entries, digest.hexdigest()

    def build(self) -> tuple[Path, str]:
        """构建（或从缓存取）bundle，返回 (本地路径, sha256)。"""
        entries, inputs_digest = self._inputs()
        bundles = self.cache_dir / "bundles"
        bundles.mkdir(parents=True, exist_ok=True)
        index = bundles / f"inputs-{inputs_digest}.sha256"
        if index.is_file():
            sha = index.read_text().strip()
            cached = bundles / f"bundle-{sha}.tar.gz"
            if _SHA256_RE.match(sha) and cached.is_file() and _sha256_file(cached) == sha:
                return cached, sha
        tmp = bundles / f"build-{os.getpid()}.tar.gz.tmp"
        try:
            self._write_tar(entries, tmp)
            sha = _sha256_file(tmp)
            final = bundles / f"bundle-{sha}.tar.gz"
            os.replace(tmp, final)
        finally:
            tmp.unlink(missing_ok=True)
        index.write_text(sha + "\n")
        return final, sha

    @staticmethod
    def _write_tar(entries: list[tuple[str, Path, int]], dest: Path) -> None:
        dirs = set()
        for name, _path, _mode in entries:
            parts = name.split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:depth]))
        with open(dest, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                    for directory in sorted(dirs):
                        tar.addfile(_tarinfo(directory, tarfile.DIRTYPE, DIR_MODE))
                    for name, path, mode in sorted(entries):
                        size = path.stat().st_size
                        with open(path, "rb") as handle:
                            tar.addfile(_tarinfo(name, tarfile.REGTYPE, mode, size), handle)

    def ensure_uploaded(self, store: Store) -> BundleRef:
        path, sha = self.build()
        key = layout.tools_bundle_key(sha)
        if not store.exists(key):
            store.put_file(key, path)
        return BundleRef(key=key, sha256=sha)

    # ------------------------------------------------------------ 启动命令

    def bootstrap_command(self, *, bundle_url: str, bundle_sha256: str, launch_url: str) -> str:
        if not _SHA256_RE.match(bundle_sha256 or ""):
            raise BundleError("bundle_sha256 必须是 64 位小写十六进制")
        for what, url in (("bundle_url", bundle_url), ("launch_url", launch_url)):
            if not url or not url.startswith(("https://", "http://", "file://")):
                raise BundleError(f"{what} 必须是 http(s) 地址")
            if any(ch in url for ch in "\0\r\n"):
                raise BundleError(f"{what} 含控制字符")
        command = " ".join(
            [
                "exec python3 -c",
                shlex.quote(_BOOTSTRAP_LOADER),
                shlex.quote(bundle_url),
                bundle_sha256,
                shlex.quote(launch_url),
            ]
        )
        if len(command) > MAX_BOOTSTRAP_LEN:
            raise BundleError(
                f"启动命令有 {len(command)} 个字符，超过 {MAX_BOOTSTRAP_LEN}"
                "（Vast onstart 上限 4048）。预签名 URL 太长，请缩短 bucket 或 endpoint。"
            )
        return command


# ---------------------------------------------------------------- bootstrap 脚本
# 在实例上以 `python3 -c` 运行（Python ≥ 3.8，只用标准库），参数：bundle URL、sha256、启动文档 URL。
# 1. 工作目录：$DGPU_WORK_ROOT；否则有 AutoDL 数据盘 /root/autodl-tmp 就用它下面的 dgpu/，
#    再否则 /workspace/dgpu（states.DEFAULT_WORK_ROOT）。
# 2. 非阻塞地拿 <root>/boot.lock：被占用（EAGAIN）说明同一个任务的 runner 还活着，什么都不做
#    （幂等）。文件系统不支持 flock（其他错误）时记一行继续，由 boot 按 pid 文件判断。
# 3. bundle 不在本地就下载（重试 10 次，指数退避），校验 sha256，解压到 bundle-<sha256>/。
# 4. 启动文档不在本地就下载到 launch.json（0600）。容器重启时预签名 URL 可能已过期，
#    所以已有的文件直接复用。
# 5. 带着锁 exec `python3 -m dgpu.runner.boot`：Vast 上它启动 watchdog 和 runner 后返回，
#    AutoDL 上它留在前台直到两者都退出（容器只在 cmd 运行期间存在）。
# 诊断信息写进 stderr（平台日志）和 logs/bootstrap.log（0600，runner 启动后随日志上传）：
# Python 版本、工作目录、每次下载失败的异常类型和说明、放弃时的磁盘余量。日志里的 URL
# 一律替换成 <url>（预签名 URL 带签名）。

BOOTSTRAP_SOURCE = r"""
import fcntl, hashlib, os, re, shutil, sys, tarfile, time, urllib.request
B, H, L = sys.argv[1:4]
R = os.environ.get("DGPU_WORK_ROOT") or (
    "/root/autodl-tmp/dgpu" if os.path.isdir("/root/autodl-tmp") else "/workspace/dgpu")
os.makedirs(R + "/logs", 0o700, exist_ok=True)


def log(msg):
    msg = re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://\S+", "<url>", msg)
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " [bootstrap] " + msg
    print(line, file=sys.stderr, flush=True)
    try:
        fd = os.open(R + "/logs/bootstrap.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


log("python %s, uid %d, work root %s" % (sys.version.split()[0], os.getuid(), R))
lock = os.open(R + "/boot.lock", os.O_RDWR | os.O_CREAT, 0o600)
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    log("a runner is already active in " + R + "; nothing to do")
    sys.exit(0)
except OSError as e:
    log("flock unsupported here (%s); continuing without the lock" % e)


def get(url, path, sha=None):
    wait = float(os.environ.get("DGPU_BOOT_BACKOFF_S", "5"))
    for i in range(10):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha:
                h = hashlib.sha256()
                with open(path + ".part", "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                if h.hexdigest() != sha:
                    raise ValueError("sha256 mismatch")
            os.replace(path + ".part", path)
            return
        except Exception as e:
            log("download of %s failed (%s: %s), try %d/10"
                % (os.path.basename(path), type(e).__name__, str(e)[:300], i + 1))
            if i < 9:
                time.sleep(min(60.0, wait * 2 ** i))
    try:
        free = "%d MB free" % (shutil.disk_usage(R).free >> 20)
    except OSError as e:
        free = "disk usage unknown (%s)" % e
    log("giving up on %s; %s" % (os.path.basename(path), free))
    sys.exit(1)


D = R + "/bundle-" + H
if not os.path.isdir(D):
    get(B, R + "/bundle.tgz", H)
    shutil.rmtree(D + ".tmp", ignore_errors=True)
    with tarfile.open(R + "/bundle.tgz") as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(D + ".tmp", filter="data")
        else:
            t.extractall(D + ".tmp")
    os.rename(D + ".tmp", D)
    os.unlink(R + "/bundle.tgz")
F = R + "/launch.json"
if not os.path.isfile(F):
    old = os.umask(0o077)
    get(L, F)
    os.umask(old)
os.chmod(F, 0o600)
os.set_inheritable(lock, True)
os.chdir(D)
log("bundle ready; starting dgpu.runner.boot")
argv = [sys.executable, "-m", "dgpu.runner.boot", "--work-root", R, "--lock-fd", str(lock)]
os.execv(sys.executable, argv)
"""

_BOOTSTRAP_PAYLOAD = base64.b64encode(zlib.compress(BOOTSTRAP_SOURCE.encode("utf-8"), 9)).decode()
_BOOTSTRAP_LOADER = (
    'import base64,zlib;exec(zlib.decompress(base64.b64decode("' + _BOOTSTRAP_PAYLOAD + '")))'
)
