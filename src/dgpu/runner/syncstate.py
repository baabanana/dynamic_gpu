"""output/、ckpt/ 的同步记录：哪些文件、以什么状态已经传到存储。

每次成功同步一个目录，就把同步开始前拍下的快照（相对路径 → [大小, mtime_ns]）记为“已上传”，
存在 run/synced-<目录>.json（容器重启后沿用）。之后：

- 文件的大小或 mtime 和记录不一样（包括被等大小地改写过），就算没有上传；
- 记录里没有的文件也算没有上传；
- 记录读不出来时，本地的所有文件都算没有上传（不会报“0 个文件没传上去”）；
- 本地目录列不出来时，返回一条“无法确定”的说明，而不是空列表。

快照在复制开始之前拍：复制期间被改写的文件，快照里是旧的 mtime，之后会被算作没有上传
（宁可多报，不能漏报）。runner 和 watchdog 共用这里的函数。
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

from dgpu.runner.util import read_json, write_json_atomic

Snapshot = Dict[str, List[int]]

_LOCK = threading.Lock()


class ListError(Exception):
    """本地目录列不出来。"""


def snapshot(root: str, skip_tmp: bool) -> Snapshot:
    """root 下所有普通文件：相对路径 → [大小, mtime_ns]。目录不存在返回空字典。"""
    if not os.path.isdir(root):
        return {}
    found = {}  # type: Snapshot
    errors = []  # type: List[OSError]
    for dirpath, _dirnames, filenames in os.walk(root, onerror=errors.append):
        for name in filenames:
            if skip_tmp and name.endswith(".tmp"):
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue  # 刚被删掉或改名
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            found[rel] = [int(st.st_size), int(st.st_mtime_ns)]
    if errors:
        raise ListError(str(errors[0]))
    return found


def manifest_path(run_dir: str, name: str) -> str:
    return os.path.join(run_dir, "synced-%s.json" % name)


def load(run_dir: str, name: str) -> Optional[Snapshot]:
    """读同步记录；不存在返回空字典，损坏返回 None（拿不准）。"""
    path = manifest_path(run_dir, name)
    if not os.path.exists(path):
        return {}
    doc = read_json(path)
    if doc is None or not isinstance(doc.get("files"), dict):
        return None
    files = {}  # type: Snapshot
    for rel, value in doc["files"].items():
        if isinstance(value, list) and len(value) == 2:
            try:
                files[str(rel)] = [int(value[0]), int(value[1])]
            except (TypeError, ValueError):
                return None
        else:
            return None
    return files


def save(run_dir: str, name: str, files: Snapshot) -> None:
    with _LOCK:
        write_json_atomic(manifest_path(run_dir, name), {"files": files})


def record_success(run_dir: str, name: str, before: Snapshot) -> None:
    """一次同步成功：把同步开始前的快照记为已上传。

    只合并，不删除：本地删掉的文件存储里还在（rclone copy 不删远端），留着无害。
    """
    with _LOCK:
        current = load(run_dir, name) or {}
        current.update(before)
        write_json_atomic(manifest_path(run_dir, name), {"files": current})


def record_downloaded(run_dir: str, name: str, before: Snapshot, after: Snapshot) -> None:
    """从存储恢复（下载）之后：新出现的和被改写的文件就是存储里的版本，记为已上传。"""
    changed = {rel: meta for rel, meta in after.items() if before.get(rel) != meta}
    if changed:
        record_success(run_dir, name, changed)


def pending(root: str, skip_tmp: bool, run_dir: str, name: str) -> List[str]:
    """root 下没有按当前内容上传过的文件（相对路径，前面加上 name/）。"""
    try:
        local = snapshot(root, skip_tmp)
    except ListError as exc:
        return ["%s/（无法列出本地文件，不确定哪些没有上传：%s）" % (name, exc)]
    if not local:
        return []
    uploaded = load(run_dir, name)
    if uploaded is None:
        uploaded = {}  # 记录损坏：全部算没有上传
    return [
        "%s/%s" % (name, rel) for rel, meta in sorted(local.items()) if uploaded.get(rel) != meta
    ]


def total_bytes(run_dir: str, names: Tuple[str, ...]) -> int:
    total = 0
    for name in names:
        for meta in (load(run_dir, name) or {}).values():
            total += meta[0]
    return total


def limit_list(items: List[str], limit: int) -> List[str]:
    if len(items) > limit:
        extra = len(items) - limit
        return items[:limit] + ["...（另有 %d 个文件）" % extra]
    return items
