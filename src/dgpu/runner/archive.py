"""安全解压代码包：拒绝绝对路径、含 .. 的路径、指向外面的链接和设备文件。

Python 3.8 早期版本没有 tarfile 的 extraction filter，所以先自己检查每个条目；
有 filter 的版本再叠加 "data" filter。
"""

from __future__ import annotations

import os
import tarfile


class UnsafeArchive(Exception):
    pass


def _parts(name: str):
    return [part for part in name.replace("\\", "/").split("/") if part not in ("", ".")]


def _check_path(name: str, what: str) -> None:
    if name.startswith("/") or name.startswith("\\"):
        raise UnsafeArchive("%s is an absolute path: %r" % (what, name))
    if ".." in _parts(name):
        raise UnsafeArchive("%s contains '..': %r" % (what, name))


def safe_extract(archive_path: str, dest: str) -> int:
    """把 tar（可压缩）解压到 dest，返回条目数。不安全时抛 UnsafeArchive，什么都不写。"""
    os.makedirs(dest, exist_ok=True)
    uid, gid = os.getuid(), os.getgid()
    with tarfile.open(archive_path, "r:*") as tar:
        members = []
        for member in tar.getmembers():
            _check_path(member.name, "entry")
            if not _parts(member.name):
                continue  # 根目录 "." 本身
            if member.issym() or member.islnk():
                # 符号链接相对它所在目录，硬链接相对包的根目录；都不允许 .. 和绝对路径，
                # 也就只能指向包内
                _check_path(member.linkname, "link target of %r" % member.name)
            elif not (member.isfile() or member.isdir()):
                raise UnsafeArchive("unsupported entry type for %r" % member.name)
            if member.isdir():
                member.mode = (member.mode & 0o777) | 0o700
            elif not member.issym():
                member.mode = (member.mode & 0o777) | 0o600
            member.uid, member.gid = uid, gid
            member.uname = member.gname = ""
            members.append(member)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, members=members, filter="data")
        else:
            tar.extractall(dest, members=members)
    return len(members)
