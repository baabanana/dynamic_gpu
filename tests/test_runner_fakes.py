"""Fake binaries for the runner tests: rclone and nvidia-smi.

No tests here. The harness writes tiny wrapper scripts that import this module and call
fake_rclone_main / fake_nvidia_smi_main, so this module must stay stdlib-only and cheap to
import (no pytest, no dgpu).

Fake rclone
- The remote "dgpu:<bucket>/<key>" maps to $FAKE_RCLONE_ROOT/<bucket>/<key>.
- Requires the S3 remote to be configured through RCLONE_CONFIG_DGPU_* (fails otherwise) and
  records whether the secret ever appeared on the command line.
- Behaves like rclone on S3 where it matters: `cat` of a missing key prints nothing and exits
  0; listing or copying a missing prefix exits 3 ("directory not found").
- Control files in $FAKE_RCLONE_ROOT/.control/:
    rules.json       [{"op", "match", "mode": fail|hang|delay, "times": n (-1 = forever),
                       "delay": seconds}] - first matching rule with times != 0 applies
    offline          if present, every call fails like a network outage
    frozen           remote prefixes ("<bucket>/<key prefix>"), one per line; writes under
                     them are recorded in violations.jsonl
    calls.jsonl      one record per finished call (op, args, rc, times, payload of JSON rcats)
    gpu_util         utilization reported by the fake nvidia-smi (default 50)
    no_gpu           if present, the fake nvidia-smi fails
"""

from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import shutil
import sys
import time

REMOTE = "dgpu"
ENV_PREFIX = "RCLONE_CONFIG_DGPU_"
VALUE_FLAGS = {
    "--contimeout",
    "--timeout",
    "--low-level-retries",
    "--retries",
    "--exclude",
    "--config",
}
BOOL_FLAGS = {"-q", "-R", "--files-only", "--json", "--update", "--local-no-check-updated"}
WRITE_OPS = {"copy", "copyto", "move", "rcat", "deletefile"}


def _root():
    return os.environ["FAKE_RCLONE_ROOT"]


def _control(name):
    return os.path.join(_root(), ".control", name)


def _append_locked(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line + "\n")
        handle.flush()


def _take_rule(op, joined):
    path = _control("rules.json")
    if not os.path.exists(path):
        return None
    with open(_control("rules.lock"), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with open(path, encoding="utf-8") as handle:
                rules = json.load(handle)
        except (OSError, ValueError):
            return None
        for rule in rules:
            if rule.get("op") not in (None, op):
                continue
            if rule.get("match") and rule["match"] not in joined:
                continue
            times = int(rule.get("times", -1))
            if times == 0:
                continue
            if times > 0:
                rule["times"] = times - 1
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(rules, handle)
                os.replace(tmp, path)
            return dict(rule)
    return None


def _local(spec):
    """Remote spec -> (local path, remote-relative key or None)."""
    if spec.startswith(REMOTE + ":"):
        rel = spec[len(REMOTE) + 1 :].strip("/")
        return os.path.join(_root(), rel), rel
    if ":" in spec.split("/", 1)[0]:
        raise ValueError(f"unknown remote in {spec!r}")
    return spec, None


def _excluded(rel, patterns):
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) for pat in patterns)


def _walk(path):
    files = []
    for dirpath, _dirnames, filenames in os.walk(path):
        if os.path.basename(dirpath) == ".control" or "/.control" in dirpath:
            continue
        for name in filenames:
            full = os.path.join(dirpath, name)
            files.append((os.path.relpath(full, path).replace(os.sep, "/"), full))
    return sorted(files)


class _StoreLock:
    """Lock shared with the integration tests' directory-backed Store: its conditional writes
    check the current object and replace it under this lock, so they are atomic with respect
    to the writes made here, as on S3."""

    def __enter__(self):
        path = _control("store.lock")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.handle = open(path, "a")  # noqa: SIM115 - closed in __exit__
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.handle.close()


def _replace_remote(tmp, dst):
    if dst.startswith(_root() + os.sep):
        with _StoreLock():
            os.replace(tmp, dst)
    else:
        os.replace(tmp, dst)


def _atomic_copy(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = f"{dst}.fake-{os.getpid()}"
    shutil.copy2(src, tmp)
    _replace_remote(tmp, dst)


def _atomic_write(dst, data):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = f"{dst}.fake-{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(data)
    _replace_remote(tmp, dst)


class Failure(Exception):
    def __init__(self, rc, message):
        super().__init__(message)
        self.rc = rc


def _copy_tree(src, dst, excludes, update, move=False):
    src_path, _ = _local(src)
    dst_path, dst_key = _local(dst)
    if os.path.isfile(src_path):
        items = [(os.path.basename(src_path), src_path)]
    elif os.path.isdir(src_path):
        items = _walk(src_path)
    else:
        raise Failure(3, f"directory not found: {src}")
    written = []
    for rel, full in items:
        if _excluded(rel, excludes):
            continue
        target = os.path.join(dst_path, rel)
        if os.path.exists(target):
            src_stat, dst_stat = os.stat(full), os.stat(target)
            if update and dst_stat.st_mtime >= src_stat.st_mtime:
                if move:
                    os.unlink(full)
                continue
            if dst_stat.st_size == src_stat.st_size and int(dst_stat.st_mtime) == int(
                src_stat.st_mtime
            ):
                if move:
                    os.unlink(full)
                continue
        _atomic_copy(full, target)
        written.append((dst_key + "/" + rel) if dst_key is not None else None)
        if move:
            os.unlink(full)
    return [key for key in written if key]


def fake_rclone_main(argv):
    started = time.time()
    record = {
        "op": None,
        "args": [],
        "rc": None,
        "ts_start": started,
        "pid": os.getpid(),
        "argv0": os.path.abspath(sys.argv[0]),
    }
    secret = os.environ.get(ENV_PREFIX + "SECRET_ACCESS_KEY", "")
    record["has_creds"] = (
        os.environ.get(ENV_PREFIX + "TYPE") == "s3"
        and bool(os.environ.get(ENV_PREFIX + "ACCESS_KEY_ID"))
        and bool(secret)
        and bool(os.environ.get(ENV_PREFIX + "ENDPOINT"))
    )
    record["secret_in_argv"] = bool(secret) and any(secret in arg for arg in argv)
    positional, excludes, flags = [], [], set()
    index = 0
    rc = 0
    try:
        while index < len(argv):
            arg = argv[index]
            if arg in VALUE_FLAGS:
                if index + 1 >= len(argv):
                    raise Failure(1, f"flag needs an argument: {arg}")
                if arg == "--exclude":
                    excludes.append(argv[index + 1])
                index += 2
                continue
            if arg in BOOL_FLAGS:
                flags.add(arg)
            elif arg.startswith("-"):
                raise Failure(1, f"unknown flag: {arg}")
            else:
                positional.append(arg)
            index += 1
        if not positional:
            raise Failure(1, "no command")
        op, args = positional[0], positional[1:]
        record["op"], record["args"] = op, args
        if not record["has_creds"]:
            raise Failure(1, "didn't find section in config file (remote dgpu not configured)")
        rule = _take_rule(op, " ".join(args))
        if rule is not None:
            record["rule"] = rule.get("mode")
            if rule.get("mode") == "hang":
                time.sleep(3600)
            elif rule.get("mode") == "delay":
                time.sleep(float(rule.get("delay", 1.0)))
            else:
                raise Failure(int(rule.get("rc", 1)), f"injected failure for {op}")
        if os.path.exists(_control("offline")):
            raise Failure(1, f"Failed to {op}: dial tcp: i/o timeout")
        written = _dispatch(op, args, excludes, flags, record)
        if written:
            record["written"] = written
            _check_frozen(op, written)
    except Failure as exc:
        sys.stderr.write(f"ERROR : {exc}\n")
        rc = exc.rc
    except Exception as exc:  # report like rclone would, but keep the cause visible
        sys.stderr.write(f"ERROR : {type(exc).__name__}: {exc}\n")
        rc = 2
    record["rc"] = rc
    record["ts_end"] = time.time()
    try:
        _append_locked(_control("calls.jsonl"), json.dumps(record))
    except OSError:
        pass
    return rc


def _check_frozen(op, written):
    path = _control("frozen")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        prefixes = [line.strip() for line in handle if line.strip()]
    for key in written:
        if any(key.startswith(prefix) for prefix in prefixes):
            _append_locked(
                _control("violations.jsonl"), json.dumps({"op": op, "key": key, "ts": time.time()})
            )


def _dispatch(op, args, excludes, flags, record):
    if op in ("copy", "move"):
        src, dst = args
        return _copy_tree(src, dst, excludes, "--update" in flags, move=(op == "move"))
    if op == "copyto":
        src, dst = args
        src_path, _ = _local(src)
        dst_path, dst_key = _local(dst)
        if not os.path.isfile(src_path):
            raise Failure(3, f"directory not found: {src}")
        _atomic_copy(src_path, dst_path)
        return [dst_key] if dst_key else []
    if op == "cat":
        path, _ = _local(args[0])
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                sys.stdout.buffer.write(handle.read())
            sys.stdout.flush()
        return []
    if op == "rcat":
        path, key = _local(args[0])
        data = sys.stdin.buffer.read()
        if key and key.endswith(".json"):
            record["payload"] = data.decode("utf-8", "replace")
        _atomic_write(path, data)
        return [key] if key else []
    if op == "lsjson":
        path, _ = _local(args[0])
        if not os.path.isdir(path):
            raise Failure(3, f"directory not found: {args[0]}")
        items = []
        for rel, full in _walk(path):
            stat = os.stat(full)
            items.append(
                {
                    "Path": rel,
                    "Name": rel.rsplit("/", 1)[-1],
                    "Size": stat.st_size,
                    "ModTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                    "IsDir": False,
                }
            )
        sys.stdout.write(json.dumps(items))
        return []
    if op == "size":
        path, _ = _local(args[0])
        if not os.path.isdir(path):
            raise Failure(3, f"directory not found: {args[0]}")
        files = _walk(path)
        total = sum(os.path.getsize(full) for _rel, full in files)
        sys.stdout.write(json.dumps({"count": len(files), "bytes": total, "sizeless": 0}))
        return []
    if op == "deletefile":
        path, key = _local(args[0])
        if not os.path.isfile(path):
            raise Failure(4, f"file not found: {args[0]}")
        os.unlink(path)
        return [key] if key else []
    raise Failure(1, f"unknown command {op!r}")


def fake_nvidia_smi_main(argv):
    root = os.environ.get("FAKE_RCLONE_ROOT", "")
    control = os.path.join(root, ".control")
    if os.path.exists(os.path.join(control, "no_gpu")):
        sys.stdout.write(
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"
        )
        return 9
    util = "50"
    try:
        with open(os.path.join(control, "gpu_util"), encoding="utf-8") as handle:
            util = handle.read().strip() or util
    except OSError:
        pass
    sys.stdout.write(f"0, NVIDIA GeForce RTX 3090, {util}, 24576\n")
    return 0
