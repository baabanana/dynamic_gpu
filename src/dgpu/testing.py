"""所有工作包共用的测试替身：MemoryStore、FakeBackend、FakeBundler。

它们实现 contracts.py 里的协议，行为（尤其是条件写、幂等销毁、标签过滤）
与真实实现的约定一致。不要在各自的测试里另写一套。
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from dgpu import layout
from dgpu.contracts import (
    INST_GONE,
    INST_LOADING,
    Balance,
    BundleRef,
    GpuRequest,
    InstanceRef,
    InstanceState,
    LaunchSpec,
    Offer,
    OfferUnavailable,
    PreconditionFailed,
    Store,
    StoreProfile,
)

TEST_PROFILE = StoreProfile(
    name="r2",
    endpoint="https://test.invalid",
    bucket="test-bucket",
    access_key_id="test-key",
    secret_access_key="test-secret",
)


class MemoryStore:
    """内存里的 Store。ETag 是内容的 md5。"""

    def __init__(self, profile: StoreProfile = TEST_PROFILE):
        self.profile = profile
        self.objects: dict[str, bytes] = {}

    @staticmethod
    def _etag(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def get_json(self, key: str) -> tuple[dict[str, Any], str] | None:
        data = self.objects.get(key)
        if data is None:
            return None
        return json.loads(data), self._etag(data)

    def put_json(
        self,
        key: str,
        obj: dict[str, Any],
        *,
        if_match: str | None = None,
        if_none_match: bool = False,
    ) -> str:
        current = self.objects.get(key)
        if if_none_match and current is not None:
            raise PreconditionFailed(f"{key} already exists")
        if if_match is not None and (current is None or self._etag(current) != if_match):
            raise PreconditionFailed(f"{key} was modified")
        data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
        self.objects[key] = data
        return self._etag(data)

    def get_bytes(self, key: str, *, start: int = 0) -> bytes | None:
        data = self.objects.get(key)
        return None if data is None else data[start:]

    def put_bytes(self, key: str, data: bytes) -> None:
        self.objects[key] = bytes(data)

    def put_file(self, key: str, path: Path) -> None:
        self.objects[key] = Path(path).read_bytes()

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list(self, prefix: str) -> list[str]:
        return sorted(key for key in self.objects if key.startswith(prefix))

    def list_dirs(self, prefix: str) -> list[str]:
        names = set()
        for key in self.objects:
            if key.startswith(prefix):
                head, sep, _ = key[len(prefix) :].partition("/")
                if sep:
                    names.add(head)
        return sorted(names)

    def download_prefix(self, prefix: str, dest: Path) -> int:
        count = 0
        for key in self.list(prefix):
            target = Path(dest) / key[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.objects[key])
            count += 1
        return count

    def presign_get(self, key: str, ttl_s: int) -> str:
        return f"https://presigned.invalid/{self.profile.bucket}/{key}?ttl={ttl_s}"

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class FakeBackend:
    """内存里的 Backend，可以编排失败。

    - offers：search 返回的列表。
    - unavailable_offer_ids：create 时抛 OfferUnavailable 的 offer。
    - foreign：不带 dgpu 标签的实例，list 绝不能返回、gc 绝不能销毁。
    - set_phase：模拟平台侧状态变化。
    - calls：按顺序记录的调用，断言用。
    """

    def __init__(self, name: str = "vast", offers: list[Offer] | None = None):
        self.name = name
        self.offers: list[Offer] = list(offers or [])
        self.unavailable_offer_ids: set[str] = set()
        self.instances: dict[str, InstanceState] = {}
        self.launches: dict[str, LaunchSpec] = {}
        self.foreign: dict[str, InstanceState] = {}
        self.destroyed: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.balance_value: Balance | None = Balance(100.0, "USD" if name == "vast" else "CNY")
        self._ids = itertools.count(1)

    def search(self, req: GpuRequest) -> list[Offer]:
        self.calls.append(("search", ""))
        matches = [
            offer
            for offer in self.offers
            if offer.gpu_name in req.names
            and offer.gpu_count >= req.count
            and offer.price_per_hour <= req.max_price_per_hour
        ]
        return sorted(matches, key=lambda offer: offer.est_total)

    def create(self, offer: Offer, launch: LaunchSpec) -> InstanceRef:
        self.calls.append(("create", offer.id))
        if offer.id in self.unavailable_offer_ids:
            raise OfferUnavailable(f"offer {offer.id} is no longer available")
        ref = InstanceRef(backend=self.name, id=str(next(self._ids)), label=launch.label)
        self.instances[ref.id] = InstanceState(
            ref=ref, phase=INST_LOADING, raw_status="loading", price_per_hour=offer.price_per_hour
        )
        self.launches[ref.id] = launch
        return ref

    def get(self, ref: InstanceRef) -> InstanceState:
        self.calls.append(("get", ref.id))
        return self.instances.get(ref.id) or InstanceState(ref=ref, phase=INST_GONE)

    def destroy(self, ref: InstanceRef) -> None:
        self.calls.append(("destroy", ref.id))
        if ref.id in self.foreign:
            raise AssertionError(f"tried to destroy a non-dgpu instance: {ref.id}")
        if self.instances.pop(ref.id, None) is not None:
            self.destroyed.append(ref.id)

    def list(self, label_prefix: str = layout.LABEL_PREFIX) -> list[InstanceState]:
        self.calls.append(("list", label_prefix))
        return [
            state for state in self.instances.values() if state.ref.label.startswith(label_prefix)
        ]

    def balance(self) -> Balance | None:
        return self.balance_value

    # ---- 测试辅助

    def set_phase(self, instance_id: str, phase: str, raw_status: str = "") -> None:
        state = self.instances[instance_id]
        self.instances[instance_id] = InstanceState(
            ref=state.ref,
            phase=phase,
            raw_status=raw_status or phase,
            price_per_hour=state.price_per_hour,
            created_at=state.created_at,
        )

    def add_foreign(self, instance_id: str, label: str = "someone-elses-work") -> None:
        ref = InstanceRef(backend=self.name, id=instance_id, label=label)
        self.foreign[instance_id] = InstanceState(ref=ref, phase="running")


def make_offer(
    offer_id: str = "o1",
    *,
    backend: str = "vast",
    gpu_name: str = "RTX 3090",
    price: float = 0.2,
    gpu_count: int = 1,
) -> Offer:
    return Offer(
        backend=backend,
        id=offer_id,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        price_per_hour=price,
        currency="USD" if backend == "vast" else "CNY",
        est_total=price,
    )


class FakeBundler:
    """不构建真实 bundle，只在存储里放一个占位对象。"""

    SHA256 = "0" * 64

    def __init__(self) -> None:
        self.commands: list[dict[str, str]] = []

    def ensure_uploaded(self, store: Store) -> BundleRef:
        key = layout.tools_bundle_key(self.SHA256)
        if not store.exists(key):
            store.put_bytes(key, b"fake-bundle")
        return BundleRef(key=key, sha256=self.SHA256)

    def bootstrap_command(self, *, bundle_url: str, bundle_sha256: str, launch_url: str) -> str:
        self.commands.append(
            {"bundle_url": bundle_url, "bundle_sha256": bundle_sha256, "launch_url": launch_url}
        )
        return f"echo bootstrap {bundle_sha256[:8]}"
