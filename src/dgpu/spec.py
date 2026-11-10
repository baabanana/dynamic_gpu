"""解析和校验 job.yaml（设计 4.2），得到 contracts.JobSpec。

所有错误都抛 SpecError，消息用中文写明：哪个字段、出了什么问题、怎么改。
只做纯校验，不访问网络和存储；路径相对 job.yaml 所在目录解析。
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import yaml

from dgpu import states
from dgpu.contracts import (
    BACKEND_AUTODL,
    BACKEND_VAST,
    BACKENDS,
    CURRENCY_BY_BACKEND,
    DataRef,
    GpuSpec,
    JobSpec,
    Limits,
    SpecError,
)

SPEC_VERSION = 1

TOP_LEVEL_KEYS = (
    "version",
    "name",
    "backend",
    "image",
    "code",
    "setup",
    "run",
    "data",
    "gpu",
    "limits",
    "env",
)
GPU_KEYS = ("names", "count")
LIMIT_KEYS = ("max_hours", "max_usd_per_hour", "max_cny_per_hour")
PRICE_KEY_BY_BACKEND = {BACKEND_VAST: "max_usd_per_hour", BACKEND_AUTODL: "max_cny_per_hour"}

# runner 自己设置的变量和会被 runner 从训练环境里清掉的变量，不允许在 env 里写
RESERVED_ENV_PREFIXES = ("DGPU_",) + tuple(states.SCRUBBED_ENV_PREFIXES)

DATA_SCHEME = "r2"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_URL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://(.*)$")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------- 入口


def load_spec(path: Path, *, backend_override: str | None = None) -> JobSpec:
    """读取并校验 job.yaml。

    backend_override 来自命令行 --backend，优先于文件里的 backend。
    code 和本地 data 路径相对 job.yaml 所在目录解析为绝对路径，且必须是已存在的目录。
    """
    path = Path(os.path.abspath(Path(path).expanduser()))
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SpecError(f"找不到任务文件 {path}。检查路径是否写对。") from None
    except IsADirectoryError:
        raise SpecError(f"{path} 是目录，不是任务文件。传入 job.yaml 文件的路径。") from None
    except UnicodeDecodeError:
        raise SpecError(f"任务文件 {path} 不是 UTF-8 编码的文本。用 UTF-8 重新保存。") from None
    except OSError as exc:
        raise SpecError(f"无法读取任务文件 {path}：{exc.strerror}。检查文件权限。") from None

    try:
        doc = yaml.load(text, Loader=_StrictLoader)  # SafeLoader 的子类，不执行任意构造
    except yaml.YAMLError as exc:
        raise SpecError(
            f"任务文件 {path} 不是合法的 YAML：{exc}\n按提示的行列改正；"
            "含冒号、# 或以特殊字符开头的值用引号括起来。"
        ) from None

    try:
        return parse_spec(doc, base_dir=path.parent, backend_override=backend_override)
    except SpecError as exc:
        raise SpecError(f"任务文件 {path} 有误：{exc}") from None


def parse_spec(doc: Any, *, base_dir: Path, backend_override: str | None = None) -> JobSpec:
    """校验已解析的 YAML 文档。base_dir 是相对路径的基准目录。"""
    if not isinstance(doc, dict):
        raise SpecError(
            f"文件内容应该是 `字段: 值` 形式的映射，现在是{_kind(doc)}。"
            "参考设计文档 4.2 的示例，至少写上 version、name、backend、image、run、gpu、limits。"
        )
    unknown = [key for key in doc if key not in TOP_LEVEL_KEYS]
    if unknown:
        raise _err(
            str(unknown[0]),
            "不认识的字段",
            f"检查拼写；顶层只允许这些字段：{', '.join(TOP_LEVEL_KEYS)}",
        )

    _check_version(doc)
    backend = _select_backend(doc.get("backend"), backend_override)
    name = _required_str(doc, "name", example="name: yolo-demo")
    run = _required_str(doc, "run", example="run: python train.py")
    setup = _optional_str(doc, "setup", example="setup: pip install -r requirements.txt")
    images = _parse_images(doc.get("image"), backend)
    code = _parse_code(doc.get("code", "."), base_dir)
    data = _parse_data(doc.get("data"), base_dir)
    gpu = _parse_gpu(doc.get("gpu"))
    limits = _parse_limits(doc.get("limits"), backend)
    env = _parse_env(doc.get("env"))

    return JobSpec(
        version=SPEC_VERSION,
        name=name,
        backend=backend,
        images=images,
        code=code,
        run=run,
        limits=limits,
        gpu=gpu,
        setup=setup,
        data=data,
        env=env,
    )


# ---------------------------------------------------------------- YAML：重复的 key 报错


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader，但同一个映射里出现重复的 key 时报错，而不是静默取最后一个。"""

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        if isinstance(node, yaml.MappingNode):
            seen: set[Any] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = self.construct_object(key_node, deep=True)
                try:
                    duplicate = key in seen
                except TypeError:  # 不可哈希的 key 交给父类报错
                    continue
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}（同一个字段写了两次，只保留一个）",
                        key_node.start_mark,
                    )
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


# ---------------------------------------------------------------- 各字段


def _check_version(doc: dict) -> None:
    if "version" not in doc:
        raise _err("version", "缺少版本号", "在文件开头加一行 version: 1")
    version = doc["version"]
    if type(version) is not int or version != SPEC_VERSION:
        raise _err(
            "version",
            f"不支持的版本 {version!r}",
            f"本版本的 dgpu 只认 version: {SPEC_VERSION}（整数，不加引号）；"
            "如果这个文件是给更新版本的 dgpu 写的，请升级 dgpu",
        )


def _select_backend(from_file: Any, override: str | None) -> str:
    choices = " 或 ".join(BACKENDS)
    if from_file is not None and from_file not in BACKENDS:
        raise _err("backend", f"不认识的后端 {from_file!r}", f"写 {choices}")
    if override is not None:
        if override not in BACKENDS:
            raise _err("--backend", f"不认识的后端 {override!r}", f"写 {choices}")
        return override
    if from_file is None:
        raise _err(
            "backend",
            "没有指定后端",
            f"在 job.yaml 里写 backend: {BACKEND_VAST}（或 {BACKEND_AUTODL}），"
            "或者命令行加 --backend",
        )
    return from_file


def _required_str(doc: dict, key: str, *, example: str) -> str:
    value = doc.get(key)
    if value is None:
        raise _err(key, "缺少这个必填字段", f"加上，如 {example}")
    if not isinstance(value, str):
        raise _err(key, f"应该是字符串，现在是{_kind(value)}", f"用引号括起来，如 {example}")
    if not value.strip():
        raise _err(key, "不能为空", f"填上内容，如 {example}")
    return value.strip()


def _optional_str(doc: dict, key: str, *, example: str) -> str | None:
    value = doc.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _err(
            key,
            f"应该是字符串，现在是{_kind(value)}",
            f"写成一个字符串，如 {example}；多条命令用 | 写成多行字符串",
        )
    return value.strip() or None


def _parse_images(raw: Any, backend: str) -> dict[str, str]:
    example = "image: {vast: vastai/pytorch:<固定版本>, autodl: <AutoDL 镜像 UUID>}"
    if raw is None:
        raise _err("image", "缺少镜像", f"加上，如 {example}；只用一个后端时也可以直接写镜像字符串")
    if isinstance(raw, str):
        items = [("image", backend, raw)]
    elif isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            field = f"image.{key}"
            if key not in BACKENDS:
                raise _err(field, "不认识的后端", f"image 下只能写 {', '.join(BACKENDS)}")
            if value is None:
                continue
            items.append((field, key, value))
    else:
        raise _err("image", f"应该是字符串或映射，现在是{_kind(raw)}", f"写成 {example}")

    images: dict[str, str] = {}
    for field, key, value in items:
        if not isinstance(value, str):
            raise _err(field, f"应该是字符串，现在是{_kind(value)}", "用引号括起来")
        image = value.strip()
        if not image:
            raise _err(field, "不能为空", "填上镜像名")
        if any(ch.isspace() for ch in image):
            raise _err(field, f"镜像名 {image!r} 里有空白字符", "去掉空格")
        if key == BACKEND_VAST:
            _check_vast_image(field, image)
        images[key] = image

    if backend not in images:
        raise _err(
            f"image.{backend}",
            f"选择了 {backend} 后端，但没有它的镜像",
            f"在 image 下加上 {backend}: <镜像>"
            + ("（AutoDL 只能用平台上的镜像 UUID）" if backend == BACKEND_AUTODL else ""),
        )
    return images


def _check_vast_image(field: str, image: str) -> None:
    pinned_hint = (
        "写上固定的版本 tag，如 vastai/pytorch:2.5.1-cuda-12.1.1，或用 @sha256:<摘要> 固定"
    )
    name, at, digest = image.partition("@")
    if not name:
        raise _err(field, f"镜像 {image!r} 缺少镜像名", "写成 <镜像名>@sha256:<64 位十六进制摘要>")
    if at:
        if not _DIGEST_RE.match(digest):
            raise _err(
                field,
                f"镜像 {image!r} 的摘要格式不对",
                "写成 <镜像名>@sha256:<64 位小写十六进制>",
            )
        return
    last = name.rsplit("/", 1)[-1]  # 最后一段之前的冒号是 registry 端口
    repo, colon, tag = last.partition(":")
    if not colon:
        raise _err(
            field,
            f"镜像 {image!r} 没有写 tag（等于 latest，镜像更新后结果不可复现）",
            pinned_hint,
        )
    if not repo or not tag:
        raise _err(field, f"镜像 {image!r} 格式不对", pinned_hint)
    if tag == "latest":
        raise _err(field, f"镜像 {image!r} 用了 latest（镜像更新后结果不可复现）", pinned_hint)


def _resolve_dir(base_dir: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return Path(os.path.abspath(path))


def _parse_code(raw: Any, base_dir: Path) -> Path:
    if raw is None:
        raw = "."
    if not isinstance(raw, str) or not raw.strip():
        raise _err(
            "code",
            f"应该是目录路径字符串，现在是{_kind(raw)}",
            "写相对 job.yaml 的目录，如 code: .",
        )
    code = _resolve_dir(base_dir, raw.strip())
    if not code.exists():
        raise _err("code", f"目录 {code} 不存在", "写相对 job.yaml 所在目录的路径，默认是 .")
    if not code.is_dir():
        raise _err("code", f"{code} 不是目录", "code 要指向训练代码所在的目录")
    return code


def _parse_data(raw: Any, base_dir: Path) -> tuple[DataRef, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise _err(
            "data",
            f"应该是列表，现在是{_kind(raw)}",
            "每行一项，如\ndata:\n  - r2://my-datasets/coco8\n  - ./datasets/my-set",
        )
    refs: list[DataRef] = []
    seen: dict[str, str] = {}
    for index, item in enumerate(raw):
        field = f"data[{index}]"
        if not isinstance(item, str) or not item.strip():
            raise _err(
                field,
                f"应该是非空字符串，现在是{_kind(item)}",
                "写 r2://<bucket>/<目录> 或本地目录路径",
            )
        text = item.strip()
        match = _URL_RE.match(text)
        if match:
            ref = _parse_remote(field, text, match.group(1), match.group(2))
        else:
            ref = _parse_local(field, text, base_dir)
        if ref.name in seen:
            raise _err(
                field,
                f"目录名 {ref.name!r} 和 {seen[ref.name]!r} 重复，"
                f"两者都会拷到 $DGPU_DATA/{ref.name}",
                "改名其中一个目录，或者只保留一个",
            )
        seen[ref.name] = text
        refs.append(ref)
    return tuple(refs)


def _parse_remote(field: str, text: str, scheme: str, rest: str) -> DataRef:
    if scheme.lower() != DATA_SCHEME:
        raise _err(
            field,
            f"不支持 {scheme}:// 地址 {text!r}",
            f"存储里的数据写成 {DATA_SCHEME}://<bucket>/<目录>（用配置里该后端的存储访问），"
            "本地目录直接写路径",
        )
    bucket, _, prefix = rest.partition("/")
    if not _BUCKET_RE.match(bucket):
        raise _err(
            field,
            f"{text!r} 里的 bucket 名 {bucket!r} 不合法",
            "bucket 名是 3–63 个小写字母、数字、点或连字符，写成 r2://<bucket>/<目录>",
        )
    prefix = prefix.strip("/")
    if not prefix:
        raise _err(
            field,
            f"{text!r} 指向整个 bucket",
            "写到 bucket 里的数据目录，如 r2://my-datasets/coco8；目录名就是 $DGPU_DATA 下的名字",
        )
    segments = prefix.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise _err(field, f"{text!r} 的路径里有空段、. 或 ..", "写成 r2://<bucket>/<目录>/<子目录>")
    return DataRef(name=segments[-1], bucket=bucket, prefix="/".join(segments) + "/")


def _parse_local(field: str, text: str, base_dir: Path) -> DataRef:
    path = _resolve_dir(base_dir, text)
    if not path.exists():
        raise _err(
            field,
            f"本地数据目录 {path} 不存在",
            f"检查路径（相对 job.yaml 所在目录）；存储里的数据写成 {DATA_SCHEME}://<bucket>/<目录>",
        )
    if not path.is_dir():
        raise _err(
            field,
            f"{path} 不是目录",
            "data 的每一项都是目录；单个文件放进一个目录里再引用",
        )
    if not path.name:
        raise _err(field, f"不能把根目录 {path} 作为数据目录", "指向具体的数据目录")
    return DataRef(name=path.name, local_path=path)


def _parse_gpu(raw: Any) -> GpuSpec:
    example = "gpu: {names: [RTX 3090, RTX 4090], count: 1}"
    if raw is None:
        raise _err("gpu", "缺少 GPU 要求", f"加上，如 {example}")
    if not isinstance(raw, dict):
        raise _err("gpu", f"应该是映射，现在是{_kind(raw)}", f"写成 {example}")
    unknown = [key for key in raw if key not in GPU_KEYS]
    if unknown:
        raise _err(f"gpu.{unknown[0]}", "不认识的字段", f"gpu 下只允许 {', '.join(GPU_KEYS)}")

    names = raw.get("names")
    if names is None:
        raise _err(
            "gpu.names",
            "缺少这个必填字段",
            "列出可以接受的显卡型号，如 names: [RTX 3090, RTX 4090]",
        )
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names:
        raise _err(
            "gpu.names",
            f"应该是非空的列表，现在是{_kind(names)}",
            "列出可以接受的显卡型号，如 names: [RTX 3090, RTX 4090]",
        )
    cleaned: list[str] = []
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise _err(
                f"gpu.names[{index}]",
                f"应该是显卡型号字符串，现在是{_kind(name)}",
                "写成 RTX 4090 这样的型号名，纯数字的型号要加引号",
            )
        cleaned.append(name.strip())

    count = raw.get("count", 1)
    if type(count) is not int or count < 1:
        raise _err(
            "gpu.count", f"应该是 ≥ 1 的整数，现在是 {count!r}", "写单机要用的卡数，如 count: 1"
        )
    return GpuSpec(names=tuple(cleaned), count=count)


def _parse_limits(raw: Any, backend: str) -> Limits:
    example = "limits: {max_hours: 6, max_usd_per_hour: 0.40}"
    if raw is None:
        raise _err("limits", "缺少费用上限（必填，没有默认值）", f"加上，如 {example}")
    if not isinstance(raw, dict):
        raise _err("limits", f"应该是映射，现在是{_kind(raw)}", f"写成 {example}")
    unknown = [key for key in raw if key not in LIMIT_KEYS]
    if unknown:
        raise _err(
            f"limits.{unknown[0]}", "不认识的字段", f"limits 下只允许 {', '.join(LIMIT_KEYS)}"
        )

    max_hours = _positive_number(raw, "max_hours", unit="小时")
    if max_hours is None:
        raise _err(
            "limits.max_hours",
            "缺少这个必填字段",
            "写任务最多运行的小时数，如 max_hours: 6；超时会终止训练并销毁实例",
        )
    usd = _positive_number(raw, "max_usd_per_hour", unit="美元/小时")
    cny = _positive_number(raw, "max_cny_per_hour", unit="元/小时")

    price_key = PRICE_KEY_BY_BACKEND[backend]
    if raw.get(price_key) is None:
        raise _err(
            f"limits.{price_key}",
            f"选择了 {backend} 后端，但没有填它的每小时价格上限（{CURRENCY_BY_BACKEND[backend]}）",
            f"加上 {price_key}: <数值>；只需要填所用后端的价格上限",
        )
    return Limits(max_hours=max_hours, max_usd_per_hour=usd, max_cny_per_hour=cny)


def _positive_number(raw: dict, key: str, *, unit: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    field = f"limits.{key}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _err(field, f"应该是数字（{unit}），现在是{_kind(value)}", f"去掉引号，如 {key}: 2")
    if not math.isfinite(value) or value <= 0:
        raise _err(field, f"必须是大于 0 的有限数字，现在是 {value!r}", f"填一个正数（{unit}）")
    return float(value)


def _parse_env(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _err("env", f"应该是映射，现在是{_kind(raw)}", '写成\nenv:\n  EPOCHS: "100"')
    env: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _ENV_KEY_RE.match(key):
            raise _err(
                f"env.{key}",
                "不是合法的环境变量名",
                "只用字母、数字和下划线，且不以数字开头",
            )
        field = f"env.{key}"
        reserved = next((p for p in RESERVED_ENV_PREFIXES if key.startswith(p)), None)
        if reserved is not None:
            raise _err(
                field,
                f"以 {reserved} 开头的变量由 dgpu 自己设置或会从训练环境中移除",
                "改个名字；存储凭据不要写进 job.yaml，训练代码通过 $DGPU_DATA 等目录读写数据",
            )
        if not isinstance(value, str):
            raise _err(
                field,
                f"值是{_kind(value)}，环境变量只能是字符串"
                "（YAML 会把 100、0.1、yes 等转成别的类型）",
                f'把值用引号括起来，如 {key}: "100"',
            )
        env[key] = value
    return env


# ---------------------------------------------------------------- 工具


def _err(field: str, problem: str, fix: str) -> SpecError:
    return SpecError(f"{field}：{problem}。{fix}。")


def _kind(value: Any) -> str:
    if value is None:
        return "空值"
    if isinstance(value, bool):
        return "布尔值（true/false/yes/no）"
    if isinstance(value, int):
        return "整数"
    if isinstance(value, float):
        return "小数"
    if isinstance(value, str):
        return "字符串"
    if isinstance(value, list):
        return "列表"
    if isinstance(value, dict):
        return "映射"
    return type(value).__name__
