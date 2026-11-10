"""job.yaml 的解析和校验：每一种校验错误都有测试，错误信息要指出字段并且是中文。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from dgpu.contracts import DataRef, RunnerSettings, SpecError, build_remote_spec
from dgpu.spec import load_spec

DIGEST = "sha256:" + "a" * 64
DELETE = object()

BASE: dict[str, Any] = {
    "version": 1,
    "name": "yolo-demo",
    "backend": "vast",
    "image": {"vast": "vastai/pytorch:2.5.1-cuda-12.1", "autodl": "base-image-l2t43iu6uk"},
    "setup": "pip install ultralytics==8.4.208",
    "run": "python train.py",
    "gpu": {"names": ["RTX 3090", "RTX 4090"], "count": 1},
    "limits": {"max_hours": 6, "max_usd_per_hour": 0.4, "max_cny_per_hour": 2.0},
    "env": {"EPOCHS": "100"},
}

_CJK = re.compile(r"[一-鿿]")


def _apply(doc: dict, dotted: str, value: Any) -> dict:
    doc = copy.deepcopy(doc)
    *parents, last = dotted.split(".")
    node = doc
    for key in parents:
        node = node[key]
    if value is DELETE:
        node.pop(last, None)
    else:
        node[last] = value
    return doc


def _write(directory: Path, doc: Any) -> Path:
    path = directory / "job.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _write_text(directory: Path, text: str) -> Path:
    path = directory / "job.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _error(path: Path, *needles: str, backend: str | None = None) -> str:
    with pytest.raises(SpecError) as info:
        load_spec(path, backend_override=backend)
    message = str(info.value)
    for needle in needles:
        assert needle in message, f"{needle!r} not in {message!r}"
    assert _CJK.search(message), message
    return message


# ---------------------------------------------------------------- 合法的文件


def test_full_spec_is_parsed(tmp_path):
    (tmp_path / "datasets" / "my-small-set").mkdir(parents=True)
    doc = dict(BASE, data=["r2://my-datasets/coco8", "./datasets/my-small-set"])
    spec = load_spec(_write(tmp_path, doc))

    assert spec.version == 1
    assert spec.name == "yolo-demo"
    assert spec.backend == "vast"
    assert spec.images == {
        "vast": "vastai/pytorch:2.5.1-cuda-12.1",
        "autodl": "base-image-l2t43iu6uk",
    }
    assert spec.code == tmp_path
    assert spec.setup == "pip install ultralytics==8.4.208"
    assert spec.run == "python train.py"
    assert spec.gpu.names == ("RTX 3090", "RTX 4090")
    assert spec.gpu.count == 1
    assert spec.limits.max_hours == 6.0
    assert spec.limits.max_price("vast") == 0.4
    assert spec.limits.max_price("autodl") == 2.0
    assert spec.env == {"EPOCHS": "100"}
    assert spec.data == (
        DataRef(name="coco8", bucket="my-datasets", prefix="coco8/"),
        DataRef(name="my-small-set", local_path=tmp_path / "datasets" / "my-small-set"),
    )


def test_optional_fields_default(tmp_path):
    doc = _apply(BASE, "setup", DELETE)
    doc = _apply(doc, "env", DELETE)
    doc = _apply(doc, "gpu.count", DELETE)
    spec = load_spec(_write(tmp_path, doc))
    assert spec.code == tmp_path
    assert spec.setup is None
    assert spec.data == ()
    assert spec.env == {}
    assert spec.gpu.count == 1
    assert spec.run_as_root is False  # 默认以无特权用户运行训练


@pytest.mark.parametrize("value", [True, False])
def test_run_as_root_reaches_the_remote_spec(tmp_path, value):
    spec = load_spec(_write(tmp_path, dict(BASE, run_as_root=value)))
    assert spec.run_as_root is value
    remote = build_remote_spec(
        spec, job_id="j-1", backend="vast", data=[], runner=RunnerSettings()
    )
    assert remote["run_as_root"] is value


@pytest.mark.parametrize("value", ["yes", 1, "true", [True]])
def test_run_as_root_must_be_a_real_boolean(tmp_path, value):
    _error(_write(tmp_path, dict(BASE, run_as_root=value)), "run_as_root", "true 或 false")


def test_relative_paths_resolve_against_yaml_dir_not_cwd(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "data" / "set1").mkdir(parents=True)
    _write(project, dict(BASE, code="src", data=["data/set1"]))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    spec = load_spec(Path("..") / "project" / "job.yaml")
    assert spec.code == project / "src"
    assert spec.code.is_absolute()
    assert spec.data[0].local_path == project / "data" / "set1"


def test_local_data_name_uses_link_name_not_target(tmp_path):
    target = tmp_path / "storage" / "coco2017"
    target.mkdir(parents=True)
    (tmp_path / "coco").symlink_to(target, target_is_directory=True)
    spec = load_spec(_write(tmp_path, dict(BASE, data=["./coco"])))
    assert spec.data[0].name == "coco"


@pytest.mark.parametrize(
    ("uri", "name", "bucket", "prefix"),
    [
        ("r2://my-datasets/coco8", "coco8", "my-datasets", "coco8/"),
        ("r2://bkt/a/b/", "b", "bkt", "a/b/"),
        ("r2://bkt//a/b", "b", "bkt", "a/b/"),
        ("R2://bkt.with.dots/x", "x", "bkt.with.dots", "x/"),
    ],
)
def test_remote_data_refs(tmp_path, uri, name, bucket, prefix):
    spec = load_spec(_write(tmp_path, dict(BASE, data=[uri])))
    assert spec.data == (DataRef(name=name, bucket=bucket, prefix=prefix),)
    assert not spec.data[0].is_local


def test_single_string_for_lists_is_accepted(tmp_path):
    doc = dict(BASE, data="r2://bkt/coco8")
    doc = _apply(doc, "gpu.names", "RTX 4090")
    spec = load_spec(_write(tmp_path, doc))
    assert spec.gpu.names == ("RTX 4090",)
    assert [ref.name for ref in spec.data] == ["coco8"]


def test_backend_override_wins_and_needs_its_price(tmp_path):
    path = _write(tmp_path, BASE)
    spec = load_spec(path, backend_override="autodl")
    assert spec.backend == "autodl"
    assert spec.image_for("autodl") == "base-image-l2t43iu6uk"

    no_cny = _write(tmp_path, _apply(BASE, "limits.max_cny_per_hour", DELETE))
    assert load_spec(no_cny).backend == "vast"  # vast 不需要人民币价格
    _error(no_cny, "limits.max_cny_per_hour", "autodl", backend="autodl")


def test_backend_from_override_when_file_has_none(tmp_path):
    path = _write(tmp_path, _apply(BASE, "backend", DELETE))
    assert load_spec(path, backend_override="vast").backend == "vast"


def test_plain_string_image_applies_to_selected_backend(tmp_path):
    path = _write(tmp_path, dict(BASE, image="vastai/pytorch:2.5.1"))
    assert load_spec(path).images == {"vast": "vastai/pytorch:2.5.1"}

    # AutoDL 的镜像是平台 UUID，不检查 tag
    path = _write(tmp_path, dict(BASE, image="base-image-l2t43iu6uk"))
    spec = load_spec(path, backend_override="autodl")
    assert spec.images == {"autodl": "base-image-l2t43iu6uk"}
    # 同一个字符串给 vast 用就是没有 tag 的镜像
    _error(path, "image", "tag")


@pytest.mark.parametrize(
    "image",
    [
        "vastai/pytorch:2.5.1-cuda-12.1",
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        "localhost:5000/team/img:1.0",
        f"vastai/pytorch@{DIGEST}",
        f"ghcr.io/org/img:latest@{DIGEST}",  # 摘要已经固定了内容
    ],
)
def test_pinned_vast_images_are_accepted(tmp_path, image):
    spec = load_spec(_write(tmp_path, _apply(BASE, "image.vast", image)))
    assert spec.image_for("vast") == image


def test_autodl_image_may_be_missing_when_using_vast(tmp_path):
    spec = load_spec(_write(tmp_path, _apply(BASE, "image.autodl", DELETE)))
    assert spec.images == {"vast": "vastai/pytorch:2.5.1-cuda-12.1"}
    spec = load_spec(_write(tmp_path, _apply(BASE, "image.autodl", None)))
    assert "autodl" not in spec.images


# ---------------------------------------------------------------- 校验错误

INVALID = [
    # version
    ("version", DELETE, ["version", "version: 1"]),
    ("version", 2, ["version", "2"]),
    ("version", "1", ["version"]),
    ("version", 1.0, ["version"]),
    ("version", True, ["version"]),
    # 顶层
    ("gpus", {"names": ["RTX 3090"]}, ["gpus", "不认识"]),
    # name / run / setup
    ("name", DELETE, ["name", "必填"]),
    ("name", "  ", ["name", "不能为空"]),
    ("name", 123, ["name", "字符串"]),
    ("run", DELETE, ["run", "必填"]),
    ("run", "", ["run", "不能为空"]),
    ("run", ["python a.py", "python b.py"], ["run", "字符串"]),
    ("setup", ["pip install x"], ["setup", "字符串"]),
    # backend
    ("backend", DELETE, ["backend", "--backend"]),
    ("backend", "runpod", ["backend", "runpod"]),
    # image
    ("image", DELETE, ["image", "缺少镜像"]),
    ("image", 5, ["image", "字符串或映射"]),
    ("image", {"vast": "a/b:1", "runpod": "x"}, ["image.runpod", "不认识的后端"]),
    ("image", {"autodl": "base-image-x"}, ["image.vast", "没有它的镜像"]),
    ("image.vast", "vastai/pytorch", ["image.vast", "没有写 tag", "固定"]),
    ("image.vast", "vastai/pytorch:latest", ["image.vast", "latest"]),
    ("image.vast", "localhost:5000/pytorch", ["image.vast", "没有写 tag"]),
    ("image.vast", "vastai/pytorch:", ["image.vast", "格式不对"]),
    ("image.vast", "vastai/pytorch@sha256:abc", ["image.vast", "摘要"]),
    ("image.vast", "vastai/pytorch@md5:" + "a" * 32, ["image.vast", "摘要"]),
    ("image.vast", f"@{DIGEST}", ["image.vast", "镜像名"]),
    ("image.vast", "vastai/pytorch :2.5", ["image.vast", "空白"]),
    ("image.vast", "", ["image.vast", "不能为空"]),
    ("image.vast", 2.5, ["image.vast", "字符串"]),
    # code
    ("code", "no-such-dir", ["code", "不存在"]),
    ("code", 5, ["code", "目录"]),
    ("code", "", ["code", "目录"]),
    # data
    ("data", {"coco": "r2://bkt/coco"}, ["data", "列表"]),
    ("data", [5], ["data[0]", "字符串"]),
    ("data", [""], ["data[0]"]),
    ("data", ["r2://bkt"], ["data[0]", "整个 bucket"]),
    ("data", ["r2://bkt/"], ["data[0]", "整个 bucket"]),
    ("data", ["r2://Bad_Bucket/coco"], ["data[0]", "bucket"]),
    ("data", ["r2://bkt/a/../b"], ["data[0]", ".."]),
    ("data", ["r2://bkt/./b"], ["data[0]", ".."]),
    ("data", ["s3://bkt/coco"], ["data[0]", "s3://", "r2://"]),
    ("data", ["./no-such-dir"], ["data[0]", "不存在"]),
    (
        "data",
        ["r2://bkt/train/coco", "r2://bkt/val/coco"],
        ["data[1]", "重复", "coco"],
    ),
    # gpu
    ("gpu", DELETE, ["gpu", "缺少"]),
    ("gpu", "RTX 3090", ["gpu", "映射"]),
    ("gpu.names", DELETE, ["gpu.names", "必填"]),
    ("gpu.names", [], ["gpu.names", "非空"]),
    ("gpu.names", {"a": 1}, ["gpu.names", "列表"]),
    ("gpu.names", ["RTX 3090", 4090], ["gpu.names[1]", "引号"]),
    ("gpu.names", [" "], ["gpu.names[0]"]),
    ("gpu.count", 0, ["gpu.count", "≥ 1"]),
    ("gpu.count", -1, ["gpu.count"]),
    ("gpu.count", 1.5, ["gpu.count", "整数"]),
    ("gpu.count", True, ["gpu.count"]),
    ("gpu.count", "2", ["gpu.count"]),
    ("gpu.vram_gb", 24, ["gpu.vram_gb", "不认识"]),
    # limits
    ("limits", DELETE, ["limits", "必填"]),
    ("limits", 6, ["limits", "映射"]),
    ("limits.max_hours", DELETE, ["limits.max_hours", "必填"]),
    ("limits.max_hours", 0, ["limits.max_hours", "大于 0"]),
    ("limits.max_hours", -2, ["limits.max_hours", "大于 0"]),
    ("limits.max_hours", float("inf"), ["limits.max_hours", "有限"]),
    ("limits.max_hours", float("nan"), ["limits.max_hours"]),
    ("limits.max_hours", "6", ["limits.max_hours", "数字", "去掉引号"]),
    ("limits.max_hours", True, ["limits.max_hours", "数字"]),
    ("limits.max_usd_per_hour", DELETE, ["limits.max_usd_per_hour", "vast", "USD"]),
    ("limits.max_usd_per_hour", 0, ["limits.max_usd_per_hour", "大于 0"]),
    ("limits.max_usd_per_hour", "0.4", ["limits.max_usd_per_hour", "数字"]),
    ("limits.max_cny_per_hour", -1, ["limits.max_cny_per_hour", "大于 0"]),
    ("limits.max_price", 1, ["limits.max_price", "不认识"]),
    # env
    ("env", ["EPOCHS=100"], ["env", "映射"]),
    ("env", {"EPOCHS": 100}, ["env.EPOCHS", "整数", "引号"]),
    ("env", {"LR": 0.01}, ["env.LR", "小数", "引号"]),
    ("env", {"DEBUG": True}, ["env.DEBUG", "布尔值", "引号"]),
    ("env", {"EMPTY": None}, ["env.EMPTY", "空值", "引号"]),
    ("env", {"LIST": [1]}, ["env.LIST", "引号"]),
    ("env", {"1ABC": "x"}, ["env.1ABC", "环境变量名"]),
    ("env", {"MY-VAR": "x"}, ["env.MY-VAR", "环境变量名"]),
    ("env", {"DGPU_DATA": "/x"}, ["env.DGPU_DATA", "DGPU_"]),
    ("env", {"AWS_SECRET_ACCESS_KEY": "x"}, ["env.AWS_SECRET_ACCESS_KEY", "AWS_"]),
    ("env", {"R2_ACCESS_KEY_ID": "x"}, ["env.R2_ACCESS_KEY_ID", "R2_"]),
    ("env", {"RCLONE_CONFIG_R2_TYPE": "s3"}, ["env.RCLONE_CONFIG_R2_TYPE"]),
    ("env", {"CONTAINER_API_KEY": "x"}, ["env.CONTAINER_API_KEY"]),
]


@pytest.mark.parametrize(
    ("field", "value", "needles"),
    INVALID,
    ids=[f"{field}={value!r}" for field, value, _ in INVALID],
)
def test_invalid_field_is_rejected(tmp_path, field, value, needles):
    path = _write(tmp_path, _apply(BASE, field, value))
    message = _error(path, *needles)
    assert str(path) in message


def test_unknown_backend_override_is_rejected(tmp_path):
    _error(_write(tmp_path, BASE), "--backend", "runpod", backend="runpod")


def test_override_does_not_hide_invalid_backend_in_file(tmp_path):
    path = _write(tmp_path, _apply(BASE, "backend", "vasst"))
    _error(path, "backend", "vasst", backend="vast")


def test_code_must_be_a_directory(tmp_path):
    (tmp_path / "train.py").write_text("print(1)\n")
    _error(_write(tmp_path, dict(BASE, code="train.py")), "code", "不是目录")


def test_local_data_must_be_a_directory(tmp_path):
    (tmp_path / "data.zip").write_bytes(b"zip")
    _error(_write(tmp_path, dict(BASE, data=["data.zip"])), "data[0]", "不是目录")


@pytest.mark.parametrize("name", ["my data", "数据集", "a+b", "x:y"])
def test_data_names_the_runner_would_reject_are_refused(tmp_path, name):
    (tmp_path / name).mkdir()
    _error(_write(tmp_path, dict(BASE, data=[f"./{name}"])), "data[0]", "不支持的字符")
    _error(_write(tmp_path, dict(BASE, data=[f"r2://bkt/sets/{name}"])), "data[0]", "不支持的字符")


def test_data_names_with_dots_dashes_and_underscores_are_fine(tmp_path):
    (tmp_path / "coco8_v1.2-small").mkdir()
    spec = load_spec(
        _write(tmp_path, dict(BASE, data=["./coco8_v1.2-small", "r2://bkt/a/B.c-d_e"]))
    )
    assert [ref.name for ref in spec.data] == ["coco8_v1.2-small", "B.c-d_e"]


def test_local_and_remote_data_with_same_name_are_rejected(tmp_path):
    (tmp_path / "coco8").mkdir()
    path = _write(tmp_path, dict(BASE, data=["r2://bkt/coco8", "./coco8"]))
    _error(path, "data[1]", "重复")


_WITHOUT_ENV = yaml.safe_dump({k: v for k, v in BASE.items() if k != "env"}, allow_unicode=True)


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("EPOCHS: 100", "整数"),
        ("LR: 0.001", "小数"),
        ("DEBUG: yes", "布尔值"),
        ("FLAG: on", "布尔值"),
    ],
)
def test_unquoted_yaml_env_values_are_rejected_not_converted(tmp_path, line, kind):
    key = line.split(":")[0]
    path = _write_text(tmp_path, _WITHOUT_ENV + f"env:\n  {line}\n")
    _error(path, f"env.{key}", kind, "引号")


def test_env_strings_are_kept_verbatim(tmp_path):
    # PyYAML（YAML 1.1）把 1e-3 当字符串；原样保留，不做任何转换
    path = _write_text(tmp_path, _WITHOUT_ENV + 'env:\n  LR: 1e-3\n  EPOCHS: "100"\n')
    assert load_spec(path).env == {"LR": "1e-3", "EPOCHS": "100"}


def test_env_error_does_not_echo_the_value(tmp_path):
    path = _write(tmp_path, _apply(BASE, "env", {"TOKEN": 987654321}))
    message = _error(path, "env.TOKEN")
    assert "987654321" not in message


# ---------------------------------------------------------------- 文件层面的错误


def test_missing_file(tmp_path):
    _error(tmp_path / "nope.yaml", "找不到", "nope.yaml")


def test_path_is_a_directory(tmp_path):
    _error(tmp_path, "目录")


def test_invalid_yaml(tmp_path):
    _error(_write_text(tmp_path, "version: 1\nname: [unclosed\n"), "YAML", "line")


def test_non_utf8_file(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_bytes("name: 中文".encode("gbk"))
    _error(path, "UTF-8")


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"])
def test_top_level_must_be_a_mapping(tmp_path, text):
    _error(_write_text(tmp_path, text), "映射")


def test_duplicate_keys_are_rejected(tmp_path):
    text = yaml.safe_dump(BASE, allow_unicode=True, sort_keys=False)
    _error(_write_text(tmp_path, text + "run: python other.py\n"), "duplicate key", "run")

    nested = text.replace("  count: 1\n", "  count: 1\n  count: 2\n")
    assert nested != text
    _error(_write_text(tmp_path, nested), "duplicate key", "count")


def test_yaml_merge_keys_still_work(tmp_path):
    text = yaml.safe_dump({k: v for k, v in BASE.items() if k != "limits"}, allow_unicode=True)
    text += "limits:\n  <<: {max_hours: 6, max_usd_per_hour: 0.1}\n  max_usd_per_hour: 0.3\n"
    spec = load_spec(_write_text(tmp_path, text))
    assert spec.limits.max_usd_per_hour == 0.3
    assert spec.limits.max_hours == 6.0
