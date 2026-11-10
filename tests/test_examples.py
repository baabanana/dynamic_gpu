"""YOLO 示例、Skill 和使用说明的测试。

不安装、不导入 ultralytics 和 torch：train.py 的辅助函数用假对象测试，
main() 用注入到 sys.modules 的假 ultralytics 模块测试。
文档测试检查其中出现的命令、参数、环境变量、状态和退出码与设计、states.py 一致。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import py_compile
import re
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dgpu import contracts, layout, states

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "yolo"
JOB_YAML = EXAMPLE_DIR / "job.yaml"
TRAIN_PY = EXAMPLE_DIR / "train.py"
EXAMPLE_README = EXAMPLE_DIR / "README.md"
SKILL_MD = ROOT / "skills" / "dgpu" / "SKILL.md"
USAGE_MD = ROOT / "docs" / "usage.md"
DOCS = (SKILL_MD, USAGE_MD, EXAMPLE_README)

# 设计 4.5 的命令表，加上 init
DESIGN_COMMANDS = {
    "doctor",
    "offers",
    "submit",
    "wait",
    "status",
    "logs",
    "list",
    "retry",
    "cancel",
    "pull",
    "gc",
}
ALLOWED_COMMANDS = DESIGN_COMMANDS | {"init"}
# 文档里用到的参数：设计 4.5（--backend、--retries、-f、--force、--dry-run）、
# 4.5 的本地数据确认（--yes）、4.10 的配置文件位置（--config），以及 --help
ALLOWED_FLAGS = {
    "--backend",
    "--retries",
    "-f",
    "--force",
    "--dry-run",
    "--yes",
    "--config",
    "--help",
}

REQUIRED_SPEC_KEYS = {"version", "name", "backend", "image", "code", "run", "gpu", "limits"}
SPEC_KEYS = REQUIRED_SPEC_KEYS | {"setup", "data", "env"}

ENV_NAMES = {
    states.ENV_JOB_ID,
    states.ENV_ATTEMPT,
    states.ENV_DATA,
    states.ENV_OUTPUT,
    states.ENV_CKPT,
    states.ENV_RESUME,
}
REASONS = {value for key, value in vars(states).items() if key.startswith("REASON_")}
ALL_PHASES = set(states.ACTIVE_PHASES) | set(states.TERMINAL_PHASES) | {states.LOST}


# ---------------------------------------------------------------- 加载 train.py


def _load_train(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """按路径导入 train.py；导入期间 ultralytics 和 torch 都不可用。"""
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    name = "dgpu_example_yolo_train"
    spec = importlib.util.spec_from_file_location(name, TRAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def train(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    return _load_train(monkeypatch)


def _no_tmp_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.tmp"))


# ---------------------------------------------------------------- job.yaml


@pytest.fixture(scope="module")
def job() -> dict:
    doc = yaml.safe_load(JOB_YAML.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_job_yaml_has_the_fields_the_design_requires(job):
    assert REQUIRED_SPEC_KEYS <= set(job) <= SPEC_KEYS
    assert job["version"] == states.SCHEMA_VERSION
    assert job["backend"] in contracts.BACKENDS
    assert layout.is_valid_job_id(layout.new_job_id(job["name"]))
    assert job["name"] == layout.new_job_id(job["name"]).rsplit("-", 2)[0]


def test_job_yaml_fills_in_both_backends(job):
    images = job["image"]
    assert set(images) == set(contracts.BACKENDS)
    repo, _, tag = images["vast"].rpartition(":")
    assert repo and tag and tag != "latest", "Vast 镜像必须写固定 tag"
    assert isinstance(images["autodl"], str) and images["autodl"].strip()

    limits = contracts.Limits(**job["limits"])
    for backend in contracts.BACKENDS:
        assert limits.max_price(backend) > 0


def test_job_yaml_limits_are_small(job):
    limits = job["limits"]
    assert 0 < limits["max_hours"] <= 2
    assert limits["max_usd_per_hour"] <= 1.0
    assert limits["max_cny_per_hour"] <= 5.0


def test_job_yaml_pins_ultralytics_and_runs_train_py(job):
    assert "ultralytics==8.4.208" in job["setup"]
    code_dir = (EXAMPLE_DIR / job["code"]).resolve()
    assert code_dir == EXAMPLE_DIR
    script = job["run"].split()[-1]
    assert (code_dir / script).resolve() == TRAIN_PY


def test_job_yaml_gpu_data_and_env(job, train):
    gpu = job["gpu"]
    assert gpu["names"] and all(isinstance(name, str) for name in gpu["names"])
    assert isinstance(gpu["count"], int) and gpu["count"] >= 1

    assert job["data"], "示例要演示数据怎么到机器上"
    data_names = set()
    for item in job["data"]:
        assert isinstance(item, str)
        if not item.startswith("r2://"):
            assert not Path(item).is_absolute(), "示例的本地数据路径要相对 job.yaml"
        data_names.add(item.rstrip("/").rsplit("/", 1)[-1])

    env = job["env"]
    assert all(isinstance(value, str) for value in env.values()), "env 的值要写成字符串"
    assert set(env) <= set(train.DEFAULTS), "env 里只放 train.py 认识的参数"
    assert env["DATASET"] in data_names


# ---------------------------------------------------------------- train.py：可编译、可导入


def test_train_py_compiles(tmp_path):
    py_compile.compile(str(TRAIN_PY), cfile=str(tmp_path / "train.pyc"), doraise=True)
    # 要能在 AutoDL 公共镜像的 Python 3.8 上运行
    ast.parse(TRAIN_PY.read_text(encoding="utf-8"), feature_version=(3, 8))


def test_train_py_imports_without_ultralytics_or_torch(train):
    assert sys.modules["ultralytics"] is None
    assert sys.modules["torch"] is None
    assert callable(train.main)


def test_train_py_uses_the_dgpu_env_names(train):
    source = TRAIN_PY.read_text(encoding="utf-8")
    for name in (states.ENV_DATA, states.ENV_OUTPUT, states.ENV_CKPT, states.ENV_RESUME):
        assert name in source


# ---------------------------------------------------------------- train.py：checkpoint 原子写入


def test_atomic_copy_writes_complete_file_and_leaves_no_tmp(train, tmp_path):
    src = tmp_path / "src" / "last.pt"
    src.parent.mkdir()
    payload = bytes(range(256)) * 4096
    src.write_bytes(payload)
    dst = tmp_path / "ckpt" / "last.pt"

    train.atomic_copy(src, dst)

    assert dst.read_bytes() == payload
    assert _no_tmp_files(tmp_path) == []


def test_atomic_copy_keeps_old_checkpoint_when_copy_fails(train, tmp_path, monkeypatch):
    src = tmp_path / "new.pt"
    src.write_bytes(b"new checkpoint" * 1000)
    dst = tmp_path / "ckpt" / "last.pt"
    dst.parent.mkdir()
    dst.write_bytes(b"old checkpoint")

    def copy_half_then_fail(source, target):
        Path(target).write_bytes(Path(source).read_bytes()[:100])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(train.shutil, "copyfile", copy_half_then_fail)
    with pytest.raises(OSError):
        train.atomic_copy(src, dst)

    assert dst.read_bytes() == b"old checkpoint"
    assert _no_tmp_files(tmp_path) == []


def test_atomic_copy_missing_source_raises_and_changes_nothing(train, tmp_path):
    dst = tmp_path / "last.pt"
    dst.write_bytes(b"old")
    with pytest.raises(FileNotFoundError):
        train.atomic_copy(tmp_path / "missing.pt", dst)
    assert dst.read_bytes() == b"old"
    assert _no_tmp_files(tmp_path) == []


def test_ckpt_callback_copies_trainer_last_into_ckpt_dir(train, tmp_path):
    weights = tmp_path / "output" / "run" / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    ckpt_dir = tmp_path / "ckpt"
    callback = train.make_ckpt_callback(ckpt_dir)

    for epoch in range(3):
        last.write_bytes(f"weights after epoch {epoch}".encode())
        callback(SimpleNamespace(last=str(last), epoch=epoch))
        assert (ckpt_dir / "last.pt").read_bytes() == last.read_bytes()

    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["last.pt"]


# ---------------------------------------------------------------- train.py：续跑判断


def _loader(result):
    calls = []

    def load(path):
        calls.append(path)
        if isinstance(result, BaseException):
            raise result
        return result

    load.calls = calls
    return load


def test_resume_needs_the_flag(train, tmp_path):
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x")
    load = _loader({"epoch": 3})
    decision = train.decide_resume(False, ckpt, load)
    assert decision.resume is False
    assert load.calls == []


def test_resume_without_checkpoint_starts_fresh(train, tmp_path):
    load = _loader({"epoch": 3})
    decision = train.decide_resume(True, tmp_path / "last.pt", load)
    assert decision.resume is False
    assert "从头训练" in decision.reason
    assert load.calls == []


def test_resume_from_unfinished_checkpoint(train, tmp_path):
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x")
    decision = train.decide_resume(
        True, ckpt, _loader({"epoch": 4, "train_args": {"epochs": 10}})
    )
    assert decision.resume is True
    assert "已完成 5 轮" in decision.reason and "共 10 轮" in decision.reason


def test_resume_without_train_args_trusts_epoch(train, tmp_path):
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x")
    assert train.decide_resume(True, ckpt, _loader({"epoch": 0})).resume is True


@pytest.mark.parametrize(
    ("loaded", "hint"),
    [
        ({"epoch": -1, "train_args": {"epochs": 10}}, "epoch=-1"),  # 训练结束后精简过
        ({"epoch": 9, "train_args": {"epochs": 10}}, "全部训练完"),  # 最后一轮已保存
        ({"train_args": {"epochs": 10}}, "没有 epoch"),
        ({"epoch": True}, "没有 epoch"),
        ({"epoch": "3"}, "没有 epoch"),
        (["not", "a", "dict"], "没有 epoch"),
        (RuntimeError("truncated file"), "RuntimeError"),
    ],
)
def test_unresumable_checkpoint_starts_fresh_and_says_why(train, tmp_path, loaded, hint):
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x")
    decision = train.decide_resume(True, ckpt, _loader(loaded))
    assert decision.resume is False
    assert "从头训练" in decision.reason
    assert hint in decision.reason


# ---------------------------------------------------------------- train.py：参数和数据


def _dgpu_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        states.ENV_DATA: str(tmp_path / "data"),
        states.ENV_OUTPUT: str(tmp_path / "output"),
        states.ENV_CKPT: str(tmp_path / "ckpt"),
    }
    env.update(extra)
    return env


def test_read_settings_defaults_and_overrides(train, tmp_path):
    settings = train.read_settings(_dgpu_env(tmp_path))
    assert (settings.dataset, settings.model, settings.epochs, settings.imgsz) == (
        "coco8",
        "yolo26n.pt",
        100,
        640,
    )
    assert settings.resume_flag is False

    settings = train.read_settings(
        _dgpu_env(tmp_path, EPOCHS="3", IMGSZ="320", MODEL="yolo26s.pt", DGPU_RESUME="1")
    )
    assert (settings.model, settings.epochs, settings.imgsz) == ("yolo26s.pt", 3, 320)
    assert settings.resume_flag is True


def test_read_settings_reports_missing_dgpu_dirs(train):
    with pytest.raises(train.ExampleError) as excinfo:
        train.read_settings({states.ENV_DATA: "/data"})
    assert states.ENV_OUTPUT in str(excinfo.value)
    assert states.ENV_CKPT in str(excinfo.value)


@pytest.mark.parametrize("value", ["abc", "0", "-5", "1.5"])
def test_read_settings_rejects_bad_numbers(train, tmp_path, value):
    with pytest.raises(train.ExampleError, match="EPOCHS"):
        train.read_settings(_dgpu_env(tmp_path, EPOCHS=value))


def _make_dataset(tmp_path: Path, name: str = "coco8") -> Path:
    dataset = tmp_path / "data" / name
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        "path: coco8\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\ndownload: https://example.invalid/coco8.zip\n",
        encoding="utf-8",
    )
    return dataset


def test_prepare_data_yaml_points_to_local_copy_and_drops_download(train, tmp_path):
    dataset = _make_dataset(tmp_path)
    resolved = train.prepare_data_yaml(dataset)

    doc = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert doc["path"] == str(dataset.resolve())
    assert "download" not in doc
    assert doc["train"] == "images/train" and doc["names"] == {0: "person"}
    assert _no_tmp_files(tmp_path) == []


def test_prepare_data_yaml_without_data_yaml_is_a_clear_error(train, tmp_path):
    (tmp_path / "data" / "coco8").mkdir(parents=True)
    with pytest.raises(train.ExampleError, match="data.yaml"):
        train.prepare_data_yaml(tmp_path / "data" / "coco8")


# ---------------------------------------------------------------- train.py：main()


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """假的 ultralytics：记录调用；train() 模拟保存两轮 checkpoint 并触发回调。"""
    state = SimpleNamespace(created=[], fail=False)

    class FakeYOLO:
        def __init__(self, model):
            self.model = model
            self.callbacks = {}
            self.train_kwargs = None
            state.created.append(self)

        def add_callback(self, event, func):
            self.callbacks.setdefault(event, []).append(func)

        def train(self, **kwargs):
            self.train_kwargs = kwargs
            if state.fail:
                raise RuntimeError("CUDA out of memory")
            weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            last = weights / "last.pt"
            for epoch in range(2):
                last.write_bytes(f"epoch {epoch}".encode())
                (weights / "best.pt").write_bytes(f"best {epoch}".encode())
                for func in self.callbacks.get("on_model_save", []):
                    func(SimpleNamespace(last=last, epoch=epoch))

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    module.__version__ = "8.4.208"
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return state


def test_main_fresh_run_saves_ckpt_and_writes_done(train, tmp_path, fake_ultralytics):
    _make_dataset(tmp_path)
    env = _dgpu_env(tmp_path, EPOCHS="2")

    assert train.main(env) == 0

    (model,) = fake_ultralytics.created
    assert model.model == "yolo26n.pt"
    assert len(model.callbacks["on_model_save"]) == 1
    kwargs = model.train_kwargs
    assert kwargs["resume"] is False
    assert kwargs["epochs"] == 2 and kwargs["imgsz"] == 640
    assert kwargs["project"] == env[states.ENV_OUTPUT] and kwargs["exist_ok"] is True
    assert kwargs["data"].endswith(train.RESOLVED_YAML_NAME)

    assert (tmp_path / "ckpt" / "last.pt").read_bytes() == b"epoch 1"
    done = json.loads((tmp_path / "output" / "DONE").read_text(encoding="utf-8"))
    assert done["resumed"] is False and done["ultralytics"] == "8.4.208"
    assert _no_tmp_files(tmp_path) == []


def test_main_resumes_from_restored_checkpoint(train, tmp_path, fake_ultralytics, monkeypatch):
    _make_dataset(tmp_path)
    ckpt = tmp_path / "ckpt" / "last.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"restored")
    monkeypatch.setattr(
        train, "load_checkpoint", lambda path: {"epoch": 2, "train_args": {"epochs": 30}}
    )

    assert train.main(_dgpu_env(tmp_path, DGPU_RESUME="1")) == 0

    (model,) = fake_ultralytics.created
    assert model.model == str(ckpt)
    assert model.train_kwargs["resume"] is True
    done = json.loads((tmp_path / "output" / "DONE").read_text(encoding="utf-8"))
    assert done["resumed"] is True


def test_main_starts_fresh_from_finished_checkpoint(
    train, tmp_path, fake_ultralytics, monkeypatch, capsys
):
    _make_dataset(tmp_path)
    ckpt = tmp_path / "ckpt" / "last.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"stripped")
    monkeypatch.setattr(train, "load_checkpoint", lambda path: {"epoch": -1})

    assert train.main(_dgpu_env(tmp_path, DGPU_RESUME="1", MODEL="yolo26s.pt")) == 0

    (model,) = fake_ultralytics.created
    assert model.model == "yolo26s.pt"
    assert model.train_kwargs["resume"] is False
    assert "从头训练" in capsys.readouterr().out


def test_main_fails_without_data_yaml(train, tmp_path, fake_ultralytics, capsys):
    (tmp_path / "data" / "coco8").mkdir(parents=True)
    assert train.main(_dgpu_env(tmp_path)) == 2
    assert fake_ultralytics.created == []
    assert "data.yaml" in capsys.readouterr().err
    assert not (tmp_path / "output" / "DONE").exists()


def test_main_skips_training_when_done_exists(train, tmp_path, fake_ultralytics):
    _make_dataset(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "DONE").write_text("{}", encoding="utf-8")
    assert train.main(_dgpu_env(tmp_path)) == 0
    assert fake_ultralytics.created == []


def test_main_training_error_propagates_without_done(train, tmp_path, fake_ultralytics):
    _make_dataset(tmp_path)
    fake_ultralytics.fail = True
    with pytest.raises(RuntimeError):
        train.main(_dgpu_env(tmp_path))
    assert not (tmp_path / "output" / "DONE").exists()
    assert not (tmp_path / "ckpt" / train.COMPLETE_NAME).exists()


def test_main_records_completion_in_ckpt_before_done(train, tmp_path, fake_ultralytics):
    _make_dataset(tmp_path)
    assert train.main(_dgpu_env(tmp_path)) == 0
    ckpt = tmp_path / "ckpt"
    marker = json.loads((ckpt / train.COMPLETE_NAME).read_text(encoding="utf-8"))
    assert marker["final_weights"] == ["best.pt", "last.pt"]
    assert (ckpt / "final" / "best.pt").read_bytes() == b"best 1"
    assert (ckpt / "final" / "last.pt").read_bytes() == b"epoch 1"
    assert _no_tmp_files(tmp_path) == []


def test_retry_after_dying_between_training_and_done_does_not_retrain(
    train, tmp_path, fake_ultralytics, monkeypatch
):
    """机器在训练完之后、写 DONE 之前宕机：$DGPU_CKPT 恢复了（含完成标记），$DGPU_OUTPUT 没有。
    重试不能从头训练（精简过的 last.pt 不能续跑），而是把权重放回输出目录再写 DONE。"""
    _make_dataset(tmp_path)
    real_write_done = train.write_done

    def machine_dies(output_dir, info):
        raise SystemExit("host lost")

    monkeypatch.setattr(train, "write_done", machine_dies)
    with pytest.raises(SystemExit):
        train.main(_dgpu_env(tmp_path))
    assert (tmp_path / "ckpt" / train.COMPLETE_NAME).exists()

    # 换机重试：ckpt/ 从存储恢复，output/ 是空的，训练结束后的 last.pt 不能续跑
    shutil.rmtree(tmp_path / "output")
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(train, "write_done", real_write_done)
    monkeypatch.setattr(train, "load_checkpoint", lambda path: {"epoch": -1})
    fake_ultralytics.created.clear()

    assert train.main(_dgpu_env(tmp_path, DGPU_RESUME="1", DGPU_ATTEMPT="2")) == 0
    assert fake_ultralytics.created == []  # 没有重新训练
    weights = tmp_path / "output" / "run" / "weights"
    assert (weights / "best.pt").read_bytes() == b"best 1"
    assert (weights / "last.pt").read_bytes() == b"epoch 1"
    done = json.loads((tmp_path / "output" / "DONE").read_text(encoding="utf-8"))
    assert done["recovered"] is True and done["restored"] == ["best.pt", "last.pt"]
    assert _no_tmp_files(tmp_path) == []


def test_unreadable_completion_marker_is_ignored(train, tmp_path, fake_ultralytics):
    _make_dataset(tmp_path)
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / train.COMPLETE_NAME).write_text("{half", encoding="utf-8")
    assert train.main(_dgpu_env(tmp_path)) == 0
    assert len(fake_ultralytics.created) == 1  # 照常训练


# ---------------------------------------------------------------- 文档：命令和参数

_INVOCATION = re.compile(r"(?<![\w/.\-])dgpu((?:[ \t]+[^\s`|]+)*)")
_COMMAND_WORD = re.compile(r"[a-z][a-z-]*")


def _invocations(text: str) -> list[tuple[str, list[str]]]:
    """找出文档里的 `dgpu <命令> [参数]`，返回 (命令, 参数列表)。

    跳过 `dgpu` 后面不是命令单词的情况（中文叙述、`<命令>` 之类的占位符）。
    """
    found = []
    for match in _INVOCATION.finditer(text):
        tokens = match.group(1).split()
        if "#" in tokens:
            tokens = tokens[: tokens.index("#")]
        flags = []
        index = 0
        while index < len(tokens) and tokens[index].startswith("-"):
            flags.append(tokens[index])
            index += 2 if tokens[index] == "--config" else 1
        if index >= len(tokens) or not _COMMAND_WORD.fullmatch(tokens[index]):
            continue
        flags += [token for token in tokens[index + 1 :] if token.startswith("-")]
        found.append((tokens[index], flags))
    return found


def test_invocation_parser_finds_commands_and_flags():
    text = (
        "运行 `dgpu wait <job> --retries 1`，然后 dgpu gc --dry-run   # 只列出\n"
        "`dgpu --config ~/x.toml doctor`；dgpu 还在开发中；~/.config/dgpu/config.toml；"
        "name: dgpu\ndescription: x；`dgpu-<任务ID>-a1`"
    )
    assert _invocations(text) == [
        ("wait", ["--retries"]),
        ("gc", ["--dry-run"]),
        ("doctor", ["--config"]),
    ]


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_docs_only_use_commands_and_flags_from_the_design(doc):
    invocations = _invocations(doc.read_text(encoding="utf-8"))
    assert invocations, f"{doc} 里没有找到 dgpu 命令"
    for command, flags in invocations:
        assert command in ALLOWED_COMMANDS, f"{doc.name}: 设计里没有 dgpu {command}"
        for flag in flags:
            assert flag in ALLOWED_FLAGS, f"{doc.name}: dgpu {command} 没有参数 {flag}"


def test_skill_covers_the_whole_flow():
    commands = {command for command, _ in _invocations(SKILL_MD.read_text(encoding="utf-8"))}
    assert commands == ALLOWED_COMMANDS


def test_usage_documents_every_command():
    commands = {command for command, _ in _invocations(USAGE_MD.read_text(encoding="utf-8"))}
    assert commands == ALLOWED_COMMANDS


# ---------------------------------------------------------------- 文档：与 states.py 一致


def test_skill_frontmatter_and_length():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n", 2)[1])
    assert front["name"] == "dgpu"
    description = front["description"]
    for word in ("Vast", "AutoDL", "job.yaml", "R2"):
        assert word in description
    assert len(text.splitlines()) <= 220


@pytest.mark.parametrize("doc", (SKILL_MD, USAGE_MD), ids=lambda path: path.name)
def test_docs_list_wait_exit_codes_from_states(doc):
    text = doc.read_text(encoding="utf-8")
    for phase, code in states.EXIT_CODE_BY_PHASE.items():
        row = re.compile(rf"^\|\s*{code}\s*\|\s*{phase}\s*\|", re.MULTILINE)
        assert row.search(text), f"{doc.name} 缺少退出码 {code} = {phase}"
    assert re.search(rf"^\|\s*{states.EXIT_USAGE}\s*\|", text, re.MULTILINE)


@pytest.mark.parametrize("path", (*DOCS, TRAIN_PY), ids=lambda path: path.name)
def test_dgpu_env_names_match_states(path):
    used = set(re.findall(r"\bDGPU_[A-Z_]+\b", path.read_text(encoding="utf-8")))
    assert used <= ENV_NAMES | {contracts.ENV_CONFIG_PATH}, used - ENV_NAMES


def _first_column_after(text: str, marker: str) -> set[str]:
    """marker 之后第一张表格第一列里用反引号括起来的值。"""
    lines = text[text.index(marker) :].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    values = set()
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        match = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if match:
            values.add(match.group(1))
    return values


def test_skill_explains_every_env_var_and_failure_reason():
    text = SKILL_MD.read_text(encoding="utf-8")
    for name in ENV_NAMES:
        assert name in text
    assert _first_column_after(text, "FAILED 的原因") == REASONS


def test_usage_reasons_and_phases_match_states():
    text = USAGE_MD.read_text(encoding="utf-8")
    reasons = _first_column_after(text, "常见的失败原因")
    assert reasons and reasons <= REASONS
    for phase in ALL_PHASES:
        assert f"`{phase}`" in text


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_relative_links_resolve(doc):
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (doc.parent / target.split("#", 1)[0]).exists(), f"{doc.name}: 链接 {target} 不存在"
