"""dgpu 的 YOLO 示例：在租来的 GPU 上训练 Ultralytics YOLO，机器中断后能接着跑。

dgpu 在机器上设置这些环境变量：
  DGPU_DATA    数据目录，数据集在 $DGPU_DATA/<DATASET>
  DGPU_OUTPUT  输出目录，任务结束后用 `dgpu pull` 取回
  DGPU_CKPT    checkpoint 目录，换机续跑时会恢复
  DGPU_RESUME  DGPU_CKPT 里有恢复出来的文件时为 1

本脚本自己的参数，在 job.yaml 的 env 里填，都有默认值：
  DATASET  数据集目录名，默认 coco8，目录里要有 data.yaml
  MODEL    初始权重，默认 yolo26n.pt
  EPOCHS   训练轮数，默认 100
  IMGSZ    输入尺寸，默认 640

ultralytics 只在 main() 里 import，没装 ultralytics 也能导入本文件、测试辅助函数。
要兼容 Python 3.8（AutoDL 的公共镜像比较旧）。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CKPT_NAME = "last.pt"  # 放在 $DGPU_CKPT 里的 checkpoint 文件名
DONE_NAME = "DONE"  # 训练完成后写在 $DGPU_OUTPUT 里的标记
RUN_NAME = "run"  # Ultralytics 的输出在 $DGPU_OUTPUT/run/
DATA_YAML_NAME = "data.yaml"  # 数据集目录里的 YOLO 数据描述
RESOLVED_YAML_NAME = "data.dgpu.yaml"  # 改写成绝对路径后的副本

DEFAULTS = {"DATASET": "coco8", "MODEL": "yolo26n.pt", "EPOCHS": "100", "IMGSZ": "640"}


class ExampleError(Exception):
    """配置或数据不对。main() 打印说明后以退出码 2 结束。"""


def log(message: str) -> None:
    print(f"[train.py] {message}", flush=True)


# ---------------------------------------------------------------- 参数


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    ckpt_dir: Path
    resume_flag: bool
    dataset: str
    model: str
    epochs: int
    imgsz: int


def _positive_int(env: Mapping[str, str], name: str) -> int:
    text = env.get(name) or DEFAULTS[name]
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise ExampleError(f"{name} 必须是正整数，现在是 {text!r}")
    return value


def read_settings(env: Mapping[str, str]) -> Settings:
    """从环境变量读参数。缺少 dgpu 的目录变量时报错，不猜默认位置。"""
    missing = [name for name in ("DGPU_DATA", "DGPU_OUTPUT", "DGPU_CKPT") if not env.get(name)]
    if missing:
        names = "、".join(missing)
        raise ExampleError(
            f"没有设置 {names}。这个脚本由 dgpu 在 GPU 机器上运行；"
            "在本机试跑时要自己设置，见 examples/yolo/README.md"
        )
    return Settings(
        data_dir=Path(env["DGPU_DATA"]),
        output_dir=Path(env["DGPU_OUTPUT"]),
        ckpt_dir=Path(env["DGPU_CKPT"]),
        resume_flag=env.get("DGPU_RESUME") == "1",
        dataset=env.get("DATASET") or DEFAULTS["DATASET"],
        model=env.get("MODEL") or DEFAULTS["MODEL"],
        epochs=_positive_int(env, "EPOCHS"),
        imgsz=_positive_int(env, "IMGSZ"),
    )


# ---------------------------------------------------------------- checkpoint


def atomic_copy(src: Path, dst: Path) -> None:
    """把 src 完整复制到 dst：先写 dst.tmp，刷盘，再 rename 成 dst。

    dgpu 同步 checkpoint 目录时跳过 *.tmp，所以存储里只会出现完整的文件。
    中途失败时删掉临时文件，dst 保持原样，异常照常抛出。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copyfile(str(src), str(tmp))
        with open(str(tmp), "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(dst))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_ckpt_callback(ckpt_dir: Path) -> Callable[[Any], None]:
    """返回 Ultralytics 的 on_model_save 回调：每次保存后把 last.pt 复制进 $DGPU_CKPT。

    Ultralytics 写 last.pt 不是原子操作，后台同步可能读到半个文件，所以不直接同步它，
    而是在它写完之后（这个回调触发时）复制一份完整的。
    """
    target = ckpt_dir / CKPT_NAME

    def on_model_save(trainer: Any) -> None:
        atomic_copy(Path(trainer.last), target)
        epoch = getattr(trainer, "epoch", None)
        log(f"checkpoint 已保存到 {target}（epoch {epoch}）")

    return on_model_save


def load_checkpoint(path: Path) -> Any:
    """读 Ultralytics checkpoint。只在 GPU 机器上调用，测试里替换掉。

    weights_only=False 是因为 Ultralytics 的 checkpoint 里有 pickle 的模型对象；
    只读自己写的 checkpoint。
    """
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class ResumeDecision:
    resume: bool
    reason: str  # 打印到日志，说明为什么续跑或从头开始


def _total_epochs(ckpt: Mapping[str, Any]) -> int | None:
    args = ckpt.get("train_args")
    if isinstance(args, Mapping):
        total = args.get("epochs")
        if isinstance(total, int) and not isinstance(total, bool):
            return total
    return None


def decide_resume(
    resume_flag: bool, ckpt_path: Path, load: Callable[[Path], Any]
) -> ResumeDecision:
    """决定续跑还是从头训练。任何拿不准的情况都从头训练，并说明原因。

    能续跑的条件：DGPU_RESUME=1、checkpoint 存在且读得出来、epoch >= 0、还有没训练的轮次。
    训练正常结束后 Ultralytics 会把 last.pt 精简成 epoch=-1，这种文件不能续跑。
    """
    if not resume_flag:
        return ResumeDecision(False, "DGPU_RESUME 不是 1，从头训练")
    if not ckpt_path.is_file():
        return ResumeDecision(False, f"DGPU_RESUME=1，但没有 {ckpt_path}，从头训练")
    try:
        ckpt = load(ckpt_path)
    except Exception as exc:
        return ResumeDecision(
            False, f"{ckpt_path} 读不出来（{type(exc).__name__}: {exc}），从头训练"
        )
    epoch = ckpt.get("epoch") if isinstance(ckpt, Mapping) else None
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        return ResumeDecision(False, f"{ckpt_path} 里没有 epoch，不是训练中的 checkpoint，从头训练")
    if epoch < 0:
        return ResumeDecision(
            False, f"checkpoint 的 epoch={epoch}，是训练结束后精简过的权重，不能续跑，从头训练"
        )
    total = _total_epochs(ckpt)
    if total is not None and epoch + 1 >= total:
        return ResumeDecision(
            False, f"checkpoint 显示 {total} 轮已经全部训练完，Ultralytics 不能续跑，从头训练"
        )
    done = f"已完成 {epoch + 1} 轮"
    if total is not None:
        done += f"，共 {total} 轮"
    return ResumeDecision(True, "从 checkpoint 续跑：" + done)


# ---------------------------------------------------------------- 数据和完成标记


def prepare_data_yaml(dataset_dir: Path) -> Path:
    """读 <数据集>/data.yaml，把 path 改成数据集的绝对路径、去掉 download，另存一份。

    Ultralytics 把相对的 path 解释成它自己的 datasets 目录；download 会在找不到数据时
    自动上网下载，掩盖数据没拷过来的问题。
    """
    import yaml  # ultralytics 依赖 pyyaml

    src = dataset_dir / DATA_YAML_NAME
    if not src.is_file():
        raise ExampleError(
            f"找不到 {src}：数据集目录里要有 data.yaml，写法见 examples/yolo/README.md"
        )
    with open(str(src), encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    if not isinstance(doc, dict):
        raise ExampleError(f"{src} 不是 YAML 映射")
    doc["path"] = str(dataset_dir.resolve())
    doc.pop("download", None)
    dst = dataset_dir / RESOLVED_YAML_NAME
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(str(tmp), str(dst))
    return dst


def write_done(output_dir: Path, info: Mapping[str, Any]) -> Path:
    """在输出目录写 DONE 标记（先写 .tmp 再 rename）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / DONE_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(info), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))
    return path


# ---------------------------------------------------------------- 主流程


def main(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    try:
        settings = read_settings(env)
        done = settings.output_dir / DONE_NAME
        if done.exists():
            log(f"{done} 已存在：这个任务已经训练完，不再重复训练")
            return 0
        data_yaml = prepare_data_yaml(settings.data_dir / settings.dataset)
    except ExampleError as exc:
        print(f"[train.py] 错误：{exc}", file=sys.stderr, flush=True)
        return 2

    ckpt_path = settings.ckpt_dir / CKPT_NAME
    decision = decide_resume(settings.resume_flag, ckpt_path, load_checkpoint)
    log(decision.reason)

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(ckpt_path) if decision.resume else settings.model)
    model.add_callback("on_model_save", make_ckpt_callback(settings.ckpt_dir))
    model.train(
        data=str(data_yaml),
        epochs=settings.epochs,
        imgsz=settings.imgsz,
        project=str(settings.output_dir),
        name=RUN_NAME,
        exist_ok=True,
        resume=decision.resume,
    )

    write_done(
        settings.output_dir,
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": settings.model,
            "epochs": settings.epochs,
            "resumed": decision.resume,
            "ultralytics": getattr(ultralytics, "__version__", "unknown"),
            "weights": f"{RUN_NAME}/weights/best.pt",
        },
    )
    log(f"训练完成，结果在 {settings.output_dir / RUN_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
