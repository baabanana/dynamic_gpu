# YOLO 示例：在租来的 GPU 上训练 Ultralytics YOLO

这个例子租一张 RTX 3090 或 4090，用 YOLO26n 在 coco8（8 张图的小数据集）上训练 30 轮，把结果传回 R2，然后销毁机器。它用来确认整条链路是通的，也可以当作你自己 YOLO 任务的模板。

**尚未在真实平台验证。** 下面的命令、耗时和费用都是按设计写的估计。

| 文件 | 作用 |
|---|---|
| `job.yaml` | 任务描述：镜像、数据、GPU、费用上限 |
| `train.py` | 训练脚本：读 dgpu 的环境变量，保存 checkpoint，中断后接着跑 |

## 1. 准备

1. 装好 dgpu，`dgpu doctor` 全部通过。见 [使用说明](../../docs/usage.md)。
2. 下载 coco8，放到 `datasets/coco8`，再放一份 `data.yaml`：

   ```bash
   cd examples/yolo
   mkdir -p datasets
   curl -L -o /tmp/coco8.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip
   unzip -q /tmp/coco8.zip -d datasets/          # 得到 datasets/coco8/images、datasets/coco8/labels
   curl -L -o datasets/coco8/data.yaml \
     https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.208/ultralytics/cfg/datasets/coco8.yaml
   ```

   `data.yaml` 里的 `path` 不用改：`train.py` 运行时会把它换成机器上的实际路径，并去掉 `download`，免得数据没拷过来时 Ultralytics 自己上网下载、掩盖问题。

   `job.yaml` 的 `data` 引用了 `./datasets/coco8`：提交时它作为数据集单独上传到 R2（内容没变就不重传），不会再打进代码包。不要把 `datasets/` 提交到 git。

## 2. 运行

```bash
cd examples/yolo
dgpu offers job.yaml                  # 看有哪些机器、每小时多少钱，不花钱
dgpu submit job.yaml                  # 租机器、启动训练，输出任务 ID，如 yolo-coco8-261110083000-a1b2
dgpu wait <任务ID>                    # 等到结束；退出码 0 表示成功
dgpu pull <任务ID> ./result           # 下载输出到 ./result/output/
dgpu gc --dry-run                     # 确认没有遗留的机器在计费
```

等待期间可以看进度：`dgpu status <任务ID>`、`dgpu logs <任务ID> -f`。

拿到的结果：

```
result/output/
  DONE                     训练完成标记（JSON：轮数、是否续跑、Ultralytics 版本）
  run/weights/best.pt      最好的权重
  run/weights/last.pt      最后一轮的权重
  run/results.csv          每轮的指标
```

### 用 AutoDL

1. 在 `job.yaml` 里把 `image.autodl` 换成你选的 AutoDL 镜像 UUID。AutoDL 只能用平台上的镜像；公共镜像比较旧（例如 CUDA 11.8、Python 3.8、PyTorch 2.0），能不能跑这个例子尚未验证。不行的话，在 AutoDL 网站上装好新版 PyTorch 后保存成私有镜像。
2. 提交时加 `--backend autodl`：

   ```bash
   dgpu offers job.yaml --backend autodl
   dgpu submit job.yaml --backend autodl
   ```

AutoDL 到 R2 是跨境网络，速度尚未实测。

## 3. 费用

估算，尚未在真实平台验证：

| | Vast | AutoDL |
|---|---|---|
| 价格 | 3090 约 $0.12/小时，4090 约 $0.32/小时 | 3090 约 ¥1.32/小时，4090 约 ¥1.88/小时 |
| 计费时长 | 拉镜像、装 ultralytics、训练，合计约 10–30 分钟 | 同左 |
| 预计花费 | 几美分到十几美分 | 一元左右 |
| 最坏花费 | $0.40（`max_usd_per_hour` × `max_hours`）加少量流量费 | ¥2.50 |

价格来自 2026-11-09 的调研，会变动。以 `dgpu offers` 的报价为准。

## 4. 中断后接着跑

`train.py` 这样处理 checkpoint：

- Ultralytics 每轮结束保存 `last.pt` 后，`on_model_save` 回调把它复制到 `$DGPU_CKPT/last.pt`：先写 `last.pt.tmp`，再 rename。Ultralytics 自己写 `last.pt` 不是原子操作，直接上传可能传上去半个文件。
- 换机重跑时（`dgpu retry <任务ID>`），dgpu 把 checkpoint 拷回 `$DGPU_CKPT` 并设置 `DGPU_RESUME=1`。`train.py` 先读这个文件，确认它能续跑（`epoch >= 0`，并且还有没训练的轮次），再用 `resume=True` 接着训练。
- 读不出来，或者是训练结束后精简过的权重（Ultralytics 会把它的 `epoch` 改成 -1），就从头训练，并在日志里写明原因。
- 训练完成后，先把最终的 `best.pt`、`last.pt` 复制到 `$DGPU_CKPT/final/`，再写完成标记 `$DGPU_CKPT/COMPLETE.json`（都是先写 `.tmp` 再 rename），最后写 `$DGPU_OUTPUT/DONE`。同一台机器上再次运行时，看到 `DONE` 就直接退出。
- 机器在训练完之后、`DONE` 和输出传完之前宕机：重试时 `$DGPU_CKPT` 会恢复（`$DGPU_OUTPUT` 不恢复），`train.py` 看到完成标记就不再训练，把 `final/` 里的权重放回 `$DGPU_OUTPUT/run/weights/` 再写 `DONE`（记 `recovered: true`），由 dgpu 重新上传。完成标记要等下一次同步（约 2 分钟）才到 R2，宕机发生在这之前时，重试还是会从头训练。代价是 checkpoint 里多存一份最终权重。

续跑必须用同一个 Ultralytics 版本，所以 `setup` 里固定了 `ultralytics==8.4.208`。

## 5. 参数

在 `job.yaml` 的 `env` 里改：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `DATASET` | `coco8` | `$DGPU_DATA` 下的数据集目录名，目录里要有 `data.yaml` |
| `MODEL` | `yolo26n.pt` | 初始权重，第一次运行时 Ultralytics 会自动下载 |
| `EPOCHS` | `100`（`job.yaml` 里设成 30） | 训练轮数 |
| `IMGSZ` | `640` | 输入尺寸 |

## 6. 换成你自己的数据

1. 数据集按 YOLO 格式组织：`images/train`、`images/val`、`labels/train`、`labels/val`，加一个 `data.yaml`（写 `train`、`val`、`names`）。
2. 上传到 R2 一次（例如 `rclone copy ./my-set r2:<你的 bucket>/datasets/my-set`），然后在 `job.yaml` 里写：

   ```yaml
   data:
     - r2://<你的 bucket>/datasets/my-set
   env:
     DATASET: my-set
   ```

   也可以写本地路径（如 `/data/my-set`），dgpu 提交时会帮你传到 R2，内容没变就不重传。单个文件超过 5 GiB 时只能用上面 rclone 的办法。目录名（`my-set`）只能用字母、数字、点、下划线和连字符。
3. 按数据量调大 `max_hours`，先用 `dgpu offers` 看报价。

## 7. 本机试跑

在本机用 CPU 跑 1 轮，确认脚本和数据没问题（需要本机装 `ultralytics==8.4.208`）：

```bash
cd examples/yolo
DGPU_DATA=$PWD/datasets DGPU_OUTPUT=/tmp/yolo-smoke/output DGPU_CKPT=/tmp/yolo-smoke/ckpt \
  EPOCHS=1 IMGSZ=320 python train.py
```

输出放在代码目录之外，免得下次提交时被一起打包。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| 日志里有 `libGL.so.1: cannot open shared object file` | 镜像缺 OpenCV 需要的系统库。训练默认不以 root 运行，`setup` 里不能 `apt-get install`。把 `setup` 改成 `python -m pip install ultralytics==8.4.208 opencv-python-headless`；确实要装系统包时在 `job.yaml` 里加 `run_as_root: true`（训练代码因此能读到 R2 密钥） |
| 报 `找不到 .../data.yaml` | 数据集目录里没有 `data.yaml`，或 `DATASET` 和目录名不一致 |
| AutoDL 上下载 `yolo26n.pt` 失败 | 权重从 GitHub 下载，AutoDL 上可能很慢。先在本机下载 `yolo26n.pt` 放进代码目录 |
| 续跑后日志写“从头训练” | 日志里写了原因：没有 checkpoint、读不出来，或 checkpoint 已经训练完 |

## 9. 许可证

Ultralytics 的代码和模型采用 AGPL-3.0 或企业许可。Ultralytics 的官方立场是：用它训练出来的模型默认也适用 AGPL-3.0；用在商业产品、对外服务或公司内部业务工具里，需要购买企业许可，除非整个项目按 AGPL-3.0 开源。这是 Ultralytics 自己的说法，不是法律意见。打算商用前请自行确认，见 <https://ultralytics.com/license>。
