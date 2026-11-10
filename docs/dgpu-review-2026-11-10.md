# dgpu 对抗式评审问题清单（2026-11-10）

状态：待修复（两份评审已完成，主会话已重跑复现脚本核实；修复在集成分支合入后进行）
关联：[设计文档](dgpu-design.md)、[实施方案](dgpu-implementation-plan.md)

评审对象是提交 `752978a`（7 个工作包合并后、集成之前）。所有问题都在本地模拟中发现，没有连接过真实平台。
“核实”一列：`复现` 表示主会话重跑了评审的复现脚本并得到相同结果；`读代码` 表示只有推理，没有复现。

修完之前不连接真实平台。每条修复都要配回归测试，并在本表回填状态。

## A. 编排、CLI、后端、存储

| 编号 | 级别 | 问题 | 位置 | 核实 | 修复方向 | 状态 |
|---|---|---|---|---|---|---|
| A1 | 严重 | `submit` / `retry` 启动阶段收到 SIGTERM 或 SIGHUP 时不做任何清理，实例留着计费；之后 `gc --dry-run` 还显示“在启动窗口内，保留” | `cli.py`（无信号处理）、`orchestrator.py` `_abort_launch` | 复现 | `main()` 把 SIGTERM、SIGHUP 转成和 Ctrl-C 相同的中止路径 | 待修 |
| A2 | 严重 | 创建请求未返回时按 Ctrl-C：只查一次标签就认定没创建，状态写“实例已销毁”，启动文档被删，实例随后出现且不会自毁，也没有提示运行 `gc` | `orchestrator.py` `_abort_launch` | 复现 | 结果未知时按后端的延迟重复查询；查不到就如实写“创建结果未知”，提示 `gc` | 待修 |
| A3 | 高 | `wait --retries` 在 LOST 时不看取消标记，`retry` 的 bump 还把 `cancel` 清掉，导致取消后又开新实例 | `orchestrator.py` `wait`、`retry` | 复现 | 有取消请求时 `wait` 不重试；`retry` 遇到未完成的取消时拒绝 | 待修 |
| A4 | 高 | `gc` 的孤儿规则：存储里没有任务目录且实例超过 1 小时就销毁。另一份配置（另一个桶）或共用平台账户的同事的任务会被删；手工命名成 `dgpu-x-a1` 的实例也会被删 | `orchestrator.py` `_gc_decide` | 复现 | 孤儿只报告，不自动销毁（显式参数才销毁）；`gc` 只认 `new_job_id` 形状的任务 ID | 待修 |
| A5 | 高 | 创建结果不明且标签查询为空后，换下一个 offer 时沿用同一个 attempt 和标签。先前的实例如果晚到，就有两个 runner 写同一个 attempt，`gc` 也分辨不出 | `orchestrator.py` `_try_offer` | 复现 | 只要创建结果不是明确的“被拒绝”，下一次创建前都 bump attempt | 待修 |
| A6 | 中 | 按旧判断销毁所有 attempt：`wait` 的收尾会销毁并发 `retry` 刚启动的实例；两个 `retry` 并发时输的一方销毁赢的一方的实例 | `orchestrator.py` `_cleanup_after_end`、`_destroy_job_instances` | 复现 | 销毁时带上所依据的 attempt，只销毁不大于它的实例；`attempt.json` 已前进则跳过收尾 | 待修 |
| A7 | 中 | `wait` 收尾清理遗留实例时吞掉平台错误，返回 SUCCEEDED 且无提示 | `orchestrator.py` `_cleanup_after_end` | 复现 | 清理失败时明确提示并指向 `gc`；`store.delete` 放进同一处理 | 待修 |
| A8 | 中 | 本地数据大小估算不跟随软链接，上传却跟随：5 GB 确认、磁盘大小计算、AutoDL 50 GB 检查都被绕过 | `orchestrator.py` 数据大小估算、`packaging._walk_dataset` | 复现 | 大小取自 `hash_dataset` 的清单，并复用该清单上传 | 待修 |
| A9 | 中 | LOST 和 `gc` 的过期判断比较的是本机时钟和实例时钟；10 分钟阈值与“运行中存储断连继续训练”“最终同步可重试 20 分钟”矛盾，`wait --retries` 会销毁仍在训练或上传的实例 | `orchestrator.py` `_view`、`_gc_decide` | 读代码 | 用存储的 `LastModified` 计算过期（`Store` 契约要加）；自动重试前要求平台也显示实例已死，或阈值大于最终同步时限 | 待修 |
| A10 | 低 | 打包时跳过的文件（指向目录外的软链接等）不提示 | `orchestrator.py`（丢弃 `PackResult`） | 读代码 | 打印 `skipped` | 待修 |
| A11 | 低 | 费用上限只算 `价格 × max_hours`，没算收尾宽限、失败的 offer 和重试 | `orchestrator.py` | 读代码 | 按截止时间公式和 `max_offers` 计算并说明 | 待修 |
| A12 | 低 | `retry` 在新 attempt 启动即失败时仍退出 0；`wait` 把存储瞬时错误变成退出码 2（文档里是“用法错误”） | `cli.py` | 读代码 | 修正退出码；`wait` 对瞬时存储错误重试 | 待修 |

需要真实平台才能确认的怀疑（并入探针）：

- 刚创建的实例短时间内查不到时会被当成已销毁（Vast 404、AutoDL 列表里没有）。建议创建后约 60 秒内忽略 GONE，销毁时总是发送 DELETE。
- 标签长度或字符集被平台截断、改写。创建后回读标签。
- AutoDL Job 在 `cmd` 非 0 退出时是否重启容器（探针 A1）。
- AutoDL 容器列表分页从 0 还是从 1 开始。
- Vast 搜索到创建之间价格变化。创建后比较实际价格，超过上限就销毁。

## B. 容器内 runner、watchdog、bundle

| 编号 | 级别 | 问题 | 位置 | 核实 | 修复方向 | 状态 |
|---|---|---|---|---|---|---|
| B1 | 严重 | `not_uploaded` 只比较文件大小：checkpoint 被等大小改写后，最终上传失败时报告“0 个文件没有传上去”，存储里是旧文件 | `runner/main.py` | 复现 | 每次成功同步记录（路径、大小、mtime_ns），之后变化的都算未上传；拿不准时报“未知”而不是 0 | 待修 |
| B2 | 高 | 最终同步对 output 和 ckpt 各用了整个时限，总时长可达设定的 2 倍；watchdog 硬截止只留“时限 + 10 分钟”，会杀掉正在上传的 runner，把成功的训练记成 TIMED_OUT，且不上传结果 | `runner/main.py`、`runner/watchdog.py`、`runner/util.py` | 复现 | 每个目标只用剩余时间；watchdog 余量由 runner 各收尾阶段的上限之和推出；硬截止时做一次有界的同步并计算 `not_uploaded` | 集成分支处理中（第 9 项） |
| B3 | 高 | watchdog 上传日志没有超时，存储卡住时自毁被阻塞；处理函数里抛异常会跳过自毁 | `runner/watchdog.py` | 复现 | watchdog 的每个存储调用都带超时，有总预算；自毁放进 `finally` | 集成分支处理中 |
| B4 | 高 | 训练和 `setup` 与 runner、watchdog 同一用户：能读到 `launch.json`、`/proc/*/environ` 里的凭据；`pkill -9 python` 会同时杀掉 runner 和 watchdog，实例无人销毁 | `runner/main.py`、`runner/boot.py` | 复现 | 容器内是 root 时，训练进程降权到无特权用户运行（`preexec_fn` 里 setgid/setuid），`run/` 和 `launch.json` 仅 root 可读；不是 root 时退回现状并记日志。文档不再声称凭据对训练进程不可见 | 待修（设计变更，需写进设计文档） |
| B5 | 高 | Vast 实例重启后已超过 `max_hours`（或停机期间被取消）时，本地结果不上传：是否做最终同步取决于只存在内存里的标志 | `runner/main.py` | 复现 | 按磁盘上的状态（output/ckpt 非空或持久化标记）决定 | 待修 |
| B6 | 高 | `nvidia-smi` 卡死（D 状态，SIGKILL 无效）时 `subprocess.run(timeout=)` 的 `wait` 永不返回，runner 主线程卡住而心跳照常，直到 watchdog 硬截止 | `runner/gpu.py`、`runner/main.py` | 读代码（macOS 无法复现） | 用 Popen 加截止时间轮询，超时后放弃该进程不再等待，按 GPU 故障处理；SIGKILL 之后的 wait 都要有上限 | 待修 |
| B7 | 中 | 代码下载失败时，最终同步对不存在的目录重试满 20 分钟才销毁，并写出多余的 `final_sync_failed` | `runner/main.py` | 复现 | 提前创建目录；跳过本地不存在的目标 | 待修 |
| B8 | 中 | 最终同步期间实例重启会重跑 `setup` 和整个训练 | `runner/main.py` | 复现 | 结果一确定就落盘（带“待最终同步”标记），重启后从最终同步继续 | 待修 |
| B9 | 中 | 日志超过上限后，失败原因、退出码、最终状态也被丢弃；runner 自身异常的 traceback 只在不上传的 `runner.log` 里 | `runner/logsink.py`、`runner/main.py` | 复现 | 保留训练输出尾部的环形缓冲；给 runner 的说明留空间；上传 `runner.log`、`watchdog.log`、`bootstrap.log` 的尾部 | 待修 |
| B10 | 中 | AutoDL 前台的 boot 自己没有截止时间：watchdog 死掉且 runner 卡住时，`cmd` 永不返回 | `runner/boot.py` | 复现 | boot 自己按 `spec.json` 和 `started.json` 执行硬截止，并重启死掉的 watchdog | 待修 |
| B11 | 低 | 周期同步线程没有真正停止，收尾时与最终同步抢带宽 | `runner/main.py` | 复现 | 目标之间检查停止标志 | 待修 |
| B12 | 低 | runner 被杀后，它启动的 rclone 进程还活着，可能在 watchdog 写状态之后落盘 | `runner/rclone.py` | 复现 | 记录 rclone 的进程组，watchdog 一并杀掉 | 待修 |

只有推理、未复现的问题：

- **AutoDL 退出码**（取决于探针 A1）：boot 把 runner 的退出码作为 `cmd` 的退出码。如果 Job 会重启非 0 退出的容器，失败的任务会反复重启。探针确认前，AutoDL 前台模式统一退出 0，结果以 `status.json` 为准。
- **flock 错误被当成“已有 runner 在运行”**：不支持 flock 的文件系统上什么都不会启动。只有 EAGAIN 才表示被占用。
- **fencing 窗口**：同步线程不重读 `attempt.json`；最后一次读取之后状态写入还能重试 300 秒；watchdog 硬截止路径只看本地标记。
- **卡死检测**：也作用于 `setup`（安静的下载或解压会误判）；单个 GPU 报 N/A 会让所有 GPU 的检测失效。
- **事件同步刷新**：停止训练前和日志截断回调里各有最长 120 秒的同步等待。
- **`bash -lc`**：会执行镜像的登录配置，可能重置 PATH，或重新导出已清除的变量。
- **bootstrap 诊断信息太少**：只记录异常类型，日志也不上传。

## C. 评审指出的测试缺口

- 代码下载失败；等大小的旧文件；最终同步的总时长；存储卡住时的 watchdog 硬截止路径。
- 截止时间之后或取消之后的重启；最终同步期间的重启。
- `nvidia-smi` 超时；日志截断后失败原因是否保留；flock 失败；bootstrap 的 HTTPS 或证书失败。
- watchdog 截止测试用 SIGSTOP 冻结了 runner，没有覆盖“runner 活着但收尾慢”的真实情况。
- `procs.py` 的 `/proc` 分支在 macOS 上从未执行。
- 凭据测试只断言环境变量被清除，没有验证训练进程实际读不到。

## D. 评审认为没有问题的部分

- 每条退出路径都会到达自毁；信号处理只置标志；fencing 的检查和启动在同一把锁下。
- 凭据只经环境变量传给 rclone，不出现在命令行；bundle 和 rclone 都固定了哈希；tar 解压拒绝越界路径。
- 标签解析无歧义，标签查询是精确匹配；两个后端的 `list` 和 `destroy` 都强制 `dgpu-` 前缀。
- Vast 把无效 key 和“不存在”区分开；`create` 不盲目重试；条件写的 ETag 处理正确。
- 平台 key 只在请求头里，输出和事件都做了打码；配置文件强制 600 权限。
