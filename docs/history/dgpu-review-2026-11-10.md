# dgpu 对抗式评审问题清单（2026-11-10）

> **开发过程记录（归档）。** 本文记录写作时的计划或评审状态，不随代码更新；当前行为以[设计文档](../dgpu-design.md)和[使用说明](../usage.md)为准。

状态：两轮都已在本地修复（A、B 两节在 `feature/dgpu-mvp` 上修复；第二轮复核的 F1–F9 见 E 节，在 `wp/fix-round2` 上修复并配回归测试）；均未在真实平台验证，F7 和 F1 的容器 key 自毁要靠真实实例确认
关联：[设计文档](../dgpu-design.md)、[实施方案](dgpu-implementation-plan.md)

评审对象是提交 `752978a`（7 个工作包合并后、集成之前）。所有问题都在本地模拟中发现，没有连接过真实平台。
“核实”一列：`复现` 表示主会话重跑了评审的复现脚本并得到相同结果；`读代码` 表示只有推理，没有复现。

修完之前不连接真实平台。每条修复都要配回归测试，并在本表回填状态。

## A. 编排、CLI、后端、存储

| 编号 | 级别 | 问题 | 位置 | 核实 | 修复方向 | 状态 |
|---|---|---|---|---|---|---|
| A1 | 严重 | `submit` / `retry` 启动阶段收到 SIGTERM 或 SIGHUP 时不做任何清理，实例留着计费；之后 `gc --dry-run` 还显示“在启动窗口内，保留” | `cli.py`（无信号处理）、`orchestrator.py` `_abort_launch` | 复现 | `main()` 把 SIGTERM、SIGHUP 转成和 Ctrl-C 相同的中止路径 | 已修复：`test_orchestrator_review.py::test_a1_*`、`test_cli.py::test_submit_destroys_the_instance_on_sigterm_or_sighup`、`test_cli.py::test_cancel_force_on_sigterm_still_destroys`、`test_integration_e2e.py::test_signal_to_a_real_dgpu_submit_destroys_its_instance`（真实 `dgpu submit` 进程）。退出码 143 / 129；SIGKILL 仍由 gc 兜底（已写进文档） |
| A2 | 严重 | 创建请求未返回时按 Ctrl-C：只查一次标签就认定没创建，状态写“实例已销毁”，启动文档被删，实例随后出现且不会自毁，也没有提示运行 `gc` | `orchestrator.py` `_abort_launch` | 复现 | 结果未知时按后端的延迟重复查询；查不到就如实写“创建结果未知”，提示 `gc` | 已修复：`test_orchestrator_review.py::test_a2_unknown_create_outcome_is_reported_honestly_and_the_late_instance_is_collected`、`test_a2_instance_found_by_a_later_lookup_is_destroyed`。隔 0/2/5/10 秒查 4 次；查不到则写“结果未知”、作废该 attempt（晚到实例被隔离），提示 10 分钟后 `gc` |
| A3 | 高 | `wait --retries` 在 LOST 时不看取消标记，`retry` 的 bump 还把 `cancel` 清掉，导致取消后又开新实例 | `orchestrator.py` `wait`、`retry` | 复现 | 有取消请求时 `wait` 不重试；`retry` 遇到未完成的取消时拒绝 | 已修复：`test_orchestrator_review.py::test_a3_*`。“未完成”= 有取消标志且当前 attempt 没有终态；任务已结束（含 CANCELLED）时显式 retry 清除标志 |
| A4 | 高 | `gc` 的孤儿规则：存储里没有任务目录且实例超过 1 小时就销毁。另一份配置（另一个桶）或共用平台账户的同事的任务会被删；手工命名成 `dgpu-x-a1` 的实例也会被删 | `orchestrator.py` `_gc_decide` | 复现 | 孤儿只报告，不自动销毁（显式参数才销毁）；`gc` 只认 `new_job_id` 形状的任务 ID | 已修复：`test_orchestrator_gc.py::test_gc_never_auto_destroys_instances_of_unknown_jobs`、`test_gc_include_unknown_destroys_after_confirmation`、`test_gc_only_handles_labels_with_generated_job_ids`、`test_cli.py::test_gc_reports_unknown_jobs_and_include_unknown_asks_first`。新增 `layout.is_generated_job_id`、`gc --include-unknown [--yes]` |
| A5 | 高 | 创建结果不明且标签查询为空后，换下一个 offer 时沿用同一个 attempt 和标签。先前的实例如果晚到，就有两个 runner 写同一个 attempt，`gc` 也分辨不出 | `orchestrator.py` `_try_offer` | 复现 | 只要创建结果不是明确的“被拒绝”，下一次创建前都 bump attempt | 已修复：`test_orchestrator_review.py::test_a5_unclear_create_bumps_the_attempt_so_a_late_instance_is_fenced` |
| A6 | 中 | 按旧判断销毁所有 attempt：`wait` 的收尾会销毁并发 `retry` 刚启动的实例；两个 `retry` 并发时输的一方销毁赢的一方的实例 | `orchestrator.py` `_cleanup_after_end`、`_destroy_job_instances` | 复现 | 销毁时带上所依据的 attempt，只销毁不大于它的实例；`attempt.json` 已前进则跳过收尾 | 已修复：`test_orchestrator_review.py::test_a6_wait_cleanup_does_not_destroy_the_instance_of_a_concurrent_retry`、`test_a6_losing_retry_does_not_destroy_the_winners_instance`（两种交错都是确定性的） |
| A7 | 中 | `wait` 收尾清理遗留实例时吞掉平台错误，返回 SUCCEEDED 且无提示 | `orchestrator.py` `_cleanup_after_end` | 复现 | 清理失败时明确提示并指向 `gc`；`store.delete` 放进同一处理 | 已修复：`test_orchestrator_review.py::test_a7_*`、`test_cli.py::test_wait_exit_code_6_when_the_job_succeeded_but_cleanup_failed`。任务成功时退出码 6（`states.EXIT_CLEANUP_FAILED`），失败时保持任务的退出码 |
| A8 | 中 | 本地数据大小估算不跟随软链接，上传却跟随：5 GB 确认、磁盘大小计算、AutoDL 50 GB 检查都被绕过 | `orchestrator.py` 数据大小估算、`packaging._walk_dataset` | 复现 | 大小取自 `hash_dataset` 的清单，并复用该清单上传 | 已修复：`test_orchestrator_review.py::test_a8_size_estimate_follows_symlinks_like_the_upload`、`test_packaging.py::test_upload_from_a_scan_hashes_that_listing_without_walking_again`。新增 `packaging.scan_dataset` 和 `Deps.scan_data` |
| A9 | 中 | LOST 和 `gc` 的过期判断比较的是本机时钟和实例时钟；10 分钟阈值与“运行中存储断连继续训练”“最终同步可重试 20 分钟”矛盾，`wait --retries` 会销毁仍在训练或上传的实例 | `orchestrator.py` `_view`、`_gc_decide` | 读代码 | 用存储的 `LastModified` 计算过期（`Store` 契约要加）；自动重试前要求平台也显示实例已死，或阈值大于最终同步时限 | 已修复：`test_orchestrator_review.py::test_a9_*`、`test_orchestrator_gc.py::test_gc_heartbeat_rule`、`test_gc_staleness_*`、`test_store.py::test_get_json_meta_*`。新增 `Store.get_json_meta` → `StoredJson`（`LastModified` 和响应的 `Date`，没有 `Date` 时写探针对象） |
| A10 | 低 | 打包时跳过的文件（指向目录外的软链接等）不提示 | `orchestrator.py`（丢弃 `PackResult`） | 读代码 | 打印 `skipped` | 已修复：`test_orchestrator_review.py::test_a10_*`、`test_cli.py::test_submit_with_a_broken_symlink_needs_yes`、`test_packaging.py::test_symlinks_inside_are_kept_relative_and_outside_are_skipped`。新增 `PackResult.skipped_links` |
| A11 | 低 | 费用上限只算 `价格 × max_hours`，没算收尾宽限、失败的 offer 和重试 | `orchestrator.py` | 读代码 | 按截止时间公式和 `max_offers` 计算并说明 | 已修复：`test_orchestrator_review.py::test_a11_ceiling_covers_startup_and_teardown_and_mentions_extra_machines`、`test_orchestrator.py::test_insufficient_balance_stops_before_anything_is_created`。上限 = 价格 ×（启动窗口 + `instance_lifetime_s`） |
| A12 | 低 | `retry` 在新 attempt 启动即失败时仍退出 0；`wait` 把存储瞬时错误变成退出码 2（文档里是“用法错误”） | `cli.py` | 读代码 | 修正退出码；`wait` 对瞬时存储错误重试 | 已修复：`test_cli.py::test_retry_exits_with_the_phase_when_the_new_attempt_ends_before_running`、`test_wait_exit_code_7_when_storage_stays_unreachable`、`test_orchestrator_review.py::test_a12_*`。瞬时存储错误退避重试最多 10 分钟，之后退出码 7（`states.EXIT_STORE_UNREACHABLE`） |

需要真实平台才能确认的怀疑（并入探针）：

- 刚创建的实例短时间内查不到时会被当成已销毁（Vast 404、AutoDL 列表里没有）。建议创建后约 60 秒内忽略 GONE，销毁时总是发送 DELETE。
  - 状态：本地已防护。创建后 90 秒内 GONE 按等待调度处理；销毁总是发请求（AutoDL 查不到记录也发 DELETE，只在查到记录时核对名称）；销毁后 `get` 和 `list` 都确认消失，平台还从没查到过的实例观察到 90 秒。测试：`test_orchestrator_review.py::test_suspicion_a_*`、`test_backend_autodl.py::test_destroy_sends_delete_for_a_deployment_not_yet_listed`。真实平台的延迟有多长仍需探针。
- 标签长度或字符集被平台截断、改写。创建后回读标签。
  - 状态：本地已防护。两个后端的 `get` 返回平台存的标签；不一致时按实例 ID 销毁并让 submit 报错。测试：`test_orchestrator_review.py::test_suspicion_b_altered_label_fails_loudly_and_destroys_by_id`、`test_backend_vast.py::test_get_reports_the_label_the_platform_actually_stored`、`test_backend_autodl.py::test_get_reports_the_name_the_platform_actually_stored`。平台是否真的改写仍需探针。
- AutoDL Job 在 `cmd` 非 0 退出时是否重启容器（探针 A1）。
  - 状态：未处理，需要真实平台（探针 A1）。
- AutoDL 容器列表分页从 0 还是从 1 开始。
  - 状态：本地已防护，不猜：第 1 页空而 `result_total` 不为 0 时改从 0 开始；翻完条数不够时补读第 0 页并去重。测试：`test_backend_autodl.py::test_list_handles_zero_based_paging_without_guessing`、`test_get_sees_containers_on_a_zero_based_container_list`。真实行为仍需探针。
- Vast 搜索到创建之间价格变化。创建后比较实际价格，超过上限就销毁。
  - 状态：本地已防护。启动轮询里第一次拿到实例价格时和上限比较，高于上限就销毁、写 `offer_rejected`、换下一台（attempt + 1）。测试：`test_orchestrator_review.py::test_suspicion_e_*`。Vast 实例行里 `dph_total` 的口径（是否含存储）仍需探针确认。

## B. 容器内 runner、watchdog、bundle

| 编号 | 级别 | 问题 | 位置 | 核实 | 修复方向 | 状态 |
|---|---|---|---|---|---|---|
| B1 | 严重 | `not_uploaded` 只比较文件大小：checkpoint 被等大小改写后，最终上传失败时报告“0 个文件没有传上去”，存储里是旧文件 | `runner/main.py` | 复现 | 每次成功同步记录（路径、大小、mtime_ns），之后变化的都算未上传；拿不准时报“未知”而不是 0 | 已修复：`tests/test_runner_review.py::test_b1_same_size_rewrite_after_upload_is_reported_not_uploaded`、`test_b1_sync_record_rules`（同步记录 `runner/syncstate.py`） |
| B2 | 高 | 最终同步对 output 和 ckpt 各用了整个时限，总时长可达设定的 2 倍；watchdog 硬截止只留“时限 + 10 分钟”，会杀掉正在上传的 runner，把成功的训练记成 TIMED_OUT，且不上传结果 | `runner/main.py`、`runner/watchdog.py`、`runner/util.py` | 复现 | 每个目标只用剩余时间；watchdog 余量由 runner 各收尾阶段的上限之和推出；硬截止时做一次有界的同步并计算 `not_uploaded` | 已修复：集成分支的公式和按剩余时间分配之外，watchdog 硬截止时也做一次有界同步并按同步记录算出 `not_uploaded`。`tests/test_runner_review.py::test_b2_b3_b12_watchdog_hard_deadline_with_live_slow_runner_and_hung_storage`、`test_b2_watchdog_hard_deadline_uploads_results_when_storage_works`、`test_b2_runner_slow_in_teardown_within_its_limits_is_not_killed`；`test_runner_unit.py::test_final_sync_total_is_bounded_by_its_setting` |
| B3 | 高 | watchdog 上传日志没有超时，存储卡住时自毁被阻塞；处理函数里抛异常会跳过自毁 | `runner/watchdog.py` | 复现 | watchdog 的每个存储调用都带超时，有总预算；自毁放进 `finally` | 已修复：watchdog 的同步有总时限，日志、事件、状态共用 `teardown_budget_s`，上限计入 `util.watchdog_teardown_worst_case_s`（之前少算了同步和多次事件上传）。`test_b2_b3_b12_watchdog_hard_deadline_with_live_slow_runner_and_hung_storage`；`test_runner_unit.py::test_watchdog_self_destroys_even_if_*` |
| B4 | 高 | 训练和 `setup` 与 runner、watchdog 同一用户：能读到 `launch.json`、`/proc/*/environ` 里的凭据；`pkill -9 python` 会同时杀掉 runner 和 watchdog，实例无人销毁 | `runner/main.py`、`runner/boot.py` | 复现 | 容器内是 root 时，训练进程降权到无特权用户运行（`preexec_fn` 里 setgid/setuid），`run/` 和 `launch.json` 仅 root 可读；不是 root 时退回现状并记日志。文档不再声称凭据对训练进程不可见 | 已修复（换身份本身要在真实实例上验证）：`runner/privsep.py`，spec 的 `run_as_root` 可关闭；不降权时记一行和 `run_as_root` 事件。`tests/test_runner_privsep.py`、`tests/test_runner_review.py::test_b4_not_root_falls_back_with_one_line_and_an_event`；设计 4.3、4.7 已改 |
| B5 | 高 | Vast 实例重启后已超过 `max_hours`（或停机期间被取消）时，本地结果不上传：是否做最终同步取决于只存在内存里的标志 | `runner/main.py` | 复现 | 按磁盘上的状态（output/ckpt 非空或持久化标记）决定 | 已修复：是否最终同步看磁盘上的结果。`tests/test_runner_review.py::test_b5_restart_after_the_deadline_uploads_local_results`、`test_b5_restart_after_cancel_uploads_local_results` |
| B6 | 高 | `nvidia-smi` 卡死（D 状态，SIGKILL 无效）时 `subprocess.run(timeout=)` 的 `wait` 永不返回，runner 主线程卡住而心跳照常，直到 watchdog 硬截止 | `runner/gpu.py`、`runner/main.py` | 读代码（macOS 无法复现） | 用 Popen 加截止时间轮询，超时后放弃该进程不再等待，按 GPU 故障处理；SIGKILL 之后的 wait 都要有上限 | 已修复：SIGKILL 之后的等待都有上限（`procs.run_bounded` / `wait_bounded`，等不到就放弃进程）；GPU 采样在后台线程。`tests/test_runner_review.py::test_b6_bounded_wait_gives_up_on_a_process_that_survives_sigkill`、`test_b6_gpu_query_times_out_and_marks_stuck_processes`、`test_b6_sampler_pauses_while_a_query_is_stuck_and_never_blocks_the_caller`、`test_b6_hung_nvidia_smi_at_preflight_fails_preflight`、`test_b6_hung_nvidia_smi_during_training_does_not_delay_cancel`（D 状态只能用“SIGKILL 无效”的注入模拟） |
| B7 | 中 | 代码下载失败时，最终同步对不存在的目录重试满 20 分钟才销毁，并写出多余的 `final_sync_failed` | `runner/main.py` | 复现 | 提前创建目录；跳过本地不存在的目标 | 已修复：`tests/test_runner_review.py::test_b7_code_download_failure_finishes_fast_without_final_sync_failed` |
| B8 | 中 | 最终同步期间实例重启会重跑 `setup` 和整个训练 | `runner/main.py` | 复现 | 结果一确定就落盘（带“待最终同步”标记），重启后从最终同步继续 | 已修复：`run/final.json` 带 `synced` 标记。`tests/test_runner_review.py::test_b8_restart_during_final_sync_does_not_rerun_setup_or_training` |
| B9 | 中 | 日志超过上限后，失败原因、退出码、最终状态也被丢弃；runner 自身异常的 traceback 只在不上传的 `runner.log` 里 | `runner/logsink.py`、`runner/main.py` | 复现 | 保留训练输出尾部的环形缓冲；给 runner 的说明留空间；上传 `runner.log`、`watchdog.log`、`bootstrap.log` 的尾部 | 已修复：`tests/test_runner_review.py::test_b9_failure_reason_and_output_tail_survive_log_truncation`、`test_b9_diagnostic_logs_are_uploaded_next_to_the_attempt_log`、`test_b9_runner_traceback_reaches_storage`、`test_b9_log_tail_survives_a_runner_crash`（新 key `layout.diag_log_key`） |
| B10 | 中 | AutoDL 前台的 boot 自己没有截止时间：watchdog 死掉且 runner 卡住时，`cmd` 永不返回 | `runner/boot.py` | 复现 | boot 自己按 `spec.json` 和 `started.json` 执行硬截止，并重启死掉的 watchdog | 已修复：`tests/test_runner_review.py::test_b10_autodl_boot_restarts_a_dead_watchdog`、`test_b10_boot_kills_everything_at_its_own_deadline` |
| B11 | 低 | 周期同步线程没有真正停止，收尾时与最终同步抢带宽 | `runner/main.py` | 复现 | 目标之间检查停止标志 | 已修复：`tests/test_runner_review.py::test_b11_periodic_sync_stops_between_targets` |
| B12 | 低 | runner 被杀后，它启动的 rclone 进程还活着，可能在 watchdog 写状态之后落盘 | `runner/rclone.py` | 复现 | 记录 rclone 的进程组，watchdog 一并杀掉 | 已修复：rclone 登记在 `run/rclone-<组件>/`，watchdog 和 boot 按命令行核对后杀掉。`tests/test_runner_review.py::test_b12_watchdog_kills_rclone_left_running_by_a_killed_runner`、`test_b2_b3_b12_watchdog_hard_deadline_with_live_slow_runner_and_hung_storage` |

只有推理、未复现的问题（修复时逐条确认，结论在每条末尾）：

- **AutoDL 退出码**（取决于探针 A1）：boot 把 runner 的退出码作为 `cmd` 的退出码。如果 Job 会重启非 0 退出的容器，失败的任务会反复重启。探针确认前，AutoDL 前台模式统一退出 0，结果以 `status.json` 为准。——**已修复**：AutoDL 前台模式下最终状态已写进存储（或本 attempt 已被取代）时退出 0，一个状态都没记录时才非 0；Vast 不变。`tests/test_runner_e2e.py::test_nonzero_exit_on_autodl_is_recorded_and_cmd_exits_zero`、`test_b10_*`
- **flock 错误被当成“已有 runner 在运行”**：不支持 flock 的文件系统上什么都不会启动。只有 EAGAIN 才表示被占用。——**确有其事，已修复**：bootstrap、boot 只把 `BlockingIOError` 当作被占用，其他错误记一行后继续，改看 `run/runner.pid`；watchdog 同样退回 pid 文件（只记一次）。`test_flock_errors_other_than_eagain_do_not_mean_busy`、`test_flock_unsupported_job_still_runs_and_second_boot_does_nothing`、`test_bootstrap_flock_unsupported_continues`
- **fencing 窗口**：同步线程不重读 `attempt.json`；最后一次读取之后状态写入还能重试 300 秒；watchdog 硬截止路径只看本地标记。——**确有其事，已修复**：周期同步在最近一次成功读取超过 3 个心跳间隔时自己读一次，读不到就这一轮不写；最终同步前读一次；心跳线程一直运行到写最终状态之前，写之前最近一次读取不够新就再读一次；watchdog 两条接手路径都读 `attempt.json`。剩余窗口是最后一次读取到状态写成之间（不超过 `teardown_budget_s` 的一半），CLI 要心跳过期 `lost_after_min` 才会换机，覆盖得住。`test_fencing_periodic_sync_rechecks_a_stale_attempt_before_writing`、`test_fencing_watchdog_hard_deadline_checks_attempt_json`
- **卡死检测**：也作用于 `setup`（安静的下载或解压会误判）；单个 GPU 报 N/A 会让所有 GPU 的检测失效。——**两点都确有其事，已修复**：只用于 `run`；N/A 的卡不参与，其余卡照常判断。`test_stall_detection_ignores_a_quiet_setup`、`test_stall_detection_with_one_gpu_reporting_na`、`test_runner_unit.py::test_gpu_parse_and_unknown_utilization`
- **事件同步刷新**：停止训练前和日志截断回调里各有最长 120 秒的同步等待。——**确有其事，已修复**：事件只写本地队列并唤醒后台上传线程，写事件的线程不等存储。`test_stopping_is_not_delayed_by_a_hung_event_upload`、`test_events_emit_never_waits_for_storage`
- **`bash -lc`**：会执行镜像的登录配置，可能重置 PATH，或重新导出已清除的变量。——**确有其事，已修复**：仍用登录 shell（有的镜像靠 profile 配环境），profile 执行之后再清除凭据类变量、把 profile 删掉的 PATH 条目补回末尾、重新导出 `DGPU_*`。`test_login_profile_cannot_reexport_secrets_or_drop_path`
- **bootstrap 诊断信息太少**：只记录异常类型，日志也不上传。——**确有其事，已修复**：记录 Python 版本、uid、工作目录、每次失败的异常说明（URL 一律替换成 `<url>`）、放弃时的磁盘余量；`bootstrap.log`（0600）由 runner 随日志上传到 `logs/attempt-<n>.bootstrap.log`。bootstrap 在 runner 启动前就放弃时还没有凭据，只能看平台日志。`test_bootstrap_https_failure_is_explained_and_redacted`、`test_bootstrap_flock_unsupported_continues`

## C. 评审指出的测试缺口

- 代码下载失败；等大小的旧文件；最终同步的总时长；存储卡住时的 watchdog 硬截止路径。
- 截止时间之后或取消之后的重启；最终同步期间的重启。
- `nvidia-smi` 超时；日志截断后失败原因是否保留；flock 失败；bootstrap 的 HTTPS 或证书失败。
- watchdog 截止测试用 SIGSTOP 冻结了 runner，没有覆盖“runner 活着但收尾慢”的真实情况。
- `procs.py` 的 `/proc` 分支在 macOS 上从未执行。
- 凭据测试只断言环境变量被清除，没有验证训练进程实际读不到。

B 组相关的缺口已补（`tests/test_runner_review.py`、`tests/test_runner_privsep.py`）：代码下载失败（B7）、等大小的旧文件（B1）、最终同步总时长（B2）、存储卡住时的 watchdog 硬截止（B2/B3）、截止后 / 取消后 / 最终同步中的重启（B5、B8）、`nvidia-smi` 超时（B6）、截断后保留失败原因（B9）、flock 失败、bootstrap 的 HTTPS 失败、runner 活着但收尾慢时的 watchdog 截止（B2）、`/proc` 解析（`test_procs_parses_a_linux_proc_tree`，用伪造的进程表）。仍未覆盖：降权后训练进程实际读不到凭据、发不了信号——测试环境不是 root，要在真实实例上验证。

## D. 评审认为没有问题的部分

- 每条退出路径都会到达自毁；信号处理只置标志；fencing 的检查和启动在同一把锁下。
- 凭据只经环境变量传给 rclone，不出现在命令行；bundle 和 rclone 都固定了哈希；tar 解压拒绝越界路径。
- 标签解析无歧义，标签查询是精确匹配；两个后端的 `list` 和 `destroy` 都强制 `dgpu-` 前缀。
- Vast 把无效 key 和“不存在”区分开；`create` 不盲目重试；条件写的 ETag 处理正确。
- 平台 key 只在请求头里，输出和事件都做了打码；配置文件强制 600 权限。

## E. 第二轮复核（2026-11-10）

复核对象是提交 `6c7a012`（A、B 两节修复之后）。复核者重跑了全部测试，另写了模拟世界里的复现（编排一侧）和真实 runner / watchdog 进程的复现（实例一侧），并逐项读了代码。下表每条都在 `wp/fix-round2` 上修复，回归测试在修复前失败、修复后通过。仍然都只在本地模拟中验证。

| 编号 | 级别 | 问题 | 状态和测试 |
|---|---|---|---|
| F1 | 严重 | 结果未知的 `create` 之后换了下一台；先前的实例晚到时没人通知、也不会自毁（启动文档已删，bootstrap 失败后 `runtype=ssh` 的容器一直在），一直计费到 `wait` 收尾或 `gc`；紧接着收到 SIGTERM 时状态是 CANCELLED 且没有任何提示 | 已修复。编排：按标签查满 `create_lookup_s`（默认 120 秒，用户决定），窗口内出现就接管；查不到才记进 `attempt.json` 的 `suspect_labels`、换 attempt 换机；返回前和中止时再查一次，找到就销毁，查不到就提示 10 分钟后 `gc` 并写事件；`status` 列出可疑标签，`wait` 收尾和 `gc` 销毁晚到实例时说明；中止时再按一次 Ctrl-C 跳过等待。实例：bootstrap / boot 在 runner 起来之前放弃时，Vast 上用容器 key 自毁，AutoDL 上退出。`test_orchestrator_round2.py::test_f1_*`；`test_runner_bootstrap.py::test_bootstrap_self_destroys_when_the_launch_document_is_gone`、`test_bootstrap_refuses_a_bundle_with_the_wrong_checksum`、`test_bootstrap_self_destroys_when_the_bundle_code_cannot_be_imported`、`test_bootstrap_on_autodl_gives_up_by_exiting`、`test_bootstrap_self_destroy_retries_a_failing_api_and_stays_bounded`、`test_boot_gives_up_before_the_runner_starts`、`test_bootstrap_starts_the_runner_once_and_a_second_run_does_nothing`（“已有 runner”不自毁）；`test_bundle.py::test_bootstrap_command_fits_with_the_longest_job_id_and_bucket_name` |
| F2 | 高 | 代码或数据下载失败、且存储对传输不可用时，最终同步对着空的 `output/`、`ckpt/` 重试到 `final_sync_timeout_min`，还写出多余的 `final_sync_failed`；第 5 节 B7 那一行在存储不可用时不成立 | 已修复：是否最终同步只看本地有没有结果；`sync_dir` 没有要传的东西时不调用 rclone。`test_runner_round2.py::test_f2_*` |
| F3 | 中 | 平台明确拒绝创建（如 Vast 4xx 坏镜像）被当作“结果未知”：查 17 秒标签、作废 attempt、提示 `gc` | 已修复：`BackendError.created`（False = 确定没创建），两个后端在平台回答了拒绝或请求没发出时设置；编排不查标签、不提示 `gc`，直接报平台的原因；超时、断连、5xx 仍是未知。`test_orchestrator_round2.py::test_f3_*`、`test_backend_vast.py::test_f3_*`、`test_backend_autodl.py::test_f3_*` |
| F4 | 低 | `attempt.json` / `status.json` 建好之后、换机循环之前的信号或异常让状态永远是 PROVISIONING，之后显示 LOST，`wait --retries` 会为用户中断的任务开机器 | 已修复：这段也走中止路径，写 CANCELLED（信号）或 FAILED（出错）。`test_orchestrator_round2.py::test_f4_*` |
| F5 | 中 | `wait --retries` 确认 LOST 后换机在销毁之前就失败（没有机器、余额不够），已退出但仍按磁盘计费的旧实例没人管，也没有提示 | 已修复：确认失联后先销毁 attempt ≤ 失联 attempt 的实例，不管换机成不成功；销毁失败提示 `gc` 并写 `destroy_failed`。`test_orchestrator_round2.py::test_f5_*` |
| F6 | 中 | 容器停机超过 watchdog 截止后重启，watchdog 抢先接手，用自己较短的同步时限（默认 10 分钟，runner 是 20 分钟），大结果可能被报成没上传 | 已修复：这种情况下 watchdog 给满 `final_sync_timeout_min` 并重试到时限；上限计入 `watchdog_teardown_worst_case_s` 和 `instance_lifetime_s`（`deadline_at` 随之一致，费用上限的 2 小时示例从约 3.4 小时变成约 3.6 小时）。`test_runner_round2.py::test_f6_*` |
| F7 | 待真实平台 | 默认的 `vastai/pytorch` 镜像上降权是生效还是退回 root（venv 禁用用户 site 时会退回）不知道 | 行为不变，只让原因看得见：`dgpu status` 的“训练进程：”一行、attempt 日志一行、`run_as_root` 事件的 `reason_text`。YOLO 示例改用固定 tag `vastai/pytorch:2.14.1-cu126-cuda-12.9-mini-py312-2026-10-23`。`test_runner_round2.py::test_f7_fallback_reason_reaches_the_heartbeat_the_event_and_one_log_line`、`test_orchestrator_round2.py::test_f7_status_shows_whether_the_training_runs_as_root_and_why`、`test_examples.py::test_job_yaml_fills_in_both_backends`。真实结果看探针第 5 步 |
| F8 | 中 | 降权的加固：大数据集上 `chown_tree` 不能被取消或截止打断，容器重启后还会再遍历一遍；CAP_KILL、CAP_DAC_OVERRIDE 是假设没有验证 | 已修复：chown 每 256 个条目检查取消和截止；`run/owner-<目录>.json` 标记让重启后不再遍历；降权前真的以训练用户起一个子进程，确认能读它的 0600 文件、能发信号 0 和 SIGKILL，不行就以 `capability_missing` 退回 root；每种原因都有 `reason_text`。`test_runner_privsep.py::test_f8_*` |
| F9 | 低 | 文档漂移：4.3 说 `run_as_root` 的 CLI 解析待实现（已实现）；4.5、第 9 节写“约 42 分钟”（公式算出 37.2 分钟）；第 5 节 B7 那一行；4.6、4.8 说晚到的实例会自己隔离（启动文档已删，runner 起不来） | 已修复（设计 4.3–4.8、4.10、第 5、6、8、9 节，`docs/usage.md`、`skills/dgpu/SKILL.md`） |

复核者的结论：**可以做有人值守的小额 Vast 探针；F1、F2 修复前不做无人值守运行。** F1、F2 现在已在本地修复，但容器 key 自毁、bootstrap 放弃路径都还没在真实实例上跑过，先按下面的顺序做探针。

建议的探针顺序（每一步之后都看 Vast 控制台并跑 `dgpu gc --dry-run`）：

0. `dgpu doctor`、`dgpu gc --dry-run`、`dgpu offers`。
1. `submit` 一个 `sleep 600`、`max_hours: 0.05` 的任务，实例一创建就按 Ctrl-C：实例应该消失，状态 CANCELLED，说明里有实例 ID。
2. 同样的任务，改用 SIGTERM：同上，退出码 143。
3. 同样的任务，让它自己到 TIMED_OUT，**不要**跑 `dgpu wait`（它的收尾会掩盖自毁是否生效）；看 `dgpu status` 和控制台，实例应该自己消失。
4. 记录：标签回读、`dph_total` 和上限的比较、`start_date`、心跳延迟、诊断日志（`logs/attempt-<n>.{runner,watchdog,bootstrap}.log`）。
5. 降权检查任务：`id -u`、试着读 `launch.json`、`kill -0` runner 的 pid、`nvidia-smi`、在 `/etc/environment` 里 grep `CONTAINER_API_KEY`；看有没有 `run_as_root` 事件和 `dgpu status` 的“训练进程：”一行。
6. `cancel`，然后 `cancel --force`。
7. 实例创建后 `kill -9` 掉 `submit`：`gc --dry-run` 应该在启动窗口内保留它，20 分钟后 `gc` 销毁它。
8. 以上都正常之后，再跑真实任务。

