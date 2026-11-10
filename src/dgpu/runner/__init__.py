"""容器内 runner。只用标准库，兼容 Python 3.8。

模块：
  boot      bootstrap 之后的入口：拿锁（幂等），先启动 watchdog 再启动 runner；
            Vast 上立即返回，AutoDL 上留在前台直到两者都退出
  main      runner 主流程（设计 4.7）和收尾、自毁
  watchdog  独立进程：runner 消失时接手收尾，到硬截止时间结束一切并自毁
  rclone    rclone 子进程封装（凭据只进 rclone 的环境变量）
  events    事件先落本地队列再上传，断网不丢
  logsink   持续读取训练进程输出、日志上限
  procs     进程组与进程树的结束（SIGTERM → 宽限 → SIGKILL）
  destroy   自毁：Vast 用容器 key 调 API，AutoDL 结束进程即可
  gpu       nvidia-smi 查询
  archive   安全解压代码包
  util      时钟、调参、工作目录布局、JSON 文件
"""
