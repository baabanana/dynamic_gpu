# AutoDL 及国内 GPU 租用平台：小型 YOLO 程序化短时训练（数据在 Cloudflare R2）

调研日期：2026-11-09。除特别标注外，AutoDL 官方文档页面均无发布/更新时间戳，下文"文档所述"指 2026-11-09 抓取到的内容。"用户轶事/第三方文章"与"官方文档"已分开标注。本次调研中部分页面（LINUX DO、NodeSeek）抓取返回 403，仅看到搜索摘要，已在文中注明。

---

## 1. AutoDL 官方 API：能否自动化"创建实例 → 跑命令 → 关机 → 释放"

### Takeaway
普通"容器实例"没有 API；只有"容器实例 Pro"（个人实名或企业认证即可）和"弹性部署"（仅企业认证）两套官方 API，且 Pro 的可选 GPU 清单里没有 3060/3080 等便宜卡，只有 4090D、5090-32G、4080S-32G、3090-48G、4090-48G、PRO6000、H800。Pro API 可完整覆盖 create → power_on(带命令) → power_off → release，并支持创建/开机时指定 `start_command`；实例内也可用 `/usr/bin/shutdown` 自行关机。

### Cited Findings

**文档结构与产品差异**
- AutoDL 文档目录的"API文档"分组只有三项：`common_api/`、`instance_pro_api/`（容器实例Pro API）、`esd_api_doc/`（弹性部署API）；没有名为"开放API"的、面向普通容器实例的章节。 — [AutoDL 文档首页](https://www.autodl.com/docs/)
- 文档明确对比：容器实例Pro 支持 API、可自动化创建/管理/销毁；普通容器实例不支持 API。 — [容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)
- Pro 采用存算分离，与物理机不绑定，文档称"只要总GPU库存充足随时开机"；普通容器实例磁盘与物理机绑定，所在机器无空闲 GPU 时需等待或克隆。 — [容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)
- Pro：系统盘默认 30GB、付费最大扩到 500GB；**没有数据盘**（文档称"可能会支持"）；系统盘是网络存储，IO 低于本地 NVMe；只支持文件存储挂载；连续关机 60 天后释放（普通实例为 15 天）。 — [容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)

**容器实例 Pro API（与用户需求最匹配）**
- 资格："使用容器实例Pro API需进行个人实名认证或者企业认证"。Base URL `https://api.autodl.com`；token 在 控制台→设置→开发者Token，放在 `Authorization` 请求头。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- 端点（均在 `https://api.autodl.com` 下）： — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
  - 创建 `POST /api/v1/dev/instance/pro/create`
  - 详情 `GET /api/v1/dev/instance/pro/snapshot`（返回 SSH host/port/密码、Jupyter token/域名、6006/6008 服务域名、折后/原价）
  - 状态 `GET /api/v1/dev/instance/pro/status`
  - 列表 `POST /api/v1/dev/instance/pro/list`
  - 开机 `POST /api/v1/dev/instance/pro/power_on`
  - 关机 `POST /api/v1/dev/instance/pro/power_off`
  - 释放 `POST /api/v1/dev/instance/pro/release`
  - 保存镜像 `POST /api/v1/dev/instance/pro/image/save`；私有镜像列表 `POST /api/v1/dev/instance/pro/image/private/list`
- 创建参数：`data_center_list`（可选，如 `westDC3`、`beijingDC2`，缺省由系统选）、`req_gpu_amount`（必填，1–4）、`expand_system_disk_by_gb`（必填，0–500）、`gpu_spec_uuid`（必填，如 `pro6000-p`）、`image_uuid`（必填）、`cuda_v_from`（必填，整数，113 即 11.3）、`instance_name`、`start_command`（可选）。默认按量计费，API 不能选其他计费方式。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- `start_command` 原文："实例开机后执行此命令，该命令执行成功与否不影响实例开机运行，不会因命令执行失败而关机"；开机接口（`power_on`）也可传 `start_command`，覆盖创建时的值。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- 开机接口 `payload` 必须为 `"gpu"`；文档称 API 不支持无卡模式开机。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- 释放前须先关机："在释放实例前请先关机实例，否则可能无法释放"。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- Pro 可用的 `gpu_spec_uuid`（附录）：H800-80G=`h800`；4090-48G=`v-48g`；PRO6000-96G=`pro6000-p`；4080(S)-32G=`v-32g-p`；3090-48G=`v-48g-350w`；5090-32G=`5090-p`；4090D=`4090D`。附录里没有 3060/3080/24G 的 3090/4090。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- 公共基础镜像（附录，仅列到 CUDA 11.x 的 PyTorch 1.9–2.0、TF、Miniconda、TensorRT 等；更新的镜像需联系客服）。例：`base-image-l2t43iu6uk` = cuda11.8-cudnn8-ubuntu20.04-py38-torch2.0.0。文档同时说私有镜像需在网页上保存。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- Pro API 页**没有**库存查询端点。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- `common_api`（无资格声明）仅两个端点：`POST /api/v1/dev/wallet/balance`（余额，数值除以 1000 为元）和 `POST /api/v1/dev/exclusive_nfs/mount`（切换专属 NFS/标准文件存储）。 — [通用API](https://www.autodl.com/docs/common_api/)

**弹性部署 API**
- 资格："使用弹性部署API需先认证企业"；弹性部署功能页写"该功能仅面向企业用户"。Base URL `https://api.autodl.com`，同样用 Developer Token。 — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)；[弹性部署概述](https://www.autodl.com/docs/elastic_deploy/)
- 端点： — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
  - 创建部署 `POST /api/v1/dev/deployment`（`deployment_type`: `ReplicaSet`/`Job`/`Container`）
  - 部署列表 `POST /api/v1/dev/deployment/list`；容器列表 `POST /api/v1/dev/deployment/container/list`；容器事件 `POST /api/v1/dev/deployment/container/event/list`
  - 停止容器 `PUT /api/v1/dev/deployment/container/stop`
  - 设置副本数 `PUT /api/v1/dev/deployment/replica_num`
  - 停止部署 `PUT /api/v1/dev/deployment/operate`（`operate: "stop"`）
  - **删除部署（释放）`DELETE /api/v1/dev/deployment`**（运行中会先停后删）
  - 地区 GPU 库存 `POST /api/v1/dev/machine/region/gpu_stock`；时长包余额 `GET /api/v1/dev/deployment/ddp/overview`；调度黑名单 `POST/GET /api/v1/dev/deployment/blacklist`
- 创建时 `container_template` 含：`dc_list`、`cuda_v_from/to`、`gpu_name_set`、`gpu_num`、CPU/内存范围、`price_from/price_to`（单位"元×1000"，0.1 元填 100）、`image_uuid`、`cmd`（必填，"启动容器命令"）、`cmd_before_shutdown`（停止前先执行，超时 5 秒直接停止）。 — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
- 容器生命周期：容器只在 `cmd` 运行期间存在，"cmd执行结束那么该容器将会退出关机"；Job/Container 类型在容器结束后调度结束；弹性容器关机后数据立即释放、不保留；数据盘固定 50GB 不可扩。 — [弹性部署概述](https://www.autodl.com/docs/elastic_deploy/)
- 弹性部署私有镜像不能从外部导入，必须在 AutoDL 网站创建并保存。 — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
- 弹性部署还支持 `reuse_container`（复用已停止容器，加快创建）。 — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
- 区域码：westDC2（西北企业区，推荐）、westDC3、beijingDC1/2、beijingDC4（L20 区）、beijingDC3（V100 区）、neimengDC1/3、foshanDC1、chongqingDC1、yangzhouDC1（3090 区）。 — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
- 社区封装包 `autodl-api`（PyPI，1.0.1），自述为弹性部署 API 的封装，**非官方**。 — [PyPI autodl-api](https://pypi.org/project/autodl-api/1.0.1/)
- 备注：`private.autodl.com/docs/esd_api_doc/` 是同一文档的私有云版本，其 HOST 是 `private.autodl.com`；公有云文档为 `api.autodl.com`，勿混用。 — [搜索结果所指的私有云文档](https://private.autodl.com/docs/esd_api_doc/)

**实例内自行关机 / 开机自动执行**
- 文档"省钱"页明确给出实例内关机命令："最好使用命令完整路径：`/usr/bin/shutdown`"；`;` 拼接表示前面成败都执行 shutdown，`&&` 表示前面成功才执行；也可在 Python 中 `os.system("/usr/bin/shutdown")`；并提醒自动关机后标准输出日志不再可见。 — [AutoDL 省钱技巧](https://www.autodl.com/docs/save_money/)
- 该页还建议"不知道任务要跑多久时，在命令后拼接关机命令"。该页面写的是通用实例，未专门说明 Pro 实例是否同样有效。 — [AutoDL 省钱技巧](https://www.autodl.com/docs/save_money/)
- 普通容器实例的守护进程文档只提 `screen`/`tmux`/JupyterLab 终端，没有提到开机自启动脚本。 — [守护进程](https://www.autodl.com/docs/daemon/)
- 计费页：按量计费"实例开机开始计费，关机结束计费"，整点扣费一次、关机时扣费一次，最低 0.01 元。 — [充值与计费](https://www.autodl.com/docs/price/)

### Inferences
- 对"短时程序化训练"，唯一可全程 API 化且单账号个人实名就能用的官方路径是 **容器实例Pro API**（create 带 `start_command` → 轮询 `status` → 实例内脚本末尾 `/usr/bin/shutdown` → 外部检测到关机后调 `release`）。该流程由文档字段拼出，`/usr/bin/shutdown` 在 Pro 上是否同样生效、关机后 Pro 状态字符串叫什么，文档均未说明，需要实测。
- 弹性部署的"cmd 结束即退出并释放数据"最适合短任务，但需企业认证；且数据立即释放，故结果必须在 `cmd` 内传出（例如上传到 R2/OSS）。
- Pro API 没有库存查询端点，只能靠创建失败的返回码判断，需要实测错误码。
- 需要外部 orchestrator 主动调 `release`：实例内部只能关机，无法保证自身被释放（文档只说关机，Pro 关机 60 天后才自动释放）；Pro 系统盘扩容部分在关机后是否继续计费未见文档（普通实例的扩容数据盘文档说关机也计费）。因此关机后尽快释放才能保证计费彻底终止。

### Gaps
- Pro 创建失败（无库存、余额不足等）的错误码与响应格式：文档只写了 `code/data/msg/request_id`，未列错误码表。
- Pro 实例关机后的状态字符串、`/usr/bin/shutdown` 在 Pro 上的行为、系统盘扩容部分关机期间是否计费：官方文档未写明。
- 普通容器实例是否有任何非官方/逆向 API：未调研，且官方明确称不支持。
- 官方文档无更新日期，无法确认 Pro API 文档的版本时间。弹性部署有 `elastic_deploy_update_log/` 更新日志页，本次未读。

---

## 2. AutoDL 定价（2026 年 11 月）与计费/释放规则

### Takeaway
官网首页（2026-11-09 实测）标价：4090 ¥1.88/时、3090 ¥1.32/时、5090 ¥2.78/时，并有"会员95折"标签；首页横幅称"平台将于 11 月 12 日调整部分 GPU 价格"，所以这些数字三天后可能变化。无卡模式 ¥0.1/时（0.5 核/2GB，每主账号同时一个）；关机后按量计费停止，但扩容数据盘继续计费；普通实例连续关机 15 天释放，Pro 为 60 天。Pro 专有卡型（5090-32G、4090D、3090-48G 等）的价格本次未能取得。

### Cited Findings
- 官网首页"炼丹会员及租用价格"区块（JS 渲染，本次用浏览器渲染后截图查看，2026-11-09）显示每小时标价：H800/80GB ¥9.98；H20/96GB ¥9.98；A800-80GB ¥5.59；RTX 5090/32GB ¥2.78；RTX 4090/24GB ¥1.88；RTX 3090/24GB ¥1.32；RTX A4000/16GB ¥0.92；RTX 2080Ti/11GB ¥0.88；每项均带"会员95折"标签。 — [AutoDL 首页](https://www.autodl.com/home)
- 同页橙色横幅（2026-11-09 可见）："因供应侧成本剧烈波动，平台将于11月12日调整部分GPU价格。若后续价格出现回落平台将第一时间下调价格。" — [AutoDL 首页](https://www.autodl.com/home)
- 首页算力排名列出的在售卡型含 RTX 3060/12GB、3070/8GB、3080/10GB、3080Ti/12GB、3090、3090Ti、4090、4090D、5090、A4000、L40 等（只是卡型清单，无价格）。 — [AutoDL 首页](https://www.autodl.com/home)
- 算力市场页（`/market/list`）需要登录，未登录跳转登录页，因此未取得 3060/3080 以及实时库存的价格（本次不登录）。 — [AutoDL 算力市场（重定向到登录页）](https://www.autodl.com/market/list)
- 官方 GPU 选型页的规格表（无价格）：3060 12GB、3080Ti 12GB、3090 24GB、4090 24GB；不含 5090；页面不显示日期。 — [GPU 选型](https://www.autodl.com/docs/gpu/)
- 无卡模式："0.5核；2GB内存；无GPU卡，价格统一为￥0.1/小时"；同一主账号同时只能一个无卡实例；无卡模式释放 GPU，正常开机时若 GPU 被他人租用可能出现空闲 GPU 不足。 — [AutoDL 省钱技巧](https://www.autodl.com/docs/save_money/)
- 按量计费：开机计费、关机结束；整点与关机时扣费；扩容数据盘"无论实例是否开机，均会计费"，单价以各主机显示为准（无固定数）；欠费超 50 元平台有权删除数据；余额不足时平台保留实例并继续计费，不会立刻释放。 — [充值与计费](https://www.autodl.com/docs/price/)
- 文件存储：¥0.01/GB/日，前 20GB 免费，按当日峰值计、次日早晨扣费；单用户默认 200GB 上限，inode 上限 20 万；需按地区初始化，只能挂到同地区实例（挂载点 `/root/autodl-fs`）；"账号连续3个月未登录或欠费50元以上，文件存储数据将被清空"。 — [文件存储](https://www.autodl.com/docs/fs/)
- 普通容器实例默认 50GB 免费数据盘；释放规则："连续关机15天释放实例，届时实例所有数据将被清空，释放后无法恢复"；主机下架时实例也会被释放，界面至少提前 7 天提示；重置系统只清系统盘，数据盘 `/root/autodl-tmp` 不受影响。 — [实例数据](https://www.autodl.com/docs/instance_data/)；[容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)
- Pro 实例连续关机 60 天后释放；Pro 付费磁盘价格高于普通容器实例，但文档未给具体单价。 — [容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)
- 带宽与流量不单独计费，同地区实例共享约 3–10Gbps 带宽（上下行相等）。 — [网络](https://www.autodl.com/docs/network/)
- Pro API 的实例详情会返回折后价与原价，可在创建后程序化读取实际单价。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- **第三方（非官方，仅供参考）**：2026-06-10 的一篇平台横评称 AutoDL 4090 约 ¥1.61–2.50/时（按量），恒源云约 ¥1.32–2.60，晨涧云约 ¥1.80–2.20。 — [AtomGit/CSDN 2026-06-10 抢卡攻略](https://gitcode.csdn.net/6a29515610ee7a33f27a8b4e.html)
- **第三方（非官方）**：PromptQuorum 2026 年对比文称 AutoDL 4090 低至 ¥2.68/时、3090 低至 ¥1.68/时，并称按秒计费、现货夜间可便宜 30–50%——与官方文档"整点扣费"和首页当前价不一致，可信度低。 — [PromptQuorum（经搜索结果转述）](https://www.promptquorum.com/zh/local-llms/alibaba-cloud-vs-tencent-cloud-gpu-ai-2026)
- 第三方旧文：AutoDL 曾把 3090 租价从 4.5 元/时打到 1.32 元/时，属历史促销，日期较早。 — [36氪](https://www.36kr.com/p/3004556400074761)

### Inferences
- YOLO 小模型（如 YOLOv8n/s）在 4090/3090 上跑 1 小时的算力成本约 ¥1.3–1.9，加上开机/镜像/数据拉取时间，单次短任务大约几元量级（粗略估计，需实测）。
- 首页价格可能是"起步价/最低价"而非每台主机的实际价格（页面没明说，且我没有看到每台主机报价）；横幅提示的 11 月 12 日调价意味着应在实际使用前再读一次价格；Pro API 的创建返回/详情返回价格，可用于程序化防价格漂移（设预算上限）。
- 首页截图在右侧被截断（页面有横向滚动条），可能还有未捕获的第三列卡型，3060/3080 的价格不能断言不存在。

### Gaps
- Pro 专有卡型（5090-32G、4090D、4080S-32G、3090-48G、4090-48G）的官方每小时价格：未取得（市场页需登录，Pro 文档无价格）。
- 3060/3080/3080Ti 的当前价格：未取得。
- 数据盘扩容、Pro 系统盘扩容的具体单价：文档称以各主机显示为准，未取得。
- 11 月 12 日调价的具体幅度：未取得。
- 会员（炼丹会员）95 折如何获得、是否对 API 创建实例适用：未核实。

---

## 3. GPU 可用性：缺货频率，API 创建会否因无空闲 GPU 失败

### Takeaway
第三方 2026-06 的文章称 AutoDL 的 4090 在高峰期"经常秒空"；官方文档承认普通实例会因所在主机 GPU 被占而无法有卡开机，而 Pro 号称"总库存充足即可开机"。API 创建在无库存时的具体行为（报错/排队）官方 Pro 文档未写；弹性部署文档有库存查询端点但同样未写排队行为。

### Cited Findings
- 文档：普通容器实例的磁盘与物理机绑定，所在机器没有 GPU 时需等待或克隆（迁移）；关机后别人可能占用 GPU；用无卡模式开机会释放 GPU，之后正常开机可能"空闲GPU不足"。 — [容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)；[AutoDL 省钱技巧](https://www.autodl.com/docs/save_money/)
- 文档：Pro"只要总GPU库存充足随时开机"。 — [容器实例Pro 概述](https://www.autodl.com/docs/instance_pro/)
- 弹性部署提供 `POST /api/v1/dev/machine/region/gpu_stock`，按地区返回各 GPU 型号的 `idle_gpu_num` / `total_gpu_num`（文档示例：westDC2 的 RTX 4090 空闲 215、总 2285；是文档示例，不是实时数据）；文档警告库存过滤按"每机一卡"计算，多卡容器即使显示有库存也可能调度不上。 — [弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
- 弹性部署页没有说明无匹配资源时是排队还是 pending。 — [弹性部署概述](https://www.autodl.com/docs/elastic_deploy/)
- Pro API 页无库存端点，也没有"库存不足"错误码说明。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)
- 第三方文章（AtomGit/CSDN，2026-06-10，作者署名"AI算力小知识"）把 AutoDL 的难抢程度评为"极难抢"，称"用户基数太大，4090高峰期经常秒空"，并引用用户群"刷不到就要等大半天"；称晨涧云"较容易"、恒源云"较紧张"。该文属营销/攻略类，不是 AutoDL 官方数据，且未涉及 API。 — [AtomGit/CSDN 2026-06-10](https://gitcode.csdn.net/6a29515610ee7a33f27a8b4e.html)
- AutoDL 官方 FAQ 页 `qa3` 只讲"不能调用 GPU"，没有"开机 GPU 不足"的说明，其他报错让联系客服。 — [AutoDL FAQ qa3](https://autodl.com/docs/qa3/)

### Inferences
- 对于程序化调度，应按"可能创建失败"设计：重试 + 换 `data_center_list` + 换 `gpu_spec_uuid`（5090-p/4090D/v-32g-p）+ 兜底切换平台。
- 如果拿到企业认证，弹性部署可以先调 `gpu_stock` 判断，再提交 `Job`。个人账号没有库存端点（Pro 文档），只能"试创建"。

### Gaps
- AutoDL API 创建时无库存的实际返回（立即报错 vs 排队）：无官方文档，也未找到可靠用户报告。
- 非高峰时段的实际缺货率、Pro 卡型（5090-32G/4090D）的缺货率：没有可靠数据，仅有 2026-06 的一篇文章对 4090 的描述。

---

## 4. 容器权限：能否 FUSE（rclone mount / s3fs）

### Takeaway
没有找到任何 AutoDL 官方文档或用户报告证实或否认 `/dev/fuse` 可用；已知的是容器内不支持 Docker。保守做法是不依赖 FUSE，用 `rclone copy/sync` 或 `aws s3 cp`/boto3 把数据集先拉到本地盘/系统盘再训练。

### Cited Findings
- 官方：AutoDL 实例是用 Docker 技术隔离的 Container，"容器实例内不支持使用Docker，如需使用Docker请联系客服租用裸金属服务器"；该页未涉及特权、FUSE、内核模块、systemd。 — [AutoDL 实例概要](https://www.autodl.com/docs/env/)
- 官方文档目录中没有任何以 FUSE 或挂载对象存储为主题的页面；存储相关仅有文件存储、公网网盘等。 — [AutoDL 文档首页](https://www.autodl.com/docs/)
- 官方推荐的外部网盘方案 AutoPanel 是浏览器/Server 架构的第三方网盘客户端，支持阿里云盘、夸克、百度，以及 OSS 类（MinIO、阿里云 OSS、腾讯云 COS、"兼容S3协议的OSS"）；典型速度 "5~20MB/s"；文中未提 rclone，也不是 FUSE 挂载。 — [公网网盘](https://www.autodl.com/docs/netdisk/)
- 通用背景（非 AutoDL）：Docker 里挂载 FUSE 通常需要 `--device /dev/fuse` 加 `SYS_ADMIN` 能力或 `--privileged`，否则报 "fuse: device not found"；阿里云 ossfs FAQ 也是这么说。 — [rclone 论坛](https://forum.rclone.org/t/cannot-mount-in-docker-container-fuse-device-not-found/50187)；[阿里云 ossfs FAQ](https://help.aliyun.com/zh/oss/developer-reference/ossfs-faq)
- 通用背景：其他 GPU 容器平台（RunPod 的 SkyPilot issue）也有因缺少 FUSE 设备/权限而 JuiceFS 挂载失败的报告。 — [SkyPilot issue（经搜索结果）](https://www.goodfirstissue.org/skypilot-org/skypilot/issues/8592)
- Cloudflare 官方有 rclone 配置 R2 的示例：provider 选 Cloudflare R2，endpoint 为 `https://<账户ID>.r2.cloudflarestorage.com`。 — [Cloudflare R2 + rclone](https://developers.cloudflare.com/r2/examples/rclone/)

### Inferences
- AutoDL 容器以 Docker 隔离、官方称不支持 Docker，一般意味着不是特权容器；因此 FUSE 很可能不可用，但这属于推断，必须用 `ls -l /dev/fuse` 与 `rclone mount` 实测确认。
- 对 YOLO 小数据集，不需要 FUSE：训练前 `rclone copy r2:bucket/dataset /root/autodl-tmp/data`（Pro 无数据盘，改拷到系统盘 `/root` 或挂载的文件存储）；训练后 `rclone copy runs/ r2:bucket/runs/` 再关机。rclone 的 `copy/sync` 是纯用户态，不需要特权。

### Gaps
- AutoDL 容器内 `/dev/fuse` 是否存在、能否 `mount`：无来源，需实测或问客服。
- Pro 实例（存算分离、网络系统盘）与普通实例在内核/特权上有无差异：未找到。

---

## 5. 网络：学术加速、境外流量政策，以及 AutoDL（国内网络）访问 R2 的速度与稳定性

### Takeaway
AutoDL 的"学术资源加速"只代理 github/huggingface 等，**不包含 R2/Cloudflare**，且"不承诺稳定性保证"。官方用户协议（自定义服务协议）禁止用专线/VPN 搞跨境经营活动，未提个人下载境外资源。关于 R2 在国内的速度，没有 AutoDL 内的实测，只有零散用户轶事（差异极大）和 Cloudflare 官方对"R2 与中国大陆"的限制说明；稳妥方案是先在 AutoDL 实例上用 rclone 实测一次，并准备国内对象存储（OSS/COS）镜像。

### Cited Findings

**AutoDL 官方网络与加速**
- 学术资源加速内置支持 github.com、githubusercontent.com、githubassets.com、huggingface.co；启用 `source /etc/network_turbo`，关闭 `unset http_proxy && unset https_proxy`；"仅限学术用途"；"不承诺稳定性保证"，遇恶意攻击随时停止；未提及 S3/Cloudflare/境外流量限制，也没说哪些地区或实例类型可用。 — [学术资源加速](https://www.autodl.com/docs/network_turbo/)
- 带宽：同地区共享约 3–10Gbps，不单独计流量费。 — [网络](https://www.autodl.com/docs/network/)
- 自定义服务协议第 2(3)D 条列出的违规情形："自行建立或租用专线（含虚拟专用网络VPN）等其他信道开展跨境经营活动的"；第 3(5) 条重复；另外禁挖矿；该协议未提翻墙/代理/爬虫。该页是"自定义服务"协议，不是主用户协议。 — [自定义服务协议](https://www.autodl.com/docs/service_agreement/)

**R2 本身的官方说法**
- Cloudflare 中国网络文档（页面标注更新于 2026-10-28）脚注："R2 buckets cannot be created within Mainland China"，"custom domains are not supported within Mainland China"，"R2 can be extended into Mainland China through Global Acceleration"。这是关于 Cloudflare China Network 产品的可用性说明，页面未提 Enterprise/ICP 要求，未直接说明从大陆访问海外 R2 的速度或可达性。 — [Cloudflare China Network：Available products](https://developers.cloudflare.com/china-network/reference/available-products)
- r2.dev 公共开发域名"is rate-limited and should only be used for development purposes"；自定义域名才能用 Cloudflare 缓存/WAF；该页未提及中国。 — [R2 Public buckets](https://developers.cloudflare.com/r2/buckets/public-buckets/)

**用户/第三方轶事（非官方，条件各异）**
- 一家 R2 用户（API易）的排障文档称"Cloudflare 在大陆没有被封锁，但国际出口高峰期可能拥塞"，"这是跨境网络的通用问题，不是 R2 特有问题"；建议改 DNS 为 1.1.1.1、并行下载+短超时+重试、必要时用香港/新加坡/带国际优质线路的节点或"镜像到自有对象存储"。该页无发布日期，无实测速度。 — [APIYI：CDN 下载慢](https://docs.apiyi.com/en/faq/cdn-download-slow)
- NodeSeek 测评（搜索摘要，页面抓取 403，未看到原文）：中国移动网络经 Workers 代理，R2 下载"白天最高约100MB/s"，另述测试时间在晚高峰 22:00。 — [NodeSeek：测评 R2 与 B2](https://www.nodeseek.com/post-84753-1)
- LINUX DO 帖（搜索摘要，抓取 403）："官方公共 URL 播放流畅，绑定自定义域名后会卡顿"，回复称"S3 线路有亚太优化"、公共开发 URL 有速率限制；另一指南称官方提供的 r2.dev 文件访问路径在国内无法访问，需绑定自定义域名。均为用户意见。 — [LINUX DO：R2 国内访问速度](https://linux.do/t/topic/952194)
- 另有来源未能定位的搜索摘要称"亚洲桶 10–25MB/s、欧洲桶约 65MB/s、美国桶约 50MB/s，在 VPS 上测"，以及"国内很不稳定"的个别说法；未找到对应页面 URL，不作为已验证事实（见 Gaps）。

### Inferences
- S3 API 端点 `<account>.r2.cloudflarestorage.com` 与 r2.dev/自定义域名走的路径不同；上面的轶事在不同端点、不同运营商、不同时段结果相差一个数量级，因此无法用它们预测 AutoDL 机房（北京/内蒙/重庆/佛山/西北）到 R2 的实际速度。
- 学术加速不覆盖 R2，且协议禁止用 VPN/专线搞跨境经营活动；在 AutoDL 内自己搭代理去访问 R2 既不稳妥、也可能违反规则（协议只列了"跨境经营活动"，个人科研用途是否算未明确；此处是推断，不是官方解释）。
- 最稳的架构：一次性把数据集从 R2 同步到国内对象存储（阿里云 OSS 或腾讯云 COS，选与 AutoDL 机房同地区），训练实例从 OSS/COS 拉取、训练结果再回传；AutoDL 自身的 AutoPanel 也直接支持 OSS/COS/S3 兼容存储，文档示例用的就是 `oss-cn-beijing`、`cos.ap-beijing`。小数据集（几 GB 以内）也可以直接 `rclone copy` 从 R2 拉，先实测。
- 结果回传 R2 方向（国内→境外）上传同样面临跨境出口问题，建议训练产物先存国内（OSS/AutoDL 文件存储），再由另一台服务器/本机异步同步到 R2。

### Gaps
- 没有找到从 AutoDL 实例访问 R2（`*.r2.cloudflarestorage.com`）的任何实测或用户报告，也没有找到 Cloudflare 官方关于 R2 在大陆网络可达性的专门说明。
- NodeSeek 与 LINUX DO 两页抓取 403，只有搜索摘要，细节（测试方法、运营商、日期）无法核实。
- 未读 AutoDL 主用户服务协议（`agreements/`）全文，不能确定对跨境访问的完整表述。
- 未验证 OSS/COS 与 AutoDL 各机房之间的内网/就近传输速度，也没找到 AutoDL 是否对出站带宽有额外限制。

---

## 6. 备选平台：API、价格、权限

### Takeaway
最成熟的备选是 **优云智算 Compshare（UCloud）**：有完整的实例 API（创建/开机/关机/删除/容量检查/价格查询/定时关机），支持 Postpay 和 Spot，但创建接口没有"启动命令"参数，需靠 SSH 或镜像自带自启动；小时价未取到。阿里云 PAI-DLC 提供竞价任务（最高省约 90%），任务制天然"跑完即结束"，但 API 细节未取到。矩池云、恒源云、智星云、潞晨云：没找到公开的 GPU 实例管理 API。腾讯云、PPIO 的细节有限。

### Cited Findings

**优云智算 Compshare（UCloud 体系）**
- 创建：`POST https://api.compshare.cn`，`Action=CreateCompShareInstance`；用 UCloud SDK，公私钥在控制台 API 管理页获取；必填 Region、Zone、GpuType、GPU、CPU、Memory(MB)、CompShareImageId、Disks（须含一块系统盘）。 — [CreateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/createcompshareinstance)
- ChargeType：`Month`/`Day`/`Postpay`（按量）/`Spot`（可能被回收）；Spot 支持 4090、5090、A100、3090、3080Ti、2080Ti、V100S，不支持 4090_48G、A800、H20、P40、2080。 — [CreateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/createcompshareinstance)
- 创建页没有启动命令/脚本参数（`Remark` 只是备注）；文档导航里有单独的"自启动"页，本次未读到内容。 — [CreateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/createcompshareinstance)；[StartCompShareInstance](https://www.compshare.cn/docs/gpus/instance/startcompshareinstance)
- 创建前须调 `CheckCompShareResourceCapacity`；无库存时接口仍返回 `RetCode=0`，对应规格的 `ResourceEnough=false`。 — [CheckCompShareResourceCapacity](https://www.compshare.cn/docs/gpus/instance/checkcompshareresourcecapacity)
- 配额按账户等级：Level0（未充值）1 台、Level1 5 台、Level2 20 台、Level3 50 台、Level4 100 台；账户有未支付订单时无法创建新实例；页面未说明实名要求。 — [CreateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/createcompshareinstance)
- 价格：`GetCompShareInstancePrice`，返回折后/原价/目录价；文档示例是按月：4090×1、16 核 64GB，实例 1068 元、系统盘 120 元（月付示例，不是小时价）；系统盘前 100GB SSD 免费（创建页说明）。 — [GetCompShareInstancePrice](https://www.compshare.cn/docs/gpus/instance/getcompshareinstanceprice)；[CreateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/createcompshareinstance)
- 关机 `StopCompShareInstance`：参数 Region/Zone/UHostId；成功后 Stopping→Stopped；同实例 10 分钟操作锁；抢占式实例默认不能手动关机；页面未写关机后是否计费。 — [StopCompShareInstance](https://www.compshare.cn/docs/gpus/instance/stopcompshareinstance)（另有搜索摘要称需传 `Force=true`，与抓取到的页面正文"未列出 Force 参数"不一致，见 Gaps）
- 删除 `TerminateCompShareInstance` 需实例处于 Stopped；系统盘数据清除，数据盘默认保留，`ReleaseUDisk=true` 同时删除（搜索结果摘要）。 — [TerminateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/terminatecompshareinstance)
- 开机 `StartCompShareInstance` 支持 `WithoutGpuSpec`（`A`=2核4GB，`B`=8核16GB，区分大小写）无卡开机；另有定时关机 `UpdateCompShareStopScheduler`（时间需晚于当前至少 5 分钟）。 — [StartCompShareInstance](https://www.compshare.cn/docs/gpus/instance/startcompshareinstance)；[UpdateCompShareStopScheduler](https://www.compshare.cn/docs/gpus/instance/updatecompsharestopscheduler)
- 第三方文章（2026-04～06，营销性）：智星云 4090 时租约 ¥1.50、恒源云 ¥1.32、阿里云 ¥1.86；2026-08 SMM 报道长三角 4090 最低单卡小时价升至 ¥1.25。优云智算自己的 4090/3090 小时价未取到。 — [搜索结果汇总的第三方比价](https://gitcode.csdn.net/6a29515610ee7a33f27a8b4e.html)；[SMM 2026-08](https://news.metal.com/newscontent/104055758-smm-computing-power-midday-review-the-lowest-price-of-the-4090-in-the-yangtze-river-delta-rises-and-bargaining-power-for)

**阿里云 PAI-DLC（任务制）**
- 竞价资源：按量计费、跟随实时市场价；官方称相比按量付费最高节省约 90%；出价为市场价的 1–9 折离散选项；库存不足时任务等待，被回收则任务失败；通用计算竞价仅支持 PyTorch；DLC 新用户 100 CU·H 免费额度不能抵扣竞价。 — [使用竞价任务](https://help.aliyun.com/zh/pai/use-preemptible-job)；[PAI 计费概述](https://help.aliyun.com/zh/pai/product-overview/billing-overview)
- `CreateJob` 的竞价相关参数、任务结束后资源是否自动释放：本次未取到原文。

**腾讯云 CVM GPU**
- 购买页的 GPU 实例支持包年包月、竞价实例、按量计费；GN7 在选型表里是"✓ 支持"深度学习训练而非"★ 推荐"。未取到竞价价格和 RunInstances 竞价参数。 — [腾讯云 GPU 实例选型](https://cloud.tencent.com/document/product/560/8123)
- 腾讯云 HAI 本次未获得资料。

**PPIO 派欧云（候选，信息有限）**
- REST 创建 GPU 实例，Bearer 鉴权；字段含 `productId`、`gpuNum`(1–8)、`rootfsSize`、`imageUrl`、`command`（启动命令，≤2047 字符）、`envs`、`networkStorages`（云存储挂载，最多 30）、`billingMode`（onDemand/monthly/spot）。页面未写实名要求，也未写国内地域与停止/删除端点。 — [PPIO：创建 GPU 实例](https://ppio.com/docs/gpus/instance/reference-create-gpu-instance)

**矩池云 / 恒源云 / 智星云 / 潞晨云**
- 搜索"矩池云 API"只得到其大模型 API（`token.matpool.com/v1`，OpenAI/Anthropic 兼容），未找到 GPU 机器租用/创建的公开 API。 — [矩池云：其他模型 API](https://matpool.com/supports/doc-API-other-models/)
- 搜索"恒源云 开放 API"未找到恒源云自己的 API/SDK 文档（结果全是别家）。

### Inferences
- 若 AutoDL Pro 因库存/价格不满足，Compshare 是最接近的"纯 API 租卡"替代：有容量检查端点、Postpay、Spot、Stop/Terminate；缺点是创建接口无启动命令，需要 SSH 进入或自启动机制来跑训练，再由实例内 `shutdown` 或外部 API 关机+终止。
- PAI-DLC 是任务制，适合"提交 YOLO 训练 → 完成即释放"，且与阿里云 OSS 同云，容易做 R2→OSS 镜像；但个人用户配置较重（工作空间、OSS、资源配额）。
- 矩池云/恒源云/智星云/潞晨云在公开资料中没有 GPU 实例 API，不适合"全自动创建-释放"的需求（仅"未找到"，并不证明没有）。

### Gaps
- Compshare 4090/3090/5090 的小时价、关机计费规则（计费方式说明页、到期/欠费说明页）、自启动页内容、是否要求实名：未读到。
- Compshare 关于 Spot 手动关机：搜索摘要称 `Force=true`，抓取的页面称无 Force 参数，需实际调用验证。
- 阿里云 PAI-DLC `CreateJob` 参数与价格、腾讯云竞价实例价格、腾讯云 HAI、潞晨云、智星云、矩池云（机器 API）：本次未能取得可靠资料。
- 各备选平台的 FUSE/特权、对 R2 的可达性：均未找到资料。

---

## 7. 合规与账号要求（中国用户）

### Takeaway
AutoDL 为实名制平台：个人（身份证+人脸视频、银行卡+人脸、或其他证件）或企业认证；容器实例Pro API 需要个人实名或企业认证，弹性部署 API 只对企业认证开放。Compshare 以充值等级限制实例配额。

### Cited Findings
- AutoDL 实名认证页：个人有三种方式（身份证：录入信息并拍摄人脸视频；银行卡：卡号+预留手机号+人脸图像；其他个人证件如海外护照、港澳居民来往内地通行证、境外驾照）；企业有银行对公账户认证、企业证件认证（法人或被授权人身份证及授权书）。页面未说明实名在租用/充值前是否强制，也没写外国用户的单独规则。 — [AutoDL 实名认证](https://www.autodl.com/docs/real_name_cert/)
- Pro API：需要个人实名或企业认证；弹性部署 API：需要企业认证（见第 1 节）。 — [容器实例Pro API](https://www.autodl.com/docs/instance_pro_api/)；[弹性部署API](https://www.autodl.com/docs/esd_api_doc/)
- 自定义服务功能：协议要求实名认证后方可使用，服务链接不得转发给第三方，仅供甲方独自用于研究。 — [自定义服务协议](https://www.autodl.com/docs/service_agreement/)
- 登录入口有微信、短信、子账号方式；子账号属于企业高级功能（文档目录中的 `c_user/`）。 — [AutoDL 登录页](https://www.autodl.com/login?url=%2Fmarket%2Flist)；[文档首页](https://www.autodl.com/docs/)
- 账户长期未登录（3 个月）或欠费超 50 元会清空文件存储数据；欠费超 50 元平台可删数据。 — [文件存储](https://www.autodl.com/docs/fs/)；[充值与计费](https://www.autodl.com/docs/price/)
- Compshare：配额随充值等级提高，未充值账号最多 1 台实例；有未支付订单无法创建。 — [CreateCompShareInstance](https://www.compshare.cn/docs/gpus/instance/createcompshareinstance)

### Inferences
- 个人开发者拿个人实名即可走 Pro API；若需要弹性部署（Job 型、库存端点、`cmd` 结束即释放），必须个人先转企业认证，门槛明显更高。
- 开发者 token 是长期凭据，写入自动化脚本需注意保管（官方未写 token 有效期或权限范围，未取到）。

### Gaps
- 是否必须实名才能充值/租卡、境外手机号/外籍用户能否注册：AutoDL 文档未明确。
- 发票、对公转账、API token 权限范围/过期时间：未调研。
- 阿里云、腾讯云的实名/企业要求：本次未调研（国内公有云通常要求实名，但此处没有取得来源，不作事实陈述）。
