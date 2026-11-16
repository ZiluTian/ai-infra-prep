# 附录 C:NCCL 环境变量速查(30 个常用)

> 原则:**生产默认值几乎永远是对的;动任何一个都先在小集群验证(Ch18 §18.4 基线 + 带载基线)。**
> 每项:含义 → 什么时候需要动 → 动错的后果。排障顺序与 Ch18/Ch24 §24.4 一致。

## C.1 调试与日志(排障第一组)

| 变量 | 含义 | 何时动 | 动错的后果 |
|---|---|---|---|
| `NCCL_DEBUG=INFO/WARN/ERROR` | 日志级别;INFO 打全量 | 任何通信问题第一步(Ch28 D27 证据链②) | INFO 日志量极大,长任务别开 |
| `NCCL_DEBUG_SUBSYS` | 子系统过滤(INIT/COLL/P2P/NET/SHM) | 只看某类(如 `P2P,NET`) | 过滤错 = 看不到目标日志 |
| `NCCL_DEBUG_FILE` | 日志落盘路径(每 rank 一文件) | 多机排障(日志别丢终端) | 路径不可写 = 静默丢日志 |
| `NCCL_DEBUG_TIMESTAMP_FORMAT` / `NCCL_DEBUG_TIMESTAMP_LEVELS` | 日志时间戳格式/级别(2.29.7 起,旧名 `NCCL_DEBUG_TIMESTAMP` 已不存在) | 尖刺对齐(Ch22 §22.2) | — |

## C.2 bootstrap 与网络选口(起不来/慢的三类后果之一)

| 变量 | 含义 | 何时动 | 动错的后果 |
|---|---|---|---|
| `NCCL_SOCKET_IFNAME` | bootstrap 用哪张网口 | **多网卡机器必设**(Ch24 §24.4 坑) | 选错 = 初始化失败/跨公网 bootstrap(慢+不稳) |
| `NCCL_SOCKET_NTHREADS` | bootstrap 线程数 | 默认 3 卡住时试 4-8 | 过大 = CPU 抖动 |
| `NCCL_IB_HCA` | 数据面指定 IB 设备(可多值) | rail-optimized 绑定 GPU i ↔ NIC i(Ch18 §18.2) | 漏配 = 走错口,带宽打对折 |
| `NCCL_TOPO_FILE` | 拓扑覆盖文件 | 特殊布线(Ch18 §18.3) | 写错 = 路由劣化,难排查 |

## C.3 IB / RoCE(跨机性能主战场)

| 变量 | 含义 | 何时动 | 动错的后果 |
|---|---|---|---|
| `NCCL_IB_DISABLE=1/0` | 是否禁用 IB/RoCE(2.29.7 核验:现行为 `NCCL_IB_DISABLE`,旧名 `NCCL_IB_ENABLE` 不存在,极性相反) | 验证"是不是网络问题"时先置 1 再看(走 SHM/TCP 对照) | 生产置 1 = 性能崩(PCIe/SHM 路径) |
| `NCCL_IB_GID_INDEX` | RoCE 的 GID(常见 3) | RoCE 集群必配,与交换机侧核对 | 错 = 跨机直接不通 |
| `NCCL_IB_TC` / `NCCL_IB_SL` | DSCP 流量类 / 服务级(QoS) | PFC/QoS 规划(Ch18 §18.2;D33 分优先级) | 与交换机错配 = PFC 风暴/饿死 |
| `NCCL_IB_TIMEOUT=18~22` | IB 超时(2^(n+1)×4.096μs 量级) | 断链"超时失败而非 hang"(Ch29 案例 9) | 太小 = 误判故障;太大 = hang 变长 |
| `NCCL_IB_RETRY_CNT` | 重传次数 | 与 TIMEOUT 配合 | 0 = 一抖即断 |
| `NCCL_IB_QPS_PER_CONNECTION` | 每连接 QP 数 | 多口聚合/拥塞时调 2-4 | 过大 = 交换机资源压力 |
| `NCCL_CROSS_NIC=0/1` | 允许跨 NIC 通信 | rail 绑定失败兜底(默认禁) | 开 = 跨 UPI,带宽 -30% 量级 |
| `NCCL_IB_ADAPTIVE_ROUTING` | IB 自适应路由 | IB 拥塞抖动(交换机开启前提) | RoCE 无效;乱开无意义 |
| `NCCL_IB_PCI_RELAXED_ORDERING` | relaxed ordering | 特定平台测速 | 跨设备正确性问题(个别平台) |

## C.4 GDR(跨机带宽的隐形开关)

| 变量 | 含义 | 何时动 | 动错的后果 |
|---|---|---|---|
| `NCCL_NET_GDR_LEVEL=LOC/PIX/PXB/PHB/SYS(或 0-9)` | GDR 允许级别(按 PCI 拓扑距离逐级放开) | 跨机 busbw 只有预期 ~50%(Ch28 D30 指纹)时排查 | 误关 = 退回 CPU bounce,带宽 5 折 |
| `NCCL_NET_GDR_READ` | 远端 GDR 读 | 与 GDR_LEVEL 配合 | 同上 |
| (IOMMU) `iommu=pt` | 内核直通模式(GDR 硬件前提) | GDR 不生效的第一查项 | 非 NCCL 变量,列此备忘 |

## C.5 算法/协议/通道(压测证明才动)

| 变量 | 含义 | 何时动 | 动错的后果 |
|---|---|---|---|
| `NCCL_ALGO` | 强制 Ring/Tree/NVLS/CollNet(逗号列表) | 对比实验(Ch28 D31);生产默认 = **不设**(NCCL 自动选;`Auto` 不是合法值,2.29.7 会 WARN 且初始化失败) | 小消息强制 ring = 延迟 O(N) 恶化 |
| `NCCL_PROTO` | Simple/LL/LL128 | 小消息延迟调优(LL/LL128);生产默认 = 不设 | 大消息用 LL = 带宽减半量级 |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | 通道数(上下界;2.29.7 核验:`NCCL_NCHANNELS` 已不存在,P2P 另有 `NCCL_MIN/MAX_P2P_NCHANNELS`) | 带宽打不满且 topo 正常时(Ch18 §18.4 故障 4) | 过大 = SM/带宽争抢,反降 |
| `NCCL_BUFFSIZE` | 每通道 buffer | 大消息吞吐微调 | 过小 = 流水线断;显存代价 |
| `NCCL_NVLS_ENABLE=1/0` | NVSwitch SHARP 规约 | H100 节点内小-中消息(Ch28 D31/D11) | 老硬件开 = 无收益或报错 |
| `NCCL_COLLNET_ENABLE=1` | IB SHARP(跨节点硬件规约) | IB 交换机支持 SHARP 时 | 不支持 = 初始化失败 |
| `NCCL_P2P_LEVEL` / `NCCL_P2P_DISABLE` | P2P 范围/禁用 | NVLink 降级排查(强制对照) | 生产禁 P2P = 性能崩 |
| `NCCL_SHM_DISABLE` | 禁共享内存(同节点 P2P 备选路径) | 对照实验 | 禁 = 同节点走 PCIe/显存拷贝 |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | 通信错误异步上报(不死等)。**注意:这是 PyTorch(ProcessGroupNCCL)变量,不是 NCCL 库变量**;旧名 `NCCL_ASYNC_ERROR_HANDLING` 在 torch 2.13 已打 deprecation 警告 | 生产容错(Ch29 案例 9:超时而非 hang) | 关 = 错误 hang 到 watchdog |
| `NCCL_IGNORE_DISABLED_P2P=1` | 抑制"P2P is disabled between GPUs"INFO 刷屏(2.29.7 新增核验) | 消费卡(4090/5090 节点内 P2P 被驱动禁用,NCCL 回退 SHM,dmesg 级日志噪音) | 纯日志控制,不影响性能 |
| `NCCL_OOB_NET_IFNAME`? (旧版) / 默认走 `NCCL_SOCKET_IFNAME` | OOB 通道选口 | 多口 | 与 bootstrap 混 = 排障迷惑 |

## C.6 排障速查(症状 → 先看谁)

| 症状 | 第一组 |
|---|---|
| 起不来(初始化失败) | `NCCL_DEBUG=INFO` + `NCCL_SOCKET_IFNAME` + `NCCL_IB_HCA` |
| 跨机慢(50% 带宽) | GDR 组(`NCCL_NET_GDR_LEVEL`)+ `NCCL_IB_HCA` 绑定 |
| 偶发 10× 尖刺 | `NCCL_IB_TC/SL`(QoS)+ PFC 计数(交换机侧,Ch18 §18.4 故障 1) |
| 断链后 hang | `NCCL_IB_TIMEOUT` + `TORCH_NCCL_ASYNC_ERROR_HANDLING` |
| 节点内 busbw 异常 | `NCCL_P2P_*`(NVLink 对照)+ `nvidia-smi topo -m`(Ch28 D27);消费卡先确认 P2P 是否被禁(INFO 的 "P2P is disabled" → 回退 SHM,带宽减半属正常,不是故障) |

## C.7 版本核验(2026-11-16,NCCL 2.29.7+cuda13.2,linuxgpu01 4×RTX5090)

- 本表 30 项逐一对照 libnccl 2.29.7 实际编译的变量名(strings,共 247 个):**26 项一致**。
- **修正 4 项**:`NCCL_DEBUG_TIMESTAMP` → `NCCL_DEBUG_TIMESTAMP_FORMAT`/`_LEVELS`;`NCCL_IB_ENABLE` → `NCCL_IB_DISABLE`(极性相反);`NCCL_NCHANNELS` → `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`;`NCCL_ASYNC_ERROR_HANDLING` → `TORCH_NCCL_ASYNC_ERROR_HANDLING`(PyTorch 变量,非 NCCL 变量)。
- **新增 1 项**:`NCCL_IGNORE_DISABLED_P2P`(消费卡 P2P 禁用场景)。
- **值合法性**:`NCCL_ALGO/NCCL_PROTO=Auto` 不是合法 token(2.29.7 WARN + 初始化 invalid usage);生产默认 = 不设。
- **实测行为**(验证机,非书中实测口径):5090 节点内 P2P 被禁用 → NCCL 走 `SHM/direct/direct`;all_reduce busbw 峰值 ≈ 34.8 GB/s(8G 消息),alltoall ≈ 26.9 GB/s(N=2,torchrun,NCCL 2.29.7)。
