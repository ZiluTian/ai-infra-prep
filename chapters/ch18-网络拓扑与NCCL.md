# 第 18 章 网络拓扑与 NCCL

> 难度 ★★★★ | 目标 ~30 页 | 前置:Ch14
> 分层:18.1-18.2 与 18.4 基础项 [基础];18.3 与 18.4 全部 [进阶]。
>
> **读完本章你能做什么**
> 1. 读懂 `nvidia-smi topo -m` 与 rail-optimized 布线,说出"GPU:NIC = 1:1"的含义
> 2. 对比 IB NDR 与 RoCEv2 的取舍(成本 / 可靠性 / 运维),给出选型判据
> 3. 讲清 NCCL 的拓扑探测、ring/tree/NVLS 算法选择(小消息/大消息),列出关键环境变量(附录 C)
> 4. 用 nccl-tests 判读 busbw,给出"节点内/跨机正常值",列出四类常见故障的排查路径
> 5. [进阶]解释 P/D 的 KV 传输与 NCCL 集合通信的带宽竞争及隔离手段

## 18.1 节点内拓扑

**层级**:

```
PCIe Gen4 x16 (~25-32 GB/s 单向)  <  NVLink4 (H100: 18 links × 50GB/s 双向 = 900GB/s/GPU)
NVSwitch: HGX 节点内 full-bisection —— 任意两卡之间都是 900GB/s(无降速路径)
```

**`nvidia-smi topo -m` 读法(面试要会认符号)**:

- `NV#`:NVLink 直连,# = link 数(8 卡 HGX 两两 `NV18`)
- `PIX/PHB`:同一 PCIe switch / 同一 root complex;`SYS`:跨 CPU socket(走 **UPI**(Inter-Processor Link,两个 CPU socket 间的互联,带宽低于 socket 内 L3),**慢一截,NUMA 敏感**)
- `P2P`:支持 P2P 拷贝(GDR 的前提之一)
- **典型坑:消费卡(4090)无 NVLink** → 矩阵全 `SYS/PIX` → TP 只能走 PCIe(~1/18 带宽)→ TP2 勉强、TP>2 不可行。**"这台机器能不能 TP8" 第一步就看这张表**
- 同节点 8 卡若出现 `SYS` 跨 socket:GDR/拷贝路径经过 UPI,带宽打折 + 延迟抖动 → 检查 CPU 亲和与 NIC 挂载(NUMA,18.4)

## 18.2 节点间网络

**InfiniBand NDR**:单口 400Gb/s ≈ 50GB/s;**8 卡节点 = 8 个 IB 口(GPU:NIC = 1:1)**,标准 rail-optimized 布线——GPU i 只与 NIC i 直连(同 rail),集合通信尽量留在本 rail 内完成,跨 rail 走 leaf switch。优点:原生无损(信用流控)、自适应路由、生态成熟(NCCL 一等公民)。缺点:贵(卡 + 交换机 + 线缆)。

**RoCEv2(RDMA over Ethernet)**:

- 无损靠**配置**实现:PFC(Priority Flow Control,逐优先级暂停帧,逐跳保证不丢)+ ECN(Explicit Congestion Notification,拥塞标记,触发 DCTCP/DCQCN 拥塞控制降速)
- 优点:以太网交换机便宜、生态通用、可复用现有 DC 网络
- 缺点:**PFC 风暴**(一个 pause 帧级联 → 整条优先级队列停摆,表现为偶发巨量延迟尖刺)、丢包即重传(延迟爆炸)、MTU/QoS 配置面大
- 取舍口诀:**规模大、SLA 高、运维团队懂 IB → IB;成本敏感、以太网运维成熟 → RoCE,但必须做 PFC/ECN 调优与压测**

**GPU:NIC 配比**:标准 1:1(8 卡 8 口)。少配(8 卡 4 口)→ 大消息集合通信聚合带宽腰斩,decode 的 all-to-all(EP)与 P/D 的 KV 传输最先受损。

## 18.3 NCCL 深究

**拓扑探测(启动期)**:NCCL 构建通信图——节点内 P2P(NVLink/PCIe)通道 + 节点间 NET(IB/RoCE)通道;bootstrap 走 TCP(由 `NCCL_SOCKET_IFNAME` 指定网卡),数据面走探测出的最优路径。`NCCL_DEBUG=INFO` 打印选定的 transport 与 channel 数——**排障第一命令**。

**算法选择(核心:消息大小决定算法)**:

| 算法 | 特性 | 适用 |
|---|---|---|
| **ring** | 带宽最优:busbw 系数 2(N−1)/N;延迟 O(N) | **大消息**(默认首选) |
| **tree** | 延迟 O(log N);带宽系数略低 | **小消息**(μs 地板主导) |
| **NVLS**(NVLink SHARP) | NVSwitch 网内硬件规约,节点内免软件多跳 | 节点内小-中消息(H100+ NVSwitch) |
| **CollNet** | IB 侧 SHARP(交换机做规约) | 跨节点大消息(硬件支持时) |

NCCL 按消息大小**自动切换**(阈值与拓扑有关,可用 `NCCL_ALGO` 强制)。**面试标准答案:"小消息 tree/NVLS(延迟主导),大消息 ring(带宽主导),NCCL 自动选,生产别手调除非压测证明"**。

**GPUDirect RDMA(GDR)**:NIC 直接读写 GPU HBM,绕过 CPU DRAM 的 bounce buffer;前提是 IOMMU/peer-memory 可用。GDR 失效(驱动/IOMMU/`NCCL_NET_GDR_LEVEL` 配置)→ 退回 CPU 拷贝,跨机带宽打 5 折以下(18.4 排障项)。

**关键环境变量(速查见附录 C)**:

```
NCCL_DEBUG=INFO|WARN        排障第一开关
NCCL_SOCKET_IFNAME=eth0     bootstrap 用哪张网口(多网卡环境必设)
NCCL_IB_HCA=mlx5_0,mlx5_1   指定 IB 设备(8 口节点常用)
NCCL_IB_GID_INDEX / NCCL_IB_TC / NCCL_IB_SL   RoCE 的 GID 与 DSCP/QoS
NCCL_P2P_DISABLE=1          关 P2P(排障对照用,性能大跌)
NCCL_SHM_DISABLE=1          关共享内存(同上)
NCCL_IB_DISABLE=1           强制走 TCP(性能崩塌,仅对照)
NCCL_MIN/MAX_NCHANNELS / NCCL_BUFFSIZE   通道数(上下界)与缓冲(手调风险高,附录 C)
NCCL_NET_GDR_LEVEL          GDR 启用级别
```

## 18.4 测量与排障

**nccl-tests 基线(每套集群上线必跑)**:

```bash
# mpirun:MPI 运行时,-np = 总进程数(此处每 GPU 一进程),-ppn = 每节点进程数
# 节点内 8 卡
mpirun -np 8 ./all_reduce_perf -b 8M -e 8G -f 2 -g 1
# 跨机(每节点 8 卡)
mpirun -np 64 -ppn 8 ./all_reduce_perf -b 8M -e 8G -f 2 -g 1
```

**判读:busbw 而非 algbw**(algbw = 消息量/时间,nccl-tests 原始输出;busbw 按 ring 系数 2(N−1)/N 归一,不同拓扑/算法间可比):

| 链路 | 单链带宽 | ring 期望 busbw(×0.875,N=8) |
|---|---|---|
| NVLink 900GB/s | 900 | **~780 GB/s**(节点内) |
| IB NDR 单口 50GB/s | 50 | 单链 ~44 GB/s;8 口 rail-optimized 大消息聚合后按 NIC 数近似线性上升 |

**低于期望 20% 以上 → 拓扑/配置有问题**(不是"正常波动")。

**四类常见故障(排查路径)**:

1. **RoCE PFC 风暴**:症状 = nccl-tests 偶发 10× 延迟尖刺、压测不稳;查 PFC 计数(`ethtool -S` / 交换机 pause 计数);修 = ECN + DCTCP 调参或换优先级隔离
2. **GDR 失效退回 CPU 拷贝**:症状 = 跨机 busbw 只有 GDR 预期的 ~50%;`NCCL_DEBUG=INFO` 看 transport 里 GDR 状态;查 IOMMU(`iommu=pt`)、驱动版本、`NCCL_NET_GDR_LEVEL`
3. **NUMA/跨 socket**:NIC 挂在非目标 socket → 跨 UPI;`numactl --hardware` 确认 GPU/NIC 亲和,绑定调度
4. **MTU 不一致**:IB active_MTU 4096→1024(`ibstat` 查);RoCE 全链路(两端 NIC + 每台交换机)都要 9000 MTU,一处 1500 即分片 → 小消息尚可、大消息崩

**带宽竞争(生产特有问题)**:同一 IB 口同时跑 ① NCCL 集合通信(TP/EP)② P/D 的 KV 传输(NIXL 类,RDMA)→ **互相挤占** → 表现为随机尖刺。手段:专用 NIC 隔离(KV 传输走独立口)、QoS/DSCP 分级、P 池与 D 池错时传输(Ch9)。

> **踩坑框:"busbw 达标了,服务还是慢"**
> nccl-tests 是**空载集合通信**;生产里通信与计算 overlap(Ch15)、与 KV 传输竞争(本节)、与调度间隙叠加。通信基线合格是**必要条件不是充分条件**——生产时延问题回到 nsys timeline(Ch10/Ch20)定位,别停在 nccl-tests。

## 动手

**实验 18.1(半天,多卡 + 多机)**:

1. nccl-tests 节点内 8 卡 / 跨机基线(8 级消息大小,记录 busbw 曲线)
2. 对照实验:`NCCL_P2P_DISABLE=1`、`NCCL_NET_GDR_LEVEL=0`(或等效)分别重跑,记录退化幅度(量化"P2P/GDR 值多少钱")
3. 交付物:该集群通信基线表(作为 Ch23 容量规划与 Ch26 实验的基准)

## 章末自测

1. `nvidia-smi topo -m` 里 `NV18`、`PIX`、`SYS` 各意味着什么?4090 集群能 TP 几卡?
2. IB 与 RoCE 的"无损"分别靠什么?PFC 风暴的症状与修法?
3. NCCL 为什么"小消息 tree、大消息 ring"?NVLS 在什么硬件上出现?
4. ring all-reduce 的 busbw 期望 = 0.875 × 链路带宽(N=8)——推导这个 0.875
5. GDR 失效的症状、排查命令、影响幅度?
6. P/D 的 KV 传输为什么和 NCCL 抢带宽?两种隔离手段?

## 面试钩子

- 支撑 Ch28-D27~D33:"NCCL 怎么确认走的 ring?busbw 多少算正常?""RoCE PFC 风暴怎么排查?""GDR 是什么、怎么确认开了?"
- Ch29 案例 5(P/D 集群网络预算)的通信基线来自本章。
