# Part IV 分布式推理篇:从"TP=8"到能推通信量、选拓扑、调 NCCL

> **目标**:读者能为任意模型规模推导各并行策略的通信量、结合拓扑做选型、定位通信瓶颈。
> **难度**:★★★~★★★★★ | **合计** ~170 页

---

## 第 14 章 并行策略总览与决策

> 难度 ★★★ | ~25 页 | 前置:无

### 大纲
1. **每种策略一张"卡片"(定义 + 通信 + 延迟特性 + 故障域)**
   - TP:层内切权重,每层 2 次 all-reduce,限节点内
   - PP:层间切,点对点 send/recv,有 bubble、怕 straggler,可跨机
   - DP:整模复制,KV 独立,抗过载
   - EP:MoE 专家分布
   - SP:序列维切激活(Megatron SP / Ulysses)
   - CP:长上下文序列维 attention(ring)
2. **通信量公式(本章定量核心)**
   - TP all-reduce ≈ `2 × bs × seq × hidden × bytes`(每层、×2 次)
   - EP all-to-all ≈ `bs × seq × topk × hidden × bytes / N`
   - PP 点对点 ≈ `micro_bs × hidden × bytes × 阶段数`
   - 每式配手算例子(70B、671B MoE)
3. **决策框架(决策树)**
   - 节点内:TP ≤ NVLink 域(≤8)
   - 跨机:EP(MoE)/ PP(稠密大模型)/ DP(过载)
   - 长上下文:CP / SP
   - 工作例:70B → TP8;671B MoE → EP×TP 混合;1M 上下文 → prefill CP
4. **混合并行与 gang 调度**
   - 多维并行组合(TP×EP×CP)的层级关系
   - K8s gang 调度(Volcano)、一个节点挂 = 整个并行组挂

### 分层
- [基础] 1-3;[进阶] 4。

### 动手 / 面试钩子
- 动手:"通信量计算器"脚本(输入模型与并行配置,输出每步通信量,Ch26 复用)。
- 面试钩子:支撑 Ch28-D01~D06("怎么定 TP/PP/EP?推导 TP 通信量")。

---

## 第 15 章 Tensor Parallelism 深度

> 难度 ★★★★ | ~30 页 | 前置:Ch14

### 大纲
1. **Megatron TP 布局**
   - ColumnParallel(QKV / MLP 上投影)+ RowParallel(输出 / MLP 下投影)
   - attention head 切分;GQA 下 kv heads < TP → 复制(vLLM 的处理方式)
2. **通信模式深究**
   - 每层 2 次 all-reduce;替换为 all-gather + reduce-scatter(即 SP,通信量等价但延迟摊开、激活显存下降)
   - 为什么 TP 不宜 > 8:all-reduce 延迟 × 层数 × 2;NVLink ~900GB/s vs IB 单口 ~50GB/s
3. **overlap**
   - 通信-计算 overlap:层切块、与下一块计算重叠
   - layerwise overlap(DeepSeek 实践):expert GEMM 与 all-to-all 重叠
4. **TP 源码走读**
   - vLLM / SGLang:parallel state、all-reduce 封装、forward 中通信发生位置
   - 通信与 CUDA graph 的兼容(哪些通信可以进 graph)

### 分层
- [基础] 1-2;[进阶] 3-4。

### 动手 / 面试钩子
- 动手:测 TP2/TP4/TP8 的 decode 延迟 scaling,验证通信成本的超线性增长。
- 面试钩子:支撑 Ch28-D07~D14("TP 一层几次通信?为什么跨机做 TP 不划算?")。

---

## 第 16 章 Expert Parallelism(MoE)

> 难度 ★★★★★ | ~30 页 | 前置:Ch13、Ch14
> 注:整章 [进阶]。

### 大纲
1. **EP 数据流(源码级)**
   - local router → 全局路由表 → dispatch all-to-all → 本地 expert 计算 → combine all-to-all
   - vLLM / SGLang 的 EP 实现点(DeepEP 集成、路由表在哪算)
2. **DeepEP 两种模式**
   - normal(节点内、NVLink、逐 token、高吞吐)vs low-latency(跨节点、RDMA write、分组 512、IBGDA)
   - 使用场景:prefill 用 normal,decode 用 low-latency
3. **负载均衡**
   - token 分布:Zipf 式热 expert
   - 静态:expert 放置(拓扑感知,热 expert 放高带宽域)
   - 动态:复制、capacity、迁移(成本与触发条件)
   - SLO 影响量化:热 expert straggler → p99(数值例子)
4. **EP 规模实战**
   - 节点内 EP8 vs 跨机 EP64:延迟/吞吐对比
   - DeepSeek 生产形态示例(EP×TP 混合,引用公开资料)

### 动手 / 面试钩子
- 动手:单机 EP8 vs TP8 对比(Qwen3-30B-A3B 或 DeepSeek 轻量版),测 all-to-all 占比(Ch26)。
- 面试钩子:支撑 Ch28-D15~D20("EP 与 TP 通信有何不同?MoE 负载怎么均衡?")。

---

## 第 17 章 Pipeline 与 Context Parallelism

> 难度 ★★★★ | ~30 页 | 前置:Ch14

### 大纲
1. **推理中的 PP**
   - bubble:单请求时 PP 多 (stages−1) 跳延迟;micro-batch / in-flight 摊 bubble
   - 为什么推理比训练少用 PP,但跨机大模型又不得不选
   - straggler:最慢阶段决定全局
2. **长上下文:SP vs CP**
   - Ulysses(all-to-all、切 head 维)vs Ring(P2P、KV 轮转)
   - 通信量对比与选型(带宽 vs 延迟、与 TP 的组合)
   - 2D 并行:CP × TP 的层级关系
3. **1M 上下文实战**
   - DeepSeek MLA + 1M:KV 压缩 + prefill CP
   - prefill-only CP:P/D 分离 + prefill 集群(呼应 Ch9)
   - 显存:KV/卡 = 总量 / CP size(回代 Ch3 公式验证)

### 分层
- [基础] 1-2(概念);[进阶] 全章(含 2D 并行推导)。

### 动手 / 面试钩子
- 动手:两节点 ring attention 实验,或长 prefill 的 CP 开/关耗时对比(Ch27)。
- 面试钩子:支撑 Ch28-D21~D26("PP bubble 多大?Ring 与 Ulysses 怎么选?")。

---

## 第 18 章 网络拓扑与 NCCL

> 难度 ★★★★ | ~30 页 | 前置:Ch14

### 大纲
1. **节点内拓扑**
   - PCIe / NVLink / NVSwitch(HGX H100:每卡 18 条 NVLink、full-bisection 900GB/s)
   - `nvidia-smi topo -m` 读法(含 4090 无 NVLink 的坑)
2. **节点间网络**
   - InfiniBand NDR:400Gb/s 口、rail-optimized 布线
   - RoCEv2:无损以太网(PFC / ECN)、成本与可靠性取舍
   - 选型:规模、预算、运维能力
3. **NCCL 深究**
   - 拓扑探测:P2P / SHM / NET;ring(带宽最优)vs tree(延迟最优)
   - 算法选择:小消息 tree、大消息 ring
   - 关键环境变量(附录 C 速查):`NCCL_P2P_*`、`NCCL_IB_*`、`NCCL_ALGO`、`NCCL_MIN/MAX_NCHANNELS`、`NCCL_DEBUG`
   - GPUDirect RDMA:绕开 CPU 拷贝
4. **测量与排障**
   - nccl-tests:busbw vs algbw,ring 期望 ≈ 0.875 × 链路带宽
   - 常见故障:RoCE PFC 风暴、GDR 失效退回 CPU 拷贝、NUMA、MTU 不一致
   - 通信与计算 / KV 传输(P/D)的带宽竞争

### 分层
- [基础] 1-2、4 基础项;[进阶] 3-4 全部。

### 动手 / 面试钩子
- 动手:nccl-tests 跑基线(节点内/跨机),故意关 P2P / GDR 观察性能退化幅度。
- 面试钩子:支撑 Ch28-D27~D33("NCCL 怎么确认走的 ring?busbw 多少算正常?")。

---

## 第 19 章 集群实战与硬件选型

> 难度 ★★★★ | ~25 页 | 前置:Ch14-18

### 大纲
1. **硬件对比与 serving 影响**
   - H100 vs H200 vs B200:算力 / 带宽 / 精度 / 互联对照表(附录 B)
   - H200 → decode 类 +30-40%(带宽);B200 → FP4 MoE
   - "该买什么卡"决策表(延迟导向 vs 成本导向)
2. **集群形态**
   - 8 卡 HGX、多机 rail-optimized、GPU:NIC 配比
   - gang 调度与故障域(一节点挂 = 整个 TP 组挂)
3. **集群级负载均衡**
   - 全局请求路由:前缀感知 consistent hashing、副本级负载
   - MoE 热 expert → 请求级路由(流向有富余 expert 容量的副本)
   - autoscaling:基于队列深度(KEDA)
4. **成本与效率**
   - $/百万 token、SM 活跃率、空闲成本
   - 混部与抢占(训练/推理共用集群)

### 分层
- [基础] 1;[进阶] 2-4。

### 动手 / 面试钩子
- 动手:集群利用率看板(DCGM + Grafana,Ch23 复用)。
- 面试钩子:支撑 Ch28-D34~D38 与 Ch29 多数案例("H100 vs H200 你的部署怎么选?")。
