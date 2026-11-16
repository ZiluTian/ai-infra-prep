# Part VII 面试篇:把前六 Part 转化为面试表现

> **目标**:题型覆盖 基础原理 / 框架深挖 / 分布式 / CUDA / 系统线上 / 系统设计。答案给"要点 + 深度分级(3 分钟版 / 10 分钟版)",不写逐字稿——逐字背答案在面试里是减分项。
> **难度**:★★★~★★★★★ | **合计** ~120 页

---

## 第 28 章 分类题库与参考答案

> ~60 页 | 前置:全部
> 格式:题目 → 答题要点(分条)→ 常见追问 → 难度 → 对应章节。每题标注"3 分钟版要点"与"10 分钟版展开点"。

### A 基础与指标(~15 题)
prefill/decode 差异、KV 精算(手算)、Roofline 与算术强度、TTFT/TPOT/ITL 定义、GQA/MQA、batching 三级跳、goodput、负载模型、为什么 decode 是 memory-bound、KV 增长对 batch 上限的影响……

### B 框架深挖(~45 题,主战场)
- PagedAttention:实现/代价/为什么快;block 大小怎么选
- vLLM v0 vs v1:为什么重写、双进程、CUDA graph 分桶
- 调度器:waiting/running 队列、token 预算、抢占 recompute vs swap
- Chunked prefill:chunk 大小权衡、与 P/D 的边界
- 投机解码:无损性证明梗概、EAGLE vs Medusa vs n-gram、高 batch 失效
- 量化:GPTQ/AWQ/SmoothQuant 原理、W4A16 实现、FP8 E4M3/E5M2、KV 量化损失
- 前缀缓存:哈希 block、radix tree、LRU 逐出
- P/D 分离:KV 传输成本精算、何时值得、动态配比
- SGLang:RadixCache、overlap scheduling、结构化输出
- TRT-LLM:in-flight batching、plugin、何时选它

### C CUDA 与算子(~25 题)
FA v1→v3 演进(含 Hopper 特性)、FlashInfer 解决什么、FlashDecoding/split-K、Triton GEMM 怎么写、CUTLASS FP8 GEMM、DeepGEMM(TMA multicast)、CUDA Graph 本质与限制、MoE all-to-all 两种模式、grouped GEMM、H200 vs H100 带宽影响、kernel launch 开销、occupancy 与 bank conflict

### D 分布式(~30 题)
TP 通信量推导、为什么 TP ≤ 8、SP vs CP、Ring vs Ulysses 选型、PP bubble 计算、EP 负载均衡与 DeepEP、NCCL ring/tree 选择、busbw 期望值、RoCE vs IB、NVLS all-reduce、H200/B200 选型、多机 gang 调度与故障域

### E 系统与线上(~30 题)
OOM 分类与排查、p99 尖刺排查、GPU util 低的系统定位、preemption 代价、SLO 定义与 burn rate、goodput 最大化、容量规划从流量到卡数、灰度判健康与回滚、监控指标清单、故障注入设计

> **使用建议**:路线 A 读者先做 A/B/E,路线 B 读者全做并在每题后写"我的真实答案"(用自己项目校准)。

---

## 第 29 章 系统设计案例(10 例)

> ~50 页 | 前置:全部
> 每案例统一结构:**需求(带数字) → 需求分解 → 容量估算(套公式) → 架构设计 → 权衡讨论 → 预期追问**。给出"参考架构"而非唯一解,并列出评审点。

1. **100B+ MoE 生产级 serving**:64 卡、TPOT p99 < 60ms、高并发
2. **低延迟 chat**:p99 TTFT < 50ms、100 req/s、短输出
3. **长上下文 1M RAG serving**:KV 成本、prefill CP、检索命中前缀复用
4. **成本降 10×**:H100 FP16 基线 → 量化 + P/D + 硬件选型的组合拳
5. **P/D 分离集群与动态配比**:KV 传输链路、网络预算
6. **多租户 SLO 隔离**:noisy neighbor、KV 隔离、过载 shed
7. **128K 上下文 prefill 集群**:CP + prefill-only 副本
8. **MoE 热 expert 治理**:放置 + 复制 + 请求路由
9. **推理集群容错**:节点挂掉后的语义(KV 丢失)、恢复时间目标
10. **容量与成本规划**:从季度流量预测到采购决策(含 $/百万 token 模型)

---

## 第 30 章 简历与项目表达

> ~10 页 | 前置:Ch25-27(实验交付物即素材)

### 大纲
1. **推理优化项目怎么写进简历**:STAR + 量化指标(TTFT −X%、吞吐 +Y%、$/百万 token、卡数);一段"可面试"的项目描述范本
2. **源码级经验怎么讲**:讲"改了什么、在哪定位的、拿到什么数据",而不是"我读过 vLLM 源码"
3. **常见简历红旗与修正**:只有"部署过 vLLM"、无数据、无权衡、指标无法复现
4. **模拟面试流程**:30min 自我介绍 → 60min 项目 deep-dive → 45min 系统设计(Ch29 抽题)→ 白板/编码(Triton kernel 或调度模拟器)
5. **反问清单**:问面试官什么(团队 stack、集群规模、SLO、当前最大瓶颈)——以及这些答案如何反推岗位真实水平
