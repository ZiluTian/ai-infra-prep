# 第 17 章 Pipeline 与 Context Parallelism

> 难度 ★★★★ | 目标 ~30 页 | 前置:Ch14
> 分层:17.1-17.2 [基础](概念);[进阶] 全章(含 2D 并行与 1M 实战推导)。
>
> **读完本章你能做什么**
> 1. 推导推理场景 PP 的 bubble 条件:为什么"in-flight 请求足够多"时 PP 才不亏
> 2. 对比 Ulysses 与 Ring(CP)的通信形态,给出"节点内 Ulysses、跨节点 Ring"的选型判据
> 3. 用 Ch3 公式验证 1M 上下文的 KV/卡 是否装得下(CP 切分后回代)
> 4. 说出 prefill-only CP 集群 + P/D 分离中"KV reshard"这个工程痛点
> 5. [进阶]推导 CP × TP 的 2D 并行层级与通信

## 17.1 推理中的 PP

**结构**:80 层切成 S 个阶段(如 PP8 = 每阶段 10 层),token 的激活在阶段间点对点传递。

**单请求视角(低并发时 PP 的代价)**:

```
一个 token 每步要串行穿过 S−1 个边界 → 延迟 += (S−1) × 跳延迟(μs 级)
且阶段 i 必须等阶段 i−1 → 全步时间 = Σ 各阶段时间(无重叠)
```

PP8、decode 单请求:相比无 PP,**纯增加 7 跳 + 阶段串行**——低并发下 PP 是净亏。

**高并发视角(bubble 被 in-flight 填满)**:continuous batching 下,不同请求处在不同阶段——阶段 i 算请求 j 时,阶段 i+1 算请求 j−1。**PP 在推理中可用的条件:in-flight 请求数 ≫ S**(每阶段队列不空)。此时每步时间 ≈ max(各阶段时间)+ 通信,与单卡 forward 同量级。

**定量(Ch14 (14-2))**:PP8、bs=256、hidden 8192、FP16:每边界 256×8192×2B = 4MB,每步 7 边界 = 28MB,IB 50GB/s → ~0.5ms/步——可接受(对比跨机 TP 的 1-2ms/步,Ch15)。**PP 是"跨机稠密大模型"的默认选择,因为它是跨机通信代价最小的切法**(小消息 × 少次 × 低频)。

**straggler(PP 特有的放大效应)**:全流水线节奏 = 最慢阶段。decode 各阶段步时间若有 ±20% 抖动(stragglers:某阶段赶上热 expert、某请求长 chunk),整条流水线按最慢走 → **PP 对负载不均比 TP 敏感**,调度器要按"阶段负载"而非"请求数"均衡(进阶题)。

**何时选 PP(决策口诀)**:单节点装不下 + 稠密 → PP;单节点装不下 + MoE → EP 优先(Ch16,expert 天然可切,比 PP 更贴合数据);PP 与 EP/TP 常**混用**(Ch14 层级:PP 在阶段间,每阶段内再 TP/EP)。

## 17.2 长上下文:SP vs CP

prefill 1M token 的单卡困境:① FLOPs ≈ 2 × 1M × 激活参数(37B 级)≈ 7.4×10¹⁶ → 单卡有效 ~700TFLOPS 时 ≈ **100s 量级**;② KV 显存(17.3 算账)单卡装不下。解法:**把序列切到多卡**。

### Ulysses(all-to-all,切 head 维)

```
P 个设备,每设备持 N/P 个 token 的全部 head
① all-to-all:交换后,每设备持全部 N 个 token 的 H/P 个 head
② 本地 attention:本设备的 head 子集 × 全序列(计算完整)
③ all-to-all 换回
```

- 通信:每层 **2 次 all-to-all**,每设备消息量 ≈ `2 × N × hidden` 量级(与 P 无关,总量守恒)
- 优点:attention 计算本地完成、无 P 步串行;带宽利用率好
- 约束:**all-to-all 吃带宽 → 限节点内 NVLink**;H 必须被 P 整除 → **P ≤ head 数**(且 P 通常 ≤ 8)

### Ring / CP(P2P,KV 轮转)

```
P 个设备,每设备持 N/P 个 token(全部 head)
for step in 1..P:
    本设备用自己的 q 块 × 手上这份 KV 块算部分 attention(online softmax 合并,Ch11)
    同时把 KV 块 P2P 发给下一家、从上一家收 KV 块   ← 通信与计算重叠
```

伪代码:

```python
# ring attention:每设备持 q 块 (N/P, H, d);KV 块 (N/P, H_kv, d) 沿环轮转
o_acc, m_acc, l_acc = None, None, None      # online softmax 状态(Ch5/Ch11)
kv_local = kv_chunk[rank]
for r in range(P):
    q, kv = q_local, kv_local               # 本块:完整计算
    kv_local = p2p_send_recv(kv_local)      # 发下一家 / 收上一家(与下面计算重叠)
    part_o, m_new, l_new = attn_chunk(q, kv_local)
    o_acc, m_acc, l_acc = merge_online_softmax(o_acc, m_acc, l_acc, part_o, m_new, l_new)
o = o_acc / l_acc
```

- 通信:每层 P 步 P2P,每步 `2 × (N/P) × H_kv × d × bytes`(K+V);**总搬运量 = 每份 KV 绕环一圈 ≈ 2 × N × H_kv × d**(与 Ulysses 同量级),但**串行 P 步、每步小消息**
- 优点:P2P 延迟可被计算掩盖;不要求 head 整除;**P 可以跨节点(RDMA)**——这是 CP 能做到 P=16/32/64 的原因
- 代价:小消息 × P 步的延迟敏感(拓扑质量直接进时延)

### 选型(面试标准答案)

| | Ulysses | Ring(CP) |
|---|---|---|
| 通信原语 | all-to-all ×2/层 | P2P ×P 步/层(可重叠) |
| 拓扑 | **节点内 NVLink** | 节点内 + **跨节点 RDMA** |
| 约束 | H % P == 0 | 无(序列可任意切) |
| P 上限 | ~8 | 16-64+ |
| 延迟特征 | 带宽主导 | 延迟主导(每步 μs 地板 × P) |

**口诀:节点内切序列用 Ulysses(带宽足、无串行),跨节点切序列用 Ring(可重叠、能上 RDMA)**。

### 2D 并行:CP × TP 的层级(进阶推导)

长 prefill 的完整形态:**TP 切 head(GEMM 侧)+ CP 切序列(attention 侧)**:

```
节点内: TP8 —— QKV 投影 GEMM 列切,每卡 1/8 的 head
跨节点: CP4 —— 序列切 4 份
attention: 每卡算 (1/4 序列) × (1/8 head),KV 按 CP 环轮转
GEMM 通信: all-reduce(每层 2 次,节点内 NVLink)
CP 通信:   KV 轮转 P2P(跨节点 RDMA,每层 P 步)
```

显存:KV/卡 = 总量 / (TP × CP)(head 切 × 序列切)——**17.3 的公式回代依据**。

## 17.3 1M 上下文实战(DeepSeek MLA + P/D)

**KV 算账(Ch3 公式回代,MLA 特性)**:

MLA 把 KV 压缩为 latent(DeepSeek-V3:512 维 latent + 64 维 rope,K/V 共享,61 层 → 每 token 每层 (512+64)×2B ≈ 1.1KB,全模型 ≈ **70KB/token (BF16)**,官方量级;FP8 减半 ≈ 35KB)→
1M token 全量 KV ≈ **70 GB(BF16)/ ~35 GB(FP8)**。

- TP8 单副本:8.75GB/卡(BF16)/ 4.4GB(FP8);60GB 池 **并发 ~7(BF16)/ ~13(FP8)**——单 TP8 副本即可支撑中等并发,KV 不再是绝对约束
- CP4 × TP8:KV/卡 = 总量/32 ≈ 2.2GB(BF16)/ 1.1GB(FP8)→ **KV 容量富余(60GB 池可容纳 ~50 in-flight),1M 并发受 prefill 吞吐而非 KV 限制**(Ch3 容量公式)

**架构形态(prefill-only CP 集群,呼应 Ch9)**:

```
[Prefill 集群: CP16(2 节点×8 卡)+ MLA] ──KV──► [Decode 集群: TP8 副本 × N]
                    (P/D 分离,Ch9)
```

**工程痛点:KV reshard**。P 侧 KV 是 CP 切分(序列维分片)+ TP 切分(head 维分片);D 侧只要 TP 切分。P→D 传输前必须 **all-gather 序列维**(P 侧汇聚成完整 KV,再按 D 的 TP 布局发送)——reshard 是一次额外 all-gather + 重排,**长上下文的 P/D 传输成本里,reshard 占不小份额**(Ch9 §9.5 "传输前处理" 的具体形态)。生产做法:CP 只在 prefill 集群,D 侧不做 CP → 传输层统一在 CP 组内完成 gather。

**decode 侧 1M 的含义(接 Ch11 §11.4)**:1M KV 的 decode attention 必须走 FlashDecoding 类 split-KV(单 CTA 扫 1M 串行不可行),且 MLA 的 latent 布局有特化 kernel。

## 动手

**实验 17.1(Ch27 承接,2 节点)**:

1. 长 prefill(64K-128K)CP 开/关耗时对比(同硬件,CP2 vs 单卡),验证"通信可被计算掩盖"
2. 或:两节点 ring attention 最小实现(17.2 伪代码),对比 naive prefill 与显存
3. 用 Ch3 容量公式算"你的模型 1M 上下文 TP8 能并发几条",与实测对照

## 章末自测

1. 低并发 vs 高并发下 PP 的代价各是什么?"in-flight ≫ S" 为什么是关键条件?
2. PP 与跨机 TP 相比,通信代价为什么小(消息量 × 次数 × 频率)?
3. Ulysses 为什么限节点内?Ring 为什么能跨节点?两者通信总量为何同量级?
4. 回代 Ch3 公式:MLA 模型 1M 上下文,TP8 与 CP4×TP8 的 KV/卡各是多少?并发上限各几条?
5. [进阶]P/D + CP 架构里,"KV reshard"发生在哪、做什么、为什么不可省?

## 面试钩子

- 支撑 Ch28-D21~D26:"PP bubble 多大、何时消失?""Ring 与 Ulysses 怎么选?""1M 上下文怎么摆卡?"
- Ch29 案例 7(128K prefill 集群)与案例 3(1M RAG)的直接依据。
