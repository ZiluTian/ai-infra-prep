# 第 13 章 MoE 算子与推理

> 难度 ★★★★★ | 目标 ~30 页 | 前置:Ch12
> 注:整章 [进阶];路线 A 读者建议只精读 13.1(数据流)与 13.4(均衡链)。
>
> **读完本章你能做什么**
> 1. 画出 MoE 推理完整数据流(router → dispatch → grouped GEMM → combine),并说清 DeepSeek-V3 的配置(256 routed + 1 shared、top-8)
> 2. 精算 all-to-all 通信量,解释 DeepEP normal 与 low-latency 两种模式的差异与场景
> 3. 解释 grouped GEMM 的"不均匀 M"问题(permute/padding/零 token expert/launch 开销)
> 4. 讲清"数据相关路由 → 热 expert → straggler → p99 长尾"的完整链条与四类缓解手段
> 5. 对 vLLM MoE forward 插桩,输出每 expert 的 token 分布与 all-to-all 耗时占比(动手)

## 13.1 MoE 推理数据流

以 DeepSeek-V3 为例:671B 总参数、**256 个 routed expert + 1 个 shared expert**、每 token **top-8**(aux-loss-free 均衡)、hidden 7168、expert intermediate 2048、前 3 层 FFN 为 dense(其余层全为 MoE)。激活参数约 37B(官方量级)。

每层 MoE 的数据流(EP 场景):

```
x (tokens, 7168)
 ① router:  x @ W_gate (7168→256) → 门控(sigmoid + top-8 选择)
 ② dispatch: EP all-to-all,token → 目标 expert 所在卡(§13.2)
 ③ expert GEMM(本卡): gate_proj(7168→2048) + up_proj(7168→2048)
                     → SiLU×mul → down_proj(2048→7168)
     —— grouped GEMM:每 expert 一段, M_e 各不相同(§13.3)
 ④ combine:  反向 all-to-all,按门控权重加权求和,回原 token
 ⑤ shared expert:本地 dense MLP,不参与通信,输出相加
```

**成本结构(面试口径)**:

- 计算:每 token 只过 top-8 expert → FLOPs/token ≈ 2 × 激活参数(37B 量级),远小于 671B
- 显存:全量 671B 权重必须**全部驻留**(哪个 token 去哪个 expert 不可预知)→ 容量按总量算(Ch3),计算按激活量算(Ch1)
- 通信:每层 2 次 all-to-all(dispatch + combine),量见 13.2

## 13.2 all-to-all dispatch 与 DeepEP

**通信量精算**:

```
(13-1) all-to-all 总发送量 ≈ batch × seq × topk × hidden × bytes
       每卡每对链路 ≈ 总量 / (N−1)(all-to-all:向每个其他卡都发)
```

手算(EP8、decode、FP16、hidden 7168):batch=64 token、top-8 →
总量 = 64 × 8 × 7168 × 2B ≈ **7.3 MB/层/步**;每卡向每个对端发 7.3/7 ≈ 1.05 MB;NDR IB 50GB/s 下传输 ≈ 20μs(量级)。

**关键洞察:decode 的 all-to-all 是"延迟主导"不是"带宽主导"**——每步每对端才 ~1MB,填不满 50GB/s 的管道,端到端时间由**消息延迟地板(μs 级)× 轮数 × 同步等待**决定。这就是 MoE decode 通信与 TP all-reduce(消息大、带宽主导)的本质区别,也是 DeepEP 分两种模式的根据。

**DeepEP 两种模式**(DeepSeek 开源,配合 V3/R1 生产部署):

| | normal 模式 | low-latency 模式 |
|---|---|---|
| 网络 | 节点内 NVLink 为主,跨节点 IB | 跨节点 RDMA,**IBGDA**(GPU 直接 post work queue,不经 CPU) |
| 粒度 | 逐 token,高吞吐 | **分组发送**(如 512 一组,1D/2D 切分:先节点内归拢,再节点间一条消息) |
| 特性 | 吞吐优先,可大消息 | 延迟优先,GPU 端直接收发、与计算 overlap |
| 场景 | **prefill**(token 量大,带宽是主角) | **decode**(token 少,延迟是主角) |

2D 切分(2D partitioning)的含义:先按节点内 NVLink 域把发往同一节点的 token 聚合(节点内便宜),每个节点对只发一条 RDMA 消息(节点间贵)→ **RDMA 消息数从 O(节点×GPU) 降到 O(节点对)**,延迟大幅降低。

**占比量级**(EP8 单机 NVLink 域,decode 一步 64 token):all-to-all 合计 ~50-150μs;grouped GEMM(激活 37B)~1-2ms → 通信占 **5-10%**。占比不高,但**长尾发生在通信**(§13.4):通信占比低 ≠ 通信无害。

## 13.3 Grouped GEMM:不均匀 M 的工程问题

E 个 expert,每个 `(M_e, N) × (N, K)`,M_e 互不相同。naive 做法是 E 个独立 GEMM kernel → 三个问题:

1. **launch 开销 ×E**:decode 时 M_e = batch × topk / E,例:64 × 8 / 256 ≈ **2**——每 expert 2 行 token 的 GEMM,256 次 kernel 发射,发射本身比计算贵
2. **小 GEMM 不跑满**:M_e=2 时 tensor core 利用率极低
3. **零 token expert**:负载不均衡时相当多 expert 一步内 0 token,仍要参与调度

工程解(Ch12 §12.1 Step 3 的结构展开):

| 手段 | 做法 | 代价/适用 |
|---|---|---|
| **permute + grouped GEMM**(主流) | token 按 expert 排序成连续 `(M_total, K)`;一个 kernel 内 device 端遍历 (expert, m_tile, n_tile)(CUTLASS grouped GEMM / vLLM `fused_moe`(Triton)/ TRT-LLM MoE kernel) | permute 本身是 kernel 开销;零 token expert 的 tile 直接 early-return(Ch12) |
| **padding** | M_e 补齐到 BM 倍数,tile 网格规整 | 热 expert 偏斜时浪费放大;简单场景够用 |
| **GEMV 切换** | M_e < 16 走 GEMV kernel(FlashInfer GEMV / TC-GEMV),M_e 大走 GEMM | vLLM fused_moe 内置该判断;避免"用大 GEMM 算 2 行" |
| **gate/up 融合** | gate_proj + up_proj 一次 GEMM 出 2N 列,epilogue 内 SiLU×mul | 省一次 HBM 往返 + 一次 launch |

**精度(接 Ch8 §8.5)**:expert 权重是激进量化的主战场——per-expert per-group scale;热 expert 保高精度、冷 expert 激进(流量分布决定精度分配)。

## 13.4 负载均衡与尾延迟(本章主线)

**完整链条(必须能一口气讲完)**:

```
数据相关路由(router 看内容选 expert)
  → token 分布 Zipf 式,少数热 expert 吸走大量 token
  → EP 下热 expert 所在卡的 M_e ≫ 均值
  → 该卡 GEMM 时间 ≫ 其他卡
  → 下一层 all-to-all 要等所有卡(同步点)
  → 全层时延 = 最慢卡 → 热 expert 卡成为 straggler
  → p99 尾延迟(Ch22),而 p50 完全正常
```

**数值例子(量级)**:EP8、batch 64、top-8(总派发 64×8=512 token),理想每卡 64 token;若热 expert 集中在卡 3:卡 3 收 379 token,其余各 19(379+7×19=512,总量守恒)→ GEMM 时间比 ≈ 20× → 该层 all-to-all 完成时间 ≈ 20 × 均匀情形。**p50 看不出,p99 爆炸**——MoE 的 SLO 容量必须按 p99 设计。

**缓解手段(分层,面试按"训练侧 → 放置 → 复制 → 路由"讲)**:

1. **训练侧**(router 设计):aux-loss / router bias(V3 的 aux-loss-free:对 router 概率加 bias 项),平滑分布——治本,但只在训练期生效
2. **放置(静态)**:拓扑感知放置热 expert——热 expert 分散到尽量多的卡(避免两个热 expert 同卡);节点内 NVLink 域优先(节点内 dispatch 快);依据是**离线的 expert 热度统计**(对目标流量跑 router 统计)
3. **复制(动态)**:热 expert 复制到多卡,token 就近分流;代价:显存 ×N + 通信模式变化 + 副本一致性(权重静态,复制是纯拷贝);触发条件:持续热度监控(分钟级),不是瞬时尖峰
4. **请求级路由(接 Ch19)**:多副本时,把请求导向"有富余 expert 容量"的副本——把 expert 均衡问题抬到请求路由层解决
5. **capacity factor + drop**(了解即可):给每卡 M_e 上限,超了丢弃部分 token;有质量损失,生产一般不用,训练/离线容忍场景才有

> **踩坑框:"MoE 激活参数少,所以肯定比稠密快"**
> 算与存都更快,没错;但**时延方差更大**——decode 长尾由最热 expert 卡决定,不由均值决定。两条工程结论:① MoE 容量规划按 p99 不按 p50;② "TP8 换成 EP8 提速 30%"这种均值结论,必须附 p99 才能进生产决策。

## 动手

**实验 13.1(1-2 小时,有 GPU + vLLM/SGLang,MoE 模型)**:

1. 用 Qwen3-30B-A3B(或可用的 DeepSeek 轻量版)起服务,压测时 hook router 的 top-k 输出,累计每 expert 的 token 数 → 输出 **top-10 热 expert 表 + 分布 Gini 系数**
2. nsys/torch profiler 抓 MoE 层:dispatch/combine(all-to-all)占该层 forward 的时间占比
3. 交付物:"热 expert 表 + 通信占比",直接作为 Ch29 案例 8(热 expert 治理)的输入数据

## 章末自测

1. 画出 V3 MoE 层四段数据流;shared expert 与 routed expert 在计算/显存/通信上各有什么特点?
2. 精算:EP8、batch 64、top-8、hidden 7168、FP16,每层 all-to-all 总发送量与每卡每对端量;为什么 decode 场景它是"延迟主导"?
3. prefill 用 normal 模式、decode 用 low-latency 模式——从带宽/延迟两个维度解释
4. grouped GEMM 为什么对 MoE 是刚需?零 token expert 在 kernel 里如何被跳过?
5. 热 expert → p99 的完整链条;五个缓解手段中,哪两个在"部署后"生效?

## 面试钩子

- 支撑 Ch28-C17~C24:"MoE 的 EP all-to-all 怎么做?""DeepEP 两种模式区别?""grouped GEMM 怎么高效?""MoE 尾延迟从哪来、怎么治?"
- Ch29 案例 1(100B+ MoE 生产级 serving,64 卡)与案例 8(MoE 热 expert 治理)是本章的综合题。
