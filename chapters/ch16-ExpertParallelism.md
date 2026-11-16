# 第 16 章 Expert Parallelism(MoE)

> 难度 ★★★★★ | 目标 ~30 页 | 前置:Ch13、Ch14
> 注:整章 [进阶];与 Ch13 的分工:Ch13 讲算子(kernel/通信量),本章讲**系统**(部署形态/均衡策略/规模实战)。
>
> **读完本章你能做什么**
> 1. 源码级说出 EP 数据流:local router → 路由表 → dispatch all-to-all → 本地 expert GEMM → combine
> 2. 解释 DeepEP normal / low-latency 的 buffer 布局、消息粒度与 overlap 机制,给出"prefill 用 normal、decode 用 low-latency"的完整理由
> 3. 用数值例子量化"热 expert straggler → p99",并给出静态放置 / 动态复制 / 请求路由三类策略的成本-收益
> 4. 对比单机 EP8 与跨机 EP64 的延迟/吞吐/故障域,说出 DeepSeek 系生产形态(EP×TP 混合)

## 16.1 EP 数据流(源码级)

```
每个 rank(持有 E/N 个 expert)每层:
 ① local router:  本卡 token 的 hidden × 全局 router 权重 → top-k 目标 expert + 门控权重
                   (router 权重全量在每卡,只算不切)
 ② 路由表:        本卡"我要把 token i 发给哪个 rank"+ 反向"哪些 token 要来我卡上"
                   → 汇总为 (rank 对, token 列表, 目标 expert 在本地第几个)
 ③ dispatch:      all-to-all,token hidden 向量发往目标 rank
 ④ 本地 expert:   grouped GEMM(Ch13 §13.3),按 expert 分组
 ⑤ combine:       反向 all-to-all,带门控权重;接收方按原 token 位置归并求和
```

**vLLM / SGLang 的实现点**:

- **DeepEP 集成**:两框架均以库形式引入 DeepEP(vLLM `vllm/distributed/` 的 all-to-all 后端;SGLang 同为 DeepEP),自研 grouped GEMM 与之对接
- **路由表在哪算**:router 在**每个 rank 本地**(权重复制,计算不切)→ 路由表是"本地算、全局用"——这是 EP 与 TP 的本质差别:TP 的通信是**结果合并**,EP 的通信是**数据搬运(携带路由信息)**
- **shared expert**:本地 dense MLP,不走 all-to-all(Ch13 §13.1)

## 16.2 DeepEP 两种模式(工程细节)

Ch13 §13.2 给了模式对比表;这里补**buffer 与 overlap 的工程细节**(面试区分点):

**normal 模式(高吞吐)**:

- 节点内走 NVLink P2P(预注册 buffer 对,逐 token 拷贝),跨节点走 IB
- 消息粒度小(逐 token),流水线深 → **带宽利用率高**,但每消息固定开销大 → 适合 token 量大的 prefill
- 典型 buffer:dispatch 发送/接收 buffer 按 `max_tokens_per_rank × hidden` 预分配,避免运行时 malloc

**low-latency 模式(decode 专用)**:

- **IBGDA**:GPU 直接写 RDMA work queue(绕过 CPU 的 post 路径)→ 消息延迟从 "CPU 往返 + 网络" 降到 "纯网络 + GPU 直发"
- **分组 512**:多个 token 打包成一组消息(1D/2D 切分:先节点内 NVLink 归拢,节点间每对一条消息)→ RDMA 消息数 O(节点对),摊薄每消息延迟
- **RDMA write(单向)**:发送方直接写接收方 buffer(零接收参与),接收方 GEMM 完成后用 flag/原子计数通知 → 发送方 CPU/GPU 完全不被收发阻塞
- **overlap(关键)**:layer i 的 expert GEMM 与 layer i+1 的 all-to-all 在独立 stream 并行;路由在 layer i-1 的 router 后已知(Ch15 §15.3 形态 2)

**为什么 prefill 用 normal、decode 用 low-latency(定量)**:

| | prefill | decode |
|---|---|---|
| 每步 token 量 | batch × chunk(数千+) | batch(数十~数百) |
| 每对端消息 | 大(MB 级)→ **带宽主导** | 小(百 KB~1MB)→ **延迟主导** |
| 模式 | normal(NVLink 深流水吃满带宽) | low-latency(IBGDA + 分组 + overlap) |

## 16.3 负载均衡(SLO 视角)

Ch13 §13.4 给了链条;这里给**量化与策略成本表**。

**p99 量化(面试数值例)**:EP8、batch 64、top-8(总派发 512 token)。均匀时每卡 64 token;某步热 expert 全落在卡 3:卡 3 = 379 token,其余 = 19 个(379+7×19=512,总量守恒)。grouped GEMM 时间 ∝ M_e(小 M 时近似线性)→ 卡 3 的 GEMM 时间 ≈ 20× 均值;combine all-to-all 等最慢卡 → **该步所有卡都慢 20×**。若热 expert 出现频率 5% → p95 开始抬头、p99 爆炸,而 p50 正常。**结论:MoE 的容量与告警都锚定 p99,不用 p50。**

**三类策略的成本-收益(面试按此表讲)**:

| 策略 | 机制 | 成本 | 生效条件 | 适用 |
|---|---|---|---|---|
| **静态放置** | 离线统计 expert 热度(对目标流量跑 router)→ 热 expert 分散到多卡、同节点优先 | 一次性统计;模型更新后需重算 | 热度稳定 | 部署期默认做 |
| **动态复制** | 热 expert 复制到 k 卡,token 就近分流 | 显存 ×k(权重静态,纯拷贝);路由表变更 | 持续热度(监控分钟级触发),非瞬时尖峰 | 生产热 expert 治理(Ch29 案例 8) |
| **请求级路由** | 多副本时把请求导向"有富余 expert 容量"的副本(副本上报 expert 占用) | 路由层要维护副本状态 | DP 副本数 ≥ 2 | 多副本集群 |
| (capacity + drop) | 每卡 M_e 上限,溢出丢弃 | **质量损失** | 仅离线/容忍场景 | 生产不用 |

**训练侧**(router bias / aux-loss-free)是治本手段,推理团队只能"读结果"(热度分布)不能用它。

> **踩坑框:"EP 卡数翻倍,吞吐翻倍"**
> 不成立的两处:① all-to-all 的**每对端延迟**不随卡数下降(消息延迟地板),卡数↑ → 参与同步的 rank 数↑ → straggler 概率↑;② 热 expert 的 M_e 偏斜与卡数无关(分布决定)。**EP 扩容的收益上限 = min(带宽, 延迟地板, 最热 expert 卡的 GEMM 时间)**。

## 16.4 EP 规模实战

**单机 EP8 vs 跨机 EP64(量级对比,以 671B 级 MoE、decode 为准)**:

| 维度 | 单机 EP8(8 卡) | 跨机 EP64(8 节点) |
|---|---|---|
| dispatch 网络 | NVLink(~900GB/s,μs 延迟) | RDMA(IB 50GB/s/对,μs 延迟地板高) |
| 每卡 expert 数 | 32 | 4 |
| all-to-all 对端 | 7(同节点) | 63(跨 7 节点) |
| 延迟特征 | 通信占比 ~5-10%(Ch13) | 通信占比 ~10-20%,straggler 风险× |
| 故障域 | 1 节点 | 1 节点挂 → 整个 64 卡组降级(需 DP 冗余) |
| 适用 | 模型 ≤ 8 卡容量 | 671B 级必须(权重 671B×2B=1.3TB > 8×80GB) |

**DeepSeek 系生产形态(公开资料,量级)**:

```
DP: 多副本(每副本 = 一个并行组)
 └─ 并行组: attention TP8(节点内 NVLink)
           + expert EP(跨节点,节点内 dispatch 走 NVLink 快路径)
           + 长 prefill 走 CP(prefill 集群,Ch17)
```

要点:**attention 与 expert 用不同并行维**——attention 的 GEMM 是稠密的(适合 TP,通信规整),expert 的 GEMM 是 grouped 且数据相关(适合 EP,通信可走 low-latency all-to-all);"节点内快、节点间慢"的拓扑被映射到"TP 在节点内、EP 跨节点"。

## 动手

**实验 16.1(Ch26 承接,半天,多卡)**:单机 EP8 vs TP8(或可用的 MoE 模型,如 Qwen3-30B-A3B):

1. 同 batch 下 decode 延迟/吞吐对比;nsys 测 all-to-all 占 MoE 层时间比
2. 故意打偏路由(测试 hook:强制 top-8 集中到前 2 个 expert)→ 测 p99 放大倍数(复现 16.3 的 20× 量级)
3. 交付物:EP vs TP 对比表 + 偏斜注入的 p99 曲线

## 章末自测

1. EP 与 TP 的通信本质差别(结果合并 vs 数据搬运)?路由表在哪算?
2. low-latency 模式的三个设计(IBGDA / 分组 512 / RDMA write)各解决什么?
3. 为什么 prefill 用 normal、decode 用 low-latency(带宽/延迟主导判据)?
4. 16.3 的 p99 例子:热 expert 出现频率 5%,为什么 p50 正常而 p99 爆炸?
5. EP 卡数翻倍吞吐不翻倍的两个原因;EP64 的故障域与 DP 冗余的关系?

## 面试钩子

- 支撑 Ch28-D15~D20:"EP 与 TP 通信有何不同?""DeepEP 两种模式?""MoE 负载怎么均衡?""671B 怎么摆?"
- Ch29 案例 1(100B+ MoE 64 卡)与案例 8(热 expert 治理)是本章综合题。
