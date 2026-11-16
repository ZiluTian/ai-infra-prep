# 第 9 章 P/D 分离(Disaggregated Serving)

> 难度 ★★★★★ | 目标 ~20 页 | 前置:Ch5、Ch6、Ch18(荐)
> 注:整章 [进阶];路线 A 读者建议精读 9.1 与 9.3(决策与成本模型)。
>
> **读完本章你能做什么**
> 1. 解释 prefill/decode"资源画像冲突"为什么 chunked prefill 解决不干净
> 2. 画出 P/D 架构:路由、KV 传输、同步协议、失败语义
> 3. 精算 KV 传输成本(Ch3 公式 + RDMA 带宽),推导"何时值得"判据
> 4. 对比 DistServe / Splitwise / Mooncake / vLLM disagg / SGLang PD
> 5. [进阶]做动态 P:D 配比与传输前 KV 压缩的决策

## 9.1 为什么分离

Chunked Prefill(Ch6)已经把"prefill 卡 decode"解决到了**步级**。但还有两个**结构性**冲突:

1. **显存池竞争**:一条 128K prefill 请求在池中临时占巨大 KV,挤 decode batch → decode 侧 TPOT 抖动;decode 长驻请求占池,又让长 prefill 排不进来
2. **扩缩容不对称**:长上下文 + 高负载下,prefill 与 decode 所需卡数比往往不是 1:1(如 2:1);一体集群只能 1:1,需求多的那一侧排队

P/D 分离 = prefill 与 decode 各自独立卡池、独立扩缩容、KV 经 RDMA 从 P 搬到 D。

```
Client ──► [Router] ──► P 副本池(pure prefill,compute-bound 调优)
                     │  prefill 完成
                     ▼  KV via RDMA (zero-copy, GDR)
                   D 副本池(pure decode,高 batch 稳定) ──► 流式输出
```

## 9.2 架构与 KV 传输

**KV 传输成本精算**(Ch3 公式直接套用)。

> **RDMA / IB 速查(没有网络背景先读这)**:**RDMA**(Remote Direct Memory Access)= 远端机器的网卡直接读写本机内存,不经过远端 CPU 与内核;**GDR**(GPUDirect RDMA)= 网卡绕 CPU 直访 GPU 显存。**IB(InfiniBand)**= NVIDIA 数据中心默认的卡间专用网络:NDR400 = 每口 400Gb/s(= 50GB/s/口);rail-optimized 拓扑下每卡走自己 rail 上的 1 个对端口 ≈ **单链 ~50GB/s**,8 条 rail 并行互不争抢。KV 传输 = 每个 rank 把自己那片 KV 分片发给 D 侧对应 rank。

| 场景 | KV/请求(全模型) | KV/每 rank 分片(P/D 同 TP8) | 单链 ~50GB/s 传输时间 |
|---|---|---|---|
| 70B GQA,8K,FP16 | 2.56 GB | 0.32 GB | ~6 ms |
| 70B GQA,128K,FP16 | 40.9 GB | 5.12 GB | ~100 ms |
| 同上,FP8 KV | 20.5 GB | 2.56 GB | ~50 ms |

**结论:P/D 同 TP 布局按 per-rank 分片传输,8K 传输(~6ms)只占 prefill(250ms@TP8)的 2-3%,128K 单链 ~100ms 也只占 prefill(~3s)的 ~3% → 传输不是主要成本**。但 FP8 KV + 多链并行仍是标配:① 保 p99、与 NCCL 流抢带宽(Ch18 §18.4);② KV 字节减半,D 侧池并发翻倍(Ch3 反推);③ (谨慎地)传输前压缩。**注意:P/D TP 布局不一致(reshard)时,传输退回全模型量级(×8),重新"与 prefill 同量级"(Ch17 §17.3 痛点)。**

**同步协议(生产细节,面试区分点)**:

1. P 完成 prefill → 通知 router + D 副本
2. D 预留 KV 槽位(池内按 block 预留,Ch5)
3. RDMA 读/写搬运(GPUDirect,绕 CPU;Ch18)
4. D 开始 decode;P 侧 KV 释放
- **失败语义**:P 崩 → in-flight 丢弃,客户端重试;D 拒收(池满)→ 返回错误,P 的 KV 作废(白算了一次 prefill——这就是"P 池超调度"的成本)
- **布局匹配**:P 与 D 的 TP size 相同时 KV 直接映射;**TP 不同则需要 KV reshard**(head 重新分布,额外通信)——生产上 P/D 同 TP 是常见约束

## 9.3 成本模型(决策公式)

```
收益 = TTFT 改善(prefill 不再排队等 decode 步)
     + TPOT 稳定性(decode 池纯净,无长 prefill 尖刺)
成本 = KV 传输(带宽 + 延迟)
     + 额外 hop(调度/路由)
     + 运维复杂度(两个资源池、两套调度、KV 传输层)
```

**值得 ⇔**(a) 高并发(TTFT 主要成分是排队)+(b) 长上下文(传输成本可被 FP8/并行摊平,且收益大)+(c) TTFT 与 TPOT 双 SLO。
**不值得:短上下文 + 中低并发**——chunked prefill 已解决大部分,传输成本反成主导。

**定量例(示意)**:70B、8K 上下文:per-rank 分片 KV 0.32GB → 传输 ~6ms;prefill 8K ≈ 250ms(TP8 算力)→ 传输占 prefill 2-3%——高负载 P/D 的收益主要来自**结构分离**(decode 池纯净、双池独立扩缩),**值得**。512 上下文:per-rank 分片 20MB → <1ms;prefill ~4ms,且一体集群下 TTFT 本来就低 → **不值得**(还多付一套 KV 传输层的运维)。

## 9.4 开源实现

| 系统 | 特点 |
|---|---|
| DistServe(OSDI'24) | P/D 分离 + goodput 调度(SLO 约束下放置 P/D 副本) |
| Splitwise(ISCA'24) | 强调 prefill 相位与 decode 相位资源差异、近存 prefill |
| Mooncake(Kuaishou) | KV-centric 集群:分离 + 全局 KV 缓存池,1024 卡级,与 SGLang 集成 |
| vLLM production disagg | NIXL 传输层;`vllm serve --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` + P/D 角色(kv_role 取值 kv_producer / kv_consumer / kv_both,附录 G §G.5) |
| SGLang PD | `--disaggregation-mode` + Mooncake / 原生传输 |

源码点:KV 传输层(NIXL:block 级搬运,与 Ch5 的 block 号体系对接)、调度拆分——P 侧管 prefill 队列 + 传输任务;D 侧管 decode batch + KV 槽位预留。**调度器被拆成两半后,"全局 goodput"从引擎内指标变成跨池指标**(接 Ch29 案例 5)。

## 9.5 [进阶]

- **动态 P:D 配比**:监控 prefill 队列深度 vs decode batch 负载,弹性扩缩 P/D 池(K8s,Ch23);稳态配比由负载的输入/输出长度比决定
- **传输前 KV 压缩**:FP8(默认、近无损)→ lossless 压缩(KV 熵不高,收益有限)→ 选择性压缩(SnapKV 类,有质量风险,要按任务评估,Ch8 §8.4 流程)
- **prefill 集群 + CP**:1M 上下文 prefill 单独放 CP8 集群(Ch17),decode 仍 TP8——"长 prefill 的长尾"被隔离在 P 池
- **双池网关**:短请求走一体副本(省传输),长请求走分离池;路由按输入长度分流(Ch19 全局路由的特化)

## 动手

P/D 实验见 Ch27 实验 3(两节点:测 TTFT/TPOT 改善、网络带宽占用、P:D 配比扫描)。

## 章末自测

1. chunked prefill 已经解决"步级"冲突,P/D 分离还解决哪两个"结构性"问题?
2. 70B 128K 请求的 KV 传输成本精算(FP16 vs FP8,单链 vs 多链)
3. "何时值得 P/D 分离"判据;给一个"不值得"的反例并解释
4. P 与 D 的 TP size 不同会发生什么?为什么生产常约束相同?
5. [进阶]P 池超调度(多放 P 副本)的失败成本是什么?

## 面试钩子

- 支撑 Ch28-B41~B44:"P/D 什么时候值得?""KV 传输成本怎么算?""P/D 的 TP 不一致怎么办?"
- Ch29 案例 5(P/D 集群与动态配比)即本章的综合题。
