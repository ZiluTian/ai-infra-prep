# 第 6 章 Continuous Batching 与调度器

> 难度 ★★★★ | 目标 ~30 页 | 前置:Ch4
>
> **读完本章你能做什么**
> 1. 对比 static / dynamic / continuous batching,说清 iteration 级调度为什么是甜蜜点
> 2. 读懂 vLLM v1 调度器的每步决策(token 预算、waiting/running、抢占),能写出伪代码
> 3. 解释 Chunked Prefill 解决的问题、chunk size 的权衡曲线,以及 vLLM/SGLang 的配置入口
> 4. 说出 FCFS 之外的调度策略(priority/fair/deadline-aware)及适用场景
> 5. [进阶]做集群级调度决策:P/D 一体 vs 分离、DP attention、全局队列
>
> **定位**:第 5 章管"KV 存哪",本章管"KV 什么时候算、按什么顺序算"。调度器是推理引擎的心脏——TTFT/TPOT 的每一个尖刺,追到底都是调度决策。

## 6.1 Batching 的三级跳

**Static batching**(最早):请求凑满一个 batch 才开始,全 batch 跑完最长者才返回。

```
req A(输出 300)  |████████████████████████████|
req B(输出 20)   |██|___________________________|  ← B 结束后 GPU 在等 A
```

GPU 大量空转,尾部长请求拖累所有人。

**Dynamic batching**:batch 边界插入新请求,但 batch 内仍"共进退"(等最长)。改善有限。

**Continuous batching**(iteration 级,Orca 提出,vLLM/SGLang/TRT-LLM 全部实现):

```
iteration 1:  A B C
iteration 2:  A B C D      ← B 结束了,同一步 D 进来
iteration 3:  A C D E
```

- 每个 decode step(每请求 1 token)就是一个调度机会:结束的让位、新到的进场
- 为什么是 iteration 粒度而不是 token 级更细:GPU 一次 forward 天然处理一个 batch,step 间插入是零结构成本的;"更细"没有收益(CUDA 图/张量组装以 step 为单位)
- **定量意义(接 Ch1 §1.5)**:continuous batching 让 decode 的 batch 长期维持在"显存允许的最大并发" → 算术强度从 1 抬到数百 → decode 吞吐提升近两个量级(直到 KV 显存与调度事件限制)

## 6.2 vLLM v1 调度器剖析(源码级)

v0 时代调度逻辑散在 handler 里;v1 把调度收敛到 `vllm/v1/core/sched/scheduler.py` 的 `schedule()`——每次 GPU forward 之前调用一次。简化伪代码:

```python
# vllm/v1/core/sched/scheduler.py(简化;实际约千行,含多模态/LoRA/抢占策略等分支)
def schedule(self) -> SchedulerOutput:
    budget = TokenBudget(self.max_num_batched_tokens)   # ① 总阀门(默认 2048,v0.32.0)
    running, waiting = self.running, deque(self.waiting)
    scheduled_prefill, preempted = [], []

    # —— 1) 先保 decode:running 里每请求 1 token ——
    for req in running:
        if not budget.can_take(1):
            break
        if not self.kv_cache_manager.try_allocate(req, 1):
            self.preempt(req)                           # ② KV 不够 → 抢占
            preempted.append(req)                       #   recompute 或 swap,见下
            continue
        budget.take(1)

    # —— 2) 再放 waiting:整段 prefill 或一个 chunk ——
    while waiting and budget.remaining > 0:
        req = waiting[0]
        n = min(req.num_uncomputed_tokens, budget.remaining)   # ③ chunked prefill 入口
        if self.kv_cache_manager.try_allocate(req, n):
            waiting.popleft()
            budget.take(n)
            scheduled_prefill.append((req, n))
        else:
            break                                       # 池满,等下一 iteration

    return SchedulerOutput(decode=running, prefill=scheduled_prefill,
                           preempted=preempted)
```

四个设计点:

**① token 预算(`max_num_batched_tokens`,默认 2048)是 v1 的总阀门**。prefill chunk 与 decode token 共享这一份预算,它同时决定:每步 prefill 能吃多长(= chunk 上限)、同时能 decode 多少请求。**这一个参数就是 TTFT/TPOT 权衡曲线的主旋钮**(6.3 展开)。

**② 抢占:recompute vs swap**。KV 池满、新请求要进、decode 保不住时:

| 策略 | 动作 | 代价 | 适用 |
|---|---|---|---|
| recompute | 丢 KV,请求回 waiting,下次重新 prefill | O(N²) 重算(Ch1 §1.2) | 上下文短、CPU 带宽贵、请求"年轻" |
| swap | KV 换出到 CPU DRAM,恢复时换回 | KV 体积 / DRAM 带宽(32K 请求 ~10GB ÷ 50GB/s ≈ 200ms) | 上下文长、重算太贵 |

vLLM 默认优先 recompute(实现简单、无 CPU 带宽抖动);SGLang 两者都支持。**抢占 = 容量超售的代价函数**:池可以"看起来"比实际大,但每次超售都记在 p99 上(Ch22)。

**③ chunked prefill 的入口就是 `n = min(剩余, 预算)`**——长 prompt 不再"一口气 prefill",而是按预算切片(6.3)。

**④ 调度器只产"计划",不碰张量**:输出是 (请求, token 数, block 映射);张量组装在 ModelRunner(Ch4 §4.1 ③)。这个边界让"异步调度"(GPU 跑第 n 步时 CPU 排第 n+1 步)成为可能。

## 6.3 Chunked Prefill:长 prefill 不再卡住 decode

**问题(head-of-line blocking)**:没有 chunking 时,一条 32K 长 prompt 的 prefill 占满整个 forward(数百 ms 到秒级),期间所有 decode 请求停住 → **每个在跑请求的 ITL 出现一个与自身无关的尖刺**。这是"TTFT 尚可、TPOT p99 爆炸"的典型来源。

**方案**:prefill 切 chunk,与 decode 混在同一 forward:

```
iteration k:   [long-prefill chunk 2048] + [decode × 128]
iteration k+1: [long-prefill chunk 2048] + [decode × 130]
...
```

- 长请求的 TTFT 变慢(被切片摊开),但**全体 decode 的 TPOT 保持平稳**——用"长请求的 TTFT"换"所有人的 TPOT",这是绝大多数在线场景的正确交易
- vLLM v1:chunked prefill 是默认行为,chunk 上限 = `max_num_batched_tokens`(默认 2048;常用 4096/8192/16384;`--no-enable-chunked-prefill` 可退回整段 prefill 的 v0 语义)
- SGLang:`--chunked-prefill-size`(如 8192),且 prefill/decode 预算分离更细

**chunk size 的权衡曲线(定量直觉)**:

```
chunk 太小(512):  prefill 效率低(小 GEMM,AI 不够,Ch1 §1.5 交叉点 N≈150 之上
                  才 compute-bound),长请求 TTFT 进一步拉长
chunk 太大(16K+): decode 每步等待变长 → TPOT 尖刺回归(只是变小变稀疏)
常用值 8192:     prefill 足够 compute-bound,单步 decode 等待 < 10ms 量级
 (v0.32.0 默认 2048,偏保护 TPOT;按负载在 4096~16384 间调)
```

**配置建议(先测后定)**:短输出聊天场景用默认 2048 或更小(保 TPOT),长文档 RAG 调大(8192~16384,保 TTFT),再用 Ch2 的 goodput 曲线验证 knee 点位置。

> **案例框(合成,数字示意):一条 128K prompt 的两种命运**
> 无 chunking:prefill 占 ~9s,期间 200 个并发 decode 请求全部停摆 → p99 ITL +9s。
> chunked(8192):128K 切成 16 步,每步 decode 等待 ~50ms → p99 ITL +50ms,该长请求 TTFT +9s(可接受:它是唯一的新请求)。
> 结论:chunked prefill 是"把尖刺从 p99 摊到长请求自己的 TTFT 上"——**调度优化的本质是"谁为事件买单"的决策**。

## 6.4 调度策略:FCFS 之外

- **FCFS**(vLLM 默认):公平、可预期;对"所有请求同 SLO"足够
- **priority**:显式优先级字段(如付费等级);注意优先级反转风险(高优先级长 prefill 反复挤占 → 低优先级饿死,需配额)
- **fair(份额制)**:多租户按权重分 batch 份额(Ch23 多租户)
- **deadline-aware(EDF/LLF)**:每请求带 deadline,按剩余 slack 排序(Ch22 SLO 工程的核心工具);在线推理里通常退化为"长请求优先切 chunk"

SGLang 的补充:调度器同时管"prefill 预算 + running batch 上限",并做 **cache-aware 调度**(多副本时把请求路由到前缀命中的副本,Ch19 全局路由的引擎内版本)。
TRT-LLM:in-flight batching + `max_batch_size`,策略面较窄(编译式,Ch4 §4.4)。

## 6.5 [进阶] 集群级调度

**P/D 一体 vs 分离**(本章与 Ch9 的决策点):

```
一体:chunked prefill 已在"步"内解决大部分冲突
     → 负载特征:短上下文、中低并发、TTFT 不敏感
分离:prefill 与 decode 资源池独立,互不干扰
     → 负载特征:长上下文、高并发、TTFT 与 TPOT 都要 SLO
判据:看 knee/saturation 间距(Ch2 §2.5)——knee ≪ saturation 且尖刺来自
     prefill 混入,分离才有净收益;否则 KV 传输成本吃掉收益(Ch9 成本模型)
```

**DP attention**:同一 TP 组内不同 GPU 组处理不同请求的 attention(权重共享、KV 独立),等效"组内 DP"——提高 decode 并发上限;与 P/D 分离正交。

**全局队列**:多副本下的请求路由(least-queued vs 前缀感知,Ch19 §3);引擎内调度(本章)与集群路由是两级调度,**两级目标不一致时(局部最优 vs 全局 SLO)以全局 SLO 为准**。

## 动手:Chunked Prefill A/B

**实验 6.1(90 分钟,呼应 Ch2 负载 C 的变体)**:

```bash
# v1 中 chunked prefill 默认开;v0.32.0 默认 max_num_batched_tokens = 2048
#   组 1:--max-num-batched-tokens 2048(小 chunk,即当前默认)
#   组 2:--max-num-batched-tokens 8192(中档)
#   组 3:--max-num-batched-tokens 16384(大 chunk)
#   (可选对照组:--no-enable-chunked-prefill,长 prefill 独占整步的 v0 语义)
# 负载:50% 短请求(512 in/128 out)+ 10% 长请求(16K in/128 out),QPS=8
# 指标:全体 TPOT p50/p99、长请求 TTFT、吞吐
```

预期:大 chunk → 长请求 TTFT 好、全体 TPOT p99 差;小 chunk → 反之;8192 居中。把三组数据画在一张图上,就是面试里"chunk size 怎么选"的完整证据链。

**实验 6.2(读码)**:在 vLLM `scheduler.py` 的 `schedule()` 打断点,注入一条长 prompt + 20 个短请求,观察:(a) 长 prompt 每步实际分到多少 token;(b) 池满时抢占发生在哪个请求、策略是 recompute 还是 swap。

## 章末自测

1. static / dynamic / continuous 三种 batching 的区别?iteration 级为什么是甜蜜点?
2. vLLM v1 的 `max_num_batched_tokens` 同时控制什么?为什么它是 TTFT/TPOT 权衡的主旋钮?
3. 抢占 recompute vs swap 的代价公式各是什么?什么场景选哪个?
4. chunk size 太小/太大的各自代价?给"短聊天 vs 长文档 RAG"各推荐一档并说理由
5. [进阶]给一个"长上下文 + 高并发 + 双 SLO"的集群,判断该不该 P/D 分离(引用 knee/saturation 判据)

## 面试钩子

- 支撑 Ch28-B19~B26:"continuous batching 怎么调度?""chunked prefill 的 chunk 怎么选?""抢占 recompute vs swap?""为什么 v1 用 token 预算统一 prefill/decode?"
- Ch29 案例 2(低延迟 chat)的调度段、案例 5(P/D 分离)的决策段。
