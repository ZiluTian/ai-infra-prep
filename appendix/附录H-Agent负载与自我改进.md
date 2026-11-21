# 附录 H:Agent 负载与自我改进(推理基础设施视角)

> **定位**:本附录不教 agent 框架怎么用,只回答一个问题:**agent 类负载与"自我改进"循环对推理 serving 栈意味着什么**——负载形态变了,哪些配置、容量、指标、路由策略要跟着变。
> **与正文的关系**:Ch4 §4.5 / Ch5 §5.4 把 agent 循环列为 SGLang radix 甜区(各一行表);Ch29 案例 7 给了 agentic 长文档的容量算账。本附录把这条线展开:会话负载的参数化画像(H.1)、serving 配置推导(H.2)、两层自我改进的负载形式与成本模型(H.3 agent 级 / H.4 模型级)、两种 serving 模式对照(H.5)、面试钩子(H.6)。
> **数据纪律**:本附录新增的生态事实(§H.1.1 框架表、§H.4.3 RL 训练框架表)一律 **[官方, 2026-11, 未实机核验]**,登记于附录 G §G.2(同 TRT-LLM 先例);负载画像参数为 **[假设]**,成本模型为 **[估算]**,替换成实测前按假设使用。vLLM v0.32.0 / SGLang v0.5.22 的引擎行为**复用附录 G §G.5 已验证口径**,不重复登记。

## H.0 范围与口径

- **覆盖**:agent 框架生态的"负载发生器"视角、agent 会话的量化负载画像、agent 负载的 serving 配置推导、agent 级自我改进(Reflexion / memory / self-refine)的负载形式、模型级自我训练 / RL 后训练的 rollout serving 成本模型、GRPO 的单步更新流程、两种 serving 模式对照。
- **不覆盖**:框架本身的编排 API 用法(那是框架文档的事);训练算法的完整推导;agent 应用设计(提示词工程、工具设计)。
- **编号**:公式 (H-1)~(H-7),与附录 A §A.1 登记一致;每个公式带一个手算。
- **可复现**:§H.1.2 / §H.2.1 用 `labs/agent_load.py`(实服务待核验);§H.4.9 用 `labs/grpo_mini.py` 在 CPU 上复现 (H-6)/(H-7) 并训练一个玩具策略。两个脚本见附录 F §F.1。

## H.1 Agent = 新负载类:从 request 到 session

### H.1.1 [基础] 生态一页:框架是"负载发生器",不是 serving 栈

2026-11 时点的主流 agent 编排框架(均 **[官方, 2026-11, 未实机核验]**,仓库见附录 E §E.3):

| 框架 | 编排模型 | 与推理引擎的接口 | 对 serving 的负载含义 |
|---|---|---|---|
| LangGraph(LangChain 生态) | 图 / 状态机,节点 checkpoint 化 | OpenAI 兼容 `base_url`(provider SDK 可指向任意 OpenAI API 服务) | 图分支 = 前缀分叉;确定性节点 → 到达率较稳 |
| OpenAI Agents SDK | agent + handoff + guardrail + session | OpenAI 兼容协议;`base_url` 指向任意 OpenAI 兼容服务(即 vLLM/SGLang) | session 对象 → 长生命周期多轮上下文 |
| CrewAI | 角色化多 agent 流水线(sequential / hierarchical) | 同上 | 流水线串行 → 会话内 burst(agent 交替调用) |
| AG2(AutoGen 系) | 会话式多 agent(actor 模型) | 同上 | agent 互聊 = 上下文互相引用 → 高前缀复用 |
| LlamaIndex Workflows | 事件驱动 | 同上 | 事件扇出 → 并发短时 burst |
| sgl-router(SGLang 官方) | **非编排**:多副本路由器 | 引擎侧(位于 serving 栈内) | cache-aware 路由 = agent 负载的一阶杠杆(§H.2.2) |

**结论(基础设施视角):所有编排框架都走 OpenAI 兼容协议(chat completions),serving 栈面对的是一个协议、多种负载形态。** 框架差异不体现在协议上,而体现在三处:① 到达过程(人 pacing vs 扇出 burst);② 前缀共享结构(单链多轮 vs 树状分叉);③ 会话生命期(分钟级交互 vs 小时级长任务)。vLLM / SGLang 暴露 OpenAI 兼容 `/v1` 端点一事是本书已验证口径(附录 G §G.5 ③f:双真实服务上跑通 `labs/` 全部 HTTP 脚本)。

### H.1.2 [基础] 会话负载的参数化模型(公式 H-1)

一个 agent 会话可参数化为:**session := (S, K, {Δ_k}, {R_k})**

- S:system prompt + few-shot 示例(全会话共享,全服务共享的也可有,如租户级 system)
- K:会话轮数;Δ_k:第 k 轮新增的用户输入 token;R_k:第 k 轮的工具结果 token
- 关键语义:chat 协议下第 k 轮的请求重发**完整历史**——

**公式 (H-1) 累积 prompt 与前缀账**

```
P_k   = S + Σ_{i≤k} (Δ_i + R_i)            第 k 轮请求的 prompt token(近似)
C₀    = Σ_k P_k                           naive prefill 总账(无缓存,每轮全量重算)
C₁    ≈ P₁ + Σ_{k≥2} (Δ_k + R_k + A_{k-1}) 缓存全命中时的实际 prefill(A = 上轮模型输出)
η     = 1 − C₁ / C₀                        prefill 节省率
```

**手算**(参数为 **[假设]**,方法通用;`agent_load.py --dry-run` 可换任意参数复算):
S = 8K,Δ = 400,R = 600(每轮),K = 10,忽略 A 或视为已并入 Δ:

- P_k = 8000 + 1000k → P₁ = 9K,P₁₀ = 18K
- C₀ = 10×8000 + 1000×(1+…+10) = 80K + 55K = **135K**
- C₁ ≈ 9K + 9×1K = **18K** → **η = 1 − 18/135 ≈ 86.7%**
- 单轮命中率:第 k 轮重发 P_k 中可命中的是 (P_k − 1K) → k=1 时 ≈ 8/9 ≈ 89%,k=10 时 ≈ 17/18 ≈ 94%——**复用率随轮数单调上升**,这是 agent 负载区别于一次性请求的结构性特征(Ch5 §5.4 的 50% 复用是静态画像;这里是动态上升的)。
- 折算 FLOPs(37B 级,2×37e9 FLOPs/token,Ch17 口径):C₀ → 135K × 7.4×10¹⁰ ≈ 1.0×10¹⁶;C₁ → 18K × 7.4×10¹⁰ ≈ 1.3×10¹⁵。

### H.1.3 [进阶] 到达过程与 KV 占用

- **会话内串行,跨会话并行**:一轮之内 LLM 调用是串行的(工具等待穿插其中,工具不在 GPU 上跑);并发来自**多个会话同时存活** → 负载画像是"高并发、单会话低 QPS",与 Ch2 压测的泊松请求流不同:这里**并发数**比到达率更能描述负载。
- **到达 burst 的三种来源 [假设]**:① 会话启动(首轮长 prefill,S 可能 8-32K);② 工具扇出(一个 agent 节点并行调 m 个工具,汇合后一次 LLM 调用);③ 重试尖峰(§H.3.1 的 retry 风暴,负反馈来源)。
- **KV 占用 = Σ_session |当前上下文|,且长驻**:agent 会话的 KV 不像一次性请求"用完即释",它在两轮之间一直占着(等用户/等工具)→ **KV 池需求由"并发会话数 × 平均上下文"决定,是容量问题,不是吞吐问题**。这是 agent 负载容量规划与 Ch3 容量链接口的地方:把 λ(Ch3 的 (3-4) 需求吞吐)换成"并发会话数 × 平均上下文 token",单请求 KV 用 (3-2) 算,每卡可并发用 (3-3) 算。

### H.1.4 [进阶] 结构化输出 / tool call 的性能代价

- **机制**:tool call / JSON schema 输出走 grammar 约束解码(每步在词表上施加合法 mask)。SGLang 的 `grammar_backend='xgrammar'` 为默认配置 **[已验证, 附录 G §G.5 ③b]**;vLLM 的 guided decoding(xgrammar / outlines 类后端)**[官方, 2026-11, 未实机核验]**。
- **代价量级 [估算]**:mask 生成是 CPU 侧操作(xgrammar 类实现做了增量填充,单步增量很小),相对每步的 GPU kernel 时间通常可忽略;**主要影响是正确性(输出 100% 符合 schema),不是延迟**。高并发 + 深层 schema 时,grammar 填充的 CPU 成本**可能**进入 overlap 调度的 CPU/GPU 关键路径(Ch6 的 overlap 时序)——这是"可能"不是"必然",用 nsys 定位,不先入为主。
- 解码约束:约束解码通常要求 beam = 1(合法集在 beam 间不对称),这与 agent 场景一致(agent 调用本来就是 greedy / 低温度单路解码)。

> **踩坑(症状 → 误诊 → 正解)**
> - 症状:开了 JSON schema 约束 + temperature 0.7,输出分布"变怪了"(与无约束时不可比)。
> - 误诊:怀疑约束解码有 bug / 温度没生效。
> - 正解:约束解码把采样重归一化到**合法 token 子集**上,分布与无约束时本来就不一样——这是预期行为。对照实验要"约束 ON vs OFF、同温度",别跨温度比。

## H.2 Serving for agents:从公式推配置

### H.2.1 前缀缓存是一阶杠杆

- **默认状态(已验证口径)**:vLLM v0.32.0 `--enable-prefix-caching` 存在且默认开(实服务 config dump 实证 `enable_prefix_caching=True`);SGLang v0.5.22 RadixCache 默认开 + `radix_eviction_policy='lru'`(附录 G §G.5 ③a③b)。→ agent 负载的 prefix 缓存**不需要"开",需要"核对 + 保护"**(下面几条都是"保护")。
- **单轮 TTFT 模型**——

**公式 (H-2) 会话单轮 TTFT**

```
TTFT_k ≈ t_queue + t_prefill,k
t_prefill,k ≈ P_k / r_prefill          冷轮(前缀未命中,全量 prefill)
t_prefill,k ≈ (Δ_k + R_k + A_{k-1}) / r_prefill   热轮(前缀命中,只算增量)
```

**手算**(37B 级,8×H100 TP8;r_prefill 的估算链:Ch17 口径 1M prefill 单卡 ~100s → 单卡 ~10K tok/s,×8 卡 prefill 近线性、TP 效率 ~0.6 **[估算]** → r_prefill ≈ 50K tok/s):

- 冷轮 k=10:P₁₀ = 18K → 18K / 50K ≈ **0.36 s**(不含排队)
- 热轮 k=10:增量 1K → 1K / 50K ≈ **0.02 s**(不含排队)
- 冷/热比 ≈ 18×(= P₁₀ / 增量,与 H-1 的账同源)——**这就是 `agent_load.py` 输出的"TTFT 冷/热比",前缀缓存生效与否的代理指标。**
- **高并发稀释(Ch5 §5.4 结论,此处复用)**:命中率高 ≠ TTFT 降 50%。t_queue 主导时(高并发),复用 KV 不会让队列变短 → "开了前缀缓存 TTFT 没降多少"在高并发下是正常现象,先按 Ch20 L2 把 TTFT 分解成 排队 / prefill / 网络 三段再下结论。

### H.2.2 会话亲和与 cache-aware 路由

- **问题**:多副本部署下,同一会话的相邻两轮落到**不同副本** = 上一轮存下的前缀 KV 在本副本根本不存在,"存了但用不上"。agent 会话轮间隔短(秒级 [假设]),轮询 LB 下跨副本概率高 → 前缀缓存收益被路由随机性吃掉一半以上。
- **对策**:
  - SGLang:`sgl-router` 的 cache-aware 模式(按前缀亲和把会话粘到同一副本,偏斜时回退 shortest-queue)**[官方, 2026-11, 未实机核验]**。Ch4 §4.5 表把"cache-aware 路由"列为 SGLang 调度特色,本节是它的负载口径展开。
  - vLLM:引擎内无会话亲和概念 → 亲和在**外部路由层**实现(自研 header 亲和,或 llm-d 类 KV-aware 路由 **[官方, 2026-11, 未实机核验]**)。
- **代价(必须一起讲)**:cache-aware 用**局部性换均衡**——会话集中会让个别副本热(它的 KV 池先满,LRU 逐出更激进,§H.2.3)。所以 cache-aware 路由都要带回退机制(偏斜超阈值 → 转队列最短),这是 SGLang 官方口径 [官方, 未实机核验]。

> **案例(匿名综合,无具体数字)**
> 某多副本 agent 网关:轮询 LB 下,会话两轮常落不同副本,前缀缓存"存了但用不上",命中率低、热轮按冷轮计费(时间)。改 cache-aware 路由(会话亲和 + 偏斜回退)后,热轮命中回升,TTFT 冷/热比恢复到 H-1 预测的量级。方向与量级可用 `agent_load.py` 在本环境 A/B 验证(双副本 + 手动粘会话 vs 轮询,看 TTFT 冷/热比),具体数字以实测为准。

### H.2.3 KV 保留与 idle 会话:LRU 逐出的 TTFT 尖峰

- **机制**:缓存 block 是 LRU 逐出(Ch5 §5.4,SGLang `evict()` 与书稿伪代码一致 **[已验证, 附录 G §G.5 ③b]**;vLLM 单 block LRU)。agent 会话的低 QPS 长命 KV,在 idle 期间(用户思考 / 工具执行)会被**高 QPS 负载的 KV 挤出**。下一轮回来时前缀已不在 → 该轮按冷轮计费,**TTFT 尖峰**——尖峰只出现在"idle 超过缓存驻留时间"的会话上,均值指标看不出来,要按会话分桶看 TTFT p95。
- **杠杆(按代价从小到大)**:
  1. **副本 KV 预算**:提高单副本 KV 池占比(vLLM 的权重/KV 配比、SGLang `--mem-fraction-static` **[已验证旗标名]**)——直接抬高"idle 多久才被逐出"的上限;
  2. **chunked prefill 吃长工具结果**:工具结果可能很长(一次检索塞回 8-16K)。vLLM `max_num_batched_tokens` 默认 2048、SGLang `chunked_prefill_size` 自动默认 4096 **[已验证, 附录 G §G.5 ③a③b]**——长工具结果轮被切成多步,不阻塞其他会话的首 token;
  3. **长上下文会话走 P/D**(Ch9):上下文 >64K 的 deep-research 类会话,prefill 池 + decode 池分离,避免长 prefill 与长 KV 互相挤。
- **指标(已验证口径)**:SGLang `sglang:cache_hit_rate` / `sglang:cached_tokens_total` / `sglang:kv_used_tokens` / `sglang:num_running_reqs`;vLLM `request_prefill_time_seconds` / `request_queue_time_seconds` / `time_to_first_token_seconds`。

### H.2.4 TTFT SLO 与按会话数扩缩

- **agent 是"延迟模式"**:交互式 agent 的每次 LLM 调用都有 TTFT 预算(工具等待占 wall-time 大头,模型侧必须快且**稳** [假设])。SLO 落在 TTFT p95(每轮),不是吞吐——goodput 口径同 Ch2/Ch27 §27.5(SLO 内的 token)。
- **扩缩信号 = 并发会话数 × 平均上下文,不是 QPS**:QPS 可以很低(人 pacing),但 KV 占用很高(会话长驻)。按 QPS 扩缩会在"会话多、每会话慢"的形态下**系统性欠容**。操作化:监控 `num_running_reqs`(SGLang)× 平均序列长度,对 KV 池水位设阈值(如 80%)触发扩容;Ch19 的 N+1 规则照用(N 由"并发会话数 × 平均上下文"算出)。

### H.2.5 Agent 负载配置速查

| 负载特征 | 一阶杠杆 | 关键指标 | 关联 |
|---|---|---|---|
| 多轮共享前缀 | 前缀缓存(默认开,核对即可) | cache_hit_rate;TTFT 冷/热比 | Ch5, §H.2.1 |
| 多副本部署 | 会话亲和 / cache-aware 路由(带回退) | 每副本命中率、副本间队列不均度 | §H.2.2 |
| 长工具结果 | chunked prefill(默认预算 2048 / 4096) | TTFT 尖峰计数、chunk 步数 | Ch6 |
| idle 会话回跳 | KV 预算 / 保留;超长上下文走 P/D | kv_used_tokens、按会话分桶的 TTFT p95 | Ch3 / Ch9, §H.2.3 |
| 并发会话 burst | 按"会话数 × 上下文"扩缩,非 QPS | KV 池水位、num_running_reqs | §H.2.4 |
| 结构化输出 / tool call | grammar 引擎(SGLang 默认 xgrammar) | schema 违反率(= 0)、CPU 占用 | §H.1.4 |

## H.3 自我改进 ①:agent 级(Reflexion / memory / self-refine)

### H.3.1 模式 → 负载形式

agent 级自我改进的三种主流模式(论文见附录 E §E.1)**[官方, 2026-11]**:

1. **Reflexion 式**:执行 → (失败) → 自我反思生成文字 → 反思附加进上下文 → 重试。**负载形式:近似重复 prompt**——重试轮的 prompt = 上一轮 prompt + 反思尾缀,前缀 100% 共享 → 前缀缓存的黄金负载。
2. **Self-Refine 式**:生成 → 自评(评分 + 批评)→ 精修 → 再评。**负载形式:每任务 2-3 倍调用量**,且"自评"调用是长输入短输出(评的是完整输出)。
3. **候选并行(T-o-T 式 / best-of-N)**:同一 prompt 分 N 路候选并行生成,再选优。**负载形式:前缀树分叉**——N 个候选共享同一主干,在分叉点之后各走各的。Ch5 §5.4 已把 "tree-of-thought / 多候选" 列为 SGLang radix 的甜区;vLLM 的定长 block 哈希池对"中前缀分叉"的复用粒度较粗(Ch5 表"命中条件"行)。
4. **memory 式**:跨会话沉淀记忆(向量库 / 摘要),每轮把检索到的记忆拼进 prompt。**负载形式:prompt 单调增长**(检索片段不重复但持续追加)→ KV 驻留成本单调上升,§H.2.3 的 LRU 尖峰更易触发。

**负载乘数**——设单次尝试成功率 p,上限 n 次,每任务 1 个 judge 调用:

**公式 (H-5) 重试期望(几何截断)**

```
E[尝试] = (1 − p^n) / (1 − p)
m(每任务 LLM 调用数) ≈ E[尝试] × (1 + judge 比)
```

**手算**:p = 0.7,n = 5 → E[尝试] = (1 − 0.7⁵)/0.3 = (1 − 0.168)/0.3 ≈ **2.77**;judge 1:1 → **m ≈ 5.5 次调用/任务**。容量规划必须用 m 后的 λ′ = m × λ,否则容量按"每个任务一次调用"算,直接欠容 3-6×(p 在 0.5-0.8、n 在 3-5 的典型区间 [假设])。

三种负载增量汇总:**① 近似重复 prompt(反思尾缀)→ 前缀缓存;② 候选并行 = 前缀树分叉 → radix 甜区(§H.3.1.3);③ retry 风暴 = 负反馈负载尖峰**(失败触发重试,失败率越高重试越多 → 尖峰与故障同源叠加,需要会话级限流 + 重试预算 m 上限 + 退避;这是 Ch22 长尾治理的 agent 特化版)。

> **踩坑(症状 → 误诊 → 正解)**
> - 症状:某 agent 服务白天平稳,一到"任务失败率高"的时段负载 3× 尖峰,p99 崩。
> - 误诊:加卡 / 升配。
> - 正解:尖峰是**负反馈**——失败本身在制造负载。先限重试(会话级重试预算 + 指数退避),把 λ′ 的天花板钉死,再谈容量。加卡只是在给负反馈供血。

### H.3.2 judge 调用:长输入短输出的第二形态

LLM-as-judge / 自评调用 = **完整轨迹进、短结论出**,与 Ch29 案例 7 的"长文档进、短输出出"同构 → prefill 重、decode 轻。serving 含义:judge 调用可以**与 policy 调用不同池**(judge 池 = 高 batch prefill 池,P/D 或 chunked 友好;policy 池保 TTFT SLO)。若 judge 与 policy 同模型(自奖励风格,§H.4.4),无额外模型,但**同池**时 judge 的大 prefill 会挤 policy 的 TTFT → 至少路由分层。

### H.3.3 基础设施杠杆汇总

| 自我改进模式 | 负载增量 | 一阶杠杆 |
|---|---|---|
| Reflexion(重试 + 反思) | 近似重复 prompt × m | 前缀缓存(重试轮天然热);重试预算钉死 λ′ |
| Self-Refine(自评循环) | 调用量 ×2-3;judge 长输入短输出 | judge 分层(不同池 / 路由);前缀缓存 |
| 候选并行 best-of-N | 前缀树分叉 N 叉 | SGLang radix(任意长度前缀 + 分叉);vLLM block 对齐粒度粗 |
| memory 沉淀 | prompt 单调增长 | KV 预算 / 保留策略;超长走 P/D |

## H.4 自我改进 ②:模型级(自训练 / RL 后训练)→ rollout serving

### H.4.1 管线:每步的推理侧都是一个独立负载

模型级自我改进(自训练 / 自奖励 / RL 后训练)的通用管线 **[官方, 2026-11;论文见附录 E]**:

```
base → ① rollout(用当前策略权重批量生成)→ ② 验证(规则 / reward model / 自奖励)
     → ③ 更新权重(SFT on 验证子集,或 RL:PPO / GRPO / DAPO)→ 循环
```

**对基础设施的关键观察:管线的每一步都是一类独立的推理负载**——① rollout(长 CoT decode,吞吐模式)、② 验证(长输入短输出,§H.3.2 的 judge 形态)、③ 权重同步(I/O 密集,epoch 边界)。RL 后训练的成本大头在①,有官方口径的佐证:NVIDIA NeMo-RL v0.3 官方 blog(2025-08)的 Llama-70B GRPO 步时分解:**总步 147s,其中 generation 84s(57%)、policy 训练 28s [官方]** → generation 是 RL 步的第一成本。

GRPO 如何把 rollout 的分数变成模型权重的更新，见 §H.4.7。

### H.4.2 为什么 rollout 是推理基础设施问题

**长 CoT decode 的 KV 账**——2026 年推理模型的输出长度已到 16-128K 量级 [假设,主流推理模型口径]:

**公式 (H-3) 单序列 KV 显存**

```
KV_bytes(seq) = 2 × L × H_kv × d_h × T × b
(T = 序列长度,b = 每元素字节;与 Ch3 (3-2) 同源)
```

**手算**(70B GQA:L = 80,H_kv = 8,d_h = 128,与 Ch29 案例 7 / 附录 A §A.3 同口径):

- 每 token:2 × 80 × 8 × 128 × 2B = **320 KB/token**(FP16)
- T = 64K → 320KB × 65536 ≈ **20 GiB/seq**(FP8 KV → 10 GiB/seq)
- **容量链(Ch3 方法,与 Ch29 案例 7 同池口径:8×H100 80GB,权重 140GB(FP16,TP8 → 17.5GB/卡),KV 池 ≈ 60GB/卡 → 总池 ~480GB [假设])**:并发上限 ≈ 480 ÷ 20 ≈ **24 条 64K 长 CoT 序列 [估算]**;FP8 KV → ~48 条。
- **decode 吞吐量级 [估算]**:memory-bound(Ch10/Ch11),每步读 ≈ 权重 17.5GB/卡 + KV → 步时 ~5ms 量级 → batch 24 时聚合 ~4K tok/s。

**负载形态结论**:rollout = **吞吐模式**——无 SLO(可以跑一整夜),目标是把 batch 打满到 KV 上限,指标是 tok/s/GPU 与 $/GPU-h;与 agent 会话的延迟模式(H.5)恰好相反。

### H.4.3 训推协同框架与三个核心接口

RL 后训练框架把 vLLM/SGLang/TRT-LLM 当作 rollout 引擎嵌入训练循环。2026-11 时点主流口径(均 **[官方, 2026-11, 未实机核验]**,仓库见附录 E §E.3):

| 框架 | rollout 引擎 | 协同方式 | 备注 |
|---|---|---|---|
| verl(volcengine) | vLLM(Ray 集群内) | HybridFlow:训练与 rollout 同集群编排,权重经 NCCL 同步 | 学术/工业采用最广 |
| OpenRLHF | vLLM(独立 rollout actor) | 分离式:trainer 与 rollout worker 解耦部署 | 工程化早期方案 |
| TRL(HuggingFace) | 训练侧算法库(SFT/DPO/PPO/GRPO),配外部 vLLM | 算法层,不绑定 serving 拓扑 | 研究入口 |
| SkyRL(NovaSky AI / SkyworkAI) | vLLM(构建于 VeRL 之上) | 全异步 RL:**in-flight 权重更新** + staleness manager(capacity rule 约束生成不超龄) | agent RL(SWE-Bench 系长程任务)主战场;v0.5.0(2026-11) |
| NeMo-RL(NVIDIA) | vLLM 后端(可扩展 TRT-LLM / SGLang) | Ray 编排 + Megatron/DTensor 训练后端;async rollouts;大规模 MoE 权重传输优化(官方 blog 报告 10×,2025-08) | 与 NeMo 栈一体 |

**三个核心接口(与框架无关的抽象)**:

1. **权重同步(trainer → rollout 引擎)**:全量重载 / NCCL 广播 / 增量。成本——

**公式 (H-4) 权重同步开销**

```
t_sync ≈ W_bytes / b_load;overhead = t_sync / t_epoch
```

**手算**(70B,W = 140GB FP16 / 70GB FP8):

- NVMe 顺序读 ~3GB/s [假设] → 140/3 ≈ **47 s**(FP8 → ~24 s)
- 节点内 NCCL(NVLink 900GB/s,Ch18 实测口径)→ 140/900 ≈ **0.16 s [估算]**(跨节点受 IB 限制,量级上升)
- overhead 占比:t_epoch = 600 s [假设] 时,NVMe 路径 7.8%,NCCL 路径 <0.3% → **权重同步在"训推同节点 + NCCL"下不是矛盾;在"分离部署 + 走盘"下是 epoch 边界的固定税**。
- 增量同步(delta):研究口径 [官方, 2026-11],实践中少见——全量重载 + 量化权重(FP8/FP4 把 W_bytes 直接减半)是主流杠杆。

2. **异步管线**:train batch i 的同时生成 batch i+1 → rollout 与训练在时间上重叠(GPU 利用率的关键;NeMo-RL 官方报告 async rollouts 使多轮 RL 快 2-3× [官方])。代价 = 生成与训练用的权重**不再同 epoch** → 引出接口 3。
3. **off-policy 管理**:给 rollout 打 epoch 标签;生成落后训练超过 staleness bound 的 batch,丢弃或重加权。SkyRL 的 capacity rule(`accepted + running ≤ capacity`,由 staleness bound 推出)是这一机制的公开实现 [官方, 未实机核验]。

> **踩坑(症状 → 误诊 → 正解)**
> - 症状:异步 RL 管线 reward 曲线"看起来正常",但周期性评测掉点。
> - 误诊:怀疑学习率 / 数据 / 奖励函数。
> - 正解:先查**权重 epoch 对齐**——rollout 侧权重落后训练侧若干 epoch = 静默 off-policy,数据被污染但日志**不报错**。正解:rollout 带 epoch 标签 + staleness bound 硬约束;这是"最难查的一类故障",因为它不产生任何异常信号。

> **案例(匿名综合,无具体数字)**
> 某异步 RL 管线,rollout 引擎与 trainer 分离部署、权重走共享盘全量重载(epoch 边界 40-90 s 量级 [估算,见 H-4 手算区间])。两个独立问题:① epoch 边界的同步窗口里 rollout 停摆(GPU 空转);② 异步生成偶发跨 epoch,污染数据。改法:同步窗口与 rollout 重叠(生成用上一 epoch 权重 + staleness bound),权重改节点内 NCCL 广播。两者都是接口 1/3 的标准问题,不需要新框架。

### H.4.4 RM / judge serving

- **RM 打分 = 完整 completion 的 prefill**(打分对象是整条输出)→ 长输入、短输出(标量/短文本)→ 与 §H.3.2 的 judge 同构:高 batch prefill 池、chunked、P/D 或独立池,别和 policy rollout 混池。
- **自奖励变体(self-rewarding)**:policy 模型兼任 judge,无额外模型,但**调用量 ×2 量级** [假设]——rollout 池要按"生成 + 自评"两段算容量,且两段资源画像不同(生成 = decode 重,自评 = prefill 重,同池会互相干扰)。

### H.4.5 [估算] 1M rollouts 的 GPU-hour 算例

**全部假设(显式列出,替换成实测前按估算使用)**:70B 级;rollout = 8K prompt + 32K CoT;prefill 用 Ch17/Ch29 口径(单卡有效 ~700 TFLOPS BF16,8 卡 = 5.6 PFLOPS);decode 聚合 ~4K tok/s(batch ~24,§H.4.2);单组 TP8(8×H100)。

| 项 | 计算 | 结果 |
|---|---|---|
| prefill 总量 | 1M × 2×70e9×8K = 1.12×10²¹ FLOPs ÷ 5.6×10¹⁵ | **≈ 2×10⁵ s ≈ 55 h**(单 TP8 组) |
| decode 总量 | 1M × 32K = 3.2×10¹⁰ tok ÷ 4K tok/s | **≈ 8×10⁶ s ≈ 2200 h**(单 TP8 组) |
| decode 占比 | 2200 / (2200 + 55) | **≈ 97.5%** |
| 32 节点(32 组 TP8)并行 | 2255 h ÷ 32 | **≈ 70 h ≈ 3 天 [估算]** |

**自洽性对照**:Ch29 案例 7 的"8K prefill ≈ 250ms @ TP8"→ 1M × 0.2s ≈ 55 h,与上表 prefill 行同量级 ✓;NeMo-RL 官方步时分解中 generation 占总步 57%(§H.4.1)方向一致 ✓。

**结论(量级,可背)**:RL 后训练的推理成本 **>97% 在 decode** → 优化只盯 decode(KV 预算、KV 量化、TP 深度、batch 打满);prefill(包括同 prompt 多 rollout 的共享前缀)是**次要矛盾**——这与 agent 延迟模式恰好相反(agent 侧 prefill/TTFT 才是主矛盾,H.5)。同 prompt 的 G 个 rollout(GRPO group)仍有前缀缓存价值,但那是"省 55h 里的一部分",不改变量级。

### H.4.6 RL 后训练负载速查

| 阶段 | 负载形态 | 一阶杠杆 | 关联 |
|---|---|---|---|
| rollout 生成 | 长 CoT decode,吞吐模式,batch 打满 | KV 预算 / KV 量化 / TP / max_running_requests | Ch3 / Ch10 / Ch11, §H.4.2 |
| 同 prompt G rollouts | 前缀分叉 G 叉 | radix 前缀缓存 | Ch5, §H.3.1 |
| RM / 自奖励打分 | 长输入短输出,prefill 重 | 高 batch prefill 池 / P-D / 独立池 | Ch9, §H.3.2 |
| 权重同步(epoch 边界) | I/O 密集,窗口内停摆 | 节点内 NCCL / 量化权重 / 异步重叠 | Ch18, §H.4.3 |
| 数据管线(验证/过滤) | CPU / IO,与 GPU 解耦 | 流水化,不占 GPU 关键路径 | Ch2 |

打分、数据校验与权重同步的伪代码见 §H.4.8；可运行的算法实验见 §H.4.9。

### H.4.7 GRPO：从 rollout 奖励到权重更新

**GRPO（Group Relative Policy Optimization，组相对策略优化）**以同一个 prompt 的多条回答为一组。训练程序用同一份策略权重从原 prompt 启动 `G` 条采样序列。每条序列逐 token 生成，以上一步的序列作为上下文，直到生成结束符或达到长度上限。`G` 是训练配置。各条序列可以并行生成，共享 prompt 的前缀缓存，各自维护生成的 token。

> **案例 [假设]：手算一次 GRPO 更新。** 给 `17 × 6 = ?` 采样四条回答，校验器对正确的最终答案给 1 分，对错误答案给 0 分。采样和打分都不更新模型；它们为 trainer 生成训练样本。

| Rollout | 奖励 `rᵢ` | 优势 `Aᵢ` | 策略损失的更新方向 |
|---|---:|---|---|
| `60 + 42 = 102` | 1 | +1 | 提高采样 token 在各自上下文中的概率 |
| `17 × 6 = 112` | 0 | −1 | 降低采样 token 在各自上下文中的概率 |
| `120 − 18 = 102` | 1 | +1 | 提高采样 token 在各自上下文中的概率 |
| `17 × 6 = 96` | 0 | −1 | 降低采样 token 在各自上下文中的概率 |

**从分数到优势。** 奖励是绝对得分，优势是相对同 prompt 的组内表现。用总体标准差 `σ` 手算（非零时忽略数值稳定项）：

```text
Aᵢ = (rᵢ − mean(r)) / σ(r),  σ(r) > 0                 (H-6)
```

组均值 `(1 + 0 + 1 + 0) / 4 = 0.5`，四个偏差的平方都是 `0.25`，所以 `σ = √0.25 = 0.5`。正确回答的优势为 `(1 − 0.5) / 0.5 = +1`，错误回答为 `(0 − 0.5) / 0.5 = −1`。只有整条回答的奖励时，该回答的所有生成 token 共用一个优势值：校验器知道最终答案是否正确，却不能直接定位哪一步算错。只保留正确回答做 SFT 也是一种做法，但它没有这里对低于组均值样本的负向策略信号。

> **踩坑：平均奖励不等于有效学习信号。** `[0, 0, 0, 0]` 与 `[1, 1, 1, 1]` 的组均值不同，但标准差都为零。下方伪代码为分母加一个小量，两组的优势都为零，因此都不产生**奖励驱动的策略梯度**；若启用参考模型 KL，仍可能有正则梯度。二元奖励下，同时监控平均奖励和**混合奖励组占比**：用同时出现 0 和 1 的完整有效组数除以完整有效组总数；一般标量奖励则看组内方差非零的比例。

**从优势到策略更新。** Trainer 把 `prompt + 已生成的回答` 输入当前模型，在**同一上下文**下重新计算每个采样 token 的 log-probability。旧策略是生成该批样本的策略快照；旧 log-probability 是固定分母。

```text
ρᵢₜ = exp(log p_current − log p_old)
Jᵢₜ = min(ρᵢₜ × Aᵢ, clip(ρᵢₜ, 1−ε, 1+ε) × Aᵢ)     (H-7)
```

**手算 [假设]：**若一个正优势 token 的概率从 `0.10` 升到 `0.13`，则 `ρ = 1.30`。取 `ε = 0.20`、`A = +1`，目标取 `min(1.30, 1.20) = 1.20`；继续抬高该比值不再为这项增加收益。负优势时看相反方向：若 `A = −1`、`ρ = 0.70`，则取 `min(−0.70, −0.80) = −0.80`；继续压低也无额外收益。裁剪的是**训练目标中的贡献**，不是把模型概率硬锁在某个范围。Trainer 最小化负目标，`loss.backward()` 沿当前策略的前向计算图传递梯度，`optimizer.step()` 才真正更新权重。

**PyTorch 风格伪代码（一个 prompt、一次更新）。** `sample_with_logprobs` 从模型的 softmax 分布逐 token 采样，并返回对应的 log-probability。示例使用相同的概率分布来采样和打分，且每条回答至少包含一个 token。

```python
def response_logprobs(model, prompt_ids, response_ids):
    tokens = cat([prompt_ids, response_ids])
    logits = model(tokens[None, :]).logits[0]  # 当前前向计算各层激活

    # 位置 t 的 logits 预测位置 t+1 的 token。
    start = len(prompt_ids) - 1
    response_logits = logits[start : start + len(response_ids)]
    logprobs = log_softmax(response_logits, dim=-1)
    return logprobs.gather(-1, response_ids[:, None]).squeeze(-1)


# 采样阶段：保存旧策略实际采样 token 的 log-probability。
with no_grad():
    rollouts = [sample_with_logprobs(model, prompt_ids) for _ in range(group_size)]
    rewards = tensor([score(prompt_ids, ids) for ids, _ in rollouts], dtype=float32)
    advantages = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-8)

# 训练阶段：从相同的 token 序列重新做有梯度的前向计算。
optimizer.zero_grad()
response_losses = []
for (response_ids, old_lp), advantage in zip(rollouts, advantages):
    new_lp = response_logprobs(model, prompt_ids, response_ids)
    ratio = exp(new_lp - old_lp)
    clipped_ratio = clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    objective = minimum(ratio * advantage, clipped_ratio * advantage)
    response_losses.append(-objective.mean())

loss = stack(response_losses).mean()
loss.backward()
optimizer.step()
```

在同步且采样与训练打分一致的情形，第一次更新前 `new_lp` 与 `old_lp` 数值相等，`ratio == 1`，裁剪不生效，但梯度仍可能非零：`old_lp` 是常量，`new_lp` 随当前权重变化。这里仅演示裁剪后的策略损失；原始 GRPO 目标还包含相对参考策略的 KL 惩罚。**旧策略**提供这批数据的概率比基准，**参考策略**提供 KL 锚点，不是同一个角色。[官方：DeepSeekMath §4.1 与算法 1](https://arxiv.org/html/2402.03300#S4.SS1)

| 方法 | 更新信号从哪来 | 对 rollout serving 的影响 |
|---|---|---|
| LLM RLHF 中常见的 PPO | 训练 critic 估计给定部分回答时的预期后续回报，再构造 token 优势 | 在线采样，同时承担 critic 训练开销 |
| GRPO | 同 prompt 多条回答的组内奖励作为优势估计的基础，不训练 critic | 要为同一 prompt 采样多条回答；相对 PPO 没有固定的总算力倍数 |
| 标准离线 DPO | 利用已有的 chosen/rejected 偏好对 [官方：DPO, 2023](https://arxiv.org/abs/2305.18290) | 优化循环内不需要生成新 rollout；偏好数据仍须事先获得 |

### H.4.8 打分、数据校验与权重同步

下面以同步 GRPO 为例。每条 rollout 记录所属的 `group_id`、prompt、回答、生成 token、这些 token 在采样时的 log-probability，以及生成时的策略版本。每个 `group_id` 对应同一个 prompt 的一组采样。

**1. 奖励打分。** `reward_fn` 读取一条完整回答。规则校验器可以给 `17 × 6` 的答案打 0 或 1 分，reward model 或自评模型也可以为整条回答给出一个标量分数。

```python
def score_rollouts(rollouts, reward_fn):
    for rollout in rollouts:
        rollout.reward = reward_fn(rollout.prompt, rollout.response)
    return rollouts
```

**2. 数据校验与分组。** 本例要求每组恰好有 `G` 条记录，使用同一策略快照，并为每个生成 token 保存一个旧 log-probability。组不完整或记录无效时跳过整组。答错但得分有效的样本仍保留在组内，参与组内平均奖励和优势值的计算。

```python
def prepare_grpo_groups(rollouts, expected_version, group_size):
    groups = []
    for group in group_by(rollouts, key="group_id"):
        if len(group) != group_size:
            continue

        if any(
            r.policy_version != expected_version
            or len(r.response_tokens) == 0
            or len(r.response_tokens) != len(r.old_logprobs)
            or not is_finite(r.reward)
            for r in group
        ):
            continue

        groups.append(group)
    return groups
```

**3. 权重同步。** Trainer 完成优化器更新后导出快照。同步模式下，所有 rollout worker 确认加载新版本后，下一批采样才开始。

```python
def sync_rollout_workers(trainer, workers, new_version):
    snapshot = trainer.export_weights()
    parallel_for(
        workers,
        lambda worker: worker.load_weights(snapshot, version=new_version),
    )
    wait_until_all_workers_report_version(workers, new_version)
```

把三个阶段接回同步训练循环。`trainer.grpo_update` 对各组的已打分样本执行 §H.4.7 的优势值计算、损失计算和权重更新。

```python
for prompts in prompt_batches:
    rollouts = generate_rollouts(prompts, G, workers, version)
    scored = score_rollouts(rollouts, reward_fn)
    groups = prepare_grpo_groups(scored, version, group_size=G)

    if groups:
        trainer.grpo_update(groups)
        sync_rollout_workers(trainer, workers, new_version=version + 1)
        version += 1
```

### H.4.9 Mini Lab：在 CPU 上跑一遍 GRPO

**目标。** 亲手复现 §H.4.7 的两个结论：组内奖励无差异时没有策略梯度；第一次更新前概率比为 1，但混合奖励组仍能推动学习。再观察复用旧 rollout 时，裁剪何时生效。

**环境。** CPU + Python 3 标准库；无需 GPU、模型下载、PyTorch 或在线服务。脚本用四道乘法题，每题四个候选答案，每个候选答案相当于一个 token。每题有独立的一行 logits；初始 softmax 均匀，正确答案概率是 `1/4 = 0.25` [假设]。这不是 LLM 训练：没有多 token 信用分配、KL、跨题参数共享或跨进程权重同步。`rollout()` 保存旧 log-probability 并给出优势；`objective_and_grad()` 对同一批样本重算概率比并求解析梯度；`train()` 更新 logits，再用新策略采样。

在仓库根目录运行（Windows 若未配置 `python`，用 `py` 替换）：

```bash
python labs/grpo_mini.py --self-test
python labs/grpo_mini.py
python labs/grpo_mini.py --group-size 1
python labs/grpo_mini.py --updates-per-batch 1
```

| 对照 | 看什么 | 怎么解释 |
|---|---|---|
| `--self-test` | `PASS` | 复算 `[1, 0, 1, 0] → [+1, −1, +1, −1]`、全同奖励、正负方向裁剪；用有限差分校验解析梯度 |
| 默认：每题 8 条 rollout、每批更新 4 次、seed 7 | `expected_reward` 在第 20 步从 `0.250` 升至 `0.948`；第 1 批 `clipped_terms` 为 `2.3%` | 复用固定旧 log-probability 的样本后，当前策略开始偏离生成策略，部分项进入裁剪分支 |
| 只改 `--group-size 1` | `mixed_groups` 始终 `0/4`，`gradient_norm` 为 `0`，`expected_reward` 仍为 `0.250` | 每组只有一个分数，与自身均值相等；即使采样碰巧答对，也不会更新 |
| 只改 `--updates-per-batch 1` | `clipped_terms` 为 `0%`，但首批 `gradient_norm` 非零 | 每批只在生成该批数据的策略上求一次梯度，裁剪不生效不等于没有学习信号 |

**读数口径。** 表中运行输出为 Windows/Python 3.13 的玩具实验复现记录（附录 G §G.5），不是生产训练或 serving [实测]。`sampled_reward` 是更新前采样得到的平均奖励，会波动；`expected_reward` 是更新后在同四道**训练题**上正确选项的平均概率，不是泛化准确率。`gradient_norm` 只记每批首次更新，`clipped_terms` 则统计该批全部更新中的样本项。

**动手题。** 保持 seed 不变，只把组大小从 8 改成 2，比较前几批 `mixed_groups` 与每批生成的样本数。再想想：若每题要生成的不再是一个 token，而是长 CoT，增加组大小能换来更多有学习信号的组，但会让 rollout 引擎多承担什么成本？回看 §H.4.2 和 §H.4.5。

## H.5 两种 serving 模式对照(agent 延迟模式 vs RL 吞吐模式)

| 维度 | agent 会话(延迟模式) | RL rollout(吞吐模式) |
|---|---|---|
| 首要目标 | TTFT p95 预算(每轮) | 无 SLO;GPU-hour 预算 |
| 并发决定量 | 并发会话数 × 平均上下文(KV 长驻) | batch 打满到 KV 上限 |
| 前缀缓存角色 | 会话内多轮复用(一阶杠杆) | 同 prompt G rollouts 共享前缀(次要) |
| 路由 | 会话亲和 / cache-aware(保 KV 局部性) | 不需要(同 prompt 打散无影响) |
| 扩缩信号 | 会话数 × 上下文;KV 池水位 | epoch 剩余 GPU-hour |
| 核心指标 | TTFT p95、cache_hit_rate | tok/s/GPU、$/epoch |
| 典型配置 | chunked prefill 小预算防尖峰;会话保留 | max_running_requests 拉满;P/D 可选 |
| 权重生命周期 | 固定(模型更新走发布) | 每 epoch 换(同步成本在关键路径) |

**一句话**:同一套引擎(vLLM/SGLang),两种负载**参数方向相反**——agent 侧把延迟做平(小 chunk、保留 KV、会话粘副本),RL 侧把吞吐做满(大 batch、打满 KV、权重随 epoch 滚动)。混在一个池里 = 两边都受损;分层或分池是默认正确答案。

## H.6 面试钩子(对齐 Ch28 口径)

1. **"为什么 agent 负载是 SGLang radix 的甜区?"** ① 多轮 = 任意长度前缀复用(vLLM 定长 block 池要按 16-token 对齐,radix 任意长度);② 候选并行 / T-o-T = 前缀树分叉,radix 子树天然表达;③ 逐出是子树 LRU,多轮会话的整条历史一起进一起出。代价要一起说:radix 的 CPU 开销(树维护)与"复用表达得越细、调度器工作越多"。
2. **"多轮会话 TTFT p95 恶化,你的排查顺序?"** ① 按 Ch20 L2 把 TTFT 分解:排队 / prefill / 网络;② cache_hit_rate 掉了?→ 路由局部性(§H.2.2)或 LRU 逐出(idle 会话,§H.2.3);③ kv_used_tokens 水位?→ KV 预算 / 扩容(§H.2.4);④ 长工具结果轮?→ chunked prefill 预算。**顺序 = 从"分解"到"归因",每步都有对应指标,不猜。**
3. **"估算 1M RL rollouts 的 GPU-hours?"** 公式链:H-3 算 KV/seq(70B GQA 64K FP16 ≈ 20GiB)→ 容量链算并发(480GB 池 → ~24)→ decode 吞吐量级(~4K tok/s @ batch 24)→ 3.2×10¹⁰ tok ÷ 4K tok/s ≈ 2200 GPU-h;prefill 55 h 可忽略(占 2.5%)。结论句:decode 占 >97%,优化只盯 decode。
4. **"RL 管线里权重同步的开销与风险?"** H-4 手算:NVMe 47 s vs 节点内 NCCL <0.3 s(70B FP16);overhead 取决于部署拓扑(同节点 NCCL → 可忽略;分离走盘 → epoch 边界固定税 5-10%)。风险不是性能,是**正确性**:异步生成跨 epoch = 静默 off-policy(踩坑框),正解 = epoch 标签 + staleness bound。
5. **"agent 负载与 RL 负载,serving 配置差在哪?"** H.5 表挑三个最大差异:SLO(TTFT p95 vs 无)、路由(会话亲和 vs 不需要)、权重生命周期(固定 vs 每 epoch 换)。收尾:同引擎、参数方向相反,混池必分层。
6. **"GRPO 平均奖励上升，为什么有效梯度可能反而减少?"** 看组内分布：全对和全错都不给相对奖励信号；平均奖励不能替代混合奖励组占比。先查这两个指标，再决定是否需要调整题目难度或采样配置（§H.4.7）。

## H.7 数据纪律与版本声明

- **[官方, 2026-11, 未实机核验](附录 G §G.2 登记)**:§H.1.1 生态框架表(LangGraph / OpenAI Agents SDK / CrewAI / AG2 / LlamaIndex Workflows / sgl-router)、§H.4.1 管线口径、§H.4.3 RL 训练框架表(verl / OpenRLHF / TRL / SkyRL / NeMo-RL,含 SkyRL staleness manager 与 NeMo-RL async rollouts / 权重传输 10× / Llama-70B 步时分解等官方 blog 数据)。引用时按"官方口径、未实装核验"表述,不写"我们验证过"。
- **[假设]**:§H.1.2 / §H.1.3 的画像参数(S/Δ/R/K/τ)、§H.3.1 的 p 与 n 区间、§H.4.2 的 KV 池 480GB(与 Ch29 案例 7 同口径)、§H.4.4 的调用量倍数。
- **[估算]**:公式 H-2 的 r_prefill、H-3 的容量链结论(24 / 48 并发)、H-4 的同步带宽、§H.4.5 全算例(55h / 2200h / 97.5% / 3 天)。
- **算法公式与教学算例 (§H.4.7)**：GRPO 按 DeepSeekMath（2024）§4.1，标准离线 DPO 按原论文（2023），为[官方]论文口径，非框架版本验证；(H-6)/(H-7) 的分数与概率均为[假设]，伪代码不代表已运行的训练实验。
- **复用已验证(附录 G §G.5,不重复登记)**:vLLM v0.32.0 默认行为(prefix 缓存 / chunked prefill / block 16 / 指标名)、SGLang v0.5.22 默认行为(radix + lru / overlap / xgrammar / chunked_prefill_size / disaggregation / 指标名)、双服务 OpenAI 兼容端点可用性。
- **配套脚本**:`labs/agent_load.py`(附录 F §F.1 第 8 个)——复现 H-1 的账(naive vs 增量 prefill、η)与 H-2 的冷/热 TTFT 比。**诚实登记:2026-11-19 新增,`--dry-run` 已在开发机验证(负载构造与账目逻辑),实服务路径未核验**,待 4×5090 环境跑通后按 §G.5 口径补登记。
