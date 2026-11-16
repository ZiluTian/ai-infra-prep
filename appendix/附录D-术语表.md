# 附录 D:术语表(EN ↔ 中文)

> 全书高频术语对照。格式:英文 | 中文(一句话口径)。分组按书中出现的主章节。
> 面试里用**中文说机制 + 英文报指标**(如"前缀缓存命中率掉了,前缀感知路由要查")——两种语言切换自然 = 真用过。

## 一、阶段与指标(Ch1-2)

| 英文 | 中文 |
|---|---|
| Prefill | 预填充:一次性处理整个 prompt 的阶段,计算密集(FLOPs ∝ 输入长度) |
| Decode | 解码:逐 token 自回归生成,带宽密集(每步读全量权重 + KV) |
| TTFT (Time to First Token) | 首 token 延迟:排队 + prefill + 首个 decode 步(公式 2-1) |
| TPOT (Time per Output Token) | 每输出 token 时间:(E2E − TTFT)/(n_out − 1),decode 阶段口径 |
| ITL (Inter-Token Latency) | token 间延迟:相邻两 token 的时间差序列,尖刺定位用 |
| goodput | 有效吞吐:SLO 约束下的最大可持续输出 tokens/s(公式 2-5,生产目标函数) |
| SLO (Service Level Objective) | 服务级目标:对**分位数**(p95/p99)的承诺,不是均值承诺 |
| burn rate | 预算燃烧率:SLO 错误预算的消耗速率,双窗口(短窗快烧 + 长窗慢烧)告警 |
| load profile | 负载画像:输入/输出长度分布 + 到达率 + 前缀复用率,容量规划的第一输入 |

## 二、KV 与显存(Ch3, Ch5, Ch8)

| 英文 | 中文 |
|---|---|
| KV cache | 键值缓存:每 token 的 K/V 张量,随序列增长,显存的第一变量 |
| PagedAttention | 分页注意力:KV 按 block 非连续存放 + block table 映射,vLLM 的核心机制 |
| block table | 块表:逻辑 block → 物理 block 的映射,每请求一张 |
| prefix cache | 前缀缓存:相同前缀(系统提示词/多轮会话)的 KV 复用,命中 = 省 prefill |
| RadixCache / RadixAttention | 基数树前缀缓存:SGLang 的前缀复用结构,按最长公共前缀检索 |
| KV quantization | KV 量化:KV 存 FP8(或 INT8),显存 ÷2,近无损(长上下文退化需测) |
| sliding window attention | 滑动窗口注意力:只保留最近 W 个 token 的 KV,KV 上界与序列长度解耦 |
| KV eviction / offload | KV 逐出 / 换出:内存不够时丢 block(recompute)或换出到 CPU(swap) |

## 三、调度与批处理(Ch4, Ch6)

| 英文 | 中文 |
|---|---|
| static batching | 静态批处理:整批同进退,尾部空闲浪费(batching 的"第一级") |
| continuous batching | 连续批处理:请求级进出,完成即走、等待即入(batching 的"第二级") |
| in-flight batching | 飞行中批处理:TRT-LLM 的叫法,迭代级动态组 batch(与 continuous 同族) |
| chunked prefill | 分块预填充:长 prefill 切成 chunk 与 decode 同批,保护 TTFT/TPOT 互不尖刺 |
| token budget | 令牌预算:每步允许处理的总 token 数(max_num_batched_tokens),goodput 核心旋钮 |
| preemption | 抢占:显存不够时踢出 running 请求(重算或换出),换容量毁尾延迟 |
| recompute | 重计算:抢占后重跑 prefill(不占 CPU 内存,但有算力成本) |
| swap (KV offload) | 换出:KV 搬到 CPU 内存保留(占内存带宽,恢复快于 recompute) |
| load shedding | 负载 shed:过载时主动拒绝/降级(429),保 SLO 的最后一道闸 |
| noisy neighbor | 吵闹邻居:同池内某租户的长请求拖坏他人的延迟,池隔离的动机 |

## 四、投机解码(Ch7)

| 英文 | 中文 |
|---|---|
| speculative decoding | 投机解码:draft 猜 γ 个 + target 一次并行验证,无损(接受分布 = target 分布) |
| draft model | 草稿模型:小模型/前缀 n-gram 提供候选 token,成本必须 ≪ target 验证 |
| EAGLE | 特征级投机:用 target 中间特征 + 轻量头预测,draft 与 target 同分布 |
| Medusa | 多头投机:target 上加多个解码头并行出多个候选,无需独立 draft |
| n-gram speculation | n-gram 投机:用上下文/自重复历史提供候选,零 draft 成本,代码类负载 α 高 |
| tree attention | 树注意力:候选组织成树一次验证多分支,提高一次验证的覆盖 |
| acceptance rate (α) | 接受率:draft token 被 target 接受的概率,净加速的第一变量 |

## 五、量化(Ch8)

| 英文 | 中文 |
|---|---|
| quantization | 量化:低比特表示权重/激活,decode 从"读 140GB"变"读 35-70GB" |
| GPTQ | 逐列量化算法:基于 Hessian 的逐列误差补偿,4bit 常用 |
| AWQ | 激活感知量化:保护"显著权重"(激活大的通道),4bit 常用 |
| SmoothQuant | 平滑量化:把激活离群值难度平滑到权重,W8A8 的使能技术 |
| W4A16 / W8A8 | 权重/激活位宽组合:W4A16 省带宽, W8A8 省算力+带宽 |
| per-group / per-block scale | 组/块缩放:每 k 个权重一个缩放因子,低比特的精度保底盘 |
| E4M3 / E5M2 | FP8 两种格式:E4M3(范围小精度高,权重/激活常用)vs E5M2(范围大,梯度/离群值) |
| NVFP4 | Blackwell 的 FP4 格式:per-block scale,生态窗口期(Ch12 §12.4 决策清单) |

## 六、P/D 分离(Ch9)

| 英文 | 中文 |
|---|---|
| P/D disaggregation | P-D 分离:prefill 与 decode 分池部署,消除 KV 池竞争与扩缩容不对称 |
| disaggregated serving | 解耦式服务:P/D 分离的学术叫法(DistServe/Splitwise 一系) |
| KV transfer | KV 传输:P 完成后把 KV 搬到 D 池(带宽 + 延迟 + 与 NCCL 抢口三大成本) |
| NIXL | NVIDIA 的 KV/张量搬运传输层,block 级,与 PagedAttention 块号体系对接 |
| Mooncake | KV-centric 架构(月之暗面):以 KV 缓存为一等公民的分离式设计 |
| slot reservation | 槽位预留:D 池先占 KV 槽位 P 再开工,防止 P 白算(同步协议核心) |

## 七、并行(Ch14-17)

| 英文 | 中文 |
|---|---|
| TP (Tensor Parallelism) | 张量并行:切 GEMM(行切/列切),每层 2 次 all-reduce,天花板 = NVLink 域 |
| PP (Pipeline Parallelism) | 流水线并行:切层,点对点传激活,bubble 与 micro-batch 的权衡 |
| DP (Data Parallelism) | 数据并行:多副本各吃流量,零通信,故障域 = 1 副本 |
| EP (Expert Parallelism) | 专家并行:切 MoE expert,all-to-all 通信,负载偏斜敏感 |
| SP (Sequence Parallelism) | 序列并行(训练侧):切注意力/归一化的序列维,降激活显存 |
| CP (Context Parallelism) | 上下文并行:切序列做 attention,长 prefill/decode 的 KV 摊薄手段 |
| Ulysses | 序列并行方案:all-to-all 换 head 分布,节点内带宽足时用 |
| Ring Attention | 环形注意力:KV 沿环 P2P 轮转,可跨节点上 RDMA,与计算重叠 |

## 八、集合通信与网络(Ch13, Ch18)

| 英文 | 中文 |
|---|---|
| all-reduce | 全归约:所有 rank 求和/取回,TP 每层 2 次(节点内 NVLink 为主) |
| all-to-all | 全对全:每对 rank 都收发,EP 的 dispatch/combine |
| ring / tree (algorithm) | 环/树算法:NCCL 的两种集合通信算法,大消息 ring 带宽优、小消息 tree 延迟优 |
| NVLS (NVSwitch SHARP) | NVSwitch 内硬件规约:节点内小-中消息加速,H100 节点内默认可用 |
| busbw | 总线带宽:nccl-tests 的算法带宽口径,= algbw × 2(N−1)/N,判网络用 |
| GDR (GPUDirect RDMA) | 网卡直读显存:跨机零拷贝,不生效时带宽打对折(附录 C 排查组) |
| IBGDA | IB GPU Direct Async:GPU 直接发 IB 消息,省 CPU 转发(EP low-latency 路径) |
| PFC / ECN | 基于优先级的流控 / 显式拥塞通知:RoCE 无损以太网的两个开关,配错 = 风暴 |
| RoCE / InfiniBand | 以太网 RDMA / 专用 RDMA 网络:跨机通信的两种底座 |
| rail-optimized | 轨道优化布线:GPU i 只连 NIC i(同轨),rail 内满带宽、跨 rail 打对折 |

## 九、可靠性与容量(Ch19, Ch22, Ch29)

| 英文 | 中文 |
|---|---|
| gang scheduling | 整组调度:一个并行组(TP8/EP32)必须整组上/整组下,不能拆 |
| fault domain | 故障域:一次故障能打到多大范围(EP32 副本 = 4 节点故障域) |
| N+1 (redundancy) | N+1 冗余:容量按"挂 1 个故障域后仍达标"设计(N+1 规则,案例 9) |
| MTTR / RTO | 平均修复时间 / 恢复时间目标:进程 <2min,节点 <20min 的量级目标 |
| canary | 灰度:新版本先吃小流量,门禁过了再放量 |
| drain | 排水:节点下线前把 in-flight 请求跑完再撤(优雅下线) |
| shadow traffic | 影子流量:复制生产请求到新版本跑,不返回用户,比 canary 更安全的验证 |

## 十、硬件与 CUDA(Ch10, Ch11, Ch12)

| 英文 | 中文 |
|---|---|
| Roofline | 屋顶线模型:time = max(FLOPs/峰值算力, bytes/峰值带宽),判定 bound 的第一工具 |
| arithmetic intensity (AI) | 算术强度:FLOPs/bytes,低于 ridge 点 = memory-bound |
| memory-bound / compute-bound | 带宽受限 / 计算受限:decode 典型 memory-bound,prefill 典型 compute-bound |
| SOL (speed of light) | 光速上限:ncu 里硬件理论峰值,实测/SOL = 效率 |
| occupancy | 占用率:活跃 warp 数占上限的比例,低 = SM 空转(但低 occupancy 未必慢) |
| bank conflict | 存储体冲突:共享内存多 lane 同 bank 串行化,attention kernel 常见 |
| TMA (Tensor Memory Accelerator) | 张量内存加速器:Hopper 的异步块拷贝单元,DeepGEMM 的 multicast 靠它 |
| WGMMA | warpgroup 矩阵乘指令:Hopper FP8/FP16 GEMM 的指令级形态 |
| CUDA Graph | 计算图:把一串 kernel 录制后整体发射,消掉 per-kernel launch 开销 |
| kernel launch overhead | 内核启动开销:每 kernel 3-10μs,小 batch 时 CPU 发射成为瓶颈(公式 12-1) |
| SM / warp | 流式多处理器 / 线程束:GPU 的计算单元与 32 线程调度单位 |
| ECC / XID error | 显存纠错 / 驱动错误码:卡级故障的两种信号,XID 码决定隔离策略 |
| MFU / SM Active | 模型浮点利用率 / SM 活跃率:两种"利用率"口径,成本核算用 SM Active |
| power cap / throttle | 功耗墙 / 降频:集群供电限制下的性能天花板,验收要查 throttle 计数 |

## 十一、工具链(Ch24)

| 英文 | 中文 |
|---|---|
| DCGM | Data Center GPU Manager:卡级监控/诊断(XID、SM Active、throttle) |
| nsys | 系统级 profiler:时间线级,看 kernel 间隙、CPU-GPU 同步、launch gap |
| ncu | kernel 级 profiler:单 kernel 的 SOL/occupancy/访存效率,看 kernel 内部 |
| nccl-tests | NCCL 基准:busbw 基线 + 带载基线,网络问题的第一证据 |

## 十二、注意力与 kernel(Ch11)

| 英文 | 中文 |
|---|---|
| FlashAttention | 精确 attention 的 IO 感知实现:tiling + online softmax,不物化 N×N 矩阵 |
| FlashDecoding | 解码 attention:KV 沿 batch 维切分(split-K)并行,长 KV 下 decode 的标配 |
| FlashInfer | 变长 batch attention kernel 库:decode/prefill 混合 batch 的特化实现 |
| MLA (Multi-head Latent Attention) | 多头潜在注意力:KV 压缩为 latent(DeepSeek V3 系),KV 显存 ÷ 数倍 |
| GQA / MHA / MQA | 分组/多/单头注意力:kv_heads 数决定 KV 显存(GQA = 8 头共享一份 KV) |
| kernel fusion | 算子融合:多 kernel 合一,消中间显存读写 + launch 开销(RMSNorm+residual 典型) |
| Triton | tile 级 GPU 编程:Python 写 kernel,推理侧自定义算子的第一选择 |
| CUTLASS | NVIDIA 的 GEMM template 库:FP8/FP4 GEMM 的参考实现与起点 |

## 十三、MoE(Ch13, Ch16)

| 英文 | 中文 |
|---|---|
| MoE (Mixture of Experts) | 混合专家:每 token 只激活部分 expert,总参数大、激活参数小 |
| router / gate | 路由器:每 token 选 top-k expert 的打分层,数据相关 → EP 偏斜的根源 |
| expert | 专家:一个 FFN 子网,EP 下每卡放 1/EP 的 expert |
| grouped GEMM | 分组矩阵乘:不同 expert 的 GEMM 一次 kernel 跑完(变长 batch GEMM) |
| hot expert | 热专家:流量偏斜下 M_e 远超均值的 expert,所在卡 GEMM 放大 → p99 尖刺 |
| straggler | 掉队者:集合通信里最慢的 rank,EP 下 = 热 expert 卡,全组等它 |

## 十四、运维与成本(Ch19, Ch23, Ch29)

| 英文 | 中文 |
|---|---|
| nightly bench | 夜间基准:CI 里每晚压测,性能回归门禁(5% 回退即报警) |
| $/M token | 每百万 token 成本:卡数 × $/卡时 ÷ 实测 goodput,成本口径(Ch19 §19.4) |
| capacity planning | 容量规划:流量 → 画像 → 实测 goodput → 卡数 × 余量(附录 A 公式链) |
| autoscaling | 自动扩缩容:按队列/负载扩缩副本,回缩要滞后(防锯齿) |
| SLO-aware scaling | SLO 感知扩缩容:扩缩目标是"保 SLO 余量"不是"保利用率",带预测提前量 |
| prefix-aware routing | 前缀感知路由:同前缀请求软亲和到同副本,命中率最大化 |
| consistent hashing | 一致性哈希:副本上下线只迁移 1/N 的前缀映射,路由稳定性的基础 |
| co-location (mixed deployment) | 混部:推理空闲卡跑训练/批任务,峰谷互补(利用率核算要单独报) |

## 十五、引擎与接口(Ch4, 全书)

| 英文 | 中文 |
|---|---|
| vLLM | PagedAttention 推理引擎,本书主实验框架(v1.x 架构) |
| SGLang | RadixAttention + 结构化输出专精的推理引擎(v0.5.x 为基线) |
| TRT-LLM | NVIDIA 的图编译式引擎:in-flight batching + plugin,性能上限型选择 |
| OpenAI-compatible API | OpenAI 兼容接口:serving 的统一出口(/v1/chat/completions),压测/网关对接口 |
| structured output | 结构化输出:受约束解码(如 JSON schema),SGLang 的核心卖点 |
| sampling (top-k / top-p) | 采样:输出 logits 上的候选裁剪,top-k 截断 + top-p 核采样 |
