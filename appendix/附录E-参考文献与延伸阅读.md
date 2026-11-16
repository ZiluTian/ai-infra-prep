# 附录 E:参考文献与延伸阅读

> 用法:面试前**不需要读完所有论文**,但每篇的"对应章节"标出的机制,你要能用自己的话讲一遍(Ch30 §30.2 三段式)。
> 论文按主题分组;文档与代码仓库是"版本基线"(附录 G)的核对对象。

## E.1 论文(按主题)

### 显存管理与 KV

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP'23, vLLM) | 把 OS 分页搬进 KV:非连续 block + block table,显存浪费 <4%,吞吐 2-4× | Ch1/Ch3/Ch4(全书的显存模型起点) |
| Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024) | RadixAttention:基数树前缀复用 + 结构化程序编译,前缀命中成为一等性能因子 | Ch4/Ch6(前缀缓存 + 结构化输出) |
| *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs* (2023) | KV 逐出策略:按累计注意力权重保留 heavy-hitter token | Ch8/Ch21(KV 治理选项之一) |
| Li et al., *SnapKV* (2024) | 观察窗口 + 重要性排序压缩 KV,长上下文推理省显存 | Ch8(长上下文 KV 策略) |
| Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2023) | 流式场景保留"注意力汇聚"的前几个 token(sink),否则 streaming 崩塌 | Ch8/Ch11(滑动窗口 + sink 的组合) |
| *LMCache* (2023+, 开源) | KV 复用/offload 中间件:跨请求、跨节点的前缀 KV 共享与分层存储 | Ch9/Ch23(前缀缓存的工程化) |

### Attention kernel

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| Dao et al., *FlashAttention* (2022) / *FlashAttention-2* (ICML'23) | IO 感知 attention:tiling + online softmax,不物化 N×N;v2 把并行切到序列维 | Ch11 §11.1-11.3 |
| Shah et al., *FlashAttention-3* (2024) | Hopper 特化:WGMMA 异步流水 + FP8 attention,prefill 再提速 | Ch11 §11.3(FA v1→v3 演进) |
| *FlashInfer Technical Report* (2025) | 变长 batch attention kernel 库:decode/prefill 混合、paged KV 的特化实现 | Ch11 §11.5-11.6 |

### 并行

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| Shoeybi et al., *Megatron-LM: Training Billion-Parameter LLMs Using Model Parallelism* (2019) | TP 的列切/行切 GEMM + 每层 2 次 all-reduce 的原始设计 | Ch14/Ch15 |
| Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022) | 序列并行(SP):切 norm/attention 的序列维,激活显存 ÷TP(训练侧) | Ch14 §SP 段 |
| Liu et al., *Ring Attention with Blockwise Transformers* (2023) | KV 沿环 P2P 轮转 + 与计算重叠,长序列可扩展 | Ch17 §17.2 |

### 模型(DeepSeek 系)

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| *DeepSeek-V3 Technical Report* (2024) | 671B MoE + MLA + FP8 训练:推理侧 671B 形态(EP + attention TP + CP)的出处 | Ch13/Ch16/Ch17 |
| *DeepSeek-R1* (2025) | 长推理链负载:输出长度 ×10 量级,decode 池容量与成本的现实标尺 | Ch3/Ch19(负载画像变化) |

### P/D 分离

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI'24) | P/D 分离的定量化框架:goodput 分解 + 独立放置,双 SLO 下 2-4× goodput | Ch9(架构与判据) |
| Patel et al., *Splitwise: Efficient Generative LLM Inference with Phase Splitting* (ISCA'24) | prefill(算力机)与 decode(带宽机)分池,硬件也可异构 | Ch9 §9.3(判据的硬件版) |
| *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024/FAST'25) | KV 缓存为一等公民:全局 KV 池 + 传输与调度解耦 | Ch9/Ch23 |

### 投机解码

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (ICML'23) | 投机解码的无损性:接受/拒绝采样保证输出分布 = target 分布 | Ch7 §7.2(无损性证明梗概) |
| Chen et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (2024) | 目标模型加多个解码头并行出候选,无需独立 draft | Ch7 §7.3 |
| Li et al., *EAGLE* (2024) / *EAGLE-2* (2024) / *EAGLE-3* (2025) | 特征级自回归投机:draft 头 + 树验证,加速比与稳定性的标杆 | Ch7 §7.3(α 与加速比曲线) |
| *LLM.n-gram* (2024, 开源) | 上下文/自重复 n-gram 提供候选,零 draft 成本;代码类负载效果最好 | Ch7 §7.4(实验 7.1 的工具) |

### 量化

| 论文 | 一句话 | 书中对应 |
|---|---|---|
| Frantar & Alistarh, *GPTQ: Accurate Post-Training Quantization for LLMs* (2022) | 逐列 Hessian 误差补偿,4bit 权重的经典算法 | Ch8 §8.2 |
| Lin et al., *AWQ: Activation-aware Weight Quantization for LLMs* (2023) | 保护激活显著的通道,4bit 质量更好 | Ch8 §8.2 |
| Xiao et al., *SmoothQuant: Accurate and Efficient Quantized Inference via Channel-wise Smoothness* (ICML'23) | 把激活离群值平滑到权重,W8A8 的使能技术 | Ch8 §8.3 |

### MoE 推理(开源)

| 项目 | 一句话 | 书中对应 |
|---|---|---|
| *DeepGEMM* (deepseek-ai, 2025) | TMA multicast 的 FP8 grouped GEMM:MoE 专家计算的 kernel 标杆 | Ch12 §12.3/12.4 |
| *DeepEP* (deepseek-ai, 2025) | 两种 all-to-all 模式(normal 高吞吐 / low-latency 低延迟)的开源实现 | Ch13 §13.2/13.3 |

## E.2 文档与官方资料(必读级)

- **vLLM 官方文档** + v1 架构设计文档(`docs.vllm.ai`;v1 的 EngineCore/单进程重构、CUDA graph 分桶、调度器 token 预算行为都从这里核对)——Ch4 版本差异、Ch5 块表、Ch6 调度的核对源。
- **SGLang 官方文档**:RadixAttention、overlap scheduling、结构化输出、P/D 分离章节——Ch4/Ch6/Ch11 的 SGLang 口径核对源。
- **TensorRT-LLM 文档**:LLM API、in-flight batching、plugin 机制——"什么时候选 TRT-LLM"题(B27)的答案源。
- **NVIDIA GenAI-Perf**(原 ai-dynamo/benchmaker):LLM 压测工具(负载生成 + TTFT/TPOT 统计)——Ch24-27 实验的压测入口。
- **DCGM 文档**:XID 码表、DIAG、主机/卡级监控——Ch21 卡级故障与 Ch22 监控指纹的码表。
- **NCCL 文档**:环境变量全集(附录 C 的 30 项从这里核对默认值)、拓扑识别、SHARP 支持矩阵。
- **各框架 release notes**:vLLM v0→v1 的行为变化、SGLang 调度行为、TRT-LLM 参数——**写 Ch4 时逐条核对过,定稿前再核对一遍**(附录 G 的复核项)。

## E.3 代码仓库(走读入口)

| 仓库 | 走读重点 | 书中对应 |
|---|---|---|
| `vllm-project/vllm` | v1 的 EngineCore、scheduler(token 预算)、KV cache manager(block table)、attention 后端 | Ch4-7, Ch25 |
| `sgl-project/sglang` | RadixCache、overlap scheduling、结构化输出约束解码 | Ch4/Ch6/Ch11 |
| `NVIDIA/TensorRT-LLM` | in-flight batching、executor、plugin 接口 | Ch4(B27) |
| `flashinfer-ai/flashinfer` | paged decode/prefill kernel 的 tile 与 split 策略 | Ch11 |
| `deepseek-ai/deep_ep` | normal/low-latency 两种 all-to-all 的实现差异 | Ch13 |
| `deepseek-ai/DeepGEMM` | TMA multicast + FP8 grouped GEMM | Ch12 |
| `NVIDIA/nccl-tests` | all_reduce_perf / alltoallv_perf 基线 | Ch18, 附录 C |
| `ai-dynamo/benchmaker` (GenAI-Perf) | LLM 压测负载生成 | Ch24-27 |
| `microsoft/llm-perf` | 跨引擎压测对比(辅助口径) | Ch2 §压测口径 |
| `kvcache-ai/Mooncake` | KV 池 + 传输层设计 | Ch9 |

> 走读纪律(Ch30 §30.2):每个"读过源码"的说法必须能落到**文件 + 方法级**,且带验证它的实验。仓库的 main 分支漂移快——**走读时锁 commit,登记到附录 G**。

## E.4 使用建议

1. **面试前 7 天**:只读 E.1 里标"书中对应 Ch7/Ch9/Ch13"的论文摘要 + 你简历里用到的项目的论文(PagedAttention / DistServe / EAGLE 三篇是最高频追问源)。
2. **面试前 3 天**:E.2 的 vLLM v1 设计文档过一遍(版本行为题的弹药),附录 C 的 NCCL 变量过一遍。
3. **不要做的事**:逐字背论文。面试官问的是"你踩过什么",不是"论文摘要是什么"——论文是**你的踩坑经验的理论锚点**,不是背诵材料。
