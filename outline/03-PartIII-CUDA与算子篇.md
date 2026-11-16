# Part III CUDA 与算子篇:建立"kernel 视角"

> **目标**:读完能看懂 ncu 报告、能写 Triton kernel、能判断"这个 op 该不该自己写"。
> **难度**:★★★~★★★★★ | **合计** ~120 页

---

## 第 10 章 GPU 架构与性能建模

> 难度 ★★★ | ~30 页 | 前置:无

### 大纲
1. **硬件模型**
   - SM / warp / shared memory / L2 / HBM
   - H100 SXM:132 SM、HBM3 80GB 3.35TB/s、FP16 ~989 TFLOPS、FP8 ~1979 TFLOPS、NVLink 900GB/s
   - H200:141GB、4.8TB/s(带宽 +44%)→ decode 类负载 ~30-40% 提升
   - B200:HBM3e 7.7TB/s(2026-11-16 复核)、FP4(NVFP4)tensor core
2. **Roofline 与算术强度(全书性能分析的地基)**
   - AI = FLOPs / bytes;ridge point = peak FLOPS / peak BW
   - 手算:decode GEMV AI ≈ 1-2 → 深度 memory-bound;prefill GEMM AI ≈ 数百 → compute-bound
   - 含义:decode 优化 = 榨带宽(batch↑、量化、KV 压缩);prefill 优化 = 榨算力(FA3、FP8)
3. **Profiling 基础**
   - nsys:timeline(kernel / memcpy / gap),找 "GPU 空转"
   - ncu:单 kernel SOL(occupancy、memory %、compute %)
   - 读图示范:一份真实 decode timeline 的逐步解读

### 分层
- [基础] 1-2;[进阶] 3。

### 动手 / 面试钩子
- 动手:nsys 抓 7B decode,量 GEMM 实际带宽利用率 vs HBM 峰值(给出"只有 ~60%"的典型发现)。
- 面试钩子:支撑 Ch28-A15~A20("为什么 decode 是 memory-bound?怎么验证?")。

---

## 第 11 章 Attention 算子优化

> 难度 ★★★★ | ~30 页 | 前置:Ch10

### 大纲
1. **为什么 attention 必须融合**
   - naive:物化 n×n attention 矩阵 → O(n²) 显存流量
   - IO-aware:分块计算、中间结果不落 HBM(FlashAttention 核心思想)
2. **FlashAttention v1 → v2 → v3**
   - v1:tiling + online softmax,O(n) 显存、~5×
   - v2:work 分法(sequence 维并行)、减少 HBM 读写
   - v3(Hopper):WGMMA / TMA / warp specialization、异步流水,较 v2 再 1.5-2.0×
3. **FlashInfer:推理 attention kernel 库**
   - paged attention kernel(decode)、sampling 融合、"变长 batch"(ragged/variable-length)
   - 为什么 SGLang 选它:decode kernel 性能 + 工程完整度
   - kernel 结构走读:block 划分、online softmax、split-K(长 decode 时的 FlashDecoding)
4. **长上下文 attention**
   - Ring Attention / CP attention(衔接 Ch17)、sliding window、FlashDecoding
   - MLA(DeepSeek):KV 压缩为 latent vector,decode kernel 特化,KV 体积降 5-10×

### 分层
- [基础] 1-2、4;[进阶] 3 + MLA。

### 动手 / 面试钩子
- 动手:写一个简化版 Triton paged attention,与 flash-attention 对比正确性与性能。
- 面试钩子:支撑 Ch28-C01~C08("FA v2 与 v3 的区别?FlashInfer 解决什么问题?")。

---

## 第 12 章 Triton、CUTLASS 与 GEMM

> 难度 ★★★★★ | ~30 页 | 前置:Ch10

### 大纲
1. **Triton 编程模型**
   - block(tile)思维、自动 pipelining、autotune
   - 递进教程:vector add → 融合 RMSNorm → matmul → grouped matmul(MoE 用)
2. **GEMM 为王**
   - tensor core 利用:MMA / WGMMA;layout(TN/NT)
   - CUTLASS:Collective Builder、写一个 FP8 GEMM(per-block scale)
   - DeepGEMM:JIT + TMA multicast(配合 TP),DeepSeek 推理选它的原因
3. **kernel launch 与 CUDA Graph**
   - 3-10μs launch 开销 × 数百 kernel → 小 batch decode 是 launch-bound
   - CUDA Graph:静态图捕获、按 batch size 分桶捕获、动态 shape 时为什么用不了
4. **什么时候该自己写 kernel**
   - 决策清单:融合 op(rmsnorm+residual)、paged/ragged 布局、低精度 GEMM、自定义 epilogue(bias/relu/residual)
   - 常见坑:shared memory bank conflict、L2 locality、occupancy 权衡

### 分层
- [基础] 1;[进阶] 2-4(整章面向资深读者)。

### 动手 / 面试钩子
- 动手:写 fused RMSNorm + SiLU 的 Triton kernel,与 PyTorch 逐算子对比。
- 面试钩子:支撑 Ch28-C09~C16("Triton GEMM 怎么写的?CUDA Graph 的本质与限制?")。

---

## 第 13 章 MoE 算子与推理

> 难度 ★★★★★ | ~30 页 | 前置:Ch12
> 注:整章 [进阶],路线 A 读者建议只读 1(数据流)。

### 大纲
1. **MoE 推理数据流**
   - router(softmax + top-k)→ token dispatch → expert GEMM → combine
   - DeepSeek-V3 实例:256 routed experts + 1 shared、top-8、首尾层 dense
2. **all-to-all dispatch**
   - EP 通信:本机 token → 目标 expert 所在卡
   - DeepEP:normal(节点内 NVLink、逐 token)vs low-latency(跨节点 RDMA、分组、IBGDA)
   - 通信量精算:`batch × seq × topk × hidden × bytes`
3. **Grouped GEMM**
   - 每个 expert 的 M 维长度不同:padding vs 按 expert 排序
   - CUTLASS grouped GEMM、TRT-LLM MoE kernel、vLLM `fused_moe`
   - 零 token expert:跳过 kernel、launch 开销
4. **负载均衡**
   - 数据相关路由 → 热 expert → straggler → 长尾(衔接 Ch22)
   - 缓解:capacity factor + drop、热 expert 复制、router bias、aux-loss(训练侧)

### 动手 / 面试钩子
- 动手:插桩 vLLM MoE forward,输出每 expert token 分布 + all-to-all 耗时占比。
- 面试钩子:支撑 Ch28-C17~C24("MoE 的 EP all-to-all 怎么做?grouped GEMM 怎么高效?")。
