# Part I 基础篇:把"推理"这件事算明白

> **目标**:建立全书的量化思维。读完 3 章,读者应能:手算任意模型的 KV 显存、读懂并设计压测、解释 prefill 与 decode 为何优化手段完全不同。
> **难度**:★★~★★★ | **合计** ~75 页

---

## 第 1 章 LLM 推理的本质:为什么 serving 和训练是两回事

> 难度 ★★ | ~25 页 | 前置:无

### 大纲
1. **从一次前向到一次 serving**
   - LLM 单次 forward 结构:embedding → N × (attention + MLP) → head
   - Prefill 与 Decode:同一个模型、两种计算形态(这是全书最重要的分野)
   - 关键数据:prefill = 大 GEMM、compute-bound,算术强度(arithmetic intensity)~数百 FLOP/byte;decode = 大 GEMV、memory-bound,AI ≈ 1-2 FLOP/byte
2. **自回归生成与 KV Cache 的必然性**
   - 为什么必须缓存:不缓存则每步 O(n²) 重算
   - KV Cache 随 seq × batch 线性增长
   - GQA / MQA:Qwen2、Llama2/3 如何压缩 KV(kv heads < q heads)
3. **为什么推理"难"(定位章)**
   - 三重矛盾:显存(KV)× 计算(prefill)× 延迟(decode)
   - 为什么"堆卡"不万能:并行引入通信,TP 有规模上限
   - 小结图:推理系统的三大杠杆——显存管理 / 计算效率 / 调度

### 分层
- [基础] 全章。
- [进阶] 用 Roofline 定量推导 7B/70B decode 步的 AI(衔接 Ch10);attention 复杂度分解 O(L·n·d) vs O(L·n²·d)。

### 动手 / 面试钩子
- 动手:小脚本测量单 decode step 耗时随输出长度变化,验证 KV 的线性增长(为 Ch3 做铺垫)。
- 面试钩子:支撑 Ch28-A01~A08("为什么 decode 是 memory-bound?""GQA 省了什么?")。

---

## 第 2 章 性能指标体系与压测方法论

> 难度 ★★ | ~25 页 | 前置:Ch1

### 大纲
1. **指标的精确定义(面试高频)**
   - TTFT = 请求到首 token:排队等待 + prefill + 首个 decode
   - TPOT = (E2E − TTFT) / (输出 token 数 − 1);ITL = 每 token 间隔(波动性)
   - 吞吐:输出 tokens/s;goodput = SLO 约束下的吞吐(区别于裸吞吐)
   - p50/p95/p99:尾部才是真实 SLO
2. **负载模型(压测结果可信的前提)**
   - 到达过程:泊松 / 固定 QPS / 突发
   - prompt:长度分布(真实 trace:ShareGPT、AzureLLM)、输出长度分布、前缀复用率
   - 为什么合成负载必须声明假设——否则结果不可比
3. **工具链**
   - vLLM bench_serving、NVIDIA GenAI-Perf、MS LLMPerf
   - 采样与测量:服务端日志 vs 客户端计时;warmup;冷启动
4. **常见坑(真实案例框)**
   - GPU 降频(power cap)、显存碎片未预热、限流与重试污染数据
   - 如何写一份"可信"压测报告:固定 seed / 模型 / GPU 状态 / 三次重复

### 分层
- [基础] 1-2;[进阶] 3-4 + 设计 SLO 约束的吞吐实验(goodput 曲线)。

### 动手 / 面试钩子
- 动手:对 7B 模型做一次完整压测并出报告(模板见 Ch24)。
- 面试钩子:支撑 Ch28-E01~E06("TTFT 怎么测才准?""goodput 和吞吐的区别?")。

---

## 第 3 章 显存模型与容量规划

> 难度 ★★★ | ~25 页 | 前置:Ch1
> **定位:全书公式基石,建议最先写、最先读。**

### 大纲
1. **显存的三大块**
   - 权重 / KV Cache / 激活(+ CUDA context 与框架开销 ~1-2GB)
   - `gpu-memory-utilization` 实际在控制什么
2. **权重体积**
   - FP16:70B → 140GB;FP8 → 70GB;INT4 → ~35-40GB(含 scale 开销)
   - MoE 特殊性:总参数 vs 激活参数(DeepSeek-V3:671B 总 / 37B 激活)→ **显存看总量,算力看激活量**
3. **KV Cache 精算(本章核心公式)**
   - `KV/ token = 2 × L × h_kv × d_head × bytes`
   - 手算示例:Llama3-70B(80 层、8 kv heads、128 dim、FP16)→ 320KB/token
   - GQA/MQA、FP8 KV、sliding window 各自如何改变该值
   - 反推:8×H100-80G 能并发多少条 32K 上下文
4. **容量规划方法**
   - 从流量(req/s × 平均输出长度)到卡数;余量 1.3-1.5×
   - preemption 与 swap 在容量中的角色(衔接 Ch21)
   - MoE 容量规划:专家副本与 KV 竞争

### 分层
- [基础] 1-3;[进阶] 4 + MoE / KV offload(CPU/SSD 分级)的容量估算。

### 动手 / 面试钩子
- 动手:容量规划计算器(Python,放入 labs/,后续实验复用)。
- 面试钩子:支撑 Ch28-A09~A14("70B + 32K 上下文要多少显存?""100B MoE 怎么规划卡数?")。
