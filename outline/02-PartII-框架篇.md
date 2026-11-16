# Part II 框架篇:vLLM / SGLang / TensorRT-LLM 源码级

> **目标**:从"会拉服务"到"能读码、能改调度、能加 kernel"。以 vLLM v1 为主线,SGLang 对照(调度与 KV 管理差异),TensorRT-LLM 补充(编译式路径)。
> **难度**:★★★~★★★★★ | **合计** ~165 页

---

## 第 4 章 推理引擎架构总览

> 难度 ★★★★ | ~35 页 | 前置:Ch1-3

### 大纲
1. **请求生命周期:从 OpenAI API 到 token 输出**
   - API 层:协议、tokenization、chat template
   - EngineCore 主循环(每个 iteration):预算 → 调度 → 执行 → 采样 → 状态更新
   - ModelRunner:输入组装(block table、CUDA graph 选择)、forward、sampling
   - KV Cache Manager:分配 / 释放 / 逐出
2. **vLLM v1 架构(源码走读,主线)**
   - 双进程:API server ↔ EngineCore(为什么 v0 要重设计:调度/执行耦合、扩展性)
   - unified scheduler + 按 batch size 捕获 CUDA graph
   - 代码地图:`vllm/v1/core/{scheduler,kv_cache_manager}`、`v1/worker/gpu_model_runner`
3. **SGLang:RadixAttention + 零开销调度**
   - radix tree KV、自动前缀复用、结构化输出(压缩 FSM)
   - overlap scheduling:GPU 跑当前 batch 时 CPU 调度下一个 batch
4. **TensorRT-LLM:编译式路径**
   - Executor / runtime graph、in-flight batching、plugin 机制(自定义 attention/MoE)
   - LLM API(高层)vs raw executor;何时选 TRT-LLM(极致优化、C++ 生态、自定义 kernel)
5. **对比与选型**
   - 对比表:调度粒度 / kernel 生态 / 精度支持 / MoE 支持 / 部署形态 / 二次开发成本

### 分层
- [基础] 1、5;[进阶] 2-4(源码级,含关键类调用链与 v0→v1 变更点)。

### 动手 / 面试钩子
- 动手:带断点读 vLLM v1 scheduler,画出请求状态机图。
- 面试钩子:支撑 Ch28-B01~B10("vLLM v0 为什么重写成 v1?""SGLang 的 RadixCache 怎么实现?")。

---

## 第 5 章 PagedAttention 与 KV Cache 管理

> 难度 ★★★★ | ~30 页 | 前置:Ch3

### 大纲
1. **连续 KV 的浪费**
   - 16-100% 浪费(paper 数据);来源:按 max_len 预分配 + 内部碎片
   - OS 类比:虚拟内存 / 页表 / 4KB 页 → 16-token block
2. **PagedAttention 算法**
   - block table:逻辑 block → 物理 block
   - attention kernel 改造:按 block 分块算 Q×Kᵀ,online softmax 累加
   - 代价:kernel 开销 + 间接寻址;实测 decode 不慢反快(带宽不被浪费)
3. **block 池与生命周期**
   - 引用计数、free list、block 共享(多请求共享前缀 block)
   - 源码:vLLM v1 `BlockPool` / `KVCacheManager`;SGLang `RadixCache`
4. **Prefix Caching 与 RadixAttention**
   - 前缀 block 哈希、命中即复用(system prompt / few-shot / 多轮)
   - radix tree:节点引用计数、LRU 逐出
   - 定量示例:50% 前缀复用的负载下 TTFT 下降 60-80% 的推导
5. **[进阶]KV Cache 进阶**
   - 量化:FP8 KV(`kv-cache-dtype`)、INT4 KV、精度损失
   - 选择性保留:H2O、SnapKV、StreamingLLM
   - offload:LMCache、分级(HBM → CPU → SSD),何时值得

### 分层
- [基础] 1-4;[进阶] 5。

### 动手 / 面试钩子
- 动手:50% 共享前缀负载下开/关 prefix caching 压测对比(Ch25 复用)。
- 面试钩子:支撑 Ch28-B11~B18("PagedAttention 怎么实现?代价是什么?""前缀缓存逐出策略?")。

---

## 第 6 章 Continuous Batching 与调度器

> 难度 ★★★★ | ~30 页 | 前置:Ch4

### 大纲
1. **batching 的三级跳**
   - static(等最长)→ dynamic(batch 边界插入)→ continuous(每个 iteration 插入)
   - 为什么 iteration 级是甜蜜点:GPU 利用率 vs 实现复杂度
2. **vLLM v1 调度器剖析**
   - waiting / running 两队列;每步 token 预算(`max_num_batched_tokens`)
   - 调度决策:这一步放多少 prefill token + 多少 decode token
   - 抢占:recompute vs swap(何时选哪个、代价、KV 丢失语义)
3. **Chunked Prefill**
   - 问题:长 prefill 阻塞 decode(head-of-line blocking)→ TPOT 尖刺
   - 方案:prefill 分块、与 decode 混 batch
   - chunk size 选择:512/2048/8192,权衡曲线(TPOT 稳定性 vs prefill 效率)
   - vLLM:`--enable-chunked-prefill`;SGLang 内置 chunked prefill
4. **调度策略**
   - 默认 FCFS;priority、fair(份额)、SJF / deadline-aware(EDF)
   - SGLang:batch + prefill 预算;TRT-LLM:runtime batch 上限
5. **[进阶]集群级调度**
   - P/D 一体 vs 分离的决策(引出 Ch9)
   - request 分片(DP attention)、全局队列

### 分层
- [基础] 1-4;[进阶] 5。

### 动手 / 面试钩子
- 动手:chunked prefill 开/关,长短混合负载下对比 TPOT p99。
- 面试钩子:支撑 Ch28-B19~B26("continuous batching 怎么调度?""抢占 recompute vs swap?")。

---

## 第 7 章 Speculative Decoding

> 难度 ★★★★ | ~25 页 | 前置:Ch1、Ch6

### 大纲
1. **原理与数学**
   - 起草 γ 个 token → 一次并行验证(tree attention)
   - 接受率 α、期望收益推导(几何分布;验证开销计入后的净加速公式)
   - 为什么无损:验证分布与目标模型一致(rejection sampling 证明梗概)
2. **draft 来源谱系**
   - 小模型(Leviathan)、Medusa(多头自回归)、EAGLE(特征级自回归)、EAGLE-2/3(免训练动态树)
   - n-gram / 检索(LLM.n-gram:从上下文自取,RAG/高重复场景最佳)
   - self-drafting:layer skip / look-ahead
3. **工程实现**
   - vLLM:`--speculative-model` / EAGLE 路径;SGLang:`--speculative-algorithm EAGLE`
   - 源码:draft forward、tree mask、采样与验证衔接
   - α 监控与动态调参
4. **什么时候不 work**
   - 高 batch:GPU 已打满,验证开销 > 收益
   - 低 α(代码/数学/长尾词)、短输出
   - 经验表:batch × α × γ → 净加速

### 分层
- [基础] 1、2、4;[进阶] 3 + batch-aware 投机解码(按请求选择性投机)。

### 动手 / 面试钩子
- 动手:7B + EAGLE 投机解码,不同 batch size 下测加速比(验证"高 batch 失效"曲线)。
- 面试钩子:支撑 Ch28-B27~B33("投机解码为什么无损?什么时候不值得用?")。

---

## 第 8 章 量化与数值格式

> 难度 ★★★★ | ~25 页 | 前置:Ch3、Ch10(荐)

### 大纲
1. **数值格式**
   - FP16/BF16;FP8:E4M3(前向)/ E5M2,FP8 tensor core 算力 ≈ 2× FP16(H100)
   - FP4:NVFP4 / MXFP4(Blackwell),group scale
   - INT8/INT4:对称/非对称、per-channel / per-group scale
2. **后训练量化(PTQ)**
   - GPTQ(Hessian 二阶)、AWQ(激活感知、保护 salient 通道)、SmoothQuant(通道迁移)
   - weight-only W4A16:GEMV 友好,NF4
   - 校准集:128-512 条真实样本
3. **W8A8 / FP8 推理**
   - TransformerEngine、DeepGEMM:FP8 GEMM 与 per-tile scaling
   - KV FP8:近无损、显存减半
4. **精度评估方法**
   - perplexity 不够:任务 benchmark + 长上下文退化 + few-shot 退化
   - MoE:router / shared expert 敏感度
   - 量化模型的压测基线(接 Ch2)
5. **[进阶]**
   - QAT、动态量化、per-expert 量化
   - B200 + FP4:DeepSeek 系 NVFP4 部署实践

### 分层
- [基础] 1-3;[进阶] 4-5。

### 动手 / 面试钩子
- 动手:同一模型 GPTQ/AWQ 4bit vs FP8 vs BF16 三方压测 + 精度对比。
- 面试钩子:支撑 Ch28-B34~B40("FP8 为什么 2× 吞吐?W4A16 怎么实现?KV 量化损失?")。

---

## 第 9 章 P/D 分离(Disaggregated Serving)

> 难度 ★★★★★ | ~20 页 | 前置:Ch5、Ch6、Ch18(荐)
> 注:整章 [进阶],路线 A 读者可了解决策部分(1、3)。

### 大纲
1. **为什么分离**
   - prefill:compute-bound、短突发;decode:带宽-bound、长驻 → 资源画像冲突
   - 同机的代价:prefill 拖 decode(TPOT 尖刺)、decode 占显存(prefill 排不进来)
   - 何时值得:高并发 + 长上下文 + TTFT SLO;短上下文中等负载不值得
2. **架构与 KV 传输**
   - P 副本 → D 副本;KV 体积精确计算(Ch3 公式套用)
   - RDMA(IB/RoCE)零拷贝传输;NIXL / Mooncake Transfer Engine
   - 同步协议:请求路由、完成通知、失败与重试语义
3. **成本模型**
   - KV 传输时间 vs 省下的排队等待;示例:128K 上下文 70B GQA FP16 ≈ 40GB/请求 → 必须 FP8 KV / 传输前压缩
   - P:D 动态配比:prefill 队列深度 vs decode batch 负载
4. **开源实现**
   - DistServe / Splitwise(paper)、Mooncake(1024 卡级 KV-centric)、vLLM production disagg、SGLang PD 支持
   - 源码点:KV 传输层、调度被拆到两侧后各自怎么决策

### 动手 / 面试钩子
- 动手:两节点 P/D 实验(或单机模拟分离),测 TTFT/TPOT 改善与网络开销(Ch27)。
- 面试钩子:支撑 Ch28-B41~B44 与 Ch29 案例 5("P/D 什么时候值得?KV 传输成本怎么算?")。
