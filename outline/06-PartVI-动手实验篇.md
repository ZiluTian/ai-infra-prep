# Part VI 动手实验篇:把"优化闭环"练成肌肉记忆

> **目标**:每个实验都是一个完整项目——交付物为 调优记录表 + 瓶颈分析 + 优化报告,可直接作为面试项目素材(呼应 Ch30)。
> **难度**:★★★~★★★★★ | **合计** ~90 页 | 全部需要 GPU(环境见 Ch24)。

---

## 第 24 章 实验环境与工具链

> 难度 ★★★ | ~20 页 | 前置:无

### 大纲
1. **单卡环境(A100 40G / 4090)**
   - 容器、vLLM / SGLang 安装(二进制 vs 源码编译)、驱动/CUDA 版本矩阵
   - 4090 的坑:无 NVLink、消费卡功耗墙、NVMe 与显存上限
2. **源码环境**
   - vLLM v1 开发环境、debug 日志级别、关键断点位置(与 Ch4 代码地图对应)
3. **工具链**
   - nsys / ncu、nccl-tests、GenAI-Perf、DCGM exporter
4. **多机环境(可选,实验二/三需要)**
   - 两节点、RDMA 配置验证:`ibstat` / `rping` / GDR check
   - NCCL 基线:节点内 / 跨机 busbw 各跑一次存档
5. **实验仓库结构**
   - `labs/`:`scripts/`、`results/`(每次实验的固定目录 + 记录表模板)、`report/`

### 分层 / 动手
- 全章即动手:环境搭建 checklist,每步附验证命令与期望输出。
- 面试钩子:无(工具章),但环境熟练度是 Ch25-27 的前提。

---

## 第 25 章 实验一:70B 稠密模型部署调优

> 难度 ★★★★ | ~25 页 | 前置:Ch5、Ch6、Ch24
> 硬件:8×H100(最低 4×A100-80G,结果外推时标注)

### 大纲
1. **基线**:vLLM TP8 部署,默认参数,按 Ch2 出压测报告
2. **参数调优矩阵**:`max_num_seqs × gpu_memory_utilization × block_size × max_num_batched_tokens`(单变量控制,记录表)
3. **Prefix caching**:构造 50% 复用负载,测 TTFT 降幅
4. **Chunked prefill**:长短混合负载,TPOT p99 对比
5. **Speculative decoding**(EAGLE / n-gram):不同 batch 的加速曲线
6. **瓶颈分析**:nsys timeline 拆解 prefill / decode / 通信占比
7. **交付物**:"70B 生产级调优报告"(模板,可直接进简历)

> 每步统一格式:命令 → 期望结果 → 常见失败模式 → 思考题。

### 面试钩子
- 支撑 Ch28-B(框架题)与 Ch29 案例 1/4;Ch30 的"项目讲法"以此实验为示范。

---

## 第 26 章 实验二:MoE 多机部署

> 难度 ★★★★★ | ~25 页 | 前置:Ch13、Ch16、Ch24
> 硬件:1 节点 8 卡必须;2 节点可选(EP 跨机)

### 大纲
1. **部署**:Qwen3-30B-A3B(或 DeepSeek 轻量版),单机 TP8 vs EP8 对比
2. **插桩观测**:每 expert token 分布、all-to-all 耗时占比、grouped GEMM 效率
3. **跨机 EP(DeepEP)**:normal vs low-latency 模式
4. **MoE 上的投机解码**:EAGLE 效果(α 通常更低,验证 Ch7 预测)
5. **负载均衡**:构造偏斜负载,观察热 expert,尝试 expert 复制
6. **交付物**:"MoE 多机调优报告"

### 面试钩子
- 支撑 Ch28-C17~C24、D15~D20;Ch29 案例 8(热 expert 治理)的素材来源。

---

## 第 27 章 实验三:长上下文与 P/D 分离

> 难度 ★★★★★ | ~20 页 | 前置:Ch9、Ch17、Ch24
> 硬件:8×H100;P/D 需 2 节点或单机模拟

### 大纲
1. **128K 上下文**:KV 显存验证(Ch3 公式 vs 实测)、prefill 耗时、`max_num_batched_tokens` 的影响
2. **CP / 长上下文 attention**(两节点时):ring attention 实验
3. **P/D 分离**:vLLM disagg 或 Mooncake;测 TTFT / TPOT、网络带宽利用率
4. **综合**:定义 SLO(p95 TTFT < 2s、TPOT < 50ms),测 goodput
5. **交付物**:"生产级优化报告"(全书综合,面试项目模板)

### 面试钩子
- 支撑 Ch29 案例 3/5/7;Ch30 中"如何把一个实验讲成 45 分钟项目 deep-dive"。
