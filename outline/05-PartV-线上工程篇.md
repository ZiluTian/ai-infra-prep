# Part V 线上工程篇:从"会优化"到"生产级、稳得住"

> **目标**:本部分是中级与高级的分水岭。读者应能独立做系统化瓶颈定位、设计 OOM 与长尾治理方案、搭一套可灰度可回滚的生产部署。
> **难度**:★★★~★★★★★ | **合计** ~110 页

---

## 第 20 章 系统化性能分析

> 难度 ★★★★ | ~30 页 | 前置:Ch2、Ch10

### 大纲
1. **三层分析框架(本章方法论核心)**
   - L1 宏观:TTFT / TPOT / 吞吐 / GPU util(时序 + p99)
   - L2 组件:排队 / prefill / decode / 通信 / 采样 分段
   - L3 kernel:nsys timeline + ncu(SOL、occupancy)
   - 方法:先定位"哪个组件异常",再下钻,禁止上来就抓 kernel
2. **症状 → 原因库(案例驱动)**
   - TTFT 高:排队?长 prefill?没开 chunked prefill?前缀冷?
   - TPOT 高:batch 太小?TP 过大通信慢?launch 开销(graph 没开)?降频?
   - 吞吐低但延迟正常:碎片?抢占风暴?
   - GPU util 低:CPU 瓶颈(tokenize / 调度)、通信等待、graph 碎片
3. **工具链实操**
   - nsys:gap 分析(CPU gap vs kernel gap)、CUDA graph 生效验证
   - ncu:GEMM SOL(memory % vs compute %)
   - vLLM 内置指标:`num_requests_waiting`、`gpu_cache_usage`、prefix cache 命中率
   - DCGM:clock / power / throttle 原因(HW Slowdown / SW Power Cap)
4. **三个完整案例(每个:症状 → 假设 → 证据 → 修复 → 回归)**
   - A:GPU util 35% 但 TPOT 达标 → batch 太小 → `max_num_seqs` + 调度调整
   - B:TP8 p99 尖刺 → all-reduce 通道竞争 → NCCL 调优 + 拓扑核对
   - C:prefill 阻塞 decode → chunked prefill + P/D 决策

### 分层
- [基础] 1-2;[进阶] 3-4。

### 动手 / 面试钩子
- 动手:复现案例 A(7B,故意设小 `max_num_seqs`),完整走一遍 L1→L3 定位。
- 面试钩子:支撑 Ch28-E07~E15("GPU util 低怎么查?给我一个你的定位流程")。

---

## 第 21 章 OOM 与稳定性

> 难度 ★★★★ | ~25 页 | 前置:Ch5

### 大纲
1. **OOM 分类(先分类再排查)**
   - 权重 OOM(启动)/ KV OOM(运行时拒绝请求)/ 激活 OOM(长 prefill)/ 碎片 OOM
   - 判别:`nvidia-smi` vs vLLM `gpu_cache_usage`、`torch.cuda.memory_summary`
2. **KV OOM 处理**
   - `gpu-memory-utilization` 与框架预留下的余量
   - preemption:recompute vs swap(KV 卸 CPU,延迟代价)
   - `max_num_batched_tokens` / `max_num_seqs` 调参
3. **泄漏与"慢性 OOM"**
   - Python 引用环、CUDA caching allocator(expandable segments)、NCCL buffer
   - memory snapshot / tracemalloc + nsys memory 联合排查
4. **稳定性工程**
   - 进程 watchdog、crash → 重启 → in-flight 请求语义(KV 丢失后怎么办)
   - 故障注入:kill 节点 / 断 NIC / ECC 错误
   - graceful drain、滚动升级

### 分层
- [基础] 1-2;[进阶] 3-4。

### 动手 / 面试钩子
- 动手:故意制造 KV OOM(小显存上限 + 长负载),观察 preemption 行为与恢复曲线。
- 面试钩子:支撑 Ch28-E16~E22("线上 OOM 怎么定位?recompute 还是 swap?")。

---

## 第 22 章 长尾延迟治理

> 难度 ★★★★★ | ~25 页 | 前置:Ch20
> 注:整章 [进阶],路线 A 读者读 1-2。

### 大纲
1. **长尾来源清单**
   - 抢占重算、长 prefill 阻塞、通信 straggler、clock/power throttle、GC 停顿、MoE 热 expert、prefix cache 逐出
2. **p99 定位方法**
   - 每请求 trace(vLLM log_stats),异常请求与系统事件时间对齐
   - 时序:p99/p50 比值、"突刺型" vs "慢性长尾"
3. **缓解 playbook**
   - 架构级:chunked prefill + P/D
   - 调度级:deadline-aware(EDF / LLF)
   - 过载 shed:队列深度 + 预测耗时 → 拒绝 / 降级
   - 副本隔离(热请求 / 噪音前缀)、MoE:热 expert 复制 + 路由
4. **SLO 工程**
   - SLO 定义(如 p95 TTFT < 2s)、SLO burn rate、告警
   - SLO-aware autoscaling(在队列爆之前扩容)

### 动手 / 面试钩子
- 动手:注入 straggler(限一张卡 clock / 加通信延迟),测 p99 恶化与缓解效果。
- 面试钩子:支撑 Ch28-E23~E28("p99 尖刺怎么排查?你们 SLO 怎么定的?")。

---

## 第 23 章 生产部署与运维

> 难度 ★★★★ | ~30 页 | 前置:Ch20

### 大纲
1. **serving 栈**
   - LLM 网关:路由 / 限流 / 灰度 / shadow traffic
   - 负载均衡策略:least-queued vs 前缀感知(KV 复用)
   - 副本管理:版本、drain、健康检查
2. **监控与告警**
   - 指标清单:TTFT/TPOT p50/p95/p99、队列深度、KV 使用率、拒绝率、GPU util/clock、NCCL 错误
   - 看板与告警阈值(SLO 导向而非固定阈值)
3. **容量规划(Ch3 的生产版)**
   - 流量预测 → 卡数、余量、季节性与突发
   - 成本:/百万 token、按租户归因
4. **发布与回归**
   - 性能回归 CI:nightly bench(固定负载 + 固定硬件)
   - 灰度:5% 流量 + SLO 门禁、自动回滚
   - kernel / 框架升级:A/B + shadow

### 分层
- [基础] 1-2;[进阶] 3-4。

### 动手 / 面试钩子
- 动手:搭监控看板(Prometheus + Grafana,接 vLLM metrics,Ch19 看板复用)。
- 面试钩子:支撑 Ch28-E29~E33("灰度怎么判健康?容量规划怎么做?")。
