# 第 21 章 OOM 与稳定性

> 难度 ★★★★ | 目标 ~25 页 | 前置:Ch5
> 分层:21.1-21.2 [基础];21.3-21.4 [进阶]。
>
> **读完本章你能做什么**
> 1. 对任意 OOM 先分类(权重 / KV / 激活 / 碎片),并说出每类的判别证据
> 2. 讲清 KV OOM 的完整处理链:预留余量 → 拒绝 → 抢占(recompute vs swap)→ 调参
> 3. 用 memory snapshot + tracemalloc + nsys 排查"慢性 OOM"(泄漏)
> 4. 设计崩溃后的语义:in-flight 请求怎么办、KV 丢了怎么办、graceful drain 怎么做

## 21.1 OOM 分类(先分类,再排查)

| 类型 | 时机 | 症状 | 判别证据 |
|---|---|---|---|
| **权重 OOM** | 启动 | 进程直接起不来 | 启动日志:weight load 阶段 OOM;`nvidia-smi` 显示权重字节 > 显存 |
| **KV OOM** | 运行时 | 请求被拒绝/抢占;服务"活着但收不了新请求" | vLLM `gpu_cache_usage` ≈ 1.0;KV block 分配失败日志 |
| **激活 OOM** | 运行时 | 某条长 prefill 请求触发崩溃(整进程) | 崩溃发生在 prefill forward;激活大小 ∝ chunk×hidden(Ch17) |
| **碎片 OOM** | 运行时(慢性) | `nvidia-smi` 显示显存被占满,但 `gpu_cache_usage` 只有 0.6-0.8 | 两者**不一致**是碎片/泄漏的指纹 |

**判别第一步(零成本)**:`nvidia-smi` 的显存占用 vs 框架自报的 KV 使用率——

- 一致且都高 → 真 KV 压力(§21.2)
- nvidia-smi 高、KV 使用率低 → 碎片或泄漏(§21.3)
- 启动就 OOM → 权重(Ch3 容量公式重算:该模型该精度该 TP 根本装不下)

**激活 OOM 的特殊性**:它是**进程级崩溃**(不是拒绝请求)——PyTorch 激活是动态申请的,超了直接 throw。防御:chunked prefill 的 chunk 上限(把长 prefill 的激活峰值切小)+ `max_num_batched_tokens` 钳制 + 对超长请求单独路由(Ch19/Ch17 的 prefill 集群)。

## 21.2 KV OOM 处理链

**预留余量**:`gpu-memory-utilization=0.9` 的含义——框架启动时**一次性**吃掉 90% 显存:权重 + KV pool(全部剩余) + 激活余量 + NCCL buffer。**OOM 不会发生在"KV pool 之外"——框架在启动时就划好了池子**;运行时 OOM 只发生在池子内的分配(KV block 耗尽)或池子外的意外(泄漏/碎片)。

**处理链(按发生顺序)**:

```
KV pool 接近耗尽
 ① 新请求: 按调度器策略排队(不立即拒绝,看 max_num_seqs / 队列上限)
 ② 队列超上限: 拒绝(返回 429/503)——过载 shed 的第一道闸(Ch22)
 ③ 在跑请求不够 block: preemption
 ④ 池子整体不够: 扩容(加副本,Ch19)
```

**preemption 两种方式(面试必问:怎么选)**:

| | recompute(重算) | swap(卸到 CPU) |
|---|---|---|
| 机制 | 丢弃该请求的 KV,重新 prefill | KV block 拷贝到 CPU 内存,恢复时拷回 |
| 代价 | 重新算 prefill(FLOPs + 占 prefill 算力) | 拷贝时间 PCIe ~25-60GB/s(1GB KV ≈ 20-40ms)+ CPU 内存占用 |
| 适用 | KV 小 / 前缀可复用(prefix cache 命中则重算便宜) | KV 大 / CPU 内存充足 |
| vLLM 默认 | **recompute**(简单、无 CPU 内存依赖) | 可配 |

**经验法则:优先 recompute**——除非该请求 prefill 极长且无 prefix cache(重算代价 ≫ swap 拷贝代价),或 CPU 内存富余且 PCIe 快。**量化:1GB KV,swap 往返 ~80ms(PCIe4),重算 32K 上下文 prefill ~1-2s(7B 单卡量级;70B TP8 ~4-8s)——短上下文 recompute 反而便宜**。

**调参三旋钮**(按影响排序):

1. `max_num_batched_tokens`(每步 token 预算):压低 → 步时稳定、并发下降;提高 → 吞吐上升、长 prefill 步变尖(Ch6)
2. `max_num_seqs`(并发上限):直接决定 KV 需求峰值
3. `gpu-memory-utilization`(池子大小):提高 → 池子大但激活/NCCL 余量变小,**OOM 风险从"拒绝请求"变成"进程崩溃"**——一般 0.85-0.90,别贪

## 21.3 [进阶] 泄漏与"慢性 OOM"

**症状指纹**:服务运行数小时后,`nvidia-smi` 占用持续爬升,`gpu_cache_usage` 不涨,最终 OOM 崩溃。

**三类常见来源**:

1. **Python 引用环 / 全局缓存累积**:请求对象、tokenizer 结果、统计结构被意外持有(典型:per-request 的 stats dict 没清理)
2. **CUDA caching allocator 行为**:PyTorch 的 caching allocator 释放的是"缓存块"不是还 HBM——`nvidia-smi` 高 ≠ 泄漏,**先看 `torch.cuda.memory_allocated` 与 `memory_reserved` 的差**(reserved 高 allocated 低 = 缓存正常;两者同步爬 = 真泄漏)
3. **NCCL / 通信 buffer**:多进程组反复 create/destroy(滚动升级未正确释放 communicator)

**排查工具链**:

```
① torch.cuda.memory_summary() / memory snapshot(导出 JSON,Chrome 打开看分配树)
② tracemalloc(定位 Python 侧对象增长,配合 gc.get_objects)
③ nsys --memory(时间线上的 cudaMalloc/cudaFree 配对)
④ 二分:开 1h / 4h / 8h 对照,看爬升曲线斜率何时出现
```

**`expandable segments`**(PyTorch 2.x):caching allocator 用可扩展段减少碎片——碎片 OOM(§21.1 第四类)的缓解项之一;但**它治不了泄漏**,只治碎片。

## 21.4 [进阶] 稳定性工程

**进程 watchdog**:推理进程挂掉必须**秒级感知 + 自动重启**——

- 感知:heartbeat(网关侧 5s 超时)+ NCCL watchdog(集合通信 hang 检测,Ch18)+ 健康检查 endpoint
- 重启:K8s 自动拉新 pod;gang 组整体重启(Ch14 §14.4)
- **重启的语义问题(面试区分点)**:in-flight 请求的 KV 全部丢失 →
  - 客户端视角:流式中断,应**幂等重试**(带 request id 去重)
  - 网关视角:重试时**前缀命中 prefix cache**(若网关把 prompt 留在路由层,或副本共享缓存)可大幅降低重试代价
  - 设计目标:**崩溃恢复 MTTR < 客户端重试超时的 1/3**(Ch19 §19.2 的 10-20min 节点级 vs 秒级进程级要分清)

**故障注入(上线前必做,Ch29 案例 9)**:

| 注入 | 验证 |
|---|---|
| kill 单进程 | watchdog 感知 < 1min,自动重启,in-flight 语义正确 |
| kill 单节点 | gang 组降级,容量 −1 副本,SLO 是否仍达标(Ch19 余量设计) |
| 断 NIC / 降链路 | NCCL 超时行为、请求超时而非 hang(Ch18 §18.4) |
| ECC 错误 / XID 报错 | DCGM 告警链路,卡隔离(不自动修,只隔离 + 换) |
| 过载(3× 流量) | shed 生效(拒绝 < 崩溃),队列有界,恢复后不震荡 |

**graceful drain 与滚动升级**:

- 升级前:副本停止接新请求 → 等在跑请求完成(或超时强制迁移)→ 摘流量 → 升级
- **drain 的陷阱**:长输出请求(10 分钟生成)会拖死 drain → 设 drain 超时,超时请求走"中断 + 重试"(客户端语义)或"迁移到旧版本副本"(若架构允许)
- 滚动顺序:按 gang 组为单位滚动(不能滚半组);**至少留 N-1 组容量**(Ch19 §19.2)

> **踩坑框:"OOM 了,加显存/降利用率就好"**
> 先分类(§21.1):权重 OOM 加显存是正解,但 **KV OOM 加显存是掩盖容量问题**(应该调调度/扩容/shed),**碎片 OOM 加显存是慢性毒药**(泄漏还在,只是爆得更晚)。"加显存"是四类里只对一个有效的药。

## 动手

**实验 21.1(半天,7B 单卡)**:故意制造 KV OOM:

1. `gpu-memory-utilization=0.6`(人为缩小 KV pool)+ 长输出负载(输出 4K token × 并发 32)
2. 观察:请求拒绝?抢占?`num_preempted` 计数曲线?恢复曲线(抢占后请求的 TTFT/总时长)
3. 对比 recompute vs swap(vLLM 配置开关),记录两者对被抢占请求的延迟代价
4. 交付物:抢占行为报告(触发条件、计数、延迟代价),支撑 Ch22 的长尾分析

## 章末自测

1. 四类 OOM 各给一条"最便宜判别证据"
2. `nvidia-smi` 显存高但 `gpu_cache_usage` 低——可能是哪两类问题?下一步各用什么工具?
3. recompute vs swap 的代价公式(用"1GB KV、32K 上下文"代入比较)
4. `gpu-memory-utilization` 从 0.85 提到 0.95 的风险变化(OOM 形态怎么变)?
5. 崩溃重启后,in-flight 请求的语义谁来兜底(网关/客户端/引擎各管什么)?
6. drain 为什么会被长输出请求拖死?两种解法?

## 面试钩子

- 支撑 Ch28-E16~E22:"线上 OOM 怎么定位?""recompute 还是 swap?""服务挂了 in-flight 请求怎么办?"
- Ch29 案例 9(推理集群容错)的故障注入表即 §21.4 的展开。
