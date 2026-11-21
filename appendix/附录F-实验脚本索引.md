# 附录 F:实验脚本索引

> `labs/` 仓库现有 9 个可直接运行的脚本(简化实现,对应正文与附录中的公式或实验)。
> 正文中的"实验"(Ch16-18, Ch25-27)依赖真实 GPU 环境,其**工具与步骤**在对应章节;本附录负责"哪个脚本支撑哪个公式/实验"的映射,以及实验记录的统一模板。

## F.1 labs/ 脚本(9 个)

| 脚本 | 支撑内容 | 用法 | 输出 | 关联章节 |
|---|---|---|---|---|
| `bench_basics.py` | Ch1"两种形态" + Ch2 基础指标 | `python bench_basics.py --base-url http://localhost:8000/v1 --model qwen2.5-7b --repeats 3`(需本地 OpenAI 兼容服务) | ① decode 恒速性:固定 512 prompt,输出 32/128/512 → TTFT/TPOT/ITL p99;② prefill 增长:输出 32,prompt 256/2048/8192 → TTFT 近似线性 | Ch1/Ch2(实验 2.1 的最小版;正式压测用 GenAI-Perf) |
| `bench_load.py` | Ch25 实测法负载生成(三种画像) | `python bench_load.py --profile mixed --model qwen2.5-7b --duration 60`(需本地 OpenAI 兼容服务) | ① mixed:80% 512in/256out + 20% 8K/512;② prefix50:50% 请求共享 4K 前缀(前缀缓存命中率画像);③ skew:重复 prompt 诱导热 expert(Ch16 §16.3);per-request TTFT/TPOT + p50/p95/p99 汇总 | Ch25 实测法(goodput 曲线基线),Ch27 §27.5,Ch29 案例负载 |
| `capacity_planner.py` | 容量链 (3-1)~(3-5) | `python capacity_planner.py --preset llama3-70b --ctx 32768 --lam 20 --lout 512 --goodput 2600` | 权重/卡 → KV 池/卡 → 并发上限 → 需求吞吐 → TP 组数 × 余量 → 卡数;装不下时直接报"增大 TP 或量化" | Ch3(公式链的可执行版),Ch29 案例估算,Ch30 白板题 ③ |
| `roofline.py` | Roofline 模型(§10.2) | `python roofline.py --peak 989 --bw 3.35 --ai 1 --name "decode GEMV (batch=1)"` | roofline 图 + kernel AI 标注 + ridge 点 + memory/compute-bound 判定 + 可达 TFLOPS(峰值百分比) | Ch1/Ch10(bound 判定),Ch12(kernel 评估第一问) |
| `comm_planner.py` | TP/PP/EP 每步通信量 | `python comm_planner.py --hidden 8192 --layers 80 --tp 8 --bs 1`;`--ep 8 --topk 8 --batch 64`(MoE) | 每步通信总量/单 op 量 + NVLink 900GB/s 与 IB 50GB/s 两档延迟 + 延迟/带宽主导判读口诀 | Ch13 (13-1)/Ch14 (14-1)(14-2),Ch16 EP 算账,Ch26 实验复用,Ch29 案例 1/5 |
| `request_tracer.py` | Ch20 L2 分段:per-request 时间戳 + TTFT 三段分解 | `python request_tracer.py --base-url http://localhost:8000/v1 --model qwen2.5-7b --n 32 --concurrency 8 [--metrics http://localhost:8000/metrics]` | ① 每请求 t0/首 token/末 token + TTFT/TPOT/E2E;② p50/p95/p99 分位;③ 接 `--metrics`(vLLM Prometheus)时自动分解服务端排队/prefill/decode 段 → TTFT ≈ 排队 + prefill + 网络 | Ch20 §20.1(L2 插桩),Ch2 §2.3(分位口径) |
| `nccl_baseline.sh` | Ch18 §18.4 busbw 验收(×0.875,N=8) | `./nccl_baseline.sh --gpus 8`(节点内);`--hostfile nodes.txt`(跨机);`--link-bw/--expected` 覆盖期望值 | all_reduce/alltoallv 的 busbw 曲线(8M→8G)+ 期望对照(节点内 ~780GB/s、跨机 rail ~350GB/s)+ "低于期望 20% → 拓扑/配置问题"判读 | Ch18 §18.4,附录 C(NCCL 变量) |
| `agent_load.py` | 附录 H 公式 (H-1)/(H-2) 的可执行版 | `python agent_load.py --dry-run --sessions 3 --turns 6`(不发请求,验证负载构造与账目);实服务:`python agent_load.py --model qwen2.5-7b --sessions 4 --turns 8`(需本地 OpenAI 兼容服务) | 多轮会话(共享 system 前缀 + 逐轮 delta + 工具结果),会话内串行 / 跨会话并行;逐轮 TTFT + 冷/热 TTFT 比(前缀缓存生效代理)+ naive/增量 prefill 账(η,复现 H-1) | 附录 H §H.1.2/§H.2.1,Ch5 §5.4(50% 复用对照),Ch2 分位口径 |
| [`grpo_mini.py`](../labs/grpo_mini.py) | GRPO 组内优势和裁剪目标 (H-6)/(H-7) | 从仓库根目录运行 `python labs/grpo_mini.py` 或加 `--self-test`（仅需 CPU 和 Python 标准库） | 混合奖励组、首次更新梯度范数、裁剪项占比、训练题期望奖励；支持单 rollout 和单次更新对照 | 附录 H §H.4.7/§H.4.9 |

**使用纪律**(Ch2 的测量口径同样适用):
- 区分脚本估算、本地功能核验与 Ch24–27 的性能[实测]；`grpo_mini.py` 的结果只描述玩具策略，不是大模型训练或 serving 性能。
- `capacity_planner.py` 的 `--goodput` 必须填**实测**值(Ch27 §27.5 的 goodput 曲线 knee 点),脚本不替你造数字。
- `bench_basics.py` 的 prompt 是重复词填充(近似 token 数)——正式实验换固定语料(Ch2 §2.3 的语料纪律)。
- `hook_router.py`(Ch16 §16.3 偏斜注入)按设计**由读者自建**(Ch24 §24.5 / Ch26 §26.2 的伪代码骨架),不随 `labs/` 分发——hook 点依赖目标框架版本。

## F.2 章节实验 → 工具映射

| 实验 | 章节 | 核心工具 | 关键产出 |
|---|---|---|---|
| 环境搭建与基线 | Ch24 | docker + `nvidia-smi` + `nccl-tests`(附录 C 的调试组)+ 记录表(§24.4) | 硬件/驱动/CUDA/框架版本快照(附录 G 的登记源) |
| NCCL 基线 + 带载基线 | Ch18 §18.4 | `nccl-tests`(all_reduce_perf / alltoallv_perf)+ 附录 C 变量组 | busbw 基线表 + 故障注入(NVLink 降级/断 NIC)对照 |
| 实验 16.1:MoE 偏斜注入 | Ch16 | SGLang/vLLM + 测试 hook(强制 top-k 集中)+ DCGM per-rank 步时 | p99 vs 放置策略曲线(热 expert 静态放置的实证) |
| 实验一:70B 稠密调优 | Ch25 | vLLM + GenAI-Perf + nsys/ncu | token 预算/chunk/前缀/投机的 A/B 迭代表(§25.4 方法) |
| 实验二:MoE 多机 | Ch26 | SGLang + DeepEP + DCGM | EP 放置 + 热 expert A/B(§26.5)+ 投机解码在 MoE 上的曲线 |
| 实验三:长上下文 + P/D | Ch27 | vLLM P/D 模式 + KV 传输(NIXL 类)+ GenAI-Perf | P/D A/B + KV 传输时序(对照 Ch9 §9.2 的表) |
| 故障注入五类 | Ch21 §21.4 / Ch29 案例 9 | kill 进程 / kill 节点 / 断 NIC / ECC 注入 / 3× 过载 | 每类的"正确行为"验收(超时而非 hang、shed 生效、告警 <5min) |

## F.3 实验记录表模板(Markdown)

> 统一骨架,源自 Ch24 §24.4。**每次实验一张表;没有记录的实验 = 没有发生。**

```markdown
# 实验记录:<一句话标题>
日期:        YYYY-MM-DD
硬件:        8×H100 SXM 80GB(NVLink 900GB/s 实测) | 驱动 5xx | CUDA 12.x | IB 8×400Gb
框架/版本:   vLLM v1.x(commit/版本) | NCCL 2.2x | 模型 Qwen2.5-70B(FP8)
负载:        λ=xx req/s | 输入 p50/p99 = xx/xx | 输出均值 xx | 前缀命中 xx%
参数 diff:   相对基线改了什么(每个参数一行,含取值理由)
---
| 指标 | 基线 | 本次 | Δ |
|---|---|---|---|
| TTFT p50/p95/p99 | | | |
| TPOT p50/p95/p99 | | | |
| goodput(tok/s,SLO 内) | | | |
| 抢占计数 | | | |
| SM Active / 显存峰值 | | | |
---
结论:      (一句话:哪个假设被证实/证伪)
下一步:    (参数继续扫 / 换架构 / 进 A/B 上线)
```

**字段纪律**:
- "硬件"行必须带**实测**NVLink 带宽(`nvidia-smi topo -m`),不认型号铭牌(附录 B 的 H800 教训)。
- "参数 diff"必须与基线**一一对应**,多个变量同时改 = 记录作废(Ch2 的单变量纪律)。
- "结论"写**假设的证实/证伪**,不写"效果很好"。

## F.4 缺什么、补什么(诚实清单)

outline 规划的 6 类脚本均已独立成 `labs/` 脚本；后续补充了 `bench_load.py`、`agent_load.py` 与 `grpo_mini.py`，**共 9 个**：
- **插桩脚本**(`request_tracer.py`):per-request t0/首 token/末 token 时间戳 + p50/p95/p99 分位;加 `--metrics` 接服务端 Prometheus 端点后,可分解服务端排队/prefill/decode 段——对应 Ch20 L2 的"TTFT = 排队 + prefill + 网络"三段。
- **nccl 基线脚本**(`nccl_baseline.sh`):nccl-tests 一键构建 + all_reduce/alltoallv 节点内/跨机运行,输出 busbw 曲线与期望值判读(×0.875,N=8;验收标准 = Ch18 §18.4 的表)。
- **agent 负载脚本**(`agent_load.py`):多轮会话 + 工具循环的 OpenAI 兼容负载生成;会话内轮次串行、跨会话并行;逐轮 TTFT + 冷/热比 + prefill 账(η)。**诚实登记:2026-11-19 新增,`--dry-run` 已在开发机验证(负载构造与账目,η 与附录 H §H.1.2 手算一致),实服务路径(vLLM/SGLang 端到端)尚未核验**——核验前其输出口径与其余 7 个已核验脚本不等价(附录 G §G.5;核验后按 ③f 口径补登记)。
- **GRPO Mini Lab** (`grpo_mini.py`)：Windows/Python 3.13 的 CPU 自测和对照已通过（附录 G §G.5）；未接入真实 LLM 或 rollout 服务。
> 补脚本的原则(不变):能复现书中**至少一个带数字的结论**才值得写;否则章节内命令块足够。
