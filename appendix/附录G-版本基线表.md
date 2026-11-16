# 附录 G:版本基线表

> 全书的"源码走读"与"实测数据"以本表登记的环境为准。
> **为什么需要这张表**:推理引擎行为随版本剧烈漂移(vLLM v0→v1 是架构级重写;SGLang 的调度行为 2024→2025 多次变化)。没有版本基线的"我读过的源码"在 6 个月后就是过期信息。
> **维护规则**:① 任何"源码走读"必须登记 文件路径 + 方法名 + 版本/commit;② 换版本重跑实验,数据必须重新生成;③ 定稿前逐章复核(§G.4 清单)。

## G.1 写作基线(默认环境)

| 项 | 基线值 | 备注 |
|---|---|---|
| vLLM | **V1 引擎,发布基线 v0.32.0**(2026-10-18) | 公开发布线是 **0.x**;"V1" = 引擎架构(EngineCore 单进程重构 + CUDA graph 分桶 + token 预算调度),**不是版本号**。正文的"vLLM v1.x"一律指 V1 架构构建(消歧说明,各章引用不改)。Ch4 的 v0 vs v1 对照用 v0.6.x(v0 引擎)+ v0.32.0(V1 引擎) |
| SGLang | **v0.5.x,基线 v0.5.22**(2026-10-19;v0.5.23 已发) | RadixCache + overlap scheduling + 结构化输出成熟版;Ch4/Ch6/Ch11 的 SGLang 行为以此为准 |
| TRT-LLM | **v1.x,稳定基线 v1.2.1**(2026-04-16;开发线 v1.4.0rc3,2026-11-06) | v1.0.0(2025-09-24)起 **PyTorch 后端为默认**;TRT engine 后端已移除(最后支持 v1.3.0rc20,2026-06-30);原"v0.18+"基线作废。Ch4"何时选 TRT-LLM"(B27)参数口径 = v1.x PyTorch 后端 |
| CUDA / 驱动 | CUDA 12.x / 驱动 550+ | H100 SXM + NVLink 4 代;FP8 需要 12.0+ |
| NCCL | 2.2x(随 CUDA 发行) | 附录 C 的变量默认值以此核对 |
| OS | Ubuntu 22.04 / 24.04(LTS) | 内核参数(IOMMU/NUMA)章节内核对 |
| 硬件批次 | 实测环境:8×H100 SXM 80GB 节点(NVLink 900GB/s 实测)+ 2 节点 EP 测试床(Ch26);对照机 H800(附录 B 脚注) | 所有 Ch25-27 实测数据的硬件口径 |
| 模型 | Qwen2.5-7B(快速实验)/ Qwen2.5-70B(FP16 + FP8,主实验);671B 级为[官方量级]引用,非实测 | Ch24-27 的模型口径 |
| 压测工具 | GenAI-Perf(ai-dynamo/benchmaker)+ labs/bench_basics.py(基础版) | Ch24 环境搭建 |

## G.2 逐章登记(源码走读 + 实测)

| 章节 | 走读对象(版本) | 实测数据 | 备注 |
|---|---|---|---|
| Ch1-3 | 无(概念/公式) | 无 | 公式不依赖版本 |
| Ch4 | vLLM v0.6.x(v0 引擎)+ v0.32.0(V1 引擎)对照;SGLang v0.5.x | 无(架构行为,无数字) | 版本行为差异的最大来源,复核优先级最高 |
| Ch5-6 | vLLM v1.x(KV cache manager、scheduler) | 调度行为定性 | 文件+方法级走读 |
| Ch7 | vLLM v1.x 投机解码路径 | batch 1/8/32 加速比(7B) | 实验 7.1 |
| Ch8 | 量化工具链(GPTQ/AWQ 脚本版本) | FP8/INT4 质量门禁(任务集 Δ) | 工具版本影响量化结果,需登记 |
| Ch9-10 | 无(架构/硬件口径) | 硬件规格(附录 B) | B200 行 2026-11-16 已复核 |
| Ch11-12 | FlashInfer(版本) + Triton(版本)+ CUTLASS(版本) | kernel 性能(ncu SOL) | 三个 kernel 库版本都要登记 |
| Ch13-14 | DeepEP(版本,若用) | 通信量(估算,comm_planner.py) | |
| Ch15-18 | SGLang v0.5.x + NCCL 2.2x | nccl-tests busbw 基线(Ch18 §18.4) | 网络排障章节,版本影响最大 |
| Ch19-21 | vLLM v1.x(容量/运维行为) | 容量规划为估算;故障注入为定性 | |
| Ch22-23 | vLLM v1.x 监控面 | 监控指纹定性 | |
| Ch24-27 | 全栈快照 = §24.4 记录表(每实验一张) | **全部实测数据**(70B FP8/FP16,8×H100) | 实测的唯一出处 |
| Ch28-30 | 无新数据 | 引用 Ch25-27 实测 + Ch17/19 量级 | |

## G.3 数据归属声明

- **实测数据**:仅 Ch24-27 的环境(§G.1 硬件批次)产生;书中其他章节的"量级/估算"数字标注为[估算]或[官方],不得当作实测引用。
- **官方量级**:DeepSeek V3/R1(MLA latent、激活参数)、硬件规格(附录 B)——来源为官方报告/datasheet,引用处已标注。
- **假设**:Ch29 案例中的[假设]数字是教学设计,不是测量结果(案例开头已声明)。

## G.4 定稿前复核清单(逐章)

> 写作时登记的是"写作时基线"。**定稿前必须逐章复核以下漂移点**,并在本表脚登记复核日期:

| 复核项 | 为什么会漂移 | 影响章节 |
|---|---|---|
| vLLM v1.x 小版本行为(token 预算默认值、前缀缓存开关、P/D 模式参数名) | v1 迭代快,参数名/默认值常变 | Ch4-7, Ch25-27 |
| SGLang 调度行为(RadixCache 逐出策略、overlap 的 CPU/GPU 时序) | 2025 年内多次重构 | Ch4/Ch6/Ch11 |
| TRT-LLM in-flight batching 参数与 plugin 接口 | 小版本 API 变化 | Ch4(B27) |
| NCCL 变量默认值(附录 C 30 项) | 随 NCCL 版本 | Ch18, 附录 C |
| FlashInfer/Triton/CUTLASS 版本与 kernel 行为 | kernel 库迭代最快 | Ch11-12 |
| B200 规格行(附录 B) | 数据发布时点新 | 附录 B, Ch10/Ch28 C26 |
| 框架版本 pin(vLLM/SGLang/TRT-LLM 基线) | 发布节奏快,参数名/默认值常变 | Ch4-27 |
| GenAI-Perf 参数与输出格式 | 工具迭代 | Ch24-27 |
| Qwen2.5 系模型行为(量化后质量、前缀复用特征) | 模型更新 | Ch25 实验一 |

**复核状态(2026-11-16,验证机 linuxgpu01:4×RTX 5090 32GB / CUDA 13.0 / Py3.12——非 Ch24-27 的[实测]环境,以下均为验证日志,细节见 §G.5 ③)**:
✅ vLLM v0.32.0 实装核验(默认 token 预算 2048 / chunked+prefix 默认开 / kv_role 新取值 / MooncakeConnector / CLI `--help=all` / block 16 / attention=FLASH_ATTN / sampling=FlashInfer JIT),书稿相应修正 7 处(ch04/ch06"默认 8192"→"默认 2048")+ ch27 kv_role 1 处。✅ SGLang v0.5.22 实装核验(disaggregation 旗标 / overlap 默认开 / radix 默认开 + lru 逐出 / evict() 与 Ch5 伪代码一致 / chunked-prefill-size 5090 自动默认 4096 / 源码路径 / `--mem-fraction-static`)——与书一致,无需改书。✅ NCCL 30 项对照(2.29.7:26 一致 / 4 修正 / 1 新增,附录 C §C.7)+ ch18 正文旧变量名修正。✅ B200 规格行(§G.5 ①)。✅ 框架版本 pin(§G.5 ②)。✅ labs 端到端(bench_basics + request_tracer 在 vLLM/SGLang 双真实服务上跑通,§G.5 ③f)。
未做(如实登记):TRT-LLM v1.2.1 运行时(依赖未齐,参数名已源码级实证,§G.5 ③d)、GenAI-Perf(验证机无 docker,§G.5 ③e)、FlashInfer/Triton/CUTLASS kernel 行为复测(版本登记:flashinfer-python 0.7.0.post1 / triton 3.7.1 / torch 2.13.0+cu130;需 H100+ncu 环境)、Qwen2.5 模型行为(验证机以 Qwen3-4B 替代,行为核验不依赖权重;Ch24-27 定量数据仍用 H100 口径)。

**复核动作**:逐项跑一遍对应章节的"最小验证实验"(Ch4:双版本行为对照;Ch18:nccl-tests 基线;Ch25:一次 goodput 曲线),把日期与结论登记在本表。
**诚实声明**:本书写作时的环境快照为 §G.1;若读者按自己的版本复现且数字对不上,**先查版本再查方法**——这是全书数据口径的第一条纪律(Ch2 §2.1)。

## G.5 复核日志

- **2026-11-16**:① **B200 规格行**:180GB HBM3e / 7.7TB/s;稠密 FP16 2.25PF / FP8 4.5PF / FP4 9PF;NVLink5 1.8TB/s(来源:NVIDIA Blackwell datasheet,HGX B200 SXM)。正文 192GB/~8TB/s 统一改为 180GB/7.7TB/s(ch03/10/19/28/29 + outline)。② **框架版本 pin**:vLLM = V1 引擎(发布线 0.x,基线 v0.32.0;"v1.x" 指 V1 架构,非版本号,见 §G.1 消歧说明);SGLang = v0.5.22;TRT-LLM = v1.2.1(PyTorch 后端默认,TRT engine 后端 v1.4.0 线已移除)。③ **逐章小版本行为复核——vLLM/SGLang/NCCL 部分完成(2026-11-16,验证机 linuxgpu01)**:验证机 = 4×RTX 5090 32GB(driver 580.178 / CUDA 13.0,Py3.12),**非** Ch24-27 的[实测]环境(8×H100),以下全部为验证日志口径。HF 缓存仅 metadata 无权重 → 以 Qwen3-4B(7.6GB,huggingface_hub xet 协议下载,逐文件大小核对)替代;行为核验(参数名/默认值/调度/指标)不依赖具体权重。服务统一用 GPU 3(GPU 0/1 = 用户 sglang TP2 任务、GPU 2 = ollama ~12GB,均勿动;双服务停后 GPU 3 恢复 131MiB 无残留;用户 GPU 0/1 任务全程未受影响)。

**a) vLLM v0.32.0(pip 实装,V1 为唯一引擎——`VLLM_USE_V1` 环境变量已不存在)**:① `max_num_batched_tokens` 默认 = **2048**(源码 `config/scheduler.py` `DEFAULT_MAX_NUM_BATCHED_TOKENS=2048`,batched-DP 场景 256;`EngineArgs` 无其他覆盖)→ 书稿旧"默认 8192"改 7 处(ch04 L96 + ch06 6 处,含实验组改 2048/8192/16384 与 `--no-enable-chunked-prefill` 对照组);② chunked prefill 默认开;③ `--enable-prefix-caching` 存在且默认开(实服务 log config dump 实证 `enable_prefix_caching=True, enable_chunked_prefill=True`);④ `kv_role` 合法值 = `kv_producer/kv_both/kv_consumer`(Literal,旧 prefill/decode 不接受)→ ch27 §27.4 已改;⑤ CLI 改版:`vllm serve --help` 只列 config groups,全量参数表(2394 行)需 `--help=all`;⑥ `--kv-transfer-config` 存在,Mooncake 连接器名 `MooncakeConnector`(30+ 连接器含 Nixl/MooncakeStore/Offloading);⑦ `--block-size` 默认 None → `DEFAULT_BLOCK_SIZE=16`(ch04/ch05/ch11"16-token block"成立);⑧ 实服务 attention backend = FLASH_ATTN(FlashAttention v2),top-p/top-k sampling 走 FlashInfer(flashinfer-cubin 缺失 → 启动时 JIT 编译,需 ninja 在 PATH;两 venv 本有 ninja 1.13.2,启动脚本加 `export PATH=$VENV:$PATH` 即解决);⑨ `max_num_batched_tokens` 不出现在启动 log 的 config dump → 以源码级证据为准;⑩ `max_num_seqs` 默认 128(书稿未声称默认值,不冲突)。

**b) SGLang v0.5.22(pip 实装;实服务 server_args 全量 dump 实证,与书一致,无需改书)**:`chunked_prefill_size=4096`(5090 自动默认,与 `memory_hook.py` 公式一致)+ `max_prefill_tokens=16384` + `max_running_requests=2048` + `max_total_num_tokens=134110` + `context_len=40960`;`radix_eviction_policy='lru'` / `disable_radix_cache=False` / `disable_overlap_schedule=False`(overlap 默认开);`attention_backend='flashinfer'` / `sampling_backend='flashinfer'` / `grammar_backend='xgrammar'`;`--disaggregation-mode {null,prefill,decode}` + `--disaggregation-transfer-backend {mooncake,nixl,ascend,fake,mori,mooncake_tcp}`(与 ch09 §9.4 表一致);`--attention-backend` 可选值 20+(含 triton 可作回退);`RadixCache.evict()` = evictable_leaves + last_access_time 堆 → 与 ch05 §5.4 伪代码一致;源码路径 `sglang/srt/managers/scheduler.py` + `sglang/srt/mem_cache/radix_cache.py` 存在;**显存旗标是 `--mem-fraction-static`(无 vLLM 式 `--gpu-memory-utilization`)**;`/metrics` 默认不启用(`enable_metrics=False`,需 `--enable-metrics`;用户生产 sglang 即用此旗标)。

**c) NCCL 2.29.7**:30 项对照 26 一致 + 4 修正 + 1 新增,见附录 C §C.7;本轮另修正 ch18 正文 `NCCL_NCHANNELS` → `NCCL_MIN/MAX_NCHANNELS`(outline/04、outline/08 同步)。

**d) TRT-LLM v1.2.1**:PyPI 仅 sdist(无 cp312 wheel);pypi.nvidia.com(NVIDIA PEP 503 索引)有 **1.2.1 cp312 wheel(2.5GB)**(安装路径 `--extra-index-url https://pypi.nvidia.com`;索引文件 URL 在索引子目录 `https://pypi.nvidia.com/tensorrt-llm/<file>`)。本轮:wheel 经 curl 断点续传下载成功并安装(`--no-deps`);依赖补齐(torch ≥2.9.1 / tensorrt 10.14.1 / nvidia-modelopt / nvidia-nccl-cu13 等,约 3-5GB)在大文件传输阶段遇 TLS 抖动未完成,运行时 import 未做。**参数名已做源码级实证**(wheel 源码树):`max_batch_size` = 引擎 batch 上限(`runtime/model_runner.py` 超限时报错 "exceeds the engine or specified limit");`use_in_flight_batching` 由 paged KV 状态启用(`models/generation_mixin.py`);PyTorch 后端 = `_torch/pyexecutor/`(py_executor.py,含 `prefix_caching`);Python API executor = `GenerationExecutor` 族(`executor/executor.py`,含 Proxy/Ray/RPC 变体);1.2.1 另有 `disaggregated_params.py`(P/D 支持)。ch04 §4.4(B27)的"in-flight batching 每 iteration 组 batch(上限 `max_batch_size`)+ APC block 复用"口径成立。

**e) GenAI-Perf**:验证机 docker daemon 未运行且无 sudo → 未执行;Ch24-27 压测口径仍遵循 8×H100 环境。

**f) labs 端到端(Qwen3-4B 替代模型,GPU 3,vLLM/SGLang 双真实服务)**:
- **vLLM**(READY ~70s):bench_basics prompt 256/2048/8192(out=32)→ TTFT 0.023/0.047/0.147s(TTFT 随 prompt 长度近线性增长 = prefill 线性形态 ✓),TPOT 恒 ~0.009–0.013s(decode 恒速 ✓);request_tracer(n=32,并发 8)吞吐 ≈624 tok/s,TTFT p50/p95/p99 = 44/59/60ms,TPOT = 12.44/13.06/13.06ms,ITL p99 = 16.36ms;Prometheus 三段 Δ:queue 0.01ms + prefill 27.96ms + 网络/客户端开销 ≈18ms ≈ 客户端 TTFT 46ms(Ch20 L2 三段分解端到端 ✓)。vLLM 指标名 = `request_queue_time_seconds` / `request_prefill_time_seconds` / `request_decode_time_seconds` / `time_to_first_token_seconds`。
- **SGLang**(READY ~75s;metrics 取样轮 READY ~28s[JIT 缓存已热]):首 case TTFT 1.865s(JIT 预热),后续 0.016s;prompt 256/2048/8192 → TTFT 0.016/0.029/0.090s;TPOT ~5.7ms(flashinfer decode,快于 vLLM 默认路径);`/metrics` 默认 404,加 `--enable-metrics` 后取样成功(71.6KB):TTFT = `sglang:time_to_first_token_seconds`(histogram),TPOT 等效 = `sglang:request_time_per_output_token_seconds`,三段分解 = `sglang:per_stage_req_latency_seconds` + `sglang:queue_time_seconds`,前缀缓存 = `sglang:cache_hit_rate` / `sglang:cached_tokens_total`,负载/KV = `sglang:num_running_reqs` / `sglang:kv_used_tokens` / `sglang:gen_throughput`。
