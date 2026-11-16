# 第 10 章 GPU 架构与性能建模

> 难度 ★★★ | 目标 ~30 页 | 前置:无
>
> **读完本章你能做什么**
> 1. 说出 GPU 存储层次(HBM/L2/SMEM/寄存器)与 SM/warp 执行模型,解释"性能 = 数据搬运效率"
> 2. 用 Roofline 判断任意算子 compute-bound 还是 memory-bound(含手算 + `labs/roofline.py`)
> 3. 读 H100/H200/B200 三大卡,说清每个规格参数对 serving 的影响
> 4. 用 nsys / ncu / DCGM 完成"GPU 卡在哪"的第一级判断
> 5. [进阶]读一份 Roofline 图,指出优化方向
>
> **定位**:全书性能分析的地基。第 1 章的"两种形态"在这里获得硬件解释,第 20 章的分析框架在这里获得工具。

## 10.1 硬件模型

**执行模型**:GPU = 132 个 SM(H100)的集合。SM 内:

- 线程组织:thread → **warp(32 线程,SIMT 最小调度单位)** → block(驻留 SM)→ grid(整个 kernel)
- 每个 SM:若干 warp 调度器 + Tensor core + 寄存器文件 + **shared memory / L1(合计 ~228KB,可配)**
- 并发掩盖延迟:SM 同时驻留多个 warp,一个 warp 等访存时切换到另一个——"延迟靠并发藏,吞吐靠占用率"

**存储层次(由快到慢)**:

```
寄存器 (SM 内,最快)  >  SMEM/L1 (~228KB/SM)  >  L2 (~50MB,全芯片共享)  >  HBM (80-180GB)
```

**核心不对称**:算力集中在 SM,数据必须从 HBM 搬到 SM 才能被算。**"性能 = 数据搬运效率"**——一个 kernel 好不好,先看它每读 1 字节 HBM 换回多少 FLOP(算术强度 AI),再看它有没有把 L2/SMEM 用起来。

**Hopper 的关键新硬件**(FA3、DeepGEMM 都建立在它上面,面试要能说出名字与作用):

| 硬件 | 作用 | 谁在用 |
|---|---|---|
| **TMA**(Tensor Memory Accelerator) | 异步批量拷贝 global↔shared,免 warp 参与;支持 multicast(一次搬运广播给多 SM,TP 广播场景) | FA3、DeepGEMM |
| **WGMMA** | warpgroup 级异步 MMA,FP8 支持,MMA 与数据搬运重叠 | FA3、FP8 GEMM |
| **FP8 tensor core** | FP8 MMA,吞吐 2× FP16 | FP8 推理(Ch8) |
| warp specialization | 不同 warp 组分工(producer 搬数据 / consumer 算 MMA),软件流水 | FA3 |

B200:双 die,HBM3e 180GB / 7.7TB/s;NVFP4 tensor core(FP4 稠密 ≈ 2× FP8);NVLink5(1.8TB/s)。(2026-11-16 对照官方 datasheet 复核,详见附录 B)

## 10.2 Roofline:一个公式分两类优化

> **AI = FLOPs ÷ HBM 字节数;ridge point = 峰值 FLOPS ÷ 峰值 HBM 带宽**

H100(FP16):989 TFLOPS ÷ 3.35 TB/s ≈ **295 FLOP/byte**:

- **AI < 295(memory-bound)**:可达性能 ≈ AI × 带宽。优化家族:**抬 AI**(加大 batch、算子融合)、**减字节**(量化、KV 压缩)、**改局部性**(L2 swizzle)
- **AI > 295(compute-bound)**:可达性能 ≈ 峰值 × MFU。优化家族:**更好的 kernel**(FA3)、**更低精度**(FP8/FP4)、**提升 MFU**(occupancy、pipeline)

**手算两例(面试标准动作)**:

1. **decode GEMV,70B TP8,batch=1**:每卡权重 17.5GB/步,FLOPs ≈ 2×70e9×(1/8)×8 = 1.4e11(全模型口径)→ AI ≈ 1 → **深度 memory-bound**。时间下界 = 字节/带宽 = 17.5GB ÷ 3.35TB/s ≈ **5.2ms/token**(单卡视角)
2. **prefill N=2048(7B)**:AI ≈ N = 2048 » 295 → **compute-bound**,时间 ≈ FLOPs/峰值 ≈ 30ms(Ch1 例 1-1)

交叉点 N ≈ 295 → **N ≈ 300**:"batch × 上下文"到几百,decode 才翻成 compute-bound(Ch1 §1.5 的工程含义)。

**MFU(模型算力利用率)**:实测 FLOPs / 峰值。大 GEMM 的 MFU 常见 60-80%;**小 batch decode 的"MFU" 只有 ~3%——这不是 bug,是物理**(AI=1 时,3% 是带宽允许的上限)。面试里"MFU 低"要先问"什么 shape"。

> **踩坑框:"GPU util 100% 但吞吐低"**
> DCGM 的 `gpu_util` 只表示"有 kernel 在跑",**不表示跑在哪个 roof 上**。memory-bound 区跑满 util ≠ 高效——看 ncu 的 Memory SOL(SOL = Speed Of Light,理论上限段;Compute(SM)%/Memory% = 该 kernel 实际用量占对应硬件峰值的百分比),或按本章公式算 AI。util 是"有没有干活",roofline 才是"干得多好"。

## 10.3 三大 GPU:serving 视角的读法

| 规格 | A100 SXM | H100 SXM | H200 SXM | B200 |
|---|---|---|---|---|
| HBM | 80G HBM2e,2.0TB/s | 80G HBM3,3.35TB/s | 141G HBM3e,4.8TB/s | 180G HBM3e,7.7TB/s |
| 算力(稠密) | FP16 312T | FP16 989T / FP8 1979T | 同 H100 | FP8 ~4.5P / FP4 ~9P(量级) |
| NVLink | 600GB/s | 900GB/s | 900GB/s | 1.8TB/s |
| FP8 tensor core | **无** | 有 | 有 | 有(+FP4) |

对 serving 的影响(选型决策的依据):

- **H200 = H100 的带宽 +44%**:decode(memory-bound)→ 预期 **+30-40%**;prefill(compute-bound)→ 基本不变。**"带宽机器",长上下文高并发 decode 的最优选**
- **B200**:FP4 让 MoE 权重 1/4(单卡容量 ×4)+ 带宽翻倍 → MoE/decode 成本最优;prefill 吃 FP4/FP8 tensor core
- **H100**:均衡,生态最成熟
- **A100 的坑**:**没有 FP8 tensor core**——FP8 GEMM 会退化路径,跑 FP8 模型不等于 H100 的 FP8 性能;实验平台与生产平台结论不可直接搬

## 10.4 Profiling 基础:三个工具三步法

**第 1 步 nsys(时间线:时间花在哪)**:

```bash
nsys profile --trace=cuda,nvtx -o run --duration 60 -- python my_infer.py
# 对已运行的服务:nsys profile --trace=cuda -o run -t <pid>
```

读 timeline 找三类异常:

- **GPU gap**(kernel 之间的空隙):CPU 瓶颈(调度/tokenize)、launch 开销小 kernel 过多 → 对应 CUDA graph / 异步调度(Ch12)
- **长 memcpy**:host↔device 拷贝、KV swap → 对应 GDR / 预分配
- **kernel 过长且单一**:定位到 kernel 名,进第 2 步

**第 2 步 ncu(单 kernel:离 SOL 多远)**:

```bash
ncu --set full -k "regex:.*gemm.*" -c 5 python my_infer.py
```

看 SOL 段:Compute(SM)% / Memory % / L2 Hit。判读:Memory 90% + Compute 5% → memory-bound,换量化/融合;Compute 70% + Memory 40% → 接近 roof,换更低精度或更好 kernel;两者都低 → occupancy 问题(寄存器/SMEM 超配、launch 配置差)。

**第 3 步 DCGM(硬件状态:是不是被限制了)**:

```bash
dcgmi dmon -e 1003,1004,150,155   # SM Active, DRAM Active(PROF 字段,需采样模式), 温度, 功耗
# throttle 原因:HW Slowdown / SW Power Cap / Thermal
```

**三步法口诀**:**nsys 找"时间在哪" → ncu 找"离 roof 多远" → DCGM 确认"硬件没被限"**。顺序不能反(上来就 ncu 抓 kernel = 第 20 章批评的"盲钻")。

> **案例框(合成):decode 7ms 去哪了**
> nsys:kernel 合计 ~4ms,gap ~3ms → CPU 侧空转 → 开 v1 异步调度 + 检查 tokenize 批量 → gap 降到 0.5ms,decode 5.5ms。
> 若 kernel 合计 7ms:ncu 抓 GEMV → Memory SOL 85% → 已贴带宽 roof → 唯一出路是量化(减字节)或加大 batch(抬 AI)——**roofline 告诉你能不能救,工具告诉你在哪卡**。

## 动手

**实验 10.1(60 分钟,单卡)**:
1. `nsys` 抓 7B decode(输出 256 token),统计:总时长、kernel 时长和、gap 和
2. 对最大 GEMM kernel 跑 `ncu`,读 Memory SOL;用 `labs/roofline.py` 把该 kernel 的 AI 画上图
3. 结论一句话:"该 decode 离带宽 roof 还有 X%,可优化方向是 Y"

**实验 10.2(10 分钟)**:`python labs/roofline.py --peak 989 --bw 3.35 --ai 1` 与 `--ai 4096`,截图对比,作为 Ch1 例 1-1 的图。

## 章末自测

1. 为什么"性能 = 数据搬运效率"?SMEM/L2 在其中的角色?
2. H100 的 ridge point 是多少?decode GEMV(AI≈1)与 prefill N=2048 各落在哪侧?
3. H200 对 decode/prefill 分别提升多少?为什么?
4. A100 跑 FP8 模型的坑?
5. 三步法:给一个"TPOT 高"的工单,写出 nsys/ncu/DCGM 各查什么、判读什么
6. [进阶]MFU 3% 的 decode 是"优化失败"吗?

## 面试钩子

- 支撑 Ch28-A15~A20、C01:"为什么 decode 是 memory-bound?怎么验证?""H200 比 H100 贵在哪、值在哪?""FA3 用了 Hopper 的什么?"
- Ch20 的 L1-L3 分析框架 = 本章三步法的系统化。
