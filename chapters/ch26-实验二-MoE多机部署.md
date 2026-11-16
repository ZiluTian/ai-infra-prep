# 第 26 章 实验二:MoE 多机部署

> 难度 ★★★★★ | 目标 ~25 页 | 前置:Ch13、Ch16、Ch24
> 硬件:1 节点 8 卡必须;2 节点可选(EP 跨机)
> 模型:Qwen3-30B-A3B(激活 3B)或可用的 DeepSeek 轻量版
> 每步统一格式:命令 → 期望结果 → 常见失败模式 → 思考题。
>
> **交付物**:"MoE 多机调优报告"(Ch29 案例 8 的数据来源)。

## 26.0 实验总览

```
部署对比(TP8 vs EP8)→ 插桩观测 → (跨机 EP)→ MoE 投机解码 → 偏斜注入与治理
主线问题:MoE 比同激活规模稠密模型,收益在哪、代价在哪(通信/长尾)?
```

**负载定义**:`labs/bench_load.py` 的 mixed profile(同 Ch25)+ **偏斜 profile**(§26.5:重复性 prompt 诱导热 expert)。

## 26.1 部署:单机 TP8 vs EP8

```bash
# TP8(attention 与 expert 都切)
vllm serve <moe-model> --tensor-parallel-size 8 --port 8001 &
# EP8(vLLM 的 Fused MoE 单机路径;SGLang 可开 DeepEP 节点内模式)
vllm serve <moe-model> --enable-expert-parallel --tensor-parallel-size 8 --port 8002 &
```

**期望结果(量级,30B-A3B 类,8×H100)**:

| | TP8 | EP8 |
|---|---|---|
| 每卡显存 | 权重/8 + KV/8(均匀) | 权重/8(expert 均分)+ 注意力权重(复制)→ **略高或持平** |
| decode 吞吐 | 中 | **高 ~20-50%**(expert GEMM 不被 TP 通信切碎,grouped GEMM 本地跑满) |
| all-to-all 占比 | 0(无 EP) | 5-10%(NVLink,Ch13) |
| p99 | 平稳 | 略高(热 expert 长尾,§26.5 量化) |

**常见失败模式**:EP 路径要求 expert 数被 8 整除(Qwen3-30B-A3B 128 experts ✓,部分模型不整除 → 框架补齐/复制逻辑,读源码确认);DeepEP 需要匹配的内核版本(与 CUDA/驱动矩阵核对,§24.1)。
**思考题**:同激活规模下,EP8 吞吐收益的来源排序(GEMM 局部性?通信量?batch 效率?)——用 nsys 各段占比回答。

## 26.2 插桩观测(本实验核心)

**① 每 expert token 分布(`hook_router.py`,读者自建,伪代码骨架如下;不随书分发,见 Ch24 §24.5)**:

```python
# hook MoE 层的 top-k 路由输出(框架 hook 点:router 后 / dispatch 前)
# 累计计数: counter[e] += 每步每 token 的命中
# 输出: top-10 热 expert 表 + 分布 Gini 系数 + 直方图
```

**期望结果**:均匀路由(无偏斜负载)下 Gini 也 > 0(路由天然数据相关);偏斜 profile 下 top-3 expert 占 30%+ token。**这张表 = Ch29 案例 8 的"热度统计"输入**。

**② all-to-all 耗时占比**(nsys,EP8 部署):

```
nsys profile --trace=cuda,nvtx -o /tmp/exp2-ep --duration 120   # 稳态负载
→ MoE 层内: dispatch + combine kernel 时间 / 该层总时间
```

**期望**:单机 NVLink 下 5-15%;**若 >25% → 查 P2P 是否退化(Ch18 §18.4,`NCCL_DEBUG=INFO`)**。

**③ grouped GEMM 效率**(ncu 抓 expert GEMM kernel):

- 看:每 expert 的 M 分布(M_e 直方图)× Memory/Compute SOL
- **期望**:decode 小 batch 下大量 expert M_e ≤ 2 → GEMV 路径;热 expert M_e 大 → GEMM 路径。两种 kernel 的效率判读分开(Ch13 §13.3)
- **思考题**:M_e=1 的 expert 走 GEMV 与走 GEMM(M=1)差多少?框架的切换阈值在哪(读源码)

## 26.3 [可选:2 节点] 跨机 EP 与 DeepEP 双模式

```bash
# EP16(跨 2 节点):mpirun -H node1:8,node2:8 启动服务,--enable-expert-parallel
# DeepEP 模式切换(框架配置项;low-latency 用于 decode)
# 对照: prefill-heavy 负载 → normal;decode-heavy → low-latency
```

**期望结果(量级,IB NDR)**:

- decode 步 all-to-all:low-latency 比 normal **延迟低 2-5×**(分组 + IBGDA,Ch16 §16.2);吞吐场景 normal 反超
- 跨机 EP16 vs 单机 EP8:吞吐接近(权重分散,单卡 expert 少),**p99 恶化**(straggler 域扩大,Ch16 §16.4)
- 带宽核对:nsys 抓 IB 传输段 vs Ch18 基线(§24.4 存档)——若只有基线 50% → GDR 失效

**常见失败模式**:两节点 NCCL bootstrap 失败(SOCKET_IFNAME 未设)、low-latency 模式要求 buffer 预注册(显存账:DeepEP buffer 额外占 1-2GB/卡,`gpu_memory_utilization` 要留余量)。
**思考题**:prefill 与 decode 混在同一 EP 组里,两种模式怎么共存?(方向:框架按 step 类型切换 / P-D 分离后各池各模式,Ch9)

## 26.4 MoE 上的投机解码

```bash
# EAGLE(有权重)或 n-gram,batch 扫描 1/8/32(同 Ch25 §25.5)
```

**期望结果(验证 Ch7 §7.4 + Ch8 §8.5 预测)**:MoE 的 α **通常低于同条件稠密模型**(draft 模型与 MoE 目标的路由/专家分布错位,尤其 draft 是稠密小模型时);加速比曲线整体下移,但 batch=1 仍可能 1.5×+。**思考题**:为什么"draft 用同家族 MoE 小模型"比"稠密 draft"α 高?(路由分布接近)

## 26.5 偏斜注入与热 expert 治理

```bash
# 注入: 偏斜 profile(80% 请求为同主题长 prompt,诱导固定 top experts)
# 观测: ① 热 expert 表(§26.2①) ② p99 放大倍数(vs 均匀负载) ③ 最热卡 GEMM 时间
# 治理 A(放置): 按热 expert 表重排 expert→rank 映射(框架支持时;否则代码级)
# 治理 B(复制): 热 expert 复制 2 副本(读源码:expert 权重加载逻辑,复制 + 路由表改)
# 各测: p99 变化、显存代价、吞吐代价
```

**期望结果(复现 Ch13 §13.4 的 20× 量级)**:均匀负载 p99/p50 ≈ 1.5-2×;偏斜负载下 **p99 放大 3-10×**(最热卡 M_e 偏斜);治理 A(分散放置)p99 回落 ~30-50%;治理 B(复制)p99 进一步回落,代价 = 权重显存 ×2(热 expert)+ 少量路由开销。
**常见失败模式**:复制后路由表未同步 → token 找不到 expert(框架自研改动要过完整回归);放置优化对**瞬时**尖峰无效(热度统计是滞后的,Ch16 §16.3)。
**思考题(面试原题)**:热 expert 复制的"触发条件"应该设计成什么(监控信号 + 阈值 + 回退)?

## 26.6 交付物:"MoE 多机调优报告"

```markdown
1. 环境(节点数/网络/版本矩阵)
2. TP8 vs EP8 对比表(吞吐/延迟分位/通信占比/显存)
3. 热 expert 表 + Gini(均匀 vs 偏斜负载)
4. all-to-all 占比与 grouped GEMM 效率(M_e 直方图 × SOL)
5. (跨机)DeepEP 双模式对比 + GDR 验证
6. 投机解码 α 与加速比(vs 稠密基线)
7. 热 expert 治理: 放置/复制的 p99 收益与代价
8. 结论: 该负载下 MoE 的最优部署形态(EP? 放置策略? 投机开/关?)
```

**面试用法**:本章实验直接支撑 Ch28-C17~C24 / D15~D20;讲法 = "我量化了 MoE 的收益与长尾,并做了热 expert 治理闭环"——**有数据、有方法、有取舍**,而非"我部署过 MoE"。

## 章末自测

1. TP8 vs EP8:收益来源排序,用 nsys 哪段占比证明?
2. 插桩三件套(路由计数/通信占比/GEMM 效率)各回答什么问题?
3. DeepEP 两模式的切换判据(带宽/延迟主导);跨机 EP 的 p99 为什么恶化?
4. 热 expert 治理两手段的成本-收益各写一行;触发条件设计(监控信号 + 阈值 + 回退)
5. [进阶]MoE 服务 SLO 按 p99 还是 p50 定容量?给数字论证(§26.5 的放大倍数)
