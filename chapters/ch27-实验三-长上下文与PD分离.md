# 第 27 章 实验三:长上下文与 P/D 分离

> 难度 ★★★★★ | 目标 ~20 页 | 前置:Ch9、Ch17、Ch24
> 硬件:8×H100;P/D 需 2 节点(或单机模拟:两个副本 + 共享 KV 传输路径)
> 每步统一格式:命令 → 期望结果 → 常见失败模式 → 思考题。
>
> **交付物**:"生产级优化报告"——全书综合,Ch30 的"45 分钟项目 deep-dive"模板。

## 27.0 实验总览

```
128K KV 验证(公式 vs 实测)→ 长 prefill 调参 → (CP)→ P/D 分离 A/B → SLO goodput 综合
主线问题:长上下文 + 高负载下,"一体 vs P/D"的完整成本-收益闭环
```

## 27.1 128K 上下文:KV 显存验证(Ch3 公式 vs 实测)

```bash
# 模型: 70B GQA(TP8)+ 8K/128K 请求各 1 条, 逐条记录 KV 池增量
# 公式(Ch3 (3-1)): KV/token/卡 = 2 × layers × kv_heads/tp × head_dim × bytes
# 70B 例: 全模型 2×80×8×128×2B = 320KB/token → TP8 后 40KB/token/卡 → 128K ≈ 5.12GB/卡(全模型 40.96GB)
vllm serve <model> --tensor-parallel-size 8 --max-model-len 131072 &
# 发 1 条 128K 请求, 记录: KV 池使用率增量(vLLM metrics) vs 公式值
```

**期望结果**:实测 KV/卡 与公式 **±10% 内**(误差来源:block 对齐、框架预留、attention sink)。**这一步是全书公式体系的"地面真值校验"——公式对不上就回 Ch3 查参数(head 数/head_dim/字节数)。**
**常见失败模式**:`max_model_len` 未设 → 长请求被拒;128K 单请求 OOM(KV 池被权重挤占,`gpu_memory_utilization` 重算,Ch21 §21.1)。
**思考题**:FP8 KV(若支持)后,单卡 128K 并发上限从 1 变几条(Ch3 回代)?

## 27.2 长 prefill 调参(max_num_batched_tokens)

```bash
# 负载: 纯长 prefill(64K-128K 输入, 输出 256)
# 扫描: max_num_batched_tokens 4K / 16K / 32K
# 指标: TTFT、单请求激活峰值(nvidia-smi 采样)、decode 干扰(混入短请求时)
```

**期望结果**:chunk 越大 TTFT 越短(少切块、少调度轮)但**激活峰值越高**(Ch17 §17.1 的 FLOPs/激活账)+ 对 decode 的步时干扰越大;存在负载最优解(典型:16K 级)。**记录"TTFT vs 激活峰值"权衡表**——这就是 Ch17 §17.3 prefill 集群参数的来源。
**思考题**:为什么 chunk 大小与"节点数"(CP)是同一决策的两种表达?(CP = 把序列切到空间,chunk = 切到时间,Ch17)

## 27.3 [可选:2 节点] CP 长上下文 attention

```bash
# 框架支持 CP 时(或自测 ring attention 最小实现, Ch17 §17.2 伪代码):
# 对照: 128K prefill, CP1(单节点 TP8) vs CP2(2 节点 × TP4 或 CP2×TP8)
# 指标: prefill 耗时、NVLink/IB 带宽占用(ns)、KV/卡 显存
```

**期望结果(量级)**:CP2 prefill 耗时 ≈ 单卡/1.5-1.8(非 2×:ring P2P 延迟 + 通信未被完全掩盖,Ch17);KV/卡 减半(§27.1 公式回代)。**带宽核对**:IB 传输段 vs §24.4 基线——低于 50% → GDR/MTU 问题(Ch18)。
**常见失败模式**:head 数与 CP size 不整除(Ulysses 约束,Ch17 §17.2);CP 与 TP 组的 gang 分配(16 卡整组,Ch14)。
**思考题**:CP2 只赚 1.7× 而非 2×,缺失的 0.3× 去哪了(延迟/掩盖/归约)?

## 27.4 P/D 分离 A/B(本实验核心)

**架构**(vLLM production disagg / Mooncake,Ch9 §9.4):

```bash
# P 池: vllm serve <model> --tensor-parallel-size 8 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
# D 池: 同模型, kv_role 改 "kv_consumer";网关: 按 Ch9 协议路由(prefill 完成 → KV 传输 → decode)
# (v0.32.0 的 kv_role 取值: kv_producer/kv_both/kv_consumer;早期版本的 prefill/decode 写法已弃用,附录 G §G.5)
# 对照 A: 一体副本(8 卡, 默认 chunked prefill)
# 对照 B: P/D(8+8 卡 或 模拟: P 4 卡 + D 4 卡)
```

**负载**:§25.0 的 mixed profile(20% 长文档 = P/D 收益区)+ 前缀复用 50%。**指标(两组同载)**:

| 指标 | 一体(A) | P/D(B) | 判读 |
|---|---|---|---|
| TTFT p95(长请求) | 高(排队+干扰) | **↓ 40-60%** | P 池无 decode 干扰 |
| TPOT p99(在跑请求) | 长请求进入时尖刺 | **平稳** | D 池纯净 |
| KV 传输开销 | 0 | ~2-3% prefill 时间(8K 级,per-rank 分片,Ch9 §9.2) | 长上下文占比 |
| 总吞吐/卡 | 基线 | ±(卡数不同则归一化) | **关键:$/token,不是吞吐** |
| 网络带宽占用 | 0 | IB 占用时序(n) | 与 NCCL 竞争?(Ch18 §18.4) |

**期望结论(复现 Ch9 判据)**:长上下文 + 高负载 + 双 SLO → P/D 显著优;**若把负载改成 80% 短请求,P/D 反输**(传输成本主导)——**同一硬件,负载结构决定架构,这就是 Ch9 §9.3 决策公式的实证**。
**常见失败模式**:KV 传输布局不匹配(P/D 的 TP size 必须一致,Ch9 §9.2);D 池 KV 预留不足(传输中 slot 未预留 → 拒收);P 池超调度(白算 prefill,失败语义,Ch9 §9.2)。
**思考题**:P:D 配比 1:1 vs 2:1,你的负载(输入/输出 token 比)该选哪个(Ch9 §9.5)?

## 27.5 综合:SLO 与 goodput(全书公式闭环)

```
SLO 定义: p95 TTFT < 2s | p99 TPOT < 50ms(§25 负载 + §27.4 架构)
goodput(Ch2 (2-5)): 满足双 SLO 的并发请求速率(req/s)
扫描: 并发 32/64/128/256 × 架构 {一体, P/D} → goodput 曲线
→ 交点 = "该负载下 P/D 开始赢"的并发阈值(你的 Ch9 判据的实测版)
```

**期望结果**:goodput 曲线两线相交,交点左侧一体优、右侧 P/D 优;交点位置与 Ch9 §9.3 的解析估算**同量级**(±50%)。**这一步是本书"公式 → 实测"闭环的完成:从 Ch3 的 KV 公式到 Ch9 的传输成本,全部在真实硬件上对过账。**

## 27.6 交付物:"生产级优化报告"(全书综合模板)

```markdown
# 长上下文高并发 serving:一体 vs P/D 的完整决策
1. 需求与 SLO(带数字): 负载构成、双 SLO、并发目标
2. 容量估算: Ch3 公式(§27.1 已验真值)+ Ch9 传输成本 → 卡数预算
3. 架构候选: 一体 / P/D(1:1, 2:1)/ prefill 集群(CP)——各自成本-收益
4. 实测: §27.2-27.5 全部记录表 + goodput 曲线
5. 决策: 该负载下选什么(判据引用 Ch9 §9.3)、什么负载下会反转(短请求占比 > X%)
6. 风险与未解决项: KV 传输失败语义、P 池超调度、网络竞争(各给缓解)
```

**面试讲法(Ch30 模板)**:45 分钟 deep-dive 的结构 = **需求(2min)→ 容量估算白板推(8min)→ 三个候选架构与排除理由(10min)→ 关键实验与数据(12min)→ 权衡与风险(8min)→ 开放问题(5min)**。数据全部来自本实验的 results/ 目录,**每个数字可现场复现**——这是与"背八股"候选人的本质区别。

## 章末自测

1. §27.1 的公式验真值:实测与公式差 10% 以上时,检查哪三个参数?
2. chunk 大小、CP size、P 池卡数,三个旋钮各控制什么(时间切分/空间切分/算力)?
3. P/D A/B 的"关键指标"为什么是 $/token 而不是吞吐(卡数不同时)?
4. goodput 曲线的交点为什么是 Ch9 判据的实证?交点左/右各什么结论?
5. 把 §27.6 模板改写成"128K 上下文 prefill 集群"(Ch29 案例 7)的需求-架构-权衡大纲
