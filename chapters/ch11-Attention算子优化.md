# 第 11 章 Attention 算子优化

> 难度 ★★★★ | 目标 ~30 页 | 前置:Ch10
>
> **读完本章你能做什么**
> 1. 解释 naive attention 的 O(N²) HBM 物化问题,以及 FlashAttention 如何用 tiling + online softmax 消掉它
> 2. 说出 FA v1→v2→v3 各自的瓶颈与解法(含 Hopper 硬件依赖)
> 3. 解释 FlashInfer 解决推理 decode 的什么问题(paged / 变长 batch / split-KV)
> 4. 解释 FlashDecoding:为什么长上下文 decode 要"切 KV 并行"
> 5. [进阶]写出一个简化 Triton paged attention kernel
>
> **定位**:attention 是 LLM 里唯一"形状随上下文变化"的大算子,prefill 的算力大头、长上下文的显存与延迟大头都在这里。

## 11.1 为什么 attention 必须融合

naive 实现的 HBM 流量(每层):

```
S = QKᵀ          → 物化 N×N 矩阵:写 N² 个 float
P = softmax(S)   → 再读 N²、再写 N²
O = PV           → 再读 N²
```

attention 的 FLOPs 是 O(N²·d),但 naive 的 I/O 也是 O(N²) 级别的**全精度物化读写** → 算术强度低、显存 O(N²)(长上下文直接爆,Ch1 §1.1 注)。

**FlashAttention 的答案(IO-aware)**:把 Q 与 KV 切成 tile,在 SMEM/寄存器里完成"读 tile → 算部分 scores → online softmax 合并 → 写回 O",**S 矩阵从不落 HBM**:

- 显存:O(N²) → O(N)
- HBM 流量:O(N²) → O(N)(只搬 Q/K/V/O 各一遍)
- 速度:prefill 上 ~5×(FA1 paper);backward 重算(不存 S)
- 本质(接 Ch10):**把 AI 从 memory-bound 抬回 compute-bound**——与 decode 靠 batching 抬 AI 是同一个物理

## 11.2 FlashAttention v1 → v2 → v3

**FA v1**:tiling + online softmax 的基础形态。

**FA v2**(瓶颈:指令组合与并行度):

1. 减少非 matmul 的 FLOP(softmax 的 exp/归一化只在必要时做,省了 O(N²) 级别的小操作)
2. **forward 沿 Q(输出序列)维并行**而不是沿 K 维 → 占用率更好
3. 减少 HBM 读写(K/V 只在需要时重读)

**FA v3**(瓶颈:Hopper 的异步流水线没用满):

1. **WGMMA 异步 MMA** + **TMA 异步搬运**:数据搬(MEM 侧)与矩阵算(MMA 侧)重叠
2. **warp specialization**:producer warp 组用 TMA 搬 tile,consumer warp 组做 MMA/softmax(exp 在 SFU 上与 MMA 重叠)
3. 软件流水:p(i+1) 的 exp 与 p(i) 的 matmul 并行
4. 结果:较 v2 再 **1.5-2.0×**;并支持 FP8 attention(长上下文 prefill 额外收益)

| 版本 | 当时的瓶颈 | 解法 |
|---|---|---|
| v1 | HBM 流量(O(N²) 物化) | tiling + online softmax |
| v2 | 指令组合 / 并行度 | 减小操作、Q 维并行 |
| v3 | Hopper 异步能力未用满 | WGMMA + TMA + warp  specialization + 软件流水 |

> 面试模板:"FA 每一代的进步 = 把 IO 从瓶颈里拿走之后,下一个瓶颈浮现"——v1 拿掉 HBM 物化,v2 拿掉指令浪费,v3 拿掉同步等待。

## 11.3 FlashInfer:推理 attention kernel 库

FlashAttention 面向训练/通用 prefill;**decode 场景**(每请求 1 个 q token、KV 在 paged 块里、batch 内各请求长度不同)有专门的形态,FlashInfer 就是为它做的 kernel 库(SGLang 默认):

1. **paged KV 原生**:kernel 直接收 block table(Ch5 的产物),不要求 KV 连续
2. **变长(ragged)batch**:一个 kernel 跑完整个 batch,各请求 seq_len 不同(无 per-request padding)→ decode batch 效率
3. **split-KV(FlashDecoding)**:见 11.4
4. 与采样/其他 op 的融合(减少 kernel launch 边界)

```python
# flashinfer 用法(简化;真实 API 有 work buffer 管理等细节)
import flashinfer

wrapper = flashinfer.PageAttentionWrapper(work_buf, kv_lens, use_tensor_cores=True)
wrapper.plan(qo_indptr, kv_indptr, kv_last_page_len,   # 变长 batch 的索引描述
             max_q_len=1, max_kv_len=max_ctx, page_block_size=16)
out = wrapper.run(q, paged_kv_cache)                   # 一个 kernel 跑全 batch decode
```

**为什么 SGLang 选它**(Ch4 §4.3):decode kernel 性能 + 工程完整度(paged/变长/split-KV 全家桶),与 RadixCache 的 block 体系天然配套。

## 11.4 FlashDecoding:长上下文 decode 的"切 KV"

问题:decode 时 attention 要对 128K 的 KV 逐块扫——**单个 CTA 串行处理 128K**,时间 ∝ KV 长度,且 GPU 大部分 SM 闲着(一个请求只占一个 CTA)。

解法(FlashDecoding / split-KV):

```
128K KV → 切成 8 片,8 个 CTA 并行各扫 16K
每片产出:局部 (O_i, m_i, l_i)(online softmax 状态,Ch5 公式)
一次 log-sum-exp 归约:O = Σ exp(m_i − M)·l_i·O_i / Σ exp(m_i − M)·l_i
```

正确性:online softmax 的可加性(Ch5 §5.2 的 m/l 状态)——**"先分块后归约"与"一次算完"数学等价**。

含义:decode attention 的 KV 扫描变成"可并行的 compute",长上下文 decode 不再被单 CTA 拖死;代价是归约 + 更多 HBM 读(每片都读 Q)——**KV 足够长才值得**(典型阈值 ~8K+)。

## 11.5 长上下文 attention 的其他形态(接 Ch17)

- **sliding window**(Mistral 类):只算窗口内 + attention sink,KV 显存 O(窗口)
- **Ring Attention / CP**:序列切到多 GPU,KV 环状轮转(Ch17)
- **MLA(DeepSeek)**:KV 压缩为 latent 向量(Ch13/17),decode kernel 特化——KV 体积降 5-10×,FlashDecoding 类切分仍适用

## 11.6 [进阶] 手写简化 Triton paged attention

教育版:单块 KV、decode(1 个 q token)、说明"block 间接 + online softmax"的最小形态:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def paged_decode(q_ptr, kv_ptr, block_table_ptr, o_ptr,
                 qk_scale, D: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_h = tl.program_id(0)                       # 每个 KV head 一个 program(简化)
    q = tl.load(q_ptr + pid_h * D + tl.arange(0, D))
    phys = tl.load(block_table_ptr)                # 逻辑→物理(单块示意;完整版循环遍历所有块)
    off = phys * (BLOCK_S * D)
    k = tl.load(kv_ptr + off + tl.arange(0, BLOCK_S)[:, None] * D
                            + tl.arange(0, D)[None, :])        # [BLOCK_S, D]
    v = tl.load(kv_ptr + off + (BLOCK_S * D)          # V 拼在 K 之后(简化布局)
                            + tl.arange(0, BLOCK_S)[:, None] * D
                            + tl.arange(0, D)[None, :])
    s = tl.sum(q[None, :] * k, axis=1) * qk_scale    # [BLOCK_S]
    m = tl.max(s, 0)                                  # online softmax 状态(完整版跨块维护)
    p = tl.exp(s - m)
    l = tl.sum(p, 0)
    o = tl.sum(p[:, None] * v, axis=0) / l
    tl.store(o_ptr + pid_h * D + tl.arange(0, D), o)
```

完整生产版(对照 Ch5 核心循环):外层 `for b in blocks` 维护 (m, l, O) 跨块更新;多请求变长 batch 用 indptr 索引;split-KV 时每个 program 只扫一段 block 范围,归约在外层。**写这个 kernel 的收获不是性能,是"看懂 FA/FlashInfer 在做什么"**——面试讲 paged attention 时能现场画 block 间接 + online softmax,比背结论强一个量级。

## 动手

**实验 11.1(90 分钟,有 GPU)**:
1. 跑上面的简化 kernel,与 `torch` 直接 attention 对比正确性(同输入)
2. 把外层改成 block 循环(完整版),对比 FlashAttention(pip `flash-attn`)在 N=8K 的速度与显存:验证"显存 O(N²)→O(N)"

**实验 11.2(读码)**:flashinfer 仓库里找 paged decode kernel 的 split-KV 参数(何时启用),用一句话解释阈值逻辑。

## 章末自测

1. naive attention 的 HBM 流量为什么是 O(N²) 级别?FA 怎么消掉的(两个机制)?
2. FA2 相对 FA1 的并行度改进?FA3 用了 Hopper 哪两个硬件?
3. FlashInfer 的"变长 batch"解决什么问题?为什么 decode 特别需要?
4. FlashDecoding 的归约为什么数学上正确?什么长度下才值得启用?
5. [进阶]手写 kernel 里,online softmax 的三个状态 (m, l, O) 分别是什么?跨块怎么合并?

## 面试钩子

- 支撑 Ch28-C01~C08:"FA v2 与 v3 的区别?""FlashInfer 解决什么?""长上下文 decode 为什么慢,怎么并行?""MLA 的 KV 压缩对 decode kernel 意味着什么?"
- Ch29 案例 3(1M RAG)的 prefill/decode 算子选择。
