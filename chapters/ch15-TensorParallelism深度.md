# 第 15 章 Tensor Parallelism 深度

> 难度 ★★★★ | 目标 ~30 页 | 前置:Ch14
> 分层:15.1-15.2 [基础];15.3-15.4 [进阶]。
>
> **读完本章你能做什么**
> 1. 写出 Megatron TP 的层内布局(ColumnParallel / RowParallel),说清 GQA 下 kv heads < TP 的处理
> 2. 推导"每层 2 次 all-reduce"的完整链路,并解释 all-gather + reduce-scatter 替代(SP)
> 3. 用数字解释为什么 TP 不宜 > 8(延迟 × 层数 × 2;NVLink vs IB)
> 4. [进阶]说出通信-计算 overlap 的两种形态与 DeepSeek 的 layerwise overlap
> 5. [进阶]走读 vLLM/SGLang 的 TP 源码:parallel state、all-reduce 封装、forward 中通信位置、与 CUDA graph 的兼容

## 15.1 Megatron TP 布局

**两种基础切法**(权重矩阵 `Y = X·W`):

- **ColumnParallel(列切)**:`W` 按列切 `(K, N/tp)`;每 rank 乘出 `Y` 的 1/tp 列 → **输出是切分的(hidden states sharded)**,本层内后续计算可用,但跨层需要通信
- **RowParallel(行切)**:`W` 按行切 `(N/tp, K)`;输入 `X` 必须先是切分的,各 rank 算出部分和 → **all-reduce 后输出完整**

**标准 Transformer 层的组合(必须会画)**:

```
X (完整)
 └─ QKV 投影: ColumnParallel ──► 每 rank 持 N/tp 个 head 的 qkv(含自己的 kv head 分片)
      └─ attention 计算: 各 rank 独立(自己的 head 子集)→ O 部分和
           └─ O 投影: RowParallel ──► [all-reduce #1] ──► 完整残差和
 └─ MLP: up/gate 投影 ColumnParallel → SiLU×mul → down 投影 RowParallel ──► [all-reduce #2]
```

**每层恰好 2 次 all-reduce**(O 投影后 + down 投影后)——这是 Ch14 (14-1) 公式的结构性来源。

**GQA 的特殊处理**:GQA 的 kv heads 数常 < TP(如 70B:8 个 kv head,TP16 放不下)。工程解:**复制**——让部分 rank 复制持有 kv head(每个 rank 至少 1 个 kv head,多的 rank 拿重复份)。vLLM 在 `get_tp_group` 构建时按 `max(1, num_kv_heads // tp)` 分配并补齐;**复制只影响显存(KV 分片冗余),不影响正确性**(相同 kv head 的 attention 结果一致)。MQA(1 个 kv head)下所有 rank 复制同一份。

**显存账(接 Ch3)**:TP 切的是**权重与激活**,KV cache 按 head 切分(k head 落在哪 rank,KV 就在哪 rank)→ KV/卡 ≈ (Ch3 公式) / tp;TP 不改变"每 token 的 KV 总量",只改变分布。

## 15.2 通信模式深究

**为什么是 2 次、能不能更少?** 不能省:列切后 attention 输出必须合并(O 是行切,部分和)、MLP 同理。但**换形式**可以:

**all-gather + reduce-scatter 替代(Sequence Parallel 的通信形态)**:

```
all-reduce(x)  ≡  reduce-scatter(x) + all-gather(result)   (通信量等价)
```

Megatron SP 的用法:把 **layernorm / dropout 所在的"非 GEMM 段"也按序列维切**到 tp 个 rank——layernorm 的激活显存降到 1/tp,通信总量不变但位置前移/后移,与 GEMM 的流水更解耦。**面试口径:"SP 不是减少通信,是把序列维切进 TP 域,激活显存换布局"**。

**为什么 TP 不宜 > 8(定量,直接引用 Ch14)**:

| | 每步 all-reduce 传输 | 70B 80 层每步总量 | 延迟量级 |
|---|---|---|---|
| TP8, NVLink 900GB/s | 16KB×2×80 = 2.5MB | ~50-100μs/步 | 可接受(decode step ~6-8ms 的 1-2%) |
| 跨机 IB 50GB/s | 同左 | ×18 | 1-2ms/步(占 20-40%) |

另外 ring all-reduce 的消息延迟随 N 增长(跳数 N-1),TP16 需要两个 NVLink 域拼——NVSwitch(HGX 节点的 NVLink 交换 fabric:让 8 卡等价于全连接高速网)域是 8 卡边界,**TP 的天花板由 NVLink 拓扑画,不由算法画**。

**TP 的延迟 scaling 是超线性的**(动手实验验证):TP2→TP4→TP8,单 token 延迟并非 1/2、1/4、1/8 下降——权重字节 ÷tp(理想)− 通信增量 − all-reduce 跳数延迟 − 负载不均。**物理下界(BF16 权重 ÷ 3.35TB/s)+ 典型值 [估算]**:70B decode,TP2 ≥ 20.9ms(典型 ~22-25ms)、TP4 ≥ 10.4ms(~12-14ms)、TP8 ≥ 5.2ms(~6-8ms,含通信/KV 读/采样)——非 1/4、1/8 关系 → "TP 加卡的边际收益递减"是面试要会的结论。

## 15.3 [进阶] 通信-计算 Overlap

**形态 1:层块 overlap(通用)**:把一层切成 2-4 个块,块 k 的 all-reduce 与块 k+1 的 GEMM 在不同 stream 上重叠。代价:同步点增加、实现复杂;收益:小消息 all-reduce(μs 级)完全藏进 GEMM(ms 级)后面。

**形态 2:layerwise overlap(MoE 场景,DeepSeek 实践)**:

```
layer i:   expert GEMM(本地计算)          ──► 下一层 all-to-all 可提前发起
layer i+1: all-to-all 传输(通信 stream) ──► 与 layer i 的 GEMM 重叠
```

DeepEP low-latency 模式的 IBGDA 设计正是为此:GPU 端直接 post RDMA,通信与计算在同一 GPU 上并行,不占 CPU。**前提:token 的路由在上一层 router 后已知 → 传输目标可提前确定**。

**形态 3:NVLS(NVLink SHARP,硬件侧)**:节点内 all-reduce 由 NVSwitch 硬件做规约,省掉 ring 的多跳软件通信——NCCL 自动选择(NCCL_ALGO=NVLS,Ch18),小-中消息延迟显著降低。

## 15.4 [进阶] TP 源码走读

**vLLM(SGLang 同构,路径有异)**:

1. **parallel state**:`vllm/distributed/parallel_state.py` 的 `GroupCoordinator`(持有 torch ProcessGroup + 通信后端封装);`get_tp_group()` 全局取 TP 组。启动时按 `world_size` 拓扑切出 TP/PP/DP 组(Ch14 的层级关系在这里落实)
2. **all-reduce 封装**:小消息(经验阈值 ~几百 KB 以下)走 **custom all-reduce**(symmetric memory / P2P 直连,`vllm/distributed/device_communicators/`),大消息走 NCCL——**NCCL 的小消息延迟地板高,自研 P2P AR 在 decode 尺度上更快**(这就是"vLLM 不用 NCCL 做 decode all-reduce"的完整答案)
3. **forward 中通信位置**:`ColumnParallelLinear.forward` 无通信(输出切分),`RowParallelLinear.forward` 尾部 `tensor_model_parallel_all_reduce`(15.1 的两个点);attention 层 `unified attention` 前后无通信
4. **与 CUDA graph 的兼容(接 Ch12)**:all-reduce 是**固定 shape + 固定通信组**的 kernel 序列 → 可进图;decode 分桶捕获时,图内的通信参数(batch 桶大小)固定。动态 shape 的 prefill 不进图;EP 的 all-to-all(decode 桶)同样可进图——**"通信进 graph" 的判据 = 消息 shape 与通信拓扑静态**

**SGLang 差异点**:`python/sglang/srt/distributed/` 同构;额外有 **overlap scheduling**(CPU 调度下一 step 与 GPU 当前 step 重叠,Ch6 的调度器细节)与 RadixCache 的 block 间接(Ch12 的 graph 兼容性)。

> **踩坑框:"TP 显存不够就开更大 TP"**
> 三个检查顺序:① 是否先试了 KV 量化/压缩(Ch5,不改拓扑);② 是否先试了 DP 减并发(Ch19,不改拓扑);③ TP 加大的通信与延迟代价(15.2 超线性)。**TP 是"装不下"的最后手段,不是第一反应。**

## 动手

**实验 15.1(半天,8 卡或 ≥2 卡)**:70B(或可用最大模型)TP2 / TP4 / TP8:

1. decode 单请求延迟、batch=16 吞吐(Ch2 模板)
2. 画 "TP 规模 vs 单 token 延迟" 曲线,拟合超线性度(1/tp 理想线的偏离)
3. nsys 抓 TP4 的一个 step:all-reduce kernel 时间占比;开/关 custom AR(vLLM 环境变量)对比小消息延迟
4. 交付物:scaling 表 + 结论(该模型该负载下"最优 TP = ?")

## 章末自测

1. 画出一层 Megatron TP 布局:哪四个矩阵是列切/行切?两个 all-reduce 发生在哪?
2. 70B GQA(8 kv heads)TP16 时 kv head 怎么处理?显存代价?
3. "每层 2 次 all-reduce"能否换形式?SP 的 all-gather + reduce-scatter 换来什么?
4. 用数字链解释 TP ≤ 8;TP2→TP8 延迟为何不是 1/4?
5. [进阶]DeepSeek 的 layerwise overlap 为什么可行(router 提前知道路由)?

## 面试钩子

- 支撑 Ch28-D07~D14:"TP 一层几次通信?""vLLM 为什么不用 NCCL 做 decode all-reduce?""TP 显存不够怎么办?""GQA + 大 TP 的 kv head 处理?"
- Ch29 案例 1/2 的节点内拓扑决策都落在这章。
