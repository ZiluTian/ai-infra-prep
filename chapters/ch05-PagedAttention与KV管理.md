# 第 5 章 PagedAttention 与 KV Cache 管理

> 难度 ★★★★ | 目标 ~30 页 | 前置:Ch3
>
> **读完本章你能做什么**
> 1. 说清"连续 KV"的两种浪费,以及"拒绝率由最大连续空闲区决定"
> 2. 描述 PagedAttention 的数据结构(block table)与 kernel 改造(分块计算 + online softmax),能写出核心循环
> 3. 解释 block size 的取舍(为什么是 16)
> 4. 读懂 vLLM 的 BlockPool/KVCacheManager 与 SGLang 的 RadixCache:前缀收养、引用计数、LRU 逐出
> 5. 定量 50% 前缀复用负载下 TTFT 的改善,并解释"高并发下收益被稀释"
> 6. [进阶]评估 KV 量化 / 选择性保留(H2O、SnapKV)/ offload(LMCache)的适用边界
>
> **定位**:第 3 章回答"KV 要多少"(容量),本章回答"KV 怎么管"(管理)。PagedAttention 是 LLM 推理最重要的数据结构——vLLM 因它得名(Kwon et al., SOSP'23)。

## 5.1 连续 KV 的浪费

PagedAttention 之前,主流框架(早期 Orca、FasterTransformer)把每个请求的 KV 存在 **HBM 连续区域**,请求到达时按 max_len(最大可能上下文)分配:

两类浪费(paper 实测:**60%-80%**):

1. **内部碎片**:按 max_len 分配,实际输出远短 → 未用完部分全程占用(分 8K 只输出 2K → 75% 浪费)
2. **外部碎片**:长短请求混排,HBM 空闲空间打碎——总空闲很大,但没有一块连续的 8K → 新请求被拒

第三类后果最致命:

3. **保守超预留**:为保证"任何请求都能分到位",框架必须留一大块"连续空闲区"余量 → 有效 KV 池小于显存量。**拒绝率由"最大连续空闲区"而非总容量决定——请求越多越碎,容量"越用越小"**(呼应第 3 章:(3-3) 手算的并发上限,连续管理下实际达不到)。

OS 类比(本章核心,必须讲透):

```
OS:        逻辑地址  ──页表──►  物理页 (4KB)
PagedAttn: 逻辑block ──block table──► 物理block (如 16 tokens)
```

- 请求视角:"我的 KV 是 0..N 号逻辑 block,连续的"(模型代码完全不变)
- HBM 视角:物理 block 散落各处,按 block 粒度分配/回收
- block table:逻辑 block → 物理 block 号,每请求一份,KB 级

## 5.2 PagedAttention 算法:attention kernel 怎么改

连续 KV 时,attention kernel 直接对"该请求的连续 K/V"算 Q×Kᵀ(FlashAttention 式)。引入 block 间接后,K/V 在 HBM 上不再连续 → kernel 必须"读一块、算一块、合并":

```python
# PagedAttention decode kernel 核心循环(简化伪代码,说明数据流;
# 生产 kernel 为 CUDA/Triton,block 间并行)
# Q: [B, H, D](decode 每请求 1 个 query token)
# block_table: [B, max_blocks] 逻辑→物理; kv_blocks: [num_blocks, BLOCK, H, D]
O = zeros; m = full(-inf); l = zeros            # online softmax 状态
for b in range(num_blocks(req)):
    phys = block_table[req, b]                  # ① 间接寻址:逻辑 → 物理
    K = kv_blocks[phys]                          # [BLOCK, H, D]
    V = kv_blocks[phys]
    S = (Q @ K.T) / sqrt(D)                      # ② 本块 scores
    m_new = maximum(m, rowmax(S))
    alpha = exp(m - m_new)                       # ③ 修正因子
    P = exp(S - m_new)
    l = alpha * l + rowsum(P)
    O = alpha * O + P @ V
    m = m_new
O = O / l                                        # ④ 输出
```

要点:

- **K/V 从不整体物化,O(N²) 中间量不落 HBM**——这是 FlashAttention 的 IO-aware 思想;PagedAttention 在其上叠加 block 间接(两者正交,可组合)
- online softmax(m、l 状态)使不同 block 的 scores 能"增量合并"——否则得先拿到全部 block 的 S 才能做全局 max,要扫两遍
- **代价**:每 block 一次间接寻址(block table 查表)+ 分块合并开销
- **收益**:HBM 流量不含"浪费字节"(只读用到的 block)+ 容量稳定(无碎片拒绝)
- 实测(paper + 生产经验):**decode 不降反快**(带宽被完全利用),prefill 基本持平
- 净效果:同样显存的稳定并发,从"不可预测"变为 ≈ 第 3 章公式 (3-3) 的上限——**这是手算公式通向生产现实的桥**

**5.2.1 block size:为什么是 16**

```
太小(4/8):   block table 开销 ↑、每 block 的 kernel 间接/合并开销 ↑
             (table 访存 + 逐块 softmax merge)
太大(32/64): 内部碎片回归(平均浪费 ~BLOCK/2 个 token)、
             前缀共享粒度变粗
```

16 是经验最优(paper 默认;vLLM `--block-size` 可选 16/32/64,默认 16,SGLang page_size 同理)。长上下文 + FP8 KV 可尝试更大 block——**测,不要信**(第 2 章纪律)。

> **面试陷阱框:"PagedAttention 多一层间接寻址,不是更慢吗?"**
> 答:间接开销是 O(num_blocks) 次查表(block table 在 L1/L2,ns 级);省掉的是 O(浪费字节) 的 HBM 流量(μs 级)。**两者差数量级,且 decode 瓶颈在 HBM 带宽(Ch1 §1.5)——一切"不读浪费字节"的优化都是净赚**。这套推理模板同样适用于 FlashAttention(不物化 N²)与量化(不读全精度字节)。

## 5.3 Block 池与生命周期(源码级)

vLLM v1:`BlockPool` + `KVCacheManager`(`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block/allocator.py`)。简化核心结构:

```python
# vllm/v1/core/block/allocator.py(简化,说明结构;实际还有多组 KV cache 等状态)
class BlockPool:
    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]  # Block: id, ref_count
        self.free: set[Block] = set(self.blocks)

    def allocate(self, n: int) -> list[Block]:
        out = [self.free.pop() for _ in range(n)]
        for b in out:
            b.ref_count += 1
        return out

    def free(self, blocks):
        for b in blocks:
            b.ref_count -= 1
            if b.ref_count == 0:
                self.free.add(b)               # ① refcount 决定"释放还是保留"
```

```python
# vllm/v1/core/kv_cache_manager.py(简化)
class KVCacheManager:
    def __init__(self, block_pool, block_size=16):
        self.block_pool = block_pool
        self.block_size = block_size
        self.hash_to_block: dict[tuple, Block] = {}   # 前缀哈希 → 物理 block

    def allocate_for_prefill(self, req) -> tuple[list[Block], int]:
        # ① 前缀命中:逐块算哈希、查表,命中的块"收养"
        hit = []
        for i in range(0, len(req.prompt) // self.block_size):
            h = hash_chain(req, i)             # 含父哈希,见下文细节 1
            b = self.hash_to_block.get(h)
            if b is None:
                break
            b.ref_count += 1                   # ② 共享 = 加引用,不复制
            hit.append(b)
        # ③ 未命中部分:新分配
        n_new = math.ceil((len(req.prompt) - len(hit) * self.block_size)
                          / self.block_size)
        new = self.block_pool.allocate(n_new)
        # ④ 命中块对应的 prompt token 不占 prefill 预算(Ch4 §4.2.3 第 3 步)
        return hit + new, len(hit) * self.block_size
```

两个面试易漏的细节:

**细节 1:哈希必须带父链**。只哈希"这 16 个 token 本身",两个完全不同上下文里的相同 16 token 会撞车(错误命中 → KV 内容对不上 → 输出错乱)。所以:

```python
# 哈希链:唯一标识"前 (i+1)*16 个 token"
h_0 = hash(tokens[0:16])
h_i = hash(h_{i-1}, tokens[i*16:(i+1)*16])
```

**细节 2:前缀共享 = 引用计数,不是复制**。多请求共享同一前缀 block:block 只分配一次,ref_count = 共享请求数;某请求结束只减引用,**归零才回 free 池**。这是"50% 前缀复用 → 省一半 prefill"(5.4 定量)的底层机制。

一个完整生命周期(请求 A,512 token,与历史请求共享前 256):

```
1. A 到达:512 token 算哈希链 → 前 16 块命中
2. 16 个命中块 ref +1(收养);其余 16 块新分配
3. prefill 只算后 256 token(预算减半)
4. decode:每填满 1 块,分配新块
5. A 结束:16 个共享块 ref −1(仍被其他请求持有,保留);16 个私有块 → free
6. B(同前缀)到达:16 块再次命中,prefill 0 token,TTFT ≈ 排队 + 首步
```

## 5.4 前缀缓存与 RadixAttention:同一目标的两套数据结构

vLLM 的"定长 block + 哈希表"见 5.3。SGLang 用显式基数树:

```python
# sglang/srt/mem_cache/radix_cache.py(简化;生产实现有批量匹配、
# 节点分裂、LRU 维护等工程细节)
class RadixCache:
    def __init__(self):
        self.root = TreeNode(value=None)        # value: token 序列(block 对齐)
        self.lru = OrderedDict()                # 叶子级 LRU

    def match_prefix(self, key: list[int]):
        node, i = self.root, 0
        while i < len(key):
            nxt = node.children.get(key[i])     # 逐 token 下行(生产为批量匹配)
            if nxt is None:
                break
            node, i = nxt, i + len(nxt.value)
        return node                             # 最长匹配前缀 = node 到 root 的路径

    def cache_finished_req(self, req):
        # 请求结束:整条路径插入树(需要的节点做分裂),ref_count += 1
        # 不释放 —— 这是"复用优先"的关键:结束了也不立刻丢 KV
        insert_path(self.root, req.token_ids)

    def evict(self, num_tokens):
        # LRU 叶子逐出;ref_count == 0 才可逐;可整棵子树逐出
        ...
```

| | vLLM(哈希 block 池) | SGLang(radix tree) |
|---|---|---|
| 数据结构 | 定长 16-token block + 哈希表(带父链) | 变长节点基数树 |
| 命中条件 | 按 block 步长前缀对齐 | 任意长度前缀,天然支持分叉 |
| 逐出 | 单 block LRU | 子树 LRU |
| 甜区负载 | 通用(共享 system prompt) | 多轮对话、few-shot、tree-of-thought/多候选、agent 循环(共享主干、分叉探索;负载画像与成本模型见附录 H) |

两者目标相同:**用复用消灭重复 prefill**;差别是"能表达多细的复用"。SGLang 论文报告:高复用负载下端到端最高 ~2-6×(负载相关,复测为准)。

**定量示例(呼应 Ch2 实验 C,50% 前缀复用)**:

- 基线(无缓存):所有请求完整 prefill
- 50% 复用:TTFT ≈ 排队 + 50% prefill + 首步 → **TTFT 降约 40-60%**(取决于 prefill/排队占比)
  - 低并发:排队小,prefill 主导 → 接近 50%
  - 高并发:排队主导 → 收益被稀释(复用 KV 不会让队列变短)
- **结论(面试追问)**:前缀缓存收益 = "复用率 × 低并发"时最大;"开了前缀缓存 TTFT 没降 50%"在高并发下是正常现象,不是 bug——先分解 TTFT(排队 vs prefill,Ch2 §2.1 的服务端口径)再下结论。

## 5.5 [进阶] KV Cache 进阶:量化 / 选择 / 卸载

**5.5.1 KV 量化(接 Ch8)**

- **FP8 KV**(vLLM `--kv-cache-dtype fp8`):显存减半、并发 ×2(Ch3 公式 (3-3) 直接代入);实测质量损失通常可忽略(attention 路径对低精度的敏感度低于 GEMM 路径)
- **INT4 KV**:再减半,但质量损失明显 → 仅"显存受限 + 质量可容忍"场景
- 经验法则:**FP8 KV 优先(近免费午餐),INT4 靠后**

**5.5.2 选择性保留(计算侧压缩)**

- **H2O**(Heavy-Hitter Oracle):动态保留 attention 权重最大的"heavy hitter" token,其余丢弃 → 动态 KV 预算
- **SnapKV**:prefill 结束时对每 token 打重要性分(attention 和),保 top-k → 固定 KV 预算
- **StreamingLLM**:sliding window + attention sink(前几个 token)
- 适用:长上下文、显存受限、质量可容忍(摘要、RAG);不适用:精确检索/代码(KV 即证据)
- 面试定位:**"KV 压缩是拿质量换容量,必须在自己任务集上评估"**——不要直接搬 paper 数字进生产

**5.5.3 offload 与跨副本复用(接 Ch3.5)**

- swap:被抢占的 KV → CPU(Ch21)
- **LMCache 类**:KV 作为"缓存对象"放分级存储(HBM/DRAM/NVMe/远端),跨副本/跨实例复用 → RAG 场景"同一文档前缀全集群只算一次"
- 定量判据(Ch9 的 KV 传输成本公式直接复用):**offload 值得 ⇔ "KV 传输时间 < prefill 重算时间"**
- 常见误用:把 offload 当"长上下文加速方案"——换入换出本身 200ms 量级,对 TTFT SLO 敏感场景是负优化。先测,再上。

## 动手:前缀缓存 A/B 实验

**实验 5.1(呼应 Ch2 实验 C,60 分钟)**:

```bash
# 组 1:无前缀缓存(基线)
vllm serve qwen2.5-7b --port 8000
# 组 2:开前缀缓存(同硬件、同负载)
vllm serve qwen2.5-7b --port 8001 --enable-prefix-caching

# 负载 C:50% 共享 256-token system prompt,128 请求,QPS=4
# 工具:labs/bench_basics.py 改造,或 vllm bench serve(负载定义见 Ch2 §2.2)
# 指标:TTFT p50/p95、prefix cache 命中率(服务端指标)、吞吐
# 加测:QPS 推到 16,验证"收益被排队稀释"(5.4 的预测)
```

交付物:两组 A/B 数据 + 一句话结论("负载 X 下 TTFT −Y%,命中率 Z%,高并发下收益收窄到 W%")。

**实验 5.2(读码,无需 GPU)**:读 vLLM `kv_cache_manager.py`,回答三问:(a) block refcount 在哪里维护? (b) 逐出决策发生在哪个调用点? (c) `allocate_for_prefill` 与 decode 分配的路径差别?

## 章末自测

1. 连续 KV 的两类浪费 + 为什么"拒绝率由最大连续空闲区决定"
2. PagedAttention 的 kernel 怎么改(分块计算 + online softmax)?为什么 decode 不降反快?
3. block size 为什么 16(双向取舍)?
4. vLLM 哈希前缀缓存与 SGLang radix tree 的差异(数据结构 / 命中条件 / 逐出 / 甜区负载)?
5. FP8 KV 省多少显存、并发升多少、风险在哪?
6. [进阶]H2O / SnapKV 何时有帮助、何时有害?

## 面试钩子

- 支撑 Ch28-B11~B18:"PagedAttention 怎么实现的,代价是什么?""block size 为什么 16?""前缀缓存逐出怎么做?""什么负载下前缀缓存无效?"
- Ch29 案例 2(低延迟 chat)/ 案例 3(1M RAG)的前缀复用设计是得分点。
- Ch25 实验 3(前缀缓存 A/B)即本章动手。
