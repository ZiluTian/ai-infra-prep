# 第 12 章 Triton、CUTLASS 与 GEMM

> 难度 ★★★★★ | 目标 ~30 页 | 前置:Ch10
> 分层:12.1 [基础];12.2-12.4 [进阶](整章面向资深读者,路线 A 读者至少读 12.1 + 12.3)。
>
> **读完本章你能做什么**
> 1. [基础] 用 Triton 写出 vector add → 融合 RMSNorm → matmul → grouped matmul,理解 tile 思维 / 自动 pipelining / autotune
> 2. [进阶] 画出 Hopper GEMM 数据通路(TMA → SMEM → WGMMA → 寄存器 → epilogue),说出 CUTLASS Collective Builder 的三层结构
> 3. [进阶] 解释 DeepGEMM 被 DeepSeek 推理选中的三个理由(JIT / TMA multicast / per-block scale)
> 4. [进阶] 推导小 batch decode 为何 launch-bound,讲清 CUDA Graph 的原理、分桶捕获与动态 shape 限制
> 5. [进阶] 用决策清单判断"这个 op 该不该自己写 kernel",并说出常见坑(bank conflict / L2 / occupancy)

## 12.1 [基础] Triton 编程模型

**与 CUDA 的思维方式差别**:CUDA 是**线程级**思维(每线程算几个元素、手动管理 SMEM 与流水);Triton 是 **tile 级**思维——一个 program 负责一块(tile)数据,布局、寄存器分配、软件流水全部交给编译器:

- `tl.program_id` → 本 program 处理哪个 tile
- `tl.load/tl.store` 是向量的,自动加 mask
- `tl.dot` → 映射到 tensor core MMA 指令(编译器替你选 layout)
- **自动 pipelining**:K 方向循环的 load 被编译器做 double/triple buffering(HBM 延迟被流水掩盖),`num_stages` 控制深度
- **autotune**:同一 kernel 多个 config(BLOCK 尺寸、num_warps、num_stages),按 `key` 参数实测选最快的

**性能定位**:简单/融合类 op、GEMM 变体(grouped/scaled)上 Triton 可达手写 CUDA 的 90-100%;FA3 级别的极致 kernel(需要精确控制 TMA/WGMMA/warp 分工)仍需手写 CUDA + CUTLASS。

### 递进教程(四个 kernel)

**Step 0:vector add(感受 tile 思维)**

```python
import torch, triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                    # 本 program 管第 pid 块
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # 块内偏移
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)

x, y = torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")
o = torch.empty_like(x)                       # 输出缓冲区(kernel 写入 o_ptr)
add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
```

读法:整行代码就是"第 pid 块 → 取 2 块 → 加 → 写回"。没有线程号、没有 `__shared__`,布局由编译器决定。

**Step 1:融合 RMSNorm(LLM 每层两次的核心 op)**

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, o_ptr, R: tl.constexpr, eps):
    row = tl.program_id(0)                  # 一个 program 处理一整行(hidden 维)
    offs = tl.arange(0, R)                  # R=4096/7168,一行 = 一个 tile
    x = tl.load(x_ptr + row * R + offs).to(tl.float32)
    w = tl.load(w_ptr + offs)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / R + eps)   # 整行在寄存器内完成
    tl.store(o_ptr + row * R + offs, (y * w).to(x.dtype))
```

**融合收益的账(面试要会算)**:naive 两遍(先算均方、再归一化乘 w)→ 2 次读 + 2 次写;融合版 → **1 读 1 写**,中间量全程在寄存器。hidden=7168 时,每行省 2 次 HBM 搬运 × 7168 × 2B ≈ 28KB——一行 14KB 的数据量下,这等于把 HBM 流量降到 ~1/2。

**Step 2:matmul(GEMM 的最小教学形态)**

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)      # 累加用 FP32
    for k in range(0, K, BK):                        # 编译器自动 pipelining
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])   # [BM, BK]
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])   # [BK, BN]
        acc += tl.dot(a, b)                          # → tensor core MMA
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty))
```

与"真正的 GEMM"的差距(面试加分点):
1. **L2 swizzle**:pid 映射按 group 化(相邻 program 共享 B 的 tile),L2 命中率从 ~1 次 → 数十次
2. **split-K**:M/N 小、K 大时切 K 维并行,再 atomic/归约
3. **layout**:B(权重)离线转置成 TN 布局,避免运行时转置
4. **autotune**:

```python
@triton.autotune(
    configs=[triton.Config({"BM":128,"BN":128,"BK":64}, num_stages=3, num_warps=8),
             triton.Config({"BM":64,"BN":256,"BK":64},  num_stages=4, num_warps=8)],
    key=["M","N","K"])
@triton.jit
def matmul_kernel(...): ...
```

**Step 3:grouped matmul(MoE 专用,接 Ch13)**

问题:一个 MoE 层有 E 个 expert,每个 expert 的 GEMM 形状 `(M_e, N) × (N, K)`,`M_e` 各不相同(且很多为 0)。

```python
# vLLM fused_moe 的结构(简化):
# 前置:token 按 expert 排序 → A_perm (M_total, K);offsets[e] = expert e 的行偏移
# grid = (cdiv(M_total, BM), E, cdiv(N, BN))
@triton.jit
def grouped_gemm(a_p, w, c, offsets, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, e, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_e = tl.load(offsets + e + 1) - tl.load(offsets + e)   # expert e 的 token 数
    if pid_m * BM >= m_e:                                    # 越界 tile → 直接跳过
        return
    m0 = tl.load(offsets + e) + pid_m * BM                   # 行起始(permute 后)
    rm = m0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_p + rm[:, None] * K + (k + tl.arange(0, BK))[None, :])
        b = tl.load(w + e * N * K + rn[:, None] * K + (k + tl.arange(0, BK))[None, :])
        acc += tl.dot(a, b)
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(c.dtype.element_ty))
```

三个结构点:**① permute 前置**(token 按 expert 连续,组内就是普通 GEMM);**② 3D grid 的 e 维**(每个 expert 的 tile 独立调度);**③ 零 token expert 的 `return`**(它的所有 tile 都越界 → 整组 program 空转即退出,不占 SM)。CUTLASS 的 grouped GEMM 用 device 端 tile scheduler 做同一件事。

## 12.2 [进阶] GEMM 为王:Tensor Core 与三个 GEMM 库

**算账**:LLM forward 中 GEMM(attention 的 QKV/O 投影 + MLP)+ attention 是大头;GEMM kernel 的效率直接决定 prefill 的 MFU(Ch10)。**GEMM 是"为王"的:它同时吃算力和带宽两种 roof。**

**Hopper 上的 GEMM 数据通路(必须会画)**:

```
HBM ──TMA 异步批量拷贝──► SMEM ──WGMMA(异步,warp-group)──► 寄存器(累加)
    ▲                                              │
    └──TMA store◄── SMEM ◄── epilogue(scale/激活/转类型)◄┘
```

- **TMA**:硬件 DMA,免 warp 参与;支持 **multicast**(一份 SMEM 数据广播给多个 SM)
- **WGMMA**:4 个 warp 组成的 warpgroup 异步执行 MMA,FP8 吞吐 2× FP16(Ch8 的指令级根据);MMA 与下一 tile 的 TMA 搬运**重叠**
- **warp specialization**:producer warp 专职 TMA 搬运,consumer warp 专职 WGMMA/epilogue——FA3 同款软件流水
- **layout(TN/NT)**:decode M=1 走 GEMV 布局,prefill 走 TN 权重转置布局;vLLM/SGLang 在加载权重时**离线**转好,运行时不做 layout 变换

**CUTLASS(NVIDIA 官方模板库,3.x)**:

- **Collective Builder 三层结构**:`TileScheduler`(怎么分 tile)× `Mainloop`(TMA + WGMMA 的流水)× `Epilogue`(scale/激活/类型转换/写回)
- "写一个 FP8 per-block scale GEMM" = 选 TileShape(如 128×128×128)+ 选 scale 模式(每 128×128 块一个 FP32 scale,在 epilogue 对 FP32 累加值解标)+ build 出 kernel;性能贴近库上限
- 代价:模板实例化是**编译期**的——shape 组合爆炸时编译时间失控(引出 DeepGEMM)

**DeepGEMM(DeepSeek 开源,FP8 GEMM,V3/R1 推理在用)**——为什么不用 cuBLAS:

1. **JIT**:运行时按 `(M,N,K)` 生成特化 kernel。prefill 的 M = chunk × batch 动态变化,JIT 让每个 M 都拿到峰值 tile 配置;CUTLASS 要在编译期枚举 M
2. **TMA multicast**:一份 HBM tile 同时载入同一 GPU 内多个 CTA 的 SMEM(多 SM 共享一次搬运,省 HBM 读流量)——**单 GPU 内机制,不是跨 rank / 跨卡**(跨 rank 的权重共享靠 TP 切分本身,Ch15)
3. **per-block FP8 scale**:cuBLAS FP8 GEMM 支持 per-tensor scale,不支持 per-block;而 **per-block 是 FP8 量化的精度关键**(Ch8 §8.1:outlier 块需要独立 scale,per-tensor 一个数管全部 → 精度崩)。这一条单独就决定了选型。

**选型速查表**:

| 场景 | 选择 | 理由 |
|---|---|---|
| prefill 大 M,FP16/BF16 | cuBLAS / CUTLASS | 库已贴 roof |
| prefill 大 M,FP8 | DeepGEMM / CUTLASS | per-block scale + 动态 M |
| decode 小 M | GEMV kernel(FlashInfer / SGLang) | M<16 时 GEMM 不划算 |
| MoE grouped | CUTLASS grouped / vLLM `fused_moe`(Triton) | 12.1 Step 3 |
| 新精度(FP4/新布局) | Triton / CUTLASS 首发 | 库滞后,自写窗口期 |

## 12.3 [进阶] Kernel launch 与 CUDA Graph

**问题:小 batch decode 是 launch-bound**

```
(12-1) 每步 CPU 发射开销 ≈ N_kernel × t_launch(3-10 μs/kernel)
```

手算:7B 模型 forward ≈ 400 个 kernel(每层:QKV GEMM、attention、O GEMM、2× MLP GEMM、RMSNorm×2、残差……)→ 400 × 5μs ≈ **2ms 纯发射**,而 GPU 实际算 3-5ms → CPU 发射跟不上 GPU 执行,timeline 出现大量 gap。**batch=1 时 decode step 的真实瓶颈常是 CPU,不是 GPU。** 三个对策:

1. **融合 kernel**(减少 N_kernel:12.1 的 RMSNorm 融合、fused_moe)
2. **CUDA Graph**(整图一次发射)
3. **异步调度**(vLLM v1 双进程模型:CPU 排下一步时 GPU 跑这一步,Ch4)

**CUDA Graph 原理**:把一段 kernel 发射序列(含参数与依赖)捕获成 DAG;**replay = 一次 driver 调用**,GPU 按图执行,省去每个 kernel 的 CPU 发射路径。

**核心限制:静态 shape**。图绑死了张量形状与指针 → 变 batch 就得重新捕获。工程解法(vLLM/SGLang 同款):

1. **按 batch size 分桶捕获**:capture batch = 1,2,4,8,…,512 各一张图;运行时 pad 到最近的桶,用对应图
2. **KV 用间接寻址**:图里记的是"从 block table 这个地址读块号",不是"读 123 号块"——block table 内容每步可变,图仍复用(Ch5 的 paged 布局天然适配)
3. **capture 成本**:每桶首次 ~秒级(启动期一次性)+ 每桶独占中间 buffer(显存冗余);**动态控制流(依赖 batch 内容的分支)不可捕获**,必须走静态形态

效果(量级,实测依负载):小 batch 低负载区,step 的 CPU 侧开销从 ~1-2ms 降到 ~0.1ms,端到端 +20-50%。

> **踩坑框:"CUDA Graph 万能加速"**——三个真实成本:① capture 时间与启动延迟;② 每桶显存冗余;③ 形状必须走 padding,极不规则的 ragged 场景(变长序列 prefill)收益有限——**CUDA Graph 的主场是 decode,不是 prefill**。

## 12.4 [进阶] 什么时候该自己写 kernel

**该写(满足其一)**:

1. **融合**:相邻 op 的中间量落 HBM(rmsnorm+residual、SiLU×gate、rope+quant)——收益 = 省下的中间字节,用 roofline 先算值不值(Ch10)
2. **布局特化**:paged KV / ragged / grouped MoE——通用库没有对应 API(FlashInfer、fused_moe 本身就是"自写"的产物)
3. **低精度/新格式**:FP8 per-block、FP4——库滞后,JIT 或 Triton 有窗口期
4. **自定义 epilogue**:GEMM 后直接 bias/激活/残差/量化,省一次 HBM 往返

**不该写**:大 M GEMM(cuBLAS 贴 roof)、标准 attention(FA/FlashInfer 已是上限,除非 MLA 这类特化形状)、**timeline 占比 <5% 的 op**(收益撑不起维护成本)。

**常见坑**:

- **SMEM bank conflict**:32 个线程同 bank 同列访问 → 串行化;解法是 padding/swizzle(ncu 的 `smsp__sass_data_bank_conflicts` 看)
- **L2 locality**:pid 映射顺序决定 B 矩阵 tile 的复用次数;group swizzle 是标配
- **occupancy 权衡**:每线程寄存器↑ → 驻留 warp 数↓;SMEM 超配同样压 occupancy。ncu 报告直接给 "occupancy limited by registers/smem",别凭感觉加 `__launch_bounds__`
- **"单 kernel 快 ≠ 端到端快"**:kernel benchmark 赢了,但引入 layout 转换/额外 HBM 往返,nsys timeline 整体更差——**验收口径永远是 timeline,不是 kernel 微基准**

> **面试模板:"你写过什么 kernel?收益多少?"**
> 结构:瓶颈定位(哪个 op、timeline 占比、roofline 判 bound)→ 选型(Triton 还是 CUDA,为什么)→ 收益(X→Y,用 ncu 的 SOL 验证)→ 成本(维护、shape 覆盖、capture/分桶等工程代价)。四段缺一不可——只说"快了 20%"是最常见的减分答案。

## 动手

**实验 12.1(半天,有 GPU)**:fused RMSNorm + SiLU×Mul 的 Triton kernel:

1. 写 12.1 Step 1 的 `rmsnorm_kernel`,与 PyTorch `F.rms_norm` 逐元素对比正确性 + 有效带宽(GB/s)
2. 写 MLP gating 的 `silu(x@gate) * (x@up)`:先逐算子版,再融合版,测 HBM 有效带宽
3. 接入一个 7B 的 forward(替换对应 op),测 decode step 时间变化
4. 交付物:表格(版本 / kernel 数 / 有效带宽 / step 时间 / ncu Memory SOL)

**实验 12.2(读码,1 小时)**:在 vLLM v1 源码里找 CUDA graph 的捕获逻辑(分桶大小列表、block table 的间接寻址、capture 发生在哪一步),回答:① 为什么不同请求能用同一张图?② prefill 为什么不进图?

## 章末自测

1. Triton 与 CUDA 的"思维单位"各是什么?自动 pipelining 与 autotune 分别解决什么?
2. 算:RMSNorm 融合相对 naive 两遍版(2 读 + 2 写),HBM 流量降为几分之一?(提示:融合版 1 读 1 写)
3. [进阶]画出 Hopper GEMM 数据通路;WGMMA 的"异步"具体与什么重叠?
4. [进阶]DeepGEMM 的三个理由;per-block FP8 scale 为什么是量化精度关键?
5. 为什么小 batch decode 是 launch-bound?三个对策各有什么代价?
6. [进阶]CUDA Graph 如何做到"同一张图处理不同请求"?它的三个真实成本?
7. 一个 op 占 timeline 8%,纯 elementwise,ncu 显示 Memory SOL 95%——该写 kernel 吗?为什么?

## 面试钩子

- 支撑 Ch28-C09~C16:"Triton GEMM 怎么写?""CUTLASS FP8 GEMM 的 scale 在哪解?""DeepGEMM 的 TMA multicast 在什么场景起作用?""CUDA Graph 的本质与限制?""launch 开销怎么消?"
- Ch29 案例 2(低延迟 chat:TTFT/TPOT 双 SLO)的算子侧三件套:融合 + CUDA Graph 分桶 + FP8。
