# 附录 A:公式速查

> 全书编号公式汇总(编号与正文一致;Ch3 中 (3-1) 在"权重"与"KV/token"两处沿用,以上下文为准)。
> 用法:Ch29 案例的"容量估算"段全部从这里出发。**先套公式,再填参数,最后标 [假设]/[实测]。**

## A.1 编号公式

| 编号 | 公式 | 含义 | 出处 |
|---|---|---|---|
| (1-1)/(3-1) | KV/token = 2 × L × h_kv × d_head × bytes | 每 token KV 显存(全模型口径) | Ch1/Ch3 |
| (3-1) | 权重字节 = P_total × bytes | 权重显存 | Ch3 |
| (3-2) | 单请求 KV = (KV/token) × n_ctx | 单请求 KV(n_ctx = prompt + 输出) | Ch3 |
| (3-3) | 每卡可并发 = KV 池(每卡) ÷ 单请求每卡 KV | 并发上限(KV 口径,向下取整) | Ch3 |
| (3-4) | 需求吞吐 = λ × L_out | decode 需求(λ = req/s) | Ch3 |
| (3-5) | TP 组数 = ⌈需求吞吐 ÷ 单组 goodput⌉ × 余量(1.3~1.5);卡数 = 组数 × TP size | 容量 → 卡数 | Ch3 |
| (2-5) | goodput = SLO 约束下的最大可持续输出 tokens/s | 生产目标函数 | Ch2 |
| (12-1) | 每步 CPU 发射开销 ≈ N_kernel × t_launch(3-10μs/kernel) | launch 开销(小 batch 瓶颈) | Ch12 |
| (13-1) | all-to-all 总发送量 ≈ batch × seq × topk × hidden × bytes | EP 每层通信量 | Ch13 |
| (14-1) | 单次 all-reduce 消息 ≈ bs × seq × hidden × bytes;每层 2 次 | TP 通信量 | Ch14 |
| (14-2) | 每 PP 边界 ≈ micro_bs × hidden × bytes;每步 ≈ (stages−1) × 该消息 | PP 通信量 | Ch14 |
| (H-1) | P_k = S + Σ_{i≤k}(Δ_i + R_i);naive 账 = Σ_k P_k,缓存命中账 ≈ Σ 增量;η = 1 − 命中账/naive 账 | agent 多轮会话的累积 prompt 与前缀缓存节省率 | 附录 H §H.1.2 |
| (H-2) | TTFT_k ≈ t_queue + (P_k 冷 / 增量 热) ÷ r_prefill | 会话单轮 TTFT 模型(冷/热 TTFT 比 = 前缀缓存生效代理) | 附录 H §H.2.1 |
| (H-3) | KV/seq = 2 × L × H_kv × d_h × T × b(70B GQA FP16:320KB/token,64K → ~20GiB) | RL 长 CoT 的单序列 KV 与并发上限 | 附录 H §H.4.2 |
| (H-4) | t_sync ≈ W_bytes ÷ b_load(70B FP16 140GB ÷ NVMe 3GB/s ≈ 47s;节点内 NCCL <0.3s) | 权重同步开销(epoch 边界的固定税) | 附录 H §H.4.3 |
| (H-5) | E[尝试] = (1 − p^n)/(1 − p)(p=0.7,n=5 → 2.77) | 自我改进循环的负载乘数(m 后的 λ′) | 附录 H §H.3.1 |

## A.2 常用辅助公式(未编号,正文引用)

| 公式 | 用途 |
|---|---|
| Roofline:time = max(FLOPs / peak_FLOPS, bytes / peak_BW);AI = FLOPs / bytes;ridge = 峰值算力 / 带宽 | Ch1/Ch10 判定 memory/compute-bound |
| prefill FLOPs ≈ 2 × P_active × N_tokens(MoE 用激活参数) | Ch1/Ch17 prefill 时间账 |
| 投机解码净加速 = (1+L)/(1+γ·c_d),L = α(1−α^γ)/(1−α) = 每轮期望接受数(c_d = draft 单 token 成本比;速算上限式 (1+γα)/(1+γ·c_d)) | Ch7 |
| ring all-reduce busbw 系数 = 2(N−1)/N(N=8 → 0.875 × 链路带宽) | Ch18 nccl-tests 判读 |
| KV 传输时间 ≈ KV/请求 ÷ 单链带宽;多链并行 ÷ 链数(GDR zero-copy 前提) | Ch9 |
| $/M token = (卡数 × $/卡时) ÷ (实测 goodput tok/时 ÷ 10⁶) = 卡数 × $/卡时 × 278 ÷ goodput(tok/s)(278 = 10⁶/3600) | Ch19 |
| TP 下每卡 KV = (KV/token) × (h_kv/tp)(kv head 复制时按 max(1, h_kv/tp) 修正) | Ch3/Ch15 |
| CP×TP 下每卡 KV = KV 总量 ÷ (TP × CP) | Ch17 |
| 热 expert 步放大 ≈ 1 + 0.4×(M_e 偏斜比 − 1)(MoE GEMM 占 decode 步 ~40%,量级) | Ch13/Ch29 案例 8 |
| N+1 规则:N = ⌈D/(0.8C)⌉ + 1(C = 单副本容量,D = 峰值需求) | Ch19/Ch29 案例 9 |

## A.3 常代参数(70B GQA 示例,FP16)

| 参数 | 值 |
|---|---|
| L / h_kv / d_head | 80 / 8 / 128 |
| KV/token(全模型) | 2×80×8×128×2B = 320KB;TP8 → 40KB/卡 |
| 权重 | 140GB(FP16)/ 70GB(FP8) |
| 8K 请求 KV | 2.56GB(全模型)/ 0.32GB/卡(TP8) |
| 每步 decode 步时(TP8,量级) | ~6-8ms(Ch15 §15.2 下界 + [估算] 典型值) |

硬件常量见附录 B;235B MoE / 671B MLA 的参数见 Ch29 案例 1/3 的需求段。
