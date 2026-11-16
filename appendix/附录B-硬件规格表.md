# 附录 B:硬件规格表

> 口径:GPU 单卡规格(稠密算力,非稀疏)。来源 = NVIDIA 官方 datasheet/发布公告。
> **✅ B200 行已于 2026-11-16 复核**(来源:NVIDIA Blackwell datasheet,HGX B200 SXM;第三方流传的"192GB/8TB/s"为 GB200 NVL72 或营销口径,未采用)。书中所有引用 B200 的表述("7.7TB/s""FP4 窗口期"等)以本表为准。
> 网络列是"常见节点配置"(非 GPU datasheet 内容),随采购配置变化。

| 参数 | A100 SXM 80GB | H100 SXM 80GB | H200 SXM 141GB | B200 SXM(2026-11-16 复核) |
|---|---|---|---|---|
| 架构 / 发布 | Ampere,2020 | Hopper,2022 | Hopper,2024 | Blackwell,2024-2025 |
| HBM 容量 | 80GB | 80GB | 141GB | 180GB |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s(+44%) | 7.7 TB/s |
| FP16/BF16 稠密 | 312 TFLOPS | 989 TFLOPS | 989 TFLOPS(同 H100) | 2.25 PFLOPS |
| FP8 稠密 | 不支持(无 FP8 TC) | 1979 TFLOPS | 1979 TFLOPS | 4.5 PFLOPS |
| FP4 稠密 | — | — | — | 9 PFLOPS |
| NVLink | 3 代,600GB/s | 4 代,900GB/s | 4 代,900GB/s | 5 代,1.8TB/s |
| NVSwitch 域 | 8 卡 | 8 卡 | 8 卡 | 8 卡(单机 1 NVSwitch 量级) |
| 常见 IB 配置 | 4-8×200Gb | 8×400Gb(NDR400) | 8×400Gb | 8×800Gb(XDR) |
| 内存(系统)常见 | 1-2TB | 1-2TB | 1.5-2.5TB | 2-4TB |

## B.1 书中引用口径对照(避免混淆)

- "H100 FP16 ridge ≈ 989/3.35 ≈ 295 FLOP/byte"(Ch1/Ch10 A3):用本表 FP16 稠密 ÷ HBM 带宽。
- "H200 decode +30-40%、KV 并发 ×1.76"(Ch19 §19.1):= 带宽 +44% 打折(attention/通信段不受益)× 显存 +76%。
- "B200 MoE 权重 1/4、decode ×2"(Ch8/Ch28 C26):= FP4 权重字节 ÷4 + 带宽 7.7TB/s,相对 FP8 H100 基线;生态窗口期结论见 Ch12 §12.4 决策清单。
- **H800(中国特供变体,Ch10 §10.3/Ch18 §18.1 的坑)**:H100 计算规格不变,**NVLink 降速至 400GB/s** → 节点内 all-reduce 时间 ≈ H100 的 2.25×;采购与验收必须 `nvidia-smi topo -m` 实测,不认型号铭牌。

## B.2 使用注意

1. 表中算力为**稠密(dense)**;稀疏(2:4)规格为稠密 2×,推理框架默认不用。
2. 实测有效算力 ≠ 峰值:MFU 40-60%(prefill)与 decode 的带宽 roof 才是规划口径(Ch10 §10.2 的 roofline 两条线)。
3. 同一型号不同 SKU(SXM/PCIe/HBM 容量变体)带宽差 20-40%,**采购单必须落到具体型号**(如 H100 SXM5 80GB 而非"H100")。
4. **复核日志**:B200 全行 2026-11-16 已复核(NVIDIA Blackwell datasheet,HGX B200 SXM:180GB HBM3e / 7.7TB/s;稠密 FP16 2.25PF / FP8 4.5PF / FP4 9PF;NVLink5 1.8TB/s;H200 系统内存列与各型号 NVSwitch 代际随采购单核对,见第 3 条)。
