"""容量规划计算器:模型配置 + 流量 -> 并发上限与卡数。
公式链对应书第 3 章 (3-1)~(3-5):
  (3-1) 权重 = P_total * bytes
  (3-2) KV/token = 2 * L * h_kv * d_head * kv_bytes
  (3-3) 并发 = KV池(每卡) / 单请求每卡KV
  (3-4) 需求吞吐 = lambda * L_out
  (3-5) 组数 = ceil(需求 / 单组goodput) * 余量

用法:
  python capacity_planner.py --preset llama3-70b --ctx 32768 --lam 20 --lout 512 --goodput 2600
  python capacity_planner.py --preset ds-v3-671b --kv-bytes 1 --tp 8
说明: goodput 必须用第 2 章方法实测;不传则只输出显存侧结论(并发上限)。
"""
import argparse
import math
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    P_total_B: float   # 总参数(B);显存看总量
    P_active_B: float  # 激活参数(B);算力看激活
    L: int             # 层数
    h_kv: int          # KV heads
    d_head: int


MODELS = {
    "llama3-8b": Model("Llama-3-8B", 8, 8, 32, 4, 128),
    "llama3-70b": Model("Llama-3-70B", 70, 70, 80, 8, 128),
    "ds-v3-671b": Model("DeepSeek-V3", 671, 37, 61, 128, 128),
}


def plan(preset: str, vram_gb: float, tp: int, weight_bytes: float, kv_bytes: float,
         ctx: int, lam: float, lout: float, goodput: float | None, margin: float) -> None:
    m = MODELS[preset]
    # (3-2) 权重(每卡,TP 切分)
    w_per_card = m.P_total_B * weight_bytes / tp
    overhead = 2.0  # CUDA context / NCCL buffer / 碎片余量(保守估计)
    kv_pool = 0.9 * vram_gb - w_per_card - overhead
    print(f"模型: {m.name} (总参 {m.P_total_B}B / 激活 {m.P_active_B}B, L={m.L}, h_kv={m.h_kv})")
    print(f"权重每卡: {w_per_card:.1f} GB (weight_bytes={weight_bytes})")
    if kv_pool <= 0:
        print("!! 权重装不下:增大 TP 或量化。KV 池为负,无法继续。")
        return
    print(f"KV 池每卡: {kv_pool:.1f} GB (util=0.9, 扣权重与开销 {overhead}GB)")

    # (3-3) KV 并发上限:TP 下每卡承担 1/tp 的 KV heads
    kv_token_card_gb = 2 * m.L * (m.h_kv / tp) * m.d_head * kv_bytes / 1e9
    per_req_card_gb = kv_token_card_gb * ctx
    conc = int(kv_pool / per_req_card_gb)
    print(f"KV/token(每卡): {kv_token_card_gb*1e6:.0f} KB")
    print(f"单请求 {ctx} ctx 每卡: {per_req_card_gb:.2f} GB")
    print(f"并发上限(每 TP 组): {conc} 个请求 (每卡 KV 视角)")

    # (3-4)/(3-5) 流量 -> 卡数
    demand = lam * lout
    print(f"需求吞吐: {lam} req/s x {lout} tok = {demand:,.0f} tokens/s")
    if goodput:
        groups = math.ceil(demand / goodput)
        groups_m = math.ceil(groups * margin)
        print(f"单组 goodput(实测): {goodput:,.0f} tokens/s")
        print(f"TP 组数: {groups} (x 余量 {margin} -> {groups_m});卡数: {groups_m * tp} x {vram_gb:.0f}GB")
    else:
        print("(未提供实测 goodput,跳过卡数规划;请按第 2 章 goodput 曲线取 knee 点)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="llama3-70b", choices=MODELS)
    ap.add_argument("--vram", type=float, default=80.0, help="单卡显存 GB")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--weight-bytes", type=float, default=2.0, help="2=BF16 1=FP8 0.5=INT4(另加~3%%scale)")
    ap.add_argument("--kv-bytes", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--lam", type=float, default=20.0, help="请求到达率 req/s")
    ap.add_argument("--lout", type=float, default=512, help="平均输出长度")
    ap.add_argument("--goodput", type=float, default=None, help="单 TP 组实测 goodput tokens/s")
    ap.add_argument("--margin", type=float, default=1.4)
    a = ap.parse_args()
    plan(a.preset, a.vram, a.tp, a.weight_bytes, a.kv_bytes, a.ctx, a.lam, a.lout, a.goodput, a.margin)


if __name__ == "__main__":
    main()
