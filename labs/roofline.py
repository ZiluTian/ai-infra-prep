"""Roofline:画机器 roofline 并标出给定 kernel 的 AI 位置。
对应书第 10 章 §10.2。
用法:
  python roofline.py --peak 989 --bw 3.35 --ai 1 --name "decode GEMV (batch=1)"
  python roofline.py --peak 989 --bw 3.35 --ai 4096 --name "prefill N=2048"
参数: peak=TFLOPS(稠密峰值), bw=TB/s(HBM 带宽), ai=FLOP/byte
理论: roof = min(peak, AI×BW);ridge point = peak / BW
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", type=float, default=989.0, help="稠密峰值 TFLOPS")
    ap.add_argument("--bw", type=float, default=3.35, help="HBM 带宽 TB/s")
    ap.add_argument("--ai", type=float, default=1.0, help="算术强度 FLOP/byte")
    ap.add_argument("--name", default="kernel")
    a = ap.parse_args()

    ridge = a.peak / a.bw
    attainable = min(a.peak, a.ai * a.bw)
    bound = "compute" if a.ai >= ridge else "memory"

    ai = np.logspace(-2, 4, 200)
    roof = np.minimum(a.peak, ai * a.bw)
    plt.figure(figsize=(7, 5))
    plt.loglog(ai, ai * a.bw, "b--", label="BW 线")
    plt.loglog(ai, roof, "b-", lw=2, label="roofline")
    plt.axhline(a.peak, color="b", ls=":", alpha=0.5)
    plt.loglog([a.ai], [attainable], "ro", ms=10)
    plt.annotate(f"{a.name}\nAI={a.ai:g} → {bound}-bound\n{attainable:.1f} TFLOPS",
                 (a.ai, attainable), textcoords="offset points", xytext=(15, 10))
    plt.xlabel("arithmetic intensity (FLOP/byte)")
    plt.ylabel("attainable TFLOPS")
    plt.title(f"ridge = {ridge:.1f} FLOP/byte (peak {a.peak} TFLOPS, BW {a.bw} TB/s)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"ridge={ridge:.1f}, kernel AI={a.ai:g} → {bound}-bound,"
          f" 可达 {attainable:.1f} TFLOPS(峰值的 {attainable / a.peak * 100:.0f}%)")


if __name__ == "__main__":
    main()
