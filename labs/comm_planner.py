"""通信量计算器:输入模型配置与并行配置,输出每步通信量与延迟量级。
对应书第 14 章 §14.2(公式 14-1/14-2)与第 13 章 (13-1),Ch26 实验复用。
用法:
  python comm_planner.py --hidden 8192 --layers 80 --tp 8 --bs 1
  python comm_planner.py --hidden 7168 --topk 8 --ep 8 --batch 64   # ep>0 自动走 MoE 路径(all-to-all)
单位:bytes/step;延迟按 NVLink 900GB/s 与 IB 50GB/s 两个量级给出。
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=80, help="层数(PP/TP 用)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1, help="PP 阶段数")
    ap.add_argument("--ep", type=int, default=0, help="EP 规模(>0 时按 MoE 算)")
    ap.add_argument("--topk", type=int, default=8, help="MoE top-k")
    ap.add_argument("--batch", type=int, default=1, help="decode batch(token 数)")
    ap.add_argument("--bytes", type=int, default=2, help="元素字节数(1=FP8, 2=FP16)")
    a = ap.parse_args()

    NVLINK, IB = 900e9, 50e9  # B/s

    def report(name, total_bytes, per_op, link_bw):
        t_us = total_bytes / link_bw * 1e6
        print(f"  {name}: 总量 {total_bytes/1e6:8.2f} MB | 单 op {per_op/1e6:6.3f} MB"
              f" | NVLink {t_us:7.1f} μs | IB     {total_bytes/IB*1e6:7.1f} μs")

    print(f"== 通信量估算 (hidden={a.hidden}, layers={a.layers}, "
          f"tp={a.tp}, pp={a.pp}, ep={a.ep}, batch={a.batch}, {a.bytes}B/元素) ==")

    if a.tp > 1:
        per_op = a.batch * a.hidden * a.bytes          # 一次 all-reduce 的消息量(decode, seq=1)
        total = a.layers * 2 * per_op                  # 每层 2 次 × 层数
        report(f"TP{a.tp} all-reduce ×{a.layers*2}", total, per_op, NVLINK)
        print(f"  注:all-reduce 实际耗时 ≈ ring 算法 2(N-1)/N 系数 × 上述值;小消息下延迟地板(μs 级)主导")
    if a.pp > 1:
        per_op = a.batch * a.hidden * a.bytes
        total = (a.pp - 1) * per_op                    # 每步跨 (pp-1) 个边界
        report(f"PP{a.pp} point-to-point", total, per_op, IB)
        print(f"  注:瓶颈是 {a.pp-1} 跳串行延迟(每跳 μs 级),不是带宽;单请求时还叠加 bubble")
    if a.ep > 0:
        total = a.batch * a.topk * a.hidden * a.bytes  # (13-1) 总发送量(单方向, 单层)
        per_peer = total / (a.ep - 1)
        report(f"EP{a.ep} all-to-all (topk={a.topk}, 单层)", total, per_peer, IB)
        print(f"  注:decode 下每对端消息小 → 延迟主导;prefill(大 batch×seq)才进入带宽主导")

    print("\n判读口诀:TP 看 NVLink(小消息×多次×低延迟),EP 看消息延迟(decode)与带宽(prefill),"
          "PP 看跳数延迟,DP 零通信。")


if __name__ == "__main__":
    main()
