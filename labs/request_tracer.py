"""Ch20 L2 分段插桩:per-request 时间戳(t0/首 token/末 token)+ TTFT 三段分解。
bench_basics.py 回答"什么形态"(Ch1/Ch2),本脚本回答"时间花在哪一段"(Ch20 §20.1 L2 层)。

用法:
  python request_tracer.py --base-url http://localhost:8000/v1 --model qwen2.5-7b --n 32 --concurrency 8
  # 服务端若暴露 Prometheus 端点(vLLM 默认 :8000/metrics),加:
  python request_tracer.py --base-url http://localhost:8000/v1 --model qwen2.5-7b --metrics http://localhost:8000/metrics

输出:
  1) 每请求 t0/首/末 时间戳 + TTFT/TPOT/E2E
  2) TTFT/TPOT/ITL 的 p50/p95/p99(Ch2 的分位口径)
  3) 接 --metrics 时:服务端 排队/prefill/decode 段均值 -> TTFT ≈ 排队 + prefill + 网络(三段分解)
注意:prompt 用重复词填充,长度是"近似" token 数;正式实验请用固定语料(Ch2 §2.3)。
客户端时间戳看不到服务端排队——这是必须接 --metrics 的原因;指标名随 vLLM 版本略有漂移,缺失时脚本会明确提示。
"""
import argparse
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# vLLM Prometheus 指标(直方图取 _sum/_count 算均值;名字随版本可能微调,缺失时按提示降级)
METRICS = (
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_gap_seconds",
)


def pct(values, q):
    if not values:
        return float("nan")
    vs = sorted(values)
    return vs[min(int(q * (len(vs) - 1)), len(vs) - 1)]


def one_request(client: OpenAI, model: str, prompt_len: int, max_tokens: int) -> dict:
    prompt = " ".join(["token"] * max(1, prompt_len // 2))  # 近似长度控制
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    t_first, t_last, n_out, itls = None, None, 0, []
    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            now = time.perf_counter()
            n_out += 1
            if t_first is None:
                t_first = now
            else:
                itls.append(now - t_last)
            t_last = now
    if t_first is None:
        raise RuntimeError("未收到任何 token;检查 --model 与服务端日志")
    return {
        "t0": t0, "t_first": t_first, "t_last": t_last, "n_out": n_out,
        "ttft": t_first - t0,
        "tpot": (t_last - t_first) / max(n_out - 1, 1),
        "itls": itls,
        "e2e": t_last - t0,
    }


def fetch_metrics(url: str) -> dict:
    """从 Prometheus 端点抓目标直方图的 _sum/_count,返回 {name: (sum, count)}。"""
    with urllib.request.urlopen(url, timeout=10) as r:
        text = r.read().decode("utf-8", errors="replace")
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, raw = parts[0].strip(), parts[1].strip()
        base_full = name.split("{")[0]  # 去掉 {label=...} 标签再判后缀
        for suffix in ("_sum", "_count"):
            if base_full.endswith(suffix):
                base = base_full[: -len(suffix)]
                if base in METRICS:
                    try:
                        v = float(raw)
                    except ValueError:
                        v = 0.0
                    s, c = out.get(base, (0.0, 0.0))
                    if suffix == "_sum":
                        out[base] = (s + v, c)
                    else:
                        out[base] = (s, c + v)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=32, help="总请求数")
    ap.add_argument("--concurrency", type=int, default=8, help="并发数(排队段的制造器)")
    ap.add_argument("--prompt-len", type=int, default=1024, help="近似 prompt token 数")
    ap.add_argument("--out-len", type=int, default=256, help="max_tokens")
    ap.add_argument("--metrics", default=None,
                    help="服务端 Prometheus 端点,如 http://localhost:8000/metrics")
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="EMPTY")

    m0 = fetch_metrics(a.metrics) if a.metrics else {}
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        results = list(ex.map(lambda _: one_request(client, a.model,
                                                    a.prompt_len, a.out_len), range(a.n)))
    wall = time.time() - t_start
    m1 = fetch_metrics(a.metrics) if a.metrics else {}

    # 1) 每请求
    print(f"n={a.n} concurrency={a.concurrency} prompt≈{a.prompt_len} out_max={a.out_len}  "
          f"wall={wall:.1f}s 吞吐≈{sum(r['n_out'] for r in results)/wall:.0f} tok/s")
    print(f"{'#':>3} {'n_out':>6} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'E2E(s)':>8}")
    for i, r in enumerate(results, 1):
        print(f"{i:>3} {r['n_out']:>6} {r['ttft']*1000:>10.1f}"
              f" {r['tpot']*1000:>10.2f} {r['e2e']:>8.2f}")

    # 2) 分位(Ch2 §2.3:分位 > 均值)
    ttfts = [r["ttft"] for r in results]
    tpots = [r["tpot"] for r in results]
    itls = [x for r in results for x in r["itls"]]
    print(f"TTFT  p50/p95/p99 = {pct(ttfts,0.5)*1000:7.0f}/{pct(ttfts,0.95)*1000:7.0f}/"
          f"{pct(ttfts,0.99)*1000:7.0f} ms")
    print(f"TPOT  p50/p95/p99 = {pct(tpots,0.5)*1000:7.2f}/{pct(tpots,0.95)*1000:7.2f}/"
          f"{pct(tpots,0.99)*1000:7.2f} ms")
    print(f"ITL   p50/p95/p99 = {pct(itls,0.5)*1000:7.2f}/{pct(itls,0.95)*1000:7.2f}/"
          f"{pct(itls,0.99)*1000:7.2f} ms")

    # 3) 服务端分段(三段分解)
    if not a.metrics:
        print("\n(未接 --metrics:只能看客户端 TTFT/TPOT;排队/prefill 分解需要服务端指标)")
        return
    print("\n服务端分段(Prometheus Δ,均值):")
    seg = {}
    for name in METRICS:
        if name in m0 and name in m1 and m1[name][1] > m0[name][1]:
            ds = m1[name][0] - m0[name][0]
            dc = m1[name][1] - m0[name][1]
            seg[name] = ds / dc
            print(f"  {name.split(':')[-1]:<32} {ds/dc*1000:>10.2f} ms  (Δcount={dc:.0f})")
    q = seg.get("vllm:request_queue_time_seconds")
    p = seg.get("vllm:request_prefill_time_seconds")
    if q is not None and p is not None:
        client_ttft = statistics.mean(ttfts)
        net = client_ttft - q - p
        print(f"\nTTFT 三段分解(近似):排队 {q*1000:.0f} ms + prefill {p*1000:.0f} ms"
              f" + 网络/客户端开销 {max(net,0)*1000:.0f} ms ≈ 客户端 TTFT {client_ttft*1000:.0f} ms")
        if q > client_ttft * 0.5:
            print("判读:排队占比 > 50% → 并发超容量(Ch20 §20.2 症状表第 1 行)")
    else:
        print("(未抓到 queue/prefill 指标:该版本指标名不同或端点无数据——只信客户端侧数字)")


if __name__ == "__main__":
    main()
