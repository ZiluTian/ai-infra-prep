"""Ch25 实测法:负载生成(三种画像)→ goodput 曲线与 p50/p95/p99。
对应正文:Ch25 实测法(goodput 曲线基线)、Ch27 §27.5(goodput 定义)、Ch29 案例负载。
bench_basics.py 回答"什么形态";本脚本回答"某种负载下的吞吐与分位"(Ch25 实测法:先定负载,再测吞吐)。

用法:
  python bench_load.py --model qwen2.5-7b --profile mixed --duration 60
  python bench_load.py --model qwen2.5-7b --profile prefix50 --duration 60 --concurrency 32
  python bench_load.py --model qwen2.5-7b --profile skew --duration 60   # MoE: 诱导热 expert(Ch16 §16.3)

画像:
  mixed     80% 512in/256out + 20% 8K/512(长短混合 8:2,Ch29 案例负载)
  prefix50  50% 请求共享同一 4K 前缀(前缀缓存命中率画像,Ch5 §5.4);其余 50% 为 512in/256out
  skew      4 个 prompt 循环复用(路由集中 → 热 expert,Ch16 §16.3)

输出:
  1) per-request TTFT/TPOT/E2E(内存收集)
  2) 按请求类别 + 全局:TTFT/TPOT 的 p50/p95/p99 + 输出吞吐 tok/s(Ch2 分位口径)
注意:prompt 用重复词填充,长度是"近似" token 数;正式实验请用固定语料(Ch2 §2.3)。
goodput 还要叠加 SLO 判定(Ch27 §27.5),本脚本只给原始分位与吞吐。
"""
import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def quantile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    ttft, last, n_out = None, None, 0
    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            now = time.perf_counter()
            n_out += 1
            if ttft is None:
                ttft = now - t0
            last = now
    if ttft is None:
        return None  # 未生成任何 token,不计入
    e2e = (last or t0) - t0
    return {"ttft": ttft, "tpot": (e2e - ttft) / max(n_out - 1, 1),
            "e2e": e2e, "n_out": n_out}


def build_spec(profile, rng):
    """按画像生成一个请求规格 (prompt, max_tokens, 类别)。"""
    if profile == "mixed":
        if rng.random() < 0.8:
            return " ".join(["token"] * 256), 256, "short(512in/256out)"
        return " ".join(["token"] * 4096), 512, "long(8K/512out)"
    if profile == "prefix50":
        if rng.random() < 0.5:
            prefix = " ".join(["ctx"] * 2048)  # 共享 4K 前缀(近似)
            tail = " ".join([f"q{i}" for i in range(64)])
            return prefix + " " + tail, 256, "prefix(4K共享)"
        return " ".join(["token"] * 256), 256, "other(512in/256out)"
    # skew: 4 个 prompt 循环 → 路由集中在少数 expert(Ch16 §16.3)
    k = rng.randrange(4)
    return " ".join([f"expert{k}" for _ in range(256)]), 256, f"skew-prompt{k}"


def worker(client, model, profile, stop, results, lock, rng, errs, errlock):
    while not stop.is_set():
        prompt, out, label = build_spec(profile, rng)
        try:
            r = one_request(client, model, prompt, out)
        except Exception as e:  # 连接失败/服务不可用:首个错误即收工
            with errlock:
                if not errs:
                    errs.append(f"{type(e).__name__}: {e}")
            stop.set()
            break
        if r is not None:
            with lock:
                results.append((label, r))
        time.sleep(0.2)  # 请求间隙(近似人的 think time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", choices=["mixed", "prefix50", "skew"], default="mixed")
    ap.add_argument("--duration", type=int, default=60, help="压测时长(秒)")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=300.0)

    stop = threading.Event()
    results, lock = [], threading.Lock()
    errs, errlock = [], threading.Lock()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(worker, client, a.model, a.profile, stop, results, lock,
                          random.Random(i), errs, errlock)
                for i in range(a.concurrency)]
        while time.perf_counter() - t0 < a.duration:
            time.sleep(1)
        stop.set()
        for f in futs:
            f.result()

    if errs:
        print(f"负载生成失败(服务不可达或请求异常): {errs[0]}")
        print(f"检查 --base-url(默认 {a.base_url})与服务是否启动;详见 Ch24 §24.4 环境清单。")
        raise SystemExit(1)

    labels = {}
    for label, r in results:
        labels.setdefault(label, []).append(r)

    def row(tag, rs):
        ttft = [r["ttft"] for r in rs]
        tpot = [r["tpot"] for r in rs]
        print(f"{tag:<22}{len(rs):>6}"
              f"{quantile(ttft, .50):>9.3f}{quantile(ttft, .95):>9.3f}{quantile(ttft, .99):>9.3f}"
              f"{quantile(tpot, .50):>9.4f}{quantile(tpot, .95):>9.4f}{quantile(tpot, .99):>9.4f}")

    print(f"profile={a.profile} duration={a.duration}s concurrency={a.concurrency}")
    print(f"{'class':<22}{'n':>6}{'TTFT_p50':>9}{'TTFT_p95':>9}{'TTFT_p99':>9}"
          f"{'TPOT_p50':>9}{'TPOT_p95':>9}{'TPOT_p99':>9}")
    for label in sorted(labels):
        row(label, labels[label])
    row("ALL", [r for _, r in results])
    total_out = sum(r["n_out"] for _, r in results)
    print(f"输出吞吐 ≈ {total_out / max(a.duration, 1):.0f} tok/s(客户端口径;goodput 需叠加 SLO,Ch27 §27.5)")


if __name__ == "__main__":
    main()
