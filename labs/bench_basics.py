"""Ch1/Ch2 动手实验:验证"两种形态"(TTFT 随 prompt 增长、TPOT 恒定)与基础压测。
需要本地 OpenAI 兼容服务(vLLM/SGLang)。

用法:
  python bench_basics.py --base-url http://localhost:8000/v1 --model qwen2.5-7b

输出:
  1) decode 恒速性:固定 512-token prompt,输出 32/128/512 -> TTFT/TPOT/ITL p99
  2) prefill 增长:输出 32,prompt 256/2048/8192 -> TTFT 近似线性
  3) 每配置重复 N 次取均值(第 2 章纪律:重复 + 记录)
注意:prompt 用重复词填充,长度是"近似" token 数;正式实验请用固定语料。
"""
import argparse
import statistics
import time

from openai import OpenAI


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
    ttft, last, n_out = None, None, 0
    itls = []
    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            now = time.perf_counter()
            n_out += 1
            if ttft is None:
                ttft = now - t0
            else:
                itls.append(now - last)
            last = now
    e2e = (last or t0) - t0
    tpot = (e2e - ttft) / max(n_out - 1, 1)
    itl_p99 = sorted(itls)[int(0.99 * len(itls))] if itls else float("nan")
    return {"ttft": ttft or 0.0, "tpot": tpot, "itl_p99": itl_p99, "n_out": n_out}


def run_case(client, model, prompt_len, max_tokens, repeats):
    rs = [one_request(client, model, prompt_len, max_tokens) for _ in range(repeats)]
    return {
        "ttft": statistics.mean(r["ttft"] for r in rs),
        "tpot": statistics.mean(r["tpot"] for r in rs),
        "itl_p99": statistics.mean(r["itl_p99"] for r in rs),
        "n_out": statistics.mean(r["n_out"] for r in rs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="EMPTY")

    print(f"{'case':<28}{'TTFT(s)':>10}{'TPOT(s)':>10}{'ITL_p99(s)':>12}{'n_out':>8}")
    # 1) decode 恒速性
    for out in (32, 128, 512):
        r = run_case(client, a.model, 512, out, a.repeats)
        print(f"prompt=512 out={out:<6}{r['ttft']:>10.3f}{r['tpot']:>10.4f}"
              f"{r['itl_p99']:>12.4f}{r['n_out']:>8.0f}")
    # 2) prefill 增长
    for p in (256, 2048, 8192):
        r = run_case(client, a.model, p, 32, a.repeats)
        print(f"prompt={p:<5} out=32{r['ttft']:>10.3f}{r['tpot']:>10.4f}"
              f"{r['itl_p99']:>12.4f}{r['n_out']:>8.0f}")


if __name__ == "__main__":
    main()
