"""附录 H:agent 多轮会话负载生成 → 前缀缓存生效画像(TTFT 冷/热比 + naive/增量 prefill 账)。
对应正文:附录 H §H.1.2(公式 H-1:naive vs 增量 prefill 账、η)、§H.2.1(冷/热 TTFT,公式 H-2);
对照 Ch5 §5.4(50% 复用静态画像)、Ch2 §2.3(分位口径)。

模型:agent 会话 = (共享 system 前缀 S, K 轮, 每轮 delta Δ + 工具结果 R)
  - chat 协议下第 k 轮请求重发完整历史:prompt_k ≈ S + Σ_{i≤k}(Δ_i + R_i + A_{i-1})
  - 前缀缓存生效时,热轮实际 prefill ≈ 该轮增量 → TTFT 冷/热比是缓存生效的代理指标(H-2)
  - 账:naive = Σ_k prompt_k;cached(理论上界)= 每会话末轮 prompt(全部历史只 prefill 一次);
    η = 1 − cached/naive(复现 H-1;实服务返回 usage 时用实测 prompt_tokens,否则按词数估)

用法:
  python agent_load.py --dry-run --sessions 3 --turns 6            # 不发请求:验证负载构造与账目
  python agent_load.py --model qwen2.5-7b --sessions 4 --turns 8   # 需本地 OpenAI 兼容服务(vLLM/SGLang)
  python agent_load.py --dry-run --system-tokens 8192 --turns 10   # 按附录 H §H.1.2 手算参数复算账

并发语义:会话内轮次串行(agent 语义:上一轮回答 + 工具结果回来才发下一轮);跨会话并行(多会话共存)。
注意:
  - prompt 用重复词填充,1 词 ≈ 1 token(近似 token 数,dry-run 账目按词数精确计数);正式实验请用固定语料(Ch2 §2.3)。
  - 2026-11-19 新增:--dry-run 已验证;实服务路径未核验(附录 G §G.5、附录 F §F.4)。
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def quantile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def make_text(rng, n_words, tag):
    """生成 n_words 个词的填充文本(1 词 ≈ 1 token,近似;词数可精确计数)。"""
    return " ".join(f"{tag}-{rng.randrange(1000)}" for _ in range(n_words))


def count_tokens(text):
    """填充文本的近似 token 数 = 词数(构造口径,精确)。"""
    return len(text.split())


def build_plan(a, seed, system_text):
    """构造全部会话/轮次:每轮的用户消息 + 请求消息序列(含 system) + 词数账。
    dry-run 与实服务共用同一构造,保证两边账目同构。"""
    plan = []
    sys_msg = {"role": "system", "content": system_text}
    for sid in range(a.sessions):
        msgs = [sys_msg]  # 会话历史(逐轮增长)
        for k in range(1, a.turns + 1):
            delta = make_text(random.Random(seed + sid * 10000 + k), a.delta_tokens, "d")
            tool = json.dumps({"tool": f"t{sid}-{k}",
                               "result": make_text(random.Random(seed + sid * 10000 + k + 500),
                                                   a.tool_tokens, "r")})
            user = f"{delta}\n[tool_result] {tool}"
            user_msg = {"role": "user", "content": user}
            req_msgs = msgs + [user_msg]
            plan.append({
                "sid": sid, "turn": k, "user": user, "req_msgs": req_msgs,
                "prompt_est": count_tokens("\n".join(m["content"] for m in req_msgs)),
            })
            msgs.append(user_msg)
            msgs.append({"role": "assistant",
                         "content": make_text(random.Random(seed + sid * 10000 + k + 900),
                                              a.max_out_tokens, "a")})
    return plan


def run_dry(a, plan, system_text):
    print(f"[dry-run] 不发送任何请求。sessions={a.sessions} turns={a.turns} "
          f"S≈{a.system_tokens}tok Δ≈{a.delta_tokens}tok R≈{a.tool_tokens}tok "
          f"A≈{a.max_out_tokens}tok(合成回答)")
    print(f"{'session':>8}{'turn':>6}{'prompt_est':>12}")
    for item in plan:
        print(f"{item['sid']:>8}{item['turn']:>6}{item['prompt_est']:>12}")

    print("\n首个请求体(截断):")
    first = plan[0]
    body = {"model": a.model, "stream": True,
            "messages": [{"role": m["role"], "content": m["content"][:80] + (" …" if len(m["content"]) > 80 else "")}
                         for m in first["req_msgs"][:2]]}
    print(json.dumps(body, ensure_ascii=False)[:400])

    # 账(与实服务路径同构:naive = Σ prompt_k;cached = 每会话末轮 prompt)
    by_sess = {}
    for item in plan:
        by_sess.setdefault(item["sid"], []).append(item)
    naive, cached = 0, 0
    for sid, items in sorted(by_sess.items()):
        s_naive = sum(i["prompt_est"] for i in items)
        s_cached = items[-1]["prompt_est"]
        naive += s_naive
        cached += s_cached
        print(f"session {sid}: naive={s_naive}tok cached={s_cached}tok "
              f"η={1 - s_cached / s_naive:.1%}")
    print(f"全局: naive={naive}tok cached={cached}tok η={1 - cached / naive:.1%}"
          f"(对照附录 H §H.1.2 手算;冷/热 TTFT 比需实服务模式输出,§H.2.1)")


def run_live(a, plan, system_text):
    from openai import OpenAI  # 惰性导入:dry-run 不依赖 openai 包
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=a.timeout)

    results, lock = [], threading.Lock()
    errs, errlock = [], threading.Lock()

    def session_worker(sid):
        msgs = [{"role": "system", "content": system_text}]
        for k in range(1, a.turns + 1):
            user = next(p["user"] for p in plan if p["sid"] == sid and p["turn"] == k)
            msgs.append({"role": "user", "content": user})
            prompt_est = count_tokens("\n".join(m["content"] for m in msgs))
            t0 = time.perf_counter()
            try:
                stream = client.chat.completions.create(
                    model=a.model, messages=msgs, max_tokens=a.max_out_tokens,
                    temperature=a.temperature, stream=True,
                    stream_options={"include_usage": True})
                ttft, last, parts, usage = None, None, [], None
                for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        last = now
                        parts.append(chunk.choices[0].delta.content)
            except Exception as e:
                with errlock:
                    errs.append(f"s{sid}t{k} {type(e).__name__}: {e}")
                msgs.pop()
                return
            if ttft is None:
                with lock:
                    results.append({"sid": sid, "turn": k, "prompt_est": prompt_est,
                                    "prompt_used": 0, "out": 0, "ttft": None, "e2e": None,
                                    "fail": True})
                msgs.pop()
                continue
            text = "".join(parts)
            with lock:
                results.append({"sid": sid, "turn": k, "prompt_est": prompt_est,
                                "prompt_used": usage.prompt_tokens if usage else prompt_est,
                                "out": usage.completion_tokens if usage else count_tokens(text),
                                "ttft": ttft, "e2e": (last or t0) - t0, "fail": False})
            msgs.append({"role": "assistant", "content": text})

    with ThreadPoolExecutor(max_workers=a.sessions) as ex:
        futs = [ex.submit(session_worker, sid) for sid in range(a.sessions)]
        for f in futs:
            f.result()

    if not results:
        print(f"负载生成失败(服务不可达或请求异常): {errs[0] if errs else '无结果'}")
        print(f"检查 --base-url(默认 {a.base_url})与服务是否启动;详见 Ch24 §24.4 环境清单。")
        raise SystemExit(1)

    print(f"sessions={a.sessions} turns={a.turns} model={a.model} "
          f"(prompt_used 有 usage 时为实测,否则为词数估算)")
    print(f"{'sess':>5}{'turn':>6}{'prompt':>9}{'out':>6}{'TTFT(s)':>10}{'E2E(s)':>10}")
    for r in sorted(results, key=lambda x: (x["sid"], x["turn"])):
        print(f"{r['sid']:>5}{r['turn']:>6}{r['prompt_used']:>9}{r['out']:>6}"
              f"{(r['ttft'] or float('nan')):>10.3f}{(r['e2e'] or float('nan')):>10.3f}")
    if errs:
        print(f"失败轮次 {len(errs)}: {errs[0]}(其余轮次照常统计)")

    ok = [r for r in results if not r["fail"]]
    ttft_all = [r["ttft"] for r in ok]
    cold = [r["ttft"] for r in ok if r["turn"] == 1]
    warm = [r["ttft"] for r in ok if r["turn"] > 1]
    print(f"\nTTFT p50/p95/p99(全轮,Ch2 分位口径): "
          f"{quantile(ttft_all, .5):.3f}/{quantile(ttft_all, .95):.3f}/{quantile(ttft_all, .99):.3f}")
    if cold and warm:
        ratio = quantile(warm, .5) / quantile(cold, .5)
        print(f"TTFT 冷/热比(p50 口径)= {ratio:.2f}(冷 = 各会话 turn1;热 = turn≥2;"
              f"前缀缓存生效时 热 ≈ 增量/冷轮 prompt,见 H-2;≈1 或 >1 → 查路由/KV 逐出,§H.2.2/§H.2.3)")
    naive = sum(r["prompt_used"] for r in ok)
    by_sess = {}
    for r in ok:
        by_sess[r["sid"]] = max(by_sess.get(r["sid"], 0), r["prompt_used"])
    cached = sum(by_sess.values())
    print(f"prefill 账: naive(Σ 每轮重发历史)={naive}tok;cached(理论上界,历史只算一次)={cached}tok;"
          f" η={1 - cached / naive:.1%}(复现 H-1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--system-tokens", type=int, default=2048, help="共享 system 前缀长度(近似 token)")
    ap.add_argument("--delta-tokens", type=int, default=128, help="每轮用户输入长度(近似)")
    ap.add_argument("--tool-tokens", type=int, default=192, help="每轮工具结果长度(近似)")
    ap.add_argument("--max-out-tokens", type=int, default=64, help="每轮模型输出上限(TTFT 为主,decode 从短)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="不发请求:打印负载构造与 prefill 账(无 openai 包/无服务也可运行)")
    a = ap.parse_args()

    # system 前缀:全会话共享(前缀缓存的共享主干)
    system_text = make_text(random.Random(a.seed + 7), a.system_tokens, "sys")
    plan = build_plan(a, a.seed, system_text)

    if a.dry_run:
        run_dry(a, plan, system_text)
    else:
        run_live(a, plan, system_text)


if __name__ == "__main__":
    main()
