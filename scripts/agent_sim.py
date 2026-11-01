#!/usr/bin/env python3
"""Coding-agent load simulation: N concurrent sessions, each a sequence of turns whose prompt is the previous turn's
prompt + the previous answer + new tool output. This is the shape of an agent loop (Claude Code-style): the context
grows every turn and almost all of it repeats, so prefix caching decides the TTFT of every turn after the first.

Prompts are random token ids sent to /v1/completions (no chat template), outputs fixed with ignore_eos.
Runs inside the vLLM pod (aiohttp is in the image).
Usage: agent_sim.py --sessions 4 --turns 8 --start 20000 --step 4000 --out 256 --result results/<tag>/agent_s4.json
"""
import argparse, asyncio, json, random, statistics, time

import aiohttp

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8000")
p.add_argument("--model", required=True)
p.add_argument("--sessions", type=int, default=1)
p.add_argument("--turns", type=int, default=8)
p.add_argument("--start", type=int, default=20000, help="tokens in the first turn (system prompt + files)")
p.add_argument("--step", type=int, default=4000, help="new tokens per turn (tool output / file reads)")
p.add_argument("--out", type=int, default=256, help="output tokens per turn")
p.add_argument("--vocab", type=int, default=100000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--result", required=True)
a = p.parse_args()

rng = random.Random(a.seed)
toks = lambda n: [rng.randrange(1000, a.vocab) for _ in range(n)]


async def turn(http, prompt):
    body = {"model": a.model, "prompt": prompt, "max_tokens": a.out, "min_tokens": a.out, "ignore_eos": True,
            "temperature": 0, "stream": True}
    t0 = time.perf_counter(); first = None; times = []
    async with http.post(f"{a.base}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            times.append(now)
    end = time.perf_counter()
    return {"ttft": first - t0, "e2e": end - t0, "tpot": (end - first) / max(a.out - 1, 1)}


async def session(http, sid, rows):
    ctx = toks(a.start)
    for t in range(a.turns):
        r = await turn(http, ctx)
        r.update(session=sid, turn=t, prompt_len=len(ctx))
        rows.append(r)
        print(f"s{sid} t{t} prompt={len(ctx)} ttft={r['ttft']:.2f}s tpot={r['tpot'] * 1000:.1f}ms", flush=True)
        ctx = ctx + toks(a.out) + toks(a.step)   # previous answer (approximated) + new tool output


async def main():
    rows = []
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        t0 = time.perf_counter()
        await asyncio.gather(*(session(http, s, rows) for s in range(a.sessions)))
        wall = time.perf_counter() - t0
    first = [r["ttft"] for r in rows if r["turn"] == 0]
    later = [r["ttft"] for r in rows if r["turn"] > 0]
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))] if xs else None
    summary = {
        "sessions": a.sessions, "turns": a.turns, "start": a.start, "step": a.step, "out": a.out,
        "max_prompt": max(r["prompt_len"] for r in rows), "wall_s": wall,
        "ttft_first_turn_p50": statistics.median(first), "ttft_later_turns_p50": statistics.median(later) if later else None,
        "ttft_later_turns_p90": q(later, 0.9), "tpot_p50_ms": statistics.median(r["tpot"] for r in rows) * 1000,
        "turns_per_min": len(rows) / wall * 60, "rows": sorted(rows, key=lambda r: (r["session"], r["turn"])),
    }
    json.dump(summary, open(a.result, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


asyncio.run(main())
