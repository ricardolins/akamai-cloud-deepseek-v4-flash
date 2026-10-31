# Results — DeepSeek-V4-Flash on 8× RTX PRO 6000 Blackwell (vLLM 0.32.0)

Test date: 2026-10-31. Setup and method: [GUIDE.md](GUIDE.md). Raw data: [results/](../results).
Fit for developer workloads and the V4.1-Flash analysis: [V4.1-FLASH.md](V4.1-FLASH.md).

Unless stated otherwise: random prompts of a fixed length, **512 output tokens**, prefix caching off, medians (p50).
"tok/s per user" = 1000 / TPOT. "tok/s total" = output tokens per second of the whole server.

Three layouts were measured:

| Name | Layout | GPUs per engine |
|---|---|---:|
| **TP=8** | 1 replica, tensor parallel over 8 GPUs | 8 |
| **TP=4** | 1 replica of the 2×TP=4 deployment, measured alone (other replica idle) | 4 |
| **2×TP=4** | 2 replicas of 4 GPUs, both under load, half of the users on each | 4 + 4 |

## 1. Summary

| Question | Answer |
|---|---|
| Does DeepSeek-V4-Flash run on RTX PRO 6000 (SM120) with stock vLLM? | **Yes** — vLLM v0.32.0, no patches |
| Best layout on this node | **2 replicas × TP=4**, not TP=8 |
| Single-user speed, 10k prompt | **98 tok/s**, TTFT 1.0 s (TP=4) |
| Single-user speed, 120k prompt | **90 tok/s**, TTFT 13.4 s (TP=4) |
| Full 1M-token prompt | Works: TTFT 217 s (~3.6 min), then **51 tok/s** (TP=4) |
| Peak output, 10k prompts (whole node) | **803 tok/s** with 128 users (2×TP=4) vs 423 tok/s (TP=8) |
| MTP speculative decoding (real text) | **+47% speed for 1 user** (109 → 161 tok/s), +35% throughput at 64 users; 78–80% draft acceptance |
| Failed requests | **0** in every test |

## 2. Single user, by prompt length

| Prompt | TTFT TP=8 | TTFT TP=4 | tok/s per user TP=8 | tok/s per user TP=4 | E2E TP=4 |
|---:|---:|---:|---:|---:|---:|
| 8k | 0.87 s | 0.82 s | 74 | **98** | 6.0 s |
| 10k | 1.10 s | 1.03 s | 74 | **98** | 6.3 s |
| 32k | 3.5 s | 3.3 s | 73 | **96** | 8.6 s |
| 60k | 6.7 s | 6.3 s | 71 | **93** | 11.8 s |
| 120k | 14.1 s | 13.4 s | 67 | **90** | 19.1 s |
| 240k | 31.1 s | 29.8 s | 63 | **83** | 36.0 s |
| 512k | 82.8 s | 79.8 s | 54 | **68** | 87.4 s |
| **1M** | 222.7 s | 216.7 s | 42 | **51** | 226.8 s |

What it shows:

- **Decode barely slows down with context.** From 8k to 120k the per-user speed drops only 8% (98 → 90 tok/s);
  even with **1M tokens** in context the model still writes at 51 tok/s. This is V4's compressed attention:
  each new token reads a heavily compressed KV cache instead of the full history.
- **TTFT is the prefill time and grows with the prompt.** Prefill runs at ~9,000 tokens/s up to 120k, then slows
  down (≈6,400 tok/s at 512k, ≈4,600 tok/s at 1M) because attention work grows with context.
- **One TP=4 replica (4 GPUs) beats TP=8 (8 GPUs)** in every row: ~33% faster decode, slightly faster prefill.
  The reason is communication: no NVLink, and with TP=8 the all-reduce crosses the two CPU sockets
  ([GUIDE §4.8](GUIDE.md#48-why-2tp4-beats-tp8-on-this-node)).
- ITL p50 ≈ TPOT and ITL p99 is within ~5% of p50 for a single user: generation is smooth.

## 3. Concurrency with a 10k prompt

### 3.1 Up to 8 users

| Users | TTFT TP=8 | TTFT TP=4 (1 replica) | tok/s per user TP=8 | tok/s per user TP=4 | total TP=8 | total TP=4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1 s | 1.0 s | 74 | 98 | 64 | 82 |
| 2 | 1.6 s | 1.5 s | 64 | 80 | 107 | 130 |
| 4 | 3.1 s | 3.0 s | 56 | 69 | 167 | 195 |
| 8 | 3.9 s | 4.0 s | 39 | 44 | 239 | 260 |

A single 4-GPU replica serves 8 users faster than TP=8 with twice the GPUs.

### 3.2 Throughput plateau (whole node, 8 GPUs)

| Users (total) | TP=8: TTFT p50 | TP=8: tok/s per user | TP=8: total tok/s | 2×TP=4: TTFT p50 | 2×TP=4: tok/s per user | 2×TP=4: total tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4.3 s | 23 | 303 | 3.9 s | **43** | **519** |
| 32 | 4.3 s | 12 | 347 | 4.0 s | **24** | **637** |
| 64 | 5.1 s | 7 | 391 | 4.0 s | **12** | **713** |
| 128 | 5.1 s | 3.5 | 423 | 4.0 s | **6.6** | **803** |

- **2×TP=4 delivers 1.7–1.9× the throughput of TP=8** on the same 8 GPUs, at every load level.
- The plateau is set by **prefill**: each request brings 10,000 prompt tokens and only 512 output tokens, so the
  server spends most of its time reading prompts. At ~9k prompt tok/s per engine, one engine can admit about
  0.9 requests/s → ~460 output tok/s. Two engines double that ceiling.
- TTFT p99 grows with the queue (e.g. 62 s at 128 users on 2×TP=4) while p50 stays ~4 s: under overload some
  requests wait for a free slot. ITL p99 of ~800 ms means generation pauses while a new prompt is prefilled.
  For latency-sensitive traffic, cap concurrency per replica (≈8–16) and scale out.

## 4. Long context under concurrency (60k and 120k prompts)

| Prompt | Users (total) | TTFT TP=8 | TTFT 2×TP=4 | tok/s per user TP=8 | tok/s per user 2×TP=4 | total TP=8 | total 2×TP=4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60k | 2 | 10.2 s | **6.4 s** | 48 | **95** | 49 | **87** |
| 60k | 4 | 15.0 s | **9.7 s** | 26 | **58** | 59 | **108** |
| 60k | 8 | 15.2 s | 15.0 s | 11 | **31** | 67 | **130** |
| 120k | 2 | 21.0 s | **13.4 s** | 35 | **91** | 29 | **54** |
| 120k | 4 | 30.2 s | **20.0 s** | 15 | **40** | 32 | **63** |
| 120k | 8 | 30.3 s | 30.7 s | 6 | **18** | 34 | **69** |

With long prompts the difference is even larger: at 2 users × 120k, 2×TP=4 gives each user **91 tok/s and 13.4 s
TTFT** (each replica handles one request alone) vs 35 tok/s and 21 s on TP=8 (both requests share one engine).

## 5. MTP (speculative decoding)

DeepSeek-V4-Flash ships one multi-token-prediction layer. vLLM option:
`--speculative-config={"method":"mtp","num_speculative_tokens":1}`.

### 5.1 Random prompts are the wrong test for MTP

With the random-token prompts used above, the MTP draft was accepted only **1–12%** of the time
([logs/mtp-specdecoding-metrics.log](../logs/mtp-specdecoding-metrics.log)). The model continues random text
in unpredictable ways, so the draft is almost always wrong and MTP only adds work:

| 1 user, TP=4 | TPOT without MTP | TPOT with MTP |
|---|---:|---:|
| 8k random prompt | 10.2 ms | 11.8 ms (slower) |
| 10k random prompt | 10.2 ms | 11.6 ms (slower) |
| 32k random prompt | 10.4 ms | 12.0 ms (slower) |

That run was stopped and MTP was measured on real text instead.

### 5.2 Real text (Spec-Bench), one TP=4 replica

[Spec-Bench](https://github.com/hemingkx/Spec-Bench): 480 real questions in 13 categories (multi-turn chat,
translation, summarization, QA, math, RAG). Chat endpoint (thinking on, as in production), up to 512 output
tokens, same seed → **the same questions** for both runs ([scripts/bench-realtext.sh](../scripts/bench-realtext.sh)).
Average prompt ~350–450 tokens.

| Users | TPOT no MTP | TPOT MTP | tok/s per user no MTP | tok/s per user MTP | total no MTP | total MTP | Gain | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.2 ms | **6.2 ms** | 109 | **161** | 99 | 141 | **+47%** per user | 80% |
| 4 | 13.2 ms | **9.2 ms** | 76 | **109** | 288 | **409** | **+42%** | 78% |
| 16 | 22.0 ms | 23.1 ms | 46 | 43 | 550 | **599** | +9% | 80% |
| 64 | 52.5 ms | **36.3 ms** | 19 | **28** | 1,130 | **1,527** | **+35%** | 78% |

- On real text the draft is right **~80% of the time** → **1.8 tokens per forward pass**.
- MTP makes a single user **~1.5× faster** (161 tok/s on 4 GPUs) and raises the throughput of one replica to
  **1,527 tok/s** with 64 users (short prompts).
- The 16-user point shows almost no gain; the no-MTP run has the same jump in TPOT at 16 users, so it is a
  property of the engine at that batch size (kernel / CUDA-graph selection), not of MTP. It was not investigated further.
- TTFT is not affected by MTP (65 ms vs 63 ms for 1 user).
- Not measured: MTP with long real documents (60k+), and `num_speculative_tokens` > 1.

**Recommendation:** enable MTP for real workloads (chat, code, RAG). Ignore MTP numbers from random-token benchmarks.

## 6. Sizing examples (whole node = 8 GPUs = 2 replicas × TP=4)

Measured values, no MTP (MTP adds ~35–47% decode speed on real text):

| Workload | Concurrent users per node that keep the target | Evidence |
|---|---|---|
| 10k prompt, ≥ 25 tok/s per user, TTFT ≤ 5 s | **~16–30** | 16 users: 43 tok/s, TTFT 3.9 s · 32 users: 24 tok/s, TTFT 4.0 s |
| 10k prompt, ≥ 40 tok/s per user | **~16** | 16 users: 43 tok/s |
| 120k prompt, TTFT ≤ 16 s, ≥ 25 tok/s | **2** (1 per replica) | 2 users: 13.4 s, 91 tok/s · 4 users: 20 s TTFT (too slow) |
| 60k prompt, TTFT ≤ 10 s, ≥ 25 tok/s | **~4** | 4 users: 9.7 s, 58 tok/s |
| 1M-token prompt | 1 per replica; plan for ~3.6 min TTFT | 216.7 s, 51 tok/s |
| Short chat (~400-token prompts), MTP on | **~64 per replica** at ~28 tok/s each | 1,527 tok/s per replica |

For long prompts the limit is TTFT (prefill), not memory: the KV cache holds 4.4 M tokens per replica.

### Cost reference

At **$3/GPU-hour** (Akamai Cloud RTX PRO 6000), the node costs **$24/h**. At the measured plateau with
10k-token prompts (803 output tok/s, plus 10k prompt tokens per request): ~2.9 M output tokens/h →
**~$8.3 per million output tokens** (input processing included in the same hour). For short-prompt chat with MTP,
one replica measured 1,527 tok/s; two replicas would be ~3,000 tok/s (×2 is an extrapolation, not measured) →
**~$2.2 per million output tokens**. Compare with your API price list for the same model before deciding.

## 7. Infrastructure observations

| Observation | Detail |
|---|---|
| GPU power under TP=8 | 100% "SM busy" but only **~270 W average** of 600 W: GPUs mostly wait on all-reduce over PCIe |
| GPU power under 2×TP=4 | ~310 W average on the loaded replica, peaks ~490 W; temperatures ≤ 63 °C |
| NUMA | GPUs 0–3 on socket 0, 4–7 on socket 1; Kubernetes assigned each TP=4 replica the 4 GPUs of one socket (checked with `nvidia-smi topo -m` inside both pods) |
| Start-up | ~7.5 min alone (weights 30 s + JIT/graphs 6 min); 16–17 min for 2 replicas starting together (disk contention, weights 8–9 min) |
| KV cache | TP=8: 6.26 M tokens; TP=4: 4.37 M per replica (8.7 M per node) |
| Download | 149 GB in 126 s from Hugging Face |
| Stability | No errors, restarts or failed requests during ~3 h of testing |
