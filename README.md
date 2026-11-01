# DeepSeek-V4-Flash on Akamai Cloud (LKE) — 8× RTX PRO 6000 Blackwell

Serving [**deepseek-ai/DeepSeek-V4-Flash**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
(284B-parameter MoE, 13B active, FP4 experts + FP8, 1M-token context) with **vLLM 0.32.0** on one Akamai Cloud
GPU node (LKE plan `g3-gpu-rtxpro6000-blackwell-8`: 8× RTX PRO 6000 Blackwell, 96 GB each), and measuring
**TTFT, TPOT, ITL and end-to-end latency** from 8k to **1M-token prompts**, under concurrency up to 128 users,
with and without MTP speculative decoding. **Also: [DeepSeek-V4.1-Flash](docs/V4.1-FLASH.md)** (510 GB, TP=8) on the
same node — runs only with a pending vLLM fix on SM120; much faster prefill, flat decode to 1M, ~0.55 s agent turns
with prefix caching.

## Key results

| | TP=8 (1 engine, 8 GPUs) | **TP=4 (1 engine, 4 GPUs)** |
|---|---:|---:|
| 10k prompt, 1 user: TTFT / speed | 1.1 s / 74 tok/s | **1.0 s / 98 tok/s** |
| 120k prompt, 1 user: TTFT / speed | 14.1 s / 67 tok/s | **13.4 s / 90 tok/s** |
| 512k prompt, 1 user: TTFT / speed | 82.8 s / 54 tok/s | **79.8 s / 68 tok/s** |
| **1M prompt**, 1 user: TTFT / speed | 222.7 s / 42 tok/s | **216.7 s / 51 tok/s** |

| Whole node (8 GPUs), 10k prompts | TP=8 | **2 replicas × TP=4** |
|---|---:|---:|
| 16 users: speed per user / total | 23 / 303 tok/s | **43 / 519 tok/s** |
| 128 users: total (plateau) | 423 tok/s | **803 tok/s** |
| 2 users × 120k: TTFT / speed per user | 21.0 s / 35 tok/s | **13.4 s / 91 tok/s** |

**MTP speculative decoding** (real text, Spec-Bench, one TP=4 replica): **161 vs 109 tok/s** for one user (+47%),
**1,527 vs 1,130 tok/s** with 64 users (+35%), 80% draft acceptance. On random-token prompts MTP is useless
(1–12% acceptance) — benchmark it on real text.

Takeaways:

1. **It runs on stock vLLM 0.32.0** on SM120 (RTX PRO 6000) — no forks or patches, unlike mid-2026 reports.
2. **Use 2 replicas × TP=4, not TP=8.** No NVLink and two CPU sockets make 8-way all-reduce expensive; 4 GPUs
   already hold the model (149 GB) with 4.4 M tokens of KV cache each.
3. **Decode speed barely depends on context** (98 → 90 tok/s from 10k to 120k; still 51 tok/s at 1M).
   TTFT (prefill, ~9k tok/s) is what grows — plan long-context sizing on TTFT.
4. **Enable MTP** for real workloads.
5. 0 failed requests in ~3 h of tests.

**DeepSeek-V4.1-Flash** (TP=8, vLLM 0.32.0 + PR #60762 + FlashInfer 0.7.2 — [details](docs/V4.1-FLASH.md)):

| | V4.1-Flash | V4-Flash TP=4 |
|---|---:|---:|
| 120k prompt, 1 user: TTFT / speed | **8.9 s** / 78 tok/s | 13.4 s / 90 tok/s |
| 1M prompt, 1 user: TTFT / speed | **91 s** / **78 tok/s** | 217 s / 51 tok/s |
| Real text, 1 user, speculative decoding | 155 tok/s (DSpark) | 161 tok/s (MTP) |
| Coding-agent turn, prefix caching on, 30k→86k context | **~0.55 s TTFT**, up to 16 agents | — |
| Quality on the patched stack | GSM8K 245/250; facts in a 1.01M-token document found | — |

Docs:

- [docs/GUIDE.md](docs/GUIDE.md): step by step — concepts, setup, manifests, method, difficulties and fixes, cleanup
- [docs/RESULTS.md](docs/RESULTS.md): all tables and the analysis, sizing examples and cost reference
- [docs/V4.1-FLASH.md](docs/V4.1-FLASH.md): DeepSeek-V4.1-Flash — what changes, why stock vLLM fails on SM120, the
  patched stack, results (latency, concurrency, agents, DSpark, quality), V4 vs V4.1 for developers, sources
- [reports/DeepSeek-V4-and-V4.1-Flash-benchmark-results.docx](reports/DeepSeek-V4-and-V4.1-Flash-benchmark-results.docx): Word report
  (all tables, sizing, cost, fit for developer workloads), built by `scripts/make_report_docx.js`

## Layout

```
k8s/        namespace, model download Jobs (V4, V4.1), vLLM Deployments (V4: TP=8, 2xTP=4, +MTP; V4.1: TP=8 stock,
            patched, +prefix caching, +DSpark), node cleanup pod
patches/    vLLM PR #60762 source diff (V4.1 on SM120)
scripts/    bench.sh (random prompts), bench-realtext.sh (Spec-Bench), run-tp4x2*.sh (two-replica runs),
            agent_sim.py (coding-agent sessions), gsm8k_eval.py, needle.py (quality), run-v41-phase*.sh,
            summarize.py, summarize_realtext.py, combine_replicas.py, gpu_stats.py, make_report_docx.js
results/    raw `vllm bench serve` JSON per configuration, summaries, GPU telemetry, hardware topology
logs/       server start-up logs, benchmark console logs, MTP acceptance metrics, cleanup log
docs/       GUIDE.md, RESULTS.md and V4.1-FLASH.md
reports/    Word report (generated from results/)
```

| results/ folder | Content |
|---|---|
| `tp8/` | TP=8: single user 8k–1M, 10k at 2–128 users, 60k/120k at 2–8 users, GPU telemetry |
| `tp4/` | one TP=4 replica alone: single user 8k–1M, 10k at 2–8 users |
| `tp4x2-a/`, `tp4x2-b/`, `tp4x2/` | both replicas under load: per replica and combined (totals) |
| `tp4-realtext/`, `tp4-mtp-realtext/` | Spec-Bench without / with MTP (1, 4, 16, 64 users) |
| `tp4-mtp-random/` | MTP on random prompts (aborted run kept as evidence) |
| `gpu/` | `nvidia-smi dmon` of the 2×TP=4 run |
| `v41-tp8/` | V4.1 TP=8: single user 8k–1M, concurrency 1–128, 60k/120k under load, GPU telemetry |
| `v41-quality/`, `v41-dspark-quality/` | GSM8K and 1M-token needle checks (without / with DSpark) |
| `v41-agent-prefix/`, `v41-dspark-agent/` | coding-agent simulation with prefix caching (without / with DSpark) |
| `v41-realtext/`, `v41-dspark-realtext/` | Spec-Bench without / with DSpark, spec-decode metrics |

Everything the tests created on the cluster (namespace, 149 GB + 476 GiB of weights, FlashInfer wheels, vLLM image)
was deleted at the end.
