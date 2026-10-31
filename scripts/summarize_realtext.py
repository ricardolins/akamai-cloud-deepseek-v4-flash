#!/usr/bin/env python3
"""Turn real-text (Spec-Bench) `vllm bench serve` results into a Markdown table, including MTP acceptance.
Usage: summarize_realtext.py results/<tag> [results/<tag2> ...]"""
import json, pathlib, re, sys

fmt = lambda v, p=0: "—" if v is None else f"{v:,.{p}f}"
print("| Config | Conc. | n | avg in | avg out | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (s) | tok/s per user | tok/s total | MTP acceptance | tokens/step |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
rows = []
for base in sys.argv[1:]:
    for f in pathlib.Path(base).glob("specbench_c*.json"):
        d = json.loads(f.read_text())
        c = int(re.search(r"_c(\d+)", f.stem).group(1))
        rows.append((pathlib.Path(base).name, c, d))
for cfg, c, d in sorted(rows, key=lambda r: (r[0], r[1])):
    n = d["completed"]; tpot = d.get("median_tpot_ms")
    acc = d.get("spec_decode_acceptance_rate")
    print(f"| {cfg} | {c} | {n} | {d['total_input_tokens'] / n:,.0f} | {d['total_output_tokens'] / n:,.0f} | "
          f"{fmt(d.get('median_ttft_ms'))} | {fmt(tpot, 1)} | {fmt(d.get('median_itl_ms'), 1)} | "
          f"{fmt(d.get('median_e2el_ms', 0) / 1000, 1)} | {fmt(1000 / tpot if tpot else None, 1)} | "
          f"{fmt(d.get('output_throughput'), 1)} | {'—' if acc is None else f'{acc:.0f}%'} | "
          f"{fmt(d.get('spec_decode_acceptance_length'), 2)} |")
