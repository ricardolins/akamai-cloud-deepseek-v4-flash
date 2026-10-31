#!/usr/bin/env python3
"""Combine results of the same test run at the same time on two replicas (2x TP=4) into one row per test.
Totals (output tok/s, concurrency, requests) are summed; latencies are the average of the two replicas' values
(both replicas get the same load, so their medians are close).
Usage: combine_replicas.py results/<tag-a> results/<tag-b> results/<tag-combined>"""
import json, pathlib, sys

a, b, out = map(pathlib.Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
SUM = ("completed", "output_throughput", "total_token_throughput", "request_throughput", "max_concurrency")
for fa in sorted(a.glob("*.json")):
    fb = b / fa.name
    if not fb.exists():
        continue
    da, db = json.loads(fa.read_text()), json.loads(fb.read_text())
    d = {}
    for k, va in da.items():
        vb = db.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            d[k] = va + vb if k in SUM else (va + vb) / 2
    # file name carries the per-replica concurrency; rename to the total
    stem = fa.stem.rsplit("_c", 1)
    name = f"{stem[0]}_c{int(stem[1]) * 2}.json"
    (out / name).write_text(json.dumps(d, indent=1))
    print(name)
