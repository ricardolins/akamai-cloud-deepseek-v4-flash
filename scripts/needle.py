#!/usr/bin/env python3
"""Long-context correctness check ("double needle"): a long filler document with two facts hidden at 50% and 80%
depth, then one question asking for both. Catches silent numerical corruption at long context (reported for
DeepSeek-V4.1 on SM120 with a wrong FP8 KV layout, vLLM issue #56700) that short prompts and GSM8K miss.

Filler = numbered, varied plain-English sentences (deterministic). Prompt size is reported from the server's usage.
Usage (inside the vLLM pod): needle.py --model <served-name> --tokens 100000 200000 500000 --out result.json
"""
import argparse, json, random, time, urllib.request

SUBJ = ["The archive", "A courier", "The committee", "An engineer", "The harbor office", "A gardener", "The museum",
        "A pilot", "The bakery", "A librarian", "The weather station", "A surveyor"]
VERB = ["recorded", "reviewed", "delivered", "inspected", "catalogued", "repaired", "described", "measured",
        "photographed", "summarized", "approved", "translated"]
OBJ = ["the northern bridge", "a box of letters", "the spring schedule", "an old map", "the river levels",
       "a set of keys", "the visitor log", "twelve crates", "the lighthouse lamp", "a broken clock",
       "the harvest report", "the station antenna"]
WHEN = ["on Monday", "before noon", "after the storm", "during the audit", "at dawn", "late in the evening",
        "in the third week", "without delay", "for the second time", "as planned"]
NEEDLES = [("The password for meeting room Aurora is VIOLET-7342.", "VIOLET-7342"),
           ("The internal codename of project Kestrel is BLUE HERON, led by Marisol Okafor.", "BLUE HERON")]
QUESTION = ("\n\nBased only on the document above, answer two questions: (1) What is the password for meeting room "
            "Aurora? (2) What is the internal codename of project Kestrel? Reply in the form "
            "'1: <password>; 2: <codename>'.")


def filler(n_chars, rng):
    out, i, size = [], 0, 0
    while size < n_chars:
        i += 1
        s = f"{i}. {rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(WHEN)}. "
        out.append(s); size += len(s)
    return out


def ask(base, model, prompt, extra, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], **extra}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"] or "", d["usage"]["prompt_tokens"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[100000])
    ap.add_argument("--chars-per-token", type=float, default=4.6, help="filler calibration (measured: adjust if off)")
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra, results = json.loads(a.extra), []
    for target in a.tokens:
        rng = random.Random(target)
        sents = filler(int(target * a.chars_per_token), rng)
        for (fact, _), depth in zip(NEEDLES, (0.5, 0.8)):
            sents.insert(int(len(sents) * depth), fact + " ")
        answer, ptoks, secs = ask(a.base, a.model, "".join(sents) + QUESTION, extra, a.max_tokens)
        ok = [key.lower() in answer.lower() for _, key in NEEDLES]
        r = {"target_tokens": target, "prompt_tokens": ptoks, "seconds": round(secs, 1),
             "needle_50pct": ok[0], "needle_80pct": ok[1], "answer": answer[-300:]}
        results.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"model": a.model, "extra": a.extra, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
