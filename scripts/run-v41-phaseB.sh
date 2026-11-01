#!/bin/bash
# Phase B (prefix caching ON, no speculation): coding-agent simulation + Spec-Bench real text.
cd /models
mkdir -p results/v41-agent-prefix results/v41-realtext
[ -s spec_bench.jsonl ] || curl -sL -o spec_bench.jsonl https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
for S in 1 2 4 8; do
  echo "=== $(date -u +%T) agent sessions=$S"
  python3 agent_sim.py --model deepseek-v4.1-flash --sessions $S --turns 10 --start 30000 --step 6000 --out 256 \
    --result results/v41-agent-prefix/agent_s$S.json | tail -12
done
pip install -q pandas 2>/dev/null
MODEL=deepseek-v4.1-flash TOKENIZER=deepseek-ai/DeepSeek-V4.1-Flash TOKENIZER_MODE=deepseek_v41 bash bench-realtext.sh v41-realtext
echo ALLDONE
