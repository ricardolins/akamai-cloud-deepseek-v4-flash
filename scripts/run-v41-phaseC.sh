#!/bin/bash
# Phase C (prefix caching ON + DSpark k=3): correctness first (DSpark output corruption is reported on SM120),
# then Spec-Bench real text and the agent simulation, to compare with phase B (same settings, no speculation).
cd /models
mkdir -p results/v41-dspark-quality results/v41-dspark-realtext results/v41-dspark-agent
python3 gsm8k_eval.py --model deepseek-v4.1-flash --n 250 --workers 32 --out results/v41-dspark-quality/gsm8k.json
python3 needle.py --model deepseek-v4.1-flash --tokens 100000 500000 --out results/v41-dspark-quality/needle.json
MODEL=deepseek-v4.1-flash TOKENIZER=deepseek-ai/DeepSeek-V4.1-Flash TOKENIZER_MODE=deepseek_v41 bash bench-realtext.sh v41-dspark-realtext
curl -s localhost:8000/metrics | grep -E "^vllm:spec_decode" > results/v41-dspark-realtext/spec-metrics.txt
for S in 1 4 16; do
  echo "=== $(date -u +%T) agent sessions=$S"
  python3 agent_sim.py --model deepseek-v4.1-flash --sessions $S --turns 10 --start 30000 --step 6000 --out 256 \
    --result results/v41-dspark-agent/agent_s$S.json | tail -12
done
echo ALLDONE
