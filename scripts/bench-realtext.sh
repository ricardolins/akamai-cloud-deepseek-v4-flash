#!/bin/bash
# Real-text benchmark (Spec-Bench: 480 questions, 13 categories — chat, translation, summarization, math, RAG, ...).
# Used to compare MTP vs no MTP: random-token prompts make the MTP draft almost always wrong (1-2% acceptance),
# so speculative decoding can only be judged on real text. Chat endpoint (chat template + thinking on, as in production),
# up to 512 output tokens, natural stop (no --ignore-eos). Same seed → same questions for every configuration.
# Usage: bench-realtext.sh <tag>   (dataset at /models/spec_bench.jsonl; BASE overrides the server URL;
#        MODEL / TOKENIZER / TOKENIZER_MODE override the V4-Flash defaults, e.g. for V4.1: deepseek_v41)
set -euo pipefail
TAG=$1; DIR=/models/results/$TAG; mkdir -p "$DIR"
for C in ${LEVELS:-1 4 16 64}; do
  N=$((C * 4)); [ $N -lt 20 ] && N=20
  echo "=== $(date -u +%T) specbench_c$C"
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url ${BASE:-http://127.0.0.1:8000} \
    --model ${MODEL:-deepseek-v4-flash} --tokenizer ${TOKENIZER:-deepseek-ai/DeepSeek-V4-Flash} --tokenizer-mode ${TOKENIZER_MODE:-deepseek_v4} \
    --dataset-name spec_bench --dataset-path /models/spec_bench.jsonl --spec-bench-output-len 512 \
    --num-prompts $N --max-concurrency $C --request-rate inf --seed 0 --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$DIR" --result-filename "specbench_c$C.json" 2>&1 \
    | grep -E "Successful requests|Failed requests|Output token throughput|Total generated|Median TTFT|Median TPOT|Median ITL|Median E2EL|Acceptance|acceptance|Error"
done
echo "=== done $(date -u +%T)"
