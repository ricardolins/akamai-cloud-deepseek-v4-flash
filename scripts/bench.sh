#!/bin/bash
# Latency/throughput benchmark for DeepSeek-V4-Flash on vLLM. Runs INSIDE the vLLM pod (bench client = same image).
# Fixed-length random prompts + fixed output (--ignore-eos) so TTFT / ITL / TPOT / E2E are comparable across sizes.
# Usage: bench.sh <tag> [output_len] [set]
#   set = main (default) | long | sweep | all
#   main : single user at 8k / 10k / 32k / 60k / 120k + 10k prompt at 2/4/8 users
#   long : single user at 240k / 512k / 1M (needs --max-model-len >= 1,048,576)
#   sweep: 10k prompt at concurrency 16..128 to find the throughput plateau (SWEEP env overrides the levels)
#   ctxload: 60k and 120k prompts at 2/4/8 users (long context under concurrency; CTXLOAD env overrides)
#   MODEL / TOKENIZER env vars override the served model name and tokenizer; BASE overrides the server URL
set -euo pipefail
TAG=${1:-baseline}; OUT=${2:-512}; SET=${3:-main}
DIR=/models/results/$TAG; mkdir -p "$DIR"
run() {  # run <input_len> <concurrency> <num_prompts>
  local IN=$1 C=$2 N=$3 NAME="in${1}_out${OUT}_c${2}"
  echo "=== $(date -u +%T) $NAME"
  vllm bench serve --backend vllm --base-url ${BASE:-http://127.0.0.1:8000} \
    --model ${MODEL:-deepseek-v4-flash} --tokenizer ${TOKENIZER:-deepseek-ai/DeepSeek-V4-Flash} \
    ${TOKENIZER_MODE:+--tokenizer-mode $TOKENIZER_MODE} \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 0 \
    --ignore-eos --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
    --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$DIR" --result-filename "$NAME.json" 2>&1 \
    | grep -E "Successful requests|Failed requests|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|P99 ITL|Mean E2EL|Median E2EL|Error|error"
}
if [[ $SET == main || $SET == all ]]; then
# single user (latency) at growing prompt sizes
run 8000   1 10
run 10000  1 10
run 32000  1 5
run 60000  1 5
run 120000 1 5
# 10k prompt under concurrency: per-user generation speed as load grows
run 10000  2 10
run 10000  4 16
run 10000  8 24
fi
if [[ $SET == long || $SET == all ]]; then
# very long context, DeepSeek-V4's headline feature (1M-token window)
run 240000  1 3
run 512000  1 2
run 1000000 1 2
fi
if [[ $SET == ctxload || $SET == all ]]; then
# CTXLOAD env overrides the levels (e.g. "1 2 4" per replica when two replicas share the load)
for IN in 60000 120000; do for C in ${CTXLOAD:-2 4 8}; do N=$((C * 2)); [ $N -lt 4 ] && N=4; run $IN "$C" $N; done; done
fi
if [[ $SET == sweep || $SET == all ]]; then
# throughput plateau: keep doubling concurrency; 2x requests per level so it reaches steady state
for C in ${SWEEP:-16 32 64 128}; do run 10000 "$C" $((C * 2)); done
fi
echo "=== done $(date -u +%T)"
