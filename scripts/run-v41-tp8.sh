#!/bin/bash
# Phase A (V4.1, prefix caching OFF): the same random-prompt suite as V4. Runs inside the vLLM pod.
cd /models
export MODEL=deepseek-v4.1-flash TOKENIZER=deepseek-ai/DeepSeek-V4.1-Flash
for set in main long ctxload sweep; do bash bench.sh v41-tp8 512 $set; done
echo ALLDONE
