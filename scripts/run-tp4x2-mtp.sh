#!/bin/bash
# 2x TP=4 + MTP sequence, run inside replica A (nohup): single-replica latency on A, then the throughput sweep on both.
# Usage: run-tp4x2-mtp.sh <replica-B-pod-IP>
B=http://$1:8000
bash /models/bench.sh tp4-mtp 512 main
bash /models/bench.sh tp4-mtp 512 long
echo "### both replicas: sweep"
SWEEP="8 16 32 64" bash /models/bench.sh tp4x2-mtp-a 512 sweep &
SWEEP="8 16 32 64" BASE=$B bash /models/bench.sh tp4x2-mtp-b 512 sweep; wait
echo "### ALL DONE"
