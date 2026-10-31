#!/bin/bash
# 2x TP=4 sequence, run inside replica A (nohup): single-replica latency on A, then both replicas loaded at once.
# Usage: run-tp4x2.sh <replica-B-pod-IP>
B=http://$1:8000
bash /models/bench.sh tp4 512 main
bash /models/bench.sh tp4 512 long
echo "### both replicas: ctxload"
CTXLOAD="1 2 4" bash /models/bench.sh tp4x2-a 512 ctxload &
CTXLOAD="1 2 4" BASE=$B bash /models/bench.sh tp4x2-b 512 ctxload; wait
echo "### both replicas: sweep"
SWEEP="8 16 32 64" bash /models/bench.sh tp4x2-a 512 sweep &
SWEEP="8 16 32 64" BASE=$B bash /models/bench.sh tp4x2-b 512 sweep; wait
echo "### ALL DONE"
