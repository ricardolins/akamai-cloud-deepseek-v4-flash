# Step-by-step guide — DeepSeek-V4-Flash with vLLM on 8× RTX PRO 6000 Blackwell (Akamai Cloud LKE)

This guide explains, in order, everything that was done to serve **deepseek-ai/DeepSeek-V4-Flash** on one
Akamai Cloud GPU node with 8 NVIDIA RTX PRO 6000 Blackwell GPUs, how it was measured, and what went wrong
along the way. The numbers are in [RESULTS.md](RESULTS.md).

## 1. Concepts in 2 minutes

| Term | Meaning | Why it matters here |
|---|---|---|
| **TTFT** (time to first token) | Time from sending the request to receiving the first output token. Dominated by **prefill**: the model reads the whole prompt. | Grows with prompt length. A 1M-token prompt takes minutes. |
| **TPOT** (time per output token) | Average time between output tokens after the first one. `1000 / TPOT` = tokens per second **per user**. | The "typing speed" the user sees. |
| **ITL** (inter-token latency) | Time between each pair of consecutive tokens. p50 ≈ TPOT; **p99** shows stalls. | High p99 ITL = other users' prefills interrupting your generation. |
| **E2E** (end-to-end latency) | Total time of the request: TTFT + all output tokens. | What a batch job or an agent step actually waits. |
| **Concurrency** | Requests in flight at the same time. | More users → more total tokens/s but slower per user. |
| **Prefill / decode** | Prefill processes the prompt (compute-bound, parallel). Decode generates one token at a time (memory-bandwidth-bound). | Long prompts are limited by prefill; chat speed by decode. |
| **TP** (tensor parallelism) | Every layer is split across N GPUs; the GPUs exchange partial results (all-reduce) twice per layer. | Without NVLink, the all-reduce goes over PCIe and costs time. |
| **Replica** | An independent copy of the model on its own GPUs. | 2 replicas of 4 GPUs vs 1 replica of 8 GPUs is the key comparison of this test. |
| **MoE** (mixture of experts) | Only a few "expert" sub-networks run per token. | V4-Flash has 284B parameters but only **13B active** per token → fast decode. |
| **MTP** (multi-token prediction) | A small extra layer drafts the next token; the main model verifies it in the same step. | Up to ~2 tokens per forward pass when the draft is right. |
| **KV cache** | Memory holding the attention state of every token in context. | Limits how many long requests fit at once. |

## 2. Infrastructure

| Item | Value |
|---|---|
| Cloud | Akamai Cloud (Linode), LKE (standard, not Enterprise), region `us-mia` |
| Kubernetes | v1.36.5, 1 node |
| Node plan | `g3-gpu-rtxpro6000-blackwell-8` |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7 each (768 GB total), 600 W each, compute capability 12.0 (SM120) |
| GPU interconnect | **PCIe Gen5 x16 only, no NVLink** (`nvidia-smi topo -m`: `PHB`) |
| CPU / RAM | 128 vCPUs, 1.4 TB RAM, **2 NUMA nodes**: GPUs 0–3 on NUMA 0 (CPUs 0–63), GPUs 4–7 on NUMA 1 (CPUs 64–127) — see [results/hardware-topology.txt](../results/hardware-topology.txt) |
| OS / runtime | Debian 13, kernel 6.12, containerd 2.4 |
| NVIDIA driver | 615.71.10 (CUDA 13.4), preinstalled by LKE |

### 2.1 GPU setup (done before this test)

On **standard LKE**, the NVIDIA driver is already installed on GPU nodes. Following the
[Akamai guide](https://techdocs.akamai.com/cloud-computing/docs/gpus-on-lke), only the NVIDIA GPU Operator is
installed, **with the driver and the container toolkit disabled**:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator \
  --set driver.enabled=false --set toolkit.enabled=false
```

> Do not enable the toolkit on standard LKE: it rewrites the containerd config of the node and can take the
> node down. LKE **Enterprise** is different: there the operator must install the driver.

Validation: the node reports `nvidia.com/gpu: 8` and a pod requesting 8 GPUs runs `nvidia-smi` and lists GPUs 0–7.

## 3. The model

| Item | Value |
|---|---|
| Model | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (original April 2026 release; `-0731` and `V4.1` variants exist but were not tested) |
| Architecture | MoE, 284B total / **13B active**, 43 layers, 256 routed experts (6 per token) + 1 shared |
| Precision as shipped | **FP4 experts (MXFP4) + FP8 for the rest** — no quantization needed |
| Size on disk | **149 GB** |
| Context window | **1,048,576 tokens (1M)** |
| Attention | Hybrid compressed attention (CSA / HCA) + sliding window: KV cache is ~10% of DeepSeek-V3.2 at 1M tokens |
| MTP | 1 built-in multi-token-prediction layer |
| Reasoning | Thinking on by default; vLLM splits it into `reasoning_content` |
| License | MIT |

Why it fits so easily: 149 GB of weights over 768 GB of GPU memory leaves ~600 GB for KV cache.
With TP=8 the server reported room for **6.26 million tokens** of KV cache — almost six 1M-token requests at once.

## 4. Step by step

### 4.1 Access to the cluster

```bash
export KUBECONFIG=~/Downloads/<cluster>-kubeconfig.yaml
kubectl get nodes
```

> **Difficulty:** on the test laptop a corporate proxy resolves the LKE API hostname to a fake IP, and kubectl
> fails with `TLS handshake timeout`. Workaround: resolve the real IP with a public DNS server and pin it, keeping
> the original name for TLS:
> ```bash
> dig +short <id>.us-mia-2-gw.linodelke.net @1.1.1.1
> kubectl --server=https://<IP>:443 --tls-server-name=<id>.us-mia-2-gw.linodelke.net get nodes
> ```

### 4.2 Pick the vLLM version

Community reports from mid-2026 said DeepSeek-V4-Flash did **not** run on RTX PRO 6000 (SM120) with stock vLLM:
DeepGEMM rejected the architecture and people used forks/patches. Before using a fork, the current release was checked:

- `vllm/vllm-openai:v0.32.0` (released 2026-10-18) registers `DeepseekV4ForCausalLM` natively;
- it has an SM120 sparse-MLA backend (`flashinfer_mla_sparse_sm120.py`) and SM12x MXFP4 MoE kernels;
- tokenizer mode, reasoning parser and tool parser `deepseek_v4` are built in.

**Result: v0.32.0 works out of the box on SM120 — no patches, no custom image.** At start-up it selects:

| Component | Backend chosen by vLLM |
|---|---|
| MoE experts (FP4) | `DEEPGEMM_MXFP4` |
| Attention | `FLASHINFER_MLA_SPARSE_DSV4` + `DEEPSEEK_SPARSE_SWA` + `DEEPSEEK_V4_INDEXER` |
| KV cache format | `fp8_ds_mla` (DeepSeek's FP8 MLA format), block size 256 |
| All-reduce (TP) | `PYNCCL` only (see 4.8) |
| Fused mHC kernels | TileLang, JIT-compiled at start-up (~3 min) |

### 4.3 Download the weights once (Job)

[k8s/05-download-model.yaml](../k8s/05-download-model.yaml) downloads the model with `hf download` to a
`hostPath` on the node's local disk (`/var/lib/llm-models`), so every vLLM pod afterwards loads from disk and the
server start-up time is measured separately from the download.

```bash
kubectl apply -f k8s/00-namespace.yaml -f k8s/05-download-model.yaml
kubectl -n llm-bench logs -f job/download-dsv4-flash
# → download finished in 126 s   (149 GB ≈ 1.2 GB/s from Hugging Face)
```

The vLLM image (9.2 GB) was pulled in 79 s.

### 4.4 The server manifests

All configurations use the same Deployment name, so applying one replaces the previous one.

| File | Layout | Notes |
|---|---|---|
| [10-vllm-dsv4-flash-tp8.yaml](../k8s/10-vllm-dsv4-flash-tp8.yaml) | 1 replica, **TP=8** | all 8 GPUs serve one engine |
| [11-vllm-dsv4-flash-tp8-mtp.yaml](../k8s/11-vllm-dsv4-flash-tp8-mtp.yaml) | TP=8 + MTP | prepared, not run (TP=8 lost to 2×TP=4) |
| [12-vllm-dsv4-flash-tp4x2.yaml](../k8s/12-vllm-dsv4-flash-tp4x2.yaml) | **2 replicas × TP=4** | two engines of 4 GPUs |
| [13-vllm-dsv4-flash-tp4x2-mtp.yaml](../k8s/13-vllm-dsv4-flash-tp4x2-mtp.yaml) | 2 × TP=4 + MTP | `--speculative-config={"method":"mtp","num_speculative_tokens":1}` |

Key server flags:

```
deepseek-ai/DeepSeek-V4-Flash
--tensor-parallel-size=8            # or 4 per replica
--max-model-len=1048576             # full 1M window
--kv-cache-dtype=fp8
--tokenizer-mode=deepseek_v4
--reasoning-parser=deepseek_v4
--gpu-memory-utilization=0.90
--no-enable-prefix-caching          # benchmark: every prompt must really be prefilled
```

`HF_HUB_OFFLINE=1` makes vLLM use only the pre-downloaded weights. `/dev/shm` is a 64 GiB memory volume (NCCL).
The startup probe allows 60 minutes.

### 4.5 Start-up

| | TP=8 | 2×TP=4 | 2×TP=4 + MTP |
|---|---:|---:|---:|
| Weight loading | 29 s | 488 s | 550 s |
| Engine init (profiling, KV cache, TileLang JIT, CUDA graphs) | 366 s | 377 s | 399 s |
| **Total until ready** | **~7.5 min** | **~16 min** | **~17 min** |
| Weights per GPU | 19.8 GiB | 37.8 GiB | 38.9 GiB |
| KV cache memory per GPU | 60.0 GiB | 41.9 GiB | — |
| KV cache capacity | **6.26 M tokens** | **4.37 M tokens per replica** (8.7 M total) | 4.19 M per replica |
| 1M-token requests that fit at once | 5.97 | 4.16 per replica | 3.99 per replica |

> **Observation:** weight loading takes ~30 s when one engine starts alone (29 s for TP=8, 33 s for a single TP=4
> replica) but **8–9 min when two replicas start at the same time** — both read the same 149 GB from the node's disk
> in parallel. For production: stagger replica start-ups.

Smoke test:

```bash
kubectl -n llm-bench exec deploy/dsv4-flash -- curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Quanto é 17*23? Responda em uma frase."}],"max_tokens":300}'
# reasoning_content: "...17*20 + 17*3 = 340 + 51 = 391..."   content: "17 vezes 23 é igual a 391."
```

### 4.6 The benchmark

The client is `vllm bench serve`, run **inside the vLLM pod** (same image, no network in between), by
[scripts/bench.sh](../scripts/bench.sh):

- random prompts of a fixed length (`--random-range-ratio 0`), **512 output tokens**, `--ignore-eos`, so every
  request does exactly the same work;
- prefix caching off: every prompt is really prefilled;
- 1 warm-up request before each test; then N requests with a fixed number in flight (`--max-concurrency`);
- metrics: TTFT, TPOT, ITL, E2E at p50 / p90 / p99, plus output tokens/s.

| Set | Tests |
|---|---|
| `main` | 1 user: 8k, 10k, 32k, 60k, 120k · 10k prompt with 2, 4, 8 users |
| `long` | 1 user: 240k, 512k, 1M |
| `ctxload` | 60k and 120k prompts with 2, 4, 8 users |
| `sweep` | 10k prompt with 16, 32, 64, 128 users (throughput plateau) |

For **2×TP=4**, the single-user tests run on one replica (the other idle), and the concurrent tests run on
**both replicas at the same time with half of the users each** (e.g. 64 users = 32 + 32), driven by
[scripts/run-tp4x2.sh](../scripts/run-tp4x2.sh); [scripts/combine_replicas.py](../scripts/combine_replicas.py) sums
throughput and averages latency. GPU power/SM/clock are sampled every 2 s with `nvidia-smi dmon`
([scripts/gpu_stats.py](../scripts/gpu_stats.py)).

```bash
kubectl -n llm-bench cp scripts/bench.sh <pod>:/models/bench.sh
kubectl -n llm-bench exec <pod> -- bash -c '(nohup bash /models/bench.sh tp8 512 main > /models/results/tp8-main.log 2>&1 &)'
python3 scripts/summarize.py results/tp8
```

Results are written to `/models/results/<tag>` on the node (hostPath) and copied into [results/](../results).

### 4.7 MTP needs real text

The first MTP run used the same random-token prompts and MTP came out **slower** (TPOT 11.8 ms vs 10.2 ms). The
server log explains it: `SpecDecoding metrics: ... Per-position acceptance rate: 0.006–0.12`. A draft of random text
is almost never right, so MTP only adds work. The run was stopped and MTP was compared on **real text**:
[Spec-Bench](https://github.com/hemingkx/Spec-Bench) (480 questions, 13 categories) through the chat endpoint,
same seed and questions with and without MTP ([scripts/bench-realtext.sh](../scripts/bench-realtext.sh)):

```bash
curl -sL -o /models/spec_bench.jsonl \
  https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
pip install pandas                      # the vLLM image lacks it; vllm bench needs it for jsonl datasets
bash /models/bench-realtext.sh tp4-mtp-realtext
python3 scripts/summarize_realtext.py results/tp4-realtext results/tp4-mtp-realtext
```

The client needs `--tokenizer-mode deepseek_v4`: DeepSeek-V4 has no Hugging Face `chat_template`; its prompt format
lives in vLLM's `deepseek_v4` tokenizer.

On real text MTP is accepted ~80% of the time (1.8 tokens per step).

### 4.8 Why 2×TP=4 beats TP=8 on this node

1. **No NVLink.** vLLM disables its fast all-reduce paths on this hardware (start-up warnings):
   `SymmMemCommunicator: Device capability 12.0 not supported`,
   `FlashInfer All Reduce is disabled ... world_size=8`,
   `Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs`.
   Every layer's all-reduce goes through NCCL over PCIe.
2. **Two NUMA nodes.** GPUs 0–3 hang off CPU socket 0 and GPUs 4–7 off socket 1. With TP=8 every all-reduce
   crosses the socket interconnect; each TP=4 replica stays inside one socket.
3. **13B active parameters** are small: a 4-GPU slice already has plenty of compute and bandwidth, so splitting
   further mostly adds communication.

Evidence: with TP=8 the GPUs show 100% "busy" but draw only ~270 W of 600 W on average — they are mostly waiting
on communication. Single-user decode is **74 tok/s on TP=8 vs 98 tok/s on one TP=4 replica** (half the GPUs).

## 5. Difficulties and how they were solved

| # | Problem | Cause | Solution |
|---|---|---|---|
| 1 | `kubectl`: `TLS handshake timeout` | Local corporate proxy answers DNS for the LKE API with a fake IP | Pin the real API IP with `--server` + `--tls-server-name` (4.1) |
| 2 | Uncertainty whether V4-Flash runs on SM120 | Mid-2026 reports needed forks/patches (DeepGEMM `Unsupported architecture`) | Checked vLLM v0.32.0: native support; no patches needed |
| 3 | `kubectl logs -f` stopped after 30 s | Client `--request-timeout=30s` also applies to streams | `--request-timeout=0` for log streaming |
| 4 | Benchmarks dying with the laptop connection | Long `kubectl exec` sessions | Run the benchmark with `nohup` **inside** the pod; poll the log file |
| 5 | Running two replicas' tests from inside one pod | Each replica only knows `127.0.0.1` | `BASE` env var in `bench.sh`: replica A drives replica B by pod IP |
| 6 | Slow start of 2 replicas (16–17 min) | Both read 149 GB from local disk at the same time (weight load 8–9 min) | Documented; start replicas one at a time if start-up time matters |
| 7 | TP=8 slower than expected | No NVLink + all-reduce across 2 CPU sockets | Use 2 replicas × TP=4 (4.8) |
| 8 | MTP slower than no MTP | Random-token prompts: draft accepted 1–12% | Measure MTP on real text (Spec-Bench): 80% acceptance, +35–47% (4.7) |
| 9 | `vllm bench` with Spec-Bench: `No module named 'pandas'` | Not in the vLLM image | `pip install pandas` in the pod |
| 10 | `Cannot use chat template functions because tokenizer.chat_template is not set` | DeepSeek-V4 ships its prompt encoder as code, not as an HF template | `--tokenizer-mode deepseek_v4` on the benchmark client |
| 11 | Cleanup: `crictl: No such file or directory` | LKE nodes ship `ctr`, not `crictl` | `ctr -n k8s.io images rm ...` ([k8s/90-cleanup-node.yaml](../k8s/90-cleanup-node.yaml)) |

## 6. Cleanup

Everything created by this test was removed at the end:

```bash
kubectl delete namespace llm-bench          # vLLM deployments, service, download Job
kubectl apply -f k8s/90-cleanup-node.yaml   # privileged pod: rm -rf /var/lib/llm-models (149 GB),
                                            # ctr images rm vllm/vllm-openai:v0.32.0 (9.2 GB),
                                            # and a host timer that removes its own image after it exits
kubectl delete pod cleanup-node
```

Output: [logs/cleanup-node.log](../logs/cleanup-node.log). The cleanup pod reuses an image already on the node
(`nvidia/cuda:12.9.1-base`, left from the GPU validation), and a `systemd-run` timer on the host removes that
image two minutes after the pod is gone. The GPU Operator (namespace `gpu-operator`) belongs to the cluster
setup and was kept.
