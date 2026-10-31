# DeepSeek-V4-Flash on Akamai Cloud (LKE) — 8× RTX PRO 6000 Blackwell

Benchmark of [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
(284B-parameter MoE, 13B active, FP4 experts + FP8, 1M-token context) served with vLLM on one
Akamai Cloud GPU node with 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each).

Work in progress — results and the full guide are added as the tests run.

| Folder | Content |
|---|---|
| `k8s/` | Kubernetes manifests (namespace, model download Job, vLLM Deployments) |
| `scripts/` | Benchmark runner and result summarizers |
| `results/` | Raw `vllm bench serve` JSON per test |
| `logs/` | Server start-up logs and errors hit along the way |
| `docs/` | Step-by-step guide and results analysis |
