# TR-Hash-i64

[![CI](https://github.com/Complexity-ML/TR-Hash-i64/actions/workflows/ci.yml/badge.svg)](https://github.com/Complexity-ML/TR-Hash-i64/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](pyproject.toml)

An independent inference engine and OpenAI-compatible server for Complexity-ML's deterministic token-routed language models. For the current SFT release, TR-Hash-i64 loads the complete persisted top-k routing tables verbatim and serves the model on CUDA, Apple MPS, or CPU.

> TR-Hash-i64 is not a vLLM fork and is not affiliated with the vLLM project.

## Why this runtime exists

A TR-Hash layer does not run a learned router at inference time. Every expert assignment is an integer lookup into a persisted `[top_k, vocab_size]` table:

```text
expert_ids = topk_token_to_expert[:, token_id]
```

That contract shapes the runtime:

- exact loading of the current SFT checkpoint's layer-specific multi-hash route tables;
- deterministic top-2 expert dispatch;
- continuous batching with a paged KV cache and prefix reuse;
- selective LM-head projection during prefill;
- tensor-only CUDA Graph decode for common batch sizes;
- a dedicated CPU path with safe dynamic INT8 behavior;
- OpenAI-compatible HTTP, SSE, and WebSocket interfaces.

Training and checkpoint creation remain outside this repository. This project is the production inference boundary.

## Models

| CLI name | Checkpoint | Parameters | Role |
| --- | --- | ---: | --- |
| `tr-hash-moe-200m` | [`AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT`](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT) | 201.2M | Current public SFT assistant |
| `tr-hash-moe-500m` | [`Pacific-i64/TR-HASH-MOE-500M-HF`](https://huggingface.co/Pacific-i64/TR-HASH-MOE-500M-HF) | 492.1M | Earlier research release |
| `tr-moe-306` | [`Pacific-i64/TR-MOE-306`](https://huggingface.co/Pacific-i64/TR-MOE-306) | 306.5M | Routed comparison checkpoint |
| `dense-306` | [`Pacific-i64/Dense-306`](https://huggingface.co/Pacific-i64/Dense-306) | 306.5M | Width-matched dense baseline |

## Quick start

TR-Hash-i64 requires Python 3.10–3.12 and PyTorch 2.0 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Complexity-ML/TR-Hash-i64.git@main"
```

Start the current SFT release on CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 tr-hash-i64 serve tr-hash-moe-200m \
  --device cuda \
  --host 127.0.0.1 \
  --port 7860 \
  --max-batch-size 16
```

Or run dynamic INT8 on an x86 CPU:

```bash
TR_HASH_I64_CPU_THREADS=8 tr-hash-i64 serve tr-hash-moe-200m \
  --device cpu \
  --host 127.0.0.1 \
  --port 7860 \
  --quantization int8 \
  --max-batch-size 4 \
  --max-kv-blocks 128
```

Wait for readiness, then send a chat request:

```bash
curl --fail http://127.0.0.1:7860/ready

curl http://127.0.0.1:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tr-hash-moe-200m",
    "messages": [{"role": "user", "content": "Explain deterministic token routing."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

The model snapshot is downloaded from Hugging Face on first use. Pass `--checkpoint /path/to/model` to use a local Hugging Face-format directory.

## Runtime features

| Area | Support |
| --- | --- |
| Scheduling | Continuous batching, chunked prefill, queue backpressure, cancellation |
| KV state | Paged KV cache, prefix caching, copy-on-write, optional supported FP8 cache |
| Accelerators | CUDA eager, CUDA Graphs, Apple MPS traced decode, explicit no-fallback device selection |
| CPU | Dedicated async engine, dynamic INT8, selective prefill logits |
| Quantization | Dynamic INT8/INT4 and compatible pre-quantized AWQ/GPTQ checkpoints |
| Parallelism | Tensor parallel, pipeline parallel, optional disaggregated prefill/decode |
| Serving | OpenAI-compatible completions/chat, SSE, WebSocket, batching, monitoring |
| Extensions | LoRA, RAG endpoints, optional sandbox and agent event APIs |
| Operations | Supervisor service manager, profiles, watchdog, diagnostics, atomic upgrade/rollback |

CUDA Graphs are enabled by default on CUDA and retain eager fallback for unsupported shapes. Explicit device choices fail rather than silently switching to another backend.

## Documentation

- [Getting started](docs/getting-started.md) — install, CUDA/CPU/MPS serving, local checkpoints
- [Architecture](docs/architecture.md) — routing contract, prefill, decode, KV cache, fusion
- [API guide](docs/api.md) — completions, chat, streaming, authentication, endpoint map
- [Production operations](docs/operations.md) — Supervisor, profiles, watchdog, upgrades
- [Benchmarks and evidence](docs/benchmarks.md) — protocols, provenance, raw results

The running server also exposes its generated OpenAPI document at `GET /docs`.

## Measured results

Results below use `AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT` and are backed by checked-in JSON artifacts.

### CPU prefill

On the production CPU path with dynamic INT8, 8 threads, and a 128-token prompt:

| Baseline | Optimized | Change |
| ---: | ---: | ---: |
| 158.799 ms | 143.404 ms | **−9.69% latency** |

The final-logit sum was identical. See [`benchmarks/results/cpu_sft_int8_prefill_2026-09-03.json`](benchmarks/results/cpu_sft_int8_prefill_2026-09-03.json).

### Dense top-k CUDA dispatch

On an RTX 5060 Ti at 150 W, a matched three-trial comparison measured median CUDA Graph choice-token throughput of 440.47 token/s before and 457.95 token/s after the isolated dense top-k dispatch change: **+3.97%**.

This number is scoped to the source hashes and protocol in [`benchmarks/results/rtx5060ti_sft_dense_topk_2026-09-03.json`](benchmarks/results/rtx5060ti_sft_dense_topk_2026-09-03.json). Later current-tree validation is recorded as a functional smoke, not as a new matched performance claim.

## Production service

For a durable Supervisor-managed deployment:

```bash
sudo tr-hash-i64 service install public-demo tr-hash-moe-200m \
  --checkpoint /models/TR-HASH-MoE-200M-160B-SFT \
  --directory /opt/TR-Hash-i64 \
  --host 0.0.0.0 \
  --port 7860 \
  --devices 0 \
  --api-key-file /etc/tr-hash-i64/api.key \
  --profile balanced
```

```bash
tr-hash-i64 service status public-demo
tr-hash-i64 service doctor public-demo
tr-hash-i64 service logs public-demo -f
```

See [Production operations](docs/operations.md) before exposing a service publicly. Host-level eGPU recovery and stable UUID/power policy live in the separate [TR-Hash-Server](https://github.com/Complexity-ML/tr-hash-server) project.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Useful CLI discovery:

```bash
tr-hash-i64 --help
tr-hash-i64 serve --help
tr-hash-i64 service --help
tr-hash-i64 list
```

Hardware-specific CUDA validation and raw benchmark results are tracked separately under `benchmarks/results/`.

## License

Apache-2.0, as declared in [`pyproject.toml`](pyproject.toml).
