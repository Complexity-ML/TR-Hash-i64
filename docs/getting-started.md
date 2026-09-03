# Getting started

TR-Hash-i64 serves deterministic token-routed causal language models on CUDA, Apple MPS, and CPU. This guide starts the current 201.2M SFT release locally.

## Requirements

- Python 3.10–3.12
- PyTorch 2.0 or newer
- Git
- For CUDA: a PyTorch build compatible with the installed NVIDIA driver

Create an isolated environment and install the engine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Complexity-ML/TR-Hash-i64.git@main"
```

For reproducible deployment, replace `main` with a full commit SHA.

## Check the installation

```bash
tr-hash-i64 list
tr-hash-i64 --help
```

`list` prints the built-in model registry. The first `serve` downloads the selected Hugging Face snapshot when necessary.

## Start the current SFT model

### CUDA

```bash
CUDA_VISIBLE_DEVICES=0 tr-hash-i64 serve tr-hash-moe-200m \
  --device cuda \
  --host 127.0.0.1 \
  --port 7860 \
  --max-batch-size 16
```

CUDA Graphs are enabled by default. Use `--no-cuda-graphs` only when capture is unsupported or the extra static buffers do not fit. An explicit `--device cuda` fails instead of silently falling back to another device.

### CPU with dynamic INT8

```bash
TR_HASH_I64_CPU_THREADS=8 tr-hash-i64 serve tr-hash-moe-200m \
  --device cpu \
  --host 127.0.0.1 \
  --port 7860 \
  --quantization int8 \
  --max-batch-size 4 \
  --max-kv-blocks 128
```

Dynamic INT8 packs linear layers with PyTorch's x86 quantization backend. The deterministic routing tables remain integer lookup tables. AWQ and GPTQ are for compatible pre-quantized checkpoints, not runtime quantization of the public SFT release.

### Apple silicon

```bash
tr-hash-i64 serve tr-hash-moe-200m \
  --device mps \
  --host 127.0.0.1 \
  --port 7860
```

MPS uses the same engine interface and a static traced decode runner where supported.

## Use a local checkpoint

A local Hugging Face-format directory can override the registered snapshot:

```bash
tr-hash-i64 serve tr-hash-moe-200m \
  --checkpoint /models/TR-HASH-MoE-200M-160B-SFT \
  --device cuda \
  --port 7860
```

For checkpoints that contain full `topk_token_to_expert` tables, the loader reads those routes verbatim. Loader diagnostics report missing, unexpected, and unloaded tensors. Legacy checkpoints that contain only a primary `token_to_expert` table retain their compatibility conversion; see [Architecture](architecture.md).

## Confirm readiness

Model loading can take time. Admit traffic only after readiness succeeds:

```bash
curl --fail http://127.0.0.1:7860/live
curl --fail http://127.0.0.1:7860/ready
curl http://127.0.0.1:7860/v1/models
```

`/live` reports process liveness. `/ready` remains unavailable until the model and serving loop are ready.

## First chat request

```bash
curl http://127.0.0.1:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tr-hash-moe-200m",
    "messages": [{"role": "user", "content": "Explain deterministic token routing."}],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": false
  }'
```

See [API guide](api.md) for streaming, authentication, context management, and the complete endpoint map.

## Useful capacity controls

| Option | Purpose |
| --- | --- |
| `--max-batch-size` | Maximum concurrent scheduler batch |
| `--chunk-size` | Maximum tokens in a chunked-prefill step |
| `--max-kv-blocks` | Paged KV cache capacity; `0` selects automatically |
| `--no-prefix-caching` | Disable reusable prompt-prefix blocks |
| `--kv-cache-dtype fp8` | Reduce supported CUDA KV-cache storage |
| `--max-pending` | Reject requests above a bounded queue depth |
| `--rate-limit` | Requests per minute per client IP; `0` disables it |

Inspect the authoritative options for the installed version with:

```bash
tr-hash-i64 serve --help
```

## Next steps

- [Architecture](architecture.md)
- [API guide](api.md)
- [Production operations](operations.md)
- [Benchmarks and evidence](benchmarks.md)
