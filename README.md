# TR-Hash-i64

An independent inference server for deterministic token-routed models from
Complexity-ML -- not a fork of, or affiliated with, the vLLM project. The
Python package and CLI command are still named `tr_hash_i64` / `tr-hash-i64`
internally (see Install/Serve below); only this repository's name has
changed so far.

- `tr-hash-moe-200m` → [`AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT`](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT) — the current public chat release and live endpoint.
- `tr-hash-moe-500m` → [`Pacific-i64/TR-HASH-MOE-500M-HF`](https://huggingface.co/Pacific-i64/TR-HASH-MOE-500M-HF) — the earlier research release.

The runtime loads each release's persisted layer-specific multi-hash tables
exactly, uses deterministic top-2 routing and applies the trained shared/routed
output scales.

The earlier 306.5M routed/dense comparison pair (`tr-moe-306` →
[`Pacific-i64/TR-MOE-306`](https://huggingface.co/Pacific-i64/TR-MOE-306),
`dense-306` → [`Pacific-i64/Dense-306`](https://huggingface.co/Pacific-i64/Dense-306))
is no longer served in production, but the checkpoints stay on the Hub and
both names remain registered — `tr-hash-i64 serve tr-moe-306` /
`... serve dense-306` still work for local/self-hosted use.

### Routing

Every expert assignment is a lookup into `topk_token_to_expert`, a
`[top_k, vocab]` table of independent hash channels computed at training
time (multi-hash rendezvous routing) and loaded verbatim from the checkpoint
— nothing is recomputed or approximated at inference. On GPU, decode uses a
CUDA-graph-safe path (`decode_step`): KV cache writes/reads and expert
dispatch are tensor-only, with no per-token `.item()` calls, so the whole
decode step can be captured once and replayed with near-zero launch overhead.



## Install

```bash
pip install git+https://github.com/Complexity-ML/TR-Hash-i64.git@main
```

## Serve

The model snapshot is downloaded automatically from Hugging Face:

```bash
tr-hash-i64 serve tr-hash-moe-500m \
  --host 0.0.0.0 \
  --port 7860 \
  --quantization none
```

Use `tr-hash-moe-200m` for the current full-SFT assistant.

Use `dense-306` for the matched dense baseline. A local directory can replace
the Hub snapshot:

```bash
tr-hash-i64 serve dense-306 \
  --checkpoint /models/Dense-306 \
  --port 7860
```

For a Linux x86 CPU deployment, dynamic INT8 packs every `nn.Linear` weight
with the PyTorch x86/FBGEMM backend while leaving the token-routing tables as
integers:

```bash
TR_HASH_I64_CPU_THREADS=8 \
tr-hash-i64 serve tr-moe-306 \
  --port 7860 \
  --quantization int8 \
  --max-batch-size 4 \
  --max-kv-blocks 128
```

The CPU engine includes continuous batching, a paged KV cache, prefix caching,
request streaming and queue backpressure. The same command automatically uses
CUDA when a GPU is available.

## Supervised production service

TR-Hash-i64 can install the complete inference process group as a Supervisor
service. Supervisor does not accelerate a matrix multiplication by itself. It
keeps the tuned server warm and makes the operational performance settings
durable: CUDA Graphs remain enabled, prefix caching stays active, the selected
GPUs remain pinned, and a failed TP/PP group is restarted as one unit.

Keep API credentials in a root-owned mode-`0600` file rather than in the
process command line:

```bash
sudo install -d -m 700 /etc/tr-hash-i64
sudo install -m 600 /dev/null /etc/tr-hash-i64/api.key
sudoedit /etc/tr-hash-i64/api.key

sudo tr-hash-i64 service install public-demo tr-hash-moe-200m \
  --checkpoint /models/TR-HASH-MoE-200M-160B-SFT \
  --directory /opt/TR-Hash-i64 \
  --host 0.0.0.0 \
  --port 7860 \
  --devices 0 \
  --api-key-file /etc/tr-hash-i64/api.key \
  --max-batch-size 32 \
  --chunk-size 512 \
  --max-kv-blocks 512 \
  --max-pending 128
```

Lifecycle and logs are available through one small command surface:

```bash
tr-hash-i64 service list
tr-hash-i64 service status public-demo
tr-hash-i64 service restart public-demo
tr-hash-i64 service logs public-demo --follow
sudo tr-hash-i64 service remove public-demo
```

The generated Supervisor definition uses `autorestart=unexpected`,
`stopasgroup=true`, `killasgroup=true`, private configuration permissions and
rotated combined logs. Distributed launch disables rank-local torchrun
restarts, so Supervisor never leaves a partially replaced TP/PP group behind.
Use `GET /live` for process liveness and `GET /ready` before admitting traffic;
readiness is withdrawn while the model is loading or the server is draining.

## OpenAI-compatible API

```bash
curl http://127.0.0.1:7860/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tr-hash-moe-500m",
    "prompt": "The meaning of life is",
    "max_tokens": 64,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "stream": false
  }'
```

`top_k`, `top_p` and `repetition_penalty` are supported by both completion
endpoints and are applied per request after temperature scaling.

Useful endpoints:

- `GET /health`
- `GET /live`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/metrics`
- `GET /v1/monitor`
- `GET /v1/experts`

CORS is enabled so the two Hugging Face Space endpoints can be called by the
Complexity website.

### Rolling chat context

`POST /v1/chat/completions` manages long conversations automatically. Before
generation, the server measures the fully rendered prompt with the model's
tokenizer and enforces:

```text
prompt_tokens + max_tokens <= max_seq_len
```

When the conversation does not fit, the server keeps system instructions and
the two newest user turns, converts older turns into a deterministic extractive
summary, then removes the oldest unrepresented messages. An oversized essential
message is reduced to a head-and-tail view as a final fallback. This processing
is local and does not trigger a second model request.

Use `--context-compact-tokens N` to compact earlier than the physical context
limit. For example, `--context-compact-tokens 1024` starts rolling compaction
once the rendered chat prompt exceeds 1,024 tokens while preserving the
separate output-token reservation.

Every non-streaming chat response includes `context_metrics`; streaming
responses expose the same object in the first SSE event:

```json
{
  "context_metrics": {
    "policy": "rolling_summary",
    "compressed": true,
    "original_tokens": 3184,
    "prompt_tokens": 1792,
    "summary_tokens": 143,
    "tokens_saved": 1392,
    "retained_messages": 5,
    "summarized_messages": 4,
    "dropped_messages": 9
  }
}
```

Aggregate measurements are available under `context` in `GET /v1/metrics` and
`GET /v1/monitor`. Send `"context_management": false` to disable compression
for a request; an over-budget request is then rejected instead of shortened.
Raw completions, batches and WebSocket completions always receive the same
exact total-token validation.

## Verify

```bash
tr-hash-i64 list
python -m pytest -q
```

The loader reports missing and unloaded tensors. Release validation uses a
strict load plus a real cached generation for both 306.5M checkpoints.
