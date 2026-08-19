# TR-Hash-i64

An independent inference server for deterministic token-routed models from
Complexity-ML -- not a fork of, or affiliated with, the vLLM project. The
Python package and CLI command are still named `tr_hash_i64` / `tr-hash-i64`
internally (see Install/Serve below); only this repository's name has
changed so far.

- `tr-hash-moe-500m` → [`Pacific-i64/TR-HASH-MOE-500M-HF`](https://huggingface.co/Pacific-i64/TR-HASH-MOE-500M-HF) — the only model with a live public endpoint right now.

The 500M runtime loads its layer-specific balanced hash tables exactly, uses
top-2 0.5/0.5 routing and applies the trained shared/routed output scales.

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
