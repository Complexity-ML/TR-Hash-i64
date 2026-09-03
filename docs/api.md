# API guide

TR-Hash-i64 exposes OpenAI-compatible completion and chat endpoints plus runtime inspection and control endpoints. The server publishes its generated OpenAPI document at `GET /docs`.

Examples below assume `http://127.0.0.1:7860`.

## Authentication

Start the server with either `--api-key` or the preferred file-based option:

```bash
tr-hash-i64 serve tr-hash-moe-200m \
  --api-key-file /etc/tr-hash-i64/api.key \
  --port 7860
```

Then send:

```text
Authorization: Bearer <key>
```

Use a root-owned mode-`0600` key file for supervised services. Do not put production credentials in command history.

## Chat completions

```bash
curl http://127.0.0.1:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tr-hash-moe-200m",
    "messages": [
      {"role": "system", "content": "Answer precisely."},
      {"role": "user", "content": "What is token routing?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "stream": false
  }'
```

For server-sent events, set `"stream": true`. The stream ends with `data: [DONE]`.

## Text completions

```bash
curl http://127.0.0.1:7860/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tr-hash-moe-200m",
    "prompt": "Deterministic token routing means",
    "max_tokens": 64,
    "temperature": 0.7,
    "seed": 42
  }'
```

Common client sampling fields include `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`, and `logit_bias`.

The server enforces:

```text
prompt_tokens + max_tokens <= max_seq_len
```

## Rolling chat context

Chat requests can compact older turns when the rendered conversation exceeds the configured budget. Start the server with, for example:

```bash
tr-hash-i64 serve tr-hash-moe-200m --context-compact-tokens 1024
```

The deterministic local policy keeps system instructions and recent user turns, summarizes older represented turns, then drops the oldest material if required. It does not make a second model request.

Non-streaming responses include `context_metrics`; streaming responses include the same object in the first event. Set `"context_management": false` on a request to reject an oversized conversation instead of compacting it.

Provide a stable `user` or `conversation_id`, or an `X-Conversation-Id`/`X-Session-Id` header, when cache affinity should follow a conversation. Anonymous chats receive isolated request-local cache scopes.

## Health and discovery

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/live` | Process liveness |
| GET | `/ready` | Model and serving readiness |
| GET | `/health` | General health response |
| GET | `/v1/models` | Metadata for the running model |
| GET | `/v1/models/{id}` | Model details |
| GET | `/docs` | Generated OpenAPI document |

## Generation and utilities

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/completions` | Text completion, optionally SSE |
| POST | `/v1/chat/completions` | Chat completion, optionally SSE |
| GET | `/v1/ws/completions` | WebSocket completion streaming |
| POST | `/v1/batch` | Batch completions |
| POST | `/v1/cancel/{request_id}` | Cancel a running request |
| POST | `/v1/tokenize` | Tokenize text |
| POST | `/v1/embeddings` | Produce text embeddings |

## Monitoring and cache control

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/usage` | Token usage counters |
| GET | `/v1/metrics` | Latency, usage, engine, and context metrics |
| GET | `/v1/monitor` | Live scheduler and device snapshot |
| GET | `/v1/experts` | Expert-routing distribution |
| GET | `/v1/logs` | Recent request records |
| GET | `/v1/cache/stats` | KV/prefix-cache statistics |
| POST | `/v1/cache/purge` | Purge reusable prefix blocks |
| POST | `/v1/priority` | Set API-key priority |

## Optional endpoint groups

These routes are present but require the matching server configuration:

- LoRA: `/v1/lora/load`, `/v1/lora/unload`, `/v1/lora/list`
- RAG with `--rag-index`: `/v1/rag/index`, `/v1/rag/search`, `/v1/rag/stats`
- sandbox with `--sandbox`: `/v1/execute`
- agent events: `/v1/agent/events`, `/v1/agent/history`

Sandbox execution is disabled unless explicitly enabled. Review the isolation settings before exposing it outside a trusted host.

## CORS and errors

CORS support is enabled for browser clients. Production deployments should still restrict network exposure, require authentication, and place TLS at a trusted reverse proxy.

Validation failures return an HTTP error rather than silently clipping invalid generation settings. Queue limits and rate limits can return overload responses before work enters the scheduler.
