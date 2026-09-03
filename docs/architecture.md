# Architecture

TR-Hash-i64 is the production inference runtime for Complexity-ML's deterministic token-routed models. It owns checkpoint loading, KV-cache state, scheduling, generation, and serving; model training remains outside this repository.

## Model contract

The current public release is:

- CLI name: `tr-hash-moe-200m`
- checkpoint: `AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT`
- parameters: 201.2M
- architecture: grouped-query attention, shared SwiGLU path, and fixed top-2 token-routed experts

In this release, each routed layer stores `topk_token_to_expert` as a `[top_k, vocab_size]` integer table. For a token ID, routing is a lookup into each persisted channel:

```text
expert_ids = topk_token_to_expert[:, token_id]
```

The current SFT routes were produced during training and are loaded verbatim. The inference runtime does not train a router or score experts. This is the main semantic difference from a learned-gating MoE server.

Legacy compatibility checkpoints may contain only a primary `token_to_expert` table; for those checkpoints, the loader derives additional cyclic routes. Dense checkpoints have no routed experts. Neither case should be described as the current release's persisted multi-hash contract.

## Request path

```text
HTTP / WebSocket
       │
       ▼
  API validation ── tokenizer / chat template / context budget
       │
       ▼
 continuous-batching scheduler
       │
       ├── prefill ── chunking ── paged KV writes
       │
       └── decode  ── cached attention ── token-table expert dispatch
                                      │
                                      ▼
                              sampling / streaming
```

The scheduler batches requests without requiring identical prompt or output lengths. KV blocks are allocated per sequence and released on completion, cancellation, timeout, or engine failure.

## Prefill

Prefill supports variable prompt lengths and chunking. The engine projects only the final hidden row needed for sampling rather than materializing vocabulary logits for every prompt token. The same selective projection is used by the main engine, the CPU engine, and the disaggregated prefill worker.

Requests that cannot obtain a KV slot are removed before token, position, sequence, and logits-index tensors are built. This keeps batch metadata aligned. Multimodal `pixel_values` are forwarded on the prefill that consumes them.

## Decode and CUDA Graphs

CUDA decode uses fixed-shape buffers captured for common batch sizes. Graph-safe mode selects tensor-only implementations during both warmup and capture, including:

- KV-cache updates and reads;
- RMS normalization;
- deterministic expert dispatch;
- sampling inputs and output buffers.

The warmup and capture paths use the same dispatch contract. Unsupported shapes retain eager execution rather than changing model semantics.

For top-k routed layers, the dense graph-safe path computes each expert output once per batch and reuses it for every route assigned to that expert. This removes repeated expert launches while preserving the weighted route merge.

## CPU engine

`CPUEngine` shares scheduling, paged KV cache, sampling, cancellation, and streaming behavior with the main engine but executes model steps on CPU without CUDA dependencies. The async wrapper runs blocking model steps in an executor so the event loop can continue handling clients.

Compatible CPU top-k batches group routes by expert. Dynamically quantized INT8 layers keep independent route calls because changing the activation group changes quantization scales and can change outputs.

## Checkpoint loading and projection fusion

The loader supports native Hugging Face-style exports and validates tensor coverage. Compatible non-quantized projections may be fused at load time:

- query, key, and value projections;
- gate and up projections.

AWQ and GPTQ layouts are excluded from the standard fusion path. CPU post-load quantization preserves its required fusion/quantization ordering.

## KV cache and prefix reuse

The paged KV cache separates logical sequence positions from physical blocks. The block pool provides allocation, reference counting, prefix reuse, copy-on-write for shared blocks, and LRU reclamation.

Prefix-cache namespaces can include the API key plus a conversation or user identifier. This prevents unrelated tenants or chats from sharing cache entries solely because their rendered prompt prefixes match.

## Parallel execution

The CLI exposes tensor parallelism (`--tp`), pipeline parallelism (`--pp`), and an optional disaggregated prefill/decode mode. Distributed workers still load the same persisted route tables; parallelism changes execution placement, not routing decisions.

## Runtime boundaries

- TR-Hash-i64 is an independent runtime, not a vLLM fork or affiliated project.
- Registered dense checkpoints use the same serving interfaces but do not gain token routing.
- CUDA Graphs, MPS tracing, quantization, and projection fusion are execution optimizations. None may synthesize or alter persisted expert routes.
- Configuration alone is not benchmark evidence. Numerical claims in this repository link to raw result files and an explicit protocol.

See [Benchmarks and evidence](benchmarks.md) for measured results and their scope.
