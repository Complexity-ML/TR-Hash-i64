#!/usr/bin/env python3
"""Evaluate PIQA choice likelihoods through TR-Hash-i64 decode, eager vs CUDA Graph."""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch

from tr_hash_i64.core.loader import load_model_by_name
from tr_hash_i64.core.tokenizer import I64Tokenizer
from tr_hash_i64.engine.i64_engine import I64Engine


PIQA_ARCHIVE = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"


def _prefill_last_logits(model, model_kwargs, last_indices):
    """Project only final context rows when the model supports selection."""
    if getattr(model, "supports_logits_indices", False):
        logits_indices = torch.tensor(
            last_indices,
            dtype=torch.long,
            device=model_kwargs["token_ids"].device,
        )
        return model(logits_indices=logits_indices, **model_kwargs)
    return model(**model_kwargs)[last_indices]


def load_piqa_validation(cache_dir: Path):
    """Load the official PIQA dev examples without the retired HF dataset script."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "physicaliqa-train-dev.zip"
    if not archive.exists():
        urlretrieve(PIQA_ARCHIVE, archive)
    with ZipFile(archive) as zf:
        inputs = zf.read("physicaliqa-train-dev/dev.jsonl").decode().splitlines()
        labels = zf.read("physicaliqa-train-dev/dev-labels.lst").decode().splitlines()
    rows = []
    for line, label in zip(inputs, labels):
        row = json.loads(line)
        row["label"] = int(label)
        rows.append(row)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 means full validation split")
    p.add_argument("--output", default="piqa_cuda_graph_results.json")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--max-batch-size", type=int, default=64)
    return p.parse_args()


class ChoiceScorer:
    def __init__(self, model_dir: str, dtype: torch.dtype, max_seq_len: int, max_batch_size: int):
        self.model_dir = model_dir
        self.max_batch_size = max_batch_size
        self.tokenizer = I64Tokenizer(str(Path(model_dir) / "tokenizer.json"))
        self.model = load_model_by_name(
            "tr-hash-moe-200m",
            dtype=dtype,
            device="cuda",
            checkpoint_override=model_dir,
            quantization="none",
        ).eval()
        self.model.requires_grad_(False)
        self.engine = I64Engine(
            model=self.model,
            num_experts=self.model.config.num_experts,
            vocab_size=self.model.config.vocab_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_kv_blocks=max(256, max_batch_size * 16),
            enable_prefix_caching=False,
            device="cuda",
        )
        self.engine.warmup_and_capture_graphs()
        if self.engine.cuda_graph_runner is None or not self.engine.cuda_graph_runner.is_captured:
            raise RuntimeError("CUDA Graph capture failed")
        if max_batch_size not in self.engine.cuda_graph_runner._captured_sizes:
            raise RuntimeError(f"batch-{max_batch_size} CUDA Graph was not captured")

    def encode_pair(self, context: str, continuation: str):
        # Match lm-eval's convention: move trailing context whitespace to continuation.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        context = context.rstrip()
        continuation = " " + continuation.strip()
        ctx = self.tokenizer.encode(context)
        whole = self.tokenizer.encode(context + continuation)
        if not ctx or len(whole) <= len(ctx) or whole[: len(ctx)] != ctx:
            # Tokenizer boundary fallback. This is rare, but preserves a non-empty context.
            cont = self.tokenizer.encode(continuation)
            if not ctx:
                ctx = [self.tokenizer.bos_token_id]
            whole = ctx + cont
        return ctx, whole[len(ctx) :]

    @torch.inference_mode()
    def score_batch(self, pairs, use_graph: bool):
        encoded = [self.encode_pair(context, continuation) for context, continuation in pairs]
        if any(not cont for _, cont in encoded):
            raise ValueError("PIQA continuation encoded to zero tokens")
        batch_size = len(encoded)
        cache = self.engine.kv_cache
        for seq_id, (ctx, cont) in enumerate(encoded):
            cache.free_sequence(seq_id)
            total_len = len(ctx) + len(cont)
            cache.allocate_blocks(seq_id, math.ceil(total_len / cache.block_size))

        ctx_lengths = [len(ctx) for ctx, _ in encoded]
        flat_ctx = [token for ctx, _ in encoded for token in ctx]
        flat_pos = [position for ctx, _ in encoded for position in range(len(ctx))]
        last_indices = []
        offset = 0
        for length in ctx_lengths:
            last_indices.append(offset + length - 1)
            offset += length
        logits_by_seq = _prefill_last_logits(
            self.model,
            {
                "token_ids": torch.tensor(flat_ctx, dtype=torch.int64, device="cuda"),
                "positions": torch.tensor(flat_pos, dtype=torch.int32, device="cuda"),
                "kv_cache": cache,
                "seq_ids": list(range(batch_size)),
                "tokens_per_seq": ctx_lengths,
            },
            last_indices,
        )
        scores = torch.zeros(batch_size, dtype=torch.float32, device="cuda")
        cont_lengths = [len(cont) for _, cont in encoded]
        decode_calls = 0
        decode_forward_steps = 0

        for j in range(max(cont_lengths)):
            active = [i for i, length in enumerate(cont_lengths) if j < length]
            active_t = torch.tensor(active, dtype=torch.long, device="cuda")
            targets = torch.tensor(
                [encoded[i][1][j] for i in active], dtype=torch.int64, device="cuda"
            )
            token_logprobs = torch.log_softmax(logits_by_seq[active_t], dim=-1)
            scores.index_add_(0, active_t, token_logprobs[torch.arange(len(active), device="cuda"), targets])

            next_active = [i for i, length in enumerate(cont_lengths) if j + 1 < length]
            if not next_active:
                continue
            token = torch.tensor(
                [encoded[i][1][j] for i in next_active], dtype=torch.int64, device="cuda"
            )
            position = torch.tensor(
                [ctx_lengths[i] + j for i in next_active], dtype=torch.int32, device="cuda"
            )
            if use_graph:
                cache.enter_graph_mode(next_active)
                next_logits = self.engine.cuda_graph_runner.run(
                    token,
                    position,
                    torch.zeros(len(next_active), dtype=torch.int32, device="cuda"),
                )
                cache.exit_graph_mode(seq_ids=next_active)
            else:
                next_logits = self.model.decode_step(
                    token_ids=token,
                    positions=position,
                    kv_cache=cache,
                    seq_ids_tensor=torch.tensor(next_active, dtype=torch.long, device="cuda"),
                )
            next_active_t = torch.tensor(next_active, dtype=torch.long, device="cuda")
            logits_by_seq.index_copy_(0, next_active_t, next_logits)
            decode_calls += len(next_active)
            decode_forward_steps += 1

        values = scores.tolist()
        for seq_id in range(batch_size):
            cache.free_sequence(seq_id)
        return values, cont_lengths, decode_calls, decode_forward_steps


def run_mode(scorer, rows, use_graph: bool):
    name = "cuda_graph" if use_graph else "eager"
    torch.cuda.synchronize()
    started = time.perf_counter()
    correct = 0
    correct_norm = 0
    choice_tokens = 0
    decode_calls = 0
    scores = []
    decode_forward_steps = 0
    example_batch_size = max(1, scorer.max_batch_size // 2)
    for start in range(0, len(rows), example_batch_size):
        chunk = rows[start : start + example_batch_size]
        pairs = [(row["goal"], solution) for row in chunk for solution in (row["sol1"], row["sol2"])]
        batch_scores, lengths, batch_decode_calls, batch_forward_steps = scorer.score_batch(pairs, use_graph)
        for local_idx, row in enumerate(chunk):
            s0, s1 = batch_scores[2 * local_idx : 2 * local_idx + 2]
            n0, n1 = lengths[2 * local_idx : 2 * local_idx + 2]
            label = int(row["label"])
            pred = int(s1 > s0)
            pred_norm = int((s1 / max(n1, 1)) > (s0 / max(n0, 1)))
            correct += pred == label
            correct_norm += pred_norm == label
            choice_tokens += n0 + n1
            scores.append((s0, s1, n0, n1, pred, pred_norm))
        decode_calls += batch_decode_calls
        decode_forward_steps += batch_forward_steps
        done = min(start + example_batch_size, len(rows))
        if done % 104 == 0 or done == len(rows):
            print(f"{name}: {done}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "mode": name,
        "examples": len(rows),
        "accuracy": correct / len(rows),
        "accuracy_norm": correct_norm / len(rows),
        "correct": correct,
        "correct_norm": correct_norm,
        "choice_tokens": choice_tokens,
        "decode_calls": decode_calls,
        "decode_forward_steps": decode_forward_steps,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "choice_tokens_per_second": choice_tokens / elapsed,
        "scores": scores,
    }


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rows = load_piqa_validation(Path.home() / ".cache" / "tr_hash_i64" / "piqa")
    if args.limit > 0:
        rows = rows[: args.limit]
    scorer = ChoiceScorer(args.model_dir, dtype, args.max_seq_len, args.max_batch_size)
    print(
        "model_loaded",
        sum(p.numel() for p in scorer.model.parameters()),
        "graphs",
        sorted(scorer.engine.cuda_graph_runner._captured_sizes),
        "examples",
        len(rows),
        flush=True,
    )

    # A few unmeasured examples populate allocator/cache paths before timing.
    warmup_pairs = [
        (row["goal"], solution)
        for row in rows[: min(8, len(rows))]
        for solution in (row["sol1"], row["sol2"])
    ][: args.max_batch_size]
    scorer.score_batch(warmup_pairs, False)
    scorer.score_batch(warmup_pairs, True)

    eager = run_mode(scorer, rows, False)
    graph = run_mode(scorer, rows, True)
    diffs = [
        max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        for a, b in zip(eager.pop("scores"), graph.pop("scores"))
    ]
    result = {
        "model": args.model_id,
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "max_batch_size": args.max_batch_size,
        "dataset": "ybisk/piqa",
        "split": "validation",
        "cuda": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "graphs": sorted(scorer.engine.cuda_graph_runner._captured_sizes),
        },
        "eager": eager,
        "cuda_graph": graph,
        "speedup": eager["elapsed_seconds"] / graph["elapsed_seconds"],
        "max_abs_choice_score_diff": max(diffs) if diffs else 0.0,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
