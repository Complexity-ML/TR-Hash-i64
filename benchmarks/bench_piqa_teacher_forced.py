#!/usr/bin/env python3
"""PIQA causal-choice evaluation matching the MLX teacher-forcing protocol."""

import argparse
import json
import time
from pathlib import Path

import torch

from bench_piqa_cuda_graph import load_piqa_validation
from tr_hash_i64.core.loader import load_model_by_name
from tr_hash_i64.core.tokenizer import I64Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="piqa_teacher_forced_results.json")
    return parser.parse_args()


def encode_choices(tokenizer, rows, max_length):
    encoded = []
    raw = tokenizer.tokenizer
    for example_index, row in enumerate(rows):
        context_ids = raw.encode(row["goal"], add_special_tokens=False).ids
        for choice_index, solution in enumerate((row["sol1"], row["sol2"])):
            continuation_ids = raw.encode(
                " " + solution.lstrip(), add_special_tokens=False
            ).ids
            ids = context_ids + continuation_ids
            if len(ids) > max_length:
                removed = len(ids) - max_length
                ids = ids[removed:]
                completion_start = len(context_ids) - removed
            else:
                completion_start = len(context_ids)
            if completion_start < 1 or not continuation_ids:
                raise ValueError(f"invalid scoring boundary for PIQA row {example_index}")
            encoded.append(
                {
                    "example_index": example_index,
                    "choice_index": choice_index,
                    "ids": ids,
                    "completion_start": completion_start,
                }
            )
    encoded.sort(key=lambda item: len(item["ids"]))
    return encoded


@torch.inference_mode()
def main():
    args = parse_args()
    rows = load_piqa_validation(Path("/workspace/datasets/piqa"))
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = I64Tokenizer(str(Path(args.model_dir) / "tokenizer.json"))
    model = load_model_by_name(
        "tr-hash-moe-200m",
        dtype=torch.float16,
        device="cuda",
        checkpoint_override=args.model_dir,
        quantization="none",
    ).eval()
    model.requires_grad_(False)
    encoded = encode_choices(tokenizer, rows, args.max_length)
    scores = [[None, None] for _ in rows]
    pad_id = tokenizer.pad_token_id

    # Warm the exact equal-width SDPA + LM-head path used below.
    warm_width = max(len(item["ids"]) for item in encoded[: args.batch_size])
    warm = torch.full(
        (len(encoded[: args.batch_size]), warm_width),
        pad_id,
        dtype=torch.int64,
        device="cuda",
    )
    for i, item in enumerate(encoded[: args.batch_size]):
        warm[i, : len(item["ids"])] = torch.tensor(item["ids"], device="cuda")
    model(warm)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for offset in range(0, len(encoded), args.batch_size):
        batch = encoded[offset : offset + args.batch_size]
        width = max(len(item["ids"]) for item in batch)
        tokens = torch.full(
            (len(batch), width), pad_id, dtype=torch.int64, device="cuda"
        )
        for i, item in enumerate(batch):
            tokens[i, : len(item["ids"])] = torch.tensor(
                item["ids"], dtype=torch.int64, device="cuda"
            )

        logits = model(tokens).reshape(len(batch), width, -1)[:, :-1]
        labels = tokens[:, 1:]
        selected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        log_probs = selected - torch.logsumexp(logits, dim=-1)
        positions = torch.arange(1, width, device="cuda").unsqueeze(0)
        starts = torch.tensor(
            [item["completion_start"] for item in batch], device="cuda"
        ).unsqueeze(1)
        lengths = torch.tensor(
            [len(item["ids"]) for item in batch], device="cuda"
        ).unsqueeze(1)
        mask = (positions >= starts) & (positions < lengths)
        totals = torch.where(mask, log_probs, 0.0).sum(dim=1)
        counts = mask.sum(dim=1)
        normalized = totals / counts
        torch.cuda.synchronize()

        for item, total, norm in zip(
            batch, totals.tolist(), normalized.tolist(), strict=True
        ):
            scores[item["example_index"]][item["choice_index"]] = {
                "loglikelihood": total,
                "loglikelihood_normalized": norm,
            }
        completed = offset + len(batch)
        if completed == len(encoded) or completed % max(args.batch_size, 400) < args.batch_size:
            elapsed = time.perf_counter() - started
            print(
                f"scored {completed}/{len(encoded)} choices "
                f"({completed / elapsed:.1f} choices/s)",
                flush=True,
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    raw_correct = 0
    norm_correct = 0
    for row, pair in zip(rows, scores, strict=True):
        label = int(row["label"])
        raw_correct += int(pair[1]["loglikelihood"] > pair[0]["loglikelihood"]) == label
        norm_correct += int(
            pair[1]["loglikelihood_normalized"]
            > pair[0]["loglikelihood_normalized"]
        ) == label

    result = {
        "model": "AETHORIA-AI/TR-HASH-200M-130B",
        "dataset": "ybisk/piqa",
        "split": "validation",
        "protocol": "causal_choice_loglikelihood_teacher_forcing",
        "batch_size": args.batch_size,
        "examples": len(rows),
        "choices": len(encoded),
        "correct": raw_correct,
        "accuracy": raw_correct / len(rows),
        "correct_norm": norm_correct,
        "accuracy_norm": norm_correct / len(rows),
        "elapsed_seconds": elapsed,
        "choices_per_second": len(encoded) / elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
