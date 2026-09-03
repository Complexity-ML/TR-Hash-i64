"""Benchmark the production CPU prefill path on a real checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from tr_hash_i64.core.loader import load_model_by_name


def checkpoint_sha256(model_dir: Path) -> str:
    """Hash all safetensors files, including names and sizes, in stable order."""
    digest = hashlib.sha256()
    paths = sorted(model_dir.rglob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors files found under {model_dir}")
    for path in paths:
        relative = path.relative_to(model_dir).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    weights_sha256 = checkpoint_sha256(model_dir)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = load_model_by_name(
        "tr-hash-moe-200m",
        dtype=torch.float32,
        device="cpu",
        checkpoint_override=str(model_dir),
        quantization=None if args.quantization == "none" else args.quantization,
    ).eval()
    vocab_size = int(model.config.vocab_size)
    token_ids = (torch.arange(args.prompt_len, dtype=torch.long) * 17 + 3) % vocab_size
    positions = torch.arange(args.prompt_len, dtype=torch.int32)
    last_index = torch.tensor([args.prompt_len - 1], dtype=torch.long)

    def run() -> torch.Tensor:
        kwargs = {
            "token_ids": token_ids,
            "positions": positions,
            "kv_cache": None,
            "seq_ids": None,
            "tokens_per_seq": [args.prompt_len],
        }
        if getattr(model, "supports_logits_indices", False):
            kwargs["logits_indices"] = last_index
        with torch.no_grad():
            logits = model(**kwargs)
        return logits[-1]

    for _ in range(2):
        run()
    trials = []
    reference = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run()
        trials.append((time.perf_counter() - start) * 1000.0)
        if reference is None:
            reference = result.clone()

    assert reference is not None
    payload = {
        "model": args.model_id,
        "model_dir": str(model_dir),
        "checkpoint_sha256": weights_sha256,
        "device": "cpu",
        "quantization": args.quantization,
        "threads": args.threads,
        "prompt_len": args.prompt_len,
        "repeats": args.repeats,
        "supports_logits_indices": bool(
            getattr(model, "supports_logits_indices", False)
        ),
        "median_ms": statistics.median(trials),
        "trials_ms": trials,
        "last_logits_sum": float(reference.float().sum().item()),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
