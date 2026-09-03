# Benchmarks and evidence

Benchmark claims in this repository are tied to raw result files. A configuration or enabled flag is not evidence that a kernel ran or that performance improved.

The current checked-in evidence uses:

- model: `AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT`
- checkpoint manifest SHA-256: `fb66baf5bb567eb5a73ce7e53b5d351a0963356808d41af904d776d50428e327`
- host label: `tr-hash-server`

## CPU production prefill

Reproduce the production model path with:

```bash
python benchmarks/bench_cpu_prefill.py \
  --model-dir /path/to/TR-HASH-MoE-200M-160B-SFT \
  --model-id AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT \
  --quantization int8 \
  --threads 8 \
  --prompt-len 128 \
  --repeats 9 \
  --output cpu-prefill.json
```

The script performs two fixed warmup passes before the measured repetitions.

Checked-in matched result:

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Median prefill | 158.799 ms | 143.404 ms |
| Relative latency | — | −9.69% |
| Throughput ratio | 1.000× | 1.107× |
| Final-logit sum | −151061.375 | −151061.375 |

The measured production gain comes from projecting only the final hidden row through the LM head during prefill. Dynamic INT8 keeps independent top-k route calls because grouping changes activation quantization scales.

Evidence:

- `benchmarks/results/cpu_sft_int8_prefill_2026-09-03.json`
- `benchmarks/results/raw/cpu_sft_int8_production_before.json`
- `benchmarks/results/raw/cpu_sft_int8_production_rebased_after.json`

## CUDA Graph PIQA scorer

Run the eager and CUDA Graph scorer under one process:

```bash
python benchmarks/bench_piqa_cuda_graph.py \
  --model-dir /path/to/TR-HASH-MoE-200M-160B-SFT \
  --model-id AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT \
  --limit 16 \
  --max-seq-len 256 \
  --max-batch-size 16 \
  --output piqa.json
```

The recorded hardware was an NVIDIA GeForce RTX 5060 Ti at a 150 W limit, PyTorch `2.11.0+cu128`, CUDA runtime 12.8, compute capability 12.0.

A matched three-trial comparison of the isolated dense top-k dispatch change reported:

| Metric | Baseline median | Optimized median |
| --- | ---: | ---: |
| CUDA Graph elapsed time | 3.2874 s | 3.1619 s |
| Choice-token throughput | 440.47 token/s | 457.95 token/s |
| Throughput change | — | +3.97% |

Accuracy, normalized accuracy, and eager-versus-graph score difference were unchanged in those trials.

This claim is scoped to the recorded source hashes in the result manifest. The current tree includes later graph-safe integration changes; its checked-in four-example run is a functional post-rebase smoke, not a rerun of the three-trial performance comparison.

Evidence:

- `benchmarks/results/rtx5060ti_sft_dense_topk_2026-09-03.json`
- `benchmarks/results/raw/piqa_sft_baseline_run*.json`
- `benchmarks/results/raw/piqa_sft_optimized_run*.json`
- `benchmarks/results/raw/piqa_sft_final_rebased_cuda_smoke.json`

## Interpreting results

For a before/after comparison, keep all of these fixed:

- physical host and selected device;
- GPU power limit and driver/runtime versions;
- exact checkpoint and manifest hash;
- dtype and quantization mode;
- prompt, sequence length, batch size, and request count;
- CPU thread count;
- warmup and measured-run counts;
- CUDA Graph or eager execution policy.

Report isolated-operation improvements separately from end-to-end serving throughput. A smoke run validates execution and numerical agreement but does not establish a stable performance gain.

## Other benchmark entry points

| Script | Purpose |
| --- | --- |
| `benchmarks/bench_e2e.py` | End-to-end generation |
| `benchmarks/bench_engine.py` | Engine-level throughput |
| `benchmarks/bench_i64_routing.py` | Routing lookup behavior |
| `benchmarks/bench_comparative.py` | Comparative runtime checks |
| `benchmarks/bench_piqa_teacher_forced.py` | Teacher-forced PIQA scoring |

Read each script's `--help` before running it. Keep raw JSON outputs when publishing derived aggregates.
