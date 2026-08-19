"""
tr-hash-i64 :: Integer Softmax

Fixed-point INT32 softmax LUT, plus the re-exported integer SiLU LUT
(actual model routing lives in layers/token_routed_mlp.py — deterministic
hash / multi-hash lookup against the checkpoint's route table, not a
learned gate or a modulo formula).

Integer softmax (fixed-point):
    logits_i32 = round(logits * 2^15)           # Q15 fixed-point
    shifted = logits_i32 - max(logits_i32)       # numerical stability
    exp_i32 = lut_exp(shifted)                   # lookup table exp
    weights_i32 = exp_i32 / sum(exp_i32)         # integer division
    Same top-k selection, same expert dispatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Fixed-point integer softmax
# =========================================================================
#
# Q7 input quantization (scale=128), 1025-entry LUT for exp().
# Covers exp(-8.0) to exp(0.0). exp(-8) ≈ 0.0003 — negligible for softmax.
# Output scaled to Q16 (65536) for integer precision during normalization.

_Q_IN = 128             # 2^7 — input quantization scale
_Q_OUT = 1 << 16        # 2^16 — output scale for exp LUT values
_LUT_MIN = -1024        # minimum shifted value (= -8.0 in float at Q7)
_LUT_SIZE = -_LUT_MIN + 1  # 1025 entries: [-1024, 0]

def _build_exp_lut() -> torch.Tensor:
    """Build exp() LUT: integer index in [-1024, 0] → exp(index/128) * 2^16."""
    indices = torch.arange(_LUT_MIN, 1, dtype=torch.float32)
    return (torch.exp(indices / _Q_IN) * _Q_OUT).to(torch.int32)

_EXP_LUT = _build_exp_lut()


def softmax_integer(logits: torch.Tensor) -> torch.Tensor:
    """
    Fixed-point INT32 softmax — drop-in replacement for F.softmax(x, dim=-1).

    1. Quantize float logits to Q7 (×128 → INT32)
    2. Subtract row-max for stability (all values ≤ 0)
    3. Clamp to [-1024, 0] — below that, exp() ≈ 0
    4. exp() via 1025-entry LUT (Q16 output)
    5. Normalize: weight_i = exp_i / sum(exp)
    6. Return float (experts still compute in float)
    """
    # Q7 quantization — float32 for precision (bf16/fp16 mantissa too short)
    logits_i32 = (logits.float() * _Q_IN).round().to(torch.int32)

    # Subtract row max → all values ≤ 0
    row_max = logits_i32.max(dim=-1, keepdim=True).values
    shifted = logits_i32 - row_max

    # Clamp to LUT range — values below -1024 map to exp(-8)≈0
    shifted = shifted.clamp(min=_LUT_MIN)

    # LUT lookup: index = shifted - LUT_MIN maps [-1024,0] → [0,1024]
    lut = _EXP_LUT.to(shifted.device)
    table_idx = (shifted - _LUT_MIN).long()
    exp_vals = lut[table_idx]  # INT32, Q16 scaled

    # Normalize — integer division then back to float
    exp_sum = exp_vals.sum(dim=-1, keepdim=True).clamp(min=1)
    weights = exp_vals.float() / exp_sum.float()

    return weights


# =========================================================================
# Fixed-point SiLU LUT — re-exported from integer_activations
# =========================================================================

from tr_hash_i64.layers.integer_activations import silu_integer, silu_multiply_integer
