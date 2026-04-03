"""
vllm-i64 :: Token-Routed MLP (Generic)

Pure i64 deterministic expert routing. Model-agnostic.
Supports Tensor Parallelism: experts sharded on intermediate dim.
Supports Shared Lexical Expert: dense SwiGLU applied to all tokens.

Integer:
  - Routing: token_id → expert_id via Zipf-balanced mapping (replicated, all ranks)
  - Scatter/gather: argsort indices

Float:
  - Expert SwiGLU compute only
  - Shared expert SwiGLU (all tokens)

Complexity-ML - 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from vllm_i64.parallel.tensor_parallel import get_tp, all_reduce
from vllm_i64.kernels.fused_experts import (
    fused_token_routed_forward,
    fused_token_routed_forward_int8,
    fused_token_routed_forward_int4,
)


class TokenRoutedMLP(nn.Module):
    """
    Generic token-routed MLP with TP support and Shared Lexical Expert.

    Routing (i64, replicated on all ranks):
        expert_id = token_to_expert[token_id]  (Zipf-balanced or modulo)

    Expert compute (float, sharded across TP ranks):
        gate_up: (E, hidden, 2 * inter_per_tp) — ColumnParallel
        down:    (E, inter_per_tp, hidden)      — RowParallel + all_reduce

    Shared expert (float, all tokens):
        shared_gate/shared_up/shared_down — dense SwiGLU
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
        shared_expert: bool = False,
        shared_intermediate_size: int = 0,
    ):
        super().__init__()
        tp = get_tp()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.tp_size = tp.tp_size

        self.full_expert_inter = intermediate_size // num_experts
        self.expert_inter = self.full_expert_inter // tp.tp_size

        # Expert weights — separate gate/up (matches framework checkpoint format)
        self.gate_proj_w = nn.Parameter(
            torch.empty(num_experts, hidden_size, self.expert_inter)
        )
        self.up_proj_w = nn.Parameter(
            torch.empty(num_experts, hidden_size, self.expert_inter)
        )
        self.down_proj_w = nn.Parameter(
            torch.empty(num_experts, self.expert_inter, hidden_size)
        )

        # Shared Lexical Expert: dense SwiGLU applied to all tokens
        self.use_shared_expert = shared_expert
        if shared_expert:
            shared_size = shared_intermediate_size if shared_intermediate_size > 0 else self.full_expert_inter
            self.shared_gate = nn.Linear(hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, hidden_size, bias=False)

        # i64 routing table (replicated — cheap)
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        nn.init.kaiming_uniform_(self.gate_proj_w, a=5**0.5)
        nn.init.kaiming_uniform_(self.up_proj_w, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj_w, a=5**0.5)

    def route(self, token_ids: Optional[torch.Tensor], num_tokens: int, device: torch.device) -> torch.Tensor:
        """Pure i64 routing. Override in subclasses."""
        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=device)
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        return self.token_to_expert[token_ids_clamped]

    def expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Sparse dispatch with loop — matches framework supplementary code."""
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (expert_ids == e)
            if not mask.any():
                continue
            x_e = x[mask]
            gate_e = x_e @ self.gate_proj_w[e]
            up_e = x_e @ self.up_proj_w[e]
            inter_e = F.silu(gate_e) * up_e
            output[mask] = (inter_e @ self.down_proj_w[e]).to(output.dtype)
        return all_reduce(output)

    def forward(self, x, token_ids=None, **kwargs):
        expert_ids = self.route(token_ids, x.shape[0], x.device)
        output = self.expert_forward(x, expert_ids)

        # Shared expert: dense SwiGLU applied to all tokens
        if self.use_shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(x)) * self.shared_up(x)
            ).to(output.dtype)
            output = output + shared_out

        return output

    def load_full_weights(self, full_gate: torch.Tensor, full_up: torch.Tensor, full_down: torch.Tensor):
        """Load from unsharded checkpoint."""
        self.gate_proj_w.data.copy_(full_gate)
        self.up_proj_w.data.copy_(full_up)
        self.down_proj_w.data.copy_(full_down)
