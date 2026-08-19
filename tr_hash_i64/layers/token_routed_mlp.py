"""Deterministic token-routed SwiGLU used by the 306.5M TR-MoE model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from tr_hash_i64.parallel.tensor_parallel import get_tp, all_reduce
class TokenRoutedMLP(nn.Module):
    """
    Generic token-routed MLP with TP support and Shared Lexical Expert.

    Routing (i64, replicated on all ranks):
        expert_id = topk_token_to_expert[route_idx, token_id]
        Each route_idx is an independent hash channel computed at training
        time (multi-hash rendezvous routing) and loaded verbatim from the
        checkpoint — never recomputed at inference.

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
        top_k: int = 1,
        top_k_primary_weight: Optional[float] = None,
        use_shared_routed_gates: bool = False,
        shared_gate_init: float = 1.0,
        routed_gate_init: float = 1.0,
        shared_output_scale: float = 1.0,
        routed_output_scale: float = 1.0,
    ):
        super().__init__()
        tp = get_tp()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.tp_size = tp.tp_size
        self.tp_rank = tp.tp_rank
        self.top_k = max(1, int(top_k))
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})"
            )
        if top_k_primary_weight is None:
            top_k_primary_weight = 0.95
        self.primary_weight = (
            min(1.0, max(0.0, float(top_k_primary_weight)))
            if self.top_k > 1 else 1.0
        )
        self.shared_output_scale = float(shared_output_scale)
        self.routed_output_scale = float(routed_output_scale)

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
        self.use_shared_routed_gates = bool(use_shared_routed_gates)
        if shared_expert:
            shared_size = shared_intermediate_size if shared_intermediate_size > 0 else self.full_expert_inter
            self.shared_gate = nn.Linear(hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, hidden_size, bias=False)
            if self.use_shared_routed_gates:
                self.shared_output_gate = nn.Parameter(
                    torch.tensor(float(shared_gate_init))
                )
                self.routed_output_gate = nn.Parameter(
                    torch.tensor(float(routed_gate_init))
                )

        # The exported checkpoint stores the primary route. Additional routes
        # are derived cyclically, exactly as in the paper evaluation runtime.
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )
        self.register_buffer(
            "topk_token_to_expert",
            torch.stack(
                [
                    (torch.arange(vocab_size, dtype=torch.long) + route_idx)
                    % num_experts
                    for route_idx in range(self.top_k)
                ]
            ),
        )

        nn.init.kaiming_uniform_(self.gate_proj_w, a=5**0.5)
        nn.init.kaiming_uniform_(self.up_proj_w, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj_w, a=5**0.5)

    def route(
        self,
        token_ids: Optional[torch.Tensor],
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a ``[top_k, num_tokens]`` integer expert lookup."""
        if token_ids is None:
            primary = torch.zeros(num_tokens, dtype=torch.long, device=device)
        else:
            token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
            return self.topk_token_to_expert[:, token_ids_clamped]
        return torch.stack(
            [
                (primary + route_idx) % self.num_experts
                for route_idx in range(self.top_k)
            ]
        )

    def expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Sparse dispatch with loop — matches framework supplementary code."""
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            return self._dense_expert_forward(x, expert_ids)

        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (expert_ids == e)
            if not mask.any():
                continue
            x_e = x[mask]
            if hasattr(self, "expert_gate_up"):
                gate_e, up_e = self.expert_gate_up[e](x_e).chunk(2, dim=-1)
                inter_e = F.silu(gate_e) * up_e
                output[mask] = self.expert_down[e](inter_e).to(output.dtype)
                continue
            if hasattr(self, "gate_up_proj_w"):
                gate_e, up_e = (x_e @ self.gate_up_proj_w[e]).chunk(2, dim=-1)
            else:
                gate_e = x_e @ self.gate_proj_w[e]
                up_e = x_e @ self.up_proj_w[e]
            inter_e = F.silu(gate_e) * up_e
            output[mask] = (inter_e @ self.down_proj_w[e]).to(output.dtype)
        return all_reduce(output)

    def _dense_expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Graph-safe expert dispatch: every expert computes densely for all
        tokens, then results are masked-accumulated with a float multiply.

        expert_forward's `x[mask]` / `output[mask] = ...` gather-scatter has
        a token count that depends on routing values, so its output shape
        isn't fixed across replays — incompatible with CUDA graph capture.
        This path avoids that by never changing tensor shapes, at the cost
        of computing every expert for every token (fine for decode's small
        per-step batches and small num_experts).
        """
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            if hasattr(self, "expert_gate_up"):
                gate_e, up_e = self.expert_gate_up[e](x).chunk(2, dim=-1)
                out_e = self.expert_down[e](F.silu(gate_e) * up_e)
            elif hasattr(self, "gate_up_proj_w"):
                gate_e, up_e = (x @ self.gate_up_proj_w[e]).chunk(2, dim=-1)
                out_e = (F.silu(gate_e) * up_e) @ self.down_proj_w[e]
            else:
                gate_e = x @ self.gate_proj_w[e]
                up_e = x @ self.up_proj_w[e]
                out_e = (F.silu(gate_e) * up_e) @ self.down_proj_w[e]
            mask = (expert_ids == e).unsqueeze(-1).to(output.dtype)
            output = output + out_e.to(output.dtype) * mask
        return all_reduce(output)

    def forward(self, x, token_ids=None, **kwargs):
        routes = self.route(token_ids, x.shape[0], x.device)
        output = self.primary_weight * self.expert_forward(x, routes[0])
        if self.top_k > 1:
            secondary_weight = (1.0 - self.primary_weight) / (self.top_k - 1)
            for route_idx in range(1, self.top_k):
                output = output + secondary_weight * self.expert_forward(
                    x, routes[route_idx]
                )

        # Shared expert: dense SwiGLU applied to all tokens
        if self.use_shared_expert:
            if hasattr(self, "shared_gate_up"):
                shared_gate, shared_up = self.shared_gate_up(x).chunk(2, dim=-1)
            else:
                shared_gate = self.shared_gate(x)
                shared_up = self.shared_up(x)
            shared_out = self.shared_down(
                F.silu(shared_gate) * shared_up
            ).to(output.dtype)
            if self.use_shared_routed_gates:
                output = (
                    self.shared_output_gate * shared_out
                    + self.routed_output_gate * output
                )
            else:
                output = (
                    self.routed_output_scale * output
                    + self.shared_output_scale * shared_out
                )
        else:
            output = self.routed_output_scale * output

        return output

    def fuse_shared_gate_up(self) -> bool:
        """Fuse the shared SwiGLU input projections after weight loading."""

        if not self.use_shared_expert or hasattr(self, "shared_gate_up"):
            return False
        fused = nn.Linear(
            self.shared_gate.in_features,
            self.shared_gate.out_features + self.shared_up.out_features,
            bias=False,
            device=self.shared_gate.weight.device,
            dtype=self.shared_gate.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat([self.shared_gate.weight, self.shared_up.weight], dim=0)
            )
        self.shared_gate_up = fused
        del self.shared_gate
        del self.shared_up
        return True

    def materialize_expert_linears(self) -> bool:
        """Convert raw expert matrices to fuseable/quantizable linear modules."""

        if hasattr(self, "expert_gate_up"):
            return False
        gate_up_modules = []
        down_modules = []
        for expert_idx in range(self.num_experts):
            gate_up = nn.Linear(
                self.hidden_size,
                2 * self.expert_inter,
                bias=False,
                device=self.gate_proj_w.device,
                dtype=self.gate_proj_w.dtype,
            )
            down = nn.Linear(
                self.expert_inter,
                self.hidden_size,
                bias=False,
                device=self.down_proj_w.device,
                dtype=self.down_proj_w.dtype,
            )
            with torch.no_grad():
                gate_up.weight.copy_(
                    torch.cat(
                        [
                            self.gate_proj_w[expert_idx].T,
                            self.up_proj_w[expert_idx].T,
                        ],
                        dim=0,
                    )
                )
                down.weight.copy_(self.down_proj_w[expert_idx].T)
            gate_up_modules.append(gate_up)
            down_modules.append(down)
        self.expert_gate_up = nn.ModuleList(gate_up_modules)
        self.expert_down = nn.ModuleList(down_modules)
        del self.gate_proj_w
        del self.up_proj_w
        del self.down_proj_w
        return True

    def load_full_weights(self, full_gate: torch.Tensor, full_up: torch.Tensor, full_down: torch.Tensor):
        """Load unsharded expert weights and take this TP rank's slice."""
        start = self.tp_rank * self.expert_inter
        end = start + self.expert_inter
        self.gate_proj_w.data.copy_(full_gate[:, :, start:end])
        self.up_proj_w.data.copy_(full_up[:, :, start:end])
        self.down_proj_w.data.copy_(full_down[:, start:end, :])
