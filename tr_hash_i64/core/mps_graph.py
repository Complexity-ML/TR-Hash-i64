"""
tr-hash-i64 :: MPS "Graph" (TorchScript trace/freeze)

MPS has no CUDA-graph-equivalent capture/replay API (torch.mps exposes no
`graph`/`CUDAGraph`). This gets the same practical goal -- amortize the
per-op Python/Metal dispatch overhead that dominates small-batch decode --
by freezing the decode step into a static TorchScript graph per batch size
instead. Same public interface as CUDAGraphRunner (capture/capture_common_
sizes/run/is_captured) so the engine can treat them interchangeably.

Only safe because decode_step's dispatch is already forced onto the
tensor-only, fixed-shape path on MPS (naive_paged_decode_attention /
TokenRoutedMLP.expert_forward) -- tracing the old .item()-driven eager path
bakes in whatever data-dependent branch happened to fire at trace time,
which is wrong on later replays with different sequence lengths.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Set


class _ForwardFnModule(nn.Module):
    """torch.jit.freeze requires a ScriptModule; tracing a bare closure
    doesn't produce one, so wrap it in a minimal nn.Module first."""

    def __init__(self, forward_fn: Callable):
        super().__init__()
        self._forward_fn = forward_fn

    def forward(self, token_ids, positions, expert_ids):
        return self._forward_fn(token_ids, positions, expert_ids)


class MPSGraphRunner:
    """
    Freezes and replays decode steps as TorchScript graphs, per batch size.

    Usage:
        runner = MPSGraphRunner(model_forward, max_batch=64)
        runner.capture_common_sizes()
        output = runner.run(real_input)
    """

    def __init__(
        self,
        forward_fn: Callable,
        max_batch_size: int = 64,
        device: str = "mps",
    ):
        self.forward_fn = forward_fn
        self.max_batch_size = max_batch_size
        self.device = device

        self.graphs: Dict[int, torch.jit.ScriptModule] = {}
        self.static_inputs: Dict[int, dict] = {}
        self._captured_sizes: Set[int] = set()
        self._wrapper = _ForwardFnModule(forward_fn).eval()

    def capture(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
    ):
        """Trace and freeze the forward pass for the given batch size."""
        if token_ids.device.type != "mps":
            raise ValueError("MPS graph requires MPS tensors")
        bs = token_ids.shape[0]

        static_in = {
            "token_ids": token_ids.clone(),
            "positions": positions.clone(),
            "expert_ids": expert_ids.clone(),
        }

        with torch.no_grad():
            for _ in range(3):
                self.forward_fn(
                    static_in["token_ids"],
                    static_in["positions"],
                    static_in["expert_ids"],
                )
            torch.mps.synchronize()

            traced = torch.jit.trace(
                self._wrapper,
                (static_in["token_ids"], static_in["positions"], static_in["expert_ids"]),
                check_trace=False,
            )
            traced = torch.jit.freeze(traced)
            for _ in range(3):
                traced(static_in["token_ids"], static_in["positions"], static_in["expert_ids"])
            torch.mps.synchronize()

        self.graphs[bs] = traced
        self.static_inputs[bs] = static_in
        self._captured_sizes.add(bs)

    def capture_common_sizes(self):
        """Trace graphs for common batch sizes up to max_batch_size."""
        sizes = [bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= self.max_batch_size]
        for bs in sizes:
            token_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
            positions = torch.zeros(bs, dtype=torch.int32, device=self.device)
            expert_ids = torch.zeros(bs, dtype=torch.int32, device=self.device)
            self.capture(token_ids, positions, expert_ids)

    def _find_best_size(self, batch_size: int) -> Optional[int]:
        """Find the smallest captured size >= batch_size."""
        candidates = [s for s in self._captured_sizes if s >= batch_size]
        return min(candidates) if candidates else None

    def run(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run with the best matching traced graph.

        Pads to the nearest captured size, runs the frozen graph, slices
        the output. Falls back to direct forward if no suitable graph exists.
        """
        batch_size = token_ids.shape[0]
        graph_size = self._find_best_size(batch_size)

        if graph_size is None:
            return self.forward_fn(token_ids, positions, expert_ids)

        static_in = self.static_inputs[graph_size]

        static_in["token_ids"].zero_()
        static_in["positions"].zero_()
        static_in["expert_ids"].zero_()

        static_in["token_ids"][:batch_size].copy_(token_ids)
        static_in["positions"][:batch_size].copy_(positions)
        static_in["expert_ids"][:batch_size].copy_(expert_ids)

        with torch.no_grad():
            out = self.graphs[graph_size](
                static_in["token_ids"], static_in["positions"], static_in["expert_ids"]
            )

        return out[:batch_size].clone()

    @property
    def is_captured(self) -> bool:
        return len(self._captured_sizes) > 0
