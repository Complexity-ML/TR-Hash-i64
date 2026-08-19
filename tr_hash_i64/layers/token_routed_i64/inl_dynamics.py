# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Mu-Guidance — cross-layer top-down equilibrium signal.

Provides a learnable mu vector with contextual projection that biases
attention and routing in downstream layers.

GitHub: https://github.com/Complexity-ML/complexity-deep
"""

import torch
import torch.nn as nn


class MuGuidance(nn.Module):
    """
    Mu-Guidance — learnable equilibrium with contextual projection.

    Receives full hidden_size tensors (post all-reduce from attention).
    Weights are replicated across TP ranks.
    """

    def __init__(self, hidden_size: int, mu_min: float = 0.0, mu_max: float = 2.0):
        super().__init__()
        self.mu = nn.Parameter(torch.full((hidden_size,), (mu_min + mu_max) / 2))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)
        self.mu_min = mu_min
        self.mu_max = mu_max

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mu_clamped = torch.clamp(self.mu, self.mu_min, self.mu_max)
        return mu_clamped + self.mu_proj(hidden_states)
