"""
Complexity Deep Model for vllm-i64.

Uses vllm-i64 attention backends (naive_varlen, paged KV cache).
Weights loaded from complexity-framework checkpoint format.

Complexity-ML — 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from vllm_i64.layers.token_routed_mlp import TokenRoutedMLP
from vllm_i64.layers.attention import (
    is_flash_attn_available, flash_prefill_attention,
    naive_varlen_attention, naive_cached_attention,
)


# =========================================================================
# RoPE
# =========================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        """positions: [N] integer tensor. Returns cos, sin of shape [N, dim]."""
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        emb = torch.cat([freqs, freqs], dim=-1)  # [N, dim]
        return emb.cos(), emb.sin()


def _rotate_half(x):
    """Rotate half the hidden dims: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x, cos, sin):
    """x: [N, heads, head_dim], cos/sin: [N, head_dim]."""
    cos = cos.to(dtype=x.dtype).unsqueeze(1)  # [N, 1, head_dim]
    sin = sin.to(dtype=x.dtype).unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


# =========================================================================
# Mu-Guidance
# =========================================================================

class MuGuidance(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

    def forward(self, h):
        mu_clamped = torch.clamp(self.mu, 0.0, 2.0)
        return mu_clamped + self.mu_proj(h)


# =========================================================================
# Attention — uses vllm-i64 attention backends
# =========================================================================

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.use_mu_guidance = (
            getattr(config, "use_mu_guidance", False)
            and not getattr(config, "disable_mu_guidance", False)
        )
        if self.use_mu_guidance:
            self.mu_to_q = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )
            self.mu_to_k = nn.Linear(
                self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
            self.mu_to_v = nn.Linear(
                self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )

        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=getattr(config, 'max_position_embeddings', 4096),
            theta=getattr(config, 'rope_theta', 10000.0),
        )

    def fuse_qkv(self) -> bool:
        """Fuse three input projections into one CPU-friendly linear layer."""

        if hasattr(self, "qkv_proj"):
            return False
        q_out = self.q_proj.out_features
        k_out = self.k_proj.out_features
        v_out = self.v_proj.out_features
        fused = nn.Linear(
            self.hidden_size,
            q_out + k_out + v_out,
            bias=False,
            device=self.q_proj.weight.device,
            dtype=self.q_proj.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat(
                    [
                        self.q_proj.weight,
                        self.k_proj.weight,
                        self.v_proj.weight,
                    ],
                    dim=0,
                )
            )
        self.qkv_splits = (q_out, k_out, v_out)
        self.qkv_proj = fused
        del self.q_proj
        del self.k_proj
        del self.v_proj
        return True

    def forward(self, hidden, positions, mu_prev=None,
                kv_cache=None, layer_idx=0,
                seq_ids=None, tokens_per_seq=None, **kwargs):
        """
        Args:
            hidden: [N, hidden_size] (flattened tokens)
            positions: [N] integer positions
            mu_prev: [N, hidden_size] or None
            kv_cache: PagedKVCache or None
            tokens_per_seq: [num_seqs] token counts
        """
        bsz = hidden.shape[0]

        if hasattr(self, "qkv_proj"):
            q, k, v = self.qkv_proj(hidden).split(self.qkv_splits, dim=-1)
        else:
            q = self.q_proj(hidden)
            k = self.k_proj(hidden)
            v = self.v_proj(hidden)

        if self.use_mu_guidance and mu_prev is not None:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        # Reshape: [N, heads, head_dim]
        q = q.view(bsz, self.num_heads, self.head_dim)
        k = k.view(bsz, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        cos, sin = self.rope(positions)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # KV Cache path
        if kv_cache is not None and seq_ids is not None and tokens_per_seq is not None:
            return self._cached_attention(
                q, k, v, kv_cache, layer_idx, seq_ids, tokens_per_seq, positions)

        # Standard prefill (no cache)
        scale = 1.0 / math.sqrt(self.head_dim)
        tps = tokens_per_seq if tokens_per_seq is not None else [bsz]

        if is_flash_attn_available() and q.is_cuda:
            out = flash_prefill_attention(q, k, v, tps, softmax_scale=scale)
        else:
            out = naive_varlen_attention(q, k, v, tps, self.num_kv_groups, softmax_scale=scale)

        out = out.to(q.dtype).reshape(bsz, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def _cached_attention(self, q, k, v, kv_cache, layer_idx, seq_ids, tokens_per_seq, positions):
        """Attention with paged KV cache."""
        bsz = q.shape[0]

        # Write new K/V to cache — each token at its own position
        outputs = []
        offset = 0
        scale = 1.0 / math.sqrt(self.head_dim)

        for i, sid in enumerate(seq_ids):
            n = tokens_per_seq[i]
            seq_positions = positions[offset:offset + n]

            # Write all tokens for this sequence to cache
            for j in range(n):
                pos = seq_positions[j].item()
                kv_cache.write_kv(layer_idx, sid, pos, k[offset + j], v[offset + j])

            # Read full K/V history from cache
            k_full, v_full = kv_cache.read_kv(layer_idx, sid)

            q_i = q[offset:offset + n]
            out_i = naive_cached_attention(
                q_i, k_full, v_full,
                self.num_kv_groups,
                seq_positions,
                softmax_scale=scale,
            )
            outputs.append(out_i)
            offset += n

        out = torch.cat(outputs, dim=0)
        out = out.to(q.dtype).reshape(bsz, self.num_heads * self.head_dim)
        return self.o_proj(out)


# =========================================================================
# Dense SwiGLU MLP
# =========================================================================

class DenseSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x, token_ids=None, **kwargs):
        if hasattr(self, "gate_up_proj"):
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def fuse_gate_up(self) -> bool:
        """Fuse SwiGLU input projections after checkpoint loading."""

        if hasattr(self, "gate_up_proj"):
            return False
        fused = nn.Linear(
            self.gate_proj.in_features,
            self.gate_proj.out_features + self.up_proj.out_features,
            bias=False,
            device=self.gate_proj.weight.device,
            dtype=self.gate_proj.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
            )
        self.gate_up_proj = fused
        del self.gate_proj
        del self.up_proj
        return True


# =========================================================================
# MoE MLP
# =========================================================================

class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tr_mlp = TokenRoutedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            vocab_size=config.vocab_size,
            shared_expert=getattr(config, 'shared_expert', False),
            shared_intermediate_size=getattr(config, 'shared_intermediate_size', None) or 0,
            top_k=getattr(config, 'top_k', 1),
            top_k_primary_weight=getattr(config, 'top_k_primary_weight', None),
            use_shared_routed_gates=getattr(config, 'use_shared_routed_gates', False),
            shared_gate_init=getattr(config, 'shared_gate_init', 1.0),
            routed_gate_init=getattr(config, 'routed_gate_init', 1.0),
        )

    def forward(self, x, token_ids=None, **kwargs):
        return self.tr_mlp(x, token_ids=token_ids)


# =========================================================================
# Decoder Layer
# =========================================================================

class ComplexityDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

        if config.num_experts > 1 and getattr(config, 'use_token_routed_mlp', True):
            # Keep checkpoint keys as ``layers.N.mlp.*``. The old wrapper
            # inserted an artificial ``tr_mlp`` segment and left the real
            # exported expert tensors unloaded.
            self.mlp = TokenRoutedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                vocab_size=config.vocab_size,
                shared_expert=getattr(config, 'shared_expert', False),
                shared_intermediate_size=getattr(
                    config, 'shared_intermediate_size', None
                ) or 0,
                top_k=getattr(config, 'top_k', 1),
                top_k_primary_weight=getattr(
                    config, 'top_k_primary_weight', None
                ),
                use_shared_routed_gates=getattr(
                    config, 'use_shared_routed_gates', False
                ),
                shared_gate_init=getattr(config, 'shared_gate_init', 1.0),
                routed_gate_init=getattr(config, 'routed_gate_init', 1.0),
            )
        else:
            self.mlp = DenseSwiGLUMLP(config)

        self.mu_guidance = None
        if getattr(config, 'use_mu_guidance', False) and not getattr(config, 'disable_mu_guidance', False):
            self.mu_guidance = MuGuidance(config.hidden_size)

    def forward(self, hidden, positions, token_ids=None, mu_prev=None,
                kv_cache=None, layer_idx=0, seq_ids=None, tokens_per_seq=None, **kwargs):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(
            hidden, positions, mu_prev=mu_prev,
            kv_cache=kv_cache, layer_idx=layer_idx,
            seq_ids=seq_ids, tokens_per_seq=tokens_per_seq,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden, token_ids=token_ids)
        hidden = residual + hidden

        mu_current = None
        if self.mu_guidance is not None:
            mu_current = self.mu_guidance(hidden)

        return hidden, mu_current


# =========================================================================
# Complexity Deep Model
# =========================================================================

class ComplexityDeepModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ComplexityDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', True)

        self._has_mu = getattr(config, 'use_mu_guidance', False)
        if self._has_mu and not getattr(config, 'disable_mu_guidance', False):
            self.mu_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, token_ids, positions=None,
                kv_cache=None, seq_ids=None, tokens_per_seq=None, **kwargs):
        """
        Engine-compatible forward.

        Args:
            token_ids: [N] flattened token IDs
            positions: [N] integer positions
            kv_cache: PagedKVCache or None
            seq_ids: list of sequence IDs
            tokens_per_seq: list of token counts per sequence
        """
        if token_ids.dim() == 2:
            # Standalone mode: [batch, seq_len] -> flatten
            batch_size, seq_len = token_ids.shape
            token_ids = token_ids.view(-1)
            if positions is None:
                positions = torch.arange(seq_len, device=token_ids.device).repeat(batch_size)
            if tokens_per_seq is None:
                tokens_per_seq = [seq_len] * batch_size
        elif positions is None:
            positions = torch.arange(token_ids.shape[0], device=token_ids.device)
            if tokens_per_seq is None:
                tokens_per_seq = [token_ids.shape[0]]

        N = token_ids.shape[0]
        hidden = self.embed_tokens(token_ids.long())

        mu_prev = None
        if self._has_mu and hasattr(self, 'mu_init'):
            mu_prev = self.mu_init.view(1, -1).expand(N, -1)

        for i, layer in enumerate(self.layers):
            hidden, mu_current = layer(
                hidden, positions,
                token_ids=token_ids,
                mu_prev=mu_prev,
                kv_cache=kv_cache,
                layer_idx=i,
                seq_ids=seq_ids,
                tokens_per_seq=tokens_per_seq,
            )
            if mu_current is not None:
                mu_prev = torch.clamp(mu_current, -2.0, 2.0)

        hidden = self.norm(hidden)

        if self.tie_word_embeddings:
            logits = F.linear(hidden.float(), self.embed_tokens.weight.float())
        else:
            logits = self.lm_head(hidden)

        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def load_framework_checkpoint(self, state_dict: dict):
        """Load weights from complexity-framework checkpoint format."""
        mapped = {}
        for k, v in state_dict.items():
            new_key = k
            # rotary_emb -> rope
            new_key = new_key.replace("self_attn.rotary_emb.", "self_attn.rope.")
            mapped[new_key] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        return missing, unexpected
