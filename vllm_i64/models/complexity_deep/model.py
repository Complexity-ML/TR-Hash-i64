"""
Complexity Deep Model for vllm-i64.

Rewritten to match the complexity-framework forward pass exactly.
Supports both Token-Routed MoE and Dense SwiGLU models.

Complexity-ML — 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from vllm_i64.layers.token_routed_mlp import TokenRoutedMLP


# =========================================================================
# RoPE
# =========================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. q,k: [batch, heads, seq, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim/2]
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rotate(q, cos, sin), rotate(k, cos, sin)


# =========================================================================
# Mu-Guidance
# =========================================================================

class MuGuidance(nn.Module):
    """Mu-Guidance — matches framework MuProjection."""
    def __init__(self, hidden_size):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

    def forward(self, h):
        return torch.clamp(self.mu + self.mu_proj(h), -2.0, 2.0)


# =========================================================================
# GQA Attention — matches framework gqa.py
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

        self.mu_to_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.mu_to_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.mu_to_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=getattr(config, 'max_position_embeddings', 4096),
            theta=getattr(config, 'rope_theta', 10000.0),
        )

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, mu_prev=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if mu_prev is not None:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)

        if past_key_value is not None:
            cos = cos[kv_seq_len - seq_len:]
            sin = sin[kv_seq_len - seq_len:]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_past_key_value = (k, v) if use_cache else None

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past_key_value is None and seq_len > 1),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_past_key_value


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
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =========================================================================
# MoE MLP — wraps TokenRoutedMLP for [batch, seq, hidden] format
# =========================================================================

class MoEMLP(nn.Module):
    """Wraps TokenRoutedMLP to handle [batch, seq, hidden] input."""
    def __init__(self, config):
        super().__init__()
        self.tr_mlp = TokenRoutedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            vocab_size=config.vocab_size,
            shared_expert=getattr(config, 'shared_expert', False),
            shared_intermediate_size=getattr(config, 'shared_intermediate_size', None) or 0,
        )

    def forward(self, x, token_ids=None, **kwargs):
        B, S, H = x.shape
        flat_x = x.view(-1, H)
        flat_ids = token_ids.view(-1) if token_ids is not None else None
        out = self.tr_mlp(flat_x, token_ids=flat_ids)
        return out.view(B, S, H)


# =========================================================================
# Decoder Layer — matches framework block.py
# =========================================================================

class ComplexityDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

        if config.num_experts > 1 and getattr(config, 'use_token_routed_mlp', True):
            self.mlp = MoEMLP(config)
        else:
            self.mlp = DenseSwiGLUMLP(config)

        self.mu_guidance = None
        if getattr(config, 'use_mu_guidance', False) and not getattr(config, 'disable_mu_guidance', False):
            self.mu_guidance = MuGuidance(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, token_ids=None, mu_prev=None, **kwargs):
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states, attention_mask=attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
            mu_prev=mu_prev,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        hidden_states = residual + hidden_states

        # Mu-Guidance after MLP
        mu_current = None
        if self.mu_guidance is not None:
            mu_current = self.mu_guidance(hidden_states)

        return hidden_states, new_kv, mu_current


# =========================================================================
# Complexity Deep Model — matches framework builder.py
# =========================================================================

class ComplexityDeepModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ComplexityDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', True)

        # Mu init
        self._has_mu = getattr(config, 'use_mu_guidance', False)
        if self._has_mu and not getattr(config, 'disable_mu_guidance', False):
            self.mu_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, input_ids=None, positions=None, token_ids=None,
                kv_cache=None, seq_ids=None, tokens_per_seq=None, **kwargs):
        """
        Args:
            input_ids: [batch, seq_len] or [N] (flattened)
            token_ids: alias for input_ids (used by I64Engine)
            positions: not used (RoPE computed from KV cache length)
            kv_cache: list of (k, v) per layer, or None
            seq_ids: not used
            tokens_per_seq: not used
        """
        if input_ids is None:
            input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Mu init
        mu_prev = None
        if self._has_mu and hasattr(self, 'mu_init'):
            mu_prev = self.mu_init.expand(batch_size, seq_len, -1)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            past_kv = None
            if kv_cache is not None and i < len(kv_cache):
                past_kv = kv_cache[i]

            hidden_states, new_kv, mu_current = layer(
                hidden_states,
                past_key_value=past_kv,
                use_cache=True,
                token_ids=input_ids,
                mu_prev=mu_prev,
            )
            new_kv_cache.append(new_kv)
            if mu_current is not None:
                mu_prev = mu_current

        # Store KV cache for next call
        self._kv_cache = new_kv_cache

        hidden_states = self.norm(hidden_states)

        # Logits
        if self.tie_word_embeddings:
            logits = F.linear(hidden_states.float(), self.embed_tokens.weight.float())
        else:
            logits = self.lm_head(hidden_states)

        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def load_framework_checkpoint(self, state_dict: dict):
        """Load weights from complexity-framework checkpoint format."""
        mapped = {}
        for k, v in state_dict.items():
            new_key = k
            # MoE MLP: gate_proj_w -> mlp.tr_mlp.gate_proj_w
            for proj in ("gate_proj_w", "up_proj_w", "down_proj_w", "token_to_expert"):
                if f"mlp.{proj}" in new_key and "tr_mlp" not in new_key:
                    new_key = new_key.replace(f"mlp.{proj}", f"mlp.tr_mlp.{proj}")
                    break
            # Shared expert: shared_gate -> mlp.tr_mlp.shared_gate
            for proj in ("shared_gate", "shared_up", "shared_down"):
                if f"mlp.{proj}" in new_key and "tr_mlp" not in new_key:
                    new_key = new_key.replace(f"mlp.{proj}", f"mlp.tr_mlp.{proj}")
                    break
            # mu_guidance -> mu_guidance (same name, no change needed)
            # rotary_emb -> rotary_emb (same name)
            mapped[new_key] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        return missing, unexpected
