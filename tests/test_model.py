"""
tr-hash-i64 :: Test Model Forward Pass

Tests ComplexityDeepModel end-to-end on CPU:
  - Correct output shapes
  - Attention with GQA + RoPE + QK Norm
  - TokenRoutedMLP routing (MoE)
  - DenseSwiGLUMLP (dense baseline)
  - Full forward: token_ids → logits

Complexity-ML — 2026
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tr_hash_i64.models.complexity_deep.config import ComplexityDeepConfig
from tr_hash_i64.models.complexity_deep.model import (
    ComplexityDeepModel,
    ComplexityDecoderLayer,
    Attention,
    DenseSwiGLUMLP,
    RotaryEmbedding,
    apply_rotary,
)


# ── Fixtures ──

@pytest.fixture
def moe_config():
    return ComplexityDeepConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        num_experts=4,
        vocab_size=256,
        max_position_embeddings=128,
        use_token_routed_mlp=True,
        shared_expert=True,
        use_qk_norm=True,
    )


@pytest.fixture
def dense_config():
    return ComplexityDeepConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        num_experts=1,
        vocab_size=256,
        max_position_embeddings=128,
        use_token_routed_mlp=False,
        shared_expert=False,
        use_qk_norm=True,
    )


@pytest.fixture
def moe_model(moe_config):
    m = ComplexityDeepModel(moe_config)
    m.eval()
    return m


@pytest.fixture
def dense_model(dense_config):
    m = ComplexityDeepModel(dense_config)
    m.eval()
    return m


# ── RoPE ──

class TestRotaryEmbedding:
    def test_output_shapes(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=128)
        cos, sin = rope(torch.arange(10))
        assert cos.shape == (10, 32)
        assert sin.shape == (10, 32)

    def test_apply_rotary(self):
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(torch.arange(8))
        q = torch.randn(8, 4, 32)
        k = torch.randn(8, 2, 32)
        q_rot = apply_rotary(q, cos, sin)
        k_rot = apply_rotary(k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ── Attention ──

class TestAttention:
    def test_output_shape(self, moe_config):
        attn = Attention(moe_config)
        attn.eval()
        N = 20  # flattened tokens (e.g. 2 seqs of 10)
        x = torch.randn(N, 128)
        positions = torch.arange(N)
        with torch.no_grad():
            out = attn(x, positions, tokens_per_seq=[10, 10])
        assert out.shape == (N, 128)

    def test_single_token(self, moe_config):
        attn = Attention(moe_config)
        attn.eval()
        x = torch.randn(1, 128)
        positions = torch.tensor([0])
        with torch.no_grad():
            out = attn(x, positions, tokens_per_seq=[1])
        assert out.shape == (1, 128)


# ── Dense MLP ──

class TestDenseSwiGLUMLP:
    def test_output_shape(self, dense_config):
        mlp = DenseSwiGLUMLP(dense_config)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == (2, 10, 128)

    def test_gate_up_fusion_preserves_output(self, dense_config):
        mlp = DenseSwiGLUMLP(dense_config).eval()
        x = torch.randn(7, 128)
        with torch.inference_mode():
            expected = mlp(x)
            assert mlp.fuse_gate_up() is True
            actual = mlp(x)
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


# ── Decoder Layer ──

class TestDecoderLayer:
    def test_moe_layer(self, moe_config):
        layer = ComplexityDecoderLayer(moe_config)
        layer.eval()
        N = 8
        x = torch.randn(N, 128)
        positions = torch.arange(N)
        token_ids = torch.randint(0, 256, (N,))
        with torch.no_grad():
            out = layer(x, positions, token_ids=token_ids, tokens_per_seq=[N])
        assert out.shape == (N, 128)

    def test_dense_layer(self, dense_config):
        layer = ComplexityDecoderLayer(dense_config)
        layer.eval()
        N = 8
        x = torch.randn(N, 128)
        positions = torch.arange(N)
        with torch.no_grad():
            out = layer(x, positions, tokens_per_seq=[N])
        assert out.shape == (N, 128)


# ── Full Model ──

class TestComplexityDeepModel:
    def test_moe_forward(self, moe_model):
        ids = torch.randint(0, 256, (5,))
        with torch.no_grad():
            logits = moe_model(ids)
        assert logits.shape == (5, 256)

    def test_moe_batch(self, moe_model):
        ids = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            logits = moe_model(ids)
        # Model flattens [2, 8] -> [16] internally, output is [16, vocab]
        assert logits.shape == (16, 256)

    def test_dense_forward(self, dense_model):
        ids = torch.randint(0, 256, (5,))
        with torch.no_grad():
            logits = dense_model(ids)
        assert logits.shape == (5, 256)

    def test_num_parameters(self, moe_model):
        assert moe_model.num_parameters() > 0

    def test_deterministic(self, moe_model):
        ids = torch.tensor([1, 2, 3, 4, 5])
        with torch.no_grad():
            logits1 = moe_model(ids)
            logits2 = moe_model(ids)
        assert torch.allclose(logits1, logits2)


# ── decode_step (CUDA-graph-safe decode path) ──

class TestDecodeStep:
    """
    decode_step() is the graph-capturable counterpart to forward()'s
    _cached_attention: tensor-only KV writes/reads instead of the
    per-token .item() loop that breaks CUDA graph stream capture.
    It must produce the same numbers as forward() for the token it decodes.
    """

    def _make_kv_cache(self, config):
        from tr_hash_i64.core.kv_cache import PagedKVCache
        return PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            block_size=16,
            num_blocks=8,
            max_seqs=4,
            dtype=torch.float32,
            device="cpu",
        )

    def test_graph_safe_context_selects_dense_expert_dispatch(self, moe_model, monkeypatch):
        from tr_hash_i64.core.graph_context import graph_safe_mode

        mlp = moe_model.layers[0].mlp
        called = []
        original_dense = mlp._dense_expert_forward

        def tracked_dense(x, expert_ids):
            called.append(True)
            return original_dense(x, expert_ids)

        monkeypatch.setattr(mlp, "_dense_expert_forward", tracked_dense)
        with torch.no_grad(), graph_safe_mode():
            moe_model(torch.tensor([1, 2, 3]))

        assert called

    def test_matches_cached_forward(self, moe_config, moe_model):
        prefill_ids = torch.tensor([3, 9, 12])
        next_id = torch.tensor([7])

        # Reference: forward()'s existing _cached_attention path end to end.
        kv_ref = self._make_kv_cache(moe_config)
        with torch.no_grad():
            moe_model(
                prefill_ids, positions=torch.arange(3),
                kv_cache=kv_ref, seq_ids=[0], tokens_per_seq=[3],
            )
            logits_ref = moe_model(
                next_id, positions=torch.tensor([3]),
                kv_cache=kv_ref, seq_ids=[0], tokens_per_seq=[1],
            )

        # Same prefill, then decode_step() for the new token.
        kv_new = self._make_kv_cache(moe_config)
        with torch.no_grad():
            moe_model(
                prefill_ids, positions=torch.arange(3),
                kv_cache=kv_new, seq_ids=[0], tokens_per_seq=[3],
            )
            logits_new = moe_model.decode_step(
                token_ids=next_id,
                positions=torch.tensor([3]),
                kv_cache=kv_new,
                seq_ids_tensor=torch.tensor([0]),
            )

        assert torch.allclose(logits_ref, logits_new, atol=1e-5, rtol=1e-4)

    def test_accepts_unused_velocity_buf(self, moe_config, moe_model):
        """Engine always passes velocity_buf; model has no mu-guidance state
        to put there, so decode_step must accept and ignore it."""
        kv = self._make_kv_cache(moe_config)
        with torch.no_grad():
            moe_model(
                torch.tensor([1, 2]), positions=torch.arange(2),
                kv_cache=kv, seq_ids=[0], tokens_per_seq=[2],
            )
            logits = moe_model.decode_step(
                token_ids=torch.tensor([5]),
                positions=torch.tensor([2]),
                kv_cache=kv,
                seq_ids_tensor=torch.tensor([0]),
                velocity_buf=torch.zeros(1, moe_config.hidden_size),
            )
        assert logits.shape == (1, moe_config.vocab_size)
