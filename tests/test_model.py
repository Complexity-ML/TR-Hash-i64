"""
tr-hash-i64 :: Test Model Forward Pass

Tests ComplexityDeepModel end-to-end on CPU:
  - Correct output shapes
  - Mu-Guidance produces mu_current
  - Attention with GQA + RoPE + QK Norm
  - TokenRoutedMLP routing (MoE)
  - DenseSwiGLUMLP (dense baseline)
  - Full forward: token_ids → logits
  - Weight loading from framework checkpoints

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
    MuGuidance,
    MoEMLP,
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
        use_mu_guidance=True,
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
        use_mu_guidance=False,
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


# ── Mu-Guidance ──

class TestMuGuidance:
    def test_output_shape(self):
        mu = MuGuidance(hidden_size=128)
        h = torch.randn(2, 10, 128)
        out = mu(h)
        assert out.shape == (2, 10, 128)

    def test_clamped(self):
        mu = MuGuidance(hidden_size=128)
        h = torch.randn(1, 5, 128) * 100
        out = mu(h)
        assert out.min() >= -2.0
        assert out.max() <= 2.0

    def test_zero_init_proj(self):
        mu = MuGuidance(hidden_size=64)
        assert mu.mu_proj.weight.abs().sum() == 0


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

    def test_with_mu(self, moe_config):
        attn = Attention(moe_config)
        attn.eval()
        N = 5
        x = torch.randn(N, 128)
        mu = torch.randn(N, 128)
        positions = torch.arange(N)
        with torch.no_grad():
            out = attn(x, positions, mu_prev=mu, tokens_per_seq=[N])
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


# ── MoE MLP ──

class TestMoEMLP:
    def test_output_shape(self, moe_config):
        mlp = MoEMLP(moe_config)
        x = torch.randn(2, 10, 128)
        token_ids = torch.randint(0, 256, (2, 10))
        out = mlp(x, token_ids=token_ids)
        assert out.shape == (2, 10, 128)

    def test_without_token_ids(self, moe_config):
        mlp = MoEMLP(moe_config)
        x = torch.randn(1, 5, 128)
        out = mlp(x, token_ids=None)
        assert out.shape == (1, 5, 128)

    def test_top2_routes_are_distinct_and_weighted(self, moe_config):
        moe_config.top_k = 2
        moe_config.top_k_primary_weight = 0.5
        mlp = MoEMLP(moe_config).tr_mlp
        token_ids = torch.tensor([1, 2, 3])
        routes = mlp.route(token_ids, 3, token_ids.device)
        assert routes.shape == (2, 3)
        assert torch.equal(routes[1], (routes[0] + 1) % moe_config.num_experts)

    def test_shared_gate_up_fusion_preserves_output(self, moe_config):
        mlp = MoEMLP(moe_config).tr_mlp.eval()
        x = torch.randn(7, 128)
        token_ids = torch.arange(7)
        with torch.inference_mode():
            expected = mlp(x, token_ids=token_ids)
            assert mlp.fuse_shared_gate_up() is True
            actual = mlp(x, token_ids=token_ids)
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)

    def test_expert_linear_materialization_preserves_output(self, moe_config):
        mlp = MoEMLP(moe_config).tr_mlp.eval()
        x = torch.randn(7, 128)
        token_ids = torch.arange(7)
        with torch.inference_mode():
            expected = mlp(x, token_ids=token_ids)
            assert mlp.materialize_expert_linears() is True
            actual = mlp(x, token_ids=token_ids)
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
        mu = torch.randn(N, 128)
        with torch.no_grad():
            out, mu_current = layer(x, positions, token_ids=token_ids, mu_prev=mu, tokens_per_seq=[N])
        assert out.shape == (N, 128)
        assert mu_current is not None
        assert mu_current.shape == (N, 128)

    def test_dense_layer(self, dense_config):
        layer = ComplexityDecoderLayer(dense_config)
        layer.eval()
        N = 8
        x = torch.randn(N, 128)
        positions = torch.arange(N)
        with torch.no_grad():
            out, mu_current = layer(x, positions, tokens_per_seq=[N])
        assert out.shape == (N, 128)
        assert mu_current is None

    def test_has_mu_guidance(self, moe_config):
        layer = ComplexityDecoderLayer(moe_config)
        assert layer.mu_guidance is not None

    def test_no_mu_guidance_dense(self, dense_config):
        layer = ComplexityDecoderLayer(dense_config)
        assert layer.mu_guidance is None


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

    def test_has_mu_init(self, moe_model):
        assert hasattr(moe_model, 'mu_init')
        assert moe_model.mu_init.shape == (1, 1, 128)

    def test_no_mu_init_dense(self, dense_model):
        assert not hasattr(dense_model, 'mu_init')

    def test_num_parameters(self, moe_model):
        assert moe_model.num_parameters() > 0

    def test_deterministic(self, moe_model):
        ids = torch.tensor([1, 2, 3, 4, 5])
        with torch.no_grad():
            logits1 = moe_model(ids)
            logits2 = moe_model(ids)
        assert torch.allclose(logits1, logits2)


# ── Weight Loading ──

class TestWeightLoading:
    def test_moe_load_framework(self, moe_config):
        model = ComplexityDeepModel(moe_config)
        state = model.state_dict()
        missing, unexpected = model.load_framework_checkpoint(state)
        assert len(missing) == 0
        assert len(unexpected) == 0

    def test_dense_load_framework(self, dense_config):
        model = ComplexityDeepModel(dense_config)
        state = model.state_dict()
        missing, unexpected = model.load_framework_checkpoint(state)
        assert len(missing) == 0
        assert len(unexpected) == 0
