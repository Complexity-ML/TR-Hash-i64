"""
vllm-i64 :: Test Model Forward Pass

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

from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
from vllm_i64.models.complexity_deep.model import (
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
        x = torch.randn(2, 10, 128)
        with torch.no_grad():
            out, kv = attn(x)
        assert out.shape == (2, 10, 128)
        assert kv is None

    def test_with_mu(self, moe_config):
        attn = Attention(moe_config)
        attn.eval()
        x = torch.randn(1, 5, 128)
        mu = torch.randn(1, 5, 128)
        with torch.no_grad():
            out, _ = attn(x, mu_prev=mu)
        assert out.shape == (1, 5, 128)

    def test_kv_cache(self, moe_config):
        attn = Attention(moe_config)
        attn.eval()
        x = torch.randn(1, 5, 128)
        with torch.no_grad():
            _, kv = attn(x, use_cache=True)
        assert kv is not None
        assert kv[0].shape[2] == 5  # seq_len cached

        # Decode step with cache
        x2 = torch.randn(1, 1, 128)
        with torch.no_grad():
            out2, kv2 = attn(x2, past_key_value=kv, use_cache=True)
        assert out2.shape == (1, 1, 128)
        assert kv2[0].shape[2] == 6  # 5 + 1


# ── Dense MLP ──

class TestDenseSwiGLUMLP:
    def test_output_shape(self, dense_config):
        mlp = DenseSwiGLUMLP(dense_config)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == (2, 10, 128)


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


# ── Decoder Layer ──

class TestDecoderLayer:
    def test_moe_layer(self, moe_config):
        layer = ComplexityDecoderLayer(moe_config)
        layer.eval()
        x = torch.randn(1, 8, 128)
        token_ids = torch.randint(0, 256, (1, 8))
        mu = torch.randn(1, 8, 128)
        with torch.no_grad():
            out, kv, mu_current = layer(x, token_ids=token_ids, mu_prev=mu)
        assert out.shape == (1, 8, 128)
        assert mu_current is not None
        assert mu_current.shape == (1, 8, 128)

    def test_dense_layer(self, dense_config):
        layer = ComplexityDecoderLayer(dense_config)
        layer.eval()
        x = torch.randn(1, 8, 128)
        with torch.no_grad():
            out, kv, mu_current = layer(x)
        assert out.shape == (1, 8, 128)
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
        assert logits.shape == (2, 8, 256)

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
