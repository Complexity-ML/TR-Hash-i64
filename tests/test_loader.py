"""
tr-hash-i64 :: Test Weight Loader + Config

Tests:
  - ComplexityDeepConfig.from_json() parsing
  - ComplexityDeepConfig defaults
  - load_checkpoint with synthetic state_dict
  - TP-aware loading (ColumnParallel/RowParallel detection)
  - Tied embedding handling
  - Registry lookup
"""

import torch
import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tr_hash_i64.models.complexity_deep.config import ComplexityDeepConfig
from tr_hash_i64.models.complexity_deep.model import ComplexityDeepModel
from tr_hash_i64.core.loader import (
    load_checkpoint,
    _fuse_projection_layers,
    _get_module_for_param,
    _projection_fusion_allowed,
)
from tr_hash_i64.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class TestComplexityDeepConfig:
    def test_defaults(self):
        config = ComplexityDeepConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 1024
        assert config.num_experts == 4
        assert config.head_dim == 64  # 1024 / 16

    def test_head_dim_property(self):
        config = ComplexityDeepConfig(hidden_size=512, num_attention_heads=8)
        assert config.head_dim == 64

    def test_expert_intermediate_property(self):
        config = ComplexityDeepConfig(intermediate_size=5632, num_experts=4)
        assert config.expert_intermediate_size == 1408

    def test_from_json(self):
        data = {
            "model_type": "complexity-deep",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "num_experts": 2,
            "parameters": "skip_this",
            "innovations": "skip_this_too",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            config = ComplexityDeepConfig.from_json(path)
            assert config.vocab_size == 1000
            assert config.hidden_size == 256
            assert config.num_hidden_layers == 4
            assert config.num_experts == 2
            # "parameters" and "innovations" should be skipped
        finally:
            os.unlink(path)

    def test_from_json_ignores_unknown_keys(self):
        data = {"vocab_size": 500, "some_future_field": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            config = ComplexityDeepConfig.from_json(path)
            assert config.vocab_size == 500
            assert not hasattr(config, "some_future_field")
        finally:
            os.unlink(path)

    def test_from_json_ignores_derived_head_dim(self):
        data = {
            "hidden_size": 640,
            "num_attention_heads": 10,
            "head_dim": 64,
            "architectures": ["TRHashForCausalLM"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            config = ComplexityDeepConfig.from_json(path)
            assert config.head_dim == 64
        finally:
            os.unlink(path)

    def test_top2_and_output_gates_from_json(self):
        data = {
            "mlp_type": "token_routed",
            "num_experts": 4,
            "top_k": 2,
            "top_k_primary_weight": 0.5,
            "use_shared_routed_gates": True,
            "shared_gate_init": 0.5,
            "routed_gate_init": 0.5,
            "shared_output_scale": 1.0,
            "routed_output_scale": 2.0,
            "routing_strategy": "token_id_balanced_hash",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            config = ComplexityDeepConfig.from_json(path)
            assert config.top_k == 2
            assert config.top_k_primary_weight == 0.5
            assert config.use_shared_routed_gates is True
            assert config.shared_output_scale == 1.0
            assert config.routed_output_scale == 2.0
            assert config.routing_strategy == "token_id_balanced_hash"
        finally:
            os.unlink(path)


class TestPublicRegistry:
    def test_public_models_include_tr_hash_500m(self):
        from tr_hash_i64.core.registry import list_models

        assert [item["name"] for item in list_models()] == [
            "tr-hash-moe-100m-agentic-sft",
            "tr-hash-moe-500m",
            "tr-hash-moe-200m",
            "tr-moe-306",
            "dense-306",
        ]

    def test_tr_hash_100m_agentic_registry_entry(self):
        from tr_hash_i64.core.registry import get_model_entry

        entry = get_model_entry("tr-hash-moe-100m-agentic-sft")
        assert entry.checkpoint == "AETHORIA-AI/TR-HASH-MoE-100M-70B-Agentic-SFT"
        assert entry.parameters == "100.4M"
        assert "Agentic SFT" in entry.description

    def test_tr_hash_500m_registry_entry(self):
        from tr_hash_i64.core.registry import get_model_entry

        entry = get_model_entry("tr-hash-moe-500m")
        assert entry.checkpoint == "Pacific-i64/TR-HASH-MOE-500M-HF"
        assert entry.parameters == "492.1M"

    def test_tr_hash_200m_registry_entry_is_stable_sft_v1(self):
        from tr_hash_i64.core.registry import get_model_entry

        entry = get_model_entry("tr-hash-moe-200m")
        assert entry.checkpoint == "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
        assert entry.parameters == "201.2M"
        assert "SFT v1" in entry.description


class TestGetModuleForParam:
    def test_finds_linear(self):
        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2, vocab_size=32,
        )
        model = ComplexityDeepModel(config)
        # q_proj is nn.Linear
        module = _get_module_for_param(model, "layers.0.self_attn.q_proj.weight")
        assert module is not None

    def test_returns_none_for_missing(self):
        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2, vocab_size=32,
        )
        model = ComplexityDeepModel(config)
        module = _get_module_for_param(model, "nonexistent.layer.weight")
        assert module is None


class TestProjectionFusion:
    @pytest.mark.parametrize("quantization", ["awq", "gptq"])
    def test_excludes_prequantized_formats(self, quantization):
        assert not _projection_fusion_allowed("cpu", quantization)
        assert not _projection_fusion_allowed("cuda", quantization)

    @pytest.mark.parametrize("quantization", [None, "none"])
    def test_allows_unquantized_accelerator_formats(self, quantization):
        assert _projection_fusion_allowed("cuda", quantization)
        assert _projection_fusion_allowed("mps", quantization)

    @pytest.mark.parametrize("quantization", ["int8", "int4", "fp8"])
    def test_leaves_accelerator_post_quantization_layout_unfused(self, quantization):
        assert not _projection_fusion_allowed("cuda", quantization)

    @pytest.mark.parametrize("quantization", [None, "none", "int8", "int4"])
    def test_preserves_cpu_fusion_behavior(self, quantization):
        assert _projection_fusion_allowed("cpu", quantization)

    def test_preserves_model_output(self):
        config = ComplexityDeepConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            num_experts=2,
            vocab_size=32,
            max_position_embeddings=32,
            top_k=2,
        )
        model = ComplexityDeepModel(config).eval()
        token_ids = torch.tensor([2, 7, 11, 19])
        positions = torch.arange(token_ids.numel())

        with torch.inference_mode():
            expected = model(token_ids, positions, tokens_per_seq=[4])
            fused_count = _fuse_projection_layers(model)
            actual = model(token_ids, positions, tokens_per_seq=[4])

        assert fused_count == 3
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class TestLoadCheckpoint:
    def _make_small_model(self):
        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2,
            vocab_size=32, max_position_embeddings=32, top_k=2,
        )
        return ComplexityDeepModel(config), config

    def test_load_synthetic_checkpoint(self):
        model, config = self._make_small_model()

        # Create a synthetic state_dict from the model itself
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            assert stats["loaded"] > 0
            assert stats["tp_size"] == 1
        finally:
            os.unlink(path)

    def test_tied_embeddings(self):
        model, config = self._make_small_model()

        # Simulate checkpoint with lm_head.weight (tied to embed_tokens)
        state_dict = {"lm_head.weight": torch.randn(config.vocab_size, config.hidden_size)}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            # lm_head.weight should be loaded into embed_tokens.weight
            assert "lm_head.weight" not in stats.get("missing", set())
        finally:
            os.unlink(path)

    def test_missing_checkpoint_raises(self):
        model, _ = self._make_small_model()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, "/nonexistent/path/model.pt")

    def test_skips_rotary_inv_freq(self):
        model, _ = self._make_small_model()

        state_dict = {
            "layers.0.self_attn.rope.rotary_emb.inv_freq": torch.randn(16),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            assert stats["skipped"] >= 1
        finally:
            os.unlink(path)

    def test_loads_native_tr_hash_top2_table_exactly(self):
        model, config = self._make_small_model()
        route_table = torch.stack(
            [
                torch.arange(config.vocab_size) % config.num_experts,
                (torch.arange(config.vocab_size) * 7 + 1) % config.num_experts,
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(
                {"layers.0.mlp.topk_token_to_expert": route_table},
                f.name,
            )
            path = f.name

        try:
            load_checkpoint(model, path, dtype=torch.float32)
            mlp = model.layers[0].mlp
            assert torch.equal(mlp.topk_token_to_expert, route_table)
            assert torch.equal(mlp.token_to_expert, route_table[0])
            token_ids = torch.tensor([1, 7, 11])
            assert torch.equal(
                mlp.route(token_ids, len(token_ids), token_ids.device),
                route_table[:, token_ids],
            )
        finally:
            os.unlink(path)

    def test_converts_mlp_engine_checkpoint_format(self):
        """complexity-framework's TRHashEngine checkpoint layout
        (layers.N.mlp.engine.*) must convert to tr-hash-i64's native
        gate_proj_w/up_proj_w/down_proj_w/topk_token_to_expert names, not
        pass through unrecognized (which would leave expert weights and
        the route table at their random init -- silently wrong inference)."""
        model, config = self._make_small_model()
        expert_inter = config.intermediate_size // config.num_experts

        route_table = torch.stack(
            [
                torch.arange(config.vocab_size) % config.num_experts,
                (torch.arange(config.vocab_size) * 7 + 1) % config.num_experts,
            ]
        )
        expert_gate = torch.randn(config.num_experts, config.hidden_size, expert_inter)
        expert_up = torch.randn(config.num_experts, config.hidden_size, expert_inter)
        expert_down = torch.randn(config.num_experts, expert_inter, config.hidden_size)
        shared_gate = torch.randn(config.intermediate_size, config.hidden_size)
        shared_up = torch.randn(config.intermediate_size, config.hidden_size)
        shared_down = torch.randn(config.hidden_size, config.intermediate_size)

        state_dict = {
            "layers.0.mlp.engine.expert_gate": expert_gate,
            "layers.0.mlp.engine.expert_up": expert_up,
            "layers.0.mlp.engine.expert_down": expert_down,
            "layers.0.mlp.engine.route_table": route_table,
            "layers.0.mlp.engine.shared_gate.weight": shared_gate,
            "layers.0.mlp.engine.shared_up.weight": shared_up,
            "layers.0.mlp.engine.shared_down.weight": shared_down,
            # Cache-only artifacts -- must be dropped, not loaded anywhere.
            "layers.0.mlp.engine.fused_route_codes": torch.zeros(config.vocab_size, dtype=torch.uint8),
            "layers.0.mlp.engine.fused_expert_pairs": torch.zeros(6, 2, dtype=torch.int32),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            mlp = model.layers[0].mlp
            assert torch.equal(mlp.gate_proj_w.data, expert_gate)
            assert torch.equal(mlp.up_proj_w.data, expert_up)
            assert torch.equal(mlp.down_proj_w.data, expert_down)
            assert torch.equal(mlp.shared_gate.weight.data, shared_gate)
            assert torch.equal(mlp.shared_up.weight.data, shared_up)
            assert torch.equal(mlp.shared_down.weight.data, shared_down)
            assert torch.equal(mlp.topk_token_to_expert, route_table)
            assert torch.equal(mlp.token_to_expert, route_table[0])
            assert not any(
                name.endswith(("fused_route_codes", "fused_expert_pairs"))
                for name in stats.get("missing", set())
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
