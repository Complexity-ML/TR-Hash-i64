"""
vllm-i64 :: Complexity Deep Config

Config specific to Complexity Deep / Pacific-Prime models.
Mirrors checkpoints/*/config.json.

Complexity-ML - 2026
"""

import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class ComplexityDeepConfig:
    """
    Complexity Deep model config.
    Mirrors checkpoints/pacific-prime-chat/config.json.
    """
    # Architecture
    model_type: str = "complexity-deep"
    architecture: str = "DeepForCausalLM"
    version: str = "0.13.0"

    # Dimensions
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 3200
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4          # GQA

    # Positions
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Norms & activation
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"

    # Embeddings
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # Token IDs (from tokenizer: </s>=0, <pad>=1, <s>=2)
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # Token-Routed MLP (i64)
    use_token_routed_mlp: bool = True
    num_experts: int = 4
    shared_expert: bool = True
    shared_intermediate_size: Optional[int] = None  # None = expert_intermediate_size
    top_k: int = 1
    top_k_primary_weight: Optional[float] = None
    use_shared_routed_gates: bool = False
    shared_gate_init: float = 1.0
    routed_gate_init: float = 1.0
    shared_output_scale: float = 1.0
    routed_output_scale: float = 1.0
    routing_strategy: str = "modulo_cyclic"
    source_mlp_type: Optional[str] = None

    # Attention features
    use_qk_norm: bool = True
    use_sdpa: bool = True
    sliding_window: Optional[int] = None


    # Mu-Guidance
    use_mu_guidance: bool = False       # Enable mu projection between layers

    # Ablation flags (from training config)
    disable_mu_guidance: bool = False   # run3-no-mu: skip mu→Q/K/V and mu routing


    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def expert_intermediate_size(self) -> int:
        return self.intermediate_size // self.num_experts

    @staticmethod
    def from_json(path: str) -> "ComplexityDeepConfig":
        """Load from a checkpoint config.json (supports both deep and framework format)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = ComplexityDeepConfig()

        # Map framework config fields to deep config fields
        field_map = {
            "norm_eps": "rms_norm_eps",
        }

        for key, val in data.items():
            if key in ("parameters", "innovations", "extra_config"):
                continue
            mapped_key = field_map.get(key, key)
            if hasattr(config, mapped_key):
                setattr(config, mapped_key, val)

        # Framework format: detect token-routed from mlp_type
        if data.get("mlp_type") == "token_routed":
            config.use_token_routed_mlp = True
        elif data.get("mlp_type") == "swiglu":
            config.use_token_routed_mlp = False
            config.num_experts = 1

        return config
