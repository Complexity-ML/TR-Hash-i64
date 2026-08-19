"""
tr-hash-i64 :: Weight Loader

Load checkpoint weights into models.
TP-aware: detects ColumnParallelLinear / RowParallelLinear and loads
only the current rank's shard from the full checkpoint.

Handles:
  - Tied embeddings (lm_head.weight → embed_tokens.weight)
  - Token-routed MLP expert weights (gate_up_proj, down_proj) — TP sharded
  - Attention Q/K/V (ColumnParallel) — TP sharded on heads
  - Attention O (RowParallel) — TP sharded on input dim
  - token_to_expert buffer (replicated)
  - dtype conversion
"""

import json as _json
import logging
import re
import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path

from tr_hash_i64.core.registry import get_model_entry
from tr_hash_i64.parallel.tensor_parallel import (
    get_tp, ColumnParallelLinear, RowParallelLinear,
)

logger = logging.getLogger("tr_hash_i64.loader")


def resolve_checkpoint_source(source: str) -> str:
    """Resolve a local path or download a Hugging Face model repository."""

    local = Path(source).expanduser()
    if local.exists():
        # .absolute(), not .resolve() -- resolve() dereferences symlinks all
        # the way through, and the HF cache stores files under extension-less
        # content-hash blobs with the real filename only on the symlink
        # (snapshots/<rev>/model.safetensors -> blobs/<hash>). Following it
        # strips the .safetensors suffix that _load_state_dict's file-type
        # detection depends on.
        return str(local.absolute())
    if source.count("/") != 1:
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    from huggingface_hub import HfApi, snapshot_download

    logger.info("Downloading Hugging Face model: %s", source)

    repo_files = HfApi().list_repo_files(source)
    if "model.safetensors" in repo_files:
        return snapshot_download(
            repo_id=source,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "*.j2",
            ],
        )

    # Multi-checkpoint "replay" repo: token_pack_NNN_TOKENS/model.safetensors
    # per training step, nothing at the root. Pull only the highest step
    # number (the most-trained checkpoint) and skip the optimizer shards
    # (checkpoint.pt, optimizer_rank*.pt) — not needed for inference and
    # would otherwise multiply the download several times over per checkpoint.
    pack_re = re.compile(r"^(token_pack_(\d+)_\d+)/model\.safetensors$")
    packs = [
        (int(match.group(2)), match.group(1))
        for f in repo_files
        for match in [pack_re.match(f)]
        if match
    ]
    if not packs:
        raise FileNotFoundError(
            f"No model.safetensors found in {source} "
            "(checked repo root and token_pack_*/ subfolders)"
        )
    _, latest_pack = max(packs)
    logger.info("Multi-checkpoint repo — using latest: %s", latest_pack)

    # Exact root-level filenames only — "*.json" would also glob-match every
    # token_pack_*/training_state.json (one per training step, not needed).
    root_files = [
        f for f in repo_files
        if "/" not in f and f.endswith((".json", ".jinja", ".j2"))
    ]
    local_dir = Path(snapshot_download(
        repo_id=source,
        allow_patterns=root_files + [f"{latest_pack}/model.safetensors"],
    ))
    weights_link = local_dir / "model.safetensors"
    if not weights_link.exists():
        weights_link.symlink_to(local_dir / latest_pack / "model.safetensors")
    return str(local_dir)


def _quantize_cpu_dynamic(model: nn.Module) -> nn.Module:
    """Pack ``nn.Linear`` weights as INT8 using the platform's native CPU backend.

    x86/FBGEMM on x86_64, QNNPACK on ARM (Apple Silicon, mobile) — PyTorch
    ships each as native SIMD kernels for that instruction set only, so
    there's no single backend that covers both.
    """

    engines = torch.backends.quantized.supported_engines
    engine = next(
        (e for e in ("x86", "fbgemm", "qnnpack") if e in engines), None
    )
    if engine is None:
        raise RuntimeError(
            "CPU INT8 requires the x86, FBGEMM, or QNNPACK PyTorch "
            f"quantization backend; available backends: {engines}"
        )

    torch.backends.quantized.engine = engine
    model.eval()
    logger.info("CPU dynamic INT8: packing nn.Linear weights with %s", engine)
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )


def _fuse_cpu_projection_layers(model: nn.Module) -> int:
    """Apply inference-only projection fusion after all weights are loaded."""

    fused = 0
    for module in list(model.modules()):
        for method_name in (
            "fuse_qkv",
            "fuse_gate_up",
            "fuse_shared_gate_up",
            "materialize_expert_linears",
        ):
            method = getattr(module, method_name, None)
            if method is not None and method():
                fused += 1
    logger.info("CPU projection fusion: %d modules", fused)
    return fused


# =========================================================================
# Multi-format state_dict loading (safetensors + PyTorch)
# =========================================================================

def _load_safetensors_file(filepath: str) -> Dict[str, torch.Tensor]:
    """Load a single .safetensors file."""
    from safetensors.torch import load_file
    return load_file(filepath)


def _load_pytorch_file(filepath: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint file and unwrap nested state dicts."""
    state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict):
        if "model" in state_dict and "model.layers.0.input_layernorm.weight" not in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    return state_dict


def _load_sharded_safetensors(directory: Path) -> Dict[str, torch.Tensor]:
    """Load sharded safetensors from a HuggingFace model directory."""
    index_path = directory / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = _json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    state_dict = {}
    for shard_name in shard_files:
        shard_path = directory / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        state_dict.update(_load_safetensors_file(str(shard_path)))
    return state_dict


def _load_from_directory(dir_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load from a model directory. Priority:
      1. model.safetensors.index.json (sharded)
      2. model.safetensors (single file)
      3. *.safetensors (glob)
      4. *.pt / *.pth / *.bin (PyTorch)
    """
    if (dir_path / "model.safetensors.index.json").exists():
        return _load_sharded_safetensors(dir_path)

    single_st = dir_path / "model.safetensors"
    if single_st.exists():
        return _load_safetensors_file(str(single_st))

    st_files = sorted(dir_path.glob("*.safetensors"))
    if st_files:
        state_dict = {}
        for f in st_files:
            state_dict.update(_load_safetensors_file(str(f)))
        return state_dict

    pt_files = sorted(dir_path.glob("*.pt")) + sorted(dir_path.glob("*.pth")) + sorted(dir_path.glob("*.bin"))
    if pt_files:
        state_dict = {}
        for f in pt_files:
            state_dict.update(_load_pytorch_file(str(f)))
        return state_dict

    raise FileNotFoundError(f"No checkpoint files found in {dir_path}")


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Auto-detect and load state dict from any supported format.

    Supports:
      - .pt / .pth / .bin (PyTorch)
      - .safetensors (single or sharded)
      - Directories (HuggingFace model dirs)
    """
    path = Path(checkpoint_path)

    if path.is_dir():
        return _load_from_directory(path)
    elif path.suffix == ".safetensors":
        return _load_safetensors_file(str(path))
    elif path.suffix in (".pt", ".pth", ".bin"):
        return _load_pytorch_file(str(path))
    else:
        try:
            return _load_pytorch_file(str(path))
        except (OSError, RuntimeError, KeyError):
            return _load_safetensors_file(str(path))


def _convert_framework_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a complexity-framework TR-Hash checkpoint (``mlp.engine.*``
    naming, deterministic single/multi-hash routing) to tr-hash-i64's
    native parameter names. Already-native checkpoints pass through
    unchanged.
    """
    import re

    has_engine_format = any(".mlp.engine." in k for k in state_dict)
    if not has_engine_format:
        return state_dict  # Already in tr-hash-i64 format or unknown

    logger.info("Detected complexity-framework mlp.engine format, converting weights...")
    converted = {}

    for name, tensor in state_dict.items():
        # Cache-only artifacts recomputed at export/build time
        # (complexity/tr_hash/routing.py) -- never loaded.
        if name.endswith(("fused_route_codes", "fused_expert_pairs")):
            continue

        m = re.match(r"(layers\.\d+\.mlp)\.engine\.expert_gate$", name)
        if m:
            converted[f"{m.group(1)}.gate_proj_w"] = tensor
            continue
        m = re.match(r"(layers\.\d+\.mlp)\.engine\.expert_up$", name)
        if m:
            converted[f"{m.group(1)}.up_proj_w"] = tensor
            continue
        m = re.match(r"(layers\.\d+\.mlp)\.engine\.expert_down$", name)
        if m:
            converted[f"{m.group(1)}.down_proj_w"] = tensor
            continue

        m = re.match(r"(layers\.\d+\.mlp)\.engine\.route_table$", name)
        if m:
            # [top_k, vocab] -- the exact table baked at training time by
            # the multi-hash rendezvous construction (route_hash_count
            # hash channels summed per (token, expert) pair, top_k highest
            # scores kept; see complexity/tr_hash/routing.py::_multi_hash_routes).
            # Load verbatim: recomputing routes here would silently diverge
            # from the trained model.
            prefix = m.group(1)
            converted[f"{prefix}.topk_token_to_expert"] = tensor.long()
            converted[f"{prefix}.token_to_expert"] = tensor[0].long()
            continue

        m = re.match(r"(layers\.\d+\.mlp)\.engine\.(shared_gate|shared_up|shared_down)\.weight$", name)
        if m:
            converted[f"{m.group(1)}.{m.group(2)}.weight"] = tensor
            continue

        # Everything else (attention, norms, embeddings, rotary) passes
        # through as-is.
        converted[name] = tensor

    return converted


def _get_module_for_param(model: nn.Module, param_name: str) -> Optional[nn.Module]:
    """Walk the module tree to find the module owning a parameter."""
    parts = param_name.split(".")
    # The last part is the actual parameter name (e.g. "weight", "bias")
    # Walk up to the parent module
    module = model
    for part in parts[:-1]:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    strict: bool = False,
) -> dict:
    """
    TP-aware checkpoint loading.

    For ColumnParallelLinear / RowParallelLinear layers, loads the full
    weight from checkpoint and calls load_full_weight() which takes the
    current rank's shard.

    For TokenRoutedMLP expert weights, calls load_full_weights() which
    shards gate_up and down projections.

    Other params are loaded directly (replicated).

    Args:
        model: the model to load into
        checkpoint_path: path to .pt file or directory
        dtype: target dtype for weights
        device: target device
        strict: if True, require all keys match

    Returns:
        dict with loading stats
    """
    tp = get_tp()
    checkpoint_path = resolve_checkpoint_source(checkpoint_path)
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if tp.tp_rank == 0:
        logger.info("Loading checkpoint: %s", checkpoint_path)

    state_dict = _load_state_dict(checkpoint_path)

    # Auto-detect complexity-framework format and convert to tr-hash-i64 format
    state_dict = _convert_framework_weights(state_dict)

    # Build lookup of model parameters and their parent modules
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    loaded = set()           # checkpoint keys that were loaded
    loaded_params = set()    # model param names that received data
    skipped = set()
    missing = set()

    # Collect native expert weights for TP sharding
    expert_gate = {}
    expert_up = {}
    expert_down_native = {}

    for name, weight in state_dict.items():
        # Skip rotary inv_freq (computed at init)
        if "rotary_emb.inv_freq" in name:
            skipped.add(name)
            continue

        # Tied embeddings: lm_head → embed_tokens
        if name == "lm_head.weight":
            embed_name = "embed_tokens.weight"
            if embed_name in params:
                params[embed_name].data.copy_(weight.to(dtype))
                loaded.add(name)
                loaded.add(embed_name)
            continue

        # Skip embed_tokens if already loaded via lm_head
        if name in ("embed_tokens.weight", "model.embed_tokens.weight"):
            if name in loaded:
                continue

        # Native TR-Hash exports store all deterministic routes explicitly as
        # [top_k, vocab]. Loading the table is required for exact inference;
        # silently deriving cyclic routes would change the trained model.
        if "topk_token_to_expert" in name:
            buf_name = name.replace("model.", "", 1) if name not in buffers else name
            if buf_name in buffers:
                buffers[buf_name].copy_(weight.long())
                loaded.add(name)
                legacy_name = buf_name.replace(
                    "topk_token_to_expert", "token_to_expert"
                )
                if legacy_name in buffers:
                    buffers[legacy_name].copy_(weight[0].long())
            continue

        # Legacy primary routing table (replicated).
        if "token_to_expert" in name:
            buf_name = name.replace("model.", "", 1) if name not in buffers else name
            if buf_name in buffers:
                primary = weight.long()
                buffers[buf_name].copy_(primary)
                topk_name = buf_name.replace(
                    "token_to_expert", "topk_token_to_expert"
                )
                if topk_name in buffers:
                    owner_path = topk_name.rsplit(".", 1)[0]
                    owner = _get_module_for_param(model, owner_path + ".dummy")
                    num_experts = getattr(owner, "num_experts", 1)
                    for route_idx in range(buffers[topk_name].shape[0]):
                        buffers[topk_name][route_idx].copy_(
                            (primary + route_idx) % num_experts
                        )
                loaded.add(name)
            continue

        # Resolve actual param name (strip "model." prefix if needed)
        resolved_name = name
        if name not in params:
            resolved_name = name.replace("model.", "", 1)

        # TP-wrapped layers: checkpoint has "q_proj.weight" but model has
        # "q_proj.linear.weight" (ColumnParallelLinear / RowParallelLinear)
        if resolved_name not in params and resolved_name.endswith(".weight"):
            tp_name = resolved_name[:-len(".weight")] + ".linear.weight"
            if tp_name in params:
                resolved_name = tp_name

        if resolved_name not in params:
            missing.add(name)
            continue

        # Check if this belongs to a TP-sharded module
        parent_module = _get_module_for_param(model, resolved_name)
        param_leaf = resolved_name.split(".")[-1]

        # For TP-wrapped names like "q_proj.linear.weight", check the grandparent
        tp_parent = None
        if param_leaf == "weight" and ".linear.weight" in resolved_name:
            gp_name = resolved_name.rsplit(".linear.weight", 1)[0] + ".dummy"
            tp_parent = _get_module_for_param(model, gp_name)

        if isinstance(tp_parent or parent_module, ColumnParallelLinear) and param_leaf == "weight":
            # Full weight → take TP shard
            tp_mod = tp_parent if isinstance(tp_parent, ColumnParallelLinear) else parent_module
            tp_mod.load_full_weight(weight.to(dtype))
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif isinstance(tp_parent or parent_module, RowParallelLinear) and param_leaf == "weight":
            # Full weight → take TP shard
            tp_mod = tp_parent if isinstance(tp_parent, RowParallelLinear) else parent_module
            tp_mod.load_full_weight(weight.to(dtype))
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif "gate_proj_w" in resolved_name and "mlp" in resolved_name:
            prefix = resolved_name.rsplit("gate_proj_w", 1)[0]
            expert_gate[prefix] = weight
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif "up_proj_w" in resolved_name and "mlp" in resolved_name:
            prefix = resolved_name.rsplit("up_proj_w", 1)[0]
            expert_up[prefix] = weight
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif "down_proj_w" in resolved_name and "mlp" in resolved_name:
            prefix = resolved_name.rsplit("down_proj_w", 1)[0]
            expert_down_native[prefix] = weight
            loaded.add(name)
            loaded_params.add(resolved_name)

        else:
            # Regular parameter (replicated) — direct copy
            params[resolved_name].data.copy_(weight.to(dtype))
            loaded.add(name)
            loaded_params.add(resolved_name)

    native_prefixes = set(expert_gate) | set(expert_up) | set(expert_down_native)
    for prefix in native_prefixes:
        if not (
            prefix in expert_gate
            and prefix in expert_up
            and prefix in expert_down_native
        ):
            raise RuntimeError(
                f"Incomplete native expert weights for {prefix}: "
                f"gate={prefix in expert_gate}, up={prefix in expert_up}, "
                f"down={prefix in expert_down_native}"
            )
        module = model
        for part in prefix.rstrip(".").split("."):
            module = getattr(module, part)
        module.load_full_weights(
            expert_gate[prefix].to(dtype),
            expert_up[prefix].to(dtype),
            expert_down_native[prefix].to(dtype),
        )

    # Check for unloaded model params
    model_params = set(params.keys())
    unloaded = model_params - loaded_params

    stats = {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "missing_in_model": len(missing),
        "unloaded_params": len(unloaded),
        "tp_rank": tp.tp_rank,
        "tp_size": tp.tp_size,
    }

    if tp.tp_rank == 0:
        logger.info("Loaded: %d tensors (TP=%d)", stats['loaded'], tp.tp_size)
        if stats['skipped']:
            logger.info("Skipped: %d (rotary inv_freq, etc.)", stats['skipped'])
        if stats['missing_in_model']:
            logger.warning("Not in model: %d", stats['missing_in_model'])
        if stats['unloaded_params']:
            logger.warning("Unloaded params: %d", stats['unloaded_params'])
            if strict:
                raise RuntimeError(f"Missing weights: {unloaded}")

    return stats


def _detect_checkpoint_quantization(checkpoint_path: str) -> Optional[str]:
    """
    Auto-detect AWQ/GPTQ quantization from checkpoint config.json.

    Returns "awq", "gptq", or None.
    """
    from tr_hash_i64.core.awq_gptq import detect_quant_config

    result = detect_quant_config(checkpoint_path)
    if result is not None:
        return result[0]  # "awq" or "gptq"
    return None


def load_model_by_name(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    checkpoint_override: Optional[str] = None,
    quantization: Optional[str] = None,
) -> nn.Module:
    """
    Load a registered model by name with TP support.

    Each rank creates the model (with TP-aware layers), then
    load_checkpoint takes the appropriate shard from the full checkpoint.

    Supports pre-quantized checkpoints:
      - "awq": load AWQ-quantized checkpoint (auto-detected or explicit)
      - "gptq": load GPTQ-quantized checkpoint (auto-detected or explicit)
      - "int8"/"int4": post-load quantization of float checkpoints
      - "fp8": FP8 quantization for Hopper GPUs

    Args:
        model_name: registered name (e.g. "pacific-prime-chat")
        dtype: weight dtype
        device: target device
        checkpoint_override: override checkpoint path
        quantization: optional quantization method ("int8", "int4", "awq", "gptq", or None)

    Returns:
        loaded model on the correct device
    """
    import importlib

    entry = get_model_entry(model_name)
    tp = get_tp()

    # Import model class dynamically
    module_path, class_name = entry.model_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # Import config class dynamically
    config_module_path, config_class_name = entry.config_loader.rsplit(".", 1)
    config_module = importlib.import_module(config_module_path)
    config_cls = getattr(config_module, config_class_name)

    ckpt_path = resolve_checkpoint_source(
        checkpoint_override or entry.checkpoint or ""
    )

    # The model repository is the source of truth for both config and weights.
    config_path = str(Path(ckpt_path) / "config.json")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    config = config_cls.from_json(config_path)

    # Create model — TP layers auto-detect tp_size from global state
    model = model_cls(config)
    model = model.to(dtype)
    # Dynamic quantization replaces Linear parameters with packed buffers.
    # Capture the architecture's trainable parameter count before packing so
    # the public API reports the checkpoint size, not only the parameters that
    # remain as regular nn.Parameter objects afterwards.
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Auto-detect AWQ/GPTQ from checkpoint if quantization not explicitly set
    effective_quant = quantization
    if ckpt_path and effective_quant in (None, "none"):
        detected = _detect_checkpoint_quantization(ckpt_path)
        if detected is not None:
            effective_quant = detected
            if tp.tp_rank == 0:
                logger.info("Auto-detected quantization: %s", detected)

    # Load weights — dispatch by quantization format
    if ckpt_path and effective_quant in ("awq", "gptq"):
        # Pre-quantized checkpoint: use AWQ/GPTQ loader
        from tr_hash_i64.core.awq_gptq import load_awq_checkpoint, load_gptq_checkpoint

        if effective_quant == "awq":
            stats = load_awq_checkpoint(model, ckpt_path, device=device, dtype=dtype)
        else:
            stats = load_gptq_checkpoint(model, ckpt_path, device=device, dtype=dtype)

        if tp.tp_rank == 0:
            logger.info("Loaded %s checkpoint: %d quantized layers, %d non-quantized weights",
                        effective_quant.upper(), stats['quant_layers'], stats['non_quant_loaded'])
    elif ckpt_path:
        # Standard float checkpoint
        load_checkpoint(model, ckpt_path, dtype=dtype, device=device)

    if device == "cpu":
        _fuse_cpu_projection_layers(model)

    # Post-load quantization (only for float checkpoints, not pre-quantized)
    if effective_quant and effective_quant not in ("none", "awq", "gptq"):
        if effective_quant == "int8" and device == "cpu":
            model = _quantize_cpu_dynamic(model)
        elif effective_quant == "fp8":
            _quantize_dense_mlp_fp8(model)
        else:
            _quantize_experts(model, effective_quant)
            _quantize_dense_mlp(model, effective_quant)
            _quantize_attention(model, effective_quant)
            _quantize_mu_attention(model, effective_quant)
            _quantize_i64_dynamics(model, effective_quant)
            _quantize_lm_head(model, effective_quant)
            _quantize_rmsnorm(model, effective_quant)
        if tp.tp_rank == 0:
            logger.info("Quantized weights: %s", effective_quant)

    model._tr_hash_i64_parameter_count = parameter_count
    model._tr_hash_i64_quantization = effective_quant or "none"

    # Disable grad tracking — inference only, avoids autograd view conflicts
    # with quantized buffers that are views of original parameters.
    model.requires_grad_(False)

    return model.to(device)


def _quantize_dense_mlp(model: nn.Module, method: str):
    """Apply post-load quantization to DenseMLP layers (2D weights, no expert dim)."""
    from tr_hash_i64.layers.dense_mlp import DenseMLP
    from tr_hash_i64.core.quantization import quantize_int8, quantize_int4

    if method not in ("int8", "int4"):
        return

    for name, module in model.named_modules():
        if not isinstance(module, DenseMLP):
            continue

        gate_w = module.gate_proj.linear.weight.data
        up_w = module.up_proj.linear.weight.data
        down_w = module.down_proj.linear.weight.data

        if method == "int8":
            gq, gs = quantize_int8(gate_w)
            uq, us = quantize_int8(up_w)
            dq, ds = quantize_int8(down_w)
            module.register_buffer("gate_int8", gq)
            module.register_buffer("gate_scale", gs)
            module.register_buffer("up_int8", uq)
            module.register_buffer("up_scale", us)
            module.register_buffer("down_int8", dq)
            module.register_buffer("down_scale", ds)
            # Fused gate+up for single-matmul path
            module.register_buffer("gate_up_int8", torch.cat([gq, uq], dim=0))
            module.register_buffer("gate_up_scale", torch.cat([gs, us]))
            module.gate_up_inter = gq.shape[0]

        elif method == "int4":
            gp, gs, gz = quantize_int4(gate_w)
            up, us, uz = quantize_int4(up_w)
            dp, ds_, dz = quantize_int4(down_w)
            module.register_buffer("gate_int4", gp)
            module.register_buffer("gate_scale_int4", gs)
            module.register_buffer("gate_zero", gz)
            module.register_buffer("up_int4", up)
            module.register_buffer("up_scale_int4", us)
            module.register_buffer("up_zero", uz)
            module.register_buffer("down_int4", dp)
            module.register_buffer("down_scale_int4", ds_)
            module.register_buffer("down_zero", dz)


def _quantize_experts(model: nn.Module, method: str):
    """Apply post-load quantization to expert MLP weights."""
    from tr_hash_i64.core.quantization import quantize_int8, quantize_int4

    if method not in ("int8", "int4"):
        return

    for name, module in model.named_modules():
        if not (hasattr(module, 'gate_up_proj') and hasattr(module, 'down_proj')):
            continue

        gate_up = module.gate_up_proj
        down = module.down_proj

        # gate_up_proj / down_proj are nn.Parameter (not nn.Linear)
        gu_data = gate_up.data if isinstance(gate_up, nn.Parameter) else gate_up.weight.data
        dn_data = down.data if isinstance(down, nn.Parameter) else down.weight.data

        if method == "int8":
            num_experts = gu_data.shape[0]
            gu_q, gu_s, dn_q, dn_s = [], [], [], []
            for e in range(num_experts):
                q, s = quantize_int8(gu_data[e])
                gu_q.append(q)
                gu_s.append(s)
                q, s = quantize_int8(dn_data[e])
                dn_q.append(q)
                dn_s.append(s)
            module.register_buffer("gate_up_int8", torch.stack(gu_q))
            module.register_buffer("gate_up_scale", torch.stack(gu_s))
            module.register_buffer("down_int8", torch.stack(dn_q))
            module.register_buffer("down_scale", torch.stack(dn_s))

        elif method == "int4":
            num_experts = gu_data.shape[0]
            gu_p, gu_s, gu_z = [], [], []
            dn_p, dn_s, dn_z = [], [], []
            for e in range(num_experts):
                p, s, z = quantize_int4(gu_data[e])
                gu_p.append(p)
                gu_s.append(s)
                gu_z.append(z)
                p, s, z = quantize_int4(dn_data[e])
                dn_p.append(p)
                dn_s.append(s)
                dn_z.append(z)
            module.register_buffer("gate_up_int4", torch.stack(gu_p))
            module.register_buffer("gate_up_scale_int4", torch.stack(gu_s))
            module.register_buffer("gate_up_zero", torch.stack(gu_z))
            module.register_buffer("down_int4", torch.stack(dn_p))
            module.register_buffer("down_scale_int4", torch.stack(dn_s))
            module.register_buffer("down_zero", torch.stack(dn_z))


def _quantize_attention(model: nn.Module, method: str):
    """
    Apply post-load INT8 quantization to LlamaAttention layers.

    Fused QKV: cat([Q, K, V], dim=0) → single INT8 matmul, split output.
    O: separate INT8 matmul + all_reduce (RowParallel replacement).
    """
    from tr_hash_i64.models.llama.model import LlamaAttention
    from tr_hash_i64.core.quantization import quantize_int8

    if method != "int8":
        return  # Only INT8 for attention (INT4 attention has too much error)

    for name, module in model.named_modules():
        if not isinstance(module, LlamaAttention):
            continue

        # Fused QKV — single matmul for all three projections
        q_w = module.q_proj.linear.weight.data
        k_w = module.k_proj.linear.weight.data
        v_w = module.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        module.register_buffer("qkv_int8", qkv_q)
        module.register_buffer("qkv_scale", qkv_s)
        module.q_size = q_w.shape[0]
        module.kv_size = k_w.shape[0]

        # QKV bias (if present)
        if module.q_proj.linear.bias is not None:
            qkv_bias = torch.cat([
                module.q_proj.linear.bias.data,
                module.k_proj.linear.bias.data,
                module.v_proj.linear.bias.data,
            ])
            module.register_buffer("qkv_bias", qkv_bias)

        # O projection — INT8 matmul (replaces RowParallelLinear)
        o_w = module.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        module.register_buffer("o_int8", o_q)
        module.register_buffer("o_scale", o_s)

        if module.o_proj.linear.bias is not None:
            module.register_buffer("o_bias", module.o_proj.linear.bias.data.clone())

        # Free float weights — INT8 buffers replace them
        module.q_proj.linear.weight = None
        module.k_proj.linear.weight = None
        module.v_proj.linear.weight = None
        module.o_proj.linear.weight = None


def _quantize_mu_attention(model: nn.Module, method: str):
    """
    Apply INT8 quantization to MuGuidedAttention layers.

    Fused QKV: cat([Q, K, V], dim=0) → single INT8 matmul, split output.
    Fused mu_QKV: cat([mu_to_q, mu_to_k, mu_to_v], dim=0) → single INT8 matmul.
    O: separate INT8 matmul.
    """
    from tr_hash_i64.models.complexity_deep.model import MuGuidedAttention
    from tr_hash_i64.core.quantization import quantize_int8

    if method != "int8":
        return

    for name, module in model.named_modules():
        if not isinstance(module, MuGuidedAttention):
            continue

        # Fused QKV
        q_w = module.q_proj.linear.weight.data
        k_w = module.k_proj.linear.weight.data
        v_w = module.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        module.register_buffer("qkv_int8", qkv_q)
        module.register_buffer("qkv_scale", qkv_s)
        module.q_size = q_w.shape[0]
        module.kv_size = k_w.shape[0]

        # QKV bias (if present)
        if module.q_proj.linear.bias is not None:
            qkv_bias = torch.cat([
                module.q_proj.linear.bias.data,
                module.k_proj.linear.bias.data,
                module.v_proj.linear.bias.data,
            ])
            module.register_buffer("qkv_bias", qkv_bias)

        # Fused mu projections: cat([mu_to_q, mu_to_k, mu_to_v])
        mu_q_w = module.mu_to_q.linear.weight.data
        mu_k_w = module.mu_to_k.linear.weight.data
        mu_v_w = module.mu_to_v.linear.weight.data
        mu_qkv_w = torch.cat([mu_q_w, mu_k_w, mu_v_w], dim=0)
        mu_qkv_q, mu_qkv_s = quantize_int8(mu_qkv_w)
        module.register_buffer("mu_qkv_int8", mu_qkv_q)
        module.register_buffer("mu_qkv_scale", mu_qkv_s)

        # O projection
        o_w = module.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        module.register_buffer("o_int8", o_q)
        module.register_buffer("o_scale", o_s)
        if module.o_proj.linear.bias is not None:
            module.register_buffer("o_bias", module.o_proj.linear.bias.data.clone())

        # Free float weights
        module.q_proj.linear.weight = None
        module.k_proj.linear.weight = None
        module.v_proj.linear.weight = None
        module.o_proj.linear.weight = None
        module.mu_to_q.linear.weight = None
        module.mu_to_k.linear.weight = None
        module.mu_to_v.linear.weight = None


def _quantize_i64_dynamics(model: nn.Module, method: str):
    """
    Quantize INLDynamics controller weights to INT8.

    Enables integer dynamics: INT8 controller matmuls + LUT activations
    (sigmoid, softplus, silu). Zero-FLOP activations after quantization.
    """
    from tr_hash_i64.models.complexity_deep.model import INLDynamics
    from tr_hash_i64.core.quantization import quantize_int8

    if method != "int8":
        return

    for name, module in model.named_modules():
        if not isinstance(module, INLDynamics):
            continue

        # Controller in
        ciq, cis = quantize_int8(module.controller_in.weight.data)
        module.register_buffer("ctrl_in_int8", ciq)
        module.register_buffer("ctrl_in_scale", cis)
        module.register_buffer("ctrl_in_bias", module.controller_in.bias.data.clone())

        # Controller out
        coq, cos = quantize_int8(module.controller_out.weight.data)
        module.register_buffer("ctrl_out_int8", coq)
        module.register_buffer("ctrl_out_scale", cos)
        module.register_buffer("ctrl_out_bias", module.controller_out.bias.data.clone())

        # mu_proj
        mpq, mps = quantize_int8(module.mu_proj.weight.data)
        module.register_buffer("mu_proj_int8", mpq)
        module.register_buffer("mu_proj_scale", mps)

        # Free float weights
        module.controller_in.weight = None
        module.controller_out.weight = None
        module.mu_proj.weight = None


def _quantize_lm_head(model: nn.Module, method: str):
    """
    Quantize lm_head (or tied embed_tokens) to INT8.

    lm_head is a large matmul: (batch, hidden) × (vocab, hidden)^T.
    INT8 accelerates logit computation significantly for large vocabularies.
    """
    from tr_hash_i64.core.quantization import quantize_int8

    if method != "int8":
        return

    # Tied embeddings: quantize embed_tokens weight for use as lm_head
    if getattr(model, 'tie_word_embeddings', False) and model.embed_tokens is not None:
        w = model.embed_tokens.weight.data
        wq, ws = quantize_int8(w)
        model.register_buffer("embed_int8", wq)
        model.register_buffer("embed_scale", ws)
        return

    # Separate lm_head
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        w = model.lm_head.weight.data
        wq, ws = quantize_int8(w)
        model.register_buffer("lm_head_int8", wq)
        model.register_buffer("lm_head_scale", ws)
        model.lm_head.weight = None


def _quantize_rmsnorm(model: nn.Module, method: str):
    """
    Quantize all RMSNorm weights to Q12 INT16 for integer forward path.

    Enables integer RMSNorm: float rsqrt (irreducible) + INT32 weight multiply.
    Only applied for INT8 quantization (full integer pipeline).
    """
    from tr_hash_i64.layers.rmsnorm import RMSNorm, quantize_rmsnorm

    if method != "int8":
        return  # Only INT8 triggers full integer pipeline

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            quantize_rmsnorm(module)


def _quantize_dense_mlp_fp8(model: nn.Module):
    """
    Quantize DenseMLP weights to FP8 E4M3 for H100/Ada tensor cores.

    ~2x throughput vs FP16 on Hopper GPUs. Falls back to float dequant on older GPUs.
    """
    from tr_hash_i64.layers.dense_mlp import DenseMLP
    from tr_hash_i64.core.fp8 import quantize_fp8

    for name, module in model.named_modules():
        if not isinstance(module, DenseMLP):
            continue

        # Gate projection
        w = module.gate_proj.linear.weight.data
        w_fp8, w_scale = quantize_fp8(w)
        module.register_buffer('gate_fp8', w_fp8)
        module.register_buffer('gate_fp8_scale', w_scale)

        # Up projection
        w = module.up_proj.linear.weight.data
        w_fp8, w_scale = quantize_fp8(w)
        module.register_buffer('up_fp8', w_fp8)
        module.register_buffer('up_fp8_scale', w_scale)

        # Down projection
        w = module.down_proj.linear.weight.data
        w_fp8, w_scale = quantize_fp8(w)
        module.register_buffer('down_fp8', w_fp8)
        module.register_buffer('down_fp8_scale', w_scale)
