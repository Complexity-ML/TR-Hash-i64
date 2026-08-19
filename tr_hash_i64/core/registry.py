"""Built-in model registry for the public 306.5M inference pair."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("tr_hash_i64.registry")


@dataclass(frozen=True)
class ModelEntry:
    """A model that can be selected by the CLI and exposed by the API."""

    name: str
    model_class: str
    config_loader: str
    config_path: Optional[str] = None
    checkpoint: Optional[str] = None
    parameters: str = ""
    description: str = ""


_REGISTRY: Dict[str, ModelEntry] = {}

_COMPLEXITY_DEEP = (
    "tr_hash_i64.models.complexity_deep.model.ComplexityDeepModel",
    "tr_hash_i64.models.complexity_deep.config.ComplexityDeepConfig",
)

_ARCHITECTURE_MAP: Dict[str, Tuple[str, str]] = {
    "DeepForCausalLM": _COMPLEXITY_DEEP,
}


def register_model(
    name: str,
    model_class: str,
    config_loader: str,
    config_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    parameters: str = "",
    description: str = "",
) -> None:
    """Register a model entry."""

    _REGISTRY[name] = ModelEntry(
        name=name,
        model_class=model_class,
        config_loader=config_loader,
        config_path=config_path,
        checkpoint=checkpoint,
        parameters=parameters,
        description=description,
    )


def get_model_entry(name: str) -> ModelEntry:
    """Return a registered model or raise with the complete supported list."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown model: {name}. Available: {available}"
        ) from exc


def get_checkpoint_path(name: str) -> Optional[str]:
    return get_model_entry(name).checkpoint


def list_models() -> list[dict[str, str]]:
    return [
        {
            "name": entry.name,
            "model_class": entry.model_class,
            "parameters": entry.parameters,
            "description": entry.description,
        }
        for entry in _REGISTRY.values()
    ]


def resolve_architecture(
    checkpoint_path: str,
) -> Optional[Tuple[str, str, str]]:
    """Resolve a local Hugging Face directory from its ``architectures`` key."""

    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.exists():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    for architecture in data.get("architectures", []):
        resolved = _ARCHITECTURE_MAP.get(architecture)
        if resolved is not None:
            logger.info(
                "Auto-detected architecture: %s -> %s",
                architecture,
                resolved[0].rsplit(".", 1)[-1],
            )
            return resolved[0], resolved[1], str(config_path)
    return None


register_model(
    name="tr-hash-moe-500m",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    checkpoint="Pacific-i64/TR-HASH-MOE-500M-HF",
    parameters="492.1M",
    description="Balanced token-ID hash top-2 residual experts with shared SwiGLU",
)

register_model(
    name="tr-hash-moe-200m",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    checkpoint="AETHORIA-AI/tr-hash-200m-70b-replay-checkpoints",
    parameters="201.2M",
    description=(
        "In-progress 200M multi-hash MoE replay run — resolves to the "
        "latest token_pack_NNN_* checkpoint in the repo, not a stable release"
    ),
)

register_model(
    name="tr-moe-306",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    checkpoint="Pacific-i64/TR-MOE-306",
    parameters="306.5M",
    description="Fixed top-2 token-routed residual experts with shared SwiGLU",
)

register_model(
    name="dense-306",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    checkpoint="Pacific-i64/Dense-306",
    parameters="306.5M",
    description="Width-matched dense SwiGLU baseline",
)
