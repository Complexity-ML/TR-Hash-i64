"""
tr-hash-i64 :: Core

Generic infrastructure — not tied to any specific model.
  - registry: model registration
  - tokenizer: text ↔ i64 conversion
  - sampling: token sampling strategies
  - kv_cache: paged KV cache with integer block management
  - loader: checkpoint weight loading
"""

from tr_hash_i64.core.registry import register_model, get_model_entry, get_checkpoint_path, list_models
from tr_hash_i64.core.tokenizer import I64Tokenizer, load_tokenizer
from tr_hash_i64.core.sampling import SamplingParams, sample_token, sample_batch
from tr_hash_i64.core.kv_cache import PagedKVCache
from tr_hash_i64.core.context_manager import ContextManager, ContextPlan, ContextWindowError
from tr_hash_i64.core.loader import load_checkpoint, load_model_by_name

__all__ = [
    "register_model", "get_model_entry", "get_checkpoint_path", "list_models",
    "I64Tokenizer", "load_tokenizer",
    "SamplingParams", "sample_token", "sample_batch",
    "PagedKVCache",
    "ContextManager",
    "ContextPlan",
    "ContextWindowError",
    "load_checkpoint", "load_model_by_name",
]
