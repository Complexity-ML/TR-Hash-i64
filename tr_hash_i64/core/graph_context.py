"""Execution context shared by CUDA-graph-safe model paths.

CUDA graph warm-up must execute exactly the same Python dispatch path as the
subsequent capture.  ``torch.cuda.is_current_stream_capturing()`` only becomes
true *inside* capture, so it cannot by itself select graph-safe kernels during
warm-up.
"""

from contextlib import contextmanager
from contextvars import ContextVar


_GRAPH_SAFE_MODE: ContextVar[bool] = ContextVar("tr_hash_graph_safe_mode", default=False)


def is_graph_safe_mode() -> bool:
    """Return whether graph-safe fixed-shape dispatch is explicitly enabled."""

    return _GRAPH_SAFE_MODE.get()


@contextmanager
def graph_safe_mode():
    """Select graph-safe kernels for both warm-up and graph capture."""

    token = _GRAPH_SAFE_MODE.set(True)
    try:
        yield
    finally:
        _GRAPH_SAFE_MODE.reset(token)
