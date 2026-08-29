from types import SimpleNamespace

import pytest

from tr_hash_i64.cli import _select_device


def _torch(*, cuda: bool = False, mps: bool = False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_auto_selects_available_accelerator():
    assert _select_device(_torch(cuda=True), "auto") == "cuda"
    assert _select_device(_torch(mps=True), "auto") == "mps"
    assert _select_device(_torch(), "auto") == "cpu"


def test_explicit_cuda_never_falls_back_to_cpu():
    with pytest.raises(SystemExit, match="Refusing to fall back to CPU"):
        _select_device(_torch(), "cuda")


def test_explicit_cpu_is_respected_even_when_cuda_exists():
    assert _select_device(_torch(cuda=True), "cpu") == "cpu"
