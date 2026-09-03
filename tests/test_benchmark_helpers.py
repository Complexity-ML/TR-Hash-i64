import torch

from benchmarks.bench_cpu_prefill import checkpoint_sha256
from benchmarks.bench_piqa_cuda_graph import _prefill_last_logits


def test_prefill_helper_projects_only_last_rows_for_supporting_model():
    class RecordingModel:
        supports_logits_indices = True

        def __init__(self):
            self.indices = None

        def __call__(self, token_ids, logits_indices=None, **_kwargs):
            self.indices = logits_indices
            logits = torch.arange(token_ids.numel() * 5).reshape(token_ids.numel(), 5)
            if logits_indices is not None:
                logits = logits.index_select(0, logits_indices)
            return logits

    model = RecordingModel()
    token_ids = torch.tensor([1, 2, 3, 4, 5, 6])

    logits = _prefill_last_logits(
        model,
        {"token_ids": token_ids},
        [1, 5],
    )

    assert model.indices is not None
    assert model.indices.tolist() == [1, 5]
    assert logits.shape == (2, 5)


def test_checkpoint_hash_is_stable_and_content_sensitive(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"first")

    first = checkpoint_sha256(tmp_path)
    assert checkpoint_sha256(tmp_path) == first

    shard.write_bytes(b"second")
    assert checkpoint_sha256(tmp_path) != first
