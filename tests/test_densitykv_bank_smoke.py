from __future__ import annotations

import torch
import pytest

from kv_cache import DensityKVBankConfig, DensityLimitedKVBank
import kv_cache.density_bank as density_bank_module


def test_published_policy_uses_a_synchronized_head_prefix() -> None:
    config = DensityKVBankConfig(
        max_entries=4,
        density_scale=1.0,
        density_growth_limit=2.0,
        work_chunk_size=2,
        compute_dtype="float32",
        fast_impl="torch",
    )
    bank = DensityLimitedKVBank(
        groups=2,
        key_dim=4,
        value_dim=3,
        config=config,
        device="cpu",
    )
    keys = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        ]
    )
    values = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)

    first = bank.update(keys, values)
    second = bank.update(keys + 2.0, values + 10.0)

    assert first.accepted_count.tolist() == [2, 2]
    assert second.accepted_count.tolist() == [2, 2]
    assert bank.counts.tolist() == [4, 4]
    assert bool((bank.counts <= config.max_entries).all())


def test_auto_triton_failure_falls_back_to_torch(monkeypatch) -> None:
    def fail_triton(*args, **kwargs):
        raise OSError("temporary compiler cache is unavailable")

    monkeypatch.setattr(density_bank_module, "triton_density_sum", fail_triton)
    bank = DensityLimitedKVBank(
        groups=1,
        key_dim=2,
        value_dim=2,
        config=DensityKVBankConfig(
            fast_impl="auto",
            compute_dtype="float32",
        ),
        device="cpu",
    )
    squared_distance = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    with pytest.warns(RuntimeWarning, match="falling back to PyTorch"):
        actual = bank._density_sum(squared_distance, exclude_diagonal=True)
    expected = bank._density_contribution(squared_distance)
    expected.diagonal(dim1=-2, dim2=-1).zero_()

    torch.testing.assert_close(actual, expected.sum(dim=-1))
    assert bank._auto_triton_failed
