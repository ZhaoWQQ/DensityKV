from __future__ import annotations

import torch

from kv_cache import DensityKVBankConfig, DensityLimitedKVBank


def test_final_b3_policy_uses_a_synchronized_head_prefix() -> None:
    config = DensityKVBankConfig(
        max_entries=4,
        density_scale=1.0,
        process_all_candidates=True,
        update_chunk_size=2,
        full_update_mode="legacy_chunk_batch",
        legacy_density_gated_bootstrap_v2=True,
        legacy_bootstrap_v2_gate="all_anchor_growth_ratio",
        legacy_bootstrap_density_limit=2.0,
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
