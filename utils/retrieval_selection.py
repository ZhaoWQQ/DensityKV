"""Deterministic helpers for paper retrieval ablations."""

from __future__ import annotations

import random


def deterministic_random_indices(
    *,
    num_eligible: int,
    count: int,
    batch_size: int,
    seed: int,
    query_frame: int,
) -> list[list[int]]:
    """Sample history indices reproducibly without touching global RNG state."""
    if num_eligible < 0:
        raise ValueError("num_eligible must be non-negative")
    if count < 0 or count > num_eligible:
        raise ValueError("count must lie in [0, num_eligible]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        local_seed = (
            int(seed)
            + int(query_frame) * 1_000_003
            + batch_index * 97_409
        )
        rows.append(random.Random(local_seed).sample(range(num_eligible), count))
    return rows
