"""Aggregation for auditable LongLive-RAG retrieval timing traces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


KINDS = ("latent_encoding", "topk_search")


def summarize_retrieval_profile(
    records: Sequence[Mapping[str, Any]],
    *,
    num_blocks: int,
    warmup_blocks: int,
) -> dict[str, Any]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if warmup_blocks < 0 or warmup_blocks >= num_blocks:
        raise ValueError("warmup_blocks must lie in [0, num_blocks)")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        kind = str(record.get("kind", ""))
        if kind not in KINDS:
            raise ValueError(f"record {index} has unsupported kind {kind!r}")
        block_index = int(record.get("block_index", -1))
        milliseconds = float(record.get("milliseconds", -1.0))
        if block_index < 0 or block_index >= num_blocks:
            raise ValueError(f"record {index} has invalid block_index")
        if milliseconds < 0.0:
            raise ValueError(f"record {index} has negative milliseconds")
        normalized.append(
            {
                "kind": kind,
                "block_index": block_index,
                "query_frame": int(record.get("query_frame", 0)),
                "milliseconds": milliseconds,
                "included_after_warmup": block_index >= warmup_blocks,
            }
        )

    measured_blocks = num_blocks - warmup_blocks
    included = [row for row in normalized if row["included_after_warmup"]]
    totals = {
        kind: sum(row["milliseconds"] for row in included if row["kind"] == kind)
        for kind in KINDS
    }
    counts = {
        kind: sum(row["kind"] == kind for row in included)
        for kind in KINDS
    }
    full_total = sum(row["milliseconds"] for row in normalized)
    measured_total = sum(totals.values())
    return {
        "schema_version": 1,
        "num_blocks": num_blocks,
        "warmup_blocks": warmup_blocks,
        "measured_blocks": measured_blocks,
        "aggregation_rule": (
            "sum synchronized CUDA-event durations after warmup and divide by "
            "the number of measured AR blocks; blocks without an eligible Top-K "
            "search contribute zero search time"
        ),
        "latent_encoding_ms_per_block": totals["latent_encoding"] / measured_blocks,
        "topk_search_ms_per_block": totals["topk_search"] / measured_blocks,
        "total_retrieval_ms_per_block": measured_total / measured_blocks,
        "total_retrieval_ms_per_rollout": full_total,
        "operation_counts_after_warmup": counts,
        "operation_mean_ms_after_warmup": {
            kind: (totals[kind] / counts[kind] if counts[kind] else 0.0)
            for kind in KINDS
        },
        "measured_total_retrieval_ms": measured_total,
        "raw_samples": normalized,
    }
