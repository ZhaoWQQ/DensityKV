"""Infinity-RoPE adapter for the LongLive-RAG paper baseline.

The implementation follows ``yesiltepe-hidir/infinity-rope`` at revision
``63d7b84f043a2536ccb62d5171ed54dfadc0a721``.  The LongLive-RAG main table
uses one static prompt per video, so only Block-Relativistic RoPE applies.
KV Flush is a prompt-switch operator and RoPE Cut is a scene-cut operator;
both must remain disabled for this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


UPSTREAM_REPOSITORY = "https://github.com/yesiltepe-hidir/infinity-rope"
UPSTREAM_REVISION = "63d7b84f043a2536ccb62d5171ed54dfadc0a721"


@dataclass(frozen=True)
class InfinityRoPEConfig:
    enabled: bool = False
    block_relativistic_rope: bool = False
    kv_flush: bool = False
    rope_cut: bool = False
    static_single_prompt: bool = True
    attention_budget_frames: int = 12
    sink_frames: int = 1
    upstream_revision: str = UPSTREAM_REVISION


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def parse_infinity_rope_config(config: Any) -> InfinityRoPEConfig:
    """Normalize and validate the explicit model config."""

    parsed = InfinityRoPEConfig(
        enabled=bool(_config_value(config, "enabled", False)),
        block_relativistic_rope=bool(
            _config_value(config, "block_relativistic_rope", False)
        ),
        kv_flush=bool(_config_value(config, "kv_flush", False)),
        rope_cut=bool(_config_value(config, "rope_cut", False)),
        static_single_prompt=bool(
            _config_value(config, "static_single_prompt", True)
        ),
        attention_budget_frames=int(
            _config_value(config, "attention_budget_frames", 12)
        ),
        sink_frames=int(_config_value(config, "sink_frames", 1)),
        upstream_revision=str(
            _config_value(config, "upstream_revision", UPSTREAM_REVISION)
        ),
    )
    if not parsed.enabled:
        return parsed
    if not parsed.block_relativistic_rope:
        raise ValueError(
            "Infinity-RoPE main-table baseline requires block_relativistic_rope=true"
        )
    if not parsed.static_single_prompt:
        raise ValueError(
            "this adapter is scoped to the static-single-prompt main-table baseline"
        )
    if parsed.kv_flush:
        raise ValueError(
            "KV Flush is a prompt-switch operator and is excluded from the static baseline"
        )
    if parsed.rope_cut:
        raise ValueError(
            "RoPE Cut is a scene-transition operator and is excluded from the static baseline"
        )
    if parsed.attention_budget_frames <= 0:
        raise ValueError("attention_budget_frames must be positive")
    if parsed.sink_frames < 0:
        raise ValueError("sink_frames must be non-negative")
    if parsed.sink_frames >= parsed.attention_budget_frames:
        raise ValueError("sink_frames must be smaller than the attention budget")
    if parsed.upstream_revision != UPSTREAM_REVISION:
        raise ValueError(
            "Infinity-RoPE source revision mismatch: "
            f"expected {UPSTREAM_REVISION}, got {parsed.upstream_revision}"
        )
    return parsed


def attach_infinity_rope_adapter(
    model: Any,
    config: Any,
    *,
    local_attn_size: Any,
    sink_size: int,
) -> InfinityRoPEConfig:
    """Attach one shared immutable adapter config to all self-attention layers."""

    parsed = parse_infinity_rope_config(config)
    if parsed.enabled:
        local_values = (
            list(local_attn_size)
            if not isinstance(local_attn_size, int)
            and hasattr(local_attn_size, "__iter__")
            else [int(local_attn_size)]
        )
        if any(value != parsed.attention_budget_frames for value in local_values):
            raise ValueError(
                "Infinity-RoPE attention budget mismatch: "
                f"model={local_values}, config={parsed.attention_budget_frames}"
            )
        if int(sink_size) != parsed.sink_frames:
            raise ValueError(
                "Infinity-RoPE sink mismatch: "
                f"model={sink_size}, config={parsed.sink_frames}"
            )

    model.infinity_rope_config = parsed
    for block in getattr(model, "blocks", []):
        block.self_attn.infinity_rope_config = parsed
    return parsed


def infinity_rope_enabled(owner: Any) -> bool:
    config = getattr(owner, "infinity_rope_config", None)
    return bool(config is not None and config.enabled)


def block_relative_frame_indices(
    *,
    current_start_frame: int,
    num_query_frames: int,
    resident_cache_frames: int,
    cache_capacity_frames: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official moving-frame positions for query and resident cache K.

    Before the cache fills, query positions advance normally.  Once rolling
    starts, the current block stays at the right edge while resident keys are
    re-indexed from zero on every attention call.
    """

    current_start_frame = int(current_start_frame)
    num_query_frames = int(num_query_frames)
    resident_cache_frames = int(resident_cache_frames)
    cache_capacity_frames = int(cache_capacity_frames)
    if num_query_frames <= 0:
        raise ValueError("num_query_frames must be positive")
    if cache_capacity_frames <= 0:
        raise ValueError("cache_capacity_frames must be positive")
    if num_query_frames > cache_capacity_frames:
        raise ValueError("query block cannot exceed the cache capacity")
    if not 0 <= resident_cache_frames <= cache_capacity_frames:
        raise ValueError("resident_cache_frames is outside the cache capacity")

    query_start = min(
        current_start_frame,
        cache_capacity_frames - num_query_frames,
    )
    query_indices = torch.arange(
        query_start,
        query_start + num_query_frames,
        dtype=torch.long,
        device=device,
    )
    key_indices = torch.arange(
        resident_cache_frames,
        dtype=torch.long,
        device=device,
    )
    return query_indices, key_indices


def apply_block_relativistic_rope(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply the upstream 3D RoPE kernel with explicit temporal positions."""

    if x.ndim != 4:
        raise ValueError("x must have shape [batch, tokens, heads, head_dim]")
    if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
        raise ValueError("grid_sizes must have shape [batch, 3]")
    if x.shape[0] != grid_sizes.shape[0]:
        raise ValueError("x and grid_sizes batch dimensions differ")
    if x.shape[-1] % 2:
        raise ValueError("RoPE head_dim must be even")

    num_heads = x.size(2)
    complex_channels = x.size(3) // 2
    split_sizes = [
        complex_channels - 2 * (complex_channels // 3),
        complex_channels // 3,
        complex_channels // 3,
    ]
    temporal_freqs, height_freqs, width_freqs = freqs.split(split_sizes, dim=1)
    frame_indices = frame_indices.to(device=freqs.device, dtype=torch.long)
    if frame_indices.numel() < int(grid_sizes[:, 0].max().item()):
        raise ValueError("frame_indices is shorter than the temporal grid")
    if frame_indices.numel() and (
        int(frame_indices.min().item()) < 0
        or int(frame_indices.max().item()) >= temporal_freqs.shape[0]
    ):
        raise ValueError("frame_indices exceeds the available RoPE frequencies")

    output = []
    for sample_index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        seq_len = frames * height * width
        x_sample = torch.view_as_complex(
            x[sample_index, :seq_len]
            .to(torch.float64)
            .reshape(seq_len, num_heads, -1, 2)
        )
        sample_frame_indices = frame_indices[:frames]
        sample_freqs = torch.cat(
            [
                temporal_freqs[sample_frame_indices]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                height_freqs[:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                width_freqs[:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        rotated = torch.view_as_real(x_sample * sample_freqs).flatten(2)
        output.append(torch.cat([rotated, x[sample_index, seq_len:]]))
    return torch.stack(output).type_as(x)


def apply_infinity_rope_or_legacy(
    owner: Any,
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    frame_indices: torch.Tensor,
    legacy_apply: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Use BR-RoPE when enabled; otherwise return the untouched legacy path."""

    if not infinity_rope_enabled(owner):
        return legacy_apply()
    return apply_block_relativistic_rope(
        x,
        grid_sizes,
        freqs,
        frame_indices=frame_indices,
    )


def cache_key_for_storage(
    owner: Any,
    *,
    raw_key: torch.Tensor,
    legacy_roped_key: torch.Tensor,
) -> torch.Tensor:
    """Store raw K for dynamic re-rotation, preserving legacy storage when off."""

    return raw_key if infinity_rope_enabled(owner) else legacy_roped_key
