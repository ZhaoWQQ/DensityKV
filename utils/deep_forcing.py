"""Deep Forcing persistent-context helpers for the native Wan KV cache.

This module ports the public PC implementation from
``cvlab-kaist/DeepForcing`` at the pinned revision below.  The LongLive-RAG
paper does not publish its private Recent/reuse settings, so the Table 2
adapter combines the paper's 12-frame attention budget with Deep Forcing's
public defaults: one sink frame, four recent frames, and seven-frame-equivalent
token-level Top-C with a reuse limit of seven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import math
import torch


UPSTREAM_REPOSITORY = "https://github.com/cvlab-kaist/DeepForcing"
UPSTREAM_REVISION = "ea9961aa4bd5b554daeecdcf59e37c495fba9df6"
OFFICIAL_CODE_COMPAT = "official_code_compat"


@dataclass(frozen=True)
class DeepForcingConfig:
    enabled: bool = False
    capacity_frames: int = 12
    recent_frames: int = 4
    fusion: str = "sum"
    keep_sinks: bool = True
    topc_max_reuse: int = 7
    rope_semantics: str = OFFICIAL_CODE_COMPAT
    upstream_revision: str = UPSTREAM_REVISION


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def parse_deep_forcing_config(config: Any) -> DeepForcingConfig:
    parsed = DeepForcingConfig(
        enabled=bool(_config_value(config, "enabled", False)),
        capacity_frames=int(_config_value(config, "capacity_frames", 12)),
        recent_frames=int(_config_value(config, "recent_frames", 4)),
        fusion=str(_config_value(config, "fusion", "sum")),
        keep_sinks=bool(_config_value(config, "keep_sinks", True)),
        topc_max_reuse=int(_config_value(config, "topc_max_reuse", 7)),
        rope_semantics=str(
            _config_value(config, "rope_semantics", OFFICIAL_CODE_COMPAT)
        ),
        upstream_revision=str(
            _config_value(config, "upstream_revision", UPSTREAM_REVISION)
        ),
    )
    if not parsed.enabled:
        return parsed
    if parsed.capacity_frames <= 0:
        raise ValueError("Deep Forcing capacity_frames must be positive")
    if parsed.recent_frames <= 0:
        raise ValueError("Deep Forcing recent_frames must be positive")
    if parsed.recent_frames >= parsed.capacity_frames:
        raise ValueError("Deep Forcing recent_frames must be below capacity")
    if parsed.fusion not in {"sum", "max"}:
        raise ValueError("Deep Forcing fusion must be 'sum' or 'max'")
    if parsed.topc_max_reuse < 0:
        raise ValueError("Deep Forcing topc_max_reuse must be non-negative")
    if parsed.rope_semantics != OFFICIAL_CODE_COMPAT:
        raise ValueError(
            "the Table 2 adapter only exposes official_code_compat RoPE semantics"
        )
    if parsed.upstream_revision != UPSTREAM_REVISION:
        raise ValueError(
            "Deep Forcing source revision mismatch: "
            f"expected {UPSTREAM_REVISION}, got {parsed.upstream_revision}"
        )
    return parsed


def attach_deep_forcing_adapter(
    model: Any,
    config: Any,
    *,
    local_attn_size: Any,
    sink_size: int,
) -> DeepForcingConfig:
    """Attach one immutable config to every native self-attention layer."""

    parsed = parse_deep_forcing_config(config)
    if parsed.enabled:
        infinity = getattr(model, "infinity_rope_config", None)
        if bool(getattr(infinity, "enabled", False)):
            raise ValueError("Deep Forcing and Infinity-RoPE are mutually exclusive")
        local_values = (
            list(local_attn_size)
            if not isinstance(local_attn_size, int)
            and hasattr(local_attn_size, "__iter__")
            else [int(local_attn_size)]
        )
        if any(int(value) != parsed.capacity_frames for value in local_values):
            raise ValueError(
                "Deep Forcing physical/attention budget mismatch: "
                f"model={local_values}, config={parsed.capacity_frames}"
            )
        if int(sink_size) != 1:
            raise ValueError("Deep Forcing Table 2 baseline requires sink_size=1")
        if not parsed.keep_sinks:
            raise ValueError("Deep Forcing Table 2 baseline must keep the sink")
        if parsed.capacity_frames - int(sink_size) - parsed.recent_frames <= 0:
            raise ValueError("Deep Forcing has no room for Top-C tokens")

    model.deep_forcing_config = parsed
    for block in getattr(model, "blocks", []):
        block.self_attn.deep_forcing_config = parsed
    return parsed


def deep_forcing_enabled(owner: Any) -> bool:
    config = getattr(owner, "deep_forcing_config", None)
    return bool(config is not None and config.enabled)


def update_recent_queries(
    previous: torch.Tensor | None,
    new_queries: torch.Tensor,
    window_tokens: int,
) -> torch.Tensor:
    """Append detached RoPE-applied Q and retain the most recent R tokens."""

    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    new_queries = new_queries.detach()
    combined = new_queries if previous is None else torch.cat([previous, new_queries], dim=1)
    return combined[:, -window_tokens:]


def persistent_context_scores(
    recent_queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    fusion: str = "sum",
) -> torch.Tensor:
    """Compute the official layer-wise score shared by all attention heads."""

    if recent_queries.ndim != 4 or keys.ndim != 4:
        raise ValueError("recent_queries and keys must be [B,T,H,D]")
    if recent_queries.shape[0] != keys.shape[0]:
        raise ValueError("query/key batch sizes differ")
    if recent_queries.shape[2:] != keys.shape[2:]:
        raise ValueError("query/key head shapes differ")
    num_heads = recent_queries.shape[2]
    head_dim = recent_queries.shape[3]
    scale = 1.0 / (num_heads * math.sqrt(head_dim))
    key_flat_t = keys.reshape(keys.shape[0], keys.shape[1], -1).transpose(1, 2)
    query_flat = recent_queries.reshape(recent_queries.shape[0], recent_queries.shape[1], -1)
    if fusion == "sum":
        query_sum = query_flat.sum(dim=1, keepdim=True)
        return torch.bmm(query_sum, key_flat_t).squeeze(1).float() * scale
    if fusion == "max":
        return torch.bmm(query_flat, key_flat_t).float().amax(dim=1) * scale
    raise ValueError(f"unsupported Deep Forcing fusion: {fusion}")


def select_persistent_indices(
    scores: torch.Tensor,
    *,
    total_tokens: int,
    recent_tokens: int,
    sink_tokens: int,
    top_c: int,
    topc_counts: torch.Tensor | None = None,
    topc_max_reuse: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return chronological keep indices and selected Top-C indices per batch."""

    if scores.ndim != 2 or scores.shape[1] != total_tokens:
        raise ValueError("scores must have shape [B,total_tokens]")
    device = scores.device
    recent_start = max(0, total_tokens - recent_tokens)
    candidate_start = min(sink_tokens, recent_start)
    candidate_indices = torch.arange(candidate_start, recent_start, device=device)
    sink_indices = torch.arange(min(sink_tokens, total_tokens), device=device)
    recent_indices = torch.arange(recent_start, total_tokens, device=device)
    keep_lists: list[torch.Tensor] = []
    selected_lists: list[torch.Tensor] = []

    for batch_index in range(scores.shape[0]):
        candidate_scores = scores[batch_index, candidate_start:recent_start]
        allowed_indices = candidate_indices
        if topc_counts is not None and topc_max_reuse > 0:
            candidate_counts = topc_counts[
                batch_index, candidate_start:recent_start
            ]
            allowed = candidate_counts < topc_max_reuse
            candidate_scores = candidate_scores[allowed]
            allowed_indices = candidate_indices[allowed]
        select_count = min(max(int(top_c), 0), int(allowed_indices.numel()))
        if select_count:
            top_local = torch.topk(candidate_scores, k=select_count, dim=0).indices
            selected = torch.sort(allowed_indices[top_local]).values
        else:
            selected = torch.empty(0, dtype=torch.long, device=device)
        keep = torch.unique(
            torch.cat([sink_indices, selected, recent_indices]), sorted=True
        )
        keep_lists.append(keep)
        selected_lists.append(selected)
    return keep_lists, selected_lists


def pack_persistent_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    keep_lists: Sequence[torch.Tensor],
    selected_lists: Sequence[torch.Tensor],
    *,
    capacity_tokens: int,
    source_counts: torch.Tensor,
    source_abs_frames: torch.Tensor,
    topc_max_reuse: int,
) -> dict[str, torch.Tensor | int]:
    """Pack K/V and metadata into the front of a fixed physical cache."""

    batch, _, heads, dim = keys.shape
    if values.shape != keys.shape:
        raise ValueError("Deep Forcing expects K and V to have identical shapes")
    if len(keep_lists) != batch or len(selected_lists) != batch:
        raise ValueError("one keep/selection list is required per batch item")
    active_tokens = max((int(indices.numel()) for indices in keep_lists), default=0)
    if active_tokens > capacity_tokens:
        raise ValueError("packed cache exceeds physical capacity")

    packed_k = keys.new_zeros((batch, capacity_tokens, heads, dim))
    packed_v = values.new_zeros((batch, capacity_tokens, heads, dim))
    packed_counts = source_counts.new_zeros((batch, capacity_tokens))
    packed_abs = source_abs_frames.new_full((batch, capacity_tokens), -1)
    protected = torch.zeros(
        (batch, capacity_tokens), dtype=torch.bool, device=keys.device
    )

    for batch_index, keep in enumerate(keep_lists):
        keep_len = int(keep.numel())
        if keep_len == 0:
            continue
        packed_k[batch_index, :keep_len] = keys[batch_index, keep]
        packed_v[batch_index, :keep_len] = values[batch_index, keep]
        packed_counts[batch_index, :keep_len] = source_counts[batch_index, keep]
        packed_abs[batch_index, :keep_len] = source_abs_frames[batch_index, keep]
        selected = selected_lists[batch_index]
        if selected.numel():
            packed_positions = torch.searchsorted(keep, selected)
            protected[batch_index, packed_positions] = True
            if topc_max_reuse > 0:
                packed_counts[batch_index, packed_positions] += 1

    return {
        "k": packed_k,
        "v": packed_v,
        "topc_select_counts": packed_counts,
        "abs_frame_idx": packed_abs,
        "protected_mask": protected,
        "active_tokens": active_tokens,
    }


def should_compress_cache(
    *,
    current_end: int,
    global_end: int,
    local_end: int,
    num_new_tokens: int,
    capacity_tokens: int,
) -> bool:
    """PC fires once: the first forward of a new block that would overflow."""

    return current_end > global_end and local_end + num_new_tokens > capacity_tokens


def initial_sink_relocation_delta(
    physical_cache_frames: int,
    capacity_frames: int,
) -> int:
    """Generalized replacement for the upstream hard-coded ``21-capacity``."""

    return int(physical_cache_frames) - int(capacity_frames)


def official_compat_topc_relocation_bounds(
    *,
    sink_tokens: int,
    local_end: int,
    max_attention_tokens: int,
) -> tuple[int, int]:
    """Return the public implementation's Top-C relocation slice.

    Under the released fixed-capacity configuration ``local_end == max_attention``
    and this deliberately returns an empty ``[sink,sink)`` interval.
    """

    tail_start = max(
        int(sink_tokens),
        int(local_end) - int(max_attention_tokens) + int(sink_tokens),
    )
    return int(sink_tokens), tail_start


def rotate_temporal_rope_delta_(
    keys: torch.Tensor,
    freqs: torch.Tensor,
    delta_frames: int,
) -> None:
    """In-place temporal-only RoPE phase shift; spatial channels stay untouched."""

    if delta_frames == 0 or keys.numel() == 0:
        return
    if keys.ndim != 4 or keys.shape[-1] % 2:
        raise ValueError("keys must be [B,T,H,D] with an even head dimension")
    complex_channels = keys.shape[-1] // 2
    temporal_channels = complex_channels - 2 * (complex_channels // 3)
    spatial_channels = complex_channels // 3
    temporal_freqs, _, _ = freqs.split(
        [temporal_channels, spatial_channels, spatial_channels], dim=1
    )
    shift = min(abs(int(delta_frames)), temporal_freqs.shape[0] - 1)
    multiplier = temporal_freqs[shift]
    if delta_frames < 0:
        multiplier = torch.conj(multiplier)
    real_imag = keys[..., : 2 * temporal_channels]
    complex_values = torch.view_as_complex(
        real_imag.to(torch.float64).reshape(-1, temporal_channels, 2)
    )
    rotated = complex_values * multiplier.view(1, temporal_channels).to(
        complex_values.dtype
    )
    real_imag.copy_(
        torch.view_as_real(rotated)
        .reshape(*keys.shape[:3], temporal_channels, 2)
        .flatten(-2)
        .to(real_imag.dtype)
    )


def prepare_deep_forcing_cache(
    owner: Any,
    kv_cache: dict[str, Any],
    *,
    roped_query: torch.Tensor,
    roped_key: torch.Tensor,
    value: torch.Tensor,
    freqs: torch.Tensor,
    current_start: int,
    current_end: int,
    frame_seq_length: int,
    sink_tokens: int,
    max_attention_tokens: int,
) -> dict[str, Any]:
    """Build one native-cache update without mutating K/V before layer commit."""

    config: DeepForcingConfig = owner.deep_forcing_config
    capacity_tokens = config.capacity_frames * frame_seq_length
    physical_tokens = int(kv_cache["k"].shape[1])
    if physical_tokens != capacity_tokens:
        raise ValueError(
            "Deep Forcing requires physical cache == configured capacity: "
            f"physical={physical_tokens}, configured={capacity_tokens}"
        )
    if int(max_attention_tokens) != capacity_tokens:
        raise ValueError(
            "Deep Forcing requires attention budget == physical capacity: "
            f"attention={max_attention_tokens}, capacity={capacity_tokens}"
        )

    batch = roped_key.shape[0]
    local_end_before = int(kv_cache["local_end_index"].item())
    global_end = int(kv_cache["global_end_index"].item())
    num_new_tokens = int(roped_key.shape[1])
    current_start_frame = int(current_start // frame_seq_length)
    is_recompute = current_end <= global_end and current_start > 0

    recent_window_tokens = config.recent_frames * frame_seq_length
    win_q = update_recent_queries(
        kv_cache.get("win_q"), roped_query, recent_window_tokens
    )
    kv_cache["win_q"] = win_q

    counts = kv_cache.get("topc_select_counts")
    if counts is None or tuple(counts.shape) != (batch, physical_tokens):
        counts = torch.zeros(
            (batch, physical_tokens), dtype=torch.long, device=roped_key.device
        )
    abs_frames = kv_cache.get("abs_frame_idx")
    if abs_frames is None or tuple(abs_frames.shape) != (batch, physical_tokens):
        abs_frames = torch.full(
            (batch, physical_tokens), -1, dtype=torch.long, device=roped_key.device
        )

    compress = should_compress_cache(
        current_end=current_end,
        global_end=global_end,
        local_end=local_end_before,
        num_new_tokens=num_new_tokens,
        capacity_tokens=capacity_tokens,
    )
    new_abs = current_start_frame + torch.div(
        torch.arange(num_new_tokens, device=roped_key.device),
        frame_seq_length,
        rounding_mode="floor",
    )
    new_abs = new_abs.unsqueeze(0).expand(batch, -1)

    if compress:
        keys_augmented = torch.cat(
            [kv_cache["k"][:, :local_end_before], roped_key], dim=1
        )
        values_augmented = torch.cat(
            [kv_cache["v"][:, :local_end_before], value], dim=1
        )
        counts_augmented = torch.cat(
            [
                counts[:, :local_end_before],
                counts.new_zeros((batch, num_new_tokens)),
            ],
            dim=1,
        )
        abs_augmented = torch.cat(
            [abs_frames[:, :local_end_before], new_abs], dim=1
        )
        total_augmented = int(keys_augmented.shape[1])
        recent_tokens = min(int(win_q.shape[1]), total_augmented)
        recent_q = win_q[:, -recent_tokens:]
        scores = persistent_context_scores(
            recent_q, keys_augmented, fusion=config.fusion
        )
        scores[:, total_augmented - recent_tokens :] = -float("inf")
        forced_sink = min(sink_tokens if config.keep_sinks else 0, total_augmented)
        if forced_sink:
            scores[:, :forced_sink] = -float("inf")
        top_c = max(0, capacity_tokens - forced_sink - recent_tokens)
        keep_lists, selected_lists = select_persistent_indices(
            scores,
            total_tokens=total_augmented,
            recent_tokens=recent_tokens,
            sink_tokens=forced_sink,
            top_c=top_c,
            topc_counts=counts_augmented,
            topc_max_reuse=config.topc_max_reuse,
        )
        packed = pack_persistent_cache(
            keys_augmented,
            values_augmented,
            keep_lists,
            selected_lists,
            capacity_tokens=capacity_tokens,
            source_counts=counts_augmented,
            source_abs_frames=abs_augmented,
            topc_max_reuse=config.topc_max_reuse,
        )
        temp_k = packed["k"]
        temp_v = packed["v"]
        temp_counts = packed["topc_select_counts"]
        temp_abs = packed["abs_frame_idx"]
        protected_mask = packed["protected_mask"]
        local_end = int(packed["active_tokens"])
        local_start = max(0, local_end - num_new_tokens)

        _, top_end = official_compat_topc_relocation_bounds(
            sink_tokens=forced_sink,
            local_end=local_end,
            max_attention_tokens=max_attention_tokens,
        )
        tail_length_tokens = local_end - top_end
        tail_length_frames = tail_length_tokens // frame_seq_length
        sink_length_frames = forced_sink // frame_seq_length
        desired_sink_start = (
            current_start_frame - tail_length_frames - sink_length_frames
        )
        sink_base = kv_cache.get("sink_base_abs_start_frame")
        if sink_base is None:
            sink_delta = initial_sink_relocation_delta(
                physical_tokens // frame_seq_length,
                config.capacity_frames,
            )
        else:
            sink_delta = desired_sink_start - int(sink_base.item())
        if forced_sink:
            rotate_temporal_rope_delta_(temp_k[:, :forced_sink], freqs, sink_delta)
            sink_offsets = torch.arange(forced_sink, device=temp_k.device)
            temp_abs[:, :forced_sink] = desired_sink_start + torch.div(
                sink_offsets, frame_seq_length, rounding_mode="floor"
            )

        top_start, top_end = official_compat_topc_relocation_bounds(
            sink_tokens=forced_sink,
            local_end=local_end,
            max_attention_tokens=max_attention_tokens,
        )
        topc_base_value = None
        if top_end > top_start:
            top_length_tokens = top_end - top_start
            top_length_frames = math.ceil(top_length_tokens / frame_seq_length)
            tail_start_abs = current_start_frame - tail_length_frames
            desired_top_start = tail_start_abs - top_length_frames
            topc_base = kv_cache.get("topc_base_abs_start_frame")
            topc_delta = (
                0
                if topc_base is None
                else desired_top_start - int(topc_base.item())
            )
            rotate_temporal_rope_delta_(
                temp_k[:, top_start:top_end], freqs, topc_delta
            )
            topc_base_value = desired_top_start
        sink_base_value = desired_sink_start
    else:
        local_end = local_end_before + current_end - global_end
        local_start = local_end - num_new_tokens
        if local_start < 0 or local_end > capacity_tokens:
            raise RuntimeError(
                "Deep Forcing direct overwrite lies outside the physical cache"
            )
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        temp_counts = counts.clone()
        temp_abs = abs_frames.clone()
        protected_mask = kv_cache.get("protected_mask")
        if protected_mask is None:
            protected_mask = torch.zeros(
                (batch, capacity_tokens),
                dtype=torch.bool,
                device=roped_key.device,
            )
        else:
            protected_mask = protected_mask.clone()
        temp_k[:, local_start:local_end] = roped_key
        temp_v[:, local_start:local_end] = value
        temp_counts[:, local_start:local_end] = 0
        temp_abs[:, local_start:local_end] = new_abs
        protected_mask[:, local_start:local_end] = False
        sink_base_value = None
        topc_base_value = None

    return {
        "action": "deep_forcing_replace",
        "new_k": temp_k,
        "new_v": temp_v,
        "topc_select_counts": temp_counts,
        "abs_frame_idx": temp_abs,
        "protected_mask": protected_mask,
        "protected_len": protected_mask.sum(dim=1, dtype=torch.long),
        "local_start_index": local_start,
        "local_end_index": local_end,
        "current_end": current_end,
        "is_recompute": is_recompute,
        "compressed": compress,
        "sink_base_abs_start_frame": sink_base_value,
        "topc_base_abs_start_frame": topc_base_value,
    }
