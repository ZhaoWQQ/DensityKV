"""Non-invasive temporal provenance diagnostics for causal self-attention."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def temporal_attention_trace_config_enabled(cfg: Any) -> bool:
    return bool(_cfg_get(cfg, "enabled", False))


@torch.no_grad()
def attach_temporal_attention_trace(
    model: nn.Module,
    cfg: Any,
    *,
    batch_size: int,
    frame_seq_length: int,
    max_frames: int,
) -> int:
    """Attach fixed-size diagnostic accumulators to selected self-attention layers."""
    if not temporal_attention_trace_config_enabled(cfg):
        return 0
    if batch_size <= 0 or frame_seq_length <= 0 or max_frames <= 0:
        raise ValueError("temporal attention trace dimensions must be positive")

    blocks = list(getattr(model, "blocks", []))
    configured_layers = _cfg_get(cfg, "layers", "all")
    if configured_layers == "all" or configured_layers is None:
        layers = set(range(len(blocks)))
    else:
        layers = {int(value) for value in configured_layers}
    denoise_calls = tuple(
        int(value) for value in _cfg_get(cfg, "denoise_calls", (3,))
    )
    if not denoise_calls or min(denoise_calls) < 0:
        raise ValueError("temporal attention trace denoise_calls must be non-negative")
    query_chunk_size = int(_cfg_get(cfg, "query_chunk_size", 32))
    if query_chunk_size <= 0:
        raise ValueError("temporal attention trace query_chunk_size must be positive")

    parameter = next(model.parameters())
    attached = 0
    for layer_index, block in enumerate(blocks):
        attn = getattr(block, "self_attn", None)
        if attn is None or layer_index not in layers:
            continue
        num_heads = int(getattr(attn, "num_heads"))
        configured_heads = _cfg_get(cfg, "heads", "all")
        if configured_heads == "all" or configured_heads is None:
            heads = tuple(range(num_heads))
        else:
            heads = tuple(int(value) for value in configured_heads)
        if any(head < 0 or head >= num_heads for head in heads):
            raise ValueError(
                f"temporal attention trace heads out of range at layer {layer_index}: {heads}"
            )

        attn.temporal_attention_trace_enabled = True
        attn.temporal_attention_trace_layer = int(layer_index)
        attn.temporal_attention_trace_heads = heads
        attn.temporal_attention_trace_denoise_calls = denoise_calls
        attn.temporal_attention_trace_query_chunk_size = query_chunk_size
        attn.temporal_attention_trace_frame_seq_length = int(frame_seq_length)
        attn.temporal_attention_trace_max_frames = int(max_frames)
        attn.temporal_attention_trace_batch_size = int(batch_size)
        attn.temporal_attention_trace_last_query_frame = -1
        attn.temporal_attention_trace_denoise_call = -1
        attn.temporal_attention_trace_mass = torch.zeros(
            batch_size,
            num_heads,
            max_frames,
            max_frames,
            device=parameter.device,
            dtype=torch.float32,
        )
        attn.temporal_attention_trace_source_token_count = torch.zeros(
            batch_size,
            num_heads,
            max_frames,
            max_frames,
            device=parameter.device,
            dtype=torch.int32,
        )
        attn.temporal_attention_trace_count = torch.zeros(
            batch_size,
            max_frames,
            device=parameter.device,
            dtype=torch.int16,
        )
        # Keep only DensityKV slot provenance without enabling verbose lineage logs.
        attn.density_kv_provenance_enabled = True
        attached += 1

    model.temporal_attention_trace_enabled = attached > 0
    model.temporal_attention_trace_attached_layers = attached
    return attached


@torch.no_grad()
def reset_temporal_attention_trace(model: nn.Module) -> int:
    reset = 0
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "self_attn", None)
        if attn is None or not bool(
            getattr(attn, "temporal_attention_trace_enabled", False)
        ):
            continue
        attn.temporal_attention_trace_mass.zero_()
        attn.temporal_attention_trace_source_token_count.zero_()
        attn.temporal_attention_trace_count.zero_()
        attn.temporal_attention_trace_last_query_frame = -1
        attn.temporal_attention_trace_denoise_call = -1
        reset += 1
    return reset


def _expand_frame_ids(
    frame_ids: torch.Tensor,
    *,
    frame_seq_length: int,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor:
    token_ids = frame_ids.repeat_interleave(frame_seq_length)
    return token_ids.view(1, 1, -1).expand(batch_size, num_heads, -1)


@torch.no_grad()
def build_temporal_key_source_frames(
    attn_module: nn.Module,
    *,
    batch_size: int,
    key_tokens: int,
    sink_tokens: int,
    warmup_tokens: int,
    memory_tokens: int,
    local_tokens: int,
    current_start_frame: int,
    num_query_frames: int,
    frame_seq_length: int,
    memory_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return source latent-frame ids for K in the exact concatenation order."""
    num_heads = int(getattr(attn_module, "num_heads"))
    device = next(attn_module.parameters()).device
    for label, tokens in (
        ("sink", sink_tokens),
        ("warmup", warmup_tokens),
        ("local", local_tokens),
    ):
        if tokens < 0 or tokens % frame_seq_length:
            raise ValueError(f"{label} token count is not frame aligned: {tokens}")
    if memory_tokens < 0:
        raise ValueError(f"memory token count must be non-negative: {memory_tokens}")

    parts = []
    sink_frames = sink_tokens // frame_seq_length
    if sink_frames:
        parts.append(
            _expand_frame_ids(
                torch.arange(sink_frames, device=device, dtype=torch.long),
                frame_seq_length=frame_seq_length,
                batch_size=batch_size,
                num_heads=num_heads,
            )
        )
    if warmup_tokens:
        parts.append(
            torch.full(
                (batch_size, num_heads, warmup_tokens),
                -1,
                device=device,
                dtype=torch.long,
            )
        )

    if memory_tokens:
        bank = getattr(attn_module, "density_kv_bank", None)
        if bank is not None and bool(getattr(attn_module, "density_kv_enabled", False)):
            # A DensityKV bank is a token coreset, so its active length is not
            # generally divisible by the number of spatial tokens per frame.
            # Each slot carries the exact source token index recorded at commit.
            source = bank.source_index[:, :memory_tokens].view(
                batch_size, num_heads, memory_tokens
            )
            source_frames = torch.where(
                source >= 0,
                torch.div(source, frame_seq_length, rounding_mode="floor")
                + sink_frames,
                torch.full_like(source, -1),
            )
            counts = bank.counts.view(batch_size, num_heads)
            positions = torch.arange(memory_tokens, device=device).view(1, 1, -1)
            source_frames = source_frames.masked_fill(
                positions >= counts.unsqueeze(-1), -1
            )
            parts.append(source_frames)
        elif memory_indices is not None:
            if memory_indices.ndim != 2 or memory_indices.shape[0] != batch_size:
                raise ValueError("memory_indices must have shape [B, selected_frames]")
            selected = memory_indices.to(device=device, dtype=torch.long) + sink_frames
            expanded = selected.repeat_interleave(frame_seq_length, dim=1)
            if expanded.shape[1] != memory_tokens:
                raise ValueError("retrieved memory provenance does not match K length")
            parts.append(expanded.unsqueeze(1).expand(-1, num_heads, -1))
        else:
            parts.append(
                torch.full(
                    (batch_size, num_heads, memory_tokens),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
            )

    local_frames = local_tokens // frame_seq_length
    if local_frames:
        current_end_frame = int(current_start_frame) + int(num_query_frames)
        local_start_frame = current_end_frame - local_frames
        parts.append(
            _expand_frame_ids(
                torch.arange(
                    local_start_frame,
                    current_end_frame,
                    device=device,
                    dtype=torch.long,
                ),
                frame_seq_length=frame_seq_length,
                batch_size=batch_size,
                num_heads=num_heads,
            )
        )

    source_frames = (
        torch.cat(parts, dim=-1)
        if parts
        else torch.empty(
            batch_size, num_heads, 0, device=device, dtype=torch.long
        )
    )
    if source_frames.shape[-1] != key_tokens:
        duplicate_tokens = int(
            getattr(attn_module, "density_kv_last_warmup_duplicate_tokens", 0)
        )
        if duplicate_tokens and key_tokens == 2 * source_frames.shape[-1]:
            source_frames = torch.cat((source_frames, source_frames), dim=-1)
        else:
            raise ValueError(
                "temporal provenance does not match attention K length: "
                f"sources={source_frames.shape[-1]}, keys={key_tokens}"
            )
    return source_frames


@torch.no_grad()
def compute_temporal_attention_histogram(
    query_bthd: torch.Tensor,
    keys_bthd: torch.Tensor,
    source_frames_bhk: torch.Tensor,
    *,
    frame_seq_length: int,
    max_frames: int,
    heads: tuple[int, ...],
    query_chunk_size: int,
) -> torch.Tensor:
    """Aggregate full-softmax mass into [B,H,query-frame,source-frame]."""
    if query_bthd.ndim != 4 or keys_bthd.ndim != 4:
        raise ValueError("temporal attention trace expects [B,T,H,D] Q/K")
    batch_size, query_tokens, num_heads, head_dim = query_bthd.shape
    if keys_bthd.shape[0] != batch_size or keys_bthd.shape[2:] != (
        num_heads,
        head_dim,
    ):
        raise ValueError("temporal attention trace Q/K shapes do not align")
    if source_frames_bhk.shape != (
        batch_size,
        num_heads,
        keys_bthd.shape[1],
    ):
        raise ValueError("temporal attention provenance shape does not match K")
    if query_tokens % frame_seq_length:
        raise ValueError("query token count is not frame aligned")

    head_index = torch.tensor(heads, device=query_bthd.device, dtype=torch.long)
    query = query_bthd.index_select(2, head_index).permute(0, 2, 1, 3).contiguous()
    keys = keys_bthd.index_select(2, head_index).permute(0, 2, 1, 3).contiguous()
    source = source_frames_bhk.index_select(1, head_index)
    valid_source = (source >= 0) & (source < max_frames)
    safe_source = source.clamp(0, max_frames - 1)
    key_transpose = keys.transpose(-1, -2)
    num_query_frames = query_tokens // frame_seq_length
    result = torch.zeros(
        batch_size,
        len(heads),
        num_query_frames,
        max_frames,
        device=query_bthd.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(head_dim)

    for query_frame in range(num_query_frames):
        frame_start = query_frame * frame_seq_length
        frame_end = frame_start + frame_seq_length
        frame_mass = result[:, :, query_frame]
        for start in range(frame_start, frame_end, query_chunk_size):
            end = min(start + query_chunk_size, frame_end)
            logits = torch.matmul(query[:, :, start:end], key_transpose)
            weights = torch.softmax(logits.float() * scale, dim=-1)
            chunk_histogram = torch.zeros(
                batch_size,
                len(heads),
                end - start,
                max_frames,
                device=query_bthd.device,
                dtype=torch.float32,
            )
            scatter_index = safe_source.unsqueeze(2).expand(
                -1, -1, end - start, -1
            )
            chunk_histogram.scatter_add_(
                -1,
                scatter_index,
                weights * valid_source.unsqueeze(2),
            )
            frame_mass.add_(chunk_histogram.sum(dim=2))
        frame_mass.div_(frame_seq_length)
    return result


@torch.no_grad()
def compute_temporal_source_token_count(
    source_frames_bhk: torch.Tensor,
    *,
    max_frames: int,
    heads: tuple[int, ...],
) -> torch.Tensor:
    """Count visible K tokens by provenance frame for each selected head."""
    if source_frames_bhk.ndim != 3:
        raise ValueError("temporal source counts expect [B,H,K] provenance")
    head_index = torch.tensor(
        heads,
        device=source_frames_bhk.device,
        dtype=torch.long,
    )
    source = source_frames_bhk.index_select(1, head_index)
    valid = (source >= 0) & (source < max_frames)
    safe_source = source.clamp(0, max_frames - 1)
    counts = torch.zeros(
        source.shape[0],
        len(heads),
        max_frames,
        device=source.device,
        dtype=torch.int32,
    )
    counts.scatter_add_(-1, safe_source, valid.to(torch.int32))
    return counts


@torch.no_grad()
def record_temporal_attention(
    attn_module: nn.Module,
    query_bthd: torch.Tensor,
    keys_bthd: torch.Tensor,
    source_frames_bhk: torch.Tensor,
    *,
    current_start_frame: int,
    clean_cache_commit: bool,
) -> bool:
    if not bool(getattr(attn_module, "temporal_attention_trace_enabled", False)):
        return False

    last_query = int(
        getattr(attn_module, "temporal_attention_trace_last_query_frame", -1)
    )
    if int(current_start_frame) != last_query:
        attn_module.temporal_attention_trace_last_query_frame = int(
            current_start_frame
        )
        attn_module.temporal_attention_trace_denoise_call = -1
    if clean_cache_commit:
        return False
    call_index = int(
        getattr(attn_module, "temporal_attention_trace_denoise_call", -1)
    ) + 1
    attn_module.temporal_attention_trace_denoise_call = call_index
    if call_index not in tuple(attn_module.temporal_attention_trace_denoise_calls):
        return False

    histogram = compute_temporal_attention_histogram(
        query_bthd.detach(),
        keys_bthd.detach(),
        source_frames_bhk,
        frame_seq_length=int(attn_module.temporal_attention_trace_frame_seq_length),
        max_frames=int(attn_module.temporal_attention_trace_max_frames),
        heads=tuple(attn_module.temporal_attention_trace_heads),
        query_chunk_size=int(attn_module.temporal_attention_trace_query_chunk_size),
    )
    source_token_count = compute_temporal_source_token_count(
        source_frames_bhk,
        max_frames=int(attn_module.temporal_attention_trace_max_frames),
        heads=tuple(attn_module.temporal_attention_trace_heads),
    )
    max_frames = int(attn_module.temporal_attention_trace_max_frames)
    num_query_frames = histogram.shape[2]
    end_frame = min(int(current_start_frame) + num_query_frames, max_frames)
    used_frames = end_frame - int(current_start_frame)
    if used_frames <= 0:
        return False
    heads = tuple(attn_module.temporal_attention_trace_heads)
    for selected_index, head_index in enumerate(heads):
        attn_module.temporal_attention_trace_mass[
            :, head_index, current_start_frame:end_frame
        ].add_(histogram[:, selected_index, :used_frames])
        attn_module.temporal_attention_trace_source_token_count[
            :, head_index, current_start_frame:end_frame
        ].add_(source_token_count[:, selected_index].unsqueeze(1))
    attn_module.temporal_attention_trace_count[
        :, current_start_frame:end_frame
    ].add_(1)
    return True


@torch.no_grad()
def export_temporal_attention_trace(model: nn.Module) -> dict[str, Any] | None:
    layers = {}
    frame_seq_length = None
    for layer_index, block in enumerate(getattr(model, "blocks", [])):
        attn = getattr(block, "self_attn", None)
        if attn is None or not bool(
            getattr(attn, "temporal_attention_trace_enabled", False)
        ):
            continue
        count = attn.temporal_attention_trace_count
        averaged = attn.temporal_attention_trace_mass / count.clamp_min(1).to(
            torch.float32
        ).unsqueeze(1).unsqueeze(-1)
        averaged_source_token_count = (
            attn.temporal_attention_trace_source_token_count.to(torch.float32)
            / count.clamp_min(1)
            .to(torch.float32)
            .unsqueeze(1)
            .unsqueeze(-1)
        )
        layers[str(layer_index)] = {
            "heads": tuple(int(value) for value in attn.temporal_attention_trace_heads),
            "mass": averaged.cpu().to(torch.float16),
            "source_token_count": averaged_source_token_count.cpu().to(
                torch.float16
            ),
            "count": count.cpu(),
        }
        frame_seq_length = int(attn.temporal_attention_trace_frame_seq_length)
    if not layers:
        return None
    first = next(iter(layers.values()))
    return {
        "format": "temporal_attention_trace_v2",
        "definition": (
            "Full-softmax attention mass from current query tokens to K tokens, "
            "grouped by query latent frame and K provenance latent frame, with "
            "the corresponding visible K-token count per provenance frame."
        ),
        "frame_seq_length": int(frame_seq_length),
        "max_frames": int(first["mass"].shape[-1]),
        "layers": layers,
    }
