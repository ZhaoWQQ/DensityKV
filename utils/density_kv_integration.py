"""Attach and access the independent density-limited KV bank."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kv_cache import DensityKVBankConfig, DensityKVBankStats, DensityLimitedKVBank


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def density_kv_config_enabled(cfg: Any) -> bool:
    return bool(_cfg_get(cfg, "enabled", False))


def density_kv_enabled(attn_module: nn.Module) -> bool:
    return bool(getattr(attn_module, "density_kv_enabled", False)) and isinstance(
        getattr(attn_module, "density_kv_bank", None),
        DensityLimitedKVBank,
    )


def density_kv_eviction_rope_indices(
    attn_module: nn.Module,
    *,
    num_evicted_frames: int,
    sink_tokens: int,
    frame_seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the canonical temporal RoPE positions used by the paper policy."""
    del attn_module, sink_tokens, frame_seq_length
    return torch.zeros(num_evicted_frames, dtype=torch.long, device=device)


def attach_density_kv_banks(
    model: nn.Module,
    cfg: Any,
    *,
    batch_size: int,
) -> int:
    if not density_kv_config_enabled(cfg):
        return 0
    if batch_size <= 0:
        raise ValueError("density KV batch_size must be positive")
    capacity = int(_cfg_get(cfg, "capacity", 8192))
    local_window_frames = int(_cfg_get(cfg, "local_window_frames", -1))
    logical_precommit = bool(_cfg_get(cfg, "logical_precommit", True))
    lineage_cfg = _cfg_get(cfg, "lineage_trace", None)
    lineage_enabled = bool(_cfg_get(lineage_cfg, "enabled", False))
    lineage_detailed = bool(_cfg_get(lineage_cfg, "detailed", False))
    lineage_layers = tuple(
        int(value) for value in _cfg_get(lineage_cfg, "layers", (0,))
    )
    lineage_heads = tuple(
        int(value) for value in _cfg_get(lineage_cfg, "heads", (0,))
    )
    lineage_frame_seq_length = int(
        _cfg_get(lineage_cfg, "frame_seq_length", 1560)
    )
    if local_window_frames < -1:
        raise ValueError("density KV local_window_frames must be -1 or non-negative")
    if lineage_frame_seq_length <= 0:
        raise ValueError("density KV lineage frame_seq_length must be positive")
    bank_config = DensityKVBankConfig(
        max_entries=capacity,
        density_scale=float(_cfg_get(cfg, "density_scale", 8.0)),
        riesz_power=float(_cfg_get(cfg, "riesz_power", 2.0)),
        riesz_eps=float(_cfg_get(cfg, "riesz_eps", 1.0)),
        density_growth_limit=float(_cfg_get(cfg, "density_growth_limit", 2.0)),
        density_baseline_floor=float(
            _cfg_get(cfg, "density_baseline_floor", 1.0e-6)
        ),
        work_chunk_size=int(_cfg_get(cfg, "work_chunk_size", 128)),
        compute_dtype=str(_cfg_get(cfg, "compute_dtype", "bfloat16")),
        fast_impl=str(_cfg_get(cfg, "fast_impl", "auto")),
    )
    parameter = next(model.parameters())
    attached = 0
    for layer_index, block in enumerate(getattr(model, "blocks", [])):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        num_heads = int(getattr(attn, "num_heads"))
        head_dim = int(getattr(attn, "head_dim"))
        bank = DensityLimitedKVBank(
            groups=batch_size * num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            config=bank_config,
            device=parameter.device,
        )
        invalid_heads = [head for head in lineage_heads if head < 0 or head >= num_heads]
        if invalid_heads:
            raise ValueError(f"density KV lineage heads out of range: {invalid_heads}")
        bank.lineage_trace_enabled = bool(
            lineage_enabled and layer_index in lineage_layers
        )
        bank.lineage_trace_detailed = bool(
            lineage_detailed and layer_index in lineage_layers
        )
        bank.lineage_trace_heads = lineage_heads
        bank.lineage_trace_events = []
        bank.lineage_layer_index = layer_index
        bank.lineage_frame_seq_length = lineage_frame_seq_length
        attn.density_kv_bank = bank
        attn.density_kv_enabled = True
        attn.density_kv_batch_size = int(batch_size)
        attn.density_kv_active_count = 0
        attn.density_kv_last_stats = None
        attn.density_kv_last_processed_count = 0
        attn.density_kv_local_window_frames = local_window_frames
        attn.density_kv_last_memory_tokens = 0
        attn.density_kv_last_local_tokens = 0
        attn.density_kv_last_attention_tokens = 0
        attn.density_kv_source_tokens_seen = 0
        attn.density_kv_pending_update = None
        attn.density_kv_logical_precommit = logical_precommit
        attn.density_kv_layer_index = layer_index
        attn.density_kv_trace_query_frame = -1
        attached += 1
    model.density_kv_enabled = attached > 0
    model.density_kv_capacity = capacity
    model.density_kv_attached_layers = attached
    return attached


@torch.no_grad()
def reset_density_kv_banks(model: nn.Module) -> int:
    """Clear all attached banks before starting a new video sample."""
    reset_count = 0
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "self_attn", None)
        if attn is None or not density_kv_enabled(attn):
            continue
        attn.density_kv_bank.clear()
        attn.density_kv_active_count = 0
        attn.density_kv_last_stats = None
        attn.density_kv_last_processed_count = 0
        attn.density_kv_last_memory_tokens = 0
        attn.density_kv_last_local_tokens = 0
        attn.density_kv_last_attention_tokens = 0
        attn.density_kv_source_tokens_seen = 0
        attn.density_kv_pending_update = None
        bank = getattr(attn, "density_kv_bank")
        if hasattr(bank, "lineage_trace_events"):
            bank.lineage_trace_events.clear()
        attn.density_kv_trace_query_frame = -1
        reset_count += 1
    return reset_count


@torch.no_grad()
def stage_density_kv_bank_update(
    attn_module: nn.Module,
    keys_bthd: torch.Tensor | None,
    values_bthd: torch.Tensor | None,
) -> bool:
    """Hold evicted clean history until the block's clean-cache commit."""
    if (
        keys_bthd is None
        or values_bthd is None
        or not density_kv_enabled(attn_module)
    ):
        return False
    if getattr(attn_module, "density_kv_pending_update", None) is not None:
        raise RuntimeError("density KV update was staged twice before clean commit")
    attn_module.density_kv_pending_update = (
        keys_bthd.detach(),
        values_bthd.detach(),
    )
    return True


def density_kv_logical_eviction_slice(
    attn_module: nn.Module,
    *,
    current_end: int,
    local_end_index: int,
    sink_tokens: int,
    frame_seq_length: int,
) -> tuple[int, int] | None:
    """Map newly expired logical-local history to the physical cache slice."""
    local_frames = int(getattr(attn_module, "density_kv_local_window_frames", -1))
    if local_frames < 0:
        return None
    if frame_seq_length <= 0:
        raise ValueError("density KV frame_seq_length must be positive")
    if sink_tokens < 0 or sink_tokens % frame_seq_length:
        raise ValueError("density KV sink must be frame aligned")
    if local_end_index < sink_tokens:
        raise ValueError("density KV physical cache ends before its sink")

    source_tokens_seen = int(
        getattr(attn_module, "density_kv_source_tokens_seen", 0)
    )
    source_start = sink_tokens + source_tokens_seen
    source_end = max(
        sink_tokens,
        int(current_end) - local_frames * frame_seq_length,
    )
    if source_end <= source_start:
        return None
    if source_start % frame_seq_length or source_end % frame_seq_length:
        raise ValueError("density KV logical eviction must be frame aligned")

    physical_non_sink_tokens = int(local_end_index) - sink_tokens
    physical_source_start = int(current_end) - physical_non_sink_tokens
    if source_start < physical_source_start or source_end > int(current_end):
        raise RuntimeError(
            "density KV logical eviction fell outside the clean physical cache: "
            f"source=[{source_start},{source_end}), "
            f"physical=[{physical_source_start},{current_end})"
        )
    local_start = sink_tokens + source_start - physical_source_start
    local_end = sink_tokens + source_end - physical_source_start
    if local_start < sink_tokens or local_end > local_end_index:
        raise RuntimeError("density KV logical eviction mapped outside cache storage")
    return int(local_start), int(local_end)


@torch.no_grad()
def commit_staged_density_kv_bank_update(
    attn_module: nn.Module,
    query_bthd: torch.Tensor | None = None,
) -> DensityKVBankStats | None:
    """Publish a staged update only after every noisy timestep has finished."""
    del query_bthd
    pending = getattr(attn_module, "density_kv_pending_update", None)
    if pending is None:
        return None
    attn_module.density_kv_pending_update = None
    return update_density_kv_bank(attn_module, pending[0], pending[1])


def get_density_kv_memory(
    attn_module: nn.Module,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not density_kv_enabled(attn_module):
        return None
    expected_batch = int(getattr(attn_module, "density_kv_batch_size", batch_size))
    if batch_size != expected_batch:
        raise ValueError(
            f"density KV bank was initialized for batch {expected_batch}, got {batch_size}"
        )
    count = int(getattr(attn_module, "density_kv_active_count", 0))
    if count <= 0:
        return None
    bank = getattr(attn_module, "density_kv_bank")
    num_heads = int(getattr(attn_module, "num_heads"))
    key_dim = int(getattr(attn_module, "head_dim"))
    stored_keys = bank.keys[:, :count]
    stored_values = bank.values[:, :count]
    keys = (
        stored_keys
        .view(batch_size, num_heads, count, key_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    values = (
        stored_values
        .view(batch_size, num_heads, count, key_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    counts = bank.counts.view(batch_size, num_heads).long()
    return keys, values, counts


@torch.no_grad()
def update_density_kv_bank(
    attn_module: nn.Module,
    keys_bthd: torch.Tensor | None,
    values_bthd: torch.Tensor | None,
) -> DensityKVBankStats | None:
    if (
        keys_bthd is None
        or values_bthd is None
        or not density_kv_enabled(attn_module)
    ):
        return None
    if keys_bthd.ndim != 4 or values_bthd.shape != keys_bthd.shape:
        raise ValueError("density KV updates require matching [B,T,H,D] K/V tensors")
    batch_size, num_tokens, num_heads, head_dim = keys_bthd.shape
    if batch_size != int(getattr(attn_module, "density_kv_batch_size", batch_size)):
        raise ValueError("density KV update batch size changed during inference")
    source_tokens_seen = int(
        getattr(attn_module, "density_kv_source_tokens_seen", 0)
    )
    bank = getattr(attn_module, "density_kv_bank")
    keys_ngd = keys_bthd.permute(1, 0, 2, 3).reshape(
        num_tokens,
        batch_size * num_heads,
        head_dim,
    )
    values_ngd = values_bthd.permute(1, 0, 2, 3).reshape_as(keys_ngd)
    stats = bank.update(keys_ngd, values_ngd)
    _record_density_kv_lineage(
        attn_module,
        stats,
        source_base=source_tokens_seen,
        query_frame=int(getattr(attn_module, "density_kv_trace_query_frame", -1)),
        batch_size=batch_size,
        num_heads=num_heads,
    )
    attn_module.density_kv_active_count = int(bank.counts.max().item())
    attn_module.density_kv_last_stats = stats
    attn_module.density_kv_last_processed_count = num_tokens
    attn_module.density_kv_source_tokens_seen = source_tokens_seen + num_tokens
    return stats


@torch.no_grad()
def _record_density_kv_lineage(
    attn_module: nn.Module,
    stats: DensityKVBankStats,
    *,
    source_base: int,
    query_frame: int,
    batch_size: int,
    num_heads: int,
) -> None:
    bank = getattr(attn_module, "density_kv_bank")
    trace_enabled = bool(getattr(bank, "lineage_trace_enabled", False))
    provenance_enabled = bool(
        getattr(attn_module, "density_kv_provenance_enabled", False)
    )
    if not trace_enabled and not provenance_enabled:
        return

    def matrix(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(1) if tensor.ndim == 1 else tensor

    candidate_index = matrix(stats.candidate_index).long()
    target_slot = matrix(stats.target_slot).long()
    accepted = matrix(stats.accepted_entry_mask).bool()
    candidate_density = matrix(stats.candidate_density)
    replacement_density = matrix(stats.replacement_density)
    evicted_density = matrix(stats.evicted_density)
    energy_delta = matrix(stats.energy_delta)
    if not (
        candidate_index.shape
        == target_slot.shape
        == accepted.shape
        == candidate_density.shape
        == replacement_density.shape
        == evicted_density.shape
        == energy_delta.shape
    ):
        raise RuntimeError("density KV lineage stats have inconsistent entry shapes")

    candidate_source = candidate_index + int(source_base)
    safe_slot = target_slot.clamp_min(0)
    target_source_before = torch.gather(bank.source_index, 1, safe_slot)
    target_insert_before = torch.gather(bank.insert_query_frame, 1, safe_slot)
    has_target = target_slot >= 0
    target_source_before = torch.where(
        has_target,
        target_source_before,
        torch.full_like(target_source_before, -1),
    )
    target_insert_before = torch.where(
        has_target,
        target_insert_before,
        torch.full_like(target_insert_before, -1),
    )
    committed = accepted & has_target
    detailed = bool(getattr(bank, "lineage_trace_detailed", False))

    decision_group = getattr(stats, "trace_decision_group", None)
    entry_reason = getattr(stats, "trace_entry_reason", None)
    candidate_energy_share = getattr(
        stats, "trace_candidate_energy_share", None
    )
    victim_energy_share = getattr(stats, "trace_victim_energy_share", None)
    anchor_slot = getattr(stats, "trace_anchor_slot", None)
    reference_density = getattr(stats, "trace_reference_density", None)
    group_reason = getattr(stats, "trace_group_reason", None)
    group_accepted = getattr(stats, "trace_group_accepted", None)
    group_candidate_count = getattr(
        stats, "trace_group_candidate_count", None
    )
    group_added_energy = getattr(stats, "trace_group_added_energy", None)
    group_removed_energy = getattr(stats, "trace_group_removed_energy", None)

    trace_heads = (
        tuple(getattr(bank, "lineage_trace_heads", (0,)))
        if trace_enabled
        else ()
    )
    for batch_index in range(batch_size):
        for head_index in trace_heads:
            group_index = batch_index * num_heads + head_index
            event = {
                "query_frame": int(query_frame),
                "source_base": int(source_base),
                "layer": int(getattr(bank, "lineage_layer_index", -1)),
                "batch": int(batch_index),
                "head": int(head_index),
                "mode": "density_growth",
                "candidate_index": candidate_index[group_index]
                .to(device="cpu", dtype=torch.int32),
                "accepted": accepted[group_index]
                .to(device="cpu", dtype=torch.uint8),
                "target_slot": target_slot[group_index]
                .to(device="cpu", dtype=torch.int32),
                "candidate_density": candidate_density[group_index]
                .to(device="cpu", dtype=torch.float16),
                "replacement_density": replacement_density[group_index]
                .to(device="cpu", dtype=torch.float16),
                "evicted_density": evicted_density[group_index]
                .to(device="cpu", dtype=torch.float16),
                "energy_delta": energy_delta[group_index]
                .to(device="cpu", dtype=torch.float16),
            }
            if not detailed:
                event["candidate_source"] = candidate_source[group_index].to(
                    device="cpu", dtype=torch.int64
                )
                event["target_source_before"] = target_source_before[group_index].to(
                    device="cpu", dtype=torch.int64
                )
                event["target_insert_query_frame"] = target_insert_before[
                    group_index
                ].to(device="cpu", dtype=torch.int32)
            if detailed:
                if decision_group is not None:
                    event["decision_group"] = decision_group[group_index].to(
                        device="cpu", dtype=torch.int16
                    )
                if entry_reason is not None:
                    event["entry_reason"] = entry_reason[group_index].to(
                        device="cpu", dtype=torch.int8
                    )
                if candidate_energy_share is not None:
                    event["candidate_energy_share"] = candidate_energy_share[
                        group_index
                    ].to(device="cpu", dtype=torch.float16)
                if victim_energy_share is not None:
                    event["victim_energy_share"] = victim_energy_share[
                        group_index
                    ].to(device="cpu", dtype=torch.float16)
                if anchor_slot is not None:
                    event["anchor_slot"] = anchor_slot[group_index].to(
                        device="cpu", dtype=torch.int32
                    )
                if reference_density is not None:
                    event["reference_density"] = reference_density[group_index].to(
                        device="cpu", dtype=torch.float16
                    )
                if group_reason is not None:
                    event["group_reason"] = group_reason[group_index].to(
                        device="cpu", dtype=torch.int8
                    )
                if group_accepted is not None:
                    event["group_accepted"] = group_accepted[group_index].to(
                        device="cpu", dtype=torch.uint8
                    )
                if group_candidate_count is not None:
                    event["group_candidate_count"] = group_candidate_count[
                        group_index
                    ].to(device="cpu", dtype=torch.int16)
                if group_added_energy is not None:
                    event["group_added_energy"] = group_added_energy[group_index].to(
                        device="cpu", dtype=torch.float32
                    )
                if group_removed_energy is not None:
                    event["group_removed_energy"] = group_removed_energy[
                        group_index
                    ].to(device="cpu", dtype=torch.float32)
            bank.lineage_trace_events.append(event)

    if bool(committed.any().item()):
        # A legacy chunked update mutates the bank sequentially, so several
        # accepted candidates can target the same slot in one call. Advanced
        # indexing with duplicate destinations has undefined write order on
        # CUDA. Match the K/V mutation by selecting the final accepted write
        # for every (group, slot) pair explicitly.
        entry_count = int(candidate_source.shape[1])
        entry_order = torch.arange(
            entry_count,
            device=bank.device,
            dtype=torch.long,
        ).view(1, -1).expand_as(candidate_source)
        committed_order = torch.where(
            committed,
            entry_order,
            torch.full_like(entry_order, -1),
        )
        last_order = torch.full_like(bank.source_index, -1)
        last_order.scatter_reduce_(
            1,
            safe_slot,
            committed_order,
            reduce="amax",
            include_self=True,
        )
        written = last_order >= 0
        final_source = torch.gather(
            candidate_source,
            1,
            last_order.clamp_min(0),
        )
        bank.source_index[written] = final_source[written]
        bank.insert_query_frame[written] = int(query_frame)


@torch.no_grad()
def export_density_kv_lineage(model: nn.Module) -> dict[str, Any] | None:
    """Collect compact lineage events and final traced bank state for torch.save."""
    layers = {}
    for layer_index, block in enumerate(getattr(model, "blocks", [])):
        attn = getattr(block, "self_attn", None)
        bank = getattr(attn, "density_kv_bank", None) if attn is not None else None
        if bank is None or not bool(getattr(bank, "lineage_trace_enabled", False)):
            continue
        count = int(bank.counts.max().item())
        final_heads = {}
        batch_size = int(getattr(attn, "density_kv_batch_size", 1))
        num_heads = int(getattr(attn, "num_heads"))
        for batch_index in range(batch_size):
            for head_index in tuple(getattr(bank, "lineage_trace_heads", (0,))):
                group_index = batch_index * num_heads + head_index
                head_count = int(bank.counts[group_index].item())
                final_heads[f"b{batch_index}_h{head_index}"] = {
                    "count": head_count,
                    "source_index": bank.source_index[
                        group_index, :head_count
                    ].cpu(),
                    "insert_query_frame": bank.insert_query_frame[
                        group_index, :head_count
                    ].cpu(),
                    "density": bank.density[group_index, :head_count].cpu(),
                    "density_baseline": bank.density_baseline[
                        group_index, :head_count
                    ].cpu(),
                }
        layers[str(layer_index)] = {
            "events": list(getattr(bank, "lineage_trace_events", [])),
            "final_count": count,
            "final_heads": final_heads,
        }
    if not layers:
        return None
    detailed = any(
        "decision_group" in event
        for layer in layers.values()
        for event in layer["events"][:1]
    )
    return {
        "format": "density_kv_lineage_v2" if detailed else "density_kv_lineage_v1",
        "frame_seq_length": 1560,
        "spatial_height": 30,
        "spatial_width": 52,
        "source_frame_offset": 1,
        "decision_reason_codes": {
            "9": "insertion_density_growth",
        },
        "layers": layers,
    }
