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
    """Return the temporal RoPE positions assigned when K enters the bank."""
    mode = str(getattr(attn_module, "density_kv_temporal_rope_mode", "zero"))
    if mode == "zero":
        return torch.zeros(num_evicted_frames, dtype=torch.long, device=device)
    if mode == "eviction_position":
        if frame_seq_length <= 0 or sink_tokens % frame_seq_length != 0:
            raise ValueError("density KV eviction positions require whole sink frames")
        start_frame = sink_tokens // frame_seq_length
        return torch.arange(
            start_frame,
            start_frame + num_evicted_frames,
            dtype=torch.long,
            device=device,
        )
    raise ValueError(f"unknown density KV temporal RoPE mode: {mode}")


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
    max_candidates = int(_cfg_get(cfg, "max_candidates_per_update", 512))
    max_admissions = int(_cfg_get(cfg, "max_admissions_per_update", 128))
    local_window_frames = int(_cfg_get(cfg, "local_window_frames", -1))
    warmup_noise_history_frames = int(
        _cfg_get(cfg, "warmup_noise_history_frames", 0)
    )
    warmup_noise_seed = int(_cfg_get(cfg, "warmup_noise_seed", 0))
    warmup_duplicate_current = bool(
        _cfg_get(cfg, "warmup_duplicate_current", False)
    )
    source_token_limit = int(_cfg_get(cfg, "source_token_limit", -1))
    verify_fixed_prefix = bool(_cfg_get(cfg, "verify_fixed_prefix", False))
    logical_precommit = bool(
        _cfg_get(cfg, "logical_precommit", not verify_fixed_prefix)
    )
    temporal_rope_mode = str(_cfg_get(cfg, "temporal_rope_mode", "zero"))
    distance_metric = str(_cfg_get(cfg, "distance_metric", "squared_l2"))
    query_response_rank = int(_cfg_get(cfg, "query_response_rank", 64))
    query_response_rescale_trace = bool(
        _cfg_get(cfg, "query_response_rescale_trace", True)
    )
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
    bootstrap_report_commits = int(
        _cfg_get(cfg, "bootstrap_report_commits", 2)
    )
    bootstrap_report_quantiles = bool(
        _cfg_get(cfg, "bootstrap_report_quantiles", False)
    )
    if local_window_frames < -1:
        raise ValueError("density KV local_window_frames must be -1 or non-negative")
    if warmup_noise_history_frames < 0:
        raise ValueError("density KV warmup_noise_history_frames must be non-negative")
    if warmup_noise_history_frames > 0 and warmup_duplicate_current:
        raise ValueError(
            "density KV Gaussian warmup and duplicate-current warmup are mutually exclusive"
        )
    if source_token_limit == 0 or source_token_limit < -1:
        raise ValueError("density KV source_token_limit must be -1 or positive")
    if verify_fixed_prefix and logical_precommit:
        raise ValueError(
            "fixed-prefix verification requires density KV logical_precommit=false"
        )
    if lineage_frame_seq_length <= 0:
        raise ValueError("density KV lineage frame_seq_length must be positive")
    if bootstrap_report_commits < 0:
        raise ValueError("density KV bootstrap_report_commits must be non-negative")
    if temporal_rope_mode not in {"zero", "eviction_position"}:
        raise ValueError(
            "density KV temporal_rope_mode must be zero or eviction_position"
        )
    if distance_metric not in {"squared_l2", "query_response"}:
        raise ValueError(
            "density KV distance_metric must be squared_l2 or query_response"
        )
    bank_config = DensityKVBankConfig(
        max_entries=capacity,
        density_scale=float(_cfg_get(cfg, "density_scale", 8.0)),
        riesz_power=float(_cfg_get(cfg, "riesz_power", 2.0)),
        riesz_eps=float(_cfg_get(cfg, "riesz_eps", 1.0)),
        replacement_ratio=float(_cfg_get(cfg, "replacement_ratio", 1.0)),
        evict_densest_when_full=bool(
            _cfg_get(cfg, "evict_densest_when_full", True)
        ),
        max_candidates_per_update=max_candidates,
        max_admissions_per_update=max_admissions,
        process_all_candidates=bool(
            _cfg_get(cfg, "process_all_candidates", False)
        ),
        update_chunk_size=int(_cfg_get(cfg, "update_chunk_size", 512)),
        full_update_mode=str(_cfg_get(cfg, "full_update_mode", "frozen_snapshot")),
        legacy_chunk_grouping=str(
            _cfg_get(cfg, "legacy_chunk_grouping", "contiguous")
        ),
        legacy_drop_tail_when_full=bool(
            _cfg_get(cfg, "legacy_drop_tail_when_full", False)
        ),
        legacy_repeat_tail_when_full=int(
            _cfg_get(cfg, "legacy_repeat_tail_when_full", 0)
        ),
        legacy_warmup_chunk_size=int(
            _cfg_get(cfg, "legacy_warmup_chunk_size", -1)
        ),
        legacy_density_growth_gate=bool(
            _cfg_get(cfg, "legacy_density_growth_gate", False)
        ),
        legacy_density_gated_bootstrap=bool(
            _cfg_get(cfg, "legacy_density_gated_bootstrap", False)
        ),
        legacy_density_gated_bootstrap_v2=bool(
            _cfg_get(cfg, "legacy_density_gated_bootstrap_v2", False)
        ),
        legacy_density_gated_bootstrap_v4=bool(
            _cfg_get(cfg, "legacy_density_gated_bootstrap_v4", False)
        ),
        legacy_bootstrap_density_limit=float(
            _cfg_get(cfg, "legacy_bootstrap_density_limit", 1.0)
        ),
        legacy_bootstrap_v4_ratio_limit=float(
            _cfg_get(cfg, "legacy_bootstrap_v4_ratio_limit", 1.0)
        ),
        legacy_bootstrap_v4_seed=int(
            _cfg_get(cfg, "legacy_bootstrap_v4_seed", 0)
        ),
        legacy_bootstrap_v4_warmup_tokens=int(
            _cfg_get(cfg, "legacy_bootstrap_v4_warmup_tokens", 0)
        ),
        legacy_bootstrap_v2_gate=str(
            _cfg_get(
                cfg,
                "legacy_bootstrap_v2_gate",
                "full_union_candidate_ratio",
            )
        ),
        legacy_bootstrap_absolute_density_limit=float(
            _cfg_get(cfg, "legacy_bootstrap_absolute_density_limit", -1.0)
        ),
        legacy_bootstrap_tail_cleanup_size=int(
            _cfg_get(cfg, "legacy_bootstrap_tail_cleanup_size", 0)
        ),
        legacy_normalized_group_count=int(
            _cfg_get(cfg, "legacy_normalized_group_count", -1)
        ),
        legacy_cleanup_divisor=int(
            _cfg_get(cfg, "legacy_cleanup_divisor", -1)
        ),
        legacy_cleanup_alignment=int(
            _cfg_get(cfg, "legacy_cleanup_alignment", 8)
        ),
        union_work_chunk_size=int(_cfg_get(cfg, "union_work_chunk_size", -1)),
        append_density_growth_limit=float(
            _cfg_get(cfg, "append_density_growth_limit", 2.0)
        ),
        append_density_baseline_floor=float(
            _cfg_get(cfg, "append_density_baseline_floor", 1.0e-6)
        ),
        append_growth_chunk_size=int(
            _cfg_get(cfg, "append_growth_chunk_size", 4096)
        ),
        append_group_count_reduce=str(
            _cfg_get(cfg, "append_group_count_reduce", "min")
        ),
        append_max_entries=int(_cfg_get(cfg, "append_max_entries", -1)),
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
        if distance_metric == "query_response" and not (
            0 < query_response_rank <= head_dim
        ):
            raise ValueError(
                "density KV query_response_rank must be in [1, head_dim]"
            )
        geometry_dim = (
            query_response_rank
            if distance_metric == "query_response"
            else head_dim
        )
        payload_dim = 2 * head_dim if distance_metric == "query_response" else head_dim
        bank = DensityLimitedKVBank(
            groups=batch_size * num_heads,
            key_dim=geometry_dim,
            value_dim=payload_dim,
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
        bank.bootstrap_report_commits = bootstrap_report_commits
        bank.bootstrap_report_quantiles = bootstrap_report_quantiles
        attn.density_kv_bank = bank
        attn.density_kv_enabled = True
        attn.density_kv_batch_size = int(batch_size)
        attn.density_kv_active_count = 0
        attn.density_kv_last_stats = None
        attn.density_kv_last_processed_count = 0
        attn.density_kv_local_window_frames = local_window_frames
        attn.density_kv_warmup_noise_history_frames = warmup_noise_history_frames
        attn.density_kv_warmup_noise_seed = warmup_noise_seed
        attn.density_kv_warmup_noise_k = None
        attn.density_kv_warmup_noise_v = None
        attn.density_kv_last_warmup_noise_tokens = 0
        attn.density_kv_warmup_duplicate_current = warmup_duplicate_current
        attn.density_kv_last_warmup_duplicate_tokens = 0
        attn.density_kv_last_memory_tokens = 0
        attn.density_kv_last_local_tokens = 0
        attn.density_kv_last_attention_tokens = 0
        attn.density_kv_variable_head_lengths = (
            bank_config.append_group_count_reduce == "masked_max"
        )
        attn.density_kv_source_token_limit = source_token_limit
        attn.density_kv_source_tokens_seen = 0
        attn.density_kv_frozen = False
        attn.density_kv_pending_update = None
        attn.density_kv_logical_precommit = logical_precommit
        attn.density_kv_verify_fixed_prefix = verify_fixed_prefix
        attn.density_kv_temporal_rope_mode = temporal_rope_mode
        attn.density_kv_distance_metric = distance_metric
        attn.density_kv_query_response_rank = query_response_rank
        attn.density_kv_query_response_rescale_trace = query_response_rescale_trace
        attn.density_kv_query_transform = None
        attn.density_kv_query_retained_variance = None
        attn.density_kv_layer_index = layer_index
        attn.density_kv_verified_token_counts = set()
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
        attn.density_kv_warmup_noise_k = None
        attn.density_kv_warmup_noise_v = None
        attn.density_kv_last_warmup_noise_tokens = 0
        attn.density_kv_last_warmup_duplicate_tokens = 0
        attn.density_kv_source_tokens_seen = 0
        attn.density_kv_frozen = False
        attn.density_kv_pending_update = None
        attn.density_kv_query_transform = None
        attn.density_kv_query_retained_variance = None
        bank = getattr(attn, "density_kv_bank")
        if hasattr(bank, "lineage_trace_events"):
            bank.lineage_trace_events.clear()
        attn.density_kv_trace_query_frame = -1
        reset_count += 1
    return reset_count


def apply_density_kv_warmup_duplicate(
    attn_module: nn.Module,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    current_start_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Double first-block active KV without changing exact attention semantics."""
    if keys.shape != values.shape:
        raise ValueError("warmup duplication requires matching K/V shapes")
    enabled = bool(
        getattr(attn_module, "density_kv_warmup_duplicate_current", False)
    )
    if not enabled or int(current_start_frame) != 0:
        attn_module.density_kv_last_warmup_duplicate_tokens = 0
        return keys, values
    original_tokens = int(keys.shape[1])
    attn_module.density_kv_last_warmup_duplicate_tokens = original_tokens
    return torch.cat((keys, keys), dim=1), torch.cat((values, values), dim=1)


@torch.no_grad()
def get_density_kv_warmup_noise(
    attn_module: nn.Module,
    *,
    batch_size: int,
    current_start_frame: int,
    frame_seq_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return fixed virtual-history Gaussian KV for the short first prefix.

    The virtual frames are attention-only: they are never committed to the
    local cache or Density-KV bank.  Real generated history replaces them in
    FIFO order as ``current_start_frame`` advances.
    """
    configured_frames = int(
        getattr(attn_module, "density_kv_warmup_noise_history_frames", 0)
    )
    remaining_frames = max(configured_frames - int(current_start_frame), 0)
    if remaining_frames <= 0:
        attn_module.density_kv_warmup_noise_k = None
        attn_module.density_kv_warmup_noise_v = None
        attn_module.density_kv_last_warmup_noise_tokens = 0
        return None
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")

    total_tokens = configured_frames * int(frame_seq_length)
    expected_shape = (
        int(batch_size),
        total_tokens,
        int(getattr(attn_module, "num_heads")),
        int(getattr(attn_module, "head_dim")),
    )
    noise_k = getattr(attn_module, "density_kv_warmup_noise_k", None)
    noise_v = getattr(attn_module, "density_kv_warmup_noise_v", None)
    if (
        noise_k is None
        or noise_v is None
        or tuple(noise_k.shape) != expected_shape
        or noise_k.device != device
        or noise_k.dtype != dtype
    ):
        generator = torch.Generator(device=device)
        generator.manual_seed(
            int(getattr(attn_module, "density_kv_warmup_noise_seed", 0))
            + 1009 * int(getattr(attn_module, "density_kv_layer_index", 0))
        )
        noise_k = torch.randn(
            expected_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noise_v = torch.randn(
            expected_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        attn_module.density_kv_warmup_noise_k = noise_k
        attn_module.density_kv_warmup_noise_v = noise_v

    remaining_tokens = remaining_frames * int(frame_seq_length)
    # Keep the newest virtual frames so old Gaussian history leaves first.
    noise_k = noise_k[:, total_tokens - remaining_tokens :]
    noise_v = noise_v[:, total_tokens - remaining_tokens :]
    attn_module.density_kv_last_warmup_noise_tokens = remaining_tokens
    return noise_k, noise_v


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
    if local_frames < 0 or bool(getattr(attn_module, "density_kv_frozen", False)):
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
    initialize_density_kv_query_metric(attn_module, query_bthd)
    pending = getattr(attn_module, "density_kv_pending_update", None)
    if pending is None:
        return None
    attn_module.density_kv_pending_update = None
    return update_density_kv_bank(attn_module, pending[0], pending[1])


@torch.no_grad()
def initialize_density_kv_query_metric(
    attn_module: nn.Module,
    query_bthd: torch.Tensor | None,
) -> bool:
    """Freeze the first clean-query covariance as the bank's K geometry."""
    metric = str(getattr(attn_module, "density_kv_distance_metric", "squared_l2"))
    if metric == "squared_l2":
        return False
    if metric != "query_response":
        raise ValueError(f"unknown density KV distance metric: {metric}")
    if getattr(attn_module, "density_kv_query_transform", None) is not None:
        return False
    if query_bthd is None:
        raise RuntimeError(
            "query-response density KV requires clean RoPE query at first commit"
        )
    if query_bthd.ndim != 4:
        raise ValueError("density KV query metric requires [B,T,H,D] queries")
    batch_size, num_tokens, num_heads, head_dim = query_bthd.shape
    expected_batch = int(getattr(attn_module, "density_kv_batch_size", batch_size))
    if batch_size != expected_batch:
        raise ValueError("density KV query metric batch size changed")
    if num_tokens <= 0:
        raise ValueError("density KV query metric requires at least one query")
    rank = int(getattr(attn_module, "density_kv_query_response_rank", head_dim))
    if not 0 < rank <= head_dim:
        raise ValueError("density KV query response rank is invalid")

    groups = batch_size * num_heads
    queries = (
        query_bthd.detach()
        .permute(0, 2, 1, 3)
        .reshape(groups, num_tokens, head_dim)
        .float()
    )
    covariance = torch.bmm(queries.transpose(1, 2), queries)
    covariance.mul_(1.0 / float(num_tokens * head_dim))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min_(0.0)
    kept_values = eigenvalues[:, -rank:]
    kept_vectors = eigenvectors[:, :, -rank:]
    total_variance = eigenvalues.sum(dim=1)
    kept_variance = kept_values.sum(dim=1)
    retained = kept_variance / total_variance.clamp_min(1.0e-12)
    if bool(
        getattr(attn_module, "density_kv_query_response_rescale_trace", True)
    ):
        kept_values = kept_values * (
            total_variance / kept_variance.clamp_min(1.0e-12)
        ).unsqueeze(1)
    transform = kept_vectors * kept_values.sqrt().unsqueeze(1)
    bank = getattr(attn_module, "density_kv_bank")
    attn_module.density_kv_query_transform = transform.to(
        device=bank.device,
        dtype=bank.keys.dtype,
    ).contiguous()
    attn_module.density_kv_query_retained_variance = retained
    print(
        f"[density-kv] layer={int(getattr(attn_module, 'density_kv_layer_index', -1))} "
        f"query-response rank={rank}/{head_dim}; retained variance "
        f"mean={retained.mean().item():.4f}, min={retained.min().item():.4f}, "
        f"max={retained.max().item():.4f}"
    )
    return True


def _project_density_kv_keys(
    attn_module: nn.Module,
    keys_bthd: torch.Tensor,
) -> torch.Tensor:
    """Map K to its low-rank vector of clean-query attention responses."""
    transform = getattr(attn_module, "density_kv_query_transform", None)
    if transform is None:
        raise RuntimeError("density KV query-response transform is not initialized")
    batch_size, num_tokens, num_heads, head_dim = keys_bthd.shape
    groups = batch_size * num_heads
    keys_gtd = keys_bthd.permute(0, 2, 1, 3).reshape(
        groups, num_tokens, head_dim
    )
    response_gtr = torch.bmm(keys_gtd.to(dtype=transform.dtype), transform)
    return response_gtr.permute(1, 0, 2).contiguous()


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
    metric = str(getattr(attn_module, "density_kv_distance_metric", "squared_l2"))
    if metric == "query_response":
        payload = bank.values[:, :count]
        if payload.shape[-1] != 2 * key_dim:
            raise RuntimeError("query-response density KV payload has invalid width")
        stored_keys, stored_values = payload.split(key_dim, dim=-1)
    else:
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


def pack_density_kv_attention_by_head(
    keys_bthd: torch.Tensor,
    values_bthd: torch.Tensor,
    memory_counts_bh: torch.Tensor,
    *,
    fixed_prefix_tokens: int,
    memory_tokens: int,
    local_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack variable per-head memory prefixes for FlashAttention varlen."""
    if keys_bthd.ndim != 4 or values_bthd.shape != keys_bthd.shape:
        raise ValueError("attention packing requires matching [B,T,H,D] K/V")
    batch_size, total_tokens, num_heads, head_dim = keys_bthd.shape
    if memory_counts_bh.shape != (batch_size, num_heads):
        raise ValueError("memory counts must have shape [B,H]")
    if fixed_prefix_tokens + memory_tokens + local_tokens != total_tokens:
        raise ValueError("density attention token partitions do not match K/V length")
    source_keys = keys_bthd.permute(0, 2, 1, 3).contiguous()
    source_values = values_bthd.permute(0, 2, 1, 3).contiguous()
    positions = torch.arange(total_tokens, device=keys_bthd.device).view(1, 1, -1)
    memory_end = fixed_prefix_tokens + memory_counts_bh.unsqueeze(-1)
    valid_lengths = memory_end + local_tokens
    local_source_start = fixed_prefix_tokens + memory_tokens
    gather_index = torch.where(
        positions < memory_end,
        positions,
        local_source_start + positions - memory_end,
    ).clamp_(0, max(total_tokens - 1, 0))
    gather_index = gather_index.expand(batch_size, num_heads, total_tokens)
    gather_index = gather_index.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    packed_keys = torch.gather(source_keys, 2, gather_index)
    packed_values = torch.gather(source_values, 2, gather_index)
    return (
        packed_keys.reshape(batch_size * num_heads, total_tokens, 1, head_dim),
        packed_values.reshape(batch_size * num_heads, total_tokens, 1, head_dim),
        valid_lengths.reshape(-1).to(torch.int32),
    )


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
    source_token_limit = int(
        getattr(attn_module, "density_kv_source_token_limit", -1)
    )
    source_tokens_seen = int(
        getattr(attn_module, "density_kv_source_tokens_seen", 0)
    )
    if source_token_limit >= 0:
        remaining = max(source_token_limit - source_tokens_seen, 0)
        if remaining == 0:
            attn_module.density_kv_last_stats = None
            attn_module.density_kv_last_processed_count = 0
            attn_module.density_kv_frozen = True
            return None
        if num_tokens > remaining:
            keys_bthd = keys_bthd[:, :remaining]
            values_bthd = values_bthd[:, :remaining]
            num_tokens = remaining
    bank = getattr(attn_module, "density_kv_bank")
    active_count_before = int(bank.counts.max().item())
    metric = str(getattr(attn_module, "density_kv_distance_metric", "squared_l2"))
    if metric == "query_response":
        keys_ngd = _project_density_kv_keys(attn_module, keys_bthd)
        original_keys_ngd = keys_bthd.permute(1, 0, 2, 3).reshape(
            num_tokens,
            batch_size * num_heads,
            head_dim,
        )
        original_values_ngd = values_bthd.permute(1, 0, 2, 3).reshape_as(
            original_keys_ngd
        )
        payload_ngd = torch.cat((original_keys_ngd, original_values_ngd), dim=-1)
        stats = bank.update(keys_ngd, payload_ngd)
    else:
        keys_ngd = keys_bthd.permute(1, 0, 2, 3).reshape(
            num_tokens,
            batch_size * num_heads,
            head_dim,
        )
        values_ngd = values_bthd.permute(1, 0, 2, 3).reshape_as(keys_ngd)
        stats = bank.update(keys_ngd, values_ngd)
    if (
        (
            bank.config.legacy_density_gated_bootstrap_v4
            or bank.config.legacy_density_gated_bootstrap_v2
            or (
                bank.config.legacy_density_gated_bootstrap
                and active_count_before < bank.config.max_entries
            )
        )
    ):
        report_count = int(getattr(bank, "_bootstrap_report_count", 0))
        report_limit = int(getattr(bank, "bootstrap_report_commits", 2))
        if report_count < report_limit:
            accepted_count = stats.accepted_count.float()
            admitted_count = getattr(
                stats, "bootstrap_admitted_count", stats.accepted_count
            ).float()
            tail_replaced_count = getattr(
                stats,
                "bootstrap_tail_replaced_count",
                torch.zeros_like(stats.accepted_count),
            ).float()
            reference_mean = getattr(
                stats,
                "bootstrap_reference_mean",
                torch.full_like(stats.accepted_count, torch.nan, dtype=torch.float32),
            ).float()
            quantile_text = ""
            if bool(getattr(bank, "bootstrap_report_quantiles", False)):
                density_quantiles = torch.quantile(
                    stats.candidate_density.float().flatten(),
                    torch.tensor(
                        [0.1, 0.5, 0.9],
                        device=stats.candidate_density.device,
                    ),
                )
                ratio_quantiles = torch.quantile(
                    stats.trace_saturation_ratio.float().flatten(),
                    torch.tensor(
                        [0.1, 0.5, 0.9],
                        device=stats.trace_saturation_ratio.device,
                    ),
                )
                quantile_text = (
                    f"density_p10_p50_p90="
                    f"{density_quantiles[0].item():.6g},"
                    f"{density_quantiles[1].item():.6g},"
                    f"{density_quantiles[2].item():.6g} "
                    f"ratio_p10_p50_p90="
                    f"{ratio_quantiles[0].item():.6g},"
                    f"{ratio_quantiles[1].item():.6g},"
                    f"{ratio_quantiles[2].item():.6g} "
                )
            print(
                "[density-kv-bootstrap] "
                f"layer={int(getattr(bank, 'lineage_layer_index', -1))} "
                f"commit={report_count} before={active_count_before} "
                f"candidates={num_tokens} "
                f"gate={getattr(stats, 'bootstrap_gate_mode', 'frozen_baseline')} "
                f"ratio_limit={bank.config.legacy_bootstrap_density_limit:.6g} "
                f"v4_ratio_limit={bank.config.legacy_bootstrap_v4_ratio_limit:.6g} "
                f"absolute_limit={bank.config.legacy_bootstrap_absolute_density_limit:.6g} "
                f"reference_mean={reference_mean.mean().item():.6g} "
                f"{quantile_text}"
                f"warmup_admitted_mean="
                f"{getattr(stats, 'bootstrap_warmup_admitted_count', torch.zeros_like(admitted_count)).float().mean().item():.1f} "
                f"admitted_mean={admitted_count.mean().item():.1f} "
                f"tail_replaced_mean={tail_replaced_count.mean().item():.1f} "
                f"accepted_min={int(accepted_count.min().item())} "
                f"accepted_mean={accepted_count.mean().item():.1f} "
                f"accepted_max={int(accepted_count.max().item())} "
                f"after={int(bank.counts.max().item())}",
                flush=True,
            )
            bank._bootstrap_report_count = report_count + 1
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
    attn_module.density_kv_frozen = (
        source_token_limit >= 0
        and attn_module.density_kv_source_tokens_seen >= source_token_limit
    )
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
                "mode": str(bank.config.full_update_mode),
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
            "0": "capacity_fill",
            "1": "legacy_joint_energy",
            "2": "frozen_individual_energy",
            "3": "no_target_capacity",
            "4": "append_density_ratio",
        },
        "layers": layers,
    }
