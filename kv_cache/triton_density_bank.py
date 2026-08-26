"""Triton kernels for the independent density-limited KV bank."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional on CPU hosts.
    triton = None
    tl = None


def _ensure_python_include_for_triton() -> None:
    """Expose the active conda Python.h when sysconfig points at /usr."""
    include_dir = Path(sys.prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    if not (include_dir / "Python.h").exists():
        return
    current = os.environ.get("CPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if str(include_dir) not in entries:
        os.environ["CPATH"] = os.pathsep.join([str(include_dir), *entries])


@dataclass
class TritonDensitySelection:
    candidate_density: torch.Tensor
    candidate_index: torch.Tensor
    target_slot: torch.Tensor
    target_density: torch.Tensor


if triton is not None:

    @triton.jit
    def _density_partial_kernel(
        distance_ptr,
        partial_ptr,
        num_rows: tl.constexpr,
        num_columns: tl.constexpr,
        num_tiles: tl.constexpr,
        block_columns: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
        exclude_diagonal: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * block_columns + tl.arange(0, block_columns)
        valid = offsets < num_columns
        distance = tl.load(
            distance_ptr + row * num_columns + offsets,
            mask=valid,
            other=float("inf"),
        ).to(tl.float32)
        contribution = tl.exp(
            -riesz_power * tl.log(riesz_eps + distance * inverse_scale_sq)
        )
        if exclude_diagonal:
            row_in_group = row % num_rows
            valid = valid & (offsets != row_in_group)
        partial = tl.sum(tl.where(valid, contribution, 0.0), axis=0)
        tl.store(partial_ptr + row * num_tiles + tile, partial)

    @triton.jit
    def _candidate_density_kernel(
        distance_ptr,
        count_ptr,
        output_ptr,
        num_candidates: tl.constexpr,
        max_entries: tl.constexpr,
        block_entries: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = row // num_candidates
        offsets = tl.arange(0, block_entries)
        count = tl.load(count_ptr + group).to(tl.int32)
        valid = offsets < count
        distance = tl.load(
            distance_ptr + row * max_entries + offsets,
            mask=offsets < max_entries,
            other=float("inf"),
        ).to(tl.float32)
        normalized = distance * inverse_scale_sq
        contribution = tl.exp(-riesz_power * tl.log(riesz_eps + normalized))
        density = tl.sum(tl.where(valid, contribution, 0.0), axis=0)
        tl.store(output_ptr + row, density)


    @triton.jit
    def _select_sparse_dense_kernel(
        candidate_density_ptr,
        stored_density_ptr,
        count_ptr,
        candidate_index_ptr,
        candidate_value_ptr,
        target_slot_ptr,
        target_density_ptr,
        num_candidates: tl.constexpr,
        max_entries: tl.constexpr,
        block_candidates: tl.constexpr,
        block_entries: tl.constexpr,
    ):
        group = tl.program_id(0)
        candidate_offsets = tl.arange(0, block_candidates)
        candidate_mask = candidate_offsets < num_candidates
        candidate_values = tl.load(
            candidate_density_ptr + group * num_candidates + candidate_offsets,
            mask=candidate_mask,
            other=float("inf"),
        ).to(tl.float32)
        candidate_value = tl.min(candidate_values, axis=0)
        candidate_choices = tl.where(
            candidate_mask & (candidate_values == candidate_value),
            candidate_offsets,
            num_candidates,
        )
        candidate_index = tl.min(candidate_choices, axis=0)

        count = tl.load(count_ptr + group).to(tl.int32)
        entry_offsets = tl.arange(0, block_entries)
        entry_mask = entry_offsets < count
        stored_values = tl.load(
            stored_density_ptr + group * max_entries + entry_offsets,
            mask=entry_offsets < max_entries,
            other=-float("inf"),
        ).to(tl.float32)
        stored_values = tl.where(entry_mask, stored_values, -float("inf"))
        dense_value = tl.max(stored_values, axis=0)
        dense_choices = tl.where(
            entry_mask & (stored_values == dense_value),
            entry_offsets,
            max_entries,
        )
        dense_index = tl.min(dense_choices, axis=0)
        full = count >= max_entries
        target_slot = tl.where(full, dense_index, count)
        target_density = tl.where(full, dense_value, 0.0)

        tl.store(candidate_index_ptr + group, candidate_index)
        tl.store(candidate_value_ptr + group, candidate_value)
        tl.store(target_slot_ptr + group, target_slot)
        tl.store(target_density_ptr + group, target_density)


    @triton.jit
    def _mutate_density_kv_kernel(
        keys_ptr,
        values_ptr,
        density_ptr,
        count_ptr,
        selected_keys_ptr,
        selected_values_ptr,
        new_distance_ptr,
        old_distance_ptr,
        target_slot_ptr,
        accepted_ptr,
        was_full_ptr,
        replacement_density_ptr,
        max_entries: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        block_width: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
    ):
        group = tl.program_id(0)
        offsets = tl.arange(0, block_width)
        count = tl.load(count_ptr + group).to(tl.int32)
        target = tl.load(target_slot_ptr + group).to(tl.int32)
        accepted = tl.load(accepted_ptr + group).to(tl.int1)
        was_full = tl.load(was_full_ptr + group).to(tl.int1)

        entry_mask = offsets < max_entries
        active = offsets < count
        not_target = offsets != target
        density_offsets = group * max_entries + offsets
        density = tl.load(
            density_ptr + density_offsets,
            mask=entry_mask,
            other=0.0,
        ).to(tl.float32)
        new_distance = tl.load(
            new_distance_ptr + density_offsets,
            mask=entry_mask,
            other=float("inf"),
        ).to(tl.float32)
        old_distance = tl.load(
            old_distance_ptr + density_offsets,
            mask=entry_mask,
            other=float("inf"),
        ).to(tl.float32)
        new_contribution = tl.exp(
            -riesz_power * tl.log(riesz_eps + new_distance * inverse_scale_sq)
        )
        old_contribution = tl.exp(
            -riesz_power * tl.log(riesz_eps + old_distance * inverse_scale_sq)
        )
        pair_mask = active & not_target
        new_contribution = tl.where(pair_mask, new_contribution, 0.0)
        old_contribution = tl.where(pair_mask & was_full, old_contribution, 0.0)
        updated_density = density + new_contribution - old_contribution
        replacement_density = tl.load(replacement_density_ptr + group).to(tl.float32)
        updated_density = tl.where(offsets == target, replacement_density, updated_density)
        updated_density = tl.where(active | (offsets == target), updated_density, density)
        tl.store(
            density_ptr + density_offsets,
            updated_density,
            mask=entry_mask & accepted,
        )

        key_offsets = offsets
        selected_key = tl.load(
            selected_keys_ptr + group * key_dim + key_offsets,
            mask=key_offsets < key_dim,
            other=0.0,
        )
        target_key_offsets = (group * max_entries + target) * key_dim + key_offsets
        tl.store(
            keys_ptr + target_key_offsets,
            selected_key,
            mask=(key_offsets < key_dim) & accepted,
        )

        value_offsets = offsets
        selected_value = tl.load(
            selected_values_ptr + group * value_dim + value_offsets,
            mask=value_offsets < value_dim,
            other=0.0,
        )
        target_value_offsets = (group * max_entries + target) * value_dim + value_offsets
        tl.store(
            values_ptr + target_value_offsets,
            selected_value,
            mask=(value_offsets < value_dim) & accepted,
        )

        tl.store(
            count_ptr + group,
            count + 1,
            mask=accepted & (~was_full),
        )


    @triton.jit
    def _mutate_density_kv_tiled_kernel(
        keys_ptr,
        values_ptr,
        density_ptr,
        count_ptr,
        selected_keys_ptr,
        selected_values_ptr,
        new_distance_ptr,
        old_distance_ptr,
        target_slot_ptr,
        accepted_ptr,
        was_full_ptr,
        replacement_density_ptr,
        max_entries: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        tile_width: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
    ):
        group = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * tile_width + tl.arange(0, tile_width)
        count = tl.load(count_ptr + group).to(tl.int32)
        target = tl.load(target_slot_ptr + group).to(tl.int32)
        accepted = tl.load(accepted_ptr + group).to(tl.int1)
        was_full = tl.load(was_full_ptr + group).to(tl.int1)

        entry_mask = offsets < max_entries
        active = offsets < count
        not_target = offsets != target
        density_offsets = group * max_entries + offsets
        density = tl.load(
            density_ptr + density_offsets,
            mask=entry_mask,
            other=0.0,
        ).to(tl.float32)
        new_distance = tl.load(
            new_distance_ptr + density_offsets,
            mask=entry_mask,
            other=float("inf"),
        ).to(tl.float32)
        old_distance = tl.load(
            old_distance_ptr + density_offsets,
            mask=entry_mask,
            other=float("inf"),
        ).to(tl.float32)
        new_contribution = tl.exp(
            -riesz_power * tl.log(riesz_eps + new_distance * inverse_scale_sq)
        )
        old_contribution = tl.exp(
            -riesz_power * tl.log(riesz_eps + old_distance * inverse_scale_sq)
        )
        pair_mask = active & not_target
        updated_density = density + tl.where(pair_mask, new_contribution, 0.0)
        updated_density -= tl.where(pair_mask & was_full, old_contribution, 0.0)
        replacement_density = tl.load(replacement_density_ptr + group).to(tl.float32)
        updated_density = tl.where(offsets == target, replacement_density, updated_density)
        updated_density = tl.where(active | (offsets == target), updated_density, density)
        tl.store(
            density_ptr + density_offsets,
            updated_density,
            mask=entry_mask & accepted,
        )

        selected_key = tl.load(
            selected_keys_ptr + group * key_dim + offsets,
            mask=offsets < key_dim,
            other=0.0,
        )
        target_key_offsets = (group * max_entries + target) * key_dim + offsets
        tl.store(
            keys_ptr + target_key_offsets,
            selected_key,
            mask=(offsets < key_dim) & accepted,
        )

        selected_value = tl.load(
            selected_values_ptr + group * value_dim + offsets,
            mask=offsets < value_dim,
            other=0.0,
        )
        target_value_offsets = (group * max_entries + target) * value_dim + offsets
        tl.store(
            values_ptr + target_value_offsets,
            selected_value,
            mask=(offsets < value_dim) & accepted,
        )

        if tile == 0:
            tl.store(
                count_ptr + group,
                count + 1,
                mask=accepted & (~was_full),
            )


def triton_density_sum(
    distances: torch.Tensor,
    *,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
    exclude_diagonal: bool = False,
) -> torch.Tensor | None:
    if triton is None or not distances.is_cuda or distances.ndim != 3:
        return None
    _ensure_python_include_for_triton()
    groups, num_rows, num_columns = distances.shape
    if num_columns == 0:
        return torch.zeros(
            groups,
            num_rows,
            device=distances.device,
            dtype=torch.float32,
        )
    if exclude_diagonal and num_rows != num_columns:
        return None
    block_columns = min(1024, triton.next_power_of_2(num_columns))
    num_tiles = triton.cdiv(num_columns, block_columns)
    partial = torch.empty(
        groups,
        num_rows,
        num_tiles,
        device=distances.device,
        dtype=torch.float32,
    )
    _density_partial_kernel[(groups * num_rows, num_tiles)](
        distances,
        partial,
        num_rows=num_rows,
        num_columns=num_columns,
        num_tiles=num_tiles,
        block_columns=block_columns,
        inverse_scale_sq=1.0 / (float(density_scale) ** 2),
        riesz_power=float(riesz_power),
        riesz_eps=float(riesz_eps),
        exclude_diagonal=bool(exclude_diagonal),
        num_warps=8 if block_columns >= 512 else 4,
    )
    return partial.sum(dim=-1)


def triton_select_density_candidate(
    distances: torch.Tensor,
    stored_density: torch.Tensor,
    counts: torch.Tensor,
    *,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
) -> TritonDensitySelection | None:
    if triton is None or not distances.is_cuda:
        return None
    _ensure_python_include_for_triton()
    if distances.ndim != 3 or stored_density.ndim != 2 or counts.ndim != 1:
        return None
    groups, num_candidates, max_entries = distances.shape
    if stored_density.shape != (groups, max_entries) or counts.shape != (groups,):
        return None
    if num_candidates <= 0 or num_candidates > 4096:
        return None

    if max_entries > 4096:
        scores = triton_density_sum(
            distances,
            density_scale=density_scale,
            riesz_power=riesz_power,
            riesz_eps=riesz_eps,
        )
        if scores is None:
            return None
        candidate_value, candidate_index = scores.min(dim=1)
        slots = torch.arange(max_entries, device=distances.device).unsqueeze(0)
        active_density = stored_density.masked_fill(
            slots >= counts.long().unsqueeze(1),
            -float("inf"),
        )
        target_density, dense_index = active_density.max(dim=1)
        full = counts.long() >= max_entries
        return TritonDensitySelection(
            candidate_density=candidate_value,
            candidate_index=candidate_index,
            target_slot=torch.where(full, dense_index, counts.long()),
            target_density=torch.where(
                full,
                target_density,
                torch.zeros_like(target_density),
            ),
        )

    scores = torch.empty(
        groups,
        num_candidates,
        device=distances.device,
        dtype=torch.float32,
    )
    block_entries = triton.next_power_of_2(max_entries)
    _candidate_density_kernel[(groups * num_candidates,)](
        distances,
        counts,
        scores,
        num_candidates=num_candidates,
        max_entries=max_entries,
        block_entries=block_entries,
        inverse_scale_sq=1.0 / (float(density_scale) ** 2),
        riesz_power=float(riesz_power),
        riesz_eps=float(riesz_eps),
        num_warps=8 if block_entries >= 512 else 4,
    )

    candidate_index = torch.empty(groups, device=distances.device, dtype=torch.int32)
    candidate_value = torch.empty(groups, device=distances.device, dtype=torch.float32)
    target_slot = torch.empty(groups, device=distances.device, dtype=torch.int32)
    target_density = torch.empty(groups, device=distances.device, dtype=torch.float32)
    block_candidates = triton.next_power_of_2(num_candidates)
    _select_sparse_dense_kernel[(groups,)](
        scores,
        stored_density,
        counts,
        candidate_index,
        candidate_value,
        target_slot,
        target_density,
        num_candidates=num_candidates,
        max_entries=max_entries,
        block_candidates=block_candidates,
        block_entries=block_entries,
        num_warps=8 if max(block_candidates, block_entries) >= 512 else 4,
    )
    return TritonDensitySelection(
        candidate_density=candidate_value,
        candidate_index=candidate_index.long(),
        target_slot=target_slot.long(),
        target_density=target_density,
    )


def triton_mutate_density_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    density: torch.Tensor,
    counts: torch.Tensor,
    selected_keys: torch.Tensor,
    selected_values: torch.Tensor,
    new_distances: torch.Tensor,
    old_distances: torch.Tensor,
    target_slots: torch.Tensor,
    accepted: torch.Tensor,
    was_full: torch.Tensor,
    replacement_density: torch.Tensor,
    *,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
) -> bool:
    if triton is None or not keys.is_cuda:
        return False
    groups, max_entries, key_dim = keys.shape
    value_dim = values.shape[-1]
    if max(max_entries, key_dim, value_dim) > 4096:
        tile_width = 1024
        grid = (groups, triton.cdiv(max(max_entries, key_dim, value_dim), tile_width))
        _mutate_density_kv_tiled_kernel[grid](
            keys,
            values,
            density,
            counts,
            selected_keys,
            selected_values,
            new_distances,
            old_distances,
            target_slots,
            accepted,
            was_full,
            replacement_density,
            max_entries=max_entries,
            key_dim=key_dim,
            value_dim=value_dim,
            tile_width=tile_width,
            inverse_scale_sq=1.0 / (float(density_scale) ** 2),
            riesz_power=float(riesz_power),
            riesz_eps=float(riesz_eps),
            num_warps=8,
        )
        return True
    block_width = triton.next_power_of_2(max(max_entries, key_dim, value_dim))
    _mutate_density_kv_kernel[(groups,)](
        keys,
        values,
        density,
        counts,
        selected_keys,
        selected_values,
        new_distances,
        old_distances,
        target_slots,
        accepted,
        was_full,
        replacement_density,
        max_entries=max_entries,
        key_dim=key_dim,
        value_dim=value_dim,
        block_width=block_width,
        inverse_scale_sq=1.0 / (float(density_scale) ** 2),
        riesz_power=float(riesz_power),
        riesz_eps=float(riesz_eps),
        num_warps=8 if block_width >= 512 else 4,
    )
    return True
