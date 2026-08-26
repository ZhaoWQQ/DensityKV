"""Random-order streaming admission for DensityKV Bootstrap 4.

The operator never materializes the candidate-by-candidate potential matrix.
Candidates are shuffled once, then judged in order against:

* the existing bank potential supplied by the caller;
* previously admitted candidates; and
* every candidate that has not been judged yet.

Previously rejected candidates are dynamically masked.  The Triton path keeps
the threshold comparison inside the streaming kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional on CPU hosts.
    triton = None
    tl = None


@dataclass
class Bootstrap4AdmissionResult:
    """Admission decisions in both source and randomized order."""

    accepted_mask: torch.Tensor
    ordered_accepted_mask: torch.Tensor
    decision_potential: torch.Tensor
    decision_score: torch.Tensor
    permutation: torch.Tensor
    accepted_count: torch.Tensor
    used_triton: bool


@dataclass
class Bootstrap4ReferenceTrace:
    """White-box tensors for every ordered admission decision.

    ``pair_potential`` is intentionally materialized only by this reference
    path.  It is meant for small numerical tests and strategy inspection, not
    production-size updates.
    """

    accepted_mask: torch.Tensor
    old_potential: torch.Tensor
    admitted_prefix_potential: torch.Tensor
    unjudged_suffix_potential: torch.Tensor
    decision_potential: torch.Tensor
    visible_count: torch.Tensor
    decision_score: torch.Tensor
    threshold: torch.Tensor
    accepted_prefix_count: torch.Tensor
    pair_potential: torch.Tensor
    accepted_count: torch.Tensor


def _ensure_python_include_for_triton() -> None:
    include_dir = (
        Path(sys.prefix)
        / "include"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    if not (include_dir / "Python.h").exists():
        return
    current = os.environ.get("CPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if str(include_dir) not in entries:
        os.environ["CPATH"] = os.pathsep.join([str(include_dir), *entries])


def _as_group_vector(
    value: float | int | torch.Tensor,
    *,
    groups: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(groups)
    if result.shape != (groups,):
        raise ValueError(f"{name} must be scalar or have shape [{groups}]")
    return result.contiguous()


def make_bootstrap4_permutation(
    num_candidates: int,
    *,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a reproducible shared candidate permutation.

    The permutation is generated on CPU so a recorded seed has stable meaning
    across CUDA devices, then copied to the candidate device.
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(
        num_candidates,
        generator=generator,
        device="cpu",
    ).to(device=device)


def _validate_permutation(
    permutation: torch.Tensor,
    *,
    num_candidates: int,
    device: torch.device,
) -> torch.Tensor:
    permutation = permutation.detach().to(device=device, dtype=torch.long).contiguous()
    if permutation.shape != (num_candidates,):
        raise ValueError(
            f"permutation must have shape [{num_candidates}], "
            f"got {tuple(permutation.shape)}"
        )
    expected = torch.arange(num_candidates, device=device)
    if not torch.equal(permutation.sort().values, expected):
        raise ValueError("permutation must contain every candidate index exactly once")
    return permutation


def _restore_source_order(
    ordered: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    restored = torch.empty_like(ordered)
    restored.scatter_(
        1,
        permutation.unsqueeze(0).expand(ordered.shape[0], -1),
        ordered,
    )
    return restored


@torch.no_grad()
def bootstrap4_admission_reference_trace(
    candidate_keys_gmd: torch.Tensor,
    old_potential_gm: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    old_count: int | torch.Tensor,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
    normalize_by_visible_count: bool = True,
) -> Bootstrap4ReferenceTrace:
    """Materialized, white-box reference for randomized-order tensors.

    Inputs are already permuted.  Every pair potential and every component of
    the admission score remains available for inspection.
    """
    if candidate_keys_gmd.ndim != 3:
        raise ValueError("candidate_keys_gmd must have shape [G,M,D]")
    groups, num_candidates, _ = candidate_keys_gmd.shape
    if old_potential_gm.shape != (groups, num_candidates):
        raise ValueError("old_potential_gm must have shape [G,M]")
    if density_scale <= 0 or riesz_power <= 0 or riesz_eps <= 0:
        raise ValueError("density parameters must be positive")

    device = candidate_keys_gmd.device
    threshold_g = _as_group_vector(
        threshold,
        groups=groups,
        device=device,
        dtype=torch.float32,
        name="threshold",
    )
    old_count_g = _as_group_vector(
        old_count,
        groups=groups,
        device=device,
        dtype=torch.int32,
        name="old_count",
    )
    if bool((threshold_g <= 0).any()):
        raise ValueError("threshold must be positive")
    if bool((old_count_g < 0).any()):
        raise ValueError("old_count must be non-negative")

    accepted = torch.zeros(
        groups,
        num_candidates,
        device=device,
        dtype=torch.bool,
    )
    admitted_prefix_potential = torch.zeros(
        groups,
        num_candidates,
        device=device,
        dtype=torch.float32,
    )
    unjudged_suffix_potential = torch.zeros_like(admitted_prefix_potential)
    potential = torch.empty_like(admitted_prefix_potential)
    score = torch.empty_like(admitted_prefix_potential)
    visible_count_trace = torch.empty(
        groups,
        num_candidates,
        device=device,
        dtype=torch.int32,
    )
    accepted_prefix_count_trace = torch.empty_like(visible_count_trace)
    accepted_count = torch.zeros(groups, device=device, dtype=torch.int32)
    inverse_scale_sq = 1.0 / float(density_scale**2)

    dot = torch.bmm(
        candidate_keys_gmd,
        candidate_keys_gmd.transpose(1, 2),
    )
    key_norm = candidate_keys_gmd.float().square().sum(dim=-1)
    distance = (
        key_norm.unsqueeze(2)
        + key_norm.unsqueeze(1)
        - 2.0 * dot.float()
    ).clamp_min_(0.0)
    pair_potential = torch.exp(
        -float(riesz_power)
        * torch.log(float(riesz_eps) + distance * inverse_scale_sq)
    )
    diagonal = torch.arange(num_candidates, device=device)
    pair_potential[:, diagonal, diagonal] = 0.0

    for candidate_index in range(num_candidates):
        if candidate_index:
            prefix = (
                pair_potential[:, candidate_index, :candidate_index]
                * accepted[:, :candidate_index]
            ).sum(dim=1)
        else:
            prefix = torch.zeros(groups, device=device, dtype=torch.float32)
        suffix = pair_potential[
            :, candidate_index, candidate_index + 1 :
        ].sum(dim=1)
        current_potential = (
            old_potential_gm[:, candidate_index].float()
            + prefix
            + suffix
        )
        accepted_prefix_count_trace[:, candidate_index] = accepted_count
        visible_count = (
            old_count_g
            + accepted_count
            + (num_candidates - candidate_index - 1)
        )
        if normalize_by_visible_count:
            current_score = current_potential / visible_count.clamp_min(1).float()
        else:
            current_score = current_potential
        current_accepted = current_score < threshold_g

        admitted_prefix_potential[:, candidate_index] = prefix
        unjudged_suffix_potential[:, candidate_index] = suffix
        potential[:, candidate_index] = current_potential
        score[:, candidate_index] = current_score
        visible_count_trace[:, candidate_index] = visible_count
        accepted[:, candidate_index] = current_accepted
        accepted_count.add_(current_accepted.to(torch.int32))

    return Bootstrap4ReferenceTrace(
        accepted_mask=accepted,
        old_potential=old_potential_gm.float().clone(),
        admitted_prefix_potential=admitted_prefix_potential,
        unjudged_suffix_potential=unjudged_suffix_potential,
        decision_potential=potential,
        visible_count=visible_count_trace,
        decision_score=score,
        threshold=threshold_g.unsqueeze(1).expand(-1, num_candidates).clone(),
        accepted_prefix_count=accepted_prefix_count_trace,
        pair_potential=pair_potential,
        accepted_count=accepted_count,
    )


@torch.no_grad()
def bootstrap4_admission_reference(
    candidate_keys_gmd: torch.Tensor,
    old_potential_gm: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    old_count: int | torch.Tensor,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
    normalize_by_visible_count: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility tuple backed by the white-box non-kernel reference."""
    trace = bootstrap4_admission_reference_trace(
        candidate_keys_gmd,
        old_potential_gm,
        threshold=threshold,
        old_count=old_count,
        density_scale=density_scale,
        riesz_power=riesz_power,
        riesz_eps=riesz_eps,
        normalize_by_visible_count=normalize_by_visible_count,
    )
    return (
        trace.accepted_mask,
        trace.decision_potential,
        trace.decision_score,
        trace.accepted_count,
    )


if triton is not None:

    @triton.jit
    def _bootstrap4_riesz_potential(
        distance,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
    ):
        base = riesz_eps + distance * inverse_scale_sq
        if riesz_power == 1.0:
            return 1.0 / base
        if riesz_power == 2.0:
            inverse = 1.0 / base
            return inverse * inverse
        return tl.exp(-riesz_power * tl.log(base))

    @triton.jit
    def _bootstrap4_gemm_stripe_decision_kernel(
        dot_ptr,
        key_norm_ptr,
        old_potential_ptr,
        threshold_ptr,
        old_count_ptr,
        accepted_mask_ptr,
        decision_potential_ptr,
        decision_score_ptr,
        accepted_count_ptr,
        stripe_start,
        stripe_rows,
        dot_group_rows,
        num_candidates,
        block_rows: tl.constexpr,
        block_columns: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
        normalize_by_visible_count: tl.constexpr,
    ):
        """Consume a GEMM stripe while preserving block-row decision order."""
        group = tl.program_id(0)
        row_lanes = tl.arange(0, block_rows)
        column_lanes = tl.arange(0, block_columns)
        accepted_prefix = tl.load(accepted_count_ptr + group).to(tl.int32)
        threshold = tl.load(threshold_ptr + group).to(tl.float32)
        old_count = tl.load(old_count_ptr + group).to(tl.int32)

        for dot_row_offset in tl.range(0, stripe_rows, block_rows):
            row_start = stripe_start + dot_row_offset
            row_indices = row_start + row_lanes
            valid_rows = (
                (row_lanes < stripe_rows - dot_row_offset)
                & (row_indices < num_candidates)
            )
            query_norm = tl.load(
                key_norm_ptr + group * num_candidates + row_indices,
                mask=valid_rows,
                other=0.0,
            ).to(tl.float32)
            row_potential = tl.load(
                old_potential_ptr + group * num_candidates + row_indices,
                mask=valid_rows,
                other=0.0,
            ).to(tl.float32)

            for column_start in tl.range(
                0,
                num_candidates,
                block_columns,
            ):
                column_indices = column_start + column_lanes
                valid_columns = column_indices < num_candidates
                dot_offsets = (
                    (
                        group * dot_group_rows
                        + dot_row_offset
                        + row_lanes[:, None]
                    )
                    * num_candidates
                    + column_indices[None, :]
                )
                dot = tl.load(
                    dot_ptr + dot_offsets,
                    mask=valid_rows[:, None] & valid_columns[None, :],
                    other=0.0,
                ).to(tl.float32)
                key_norm = tl.load(
                    key_norm_ptr + group * num_candidates + column_indices,
                    mask=valid_columns,
                    other=0.0,
                ).to(tl.float32)
                distance = tl.maximum(
                    query_norm[:, None]
                    + key_norm[None, :]
                    - 2.0 * dot,
                    0.0,
                )
                contribution = _bootstrap4_riesz_potential(
                    distance,
                    inverse_scale_sq,
                    riesz_power,
                    riesz_eps,
                )
                is_decided_prefix = column_indices < row_start
                prefix_accepted = tl.load(
                    accepted_mask_ptr
                    + group * num_candidates
                    + column_indices,
                    mask=valid_columns & is_decided_prefix,
                    other=0,
                ).to(tl.int1)
                visible_column = valid_columns & (
                    (~is_decided_prefix) | prefix_accepted
                )
                pair_visible = (
                    valid_rows[:, None]
                    & visible_column[None, :]
                    & (row_indices[:, None] != column_indices[None, :])
                )
                row_potential += tl.sum(
                    tl.where(pair_visible, contribution, 0.0),
                    axis=1,
                )

            local_columns = row_start + row_lanes
            local_valid = valid_rows
            local_dot_offsets = (
                (
                    group * dot_group_rows
                    + dot_row_offset
                    + row_lanes[:, None]
                )
                * num_candidates
                + local_columns[None, :]
            )
            local_dot = tl.load(
                dot_ptr + local_dot_offsets,
                mask=valid_rows[:, None] & local_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            local_key_norm = tl.load(
                key_norm_ptr + group * num_candidates + local_columns,
                mask=local_valid,
                other=0.0,
            ).to(tl.float32)
            local_distance = tl.maximum(
                query_norm[:, None]
                + local_key_norm[None, :]
                - 2.0 * local_dot,
                0.0,
            )
            local_contribution = _bootstrap4_riesz_potential(
                local_distance,
                inverse_scale_sq,
                riesz_power,
                riesz_eps,
            )
            local_accepted = tl.zeros((block_rows,), dtype=tl.int1)

            for local_row in tl.static_range(0, block_rows):
                current_index = row_start + local_row
                current_valid = local_row < stripe_rows - dot_row_offset
                row_selector = row_lanes == local_row
                current_potential = tl.sum(
                    tl.where(row_selector, row_potential, 0.0),
                    axis=0,
                )
                rejected_local_prefix = (
                    (row_lanes < local_row)
                    & local_valid
                    & (~local_accepted)
                )
                local_row_selector = row_lanes[:, None] == local_row
                rejected_correction = tl.sum(
                    tl.where(
                        local_row_selector
                        & rejected_local_prefix[None, :],
                        local_contribution,
                        0.0,
                    ),
                    axis=None,
                )
                current_potential -= rejected_correction
                visible_count = (
                    old_count
                    + accepted_prefix
                    + (num_candidates - current_index - 1)
                )
                if normalize_by_visible_count:
                    current_score = current_potential / tl.maximum(
                        visible_count.to(tl.float32),
                        1.0,
                    )
                else:
                    current_score = current_potential
                current_accepted = current_valid & (
                    current_score < threshold
                )

                tl.store(
                    accepted_mask_ptr
                    + group * num_candidates
                    + current_index,
                    current_accepted.to(tl.int8),
                    mask=current_valid,
                )
                tl.store(
                    decision_potential_ptr
                    + group * num_candidates
                    + current_index,
                    current_potential,
                    mask=current_valid,
                )
                tl.store(
                    decision_score_ptr
                    + group * num_candidates
                    + current_index,
                    current_score,
                    mask=current_valid,
                )
                local_accepted = tl.where(
                    row_selector,
                    current_accepted,
                    local_accepted,
                )
                accepted_prefix += current_accepted.to(tl.int32)

        tl.store(accepted_count_ptr + group, accepted_prefix)

    @triton.jit
    def _bootstrap4_gemm_partial_potential_kernel(
        dot_ptr,
        key_norm_ptr,
        accepted_mask_ptr,
        partial_potential_ptr,
        row_start,
        tile_rows,
        dot_group_rows,
        dot_row_offset,
        num_candidates,
        num_partitions,
        block_rows: tl.constexpr,
        block_columns: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
    ):
        """Compute one column partition of a decision block."""
        group = tl.program_id(0)
        partition = tl.program_id(1)
        row_lanes = tl.arange(0, block_rows)
        column_lanes = tl.arange(0, block_columns)
        row_indices = row_start + row_lanes
        column_indices = partition * block_columns + column_lanes
        valid_rows = row_lanes < tile_rows
        valid_columns = column_indices < num_candidates

        query_norm = tl.load(
            key_norm_ptr + group * num_candidates + row_indices,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        key_norm = tl.load(
            key_norm_ptr + group * num_candidates + column_indices,
            mask=valid_columns,
            other=0.0,
        ).to(tl.float32)
        dot_offsets = (
            (
                group * dot_group_rows
                + dot_row_offset
                + row_lanes[:, None]
            )
            * num_candidates
            + column_indices[None, :]
        )
        dot = tl.load(
            dot_ptr + dot_offsets,
            mask=valid_rows[:, None] & valid_columns[None, :],
            other=0.0,
        ).to(tl.float32)
        distance = tl.maximum(
            query_norm[:, None] + key_norm[None, :] - 2.0 * dot,
            0.0,
        )
        contribution = _bootstrap4_riesz_potential(
            distance,
            inverse_scale_sq,
            riesz_power,
            riesz_eps,
        )
        is_decided_prefix = column_indices < row_start
        prefix_accepted = tl.load(
            accepted_mask_ptr + group * num_candidates + column_indices,
            mask=valid_columns & is_decided_prefix,
            other=0,
        ).to(tl.int1)
        visible_column = valid_columns & (
            (~is_decided_prefix) | prefix_accepted
        )
        pair_visible = (
            valid_rows[:, None]
            & visible_column[None, :]
            & (row_indices[:, None] != column_indices[None, :])
        )
        partial = tl.sum(
            tl.where(pair_visible, contribution, 0.0),
            axis=1,
        )
        partial_offsets = (
            (group * num_partitions + partition) * block_rows
            + row_lanes
        )
        tl.store(
            partial_potential_ptr + partial_offsets,
            partial,
            mask=valid_rows,
        )

    @triton.jit
    def _bootstrap4_partial_decision_kernel(
        dot_ptr,
        key_norm_ptr,
        old_potential_ptr,
        threshold_ptr,
        old_count_ptr,
        partial_potential_ptr,
        accepted_mask_ptr,
        decision_potential_ptr,
        decision_score_ptr,
        accepted_count_ptr,
        row_start,
        tile_rows,
        dot_group_rows,
        dot_row_offset,
        num_candidates,
        num_partitions,
        block_rows: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
        normalize_by_visible_count: tl.constexpr,
    ):
        """Reduce ordered partials and make one sequential decision block."""
        group = tl.program_id(0)
        row_lanes = tl.arange(0, block_rows)
        row_indices = row_start + row_lanes
        valid_rows = row_lanes < tile_rows
        query_norm = tl.load(
            key_norm_ptr + group * num_candidates + row_indices,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        row_potential = tl.load(
            old_potential_ptr + group * num_candidates + row_indices,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)

        for partition in tl.range(0, num_partitions):
            partial_offsets = (
                (group * num_partitions + partition) * block_rows
                + row_lanes
            )
            row_potential += tl.load(
                partial_potential_ptr + partial_offsets,
                mask=valid_rows,
                other=0.0,
            ).to(tl.float32)

        local_columns = row_start + row_lanes
        local_dot_offsets = (
            (
                group * dot_group_rows
                + dot_row_offset
                + row_lanes[:, None]
            )
            * num_candidates
            + local_columns[None, :]
        )
        local_dot = tl.load(
            dot_ptr + local_dot_offsets,
            mask=valid_rows[:, None] & valid_rows[None, :],
            other=0.0,
        ).to(tl.float32)
        local_key_norm = tl.load(
            key_norm_ptr + group * num_candidates + local_columns,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        local_distance = tl.maximum(
            query_norm[:, None]
            + local_key_norm[None, :]
            - 2.0 * local_dot,
            0.0,
        )
        local_contribution = _bootstrap4_riesz_potential(
            local_distance,
            inverse_scale_sq,
            riesz_power,
            riesz_eps,
        )
        local_accepted = tl.zeros((block_rows,), dtype=tl.int1)
        accepted_prefix = tl.load(accepted_count_ptr + group).to(tl.int32)
        threshold = tl.load(threshold_ptr + group).to(tl.float32)
        old_count = tl.load(old_count_ptr + group).to(tl.int32)

        for local_row in tl.static_range(0, block_rows):
            current_index = row_start + local_row
            current_valid = local_row < tile_rows
            row_selector = row_lanes == local_row
            current_potential = tl.sum(
                tl.where(row_selector, row_potential, 0.0),
                axis=0,
            )
            rejected_local_prefix = (
                (row_lanes < local_row)
                & valid_rows
                & (~local_accepted)
            )
            local_row_selector = row_lanes[:, None] == local_row
            rejected_correction = tl.sum(
                tl.where(
                    local_row_selector & rejected_local_prefix[None, :],
                    local_contribution,
                    0.0,
                ),
                axis=None,
            )
            current_potential -= rejected_correction
            visible_count = (
                old_count
                + accepted_prefix
                + (num_candidates - current_index - 1)
            )
            if normalize_by_visible_count:
                current_score = current_potential / tl.maximum(
                    visible_count.to(tl.float32),
                    1.0,
                )
            else:
                current_score = current_potential
            current_accepted = current_valid & (current_score < threshold)

            tl.store(
                accepted_mask_ptr
                + group * num_candidates
                + current_index,
                current_accepted.to(tl.int8),
                mask=current_valid,
            )
            tl.store(
                decision_potential_ptr
                + group * num_candidates
                + current_index,
                current_potential,
                mask=current_valid,
            )
            tl.store(
                decision_score_ptr
                + group * num_candidates
                + current_index,
                current_score,
                mask=current_valid,
            )
            local_accepted = tl.where(
                row_selector,
                current_accepted,
                local_accepted,
            )
            accepted_prefix += current_accepted.to(tl.int32)

        tl.store(accepted_count_ptr + group, accepted_prefix)

    @triton.jit
    def _bootstrap4_triangular_admission_kernel(
        keys_ptr,
        old_potential_ptr,
        threshold_ptr,
        old_count_ptr,
        accepted_mask_ptr,
        decision_potential_ptr,
        decision_score_ptr,
        accepted_count_ptr,
        num_candidates,
        key_dim: tl.constexpr,
        block_rows: tl.constexpr,
        block_columns: tl.constexpr,
        block_dim: tl.constexpr,
        inverse_scale_sq: tl.constexpr,
        riesz_power: tl.constexpr,
        riesz_eps: tl.constexpr,
        normalize_by_visible_count: tl.constexpr,
    ):
        group = tl.program_id(0)
        row_lanes = tl.arange(0, block_rows)
        column_lanes = tl.arange(0, block_columns)
        dim_lanes = tl.arange(0, block_dim)
        threshold = tl.load(threshold_ptr + group).to(tl.float32)
        old_count = tl.load(old_count_ptr + group).to(tl.int32)
        accepted_prefix = tl.zeros((), dtype=tl.int32)

        for row_start in tl.range(0, num_candidates, block_rows):
            row_indices = row_start + row_lanes
            valid_rows = row_indices < num_candidates
            query_offsets = (
                (group * num_candidates + row_indices[:, None]) * key_dim
                + dim_lanes[None, :]
            )
            query = tl.load(
                keys_ptr + query_offsets,
                mask=valid_rows[:, None] & (dim_lanes[None, :] < key_dim),
                other=0.0,
            )
            query_norm = tl.sum(query.to(tl.float32) * query.to(tl.float32), axis=1)
            row_potential = tl.load(
                old_potential_ptr + group * num_candidates + row_indices,
                mask=valid_rows,
                other=0.0,
            ).to(tl.float32)

            for column_start in tl.range(0, num_candidates, block_columns):
                column_indices = column_start + column_lanes
                valid_columns = column_indices < num_candidates
                key_offsets = (
                    (group * num_candidates + column_indices[:, None]) * key_dim
                    + dim_lanes[None, :]
                )
                key = tl.load(
                    keys_ptr + key_offsets,
                    mask=valid_columns[:, None]
                    & (dim_lanes[None, :] < key_dim),
                    other=0.0,
                )
                key_norm = tl.sum(
                    key.to(tl.float32) * key.to(tl.float32),
                    axis=1,
                )
                dot = tl.dot(
                    query,
                    tl.trans(key),
                    out_dtype=tl.float32,
                    input_precision="ieee",
                )
                distance = tl.maximum(
                    query_norm[:, None]
                    + key_norm[None, :]
                    - 2.0 * dot,
                    0.0,
                )
                contribution = _bootstrap4_riesz_potential(
                    distance,
                    inverse_scale_sq,
                    riesz_power,
                    riesz_eps,
                )

                is_decided_prefix = column_indices < row_start
                prefix_accepted = tl.load(
                    accepted_mask_ptr
                    + group * num_candidates
                    + column_indices,
                    mask=valid_columns & is_decided_prefix,
                    other=0,
                ).to(tl.int1)
                visible_column = valid_columns & (
                    (~is_decided_prefix) | prefix_accepted
                )
                pair_visible = (
                    valid_rows[:, None]
                    & visible_column[None, :]
                    & (row_indices[:, None] != column_indices[None, :])
                )
                row_potential += tl.sum(
                    tl.where(pair_visible, contribution, 0.0),
                    axis=1,
                )

            local_dot = tl.dot(
                query,
                tl.trans(query),
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            local_distance = tl.maximum(
                query_norm[:, None]
                + query_norm[None, :]
                - 2.0 * local_dot,
                0.0,
            )
            local_contribution = _bootstrap4_riesz_potential(
                local_distance,
                inverse_scale_sq,
                riesz_power,
                riesz_eps,
            )
            local_accepted = tl.zeros((block_rows,), dtype=tl.int1)

            for local_row in tl.static_range(0, block_rows):
                current_index = row_start + local_row
                current_valid = current_index < num_candidates
                row_selector = row_lanes == local_row
                current_potential = tl.sum(
                    tl.where(row_selector, row_potential, 0.0),
                    axis=0,
                )
                rejected_local_prefix = (
                    (row_lanes < local_row)
                    & (row_start + row_lanes < num_candidates)
                    & (~local_accepted)
                )
                local_row_selector = row_lanes[:, None] == local_row
                rejected_correction = tl.sum(
                    tl.where(
                        local_row_selector
                        & rejected_local_prefix[None, :],
                        local_contribution,
                        0.0,
                    ),
                    axis=None,
                )
                current_potential -= rejected_correction
                visible_count = (
                    old_count
                    + accepted_prefix
                    + (num_candidates - current_index - 1)
                )
                if normalize_by_visible_count:
                    current_score = current_potential / tl.maximum(
                        visible_count.to(tl.float32),
                        1.0,
                    )
                else:
                    current_score = current_potential
                current_accepted = current_valid & (current_score < threshold)

                tl.store(
                    accepted_mask_ptr
                    + group * num_candidates
                    + current_index,
                    current_accepted.to(tl.int8),
                    mask=current_valid,
                )
                tl.store(
                    decision_potential_ptr
                    + group * num_candidates
                    + current_index,
                    current_potential,
                    mask=current_valid,
                )
                tl.store(
                    decision_score_ptr
                    + group * num_candidates
                    + current_index,
                    current_score,
                    mask=current_valid,
                )
                local_accepted = tl.where(
                    row_selector,
                    current_accepted,
                    local_accepted,
                )
                accepted_prefix += current_accepted.to(tl.int32)

        tl.store(accepted_count_ptr + group, accepted_prefix)


def _bootstrap4_admission_triton_gemm_tiled(
    candidate_keys_gmd: torch.Tensor,
    old_potential_gm: torch.Tensor,
    *,
    threshold_g: torch.Tensor,
    old_count_g: torch.Tensor,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
    normalize_by_visible_count: bool,
    block_rows: int,
    block_columns: int,
    gemm_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or not candidate_keys_gmd.is_cuda:
        return None
    if candidate_keys_gmd.dtype not in {torch.bfloat16, torch.float16}:
        return None
    if block_rows not in {16, 32}:
        raise ValueError("block_rows must be 16 or 32 for the Triton path")
    if block_columns not in {16, 32, 64, 128, 256, 512}:
        raise ValueError(
            "block_columns must be 16, 32, 64, 128, 256, or 512 "
            "for the Triton path"
        )
    if gemm_rows < block_rows or gemm_rows % block_rows != 0:
        raise ValueError(
            "gemm_rows must be at least block_rows and divisible by it"
        )
    _ensure_python_include_for_triton()

    groups, num_candidates, _ = candidate_keys_gmd.shape
    ordered_mask_u8 = torch.zeros(
        groups,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=torch.uint8,
    )
    potential = torch.empty(
        groups,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=torch.float32,
    )
    score = torch.empty_like(potential)
    accepted_count = torch.zeros(
        groups,
        device=candidate_keys_gmd.device,
        dtype=torch.int32,
    )
    key_norm = (
        candidate_keys_gmd.float().square().sum(dim=-1).contiguous()
    )
    key_transpose = candidate_keys_gmd.transpose(1, 2)
    dot_buffer = torch.empty(
        groups,
        gemm_rows,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=candidate_keys_gmd.dtype,
    )
    num_warps = 8 if block_rows >= 32 or block_columns >= 128 else 4

    for gemm_start in range(0, num_candidates, gemm_rows):
        gemm_end = min(gemm_start + gemm_rows, num_candidates)
        actual_gemm_rows = gemm_end - gemm_start
        query = candidate_keys_gmd[:, gemm_start:gemm_end]
        dot = dot_buffer[:, :actual_gemm_rows]
        torch.bmm(query, key_transpose, out=dot)
        _bootstrap4_gemm_stripe_decision_kernel[(groups,)](
            dot,
            key_norm,
            old_potential_gm,
            threshold_g,
            old_count_g,
            ordered_mask_u8,
            potential,
            score,
            accepted_count,
            gemm_start,
            actual_gemm_rows,
            gemm_rows,
            num_candidates,
            block_rows=block_rows,
            block_columns=block_columns,
            inverse_scale_sq=1.0 / float(density_scale**2),
            riesz_power=float(riesz_power),
            riesz_eps=float(riesz_eps),
            normalize_by_visible_count=bool(
                normalize_by_visible_count
            ),
            num_warps=num_warps,
            num_stages=2,
        )
    return ordered_mask_u8.bool(), potential, score, accepted_count


def _bootstrap4_admission_triton_gemm_parallel(
    candidate_keys_gmd: torch.Tensor,
    old_potential_gm: torch.Tensor,
    *,
    threshold_g: torch.Tensor,
    old_count_g: torch.Tensor,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
    normalize_by_visible_count: bool,
    block_rows: int,
    block_columns: int,
    gemm_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or not candidate_keys_gmd.is_cuda:
        return None
    if candidate_keys_gmd.dtype not in {torch.bfloat16, torch.float16}:
        return None
    if block_rows not in {16, 32, 64}:
        raise ValueError(
            "block_rows must be 16, 32, or 64 for the parallel Triton path"
        )
    if block_columns not in {16, 32, 64, 128, 256, 512}:
        raise ValueError(
            "block_columns must be 16, 32, 64, 128, 256, or 512 "
            "for the Triton path"
        )
    if gemm_rows < block_rows or gemm_rows % block_rows != 0:
        raise ValueError(
            "gemm_rows must be at least block_rows and divisible by it"
        )
    _ensure_python_include_for_triton()

    groups, num_candidates, _ = candidate_keys_gmd.shape
    num_partitions = triton.cdiv(num_candidates, block_columns)
    ordered_mask_u8 = torch.zeros(
        groups,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=torch.uint8,
    )
    potential = torch.empty(
        groups,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=torch.float32,
    )
    score = torch.empty_like(potential)
    accepted_count = torch.zeros(
        groups,
        device=candidate_keys_gmd.device,
        dtype=torch.int32,
    )
    key_norm = (
        candidate_keys_gmd.float().square().sum(dim=-1).contiguous()
    )
    key_transpose = candidate_keys_gmd.transpose(1, 2)
    dot_buffer = torch.empty(
        groups,
        gemm_rows,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=candidate_keys_gmd.dtype,
    )
    partial_potential = torch.empty(
        groups,
        num_partitions,
        block_rows,
        device=candidate_keys_gmd.device,
        dtype=torch.float32,
    )
    inverse_scale_sq = 1.0 / float(density_scale**2)
    partial_warps = 8 if block_rows >= 32 or block_columns >= 128 else 4

    for gemm_start in range(0, num_candidates, gemm_rows):
        gemm_end = min(gemm_start + gemm_rows, num_candidates)
        actual_gemm_rows = gemm_end - gemm_start
        query = candidate_keys_gmd[:, gemm_start:gemm_end]
        dot = dot_buffer[:, :actual_gemm_rows]
        torch.bmm(query, key_transpose, out=dot)
        for dot_row_offset in range(0, actual_gemm_rows, block_rows):
            row_start = gemm_start + dot_row_offset
            tile_rows = min(block_rows, num_candidates - row_start)
            _bootstrap4_gemm_partial_potential_kernel[
                (groups, num_partitions)
            ](
                dot,
                key_norm,
                ordered_mask_u8,
                partial_potential,
                row_start,
                tile_rows,
                gemm_rows,
                dot_row_offset,
                num_candidates,
                num_partitions,
                block_rows=block_rows,
                block_columns=block_columns,
                inverse_scale_sq=inverse_scale_sq,
                riesz_power=float(riesz_power),
                riesz_eps=float(riesz_eps),
                num_warps=partial_warps,
                num_stages=2,
            )
            _bootstrap4_partial_decision_kernel[(groups,)](
                dot,
                key_norm,
                old_potential_gm,
                threshold_g,
                old_count_g,
                partial_potential,
                ordered_mask_u8,
                potential,
                score,
                accepted_count,
                row_start,
                tile_rows,
                gemm_rows,
                dot_row_offset,
                num_candidates,
                num_partitions,
                block_rows=block_rows,
                inverse_scale_sq=inverse_scale_sq,
                riesz_power=float(riesz_power),
                riesz_eps=float(riesz_eps),
                normalize_by_visible_count=bool(
                    normalize_by_visible_count
                ),
                num_warps=4,
                num_stages=2,
            )
    return ordered_mask_u8.bool(), potential, score, accepted_count


def _bootstrap4_admission_triton_persistent(
    candidate_keys_gmd: torch.Tensor,
    old_potential_gm: torch.Tensor,
    *,
    threshold_g: torch.Tensor,
    old_count_g: torch.Tensor,
    density_scale: float,
    riesz_power: float,
    riesz_eps: float,
    normalize_by_visible_count: bool,
    block_rows: int,
    block_columns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or not candidate_keys_gmd.is_cuda:
        return None
    if candidate_keys_gmd.dtype not in {torch.bfloat16, torch.float16}:
        return None
    if block_rows not in {16, 32}:
        raise ValueError("block_rows must be 16 or 32 for the Triton path")
    if block_columns not in {16, 32, 64, 128}:
        raise ValueError(
            "block_columns must be 16, 32, 64, or 128 for the Triton path"
        )
    _ensure_python_include_for_triton()

    groups, num_candidates, key_dim = candidate_keys_gmd.shape
    block_dim = max(16, triton.next_power_of_2(key_dim))
    ordered_mask_u8 = torch.zeros(
        groups,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=torch.uint8,
    )
    potential = torch.empty(
        groups,
        num_candidates,
        device=candidate_keys_gmd.device,
        dtype=torch.float32,
    )
    score = torch.empty_like(potential)
    accepted_count = torch.empty(
        groups,
        device=candidate_keys_gmd.device,
        dtype=torch.int32,
    )
    num_warps = 8 if block_rows == 32 or block_columns >= 64 else 4
    _bootstrap4_triangular_admission_kernel[(groups,)](
        candidate_keys_gmd,
        old_potential_gm,
        threshold_g,
        old_count_g,
        ordered_mask_u8,
        potential,
        score,
        accepted_count,
        num_candidates,
        key_dim=key_dim,
        block_rows=block_rows,
        block_columns=block_columns,
        block_dim=block_dim,
        inverse_scale_sq=1.0 / float(density_scale**2),
        riesz_power=float(riesz_power),
        riesz_eps=float(riesz_eps),
        normalize_by_visible_count=bool(normalize_by_visible_count),
        num_warps=num_warps,
        num_stages=2,
    )
    return ordered_mask_u8.bool(), potential, score, accepted_count


@torch.no_grad()
def randomized_triangular_density_admission(
    candidate_keys_gmd: torch.Tensor,
    old_potential_gm: torch.Tensor | None = None,
    *,
    threshold: float | torch.Tensor,
    old_count: int | torch.Tensor = 0,
    density_scale: float = 1.0,
    riesz_power: float = 2.0,
    riesz_eps: float = 1.0,
    seed: int = 0,
    permutation: torch.Tensor | None = None,
    normalize_by_visible_count: bool = True,
    implementation: str = "auto",
    kernel_variant: str = "gemm_parallel",
    block_rows: int = 64,
    block_columns: int = 512,
    gemm_rows: int = 768,
) -> Bootstrap4AdmissionResult:
    """Run randomized dynamic-triangular admission.

    ``threshold`` is consumed by the admission kernel. With
    ``normalize_by_visible_count=True`` it is a mean pair-potential cutoff;
    otherwise it is a total-potential cutoff.

    ``gemm_rows`` only controls how many candidate rows the GEMM-backed paths
    precompute at once. Decisions remain sequential in ``block_rows`` chunks,
    so changing it does not change admission semantics.
    """
    if candidate_keys_gmd.ndim != 3:
        raise ValueError("candidate_keys_gmd must have shape [G,M,D]")
    if implementation not in {"auto", "torch", "triton"}:
        raise ValueError("implementation must be auto, torch, or triton")
    if kernel_variant not in {
        "gemm_parallel",
        "gemm_tiled",
        "persistent",
    }:
        raise ValueError(
            "kernel_variant must be gemm_parallel, gemm_tiled, or persistent"
        )
    if density_scale <= 0 or riesz_power <= 0 or riesz_eps <= 0:
        raise ValueError("density parameters must be positive")
    groups, num_candidates, _ = candidate_keys_gmd.shape
    if num_candidates <= 0:
        raise ValueError("at least one candidate is required")
    device = candidate_keys_gmd.device
    if old_potential_gm is None:
        old_potential_gm = torch.zeros(
            groups,
            num_candidates,
            device=device,
            dtype=torch.float32,
        )
    if old_potential_gm.shape != (groups, num_candidates):
        raise ValueError("old_potential_gm must have shape [G,M]")
    old_potential_gm = old_potential_gm.detach().to(
        device=device,
        dtype=torch.float32,
    ).contiguous()
    threshold_g = _as_group_vector(
        threshold,
        groups=groups,
        device=device,
        dtype=torch.float32,
        name="threshold",
    )
    old_count_g = _as_group_vector(
        old_count,
        groups=groups,
        device=device,
        dtype=torch.int32,
        name="old_count",
    )
    if bool((threshold_g <= 0).any()):
        raise ValueError("threshold must be positive")
    if bool((old_count_g < 0).any()):
        raise ValueError("old_count must be non-negative")

    if permutation is None:
        permutation = make_bootstrap4_permutation(
            num_candidates,
            seed=seed,
            device=device,
        )
    else:
        permutation = _validate_permutation(
            permutation,
            num_candidates=num_candidates,
            device=device,
        )
    ordered_keys = candidate_keys_gmd.index_select(1, permutation).contiguous()
    ordered_old_potential = old_potential_gm.index_select(
        1, permutation
    ).contiguous()

    triton_result = None
    if implementation in {"auto", "triton"}:
        triton_kwargs = dict(
            threshold_g=threshold_g,
            old_count_g=old_count_g,
            density_scale=density_scale,
            riesz_power=riesz_power,
            riesz_eps=riesz_eps,
            normalize_by_visible_count=normalize_by_visible_count,
            block_rows=block_rows,
            block_columns=block_columns,
        )
        if kernel_variant == "gemm_parallel":
            triton_result = _bootstrap4_admission_triton_gemm_parallel(
                ordered_keys,
                ordered_old_potential,
                gemm_rows=gemm_rows,
                **triton_kwargs,
            )
        elif kernel_variant == "gemm_tiled":
            triton_result = _bootstrap4_admission_triton_gemm_tiled(
                ordered_keys,
                ordered_old_potential,
                gemm_rows=gemm_rows,
                **triton_kwargs,
            )
        else:
            triton_result = _bootstrap4_admission_triton_persistent(
                ordered_keys,
                ordered_old_potential,
                **triton_kwargs,
            )
    if triton_result is None:
        if implementation == "triton":
            raise RuntimeError(
                "Triton Bootstrap 4 admission was requested but is unavailable"
            )
        ordered_mask, ordered_potential, ordered_score, accepted_count = (
            bootstrap4_admission_reference(
                ordered_keys,
                ordered_old_potential,
                threshold=threshold_g,
                old_count=old_count_g,
                density_scale=density_scale,
                riesz_power=riesz_power,
                riesz_eps=riesz_eps,
                normalize_by_visible_count=normalize_by_visible_count,
            )
        )
        used_triton = False
    else:
        ordered_mask, ordered_potential, ordered_score, accepted_count = (
            triton_result
        )
        used_triton = True

    return Bootstrap4AdmissionResult(
        accepted_mask=_restore_source_order(ordered_mask, permutation),
        ordered_accepted_mask=ordered_mask,
        decision_potential=_restore_source_order(
            ordered_potential, permutation
        ),
        decision_score=_restore_source_order(ordered_score, permutation),
        permutation=permutation,
        accepted_count=accepted_count,
        used_triton=used_triton,
    )


__all__ = [
    "Bootstrap4AdmissionResult",
    "Bootstrap4ReferenceTrace",
    "bootstrap4_admission_reference",
    "bootstrap4_admission_reference_trace",
    "make_bootstrap4_permutation",
    "randomized_triangular_density_admission",
]
