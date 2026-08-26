"""Density-limited online KV coreset with optional dense-point eviction.

The manager treats K as the geometry and V as an opaque payload that always
moves with its paired K.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .bootstrap4_admission import randomized_triangular_density_admission

try:
    from .triton_density_bank import (
        triton_density_sum,
        triton_mutate_density_kv,
        triton_select_density_candidate,
    )
except Exception:  # pragma: no cover - optional CUDA optimization.
    triton_density_sum = None
    triton_mutate_density_kv = None
    triton_select_density_candidate = None


@dataclass(frozen=True)
class DensityKVBankConfig:
    max_entries: int = 1024
    density_scale: float = 1.0
    riesz_power: float = 2.0
    riesz_eps: float = 1.0
    replacement_ratio: float = 1.0
    evict_densest_when_full: bool = True
    max_candidates_per_update: int = 512
    max_admissions_per_update: int = 1
    process_all_candidates: bool = False
    update_chunk_size: int = 512
    full_update_mode: str = "frozen_snapshot"
    legacy_chunk_grouping: str = "contiguous"
    legacy_drop_tail_when_full: bool = False
    legacy_repeat_tail_when_full: int = 0
    legacy_warmup_chunk_size: int = -1
    legacy_density_growth_gate: bool = False
    legacy_density_gated_bootstrap: bool = False
    legacy_density_gated_bootstrap_v2: bool = False
    legacy_density_gated_bootstrap_v4: bool = False
    legacy_bootstrap_density_limit: float = 1.0
    legacy_bootstrap_v4_ratio_limit: float = 1.0
    legacy_bootstrap_v4_seed: int = 0
    legacy_bootstrap_v4_warmup_tokens: int = 0
    legacy_bootstrap_v2_gate: str = "full_union_candidate_ratio"
    legacy_bootstrap_absolute_density_limit: float = -1.0
    legacy_bootstrap_tail_cleanup_size: int = 0
    legacy_normalized_group_count: int = -1
    legacy_cleanup_divisor: int = -1
    legacy_cleanup_alignment: int = 8
    union_work_chunk_size: int = -1
    append_density_growth_limit: float = 2.0
    append_density_baseline_floor: float = 1.0e-6
    append_growth_chunk_size: int = 4096
    append_group_count_reduce: str = "min"
    append_max_entries: int = -1
    compute_dtype: str = "bfloat16"
    fast_impl: str = "auto"  # "auto", "torch", or "triton"

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if self.density_scale <= 0:
            raise ValueError("density_scale must be positive")
        if self.riesz_power <= 0 or self.riesz_eps <= 0:
            raise ValueError("Riesz parameters must be positive")
        if not 0 < self.replacement_ratio <= 1.0:
            raise ValueError("replacement_ratio must be in (0, 1]")
        if self.max_candidates_per_update <= 0:
            raise ValueError("max_candidates_per_update must be positive")
        if self.max_admissions_per_update <= 0:
            raise ValueError("max_admissions_per_update must be positive")
        if self.max_admissions_per_update > self.max_candidates_per_update:
            raise ValueError("max_admissions_per_update cannot exceed max_candidates_per_update")
        if self.update_chunk_size <= 0:
            raise ValueError("update_chunk_size must be positive")
        if self.legacy_warmup_chunk_size == 0 or self.legacy_warmup_chunk_size < -1:
            raise ValueError("legacy_warmup_chunk_size must be -1 or positive")
        if (
            self.legacy_normalized_group_count == 0
            or self.legacy_normalized_group_count < -1
        ):
            raise ValueError("legacy_normalized_group_count must be -1 or positive")
        if self.legacy_cleanup_divisor == 0 or self.legacy_cleanup_divisor < -1:
            raise ValueError("legacy_cleanup_divisor must be -1 or positive")
        if self.legacy_cleanup_alignment <= 0:
            raise ValueError("legacy_cleanup_alignment must be positive")
        if (
            self.legacy_cleanup_divisor > 0
            and self.legacy_normalized_group_count < 1
        ):
            raise ValueError(
                "legacy_cleanup_divisor requires legacy_normalized_group_count"
            )
        if self.legacy_repeat_tail_when_full < 0:
            raise ValueError("legacy_repeat_tail_when_full must be non-negative")
        if self.legacy_repeat_tail_when_full > self.update_chunk_size:
            raise ValueError(
                "legacy_repeat_tail_when_full cannot exceed update_chunk_size"
            )
        if self.legacy_repeat_tail_when_full and self.legacy_drop_tail_when_full:
            raise ValueError("legacy repeat-tail and drop-tail modes are exclusive")
        if self.legacy_normalized_group_count > 0 and (
            self.legacy_repeat_tail_when_full or self.legacy_drop_tail_when_full
        ):
            raise ValueError(
                "normalized legacy grouping is exclusive with repeat/drop tail"
            )
        if (
            self.legacy_repeat_tail_when_full
            and self.legacy_chunk_grouping != "contiguous"
        ):
            raise ValueError("legacy repeat-tail requires contiguous grouping")
        if self.union_work_chunk_size == 0 or self.union_work_chunk_size < -1:
            raise ValueError("union_work_chunk_size must be -1 or positive")
        if self.full_update_mode not in {
            "frozen_snapshot",
            "legacy_chunk_batch",
            "append_only_density",
            "union_prune_density",
            "gated_union_prune_density",
        }:
            raise ValueError(
                "full_update_mode must be frozen_snapshot, legacy_chunk_batch, "
                "append_only_density, union_prune_density, or "
                "gated_union_prune_density"
            )
        if (
            self.legacy_density_growth_gate
            and self.full_update_mode != "legacy_chunk_batch"
        ):
            raise ValueError(
                "legacy_density_growth_gate requires legacy_chunk_batch mode"
            )
        if (
            self.legacy_density_gated_bootstrap
            and self.full_update_mode != "legacy_chunk_batch"
        ):
            raise ValueError(
                "legacy_density_gated_bootstrap requires legacy_chunk_batch mode"
            )
        if (
            self.legacy_density_gated_bootstrap_v2
            and self.full_update_mode != "legacy_chunk_batch"
        ):
            raise ValueError(
                "legacy_density_gated_bootstrap_v2 requires legacy_chunk_batch mode"
            )
        if (
            self.legacy_density_gated_bootstrap_v4
            and self.full_update_mode != "legacy_chunk_batch"
        ):
            raise ValueError(
                "legacy_density_gated_bootstrap_v4 requires legacy_chunk_batch mode"
            )
        bootstrap_modes = (
            int(self.legacy_density_gated_bootstrap)
            + int(self.legacy_density_gated_bootstrap_v2)
            + int(self.legacy_density_gated_bootstrap_v4)
        )
        if bootstrap_modes > 1:
            raise ValueError("legacy bootstrap v1, v2, and v4 are mutually exclusive")
        if (
            self.legacy_density_gated_bootstrap_v4
            and self.append_group_count_reduce != "masked_max"
        ):
            raise ValueError(
                "legacy bootstrap v4 requires append_group_count_reduce=masked_max"
            )
        if self.legacy_bootstrap_v2_gate not in {
            "full_union_candidate_ratio",
            "current_old_ratio",
            "absolute_candidate_density",
            "anchor_growth_ratio",
            "all_anchor_growth_ratio",
        }:
            raise ValueError(
                "legacy_bootstrap_v2_gate must be "
                "full_union_candidate_ratio, current_old_ratio, "
                "absolute_candidate_density, anchor_growth_ratio, or "
                "all_anchor_growth_ratio"
            )
        if (
            self.legacy_bootstrap_v2_gate != "full_union_candidate_ratio"
            and not self.legacy_density_gated_bootstrap_v2
        ):
            raise ValueError(
                "non-default legacy_bootstrap_v2_gate requires "
                "legacy_density_gated_bootstrap_v2"
            )
        if (
            self.legacy_bootstrap_v2_gate == "absolute_candidate_density"
            and self.legacy_bootstrap_absolute_density_limit <= 0
        ):
            raise ValueError(
                "absolute_candidate_density requires a positive "
                "legacy_bootstrap_absolute_density_limit"
            )
        if (
            self.legacy_bootstrap_v2_gate != "absolute_candidate_density"
            and self.legacy_bootstrap_absolute_density_limit != -1.0
        ):
            raise ValueError(
                "legacy_bootstrap_absolute_density_limit is only valid for "
                "absolute_candidate_density"
            )
        if self.legacy_bootstrap_tail_cleanup_size < 0:
            raise ValueError(
                "legacy_bootstrap_tail_cleanup_size must be non-negative"
            )
        if (
            self.legacy_bootstrap_tail_cleanup_size
            and not self.legacy_density_gated_bootstrap
        ):
            raise ValueError(
                "legacy bootstrap tail cleanup requires "
                "legacy_density_gated_bootstrap"
            )
        if (
            self.legacy_bootstrap_tail_cleanup_size
            > self.update_chunk_size
        ):
            raise ValueError(
                "legacy bootstrap tail cleanup cannot exceed update_chunk_size"
            )
        if (
            self.legacy_normalized_group_count > 0
            and self.full_update_mode != "legacy_chunk_batch"
        ):
            raise ValueError(
                "legacy normalized grouping requires legacy_chunk_batch mode"
            )
        if (
            self.full_update_mode == "gated_union_prune_density"
            and self.append_group_count_reduce != "masked_max"
        ):
            raise ValueError(
                "gated_union_prune_density requires "
                "append_group_count_reduce=masked_max"
            )
        if self.legacy_chunk_grouping not in {
            "contiguous",
            "interleaved",
            "scrambled",
        }:
            raise ValueError(
                "legacy_chunk_grouping must be contiguous, interleaved, or scrambled"
            )
        if self.append_density_growth_limit <= 0:
            raise ValueError("append_density_growth_limit must be positive")
        if self.legacy_bootstrap_density_limit <= 0:
            raise ValueError("legacy_bootstrap_density_limit must be positive")
        if self.legacy_bootstrap_v4_ratio_limit <= 0:
            raise ValueError(
                "legacy_bootstrap_v4_ratio_limit must be positive"
            )
        if self.legacy_bootstrap_v4_warmup_tokens < 0:
            raise ValueError(
                "legacy_bootstrap_v4_warmup_tokens must be non-negative"
            )
        if (
            self.legacy_bootstrap_v4_warmup_tokens
            and not self.legacy_density_gated_bootstrap_v4
        ):
            raise ValueError(
                "legacy_bootstrap_v4_warmup_tokens requires "
                "legacy_density_gated_bootstrap_v4"
            )
        if self.legacy_bootstrap_v4_warmup_tokens > self.max_entries:
            raise ValueError(
                "legacy_bootstrap_v4_warmup_tokens cannot exceed max_entries"
            )
        if self.append_density_baseline_floor <= 0:
            raise ValueError("append_density_baseline_floor must be positive")
        if self.append_growth_chunk_size <= 0:
            raise ValueError("append_growth_chunk_size must be positive")
        if self.append_group_count_reduce not in {"min", "masked_max"}:
            raise ValueError(
                "append_group_count_reduce must be min or masked_max"
            )
        if self.append_max_entries == 0 or self.append_max_entries < -1:
            raise ValueError("append_max_entries must be -1 or positive")
        if self.compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("compute_dtype must be bfloat16, float16, or float32")
        if self.fast_impl not in {"auto", "torch", "triton"}:
            raise ValueError("fast_impl must be auto, torch, or triton")


@dataclass
class DensityKVBankStats:
    accepted: torch.Tensor
    added: torch.Tensor
    replaced: torch.Tensor
    rejected_full: torch.Tensor
    candidate_index: torch.Tensor
    target_slot: torch.Tensor
    candidate_density: torch.Tensor
    replacement_density: torch.Tensor
    evicted_density: torch.Tensor
    energy_delta: torch.Tensor
    accepted_count: torch.Tensor
    accepted_entry_mask: torch.Tensor

    def summary(self) -> dict[str, float]:
        return {
            "accepted": float(self.accepted.sum().item()),
            "accepted_entries": float(self.accepted_count.sum().item()),
            "added": float(self.added.sum().item()),
            "replaced": float(self.replaced.sum().item()),
            "rejected_full": float(self.rejected_full.sum().item()),
            "mean_candidate_density": float(self.candidate_density.float().mean().item()),
            "mean_energy_delta": float(self.energy_delta.float().mean().item()),
        }


@dataclass
class DensityKVBankView:
    keys: torch.Tensor
    values: torch.Tensor
    density: torch.Tensor
    active_mask: torch.Tensor
    counts: torch.Tensor


class DensityLimitedKVBank(nn.Module):
    """Maintain a bounded per-group set of exact K/V pairs.

    Shape convention:
        incoming keys:   [N, G, Dk]
        incoming values: [N, G, Dv]
        stored keys:     [G, M, Dk]
        stored values:   [G, M, Dv]

    The compatibility path admits a bounded number of low-density candidates
    per update. With ``process_all_candidates=True``, every incoming KV pair is
    scored against the same pre-update bank snapshot. Chunking bounds temporary
    GEMM memory but cannot alter admission. Candidate-to-candidate interactions
    from the current update are excluded from admission, then included when the
    accepted replacements are committed and density is rebuilt exactly:

        E(S - e + x) - E(S) = Phi_{S\\e}(x) - Phi_S(e).

    V never participates in distance or density calculations.
    """

    def __init__(
        self,
        groups: int,
        key_dim: int,
        value_dim: int,
        config: DensityKVBankConfig | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if groups <= 0 or key_dim <= 0 or value_dim <= 0:
            raise ValueError("groups, key_dim, and value_dim must be positive")
        self.config = config or DensityKVBankConfig()
        self.groups = int(groups)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        storage_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.compute_dtype]
        shape = (self.groups, self.config.max_entries)
        self.register_buffer(
            "keys",
            torch.zeros(*shape, self.key_dim, device=device, dtype=storage_dtype),
            persistent=True,
        )
        self.register_buffer(
            "values",
            torch.zeros(*shape, self.value_dim, device=device, dtype=storage_dtype),
            persistent=True,
        )
        self.register_buffer(
            "density",
            torch.zeros(*shape, device=device, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "density_baseline",
            torch.zeros(*shape, device=device, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "counts",
            torch.zeros(self.groups, device=device, dtype=torch.int32),
            persistent=True,
        )
        self.register_buffer(
            "source_index",
            torch.full(shape, -1, device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "insert_query_frame",
            torch.full(shape, -1, device=device, dtype=torch.int32),
            persistent=False,
        )
        self._legacy_baseline_initialized = False
        self._bootstrap4_commit_index = 0

    @property
    def device(self) -> torch.device:
        return self.keys.device

    def clear(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.density.zero_()
        self.density_baseline.zero_()
        self.counts.zero_()
        self.source_index.fill_(-1)
        self.insert_query_frame.fill_(-1)
        self._legacy_baseline_initialized = False
        self._bootstrap4_commit_index = 0

    def active_mask(self) -> torch.Tensor:
        slots = torch.arange(self.keys.shape[1], device=self.device)
        return slots.unsqueeze(0) < self.counts.long().unsqueeze(1)

    def _ensure_storage_capacity(self, required: int) -> None:
        current = int(self.keys.shape[1])
        if required <= current:
            return
        growth = self.config.append_growth_chunk_size
        expanded = ((required + growth - 1) // growth) * growth
        expanded = max(expanded, current + growth)
        extra = expanded - current
        self.keys = torch.cat(
            (
                self.keys,
                torch.zeros(
                    self.groups,
                    extra,
                    self.key_dim,
                    device=self.device,
                    dtype=self.keys.dtype,
                ),
            ),
            dim=1,
        )
        self.values = torch.cat(
            (
                self.values,
                torch.zeros(
                    self.groups,
                    extra,
                    self.value_dim,
                    device=self.device,
                    dtype=self.values.dtype,
                ),
            ),
            dim=1,
        )
        self.density = torch.cat(
            (
                self.density,
                torch.zeros(
                    self.groups,
                    extra,
                    device=self.device,
                    dtype=self.density.dtype,
                ),
            ),
            dim=1,
        )
        self.density_baseline = torch.cat(
            (
                self.density_baseline,
                torch.zeros(
                    self.groups,
                    extra,
                    device=self.device,
                    dtype=self.density_baseline.dtype,
                ),
            ),
            dim=1,
        )
        self.source_index = torch.cat(
            (
                self.source_index,
                torch.full(
                    (self.groups, extra),
                    -1,
                    device=self.device,
                    dtype=self.source_index.dtype,
                ),
            ),
            dim=1,
        )
        self.insert_query_frame = torch.cat(
            (
                self.insert_query_frame,
                torch.full(
                    (self.groups, extra),
                    -1,
                    device=self.device,
                    dtype=self.insert_query_frame.dtype,
                ),
            ),
            dim=1,
        )

    def view(self) -> DensityKVBankView:
        return DensityKVBankView(
            keys=self.keys,
            values=self.values,
            density=self.density,
            active_mask=self.active_mask(),
            counts=self.counts,
        )

    def _density_contribution(self, squared_distance: torch.Tensor) -> torch.Tensor:
        normalized = squared_distance.float() / (self.config.density_scale ** 2)
        return 1.0 / (self.config.riesz_eps + normalized).pow(self.config.riesz_power)

    def _density_sum(
        self,
        squared_distance: torch.Tensor,
        *,
        exclude_diagonal: bool = False,
    ) -> torch.Tensor:
        if self.config.fast_impl in {"auto", "triton"} and triton_density_sum is not None:
            result = triton_density_sum(
                squared_distance,
                density_scale=self.config.density_scale,
                riesz_power=self.config.riesz_power,
                riesz_eps=self.config.riesz_eps,
                exclude_diagonal=exclude_diagonal,
            )
            if result is not None:
                return result
        contribution = self._density_contribution(squared_distance)
        if exclude_diagonal:
            if squared_distance.shape[-2] != squared_distance.shape[-1]:
                raise ValueError("diagonal exclusion requires a square distance matrix")
            diagonal = torch.eye(
                squared_distance.shape[-1],
                device=squared_distance.device,
                dtype=torch.bool,
            ).unsqueeze(0)
            contribution = contribution.masked_fill(diagonal, 0.0)
        return contribution.sum(dim=-1)

    @staticmethod
    def _squared_l2(query_gnd: torch.Tensor, key_gmd: torch.Tensor) -> torch.Tensor:
        dot = torch.bmm(query_gnd, key_gmd.transpose(1, 2))
        query_norm = query_gnd.float().square().sum(dim=-1, keepdim=True)
        key_norm = key_gmd.float().square().sum(dim=-1).unsqueeze(1)
        return (query_norm + key_norm - 2.0 * dot.float()).clamp_min_(0.0)

    def _thin_candidates(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        limit = self.config.max_candidates_per_update
        if keys.shape[0] <= limit:
            return keys, values
        indices = torch.linspace(
            0,
            keys.shape[0] - 1,
            limit,
            device=keys.device,
        ).round().long().unique()
        return keys.index_select(0, indices), values.index_select(0, indices)

    def _prepare_inputs(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        thin: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.ndim != 3 or keys.shape[1:] != (self.groups, self.key_dim):
            raise ValueError(
                f"keys must have shape [N, {self.groups}, {self.key_dim}], "
                f"got {tuple(keys.shape)}"
            )
        if values.ndim != 3 or values.shape != (keys.shape[0], self.groups, self.value_dim):
            raise ValueError(
                f"values must have shape [N, {self.groups}, {self.value_dim}], "
                f"got {tuple(values.shape)}"
            )
        if keys.shape[0] == 0:
            raise ValueError("a KV chunk must contain at least one candidate")
        keys = keys.detach().to(device=self.device, dtype=self.keys.dtype).contiguous()
        values = values.detach().to(device=self.device, dtype=self.values.dtype).contiguous()
        should_thin = thin and not self.config.process_all_candidates
        return self._thin_candidates(keys, values) if should_thin else (keys, values)

    def _select_torch(
        self,
        distances_gnm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active = self.active_mask()
        contribution = self._density_contribution(distances_gnm)
        candidate_density = (
            contribution * active.unsqueeze(1).to(dtype=contribution.dtype)
        ).sum(dim=-1)
        sparse_density, sparse_index = candidate_density.min(dim=1)
        masked_density = self.density.masked_fill(~active, -float("inf"))
        dense_density, dense_slot = masked_density.max(dim=1)
        full = self.counts.long() >= self.config.max_entries
        target_slot = torch.where(full, dense_slot, self.counts.long())
        target_density = torch.where(full, dense_density, torch.zeros_like(dense_density))
        return sparse_index, sparse_density, target_slot, target_density

    def _select(
        self,
        distances_gnm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        use_triton = self.config.fast_impl in {"auto", "triton"}
        if use_triton and triton_select_density_candidate is not None:
            result = triton_select_density_candidate(
                distances_gnm,
                self.density,
                self.counts,
                density_scale=self.config.density_scale,
                riesz_power=self.config.riesz_power,
                riesz_eps=self.config.riesz_eps,
            )
            if result is not None:
                return (
                    result.candidate_index,
                    result.candidate_density,
                    result.target_slot,
                    result.target_density,
                    True,
                )
        if self.config.fast_impl == "triton":
            raise RuntimeError("Triton density selection was requested but is unavailable")
        selected = self._select_torch(distances_gnm)
        return *selected, False

    def _mutate_torch(
        self,
        selected_keys: torch.Tensor,
        selected_values: torch.Tensor,
        new_distances: torch.Tensor,
        old_distances: torch.Tensor,
        target_slots: torch.Tensor,
        accepted: torch.Tensor,
        was_full: torch.Tensor,
        replacement_density: torch.Tensor,
    ) -> None:
        active = self.active_mask()
        slots = torch.arange(self.config.max_entries, device=self.device).unsqueeze(0)
        pair_mask = active & (slots != target_slots.unsqueeze(1))
        new_contribution = self._density_contribution(new_distances) * pair_mask
        old_contribution = (
            self._density_contribution(old_distances)
            * pair_mask
            * was_full.unsqueeze(1)
        )
        updated = self.density + new_contribution - old_contribution
        updated.scatter_(1, target_slots.unsqueeze(1), replacement_density.unsqueeze(1))
        self.density.copy_(torch.where(accepted.unsqueeze(1), updated, self.density))
        groups = torch.arange(self.groups, device=self.device)
        accepted_groups = groups[accepted]
        accepted_slots = target_slots[accepted]
        if accepted_groups.numel() > 0:
            self.keys[accepted_groups, accepted_slots] = selected_keys[accepted]
            self.values[accepted_groups, accepted_slots] = selected_values[accepted]
        self.counts.add_((accepted & ~was_full).to(self.counts.dtype))

    @staticmethod
    def _gather_rows(values_gnd: torch.Tensor, indices_gr: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values_gnd,
            1,
            indices_gr.unsqueeze(-1).expand(-1, -1, values_gnd.shape[-1]),
        )

    @staticmethod
    def _gather_matrix_columns(matrix_grm: torch.Tensor, indices_gr: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            matrix_grm,
            2,
            indices_gr.unsqueeze(1).expand(-1, matrix_grm.shape[1], -1),
        )

    @torch.no_grad()
    def _append_full_union_batch_chunked(
        self,
        keys_gnd: torch.Tensor,
        values_gnd: torch.Tensor,
        *,
        active_count: int,
    ) -> DensityKVBankStats:
        """Append an uncompressed union-prune batch with bounded work memory."""
        groups, num_candidates, _ = keys_gnd.shape
        internal_density = self._density_over_union(keys_gnd)
        external_density = torch.zeros_like(internal_density)
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        if active_count > 0:
            for start in range(0, num_candidates, work_chunk_size):
                end = min(start + work_chunk_size, num_candidates)
                distances = self._squared_l2(
                    keys_gnd[:, start:end], self.keys[:, :active_count]
                )
                contribution = self._density_contribution(distances)
                external_density[:, start:end] = contribution.sum(dim=-1)
                self.density[:, :active_count].add_(contribution.sum(dim=1))
        new_density = external_density + internal_density
        end = active_count + num_candidates
        self.keys[:, active_count:end].copy_(keys_gnd)
        self.values[:, active_count:end].copy_(values_gnd)
        self.density[:, active_count:end].copy_(new_density)
        self.counts.fill_(end)

        selected_indices = torch.arange(
            num_candidates, device=self.device
        ).unsqueeze(0).expand(groups, -1)
        target_slots = torch.arange(
            active_count, end, device=self.device
        ).unsqueeze(0).expand(groups, -1)
        accepted = torch.ones(groups, device=self.device, dtype=torch.bool)
        accepted_count = torch.full(
            (groups,), num_candidates, device=self.device, dtype=torch.int32
        )
        added_energy = new_density.sum(dim=1) - 0.5 * internal_density.sum(dim=1)
        candidate_energy_share = external_density + 0.5 * internal_density
        stats = DensityKVBankStats(
            accepted=accepted,
            added=accepted,
            replaced=torch.zeros_like(accepted),
            rejected_full=torch.zeros_like(accepted),
            candidate_index=selected_indices,
            target_slot=target_slots,
            candidate_density=new_density,
            replacement_density=new_density,
            evicted_density=torch.zeros_like(new_density),
            energy_delta=added_energy.unsqueeze(1).expand_as(new_density),
            accepted_count=accepted_count,
            accepted_entry_mask=torch.ones_like(selected_indices, dtype=torch.bool),
        )
        stats.trace_decision_group = torch.zeros_like(selected_indices)
        stats.trace_group_reason = torch.zeros(
            groups, 1, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = torch.ones(
            groups, 1, device=self.device, dtype=torch.bool
        )
        stats.trace_group_candidate_count = torch.full(
            (groups, 1), num_candidates, device=self.device, dtype=torch.int32
        )
        stats.trace_group_added_energy = added_energy.unsqueeze(1)
        stats.trace_group_removed_energy = torch.zeros_like(
            stats.trace_group_added_energy
        )
        stats.trace_candidate_energy_share = candidate_energy_share
        stats.trace_victim_energy_share = torch.zeros_like(candidate_energy_share)
        return stats

    @torch.no_grad()
    def _update_batch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        active_count: int | None = None,
        counts_synchronized: bool = False,
        admission_limit: int | None = None,
    ) -> DensityKVBankStats:
        """Vectorized multi-entry update for a clean KV chunk."""
        if not counts_synchronized and not torch.equal(
            self.counts,
            self.counts[:1].expand_as(self.counts),
        ):
            raise RuntimeError("batch KV updates require synchronized group counts")
        if active_count is None:
            active_count = int(self.counts[0].item())
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        groups, num_candidates, _ = keys_gnd.shape
        free_entries = max(self.config.max_entries - active_count, 0)
        selection_limit = free_entries if free_entries > 0 else self.config.max_entries
        num_selected = min(
            self.config.max_admissions_per_update
            if admission_limit is None
            else int(admission_limit),
            num_candidates,
            selection_limit,
        )
        if (
            self.config.full_update_mode == "union_prune_density"
            and free_entries > 0
            and num_selected == num_candidates
        ):
            return self._append_full_union_batch_chunked(
                keys_gnd,
                values_gnd,
                active_count=active_count,
            )

        if active_count > 0:
            use_growth_gate = (
                self.config.legacy_density_growth_gate
                and active_count >= self.config.max_entries
            )
            if use_growth_gate and not self._legacy_baseline_initialized:
                self.density_baseline[:, :active_count].copy_(
                    self.density[:, :active_count].clamp_min(
                        self.config.append_density_baseline_floor
                    )
                )
                self._legacy_baseline_initialized = True
            distances_to_store = self._squared_l2(
                keys_gnd,
                self.keys[:, :active_count],
            )
            density_to_store = self._density_sum(distances_to_store)
            if use_growth_gate:
                nearest_store = distances_to_store.argmin(dim=-1)
                reference_baseline = torch.gather(
                    self.density_baseline[:, :active_count],
                    1,
                    nearest_store,
                ).clamp_min(self.config.append_density_baseline_floor)
                saturation_ratio = density_to_store / reference_baseline
            else:
                nearest_store = None
                reference_baseline = None
                saturation_ratio = None
        else:
            use_growth_gate = False
            distances_to_store = torch.empty(
                groups,
                num_candidates,
                0,
                device=self.device,
                dtype=torch.float32,
            )
            density_to_store = torch.zeros(
                groups,
                num_candidates,
                device=self.device,
                dtype=torch.float32,
            )
            nearest_store = None
            reference_baseline = None
            saturation_ratio = None

        candidate_pair_distances = self._squared_l2(keys_gnd, keys_gnd)
        density_in_chunk = self._density_sum(
            candidate_pair_distances,
            exclude_diagonal=True,
        )
        selection_score = density_to_store + density_in_chunk
        if free_entries > 0 and num_selected == num_candidates:
            # A no-compression update must preserve the source order.  Attention
            # is permutation invariant in exact arithmetic, but changing the
            # reduction order creates avoidable autoregressive numerical drift.
            selected_indices = torch.arange(
                num_candidates,
                device=self.device,
            ).unsqueeze(0).expand(groups, -1)
        else:
            selected_indices = torch.topk(
                selection_score,
                k=num_selected,
                dim=1,
                largest=False,
                sorted=False,
            ).indices
        selected_keys = self._gather_rows(keys_gnd, selected_indices).contiguous()
        selected_values = self._gather_rows(values_gnd, selected_indices).contiguous()
        selected_score = torch.gather(selection_score, 1, selected_indices)
        if use_growth_gate:
            assert nearest_store is not None
            assert reference_baseline is not None
            assert saturation_ratio is not None
            selected_nearest_store = torch.gather(
                nearest_store, 1, selected_indices
            )
            selected_reference_baseline = torch.gather(
                reference_baseline, 1, selected_indices
            )
            selected_saturation_ratio = torch.gather(
                saturation_ratio, 1, selected_indices
            )
            growth_gate_accepted = (
                selected_saturation_ratio.mean(dim=1)
                < self.config.append_density_growth_limit
            )
        else:
            selected_nearest_store = None
            selected_reference_baseline = None
            selected_saturation_ratio = None
            growth_gate_accepted = torch.ones(
                groups, device=self.device, dtype=torch.bool
            )
        selected_pair_distances = self._squared_l2(selected_keys, selected_keys)
        selected_internal_density = self._density_sum(
            selected_pair_distances,
            exclude_diagonal=True,
        )

        if free_entries > 0:
            if active_count > 0:
                selected_to_store = self._gather_rows(
                    distances_to_store,
                    selected_indices,
                )
                selected_to_store_contribution = self._density_contribution(selected_to_store)
                self.density[:, :active_count].add_(
                    selected_to_store_contribution.sum(dim=1)
                )
                new_density = (
                    selected_to_store_contribution.sum(dim=-1)
                    + selected_internal_density
                )
            else:
                new_density = selected_internal_density
            end = active_count + num_selected
            self.keys[:, active_count:end].copy_(selected_keys)
            self.values[:, active_count:end].copy_(selected_values)
            self.density[:, active_count:end].copy_(new_density)
            self.counts.fill_(end)
            if self.config.legacy_density_growth_gate and end >= self.config.max_entries:
                self.density_baseline[:, :end].copy_(
                    self.density[:, :end].clamp_min(
                        self.config.append_density_baseline_floor
                    )
                )
                self._legacy_baseline_initialized = True
            target_slots = torch.arange(
                active_count,
                end,
                device=self.device,
            ).unsqueeze(0).expand(groups, -1)
            accepted = torch.ones(groups, device=self.device, dtype=torch.bool)
            accepted_count = torch.full(
                (groups,), num_selected, device=self.device, dtype=torch.int32
            )
            added_energy = (
                new_density.sum(dim=1)
                - 0.5 * selected_internal_density.sum(dim=1)
            )
            if active_count > 0:
                candidate_energy_share = (
                    selected_to_store_contribution.sum(dim=-1)
                    + 0.5 * selected_internal_density
                )
            else:
                candidate_energy_share = 0.5 * selected_internal_density
            stats = DensityKVBankStats(
                accepted=accepted,
                added=accepted,
                replaced=torch.zeros_like(accepted),
                rejected_full=torch.zeros_like(accepted),
                candidate_index=selected_indices,
                target_slot=target_slots,
                candidate_density=selected_score,
                replacement_density=new_density,
                evicted_density=torch.zeros_like(new_density),
                energy_delta=added_energy.unsqueeze(1).expand_as(selected_score),
                accepted_count=accepted_count,
                accepted_entry_mask=torch.ones_like(
                    selected_indices,
                    dtype=torch.bool,
                ),
            )
            stats.trace_decision_group = torch.zeros_like(selected_indices)
            stats.trace_group_reason = torch.zeros(
                groups, 1, device=self.device, dtype=torch.int8
            )
            stats.trace_group_accepted = torch.ones(
                groups, 1, device=self.device, dtype=torch.bool
            )
            stats.trace_group_candidate_count = torch.full(
                (groups, 1), num_selected, device=self.device, dtype=torch.int32
            )
            stats.trace_group_added_energy = added_energy.unsqueeze(1)
            stats.trace_group_removed_energy = torch.zeros_like(
                stats.trace_group_added_energy
            )
            stats.trace_candidate_energy_share = candidate_energy_share
            stats.trace_victim_energy_share = torch.zeros_like(
                candidate_energy_share
            )
            if self.config.legacy_density_growth_gate:
                stats.trace_gate_accepted = torch.ones_like(
                    selected_indices, dtype=torch.bool
                )
            return stats

        evicted_density, evicted_slots = torch.topk(
            self.density,
            k=num_selected,
            dim=1,
            largest=True,
            sorted=False,
        )
        old_keys = self._gather_rows(self.keys, evicted_slots).contiguous()
        selected_to_store = self._gather_rows(distances_to_store, selected_indices)
        old_to_store = self._squared_l2(old_keys, self.keys)
        selected_to_store_contribution = self._density_contribution(selected_to_store)
        old_to_store_contribution = self._density_contribution(old_to_store)
        selected_to_evicted = self._gather_matrix_columns(
            selected_to_store_contribution,
            evicted_slots,
        )
        old_pair_distances = self._squared_l2(old_keys, old_keys)
        old_internal_density = self._density_sum(
            old_pair_distances,
            exclude_diagonal=True,
        )
        selected_to_kept = (
            selected_to_store_contribution.sum(dim=-1)
            - selected_to_evicted.sum(dim=-1)
        )
        replacement_density = selected_to_kept + selected_internal_density
        removed_energy = (
            evicted_density.sum(dim=1)
            - 0.5 * old_internal_density.sum(dim=1)
        )
        added_energy = (
            selected_to_kept.sum(dim=1)
            + 0.5 * selected_internal_density.sum(dim=1)
        )
        energy_delta = added_energy - removed_energy
        candidate_energy_share = (
            selected_to_kept + 0.5 * selected_internal_density
        )
        victim_energy_share = (
            evicted_density - 0.5 * old_internal_density
        )
        accepted = (
            bool(self.config.evict_densest_when_full)
            & (added_energy < removed_energy * self.config.replacement_ratio)
            & growth_gate_accepted
        )

        proposal_density = (
            self.density
            - old_to_store_contribution.sum(dim=1)
            + selected_to_store_contribution.sum(dim=1)
        )
        proposal_density.scatter_(1, evicted_slots, replacement_density)
        self.density.copy_(
            torch.where(accepted.unsqueeze(1), proposal_density, self.density)
        )
        group_index = torch.arange(groups, device=self.device).unsqueeze(1).expand(
            -1, num_selected
        )
        accepted_entries = accepted.unsqueeze(1).expand(-1, num_selected)
        self.keys[
            group_index[accepted_entries], evicted_slots[accepted_entries]
        ] = selected_keys[accepted_entries]
        self.values[
            group_index[accepted_entries], evicted_slots[accepted_entries]
        ] = selected_values[accepted_entries]
        if use_growth_gate:
            assert selected_reference_baseline is not None
            self.density_baseline[
                group_index[accepted_entries], evicted_slots[accepted_entries]
            ] = selected_reference_baseline[accepted_entries]
        accepted_count = accepted.to(torch.int32) * num_selected
        stats = DensityKVBankStats(
            accepted=accepted,
            added=torch.zeros_like(accepted),
            replaced=accepted,
            rejected_full=~accepted,
            candidate_index=selected_indices,
            target_slot=evicted_slots,
            candidate_density=selected_score,
            replacement_density=replacement_density,
            evicted_density=evicted_density,
            energy_delta=energy_delta.unsqueeze(1).expand_as(selected_score),
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros_like(selected_indices)
        stats.trace_group_reason = torch.ones(
            groups, 1, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (groups, 1), num_selected, device=self.device, dtype=torch.int32
        )
        stats.trace_group_added_energy = added_energy.unsqueeze(1)
        stats.trace_group_removed_energy = removed_energy.unsqueeze(1)
        stats.trace_candidate_energy_share = candidate_energy_share
        stats.trace_victim_energy_share = victim_energy_share
        if use_growth_gate:
            assert selected_saturation_ratio is not None
            assert selected_nearest_store is not None
            assert selected_reference_baseline is not None
            stats.trace_gate_accepted = growth_gate_accepted.unsqueeze(1).expand_as(
                selected_indices
            )
            stats.trace_anchor_slot = selected_nearest_store
            stats.trace_reference_density = selected_reference_baseline
            stats.trace_saturation_ratio = selected_saturation_ratio
        return stats

    @torch.no_grad()
    def _update_all_candidates_frozen_snapshot(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Score every candidate against one frozen pre-update bank snapshot."""
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError("full KV updates require synchronized group counts")
        active_count = int(self.counts[0].item())
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        num_candidates = int(keys_gnd.shape[1])
        candidate_density_chunks = []
        for start in range(0, num_candidates, self.config.update_chunk_size):
            end = min(start + self.config.update_chunk_size, num_candidates)
            if active_count == 0:
                density = torch.zeros(
                    self.groups,
                    end - start,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                distances = self._squared_l2(
                    keys_gnd[:, start:end],
                    self.keys[:, :active_count],
                )
                # The current Triton reduction is tuned for the small single-
                # admission path and is slower for a 512 x 16384 tile. Keep an
                # explicit triton request reproducible, but let auto use the
                # faster vectorized PyTorch reduction for full-frame scoring.
                density = (
                    self._density_sum(distances)
                    if self.config.fast_impl == "triton"
                    else self._density_contribution(distances).sum(dim=-1)
                )
            candidate_density_chunks.append(density)
        candidate_density = torch.cat(candidate_density_chunks, dim=1)

        free_entries = max(self.config.max_entries - active_count, 0)
        if free_entries >= num_candidates:
            # Preserve token order while the update is an exact, no-compression append.
            candidate_order = torch.arange(
                num_candidates,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(self.groups, -1)
        else:
            candidate_order = torch.argsort(
                candidate_density,
                dim=1,
                descending=False,
                stable=True,
            )
        ordered_density = torch.gather(candidate_density, 1, candidate_order)
        target_slot = torch.full_like(candidate_order, -1)
        replacement_density = ordered_density.clone()
        evicted_density = torch.zeros_like(ordered_density)
        accepted_entries = torch.zeros_like(candidate_order, dtype=torch.bool)
        added_entries = torch.zeros_like(accepted_entries)
        replaced_entries = torch.zeros_like(accepted_entries)

        if free_entries > 0:
            num_added = min(free_entries, num_candidates)
            selected_indices = candidate_order[:, :num_added]
            selected_keys = self._gather_rows(keys_gnd, selected_indices).contiguous()
            selected_values = self._gather_rows(values_gnd, selected_indices).contiguous()
            selected_pair_distances = self._squared_l2(selected_keys, selected_keys)
            selected_internal_density = self._density_sum(
                selected_pair_distances,
                exclude_diagonal=True,
            )
            if active_count > 0:
                selected_to_store = self._squared_l2(
                    selected_keys,
                    self.keys[:, :active_count],
                )
                selected_to_store_contribution = self._density_contribution(
                    selected_to_store
                )
                self.density[:, :active_count].add_(
                    selected_to_store_contribution.sum(dim=1)
                )
                new_density = (
                    selected_to_store_contribution.sum(dim=-1)
                    + selected_internal_density
                )
            else:
                new_density = selected_internal_density
            end = active_count + num_added
            slots = torch.arange(
                active_count,
                end,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(self.groups, -1)
            self.keys[:, active_count:end].copy_(selected_keys)
            self.values[:, active_count:end].copy_(selected_values)
            self.density[:, active_count:end].copy_(new_density)
            self.counts.fill_(end)
            target_slot[:, :num_added] = slots
            replacement_density[:, :num_added] = new_density
            accepted_entries[:, :num_added] = True
            added_entries[:, :num_added] = True
        elif bool(self.config.evict_densest_when_full):
            num_paired = min(num_candidates, self.config.max_entries)
            evicted_density_paired, evicted_slots = torch.topk(
                self.density,
                k=num_paired,
                dim=1,
                largest=True,
                sorted=True,
            )
            selected_indices = candidate_order[:, :num_paired]
            selected_keys = self._gather_rows(keys_gnd, selected_indices).contiguous()
            selected_values = self._gather_rows(values_gnd, selected_indices).contiguous()
            old_keys = self._gather_rows(self.keys, evicted_slots).contiguous()
            paired_distance = (
                selected_keys.float() - old_keys.float()
            ).square().sum(dim=-1)
            paired_contribution = self._density_contribution(paired_distance)
            candidate_without_target = (
                ordered_density[:, :num_paired] - paired_contribution
            ).clamp_min_(0.0)
            accepted = candidate_without_target < (
                evicted_density_paired * self.config.replacement_ratio
            )
            target_slot[:, :num_paired] = evicted_slots
            replacement_density[:, :num_paired] = candidate_without_target
            evicted_density[:, :num_paired] = evicted_density_paired
            accepted_entries[:, :num_paired] = accepted
            replaced_entries[:, :num_paired] = accepted

            # Admission above reads only the frozen bank. Mutate after every
            # candidate has been scored, then rebuild all affected densities
            # exactly, including interactions among newly accepted points.
            accepted_count = accepted.sum(dim=1, dtype=torch.int32)
            max_accepted = int(accepted_count.max().item())
            if max_accepted > 0:
                accepted_order = torch.argsort(
                    accepted.to(torch.int8),
                    dim=1,
                    descending=True,
                    stable=True,
                )[:, :max_accepted]
                valid = torch.arange(
                    max_accepted,
                    device=self.device,
                ).unsqueeze(0) < accepted_count.unsqueeze(1)
                candidate_ids = torch.gather(
                    selected_indices,
                    1,
                    accepted_order,
                ).masked_fill(~valid, 0)
                slots = torch.gather(
                    evicted_slots,
                    1,
                    accepted_order,
                ).masked_fill(~valid, 0)
                new_keys = self._gather_rows(keys_gnd, candidate_ids).contiguous()
                new_values = self._gather_rows(values_gnd, candidate_ids).contiguous()
                old_keys = self._gather_rows(self.keys, slots).contiguous()
                proposal = self.density.clone()
                new_to_kept = torch.zeros(
                    self.groups,
                    max_accepted,
                    device=self.device,
                    dtype=torch.float32,
                )
                # Keep the mutation reduction order canonical so changing the
                # scoring tile cannot perturb the persisted density table.
                mutation_chunk = min(512, max_accepted)
                for start in range(0, max_accepted, mutation_chunk):
                    end = min(start + mutation_chunk, max_accepted)
                    chunk_valid = valid[:, start:end]
                    old_to_store = self._squared_l2(
                        old_keys[:, start:end],
                        self.keys,
                    )
                    old_contribution = self._density_contribution(old_to_store)
                    new_to_store = self._squared_l2(
                        new_keys[:, start:end],
                        self.keys,
                    )
                    new_contribution = self._density_contribution(new_to_store)
                    contribution_mask = chunk_valid.unsqueeze(-1)
                    proposal.add_(
                        (
                            (new_contribution - old_contribution)
                            * contribution_mask
                        ).sum(dim=1)
                    )
                    contribution_to_evicted = self._gather_matrix_columns(
                        new_contribution,
                        slots,
                    )
                    new_to_kept[:, start:end] = (
                        new_contribution.sum(dim=-1)
                        - (
                            contribution_to_evicted
                            * valid.unsqueeze(1)
                        ).sum(dim=-1)
                    ) * chunk_valid
                    del old_to_store, old_contribution
                    del new_to_store, new_contribution, contribution_to_evicted

                new_internal = torch.zeros_like(new_to_kept)
                for start in range(0, max_accepted, mutation_chunk):
                    end = min(start + mutation_chunk, max_accepted)
                    chunk_valid = valid[:, start:end]
                    pair_contribution = self._density_contribution(
                        self._squared_l2(
                            new_keys[:, start:end],
                            new_keys,
                        )
                    )
                    diagonal = torch.arange(
                        end - start,
                        device=self.device,
                    )
                    pair_contribution[:, diagonal, diagonal + start] = 0.0
                    new_internal[:, start:end] = (
                        pair_contribution
                        * valid.unsqueeze(1)
                    ).sum(dim=-1)
                    new_internal[:, start:end].mul_(chunk_valid).clamp_min_(0.0)
                    del pair_contribution

                new_density = new_to_kept + new_internal
                groups = torch.arange(
                    self.groups,
                    device=self.device,
                ).unsqueeze(1).expand_as(slots)
                proposal[groups[valid], slots[valid]] = new_density[valid]
                self.keys[groups[valid], slots[valid]] = new_keys[valid]
                self.values[groups[valid], slots[valid]] = new_values[valid]
                self.density.copy_(proposal.clamp_min_(0.0))

        accepted = accepted_entries.any(dim=1)
        added = added_entries.any(dim=1)
        replaced = replaced_entries.any(dim=1)
        rejected_full = (
            active_count >= self.config.max_entries
        ) & ~accepted
        stats = DensityKVBankStats(
            accepted=accepted,
            added=added,
            replaced=replaced,
            rejected_full=rejected_full,
            candidate_index=candidate_order,
            target_slot=target_slot,
            candidate_density=ordered_density,
            replacement_density=replacement_density,
            evicted_density=evicted_density,
            energy_delta=replacement_density - evicted_density,
            accepted_count=accepted_entries.sum(dim=1, dtype=torch.int32),
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.arange(
            num_candidates, device=self.device, dtype=torch.int32
        ).unsqueeze(0).expand(self.groups, -1)
        stats.trace_entry_reason = torch.where(
            target_slot < 0,
            torch.full_like(target_slot, 3, dtype=torch.int8),
            torch.where(
                added_entries,
                torch.zeros_like(target_slot, dtype=torch.int8),
                torch.full_like(target_slot, 2, dtype=torch.int8),
            ),
        )
        stats.trace_candidate_energy_share = replacement_density
        stats.trace_victim_energy_share = evicted_density
        return stats

    @torch.no_grad()
    def _update_all_candidates_append_only_density(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Append sparse candidates without ever replacing an existing KV."""
        if self.config.append_group_count_reduce == "masked_max":
            return self._update_all_candidates_append_only_masked_max(keys, values)
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError("append-only KV updates require synchronized group counts")
        active_count = int(self.counts[0].item())
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        num_candidates = int(keys_gnd.shape[1])

        if active_count == 0:
            candidate_density = torch.zeros(
                self.groups,
                num_candidates,
                device=self.device,
                dtype=torch.float32,
            )
            nearest_index = torch.zeros(
                self.groups,
                num_candidates,
                device=self.device,
                dtype=torch.long,
            )
            saturation_ratio = torch.zeros_like(candidate_density)
            reference_baseline = torch.zeros_like(candidate_density)
            admissible_count = torch.full(
                (self.groups,),
                num_candidates,
                device=self.device,
                dtype=torch.long,
            )
        else:
            density_chunks = []
            nearest_chunks = []
            ratio_chunks = []
            baseline_chunks = []
            old_keys = self.keys[:, :active_count]
            old_baseline = self.density_baseline[:, :active_count]
            for start in range(0, num_candidates, self.config.update_chunk_size):
                end = min(start + self.config.update_chunk_size, num_candidates)
                distances = self._squared_l2(keys_gnd[:, start:end], old_keys)
                density = self._density_contribution(distances).sum(dim=-1)
                nearest = distances.argmin(dim=-1)
                baseline = torch.gather(old_baseline, 1, nearest).clamp_min(
                    self.config.append_density_baseline_floor
                )
                density_chunks.append(density)
                nearest_chunks.append(nearest)
                ratio_chunks.append(density / baseline)
                baseline_chunks.append(baseline)
            candidate_density = torch.cat(density_chunks, dim=1)
            nearest_index = torch.cat(nearest_chunks, dim=1)
            saturation_ratio = torch.cat(ratio_chunks, dim=1)
            reference_baseline = torch.cat(baseline_chunks, dim=1)
            admissible_count = (
                saturation_ratio < self.config.append_density_growth_limit
            ).sum(dim=1)

        if self.config.append_max_entries > 0:
            remaining = max(self.config.append_max_entries - active_count, 0)
            admissible_count.clamp_max_(remaining)

        # Attention requires every head/group to expose the same token count.
        # The strictest group determines the shared append length.
        num_accepted = int(admissible_count.min().item())
        candidate_order = torch.argsort(
            saturation_ratio,
            dim=1,
            descending=False,
            stable=True,
        )
        ordered_density = torch.gather(candidate_density, 1, candidate_order)
        ordered_ratio = torch.gather(saturation_ratio, 1, candidate_order)
        ordered_anchor = torch.gather(nearest_index, 1, candidate_order)
        ordered_reference = torch.gather(reference_baseline, 1, candidate_order)
        if active_count == 0:
            ordered_anchor = torch.full_like(ordered_anchor, -1)
        target_slot = torch.full_like(candidate_order, -1)
        accepted_entries = torch.zeros_like(candidate_order, dtype=torch.bool)
        replacement_density = ordered_density.clone()

        if num_accepted > 0:
            selected_indices = candidate_order[:, :num_accepted]
            selected_keys = self._gather_rows(keys_gnd, selected_indices).contiguous()
            selected_values = self._gather_rows(values_gnd, selected_indices).contiguous()
            selected_to_old_density = ordered_density[:, :num_accepted]

            if active_count > 0:
                selected_nearest = torch.gather(
                    nearest_index,
                    1,
                    selected_indices,
                )
                inherited_baseline = torch.gather(
                    self.density_baseline[:, :active_count],
                    1,
                    selected_nearest,
                ).clamp_min(self.config.append_density_baseline_floor)
                commit_chunk = min(256, num_accepted)
                for start in range(0, num_accepted, commit_chunk):
                    end = min(start + commit_chunk, num_accepted)
                    contribution = self._density_contribution(
                        self._squared_l2(
                            selected_keys[:, start:end],
                            self.keys[:, :active_count],
                        )
                    )
                    self.density[:, :active_count].add_(contribution.sum(dim=1))
                    del contribution
            else:
                inherited_baseline = None

            new_internal_density = torch.zeros(
                self.groups,
                num_accepted,
                device=self.device,
                dtype=torch.float32,
            )
            internal_chunk = min(512, num_accepted)
            for start in range(0, num_accepted, internal_chunk):
                end = min(start + internal_chunk, num_accepted)
                contribution = self._density_contribution(
                    self._squared_l2(
                        selected_keys[:, start:end],
                        selected_keys,
                    )
                )
                diagonal = torch.arange(end - start, device=self.device)
                contribution[:, diagonal, diagonal + start] = 0.0
                new_internal_density[:, start:end] = contribution.sum(dim=-1)
                del contribution

            new_density = selected_to_old_density + new_internal_density
            if inherited_baseline is None:
                new_baseline = new_density.clamp_min(
                    self.config.append_density_baseline_floor
                )
            else:
                new_baseline = inherited_baseline

            end_count = active_count + num_accepted
            self._ensure_storage_capacity(end_count)
            self.keys[:, active_count:end_count].copy_(selected_keys)
            self.values[:, active_count:end_count].copy_(selected_values)
            self.density[:, active_count:end_count].copy_(new_density)
            self.density_baseline[:, active_count:end_count].copy_(new_baseline)
            self.counts.fill_(end_count)

            slots = torch.arange(
                active_count,
                end_count,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(self.groups, -1)
            target_slot[:, :num_accepted] = slots
            accepted_entries[:, :num_accepted] = True
            replacement_density[:, :num_accepted] = new_density

        accepted_count = torch.full(
            (self.groups,),
            num_accepted,
            device=self.device,
            dtype=torch.int32,
        )
        accepted = accepted_count > 0
        stats = DensityKVBankStats(
            accepted=accepted,
            added=accepted,
            replaced=torch.zeros_like(accepted),
            rejected_full=torch.zeros_like(accepted),
            candidate_index=candidate_order,
            target_slot=target_slot,
            candidate_density=ordered_density,
            replacement_density=replacement_density,
            evicted_density=torch.zeros_like(ordered_density),
            energy_delta=(
                ordered_ratio - self.config.append_density_growth_limit
            ),
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.arange(
            num_candidates, device=self.device, dtype=torch.int32
        ).unsqueeze(0).expand(self.groups, -1)
        stats.trace_entry_reason = torch.full_like(
            candidate_order, 4, dtype=torch.int8
        )
        stats.trace_anchor_slot = ordered_anchor
        stats.trace_reference_density = ordered_reference
        stats.trace_candidate_energy_share = ordered_ratio
        stats.trace_victim_energy_share = torch.full_like(
            ordered_ratio, self.config.append_density_growth_limit
        )
        return stats

    @torch.no_grad()
    def _update_all_candidates_append_only_masked_max(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Append each group's admissible entries and leave shorter tails masked."""
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        num_candidates = int(keys_gnd.shape[1])
        active_count = int(self.counts.max().item())
        old_mask = self.active_mask()[:, :active_count]

        if active_count == 0:
            candidate_density = torch.zeros(
                self.groups,
                num_candidates,
                device=self.device,
                dtype=torch.float32,
            )
            nearest_index = torch.zeros(
                self.groups,
                num_candidates,
                device=self.device,
                dtype=torch.long,
            )
            saturation_ratio = torch.zeros_like(candidate_density)
            reference_baseline = torch.zeros_like(candidate_density)
            admissible_count = torch.full(
                (self.groups,),
                num_candidates,
                device=self.device,
                dtype=torch.long,
            )
        else:
            density_chunks = []
            nearest_chunks = []
            ratio_chunks = []
            baseline_chunks = []
            old_keys = self.keys[:, :active_count]
            old_baseline = self.density_baseline[:, :active_count]
            old_mask_expanded = old_mask.unsqueeze(1)
            for start in range(0, num_candidates, self.config.update_chunk_size):
                end = min(start + self.config.update_chunk_size, num_candidates)
                distances = self._squared_l2(keys_gnd[:, start:end], old_keys)
                density = (
                    self._density_contribution(distances) * old_mask_expanded
                ).sum(dim=-1)
                nearest = distances.masked_fill(
                    ~old_mask_expanded,
                    torch.inf,
                ).argmin(dim=-1)
                baseline = torch.gather(old_baseline, 1, nearest).clamp_min(
                    self.config.append_density_baseline_floor
                )
                density_chunks.append(density)
                nearest_chunks.append(nearest)
                ratio_chunks.append(density / baseline)
                baseline_chunks.append(baseline)
            candidate_density = torch.cat(density_chunks, dim=1)
            nearest_index = torch.cat(nearest_chunks, dim=1)
            saturation_ratio = torch.cat(ratio_chunks, dim=1)
            reference_baseline = torch.cat(baseline_chunks, dim=1)
            admissible_count = (
                saturation_ratio < self.config.append_density_growth_limit
            ).sum(dim=1)

        if self.config.append_max_entries > 0:
            remaining = (
                self.config.append_max_entries - self.counts.long()
            ).clamp_min_(0)
            admissible_count = torch.minimum(admissible_count, remaining)

        candidate_order = torch.argsort(
            saturation_ratio,
            dim=1,
            descending=False,
            stable=True,
        )
        ordered_density = torch.gather(candidate_density, 1, candidate_order)
        ordered_ratio = torch.gather(saturation_ratio, 1, candidate_order)
        ordered_anchor = torch.gather(nearest_index, 1, candidate_order)
        ordered_reference = torch.gather(reference_baseline, 1, candidate_order)
        if active_count == 0:
            ordered_anchor = torch.full_like(ordered_anchor, -1)
        target_slot = torch.full_like(candidate_order, -1)
        accepted_entries = torch.zeros_like(candidate_order, dtype=torch.bool)
        replacement_density = ordered_density.clone()
        max_accepted = int(admissible_count.max().item())

        if max_accepted > 0:
            positions = torch.arange(max_accepted, device=self.device).unsqueeze(0)
            selected_valid = positions < admissible_count.unsqueeze(1)
            selected_indices = candidate_order[:, :max_accepted]
            selected_keys = self._gather_rows(keys_gnd, selected_indices).contiguous()
            selected_values = self._gather_rows(values_gnd, selected_indices).contiguous()
            selected_to_old_density = ordered_density[:, :max_accepted]

            if active_count > 0:
                selected_nearest = torch.gather(
                    nearest_index,
                    1,
                    selected_indices,
                )
                inherited_baseline = torch.gather(
                    self.density_baseline[:, :active_count],
                    1,
                    selected_nearest,
                ).clamp_min(self.config.append_density_baseline_floor)
                commit_chunk = min(256, max_accepted)
                for start in range(0, max_accepted, commit_chunk):
                    end = min(start + commit_chunk, max_accepted)
                    contribution = self._density_contribution(
                        self._squared_l2(
                            selected_keys[:, start:end],
                            self.keys[:, :active_count],
                        )
                    )
                    contribution.mul_(selected_valid[:, start:end].unsqueeze(-1))
                    contribution.mul_(old_mask.unsqueeze(1))
                    self.density[:, :active_count].add_(contribution.sum(dim=1))
                    del contribution
            else:
                inherited_baseline = None

            new_internal_density = torch.zeros(
                self.groups,
                max_accepted,
                device=self.device,
                dtype=torch.float32,
            )
            internal_chunk = min(512, max_accepted)
            for start in range(0, max_accepted, internal_chunk):
                end = min(start + internal_chunk, max_accepted)
                contribution = self._density_contribution(
                    self._squared_l2(selected_keys[:, start:end], selected_keys)
                )
                pair_valid = (
                    selected_valid[:, start:end].unsqueeze(-1)
                    & selected_valid.unsqueeze(1)
                )
                contribution.mul_(pair_valid)
                diagonal = torch.arange(end - start, device=self.device)
                contribution[:, diagonal, diagonal + start] = 0.0
                new_internal_density[:, start:end] = contribution.sum(dim=-1)
                del contribution
            new_density = selected_to_old_density + new_internal_density
            if inherited_baseline is None:
                new_baseline = new_density.clamp_min(
                    self.config.append_density_baseline_floor
                )
            else:
                new_baseline = inherited_baseline

            old_counts = self.counts.long()
            end_counts = old_counts + admissible_count
            self._ensure_storage_capacity(int(end_counts.max().item()))
            selected_slots = old_counts.unsqueeze(1) + positions
            groups = torch.arange(self.groups, device=self.device).unsqueeze(1)
            groups = groups.expand_as(selected_slots)
            self.keys[groups[selected_valid], selected_slots[selected_valid]] = (
                selected_keys[selected_valid]
            )
            self.values[groups[selected_valid], selected_slots[selected_valid]] = (
                selected_values[selected_valid]
            )
            self.density[groups[selected_valid], selected_slots[selected_valid]] = (
                new_density[selected_valid]
            )
            self.density_baseline[
                groups[selected_valid], selected_slots[selected_valid]
            ] = new_baseline[selected_valid]
            self.counts.copy_(end_counts.to(torch.int32))
            target_slot[:, :max_accepted] = torch.where(
                selected_valid,
                selected_slots,
                torch.full_like(selected_slots, -1),
            )
            accepted_entries[:, :max_accepted] = selected_valid
            replacement_density[:, :max_accepted] = torch.where(
                selected_valid,
                new_density,
                replacement_density[:, :max_accepted],
            )

        accepted_count = admissible_count.to(torch.int32)
        accepted = accepted_count > 0
        stats = DensityKVBankStats(
            accepted=accepted,
            added=accepted,
            replaced=torch.zeros_like(accepted),
            rejected_full=torch.zeros_like(accepted),
            candidate_index=candidate_order,
            target_slot=target_slot,
            candidate_density=ordered_density,
            replacement_density=replacement_density,
            evicted_density=torch.zeros_like(ordered_density),
            energy_delta=(
                ordered_ratio - self.config.append_density_growth_limit
            ),
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.arange(
            num_candidates, device=self.device, dtype=torch.int32
        ).unsqueeze(0).expand(self.groups, -1)
        stats.trace_entry_reason = torch.full_like(
            candidate_order, 4, dtype=torch.int8
        )
        stats.trace_anchor_slot = ordered_anchor
        stats.trace_reference_density = ordered_reference
        stats.trace_candidate_energy_share = ordered_ratio
        stats.trace_victim_energy_share = torch.full_like(
            ordered_ratio, self.config.append_density_growth_limit
        )
        return stats

    @torch.no_grad()
    def _density_over_union(self, keys_gld: torch.Tensor) -> torch.Tensor:
        """Compute exact per-point density without materializing L x L."""
        length = int(keys_gld.shape[1])
        result = torch.empty(
            self.groups,
            length,
            device=self.device,
            dtype=torch.float32,
        )
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        chunk_size = min(work_chunk_size, length)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            distances = self._squared_l2(keys_gld[:, start:end], keys_gld)
            density = self._density_sum(distances)
            rows = torch.arange(end - start, device=self.device)
            self_distance = distances[:, rows, rows + start]
            result[:, start:end] = density - self._density_contribution(self_distance)
        return result.clamp_min_(0.0)

    @torch.no_grad()
    def _density_over_union_with_candidate_nearest(
        self,
        keys_gld: torch.Tensor,
        *,
        old_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute union density and each candidate's nearest old entry.

        Rows are streamed, so the full pairwise matrix is never materialized.
        The diagonal contribution is explicitly masked before reduction.
        """
        length = int(keys_gld.shape[1])
        if old_count < 0 or old_count > length:
            raise ValueError("old_count must lie inside the union")
        num_candidates = length - old_count
        density = torch.empty(
            self.groups,
            length,
            device=self.device,
            dtype=torch.float32,
        )
        nearest_old = torch.full(
            (self.groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        chunk_size = min(work_chunk_size, length)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            distances = self._squared_l2(keys_gld[:, start:end], keys_gld)
            contribution = self._density_contribution(distances)
            diagonal = torch.arange(end - start, device=self.device)
            contribution[:, diagonal, diagonal + start] = 0.0
            density[:, start:end] = contribution.sum(dim=-1)
            candidate_start = max(start, old_count)
            if old_count > 0 and candidate_start < end:
                local_start = candidate_start - start
                nearest_old[
                    :,
                    candidate_start - old_count : end - old_count,
                ] = distances[:, local_start:, :old_count].argmin(dim=-1)
            del distances, contribution
        return density.clamp_min_(0.0), nearest_old

    @torch.no_grad()
    def _bootstrap_v2_union_density(
        self,
        candidate_keys_gnd: torch.Tensor,
        *,
        active_count: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Extend exact old density and expose current-to-old contributions."""
        candidate_internal, _ = (
            self._density_over_union_with_candidate_nearest(
                candidate_keys_gnd,
                old_count=0,
            )
        )
        num_candidates = int(candidate_keys_gnd.shape[1])
        nearest_old = torch.full(
            (self.groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        if active_count == 0:
            return (
                self.density[:, :0].clone(),
                candidate_internal,
                torch.zeros_like(candidate_internal),
                nearest_old,
            )

        old_union_density = self.density[:, :active_count].clone()
        candidate_external = torch.empty_like(candidate_internal)
        old_keys = self.keys[:, :active_count]
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        for start in range(0, num_candidates, work_chunk_size):
            end = min(start + work_chunk_size, num_candidates)
            distances = self._squared_l2(
                candidate_keys_gnd[:, start:end],
                old_keys,
            )
            contribution = self._density_contribution(distances)
            candidate_external[:, start:end] = contribution.sum(dim=-1)
            old_union_density.add_(contribution.sum(dim=1))
            nearest_old[:, start:end] = distances.argmin(dim=-1)
            del distances, contribution
        return (
            old_union_density,
            candidate_external + candidate_internal,
            candidate_external,
            nearest_old,
        )

    @torch.no_grad()
    def _bootstrap3_all_anchor_candidate_order(
        self,
        candidate_keys_gnd: torch.Tensor,
        *,
        active_count: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        """Choose the largest prefix whose violating anchors fit eviction budget.

        ``self.density`` is the exact old-to-old sum from the previous commit,
        while ``self.density_baseline`` is frozen at each point's insertion.
        Only candidate-to-old contributions are recomputed. The pairwise
        old-to-old matrix is never stored or rebuilt. Any old point that would
        cross the growth limit is a mandatory victim; remaining victim slots
        are assigned to the densest old points.
        """
        groups, num_candidates, _ = candidate_keys_gnd.shape
        candidate_order = torch.arange(
            num_candidates,
            device=self.device,
        ).unsqueeze(0).expand(groups, -1)
        candidate_external = torch.zeros(
            groups,
            num_candidates,
            device=self.device,
            dtype=torch.float32,
        )
        candidate_score = torch.zeros_like(candidate_external)
        old_increment = torch.zeros(
            groups,
            active_count,
            device=self.device,
            dtype=torch.float32,
        )
        max_accepted = min(num_candidates, self.config.max_entries)
        if active_count == 0 or max_accepted == 0:
            return (
                candidate_order,
                candidate_external,
                candidate_score,
                max_accepted,
                old_increment,
            )

        old_keys = self.keys[:, :active_count]
        old_density = self.density[:, :active_count]
        old_baseline = self.density_baseline[:, :active_count].clamp_min(
            self.config.append_density_baseline_floor
        )
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size

        for start in range(0, num_candidates, work_chunk_size):
            end = min(start + work_chunk_size, num_candidates)
            contribution = self._density_contribution(
                self._squared_l2(
                    candidate_keys_gnd[:, start:end],
                    old_keys,
                )
            )
            candidate_external[:, start:end] = contribution.sum(dim=-1)
            candidate_score[:, start:end] = (
                contribution / old_baseline.unsqueeze(1)
            ).amax(dim=-1)
            del contribution

        candidate_order = torch.argsort(
            candidate_score,
            dim=1,
            descending=False,
            stable=True,
        )
        ordered_keys = self._gather_rows(
            candidate_keys_gnd,
            candidate_order[:, :max_accepted],
        ).contiguous()
        accepted_count = 0
        best_increment = old_increment
        processed = 0
        growth_limit = self.config.legacy_bootstrap_density_limit
        while processed < max_accepted:
            end = min(processed + work_chunk_size, max_accepted)
            chunk_count = end - processed
            contribution = self._density_contribution(
                self._squared_l2(
                    ordered_keys[:, processed:end],
                    old_keys,
                )
            )
            cumulative = old_increment.unsqueeze(1) + contribution.cumsum(dim=1)
            feasible = (
                (
                    old_density.unsqueeze(1) + cumulative
                )
                / old_baseline.unsqueeze(1)
                >= growth_limit
            ).sum(dim=-1)
            prefix_count = torch.arange(
                processed + 1,
                end + 1,
                device=self.device,
            )
            eviction_budget = (
                active_count + prefix_count - self.config.max_entries
            ).clamp_min(0)
            feasible = (
                feasible <= eviction_budget.unsqueeze(0)
            ).all(dim=0)
            feasible_positions = feasible.nonzero(as_tuple=False)
            if feasible_positions.numel() > 0:
                best_position = int(feasible_positions[-1, 0].item())
                accepted_count = processed + best_position + 1
                best_increment = cumulative[:, best_position].clone()
            if chunk_count > 0:
                old_increment = cumulative[:, -1].clone()
            processed = end
            del contribution, cumulative
        return (
            candidate_order,
            candidate_external,
            candidate_score,
            accepted_count,
            best_increment,
        )

    @torch.no_grad()
    def _density_over_masked_union(
        self,
        keys_gld: torch.Tensor,
        valid_gn: torch.Tensor,
    ) -> torch.Tensor:
        """Compute density for a per-group masked union in bounded memory."""
        if valid_gn.shape != keys_gld.shape[:2]:
            raise ValueError("masked union validity must match [G,N]")
        length = int(keys_gld.shape[1])
        result = torch.full(
            (self.groups, length),
            torch.inf,
            device=self.device,
            dtype=torch.float32,
        )
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        chunk_size = min(work_chunk_size, length)
        source_valid = valid_gn.unsqueeze(1)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            contribution = self._density_contribution(
                self._squared_l2(keys_gld[:, start:end], keys_gld)
            )
            contribution.mul_(source_valid)
            diagonal = torch.arange(end - start, device=self.device)
            contribution[:, diagonal, diagonal + start] = 0.0
            density = contribution.sum(dim=-1)
            result[:, start:end] = torch.where(
                valid_gn[:, start:end],
                density,
                torch.full_like(density, torch.inf),
            )
            del contribution
        return result

    @torch.no_grad()
    def _update_all_candidates_gated_union_prune_density(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Gate against the frozen bank, then prune the admitted union globally."""
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        num_candidates = int(keys_gnd.shape[1])
        old_counts = self.counts.long().clone()
        active_count = int(self.counts.max().item())
        old_valid = self.active_mask()[:, :active_count]

        if active_count == 0:
            candidate_density = torch.zeros(
                self.groups,
                num_candidates,
                device=self.device,
                dtype=torch.float32,
            )
            nearest_index = torch.zeros(
                self.groups,
                num_candidates,
                device=self.device,
                dtype=torch.long,
            )
            reference_baseline = torch.zeros_like(candidate_density)
            saturation_ratio = torch.zeros_like(candidate_density)
            gate_accepted = torch.ones_like(candidate_density, dtype=torch.bool)
            candidate_baseline = torch.zeros_like(candidate_density)
        else:
            density_chunks = []
            nearest_chunks = []
            baseline_chunks = []
            old_keys = self.keys[:, :active_count]
            old_baseline = self.density_baseline[:, :active_count]
            old_valid_expanded = old_valid.unsqueeze(1)
            for start in range(0, num_candidates, self.config.update_chunk_size):
                end = min(start + self.config.update_chunk_size, num_candidates)
                distances = self._squared_l2(keys_gnd[:, start:end], old_keys)
                contribution = self._density_contribution(distances)
                density = (contribution * old_valid_expanded).sum(dim=-1)
                nearest = distances.masked_fill(
                    ~old_valid_expanded,
                    torch.inf,
                ).argmin(dim=-1)
                baseline = torch.gather(old_baseline, 1, nearest).clamp_min(
                    self.config.append_density_baseline_floor
                )
                density_chunks.append(density)
                nearest_chunks.append(nearest)
                baseline_chunks.append(baseline)
                del distances, contribution
            candidate_density = torch.cat(density_chunks, dim=1)
            nearest_index = torch.cat(nearest_chunks, dim=1)
            reference_baseline = torch.cat(baseline_chunks, dim=1)
            saturation_ratio = candidate_density / reference_baseline
            gate_accepted = (
                saturation_ratio < self.config.append_density_growth_limit
            )
            candidate_baseline = reference_baseline

        union_keys = torch.cat((self.keys[:, :active_count], keys_gnd), dim=1)
        union_values = torch.cat((self.values[:, :active_count], values_gnd), dim=1)
        union_valid = torch.cat((old_valid, gate_accepted), dim=1)
        union_baseline = torch.cat(
            (self.density_baseline[:, :active_count], candidate_baseline), dim=1
        )
        candidate_metadata = torch.full(
            (self.groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        union_source = torch.cat(
            (self.source_index[:, :active_count], candidate_metadata), dim=1
        )
        union_insert = torch.cat(
            (
                self.insert_query_frame[:, :active_count],
                candidate_metadata.to(dtype=self.insert_query_frame.dtype),
            ),
            dim=1,
        )
        union_length = int(union_keys.shape[1])
        valid_counts = union_valid.sum(dim=1, dtype=torch.long)
        final_counts = valid_counts.clamp_max(self.config.max_entries)
        max_final_count = int(final_counts.max().item())
        if max_final_count <= 0:
            raise RuntimeError("gated union unexpectedly rejected an empty initial bank")

        union_density = self._density_over_masked_union(union_keys, union_valid)
        selected = torch.topk(
            union_density,
            k=max_final_count,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        selected_valid = (
            torch.arange(max_final_count, device=self.device).unsqueeze(0)
            < final_counts.unsqueeze(1)
        )
        keep_mask = torch.zeros(
            self.groups,
            union_length,
            device=self.device,
            dtype=torch.bool,
        )
        groups = torch.arange(self.groups, device=self.device).unsqueeze(1)
        groups = groups.expand_as(selected)
        keep_mask[groups[selected_valid], selected[selected_valid]] = True

        all_indices = torch.arange(union_length, device=self.device).unsqueeze(0)
        all_indices = all_indices.expand(self.groups, -1)
        packed_keep = all_indices.masked_fill(
            ~keep_mask, union_length
        ).sort(dim=1).values
        keep_indices = packed_keep[:, :max_final_count]
        safe_keep = keep_indices.clamp_max(union_length - 1)
        final_valid = keep_indices < union_length
        kept_keys = self._gather_rows(union_keys, safe_keep).contiguous()
        kept_values = self._gather_rows(union_values, safe_keep).contiguous()
        final_density = torch.gather(union_density, 1, safe_keep)

        evict_mask = union_valid & ~keep_mask
        evicted_counts = evict_mask.sum(dim=1, dtype=torch.long)
        max_evicted = int(evicted_counts.max().item())
        if max_evicted > 0:
            packed_evict = all_indices.masked_fill(
                ~evict_mask, union_length
            ).sort(dim=1).values
            evicted_indices = packed_evict[:, :max_evicted]
            safe_evict = evicted_indices.clamp_max(union_length - 1)
            evicted_valid = evicted_indices < union_length
            evicted_keys = self._gather_rows(union_keys, safe_evict).contiguous()
            work_chunk_size = self.config.union_work_chunk_size
            if work_chunk_size < 0:
                work_chunk_size = self.config.update_chunk_size
            for start in range(0, max_final_count, work_chunk_size):
                end = min(start + work_chunk_size, max_final_count)
                contribution = self._density_contribution(
                    self._squared_l2(kept_keys[:, start:end], evicted_keys)
                )
                contribution.mul_(evicted_valid.unsqueeze(1))
                final_density[:, start:end].sub_(contribution.sum(dim=-1))
                del contribution
        final_density = torch.where(
            final_valid,
            final_density.clamp_min_(0.0),
            torch.zeros_like(final_density),
        )

        kept_baseline = torch.gather(union_baseline, 1, safe_keep)
        if active_count == 0:
            kept_baseline = final_density.clamp_min(
                self.config.append_density_baseline_floor
            )
        kept_baseline = torch.where(
            final_valid,
            kept_baseline,
            torch.zeros_like(kept_baseline),
        )
        kept_source = torch.gather(union_source, 1, safe_keep)
        kept_insert = torch.gather(union_insert, 1, safe_keep)
        kept_source = torch.where(
            final_valid, kept_source, torch.full_like(kept_source, -1)
        )
        kept_insert = torch.where(
            final_valid, kept_insert, torch.full_like(kept_insert, -1)
        )

        self.keys.zero_()
        self.values.zero_()
        self.density.zero_()
        self.density_baseline.zero_()
        self.source_index.fill_(-1)
        self.insert_query_frame.fill_(-1)
        self.keys[:, :max_final_count].copy_(
            torch.where(
                final_valid.unsqueeze(-1), kept_keys, torch.zeros_like(kept_keys)
            )
        )
        self.values[:, :max_final_count].copy_(
            torch.where(
                final_valid.unsqueeze(-1), kept_values, torch.zeros_like(kept_values)
            )
        )
        self.density[:, :max_final_count].copy_(final_density)
        self.density_baseline[:, :max_final_count].copy_(kept_baseline)
        self.source_index[:, :max_final_count].copy_(kept_source)
        self.insert_query_frame[:, :max_final_count].copy_(kept_insert)
        self.counts.copy_(final_counts.to(torch.int32))

        candidate_target_slot = torch.full(
            (self.groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        resulting_slots = torch.arange(max_final_count, device=self.device).unsqueeze(0)
        resulting_slots = resulting_slots.expand(self.groups, -1)
        kept_candidate = final_valid & (keep_indices >= active_count)
        kept_candidate_ids = keep_indices - active_count
        candidate_target_slot[
            groups[kept_candidate], kept_candidate_ids[kept_candidate]
        ] = resulting_slots[kept_candidate]
        accepted_entries = candidate_target_slot >= 0
        accepted_count = accepted_entries.sum(dim=1, dtype=torch.int32)
        accepted = accepted_count > 0
        safe_target = candidate_target_slot.clamp_min(0)
        replacement_density = torch.gather(final_density, 1, safe_target)
        replacement_density = torch.where(
            accepted_entries,
            replacement_density,
            torch.zeros_like(replacement_density),
        )
        old_evicted = (
            evict_mask[:, :active_count].sum(dim=1, dtype=torch.int32)
            if active_count > 0
            else torch.zeros(self.groups, device=self.device, dtype=torch.int32)
        )
        candidate_order = torch.arange(
            num_candidates, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(self.groups, -1)
        stats = DensityKVBankStats(
            accepted=accepted,
            added=final_counts > old_counts,
            replaced=old_evicted > 0,
            rejected_full=~accepted,
            candidate_index=candidate_order,
            target_slot=candidate_target_slot,
            candidate_density=candidate_density,
            replacement_density=replacement_density,
            evicted_density=(
                reference_baseline * self.config.append_density_growth_limit
            ),
            energy_delta=saturation_ratio - self.config.append_density_growth_limit,
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = candidate_order.to(torch.int32)
        stats.trace_entry_reason = torch.where(
            accepted_entries,
            torch.zeros_like(candidate_order, dtype=torch.int8),
            torch.where(
                gate_accepted,
                torch.full_like(candidate_order, 7, dtype=torch.int8),
                torch.full_like(candidate_order, 6, dtype=torch.int8),
            ),
        )
        stats.trace_anchor_slot = (
            nearest_index
            if active_count > 0
            else torch.full_like(nearest_index, -1)
        )
        stats.trace_reference_density = reference_baseline
        stats.trace_candidate_energy_share = saturation_ratio
        stats.trace_victim_energy_share = torch.full_like(
            saturation_ratio, self.config.append_density_growth_limit
        )
        stats.trace_gate_accepted = gate_accepted
        return stats

    @torch.no_grad()
    def _update_all_candidates_union_prune_density(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Append the full clean chunk, then globally prune the densest overflow."""
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError("union-prune updates require synchronized group counts")
        active_count = int(self.counts[0].item())
        num_candidates = int(keys.shape[0])
        free_entries = max(self.config.max_entries - active_count, 0)
        if num_candidates <= free_entries:
            return self._update_batch(
                keys,
                values,
                active_count=active_count,
                counts_synchronized=True,
                admission_limit=num_candidates,
            )

        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        union_keys = torch.cat((self.keys[:, :active_count], keys_gnd), dim=1)
        union_values = torch.cat((self.values[:, :active_count], values_gnd), dim=1)
        union_length = int(union_keys.shape[1])
        capacity = self.config.max_entries
        overflow = union_length - capacity
        if overflow <= 0:
            raise RuntimeError("union-prune overflow must be positive")

        union_density = self._density_over_union(union_keys)
        keep_unsorted = torch.topk(
            union_density,
            k=capacity,
            dim=1,
            largest=False,
            sorted=False,
        ).indices
        keep_mask = torch.zeros(
            self.groups,
            union_length,
            device=self.device,
            dtype=torch.bool,
        )
        keep_mask.scatter_(1, keep_unsorted, True)
        all_indices = torch.arange(union_length, device=self.device).unsqueeze(0)
        all_indices = all_indices.expand(self.groups, -1)
        keep_indices = keep_unsorted.sort(dim=1).values
        evicted_indices = all_indices.masked_select(~keep_mask).view(
            self.groups, overflow
        )

        kept_keys = self._gather_rows(union_keys, keep_indices).contiguous()
        kept_values = self._gather_rows(union_values, keep_indices).contiguous()
        evicted_keys = self._gather_rows(union_keys, evicted_indices).contiguous()
        final_density = torch.gather(union_density, 1, keep_indices)
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        chunk_size = min(work_chunk_size, capacity)
        for start in range(0, capacity, chunk_size):
            end = min(start + chunk_size, capacity)
            distances = self._squared_l2(kept_keys[:, start:end], evicted_keys)
            final_density[:, start:end] -= self._density_sum(distances)
        final_density.clamp_min_(0.0)

        candidate_target_slot = torch.full(
            (self.groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        resulting_slots = torch.arange(capacity, device=self.device).unsqueeze(0)
        resulting_slots = resulting_slots.expand(self.groups, -1)
        kept_candidate = keep_indices >= active_count
        kept_candidate_ids = keep_indices - active_count
        groups = torch.arange(self.groups, device=self.device).unsqueeze(1)
        groups = groups.expand_as(keep_indices)
        candidate_target_slot[
            groups[kept_candidate], kept_candidate_ids[kept_candidate]
        ] = resulting_slots[kept_candidate]
        accepted_entries = candidate_target_slot >= 0
        accepted_count = accepted_entries.sum(dim=1, dtype=torch.int32)
        accepted = accepted_count > 0
        safe_target = candidate_target_slot.clamp_min(0)
        replacement_density = torch.gather(final_density, 1, safe_target)
        replacement_density = torch.where(
            accepted_entries,
            replacement_density,
            torch.zeros_like(replacement_density),
        )
        candidate_density = union_density[:, active_count:]
        keep_threshold = torch.gather(union_density, 1, keep_indices).max(
            dim=1, keepdim=True
        ).values

        old_source = self.source_index[:, :active_count]
        old_insert = self.insert_query_frame[:, :active_count]
        candidate_metadata = torch.full(
            (self.groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        union_source = torch.cat((old_source, candidate_metadata), dim=1)
        union_insert = torch.cat(
            (old_insert, candidate_metadata.to(dtype=old_insert.dtype)), dim=1
        )
        kept_source = torch.gather(union_source, 1, keep_indices)
        kept_insert = torch.gather(union_insert, 1, keep_indices)

        self.keys.copy_(kept_keys)
        self.values.copy_(kept_values)
        self.density.copy_(final_density)
        self.density_baseline.zero_()
        self.counts.fill_(capacity)
        self.source_index.copy_(kept_source)
        self.insert_query_frame.copy_(kept_insert)

        candidate_order = torch.arange(
            num_candidates, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(self.groups, -1)
        stats = DensityKVBankStats(
            accepted=accepted,
            added=torch.zeros_like(accepted),
            replaced=accepted,
            rejected_full=~accepted,
            candidate_index=candidate_order,
            target_slot=candidate_target_slot,
            candidate_density=candidate_density,
            replacement_density=replacement_density,
            evicted_density=torch.zeros_like(candidate_density),
            energy_delta=candidate_density - keep_threshold,
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros_like(candidate_order)
        stats.trace_group_reason = torch.full(
            (self.groups, 1), 5, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (self.groups, 1), num_candidates, device=self.device, dtype=torch.int32
        )
        stats.trace_group_added_energy = candidate_density.sum(
            dim=1, keepdim=True
        )
        stats.trace_group_removed_energy = torch.zeros_like(
            stats.trace_group_added_energy
        )
        stats.trace_candidate_energy_share = candidate_density
        stats.trace_victim_energy_share = keep_threshold.expand_as(candidate_density)
        return stats

    @torch.no_grad()
    def _update_legacy_single_candidate(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Apply one full-bank legacy decision without batch-only overhead."""
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        distances = self._squared_l2(keys_gnd, self.keys)
        candidate_density = self._density_sum(distances).squeeze(1)
        target_density, target_slot = self.density.max(dim=1)
        groups = torch.arange(self.groups, device=self.device)
        selected_keys = keys_gnd[:, 0].contiguous()
        selected_values = values_gnd[:, 0].contiguous()
        selected_distances = distances[:, 0].contiguous()
        target_contribution = self._density_contribution(
            selected_distances[groups, target_slot]
        )
        replacement_density = (candidate_density - target_contribution).clamp_min_(0.0)
        energy_delta = replacement_density - target_density
        accepted = (
            bool(self.config.evict_densest_when_full)
            & (replacement_density < target_density * self.config.replacement_ratio)
        )

        old_keys = self.keys[groups, target_slot].unsqueeze(1).contiguous()
        old_distances = self._squared_l2(old_keys, self.keys).squeeze(1).contiguous()
        mutated = False
        if (
            self.config.fast_impl in {"auto", "triton"}
            and triton_mutate_density_kv is not None
        ):
            mutated = triton_mutate_density_kv(
                self.keys,
                self.values,
                self.density,
                self.counts,
                selected_keys,
                selected_values,
                selected_distances,
                old_distances,
                target_slot,
                accepted,
                torch.ones_like(accepted),
                replacement_density,
                density_scale=self.config.density_scale,
                riesz_power=self.config.riesz_power,
                riesz_eps=self.config.riesz_eps,
            )
        if not mutated:
            if self.config.fast_impl == "triton":
                raise RuntimeError("Triton density mutation was requested but is unavailable")
            self._mutate_torch(
                selected_keys,
                selected_values,
                selected_distances,
                old_distances,
                target_slot,
                accepted,
                torch.ones_like(accepted),
                replacement_density,
            )

        accepted_entries = accepted.unsqueeze(1)
        stats = DensityKVBankStats(
            accepted=accepted,
            added=torch.zeros_like(accepted),
            replaced=accepted,
            rejected_full=~accepted,
            candidate_index=torch.zeros(
                self.groups, 1, device=self.device, dtype=torch.long
            ),
            target_slot=target_slot.unsqueeze(1),
            candidate_density=candidate_density.unsqueeze(1),
            replacement_density=replacement_density.unsqueeze(1),
            evicted_density=target_density.unsqueeze(1),
            energy_delta=energy_delta.unsqueeze(1),
            accepted_count=accepted.to(torch.int32),
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros(
            self.groups, 1, device=self.device, dtype=torch.long
        )
        stats.trace_group_reason = torch.ones(
            self.groups, 1, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted_entries
        stats.trace_group_candidate_count = torch.ones(
            self.groups, 1, device=self.device, dtype=torch.int32
        )
        stats.trace_group_added_energy = replacement_density.unsqueeze(1)
        stats.trace_group_removed_energy = target_density.unsqueeze(1)
        stats.trace_candidate_energy_share = replacement_density.unsqueeze(1)
        stats.trace_victim_energy_share = target_density.unsqueeze(1)
        return stats

    def _legacy_normalized_group_sizes(
        self, candidate_count: int
    ) -> list[int] | None:
        if self.config.legacy_normalized_group_count < 1:
            return None
        cleanup_size = 0
        if self.config.legacy_cleanup_divisor > 0:
            alignment = self.config.legacy_cleanup_alignment
            cleanup_size = int(
                round(
                    candidate_count
                    / self.config.legacy_cleanup_divisor
                    / alignment
                )
                * alignment
            )
            cleanup_size = max(alignment, cleanup_size)
        coarse_count = candidate_count - cleanup_size
        group_count = self.config.legacy_normalized_group_count
        if coarse_count < group_count:
            raise RuntimeError(
                "normalized legacy grouping has fewer coarse candidates than groups"
            )
        base_size, larger_groups = divmod(coarse_count, group_count)
        result = [
            base_size + (group_index < larger_groups)
            for group_index in range(group_count)
        ]
        if cleanup_size:
            result.append(cleanup_size)
        if sum(result) != candidate_count:
            raise RuntimeError("normalized legacy grouping lost candidates")
        return result

    def _combine_legacy_stats(
        self,
        chunk_stats: list[DensityKVBankStats],
    ) -> DensityKVBankStats:
        if not chunk_stats:
            raise RuntimeError("cannot combine an empty legacy stats list")
        if len(chunk_stats) == 1:
            return chunk_stats[0]
        combined = DensityKVBankStats(
            accepted=torch.stack(
                [stats.accepted for stats in chunk_stats]
            ).any(dim=0),
            added=torch.stack(
                [stats.added for stats in chunk_stats]
            ).any(dim=0),
            replaced=torch.stack(
                [stats.replaced for stats in chunk_stats]
            ).any(dim=0),
            rejected_full=torch.stack(
                [stats.rejected_full for stats in chunk_stats]
            ).any(dim=0),
            candidate_index=torch.cat(
                [stats.candidate_index for stats in chunk_stats], dim=1
            ),
            target_slot=torch.cat(
                [stats.target_slot for stats in chunk_stats], dim=1
            ),
            candidate_density=torch.cat(
                [stats.candidate_density for stats in chunk_stats], dim=1
            ),
            replacement_density=torch.cat(
                [stats.replacement_density for stats in chunk_stats], dim=1
            ),
            evicted_density=torch.cat(
                [stats.evicted_density for stats in chunk_stats], dim=1
            ),
            energy_delta=torch.cat(
                [stats.energy_delta for stats in chunk_stats], dim=1
            ),
            accepted_count=torch.stack(
                [stats.accepted_count for stats in chunk_stats]
            ).sum(dim=0),
            accepted_entry_mask=torch.cat(
                [stats.accepted_entry_mask for stats in chunk_stats], dim=1
            ),
        )
        for name in (
            "trace_decision_group",
            "trace_entry_reason",
            "trace_candidate_energy_share",
            "trace_victim_energy_share",
            "trace_gate_accepted",
            "trace_saturation_ratio",
            "trace_anchor_slot",
            "trace_reference_density",
        ):
            values = [getattr(stats, name, None) for stats in chunk_stats]
            if all(value is not None for value in values):
                setattr(combined, name, torch.cat(values, dim=1))
        for name in (
            "trace_group_reason",
            "trace_group_accepted",
            "trace_group_candidate_count",
            "trace_group_added_energy",
            "trace_group_removed_energy",
        ):
            setattr(
                combined,
                name,
                torch.cat([getattr(stats, name) for stats in chunk_stats], dim=1),
            )
        return combined

    @torch.no_grad()
    def _update_legacy_density_gated_bootstrap(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Density-gate every warmup candidate instead of filling FIFO.

        The first clean commit has no historical center whose insertion density
        can serve as a local scale.  We therefore normalize each candidate's
        within-commit density by the per-group median.  Later under-capacity
        commits use the same nearest-center growth ratio as append-only density
        admission.  The strictest group determines the shared token count.
        """
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError(
                "density-gated legacy bootstrap requires synchronized group counts"
            )
        active_count = int(self.counts[0].item())
        if active_count >= self.config.max_entries:
            raise RuntimeError("density-gated bootstrap requires free bank capacity")

        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        num_candidates = int(keys_gnd.shape[1])
        floor = self.config.append_density_baseline_floor

        if active_count == 0:
            candidate_density = self._density_over_union(keys_gnd)
            reference_baseline = candidate_density.median(
                dim=1, keepdim=True
            ).values.clamp_min(floor).expand_as(candidate_density)
            saturation_ratio = candidate_density / reference_baseline
            nearest_index = torch.full(
                (self.groups, num_candidates),
                -1,
                device=self.device,
                dtype=torch.long,
            )
        else:
            density_chunks: list[torch.Tensor] = []
            nearest_chunks: list[torch.Tensor] = []
            ratio_chunks: list[torch.Tensor] = []
            baseline_chunks: list[torch.Tensor] = []
            old_keys = self.keys[:, :active_count]
            old_baseline = self.density_baseline[:, :active_count]
            for start in range(0, num_candidates, self.config.update_chunk_size):
                end = min(start + self.config.update_chunk_size, num_candidates)
                distances = self._squared_l2(keys_gnd[:, start:end], old_keys)
                density = self._density_contribution(distances).sum(dim=-1)
                nearest = distances.argmin(dim=-1)
                baseline = torch.gather(old_baseline, 1, nearest).clamp_min(floor)
                density_chunks.append(density)
                nearest_chunks.append(nearest)
                ratio_chunks.append(density / baseline)
                baseline_chunks.append(baseline)
            candidate_density = torch.cat(density_chunks, dim=1)
            nearest_index = torch.cat(nearest_chunks, dim=1)
            saturation_ratio = torch.cat(ratio_chunks, dim=1)
            reference_baseline = torch.cat(baseline_chunks, dim=1)

        admissible_count = (
            saturation_ratio < self.config.legacy_bootstrap_density_limit
        ).sum(dim=1)
        free_entries = self.config.max_entries - active_count
        admissible_count.clamp_max_(free_entries)
        num_accepted = int(admissible_count.min().item())

        candidate_order = torch.argsort(
            saturation_ratio,
            dim=1,
            descending=False,
            stable=True,
        )
        ordered_density = torch.gather(candidate_density, 1, candidate_order)
        ordered_ratio = torch.gather(saturation_ratio, 1, candidate_order)
        ordered_anchor = torch.gather(nearest_index, 1, candidate_order)
        ordered_reference = torch.gather(
            reference_baseline, 1, candidate_order
        )
        target_slot = torch.full_like(candidate_order, -1)
        accepted_entries = torch.zeros_like(candidate_order, dtype=torch.bool)
        replacement_density = ordered_density.clone()

        if num_accepted > 0:
            selected_indices = candidate_order[:, :num_accepted]
            selected_keys = self._gather_rows(
                keys_gnd, selected_indices
            ).contiguous()
            selected_values = self._gather_rows(
                values_gnd, selected_indices
            ).contiguous()

            if active_count > 0:
                selected_nearest = torch.gather(
                    nearest_index, 1, selected_indices
                )
                inherited_baseline = torch.gather(
                    self.density_baseline[:, :active_count],
                    1,
                    selected_nearest,
                ).clamp_min(floor)
                selected_external_density = torch.gather(
                    candidate_density, 1, selected_indices
                )
                commit_chunk = min(256, num_accepted)
                for start in range(0, num_accepted, commit_chunk):
                    end = min(start + commit_chunk, num_accepted)
                    contribution = self._density_contribution(
                        self._squared_l2(
                            selected_keys[:, start:end],
                            self.keys[:, :active_count],
                        )
                    )
                    self.density[:, :active_count].add_(
                        contribution.sum(dim=1)
                    )
                    del contribution
            else:
                inherited_baseline = None
                selected_external_density = torch.zeros(
                    self.groups,
                    num_accepted,
                    device=self.device,
                    dtype=torch.float32,
                )

            selected_internal_density = self._density_over_union(selected_keys)
            new_density = selected_external_density + selected_internal_density
            new_baseline = (
                new_density.clamp_min(floor)
                if inherited_baseline is None
                else inherited_baseline
            )

            end_count = active_count + num_accepted
            self.keys[:, active_count:end_count].copy_(selected_keys)
            self.values[:, active_count:end_count].copy_(selected_values)
            self.density[:, active_count:end_count].copy_(new_density)
            self.density_baseline[:, active_count:end_count].copy_(new_baseline)
            self.counts.fill_(end_count)
            if end_count >= self.config.max_entries:
                self._legacy_baseline_initialized = True

            slots = torch.arange(
                active_count,
                end_count,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(self.groups, -1)
            target_slot[:, :num_accepted] = slots
            accepted_entries[:, :num_accepted] = True
            replacement_density[:, :num_accepted] = new_density

        accepted_count = torch.full(
            (self.groups,),
            num_accepted,
            device=self.device,
            dtype=torch.int32,
        )
        accepted = accepted_count > 0
        stats = DensityKVBankStats(
            accepted=accepted,
            added=accepted,
            replaced=torch.zeros_like(accepted),
            rejected_full=torch.zeros_like(accepted),
            candidate_index=candidate_order,
            target_slot=target_slot,
            candidate_density=ordered_density,
            replacement_density=replacement_density,
            evicted_density=torch.zeros_like(ordered_density),
            energy_delta=(
                ordered_ratio - self.config.legacy_bootstrap_density_limit
            ),
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros_like(
            candidate_order, dtype=torch.int32
        )
        stats.trace_entry_reason = torch.full_like(
            candidate_order, 4, dtype=torch.int8
        )
        stats.trace_anchor_slot = ordered_anchor
        stats.trace_reference_density = ordered_reference
        stats.trace_group_reason = torch.full(
            (self.groups, 1), 4, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (self.groups, 1),
            num_candidates,
            device=self.device,
            dtype=torch.int32,
        )
        stats.trace_group_added_energy = torch.where(
            accepted,
            replacement_density[:, :num_accepted].sum(dim=1)
            if num_accepted > 0
            else torch.zeros(
                self.groups, device=self.device, dtype=torch.float32
            ),
            torch.zeros(self.groups, device=self.device, dtype=torch.float32),
        ).unsqueeze(1)
        stats.trace_group_removed_energy = torch.zeros_like(
            stats.trace_group_added_energy
        )
        stats.trace_candidate_energy_share = ordered_ratio
        stats.trace_victim_energy_share = torch.full_like(
            ordered_ratio, self.config.legacy_bootstrap_density_limit
        )
        stats.trace_gate_accepted = accepted_entries
        stats.trace_saturation_ratio = ordered_ratio
        return stats

    @torch.no_grad()
    def _update_legacy_all_anchor_growth_bootstrap(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Preserve every retained point's insertion-density growth bound."""
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError(
                "all-anchor bootstrap requires synchronized group counts"
            )
        active_count = int(self.counts[0].item())
        capacity = self.config.max_entries
        floor = self.config.append_density_baseline_floor
        growth_limit = self.config.legacy_bootstrap_density_limit
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        groups, num_candidates, _ = keys_gnd.shape
        (
            candidate_order,
            candidate_external,
            candidate_score,
            num_accepted,
            old_increment,
        ) = self._bootstrap3_all_anchor_candidate_order(
            keys_gnd,
            active_count=active_count,
        )

        selected_candidates = candidate_order[:, :num_accepted]
        selected_keys = self._gather_rows(
            keys_gnd,
            selected_candidates,
        ).contiguous()
        selected_values = self._gather_rows(
            values_gnd,
            selected_candidates,
        ).contiguous()
        selected_external = torch.gather(
            candidate_external,
            1,
            selected_candidates,
        )
        selected_internal = (
            self._density_over_union(selected_keys)
            if num_accepted > 0
            else torch.zeros(
                groups,
                0,
                device=self.device,
                dtype=torch.float32,
            )
        )
        selected_density = selected_external + selected_internal
        old_proposal_density = (
            self.density[:, :active_count] + old_increment
        )
        proposal_keys = torch.cat(
            (self.keys[:, :active_count], selected_keys),
            dim=1,
        )
        proposal_values = torch.cat(
            (self.values[:, :active_count], selected_values),
            dim=1,
        )
        proposal_density = torch.cat(
            (old_proposal_density, selected_density),
            dim=1,
        )

        free_entries = max(capacity - active_count, 0)
        evict_count = max(num_accepted - free_entries, 0)
        added_count = num_accepted - evict_count
        proposal_count = active_count + num_accepted
        final_count = proposal_count - evict_count
        keep_mask = torch.ones(
            groups,
            proposal_count,
            device=self.device,
            dtype=torch.bool,
        )
        evicted_density = torch.zeros(
            groups,
            evict_count,
            device=self.device,
            dtype=torch.float32,
        )
        evicted_old = torch.zeros(
            groups,
            evict_count,
            device=self.device,
            dtype=torch.long,
        )
        if evict_count > 0:
            old_baseline = self.density_baseline[:, :active_count].clamp_min(
                floor
            )
            must_evict = (
                old_proposal_density / old_baseline
                >= growth_limit
            )
            must_evict_count = must_evict.sum(dim=1)
            if bool((must_evict_count > evict_count).any()):
                raise RuntimeError(
                    "all-anchor bootstrap selected an infeasible prefix: "
                    f"required_evictions={must_evict_count.tolist()}, "
                    f"budget={evict_count}"
                )
            victim_score = torch.where(
                must_evict,
                torch.full_like(old_proposal_density, torch.inf),
                old_proposal_density,
            )
            evicted_density, evicted_old = torch.topk(
                victim_score,
                k=evict_count,
                dim=1,
                largest=True,
                sorted=False,
            )
            evicted_density = torch.gather(
                old_proposal_density,
                1,
                evicted_old,
            )
            keep_mask[:, :active_count].scatter_(1, evicted_old, False)

        all_indices = torch.arange(
            proposal_count,
            device=self.device,
        ).unsqueeze(0).expand(groups, -1)
        keep_indices = all_indices.masked_fill(
            ~keep_mask,
            proposal_count,
        ).sort(dim=1).values[:, :final_count]
        kept_keys = self._gather_rows(proposal_keys, keep_indices).contiguous()
        kept_values = self._gather_rows(proposal_values, keep_indices).contiguous()
        final_density = torch.gather(proposal_density, 1, keep_indices)
        if evict_count > 0:
            evicted_keys = self._gather_rows(
                self.keys[:, :active_count],
                evicted_old,
            ).contiguous()
            work_chunk_size = self.config.union_work_chunk_size
            if work_chunk_size < 0:
                work_chunk_size = self.config.update_chunk_size
            for start in range(0, final_count, work_chunk_size):
                end = min(start + work_chunk_size, final_count)
                final_density[:, start:end].sub_(
                    self._density_sum(
                        self._squared_l2(
                            kept_keys[:, start:end],
                            evicted_keys,
                        )
                    )
                )
        final_density.clamp_min_(0.0)

        proposal_baseline = torch.cat(
            (
                self.density_baseline[:, :active_count],
                torch.zeros(
                    groups,
                    num_accepted,
                    device=self.device,
                    dtype=torch.float32,
                ),
            ),
            dim=1,
        )
        kept_baseline = torch.gather(
            proposal_baseline,
            1,
            keep_indices,
        )
        kept_candidate = keep_indices >= active_count
        kept_baseline = torch.where(
            kept_candidate,
            final_density.clamp_min(floor),
            kept_baseline.clamp_min(floor),
        )
        old_kept = ~kept_candidate
        if old_kept.any():
            old_growth = final_density[old_kept] / kept_baseline[old_kept]
            tolerance = 2.0e-4 * max(growth_limit, 1.0)
            if bool((old_growth >= growth_limit + tolerance).any()):
                raise RuntimeError(
                    "all-anchor bootstrap violated a retained insertion-density "
                    f"bound: max_ratio={float(old_growth.max().item()):.8f}, "
                    f"limit={growth_limit:.8f}"
                )

        candidate_metadata = torch.full(
            (groups, num_accepted),
            -1,
            device=self.device,
            dtype=self.source_index.dtype,
        )
        proposal_source = torch.cat(
            (self.source_index[:, :active_count], candidate_metadata),
            dim=1,
        )
        proposal_insert = torch.cat(
            (
                self.insert_query_frame[:, :active_count],
                candidate_metadata.to(dtype=self.insert_query_frame.dtype),
            ),
            dim=1,
        )
        kept_source = torch.gather(proposal_source, 1, keep_indices)
        kept_insert = torch.gather(proposal_insert, 1, keep_indices)

        candidate_target = torch.full(
            (groups, num_candidates),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        if num_accepted > 0:
            result_slots = torch.arange(
                final_count,
                device=self.device,
            ).unsqueeze(0).expand(groups, -1)
            selected_position = (keep_indices - active_count).clamp_min(0)
            original_candidate = torch.gather(
                selected_candidates,
                1,
                selected_position.clamp_max(num_accepted - 1),
            )
            group_index = torch.arange(
                groups,
                device=self.device,
            ).unsqueeze(1).expand_as(keep_indices)
            candidate_target[
                group_index[kept_candidate],
                original_candidate[kept_candidate],
            ] = result_slots[kept_candidate]

        self.keys.zero_()
        self.values.zero_()
        self.density.zero_()
        self.density_baseline.zero_()
        self.source_index.fill_(-1)
        self.insert_query_frame.fill_(-1)
        self.keys[:, :final_count].copy_(kept_keys)
        self.values[:, :final_count].copy_(kept_values)
        self.density[:, :final_count].copy_(final_density)
        self.density_baseline[:, :final_count].copy_(kept_baseline)
        self.source_index[:, :final_count].copy_(kept_source)
        self.insert_query_frame[:, :final_count].copy_(kept_insert)
        self.counts.fill_(final_count)
        self._legacy_baseline_initialized = final_count >= capacity

        ordered_density = torch.gather(
            candidate_external,
            1,
            candidate_order,
        )
        ordered_score = torch.gather(
            candidate_score,
            1,
            candidate_order,
        )
        ordered_target = torch.gather(
            candidate_target,
            1,
            candidate_order,
        )
        accepted_entries = ordered_target >= 0
        ordered_replacement_density = torch.where(
            accepted_entries,
            torch.gather(
                final_density,
                1,
                ordered_target.clamp_min(0),
            ),
            torch.zeros_like(ordered_density),
        )
        ordered_evicted_density = torch.zeros_like(ordered_density)
        if evict_count > 0:
            ordered_evicted_density[
                :, added_count : added_count + evict_count
            ] = evicted_density
        accepted_count = torch.full(
            (groups,),
            num_accepted,
            device=self.device,
            dtype=torch.int32,
        )
        accepted = accepted_count > 0
        added = torch.full(
            (groups,),
            added_count > 0,
            device=self.device,
            dtype=torch.bool,
        )
        replaced = torch.full(
            (groups,),
            evict_count > 0,
            device=self.device,
            dtype=torch.bool,
        )
        rejected_full = torch.full(
            (groups,),
            active_count >= capacity and num_accepted == 0,
            device=self.device,
            dtype=torch.bool,
        )
        stats = DensityKVBankStats(
            accepted=accepted,
            added=added,
            replaced=replaced,
            rejected_full=rejected_full,
            candidate_index=candidate_order,
            target_slot=ordered_target,
            candidate_density=ordered_density,
            replacement_density=ordered_replacement_density,
            evicted_density=ordered_evicted_density,
            energy_delta=ordered_score - growth_limit,
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros_like(
            candidate_order,
            dtype=torch.int32,
        )
        stats.trace_entry_reason = torch.full_like(
            candidate_order,
            9,
            dtype=torch.int8,
        )
        stats.trace_anchor_slot = torch.full_like(candidate_order, -1)
        stats.trace_reference_density = torch.ones_like(ordered_score)
        stats.trace_group_reason = torch.full(
            (groups, 1),
            9,
            device=self.device,
            dtype=torch.int8,
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (groups, 1),
            num_candidates,
            device=self.device,
            dtype=torch.int32,
        )
        stats.trace_group_added_energy = (
            ordered_replacement_density.sum(dim=1, keepdim=True)
        )
        stats.trace_group_removed_energy = (
            evicted_density.sum(dim=1, keepdim=True)
        )
        stats.trace_candidate_energy_share = ordered_score
        stats.trace_victim_energy_share = torch.full_like(
            ordered_score,
            growth_limit,
        )
        stats.trace_gate_accepted = accepted_entries
        stats.trace_saturation_ratio = ordered_score
        stats.bootstrap_admitted_count = accepted_count
        stats.bootstrap_gate_mode = "all_anchor_growth_ratio"
        stats.bootstrap_gate_limit = growth_limit
        return stats

    @torch.no_grad()
    def _update_legacy_density_gated_bootstrap_v2(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Gate and replace in one exact old-plus-current density field.

        The configured gate changes only the admission score.  Every mode uses
        the same union-density victim selection and exact final-density update.
        """
        if self.config.legacy_bootstrap_v2_gate == "all_anchor_growth_ratio":
            return self._update_legacy_all_anchor_growth_bootstrap(keys, values)
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError(
                "density-gated bootstrap v2 requires synchronized group counts"
            )
        active_count = int(self.counts[0].item())
        capacity = self.config.max_entries
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        groups, num_candidates, _ = keys_gnd.shape
        floor = self.config.append_density_baseline_floor

        union_keys = torch.cat((self.keys[:, :active_count], keys_gnd), dim=1)
        union_values = torch.cat((self.values[:, :active_count], values_gnd), dim=1)
        (
            old_union_density,
            candidate_union_density,
            candidate_external_density,
            nearest_old,
        ) = (
            self._bootstrap_v2_union_density(
                keys_gnd,
                active_count=active_count,
            )
        )
        union_density = torch.cat(
            (old_union_density, candidate_union_density),
            dim=1,
        )
        gate_mode = self.config.legacy_bootstrap_v2_gate
        if active_count == 0:
            if gate_mode == "absolute_candidate_density":
                candidate_density = candidate_external_density
                reference_density = torch.full_like(
                    candidate_density,
                    self.config.legacy_bootstrap_absolute_density_limit,
                )
                gate_limit = 1.0
            else:
                candidate_density = candidate_union_density
                reference_density = candidate_density.median(
                    dim=1, keepdim=True
                ).values.clamp_min(floor).expand_as(candidate_density)
                gate_limit = self.config.legacy_bootstrap_density_limit
        else:
            if gate_mode == "full_union_candidate_ratio":
                candidate_density = candidate_union_density
                reference_density = torch.gather(
                    old_union_density,
                    1,
                    nearest_old,
                ).clamp_min(floor)
                gate_limit = self.config.legacy_bootstrap_density_limit
            elif gate_mode == "current_old_ratio":
                candidate_density = candidate_external_density
                reference_density = torch.gather(
                    self.density[:, :active_count],
                    1,
                    nearest_old,
                ).clamp_min(floor)
                gate_limit = self.config.legacy_bootstrap_density_limit
            elif gate_mode == "absolute_candidate_density":
                candidate_density = candidate_external_density
                reference_density = torch.full_like(
                    candidate_density,
                    self.config.legacy_bootstrap_absolute_density_limit,
                )
                gate_limit = 1.0
            elif gate_mode == "anchor_growth_ratio":
                candidate_density = torch.gather(
                    old_union_density,
                    1,
                    nearest_old,
                )
                reference_density = torch.gather(
                    self.density[:, :active_count],
                    1,
                    nearest_old,
                ).clamp_min(floor)
                gate_limit = self.config.legacy_bootstrap_density_limit
            else:  # guarded by DensityKVBankConfig validation
                raise RuntimeError(f"unsupported bootstrap v2 gate: {gate_mode}")
        saturation_ratio = candidate_density / reference_density
        candidate_order = torch.argsort(
            saturation_ratio,
            dim=1,
            descending=False,
            stable=True,
        )
        admissible_count = (
            saturation_ratio < gate_limit
        ).sum(dim=1)
        admissible_count.clamp_max_(capacity)
        num_accepted = int(admissible_count.min().item())

        ordered_density = torch.gather(candidate_density, 1, candidate_order)
        ordered_ratio = torch.gather(saturation_ratio, 1, candidate_order)
        ordered_reference = torch.gather(
            reference_density, 1, candidate_order
        )
        ordered_anchor = torch.gather(nearest_old, 1, candidate_order)
        ordered_target = torch.full_like(candidate_order, -1)
        accepted_entries = torch.zeros_like(candidate_order, dtype=torch.bool)
        ordered_replacement_density = torch.zeros_like(ordered_density)
        ordered_evicted_density = torch.zeros_like(ordered_density)

        accepted_count = torch.full(
            (groups,),
            num_accepted,
            device=self.device,
            dtype=torch.int32,
        )
        accepted = accepted_count > 0
        free_entries = max(capacity - active_count, 0)
        evict_count = max(num_accepted - free_entries, 0)
        added_count = num_accepted - evict_count

        if num_accepted > 0:
            selected_candidates = candidate_order[:, :num_accepted]
            keep_mask = torch.zeros(
                groups,
                active_count + num_candidates,
                device=self.device,
                dtype=torch.bool,
            )
            if active_count > 0:
                keep_mask[:, :active_count] = True
            if evict_count > 0:
                evicted_density, evicted_old = torch.topk(
                    union_density[:, :active_count],
                    k=evict_count,
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                keep_mask[:, :active_count].scatter_(1, evicted_old, False)
                ordered_evicted_density[
                    :, added_count : added_count + evict_count
                ] = evicted_density
            keep_mask.scatter_(
                1,
                selected_candidates + active_count,
                True,
            )

            final_count = active_count - evict_count + num_accepted
            union_length = active_count + num_candidates
            all_indices = torch.arange(
                union_length, device=self.device
            ).unsqueeze(0).expand(groups, -1)
            keep_indices = all_indices.masked_fill(
                ~keep_mask, union_length
            ).sort(dim=1).values[:, :final_count]
            excluded_count = union_length - final_count
            excluded_indices = all_indices.masked_select(~keep_mask).view(
                groups, excluded_count
            )

            kept_keys = self._gather_rows(union_keys, keep_indices).contiguous()
            kept_values = self._gather_rows(union_values, keep_indices).contiguous()
            final_density = torch.gather(union_density, 1, keep_indices)
            if excluded_count > 0:
                excluded_keys = self._gather_rows(
                    union_keys, excluded_indices
                ).contiguous()
                work_chunk_size = self.config.union_work_chunk_size
                if work_chunk_size < 0:
                    work_chunk_size = self.config.update_chunk_size
                for start in range(0, final_count, work_chunk_size):
                    end = min(start + work_chunk_size, final_count)
                    final_density[:, start:end].sub_(
                        self._density_sum(
                            self._squared_l2(
                                kept_keys[:, start:end],
                                excluded_keys,
                            )
                        )
                    )
            final_density.clamp_min_(0.0)

            candidate_target = torch.full(
                (groups, num_candidates),
                -1,
                device=self.device,
                dtype=torch.long,
            )
            result_slots = torch.arange(
                final_count, device=self.device
            ).unsqueeze(0).expand(groups, -1)
            kept_candidate = keep_indices >= active_count
            group_index = torch.arange(
                groups, device=self.device
            ).unsqueeze(1).expand_as(keep_indices)
            kept_candidate_index = keep_indices - active_count
            candidate_target[
                group_index[kept_candidate],
                kept_candidate_index[kept_candidate],
            ] = result_slots[kept_candidate]
            ordered_target = torch.gather(
                candidate_target, 1, candidate_order
            )
            accepted_entries = ordered_target >= 0
            ordered_replacement_density = torch.where(
                accepted_entries,
                torch.gather(
                    final_density,
                    1,
                    ordered_target.clamp_min(0),
                ),
                torch.zeros_like(ordered_density),
            )

            candidate_metadata = torch.full(
                (groups, num_candidates),
                -1,
                device=self.device,
                dtype=self.source_index.dtype,
            )
            union_source = torch.cat(
                (self.source_index[:, :active_count], candidate_metadata),
                dim=1,
            )
            union_insert = torch.cat(
                (
                    self.insert_query_frame[:, :active_count],
                    candidate_metadata.to(dtype=self.insert_query_frame.dtype),
                ),
                dim=1,
            )
            kept_source = torch.gather(union_source, 1, keep_indices)
            kept_insert = torch.gather(union_insert, 1, keep_indices)

            self.keys.zero_()
            self.values.zero_()
            self.density.zero_()
            self.density_baseline.zero_()
            self.source_index.fill_(-1)
            self.insert_query_frame.fill_(-1)
            self.keys[:, :final_count].copy_(kept_keys)
            self.values[:, :final_count].copy_(kept_values)
            self.density[:, :final_count].copy_(final_density)
            self.density_baseline[:, :final_count].copy_(
                final_density.clamp_min(floor)
            )
            self.source_index[:, :final_count].copy_(kept_source)
            self.insert_query_frame[:, :final_count].copy_(kept_insert)
            self.counts.fill_(final_count)
            self._legacy_baseline_initialized = final_count >= capacity

        added = torch.full(
            (groups,),
            added_count > 0,
            device=self.device,
            dtype=torch.bool,
        )
        replaced = torch.full(
            (groups,),
            evict_count > 0,
            device=self.device,
            dtype=torch.bool,
        )
        rejected_full = torch.full(
            (groups,),
            active_count >= capacity and num_accepted == 0,
            device=self.device,
            dtype=torch.bool,
        )
        stats = DensityKVBankStats(
            accepted=accepted,
            added=added,
            replaced=replaced,
            rejected_full=rejected_full,
            candidate_index=candidate_order,
            target_slot=ordered_target,
            candidate_density=ordered_density,
            replacement_density=ordered_replacement_density,
            evicted_density=ordered_evicted_density,
            energy_delta=(
                ordered_ratio - gate_limit
            ),
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros_like(
            candidate_order, dtype=torch.int32
        )
        stats.trace_entry_reason = torch.full_like(
            candidate_order, 8, dtype=torch.int8
        )
        stats.trace_anchor_slot = ordered_anchor
        stats.trace_reference_density = ordered_reference
        stats.trace_group_reason = torch.full(
            (groups, 1), 8, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (groups, 1),
            num_candidates,
            device=self.device,
            dtype=torch.int32,
        )
        stats.trace_group_added_energy = (
            ordered_replacement_density[:, :num_accepted].sum(
                dim=1, keepdim=True
            )
            if num_accepted > 0
            else torch.zeros(
                groups, 1, device=self.device, dtype=torch.float32
            )
        )
        stats.trace_group_removed_energy = (
            ordered_evicted_density[:, :num_accepted].sum(
                dim=1, keepdim=True
            )
            if num_accepted > 0
            else torch.zeros_like(stats.trace_group_added_energy)
        )
        stats.trace_candidate_energy_share = ordered_ratio
        stats.trace_victim_energy_share = torch.full_like(
            ordered_ratio, gate_limit
        )
        stats.trace_gate_accepted = accepted_entries
        stats.trace_saturation_ratio = ordered_ratio
        stats.bootstrap_admitted_count = accepted_count
        stats.bootstrap_gate_mode = gate_mode
        stats.bootstrap_gate_limit = gate_limit
        return stats

    @torch.no_grad()
    def _update_legacy_density_gated_bootstrap_v4(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Apply randomized sequential admission against an adaptive mean.

        The reference mean is frozen before any decision and computed from the
        complete old-bank plus current-candidate field. Each group then admits
        candidates independently. At capacity, accepted candidates replace
        only the currently densest old entries.
        """
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        groups, num_candidates, _ = keys_gnd.shape
        capacity = self.config.max_entries
        old_counts = self.counts.long().clone()
        active_count = int(old_counts.max().item())
        old_valid = (
            torch.arange(active_count, device=self.device).unsqueeze(0)
            < old_counts.unsqueeze(1)
        )

        old_potential = torch.zeros(
            groups,
            num_candidates,
            device=self.device,
            dtype=torch.float32,
        )
        work_chunk_size = self.config.union_work_chunk_size
        if work_chunk_size < 0:
            work_chunk_size = self.config.update_chunk_size
        if active_count > 0:
            old_keys = self.keys[:, :active_count]
            for start in range(0, num_candidates, work_chunk_size):
                end = min(start + work_chunk_size, num_candidates)
                contribution = self._density_contribution(
                    self._squared_l2(keys_gnd[:, start:end], old_keys)
                )
                contribution.mul_(old_valid.unsqueeze(1))
                old_potential[:, start:end] = contribution.sum(dim=-1)
                del contribution

        candidate_internal = self._density_over_union(keys_gnd)
        visible_count = (
            old_counts + max(num_candidates - 1, 0)
        ).clamp_min(1)
        reference_mean = (
            (old_potential + candidate_internal)
            / visible_count.unsqueeze(1).float()
        ).mean(dim=1)
        reference_mean.clamp_min_(
            self.config.append_density_baseline_floor
        )
        ratio_limit = self.config.legacy_bootstrap_v4_ratio_limit
        threshold = reference_mean * ratio_limit

        result = randomized_triangular_density_admission(
            keys_gnd,
            old_potential,
            threshold=threshold,
            old_count=old_counts,
            density_scale=self.config.density_scale,
            riesz_power=self.config.riesz_power,
            riesz_eps=self.config.riesz_eps,
            seed=(
                self.config.legacy_bootstrap_v4_seed
                + self._bootstrap4_commit_index
            ),
            normalize_by_visible_count=True,
            implementation=self.config.fast_impl,
        )
        self._bootstrap4_commit_index += 1

        ordered_rank = (
            result.ordered_accepted_mask.to(torch.int32).cumsum(dim=1) - 1
        )
        ordered_selected = (
            result.ordered_accepted_mask
            & (ordered_rank < capacity)
        )
        selected_count = ordered_selected.sum(dim=1, dtype=torch.long)
        max_selected = int(selected_count.max().item())
        source_selected = torch.zeros(
            groups,
            num_candidates,
            device=self.device,
            dtype=torch.bool,
        )
        source_selected.scatter_(
            1,
            result.permutation.unsqueeze(0).expand(groups, -1),
            ordered_selected,
        )

        candidate_order = torch.arange(
            num_candidates,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(groups, -1)
        target_slot = torch.full_like(candidate_order, -1)
        source_evicted_density = torch.zeros(
            groups,
            num_candidates,
            device=self.device,
            dtype=torch.float32,
        )
        replacement_density = torch.zeros_like(source_evicted_density)
        added_count = torch.minimum(
            selected_count,
            (capacity - old_counts).clamp_min(0),
        )
        evict_count = selected_count - added_count

        if max_selected > 0:
            ordered_positions = torch.arange(
                num_candidates,
                device=self.device,
            ).unsqueeze(0).expand(groups, -1)
            packed_positions = ordered_positions.masked_fill(
                ~ordered_selected, num_candidates
            ).sort(dim=1).values[:, :max_selected]
            selected_valid = (
                torch.arange(max_selected, device=self.device).unsqueeze(0)
                < selected_count.unsqueeze(1)
            )
            safe_positions = packed_positions.clamp_max(num_candidates - 1)
            selected_source = result.permutation[safe_positions]
            selected_keys = self._gather_rows(
                keys_gnd, selected_source
            ).contiguous()
            selected_values = self._gather_rows(
                values_gnd, selected_source
            ).contiguous()

            max_evict = int(evict_count.max().item())
            if max_evict > 0:
                old_density = self.density[:, :active_count].masked_fill(
                    ~old_valid, -torch.inf
                )
                victim_slots = torch.topk(
                    old_density,
                    k=max_evict,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                victim_valid = (
                    torch.arange(max_evict, device=self.device).unsqueeze(0)
                    < evict_count.unsqueeze(1)
                )
                victim_keys = self._gather_rows(
                    self.keys[:, :active_count], victim_slots
                ).contiguous()
                victim_density = torch.gather(
                    self.density[:, :active_count], 1, victim_slots
                )
            else:
                victim_slots = torch.empty(
                    groups, 0, device=self.device, dtype=torch.long
                )
                victim_valid = torch.empty(
                    groups, 0, device=self.device, dtype=torch.bool
                )
                victim_keys = self.keys[:, :0]
                victim_density = self.density[:, :0]

            selected_position = torch.arange(
                max_selected, device=self.device
            ).unsqueeze(0).expand(groups, -1)
            is_added = selected_position < added_count.unsqueeze(1)
            free_slots = old_counts.unsqueeze(1) + selected_position
            victim_rank = (
                selected_position - added_count.unsqueeze(1)
            ).clamp_min(0)
            safe_victim_rank = victim_rank.clamp_max(max(max_evict - 1, 0))
            if max_evict > 0:
                replacement_slots = torch.gather(
                    victim_slots, 1, safe_victim_rank
                )
                selected_evicted_density = torch.gather(
                    victim_density, 1, safe_victim_rank
                )
            else:
                replacement_slots = torch.zeros_like(selected_position)
                selected_evicted_density = torch.zeros(
                    groups,
                    max_selected,
                    device=self.device,
                    dtype=torch.float32,
                )
            selected_target = torch.where(
                is_added, free_slots, replacement_slots
            )
            selected_target = torch.where(
                selected_valid,
                selected_target,
                torch.zeros_like(selected_target),
            )
            selected_evicted_density = torch.where(
                selected_valid & ~is_added,
                selected_evicted_density,
                torch.zeros_like(selected_evicted_density),
            )

            kept_old = old_valid.clone()
            if max_evict > 0:
                kept_old.scatter_(
                    1,
                    victim_slots,
                    ~victim_valid,
                )
            final_old_density = self.density[:, :active_count].clone()
            selected_external = torch.zeros(
                groups,
                max_selected,
                device=self.device,
                dtype=torch.float32,
            )
            if active_count > 0:
                for start in range(0, active_count, work_chunk_size):
                    end = min(start + work_chunk_size, active_count)
                    old_chunk = self.keys[:, start:end]
                    selected_contribution = self._density_contribution(
                        self._squared_l2(old_chunk, selected_keys)
                    )
                    selected_contribution.mul_(
                        selected_valid.unsqueeze(1)
                    )
                    final_old_density[:, start:end].add_(
                        selected_contribution.sum(dim=-1)
                    )
                    selected_external.add_(
                        (
                            selected_contribution
                            * kept_old[:, start:end].unsqueeze(2)
                        ).sum(dim=1)
                    )
                    if max_evict > 0:
                        victim_contribution = self._density_contribution(
                            self._squared_l2(old_chunk, victim_keys)
                        )
                        victim_contribution.mul_(
                            victim_valid.unsqueeze(1)
                        )
                        final_old_density[:, start:end].sub_(
                            victim_contribution.sum(dim=-1)
                        )
                        del victim_contribution
                    del selected_contribution

            selected_internal = self._density_over_masked_union(
                selected_keys,
                selected_valid,
            )
            selected_final_density = torch.where(
                selected_valid,
                selected_external + selected_internal,
                torch.zeros_like(selected_external),
            )
            final_old_density = torch.where(
                kept_old,
                final_old_density.clamp_min_(0.0),
                torch.zeros_like(final_old_density),
            )

            if active_count > 0:
                self.density[:, :active_count].copy_(final_old_density)
            group_index = torch.arange(
                groups, device=self.device
            ).unsqueeze(1).expand(groups, max_selected)
            write_groups = group_index[selected_valid]
            write_slots = selected_target[selected_valid]
            self.keys[write_groups, write_slots] = selected_keys[selected_valid]
            self.values[write_groups, write_slots] = selected_values[selected_valid]
            self.density[write_groups, write_slots] = selected_final_density[
                selected_valid
            ]
            final_counts = old_counts + added_count
            self.counts.copy_(final_counts.to(torch.int32))
            final_active = self.active_mask()
            self.density.copy_(
                torch.where(
                    final_active,
                    self.density.clamp_min(0.0),
                    torch.zeros_like(self.density),
                )
            )
            self.density_baseline.copy_(
                torch.where(
                    final_active,
                    self.density.clamp_min(
                        self.config.append_density_baseline_floor
                    ),
                    torch.zeros_like(self.density),
                )
            )

            valid_source = selected_source[selected_valid]
            target_slot[write_groups, valid_source] = write_slots
            source_evicted_density[write_groups, valid_source] = (
                selected_evicted_density[selected_valid]
            )
            replacement_density[write_groups, valid_source] = (
                selected_final_density[selected_valid]
            )
            self._legacy_baseline_initialized = bool(
                (self.counts.long() >= capacity).all().item()
            )

        accepted = selected_count > 0
        stats = DensityKVBankStats(
            accepted=accepted,
            added=added_count > 0,
            replaced=evict_count > 0,
            rejected_full=(old_counts >= capacity) & ~accepted,
            candidate_index=candidate_order,
            target_slot=target_slot,
            candidate_density=result.decision_potential,
            replacement_density=replacement_density,
            evicted_density=source_evicted_density,
            energy_delta=result.decision_score - threshold.unsqueeze(1),
            accepted_count=selected_count.to(torch.int32),
            accepted_entry_mask=source_selected,
        )
        normalized_score = (
            result.decision_score / reference_mean.unsqueeze(1)
        )
        stats.trace_decision_group = torch.zeros_like(
            candidate_order, dtype=torch.int32
        )
        stats.trace_entry_reason = torch.where(
            source_selected,
            torch.zeros_like(candidate_order, dtype=torch.int8),
            torch.where(
                result.accepted_mask,
                torch.full_like(candidate_order, 7, dtype=torch.int8),
                torch.full_like(candidate_order, 6, dtype=torch.int8),
            ),
        )
        stats.trace_anchor_slot = torch.full_like(candidate_order, -1)
        stats.trace_reference_density = reference_mean.unsqueeze(1).expand_as(
            result.decision_score
        )
        stats.trace_candidate_energy_share = normalized_score
        stats.trace_victim_energy_share = torch.full_like(
            normalized_score, ratio_limit
        )
        stats.trace_gate_accepted = result.accepted_mask
        stats.trace_saturation_ratio = normalized_score
        stats.trace_group_reason = torch.full(
            (groups, 1), 6, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (groups, 1),
            num_candidates,
            device=self.device,
            dtype=torch.int32,
        )
        stats.trace_group_added_energy = replacement_density.sum(
            dim=1, keepdim=True
        )
        stats.trace_group_removed_energy = source_evicted_density.sum(
            dim=1, keepdim=True
        )
        stats.bootstrap_admitted_count = result.accepted_count
        stats.bootstrap_gate_mode = "current_mean_potential_ratio"
        stats.bootstrap_gate_limit = ratio_limit
        stats.bootstrap_reference_mean = reference_mean
        stats.bootstrap_threshold = threshold
        stats.bootstrap_used_triton = result.used_triton
        return stats

    @torch.no_grad()
    def _update_legacy_bootstrap_tail_cleanup(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Let a small tail group revise a still-underfull bootstrap bank."""
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError(
                "legacy bootstrap tail cleanup requires synchronized counts"
            )
        active_count = int(self.counts[0].item())
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        groups, cleanup_size, _ = keys_gnd.shape
        if cleanup_size <= 0:
            raise RuntimeError("legacy bootstrap tail cleanup is empty")
        if active_count < cleanup_size:
            raise RuntimeError(
                "legacy bootstrap tail cleanup requires at least as many "
                "active entries as cleanup candidates"
            )

        active_keys = self.keys[:, :active_count]
        active_density = self.density[:, :active_count]
        distances_to_store = self._squared_l2(keys_gnd, active_keys)
        contribution_to_store = self._density_contribution(distances_to_store)
        external_density = contribution_to_store.sum(dim=-1)
        internal_density = self._density_over_union(keys_gnd)
        candidate_density = external_density + internal_density

        nearest_store = distances_to_store.argmin(dim=-1)
        reference_baseline = torch.gather(
            self.density_baseline[:, :active_count],
            1,
            nearest_store,
        ).clamp_min(self.config.append_density_baseline_floor)
        saturation_ratio = candidate_density / reference_baseline
        if self.config.legacy_density_growth_gate:
            growth_gate_accepted = (
                saturation_ratio.mean(dim=1)
                < self.config.append_density_growth_limit
            )
        else:
            growth_gate_accepted = torch.ones(
                groups, device=self.device, dtype=torch.bool
            )

        evicted_density, evicted_slots = torch.topk(
            active_density,
            k=cleanup_size,
            dim=1,
            largest=True,
            sorted=False,
        )
        old_keys = self._gather_rows(active_keys, evicted_slots).contiguous()
        old_to_store = self._squared_l2(old_keys, active_keys)
        old_to_store_contribution = self._density_contribution(old_to_store)
        selected_to_evicted = self._gather_matrix_columns(
            contribution_to_store,
            evicted_slots,
        )
        old_internal_density = self._density_over_union(old_keys)
        selected_to_kept = (
            external_density - selected_to_evicted.sum(dim=-1)
        )
        replacement_density = selected_to_kept + internal_density
        removed_energy = (
            evicted_density.sum(dim=1)
            - 0.5 * old_internal_density.sum(dim=1)
        )
        added_energy = (
            selected_to_kept.sum(dim=1)
            + 0.5 * internal_density.sum(dim=1)
        )
        energy_delta = added_energy - removed_energy
        accepted = (
            bool(self.config.evict_densest_when_full)
            & (added_energy < removed_energy * self.config.replacement_ratio)
            & growth_gate_accepted
        )

        proposal_density = (
            active_density
            - old_to_store_contribution.sum(dim=1)
            + contribution_to_store.sum(dim=1)
        )
        proposal_density.scatter_(1, evicted_slots, replacement_density)
        active_density.copy_(
            torch.where(
                accepted.unsqueeze(1),
                proposal_density,
                active_density,
            )
        )
        group_index = torch.arange(
            groups, device=self.device
        ).unsqueeze(1).expand(-1, cleanup_size)
        accepted_entries = accepted.unsqueeze(1).expand(-1, cleanup_size)
        self.keys[
            group_index[accepted_entries], evicted_slots[accepted_entries]
        ] = keys_gnd[accepted_entries]
        self.values[
            group_index[accepted_entries], evicted_slots[accepted_entries]
        ] = values_gnd[accepted_entries]
        self.density_baseline[
            group_index[accepted_entries], evicted_slots[accepted_entries]
        ] = reference_baseline[accepted_entries]

        candidate_index = torch.arange(
            cleanup_size,
            device=self.device,
        ).unsqueeze(0).expand(groups, -1)
        accepted_count = accepted.to(torch.int32) * cleanup_size
        candidate_energy_share = (
            selected_to_kept + 0.5 * internal_density
        )
        victim_energy_share = (
            evicted_density - 0.5 * old_internal_density
        )
        stats = DensityKVBankStats(
            accepted=accepted,
            added=torch.zeros_like(accepted),
            replaced=accepted,
            rejected_full=~accepted,
            candidate_index=candidate_index,
            target_slot=evicted_slots,
            candidate_density=candidate_density,
            replacement_density=replacement_density,
            evicted_density=evicted_density,
            energy_delta=energy_delta.unsqueeze(1).expand_as(candidate_density),
            accepted_count=accepted_count,
            accepted_entry_mask=accepted_entries,
        )
        stats.trace_decision_group = torch.zeros_like(
            candidate_index, dtype=torch.int32
        )
        stats.trace_entry_reason = torch.ones_like(
            candidate_index, dtype=torch.int8
        )
        stats.trace_group_reason = torch.ones(
            groups, 1, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (groups, 1),
            cleanup_size,
            device=self.device,
            dtype=torch.int32,
        )
        stats.trace_group_added_energy = added_energy.unsqueeze(1)
        stats.trace_group_removed_energy = removed_energy.unsqueeze(1)
        stats.trace_candidate_energy_share = candidate_energy_share
        stats.trace_victim_energy_share = victim_energy_share
        stats.trace_gate_accepted = growth_gate_accepted.unsqueeze(1).expand_as(
            candidate_index
        )
        stats.trace_saturation_ratio = saturation_ratio
        stats.trace_anchor_slot = nearest_store
        stats.trace_reference_density = reference_baseline
        return stats

    @torch.no_grad()
    def _update_all_candidates_legacy_chunk_batch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> DensityKVBankStats:
        """Reproduce the nine-grid baseline's order-dependent chunk updates."""
        if self.config.legacy_density_gated_bootstrap_v4:
            warmup_tokens = self.config.legacy_bootstrap_v4_warmup_tokens
            if (
                warmup_tokens > 0
                and self._bootstrap4_commit_index == 0
                and not bool(self.counts.any().item())
            ):
                warmup_count = min(warmup_tokens, int(keys.shape[0]))
                warmup_stats = self._update_batch(
                    keys[:warmup_count],
                    values[:warmup_count],
                    active_count=0,
                    counts_synchronized=True,
                    admission_limit=warmup_count,
                )
                warmup_stats.trace_decision_group.zero_()
                warmup_stats.trace_entry_reason = torch.full_like(
                    warmup_stats.candidate_index, 9, dtype=torch.int8
                )
                warmup_stats.trace_gate_accepted = torch.ones_like(
                    warmup_stats.accepted_entry_mask
                )
                warmup_stats.trace_saturation_ratio = torch.zeros_like(
                    warmup_stats.candidate_density
                )
                warmup_stats.trace_anchor_slot = torch.full_like(
                    warmup_stats.candidate_index, -1
                )
                warmup_stats.trace_reference_density = torch.zeros_like(
                    warmup_stats.candidate_density
                )
                warmup_stats.trace_group_reason = torch.full(
                    (self.groups, 1),
                    9,
                    device=self.device,
                    dtype=torch.int8,
                )
                warmup_stats.trace_group_accepted = torch.ones(
                    self.groups, 1, device=self.device, dtype=torch.bool
                )
                warmup_stats.trace_group_candidate_count = torch.full(
                    (self.groups, 1),
                    warmup_count,
                    device=self.device,
                    dtype=torch.int32,
                )
                warmup_stats.trace_group_added_energy = (
                    warmup_stats.replacement_density.sum(dim=1, keepdim=True)
                )
                warmup_stats.trace_group_removed_energy = torch.zeros_like(
                    warmup_stats.trace_group_added_energy
                )
                if warmup_count == keys.shape[0]:
                    self._bootstrap4_commit_index += 1
                    warmup_stats.bootstrap_admitted_count = (
                        warmup_stats.accepted_count
                    )
                    warmup_stats.bootstrap_warmup_admitted_count = (
                        warmup_stats.accepted_count
                    )
                    warmup_stats.bootstrap_gate_mode = "first_latent_unconditional"
                    warmup_stats.bootstrap_gate_limit = (
                        self.config.legacy_bootstrap_v4_ratio_limit
                    )
                    return warmup_stats

                gated_stats = self._update_legacy_density_gated_bootstrap_v4(
                    keys[warmup_count:],
                    values[warmup_count:],
                )
                gated_stats.trace_decision_group.fill_(1)
                gated_stats.candidate_index = (
                    gated_stats.candidate_index + warmup_count
                )
                combined = self._combine_legacy_stats(
                    [warmup_stats, gated_stats]
                )
                combined.bootstrap_admitted_count = combined.accepted_count
                combined.bootstrap_warmup_admitted_count = (
                    warmup_stats.accepted_count
                )
                for name in (
                    "bootstrap_gate_mode",
                    "bootstrap_gate_limit",
                    "bootstrap_reference_mean",
                    "bootstrap_threshold",
                    "bootstrap_used_triton",
                ):
                    setattr(combined, name, getattr(gated_stats, name))
                return combined
            return self._update_legacy_density_gated_bootstrap_v4(keys, values)
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError("full KV updates require synchronized group counts")
        active_count = int(self.counts[0].item())
        if self.config.legacy_density_gated_bootstrap_v2:
            return self._update_legacy_density_gated_bootstrap_v2(keys, values)
        if (
            self.config.legacy_density_gated_bootstrap
            and active_count < self.config.max_entries
        ):
            cleanup_size = self.config.legacy_bootstrap_tail_cleanup_size
            if cleanup_size:
                if keys.shape[0] <= cleanup_size:
                    raise RuntimeError(
                        "legacy bootstrap tail cleanup must leave a non-empty "
                        "coarse candidate prefix"
                    )
                coarse_count = keys.shape[0] - cleanup_size
                coarse_stats = self._update_legacy_density_gated_bootstrap(
                    keys[:coarse_count],
                    values[:coarse_count],
                )
                tail_stats = self._update_legacy_bootstrap_tail_cleanup(
                    keys[coarse_count:],
                    values[coarse_count:],
                )
                tail_stats.trace_decision_group.fill_(1)
                tail_stats.candidate_index = (
                    tail_stats.candidate_index + coarse_count
                )
                combined = self._combine_legacy_stats(
                    [coarse_stats, tail_stats]
                )
                combined.bootstrap_admitted_count = (
                    coarse_stats.accepted_count
                )
                combined.bootstrap_tail_replaced_count = (
                    tail_stats.accepted_count
                )
                return combined
            return self._update_legacy_density_gated_bootstrap(keys, values)
        candidate_order = torch.arange(keys.shape[0], device=self.device)
        if (
            self.config.legacy_drop_tail_when_full
            and active_count >= self.config.max_entries
        ):
            complete_count = (
                keys.shape[0] // self.config.update_chunk_size
            ) * self.config.update_chunk_size
            if complete_count == 0:
                raise RuntimeError(
                    "legacy_drop_tail_when_full would discard every candidate"
                )
            keys = keys[:complete_count]
            values = values[:complete_count]
            candidate_order = candidate_order[:complete_count]
        if (
            self.config.legacy_chunk_grouping != "contiguous"
            and active_count >= self.config.max_entries
        ):
            if self.config.legacy_chunk_grouping == "interleaved":
                chunk_count = (
                    keys.shape[0] + self.config.update_chunk_size - 1
                ) // self.config.update_chunk_size
                positions = torch.arange(
                    self.config.update_chunk_size, device=self.device
                ).unsqueeze(1)
                chunk_offsets = (
                    torch.arange(chunk_count, device=self.device)
                    * self.config.update_chunk_size
                ).unsqueeze(0)
                candidate_order = (positions + chunk_offsets).flatten()
                candidate_order = candidate_order[candidate_order < keys.shape[0]]
            else:
                hashed = candidate_order.to(torch.int64)
                hashed = ((hashed ^ (hashed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
                hashed = ((hashed ^ (hashed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
                hashed = hashed ^ (hashed >> 16)
                candidate_order = torch.argsort(hashed, stable=True)
            keys = keys[candidate_order]
            values = values[candidate_order]
        repeat_tail = self.config.legacy_repeat_tail_when_full
        if repeat_tail and active_count >= self.config.max_entries:
            if repeat_tail > keys.shape[0]:
                raise RuntimeError(
                    "legacy_repeat_tail_when_full exceeds the candidate count"
                )
            repeated_order = candidate_order[-repeat_tail:]
            keys = torch.cat((keys, keys[-repeat_tail:]), dim=0)
            values = torch.cat((values, values[-repeat_tail:]), dim=0)
            candidate_order = torch.cat((candidate_order, repeated_order), dim=0)
        normalized_group_sizes: list[int] | None = None
        if (
            self.config.legacy_normalized_group_count > 0
            and active_count >= self.config.max_entries
        ):
            normalized_group_sizes = self._legacy_normalized_group_sizes(
                int(keys.shape[0])
            )
        offset = 0
        normalized_group_index = 0
        chunk_stats: list[DensityKVBankStats] = []
        compact_single_stats = (
            self.config.update_chunk_size == 1
            and active_count >= self.config.max_entries
            and getattr(self, "lineage_trace_enabled", None) is False
        )
        compact_accepted = torch.zeros(
            self.groups, device=self.device, dtype=torch.bool
        )
        compact_added = torch.zeros_like(compact_accepted)
        compact_replaced = torch.zeros_like(compact_accepted)
        compact_rejected = torch.zeros_like(compact_accepted)
        compact_accepted_count = torch.zeros(
            self.groups, device=self.device, dtype=torch.int32
        )
        compact_last_stats: DensityKVBankStats | None = None
        while offset < keys.shape[0]:
            free_entries = max(self.config.max_entries - active_count, 0)
            if normalized_group_sizes is not None:
                if normalized_group_index >= len(normalized_group_sizes):
                    raise RuntimeError(
                        "normalized legacy grouping ended before its candidates"
                    )
                chunk_size = normalized_group_sizes[normalized_group_index]
                normalized_group_index += 1
            else:
                chunk_limit = self.config.update_chunk_size
                if free_entries > 0 and self.config.legacy_warmup_chunk_size > 0:
                    chunk_limit = self.config.legacy_warmup_chunk_size
                chunk_size = min(
                    chunk_limit,
                    keys.shape[0] - offset,
                )
            if free_entries > 0:
                chunk_size = min(chunk_size, free_entries)
            end = offset + chunk_size
            if free_entries == 0 and chunk_size == 1:
                stats = self._update_legacy_single_candidate(
                    keys[offset:end],
                    values[offset:end],
                )
            else:
                stats = self._update_batch(
                    keys[offset:end],
                    values[offset:end],
                    active_count=active_count,
                    counts_synchronized=True,
                    admission_limit=chunk_size,
                )
            stats.trace_decision_group.fill_(len(chunk_stats))
            stats.candidate_index = candidate_order[
                stats.candidate_index + offset
            ]
            if compact_single_stats and free_entries == 0:
                compact_accepted |= stats.accepted
                compact_added |= stats.added
                compact_replaced |= stats.replaced
                compact_rejected |= stats.rejected_full
                compact_accepted_count += stats.accepted_count
                compact_last_stats = stats
            else:
                chunk_stats.append(stats)
            if free_entries > 0:
                active_count += chunk_size
            offset = end
        if (
            normalized_group_sizes is not None
            and normalized_group_index != len(normalized_group_sizes)
        ):
            raise RuntimeError("normalized legacy grouping left unused groups")

        if compact_last_stats is not None and not chunk_stats:
            compact_last_stats.accepted = compact_accepted
            compact_last_stats.added = compact_added
            compact_last_stats.replaced = compact_replaced
            compact_last_stats.rejected_full = compact_rejected
            compact_last_stats.accepted_count = compact_accepted_count
            return compact_last_stats
        return self._combine_legacy_stats(chunk_stats)

    @torch.no_grad()
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> DensityKVBankStats:
        keys, values = self._prepare_inputs(keys, values)
        if self.config.process_all_candidates:
            if self.config.full_update_mode == "append_only_density":
                return self._update_all_candidates_append_only_density(keys, values)
            if self.config.full_update_mode == "union_prune_density":
                return self._update_all_candidates_union_prune_density(keys, values)
            if self.config.full_update_mode == "gated_union_prune_density":
                return self._update_all_candidates_gated_union_prune_density(
                    keys, values
                )
            if self.config.full_update_mode == "legacy_chunk_batch":
                return self._update_all_candidates_legacy_chunk_batch(keys, values)
            return self._update_all_candidates_frozen_snapshot(keys, values)
        if self.config.max_admissions_per_update > 1:
            return self._update_batch(keys, values)
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        distances = self._squared_l2(keys_gnd, self.keys)
        candidate_index, candidate_density, target_slot, target_density, used_triton = self._select(
            distances
        )

        groups = torch.arange(self.groups, device=self.device)
        selected_keys = keys_gnd[groups, candidate_index].contiguous()
        selected_values = values_gnd[groups, candidate_index].contiguous()
        selected_distances = distances[groups, candidate_index].contiguous()
        was_full = self.counts.long() >= self.config.max_entries
        selected_target_distance = selected_distances[groups, target_slot]
        target_contribution = self._density_contribution(selected_target_distance)
        replacement_density = torch.where(
            was_full,
            candidate_density - target_contribution,
            candidate_density,
        ).clamp_min_(0.0)
        energy_delta = replacement_density - target_density
        replace = (
            was_full
            & bool(self.config.evict_densest_when_full)
            & (replacement_density < target_density * self.config.replacement_ratio)
        )
        added = ~was_full
        accepted = added | replace

        old_keys = self.keys[groups, target_slot].unsqueeze(1).contiguous()
        old_distances = self._squared_l2(old_keys, self.keys).squeeze(1).contiguous()
        mutated = False
        if used_triton and triton_mutate_density_kv is not None:
            mutated = triton_mutate_density_kv(
                self.keys,
                self.values,
                self.density,
                self.counts,
                selected_keys,
                selected_values,
                selected_distances,
                old_distances,
                target_slot,
                accepted,
                was_full,
                replacement_density,
                density_scale=self.config.density_scale,
                riesz_power=self.config.riesz_power,
                riesz_eps=self.config.riesz_eps,
            )
        if not mutated:
            if self.config.fast_impl == "triton":
                raise RuntimeError("Triton density mutation was requested but is unavailable")
            self._mutate_torch(
                selected_keys,
                selected_values,
                selected_distances,
                old_distances,
                target_slot,
                accepted,
                was_full,
                replacement_density,
            )

        return DensityKVBankStats(
            accepted=accepted,
            added=added & accepted,
            replaced=replace,
            rejected_full=was_full & ~replace,
            candidate_index=candidate_index,
            target_slot=target_slot,
            candidate_density=candidate_density,
            replacement_density=replacement_density,
            evicted_density=target_density,
            energy_delta=energy_delta,
            accepted_count=accepted.to(torch.int32),
            accepted_entry_mask=accepted,
        )

    @torch.no_grad()
    def recompute_density(self) -> torch.Tensor:
        """Rebuild the density table exactly; intended for initialization/tests."""
        distances = self._squared_l2(self.keys, self.keys)
        active = self.active_mask()
        pair_mask = active.unsqueeze(1) & active.unsqueeze(2)
        diagonal = torch.eye(
            self.keys.shape[1],
            device=self.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        contribution = self._density_contribution(distances)
        rebuilt = (contribution * (pair_mask & ~diagonal)).sum(dim=-1)
        self.density.copy_(torch.where(active, rebuilt, torch.zeros_like(rebuilt)))
        return self.density

    @torch.no_grad()
    def load_entries(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Replace the bank with the first M entries and rebuild density."""
        keys, values = self._prepare_inputs(keys, values, thin=False)
        if self.config.full_update_mode == "append_only_density":
            count = int(keys.shape[0])
            self._ensure_storage_capacity(count)
        else:
            count = min(int(keys.shape[0]), self.config.max_entries)
        self.clear()
        self.keys[:, :count].copy_(keys[:count].permute(1, 0, 2))
        self.values[:, :count].copy_(values[:count].permute(1, 0, 2))
        self.counts.fill_(count)
        self.recompute_density()
        self.density_baseline[:, :count].copy_(
            self.density[:, :count].clamp_min(
                self.config.append_density_baseline_floor
            )
        )
        self._legacy_baseline_initialized = (
            self.config.legacy_density_growth_gate
            and count >= self.config.max_entries
        )
