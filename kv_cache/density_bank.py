"""Density-guided online KV bank used by the public DensityKV release.

Keys define the retention geometry. Values remain opaque payloads and always
move with their paired keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn

try:
    from .triton_density_bank import triton_density_sum
except Exception:  # pragma: no cover - optional CUDA optimization
    triton_density_sum = None


@dataclass(frozen=True)
class DensityKVBankConfig:
    """Configuration for the published density-growth policy."""

    max_entries: int = 1024
    density_scale: float = 1.0
    riesz_power: float = 2.0
    riesz_eps: float = 1.0
    density_growth_limit: float = 2.0
    density_baseline_floor: float = 1.0e-6
    work_chunk_size: int = 512
    compute_dtype: str = "bfloat16"
    fast_impl: str = "auto"  # "auto", "torch", or "triton"

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if self.density_scale <= 0:
            raise ValueError("density_scale must be positive")
        if self.riesz_power <= 0 or self.riesz_eps <= 0:
            raise ValueError("Riesz parameters must be positive")
        if self.density_growth_limit <= 0:
            raise ValueError("density_growth_limit must be positive")
        if self.density_baseline_floor <= 0:
            raise ValueError("density_baseline_floor must be positive")
        if self.work_chunk_size <= 0:
            raise ValueError("work_chunk_size must be positive")
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
            "mean_candidate_density": float(
                self.candidate_density.float().mean().item()
            ),
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
    """Maintain a bounded token-level KV bank for independent head groups.

    Incoming tensors use ``[N, G, D]`` and stored tensors use ``[G, M, D]``,
    where ``G`` combines batch and attention-head indices. Every group shares
    one feasible admission-prefix length so their bank sizes remain aligned.
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
        self._auto_triton_failed = False
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

    def active_mask(self) -> torch.Tensor:
        slots = torch.arange(self.config.max_entries, device=self.device)
        return slots.unsqueeze(0) < self.counts.long().unsqueeze(1)

    def view(self) -> DensityKVBankView:
        return DensityKVBankView(
            keys=self.keys,
            values=self.values,
            density=self.density,
            active_mask=self.active_mask(),
            counts=self.counts,
        )

    def _density_contribution(self, squared_distance: torch.Tensor) -> torch.Tensor:
        normalized = squared_distance.float() / (self.config.density_scale**2)
        return 1.0 / (self.config.riesz_eps + normalized).pow(
            self.config.riesz_power
        )

    def _density_sum(
        self,
        squared_distance: torch.Tensor,
        *,
        exclude_diagonal: bool = False,
    ) -> torch.Tensor:
        use_triton = (
            self.config.fast_impl in {"auto", "triton"}
            and triton_density_sum is not None
            and not self._auto_triton_failed
        )
        if use_triton:
            try:
                result = triton_density_sum(
                    squared_distance,
                    density_scale=self.config.density_scale,
                    riesz_power=self.config.riesz_power,
                    riesz_eps=self.config.riesz_eps,
                    exclude_diagonal=exclude_diagonal,
                )
            except Exception as error:
                if self.config.fast_impl == "triton":
                    raise
                self._auto_triton_failed = True
                warnings.warn(
                    "Triton density reduction failed; falling back to PyTorch "
                    f"for this bank ({type(error).__name__}: {error}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                if result is not None:
                    return result
        if self.config.fast_impl == "triton" and triton_density_sum is None:
            raise RuntimeError("Triton density reduction is unavailable")
        contribution = self._density_contribution(squared_distance)
        if exclude_diagonal:
            if squared_distance.shape[-2] != squared_distance.shape[-1]:
                raise ValueError("diagonal exclusion requires a square matrix")
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

    def _prepare_inputs(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.ndim != 3 or keys.shape[1:] != (self.groups, self.key_dim):
            raise ValueError(
                f"keys must have shape [N, {self.groups}, {self.key_dim}], "
                f"got {tuple(keys.shape)}"
            )
        if values.ndim != 3 or values.shape != (
            keys.shape[0],
            self.groups,
            self.value_dim,
        ):
            raise ValueError(
                f"values must have shape [N, {self.groups}, {self.value_dim}], "
                f"got {tuple(values.shape)}"
            )
        if keys.shape[0] == 0:
            raise ValueError("a KV update must contain at least one candidate")
        keys = keys.detach().to(self.device, self.keys.dtype).contiguous()
        values = values.detach().to(self.device, self.values.dtype).contiguous()
        return keys, values

    @staticmethod
    def _gather_rows(values_gnd: torch.Tensor, indices_gr: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values_gnd,
            1,
            indices_gr.unsqueeze(-1).expand(-1, -1, values_gnd.shape[-1]),
        )

    @torch.no_grad()
    def _density_over_union(self, keys_gld: torch.Tensor) -> torch.Tensor:
        """Compute exact per-point density without materializing L by L."""
        length = int(keys_gld.shape[1])
        result = torch.empty(
            self.groups,
            length,
            device=self.device,
            dtype=torch.float32,
        )
        chunk_size = min(self.config.work_chunk_size, length)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            distances = self._squared_l2(keys_gld[:, start:end], keys_gld)
            density = self._density_sum(distances)
            rows = torch.arange(end - start, device=self.device)
            self_distance = distances[:, rows, rows + start]
            result[:, start:end] = density - self._density_contribution(
                self_distance
            )
        return result.clamp_min_(0.0)

    @torch.no_grad()
    def _candidate_order_and_prefix(
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
        """Return the largest cross-head prefix satisfying density growth."""
        groups, num_candidates, _ = candidate_keys_gnd.shape
        candidate_order = torch.arange(
            num_candidates, device=self.device
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
            self.config.density_baseline_floor
        )
        chunk_size = self.config.work_chunk_size

        for start in range(0, num_candidates, chunk_size):
            end = min(start + chunk_size, num_candidates)
            contribution = self._density_contribution(
                self._squared_l2(candidate_keys_gnd[:, start:end], old_keys)
            )
            candidate_external[:, start:end] = contribution.sum(dim=-1)
            candidate_score[:, start:end] = (
                contribution / old_baseline.unsqueeze(1)
            ).amax(dim=-1)

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
        growth_limit = self.config.density_growth_limit
        while processed < max_accepted:
            end = min(processed + chunk_size, max_accepted)
            contribution = self._density_contribution(
                self._squared_l2(ordered_keys[:, processed:end], old_keys)
            )
            cumulative = old_increment.unsqueeze(1) + contribution.cumsum(dim=1)
            violating = (
                (old_density.unsqueeze(1) + cumulative)
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
            feasible = (violating <= eviction_budget.unsqueeze(0)).all(dim=0)
            feasible_positions = feasible.nonzero(as_tuple=False)
            if feasible_positions.numel() > 0:
                best_position = int(feasible_positions[-1, 0].item())
                accepted_count = processed + best_position + 1
                best_increment = cumulative[:, best_position].clone()
            old_increment = cumulative[:, -1].clone()
            processed = end
        return (
            candidate_order,
            candidate_external,
            candidate_score,
            accepted_count,
            best_increment,
        )

    @torch.no_grad()
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> DensityKVBankStats:
        keys, values = self._prepare_inputs(keys, values)
        if not torch.equal(self.counts, self.counts[:1].expand_as(self.counts)):
            raise RuntimeError("DensityKV requires synchronized head-group counts")

        active_count = int(self.counts[0].item())
        capacity = self.config.max_entries
        floor = self.config.density_baseline_floor
        growth_limit = self.config.density_growth_limit
        keys_gnd = keys.permute(1, 0, 2).contiguous()
        values_gnd = values.permute(1, 0, 2).contiguous()
        groups, num_candidates, _ = keys_gnd.shape
        (
            candidate_order,
            candidate_external,
            candidate_score,
            num_accepted,
            old_increment,
        ) = self._candidate_order_and_prefix(
            keys_gnd,
            active_count=active_count,
        )

        selected_candidates = candidate_order[:, :num_accepted]
        selected_keys = self._gather_rows(
            keys_gnd, selected_candidates
        ).contiguous()
        selected_values = self._gather_rows(
            values_gnd, selected_candidates
        ).contiguous()
        selected_external = torch.gather(
            candidate_external, 1, selected_candidates
        )
        selected_internal = (
            self._density_over_union(selected_keys)
            if num_accepted > 0
            else torch.zeros(groups, 0, device=self.device, dtype=torch.float32)
        )
        selected_density = selected_external + selected_internal
        old_proposal_density = self.density[:, :active_count] + old_increment
        proposal_keys = torch.cat(
            (self.keys[:, :active_count], selected_keys), dim=1
        )
        proposal_values = torch.cat(
            (self.values[:, :active_count], selected_values), dim=1
        )
        proposal_density = torch.cat(
            (old_proposal_density, selected_density), dim=1
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
            old_baseline = self.density_baseline[:, :active_count].clamp_min(floor)
            must_evict = old_proposal_density / old_baseline >= growth_limit
            must_evict_count = must_evict.sum(dim=1)
            if bool((must_evict_count > evict_count).any()):
                raise RuntimeError(
                    "selected prefix cannot evict all density-growth violations"
                )
            victim_score = torch.where(
                must_evict,
                torch.full_like(old_proposal_density, torch.inf),
                old_proposal_density,
            )
            _, evicted_old = torch.topk(
                victim_score,
                k=evict_count,
                dim=1,
                largest=True,
                sorted=False,
            )
            evicted_density = torch.gather(
                old_proposal_density, 1, evicted_old
            )
            keep_mask[:, :active_count].scatter_(1, evicted_old, False)

        all_indices = torch.arange(
            proposal_count, device=self.device
        ).unsqueeze(0).expand(groups, -1)
        keep_indices = all_indices.masked_fill(
            ~keep_mask, proposal_count
        ).sort(dim=1).values[:, :final_count]
        kept_keys = self._gather_rows(proposal_keys, keep_indices).contiguous()
        kept_values = self._gather_rows(proposal_values, keep_indices).contiguous()
        final_density = torch.gather(proposal_density, 1, keep_indices)
        if evict_count > 0:
            evicted_keys = self._gather_rows(
                self.keys[:, :active_count], evicted_old
            ).contiguous()
            for start in range(0, final_count, self.config.work_chunk_size):
                end = min(start + self.config.work_chunk_size, final_count)
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
        kept_baseline = torch.gather(proposal_baseline, 1, keep_indices)
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
                raise RuntimeError("retained state exceeded its density-growth limit")

        candidate_metadata = torch.full(
            (groups, num_accepted),
            -1,
            device=self.device,
            dtype=self.source_index.dtype,
        )
        proposal_source = torch.cat(
            (self.source_index[:, :active_count], candidate_metadata), dim=1
        )
        proposal_insert = torch.cat(
            (
                self.insert_query_frame[:, :active_count],
                candidate_metadata.to(self.insert_query_frame.dtype),
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
                final_count, device=self.device
            ).unsqueeze(0).expand(groups, -1)
            selected_position = (keep_indices - active_count).clamp_min(0)
            original_candidate = torch.gather(
                selected_candidates,
                1,
                selected_position.clamp_max(num_accepted - 1),
            )
            group_index = torch.arange(
                groups, device=self.device
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

        ordered_density = torch.gather(candidate_external, 1, candidate_order)
        ordered_score = torch.gather(candidate_score, 1, candidate_order)
        ordered_target = torch.gather(candidate_target, 1, candidate_order)
        accepted_entries = ordered_target >= 0
        ordered_replacement_density = torch.where(
            accepted_entries,
            torch.gather(final_density, 1, ordered_target.clamp_min(0)),
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
            (groups,), added_count > 0, device=self.device, dtype=torch.bool
        )
        replaced = torch.full(
            (groups,), evict_count > 0, device=self.device, dtype=torch.bool
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
            candidate_order, dtype=torch.int32
        )
        stats.trace_entry_reason = torch.full_like(
            candidate_order, 9, dtype=torch.int8
        )
        stats.trace_anchor_slot = torch.full_like(candidate_order, -1)
        stats.trace_reference_density = torch.ones_like(ordered_score)
        stats.trace_group_reason = torch.full(
            (groups, 1), 9, device=self.device, dtype=torch.int8
        )
        stats.trace_group_accepted = accepted.unsqueeze(1)
        stats.trace_group_candidate_count = torch.full(
            (groups, 1),
            num_candidates,
            device=self.device,
            dtype=torch.int32,
        )
        stats.trace_group_added_energy = ordered_replacement_density.sum(
            dim=1, keepdim=True
        )
        stats.trace_group_removed_energy = evicted_density.sum(
            dim=1, keepdim=True
        )
        stats.trace_candidate_energy_share = ordered_score
        stats.trace_victim_energy_share = torch.full_like(
            ordered_score, growth_limit
        )
        stats.trace_gate_accepted = accepted_entries
        stats.trace_saturation_ratio = ordered_score
        stats.bootstrap_admitted_count = accepted_count
        stats.bootstrap_gate_mode = "insertion_density_growth"
        stats.bootstrap_gate_limit = growth_limit
        return stats

    @torch.no_grad()
    def recompute_density(self) -> torch.Tensor:
        """Rebuild the exact density table for initialization and tests."""
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
        keys, values = self._prepare_inputs(keys, values)
        count = min(int(keys.shape[0]), self.config.max_entries)
        self.clear()
        self.keys[:, :count].copy_(keys[:count].permute(1, 0, 2))
        self.values[:, :count].copy_(values[:count].permute(1, 0, 2))
        self.counts.fill_(count)
        self.recompute_density()
        self.density_baseline[:, :count].copy_(
            self.density[:, :count].clamp_min(
                self.config.density_baseline_floor
            )
        )
