"""Optional Triton reduction for Soft-Riesz density."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is unavailable on CPU hosts
    triton = None
    tl = None


def _ensure_python_include_for_triton() -> None:
    include_dir = (
        Path(sys.prefix)
        / "include"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    if not (include_dir / "Python.h").exists():
        return
    entries = [entry for entry in os.environ.get("CPATH", "").split(os.pathsep) if entry]
    if str(include_dir) not in entries:
        os.environ["CPATH"] = os.pathsep.join([str(include_dir), *entries])


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
    groups, num_rows, num_columns = distances.shape
    if exclude_diagonal and num_rows != num_columns:
        return None
    if num_columns == 0:
        return torch.zeros(
            groups,
            num_rows,
            device=distances.device,
            dtype=torch.float32,
        )

    _ensure_python_include_for_triton()
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
