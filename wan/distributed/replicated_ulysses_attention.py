"""Ulysses attention for replicated causal Wan inference.

The causal and historical-memory state remains replicated on every rank. Only the expensive
attention kernel is sequence/head parallel:

  [S, H] --slice S--> [S/P, H] --all-to-all--> [S, H/P]

The result is exchanged back and all-gathered over sequence so the surrounding
transformer block keeps its ordinary replicated layout. This makes the helper
compatible with the existing LongLive-RAG cache lifecycle.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


_GROUP: dist.ProcessGroup | None = None
_WORLD_SIZE = 1
_RANK = 0
_ORIGINAL_ATTENTION = None


def initialize_replicated_ulysses(group: dist.ProcessGroup) -> None:
    global _GROUP, _WORLD_SIZE, _RANK
    _GROUP = group
    _WORLD_SIZE = dist.get_world_size(group)
    _RANK = dist.get_rank(group)


def _require_initialized() -> dist.ProcessGroup:
    if _GROUP is None or _WORLD_SIZE <= 1:
        raise RuntimeError("replicated Ulysses group is not initialized")
    return _GROUP


def _all_to_all(tensor: torch.Tensor, *, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    group = _require_initialized()
    if tensor.shape[scatter_dim] % _WORLD_SIZE != 0:
        raise ValueError(
            f"dimension {scatter_dim} with length {tensor.shape[scatter_dim]} "
            f"must be divisible by Ulysses world size {_WORLD_SIZE}"
        )
    send = [chunk.contiguous() for chunk in torch.chunk(tensor, _WORLD_SIZE, dim=scatter_dim)]
    recv = [torch.empty_like(send[0]) for _ in range(_WORLD_SIZE)]
    dist.all_to_all(recv, send, group=group)
    return torch.cat(recv, dim=gather_dim)


def _seq_to_head(tensor: torch.Tensor) -> torch.Tensor:
    return _all_to_all(tensor, scatter_dim=2, gather_dim=1)


def _head_to_seq(tensor: torch.Tensor) -> torch.Tensor:
    return _all_to_all(tensor, scatter_dim=1, gather_dim=2)


def replicated_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run exact full attention while distributing heads across SP ranks."""
    group = _require_initialized()
    if q_lens is not None or k_lens is not None:
        raise NotImplementedError("variable-length attention is not used by causal rollout")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q/k/v shaped [B, S, H, D]")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("q/k/v head counts must match")
    if q.shape[2] % _WORLD_SIZE != 0:
        raise ValueError(
            f"head count {q.shape[2]} must be divisible by Ulysses world size {_WORLD_SIZE}"
        )
    if q.shape[1] % _WORLD_SIZE != 0 or k.shape[1] % _WORLD_SIZE != 0:
        raise ValueError(
            f"query/key lengths {(q.shape[1], k.shape[1])} must be divisible by "
            f"Ulysses world size {_WORLD_SIZE}"
        )

    q_width = q.shape[1] // _WORLD_SIZE
    k_width = k.shape[1] // _WORLD_SIZE
    q_local = q[:, _RANK * q_width:(_RANK + 1) * q_width].contiguous()
    k_local = k[:, _RANK * k_width:(_RANK + 1) * k_width].contiguous()
    v_local = v[:, _RANK * k_width:(_RANK + 1) * k_width].contiguous()

    q_heads = _seq_to_head(q_local)
    k_heads = _seq_to_head(k_local)
    v_heads = _seq_to_head(v_local)
    attention_fn = _ORIGINAL_ATTENTION if _ORIGINAL_ATTENTION is not None else original_attention()
    out_heads = attention_fn(q_heads, k_heads, v_heads, **kwargs)
    out_local = _head_to_seq(out_heads)

    gathered = [torch.empty_like(out_local) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, out_local, group=group)
    return torch.cat(gathered, dim=1)


def install_replicated_ulysses_attention(group: dist.ProcessGroup) -> None:
    """Patch only latent-memory self-attention's module-global kernel."""
    global _ORIGINAL_ATTENTION
    initialize_replicated_ulysses(group)
    import wan.modules.causal_model_latentmem as causal_latentmem

    if _ORIGINAL_ATTENTION is None:
        _ORIGINAL_ATTENTION = causal_latentmem.attention
    causal_latentmem.attention = replicated_ulysses_attention


def uninstall_replicated_ulysses_attention() -> None:
    """Restore ordinary attention after a sequence-parallel inference phase."""
    if _ORIGINAL_ATTENTION is None:
        return
    import wan.modules.causal_model_latentmem as causal_latentmem

    causal_latentmem.attention = _ORIGINAL_ATTENTION


def original_attention():
    if _ORIGINAL_ATTENTION is None:
        from wan.modules.attention import attention

        return attention
    return _ORIGINAL_ATTENTION
