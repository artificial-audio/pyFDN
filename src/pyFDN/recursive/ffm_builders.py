"""Helpers for building cascaded filter-feedback matrices (FFMs)."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .delay_lines import DiagonalDelay
from .feedback_mix import FeedbackMix
from .stage import Stage


def build_ffm_stages(
    mix_matrices: Sequence[torch.Tensor],
    delay_vectors: Sequence[torch.Tensor | Sequence[int]],
    *,
    state_prefix: str = "ffm",
) -> List[Stage]:
    """
    Build a cascaded FFM stage list: U0 -> D(m1) -> U1 -> D(m2) -> ... -> Uk.

    This matches the common paper structure:
        F_k(z) = U_k D_{m_k}(z) F_{k-1}(z), with F_0(z) = U_0

    Args:
        mix_matrices: [U0, U1, ..., UK], each of shape [N, N]
        delay_vectors: [m1, ..., mK], each of shape [N] (or any 1D sequence)
        state_prefix: Prefix for delay state keys ("{state_prefix}_d{idx}_buffers/pointer")

    Returns:
        A list of Stage instances in execution order.
    """
    if len(mix_matrices) != len(delay_vectors) + 1:
        raise ValueError(
            "mix_matrices must have exactly one more element than delay_vectors "
            f"(got {len(mix_matrices)} mix matrices and {len(delay_vectors)} delay vectors)"
        )

    if len(mix_matrices) == 0:
        return []

    n = int(mix_matrices[0].shape[0])
    if mix_matrices[0].ndim != 2 or mix_matrices[0].shape[0] != mix_matrices[0].shape[1]:
        raise ValueError(f"mix_matrices[0] must be square [N,N], got shape {tuple(mix_matrices[0].shape)}")

    for idx, u in enumerate(mix_matrices):
        if u.ndim != 2 or u.shape[0] != u.shape[1]:
            raise ValueError(f"mix_matrices[{idx}] must be square [N,N], got shape {tuple(u.shape)}")
        if int(u.shape[0]) != n:
            raise ValueError(
                f"All mix_matrices must have the same N; mix_matrices[0] has N={n} "
                f"but mix_matrices[{idx}] has shape {tuple(u.shape)}"
            )

    stages: List[Stage] = [FeedbackMix(feedback_matrix=mix_matrices[0])]

    for k, (m_k, u_k) in enumerate(zip(delay_vectors, mix_matrices[1:]), start=1):
        m_k_t = torch.as_tensor(m_k, dtype=torch.long)
        if m_k_t.ndim != 1:
            raise ValueError(f"delay_vectors[{k}] must be 1D [N], got shape {tuple(m_k_t.shape)}")
        if int(m_k_t.numel()) != n:
            raise ValueError(
                f"delay_vectors[{k}] must have length N={n}, got length {int(m_k_t.numel())}"
            )
        stages.append(DiagonalDelay(m_k_t, state_key=f"{state_prefix}_d{k}"))
        stages.append(FeedbackMix(feedback_matrix=u_k))

    return stages

