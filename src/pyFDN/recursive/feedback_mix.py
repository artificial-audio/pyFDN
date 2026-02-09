"""Feedback mixing stage."""

from __future__ import annotations
import math
from typing import Dict, Literal, Optional, Tuple, TYPE_CHECKING
import torch

from .stage import Stage

if TYPE_CHECKING:
    from .cost import CostContext, StageCost


class FeedbackMix(Stage):
    """
    Applies feedback matrix to mix between delay lines.
    
    This stage:
    - Reads ctx["lines"]
    - Applies feedback matrix A
    - Writes result back to ctx["lines"] (in-place modification)
    - Is purely feedforward (no state)
    
    Operation: ctx["lines"] = ctx["lines"] @ A.T
    """
    
    def __init__(
        self,
        feedback_matrix: Optional[torch.Tensor] = None,
        num_lines: int = 4,
        mix_type: Literal["dense", "hadamard", "householder"] = "dense",
        householder_vector: Optional[torch.Tensor] = None,
    ):
        """
        Initialize feedback mixing stage.
        
        Args:
            feedback_matrix: Feedback matrix A of shape [N, N]
                           If None, creates identity matrix (no mixing)
            num_lines: Number of feedback lines (N), used if feedback_matrix is None
            mix_type: Matrix structure:
                - "dense": generic dense matrix multiply (default)
                - "hadamard": normalized Hadamard transform, no matrix input required
                - "householder": Householder reflection defined by a vector
            householder_vector: Householder vector v of shape [N] or [N, 1]
                               used when mix_type="householder"
        """
        super().__init__(state_keys=set())  # Stateless
        self.mix_type = mix_type

        if self.mix_type == "dense":
            if feedback_matrix is None:
                # Default: identity matrix (no mixing)
                self.A = torch.eye(num_lines, dtype=torch.float32)
            else:
                self.A = feedback_matrix.float()
            if self.A.dim() != 2 or self.A.shape[0] != self.A.shape[1]:
                raise ValueError(
                    f"Feedback matrix must be square, got shape {self.A.shape}"
                )
            self.num_lines = int(self.A.shape[0])
            self.v = None
            self._v_norm_sq = None

        elif self.mix_type == "hadamard":
            if feedback_matrix is not None:
                raise ValueError("mix_type='hadamard' does not use feedback_matrix")
            if num_lines < 1 or (num_lines & (num_lines - 1)) != 0:
                raise ValueError(
                    f"Hadamard mix requires num_lines to be a power of two, got {num_lines}"
                )
            self.num_lines = int(num_lines)
            self._hadamard_scale = 1.0 / math.sqrt(float(self.num_lines))
            self.A = None
            self.v = None
            self._v_norm_sq = None

        elif self.mix_type == "householder":
            if householder_vector is None:
                raise ValueError("mix_type='householder' requires householder_vector")
            if feedback_matrix is not None:
                raise ValueError("mix_type='householder' does not use feedback_matrix")
            v = householder_vector.float().reshape(-1)
            if v.numel() == 0:
                raise ValueError("householder_vector must have at least one element")
            v_norm_sq = torch.dot(v, v)
            if float(v_norm_sq.item()) <= 0.0:
                raise ValueError("householder_vector must have non-zero norm")
            self.v = v
            self._v_norm_sq = v_norm_sq
            self.num_lines = int(v.numel())
            self.A = None

        else:
            raise ValueError(
                f"Unknown mix_type '{mix_type}'. Expected 'dense', 'hadamard', or 'householder'."
            )
    
    def init_state(self, batch_size: int, block_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        if self.A is not None:
            self.A = self.A.to(device)
        if self.v is not None:
            self.v = self.v.to(device)
            self._v_norm_sq = self._v_norm_sq.to(device)
        return {}

    def _apply_hadamard(self, lines: torch.Tensor) -> torch.Tensor:
        """Fast Walsh-Hadamard transform along the line axis with orthonormal scaling."""
        mixed = lines
        batch_size, n, block_size = mixed.shape
        h = 1
        while h < n:
            mixed = mixed.reshape(batch_size, n // (2 * h), 2, h, block_size)
            a = mixed[:, :, 0, :, :]
            b = mixed[:, :, 1, :, :]
            mixed = torch.stack((a + b, a - b), dim=2).reshape(batch_size, n, block_size)
            h *= 2
        return mixed * self._hadamard_scale
    
    def step_block(
        self,
        lines: Optional[torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int,
        x_block: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply feedback matrix to lines.
        
        Computes: lines = lines @ A.T
        """
        if lines is None:
            raise RuntimeError("FeedbackMix requires `lines` to be set")

        if int(lines.shape[1]) != self.num_lines:
            raise ValueError(
                f"Expected lines with N={self.num_lines}, got shape {tuple(lines.shape)}"
            )

        if self.mix_type == "dense":
            # Apply dense feedback matrix: [B, N, T] @ [N, N] -> [B, N, T]
            # einsum('bnt,nm->bmt') computes lines @ A.T efficiently without transposing
            mixed = torch.einsum("bnt,nm->bmt", lines, self.A.T)  # [B, N, T]

        elif self.mix_type == "hadamard":
            # Apply normalized Hadamard transform in O(N log N) using FWHT.
            mixed = self._apply_hadamard(lines)

        else:  # self.mix_type == "householder"
            # Apply Householder reflection: x -> x - 2 v (v^T x) / (v^T v)
            proj = torch.einsum("bnt,n->bt", lines, self.v)
            correction = (2.0 / self._v_norm_sq) * torch.einsum("bt,n->bnt", proj, self.v)
            mixed = lines - correction

        return mixed, None

    def estimate_cost(self, ctx: "CostContext") -> "StageCost":
        from .cost import StageCost, nbytes_for_shape

        B = int(ctx.batch_size)
        T = int(ctx.block_size)
        N = int(self.num_lines)

        lines_shape = ctx.lines_in if ctx.lines_in is not None else (B, N, T)
        lines_bytes = nbytes_for_shape(lines_shape, ctx.dtype)
        matrix_bytes = 0

        if self.mix_type == "dense":
            matrix_bytes = int(self.A.numel()) * int(self.A.element_size())
            flops = float(2 * B * T * N * N)
        elif self.mix_type == "hadamard":
            flops = float(B * T * N * (math.log2(N) + 1.0))
        else:  # householder
            vec_bytes = int(self.v.numel()) * int(self.v.element_size())
            matrix_bytes = vec_bytes
            flops = float(6 * B * T * N)

        read_dsp = float(lines_bytes + matrix_bytes)
        write_dsp = float(lines_bytes)

        return StageCost(
            flops=flops,
            bytes_read_dsp=read_dsp,
            bytes_write_dsp=write_dsp,
            bytes_read_impl=read_dsp,
            bytes_write_impl=write_dsp,
            int_ops=0.0,
        )
