"""Feedback mixing stage."""

from __future__ import annotations
from typing import Dict, Optional
import torch

from .stage import Stage


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
        num_lines: int = 4
    ):
        """
        Initialize feedback mixing stage.
        
        Args:
            feedback_matrix: Feedback matrix A of shape [N, N]
                           If None, creates identity matrix (no mixing)
            num_lines: Number of feedback lines (N), used if feedback_matrix is None
        """
        super().__init__(state_keys=set())  # Stateless
        
        if feedback_matrix is None:
            # Default: identity matrix (no mixing)
            self.A = torch.eye(num_lines, dtype=torch.float32)
        else:
            self.A = feedback_matrix.float()
        
        if self.A.dim() != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(
                f"Feedback matrix must be square, got shape {self.A.shape}"
            )
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        # Move matrix to device
        self.A = self.A.to(device)
        return {}
    
    def step_block(
        self,
        ctx: Dict[str, torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int
    ) -> None:
        """
        Apply feedback matrix to lines.
        
        Computes: ctx["lines"] = ctx["lines"] @ A.T
        """
        if "lines" not in ctx:
            raise RuntimeError("FeedbackMix requires ctx['lines'] to be set")
        
        lines = ctx["lines"]  # [B, N, T]
        
        # Apply feedback matrix using einsum: [B, N, T] @ [N, N] -> [B, N, T]
        # einsum('bnt,nm->bmt') computes lines @ A.T efficiently without transposing
        mixed = torch.einsum('bnt,nm->bmt', lines, self.A.T)  # [B, N, T]
        
        # Update in-place
        ctx["lines"] = mixed
