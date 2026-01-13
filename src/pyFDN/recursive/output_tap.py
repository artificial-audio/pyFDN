"""Output summation stage."""

from __future__ import annotations
from typing import Dict, Optional
import torch

from .stage import Stage


class OutputTap(Stage):
    """
    Produces final output by summing weighted line signals.
    
    This stage:
    - Reads ctx["lines"]
    - Optionally reads ctx["x"] for direct path
    - Applies output matrix C (and optional direct matrix D)
    - Writes ctx["y"] as final output
    - Is purely feedforward (no state)
    
    Operation: ctx["y"] = ctx["lines"] @ C.T [+ ctx["x"] @ D.T]
    """
    
    def __init__(
        self,
        output_matrix: Optional[torch.Tensor] = None,
        direct_matrix: Optional[torch.Tensor] = None,
        num_lines: int = 4,
        num_outputs: int = 1,
        num_inputs: int = 1
    ):
        """
        Initialize output summation stage.
        
        Args:
            output_matrix: Output gain matrix C of shape [N_out, N]
                         If None, creates averaging matrix (all lines contribute equally)
            direct_matrix: Direct-path matrix D of shape [N_out, N_in]
                         If None, no direct path is used
            num_lines: Number of feedback lines (N), used if output_matrix is None
            num_outputs: Number of output channels (N_out), used if matrices are None
            num_inputs: Number of input channels (N_in), used if direct_matrix is None
        """
        super().__init__(state_keys=set())  # Stateless
        
        if output_matrix is None:
            # Default: average all lines equally
            self.C = torch.ones(num_outputs, num_lines, dtype=torch.float32) / num_lines
        else:
            self.C = output_matrix.float()
        
        if self.C.dim() != 2:
            raise ValueError(
                f"Output matrix must be 2D [N_out, N], got shape {self.C.shape}"
            )
        
        # Direct path is optional
        if direct_matrix is None:
            self.D = None
        else:
            self.D = direct_matrix.float()
            if self.D.dim() != 2:
                raise ValueError(
                    f"Direct matrix must be 2D [N_out, N_in], got shape {self.D.shape}"
                )
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        # Move matrices to device
        self.C = self.C.to(device)
        if self.D is not None:
            self.D = self.D.to(device)
        return {}
    
    def step_block(
        self,
        ctx: Dict[str, torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int
    ) -> None:
        """
        Compute output as weighted sum of lines (and optional direct path).
        
        Computes: ctx["y"] = ctx["lines"] @ C.T [+ ctx["x"] @ D.T]
        """
        if "lines" not in ctx:
            raise RuntimeError("OutputTap requires ctx['lines'] to be set")
        
        lines = ctx["lines"]  # [B, T, N]
        
        # Apply output matrix: [B, T, N] @ [N, N_out] -> [B, T, N_out]
        y = torch.matmul(lines, self.C.T)
        
        # Add direct path if present
        if self.D is not None:
            if "x" not in ctx:
                raise RuntimeError(
                    "OutputTap with direct path requires ctx['x'] to be set"
                )
            x = ctx["x"]  # [B, T, N_in]
            # [B, T, N_in] @ [N_in, N_out] -> [B, T, N_out]
            y = y + torch.matmul(x, self.D.T)
        
        ctx["y"] = y
