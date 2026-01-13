"""Input injection stage."""

from __future__ import annotations
from typing import Dict, Optional
import torch

from .stage import Stage


class InputTap(Stage):
    """
    Injects external input into feedback lines.
    
    This stage:
    - Reads ctx["x"] (external input)
    - Reads ctx["lines"] (current line signals)
    - Adds weighted input to lines
    - Writes result back to ctx["lines"] (in-place modification)
    - Is purely feedforward (no state)
    
    Operation: ctx["lines"] = ctx["lines"] + ctx["x"] @ B.T
    """
    
    def __init__(
        self,
        input_matrix: Optional[torch.Tensor] = None,
        num_lines: int = 4,
        num_inputs: int = 1
    ):
        """
        Initialize input injection stage.
        
        Args:
            input_matrix: Input gain matrix B of shape [N, N_in]
                        If None, creates a matrix that feeds input to all lines equally
            num_lines: Number of feedback lines (N), used if input_matrix is None
            num_inputs: Number of input channels (N_in), used if input_matrix is None
        """
        super().__init__(state_keys=set())  # Stateless
        
        if input_matrix is None:
            # Default: feed input to all lines with gain 1.0
            self.B = torch.ones(num_lines, num_inputs, dtype=torch.float32)
        else:
            self.B = input_matrix.float()
        
        if self.B.dim() != 2:
            raise ValueError(
                f"Input matrix must be 2D [N, N_in], got shape {self.B.shape}"
            )
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        # Move matrix to device
        self.B = self.B.to(device)
        return {}
    
    def step_block(
        self,
        ctx: Dict[str, torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int
    ) -> None:
        """
        Add weighted external input to lines.
        
        Computes: ctx["lines"] = ctx["lines"] + ctx["x"] @ B.T
        """
        if "lines" not in ctx:
            raise RuntimeError("InputTap requires ctx['lines'] to be set")
        if "x" not in ctx:
            raise RuntimeError("InputTap requires ctx['x'] to be set")
        
        lines = ctx["lines"]  # [B, T, N]
        x = ctx["x"]          # [B, T, N_in]
        
        # Apply input matrix: [B, T, N_in] @ [N_in, N] -> [B, T, N]
        input_contrib = torch.matmul(x, self.B.T)
        
        # Add to lines in-place
        ctx["lines"] = lines + input_contrib
