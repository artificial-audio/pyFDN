"""Input injection stage."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from .stage import Stage


class InputTap(Stage):
    """
    Computes injected signal from external input for feedback lines.
    
    This stage:
    - Reads ctx["x"] (external input)
    - Does NOT require ctx["lines"]
    - Produces ctx["inject"] (input contribution for all lines)
    - Is purely feedforward (no state)
    
    Operation: ctx["inject"] = ctx["x"] @ B.T

    The injection is then added to the delay-line signals by `DelayRead`,
    which must appear later in the stage list. This allows `InputTap`
    to be placed as the **first stage** in the recursion core.
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
            self.num_lines = self.B.shape[0]
            self.num_inputs = self.B.shape[1]
        
        if self.B.dim() != 2:
            raise ValueError(
                f"Input matrix must be 2D [N, N_in], got shape {self.B.shape}"
            )
    
    def init_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        # Move matrix to device
        self.B = self.B.to(device)
        return {}
    
    def step_block(
        self,
        lines: Optional[torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int,
        x_block: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute injected signal from external input and add it to the lines.
        
        If `lines` is None, a new tensor is created from the injected signal.
        Otherwise the injected signal is added to the existing `lines`.
        
        Operation:
            inject = x_block @ B.T
            lines  = (lines or 0) + inject
        """
        if x_block is None:
            raise RuntimeError("InputTap requires external input `x_block` to be provided")

        # Apply input matrix using einsum: [B, N_in, T] @ [N_in, N] -> [B, N, T]
        # einsum('bnt,nm->bmt') computes x @ B.T efficiently without transposing
        inject = torch.einsum('bnt,nm->bmt', x_block, self.B.T)  # [B, N, T]

        if lines is None:
            new_lines = inject
        else:
            if lines.shape != inject.shape:
                raise RuntimeError(
                    f"InputTap: existing lines have shape {lines.shape}, "
                    f"but injected signal has shape {inject.shape}"
                )
            new_lines = lines + inject

        return new_lines, None
