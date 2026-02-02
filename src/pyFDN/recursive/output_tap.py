"""Output summation stage."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import json

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
        """
        super().__init__(state_keys=set())  # Stateless
        
        if output_matrix is None:
            # Default: average all lines equally
            self.C = torch.ones(num_outputs, num_lines, dtype=torch.float32) / num_lines
        else:
            self.C = output_matrix.float()
            self.num_outputs, self.num_lines = self.C.shape
        
        if self.C.dim() != 2:
            raise ValueError(
                f"Output matrix must be 2D [N_out, N], got shape {self.C.shape}"
            )
        
        # Direct path is optional
        if direct_matrix is None:
            self.D = None
        else:
            self.D = direct_matrix.float()
            if self.D.shape[0] != self.num_outputs:
                raise ValueError(
                    f"Direct matrix must have {self.num_outputs} output channels, got shape {self.D.shape}"
                )
            if self.D.shape[1] != self.num_inputs:
                raise ValueError(
                    f"Direct matrix must have {self.num_inputs} input channels, got shape {self.D.shape}"
                )
            if self.D.dim() != 2:
                raise ValueError(
                    f"Direct matrix must be 2D [N_out, N_in], got shape {self.D.shape}"
                )
        
    
    def init_state(self, batch_size: int, block_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        # Move matrices to device
        self.C = self.C.to(device)
        if self.D is not None:
            self.D = self.D.to(device)
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
        Compute output as weighted sum of lines (and optional direct path).
        
        Returns the unchanged `lines` tensor together with the output block.
        """
        if lines is None:
            raise RuntimeError("OutputTap requires `lines` to be set")

        # Apply output matrix using einsum: [B, N, T] @ [N, N_out] -> [B, N_out, T]
        # einsum('bnt,no->bot', lines, self.C.T) computes lines @ C.T efficiently without transposing
        y = torch.einsum('bnt,no->bot', lines, self.C.T)  # [B, N_out, T]
        
        # Add direct path if present
        if self.D is not None:
            if x_block is None:
                raise RuntimeError(
                    "OutputTap with direct path requires external input `x_block`"
                )
            x = x_block  # [B, N_in, T]
            # Apply direct matrix using einsum: [B, N_in, T] @ [N_in, N_out] -> [B, N_out, T]
            direct = torch.einsum('bnt,no->bot', x, self.D.T)  # [B, N_out, T]
            y = y + direct

        # region agent log
        try:
            with open("/Users/wu12recu/Documents/GitHub/pyFDN/.cursor/debug.log", "a") as _f:
                _f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "post-refactor",
                    "hypothesisId": "H3",
                    "location": "recursive/output_tap.py:OutputTap.step_block",
                    "message": "OutputTap produced block output",
                    "data": {
                        "block_size": int(block_size),
                        "lines_abs_max": float(lines.abs().max().item()),
                        "y_abs_max": float(y.abs().max().item()),
                    },
                    "timestamp": __import__("time").time(),
                }) + "\n")
        except Exception:
            pass
        # endregion

        return lines, y
