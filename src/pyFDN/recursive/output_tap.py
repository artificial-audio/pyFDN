"""Output summation stage."""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import torch

from .stage import Stage

if TYPE_CHECKING:
    from .cost import CostContext, StageCost


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
            self.num_outputs = num_outputs
            self.num_lines = num_lines
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
            self.num_inputs = self.D.shape[1]
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

        return lines, y

    def estimate_cost(self, ctx: "CostContext") -> "StageCost":
        from .cost import StageCost, nbytes_for_shape

        B = int(ctx.batch_size)
        T = int(ctx.block_size)
        N_out = int(self.C.shape[0])
        N = int(self.C.shape[1])

        lines_shape = ctx.lines_in if ctx.lines_in is not None else (B, N, T)
        lines_bytes = nbytes_for_shape(lines_shape, ctx.dtype)
        c_bytes = int(self.C.numel()) * int(self.C.element_size())

        y_shape = ctx.y_shape if ctx.y_shape is not None else (B, N_out, T)
        y_bytes = nbytes_for_shape(y_shape, ctx.dtype)

        flops = 2 * B * T * N * N_out
        read_dsp = lines_bytes + c_bytes
        write_dsp = y_bytes

        if self.D is not None:
            N_in = int(self.D.shape[1])
            x_bytes = nbytes_for_shape((B, N_in, T), ctx.dtype)
            d_bytes = int(self.D.numel()) * int(self.D.element_size())
            flops += 2 * B * T * N_in * N_out + B * N_out * T
            read_dsp += x_bytes + d_bytes

        return StageCost(
            flops=float(flops),
            bytes_read_dsp=float(read_dsp),
            bytes_write_dsp=float(write_dsp),
            bytes_read_impl=float(read_dsp),
            bytes_write_impl=float(write_dsp),
            int_ops=0.0,
        )
