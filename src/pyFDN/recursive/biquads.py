"""Parallel biquad filter bank stage."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from .stage import Stage


class Biquads(Stage):
    """
    Parallel bank of biquad IIR filters applied to feedback lines.

    This stage:
    - Operates on the `lines` tensor (feedback-line signals for the current block)
    - Applies biquad filtering to each line independently
    - Maintains IIR filter state across blocks
    - Returns the filtered lines for the next stage

    Filter structure: Transposed Direct Form II biquad
        a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    State per line: [z1, z2] (two delay elements for DF2T)
    """

    def __init__(
        self,
        num_lines: int = 4,
        biquad_coeffs: Optional[torch.Tensor] = None,
    ):
        """
        Initialize parallel biquad filter bank.

        Args:
            num_lines: Number of filter lines (N)
            biquad_coeffs: Filter coefficients of shape [N, 6] or [N, num_sections, 6]
                          where each row is [a0, a1, a2, b0, b1, b2]
                          If None, creates simple one-pole lowpass filters
        """
        super().__init__(state_keys={"biquad_state"})
        self.num_lines = num_lines
        
        # Initialize filter coefficients
        if biquad_coeffs is None:
            # Default: simple one-pole lowpass (y[n] = 0.9*y[n-1] + 0.1*x[n])
            # As biquad: a0=1.0, a1=-0.9, a2=0, b0=0.1, b1=0, b2=0
            self.coeffs = torch.tensor(
                [[1.0, -0.9, 0.0, 0.1, 0.0, 0.0]],
                dtype=torch.float32
            ).repeat(num_lines, 1)  # [N, 6]
            # Add section dimension to match expected 3D shape [N, num_sections, 6]
            self.coeffs = self.coeffs.unsqueeze(1)  # [N, 1, 6]
            self.num_sections = 1
        else:
            self.coeffs = biquad_coeffs.float()
            if self.coeffs.dim() == 2:
                # [N, 6] -> add section dimension
                self.coeffs = self.coeffs.unsqueeze(1)  # [N, 1, 6]
            if self.coeffs.shape[-1] != 6:
                raise ValueError(
                    f"Biquad coefficients must have 6 values [a0, a1, a2, b0, b1, b2], "
                    f"got {self.coeffs.shape[-1]} values"
                )
            self.num_sections = self.coeffs.shape[1]
    
    def init_state(self, batch_size: int, block_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize biquad filter states.
        
        State shape: [B, N, num_sections, 2] for DF2T states [z1, z2]
        """
        # Move coefficients to device
        self.coeffs = self.coeffs.to(device)
        
        return {
            "biquad_state": torch.zeros(
                batch_size, self.num_lines, self.num_sections, 2,
                device=device, dtype=torch.float32
            )
        }
    
    def step_block(
        self,
        lines: Optional[torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int,
        x_block: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply biquad filtering to the feedback-line tensor.

        Reads `lines` [B, N, T], updates biquad state in `next_state`, and
        returns the filtered lines for the next stage.
        """
        if lines is None:
            raise RuntimeError("Biquads requires `lines` to be set")

        x = lines  # [B, N, T]
        filter_state = state_t["biquad_state"].clone()  # [B, N, num_sections, 2]
        
        B, N, T = x.shape
        
        # Process each section sequentially (cascaded biquads)
        y = x.clone()
        
        for section_idx in range(self.num_sections):
            # Get coefficients for this section: [a0, a1, a2, b0, b1, b2]
            a0, a1, a2, b0, b1, b2 = self.coeffs[:, section_idx].unbind(dim=1)  # Each: [N]
            
            # Get DF2T state for this section: [B, N, 2] -> [z1, z2]
            state = filter_state[:, :, section_idx, :]
            z1 = state[:, :, 0]  # [B, N]
            z2 = state[:, :, 1]  # [B, N]

            # Normalize coefficients by a0 for stable DF2T update
            inv_a0 = 1.0 / a0  # [N]
            b0n = b0 * inv_a0
            b1n = b1 * inv_a0
            b2n = b2 * inv_a0
            a1n = a1 * inv_a0
            a2n = a2 * inv_a0
            
            # Process block sample by sample (IIR requires sequential processing)
            output = torch.zeros_like(y)  # [B, N, T]
            
            for t in range(T):
                x_n = y[:, :, t]  # [B, N] - input for this section

                # Transposed Direct Form II:
                # y[n]  = b0n*x[n] + z1
                # z1'   = b1n*x[n] - a1n*y[n] + z2
                # z2'   = b2n*x[n] - a2n*y[n]
                y_n = b0n.unsqueeze(0) * x_n + z1
                z1 = b1n.unsqueeze(0) * x_n - a1n.unsqueeze(0) * y_n + z2
                z2 = b2n.unsqueeze(0) * x_n - a2n.unsqueeze(0) * y_n

                output[:, :, t] = y_n
            
            # Update state for this section
            filter_state[:, :, section_idx, 0] = z1
            filter_state[:, :, section_idx, 1] = z2
            
            # Output of this section becomes input to next section
            y = output
        
        # Save updated state
        next_state["biquad_state"] = filter_state

        return y, None
