"""Parallel biquad filter bank stage."""

from __future__ import annotations
from typing import Dict, Optional
import torch

from .stage import Stage


class Biquads(Stage):
    """
    Parallel bank of biquad IIR filters applied to feedback lines.
    
    This stage:
    - Operates on ctx["lines"] (or another specified context key)
    - Applies biquad filtering to each line independently
    - Maintains IIR filter state across blocks
    - Modifies ctx["lines"] in-place
    
    Filter structure: Direct Form I biquad
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    
    State per line: [x[n-1], x[n-2], y[n-1], y[n-2]]
    """
    
    def __init__(
        self,
        num_lines: int = 4,
        biquad_coeffs: Optional[torch.Tensor] = None,
        context_key: str = "lines"
    ):
        """
        Initialize parallel biquad filter bank.
        
        Args:
            num_lines: Number of filter lines (N)
            biquad_coeffs: Filter coefficients of shape [N, 5] or [N, num_sections, 5]
                          where each row is [b0, b1, b2, a1, a2]
                          If None, creates simple one-pole lowpass filters
            context_key: Context dictionary key to filter (default: "lines")
        """
        super().__init__(state_keys={"biquad_state"})
        self.num_lines = num_lines
        self.context_key = context_key
        
        # Initialize filter coefficients
        if biquad_coeffs is None:
            # Default: simple one-pole lowpass (y[n] = 0.9*y[n-1] + 0.1*x[n])
            # As biquad: b0=0.1, b1=0, b2=0, a1=-0.9, a2=0
            self.coeffs = torch.tensor(
                [[0.1, 0.0, 0.0, -0.9, 0.0]],
                dtype=torch.float32
            ).repeat(num_lines, 1)  # [N, 5]
            self.num_sections = 1
        else:
            self.coeffs = biquad_coeffs.float()
            if self.coeffs.dim() == 2:
                # [N, 5] -> add section dimension
                self.coeffs = self.coeffs.unsqueeze(1)  # [N, 1, 5]
            self.num_sections = self.coeffs.shape[1]
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize biquad filter states.
        
        State shape: [B, N, num_sections, 4] for [x[n-1], x[n-2], y[n-1], y[n-2]]
        """
        # Move coefficients to device
        self.coeffs = self.coeffs.to(device)
        
        return {
            "biquad_state": torch.zeros(
                batch_size, self.num_lines, self.num_sections, 4,
                device=device, dtype=torch.float32
            )
        }
    
    def step_block(
        self,
        ctx: Dict[str, torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int
    ) -> None:
        """
        Apply biquad filtering to lines in-place.
        
        Filters ctx[self.context_key] and updates it with filtered output.
        """
        if self.context_key not in ctx:
            raise RuntimeError(
                f"Biquads requires ctx['{self.context_key}'] to be set"
            )
        
        x = ctx[self.context_key]  # [B, T, N]
        filter_state = state_t["biquad_state"].clone()  # [B, N, num_sections, 4]
        
        B, T, N = x.shape
        
        # Process each section sequentially (cascaded biquads)
        y = x.clone()
        
        for section_idx in range(self.num_sections):
            # Get coefficients for this section
            b0, b1, b2, a1, a2 = self.coeffs[:, section_idx].unbind(dim=1)  # Each: [N]
            
            # Get state for this section: [B, N, 4]
            state = filter_state[:, :, section_idx, :]
            x_state = state[:, :, :2]  # [B, N, 2] - [x[n-1], x[n-2]]
            y_state = state[:, :, 2:]  # [B, N, 2] - [y[n-1], y[n-2]]
            
            # Process block sample by sample (IIR requires sequential processing)
            output = torch.zeros_like(y)  # [B, T, N]
            
            for t in range(T):
                x_n = y[:, t, :]  # [B, N] - input for this section
                
                # Compute output: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
                y_n = (
                    b0 * x_n
                    + b1 * x_state[:, :, 0]
                    + b2 * x_state[:, :, 1]
                    + a1 * y_state[:, :, 0]  # Note: a1 already has negative sign in coeffs
                    + a2 * y_state[:, :, 1]
                )
                
                output[:, t, :] = y_n
                
                # Update state (shift)
                x_state[:, :, 1] = x_state[:, :, 0]  # x[n-2] = x[n-1]
                x_state[:, :, 0] = x_n               # x[n-1] = x[n]
                y_state[:, :, 1] = y_state[:, :, 0]  # y[n-2] = y[n-1]
                y_state[:, :, 0] = y_n               # y[n-1] = y[n]
            
            # Update state for this section
            filter_state[:, :, section_idx, :2] = x_state
            filter_state[:, :, section_idx, 2:] = y_state
            
            # Output of this section becomes input to next section
            y = output
        
        # Update context in-place
        ctx[self.context_key] = y
        
        # Save updated state
        next_state["biquad_state"] = filter_state
