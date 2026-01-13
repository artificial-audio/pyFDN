"""Delay line stages for feedback delays."""

from __future__ import annotations
from typing import Dict
import torch

from .stage import Stage


class DelayRead(Stage):
    """
    Reads delayed samples from circular delay buffers.
    
    This stage:
    - Reads from the shared "delay_buffers" state
    - Produces ctx["lines"] with delayed samples
    - Should appear first in the stage pipeline
    
    Shares state with DelayWrite stage.
    """
    
    def __init__(
        self,
        delay_length: int = 1024,
        num_lines: int = 4
    ):
        """
        Initialize delay read stage.
        
        Args:
            delay_length: Length of delay buffer in samples (L)
            num_lines: Number of delay lines (N)
        """
        super().__init__(state_keys={"delay_buffers", "delay_pointer"})
        self.delay_length = delay_length
        self.num_lines = num_lines
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize delay buffers and pointer."""
        return {
            "delay_buffers": torch.zeros(
                batch_size, self.delay_length, self.num_lines,
                device=device, dtype=torch.float32
            ),
            "delay_pointer": torch.zeros(
                batch_size, dtype=torch.long, device=device
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
        Read delayed samples from buffers.
        
        Produces ctx["lines"] of shape [B, T, N] containing delayed samples.
        """
        buffers = state_t["delay_buffers"]  # [B, L, N]
        pointer = state_t["delay_pointer"]  # [B]
        
        B, L, N = buffers.shape
        T = block_size
        
        # Generate indices for reading: (pointer + 0), (pointer + 1), ..., (pointer + T-1)
        # All modulo L
        time_offsets = torch.arange(T, device=buffers.device)  # [T]
        read_indices = (pointer.unsqueeze(1) + time_offsets.unsqueeze(0)) % L  # [B, T]
        
        # Gather samples from buffers
        # We need to expand indices to match the buffer shape for gathering
        read_indices_expanded = read_indices.unsqueeze(2).expand(B, T, N)  # [B, T, N]
        
        # Gather along the L dimension (dim=1 of buffers)
        lines = torch.gather(buffers, 1, read_indices_expanded)  # [B, T, N]
        
        ctx["lines"] = lines
        
        # Note: We don't update the pointer here - that's done by DelayWrite


class DelayWrite(Stage):
    """
    Writes processed samples back into circular delay buffers.
    
    This stage:
    - Reads ctx["lines"] (processed samples)
    - Writes to the shared "delay_buffers" state
    - Advances the buffer pointer
    - Should appear after all processing stages
    
    Shares state with DelayRead stage.
    """
    
    def __init__(self):
        """
        Initialize delay write stage.
        
        Note:
            Delay parameters (length, num_lines) are determined by the shared
            state initialized by DelayRead.
        """
        super().__init__(state_keys={"delay_buffers", "delay_pointer"})
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        DelayWrite doesn't initialize state - it shares state with DelayRead.
        
        Returns empty dict.
        """
        return {}
    
    def step_block(
        self,
        ctx: Dict[str, torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int
    ) -> None:
        """
        Write processed samples to buffers and advance pointer.
        
        Reads ctx["lines"] of shape [B, T, N] and writes to delay buffers.
        """
        if "lines" not in ctx:
            raise RuntimeError("DelayWrite requires ctx['lines'] to be set by previous stages")
        
        lines = ctx["lines"]  # [B, T, N]
        buffers = state_t["delay_buffers"].clone()  # [B, L, N] - clone to avoid mutating state_t
        pointer = state_t["delay_pointer"]  # [B]
        
        B, T, N = lines.shape
        L = buffers.shape[1]
        
        # Generate indices for writing: (pointer + 0), (pointer + 1), ..., (pointer + T-1)
        # All modulo L
        time_offsets = torch.arange(T, device=buffers.device)  # [T]
        write_indices = (pointer.unsqueeze(1) + time_offsets.unsqueeze(0)) % L  # [B, T]
        
        # Expand indices to match the shape for scattering
        write_indices_expanded = write_indices.unsqueeze(2).expand(B, T, N)  # [B, T, N]
        
        # Scatter samples into buffers
        buffers.scatter_(1, write_indices_expanded, lines)
        
        # Advance pointer
        new_pointer = (pointer + T) % L
        
        # Write updated state
        next_state["delay_buffers"] = buffers
        next_state["delay_pointer"] = new_pointer
