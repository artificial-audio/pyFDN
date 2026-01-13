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
                batch_size, self.num_lines, self.delay_length,
                device=device, dtype=torch.float32
            ),
            "delay_pointer": torch.zeros(
                batch_size, self.num_lines, dtype=torch.long, device=device
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
        
        Produces ctx["lines"] of shape [B, N, T] containing delayed samples.
        """
        buffers = state_t["delay_buffers"]  # [B, N, L]
        pointer = state_t["delay_pointer"]  # [B, N]
        
        B, N, L = buffers.shape
        T = block_size
        
        # Generate indices for reading delayed samples
        # Read from (pointer - delay_length) % L to get samples delayed by delay_length
        # pointer: [B, N], time_offsets: [T] -> read_indices: [B, N, T]
        time_offsets = torch.arange(T, device=buffers.device).view(1, 1, T)  # [1, 1, T]
        # Read from positions offset by delay_length behind the write pointer
        read_indices = (pointer.unsqueeze(2) - self.delay_length + time_offsets) % L  # [B, N, T]
        
        # Gather samples from buffers along the L dimension (dim=2)
        # buffers: [B, N, L], read_indices: [B, N, T] -> lines: [B, N, T]
        ctx["lines"] = torch.gather(buffers, 2, read_indices)  # [B, N, T]
        
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
        
        Reads ctx["lines"] of shape [B, N, T] and writes to delay buffers.
        """
        if "lines" not in ctx:
            raise RuntimeError("DelayWrite requires ctx['lines'] to be set by previous stages")
        
        lines = ctx["lines"]  # [B, N, T]
        buffers = state_t["delay_buffers"].clone()  # [B, N, L] - clone to avoid mutating state_t
        pointer = state_t["delay_pointer"]  # [B, N]
        
        B, N, T = lines.shape
        L = buffers.shape[2]
        
        # Generate indices for writing: (pointer + 0), (pointer + 1), ..., (pointer + T-1)
        # All modulo L
        # pointer: [B, N], time_offsets: [T] -> write_indices: [B, N, T]
        time_offsets = torch.arange(T, device=buffers.device).view(1, 1, T)  # [1, 1, T]
        write_indices = (pointer.unsqueeze(2) + time_offsets) % L  # [B, N, T]
        
        # Scatter samples into buffers along the L dimension (dim=2)
        buffers.scatter_(2, write_indices, lines)
        
        # Advance pointer (one per delay line)
        new_pointer = (pointer + T) % L  # [B, N]
        
        # Write updated state
        next_state["delay_buffers"] = buffers
        next_state["delay_pointer"] = new_pointer
