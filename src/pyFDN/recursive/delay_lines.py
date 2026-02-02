"""Delay line stages for feedback delays."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from .stage import Stage


class DelayRead(Stage):
    """
    Reads delayed samples from circular delay buffers.
    
    This stage:
    - Reads from the shared "delay_buffers" state
    - Produces `lines` with delayed samples
    - Should appear *after* DelayWrite in the stage pipeline
    
    Shares delay-line state with DelayWrite, but does not initialize it.
    """
    
    def __init__(
        self,
        delay_lengths: torch.Tensor = torch.tensor([81, 100, 121, 169]),
        num_lines: int = 4
    ):
        """
        Initialize delay read stage.
        
        Args:
            delay_length: Length of delay buffer in samples (L)
            num_lines: Number of delay lines (N)
        """
        # Stateless: delay state is owned/initialized by DelayWrite
        super().__init__(state_keys=set())
        self.delay_lengths = delay_lengths
        self.num_lines = num_lines
    
    def init_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize delay buffers and pointer.
        
        Args:
            batch_size: Batch size
            block_size: Block size
            device: Device
        Returns:
            Dict[str, torch.Tensor]: State dictionary
                "delay_buffers": Delay buffers of shape [B, N, L]
                "delay_pointer": Delay pointer of shape [B, N]
        """
        max_delay = self.delay_lengths.max().item()
        buffer_size = max_delay + block_size
        return {
            "delay_buffers": torch.zeros(
                batch_size,
                self.num_lines,
                buffer_size,
                device=device,
                dtype=torch.float32,
            ),
            "delay_pointer": torch.zeros(
                batch_size,
                self.num_lines,
                device=device,
                dtype=torch.long,
            ),
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
        Read delayed samples from buffers and add optional injection.
        
        Produces `lines` of shape [B, N, T] containing delayed samples.
        
        Args:
            lines: Optional incoming lines tensor of shape [B, N, T]
            state_t: State at start of block
            next_state: State at end of block
            block_size: Block size
            x_block: Optional external input block of shape [B, N_in, T]
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - `new_lines`: Updated lines tensor [B, N, T]
                - `None`: No output block needed
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
        read_indices = (pointer.unsqueeze(2) - self.delay_lengths.unsqueeze(0).unsqueeze(-1) + time_offsets) % L  # [B, N, T]
        
        # Gather samples from buffers along the L dimension (dim=2)
        # buffers: [B, N, L], read_indices: [B, N, T] -> delayed: [B, N, T]
        delayed = torch.gather(buffers, 2, read_indices)  # [B, N, T]

        # For the canonical topology (InputTap -> DelayWrite -> DelayRead -> OutputTap),
        # DelayRead should output the *purely delayed* signal from the buffers.
        # Any input injection affects the buffers via DelayWrite, not by being
        # summed here, so we ignore any incoming `lines` value.
        new_lines = delayed

        return new_lines, None


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
    
    def init_state(self, batch_size: int, block_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        DelayWrite does not initialize state.
        """
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
        Write processed samples to buffers and advance pointer.
        
        Reads `lines` of shape [B, N, T] and writes to delay buffers.
        """
        if lines is None:
            raise RuntimeError("DelayWrite requires `lines` to be provided by previous stages")
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

        return lines, None

# TODO: find ways to combine DelayRead and DelayWrite into a single stage
class Delay(Stage):
    """
    Reads delayed samples from circular delay buffers.
    
    This stage:
    - Reads from the shared "delay_buffers" state
    - Produces `lines` with delayed samples
    - Should appear *after* DelayWrite in the stage pipeline
    
    Shares delay-line state with DelayWrite, but does not initialize it.
    """
    
    def __init__(
        self,
        delay_lengths: torch.Tensor = torch.tensor([81, 100, 121, 169]),
        num_lines: int = 4
    ):
        """
        Initialize delay read stage.
        
        Args:
            delay_length: Length of delay buffer in samples (L)
            num_lines: Number of delay lines (N)
        """
        # Stateless: delay state is owned/initialized by DelayWrite
        super().__init__(state_keys=set())
        self.delay_lengths = delay_lengths
        self.num_lines = num_lines
    
    def init_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize delay buffers and pointer.
        
        Args:
            batch_size: Batch size
            block_size: Block size
            device: Device
        Returns:
            Dict[str, torch.Tensor]: State dictionary
                "delay_buffers": Delay buffers of shape [B, N, L]
                "delay_pointer": Delay pointer of shape [B, N]
        """
        max_delay = self.delay_lengths.max().item()
        buffer_size = max_delay + block_size
        return {
            "delay_buffers": torch.zeros(
                batch_size,
                self.num_lines,
                buffer_size,
                device=device,
                dtype=torch.float32,
            ),
            "delay_pointer": torch.zeros(
                batch_size,
                self.num_lines,
                device=device,
                dtype=torch.long,
            ),
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
        Read delayed samples from buffers and add optional injection.
        
        Produces `lines` of shape [B, N, T] containing delayed samples.
        
        Args:
            lines: Optional incoming lines tensor of shape [B, N, T]
            state_t: State at start of block
            next_state: State at end of block
            block_size: Block size
            x_block: Optional external input block of shape [B, N_in, T]
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - `new_lines`: Updated lines tensor [B, N, T]
                - `None`: No output block needed
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
        read_indices = (pointer.unsqueeze(2) - self.delay_lengths.unsqueeze(0).unsqueeze(-1) + time_offsets) % L  # [B, N, T]
        
        # Gather samples from buffers along the L dimension (dim=2)
        # buffers: [B, N, L], read_indices: [B, N, T] -> delayed: [B, N, T]
        delayed = torch.gather(buffers, 2, read_indices)  # [B, N, T]

        # Delay is read-only: do not write to buffers or update pointer.
        return delayed, None