"""Delay line stages for feedback delays."""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import torch

from .stage import Stage


class DelayRead(Stage):
    """
    Reads delayed samples from circular delay buffers.
    
    This stage:
    - Initializes and reads from a circular delay buffer state
    - Produces `lines` with delayed samples
    - Should appear *before* DelayWrite in the stage pipeline
    
    Shares delay-line state with DelayWrite and initializes it.
    """
    
    def __init__(
        self,
        delay_lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        delay_length: int | None = None,
        num_lines: int | None = None,
        state_key: str = "delay",
    ):
        """
        Initialize delay read stage.
        
        Args:
            delay_lengths: Per-line delay lengths in samples, shape [N]
            delay_length: Convenience alias for uniform delay length across lines
            num_lines: Number of delay lines (N). If omitted, inferred.
            state_key: Prefix for state keys (allows multiple independent delay banks)
        """
        self.state_key = str(state_key)
        self.buffers_key = f"{self.state_key}_buffers"
        self.pointer_key = f"{self.state_key}_pointer"
        super().__init__(state_keys={self.buffers_key, self.pointer_key})

        if delay_lengths is not None and delay_length is not None:
            raise ValueError("Provide only one of delay_lengths or delay_length")

        if delay_lengths is None:
            if delay_length is None:
                delay_lengths = (81, 100, 121, 169)
            else:
                inferred_num_lines = int(num_lines) if num_lines is not None else 1
                delay_lengths = [int(delay_length)] * inferred_num_lines

        delay_lengths_t = torch.as_tensor(delay_lengths, dtype=torch.long)
        if delay_lengths_t.ndim != 1:
            raise ValueError(f"delay_lengths must be 1D [N], got shape {tuple(delay_lengths_t.shape)}")
        if torch.any(delay_lengths_t <= 0):
            raise ValueError("delay_lengths must be positive integers for DelayRead/DelayWrite")

        inferred_num_lines = int(delay_lengths_t.numel())
        if num_lines is None:
            num_lines = inferred_num_lines
        if int(num_lines) != inferred_num_lines:
            raise ValueError(f"num_lines ({num_lines}) must match len(delay_lengths) ({inferred_num_lines})")

        self.delay_lengths = delay_lengths_t
        self.num_lines = int(num_lines)
    
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
                "{state_key}_buffers": Delay buffers of shape [B, N, L]
                "{state_key}_pointer": Delay pointer of shape [B, N]
        """
        self.delay_lengths = self.delay_lengths.to(device)
        max_delay = self.delay_lengths.max().item()
        buffer_size = max_delay + block_size
        return {
            self.buffers_key: torch.zeros(
                batch_size,
                self.num_lines,
                buffer_size,
                device=device,
                dtype=torch.float32,
            ),
            self.pointer_key: torch.zeros(
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
        buffers = state_t[self.buffers_key]  # [B, N, L]
        pointer = state_t[self.pointer_key]  # [B, N]
        
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

        # DelayRead outputs the *purely delayed* signal from the buffers.
        # Any input injection affects the buffers via DelayWrite (later in the
        # pipeline), so we ignore any incoming `lines` value.
        new_lines = delayed

        return new_lines, None


class DelayWrite(Stage):
    """
    Writes processed samples back into circular delay buffers.
    
    This stage:
    - Reads ctx["lines"] (processed samples)
    - Writes to the shared delay-buffer state ("{state_key}_buffers"/"{state_key}_pointer")
    - Advances the buffer pointer
    - Should appear after all processing stages
    
    Shares state with DelayRead stage.
    """
    
    def __init__(self, *, state_key: str = "delay"):
        """
        Initialize delay write stage.
        
        Note:
            Delay parameters (length, num_lines) are determined by the shared
            state initialized by DelayRead.
        """
        self.state_key = str(state_key)
        self.buffers_key = f"{self.state_key}_buffers"
        self.pointer_key = f"{self.state_key}_pointer"
        super().__init__(state_keys={self.buffers_key, self.pointer_key})
    
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
        buffers_t = state_t[self.buffers_key]  # [B, N, L]
        if buffers_t.dtype != lines.dtype:
            buffers_t = buffers_t.to(dtype=lines.dtype)
        buffers = buffers_t.clone()  # clone to avoid mutating state_t
        pointer = state_t[self.pointer_key]  # [B, N]
        
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
        next_state[self.buffers_key] = buffers
        next_state[self.pointer_key] = new_pointer

        return lines, None

class DiagonalDelay(Stage):
    """
    Per-line (diagonal) delay operator with internal state.

    Unlike DelayRead/DelayWrite, this stage *combines* read and write in one step,
    enabling correct sample-accurate behavior when any delay length is smaller
    than block_size (within-block causality).

    Intended use: cascaded filter-feedback-matrix (FFM) constructions where
    diagonal delays D_{m_k}(z) appear between mixing matrices.
    """

    def __init__(
        self,
        delay_lengths: torch.Tensor | Sequence[int],
        *,
        state_key: str = "ffm_delay",
    ):
        self.state_key = str(state_key)
        self.buffers_key = f"{self.state_key}_buffers"
        self.pointer_key = f"{self.state_key}_pointer"
        super().__init__(state_keys={self.buffers_key, self.pointer_key})

        delay_lengths_t = torch.as_tensor(delay_lengths, dtype=torch.long)
        if delay_lengths_t.ndim != 1:
            raise ValueError(f"delay_lengths must be 1D [N], got shape {tuple(delay_lengths_t.shape)}")
        if torch.any(delay_lengths_t < 0):
            raise ValueError("delay_lengths must be non-negative integers for DiagonalDelay")

        self.delay_lengths = delay_lengths_t
        self.num_lines = int(delay_lengths_t.numel())

    def init_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        self.delay_lengths = self.delay_lengths.to(device)
        max_delay = int(self.delay_lengths.max().item()) if self.num_lines > 0 else 0
        buffer_size = max_delay + int(block_size)
        if buffer_size <= 0:
            raise ValueError(f"Computed buffer_size must be positive, got {buffer_size}")

        return {
            self.buffers_key: torch.zeros(
                batch_size,
                self.num_lines,
                buffer_size,
                device=device,
                dtype=torch.float32,
            ),
            self.pointer_key: torch.zeros(
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
        if lines is None:
            raise RuntimeError("DiagonalDelay requires `lines` to be set")

        x = lines  # [B, N, T]
        buffers_t = state_t[self.buffers_key]  # [B, N, L]
        if buffers_t.dtype != x.dtype:
            buffers_t = buffers_t.to(dtype=x.dtype)
        buffers = buffers_t.clone()  # avoid mutating state_t
        pointer = state_t[self.pointer_key]  # [B, N]

        B, N, T = x.shape
        if N != self.num_lines:
            raise RuntimeError(f"DiagonalDelay expected N={self.num_lines} lines, got N={N}")
        if T != int(block_size):
            raise RuntimeError(f"DiagonalDelay expected block_size={block_size}, got T={T}")

        L = buffers.shape[2]

        # Write x into the circular buffer first so reads can see within-block samples.
        time_offsets = torch.arange(T, device=buffers.device, dtype=pointer.dtype).view(1, 1, T)  # [1, 1, T]
        write_indices = (pointer.unsqueeze(2) + time_offsets) % L  # [B, N, T]
        buffers.scatter_(2, write_indices, x)

        # Now read delayed samples from the updated buffer.
        read_indices = (
            pointer.unsqueeze(2)
            - self.delay_lengths.unsqueeze(0).unsqueeze(-1)
            + time_offsets
        ) % L  # [B, N, T]
        y = torch.gather(buffers, 2, read_indices)  # [B, N, T]

        # Advance pointer and commit state.
        new_pointer = (pointer + T) % L
        next_state[self.buffers_key] = buffers
        next_state[self.pointer_key] = new_pointer

        return y, None


# Backwards compatible alias (previously a WIP combined-delay stage).
Delay = DiagonalDelay
