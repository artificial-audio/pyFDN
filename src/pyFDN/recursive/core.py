"""Core coordinator for recursive DSP processing."""

from __future__ import annotations
from typing import Dict, List, Optional
import torch

from .stage import Stage


class RecursionCore:
    """
    Coordinator for recursive DSP processing using a sequence of stages.
    
    This class manages:
    - Ordered list of processing stages
    - Global state dictionary shared across all stages
    - Block-based processing of input signals
    - Propagation of a single block-local context tensor (`lines`)
    
    Typical usage:
        stages = [
            DelayRead(...),         # read delayed lines -> lines
            InputTap(...),          # inject external input into lines
            FeedbackMix(...),       # process/mix lines inside the loop
            ...                     # (optional additional processing on lines)
            DelayWrite(...),        # write updated lines back to delay buffers
            OutputTap(...),         # produce final output from lines (+ optional direct path)
        ]
        core = RecursionCore(stages, block_size=512)
        output = core.process(input_signal)
    """
    
    def __init__(
        self,
        stages: List[Stage],
        block_size: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the recursion core.
        
        Args:
            stages: Ordered list of Stage instances to execute sequentially
            block_size: Maximum number of samples per processing block
            device: PyTorch device for all tensors. If None, uses CPU.
        """
        self.stages = stages
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = int(block_size)
        self.device = device or torch.device("cpu")
        self._validate_stages()
        
    def _validate_stages(self) -> None:
        """
        Validate stage configuration.
        
        Note: Multiple stages can claim the same state keys (e.g., DelayRead and
        DelayWrite both use the same delay-buffer keys), but only one should
        initialize them.
        We detect conflicts during init_state if multiple stages try to initialize
        the same key.
        """
        # Just track state keys for informational purposes
        # Actual conflicts will be caught in init_state
        pass
    
    def init_state(
        self, 
        batch_size: int,
        block_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize global state by merging all stage states.
        
        Args:
            batch_size: Number of parallel signals (B dimension)
            
        Returns:
            Merged state dictionary containing all stage states
        """
        resolved_block_size = self.block_size if block_size is None else int(block_size)
        resolved_device = self.device if device is None else device

        state: Dict[str, torch.Tensor] = {}
        for stage in self.stages:
            stage_state = stage.init_state(batch_size, resolved_block_size, resolved_device)
            # Ensure no key conflicts (should already be caught by validation)
            for key in stage_state:
                if key in state:
                    raise RuntimeError(
                        f"State key '{key}' initialized by multiple stages"
                    )
                state[key] = stage_state[key]
        return state
    
    def process(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Process input signal through the recursive system.
        
        Args:
            input_signal: Input tensor of shape [T_total, N_in] or [B, N_in, T_total]
            
        Returns:
            Output tensor of same shape as input (with N_out channels instead of N_in)
            
        Note:
            - Automatically adds batch dimension if input is 2D
            - Handles variable-length final block if T_total % block_size != 0
            - Maintains device and dtype consistency
            - Input/output dimensions: [B, N, T] (batch, channel, sample)
        """
        # Ensure input is 3D [B, N_in, T_total]
        if input_signal.dim() == 2:
            # Assume [T, N_in] -> transpose to [N_in, T], then add batch dim -> [1, N_in, T]
            input_signal = input_signal.T.unsqueeze(0)  # [1, N_in, T]
            squeeze_output = True
        elif input_signal.dim() == 3:
            # Assume new format [B, N_in, T_total]
            squeeze_output = False
        else:
            raise ValueError(
                f"Input must be 2D [T, N_in] or 3D [B, N_in, T_total], "
                f"got {input_signal.dim()}D"
            )
        
        # Move to correct device
        input_signal = input_signal.to(self.device)
        
        B, N_in, T_total = input_signal.shape
        
        # Initialize state
        state = self.init_state(B, self.block_size, self.device)

        # Block-local feedback-line context and output
        lines: Optional[torch.Tensor] = None
        y_block: Optional[torch.Tensor] = None
        
        # Prepare output buffer (will be allocated on first block)
        output_blocks: List[torch.Tensor] = []
        
        # Process blocks
        block_size = self.block_size
        num_blocks = (T_total + block_size - 1) // block_size
        
        for block_idx in range(num_blocks):
            # Block-local context must not carry across blocks. Persistent state
            # belongs in `state` / `next_state` only.
            lines = None
            y_block = None

            # Determine current block size
            start_t = block_idx * block_size
            end_t = min(start_t + block_size, T_total)
            current_block_size = end_t - start_t

            # Extract input block: [B, N_in, T]
            x_block = input_signal[:, :, start_t:end_t]  # [B, N_in, T]

            # If this is the last (possibly short) block, pad at end to full block_size
            if current_block_size < block_size:
                pad_width = block_size - current_block_size
                pad_shape = list(x_block.shape[:-1]) + [pad_width]
                padding = torch.zeros(*pad_shape, device=x_block.device, dtype=x_block.dtype)
                x_block = torch.cat([x_block, padding], dim=-1)
                current_block_size = block_size  # Ensure block_size consistency

            # Initialize next_state accumulator
            next_state: Dict[str, torch.Tensor] = {}

            # Run all stages in sequence
            for stage in self.stages:
                lines, y_candidate = stage.step_block(
                    lines, state, next_state, block_size, x_block
                )
                if y_candidate is not None:
                    y_block = y_candidate
            
            # Merge state_t and next_state for next block
            for key in state:
                if key in next_state:
                    state[key] = next_state[key]
                # else: carry over previous state value
            
            # Extract output block
            if y_block is None:
                raise RuntimeError(
                    "No output produced - ensure pipeline includes OutputTap stage"
                )
            output_blocks.append(y_block)
        
        # Concatenate all output blocks along time dimension (dim=2)
        output = torch.cat(output_blocks, dim=2)  # [B, N_out, T_blocks]
        # Trim to input length when last block was zero-padded
        output = output[:, :, :T_total]

        # Remove batch dimension if input was 2D
        if squeeze_output:
            output = output.squeeze(0)  # [N_out, T_total]
            # Transpose back to [T_total, N_out] for backward compatibility
            output = output.T  # [T_total, N_out]
        
        return output
    
    def __repr__(self) -> str:
        stage_list = "\n  ".join(str(s) for s in self.stages)
        return f"RecursionCore(\n  {stage_list}\n)"
