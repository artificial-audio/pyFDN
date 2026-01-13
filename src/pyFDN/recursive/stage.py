"""Abstract base class for recursive DSP processing stages."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Set
import torch


class Stage(ABC):
    """
    Abstract base class for a processing stage in a recursive DSP system.
    
    Each stage represents one block-processing operation that can read from and write to:
    - ctx: Signal context dictionary for the current block (local to this block)
    - state_t: Global state dictionary at the start of the block (read-only)
    - next_state: Accumulating state updates for the next block (write-only for owned keys)
    
    Stages operate on blocks of audio with consistent tensor shapes:
    - B: batch size
    - T: time samples in block
    - N: number of feedback lines/channels
    - N_in: number of input channels
    - N_out: number of output channels
    """
    
    def __init__(self, state_keys: Set[str] | None = None):
        """
        Initialize a processing stage.
        
        Args:
            state_keys: Set of state dictionary keys this stage owns and can modify.
                       If None or empty, the stage is purely feedforward.
        """
        self.state_keys = state_keys or set()
    
    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize state for this stage.
        
        Args:
            batch_size: Number of parallel signals (B dimension)
            device: PyTorch device to create tensors on
            
        Returns:
            Dictionary mapping this stage's state keys to initial state tensors.
            For purely feedforward stages, return an empty dict.
        """
        pass
    
    @abstractmethod
    def step_block(
        self,
        ctx: Dict[str, torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int
    ) -> None:
        """
        Process one block of audio.
        
        This method should:
        1. Read any needed signals from ctx and state_t
        2. Perform processing
        3. Write results back to ctx and next_state (only for owned state keys)
        
        Args:
            ctx: Signal context dictionary for this block. May contain:
                - "x": External input block [B, T, N_in]
                - "lines": Feedback line signals [B, T, N]
                - "y": Output block [B, T, N_out]
                - Other stage-specific entries
            state_t: Global state at start of block (read-only, do not modify)
            next_state: Accumulating state updates (write only your owned keys)
            block_size: Number of time samples in this block (T dimension)
            
        Returns:
            None. Modifications are made in-place to ctx and next_state.
            
        Note:
            - You may read any entry from state_t
            - You may only write to next_state keys listed in self.state_keys
            - You may read and write to ctx
            - Do not directly mutate state_t
        """
        pass
    
    def __repr__(self) -> str:
        state_info = f" (state keys: {self.state_keys})" if self.state_keys else " (stateless)"
        return f"{self.__class__.__name__}(){state_info}"
