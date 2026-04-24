"""Abstract base class for recursive DSP processing stages."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Stage(ABC):
    """
    Abstract base class for a processing stage in a recursive DSP system.

    Each stage represents one block-processing operation that can read from and write to:
    - lines: Feedback-line signals tensor for the current block (local to this block)
    - state_t: Global state dictionary at the start of the block (read-only)
    - next_state: Accumulating state updates for the next block (write-only for owned keys)

    Stages operate on blocks of audio with consistent tensor shapes:
    - B: batch size
    - T: time samples in block
    - N: number of feedback lines/channels
    - N_in: number of input channels
    - N_out: number of output channels
    """

    def __init__(self, state_keys: set[str] | None = None):
        """
        Initialize a processing stage.

        Args:
            state_keys: Set of state dictionary keys this stage owns and can modify.
                       If None or empty, the stage is purely feedforward.
        """
        self.state_keys = state_keys or set()

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        block_size: int | torch.device | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
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
        lines: torch.Tensor | None,
        state_t: dict[str, torch.Tensor],
        next_state: dict[str, torch.Tensor],
        block_size: int,
        x_block: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Process one block of audio.

        This method should:
        1. Read any needed signals from `lines`, `x_block` and `state_t`
        2. Perform processing
        3. Return updated `lines` and (optionally) an output block `y_block`

        Args:
            lines: Feedback-line signals for this block, shape [B, N, T].
                   May be None for the first stage (e.g. `DelayRead`) which
                   is responsible for creating the initial lines tensor.
            state_t: Global state at start of block (read-only, do not modify)
            next_state: Accumulating state updates (write only your owned keys)
            block_size: Number of time samples in this block (T dimension)
            x_block: Optional external input block, shape [B, N_in, T].
                     Only stages that use the external input (e.g. `InputTap`,
                     `OutputTap` with direct paths) need to read this.

        Returns:
            A tuple `(new_lines, y_block)` where:
                - `new_lines` is the updated feedback-line tensor [B, N, T]
                - `y_block` is an optional output tensor [B, N_out, T]
                  (typically produced only by `OutputTap`-like stages)

        Note:
            - You may read any entry from `state_t`
            - You may only write to `next_state` keys listed in `self.state_keys`
            - Do not directly mutate `state_t`
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        state_info = (
            f" (state keys: {self.state_keys})" if self.state_keys else " (stateless)"
        )
        return f"{self.__class__.__name__}(){state_info}"
