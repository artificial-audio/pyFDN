from __future__ import annotations
import numpy as np
from typing import Optional
from numpy.typing import ArrayLike

class FeedbackDelay:
    """Vectorised block delay lines for the FDN."""

    def __init__(self, delays: ArrayLike, max_block_size: int) -> None:
        delays_arr = np.asarray(delays, dtype=int).reshape(-1)
        if delays_arr.ndim != 1:
            raise ValueError("Delays must be a 1-D array")
        if np.any(delays_arr <= 0):
            raise ValueError("Delays must be positive integers")

        self.delays = delays_arr
        self.num_delays = delays_arr.size
        self.max_block_size = int(max_block_size)
        self.max_delay = int(np.max(delays_arr))
        self.buffer = np.zeros((self.num_delays, self.max_delay), dtype=float)
        self.pointers = np.zeros(self.num_delays, dtype=int)
        self._last_indices: Optional[np.ndarray] = None

    def get_values(self, block_size: int) -> np.ndarray:
        if block_size > self.max_block_size:
            raise ValueError("Block size exceeds configured maximum")
        offsets = (self.pointers[:, None] + np.arange(block_size)[None, :]) % self.delays[:, None]
        self._last_indices = offsets
        gathered = self.buffer[np.arange(self.num_delays)[:, None], offsets]
        return gathered.T

    def set_values(self, block: ArrayLike) -> None:
        if self._last_indices is None:
            raise RuntimeError("get_values must be called before set_values")
        block_arr = np.asarray(block, dtype=float)
        if block_arr.shape != (self._last_indices.shape[1], self.num_delays):
            raise ValueError("Block shape mismatch when writing delay values")
        self.buffer[np.arange(self.num_delays)[:, None], self._last_indices] = block_arr.T

    def advance(self, block_size: int) -> None:
        self.pointers = (self.pointers + block_size) % self.delays
        self._last_indices = None

