"""Bank of per-channel SOS filter cascades with persistent state.

Used for per-delay-line absorption filters in ``process_fdn``; the SOS-cascade
counterpart of :class:`pyFDN.dsp.dfilt_matrix.FIRMatrixFilter`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import sosfilt, sosfilt_zi


class SOSFilterBank:
    """Apply one SOS filter cascade per channel, block by block.

    Parameters
    ----------
    sos : array
        Per-channel SOS bank of shape ``(n_sections, 6, N)`` where
        ``N = num_channels``. Section rows are ``[b0, b1, b2, a0, a1, a2]``.
        This is the canonical SOS bank layout in pyFDN: it matches the FLAMO
        ``parallelSOSFilter`` input and the output of
        :func:`pyFDN.first_order_absorption`, :func:`pyFDN.one_pole_absorption`,
        and :func:`pyFDN.absorption_geq`.
    num_channels : int
        Number of channels N.

    Filter state persists across calls to :meth:`filter`, so a long signal
    can be processed in consecutive blocks.
    """

    def __init__(self, sos: ArrayLike, num_channels: int):
        n = int(num_channels)
        sos_arr = np.asarray(sos, dtype=float)
        if sos_arr.ndim == 3 and sos_arr.shape[1] == 6 and sos_arr.shape[2] == n:
            # Canonical (n_sections, 6, N) -> (N, n_sections, 6) for scipy sosfilt.
            sos_arr = sos_arr.transpose(2, 0, 1)
        else:
            raise ValueError(
                f"sos must have shape (n_sections, 6, N); got {sos_arr.shape} for N={n}"
            )
        self.sos = np.ascontiguousarray(sos_arr)  # internal: (N, n_sections, 6)
        self.num_channels = n
        self._state = [sosfilt_zi(self.sos[i]) * 0.0 for i in range(n)]

    def filter(self, block: ArrayLike) -> np.ndarray:
        """Filter a block of shape (block_size, N) channel-wise."""
        x = np.asarray(block, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.num_channels:
            raise ValueError(f"block must have shape (block_size, {self.num_channels})")
        out = np.empty_like(x)
        for i in range(self.num_channels):
            out[:, i], self._state[i] = sosfilt(
                self.sos[i], np.ascontiguousarray(x[:, i]), zi=self._state[i]
            )
        return out
