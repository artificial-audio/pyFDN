"""Matrix of FIR filters with persistent state (dfiltMatrix translation).

The MATLAB toolbox wraps every matrix entry in a ``dfilt`` object; here only
the FIR-matrix case is needed (used for paraunitary / velvet feedback
matrices). Scalar matrices are handled directly by ``process_fdn`` and IIR
per-delay-line absorption is handled by
:class:`pyFDN.dsp.sos_filter_bank.SOSFilterBank`, so neither needs this class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lfilter


class FIRMatrixFilter:
    """Apply a matrix of FIR filters to a multichannel signal, block by block.

    Parameters
    ----------
    coefficients : (n_out, n_in, order) array
        FIR coefficients per matrix entry in z^{-1} convention
        (``coefficients[i, j, k]`` is the tap of ``z^{-k}`` from input ``j``
        to output ``i``).

    Filter state persists across calls to :meth:`filter`, so a long signal
    can be processed in consecutive blocks.
    """

    def __init__(self, coefficients: ArrayLike):
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 3:
            raise ValueError("coefficients must have shape (n_out, n_in, order)")
        self.coefficients = coeffs
        self.n_out, self.n_in, self.order = coeffs.shape
        self._state = np.zeros((self.n_out, self.n_in, max(self.order - 1, 1)))

    def filter(self, block: ArrayLike) -> np.ndarray:
        """Filter a block of shape (block_size, n_in) to (block_size, n_out)."""
        x = np.asarray(block, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"block must have shape (block_size, {self.n_in})")
        out = np.zeros((x.shape[0], self.n_out))
        if self.order == 1:
            return x @ self.coefficients[:, :, 0].T
        for i in range(self.n_out):
            for j in range(self.n_in):
                y, self._state[i, j] = lfilter(
                    self.coefficients[i, j], [1.0], x[:, j], zi=self._state[i, j]
                )
                out[:, i] += y
        return out
