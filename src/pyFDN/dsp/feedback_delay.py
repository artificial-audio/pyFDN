import numpy as np

class FeedbackDelay:
    """
    Array of feedback delay processors.

    Translated from MATLAB class.
    """

    def __init__(self, max_block_size, delays):
        self.delays = np.array(delays)
        self.num_delays = len(delays)
        # Allocate delay lines with extra space for max_block_size
        self.values = np.zeros((max(delays) + max_block_size, self.num_delays))
        self.pointers = np.ones(self.num_delays, dtype=int)  # 1-based in MATLAB, 0-based in Python
        self.pointers -= 1  # convert to 0-based indexing

    def set_values(self, val):
        """
        Set values at current pointer positions.
        val: shape [blockSize, num_delays]
        """
        blkSz = val.shape[0]
        row_idx, col_idx = self._get_index(blkSz)
        self.values[row_idx, col_idx] = val

    def get_values(self, blkSz):
        """
        Get values from current pointer positions.
        """
        row_idx, col_idx = self._get_index(blkSz)
        return self.values[row_idx, col_idx]

    def next(self, blkSz):
        """
        Move pointers forward by block size (with wrap-around modulo delay).
        """
        self.pointers = self._mod_delay(self.pointers + blkSz)

    # ---------------- Internal helpers ---------------- #
    def _get_index(self, blkSz):
        """
        Compute row and column indices for current block.
        Returns:
            row_idx, col_idx : arrays of shape [blkSz, num_delays]
        """
        # Row indices: pointers + 0..blkSz-1
        row_idx = self.pointers + np.arange(blkSz)[:, None]  # shape [blkSz, num_delays]
        row_idx = self._mod_delay(row_idx)

        # Column indices: broadcast 0..num_delays-1
        col_idx = np.arange(self.num_delays)[None, :]  # shape [1, num_delays]
        col_idx = np.tile(col_idx, (blkSz, 1))

        return row_idx, col_idx

    def _mod_delay(self, idx):
        """
        Wrap-around modulo each delay value.
        idx: array
        """
        # idx - delays > 0 → wrap around
        wrapped = idx.copy()
        for i, delay in enumerate(self.delays):
            wrapped[:, i] = np.mod(idx[:, i], delay)
        return wrapped
