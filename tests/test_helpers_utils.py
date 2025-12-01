"""Tests for helper utility functions."""

import numpy as np
import pytest

from pyFDN.auxiliary.utils import ensure_3d, last_nonzero_indices


def test_ensure_3d_promotes_2d_arrays():
    mat = np.array([[1.0, 2.0]])
    promoted = ensure_3d(mat)
    assert promoted.shape == (1, 2, 1)
    assert promoted[..., 0].tolist() == mat.tolist()


def test_ensure_3d_rejects_invalid_rank():
    with pytest.raises(ValueError):
        ensure_3d(np.array([1.0]))


def test_last_nonzero_indices_reports_positions():
    tensor = np.array(
        [
            [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    indices = last_nonzero_indices(tensor)
    assert indices.shape == (2, 2)
    assert indices[0, 0] == 3
    assert indices[1, 0] == 2
    assert np.all(indices[:, 1] == 0)
