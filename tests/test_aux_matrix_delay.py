"""Tests for matrix delay helpers."""

import numpy as np
import pytest

from pyFDN.auxiliary.matrix_delay_approximation import matrix_delay_approximation
from pyFDN.auxiliary.mgrpdelay import mgrpdelay


def test_mgrpdelay_handles_zero_rows():
    matrix = np.zeros((1, 2, 4))
    gd, freq = mgrpdelay(matrix)
    assert gd.shape == (1, 2, freq.size)
    assert np.isnan(gd).all()
    assert np.isclose(freq[-1], np.pi)


def test_matrix_delay_approximation_uses_nanmean(monkeypatch):
    mock_gd = np.array(
        [
            [[1.0, np.inf], [2.0, 4.0]],
            [[np.nan, 3.0], [5.0, 7.0]],
        ]
    )
    mock_freq = np.array([0.0, 1.0])

    monkeypatch.setattr(
        "pyFDN.auxiliary.matrix_delay_approximation.mgrpdelay",
        lambda _: (mock_gd, mock_freq),
    )

    approx, err = matrix_delay_approximation(np.ones((2, 2, 1)))
    assert approx.shape == (2,)
    assert err.shape == (2, 2)
    assert np.all(np.isfinite(approx))
    assert np.all(np.isfinite(err))
