"""Tests for auxiliary.delay module."""

import numpy as np
import pytest
import torch

from pyFDN.auxiliary.delay import (
    flamo_delay_feedback_matrix,
    matrix_delay_approximation,
    mgrpdelay,
    swap_flamo_recursion_paths,
)
from pyFDN.translate.dss_to_flamo import dss_to_flamo


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
        "pyFDN.auxiliary.delay.mgrpdelay",
        lambda _: (mock_gd, mock_freq),
    )

    approx, err = matrix_delay_approximation(np.ones((2, 2, 1)))
    assert approx.shape == (2,)
    assert err.shape == (2, 2)
    assert np.all(np.isfinite(approx))
    assert np.all(np.isfinite(err))


def _small_flamo_fdn():
    return dss_to_flamo(
        np.eye(2),
        np.ones((2, 1)),
        np.ones((1, 2)),
        np.zeros((1, 1)),
        np.array([11, 13]),
        48000,
        nfft=128,
        device="cpu",
    )


def test_flamo_delay_feedback_matrix_copies_and_rewires_model():
    model = _small_flamo_fdn()
    original_loop = model.get_core().branchA.feedback_loop

    result = flamo_delay_feedback_matrix(model, [5, 7], [1, 2], [3, 4])
    result_loop = result.get_core().branchA.feedback_loop

    assert result is not model
    assert not hasattr(original_loop.feedback, "delay_in")
    assert hasattr(result_loop.feedback, "delay_in")
    assert hasattr(result_loop.feedback, "matrix")
    assert hasattr(result_loop.feedback, "delay_out")

    for module, samples in (
        (result_loop.feedforward.delay, [5, 7]),
        (result_loop.feedback.delay_in, [1, 2]),
        (result_loop.feedback.delay_out, [3, 4]),
    ):
        values = torch.as_tensor(samples, dtype=module.param.dtype)
        torch.testing.assert_close(module.param.detach().cpu(), module.sample2s(values))


def test_flamo_delay_feedback_matrix_validates_delays():
    model = _small_flamo_fdn()
    with pytest.raises(ValueError, match="equal lengths"):
        flamo_delay_feedback_matrix(model, [5, 7], [1], [3, 4])
    with pytest.raises(ValueError, match="non-negative"):
        flamo_delay_feedback_matrix(model, [5, -1], [1, 2], [3, 4])


def test_swap_flamo_recursion_paths_copies_and_swaps():
    model = flamo_delay_feedback_matrix(_small_flamo_fdn(), [5, 7], [1, 2], [3, 4])
    original_loop = model.get_core().branchA.feedback_loop

    result = swap_flamo_recursion_paths(model)
    result_loop = result.get_core().branchA.feedback_loop

    assert result is not model
    assert hasattr(result_loop.feedforward, "delay_in")
    assert not hasattr(result_loop.feedback, "delay_in")
    assert hasattr(original_loop.feedback, "delay_in")
