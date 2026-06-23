"""Tests for time-varying feedback matrix processing."""

import numpy as np
import pytest

from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix


def _make_time_varying_matrix() -> TimeVaryingMatrix:
    tvm = TimeVaryingMatrix(
        N=4,
        cycles_per_second=2.0,
        amplitude=0.5,
        fs=48000.0,
        spread=0.0,
    )
    tvm.phase = np.array([0.25, 1.25])
    tvm.frequency = np.array([2.0, 3.0])
    tvm.angle_amplitude = np.array([0.5, 0.25])
    tvm.sample_index = 0
    return tvm


def _explicit_pairwise_rotation(tvm: TimeVaryingMatrix, x: np.ndarray) -> np.ndarray:
    length = x.shape[0]
    sample_indices = tvm.sample_index + np.arange(length)
    time = sample_indices[:, np.newaxis] / tvm.fs
    angles = tvm.angle_amplitude * np.sin(2 * np.pi * tvm.frequency * time + tvm.phase)
    cos = np.cos(angles)
    sin = np.sin(angles)

    x_pairs = x.reshape(length, tvm.num_pairs, 2)
    out = np.empty_like(x)
    out_pairs = out.reshape(length, tvm.num_pairs, 2)
    out_pairs[..., 0] = cos * x_pairs[..., 0] - sin * x_pairs[..., 1]
    out_pairs[..., 1] = sin * x_pairs[..., 0] + cos * x_pairs[..., 1]
    return out


def test_time_varying_matrix_filter_matches_pairwise_rotations():
    tvm = _make_time_varying_matrix()
    x = np.random.default_rng(0).standard_normal((32, 4))

    expected = _explicit_pairwise_rotation(tvm, x)
    out = tvm.filter(x)

    np.testing.assert_allclose(out, expected)
    assert tvm.sample_index == x.shape[0]


def test_time_varying_matrix_filter_keeps_state_across_blocks():
    x = np.random.default_rng(1).standard_normal((50, 4))
    single_block = _make_time_varying_matrix()
    split_block = _make_time_varying_matrix()

    expected = single_block.filter(x)
    out = np.vstack([split_block.filter(x[:17]), split_block.filter(x[17:])])

    np.testing.assert_allclose(out, expected)
    assert split_block.sample_index == x.shape[0]


def test_time_varying_matrix_filter_rejects_wrong_shape():
    tvm = _make_time_varying_matrix()

    with pytest.raises(ValueError, match="shape"):
        tvm.filter(np.zeros(4))

    with pytest.raises(ValueError, match="shape"):
        tvm.filter(np.zeros((8, 3)))
