"""Tests for translation helpers."""

import numpy as np
import pytest

from pyFDN import dss2impz
from pyFDN import dss2ss


def test_dss2ss_raises_for_inconsistent_delay_blocks():
    delays = np.array([3, 4])
    A = np.eye(2)
    bb = np.ones((2, 1))
    cc = np.ones((1, 2))
    dd = np.eye(1)

    with pytest.raises(ValueError):
        dss2ss(delays, A, bb, cc, dd)


def test_dss2impz_produces_delayed_impulse():
    ir_len = 8
    delays = np.array([3])
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])

    impulse = dss2impz(
        ir_len,
        delays,
        A,
        B,
        C,
        D,
        input_type="mergeInput",
    )

    assert impulse.shape == (ir_len,)
    ir_vector = impulse
    expected = np.zeros(ir_len)
    expected[delays[0]] = 1.0
    assert np.allclose(ir_vector, expected)
