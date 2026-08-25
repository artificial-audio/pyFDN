"""Tests for extracting plotting data from FLAMO model graphs."""

import numpy as np
import pytest

import pyFDN


def test_extract_build_from_dss_model():
    pytest.importorskip("flamo")

    A = np.array([[0.0, 0.5], [-0.5, 0.0]])
    B = np.array([[1.0], [2.0]])
    C = np.array([[3.0, 4.0]])
    D = np.array([[0.25]])
    delays = np.array([11, 17])
    attenuation = np.zeros((1, 6, 2))
    attenuation[:, 0, :] = [0.8, 0.9]
    attenuation[:, 3, :] = 1.0
    post_eq = np.zeros((1, 6, 1))
    post_eq[:, 0, :] = 0.7
    post_eq[:, 3, :] = 1.0

    model = pyFDN.dss_to_flamo(
        A,
        B,
        C,
        D,
        delays,
        48_000,
        nfft=128,
        post_delay=attenuation,
        post_output=post_eq,
    )
    parameters = pyFDN.extract_build(model)

    np.testing.assert_array_equal(parameters.delays, delays)
    np.testing.assert_allclose(parameters.A, A)
    np.testing.assert_allclose(parameters.B, B)
    np.testing.assert_allclose(parameters.C, C)
    np.testing.assert_allclose(parameters.D, D)
    np.testing.assert_allclose(parameters.post_delay, attenuation)
    np.testing.assert_allclose(parameters.post_output, post_eq)
    assert parameters.fs == 48_000
