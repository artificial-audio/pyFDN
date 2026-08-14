"""Tests for the time-domain block-based graph engine (``pyFDN.td``)
Operators only

The td engine filtering correctness of each operator is checked against
:func:`pyFDN.process_fdn`, when an fdn is built, or individual ``pyFDN`` modules.
This needs no FLAMO install.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import pyFDN
from pyFDN import td
from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix as _DSPTimeVaryingMatrix

# ====================== AUXILIARY =====================


def _time_varying_parameters(N: int, fs: float) -> dict[str, Any]:
    """parameters for the time_varying matrix"""

    tvm_kwargs = {
        "N": N,
        "cycles_per_second": 1.3,
        "amplitude": 0.2,
        "fs": fs,
        "spread": 0.1,
    }
    return tvm_kwargs


def _absorption_sos(delays: np.ndarray, fs: float) -> np.ndarray:
    """Per-delay first-order shelving absorption, shape (n_sections, 6, N)."""
    return pyFDN.first_order_absorption(1.5, 0.4, delays, fs, None)


def _noise(rng: np.random.Generator, length: int, channels: int) -> np.ndarray:
    """Gaussian noise"""
    x = rng.standard_normal((length, channels))
    return x


# ============================================================
# ====================== TEST OPERATORS ======================


def test_identity() -> None:
    """Test td Identity operator filter"""

    # parameters
    _, _, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)

    # reference
    I_ref = np.asarray(
        pyFDN.fdn_matrix_gallery(N=N, matrix_type="parallel"), dtype=float
    )

    # input vector
    in_sig = _noise(rng=rng, length=1, channels=N)

    # td engine
    I_td = td.Identity(channels=N)

    # Test
    out_sig_ref = in_sig @ I_ref.T
    out_sig_td = I_td.filter(in_sig)
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-12, rtol=0)


def test_gain() -> None:
    """Test td Gain operator filter"""

    # parameters
    _, _, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)

    # reference
    gain_ref = np.asarray(
        pyFDN.fdn_matrix_gallery(N=N, matrix_type="orthogonal"), dtype=float
    )

    # input vector
    in_sig = _noise(rng=rng, length=1, channels=N)

    # td engine
    gain_td = td.Gain(matrix=gain_ref)

    # Test
    out_sig_ref = in_sig @ gain_ref.T
    out_sig_td = gain_td.filter(in_sig)
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-12, rtol=0)


def test_delay() -> None:
    """Test td Delay operator filtering"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    max_block_size = 256

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    delay_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=max_block_size
    )

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    delay_td = td.Delay(fdnbuild.delays)

    # filter signal
    out_sig_ref = np.zeros_like(in_sig, dtype=float)
    out_sig_td = np.zeros_like(in_sig, dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        out_sig_ref[start : start + block_size, :] = delay_ref.get_values(block_size)
        delay_ref.set_values(block_in)
        delay_ref.advance(block_size)

        # td
        out_sig_td[start : start + block_size, :] = delay_td.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_sosbank() -> None:
    """Test td SOSBank operator filtering"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    sos = _absorption_sos(fdnbuild.delays, fs)
    absorption_ref = pyFDN.SOSFilterBank(sos, N)

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    absorption_td = td.SOSBank(sos)

    # Test
    out_sig_ref = absorption_ref.filter(in_sig)
    out_sig_td = absorption_td.filter(in_sig)
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-12, rtol=0)


def test_matrixfir() -> None:
    """Test td MatrixFIR operator filtering"""

    # parameters
    n_in, n_out, _ = 2, 3, 4
    rng = np.random.default_rng(seed=5)

    # reference
    taps = 100
    coeffs = rng.standard_normal((n_out, n_in, taps))
    matrixFIR_ref = pyFDN.FIRMatrixFilter(coeffs)

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in)

    # td engine
    matrixFIR_td = td.MatrixFIR(coeffs)

    # filter signal
    out_sig_ref = matrixFIR_ref.filter(in_sig)
    out_sig_td = matrixFIR_td.filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-10, rtol=0)


def test_matrixconv() -> None:
    """Test td MatrixConvolver operator filtering against td MatrixFIR"""

    # parameters
    n_in, n_out, _ = 2, 3, 4
    rng = np.random.default_rng(seed=5)

    # reference
    taps = 100
    coeffs = rng.standard_normal((n_out, n_in, taps))

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in)

    # td engine
    matrixFIR_td = td.MatrixFIR(coeffs)
    matrixConv_td = td.MatrixConvolver(coeffs)

    # filter signal
    out_sig_ref = matrixFIR_td.filter(in_sig)
    out_sig_td = matrixConv_td.filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-10, rtol=0)


def test_tvmatrix() -> None:
    """Test td TimeVaryingMatrix operator against vanilla pyfdn TimeVaringMatrix"""

    # parameters
    _, _, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    tvmatrix_kwargs = _time_varying_parameters(N=N, fs=fs)
    np.random.seed(3)
    tvmatrix_ref = _DSPTimeVaryingMatrix(**tvmatrix_kwargs)

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    np.random.seed(3)
    tvmatrix_td = td.TimeVaryingMatrix(_DSPTimeVaryingMatrix(**tvmatrix_kwargs))

    # filter signal
    out_sig_ref = tvmatrix_ref.filter(in_sig)
    out_sig_td = tvmatrix_td.filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-10, rtol=0)
