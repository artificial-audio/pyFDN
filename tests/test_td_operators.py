"""Tests for the time-domain block-based graph engine (``pyFDN.td``)
Operators only

The filtering correctness of each operator is checked against an independent
reference: a direct scipy/numpy computation of the same operation, or another
``pyFDN`` component built on a different mechanism.
This needs no FLAMO install.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.signal import sosfilt

import pyFDN
from pyFDN import td

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
    return pyFDN.decay_to_first_order_shelf(1.5, 0.4, None, delays, fs)


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
    delay_ref = td.RecursionState(delays=fdnbuild.delays, max_block_size=max_block_size)

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

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # reference: scipy sosfilt, one cascade per channel
    out_sig_ref = np.column_stack(
        [
            sosfilt(
                np.ascontiguousarray(sos[:, :, i]),  # (n_sections, 6) for channel i
                np.ascontiguousarray(in_sig[:, i]),
            )
            for i in range(N)
        ]
    )

    # td engine
    absorption_td = td.SOSBank(sos)

    # Test
    out_sig_td = absorption_td.filter(in_sig)
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-12, rtol=0)


def test_sosbank_block_consistency_and_reset() -> None:
    """Test that SOSBank state persists across blocks, and that reset()
    brings it back to the initial (zero) state"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    block = 128

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    sos = _absorption_sos(fdnbuild.delays, fs)

    # input signal
    n_samples = 1024
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    absorption_td = td.SOSBank(sos)
    out_sig_one_shot = absorption_td.filter(in_sig)

    absorption_td.reset()
    out_sig_blockwise = np.vstack(
        [
            absorption_td.filter(in_sig[i : i + block])
            for i in range(0, n_samples, block)
        ]
    )

    # Test
    np.testing.assert_allclose(out_sig_blockwise, out_sig_one_shot, atol=1e-12, rtol=0)


def test_matrixfir() -> None:
    """Test td MatrixFIR operator filtering"""

    # parameters
    n_in, n_out, _ = 2, 3, 4
    rng = np.random.default_rng(seed=5)

    # reference
    taps = 100
    coeffs = rng.standard_normal((n_out, n_in, taps))

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in)

    # reference: explicit convolution sum over the matrix entries
    out_sig_ref = np.zeros((n_samples, n_out))
    for i in range(n_out):
        for j in range(n_in):
            out_sig_ref[:, i] += np.convolve(in_sig[:, j], coeffs[i, j])[:n_samples]

    # td engine
    matrixFIR_td = td.MatrixFIR(coeffs)
    out_sig_td = matrixFIR_td.filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-10, rtol=0)


def test_matrixfir_block_consistency_and_reset() -> None:
    """Test that MatrixFIR state persists across blocks, and that reset()
    brings it back to the initial (zero) state"""

    # parameters
    n_in, n_out, _ = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    taps, block = 8, 32

    # input signal
    n_samples = 200
    coeffs = rng.standard_normal((n_out, n_in, taps))
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in)

    # td engine
    matrixFIR_td = td.MatrixFIR(coeffs)
    out_sig_one_shot = matrixFIR_td.filter(in_sig)

    matrixFIR_td.reset()
    out_sig_blockwise = np.vstack(
        [matrixFIR_td.filter(in_sig[i : i + block]) for i in range(0, n_samples, block)]
    )

    # Test
    np.testing.assert_allclose(out_sig_blockwise, out_sig_one_shot, atol=1e-12, rtol=0)


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
    """Test td TimeVaryingMatrix operator against an explicit per-sample
    block-diagonal rotation matrix"""

    # parameters
    _, _, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # td engine
    np.random.seed(3)
    tvmatrix_td = td.TimeVaryingMatrix(**_time_varying_parameters(N=N, fs=fs))

    # input signal
    n_samples = 400
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # reference: build the rotation matrix sample by sample and apply it
    out_sig_ref = np.zeros_like(in_sig)
    for n in range(n_samples):
        angles = tvmatrix_td.angle_amplitude * np.sin(
            2 * np.pi * tvmatrix_td.frequency * (n / fs) + tvmatrix_td.phase
        )
        rotation = np.zeros((N, N))
        for pair, angle in enumerate(angles):
            rotation[2 * pair : 2 * pair + 2, 2 * pair : 2 * pair + 2] = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
        out_sig_ref[n] = rotation @ in_sig[n]

    # filter signal
    out_sig_td = tvmatrix_td.filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_ref, out_sig_td, atol=1e-12, rtol=0)


def test_tvmatrix_is_orthogonal_and_resets() -> None:
    """Test that TimeVaryingMatrix preserves the signal norm at every sample
    (it is orthogonal), and that reset() rewinds the modulation clock"""

    # parameters
    _, _, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # td engine
    np.random.seed(3)
    tvmatrix_td = td.TimeVaryingMatrix(**_time_varying_parameters(N=N, fs=fs))

    # input signal
    n_samples = 400
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # filter signal
    out_sig = tvmatrix_td.filter(in_sig)

    # Test: orthogonal -> per-sample norm preserved
    np.testing.assert_allclose(
        np.linalg.norm(out_sig, axis=1),
        np.linalg.norm(in_sig, axis=1),
        atol=1e-12,
        rtol=0,
    )

    # Test: reset rewinds the clock, so the same input gives the same output
    assert tvmatrix_td.sample_index == n_samples
    tvmatrix_td.reset()
    assert tvmatrix_td.sample_index == 0
    np.testing.assert_allclose(tvmatrix_td.filter(in_sig), out_sig, atol=1e-12, rtol=0)


def test_tvmatrix_rejects_odd_channel_count() -> None:
    """Test that TimeVaryingMatrix rejects channel counts it cannot pair up"""

    # Test
    with pytest.raises(ValueError, match="N must be even"):
        td.TimeVaryingMatrix(
            N=3, cycles_per_second=1.3, amplitude=0.2, fs=48_000.0, spread=0.1
        )

    with pytest.raises(ValueError, match="N must be a positive integer"):
        td.TimeVaryingMatrix(
            N=0, cycles_per_second=1.3, amplitude=0.2, fs=48_000.0, spread=0.1
        )


def test_absolute_value() -> None:
    """Test td AbsoluteValue operator filtering"""

    # parameters
    _, _, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    abs_td = td.AbsoluteValue(channels=N)

    # Test: memoryless and stateless, so blockwise equals one-shot
    out_sig_td = abs_td.filter(in_sig)
    np.testing.assert_allclose(out_sig_td, np.abs(in_sig), atol=1e-12, rtol=0)
    assert np.all(out_sig_td >= 0.0)

    block = 128
    out_sig_blockwise = np.vstack(
        [abs_td.filter(in_sig[i : i + block]) for i in range(0, n_samples, block)]
    )
    np.testing.assert_allclose(out_sig_blockwise, out_sig_td, atol=1e-12, rtol=0)

    # Test: channel count is enforced, like every other operator
    with pytest.raises(ValueError, match="expects 4 input channels"):
        abs_td.filter(_noise(rng=rng, length=16, channels=N + 1))
