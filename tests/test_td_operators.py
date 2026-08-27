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


# ============================================================
# =============== NONLINEAR / PITCH-SHIFT OPERATORS ==========


def _shimmer_operators(N: int, fs: float) -> dict[str, td.TimeOperator]:
    """One instance of every stateful nonlinear operator, active on channels 0-1."""
    active = [0, 1]
    return {
        "ControllableFullWaveRect": td.ControllableFullWaveRect(N, 0.5, active),
        "SDFD": td.SDFD(N, 0.3, active),
        "RingModulator": td.RingModulator(N, 200.0, np.sqrt(2), fs, active),
        "PitchShift": td.PitchShift(N, 4096, 1024, 700.0, fs, active),
        "GranularPitchShift": td.GranularPitchShift(
            N, 8192, 2048, 700.0, active, seed=1
        ),
    }


@pytest.mark.parametrize("name", list(_shimmer_operators(4, 48_000.0)))  # type: ignore[misc]
def test_shimmer_operator_block_consistency_and_reset(name: str) -> None:
    """State persists across blocks, and reset() restores the initial state.

    Every one of these carries state (a previous sample, a modulation phase, a
    circular buffer, an internal DC blocker), so a blockwise render may only
    match the one-shot render if all of it is carried across the boundary --
    and only if reset() puts all of it back.
    """

    # parameters
    N, fs, block = 4, 48_000.0, 128
    rng = np.random.default_rng(seed=5)

    # input signal
    n_samples = 2048
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    operator = _shimmer_operators(N, fs)[name]
    out_sig_one_shot = operator.filter(in_sig)

    operator.reset()
    out_sig_blockwise = np.vstack(
        [operator.filter(in_sig[i : i + block]) for i in range(0, n_samples, block)]
    )

    # Test
    np.testing.assert_allclose(out_sig_blockwise, out_sig_one_shot, atol=1e-12, rtol=0)

    # Test: inactive channels are passed through untouched, bit for bit
    np.testing.assert_array_equal(out_sig_one_shot[:, 2:], in_sig[:, 2:])

    # Test: channel count is enforced, like every other operator
    with pytest.raises(ValueError, match="expects 4 input channels"):
        operator.filter(_noise(rng=rng, length=16, channels=N + 1))


def test_dc_blocker_matches_difference_equation() -> None:
    """Test DCBlocker against a direct evaluation of y[n] = x[n] - x[n-1] + R y[n-1]"""

    # parameters
    N, R = 3, 0.99
    rng = np.random.default_rng(seed=5)

    # input signal
    n_samples = 1000
    in_sig = _noise(rng=rng, length=n_samples, channels=N) + 0.4  # offset to remove

    # reference
    out_sig_ref = np.zeros_like(in_sig)
    for n in range(n_samples):
        prev_x = in_sig[n - 1] if n else np.zeros(N)
        prev_y = out_sig_ref[n - 1] if n else np.zeros(N)
        out_sig_ref[n] = in_sig[n] - prev_x + R * prev_y

    # td engine
    out_sig_td = td.DCBlocker(N, R=R).filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-10, rtol=0)

    # Test: a constant input decays away, i.e. DC really is rejected
    settled_dc = td.DCBlocker(N, R=R).filter(np.ones((2000, N)))[-1]
    assert np.abs(settled_dc).max() < 1e-6


def test_dc_blocker_compensation_gain_is_bounded() -> None:
    """Test that the energy compensation cannot wind up without limit.

    A near-DC input is exactly what the blocker removes, so its output power
    goes to zero and the uncompensated input/output ratio diverges. Uncapped,
    the gain undoes the blocking and runs away inside a feedback loop.
    """

    # parameters
    max_gain = 4.0

    # input signal: pure DC, the worst case for the compensation
    in_sig = np.ones((48_000, 1))

    # td engine
    blocker = td.DCBlocker(1, correct_loss=True, max_gain=max_gain)
    out_sig_td = blocker.filter(in_sig)

    # Test
    assert blocker.gain.max() <= max_gain
    assert np.abs(out_sig_td[-1000:]).max() < 1e-9  # DC really is blocked


def test_controllable_full_wave_rect_blocks_the_dc_it_injects() -> None:
    """Test that full-wave rectification does not leave a DC offset behind"""

    # parameters
    N = 2
    rng = np.random.default_rng(seed=5)

    # input signal
    n_samples = 20_000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # td engine
    rect_td = td.ControllableFullWaveRect(N, alpha=1.0, active_channels=[0])
    out_sig_td = rect_td.filter(in_sig)

    # Test: |x| has a large positive mean, the operator's output does not
    assert np.abs(in_sig[:, 0]).mean() > 0.5
    assert abs(out_sig_td[2000:, 0].mean()) < 1e-2


def test_sdfd_is_a_unit_delay_at_zero_depth() -> None:
    """Test SDFD at d = 0, where both branches collapse onto x[n - 1]"""

    # parameters
    N = 2

    # input signal
    in_sig = np.arange(1, 11, dtype=float)[:, np.newaxis].repeat(N, axis=1)

    # reference
    out_sig_ref = np.vstack([np.zeros((1, N)), in_sig[:-1]])

    # td engine
    out_sig_td = td.SDFD(N, d=0.0, active_channels=[0, 1]).filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_ring_modulator_matches_sine_product() -> None:
    """Test RingModulator against an explicit sine multiplication"""

    # parameters
    N, fs, mod_freq, mod_amp = 3, 48_000.0, 700.0, np.sqrt(2)
    rng = np.random.default_rng(seed=5)

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N)

    # reference
    n = np.arange(n_samples)[:, np.newaxis]
    out_sig_ref = in_sig.copy()
    out_sig_ref[:, :1] *= mod_amp * np.sin(2 * np.pi * mod_freq * n / fs)

    # td engine
    out_sig_td = td.RingModulator(N, mod_freq, mod_amp, fs, [0]).filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("cents", [1200.0, -1200.0])  # type: ignore[misc]
@pytest.mark.parametrize("operator_name", ["PitchShift", "GranularPitchShift"])  # type: ignore[misc]
def test_pitch_shifters_transpose_a_sine(operator_name: str, cents: float) -> None:
    """Test that a sine comes back out at the transposed frequency"""

    # parameters
    fs, f0 = 48_000.0, 500.0

    # input signal
    n_samples = 24_000
    in_sig = np.sin(2 * np.pi * f0 * np.arange(n_samples) / fs)[:, np.newaxis]

    # td engine
    if operator_name == "PitchShift":
        shifter: td.TimeOperator = td.PitchShift(1, 8192, 2048, cents, fs, [0])
    else:
        shifter = td.GranularPitchShift(1, 8192, 2048, cents, [0], seed=0)
    out_sig_td = shifter.filter(in_sig)

    # Test: the dominant partial sits at f0 * 2 ** (cents / 1200), give or take
    # the spread the window/grain modulation adds around it
    settled = out_sig_td[8000:, 0] * np.hanning(n_samples - 8000)
    spectrum = np.abs(np.fft.rfft(settled))
    freqs = np.fft.rfftfreq(settled.size, 1.0 / fs)
    np.testing.assert_allclose(
        freqs[np.argmax(spectrum)], f0 * 2.0 ** (cents / 1200.0), rtol=0.05
    )


def test_granular_pitch_shift_seed_controls_the_grain_positions() -> None:
    """Test that seeding makes the random grain placement reproducible"""

    # parameters
    rng = np.random.default_rng(seed=5)

    # input signal
    n_samples = 8000
    in_sig = _noise(rng=rng, length=n_samples, channels=1)

    # td engine
    def render(seed: int) -> np.ndarray:
        return td.GranularPitchShift(1, 8192, 2048, 700.0, [0], seed=seed).filter(
            in_sig
        )

    # Test
    np.testing.assert_array_equal(render(0), render(0))
    assert not np.allclose(render(0), render(1))


def test_pitch_shifters_reject_buffers_that_cannot_hold_a_window() -> None:
    """Test the buffer-size guards of both pitch shifters"""

    with pytest.raises(ValueError, match="max_delay_samps must be"):
        td.PitchShift(1, 1024, 1024, 700.0, 48_000.0, [0])
    with pytest.raises(ValueError, match="max_delay_samps must be"):
        td.GranularPitchShift(1, 1024, 1024, 700.0, [0])
    with pytest.raises(ValueError, match="fade_ratio must be"):
        td.GranularPitchShift(1, 8192, 1024, 700.0, [0], fade_ratio=0.8)
