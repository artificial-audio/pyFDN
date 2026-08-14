"""Tests for the time-domain block-based graph engine (``pyFDN.td``)
Graph processing

The audio processing of graphs built with ``pyFDN.td`` is tested agains
:func:`pyFDN.process_fdn`.
This needs no FLAMO install.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pyFDN
from pyFDN import td
from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix as _DSPTimeVaryingMatrix
from pyFDN.td.connectors import _RECURSION_BLOCK_SIZE

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


def _impulse(length: int, channels: int) -> np.ndarray:
    """Unit impulse"""
    x = np.zeros((length, channels))
    x[0, :] = 1.0
    return x


# ============================================================
# ====================== TEST CONNECTORS =====================


def test_series_filter() -> None:
    """Test td Series connector filtering"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    max_block_size = 256

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    B_ref = fdnbuild.B
    C_ref = fdnbuild.C
    delays_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=max_block_size
    )

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in)

    # td engine
    in_gains_td = td.Gain(B_ref)
    delays_td = td.Delay(delays=fdnbuild.delays)
    out_gains_td = td.Gain(C_ref)
    series_conn = td.Series([in_gains_td, delays_td, out_gains_td])

    # filter signal
    out_sig_ref = np.zeros((n_samples, n_out), dtype=float)
    out_sig_td = np.zeros((n_samples, n_out), dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        out_sig_ref[start : start + block_size, :] = (
            delays_ref.get_values(block_size) @ C_ref.T
        )
        delays_ref.set_values(block_in @ B_ref.T)
        delays_ref.advance(block_size)

        # td
        out_sig_td[start : start + block_size, :] = series_conn.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_parallel_filter() -> None:
    """Test td Parallel connector filtering"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    max_block_size = 256

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    B_ref = fdnbuild.B
    C_ref = fdnbuild.C
    D_ref = fdnbuild.D
    delays_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=max_block_size
    )

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in)

    # td engine
    in_gains_td = td.Gain(B_ref)
    delays_td = td.Delay(delays=fdnbuild.delays)
    out_gains_td = td.Gain(C_ref)
    series_conn = td.Series([in_gains_td, delays_td, out_gains_td])
    parallel_conn = td.Parallel([series_conn, td.Gain(D_ref)], sum_output=True)

    # filter signal
    out_sig_ref = np.zeros((n_samples, n_out), dtype=float)
    out_sig_td = np.zeros((n_samples, n_out), dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        out_sig_ref[start : start + block_size, :] = (
            delays_ref.get_values(block_size) @ C_ref.T + block_in @ D_ref.T
        )
        delays_ref.set_values(block_in @ B_ref.T)
        delays_ref.advance(block_size)

        # td engine
        out_sig_td[start : start + block_size, :] = parallel_conn.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_recursion_filter_forward_delay() -> None:
    """Test td Recursion connector filtering.
    The delay lines are positioned along the forward path.
    The inherent delay is positioned along the forward path."""

    # parameters
    n_in, n_out, N = 4, 4, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    A_ref = 0.2 * fdnbuild.A
    I_ref = pyFDN.fdn_matrix_gallery(N=N, matrix_type="parallel")
    print(fdnbuild.delays)

    # input signal
    n_samples = 4000
    in_sig = _impulse(length=n_samples, channels=N)

    # td engine
    delays_td = td.Delay(delays=fdnbuild.delays)
    A_td = td.Gain(A_ref)
    recursion_conn = td.Recursion(
        forward=delays_td, feedback=A_td, delay_position="forward"
    )

    # filter signal
    out_sig_ref = np.zeros((n_samples, N), dtype=float)
    out_sig_td = np.zeros((n_samples, N), dtype=float)

    out_sig_ref = pyFDN.process_fdn(
        input_signal=in_sig,
        delays=fdnbuild.delays + _RECURSION_BLOCK_SIZE,
        A=A_ref,
        B=I_ref,
        C=I_ref,
        D=0 * I_ref,
    )

    out_sig_td = recursion_conn.filter(in_sig)

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_recursion_filter_forward_delay_in_block_processing() -> None:
    """Test td Recursion connector filtering in a block-processing pipeline.
    The delay lines are positioned along the forward path.
    The inherent delay is positioned along the forward path."""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    A_ref = 0.2 * fdnbuild.A
    # reference with the inclusion of the Recursion inherent delay
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    total_delay = fdnbuild.delays + rec_inherent_delay
    max_block_size = int(np.min(total_delay))
    delay_ref = pyFDN.FeedbackDelay(delays=total_delay, max_block_size=max_block_size)
    print(fdnbuild.delays)

    # input signal
    n_samples = 4000
    in_sig = _impulse(length=n_samples, channels=N)

    # td engine
    delays_td = td.Delay(delays=fdnbuild.delays)
    A_td = td.Gain(A_ref)
    recursion_conn = td.Recursion(
        forward=delays_td, feedback=A_td, delay_position="forward"
    )

    # filter signal
    out_sig_ref = np.zeros((n_samples, N), dtype=float)
    out_sig_td = np.zeros((n_samples, N), dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        fw_out = delay_ref.get_values(block_size)
        fb_out = fw_out @ A_ref.T
        delay_ref.set_values(block_in + fb_out)
        out_sig_ref[start : start + block_size] = fw_out
        delay_ref.advance(block_size)

        # td engine
        out_sig_td[start : start + block_size] = recursion_conn.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_recursion_filter_feedback_delay_in_block_processing() -> None:
    """Test td Recursion connector filtering in a block-processing pipeline.
    The delay lines are positioned along the forward path.
    The inherent delay is positioned along the feedback path."""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    A_ref = 0.2 * fdnbuild.A
    # reference with the inclusion of the Recursion inherent delay
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    max_block_size = min([int(np.min(fdnbuild.delays)), rec_inherent_delay])
    delay_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=max_block_size
    )
    add_delay_ref = pyFDN.FeedbackDelay(
        delays=np.ones(N) * rec_inherent_delay, max_block_size=rec_inherent_delay
    )
    print(fdnbuild.delays)

    # input signal
    n_samples = 4000
    in_sig = _impulse(length=n_samples, channels=N)

    # td engine
    delays_td = td.Delay(delays=fdnbuild.delays)
    A_td = td.Gain(A_ref)
    with pytest.warns(
        UserWarning,
        match="Minimum delay in feedback path cannot be shorter than Recursion internal block size, and cannot be compensated",
    ):
        recursion_conn = td.Recursion(
            forward=delays_td, feedback=A_td, delay_position="feedback"
        )

    # filter signal
    out_sig_ref = np.zeros((n_samples, N), dtype=float)
    out_sig_td = np.zeros((n_samples, N), dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        state = add_delay_ref.get_values(block_size)

        fw_out = delay_ref.get_values(block_size)
        fb_out = fw_out @ A_ref.T
        out_sig_ref[start : start + block_size] = fw_out

        add_delay_ref.set_values(fb_out)
        add_delay_ref.advance(block_size)

        delay_ref.set_values(block_in + state)
        delay_ref.advance(block_size)

        # td engine
        out_sig_td[start : start + block_size] = recursion_conn.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_recursion_filter_forward_delay_compensation() -> None:
    """Test td Recursion connector filtering in a block-processing pipeline.
    The delay lines are positioned along the forward path.
    The inherent delay is positioned along the forward path.
    We compensate for that delay shortening the delay lines."""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    A_ref = 0.2 * fdnbuild.A
    max_block_size = min(int(2**12), int(np.min(fdnbuild.delays)))
    delay_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=max_block_size
    )
    print(fdnbuild.delays)

    # input signal
    n_samples = 4000
    in_sig = _impulse(length=n_samples, channels=N)

    # td engine
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    delays_td = td.Delay(delays=fdnbuild.delays - rec_inherent_delay)
    A_td = td.Gain(A_ref)
    recursion_conn = td.Recursion(
        forward=delays_td, feedback=A_td, delay_position="forward"
    )

    # filter signal
    out_sig_ref = np.zeros((n_samples, N), dtype=float)
    out_sig_td = np.zeros((n_samples, N), dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        fw_out = delay_ref.get_values(block_size)
        fb_out = fw_out @ A_ref.T
        out_sig_ref[start : start + block_size] = fw_out
        delay_ref.set_values(block_in + fb_out)
        delay_ref.advance(block_size)

        # td engine
        out_sig_td[start : start + block_size] = recursion_conn.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


def test_recursion_filter_feedback_delay_compensation() -> None:
    """Test td Recursion connector filtering in a block-processing pipeline.
    The delay lines are positioned along the feedback path.
    The inherent delay is positioned along the feedback path.
    We compensate for that delay shortening the delay lines."""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    A_ref = 0.2 * fdnbuild.A
    max_block_size = min(int(2**12), int(np.min(fdnbuild.delays)))
    delay_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=max_block_size
    )
    print(fdnbuild.delays)

    # input signal
    n_samples = 4000
    in_sig = _impulse(length=n_samples, channels=N)

    # td engine
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    delays_td = td.Delay(delays=fdnbuild.delays - rec_inherent_delay)
    A_td = td.Gain(A_ref)
    recursion_conn = td.Recursion(
        forward=A_td, feedback=delays_td, delay_position="feedback"
    )

    # filter signal
    out_sig_ref = np.zeros((n_samples, N), dtype=float)
    out_sig_td = np.zeros((n_samples, N), dtype=float)

    start = 0
    while start < n_samples:
        block_size = min(max_block_size, n_samples - start)
        block_in = in_sig[start : start + block_size, :]

        # reference
        fb_out = delay_ref.get_values(block_size)
        fw_out = (block_in + fb_out) @ A_ref.T
        out_sig_ref[start : start + block_size] = fw_out
        delay_ref.set_values(fw_out)
        delay_ref.advance(block_size)

        # td engine
        out_sig_td[start : start + block_size] = recursion_conn.filter(block_in)

        start += block_size

    # Test
    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-12, rtol=0)


# ============================================================
# ===================== TEST LARGE GRAPHS ====================


def test_res_mcr_block_process() -> None:
    """Test that the recursion with a MatrixConvolver feedback
    (the RES room coupling) is exact.
    Uses some system latency as well."""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    sys_latency = 10

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=N) * 0.1  # microphone noise

    # reference
    # Physical room
    rir_len = 200
    rirs = rng.standard_normal((N, N, rir_len)) * 0.5  # small -> stable loop
    feedback_ref = pyFDN.FIRMatrixFilter(rirs)
    # virtual room
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    max_block_size = int(np.min(fdnbuild.delays + sys_latency))
    delay_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays + sys_latency, max_block_size=max_block_size
    )
    mixing_ref = 0.5 * fdnbuild.A

    # td engine
    # Physical room
    feedback_td = td.MatrixConvolver(rirs)
    # Virtual room
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    latency_td = td.Delay(np.ones_like(fdnbuild.delays) * sys_latency)
    delay_td = td.Delay(fdnbuild.delays - rec_inherent_delay)
    mixing_td = td.Gain(mixing_ref)
    forward_td = td.Series([latency_td, delay_td, mixing_td])
    # full RES system
    res = td.Recursion(
        forward=forward_td,
        feedback=feedback_td,
    )

    # process signal
    out_sig_ref = np.zeros((n_samples, N))
    out_sig_td = np.zeros((n_samples, N))

    n = 0
    while n < n_samples:
        block_size = min(max_block_size, n_samples - n)
        block_in = in_sig[n : n + block_size, :]

        # --------- Reference processing ---------
        # delay lines
        delay_out = delay_ref.get_values(block_size)
        # mixing matrix
        mix_out = delay_out @ mixing_ref.T
        # output
        out_sig_ref[n : n + block_size, :] = mix_out
        # feedback convolution
        feedback = feedback_ref.filter(mix_out)
        # update buffers
        delay_ref.set_values(block_in + feedback)
        delay_ref.advance(block_size)

        # ------------- td processing ------------
        out_sig_td[n : n + block_size, :] = res.filter(block_in)

        n += block_size

    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-10, rtol=0)


def test_res_tvfdn_block_process():
    """Test a time-varying standard FDN in a RES.
    The internal Recursion (FDN) is compensated.
    The external Recursion (acoustic feedback) is not compensated.
    No system latency."""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    # Time variation parameters
    tvm_kwargs = _time_varying_parameters(N=N, fs=fs)

    # input signal
    n_samples = 4000
    in_sig = _noise(rng=rng, length=n_samples, channels=n_in) * 0.1  # microphone noise

    # reference
    # Physical room
    rir_len = 200
    rirs = rng.standard_normal((n_in, n_out, rir_len)) * 0.5  # small -> stable loop
    acoustic_feedback_ref = pyFDN.FIRMatrixFilter(rirs)
    # reference with the inclusion of the Recursion inherent delay
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    max_block_size = rec_inherent_delay
    add_delay_ref = pyFDN.FeedbackDelay(
        delays=np.ones(2) * rec_inherent_delay, max_block_size=rec_inherent_delay
    )
    # Virtual room
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )
    delay_ref = pyFDN.FeedbackDelay(
        delays=fdnbuild.delays, max_block_size=np.min(fdnbuild.delays)
    )
    A_ref = fdnbuild.A
    B_ref = fdnbuild.B
    C_ref = fdnbuild.C
    sos = _absorption_sos(fdnbuild.delays, fs)
    absorp_ref = pyFDN.SOSFilterBank(sos, N)

    # Generate tv matrices
    # Identical modulation in both renders: seed the global RNG the TVM draws from.
    np.random.seed(7)
    tvm_ref = _DSPTimeVaryingMatrix(**tvm_kwargs)
    np.random.seed(7)
    tvm_td = td.TimeVaryingMatrix(_DSPTimeVaryingMatrix(**tvm_kwargs))

    # td engine
    # Physical room
    acoustic_feedback_td = td.MatrixConvolver(rirs)
    # Virtual room
    input_gains_td = td.Gain(fdnbuild.B)
    rec_inherent_delay = _RECURSION_BLOCK_SIZE
    delay_lines_td = td.Delay(fdnbuild.delays - rec_inherent_delay)
    absorp_filters_td = td.SOSBank(sos)
    output_gains_td = td.Gain(fdnbuild.C)
    feedback_matrix_td = td.Gain(fdnbuild.A)

    res_dsp = td.Series(
        [
            input_gains_td,
            td.Recursion(
                forward=td.Series([delay_lines_td, absorp_filters_td]),
                feedback=td.Series([feedback_matrix_td, tvm_td]),
                delay_position="forward",
            ),
            output_gains_td,
        ]
    )
    # full RES system
    with pytest.warns(
        UserWarning,
        match="Minimum delay in feedback path cannot be shorter than Recursion internal block size, and cannot be compensated",
    ):
        res = td.Recursion(
            forward=res_dsp, feedback=acoustic_feedback_td, delay_position="feedback"
        )

    # process signal
    out_sig_ref = np.zeros((n_samples, n_out))
    out_sig_td = np.zeros((n_samples, n_out))

    n = 0
    while n < n_samples:
        block_size = min(max_block_size, n_samples - n)
        block_in = in_sig[n : n + block_size, :]

        # --------- reference processing ---------
        # current system state - inherently delays by Recursion
        state = add_delay_ref.get_values(block_size)

        # fdn
        delay_out = delay_ref.get_values(block_size)
        delay_out = absorp_ref.filter(delay_out)
        internal_feedback = tvm_ref.filter(delay_out @ A_ref.T)

        delay_ref.set_values((block_in + state) @ B_ref.T + internal_feedback)
        fdn_out = delay_out @ C_ref.T

        out_sig_ref[n : n + block_size] = fdn_out
        delay_ref.advance(block_size)

        # acoustic feedback
        ac_fb_out = acoustic_feedback_ref.filter(fdn_out)

        # future system state
        add_delay_ref.set_values(ac_fb_out)
        add_delay_ref.advance(block_size)

        # ------------- td processing ------------
        out_sig_td[n : n + block_size, :] = res.filter(block_in)

        n += block_size

    np.testing.assert_allclose(out_sig_td, out_sig_ref, atol=1e-10, rtol=0)
