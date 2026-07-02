"""Tests for the time-domain FLAMO-graph engine (``pyFDN.td``).

Two angles:

* The hand-built operator tree is checked against :func:`pyFDN.process_fdn`,
  which needs no flamo install -- this exercises the recursion / series /
  parallel engine directly.
* The compiler is checked end-to-end by building a FLAMO model with
  :func:`pyFDN.dss_to_flamo`, compiling it, and matching both
  :func:`pyFDN.process_fdn` (exact, same time-domain math) and the FLAMO
  frequency-domain render (loose, independent implementation).
"""

from __future__ import annotations

import numpy as np
import pytest

import pyFDN
from pyFDN import td


def _fdn_params(seed: int = 5):
    rng = np.random.default_rng(seed)
    fs = 48_000.0
    delays = np.array([373, 421, 547, 661])
    n = delays.size
    A = pyFDN.random_orthogonal(n)
    B = np.ones((n, 1)) / n
    C = np.ones((1, n))
    D = np.full((1, 1), 0.25)
    return rng, fs, delays, A, B, C, D


def _absorption_sos(delays: np.ndarray, fs: float) -> np.ndarray:
    # Per-delay first-order shelving absorption, shape (n_sections, 6, N).
    return pyFDN.first_order_absorption(1.5, 0.4, delays, fs, None)


def _impulse(length: int) -> np.ndarray:
    x = np.zeros(length)
    x[0] = 1.0
    return x


def test_engine_matches_process_fdn_handbuilt():
    """Hand-built operator tree == process_fdn (no flamo needed)."""
    _, fs, delays, A, B, C, D = _fdn_params()
    sos = _absorption_sos(delays, fs)
    impulse = _impulse(int(fs) // 12)

    # core = D direct path summed with B -> [delay -> absorption ; fb = A] -> C
    fdn_branch = td.Series(
        [
            td.Gain(B),
            td.Recursion(
                td.Series([td.Delay(delays), td.SOSBank(sos)]),
                td.Gain(A),
            ),
            td.Gain(C),
        ]
    )
    core = td.Parallel([fdn_branch, td.Gain(D)], sum_output=True)

    ir_td = core.process(impulse).squeeze()
    ir_ref = pyFDN.process_fdn(
        impulse, delays, A, B, C, D, absorption=pyFDN.SOSFilterBank(sos, delays.size)
    )

    assert ir_td.shape == ir_ref.shape
    np.testing.assert_allclose(ir_td, ir_ref, atol=1e-12, rtol=0)


def test_engine_lossless_handbuilt():
    """A lossless loop (no in-loop filter, Identity residual) matches process_fdn."""
    _, _fs, delays, A, B, C, D = _fdn_params(seed=1)
    impulse = _impulse(4000)

    fdn_branch = td.Series(
        [td.Gain(B), td.Recursion(td.Delay(delays), td.Gain(A)), td.Gain(C)]
    )
    core = td.Parallel([fdn_branch, td.Gain(D)])

    ir_td = core.process(impulse).squeeze()
    ir_ref = pyFDN.process_fdn(impulse, delays, A, B, C, D)
    np.testing.assert_allclose(ir_td, ir_ref, atol=1e-12, rtol=0)


def test_reset_restores_initial_state():
    _, fs, delays, A, B, C, D = _fdn_params()
    sos = _absorption_sos(delays, fs)
    impulse = _impulse(3000)
    op = td.Series(
        [
            td.Gain(B),
            td.Recursion(td.Series([td.Delay(delays), td.SOSBank(sos)]), td.Gain(A)),
            td.Gain(C),
        ]
    )
    first = op.process(impulse).copy()
    op.reset()
    second = op.process(impulse)
    np.testing.assert_allclose(first, second, atol=0, rtol=0)


def test_time_varying_matrix_matches_process_fdn_extra_matrix():
    """td TimeVaryingMatrix on the feedback path == process_fdn extra_matrix."""
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix as DspTVM

    _, fs, delays, A, B, C, D = _fdn_params()
    impulse = _impulse(8000)
    tvm_kwargs = {
        "N": delays.size,
        "cycles_per_second": 1.3,
        "amplitude": 0.2,
        "fs": fs,
        "spread": 0.1,
    }

    # Identical modulation in both renders: seed the global RNG the TVM draws from.
    np.random.seed(7)
    tvm_td = DspTVM(**tvm_kwargs)
    np.random.seed(7)
    tvm_ref = DspTVM(**tvm_kwargs)

    op = td.Series(
        [
            td.Gain(B),
            td.Recursion(
                td.Delay(delays),
                td.Series([td.Gain(A), td.TimeVaryingMatrix(tvm_td)]),
            ),
            td.Gain(C),
        ]
    )
    ir_td = op.process(impulse).squeeze()
    ir_ref = pyFDN.process_fdn(impulse, delays, A, B, C, D, extra_matrix=tvm_ref)
    # process_fdn adds the direct path D; the hand-built tree above omits it.
    ir_ref = ir_ref - D[0, 0] * impulse
    np.testing.assert_allclose(ir_td, ir_ref, atol=1e-12, rtol=0)


def test_time_varying_matrix_modulates_response():
    """The time-varying loop differs from the static loop (modulation is active)."""
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix as DspTVM

    _, fs, delays, A, B, C, _D = _fdn_params()
    impulse = _impulse(8000)
    np.random.seed(3)
    tvm = DspTVM(N=delays.size, cycles_per_second=2.0, amplitude=0.3, fs=fs, spread=0.0)

    static = td.Series(
        [td.Gain(B), td.Recursion(td.Delay(delays), td.Gain(A)), td.Gain(C)]
    )
    varying = td.Series(
        [
            td.Gain(B),
            td.Recursion(
                td.Delay(delays), td.Series([td.Gain(A), td.TimeVaryingMatrix(tvm)])
            ),
            td.Gain(C),
        ]
    )
    ir_static = static.process(impulse).squeeze()
    ir_varying = varying.process(impulse).squeeze()
    # Same up to the first echo (loop not yet engaged), diverging afterwards.
    assert np.max(np.abs(ir_static[: delays.min()] - ir_varying[: delays.min()])) == 0.0
    assert np.max(np.abs(ir_static - ir_varying)) > 1e-3


def test_matrix_convolver_matches_convolution():
    """MatrixConvolver equals the linear matrix convolution, whole and streamed."""
    rng = np.random.default_rng(0)
    n_out, n_in, taps, length = 3, 2, 17, 400
    coeffs = rng.standard_normal((n_out, n_in, taps))
    x = rng.standard_normal((length, n_in))

    reference = np.zeros((length, n_out))
    for i in range(n_out):
        for j in range(n_in):
            reference[:, i] += np.convolve(x[:, j], coeffs[i, j])[:length]

    whole = td.MatrixConvolver(coeffs).process(x)
    np.testing.assert_allclose(whole, reference, atol=1e-10, rtol=0)

    # Block-by-block streaming must reproduce the whole-signal result.
    conv = td.MatrixConvolver(coeffs)
    streamed = np.concatenate(
        [conv.process(x[s : s + 33]) for s in range(0, length, 33)], axis=0
    )
    np.testing.assert_allclose(streamed, reference, atol=1e-10, rtol=0)

    # Same coefficients, same result as the lfilter-based MatrixFIR.
    np.testing.assert_allclose(
        whole, td.MatrixFIR(coeffs).process(x), atol=1e-10, rtol=0
    )


def test_recursion_with_convolver_feedback():
    """Recursion with a MatrixConvolver feedback (the RES room coupling) is exact."""
    rng = np.random.default_rng(1)
    n, taps, latency, length = 2, 4, 5, 300
    mix = 0.5 * pyFDN.random_orthogonal(n)
    coupling = 0.1 * rng.standard_normal((n, n, taps))  # small -> stable loop
    x = rng.standard_normal((length, n))

    loop = td.Recursion(
        td.Series([td.Delay(np.full(n, latency)), td.Gain(mix)]),
        td.MatrixConvolver(coupling),
    )
    y_td = loop.process(x)

    # Reference: y[n] = mix @ (x[n-L] + (coupling * y)[n-L]).
    y_ref = np.zeros((length, n))
    for k in range(length):
        if k - latency < 0:
            continue
        coupled = np.zeros(n)
        for tau in range(taps):
            if k - latency - tau >= 0:
                coupled += coupling[:, :, tau] @ y_ref[k - latency - tau]
        y_ref[k] = mix @ (x[k - latency] + coupled)
    np.testing.assert_allclose(y_td, y_ref, atol=1e-10, rtol=0)


def test_nested_recursion_matches_analytic():
    """A recursion inside a recursion (RES: FDN within the loop) is exact."""
    a, b, outer_delay, inner_delay, length = 0.5, 0.4, 7, 11, 250
    inner = td.Recursion(td.Delay([inner_delay]), td.Gain([[b]]))  # comb, gain b
    outer = td.Recursion(td.Series([td.Delay([outer_delay]), inner]), td.Gain([[a]]))
    impulse = _impulse(length).reshape(-1, 1)
    y_td = outer.process(impulse).squeeze()

    # y[n] = x[n-L-d] + a*y[n-L-d] + b*y[n-d]  (L=outer_delay, d=inner_delay)
    x = _impulse(length)
    y_ref = np.zeros(length)
    for n in range(length):
        if n - outer_delay - inner_delay >= 0:
            y_ref[n] += x[n - outer_delay - inner_delay]
            y_ref[n] += a * y_ref[n - outer_delay - inner_delay]
        if n - inner_delay >= 0:
            y_ref[n] += b * y_ref[n - inner_delay]
    np.testing.assert_allclose(y_td, y_ref, atol=1e-10, rtol=0)


def test_compile_flamo_matches_process_fdn():
    """Compile a real FLAMO graph and match the time-domain reference exactly."""
    pytest.importorskip("flamo")
    import torch

    _, fs, delays, A, B, C, D = _fdn_params()
    sos = _absorption_sos(delays, fs)
    impulse = _impulse(int(fs) // 12)

    model = pyFDN.dss_to_flamo(
        A, B, C, D, delays, fs, nfft=2**15, sos_filter=sos, dtype=torch.float64
    )
    ir_td = td.process(model, impulse)
    ir_ref = pyFDN.process_fdn(
        impulse, delays, A, B, C, D, absorption=pyFDN.SOSFilterBank(sos, delays.size)
    )
    np.testing.assert_allclose(ir_td, ir_ref, atol=1e-10, rtol=0)


def test_compile_flamo_matches_flamo_render():
    """Compiled time-domain render matches FLAMO's frequency-domain render.

    FLAMO renders by circular convolution over ``nfft`` samples, so the reverb
    tail must decay well within ``nfft`` to avoid wrap-around aliasing. A short
    reverberation time and a long ``nfft`` keep the wrap below tolerance; the
    residual is FLAMO's frequency-sampling of the IIR absorption, not the engine.
    """
    pytest.importorskip("flamo")
    import torch

    _, fs, delays, A, B, C, D = _fdn_params()
    # Fast, frequency-flat decay so the tail is deep below tolerance within nfft.
    sos = pyFDN.first_order_absorption(0.3, 0.3, delays, fs, None)
    ir_len = 12_000

    model = pyFDN.dss_to_flamo(
        A, B, C, D, delays, fs, nfft=2**17, sos_filter=sos, dtype=torch.float64
    )
    ir_td = td.process(model, _impulse(ir_len))
    ir_flamo = pyFDN.flamo_time_response(model).squeeze().astype(np.float64)[:ir_len]
    np.testing.assert_allclose(ir_td, ir_flamo, atol=1e-6, rtol=0)
