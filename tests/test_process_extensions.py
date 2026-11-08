"""Tests for FIR feedback matrices and absorption filters (infrastructure for
the paraunitary / scattering / pole-boundary examples)."""

from __future__ import annotations

import numpy as np
import pytest

import pyFDN
from pyFDN import td
from pyFDN.auxiliary.math import general_char_poly
from pyFDN.dss import DSS
from pyFDN.translate.dss_to_impz import dss_to_impz

FDNFixture = dict[str, np.ndarray]


@pytest.fixture()  # type: ignore[misc]
def paraunitary_fdn() -> FDNFixture:
    np.random.seed(0)
    fbm, _ = pyFDN.cascaded_paraunitary_matrix(4, 2, matrix_type="random")
    return {
        "A": fbm,
        "delays": np.array([19, 31, 42, 57]),
        "B": np.eye(4, 1),
        "C": np.eye(1, 4),
        "D": np.zeros((1, 1)),
    }


def _evaluate_loop(A: np.ndarray, delays: np.ndarray, z: complex) -> np.ndarray:
    """P(z) = diag(z^m) - A(z) for polynomial A in z^{-1} convention."""
    ks = np.arange(A.shape[2])
    a_val = np.sum(A * (z**-ks)[None, None, :], axis=2)
    return np.diag(z ** delays.astype(float)) - a_val


def test_general_char_poly_polynomial_matches_det(
    paraunitary_fdn: FDNFixture,
) -> None:
    """Regression: polyphase GCP coefficients land at p_ind + j (not p_ind - j)."""
    A = paraunitary_fdn["A"]
    delays = paraunitary_fdn["delays"]
    p = general_char_poly(delays, A)
    for z in [0.9 * np.exp(0.7j), 1.1 * np.exp(2.1j), 1.3 * np.exp(4.0j)]:
        det_p = np.linalg.det(_evaluate_loop(A, delays, z))
        gcp_val = np.polyval(p[::-1], 1.0 / z) * z ** int(delays.sum())
        assert det_p == pytest.approx(gcp_val, rel=1e-8)


def test_fir_feedback_on_the_td_graph_matches_frequency_inversion(
    paraunitary_fdn: FDNFixture,
) -> None:
    A = paraunitary_fdn["A"]
    delays = paraunitary_fdn["delays"]
    B, C, D = paraunitary_fdn["B"], paraunitary_fdn["C"], paraunitary_fdn["D"]
    ir_len = 1024
    graph = pyFDN.dss_to_td(delays, np.eye(4), B, C, D, post_matrix=td.MatrixFIR(A))
    ir_time = graph.impulse_response(ir_len)[:, 0, 0]

    # inverse z-transform on circle |z| = r > 1 (lossless system: unit-circle poles)
    nfft = 2**14
    r = 1.002  # r^{-nfft} ≈ 6e-15 keeps circular aliasing below tolerance
    zs = r * np.exp(2j * np.pi * np.arange(nfft) / nfft)
    H = np.array(
        [(C @ np.linalg.solve(_evaluate_loop(A, delays, z), B))[0, 0] for z in zs]
    )
    ir_freq = (np.fft.ifft(H) * r ** np.arange(nfft)).real[:ir_len]
    np.testing.assert_allclose(ir_time, ir_freq, atol=1e-9)


def test_dss_to_pr_rejects_polynomial_matrix(
    paraunitary_fdn: FDNFixture,
) -> None:
    """Polynomial (FIR) feedback matrices are not supported by the direct path;
    use the FLAMO-based decomposition instead."""
    with pytest.raises(ValueError):
        pyFDN.dss_to_pr(
            paraunitary_fdn["delays"],
            paraunitary_fdn["A"],
            paraunitary_fdn["B"],
            paraunitary_fdn["C"],
            paraunitary_fdn["D"],
            mode="roots",
        )


def test_fir_matrix_filter_block_consistency() -> None:
    np.random.seed(3)
    coeffs = np.random.randn(3, 2, 8)
    x = np.random.randn(200, 2)

    one_shot = td.MatrixFIR(coeffs).process_block(x)
    blockwise = td.MatrixFIR(coeffs)
    parts = [blockwise.process_block(x[i : i + 32]) for i in range(0, 200, 32)]
    np.testing.assert_allclose(one_shot, np.vstack(parts), atol=1e-12)

    # order-1 (static) matrix degenerates to a matrix multiply
    static = td.MatrixFIR(coeffs[:, :, :1]).process_block(x)
    np.testing.assert_allclose(static, x @ coeffs[:, :, 0].T, atol=1e-12)


def test_sos_filter_bank_block_consistency_and_shapes() -> None:
    from scipy.signal import sosfilt

    np.random.seed(4)
    n = 3
    fs = 48000
    delays = np.array([100, 200, 300])
    sos = pyFDN.decay_to_first_order_shelf(0.3, 0.1, delays, fs)  # (1, 6, N)
    assert sos.shape == (1, 6, n)
    x = np.random.randn(200, n)

    # block-wise filtering with persistent state matches one-shot sosfilt
    bank = td.SOSBank(sos)
    parts = [bank.process_block(x[i : i + 32]) for i in range(0, 200, 32)]
    blockwise = np.vstack(parts)
    for i in range(n):
        one_shot = sosfilt(
            np.ascontiguousarray(sos[:, :, i]),  # (n_sections, 6) for channel i
            np.ascontiguousarray(x[:, i]),
        )
        np.testing.assert_allclose(blockwise[:, i], one_shot, atol=1e-12)

    # only the canonical (n_sections, 6, N) layout is accepted; others raise
    for bad in (sos[0], sos.transpose(2, 0, 1), np.zeros((5, n))):
        with pytest.raises(ValueError, match="shape"):
            td.SOSBank(bad)


def test_paraunitary_from_elementals_is_paraunitary() -> None:
    np.random.seed(5)
    n, degree = 3, 5
    matrix, v = pyFDN.paraunitary_from_elementals(n, degree)
    assert matrix.shape == (n, n, degree)
    assert v.shape == (n, degree - 1)
    is_p, _, _ = pyFDN.is_paraunitary(matrix)
    assert is_p


def test_process_dss_absorption_matches_flamo() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("flamo")

    np.random.seed(2)
    n = 4
    fs = 48000
    delays = np.array([101, 143, 165, 177])
    A = pyFDN.random_orthogonal(n)
    B = np.ones((n, 1))
    C = np.ones((1, n))
    D = np.zeros((1, 1))
    # short RTs so the IR decays well within nfft (FLAMO is circular)
    sos = pyFDN.decay_to_first_order_shelf(0.15, 0.05, delays, fs)  # (1, 6, N)

    # constract the SOSFilter
    absorption = td.SOSBank(sos)

    ir_len = 4096
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0
    ir_td = pyFDN.dss_to_td(delays, A, B, C, D, post_delay=absorption).process_signal(
        impulse, squeeze=True
    )

    model = pyFDN.dss_to_flamo(
        delays,
        A,
        B,
        C,
        D,
        fs,
        nfft=2**14,
        post_delay=sos,
        dtype=torch.float64,
    )
    ir_flamo = pyFDN.flamo_time_response(model).squeeze()[:ir_len]
    np.testing.assert_allclose(ir_td, ir_flamo, atol=1e-6)


def test_dss_to_flamo_output_filter_matches_sosfilt() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("flamo")
    from scipy.signal import sosfilt

    np.random.seed(3)
    n = 4
    fs = 48000
    delays = np.array([101, 143, 165, 177])
    A = pyFDN.random_orthogonal(n) * 0.6  # decays well within nfft (FLAMO is circular)
    B = np.ones((n, 1))
    C = np.ones((1, n))
    D = np.zeros((1, 1))

    eq_sos = pyFDN.gain_to_geq(np.linspace(-6.0, 6.0, 10), fs=fs)

    def build(output_filter):
        return pyFDN.dss_to_flamo(
            delays,
            A,
            B,
            C,
            D,
            fs,
            nfft=2**14,
            post_output=output_filter,
            dtype=torch.float64,
        )

    ir_len = 4096
    ir_plain = pyFDN.flamo_time_response(build(None)).squeeze()[:ir_len]
    ir_eq = pyFDN.flamo_time_response(build(eq_sos[:, :, np.newaxis])).squeeze()[
        :ir_len
    ]
    np.testing.assert_allclose(ir_eq, sosfilt(eq_sos, ir_plain), atol=1e-8)


# ============================================================================
# DSS systems with filter hooks (dss_to_td / dss_to_flamo)
# ============================================================================

FS = 48_000.0


def _siso_dss() -> DSS:
    n = 4
    return DSS(
        delays=np.array([101, 143, 165, 177]),
        A=pyFDN.random_orthogonal(n),
        B=np.ones((n, 1)),
        C=np.ones((1, n)),
        D=np.zeros((1, 1)),
    )


def _decay(dss: DSS, rt: float) -> np.ndarray:
    return pyFDN.decay_to_first_order_shelf(rt, rt, dss.delays, FS)


def _flat_gain_sos(gain: float, n_channels: int) -> np.ndarray:
    """A flat gain as a one-section per-channel SOS bank."""
    sos = np.zeros((1, 6, n_channels))
    sos[:, 0, :] = gain
    sos[:, 3, :] = 1.0
    return sos


def test_dss_to_td_sos_arrays_match_sosbank_objects() -> None:
    """SOS arrays given as hooks become the same SOSBank nodes."""
    dss = _siso_dss()
    filters = {
        "post_delay": pyFDN.decay_to_first_order_shelf(0.3, 0.1, dss.delays, FS),
        "post_matrix": _flat_gain_sos(0.8, 4),
        "post_output": _flat_gain_sos(0.5, 1),
    }
    signal = np.zeros(2048)
    signal[0] = 1.0

    got = pyFDN.dss_to_td(*dss, **filters).process_signal(signal, squeeze=True)
    sosbanks = {k: td.SOSBank(v) for k, v in filters.items()}
    expected = pyFDN.dss_to_td(*dss, **sosbanks).process_signal(signal, squeeze=True)
    np.testing.assert_allclose(got, expected)


def test_dss_to_td_fir_feedback_mimo_and_direct_path() -> None:
    rng = np.random.default_rng(17)
    n, num_inputs, num_outputs = 4, 2, 3
    fir, _ = pyFDN.cascaded_paraunitary_matrix(n, 2, matrix_type="random")
    dss = DSS(
        delays=np.array([19, 31, 42, 57]),
        A=np.eye(n),
        B=rng.standard_normal((n, num_inputs)),
        C=rng.standard_normal((num_outputs, n)),
        D=rng.standard_normal((num_outputs, num_inputs)),
    )
    signal = rng.standard_normal((512, num_inputs))

    small = pyFDN.dss_to_td(*dss, post_matrix=td.MatrixFIR(0.8 * fir), block_size=16)
    default = pyFDN.dss_to_td(*dss, post_matrix=td.MatrixFIR(0.8 * fir))
    np.testing.assert_allclose(
        small.process_signal(signal), default.process_signal(signal), atol=1e-12
    )


def test_a_lossless_fir_fdn_renders_through_flamo_with_alias_decay() -> None:
    """A FIR matrix module carries the model's alias decay, like every hook."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("flamo")
    fir, _ = pyFDN.cascaded_paraunitary_matrix(4, 2, matrix_type="random")
    dss = DSS(
        np.array([19, 31, 42, 57]),
        np.eye(4),
        np.eye(4, 1),
        np.eye(1, 4),
        np.zeros((1, 1)),
    )
    nfft, alias = 2**12, 60.0
    model = pyFDN.dss_to_flamo(
        *dss,
        48_000.0,
        nfft=nfft,
        alias_decay_db=alias,
        dtype=torch.float64,
        device="cpu",
        post_matrix=pyFDN.fir_matrix_module(
            fir, nfft, alias_decay_db=alias, dtype=torch.float64, device="cpu"
        ),
    )
    reference = pyFDN.dss_to_td(*dss, post_matrix=td.MatrixFIR(fir)).impulse_response(
        nfft
    )
    error = np.linalg.norm(pyFDN.flamo_time_response(model) - reference)
    # the residual is the alias decay, by construction
    assert 20 * np.log10(error / np.linalg.norm(reference)) < -alias + 2


def test_dss_to_td_process_block_matches_process_signal() -> None:
    dss = _siso_dss()
    signal = np.zeros(2048)
    signal[0] = 1.0

    graph = pyFDN.dss_to_td(*dss, post_delay=_decay(dss, 0.3), block_size=32)
    whole = graph.process_signal(signal)
    streamed_graph = pyFDN.dss_to_td(*dss, post_delay=_decay(dss, 0.3), block_size=32)
    streamed = np.vstack(
        [
            streamed_graph.process_block(signal[start : start + 47])
            for start in range(0, signal.size, 47)
        ]
    )
    np.testing.assert_allclose(streamed, whole, atol=1e-12)


def test_dss_to_td_reset_restores_initial_state() -> None:
    dss = _siso_dss()
    graph = pyFDN.dss_to_td(*dss, post_delay=_decay(dss, 0.3), block_size=32)
    signal = np.zeros(2048)
    signal[0] = 1.0

    first = graph.process_signal(signal)
    assert not np.allclose(graph.process_signal(signal), first)
    graph.reset()
    np.testing.assert_allclose(graph.process_signal(signal), first)


def test_dss_to_td_rejects_invalid_block_size() -> None:
    dss = _siso_dss()
    with pytest.raises(ValueError, match="positive integer"):
        pyFDN.dss_to_td(*dss, block_size=0)
    with pytest.raises(ValueError, match="shortest FDN delay"):
        pyFDN.dss_to_td(*dss, block_size=102)


def test_td_impulse_response_applies_absorption_decay() -> None:
    np.random.seed(3)
    dss = _siso_dss()

    ir_len = 16384
    ir_loss = dss_to_impz(*dss, ir_len).squeeze()
    graph = pyFDN.dss_to_td(*dss, post_delay=_decay(dss, 0.3))
    ir_dec = graph.impulse_response(ir_len).squeeze()

    # A lossless orthogonal FDN rings on; a 0.3 s RT is nearly gone by the
    # second half of the buffer, so its tail energy is far lower.
    tail = slice(ir_len // 2, None)
    assert np.sum(ir_dec[tail] ** 2) < 1e-2 * np.sum(ir_loss[tail] ** 2)


def test_all_three_hooks_render_alike_and_extract_back() -> None:
    """post_matrix is a hook like the other two, and both renderers honour it."""
    pytest.importorskip("flamo")
    dss = _siso_dss()
    post_matrix = _flat_gain_sos(0.5, 4)
    ir_len = 4096
    graph = pyFDN.dss_to_td(*dss, post_matrix=post_matrix)
    numpy_ir = graph.impulse_response(ir_len).squeeze()
    model = pyFDN.dss_to_flamo(
        *dss, FS, nfft=2**14, device="cpu", post_matrix=post_matrix
    )
    flamo_ir = np.asarray(pyFDN.flamo_time_response(model)).reshape(-1)
    np.testing.assert_allclose(numpy_ir, flamo_ir[:ir_len], atol=1e-4)

    # and it is not a no-op: halving the feedback path shortens the decay
    plain = dss_to_impz(*dss, ir_len).squeeze()
    assert not np.allclose(numpy_ir, plain, atol=1e-3)

    # the graph round-trips it back into the same hook
    _, filters = pyFDN.flamo_to_dss(model)
    np.testing.assert_allclose(filters["post_matrix"], post_matrix, atol=1e-6)


def test_td_impulse_response_applies_the_post_output_hook() -> None:
    """post_output becomes the graph's wet-output SOS node."""
    dss = _siso_dss()
    np.testing.assert_allclose(
        pyFDN.dss_to_td(*dss, post_output=_flat_gain_sos(0.5, 1)).impulse_response(256),
        0.5 * dss_to_impz(*dss, 256),
        atol=1e-12,
    )


def test_delay_lines_can_be_filled_and_reset_to_zero() -> None:
    rng = np.random.default_rng(0)
    delays = np.array([7, 11, 13])
    N = delays.size
    lines = [rng.standard_normal(m) for m in delays]
    B, C, D = np.ones((N, 1)), np.eye(N), np.zeros((N, 1))

    # No feedback: each line plays its content out once, in output order,
    # also when filled mid-stream.
    graph = pyFDN.dss_to_td(delays, np.zeros((N, N)), B, C, D, block_size=5)
    graph.process_signal(np.zeros(3))
    state = graph.ops[0].ops[1].state
    state.set_lines(lines)
    y = graph.process_signal(np.zeros(20))
    for i, m in enumerate(delays):
        np.testing.assert_array_equal(y[:m, i], lines[i])
        np.testing.assert_array_equal(y[m:, i], 0.0)

    graph.reset()
    np.testing.assert_array_equal(graph.process_signal(np.zeros(20)), 0.0)
    state.fill_noise(0.5, rng)
    assert np.all(graph.process_signal(np.zeros(7)) != 0.0)

    with pytest.raises(ValueError, match="lines"):
        state.set_lines([s[:-1] for s in lines])


@pytest.mark.parametrize(
    "bad_line", [np.zeros(2), [0.0, np.nan, 0.0], [0.0, np.inf, 0.0]]
)
def test_set_lines_rejects_invalid_later_line_without_changing_state(bad_line) -> None:
    state = td.RecursionState([2, 3], max_block_size=1)
    state.set_lines([[1.0, 2.0], [3.0, 4.0, 5.0]])
    state.advance(1)
    before = state.buffer.copy()
    pointers = state.pointers.copy()
    with pytest.raises(ValueError, match=r"lines\[1\]"):
        state.set_lines([[8.0, 9.0], bad_line])
    np.testing.assert_array_equal(state.buffer, before)
    np.testing.assert_array_equal(state.pointers, pointers)
    with pytest.raises(ValueError, match="one array per delay line"):
        state.set_lines([[8.0, 9.0]])
    np.testing.assert_array_equal(state.buffer, before)


@pytest.mark.parametrize("variance", [-1.0, np.nan, np.inf, -np.inf])
def test_fill_noise_rejects_invalid_variance_without_changing_state(variance) -> None:
    state = td.RecursionState([2, 3], max_block_size=1)
    state.set_lines([[1.0, 2.0], [3.0, 4.0, 5.0]])
    before = state.buffer.copy()
    with pytest.raises(ValueError, match="variance"):
        state.fill_noise(variance)
    np.testing.assert_array_equal(state.buffer, before)


def test_fill_noise_is_reproducible_and_preserves_read_positions() -> None:
    state = td.RecursionState([2, 3], max_block_size=1)
    state.advance(1)
    pointers = state.pointers.copy()
    state.fill_noise(0.5, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    expected = [rng.normal(0.0, np.sqrt(0.5), m) for m in state.delays]
    for i, m in enumerate(state.delays):
        indices = (state.pointers[i] + np.arange(m)) % m
        np.testing.assert_array_equal(state.buffer[i, indices], expected[i])
    np.testing.assert_array_equal(state.pointers, pointers)
    state.fill_noise(0.0)
    np.testing.assert_array_equal(state.buffer, 0.0)
    np.testing.assert_array_equal(state.pointers, pointers)


def test_initialized_delay_line_decays_with_feedback_and_input() -> None:
    graph = pyFDN.dss_to_td([3], [[0.5]], [[1.0]], [[1.0]], [[0.0]], block_size=2)
    graph.ops[0].ops[1].state.set_lines([[1.0, 2.0, 3.0]])
    x = np.zeros(9)
    x[0] = 4.0
    expected = np.array([1.0, 2.0, 3.0, 4.5, 1.0, 1.5, 2.25, 0.5, 0.75])
    np.testing.assert_array_equal(graph.process_signal(x).reshape(-1), expected)
