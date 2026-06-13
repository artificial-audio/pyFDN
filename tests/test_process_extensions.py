"""Tests for FIR feedback matrices and absorption filters (infrastructure for
the paraunitary / scattering / pole-boundary examples)."""

from __future__ import annotations

import numpy as np
import pytest

import pyFDN
from pyFDN.auxiliary.math import general_char_poly
from pyFDN.dsp.dfilt_matrix import FIRMatrixFilter

FDNFixture = dict[str, np.ndarray]


@pytest.fixture()  # type: ignore[misc]
def paraunitary_fdn() -> FDNFixture:
    np.random.seed(0)
    fbm, _ = pyFDN.construct_cascaded_paraunitary_matrix(4, 2, matrix_type="random")
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


def test_process_fdn_fir_matrix_matches_frequency_inversion(
    paraunitary_fdn: FDNFixture,
) -> None:
    A = paraunitary_fdn["A"]
    delays = paraunitary_fdn["delays"]
    B, C, D = paraunitary_fdn["B"], paraunitary_fdn["C"], paraunitary_fdn["D"]
    ir_len = 1024
    ir_time = pyFDN.dss_to_impz(ir_len, delays, A, B, C, D)[:, 0, 0]

    # inverse z-transform on circle |z| = r > 1 (lossless system: unit-circle poles)
    nfft = 2**14
    r = 1.002  # r^{-nfft} ≈ 6e-15 keeps circular aliasing below tolerance
    zs = r * np.exp(2j * np.pi * np.arange(nfft) / nfft)
    H = np.array(
        [(C @ np.linalg.solve(_evaluate_loop(A, delays, z), B))[0, 0] for z in zs]
    )
    ir_freq = (np.fft.ifft(H) * r ** np.arange(nfft)).real[:ir_len]
    np.testing.assert_allclose(ir_time, ir_freq, atol=1e-9)


def test_dss_to_pr_direct_rejects_polynomial_matrix(
    paraunitary_fdn: FDNFixture,
) -> None:
    """Polynomial (FIR) feedback matrices are not supported by the direct path;
    use the FLAMO-based decomposition instead."""
    with pytest.raises(ValueError):
        pyFDN.dss_to_pr_direct(
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

    one_shot = FIRMatrixFilter(coeffs).filter(x)
    blockwise = FIRMatrixFilter(coeffs)
    parts = [blockwise.filter(x[i : i + 32]) for i in range(0, 200, 32)]
    np.testing.assert_allclose(one_shot, np.vstack(parts), atol=1e-12)

    # order-1 (static) matrix degenerates to a matrix multiply
    static = FIRMatrixFilter(coeffs[:, :, :1]).filter(x)
    np.testing.assert_allclose(static, x @ coeffs[:, :, 0].T, atol=1e-12)


def test_sos_filter_bank_block_consistency_and_shapes() -> None:
    from scipy.signal import sosfilt

    np.random.seed(4)
    n = 3
    fs = 48000
    delays = np.array([100, 200, 300])
    sos = pyFDN.first_order_absorption(0.3, 0.1, delays, fs)  # (1, 6, N)
    assert sos.shape == (1, 6, n)
    x = np.random.randn(200, n)

    # block-wise filtering with persistent state matches one-shot sosfilt
    bank = pyFDN.SOSFilterBank(sos, n)
    parts = [bank.filter(x[i : i + 32]) for i in range(0, 200, 32)]
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
            pyFDN.SOSFilterBank(bad, n)


def test_construct_paraunitary_from_elementals_is_paraunitary() -> None:
    np.random.seed(5)
    n, degree = 3, 5
    matrix, v = pyFDN.construct_paraunitary_from_elementals(n, degree)
    assert matrix.shape == (n, n, degree)
    assert v.shape == (n, degree - 1)
    is_p, _, _ = pyFDN.is_paraunitary(matrix.transpose(2, 0, 1))
    assert is_p


def test_process_fdn_absorption_matches_flamo() -> None:
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
    sos = pyFDN.first_order_absorption(0.15, 0.05, delays, fs)  # (1, 6, N)

    ir_len = 4096
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0
    ir_td = pyFDN.process_fdn(impulse, delays, A, B, C, D, absorption_filters=sos)

    model = pyFDN.dss_to_flamo(
        A,
        B,
        C,
        D,
        delays,
        fs,
        nfft=2**14,
        sos_filter=sos,
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

    eq_sos, _ = pyFDN.design_geq(np.linspace(-6.0, 6.0, 10), fs=fs)
    eq_sos = eq_sos / eq_sos[:, 3:4]  # a0 = 1

    def build(output_filter):
        return pyFDN.dss_to_flamo(
            A,
            B,
            C,
            D,
            delays,
            fs,
            nfft=2**14,
            output_filter=output_filter,
            dtype=torch.float64,
        )

    ir_len = 4096
    ir_plain = pyFDN.flamo_time_response(build(None)).squeeze()[:ir_len]
    ir_eq = pyFDN.flamo_time_response(build(eq_sos[:, :, np.newaxis])).squeeze()[
        :ir_len
    ]
    np.testing.assert_allclose(ir_eq, sosfilt(eq_sos, ir_plain), atol=1e-8)
