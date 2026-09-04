"""Tests for the Kronecker feedback matrix (Coppola 2026) and its td operators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import hadamard

import pyFDN
from pyFDN import td
from pyFDN.generate.kronecker_matrix import num_kernels

KERNEL_TYPES = ["rotation", "reflection"]


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5])  # type: ignore[misc]
@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)  # type: ignore[misc]
def test_matrix_is_orthogonal_for_arbitrary_angles(M: int, kernel_type: str) -> None:
    rng = np.random.default_rng(M)
    A = pyFDN.kronecker_matrix(rng.uniform(-np.pi, np.pi, M), kernel_type)

    assert A.shape == (2**M, 2**M)
    assert np.allclose(A @ A.T, np.eye(2**M), atol=1e-12)


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5])  # type: ignore[misc]
@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)  # type: ignore[misc]
def test_fast_transform_matches_explicit_matrix(M: int, kernel_type: str) -> None:
    rng = np.random.default_rng(100 + M)
    angles = rng.uniform(-np.pi, np.pi, M)
    x = rng.standard_normal((16, 2**M))

    fast = pyFDN.kronecker_transform(x, angles, kernel_type)

    assert np.allclose(fast, x @ pyFDN.kronecker_matrix(angles, kernel_type).T)


def test_fast_transform_accepts_mixed_kernel_types() -> None:
    rng = np.random.default_rng(7)
    angles = rng.uniform(-np.pi, np.pi, 4)
    kernel_type = ["rotation", "reflection", "reflection", "rotation"]
    x = rng.standard_normal((5, 16))

    fast = pyFDN.kronecker_transform(x, angles, kernel_type)

    assert np.allclose(fast, x @ pyFDN.kronecker_matrix(angles, kernel_type).T)


def test_fast_transform_accepts_a_single_vector() -> None:
    rng = np.random.default_rng(8)
    angles = rng.uniform(-np.pi, np.pi, 3)
    x = rng.standard_normal(8)

    out = pyFDN.kronecker_transform(x, angles)

    assert out.shape == (8,)
    assert np.allclose(out, pyFDN.kronecker_matrix(angles) @ x)


def test_fast_transform_leaves_the_input_untouched() -> None:
    x = np.ones((4, 8))
    original = x.copy()

    pyFDN.kronecker_transform(x, np.full(3, 0.3))

    assert np.array_equal(x, original)


def test_per_sample_angles_match_a_per_sample_matrix() -> None:
    rng = np.random.default_rng(9)
    angles = rng.uniform(-np.pi, np.pi, (6, 3))
    x = rng.standard_normal((6, 8))

    fast = pyFDN.kronecker_transform(x, angles)

    expected = np.stack(
        [pyFDN.kronecker_matrix(angles[n]) @ x[n] for n in range(x.shape[0])]
    )
    assert np.allclose(fast, expected)


@pytest.mark.parametrize("M", [2, 3, 4, 5])  # type: ignore[misc]
def test_all_reflection_kernels_at_quarter_pi_give_hadamard(M: int) -> None:
    """Section 5.1: Ref(pi/4) is H_2, so the Kronecker product is H_{2^M}."""
    A = pyFDN.kronecker_matrix(np.full(M, np.pi / 4), "reflection")

    assert np.allclose(A, hadamard(2**M) / np.sqrt(2**M), atol=1e-12)


def test_rotation_kernels_at_quarter_pi_have_equal_magnitudes() -> None:
    """Rot(pi/4) mixes with uniform energy, as Hadamard does but with signs of its own."""
    N = 16
    A = pyFDN.kronecker_matrix(pyFDN.kronecker_angles(N))

    assert np.allclose(np.abs(A), 1 / np.sqrt(N))


def test_outermost_angle_zero_decouples_the_contiguous_halves() -> None:
    """Section 5.2: theta_M = 0 makes the matrix block diagonal (two stereo halves)."""
    N = 16
    A = pyFDN.kronecker_matrix(pyFDN.kronecker_angles(N, theta4=0.0))
    half = N // 2

    assert np.allclose(A[:half, half:], 0.0)
    assert np.allclose(A[half:, :half], 0.0)
    # ... and the two independent halves are themselves orthogonal.
    assert np.allclose(A[:half, :half] @ A[:half, :half].T, np.eye(half))


def test_outermost_angle_half_pi_is_purely_cross_coupled() -> None:
    """theta_M = pi/2 leaves only the off-diagonal blocks: the halves swap."""
    N = 16
    A = pyFDN.kronecker_matrix(pyFDN.kronecker_angles(N, theta4=np.pi / 2))
    half = N // 2

    assert np.allclose(A[:half, :half], 0.0)
    assert np.allclose(A[half:, half:], 0.0)


def test_innermost_angle_zero_decouples_even_and_odd_lines() -> None:
    """Section 5.3: theta_1 = 0 splits the network into even/odd subnetworks."""
    N = 16
    A = pyFDN.kronecker_matrix(pyFDN.kronecker_angles(N, theta1=0.0))
    even, odd = np.arange(0, N, 2), np.arange(1, N, 2)

    assert np.allclose(A[np.ix_(even, odd)], 0.0)
    assert np.allclose(A[np.ix_(odd, even)], 0.0)


def test_figure_5_matrix_combines_even_odd_and_stereo_partitioning() -> None:
    """Figure 5: N = 8 rotation kernels at theta = (0, pi/4, pi/8)."""
    A = pyFDN.kronecker_matrix([0.0, np.pi / 4, np.pi / 8])
    even, odd = np.arange(0, 8, 2), np.arange(1, 8, 2)

    # theta_1 = 0: no even/odd cross-talk.
    assert np.allclose(A[np.ix_(even, odd)], 0.0)
    # theta_3 = pi/8: partial, non-zero coupling between the two halves,
    # of magnitude sin(pi/8) / sqrt(2) against sqrt(2) / 2 * cos(pi/8) within.
    assert np.isclose(np.abs(A[:4, 4:]).max(), np.sin(np.pi / 8) / np.sqrt(2))
    assert np.isclose(np.abs(A[:4, :4]).max(), np.cos(np.pi / 8) / np.sqrt(2))
    # Figure 5 shows exactly the two magnitudes 0.27 and 0.65.
    magnitudes = np.unique(np.round(np.abs(A[np.abs(A) > 1e-12]), 2))
    assert np.allclose(magnitudes, [0.27, 0.65])


def test_kronecker_angles_defaults_and_overrides() -> None:
    angles = pyFDN.kronecker_angles(32, theta1=0.0, theta5=np.pi / 16)

    assert angles.shape == (5,)
    assert angles[0] == 0.0
    assert angles[4] == np.pi / 16
    assert np.allclose(angles[1:4], np.pi / 4)


def test_kronecker_angles_rejects_out_of_range_kernels() -> None:
    with pytest.raises(ValueError, match="out of range"):
        pyFDN.kronecker_angles(8, theta9=0.0)
    with pytest.raises(TypeError, match="Unexpected keyword"):
        pyFDN.kronecker_angles(8, gamma1=0.0)  # type: ignore[call-arg]


@pytest.mark.parametrize("N", [3, 1, 12])  # type: ignore[misc]
def test_num_kernels_requires_a_power_of_two(N: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        num_kernels(N)


def test_transform_rejects_a_channel_count_that_does_not_match_the_angles() -> None:
    with pytest.raises(ValueError, match="channels"):
        pyFDN.kronecker_transform(np.zeros((4, 8)), np.zeros(2))


def test_transform_rejects_an_angle_row_count_that_does_not_match_the_block() -> None:
    with pytest.raises(ValueError, match="rows"):
        pyFDN.kronecker_transform(np.zeros((4, 8)), np.zeros((3, 3)))


def test_unknown_kernel_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown kernel_type"):
        pyFDN.kronecker_matrix([0.1, 0.2], "shear")


def test_kernels_are_the_paper_definitions() -> None:
    assert np.allclose(pyFDN.rotation_kernel(0.0), np.eye(2))
    assert np.allclose(pyFDN.reflection_kernel(0.0), np.diag([1.0, -1.0]))
    assert np.allclose(
        pyFDN.reflection_kernel(np.pi / 4), np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    )


# --- gallery integration ----------------------------------------------------


def test_gallery_lists_and_builds_the_kronecker_type() -> None:
    assert "kronecker" in pyFDN.fdn_matrix_gallery()

    A = pyFDN.fdn_matrix_gallery(8, "kronecker")
    assert isinstance(A, np.ndarray)  # the gallery returns a name list when N is None

    assert A.shape == (8, 8)
    assert np.allclose(A @ A.T, np.eye(8))
    assert np.allclose(A, pyFDN.kronecker_matrix(pyFDN.kronecker_angles(8)))


def test_gallery_forwards_angles_and_kernel_type() -> None:
    angles = [0.0, np.pi / 4, np.pi / 8]

    A = pyFDN.fdn_matrix_gallery(
        8, "kronecker", angles=angles, kernel_type="reflection"
    )

    assert np.allclose(A, pyFDN.kronecker_matrix(angles, "reflection"))


def test_gallery_kronecker_requires_a_power_of_two_size() -> None:
    with pytest.raises(ValueError, match="power of two"):
        pyFDN.fdn_matrix_gallery(6, "kronecker")


# --- td operators -----------------------------------------------------------


def test_kronecker_matrix_operator_matches_a_static_gain() -> None:
    rng = np.random.default_rng(11)
    angles = rng.uniform(-np.pi, np.pi, 4)
    x = rng.standard_normal((32, 16))

    operator = td.KroneckerMatrix(angles)

    assert operator.in_channels == operator.out_channels == 16
    assert np.allclose(
        operator.process(x), td.Gain(pyFDN.kronecker_matrix(angles)).process(x)
    )


def test_kronecker_matrix_operator_is_block_invariant() -> None:
    rng = np.random.default_rng(12)
    angles = rng.uniform(-np.pi, np.pi, 3)
    x = rng.standard_normal((30, 8))
    operator = td.KroneckerMatrix(angles)

    whole = operator.process(x)
    streamed = np.vstack([operator.filter(x[i : i + 7]) for i in range(0, 30, 7)])

    assert np.allclose(whole, streamed)


def test_kronecker_matrix_operator_rejects_a_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="expects 8 input channels"):
        td.KroneckerMatrix(np.zeros(3)).process(np.zeros((4, 4)))


def test_time_varying_operator_is_orthogonal_at_every_sample() -> None:
    fs = 48000.0
    operator = td.TimeVaryingKroneckerMatrix(
        pyFDN.kronecker_angles(16, theta4=0.0), fs, rate=2.0, depth=np.pi
    )

    for angles in operator.angles_at([0, 137, 6000, 24000]):
        A = pyFDN.kronecker_matrix(angles)
        assert np.allclose(A @ A.T, np.eye(16), atol=1e-12)


def test_time_varying_operator_follows_the_paper_modulation_law() -> None:
    """Section 5.4: theta_{M-1}(t) = pi/4 + depth * sin(2 pi rate t)."""
    fs, rate, depth = 48000.0, 0.2, 0.2 * np.pi
    operator = td.TimeVaryingKroneckerMatrix(
        pyFDN.kronecker_angles(16, theta4=0.0),
        fs,
        rate=[0.0, 0.0, rate, 0.0],
        depth=[0.0, 0.0, depth, 0.0],
    )

    n = np.arange(0, int(fs), 997)
    angles = operator.angles_at(n)

    assert np.allclose(
        angles[:, 2], np.pi / 4 + depth * np.sin(2 * np.pi * rate * n / fs)
    )
    # Unmodulated kernels stay put, theta_M included.
    assert np.allclose(angles[:, [0, 1]], np.pi / 4)
    assert np.allclose(angles[:, 3], 0.0)


def test_time_varying_operator_advances_and_rewinds_its_clock() -> None:
    fs = 48000.0
    operator = td.TimeVaryingKroneckerMatrix(
        pyFDN.kronecker_angles(8), fs, rate=1.0, depth=0.5
    )
    x = np.random.default_rng(13).standard_normal((64, 8))

    first = operator.process(x)
    assert operator.sample_index == 64

    second = operator.process(x)
    assert not np.allclose(first, second)

    operator.reset()
    assert np.allclose(operator.process(x), first)


def test_zero_depth_time_varying_operator_equals_the_static_matrix() -> None:
    angles = pyFDN.kronecker_angles(8, theta1=0.3)
    x = np.random.default_rng(14).standard_normal((20, 8))

    modulated = td.TimeVaryingKroneckerMatrix(angles, 48000.0, rate=3.0, depth=0.0)

    assert np.allclose(modulated.process(x), td.KroneckerMatrix(angles).process(x))


def test_triangle_waveform_stays_within_the_requested_depth() -> None:
    operator = td.TimeVaryingKroneckerMatrix(
        pyFDN.kronecker_angles(8), 48000.0, rate=1.0, depth=np.pi, waveform="triangle"
    )

    angles = operator.angles_at(np.arange(48000))
    excursion = angles - np.pi / 4

    assert np.isclose(excursion.max(), np.pi, atol=1e-3)
    assert np.isclose(excursion.min(), -np.pi, atol=1e-3)


def test_unknown_waveform_is_rejected() -> None:
    with pytest.raises(ValueError, match="waveform"):
        td.TimeVaryingKroneckerMatrix(np.zeros(3), 48000.0, waveform="saw")


def test_modulation_parameter_length_is_checked() -> None:
    with pytest.raises(ValueError, match="depth has 2 entries"):
        td.TimeVaryingKroneckerMatrix(np.zeros(3), 48000.0, depth=[0.1, 0.2])


def test_kronecker_feedback_matrix_renders_a_lossless_fdn() -> None:
    """A Kronecker matrix in place of A keeps the undamped FDN energy-preserving."""
    N = 8
    delays = pyFDN.sample_delay_lengths(N, (60, 300), coprime=True, rng=3)
    A = pyFDN.kronecker_matrix(pyFDN.kronecker_angles(N, theta2=0.7))
    B = np.ones((N, 1)) / np.sqrt(N)
    C = np.ones((1, N)) / np.sqrt(N)
    D = np.zeros((1, 1))

    impulse = np.zeros(4000)
    impulse[0] = 1.0
    ir = pyFDN.process_fdn(impulse, delays, A, B, C, D)

    # Lossless: the tail neither grows nor dies away.
    assert np.isfinite(ir).all()
    late = np.asarray(ir).reshape(-1)[2000:]
    assert np.abs(late).max() > 1e-3


def test_time_varying_kronecker_matrix_drives_process_fdn() -> None:
    """With A = I the operator on ``post_matrix`` *is* the feedback matrix."""
    N = 16
    delays = pyFDN.sample_delay_lengths(N, (60, 300), coprime=True, rng=4)
    fs = 48000.0
    modulated = td.TimeVaryingKroneckerMatrix(
        pyFDN.kronecker_angles(N, theta4=0.0),
        fs,
        rate=[0.0, 0.0, 2.0, 0.0],
        depth=[0.0, 0.0, np.pi, 0.0],
    )
    static = td.KroneckerMatrix(pyFDN.kronecker_angles(N, theta4=0.0))

    impulse = np.zeros(8000)
    impulse[0] = 1.0
    common = (
        delays,
        np.eye(N),
        np.ones((N, 1)) / np.sqrt(N),
        np.ones((1, N)) / np.sqrt(N),
        np.zeros((1, 1)),
    )
    moving = pyFDN.process_fdn(impulse, *common, post_matrix=modulated)
    fixed = pyFDN.process_fdn(impulse, *common, post_matrix=static)

    assert np.isfinite(moving).all()
    assert not np.allclose(moving, fixed)
