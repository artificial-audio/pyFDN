"""Tests for the time-domain block-based graph engine (``pyFDN.td``)
Connectors only

The graph-building functionalities of individual ``pyFDN.td`` connectors are tested.
This needs no FLAMO install.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyFDN
from pyFDN import td

# ============================================================
# ======================== TEST SERIES =======================


def test_series_reject_empty_list() -> None:
    """Test if Series connector rejects empty input list of operators"""

    # Test
    with pytest.raises(ValueError, match="Series needs at least one operator"):
        td.Series([])


def test_series_reject_io_mismatch() -> None:
    """Test if Series connector rejects list of operators with
    non-matching input/output channel count connections"""
    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    A_td = td.Gain(fdnbuild.A)  # N x N
    B_td = td.Gain(fdnbuild.B)  # N x n_in
    C_td = td.Gain(fdnbuild.C)  # n_out x N
    D_td = td.Gain(fdnbuild.D)  # n_out x n_in

    # Test
    with pytest.raises(
        ValueError,
        match="Operator 1 output-channel count does not match operator 2 input-channel count",
    ):
        td.Series([B_td, A_td, D_td, C_td])


def test_series_flatten_ops() -> None:
    """Test td Series connector flattening"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    A1_td = td.Gain(fdnbuild.A)
    A2_td = td.Gain(fdnbuild.A)
    A3_td = td.Gain(fdnbuild.A)
    A4_td = td.Gain(fdnbuild.A)
    series_nested = td.Series([A1_td, td.Series([td.Series([A2_td, A3_td]), A4_td])])

    # Test
    assert series_nested.ops == [A1_td, A2_td, A3_td, A4_td]


# ============================================================
# ======================= TEST PARALLEL ======================


def test_parallel_reject_empty_list() -> None:
    """Test if Parallel connector rejects empty input list of operators"""

    # Test
    with pytest.raises(ValueError, match="Parallel needs at least one operator"):
        td.Parallel([])


def test_parallel_reject_input_mismatch() -> None:
    """Test if Parallel connector rejects list of operators with
    non-matching input channel counts"""
    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    A_td = td.Gain(fdnbuild.A)  # N x N
    B_td = td.Gain(fdnbuild.B)  # N x n_in

    # Test
    with pytest.raises(
        ValueError, match="Parallel operators must share input-channel count"
    ):
        td.Parallel([A_td, B_td])


def test_parallel_reject_output_mismatch() -> None:
    """Test if Parallel connector rejects list of operators with
    non-matching output channel counts when sum_output==True"""
    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    B_td = td.Gain(fdnbuild.B)  # N x n_in
    D_td = td.Gain(fdnbuild.D)  # n_out x n_in

    # Test
    with pytest.raises(
        ValueError, match="Summed Parallel operators must share out-channel count"
    ):
        td.Parallel([B_td, D_td], sum_output=True)


def test_parallel_flatten_ops() -> None:
    """Test td Parallel connector filtering"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    A1_td = td.Gain(fdnbuild.A)
    A2_td = td.Gain(fdnbuild.A)
    A3_td = td.Gain(fdnbuild.A)
    A4_td = td.Gain(fdnbuild.A)
    parallel_nested = td.Parallel(
        [A1_td, td.Parallel([td.Parallel([A2_td, A3_td]), A4_td])]
    )

    # Test
    assert parallel_nested.ops == [A1_td, A2_td, A3_td, A4_td]


# ============================================================
# ====================== TEST RECURSION ======================


def _loop_paths(N: int, delays: list[int]) -> tuple[td.TimeOperator, td.TimeOperator]:
    """A minimal forward/feedback pair for Recursion construction tests."""
    return td.Delay(delays), td.Gain(np.eye(N))


def test_recursion_requires_block_size() -> None:
    """Test that Recursion refuses to guess the block size:
    it must be given explicitly, because the user has to compensate for it"""

    # parameters
    N = 4
    forward, feedback = _loop_paths(N, [200, 250, 220, 190])

    # Test
    with pytest.raises(TypeError):
        td.Recursion(forward, feedback)  # type: ignore[call-arg]


def test_recursion_rejects_non_positive_block_size() -> None:
    """Test that Recursion rejects a non-positive block size"""

    # parameters
    N = 4
    forward, feedback = _loop_paths(N, [200, 250, 220, 190])

    # Test
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        td.Recursion(forward, feedback, block_size=0)


def test_recursion_rejects_invalid_delay_position() -> None:
    """Test that Recursion rejects an unknown delay_position"""

    # parameters
    N = 4
    forward, feedback = _loop_paths(N, [200, 250, 220, 190])

    # Test
    with pytest.raises(ValueError, match="delay_position value not valid"):
        td.Recursion(forward, feedback, block_size=64, delay_position="sideways")


@pytest.mark.parametrize("delay_position", ["forward", "feedback"])  # type: ignore[misc]
def test_recursion_warns_about_inserted_block_delay(delay_position: str) -> None:
    """Test that Recursion warns that its block processing inserts
    block_size samples of delay that the user has to compensate for"""

    # parameters
    N = 4
    block_size = 64
    forward, feedback = _loop_paths(N, [200, 250, 220, 190])

    # Test
    with pytest.warns(
        UserWarning,
        match=(
            f"Recursion block processing inserts {block_size} samples of delay "
            f"into the loop, on the {delay_position} path"
        ),
    ):
        td.Recursion(
            forward, feedback, block_size=block_size, delay_position=delay_position
        )


def test_recursion_state_buffer_matches_block_size() -> None:
    """Test that Recursion holds one block-long delay line per loop channel,
    on the side named by delay_position"""

    # parameters
    n_in, N = 2, 4
    block_size = 32
    rng = np.random.default_rng(seed=5)

    # td engine
    forward = td.Series([td.Gain(rng.standard_normal((N, n_in))), td.Delay([64] * N)])
    feedback = td.Gain(rng.standard_normal((n_in, N)))

    with pytest.warns(UserWarning, match="inserts 32 samples of delay"):
        forward_delayed = td.Recursion(
            forward, feedback, block_size=block_size, delay_position="forward"
        )
        feedback_delayed = td.Recursion(
            forward, feedback, block_size=block_size, delay_position="feedback"
        )

    # Test: the state sits on the loop output for "forward", on the loop input
    # for "feedback", and every line is exactly one block long.
    assert forward_delayed._state.num_delays == N
    assert feedback_delayed._state.num_delays == n_in
    for recursion in (forward_delayed, feedback_delayed):
        np.testing.assert_array_equal(
            recursion._state.delays,
            np.full(recursion._state.num_delays, block_size),
        )


def test_recursion_reject_io_mismatch() -> None:
    """Test that Recursion rejects forward/feedback paths
    whose channel counts do not close the loop"""

    # parameters
    n_in, N = 2, 4
    rng = np.random.default_rng(seed=5)

    # td engine
    forward = td.Gain(rng.standard_normal((N, n_in)))

    # Test
    with pytest.raises(
        ValueError,
        match="Forward output-channel count does not match feedback input-channel count",
    ):
        td.Recursion(forward, td.Gain(np.eye(n_in)), block_size=64)

    with pytest.raises(
        ValueError,
        match="Feedback output-channel count does not match forward input-channel count",
    ):
        td.Recursion(forward, td.Gain(np.eye(N)), block_size=64)
