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


def test_recursion_find_min_loop_delay_no_delay() -> None:
    """Test if Recursion connector creates the recursive graph
    correctly also in case no delay line is involved"""

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

    forward = td.Parallel([A1_td, A2_td], sum_output=True)
    feedback = td.Series([A3_td, A4_td])
    recursion_conn = object.__new__(td.Recursion)

    # Test
    assert recursion_conn._find_min_delay(forward) == 0
    assert recursion_conn._find_min_delay(feedback) == 0


def test_recursion_find_min_loop_delay_forward_and_feedback_delays() -> None:
    """Test if Recursion connector correctly finds the minimum
    delay line in the architecture when there are delays
    in both the forward and the feedback paths"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    delays_forward = [200, 250, 220, 190]
    delays_feedback = [210, 175, 230, 245]

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    A_td = td.Gain(fdnbuild.A)
    delay1_td = td.Delay(delays_forward)
    delay2_td = td.Delay(delays_feedback)

    forward = delay1_td
    feedback = td.Series([delay2_td, A_td])
    recursion_conn = object.__new__(td.Recursion)

    # Test
    assert recursion_conn._find_min_delay(forward) == np.min(delays_forward)
    assert recursion_conn._find_min_delay(feedback) == np.min(delays_feedback)


def test_recursion_find_min_loop_delay_multiple_forward_delays() -> None:
    """Test if Recursion connector correctly finds the minimum
    delay line in the architecture when there are multiple delay-line
    blocks in the forward path"""

    # parameters
    delays_forward_1 = [200, 250, 220, 190]
    delays_forward_2 = [210, 175, 230, 245]

    # td engine
    delay1_td = td.Delay(delays_forward_1)
    delay2_td = td.Delay(delays_forward_2)

    forward = td.Series([delay1_td, delay2_td])
    recursion_conn = object.__new__(td.Recursion)

    # Test
    assert recursion_conn._find_min_delay(forward) == np.min(delays_forward_1) + np.min(
        delays_forward_2
    )


def test_recursion_find_min_loop_delay_nested_operators():
    """Test if Recursion connector correctly finds the minimum
    delay line in the architecture when the delay-line blocks
    are nested in Parallel and Series connectors"""

    # parameters
    n_in, n_out, N = 2, 3, 4
    rng = np.random.default_rng(seed=5)
    fs = 48_000.0
    delays_forward_1 = [200, 250, 220, 190]
    delays_forward_2 = [210, 175, 230, 245]
    delays_feedback = [310, 240, 205, 215]

    # reference
    fdnbuild = pyFDN.fdn_build_gallery(
        N=N, fs=fs, num_inputs=n_in, num_outputs=n_out, rng=rng
    )

    # td engine
    A1_td = td.Gain(fdnbuild.A)
    A2_td = td.Gain(fdnbuild.A)
    delay1_td = td.Delay(delays_forward_1)
    delay2_td = td.Delay(delays_forward_2)
    delay3_td = td.Delay(delays_feedback)

    forward = td.Series(
        [
            A1_td,
            td.Parallel(
                [
                    delay1_td,
                    td.Series(
                        [
                            A2_td,
                            delay2_td,
                        ]
                    ),
                ]
            ),
        ]
    )
    feedback = td.Series(
        [
            delay3_td,
        ]
    )
    recursion_conn = object.__new__(td.Recursion)

    # Test
    assert recursion_conn._find_min_delay(forward) == np.min(
        [delays_forward_1, delays_forward_2]
    )
    assert recursion_conn._find_min_delay(feedback) == np.min(delays_feedback)
