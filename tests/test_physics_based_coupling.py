"""Regression checks for the paper's physical coupling and energy observables."""
import numpy as np
import pytest
from scipy.linalg import expm

import pyFDN
from pyFDN.td import TimeVaryingMatrix
from pyFDN.auxiliary.physics_based_coupling import (
    create_lossless_coupling_matrix,
    create_state_transition_matrix,
    get_coupling_angles,
    get_room_time_varying_matrix,
    gfdn_energy_ledger,
    match_delay_totals_to_volumes,
    run_gfdn,
    trajectory,
)


def test_two_room_rule_matches_physical_rates():
    volumes = np.array([160., 80.])
    beta = create_lossless_coupling_matrix(np.array([[0., 2.], [2., 0.]]), volumes)
    totals = np.array([32000, 16000])
    fs, lines = 48000, 16
    theta = get_coupling_angles(beta, totals / (lines * fs))[0, 1]
    h = fs * lines * volumes[0] / totals[0]
    assert h * np.sin(theta)**2 == pytest.approx(343 * 2 / 4)


def test_coupling_does_not_silently_cap_infeasible_fractions():
    with pytest.raises(ValueError, match="exceeds one"):
        get_coupling_angles(np.array([[0., 2.], [2., 0.]]), [1., 1.])
    with pytest.raises(ValueError, match="nonnegative"):
        get_coupling_angles(np.array([[0., -1.], [-1., 0.]]), [1., 1.])
    with pytest.raises(ValueError, match="positive finite"):
        get_coupling_angles(np.zeros((2, 2)), [0., 1.])


def test_allocated_storage_matches_volume_and_remains_positive():
    base = np.array([9, 13, 5, 7, 11, 17])
    delays = match_delay_totals_to_volumes(base, 2, [160, 80, 120])
    totals = delays.reshape(3, 2).sum(axis=1)
    np.testing.assert_allclose(totals / totals.sum(), np.array([4, 2, 3]) / 9)
    assert np.all(delays >= 1)
    assert abs(delays.sum() - base.sum()) <= 4.5


def test_energy_and_density_generators_are_consistent():
    volumes = np.array([160., 80., 120.])
    apertures = np.array([[0., 2., 0.], [2., 0., 1.5], [0., 1.5, 0.]])
    beta = create_lossless_coupling_matrix(apertures, volumes)
    density_generator = beta.copy()
    np.fill_diagonal(density_generator, -beta.sum(axis=1))
    energy_generator = create_state_transition_matrix(beta)
    np.testing.assert_allclose(energy_generator, np.diag(volumes) @ density_generator @ np.diag(1 / volumes))
    np.testing.assert_allclose(energy_generator.sum(axis=0), 0, atol=1e-14)
    _, energy = trajectory(energy_generator, np.array([1., 0., 0.]), np.eye(3), 100, 1000)
    np.testing.assert_allclose(energy.sum(axis=1), 1, atol=1e-13)


def test_source_wrapper_selects_inputs_and_restarts_modulation():
    phi = expm(np.array([[0., .2], [-.2, 0.]]))
    feedback = np.kron(phi, np.eye(2))
    inputs = np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]])
    delays = np.array([2, 3, 4, 5])
    np.random.seed(3)
    hook = get_room_time_varying_matrix(phi, 2, fs=100)
    responses = run_gfdn(feedback, inputs, np.eye(4), delays, 100, tv_matrix=hook)
    selected = run_gfdn(feedback, inputs, np.eye(4), delays, 100, src=[1, 0, 1], tv_matrix=hook)
    assert responses.shape == (2, 100, 4)
    assert not np.array_equal(responses[0], responses[1])
    np.testing.assert_allclose(selected, responses[[1, 0, 1]])
    scalar = run_gfdn(feedback, inputs, np.eye(4), delays, 100, src=1, tv_matrix=hook)
    np.testing.assert_allclose(scalar, responses[1:2])


def test_single_source_ledger_preserves_energy():
    phi = expm(np.array([[0., .2], [-.2, 0.]]))
    feedback = np.kron(phi, np.eye(2))
    inputs = np.array([[1.], [0.], [0.], [0.]])
    delays = np.array([2, 3, 4, 5])
    responses = run_gfdn(feedback, inputs, np.eye(4), delays, 100)
    energy, stop = gfdn_energy_ledger(responses, 2, 2, delays)
    assert energy.shape == (1, stop, 2)
    np.testing.assert_allclose(energy.sum(axis=2), 1, atol=1e-13)
    with pytest.raises(ValueError, match="longest delay"):
        gfdn_energy_ledger(responses[:, :2], 2, 2, delays)


def test_modulation_changes_only_internal_mixers():
    phi = expm(np.array([[0., .2], [-.2, 0.]]))
    f = np.kron(phi, np.eye(2))
    np.random.seed(4)
    hook = get_room_time_varying_matrix(phi, 2, fs=100)
    np.random.seed(4)
    rotation = TimeVaryingMatrix(4, 1, 3, 100, .3)
    values = np.random.default_rng(5).normal(size=(30, 4))
    observed = hook.process_signal(values @ f.T)
    expected = rotation.process_signal(values) @ f.T
    np.testing.assert_allclose(observed, expected, atol=1e-13)


def test_initial_delay_contents_play_out_and_reset():
    delays = np.array([7, 11, 13])
    rng = np.random.default_rng(0)
    lines = [rng.standard_normal(m) for m in delays]
    graph = pyFDN.dss_to_td(delays, np.zeros((3, 3)), np.ones((3, 1)),
                           np.eye(3), np.zeros((3, 1)), block_size=5)
    graph.process_signal(np.zeros(3))
    state = graph.ops[0].ops[1].state
    state.set_lines(lines)
    output = graph.process_signal(np.zeros(20))
    for i, length in enumerate(delays):
        np.testing.assert_array_equal(output[:length, i], lines[i])
        np.testing.assert_array_equal(output[length:, i], 0)
    graph.reset()
    np.testing.assert_array_equal(graph.process_signal(np.zeros(20)), 0)
    state.fill_noise(.5, rng)
    assert np.all(graph.process_signal(np.zeros(7)) != 0)
    with pytest.raises(ValueError, match="lines"):
        state.set_lines([line[:-1] for line in lines])
