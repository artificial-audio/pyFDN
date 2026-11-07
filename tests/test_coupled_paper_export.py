"""Scientific regression checks for the reproducible paper experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyFDN.td import RecursionState

source = Path(__file__).resolve().parents[1] / "examples/coupled_rooms/export_paper_figures.py"
spec = importlib.util.spec_from_file_location("coupled_paper_export", source)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_noise_start_has_unit_room_energy_and_isotropic_second_moment():
    delays = np.array([2, 3, 4, 5])
    state = RecursionState(delays, 1)
    draws = []
    for seed in range(2000):
        export.initialise_room_noise(state, delays, 2, 0, seed)
        values = np.r_[state.buffer[0, :2], state.buffer[1, :3]]
        assert values @ values == pytest.approx(1)
        np.testing.assert_array_equal(state.buffer[2:], 0)
        draws.append(values)
    draws = np.asarray(draws)
    np.testing.assert_allclose(draws.mean(axis=0), 0, atol=.035)
    np.testing.assert_allclose(draws.T @ draws / len(draws), np.eye(5)/5, atol=.02)


def test_zero_tail_cannot_dilute_lossy_error_average():
    time = np.arange(5*export.FS)/export.FS
    reference = np.exp(-np.log(1e6)*time)[:,None]
    energy = reference.copy()
    energy[time <= 1] += .01
    errors = export.energy_errors(energy, reference)
    assert errors["evaluation_stop_s"] == pytest.approx(1, abs=1/export.FS)
    assert errors["full_mae_pp"] == pytest.approx(1)
