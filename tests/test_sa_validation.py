"""Checks of the validation's state alignment and error decomposition."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

source = Path(__file__).resolve().parents[1]/'examples/coupled_rooms/validate_sa_assumptions.py'
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location('sa_validation', source)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
paper = validation.paper


def test_forced_response_matches_direct_recursion_with_complex_modes():
    generator = np.array([[-1.,0,1],[1,-1,0],[0,1,-1]])
    transition = np.eye(3)+.1*generator
    forcing = np.random.default_rng(9).normal(size=(40,3))
    expected = np.zeros((41,3))
    for n in range(40): expected[n+1] = transition@expected[n]+forcing[n]
    np.testing.assert_allclose(validation.forced_response(transition, forcing), expected, atol=1e-13)


def test_continuous_generator_keeps_complex_conjugate_modes(monkeypatch):
    monkeypatch.setattr(paper, 'FS', 100)
    generator = np.array([[-1.,0,1],[1,-1,0],[0,1,-1]])
    initial = np.array([1.,0,0])
    expected = expm(np.arange(101)[:,None,None]/100*generator)@initial
    np.testing.assert_allclose(validation.continuous_energy(generator, initial, 101), expected, atol=1e-13)


@pytest.mark.parametrize('lossy', [False, True])
@pytest.mark.parametrize('varying', [False, True])
def test_exact_energy_decomposition_uses_the_same_preupdate_state(monkeypatch, lossy, varying):
    monkeypatch.setattr(paper, 'FS', 100)
    phi = expm(np.array([[0.,.25],[-.25,0.]]))
    mixer = expm(np.array([[0.,.3],[-.3,0.]]))
    delays = np.array([3,4,5,9])
    np.random.seed(4)
    model = dict(volumes=np.array([1.,2.]), lines=2, delays=delays,
                 totals=np.array([7,14]), phi=phi, gamma=np.diag([.1,.4]),
                 feedback=paper.get_feedback_matrix(phi,block_diag(mixer,mixer),2),
                 hook=paper.get_room_time_varying_matrix(phi,2,amplitude=.2,fs=100))
    model['generator'] = validation.closed_generator(model, False)
    response = paper.render(model, source=0, lossy=lossy, varying=varying,
                            duration=1., include_outputs=True)
    terms = validation.closure_terms(model, response, lossy, varying)
    energy = response['energy']
    predicted = energy[:-1]@validation.closed_generator(model, lossy).T/100
    predicted += terms['port']+terms['cross']+terms['attenuation']
    np.testing.assert_allclose(np.diff(energy,axis=0),predicted,atol=1e-13)
    if not lossy:
        np.testing.assert_allclose(energy.sum(axis=1),1,atol=1e-13)
        np.testing.assert_allclose(terms['attenuation'],0,atol=1e-13)


def test_matrix_and_modulation_draws_are_independent():
    first = validation.validation_model('chain', 101, 701, .1)
    second = validation.validation_model('chain', 101, 702, 3.)
    third = validation.validation_model('chain', 102, 701, .1)
    np.testing.assert_array_equal(first['feedback'],second['feedback'])
    np.testing.assert_array_equal(first['delays'],third['delays'])
    assert not np.array_equal(first['feedback'],third['feedback'])


def test_common_error_window_is_selected_by_sa_reference():
    time = np.arange(3*paper.FS)/paper.FS
    sa = np.exp(-np.log(1e6)*time)[:,None]
    alternative = np.exp(-2*np.log(1e6)*time)[:,None]
    estimate = alternative+.01
    metrics = paper.energy_errors(estimate, alternative, window_reference=sa)
    assert metrics['evaluation_stop_s'] == pytest.approx(1,abs=1/paper.FS)
    assert metrics['full_mae_pp'] == pytest.approx(1)


def test_validation_cases_cover_the_declared_design_without_duplicates():
    specs = validation.case_specs()
    assert len(specs)==210
    assert len({validation.case_key(s) for s in specs})==210
    assert {s['source'] for s in specs}=={0,1,2}
