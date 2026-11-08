"""Receiving-side mixing preserves the physical block coupling exactly."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

source = Path(__file__).resolve().parents[1]/'examples/coupled_rooms/compare_static_coupling.py'
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location('static_coupling', source)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.mark.parametrize('topology', ['chain', 'complete'])
def test_receiving_rotations_preserve_each_source_vector_contribution(topology):
    original = comparison.static_model(topology, 101)
    changed = comparison.static_model(topology, 101, 501)
    alternative = comparison.static_model(topology, 101, 502)
    np.testing.assert_array_equal(original['delays'], changed['delays'])
    np.testing.assert_array_equal(original['phi'], changed['phi'])
    assert not np.allclose(changed['feedback'], alternative['feedback'])
    k = changed['lines']
    vector = np.random.default_rng(42).normal(size=k)
    for i in range(3):
        for j in range(3):
            block = changed['feedback'][i*k:(i+1)*k, j*k:(j+1)*k]
            expected = changed['phi'][i,j]**2*np.dot(vector, vector)
            assert np.dot(block@vector, block@vector) == pytest.approx(expected, abs=1e-13)


def test_case_design_is_paired_and_unique():
    specs = comparison.case_specs()
    assert len(specs) == len({comparison.case_key(s) for s in specs}) == 240
    assert {s['receiver_seed'] for s in specs if s['variant']=='receiving'} == {501,502}
    assert sum(s['matrix_seed']==101 for s in specs) == 48


@pytest.mark.parametrize('lossy', [False, True])
def test_broader_feedback_retains_exact_energy_decomposition(monkeypatch, lossy):
    paper = comparison.paper
    model = comparison.static_model('complete', 101, 501)
    monkeypatch.setattr(paper, 'FS', 100)
    model['delays'] = np.tile(np.arange(2,18), 3)
    model['totals'] = model['delays'].reshape(3,16).sum(axis=1)
    response = paper.render(model, source=1, lossy=lossy, varying=False,
                            duration=1., include_outputs=True)
    terms = comparison.validation.closure_terms(model, response, lossy, False)
    energy = response['energy']
    predicted = energy[:-1]@comparison.validation.closed_generator(model, lossy).T/100
    predicted += terms['port']+terms['cross']+terms['attenuation']
    np.testing.assert_allclose(np.diff(energy, axis=0), predicted, atol=2e-13)
    if not lossy: np.testing.assert_allclose(energy.sum(axis=1), 1., atol=1e-13)
