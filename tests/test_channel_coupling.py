"""Checks of directional coupling and separation of its two covariance errors."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

source = Path(__file__).resolve().parents[1]/'examples/coupled_rooms/compare_channel_coupling.py'
sys.path.insert(0,str(source.parent))
spec = importlib.util.spec_from_file_location('channel_coupling',source)
comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)


@pytest.mark.parametrize('profile',comparison.PROFILES)
def test_channel_profiles_preserve_rate_and_change_singular_values(profile):
    model = comparison.channel_model(101,profile,501)
    probabilities = model['channel_probabilities']; k=model['lines']
    assert probabilities.mean()==pytest.approx(model['phi'][0,1]**2,abs=1e-15)
    block = model['feedback'][:k,k:]
    np.testing.assert_allclose(np.sort(np.linalg.svd(block,compute_uv=False)**2),np.sort(probabilities),atol=1e-14)
    if profile!='uniform': assert np.std(probabilities)>0


def test_infeasible_concentration_is_rejected():
    with pytest.raises(ValueError,match='capacity'): comparison.channel_probabilities(.2,16,'rank1')


@pytest.mark.parametrize('profile',['uniform','rank1'])
@pytest.mark.parametrize('lossy',[False,True])
def test_directional_and_cross_terms_separately_reconstruct_energy(monkeypatch,profile,lossy):
    paper = comparison.paper
    model = comparison.channel_model(101,profile,501)
    monkeypatch.setattr(paper,'FS',100)
    model['delays'] = np.tile(np.arange(2,18),2)
    model['totals'] = model['delays'].reshape(2,16).sum(axis=1)
    response = paper.render(model,source=0,lossy=lossy,varying=False,duration=1.,include_outputs=True)
    terms = comparison.directional_closure_terms(model,response,lossy)
    energy = response['energy']
    predicted = energy[:-1]@comparison.validation.closed_generator(model,lossy).T/100
    predicted += sum(terms[name] for name in ['port','anisotropy','cross','attenuation'])
    np.testing.assert_allclose(np.diff(energy,axis=0),predicted,atol=2e-13)
    np.testing.assert_allclose(terms['anisotropy'].sum(axis=1),0,atol=2e-13)
    np.testing.assert_allclose(terms['cross'].sum(axis=1),0,atol=2e-13)
    if profile=='uniform': np.testing.assert_allclose(terms['anisotropy'],0,atol=2e-13)
    else: assert np.max(np.abs(terms['anisotropy']))>1e-5
    if not lossy: np.testing.assert_allclose(energy.sum(axis=1),1,atol=2e-13)


def test_design_has_paired_uniform_basis_controls_and_both_sources():
    specs = comparison.case_specs()
    assert len(specs)==len({comparison.case_key(s) for s in specs})==240
    assert sum(s['matrix_seed']==101 for s in specs)==48
    assert {s['source'] for s in specs}=={0,1}
    assert sum(s['profile']=='uniform' for s in specs)==80
