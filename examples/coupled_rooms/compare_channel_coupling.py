"""Test unequal two-room coupling angles at the same diffuse transfer rate.

The cosine-sine junction uses mean(sin(theta_k)**2) = Phi_12**2.
Diagnostics separate within-room directional bias from cross-room coherence.
No manuscript assets are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.linalg import block_diag

import validate_sa_assumptions as validation

paper = validation.paper
PROFILES = ['uniform', 'graded', 'rank4', 'rank2', 'rank1']
BASIS_SEEDS = [501, 502]


def channel_probabilities(mean, count, profile):
    """Preserve total diffuse transfer while changing its singular values."""
    if not 0 <= mean <= 1 or count < 1: raise ValueError('Invalid mean or channel count')
    if profile == 'uniform': weights = np.ones(count)
    elif profile == 'graded': weights = np.geomspace(.01, 1., count)
    elif profile in ['rank1', 'rank2', 'rank4']:
        rank = int(profile[4:])
        if rank > count: raise ValueError('Coupling rank exceeds the channel count')
        weights = np.r_[np.ones(rank), np.zeros(count-rank)]
    else: raise ValueError(f'Unknown coupling profile: {profile}')
    result = weights*(count*mean/weights.sum())
    if np.any(result > 1+1e-14): raise ValueError('Requested coupling exceeds this profile capacity')
    return np.clip(result, 0., 1.)


def channel_model(matrix_seed, profile='uniform', basis_seed=None, varying=False):
    model = paper.room_model([160,80], [[0,2],[2,0]], setting=1000, matrix_seed=12343)
    k = model['lines']
    rng = np.random.default_rng(matrix_seed)
    mixers = block_diag(*[paper.pyFDN.random_orthogonal(k, rng) for _ in range(2)])
    probabilities = channel_probabilities(model['phi'][0,1]**2, k, profile)
    cosine, sine = np.diag(np.sqrt(1-probabilities)), np.diag(np.sqrt(probabilities))
    junction = np.block([[cosine,sine],[-sine,cosine]])
    if basis_seed is not None:
        rng = np.random.default_rng(basis_seed)
        receiving = block_diag(*[paper.pyFDN.random_orthogonal(k, rng) for _ in range(2)])
        junction = receiving@junction
    model['feedback'] = junction@mixers
    model['channel_probabilities'] = probabilities
    np.testing.assert_allclose(model['feedback'].T@model['feedback'], np.eye(2*k), atol=2e-14)
    for i in range(2):
        for j in range(2):
            block = model['feedback'][i*k:(i+1)*k,j*k:(j+1)*k]
            np.testing.assert_allclose(np.sum(block**2)/k, model['phi'][i,j]**2, atol=2e-14)
    if varying and profile != 'uniform': raise ValueError('Modulation control uses uniform coupling')
    saved = np.random.get_state()
    try:
        np.random.seed(701)
        model['hook'] = paper.get_room_time_varying_matrix(model['phi'], k, amplitude=3. if varying else 0., fs=paper.FS)
    finally: np.random.set_state(saved)
    return model


def directional_closure_terms(model, response, lossy, varying=False):
    """Split true cross-room products from single-room directional imbalance."""
    if varying:
        terms = validation.closure_terms(model, response, lossy, True)
        terms['anisotropy'] = np.zeros_like(terms['cross'])
        return terms
    energy = response['energy']; outputs = response['outputs'][:len(energy)-1]
    k = model['lines']; rooms = len(model['volumes'])
    gains = (np.diag(paper.get_decay_matrix(model['gamma'], model['delays'].reshape(rooms,k), paper.FS))
             if lossy else np.ones(rooms*k))
    attenuated = outputs*gains
    written = attenuated@model['feedback'].T
    outgoing = (outputs**2).reshape(-1,rooms,k).sum(axis=2)
    after_loss = (attenuated**2).reshape(-1,rooms,k).sum(axis=2)
    incoming = (written**2).reshape(-1,rooms,k).sum(axis=2)
    separate = np.zeros_like(incoming)
    for i in range(rooms):
        for j in range(rooms):
            block = model['feedback'][i*k:(i+1)*k,j*k:(j+1)*k]
            contribution = attenuated[:,j*k:(j+1)*k]@block.T
            separate[:,i] += (contribution**2).sum(axis=1)
    squared = model['phi']**2
    port = (outgoing-energy[:-1]*k/model['totals'])@(squared-np.eye(rooms)).T
    anisotropy = separate-after_loss@squared.T
    cross = incoming-separate
    attenuation = (after_loss-outgoing)@squared.T
    if lossy: attenuation += energy[:-1]@model['gamma'].T/paper.FS
    balance = float(np.max(np.abs(np.diff(energy,axis=0)-(incoming-outgoing))))
    assert balance < 1e-9, balance
    return dict(port=port,anisotropy=anisotropy,cross=cross,attenuation=attenuation,
                outgoing=outgoing,balance_error=balance)


def case_specs():
    choices = [('fixed','uniform',None), ('modulated','uniform',None)]
    choices += [('directional',profile,basis) for profile in PROFILES for basis in BASIS_SEEDS]
    return [dict(matrix_seed=seed,source=source,lossy=lossy,variant=variant,profile=profile,basis_seed=basis)
            for seed in validation.MATRIX_SEEDS for source in range(2) for lossy in [False,True]
            for variant,profile,basis in choices]


def case_key(spec):
    return (f"m{spec['matrix_seed']}_s{spec['source']+1}_{'lossy' if spec['lossy'] else 'lossless'}_"
            f"{spec['variant']}_{spec['profile']}_b{spec['basis_seed']}")


def run_case(spec, noise_seeds=validation.NOISE_SEEDS):
    varying = spec['variant']=='modulated'
    model = channel_model(spec['matrix_seed'],spec['profile'],spec['basis_seed'],varying)
    diagnostic = spec['matrix_seed']==101
    total = None; residuals = {}; balance = 0.
    for seed in noise_seeds:
        response = paper.render(model, source=spec['source'],lossy=spec['lossy'],varying=varying,
                                seed=seed,include_outputs=diagnostic)
        energy,reference = response['energy'],response['reference']
        if total is None: total = np.zeros_like(energy)
        total += energy
        if diagnostic:
            terms = directional_closure_terms(model,response,spec['lossy'],varying)
            balance = max(balance,terms['balance_error'])
            for name in ['port','anisotropy','cross','attenuation']:
                if name not in residuals: residuals[name] = np.zeros_like(terms[name])
                residuals[name] += terms[name]/len(noise_seeds)
    mean = total/len(noise_seeds)
    generator = validation.closed_generator(model,spec['lossy'])
    initial = np.eye(2)[spec['source']]
    closed = validation.continuous_energy(generator,initial,len(mean))
    result = dict(spec=spec,initial_noise_seeds=list(noise_seeds),channel_probabilities=model['channel_probabilities'].tolist(),
                  metrics=paper.energy_errors(mean,reference),vs_closed=paper.energy_errors(mean,closed,window_reference=reference),
                  closed_vs_sa=paper.energy_errors(closed,reference),
                  feedback_sha256=hashlib.sha256(model['feedback'].tobytes()).hexdigest())
    if diagnostic:
        transition = np.eye(2)+generator/paper.FS
        components = {name:validation.forced_response(transition,value) for name,value in residuals.items()}
        values,vectors = np.linalg.eig(transition)
        euler = np.real_if_close((values[None,:]**np.arange(len(mean))[:,None]*
                  np.linalg.solve(vectors,initial)[None,:])@vectors.T).real
        error = float(np.max(np.abs(euler+sum(components.values())-mean)))
        assert error < 1e-8, error
        stop = int(round(result['metrics']['evaluation_stop_s']*paper.FS))+1
        result['closure'] = dict(balance_max_abs=balance,reconstruction_max_abs=error,
            component_mae_pp={name:float(100*np.abs(value[:stop]).mean()) for name,value in components.items()})
        indices = np.unique(np.r_[np.arange(0,len(mean),960),len(mean)-1])
        result['trace'] = dict(time_s=(indices/paper.FS).tolist())
        for name,value in dict(energy=mean,sa=reference,closed=closed,**components).items():
            result['trace'][name] = np.round(value[indices],10).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--code-revision')
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    cache = args.output/'cases'; cache.mkdir(exist_ok=True)
    paths = [Path(__file__),Path(validation.__file__),Path(paper.__file__)]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=paths[0].resolve().parents[2],capture_output=True,text=True)
    provenance = dict(code_revision=args.code_revision or (git.stdout.strip() if git.returncode==0 else None),
                      source_sha256=hashes,python=platform.python_version(),numpy=np.__version__)
    cases = {}; pending = []
    for spec in case_specs():
        key = case_key(spec); path = cache/(key+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['provenance']['source_sha256'] != hashes: raise ValueError(f'{path}: source changed; use a fresh output')
            cases[key] = saved['case']
        else: pending.append(spec)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case,spec):spec for spec in pending}
        for future in as_completed(futures):
            result = future.result(); key = case_key(futures[future]); cases[key] = result
            (cache/(key+'.json')).write_text(json.dumps(dict(provenance=provenance,case=result))+'\n')
            print(f'{len(cases)}/240 {key}: {result["metrics"]["full_mae_pp"]:.5f} pp',flush=True)
    model = channel_model(101)
    parameters = {key:model[key].tolist() for key in ['volumes','apertures','delays','totals','phi','gamma','generator']}
    report = dict(provenance=provenance,settings=dict(sample_rate=paper.FS,duration_s=5.,profiles=PROFILES,
                  basis_seeds=BASIS_SEEDS,matrix_seeds=validation.MATRIX_SEEDS,noise_seeds=validation.NOISE_SEEDS,
                  modulation_seed=701,modulation_amplitude_rad=3.,diagnostic_matrix_seed=101),
                  network_parameters=parameters,cases=[cases[case_key(s)] for s in case_specs()])
    (args.output/'channel-coupling-results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
