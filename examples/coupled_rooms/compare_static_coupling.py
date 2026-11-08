"""Compare fixed receiving-room rotations with fixed and modulated GFDNs.

The new feedback is diag(U_i) (Phi x I) diag(M_i). Its block Gram
matrices equal Phi_ij**2 I, preserving each individual room contribution
without an additional isotropy assumption. No manuscript assets are changed.
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
RECEIVER_SEEDS = [501, 502]


def static_model(topology, matrix_seed, receiver_seed=None, varying=False):
    model = validation.validation_model(topology, matrix_seed, 701, 3. if varying else 0.)
    original = model['feedback']
    if receiver_seed is not None:
        rng = np.random.default_rng(receiver_seed)
        receiving = block_diag(*[paper.pyFDN.random_orthogonal(model['lines'], rng)
                                 for _ in model['volumes']])
        model['feedback'] = receiving@original
    k, rooms = model['lines'], len(model['volumes'])
    np.testing.assert_allclose(model['feedback'].T@model['feedback'], np.eye(k*rooms), atol=2e-14)
    for i in range(rooms):
        for j in range(rooms):
            block = model['feedback'][i*k:(i+1)*k, j*k:(j+1)*k]
            np.testing.assert_allclose(block.T@block, model['phi'][i,j]**2*np.eye(k), atol=2e-14)
    return model


def case_specs():
    return [dict(topology=topology, matrix_seed=seed, source=source, lossy=lossy,
                 variant=variant, receiver_seed=receiver)
            for seed in validation.MATRIX_SEEDS for topology in validation.TOPOLOGIES
            for source in range(3) for lossy in [False, True]
            for variant, receiver in [('fixed', None), ('receiving', 501),
                                      ('receiving', 502), ('modulated', None)]]


def case_key(spec):
    return (f"{spec['topology']}_m{spec['matrix_seed']}_s{spec['source']+1}_"
            f"{'lossy' if spec['lossy'] else 'lossless'}_{spec['variant']}_u{spec['receiver_seed']}")


def run_case(spec):
    varying = spec['variant'] == 'modulated'
    model = static_model(spec['topology'], spec['matrix_seed'], spec['receiver_seed'], varying)
    diagnostic = spec['matrix_seed'] == 101
    total = None
    residuals = {}
    balance = 0.
    for seed in validation.NOISE_SEEDS:
        response = paper.render(model, source=spec['source'], lossy=spec['lossy'],
                                varying=varying, seed=seed, include_outputs=diagnostic)
        energy, reference = response['energy'], response['reference']
        if total is None: total = np.zeros_like(energy)
        total += energy
        if diagnostic:
            terms = validation.closure_terms(model, response, spec['lossy'], varying)
            balance = max(balance, terms['balance_error'])
            for name in ['port', 'cross', 'attenuation']:
                if name not in residuals: residuals[name] = np.zeros_like(terms[name])
                residuals[name] += terms[name]/len(validation.NOISE_SEEDS)
    mean = total/len(validation.NOISE_SEEDS)
    generator = validation.closed_generator(model, spec['lossy'])
    initial = np.eye(len(model['volumes']))[spec['source']]
    closed = validation.continuous_energy(generator, initial, len(mean))
    result = dict(spec=spec, initial_noise_seeds=validation.NOISE_SEEDS,
                  metrics=paper.energy_errors(mean, reference),
                  vs_closed=paper.energy_errors(mean, closed, window_reference=reference),
                  closed_vs_sa=paper.energy_errors(closed, reference),
                  feedback_sha256=hashlib.sha256(model['feedback'].tobytes()).hexdigest())
    if diagnostic:
        transition = np.eye(len(initial))+generator/paper.FS
        components = {name: validation.forced_response(transition, value)
                      for name, value in residuals.items()}
        values, vectors = np.linalg.eig(transition)
        euler = np.real_if_close((values[None,:]**np.arange(len(mean))[:,None]*
                  np.linalg.solve(vectors, initial)[None,:])@vectors.T).real
        error = float(np.max(np.abs(euler+sum(components.values())-mean)))
        assert error < 1e-8, error
        stop = int(round(result['metrics']['evaluation_stop_s']*paper.FS))+1
        result['closure'] = dict(balance_max_abs=balance, reconstruction_max_abs=error,
            component_mae_pp={name: float(100*np.abs(value[:stop]).mean())
                              for name, value in components.items()})
        indices = np.unique(np.r_[np.arange(0, len(mean), 960), len(mean)-1])
        result['trace'] = dict(time_s=(indices/paper.FS).tolist())
        for name, value in dict(energy=mean, sa=reference, closed=closed, **components).items():
            result['trace'][name] = np.round(value[indices], 10).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--code-revision')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'cases'; cache.mkdir(exist_ok=True)
    paths = [Path(__file__), Path(validation.__file__), Path(paper.__file__)]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=paths[0].resolve().parents[2], capture_output=True, text=True)
    provenance = dict(code_revision=args.code_revision or (git.stdout.strip() if git.returncode==0 else None),
                      source_sha256=hashes, python=platform.python_version(), numpy=np.__version__)
    cases = {}; pending = []
    for spec in case_specs():
        key = case_key(spec); path = cache/(key+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['provenance']['source_sha256'] != hashes:
                raise ValueError(f'{path}: source changed; use a fresh output directory')
            cases[key] = saved['case']
        else: pending.append(spec)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, spec): spec for spec in pending}
        for future in as_completed(futures):
            result = future.result(); key = case_key(futures[future])
            (cache/(key+'.json')).write_text(json.dumps(dict(provenance=provenance, case=result))+'\n')
            cases[key] = result
            print(f'{len(cases)}/240 {key}: {result["metrics"]["full_mae_pp"]:.5f} pp', flush=True)
    report = dict(provenance=provenance, settings=dict(sample_rate=paper.FS, duration_s=5.,
                  receiver_seeds=RECEIVER_SEEDS, matrix_seeds=validation.MATRIX_SEEDS,
                  noise_seeds=validation.NOISE_SEEDS, modulation_seed=701, modulation_amplitude_rad=3.,
                  diagnostic_matrix_seed=101), cases=[cases[case_key(s)] for s in case_specs()])
    (args.output/'static-coupling-results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
