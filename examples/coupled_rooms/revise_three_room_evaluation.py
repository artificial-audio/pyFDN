"""Compact three-room comparison and coupling metrics for the revised paper.

Run from the checkout root with PYTHONPATH=src and --output <directory>.
The absorption profile was selected from SA-only decay curvature, not FDN error.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

import export_paper_figures as paper
from revise_two_room_figures import comparison_handles, plot_comparison
from validate_sa_assumptions import continuous_energy

ABSORPTION = np.array([.06, .02, .005])
APERTURES = {'chain': np.array([[0., 2., 0.], [2., 0., 1.5], [0., 1.5, 0.]]),
             'complete': np.array([[0., 2., .5], [2., 0., 1.5], [.5, 1.5, 0.]])}


def model_for(topology):
    model = paper.room_model([160, 80, 120], APERTURES[topology])
    model['gamma'] = paper.create_diagonal_absorption_matrix(
        6*np.cbrt(model['volumes'])**2*ABSORPTION, model['volumes'])
    return model


def closed_generator(model, lossy):
    q = paper.FS*model['lines']*(model['phi']**2-np.eye(3))/model['totals'][None, :]
    return q-model['gamma'] if lossy else q


def late_db_error(energy, reference):
    time = np.arange(len(energy))/paper.FS
    mask = ((time >= .2) & (time <= 3))[:, None] & (reference >= 1e-6)
    error = abs(10*np.log10(np.maximum(energy, 1e-30))-10*np.log10(np.maximum(reference, 1e-30)))
    return float(error[mask].mean())


def render_panel(spec):
    topology, lossy, output = spec
    model = model_for(topology)
    print('Rendering', topology, 'lossy' if lossy else 'lossless', flush=True)
    cases = {name: paper.ensemble(model, source=0, lossy=lossy, varying=varying)
             for name, varying in [('varying', True), ('fixed', False)]}
    cases['impulse'] = paper.render(model, source=0, lossy=lossy, varying=True, initialization='impulse')
    metrics = {}
    for name, response in cases.items():
        values, reference = response['energy'], response['reference']
        assert np.isfinite(values).all() and values.min() >= -1e-12
        np.testing.assert_allclose(values[0], [1., 0., 0.], atol=1e-12, rtol=0)
        if lossy:
            assert np.diff(values.sum(axis=1)).max() < 1e-10
        else:
            np.testing.assert_allclose(values.sum(axis=1), 1., atol=1e-10, rtol=0)
        metrics[name] = dict(paper.energy_errors(values, reference),
                             later_log_energy_mae_db=late_db_error(values, reference))
    np.testing.assert_array_equal(cases['varying']['reference'], cases['fixed']['reference'])
    key = topology+('_lossy' if lossy else '_lossless')
    curves = {name: response['energy'][::48] for name, response in cases.items()}
    curves.update(reference=cases['varying']['reference'][::48],
                  varying_se=cases['varying']['se'][::48], fixed_se=cases['fixed']['se'][::48])
    np.savez_compressed(output/(key+'-comparison-curves.npz'), **curves)
    (output/(key+'-metrics.json')).write_text(json.dumps(metrics, indent=2)+'\n')
    print('Completed', key, 'TV-noise MAE', metrics['varying']['full_mae_pp'], flush=True)
    return topology, lossy, cases, metrics


def spectrum_metrics(reference, closed):
    sa = np.linalg.eigvals(reference)
    fdn = np.linalg.eigvals(closed)
    rows, cols = linear_sum_assignment(abs(sa[:, None]-fdn[None, :]))
    sa, fdn = sa[rows], fdn[cols]
    active = abs(sa) > 1e-9
    relative = abs(fdn[active]-sa[active])/abs(sa[active])
    return {'sa_eigenvalues': [{'real': float(x.real), 'imag': float(x.imag)} for x in sa],
            'closed_eigenvalues': [{'real': float(x.real), 'imag': float(x.imag)} for x in fdn],
            'maximum_nonzero_rate_relative_error_percent': float(100*relative.max()),
            'sa_relaxation_times_s': (-1/sa[active].real).tolist(),
            'closed_relaxation_times_s': (-1/fdn[active].real).tolist()}


def static_metrics(model, lossy):
    reference = model['generator']-(model['gamma'] if lossy else 0)
    closed = closed_generator(model, lossy)
    count = int(5*paper.FS)-int(max(model['delays']))
    result = spectrum_metrics(reference, closed)
    result['source_trajectories'] = []
    for source in range(3):
        initial = np.eye(3)[source]
        sa = continuous_energy(reference, initial, count)
        fdn = continuous_energy(closed, initial, count)
        display_error = abs(fdn[:int(3*paper.FS)+1:48]-sa[:int(3*paper.FS)+1:48])
        result['source_trajectories'].append(dict(source_room=source+1, **paper.energy_errors(fdn, sa),
            interval_0_3s_mae_pp=float(100*display_error.mean()),
            interval_0_3s_peak_pp=float(100*display_error.max())))
    return result


def door_fraction(t):
    return float(np.interp(t, [0., 1., 1.25, 2., 2.25, 3.], [1., 1., 0., 0., 1., 1.]))


def moving_generators(model, topology, lossy, t):
    area = APERTURES[topology].copy()
    area[1, 2] = area[2, 1] = 1.5*door_fraction(t)
    beta = paper.create_lossless_coupling_matrix(area, model['volumes'])
    reference = paper.create_state_transition_matrix(beta)
    theta = paper.get_coupling_angles(beta, model['totals']/(model['lines']*paper.FS))
    phi = expm(paper.get_coupling_matrix(theta))
    closed = paper.FS*model['lines']*(phi**2-np.eye(3))/model['totals'][None, :]
    if lossy:
        reference = reference-model['gamma']
        closed = closed-model['gamma']
    return reference, closed


def dynamic_metrics(model, topology, lossy, output):
    time = np.linspace(0., 3., 3001)

    def rhs(t, state):
        sa, fdn = moving_generators(model, topology, lossy, t)
        matrices = state.reshape(2, 3, 3)
        return np.stack([sa@matrices[0], fdn@matrices[1]]).ravel()

    solutions = []
    for tolerance, step in [(1e-10, .02), (1e-12, .01)]:
        # Stop at the known ramp boundaries, where sqrt-aperture angles are nonsmooth.
        state = np.stack([np.eye(3), np.eye(3)]).ravel()
        pieces = []
        boundaries = [0., 1., 1.25, 2., 2.25, 3.]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            grid = time[(time >= left) & (time <= right)]
            sol = solve_ivp(rhs, (left, right), state, method='DOP853', t_eval=grid,
                            rtol=tolerance, atol=tolerance*.01, max_step=step)
            assert sol.success, sol.message
            pieces.append(sol.y.T[1:] if pieces else sol.y.T)
            state = sol.y[:, -1]
        solutions.append(np.concatenate(pieces).reshape(-1, 2, 3, 3))
    integration_difference = float(np.max(abs(solutions[0]-solutions[1])))
    assert integration_difference < 1e-8, integration_difference
    propagated = solutions[1]
    assert propagated.min() > -1e-10
    if not lossy:
        np.testing.assert_allclose(propagated.sum(axis=2), 1., atol=1e-9, rtol=0)
    else:
        assert propagated.sum(axis=2).max() < 1.+1e-9
    error = abs(propagated[:, 1]-propagated[:, 0])
    result = {'interval_s': [0., 3.], 'evaluation_spacing_s': .001,
              'source_trajectories': [{'source_room': source+1,
                  'mae_pp': float(100*error[:, :, source].mean()),
                  'maximum_error_pp': float(100*error[:, :, source].max())} for source in range(3)],
              'integration_maximum_difference': integration_difference, 'fixed_delays': True,
              'fixed_H': float(paper.FS*model['lines']*model['volumes'][0]/model['totals'][0]),
              'scope': 'Closed room-energy models only; no pressure-domain dynamic-aperture render'}
    result['frozen_spectra'] = {str(t): spectrum_metrics(*moving_generators(model, topology, lossy, t))
                                for t in [0., 1.125, 1.5]}
    name = topology+('_lossy' if lossy else '_lossless')+'-door-trajectories.npz'
    np.savez_compressed(output/name, time=time, sa=propagated[:, 0], closed=propagated[:, 1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-revision', default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout='constrained')
    report = {'absorption_coefficients': ABSORPTION.tolist(), 'display_interval_s': [0., 3.],
              'audio_duration_s': 5., 'sample_rate': paper.FS, 'noise_seeds': paper.ENSEMBLE_SEEDS,
              'matrix_seed': 12314, 'audio_renders': 132, 'results': {}, 'models': {}}
    specs = [(name, lossy, args.output) for name in APERTURES for lossy in [False, True]]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for topology, lossy, cases, metrics in pool.map(render_panel, specs):
            ax = axes[list(APERTURES).index(topology), int(lossy)]
            plot_comparison(ax, cases, lossy, 0, end_time=3)
            ax.set_title(f"{topology.title()}; {'lossy' if lossy else 'lossless'}")
            if lossy:
                ax.set_ylim(-60, 2)
            for index, method in enumerate(['reference', 'varying', 'fixed', 'impulse']):
                response = cases['varying'] if method == 'reference' else cases[method]
                values = response['reference'] if method == 'reference' else response['energy']
                for room in range(3):
                    expected = values[:int(3*paper.FS)+1:48, room]
                    if lossy:
                        expected = 10*np.log10(np.maximum(expected, 1e-12))
                    np.testing.assert_array_equal(ax.lines[3*index+room].get_ydata(), expected)
            marker_sets = [set(range(*[line.get_markevery()[0], 3001, line.get_markevery()[1]]))
                           for line in ax.lines[3:]]
            for i in range(len(marker_sets)):
                for j in range(i+1, len(marker_sets)):
                    assert marker_sets[i].isdisjoint(marker_sets[j])
            report['results'][topology+('_lossy' if lossy else '_lossless')] = metrics
    figure.legend(handles=comparison_handles(3), loc='outside upper center', ncol=4)
    figure.savefig(args.output/'joint-three-room-energy.pdf')
    plt.close(figure)
    (args.output/'primary-results.json').write_text(json.dumps(report, indent=2)+'\n')
    coupling = {'static': {}, 'dynamic': {},
                'dynamic_protocol': {'duration_s': 3., 'aperture_23_area_m2': 1.5,
                    'closing_ramp_s': [1., 1.25], 'closed_interval_s': [1.25, 2.],
                    'opening_ramp_s': [2., 2.25], 'other_apertures_unchanged': True}}
    for topology in APERTURES:
        model = model_for(topology)
        report['models'][topology] = {key: model[key].tolist() if isinstance(model[key], np.ndarray) else model[key]
                                     for key in ['volumes', 'apertures', 'delays', 'totals', 'lines', 'gamma']}
        for lossy in [False, True]:
            key = topology+('_lossy' if lossy else '_lossless')
            coupling['static'][key] = static_metrics(model, lossy)
            coupling['dynamic'][key] = dynamic_metrics(model, topology, lossy, args.output)
            print('Completed coupling metrics', key, flush=True)
    (args.output/'coupling-metrics.json').write_text(json.dumps(coupling, indent=2)+'\n')
    report['code_revision'] = args.code_revision
    report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), Path(__file__).with_name('revise_two_room_figures.py'),
         Path(__file__).with_name('export_paper_figures.py'), Path(__file__).with_name('validate_sa_assumptions.py')]}
    report['file_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(args.output.iterdir()) if p.suffix in ['.npz', '.pdf', '.json']
                             and p.name != 'joint-three-room-comparison-manifest.json'}
    (args.output/'joint-three-room-comparison-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Completed four-way Fig. 4 and static/dynamic coupling metrics.', flush=True)


if __name__ == '__main__':
    main()
