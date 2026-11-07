"""Validate the SA closure, feedback robustness and modulation strength.

Run export_paper_figures.py first, then run this script with the same --output.
The validation extends its manifest/table and replaces the endpoint histogram
with the controlled validation figure. Case files checkpoint the long run.
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
from scipy.linalg import block_diag, expm
from scipy.signal import lfilter

import pyFDN
import export_paper_figures as paper

MATRIX_SEEDS = [101, 102, 103, 104, 105]
MODULATION_SEEDS = [701, 702]
NOISE_SEEDS = list(range(200, 208))
AMPLITUDES = [0., .1, .3, 1., 3.]
TOPOLOGIES = {"chain": [[0, 2, 0], [2, 0, 1.5], [0, 1.5, 0]],
              "complete": [[0, 2, .5], [2, 0, 1.5], [.5, 1.5, 0]]}


def validation_model(topology, matrix_seed, modulation_seed, amplitude):
    model = paper.room_model([160, 80, 120], TOPOLOGIES[topology])
    rng = np.random.default_rng(matrix_seed)
    mixers = block_diag(*[pyFDN.random_orthogonal(model['lines'], rng) for _ in range(3)])
    model['feedback'] = paper.get_feedback_matrix(model['phi'], mixers, model['lines'])
    saved = np.random.get_state()
    try:
        np.random.seed(modulation_seed)
        model['hook'] = paper.get_room_time_varying_matrix(
            model['phi'], model['lines'], amplitude=amplitude, fs=paper.FS)
    finally:
        np.random.set_state(saved)
    return model


def closed_generator(model, lossy):
    rooms = len(model['volumes'])
    result = paper.FS*model['lines']*(model['phi']**2-np.eye(rooms))/model['totals'][None, :]
    return result-model['gamma'] if lossy else result


def continuous_energy(generator, initial, count):
    values, vectors = np.linalg.eig(generator)
    modes = np.exp(np.arange(count)[:, None]/paper.FS*values)
    result = (modes*(np.linalg.solve(vectors, initial))[None, :])@vectors.T
    assert np.max(np.abs(result.imag)) < 1e-10
    return result.real


def forced_response(transition, forcing):
    """x[0]=0, x[n+1]=T*x[n]+forcing[n], including the final state."""
    values, vectors = np.linalg.eig(transition)
    coordinates = np.vstack([forcing, np.zeros((1, forcing.shape[1]))])@np.linalg.inv(vectors).T
    filtered = np.column_stack([lfilter([0., 1.], [1., -value], coordinates[:, k])
                                for k, value in enumerate(values)])
    result = filtered@vectors.T
    assert np.max(np.abs(result.imag)) < 1e-9
    return result.real


def closure_terms(model, response, lossy, varying):
    """Exact per-update forcing of the closed room-energy recursion."""
    energy = response['energy']
    outputs = response['outputs'][:len(energy)-1]
    k, rooms = model['lines'], len(model['volumes'])
    attenuation = (paper.get_decay_matrix(model['gamma'], model['delays'].reshape(rooms, k), paper.FS)
                   if lossy else np.eye(len(model['delays'])))
    written = outputs@(model['feedback']@attenuation).T
    if varying:
        model['hook'].reset()
        written = model['hook'].process_signal(written)
    outgoing = (outputs**2).reshape(-1, rooms, k).sum(axis=2)
    after_loss = ((outputs*np.diag(attenuation))**2).reshape(-1, rooms, k).sum(axis=2)
    incoming = (written**2).reshape(-1, rooms, k).sum(axis=2)
    balance_error = float(np.max(np.abs(np.diff(energy, axis=0)-(incoming-outgoing))))
    assert balance_error < 1e-9, balance_error
    squared = model['phi']**2
    port = (outgoing-energy[:-1]*k/model['totals'])@(squared-np.eye(rooms)).T
    cross = incoming-after_loss@squared.T
    loss = (after_loss-outgoing)@squared.T
    if lossy:
        loss += (energy[:-1]@model['gamma'].T)/paper.FS
    return dict(port=port, cross=cross, attenuation=loss, outgoing=outgoing,
                balance_error=balance_error)


def case_specs():
    result = []
    for matrix_seed in MATRIX_SEEDS:
        for topology in TOPOLOGIES:
            for source in range(3):
                for lossy in [False, True]:
                    for amplitude, modulation_seed in [(0., 701), (3., 701), (3., 702)]:
                        result.append(dict(topology=topology, matrix_seed=matrix_seed, source=source,
                                           lossy=lossy, amplitude=amplitude, modulation_seed=modulation_seed))
    for matrix_seed in MATRIX_SEEDS:
        for topology in TOPOLOGIES:
            for amplitude in [.1, .3, 1.]:
                result.append(dict(topology=topology, matrix_seed=matrix_seed, source=0,
                                   lossy=False, amplitude=amplitude, modulation_seed=701))
    return result


def case_key(spec):
    return (f"{spec['topology']}_m{spec['matrix_seed']}_s{spec['source']+1}_"
            f"{'lossy' if spec['lossy'] else 'lossless'}_a{spec['amplitude']:g}_r{spec['modulation_seed']}")


def run_case(spec):
    model = validation_model(spec['topology'], spec['matrix_seed'], spec['modulation_seed'], spec['amplitude'])
    varying = spec['amplitude'] > 0
    diagnostic = spec['matrix_seed'] == MATRIX_SEEDS[0] and spec['modulation_seed'] == MODULATION_SEEDS[0] and spec['amplitude'] in [0., 3.]
    convergence = spec['topology'] == 'complete' and spec['matrix_seed'] == 101 and spec['source'] == 0 and not spec['lossy'] and spec['modulation_seed'] == 701 and spec['amplitude'] in [0., 3.]
    seeds = list(range(200, 232)) if convergence else NOISE_SEEDS
    total = squared_sum = None
    totals = {}
    balance = 0.
    checkpoints = {}
    trials = []
    generator = closed_generator(model, spec['lossy'])
    for n, seed in enumerate(seeds, 1):
        response = paper.render(model, source=spec['source'], lossy=spec['lossy'], varying=varying,
                                seed=seed, include_outputs=diagnostic and n <= len(NOISE_SEEDS))
        energy, reference = response['energy'], response['reference']
        if total is None:
            total = np.zeros_like(energy); squared_sum = np.zeros_like(energy)
            initial = np.eye(3)[spec['source']]
            closed = continuous_energy(generator, initial, len(energy))
        total += energy; squared_sum += energy**2
        trials.append(paper.energy_errors(energy, reference))
        if diagnostic and n <= len(NOISE_SEEDS):
            terms = closure_terms(model, response, spec['lossy'], varying)
            for name in ['port', 'cross', 'attenuation', 'outgoing']:
                if name not in totals: totals[name] = np.zeros_like(terms[name])
                totals[name] += terms[name]
            balance = max(balance, terms['balance_error'])
        if n in [8, 16, 32]:
            mean = total/n
            checkpoints[str(n)] = paper.energy_errors(mean, reference)
            if n == 8:
                primary = mean.copy()
                standard_error = np.sqrt(np.maximum((squared_sum-total**2/n)/(n-1), 0)/n)
    result = dict(spec=spec, initial_noise_seeds=NOISE_SEEDS, metrics=checkpoints['8'],
                  vs_closed=paper.energy_errors(primary, closed, window_reference=reference),
                  closed_vs_sa=paper.energy_errors(closed, reference), trial_metrics=trials[:8],
                  convergence=checkpoints if convergence else {},
                  feedback_sha256=hashlib.sha256(model['feedback'].tobytes()).hexdigest())
    if diagnostic:
        terms = {key: value/8 for key, value in totals.items()}
        transition = np.eye(3)+generator/paper.FS
        components = {name: forced_response(transition, terms[name]) for name in ['port', 'cross', 'attenuation']}
        # Euler's unforced state has T^n initial, not exp(L*t).
        values, vectors = np.linalg.eig(transition)
        euler = ((values[None, :]**np.arange(len(primary))[:, None])*
                 np.linalg.solve(vectors, initial)[None, :])@vectors.T
        euler = np.real_if_close(euler).real
        reconstructed = euler+sum(components.values())
        reconstruction_error = float(np.max(np.abs(reconstructed-primary)))
        assert reconstruction_error < 1e-8, reconstruction_error
        stop = int(round(result['metrics']['evaluation_stop_s']*paper.FS))+1
        result['closure'] = dict(balance_max_abs=balance, reconstruction_max_abs=reconstruction_error,
                                component_mae_pp={name: float(100*np.abs(value[:stop]).mean()) for name, value in components.items()},
                                euler_difference_mae_pp=float(100*np.abs(euler[:stop]-closed[:stop]).mean()))
        # Twenty-ms bins reduce finite-ensemble endpoint fluctuations; normalization
        # uses initial total energy, avoiding division by empty-room energy.
        bins = stop//960
        equivalent = terms['outgoing']*model['totals']/model['lines']-primary[:-1]
        result['closure']['port_equivalent_20ms_mae_pp'] = float(100*np.abs(equivalent[:bins*960].reshape(bins, 960, 3).mean(axis=1)).mean())
        indices = np.unique(np.r_[np.arange(0, len(primary), 960), len(primary)-1])
        result['trace'] = dict(time_s=(indices/paper.FS).tolist())
        for name, value in dict(energy=primary, sa=reference, closed=closed, standard_error=standard_error, **components).items():
            result['trace'][name] = np.round(value[indices], 10).tolist()
    return result


def write_assets(output, cases, provenance):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 8.), layout='constrained')
    for panel, (ax, topology) in enumerate(zip(axes[0], TOPOLOGIES)):
        for source in range(3):
            fixed = [c['metrics']['full_mae_pp'] for c in cases if c['spec']['topology']==topology and c['spec']['source']==source and not c['spec']['lossy'] and c['spec']['amplitude']==0]
            varied = [c['metrics']['full_mae_pp'] for c in cases if c['spec']['topology']==topology and c['spec']['source']==source and not c['spec']['lossy'] and c['spec']['amplitude']==3]
            for values, offset, color, marker in [(fixed, -.14, '#555555', 'o'), (varied, .14, '#1769aa', '^')]:
                ax.scatter(source+offset+np.linspace(-.06,.06,len(values)), values, color=color, marker=marker, s=16)
        ax.set(title=f'({chr(97+panel)}) {topology.title()}: source/matrix robustness', ylabel='SA trajectory MAE (pp)', xticks=range(3), xticklabels=['R1', 'R2', 'R3'], xlabel='Initially excited room', yscale='log')
        ax.legend(handles=[Line2D([], [], color='#555555', marker='o', ls='', label='Fixed'), Line2D([], [], color='#1769aa', marker='^', ls='', label='Modulated')])
    for topology, color, marker in [('chain', '#1769aa', 'o'), ('complete', '#25805d', 's')]:
        groups = [[c['metrics']['full_mae_pp'] for c in cases if c['spec']['topology']==topology and c['spec']['source']==0 and not c['spec']['lossy'] and c['spec']['modulation_seed']==701 and c['spec']['amplitude']==a] for a in AMPLITUDES]
        medians = np.array([np.median(g) for g in groups])
        axes[1,0].errorbar(AMPLITUDES, medians, yerr=[medians-[min(g) for g in groups], [max(g) for g in groups]-medians], label=topology.title(), color=color, marker=marker, capsize=3)
    axes[1,0].set(title='(c) R1: modulation amplitude', xlabel='Nominal rotation amplitude (rad)', ylabel='SA trajectory MAE (pp)', xscale='symlog', yscale='log')
    axes[1,0].set_xscale('symlog', linthresh=.1, linscale=.5)
    axes[1,0].set_xticks(AMPLITUDES, ['0', '0.1', '0.3', '1', '3']); axes[1,0].legend()
    for amplitude, color, marker in [(0., '#555555', 'o'), (3., '#1769aa', '^')]:
        row = next(c for c in cases if c['convergence'] and c['spec']['amplitude']==amplitude)
        axes[1,1].plot([8,16,32], [row['convergence'][str(n)]['full_mae_pp'] for n in [8,16,32]], color=color, marker=marker, label='Fixed' if amplitude==0 else 'Modulated')
    axes[1,1].set(title='(d) Complete, R1: ensemble size', xlabel='Independent initializations', ylabel='SA trajectory MAE (pp)', xticks=[8,16,32], yscale='log'); axes[1,1].legend()
    for panel, (ax, amplitude) in enumerate(zip(axes[2], [0., 3.])):
        row = next(c for c in cases if c['spec']==dict(topology='complete', matrix_seed=101, source=0, lossy=False, amplitude=amplitude, modulation_seed=701))
        trace = {key: np.asarray(value) for key, value in row['trace'].items()}
        curves = [('Total', 100*(trace['energy'][:,0]-trace['sa'][:,0]), '#222222', '-'),
                  ('Rate design', 100*(trace['closed'][:,0]-trace['sa'][:,0]), '#1769aa', '--'),
                  ('Port term', 100*trace['port'][:,0], '#d36b16', '-.'),
                  ('Cross term', 100*trace['cross'][:,0], '#25805d', ':')]
        for label, values, color, style in curves: ax.plot(trace['time_s'], values, label=label, color=color, ls=style)
        ax.set(title=f"({chr(101+panel)}) Complete, R1: {'fixed' if amplitude==0 else 'modulated'}", xlabel='Time (s)', ylabel='Signed R1 correction (pp)'); ax.legend(ncol=2)
    for ax in axes.flat: ax.grid(alpha=.15)
    figure.savefig(output/'joint-energy-distributions.pdf'); plt.close(figure)
    table = [r'% BEGIN VALIDATION', r'\begin{table*}[t]', r'\centering', r'\caption{Robustness across five internal-matrix seeds, all source rooms and two modulation schedules. Each case averages eight unit-energy noise starts. Entries are median [minimum, maximum] SA-trajectory MAE in percentage points; ranges are observed variation, not confidence intervals. Modulated values use the nominal 3-rad setting. Lossy errors use the common SA reference window.}', r'\label{tab:validation-robustness}', r'\begin{tabular}{llrr}', r'\toprule', r'Case & Source room & Fixed feedback & Modulated feedback \\', r'\midrule']
    for topology in TOPOLOGIES:
        for lossy in [False, True]:
            for source in range(3):
                values = [[c['metrics']['full_mae_pp'] for c in cases if c['spec']['topology']==topology and c['spec']['source']==source and c['spec']['lossy']==lossy and c['spec']['amplitude']==a] for a in [0.,3.]]
                formatted = [f'{np.median(g):.3f} [{min(g):.3f}, {max(g):.3f}]' for g in values]
                best = int(np.median(values[1]) < np.median(values[0]))
                if not lossy:
                    formatted[best] = r'\textbf{'+formatted[best]+'}'
                table.append(f"{topology.title()}, {'lossy' if lossy else 'lossless'} & R{source+1} & "+' & '.join(formatted)+r' \\')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', r'% END VALIDATION']
    old = (output/'joint-evaluation.tex').read_text().split('% BEGIN VALIDATION')[0].rstrip()
    (output/'joint-evaluation.tex').write_text(old+'\n\n'+'\n'.join(table)+'\n')
    validation = dict(provenance=provenance, settings=dict(matrix_seeds=MATRIX_SEEDS, modulation_seeds=MODULATION_SEEDS,
                      initial_noise_seeds=NOISE_SEEDS, amplitudes=AMPLITUDES, duration_s=5., sample_rate=paper.FS,
                      diagnostics_matrix_seed=101, diagnostics_modulation_seed=701, diagnostic_trace_spacing_s=.02,
                      robustness_loss_conditions=['lossless','lossy'], amplitude_sweep_loss_condition='lossless', amplitude_sweep_source_room=1,
                      modulation_frequency_Hz=1., modulation_spread=.3), cases=cases)
    manifest_path = output/'joint-figure-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['validation'] = validation
    manifest['file_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('joint-*')) if p.suffix in ['.pdf','.tex']}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--code-revision')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not (args.output/'joint-figure-manifest.json').exists(): parser.error('Run export_paper_figures.py with this --output first.')
    cache = args.output/'validation-cases'; cache.mkdir(exist_ok=True)
    paths = [Path(__file__), Path(paper.__file__)]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    root = Path(__file__).resolve().parents[2]
    revision = args.code_revision
    if revision is None:
        git = subprocess.run(['git','rev-parse','HEAD'], cwd=root, capture_output=True, text=True)
        revision = git.stdout.strip() if git.returncode==0 else None
    provenance = dict(code_revision=revision, source_sha256=source_hashes, python=platform.python_version(),
                      numpy=np.__version__, sample_rate=paper.FS)
    specs = case_specs(); cases = {}; pending = []
    for spec in specs:
        key = case_key(spec); file = cache/(key+'.json')
        if file.exists():
            saved = json.loads(file.read_text())
            if saved['provenance']['source_sha256'] != source_hashes and not args.render_only:
                raise ValueError(f'{file}: source changed; use a fresh output directory or --render-only for unchanged numerical results')
            cases[key] = saved['case']
        else: pending.append(spec)
    if args.render_only and pending: raise ValueError('The validation run is incomplete.')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, spec): spec for spec in pending}
        for future in as_completed(futures):
            spec = futures[future]; result = future.result(); key = case_key(spec)
            (cache/(key+'.json')).write_text(json.dumps(dict(provenance=provenance, case=result))+'\n')
            cases[key] = result
            print(f'{len(cases)}/{len(specs)} {key} MAE={result["metrics"]["full_mae_pp"]:.4f} pp', flush=True)
    ordered = [cases[case_key(s)] for s in specs]
    if args.render_only:
        provenance = json.loads(next(cache.glob('*.json')).read_text())['provenance'] | {'render_source_sha256': source_hashes, 'render_code_revision': revision}
    write_assets(args.output, ordered, provenance)
    print(f'Completed {len(ordered)} cases; updated validation figure, table and manifest.', flush=True)


if __name__ == '__main__':
    main()
