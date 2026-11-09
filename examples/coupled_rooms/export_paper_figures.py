"""Reproduce the coupled-room paper's numerical figures and run manifest.

Run from the repository root:
    python examples/coupled_rooms/export_paper_figures.py --output /path/to/figures

SA verification starts from independent Gaussian delay-cell values in one
room, normalized to unit stored energy. Impulse rendering is measured
separately. No external room dataset or marimo session is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.linalg import block_diag, expm

import pyFDN
from pyFDN.build import FDNBuild
from pyFDN.auxiliary.physics_based_coupling import (
    create_diagonal_absorption_matrix, create_lossless_coupling_matrix,
    create_state_transition_matrix, get_coupling_angles, get_coupling_matrix,
    get_decay_matrix, get_feedback_matrix, get_room_time_varying_matrix,
    gfdn_energy_ledger, match_delay_totals_to_volumes, trajectory,
)

FS = 48000
ENSEMBLE_SEEDS = list(range(200, 216))
COLORS = ["#1769aa", "#d36b16", "#25805d"]
SA_LINE_STYLE = dict(ls="-", lw=3.6, alpha=.3, zorder=1)
GFDN_LINE_STYLE = dict(ls="-", lw=1.1, marker="o", ms=3.8,
                       markerfacecolor="white", markeredgewidth=.8, zorder=3)


def room_model(volumes, apertures, lines=16, setting=2000, matrix_seed=12314):
    volumes = np.asarray(volumes, dtype=float)
    if len(volumes) == 2:
        base = np.r_[pyFDN.sample_delay_lengths(lines, (setting, 3*setting), coprime=True, rng=331),
                     pyFDN.sample_delay_lengths(lines, (setting//2, 3*setting//2), coprime=True, rng=122)]
    else:
        base = np.concatenate([
            pyFDN.sample_delay_lengths(lines, (int(.5*setting*v/volumes[0]), int(1.5*setting*v/volumes[0])),
                                       coprime=True, rng=331+10*i)
            for i, v in enumerate(volumes)])
    delays = match_delay_totals_to_volumes(base, lines, volumes)
    totals = delays.reshape(len(volumes), lines).sum(axis=1)
    beta = create_lossless_coupling_matrix(np.asarray(apertures), volumes)
    generator = create_state_transition_matrix(beta)
    theta = get_coupling_angles(beta, totals/(lines*FS))
    phi = expm(get_coupling_matrix(theta))
    saved = np.random.get_state()
    try:
        np.random.seed(matrix_seed)
        hook = get_room_time_varying_matrix(phi, lines, fs=FS)
        mixers = block_diag(*[pyFDN.random_orthogonal(lines) for _ in volumes])
    finally:
        np.random.set_state(saved)
    gamma = create_diagonal_absorption_matrix(6*np.cbrt(volumes)**2*np.array([.1,.3,.2][:len(volumes)]), volumes)
    feedback = get_feedback_matrix(phi, mixers, lines)
    return dict(volumes=volumes, apertures=np.asarray(apertures), delays=delays,
                totals=totals, lines=lines, generator=generator, gamma=gamma,
                phi=phi, theta=theta, feedback=feedback, hook=hook,
                matrix_seed=matrix_seed, delay_setting_samples=setting)


def initialise_room_noise(state, delays, lines, source, seed):
    """Use the existing delay-state API; unit energy and isotropic covariance.

    Normalized Gaussian samples are uniform on the source-room sphere. They
    have covariance I/tau but are not independent after normalization.
    """
    state.fill_noise(rng=np.random.default_rng(seed))
    contents = [state.buffer[j, :length].copy() if j//lines == source else np.zeros(length)
                for j, length in enumerate(delays)]
    norm = np.sqrt(sum(np.dot(values, values) for values in contents))
    contents = [values/norm for values in contents]
    state.set_lines(contents)


def render(model, source=0, lossy=False, varying=True, initialization="noise", seed=200, duration=5., include_outputs=False):
    delays, k = model["delays"], model["lines"]
    rooms, channels = len(model["volumes"]), len(delays)
    feedback = model["feedback"]
    if lossy:
        feedback = feedback @ get_decay_matrix(model["gamma"], delays.reshape(rooms,k), FS)
    gains = np.zeros((channels,1)); gains[source*k:(source+1)*k] = 1/np.sqrt(k)
    model["hook"].reset()
    build = FDNBuild(A=feedback, B=gains, C=np.eye(channels), D=np.zeros((channels,1)),
                     delays=delays, fs=FS, post_matrix=model["hook"] if varying else None)
    graph = pyFDN.build_to_td(build)
    signal = np.zeros((int(duration*FS),1))
    if initialization == "noise":
        initialise_room_noise(graph.ops[0].ops[1].state, delays, k, source, seed)
    elif initialization == "impulse":
        signal[0,0] = 1
    else:
        raise ValueError("initialization must be noise or impulse")
    outputs = graph.process_signal(signal)
    ledger, _ = gfdn_energy_ledger(outputs[None], rooms, k, delays)
    energy = ledger[0]
    if initialization == "noise":
        # The helper observes the state AFTER an update; prepend the known
        # pre-update state to align the free decay with SA at t=0.
        energy = np.vstack([np.eye(rooms)[source], energy])
    _, reference = trajectory(model["generator"], np.eye(rooms)[source], np.eye(rooms),
                              FS, len(energy), model["gamma"] if lossy else None)
    late = slice(int(.85*len(energy)), len(energy))
    endpoint_mean = (outputs[late]**2).mean(axis=0)
    microphones = np.stack([outputs[:FS//2, r*k:(r+1)*k].sum(axis=1)/np.sqrt(k) for r in range(rooms)], axis=1)
    result = dict(energy=energy, reference=reference, endpoint_mean=endpoint_mean, microphones=microphones)
    if include_outputs:
        result["outputs"] = outputs
    return result


def energy_errors(energy, reference, window_reference=None):
    # Bound lossy error averages by the useful reference dynamic range;
    # arbitrarily long zero tails must not improve the reported accuracy.
    window = reference if window_reference is None else window_reference
    valid = np.flatnonzero(window.sum(axis=1) >= window[0].sum()*1e-6)
    stop = int(valid[-1])+1
    energy, reference = energy[:stop], reference[:stop]
    error = np.abs(energy-reference)
    early_stop = min(int(.2*FS),len(energy))
    return dict(evaluation_stop_s=(stop-1)/FS, full_mae_pp=float(100*error.mean()), early_mae_pp=float(100*error[:early_stop].mean()),
                later_mae_pp=float(100*error[early_stop:].mean()), maximum_error_pp=float(100*error.max()))


def ensemble(model, source=0, lossy=False, varying=True):
    total = squared = None
    individual = []
    for seed in ENSEMBLE_SEEDS:
        response = render(model, source, lossy, varying, seed=seed)
        energy = response["energy"]
        if total is None:
            total = np.zeros_like(energy); squared = np.zeros_like(energy)
        total += energy; squared += energy**2
        individual.append(energy_errors(energy,response["reference"]))
    count = len(ENSEMBLE_SEEDS)
    mean = total/count
    se = np.sqrt(np.maximum((squared-total**2/count)/(count-1),0)/count)
    return dict(energy=mean, se=se, reference=response["reference"],
                metrics=energy_errors(mean,response["reference"]), individual=individual)


def trajectory_handles(rooms):
    return [Line2D([], [], color=COLORS[i], label=f"Room {i+1}") for i in range(rooms)] + [
        Line2D([], [], color="black", label="GFDN", **GFDN_LINE_STYLE),
        Line2D([], [], color="black", label="SA", **SA_LINE_STYLE)]


def plot_trajectories(axes, response, lossy, title):
    # One-ms display spacing; all error metrics still use every audio sample.
    indices = np.unique(np.r_[np.arange(0,len(response["energy"]),48),len(response["energy"])-1])
    time = indices/FS
    for room in range(response["energy"].shape[1]):
        energy = response["energy"][indices,room]
        reference = response["reference"][indices,room]
        if lossy:
            energy=10*np.log10(np.maximum(energy,1e-12))
            reference=10*np.log10(np.maximum(reference,1e-12))
        axes.plot(time,reference,color=COLORS[room],**SA_LINE_STYLE)
        # Marker times are staggered between rooms; curve values stay unchanged.
        axes.plot(time,energy,color=COLORS[room],markevery=(50+200*room,600),
                  **GFDN_LINE_STYLE)
        if not lossy:
            axes.fill_between(time,response["energy"][indices,room]-response["se"][indices,room],
                              response["energy"][indices,room]+response["se"][indices,room],
                              color=COLORS[room],alpha=.15,zorder=.5)
    axes.set(title=title,xlabel="Time (s)",ylabel="Energy / injected energy (dB)" if lossy else "Fraction of injected energy")
    axes.set_ylim((-85,2) if lossy else (-.02,1.02))
    axes.grid(alpha=.15)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--code-revision",default=None)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.labelsize":9,"legend.fontsize":7,"pdf.fonttype":42})
    manifest=dict(protocol="noise-initialized SA verification; impulse response and mixing separate",
                  noise_seeds=ENSEMBLE_SEEDS,noise_normalization="unit energy in selected room; spherical Gaussian draw",
                  sample_rate=FS,duration_s=5.,plot_spacing_s=.001,early_interval_s=[0,.2],later_interval_s=[.2,5.],
                  averaging="MAE of ensemble-mean stored energy; 16 initializations at a fixed feedback realization",
                  error_window_note="200 ms is a reporting boundary, not a claimed mixing time; averages stop at a 60-dB drop in reference total energy or render end",
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                  code_revision=args.code_revision,initialization_source_commit="de4da42dc4c3a0cafbc835d02da368a68bb5b712",models={},results={})
    root=Path(__file__).resolve().parents[2]
    if manifest["code_revision"] is None:
        revision=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True)
        if revision.returncode==0:manifest["code_revision"]=revision.stdout.strip()
    manifest["source_sha256"]={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(list((root/'src/pyFDN').rglob('*.py'))+list(Path(__file__).parent.glob('*.py')))}
    models={"two":room_model([160,80],[[0,2],[2,0]],setting=1000,matrix_seed=12343),
            "chain":room_model([160,80,120],[[0,2,0],[2,0,1.5],[0,1.5,0]]),
            "complete":room_model([160,80,120],[[0,2,.5],[2,0,1.5],[.5,1.5,0]])}
    for name,model in models.items():
        manifest["models"][name]={key:model[key].tolist() if isinstance(model[key],np.ndarray) else model[key]
                                  for key in ["volumes","apertures","delays","totals","lines","theta","matrix_seed","delay_setting_samples"]}
        manifest["models"][name]["modal_relaxation_times_s"]=sorted((-1/np.linalg.eigvals(model['generator']).real[np.linalg.eigvals(model['generator']).real < -1e-10]).tolist())
    ir_figure,ir_axes=plt.subplots(2,2,figsize=(7,4),layout="constrained")
    two_figure,two_axes=plt.subplots(2,2,figsize=(7.2,5.3),layout="constrained")
    three_figure,three_axes=plt.subplots(2,2,figsize=(7.2,5.3),layout="constrained")
    for name,model in models.items():
        sources=range(2) if name=="two" else [0]
        for source in sources:
            for lossy in [False,True]:
                key=f"{name}_source{source+1}_{'lossy' if lossy else 'lossless'}"
                result=ensemble(model,source,lossy)
                impulse=render(model,source,lossy,initialization="impulse")
                manifest['results'][key]=dict(noise=result['metrics'],noise_trial_metrics=result['individual'],
                                               impulse=energy_errors(impulse['energy'],impulse['reference']))
                axes=two_axes[source,int(lossy)] if name=="two" else three_axes[['chain','complete'].index(name),int(lossy)]
                plot_trajectories(axes,result,lossy,f"{'Source R'+str(source+1) if name=='two' else name.title()}; {'lossy' if lossy else 'lossless'}")
                if name=="two":
                    for receiver in range(2):
                        ir_axes[source,receiver].plot(np.arange(len(impulse['microphones']))/FS*1000,impulse['microphones'][:,receiver],
                                                      color=COLORS[int(lossy)],label="Lossy" if lossy else "Lossless",lw=.6,rasterized=True)
                        ir_axes[source,receiver].set(title=f"Source R{source+1} → receiver R{receiver+1}",xlabel="Time (ms)",ylabel="Amplitude")
                print(key,result['metrics'],flush=True)
    ir_axes[0,-1].legend(ncol=2)
    for figure,rooms in [(two_figure,2),(three_figure,3)]:
        figure.legend(handles=trajectory_handles(rooms),loc="outside upper center",ncol=rooms+2)
    ir_figure.savefig(args.output/'joint-two-room-irs.pdf',dpi=300)
    two_figure.savefig(args.output/'joint-two-room-energy.pdf')
    three_figure.savefig(args.output/'joint-three-room-energy.pdf')
    plt.close(ir_figure);plt.close(two_figure);plt.close(three_figure)
    coupling_figure,axes=plt.subplots(2,3,figsize=(7.2,4.8),layout="constrained")
    for row,name in enumerate(['chain','complete']):
        model=models[name];h=FS*model['lines']*model['volumes'][0]/model['totals'][0]
        actual=h*model['phi']**2;np.fill_diagonal(actual,0)
        target=343*model['apertures']/4;residual=np.abs(actual-target)/target.max()
        manifest['models'][name]['conductance_max_normalized_error']=float(residual.max())
        for col,(matrix,title) in enumerate([(actual,'GFDN conductance'),(target,'SA conductance'),(100*residual,'Mismatch / max SA (%)')]):
            ax=axes[row,col];im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=12 if col==2 else 190)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>(6 if col==2 else 95) else 'black')
            ax.set(title=f'{name.title()}\n{title}',xlabel='Sending room',ylabel='Receiving room',xticks=range(3),yticks=range(3))
            ax.title.set_fontsize(8)
            ax.set_xticklabels(['R1','R2','R3']);ax.set_yticklabels(['R1','R2','R3'])
    coupling_figure.savefig(args.output/'joint-coupling-residuals.pdf');plt.close(coupling_figure)
    distribution_figure,axes=plt.subplots(3,3,figsize=(7.2,6.4),layout='constrained')
    settings=[(32,10000,False),(16,10000,False),(8,10000,False),
              (32,8000,False),(32,4000,False),(32,2000,False),
              (8,1000,True),(8,2000,True),(16,1000,True)]
    manifest['mixing_panels']=[]
    for ax,(lines,setting,varying) in zip(axes.flat,settings):
        model=room_model([160,80,120],[[0,2,0],[2,0,1.5],[0,1.5,0]],lines=lines,setting=setting,matrix_seed=1)
        result=render(model,varying=varying,initialization='impulse')
        means=result['endpoint_mean'];p=means/means.sum();positive=p>0
        divergence=float(np.sum(p[positive]*np.log(p[positive]*len(p))))
        ax.bar(np.arange(len(means)),means,color=COLORS[0],width=.8)
        ax.axhline(1/model['delays'].sum(),color='black',ls='--',lw=.7)
        ax.set(title=f"K={lines}; {'modulated' if varying else 'fixed'}\nR1 setting={setting/FS*1000:.1f} ms\nKL={divergence:.2e}",xlabel='Delay-line index',ylabel='Mean endpoint energy')
        ax.title.set_fontsize(8)
        manifest['mixing_panels'].append(dict(lines=lines,setting_samples=setting,modulated=varying,KL=divergence,
                                              delays=model['delays'].tolist(),initialization='impulse',late_fraction=.15))
    distribution_figure.savefig(args.output/'joint-energy-distributions.pdf');plt.close(distribution_figure)
    # One paired fixed/modulated free-decay comparison isolates feedback mixing.
    manifest['noise_fixed_modulated']={}
    for name in ['chain','complete']:
        result=ensemble(models[name],varying=False)
        manifest['noise_fixed_modulated'][name]=dict(fixed=result['metrics'],modulated=manifest['results'][f'{name}_source1_lossless']['noise'])
    table=[r'\begin{table*}[t]',r'\centering',r'\caption{Stored-energy errors against SA in the three-room examples. Noise rows evaluate the ensemble mean of 16 unit-energy diffuse starts; impulse rows use one unit-energy input impulse. All values are percentage points of initial energy. The 200-ms boundary is a reporting interval, not an estimated mixing time.}',r'\label{tab:joint-errors}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Case & Initialisation & Full MAE & 0--200 ms MAE & Later MAE & Maximum error \\',r'\midrule']
    for name in ['chain','complete']:
        for lossy in [False,True]:
            row=manifest['results'][f"{name}_source1_{'lossy' if lossy else 'lossless'}"]
            for initial in ['noise','impulse']:
                values=row[initial]
                table.append(f"{name.title()}, {'lossy' if lossy else 'lossless'} & {initial.title()} & "+' & '.join(f"{values[k]:.3f}" for k in ['full_mae_pp','early_mae_pp','later_mae_pp','maximum_error_pp'])+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (args.output/'joint-evaluation.tex').write_text('\n'.join(table)+'\n')
    manifest['file_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob('joint-*')) if p.suffix in ['.pdf','.tex']}
    (args.output/'joint-figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Wrote five figures, evaluation table, and provenance manifest.',flush=True)


if __name__=='__main__':
    main()
