"""Export the annotated manuscript's two-room comparisons.

Run from the repository root after the primary paper export:
    PYTHONPATH=src python examples/coupled_rooms/revise_two_room_figures.py --output figures

Only the two-room IR and energy figures are generated. SA, diffuse-start
fixed/TV feedback, and an impulse-start TV response share the same geometry.
All metrics use the full audio-rate record; the energy plots show 0--2 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from export_paper_figures import (
    COLORS, ENSEMBLE_SEEDS, FS, energy_errors, ensemble, render, room_model,
)


def plot_comparison(ax, cases, lossy, source):
    styles = {"fixed": dict(ls="-.", lw=1.),
              "impulse": dict(ls=":", lw=1., marker="^", ms=3., markevery=200),
              "varying": dict(ls="-", lw=1.4),
              "reference": dict(ls="--", lw=.9, marker="o", ms=3.,
                                markerfacecolor="white", markevery=200)}
    for method in ["fixed", "impulse", "varying", "reference"]:
        response = cases["varying"] if method == "reference" else cases[method]
        values = response["reference"] if method == "reference" else response["energy"]
        indices = np.arange(0, min(len(values), 2*FS+1), 48)
        time = indices/FS
        for room in range(2):
            curve = values[indices, room]
            if lossy:
                curve = 10*np.log10(np.maximum(curve, 1e-12))
            ax.plot(time, curve, color=COLORS[room], **styles[method])
            if method == "varying" and not lossy:
                se = response["se"][indices, room]
                ax.fill_between(time, curve-se, curve+se, color=COLORS[room], alpha=.12)
    ax.set(title=f"Initially R{source+1}; {'lossy' if lossy else 'lossless'}",
           xlabel="Time (s)", ylabel="Stored energy (dB)" if lossy else "Stored energy",
           xlim=(0, 2), ylim=(-85, 2) if lossy else (-.02, 1.02),
           xticks=[0, .5, 1, 1.5, 2])
    ax.grid(alpha=.15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":9, "axes.labelsize":9, "legend.fontsize":8,
                         "pdf.fonttype":42})
    model = room_model([160,80], [[0,2],[2,0]], setting=1000, matrix_seed=12343)
    ir_figure, ir_axes = plt.subplots(2,2,figsize=(7.2,3.6),layout="constrained")
    energy_figure, energy_axes = plt.subplots(2,2,figsize=(7.2,4.5),layout="constrained")
    manifest = dict(protocol="Paired fixed/TV diffuse starts and TV impulse starts; fixed geometry and H",
                    base_code_revision=args.code_revision, sample_rate=FS, render_duration_s=5.,
                    energy_plot_interval_s=[0,2], curve_export_spacing_s=.001,
                    noise_seeds=ENSEMBLE_SEEDS, matrix_seed=12343,
                    noise_normalization="Unit energy in source-room delay cells; spherical Gaussian draw",
                    impulse_normalization="Unit-norm input weights across the selected room; unit impulse",
                    metrics="Audio-rate stored-energy errors; same SA reference and 60-dB lossy cutoff as the primary export",
                    delays=model["delays"].tolist(), totals=model["totals"].tolist(), results={})
    for source in range(2):
        for lossy in [False,True]:
            key=f"two_source{source+1}_{'lossy' if lossy else 'lossless'}"
            varying=ensemble(model,source,lossy,varying=True)
            fixed=ensemble(model,source,lossy,varying=False)
            impulse=render(model,source,lossy,varying=True,initialization="impulse")
            cases=dict(varying=varying,fixed=fixed,impulse=impulse)
            for name,response in cases.items():
                values=response["energy"]
                assert np.isfinite(values).all() and values.min() >= -1e-12
                np.testing.assert_allclose(values[0],np.eye(2)[source],atol=1e-12)
                if not lossy:
                    np.testing.assert_allclose(values.sum(axis=1),1.,atol=1e-10)
            np.testing.assert_allclose(varying["reference"],fixed["reference"],atol=1e-12)
            plot_comparison(energy_axes[source,int(lossy)],cases,lossy,source)
            manifest["results"][key]=dict(varying_noise=varying["metrics"],fixed_noise=fixed["metrics"],
                varying_impulse=energy_errors(impulse["energy"],impulse["reference"]))
            curves={}
            for name,response in cases.items():
                curves[name]=response["energy"][::48]
            curves.update(reference=varying["reference"][::48],varying_se=varying["se"][::48],
                          fixed_se=fixed["se"][::48])
            np.savez_compressed(args.output/f"{key}-comparison-curves.npz",**curves)
            for receiver in range(2):
                ax=ir_axes[source,receiver]
                ax.plot(np.arange(len(impulse["microphones"]))/FS*1000,
                        impulse["microphones"][:,receiver],color=COLORS[int(lossy)],
                        lw=.6,rasterized=True)
                ax.set(title=f"Source R{source+1} → receiver R{receiver+1}",
                       xlabel="Time (ms)",ylabel="Amplitude",xlim=(0,500))
            print(key,manifest["results"][key],flush=True)
    handles=[Line2D([],[],color=COLORS[i],label=f"Room {i+1}") for i in range(2)]
    handles += [Line2D([],[],color="black",ls="-",label="Noise, TV"),
                Line2D([],[],color="black",ls="-.",label="Noise, fixed"),
                Line2D([],[],color="black",ls=":",marker="^",ms=3,label="Impulse, TV"),
                Line2D([],[],color="black",ls="--",marker="o",ms=3,
                       markerfacecolor="white",label="SA")]
    energy_figure.legend(handles=handles,loc="outside upper center",ncol=3)
    ir_figure.legend(handles=[Line2D([],[],color=COLORS[i],label=label)
                             for i,label in enumerate(["Lossless","Lossy"])],
                     loc="outside upper center",ncol=2)
    for figure,name in [(ir_figure,"joint-two-room-irs.pdf"),
                        (energy_figure,"joint-two-room-energy.pdf")]:
        figure.savefig(args.output/name,dpi=300)
        plt.close(figure)
    root=Path(__file__).resolve().parents[2]
    paths=sorted(list((root/"src/pyFDN").rglob("*.py"))+
                 [Path(__file__),Path(__file__).with_name("export_paper_figures.py")])
    manifest["source_sha256"]={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest["file_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.suffix in [".pdf",".npz"]}
    (args.output/"joint-two-room-comparison-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Exported the two figures and comparison provenance; conservation checks passed.",flush=True)


if __name__=="__main__":
    main()
