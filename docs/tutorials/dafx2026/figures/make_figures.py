"""Generate the static figures used by the DAFx 2026 tutorial slides.

The slides (``../slides.qmd``) contain no executed code: every figure is an SVG
produced here and checked in under ``out/``. That keeps the deck renderable
without a Python environment (important once it is frozen, see ``../README.md``)
while still making every plot reproducible from the current ``pyFDN``.

Run from anywhere::

    python docs/tutorials/dafx2026/figures/make_figures.py          # all figures
    python docs/tutorials/dafx2026/figures/make_figures.py poles ir  # a subset

Each figure is a function named ``fig_<name>`` returning a Matplotlib figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

import pyFDN  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

# Palette lifted from the pyFDN logo (docs/logo/logo_pyFDN_3.png).
NAVY = "#17365a"
BLUE = "#1f6fb4"
GREEN = "#4caf50"
YELLOW = "#fbc02d"
ORANGE = "#f57c20"
RED = "#ef3e36"
GREY = "#8a97a5"
CYCLE = [BLUE, ORANGE, GREEN, RED, YELLOW, NAVY]

FS = 48_000


def _style() -> None:
    """Slide-friendly Matplotlib defaults: large text, light grid, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.edgecolor": GREY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GREY,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "text.color": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": plt.cycler(color=CYCLE),
        }
    )


def _ir(build: pyFDN.FDNBuild, seconds: float = 3.0) -> np.ndarray:
    """Flat impulse response of a build, without the FLAMO/torch detour."""
    return np.asarray(pyFDN.build_to_impz(build, int(seconds * build.fs))).ravel()


def _flatness(magnitude: np.ndarray) -> float:
    """Spectral flatness (geometric/arithmetic mean of power, DC excluded); 1.0 is flat."""
    power = np.abs(magnitude).ravel()[1:] ** 2
    power = power[power > 0]
    if power.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


# --------------------------------------------------------------------------- #
# 1. The block diagram: what an FDN *is*                                      #
# --------------------------------------------------------------------------- #
def fig_diagram() -> plt.Figure:
    """Canonical FDN signal flow: input gains, delay lines, feedback matrix, output gains.

    Thick lines carry all ``N`` delay-line channels at once (marked with a slash);
    thin lines are the single-channel input, output and per-delay branches.
    """
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_axis_off()
    ax.set(xlim=(0, 10.4), ylim=(0, 5.4))

    y_fwd = 2.8  # forward (delay-line) path
    y_out = 4.7  # output tap running above the feedback matrix
    y_fb = 0.95  # feedback bus running underneath
    y_dir = 0.2  # direct path along the bottom
    delay_rows = [(4.05, "$z^{-m_1}$"), (3.25, "$z^{-m_2}$"), (1.75, "$z^{-m_N}$")]
    BUS = 3.4  # linewidth of an N-channel bus

    def box(x, y, w, h, label, color, fontsize=15):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.6,
                edgecolor=color,
                facecolor="white",
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=NAVY,
            zorder=4,
        )

    def wire(points, color=NAVY, lw=1.4, arrow=True):
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", zorder=2)
        if arrow:
            ax.add_patch(
                FancyArrowPatch(
                    points[-2],
                    points[-1],
                    arrowstyle="-|>",
                    mutation_scale=9 + 2.5 * lw,
                    linewidth=lw,
                    color=color,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=2,
                )
            )

    def sum_node(x, y):
        ax.add_patch(
            Circle((x, y), 0.17, facecolor="white", edgecolor=NAVY, lw=1.4, zorder=4)
        )
        ax.text(x, y, "+", ha="center", va="center", fontsize=13, color=NAVY, zorder=5)

    def dot(x, y, color=NAVY):
        ax.add_patch(Circle((x, y), 0.07, color=color, zorder=5))

    def bus_mark(x, y, label="$N$"):
        ax.plot([x - 0.1, x + 0.1], [y - 0.16, y + 0.16], color=NAVY, lw=1.1, zorder=6)
        ax.text(
            x + 0.06, y + 0.28, label, fontsize=11, color=NAVY, ha="center", zorder=6
        )

    # --- input, input gains, feedback summation ---
    ax.text(0.1, y_fwd, "$x(n)$", fontsize=16, va="center", color=NAVY)
    wire([(0.72, y_fwd), (1.35, y_fwd)])
    dot(1.05, y_fwd)
    box(1.35, y_fwd, 0.7, 0.6, r"$\mathbf{b}$", BLUE)
    wire([(2.05, y_fwd), (2.66, y_fwd)], color=BLUE, lw=BUS)
    bus_mark(2.36, y_fwd)
    sum_node(2.85, y_fwd)
    wire([(3.02, y_fwd), (3.4, y_fwd)], lw=BUS)

    # --- delay bank: fan out of the bus, one delay per channel, fan back in ---
    ax.plot(
        [3.4, 3.4], [delay_rows[-1][0], delay_rows[0][0]], color=NAVY, lw=1.4, zorder=2
    )
    ax.plot(
        [4.95, 4.95],
        [delay_rows[-1][0], delay_rows[0][0]],
        color=NAVY,
        lw=1.4,
        zorder=2,
    )
    for y, label in delay_rows:
        wire([(3.4, y), (3.62, y)])
        box(3.62, y, 1.1, 0.58, label, NAVY, fontsize=14)
        wire([(4.72, y), (4.95, y)])
    ax.text(4.17, 2.5, r"$\vdots$", ha="center", fontsize=17, color=NAVY)

    # --- delay outputs: tapped for the output, then fed to the feedback matrix ---
    wire([(4.95, y_fwd), (5.55, y_fwd)], lw=BUS)
    dot(5.25, y_fwd)
    box(5.55, y_fwd, 0.85, 2.7, r"$\mathbf{A}$", ORANGE, fontsize=19)
    ax.text(5.98, 1.62, "mixing", ha="center", fontsize=11, color=ORANGE)

    # --- feedback bus: right of A, down, back left into the summation node ---
    wire(
        [(6.4, y_fwd), (6.95, y_fwd), (6.95, y_fb), (2.85, y_fb), (2.85, y_fwd - 0.17)],
        color=ORANGE,
        lw=BUS,
    )

    # --- output tap: up from the delay outputs, through c, into the output sum ---
    wire([(5.25, y_fwd), (5.25, y_out), (7.1, y_out)], color=GREEN, lw=BUS)
    box(7.1, y_out, 0.8, 0.6, r"$\mathbf{c}^{\!\top}$", GREEN)
    wire([(7.9, y_out), (9.1, y_out), (9.1, y_fwd + 0.17)], color=GREEN)
    sum_node(9.1, y_fwd)
    wire([(9.27, y_fwd), (9.85, y_fwd)])
    ax.text(9.98, y_fwd, "$y(n)$", fontsize=16, va="center", color=NAVY)

    # --- direct path along the bottom ---
    wire(
        [(1.05, y_fwd), (1.05, y_dir), (9.1, y_dir), (9.1, y_fwd - 0.17)],
        color=GREY,
        lw=1.2,
    )
    ax.text(
        5.1, y_dir - 0.14, r"$d$   direct path", ha="center", fontsize=12, color=GREY
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. The translator hub: one representation in, five out                      #
# --------------------------------------------------------------------------- #
def fig_translators() -> plt.Figure:
    """Delay state space as the hub, with a named translator to each other form."""
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_axis_off()
    ax.set(xlim=(0, 10.5), ylim=(0, 5.4))

    targets = [
        (4.6, "impulse response", "dss_to_impz", BLUE),
        (3.65, "state space", "dss_to_ss", BLUE),
        (2.7, "matrix transfer function", "dss_to_tf", BLUE),
        (1.75, "poles & residues", "dss_to_pr", BLUE),
        (0.55, "FLAMO / torch", "dss_to_flamo", ORANGE),
    ]

    def rounded(x, y, w, h, edge, face="white", lw=1.6):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
                zorder=3,
            )
        )

    # The hub.
    rounded(0.3, 2.7, 2.5, 1.5, NAVY, face="#f2f5f8", lw=2.0)
    ax.text(
        1.55, 3.12, "delay state space", ha="center", fontsize=14, color=NAVY, zorder=4
    )
    ax.text(
        1.55,
        2.72,
        r"$\mathbf{m},\ \mathbf{A},\ \mathbf{b},\ \mathbf{c},\ d$",
        ha="center",
        fontsize=14,
        color=NAVY,
        zorder=4,
    )
    ax.text(
        1.55,
        2.3,
        "FDNBuild",
        ha="center",
        fontsize=12,
        color=ORANGE,
        zorder=4,
        style="italic",
    )

    for y, label, fn, color in targets:
        rounded(6.2, y, 3.9, 0.72, color)
        ax.text(
            8.15, y, label, ha="center", va="center", fontsize=13, color=NAVY, zorder=4
        )
        ax.add_patch(
            FancyArrowPatch(
                (2.85, 2.7),
                (6.15, y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color=color,
                connectionstyle="arc3,rad=0.0",
                shrinkA=2,
                shrinkB=2,
                zorder=2,
            )
        )
        # Label each arrow beside its own line: offset perpendicular to the arrow,
        # on an opaque backing so a near-parallel neighbour cannot run through it.
        dx, dy = 6.15 - 2.85, y - 2.7
        norm = np.hypot(dx, dy)
        nx, ny = -dy / norm, dx / norm  # left-hand normal: up-left / up-right
        lx, ly = 2.85 + 0.5 * dx + 0.33 * nx, 2.7 + 0.5 * dy + 0.33 * ny
        ax.text(
            lx,
            ly,
            fn,
            ha="center",
            va="center",
            fontsize=11.5,
            color=color,
            family="monospace",
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

    # FLAMO models translate back: to plain arrays, and to poles directly.
    ax.plot(
        [6.15, 1.55, 1.55],
        [0.55, 0.55, 1.85],
        color=ORANGE,
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.add_patch(
        FancyArrowPatch(
            (1.55, 1.6),
            (1.55, 1.94),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=ORANGE,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )
    ax.text(
        3.85,
        0.36,
        "extract_build",
        fontsize=11.5,
        color=ORANGE,
        family="monospace",
        ha="center",
        va="center",
        zorder=4,
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.1, 0.91),
            (7.1, 1.39),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=ORANGE,
            linestyle=(0, (4, 3)),
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )
    ax.text(
        7.25,
        1.15,
        "flamo_to_pr",
        fontsize=11.5,
        color=ORANGE,
        family="monospace",
        ha="left",
        va="center",
        zorder=4,
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. Impulse response and energy decay                                        #
# --------------------------------------------------------------------------- #
def fig_ir() -> plt.Figure:
    """IR (mu-law compressed) and energy decay curve of a vanilla 8-delay FDN."""
    build = pyFDN.fdn_build_gallery(8, fs=FS, rt=1.6, io_type="ones", rng=7)
    ir = _ir(build, 2.5)
    t = np.arange(ir.size) / FS

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(
        t, pyFDN.mulaw_encode(ir / np.max(np.abs(ir))), color=BLUE, linewidth=0.5
    )
    axes[0].set(
        xlabel="time [s]", ylabel="amplitude ($\\mu$-law)", title="Impulse response"
    )

    energy = pyFDN.sq_to_db(pyFDN.edc(ir))
    axes[1].plot(t, energy - energy[0], color=ORANGE)
    axes[1].axhline(-60, color=GREY, linestyle="--", linewidth=1.0)
    axes[1].text(0.05, -56, "$-60$ dB", fontsize=11, color=GREY)
    axes[1].set(
        xlabel="time [s]",
        ylabel="energy [dB]",
        ylim=(-80, 2),
        title="Energy decay curve  ($T_{60} = 1.6$ s)",
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. Feedback matrix gallery                                                  #
# --------------------------------------------------------------------------- #
def fig_matrices() -> plt.Figure:
    """Four lossless feedback matrices from ``pyFDN.fdn_matrix_gallery``."""
    types = ["orthogonal", "Hadamard", "circulant", "Householder"]
    fig, axes = plt.subplots(1, len(types), figsize=(11.5, 3.2))
    for ax, matrix_type in zip(axes, types, strict=True):
        A = pyFDN.fdn_matrix_gallery(8, matrix_type)
        ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=matrix_type, xticks=[], yticks=[])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GREY)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Poles: stability and modal density                                       #
# --------------------------------------------------------------------------- #
def fig_poles() -> plt.Figure:
    """Pole map of a small FDN and the resonance density that follows from the delays.

    A homogeneous FDN has all ``sum(delays)`` poles on a circle of radius
    ``g`` (the per-sample gain), spread near-uniformly in frequency — the two
    facts behind "decay time" and "modal density".
    """
    delays = np.array([113, 127, 131, 139])
    A = pyFDN.fdn_matrix_gallery(4, "orthogonal")
    rt = 1.0
    g = pyFDN.rt_to_gain_per_sample(rt, FS)
    A_lossy = np.diag(g ** delays.astype(float)) @ A
    B = np.ones((4, 1))
    C = np.ones((1, 4))
    D = np.zeros((1, 1))
    _residues, poles, _direct, is_conjugate, _meta = pyFDN.dss_to_pr(
        delays, A_lossy, B, C, D, mode="eig"
    )
    poles = np.asarray(poles).ravel()
    # dss_to_pr returns one pole per conjugate pair; mirror them for the z-plane.
    poles_full = np.concatenate(
        [poles, np.conj(poles[np.asarray(is_conjugate).ravel()])]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    theta = np.linspace(0, 2 * np.pi, 512)
    axes[0].plot(np.cos(theta), np.sin(theta), color=GREY, linewidth=1.0)
    axes[0].scatter(
        poles_full.real, poles_full.imag, s=9, color=BLUE, alpha=0.7, linewidths=0
    )
    axes[0].set(
        xlabel=r"$\mathrm{Re}\,z$",
        ylabel=r"$\mathrm{Im}\,z$",
        title=f"$\\sum m_i = {int(delays.sum())}$ poles, radius $|p| = g$",
        aspect="equal",
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
    )
    axes[0].annotate(
        f"$g = {g:.6f}$\n$T_{{60}} = {rt:.1f}$ s",
        xy=(0.0, 0.0),
        ha="center",
        va="center",
        fontsize=12,
        color=NAVY,
    )

    # Each pole is one resonance: zoom the magnitude response far enough in to
    # resolve them individually, and mark where the poles sit.
    ir = np.asarray(pyFDN.dss_to_impz(2**16, delays, A_lossy, B, C, D)).ravel()
    spectrum = pyFDN.lin_to_db(np.abs(np.fft.rfft(ir)))
    bin_freqs = np.fft.rfftfreq(ir.size, 1.0 / FS)
    lo, hi = 200.0, 500.0
    window = (bin_freqs >= lo) & (bin_freqs <= hi)
    # One resonance per conjugate *pair*, so count the reduced pole set.
    pole_freqs = np.abs(np.angle(poles)) / (2 * np.pi) * FS
    in_window = np.unique(pole_freqs[(pole_freqs >= lo) & (pole_freqs <= hi)])

    axes[1].plot(
        bin_freqs[window], spectrum[window] - spectrum[window].max(), color=BLUE
    )
    axes[1].eventplot(
        in_window,
        lineoffsets=-28,
        linelengths=5,
        colors=ORANGE,
        linewidths=1.4,
    )
    axes[1].set(
        xlabel="frequency [Hz]",
        ylabel="magnitude [dB]",
        xlim=(lo, hi),
        ylim=(-32, 3),
        title=f"One resonance per pole ({in_window.size} between {lo:.0f} and {hi:.0f} Hz)",
    )
    axes[1].text(
        lo + 8, -24, "pole frequencies", fontsize=11, color=ORANGE, va="bottom"
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Frequency-dependent absorption                                           #
# --------------------------------------------------------------------------- #
def fig_absorption() -> plt.Figure:
    """Target vs. measured reverberation time for a frequency-dependent FDN."""
    build = pyFDN.fdn_build_gallery(
        8, fs=FS, rt=2.2, rt_nyquist=0.45, rt_crossover=1500.0, io_type="ones", rng=3
    )
    model = pyFDN.dss_to_flamo(
        build.A,
        build.B,
        build.C,
        build.D,
        build.delays,
        build.fs,
        nfft=2**17,
        sos_filter=build.filters,
    )
    ir = np.asarray(pyFDN.flamo_time_response(model)).ravel()
    rt, bands = pyFDN.estimate_rt_bands(ir, FS, fc=1000.0, start=-4.0, n=8)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.semilogx(bands, rt, "o-", color=BLUE, label="measured (estimate_rt_bands)")
    ax.axhline(2.2, color=GREY, linestyle="--", linewidth=1.0)
    ax.axhline(0.45, color=GREY, linestyle="--", linewidth=1.0)
    ax.text(bands[0], 2.3, "target DC = 2.2 s", fontsize=11, color=GREY)
    ax.text(bands[0], 0.55, "target Nyquist = 0.45 s", fontsize=11, color=GREY)
    ax.set(xlabel="octave band centre [Hz]", ylabel="$T_{60}$ [s]", ylim=(0, 2.8))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. Colorless optimisation                                                   #
# --------------------------------------------------------------------------- #
def fig_colorless() -> plt.Figure:
    """Magnitude response of a lossless FDN before and after colorless training."""
    nfft = 2**12
    delays = pyFDN.sample_delay_lengths(
        8, (800, 3200), distribution="geometric", coprime=True, rng=1
    )
    model = pyFDN.build_fdn(
        delays=delays, rt=None, nfft=nfft, output="magnitude", device="cpu", rng=1
    )
    mag_init = np.abs(np.asarray(pyFDN.flamo_freq_response(model, fs=FS)).squeeze())
    log = pyFDN.train_fdn(
        model,
        "colorless",
        optimizer="lbfgs",
        max_steps=400,
        lr=1e-2,
        device="cpu",
        rng=1,
    )
    mag_opt = np.abs(np.asarray(pyFDN.flamo_freq_response(model, fs=FS)).squeeze())
    freqs = np.fft.rfftfreq(nfft, 1.0 / FS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    # Zoomed to a range where individual resonances stay resolvable on a slide.
    axes[0].plot(
        freqs,
        pyFDN.lin_to_db(mag_init),
        color=GREY,
        linewidth=1.2,
        label=f"random init (flatness {_flatness(mag_init):.2f})",
    )
    axes[0].plot(
        freqs,
        pyFDN.lin_to_db(mag_opt),
        color=BLUE,
        linewidth=1.2,
        label=f"colorless (flatness {_flatness(mag_opt):.2f})",
    )
    axes[0].set(
        xlabel="frequency [Hz]",
        ylabel="magnitude [dB]",
        xlim=(100, 1000),
        title="Magnitude response",
    )
    axes[0].legend(loc="upper right", fontsize=11)

    axes[1].semilogy(log.train_loss, "-o", ms=3, color=ORANGE)
    axes[1].set(xlabel="epoch", ylabel="loss", title="Training loss")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 7. Echo density / mixing time                                               #
# --------------------------------------------------------------------------- #
def fig_echo_density() -> plt.Figure:
    """Normalised echo density of two FDNs: a scalar matrix vs. a mixing one."""
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for matrix_type, color in (("orthogonal", BLUE), ("permutation", ORANGE)):
        build = pyFDN.fdn_build_gallery(8, fs=FS, rt=1.6, io_type="ones", rng=11)
        A = pyFDN.fdn_matrix_gallery(8, matrix_type)
        g = pyFDN.rt_to_gain_per_sample(1.6, FS)
        lossy = pyFDN.FDNBuild(
            A=np.diag(g ** build.delays.astype(float)) @ A,
            B=build.B,
            C=build.C,
            D=build.D,
            delays=build.delays,
            fs=build.fs,
            filters=None,
            post_eq=None,
        )
        ir = _ir(lossy, 0.6)
        # echo_density interpolates back to one value per sample; mixing time in ms,
        # and 0.0 when the density never reaches the threshold (a permutation never mixes).
        mixing_time, density = pyFDN.echo_density(ir, n=1024, fs=FS, hop=256)
        t_ms = np.arange(density.size) / FS * 1e3
        mixing = f"mixes at {mixing_time:.0f} ms" if mixing_time > 0 else "never mixes"
        ax.plot(t_ms, density, color=color, label=f"{matrix_type} — {mixing}")
    ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1.0)
    ax.set(
        xlabel="time [ms]",
        ylabel="normalised echo density",
        xlim=(0, 300),
        ylim=(0, 1.4),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


FIGURES = {
    name.removeprefix("fig_"): value
    for name, value in sorted(globals().items())
    if name.startswith("fig_")
}


def main(argv: list[str]) -> int:
    _style()
    requested = argv or list(FIGURES)
    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        print(
            f"unknown figure(s): {', '.join(unknown)}\navailable: {', '.join(FIGURES)}"
        )
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in requested:
        fig = FIGURES[name]()
        path = OUT_DIR / f"{name}.svg"
        fig.savefig(path)
        plt.close(fig)
        print(
            f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
