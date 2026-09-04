# gallery_category: Feedback Matrices
# gallery_description: Build lossless feedback matrices from Kronecker products of 2x2 kernels, then use single kernel angles to control stereo cross-coupling and time-varying modulation.

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kronecker feedback matrices

    A feedback matrix has to be orthogonal to keep an FDN lossless, and that is normally the end of the story: pick Hadamard, pick a random orthogonal matrix, and leave it alone. Nudging one entry breaks orthogonality, so there is nothing to turn.

    The Kronecker construction gives the matrix knobs. An \(N \times N\) matrix with \(N = 2^M\) is written as a Kronecker product of \(M\) two-by-two kernels, each carrying one angle:

    $$\Psi_M = K_M \otimes K_{M-1} \otimes \dots \otimes K_1, \qquad K_i = \mathrm{Rot}(\theta_i) \ \text{or}\ \widehat{\mathrm{Ref}}(\theta_i).$$

    Kronecker products of orthogonal matrices are orthogonal, so **every** setting of every angle is lossless — the angles can be swept, modulated at audio rate, or automated, and the FDN never stops being energy-preserving.

    What makes the angles useful rather than merely safe is that each one addresses one bit of the delay-line index. \(\theta_M\) mixes the two contiguous halves of the network, \(\theta_1\) mixes the even- and odd-indexed lines, and the levels in between sit at intermediate granularities. Setting one angle to zero cuts the network along that partition; sweeping it re-couples it continuously.

    This notebook reproduces the configurations from the paper's companion site: the construction itself, the fast transform that applies it, the **stereo cross-coupling** sweep on \(\theta_M\), and **matrix modulation** on \(\theta_{M-1}\). The selective-freeze configuration is deliberately left out.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    Reference: *{pyFDN.paper_link("Coppola2026FastParametric")}*. <br/>
    Companion site: <https://andrea-coppola-arturia.github.io/fastparametricmatrices/>
    """)
    return


@app.cell
def _():
    import time

    import numpy as np
    import plotly.graph_objects as go
    from scipy.linalg import hadamard
    from scipy.signal import correlate, square

    import pyFDN
    from pyFDN import td

    fs = 48000

    # Length of every rendered listening example. Three stereo players of this
    # length is about 6 MB of cell output, inside marimo's 8 MB default cap
    # (``output_max_bytes``); six 6-second players would be over twice that.
    AUDIO_SECONDS = 4.0
    return AUDIO_SECONDS, correlate, fs, go, hadamard, np, pyFDN, square, td, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The construction

    Both kernel families are one-parameter families of 2x2 orthogonal matrices:

    $$\mathrm{Rot}(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix},
    \qquad
    \widehat{\mathrm{Ref}}(\theta) = \begin{bmatrix}\cos\theta & \sin\theta\\ \sin\theta & -\cos\theta\end{bmatrix}.$$

    `pyFDN.kronecker_matrix` takes the angles innermost first — `angles[0]` is \(\theta_1\), `angles[-1]` is \(\theta_M\) — and folds them up with the recursion \(\Psi_m = K_m \otimes \Psi_{m-1}\). `pyFDN.kronecker_angles` is the convenience that fills in a default for every kernel and lets you name the ones you want to move.
    """)
    return


@app.cell
def _(np, pyFDN):
    # Figure 5 of the paper: N = 8, rotation kernels, theta = (0, pi/4, pi/8).
    figure5 = pyFDN.kronecker_matrix([0.0, np.pi / 4, np.pi / 8], "rotation")

    print(f"orthogonal: {np.allclose(figure5 @ figure5.T, np.eye(8))}")
    print(np.round(figure5, 2))
    return (figure5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two partitions are visible at once in that matrix. \(\theta_1 = 0\) leaves the innermost kernel at the identity, so no even-indexed line ever feeds an odd-indexed one — the checkerboard of zeros. \(\theta_3 = \pi/8\) is small but non-zero, so the two halves are only weakly coupled: the corner blocks are the faint \(\pm 0.27\) entries against \(\pm 0.65\) inside each half.

    The two configurations are independent, which is the point of the parameterisation. A single matrix can be split even/odd *and* partially coupled across the stereo halves at the same time.
    """)
    return


@app.cell
def _(figure5, pyFDN):
    pyFDN.plot_matrix(
        figure5,
        title="Kronecker matrix Ψ₃ at θ = (0, π/4, π/8)"
        "<br><sup>θ₁ = 0 decouples even from odd; θ₃ = π/8 leaves the halves weakly coupled</sup>",
        block_boundaries=[4],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hadamard is one point in the family

    \(\widehat{\mathrm{Ref}}(\pi/4)\) is exactly the order-2 Hadamard matrix, so putting every kernel there recovers the standard normalised Hadamard matrix of order \(N\). The parametric family therefore contains the usual FDN mixing matrix and morphs continuously away from it — with `"rotation"` kernels at the same angle you get the same equal-magnitude mixing under a different sign pattern.
    """)
    return


@app.cell
def _(hadamard, np, pyFDN):
    _N = 16
    _reflection = pyFDN.kronecker_matrix(np.full(4, np.pi / 4), "reflection")
    _rotation = pyFDN.kronecker_matrix(pyFDN.kronecker_angles(_N), "rotation")

    print(
        "all reflection kernels at π/4 == Hadamard: "
        f"{np.allclose(_reflection, hadamard(_N) / np.sqrt(_N))}"
    )
    print(
        "all rotation kernels at π/4 mix with equal magnitude: "
        f"{np.allclose(np.abs(_rotation), 1 / np.sqrt(_N))}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Applying it in O(N log₂ N)

    The recursive structure is also a divide-and-conquer algorithm. `pyFDN.kronecker_transform` never forms the matrix: it runs \(\log_2 N\) butterfly passes over the channel vector, level \(i\) mixing the channel pairs that differ in bit \(i-1\) of their index. That is the same cost as the fast Walsh–Hadamard transform, but with \(M\) free parameters instead of none.

    It matters most when the angles move. A dense matrix product has to rebuild the whole \(N \times N\) matrix whenever an angle changes; here a moving angle only changes the two-by-two kernel of one level, so per-sample modulation costs no more than a static matrix.
    """)
    return


@app.cell
def _(mo, np, pyFDN, time):
    _rng = np.random.default_rng(0)
    _rows = [
        "| N | max abs. difference | dense (ms) | butterfly (ms) |",
        "|---|---|---|---|",
    ]
    for _M in (3, 4, 5, 6):
        _size = 2**_M
        _angles = _rng.uniform(-np.pi, np.pi, _M)
        _matrix = pyFDN.kronecker_matrix(_angles)
        _x = _rng.standard_normal((4096, _size))

        _t0 = time.perf_counter()
        _dense = _x @ _matrix.T
        _dense_time = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        _fast = pyFDN.kronecker_transform(_x, _angles)
        _fast_time = time.perf_counter() - _t0

        _rows.append(
            f"| {_size} | {np.abs(_dense - _fast).max():.1e} | "
            f"{_dense_time * 1e3:.2f} | {_fast_time * 1e3:.2f} |"
        )

    mo.output.replace(mo.md("\n".join(_rows)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two agree to machine precision. The timing here is NumPy on 4096-sample blocks, where BLAS makes the dense product very hard to beat at these sizes; the paper's C++ benchmark, which multiplies one sample at a time inside a feedback loop, is where the asymptotic win shows up (7.5x over a naive product at \(N = 32\), and 24x once an angle is updated every sample).

    `pyFDN.td` wraps both cases as operators: `td.KroneckerMatrix` for a fixed angle set, `td.TimeVaryingKroneckerMatrix` for modulated ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Stereo cross-coupling on θ_M

    Split the delay lines into two halves and give each half one stereo channel: the left input feeds lines 0–15 and is read back off them, the right input feeds lines 16–31. The outermost angle is then exactly the knob that decides how much the two sides hear of each other.

    Writing \(\Psi_M = \mathrm{Rot}(\theta_M) \otimes \Psi_{M-1}\) out in blocks,

    $$\Psi_M = \begin{bmatrix} \cos\theta_M\,\Psi_{M-1} & -\sin\theta_M\,\Psi_{M-1}\\ \sin\theta_M\,\Psi_{M-1} & \cos\theta_M\,\Psi_{M-1}\end{bmatrix},$$

    so at \(\theta_M = 0\) the off-diagonal blocks vanish and the two halves are separate FDNs, at \(\theta_M = \pi/4\) all four blocks have equal weight and the network is one unified, maximally diffusive FDN, and everything in between is a continuous, always-lossless morph. The site's percentages are positions along \([0, \pi/4]\).

    ### The network

    A 32-line FDN, coprime delays from 20 ms to 200 ms, \(T_{60} = 10\) s, no damping filters and no diffusion stages, so nothing but the matrix shapes what is heard. The sorted delays are dealt alternately into the two halves so each half spans the full 20–200 ms range — otherwise the contiguous split would give one channel all the short lines and the other all the long ones.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    N_stereo = 32
    T60_stereo = 10.0  # seconds

    _sorted_delays = pyFDN.sample_delay_lengths(
        N_stereo,
        (int(0.020 * fs), int(0.200 * fs)),
        coprime=True,
        sort=True,
        rng=2026,
    )
    # Deal alternately into the two halves so both span 20-200 ms.
    delays_stereo = _sorted_delays.reshape(-1, 2).T.reshape(-1)

    absorption_stereo = np.diag(
        pyFDN.rt_to_gain_per_sample(T60_stereo, fs) ** delays_stereo
    )

    # Left input drives the first half, right input the second (Eq. 19 and 20).
    _half = N_stereo // 2
    input_stereo = np.zeros((N_stereo, 2))
    input_stereo[:_half, 0] = 1 / np.sqrt(_half)
    input_stereo[_half:, 1] = 1 / np.sqrt(_half)
    output_stereo = input_stereo.T.copy()
    direct_stereo = np.zeros((2, 2))

    print(
        f"half A: {delays_stereo[:_half].min()}–{delays_stereo[:_half].max()} samples"
    )
    print(
        f"half B: {delays_stereo[_half:].min()}–{delays_stereo[_half:].max()} samples"
    )
    return (
        N_stereo,
        absorption_stereo,
        delays_stereo,
        direct_stereo,
        input_stereo,
        output_stereo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The coupling sweep

    Six settings, the six on the companion site: 0%, 5%, 10%, 25%, 50% and 100% of the way from \(\theta_M = 0\) to \(\theta_M = \pi/4\). Only that one angle changes; every other kernel stays at \(\pi/4\).
    """)
    return


@app.cell
def _(N_stereo, np, pyFDN):
    coupling_percent = [0, 5, 10, 25, 50, 100]

    coupling_angles = {
        percent: pyFDN.kronecker_angles(N_stereo, theta5=np.pi / 4 * percent / 100)
        for percent in coupling_percent
    }

    for _percent, _angles in coupling_angles.items():
        print(f"{_percent:4d}%   θ_M = {_angles[-1]:.4f} rad")
    return coupling_angles, coupling_percent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stereo impulse responses

    A unit impulse into the left input only, so any energy appearing on the right had to cross through the coupling blocks.
    """)
    return


@app.cell
def _(
    absorption_stereo,
    coupling_angles,
    delays_stereo,
    direct_stereo,
    fs,
    input_stereo,
    np,
    output_stereo,
    pyFDN,
):
    _ir_duration = 12.0  # seconds

    _impulse = np.zeros((int(_ir_duration * fs), 2))
    _impulse[0, 0] = 1.0

    stereo_ir = {
        percent: pyFDN.process_fdn(
            _impulse,
            delays_stereo,
            pyFDN.kronecker_matrix(angles) @ absorption_stereo,
            input_stereo,
            output_stereo,
            direct_stereo,
        )
        for percent, angles in coupling_angles.items()
    }
    return (stereo_ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the energy is

    The most direct reading of the sweep: the level of the right channel relative to the left, in 100 ms windows. At 0% the right channel is digital silence for ever. At 5% it climbs slowly out of nothing and is still 15 dB down after ten seconds; by 25% the two sides are within a few dB after a second or so; at 100% they are level almost immediately.
    """)
    return


@app.cell
def _(coupling_percent, fs, go, mo, np, stereo_ir):
    def channel_balance(ir, window=0.100, hop=0.050):
        """Right-minus-left level in dB, in sliding windows."""
        w, h = int(window * fs), int(hop * fs)
        starts = np.arange(0, len(ir) - w, h)
        energy_l = np.array([np.sum(ir[s : s + w, 0] ** 2) for s in starts])
        energy_r = np.array([np.sum(ir[s : s + w, 1] ** 2) for s in starts])
        times = (starts + w / 2) / fs
        return times, 10 * np.log10((energy_r + 1e-30) / (energy_l + 1e-30))

    _figure = go.Figure()
    for _percent in coupling_percent:
        _t, _balance = channel_balance(stereo_ir[_percent])
        _figure.add_scatter(
            x=_t, y=np.maximum(_balance, -80), mode="lines", name=f"{_percent}%"
        )
    _figure.update_layout(
        title="Right-channel level relative to left"
        "<br><sup>impulse into the left input only; floor clamped at −80 dB</sup>",
        xaxis_title="Time [s]",
        yaxis_title="R − L [dB]",
        height=420,
        legend_title="coupling",
    )
    mo.output.replace(_figure)
    return (channel_balance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interaural cross-correlation

    The site's own measure. IACC is the peak magnitude of the normalised cross-correlation between the two channels over a ±1 ms lag window, evaluated in sliding 100 ms windows: near 1 the channels are near-identical and the image is narrow and centred, near 0 they are decorrelated and the image is wide.

    At 0% the right channel is silent, the correlation is undefined, and the trace is simply absent — the same gap the site leaves in that figure. Every coupled setting behaves the way the site describes: whatever the coupling strength, the reverberant tail decorrelates within a fraction of a second and settles at a low steady-state value. Coupling controls *how much* energy reaches the other side, not how similar the two sides end up sounding — that is what the balance plot above measures and this one does not.
    """)
    return


@app.cell
def _(correlate, coupling_percent, fs, go, mo, np, stereo_ir):
    def iacc(ir, window=0.100, hop=0.050, max_lag=0.001):
        """Sliding-window IACC (ISO 3382-1) of a stereo impulse response."""
        w, h, lag = int(window * fs), int(hop * fs), int(max_lag * fs)
        starts = np.arange(0, len(ir) - w, h)
        values = np.full(len(starts), np.nan)
        for index, start in enumerate(starts):
            left, right = ir[start : start + w, 0], ir[start : start + w, 1]
            norm = np.sqrt(np.sum(left**2) * np.sum(right**2))
            if norm < 1e-20:  # a silent channel leaves the IACC undefined
                continue
            rho = correlate(left, right, mode="full")[w - 1 - lag : w + lag] / norm
            values[index] = np.max(np.abs(rho))
        return (starts + w / 2) / fs, values

    _figure = go.Figure()
    for _percent in coupling_percent:
        _t, _values = iacc(stereo_ir[_percent])
        if np.all(np.isnan(_values)):
            continue  # 0%: right channel silent throughout
        _figure.add_scatter(x=_t, y=_values, mode="lines", name=f"{_percent}%")
    _figure.update_layout(
        title="Sliding-window IACC"
        "<br><sup>100 ms window, 50 ms hop, ±1 ms lag; 0% omitted (right channel silent)</sup>",
        xaxis_title="Time [s]",
        yaxis_title="IACC",
        yaxis_range=[0, 1],
        height=420,
        legend_title="coupling",
    )
    mo.output.replace(_figure)
    return (iacc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listen

    A single piano-like note hard-panned left, rendered 100% wet through each setting. Headphones recommended. The note stays put at 0%, sweeps slowly across at 5–10%, and is spread across the panorama almost at once at 100%.
    """)
    return


@app.cell
def _(AUDIO_SECONDS, fs, mo, np, pyFDN):
    def piano_note(f0, duration, fs, seed=0):
        """A plucked-string-ish tone: decaying inharmonic partials plus a click."""
        rng = np.random.default_rng(seed)
        t = np.arange(int(duration * fs)) / fs
        note = np.zeros_like(t)
        for partial in range(1, 17):
            # Slight stiffness-induced inharmonicity, and faster decay up high.
            frequency = f0 * partial * np.sqrt(1 + 4e-4 * partial**2)
            if frequency > 0.45 * fs:
                break
            note += (
                np.exp(-t * (2.0 + 0.9 * partial))
                * np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
                / partial**1.3
            )
        hammer = rng.standard_normal(len(t)) * np.exp(-t * 900) * 0.4
        note = note + hammer
        return note / np.abs(note).max() * 0.9

    dry_note = piano_note(220.0, 1.2, fs)

    # Hard-panned left, then silence for the tail. Kept to AUDIO_SECONDS so a
    # cell of three players stays inside marimo's 8 MB output limit.
    stereo_note = np.zeros((int(AUDIO_SECONDS * fs), 2))
    stereo_note[: len(dry_note), 0] = dry_note

    mo.output.replace(
        pyFDN.labeled_audio(
            "<b>Dry source: piano note, hard-panned left</b>", stereo_note, fs=fs
        )
    )
    return (stereo_note,)


@app.cell
def _(
    absorption_stereo,
    coupling_angles,
    delays_stereo,
    direct_stereo,
    input_stereo,
    output_stereo,
    pyFDN,
    stereo_note,
):
    stereo_wet = {
        percent: pyFDN.process_fdn(
            stereo_note,
            delays_stereo,
            pyFDN.kronecker_matrix(angles) @ absorption_stereo,
            input_stereo,
            output_stereo,
            direct_stereo,
        )
        for percent, angles in coupling_angles.items()
    }
    return (stereo_wet,)


@app.cell
def _(coupling_angles, fs, mo, pyFDN, stereo_wet):
    def coupling_players(percentages):
        """Stack the players for a few coupling settings.

        Split across two cells: six stereo players in a single output would
        exceed marimo's 8 MB per-cell limit (``output_max_bytes``).
        """
        return mo.vstack(
            [
                pyFDN.labeled_audio(
                    f"<b>{percent}% coupling</b> — θ_M = "
                    f"{coupling_angles[percent][-1]:.4f} rad",
                    stereo_wet[percent],
                    fs=fs,
                )
                for percent in percentages
            ]
        )

    return (coupling_players,)


@app.cell
def _(coupling_percent, coupling_players, mo):
    mo.output.replace(coupling_players(coupling_percent[:3]))
    return


@app.cell
def _(coupling_percent, coupling_players, mo):
    mo.output.replace(coupling_players(coupling_percent[3:]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Matrix modulation on θ_{M−1}

    A static FDN has fixed modes. Sustain a harmonically rich tone through one and the same handful of resonances is excited for as long as the tone lasts, which is what "metallic ringing" describes. Moving a kernel angle moves the poles, so no mode is driven long enough to stand out.

    What distinguishes this from the familiar cure — modulating the delay lengths — is that the routing coefficients move while the delays stay put. Delay modulation retunes the whole network and brings chorusing with it; matrix modulation only redistributes energy between lines, so it breaks up resonances with far less pitch movement.

    The site isolates the effect on a deliberately sparse network: 16 lines with \(\theta_M = 0\), which splits it into two independent 8×8 sub-networks, one per stereo side. Fewer modes per side makes the individual resonances easy to hear. With \(M = \log_2 16 = 4\), the modulated angle \(\theta_{M-1}\) is \(\theta_3\), and it follows

    $$\theta_3(t) = \frac{\pi}{4} + \text{depth}\cdot\sin(2\pi\,\text{rate}\,t).$$
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    N_mod = 16
    T60_mod = 3.0  # seconds

    _sorted_delays = pyFDN.sample_delay_lengths(
        N_mod, (int(0.020 * fs), int(0.200 * fs)), coprime=True, sort=True, rng=7
    )
    delays_mod = _sorted_delays.reshape(-1, 2).T.reshape(-1)

    # theta_M = 0: two independent 8x8 sub-networks, one per stereo side.
    modulation_base = pyFDN.kronecker_angles(N_mod, theta4=0.0)

    absorption_mod = np.diag(pyFDN.rt_to_gain_per_sample(T60_mod, fs) ** delays_mod)

    _half = N_mod // 2
    input_mod = np.zeros((N_mod, 2))
    input_mod[:_half, 0] = 1 / np.sqrt(_half)
    input_mod[_half:, 1] = 1 / np.sqrt(_half)
    output_mod = input_mod.T.copy()
    direct_mod = np.zeros((2, 2))

    print(f"base angles θ₁..θ₄: {np.round(modulation_base, 4)}")
    return (
        N_mod,
        absorption_mod,
        delays_mod,
        direct_mod,
        input_mod,
        modulation_base,
        output_mod,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The dry source

    Two sustained chords on a square-wave oscillator. The harmonically rich tone excites the network's modes hard, which is what surfaces the metallic ringing that the modulation is meant to remove.
    """)
    return


@app.cell
def _(AUDIO_SECONDS, fs, mo, np, pyFDN, square):
    def square_chord(freqs, start, stop, t, fade=0.01):
        """A square-wave chord gated between ``start`` and ``stop`` seconds."""
        gate = ((t >= start) & (t < stop)).astype(float)
        ramp = np.clip((t - start) / fade, 0, 1) * np.clip((stop - t) / fade, 0, 1)
        tone = sum(square(2 * np.pi * f * t) for f in freqs) / len(freqs)
        return 0.35 * tone * gate * ramp

    _t = np.arange(int(AUDIO_SECONDS * fs)) / fs
    # Two shortish chords, so that the last 1.4 s is exposed tail -- which is
    # where the ringing the modulation removes is easiest to hear.
    _chords = square_chord([220.0, 277.18, 329.63], 0.1, 1.3, _t) + square_chord(
        [196.0, 246.94, 293.66], 1.5, 2.6, _t
    )
    dry_chords = np.stack([_chords, _chords], axis=1)

    mo.output.replace(
        pyFDN.labeled_audio(
            "<b>Dry source: two chords (square-wave oscillator)</b>", dry_chords, fs=fs
        )
    )
    return (dry_chords,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Three modulation regimes

    The three settings from the site: off, the recommended subtle setting (0.2 Hz, depth 0.2π), and the aggressive upper bound (2 Hz, depth π — the full angular range).

    `td.TimeVaryingKroneckerMatrix` takes the base angles plus a rate and depth per kernel; zeros leave a kernel alone, which is how a single level is singled out. With the FDN's static feedback matrix set to the absorption alone, the operator on `post_matrix` *is* the feedback matrix.
    """)
    return


@app.cell
def _(N_mod, fs, modulation_base, np, td):
    modulation_settings = {
        "off": (0.0, 0.0),
        "subtle (0.2 Hz, 0.2π)": (0.2, 0.2 * np.pi),
        "aggressive (2 Hz, π)": (2.0, np.pi),
    }

    def modulated_matrix(rate, depth):
        """Modulate theta_{M-1} only, leaving every other kernel fixed."""
        modulated = np.arange(int(np.log2(N_mod))) == int(np.log2(N_mod)) - 2
        return td.TimeVaryingKroneckerMatrix(
            modulation_base,
            fs,
            rate=np.where(modulated, rate, 0.0),
            depth=np.where(modulated, depth, 0.0),
        )

    modulation_names = list(modulation_settings)
    return modulated_matrix, modulation_names, modulation_settings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The angle trajectories, one second of each. The subtle setting stays within a fifth of π of centre; the aggressive one sweeps the whole range twice a second, which is audible as an effect in its own right rather than as diffusion.
    """)
    return


@app.cell
def _(fs, go, mo, modulated_matrix, modulation_settings, np):
    _figure = go.Figure()
    _n = np.arange(0, 3 * fs, 64)
    for _name, (_rate, _depth) in modulation_settings.items():
        _angles = modulated_matrix(_rate, _depth).angles_at(_n)
        _figure.add_scatter(x=_n / fs, y=_angles[:, 2], mode="lines", name=_name)
    _figure.update_layout(
        title="Modulated kernel angle θ₃ = θ_{M−1}",
        xaxis_title="Time [s]",
        yaxis_title="θ₃ [rad]",
        height=340,
    )
    mo.output.replace(_figure)
    return


@app.cell
def _(
    absorption_mod,
    delays_mod,
    direct_mod,
    dry_chords,
    input_mod,
    modulated_matrix,
    modulation_settings,
    output_mod,
    pyFDN,
):
    modulated_wet = {
        name: pyFDN.process_fdn(
            dry_chords,
            delays_mod,
            absorption_mod,  # A carries the absorption only ...
            input_mod,
            output_mod,
            direct_mod,
            post_matrix=modulated_matrix(rate, depth),  # ... the operator is the matrix
        )
        for name, (rate, depth) in modulation_settings.items()
    }
    return (modulated_wet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the three spectrograms downwards. The static network leaves sharp horizontal ridges where individual modes sustain through the chord tails; subtle modulation smears them; the aggressive setting broadens them into wavering bands.
    """)
    return


@app.cell
def _(fs, mo, modulated_wet, modulation_names, pyFDN):
    mo.vstack(
        [
            pyFDN.plot_spectrogram(
                modulated_wet[name][:, 0],
                fs,
                nperseg=2048 * 8,
                noverlap=2048 * 6,
                title=f"Modulation {name} — spectrogram",
                colorscale="Magma",
                height=330,
            )
            for name in modulation_names
        ]
    )
    return


@app.cell
def _(fs, mo, modulated_wet, modulation_names, pyFDN):
    mo.output.replace(
        mo.vstack(
            [
                pyFDN.labeled_audio(
                    f"<b>Modulation {name}</b>", modulated_wet[name], fs=fs
                )
                for name in modulation_names
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why it works: the poles move

    Figure 6 of the paper measures the effect on the poles directly. Freeze the modulated matrix at a sequence of instants across one modulation period, take the poles of the resulting FDN each time, and histogram their frequencies. Without modulation every snapshot has the same poles, so the histogram is a set of spikes; with modulation the poles wander and the histogram flattens. The standard deviation of the bin counts is the summary number — lower means a more uniform pole density, which is what "less coloration" means quantitatively.

    The paper's network: 8 lines at `m = [53, 61, 71, 79, 89, 101, 113, 127]`, rotation kernels, and the middle angle \(\theta_2\) modulated with amplitude π by a 1 Hz triangle wave.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    figure6_delays = np.array([53, 61, 71, 79, 89, 101, 113, 127])
    N_figure6 = len(figure6_delays)
    num_snapshots = 21

    def pole_frequencies(angles):
        """Positive pole frequencies in Hz of the lossless FDN at these angles."""
        _, poles, _, _, _ = pyFDN.dss_to_pr(
            figure6_delays,
            pyFDN.kronecker_matrix(angles),
            np.ones((N_figure6, 1)),
            np.ones((1, N_figure6)),
            np.zeros((1, 1)),
            mode="roots",
        )
        frequency = np.angle(poles) / (2 * np.pi) * fs
        return frequency[frequency > 0]

    # One period of a unit-amplitude 1 Hz triangle wave.
    _phase = 2 * np.pi * np.arange(num_snapshots) / num_snapshots
    triangle = (2 / np.pi) * np.arcsin(np.sin(_phase))

    static_poles = np.tile(
        pole_frequencies(pyFDN.kronecker_angles(N_figure6)), num_snapshots
    )
    modulated_poles = np.concatenate(
        [
            pole_frequencies(
                pyFDN.kronecker_angles(N_figure6, theta2=np.pi / 4 + np.pi * weight)
            )
            for weight in triangle
        ]
    )
    return modulated_poles, num_snapshots, static_poles


@app.cell
def _(go, mo, modulated_poles, np, num_snapshots, static_poles):
    _bins = np.linspace(4500, 7500, 51)
    _centres = (_bins[:-1] + _bins[1:]) / 2

    _static_counts, _ = np.histogram(static_poles, _bins)
    _modulated_counts, _ = np.histogram(modulated_poles, _bins)

    _figure = go.Figure()
    _figure.add_bar(
        x=_centres,
        y=_static_counts,
        name=f"no modulation (σ = {_static_counts.std():.1f})",
    )
    _figure.add_bar(
        x=_centres,
        y=_modulated_counts,
        name=f"modulated (σ = {_modulated_counts.std():.1f})",
    )
    _figure.update_layout(
        title="Pole frequency histogram over one modulation period"
        f"<br><sup>{num_snapshots} snapshots of the 8×8 Kronecker FDN; σ is the standard deviation of the bin counts</sup>",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Number of occurrences",
        barmode="overlay",
        height=420,
    )
    _figure.update_traces(opacity=0.65)
    mo.output.replace(_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Modulation roughly halves the spread of the bin counts here, the same direction and order as the paper reports (71.0 down to 24.8 on its own network and bin grid; the absolute numbers scale with the number of snapshots and bins). Fixed poles pile into a few bins and leave others empty; moving poles fill the band, and that even filling is the reverberation-tail liveliness the modulation is after.

    ## What was left out

    The companion site's third configuration, selective freeze on \(\theta_1 = 0\), is not covered here. It needs the input matrix and the absorption to be switched at runtime — the frozen half gets zero input and unit gain — rather than a matrix construction, so it belongs with the time-varying-gain machinery rather than with the feedback matrix itself.
    """)
    return


if __name__ == "__main__":
    app.run()
