# gallery_category: Getting Started
# gallery_title: Build an FDN with process_fdn
# gallery_description: Hands-on walk-through of the FDN knobs - delays, feedback matrix, in/out gains and decay - assembled as a delay state space and simulated with pyFDN.process_fdn, the pure-NumPy time-domain path. Every step has experiments to try.

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build an FDN with `process_fdn`

    A feedback delay network is four objects and a loop:

    | | | |
    | --- | --- | --- |
    | `delays` | $N$ delay lengths in samples | how *dense* it is |
    | `A` | the $N \times N$ feedback matrix | how fast it *mixes* |
    | `B`, `C`, `D` | input, output and direct gains | how it is *driven* and *tapped* |
    | absorption | one filter per delay line | how long it *decays* |

    Together these are the **delay state space (DSS)** — the representation every
    other view in pyFDN is translated from. This notebook turns each of them in
    turn, renders the result with `pyFDN.process_fdn` (plain NumPy, no torch)
    and listens to it.

    Each code cell ends with a **Try this** block: uncomment a line, or change a
    number, and marimo re-runs everything downstream — plots and audio included.
    """)
    return


@app.cell
def _():
    import warnings

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "sphinx_gallery"  # interactive in marimo and in the docs

    import pyFDN
    from pyFDN import td

    return go, np, pyFDN, td, warnings


@app.cell
def _():
    fs = 48_000  # sampling rate in Hz
    N = 8  # number of delay lines
    ir_len_seconds = 3.0

    # Try this:
    #   N = 4    -> fewer lines: sparser, more obviously metallic
    #   N = 16   -> denser, smoother, and four times the matrix multiply
    return N, fs, ir_len_seconds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 1 — the delays set the density

    Every delay line contributes its own set of resonances, and the sum of the
    delay lengths fixes how many there are: long delays give a dense, low-pitched
    ringing, short delays a sparse and audibly pitched one. Lengths that share a
    common factor make their echoes coincide, which is heard as a metallic ring,
    so the classic choice is mutually coprime lengths spread over a range.

    `sample_delay_lengths` draws them for you; you can also just type the numbers.
    """)
    return


@app.cell
def _(N, fs, np, pyFDN):
    delays = pyFDN.sample_delay_lengths(
        N,
        delay_range=(1000, 3000),  # samples: about 21-62 ms at 48 kHz
        distribution="geometric",  # equal probability per octave
        coprime=True,  # avoid coinciding echoes
        rng=2,
    )

    print(f"delays [samples]: {delays}")
    print(f"delays [ms]:      {np.round(1000 * delays / fs, 1)}")
    print(f"mean delay:       {1000 * delays.mean() / fs:.1f} ms")

    # Try this:
    #   delay_range=(300, 600)     -> flutter, an obvious pitch, a small box
    #   delay_range=(2000, 9000)   -> sparse, granular onset, a big hall
    #   coprime=False              -> listen for the ring that comes back
    #   distribution="uniform"     -> flat in samples instead of flat per octave
    #
    # Or set them by hand, in milliseconds:
    #   delays = pyFDN.ms_to_smp(np.array([20, 27, 31, 37, 43, 53, 61, 71]), fs)
    return (delays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 2 — the feedback matrix mixes

    `A` decides where each delay line sends its output. It has one job to do
    correctly — be **lossless**, so the loop neither grows nor dies on its own —
    and one job to do well: mix the lines quickly, so a single input echo has
    spread across all $N$ lines before the ear can count the individual
    reflections. Losslessness is checkable (`is_orthogonal`); mixing speed is
    audible, and measurable as *echo density*.
    """)
    return


@app.cell
def _(N, np, pyFDN):
    print(f"  {'matrix type':32s}{'orthogonal':12s}lossless")
    for _name in pyFDN.fdn_matrix_gallery():
        _matrix = pyFDN.fdn_matrix_gallery(N, _name)
        # Orthogonality is the usual sufficient condition; the weaker one is
        # "diagonally similar to orthogonal", which is lossless just the same.
        print(
            f"  {_name:32s}{str(pyFDN.is_orthogonal(_matrix)):12s}"
            f"{pyFDN.is_unilossless(_matrix)}"
        )

    np.random.seed(0)  # "orthogonal" is drawn at random: seed it to stay reproducible
    A = pyFDN.fdn_matrix_gallery(N, "orthogonal")

    # Try this:
    #   A = pyFDN.fdn_matrix_gallery(N, "Hadamard")     -> maximal mixing, +/-1 only
    #   A = pyFDN.fdn_matrix_gallery(N, "Householder")  -> cheap: one inner product
    #   A = pyFDN.fdn_matrix_gallery(N, "permutation")  -> lossless but never mixes
    #   A = pyFDN.fdn_matrix_gallery(N, "circulant")
    return (A,)


@app.cell
def _(A, pyFDN):
    pyFDN.plot_matrix(A, title="Feedback matrix A").show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mixing is audible, and measurable

    Normalised echo density (Abel & Huang 2006) reaches 1 when the response has
    become statistically indistinguishable from Gaussian noise — that is the point
    at which the ear stops hearing separate echoes. The *mixing time* is when that
    happens.

    The comparison below changes **only** the matrix: same delays, same
    losslessness. A permutation matrix routes each line to exactly one other line,
    so the delays never combine — the result is a bank of comb filters, and it is
    the sound of a bad reverb.
    """)
    return


@app.cell
def _(A, N, delays, fs, np, pyFDN, warnings):
    _B = np.ones((N, 1)) / np.sqrt(N)
    _C = np.ones((1, N)) / np.sqrt(N)
    _D = np.zeros((1, 1))

    for _label, _matrix in [
        ("orthogonal", A),
        ("permutation", pyFDN.fdn_matrix_gallery(N, "permutation")),
    ]:
        _ir = pyFDN.dss_to_impz(2 * fs, delays, _matrix, _B, _C, _D).squeeze()
        with warnings.catch_warnings():
            # "never mixes" is a result here, not a problem: echo_density warns
            # when the density does not reach the threshold, which is exactly
            # what a permutation matrix does.
            warnings.simplefilter("ignore", UserWarning)
            _mixing_time, _ = pyFDN.echo_density(_ir, n=1024, fs=fs, hop=256)
        _mixed = f"{_mixing_time:.0f} ms" if _mixing_time else "never"
        print(f"{_label:12s} mixes after {_mixed}")

    # Try this: add "Hadamard" or "circulant" to the list and rank them.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wiring — input, output and direct gains

    `B` distributes the input over the delay lines, `C` sums them back down to the
    output, and `D` is the direct (dry) path around the loop. For a mono reverb
    the flat choice below is hard to beat; the interesting variations are
    *decorrelated* output gains, which is how one FDN feeds several loudspeakers
    from the same tail.
    """)
    return


@app.cell
def _(N, np):
    B = np.ones((N, 1)) / np.sqrt(N)  # (N, num_inputs)
    C = np.ones((1, N)) / np.sqrt(N)  # (num_outputs, N)
    D = np.zeros((1, 1))  # (num_outputs, num_inputs) — wet only

    print(f"B: {B.shape}, C: {C.shape}, D: {D.shape}")

    # Try this:
    #   D = np.full((1, 1), 0.5)                     -> mix the dry signal back in
    #   C = np.random.default_rng(0).standard_normal((2, N)) / np.sqrt(N)
    #       -> a stereo FDN: two decorrelated taps on the same tail
    #   B = np.eye(N)[:, :1]                         -> drive one delay line only
    return B, C, D


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render it — the lossless FDN

    That is a complete FDN already. `dss_to_impz` runs the delay state space in
    the time domain and hands back an impulse response.

    With an orthogonal `A` and nothing in the loop, the FDN is lossless: energy
    circulates forever. The tail below does not decay at all, which is exactly
    what the plot and the audio player should show.
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, ir_len_seconds, mo, pyFDN):
    ir_lossless = pyFDN.dss_to_impz(
        int(ir_len_seconds * fs), delays, A, B, C, D
    ).squeeze()

    mo.vstack(
        [
            pyFDN.plot_impulse_response(
                ir_lossless, fs=fs, title="Lossless FDN — the loop never lets go"
            ),
            pyFDN.labeled_audio("lossless", pyFDN.peak_normalize(ir_lossless), fs=fs),
        ]
    )
    return (ir_lossless,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 3 — absorption sets the decay

    To make it decay, attenuate the signal every time it goes round. The
    attenuation has to be **proportional to the delay length**, otherwise short
    lines die before long ones and the decay is no longer a single exponential.
    One gain per sample, raised to the delay length, does exactly that:

    `A_lossy = diag(g ** delays) @ A`

    This is *homogeneous* decay: every resonance dies at the same rate, so the
    reverberation time is the one you asked for.
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, ir_len_seconds, np, pyFDN):
    rt = 1.8  # target T60 in seconds

    g = pyFDN.rt_to_gain_per_sample(rt, fs)
    A_lossy = np.diag(g**delays) @ A
    ir_broadband = pyFDN.dss_to_impz(
        int(ir_len_seconds * fs), delays, A_lossy, B, C, D
    ).squeeze()

    print(f"gain per sample: {g:.8f}")
    print(f"per-round-trip attenuation [dB]: {np.round(pyFDN.lin_to_db(g**delays), 2)}")

    # Try this:
    #   rt = 0.4   -> a small, dry room
    #   rt = 6.0   -> a cathedral; raise ir_len_seconds to see the whole tail
    #   A_lossy = np.diag(g ** delays.mean()) @ A
    #       -> the same gain on every line: no longer homogeneous. Look at the EDC.
    return (ir_broadband,)


@app.cell
def _(fs, ir_broadband, ir_lossless, mo, pyFDN):
    mo.vstack(
        [
            pyFDN.plot_edc(
                ir_lossless,
                ir_broadband,
                fs=fs,
                labels=["lossless", "broadband decay"],
                normalize=True,
                title="Energy decay curve — the -60 dB crossing is T60",
            ),
            pyFDN.labeled_audio(
                "broadband decay", pyFDN.peak_normalize(ir_broadband), fs=fs
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 3, per band — frequency-dependent decay

    Real rooms absorb high frequencies faster than low ones, so a single number
    is not enough. Replace the scalar gain with a **filter per delay line** whose
    attenuation follows the target $T_{60}$ across frequency.
    `absorption_geq` designs those filters from a target curve at ten bands (DC,
    the eight octave bands 63 Hz – 8 kHz, and Nyquist).

    The filters live *inside* the loop, so the feedback matrix goes back to being
    the plain lossless `A` — all the decay is now in the filters. Bundling the
    parts into an `FDNBuild` lets `build_to_impz` render the whole thing.
    """)
    return


@app.cell
def _(A, B, C, D, delays, fs, ir_len_seconds, np, pyFDN):
    target_rt = np.array([2.4, 2.4, 2.3, 2.1, 1.8, 1.4, 1.0, 0.7, 0.5, 0.5])

    absorption = pyFDN.absorption_geq(target_rt, delays, fs)  # (n_sections, 6, N)

    build = pyFDN.FDNBuild(
        A=A, B=B, C=C, D=D, delays=delays, fs=fs, post_delay=absorption
    )
    ir = pyFDN.build_to_impz(build, int(ir_len_seconds * fs)).squeeze()

    print(f"absorption SOS bank: {absorption.shape}  (sections, coefficients, lines)")

    # Try this:
    #   target_rt = np.full(10, 1.5)                       -> flat decay, still filtered
    #   target_rt = np.linspace(4.0, 0.2, 10)              -> a very dark room
    #   target_rt = np.array([4, 4, 3, 2, 1, .6, .4, .3, .2, .2])
    #       -> ask for something extreme and see where the design stops delivering
    return absorption, build, ir, target_rt


@app.cell
def _(build, pyFDN):
    # Every parameter of the finished FDN in one figure: delays, A, B, C, D and
    # the absorption response of each line.
    pyFDN.plot_FDN_build(build, title="The complete FDN")

    # Try this: pyFDN.plot_db_per_sample(absorption, delays, fs=fs, nfft=2**14)
    #   -> the attenuation each line applies per sample, which is the quantity
    #      the absorption design actually solves for.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Did you get what you asked for?

    `estimate_rt_bands` fits the Schroeder decay of the rendered impulse response
    in octave bands. This is the loop that matters: *state a target, render,
    measure, compare.* Everything analytic in pyFDN is meant to be checked this
    way — nothing about the design has to be taken on faith.
    """)
    return


@app.cell
def _(fs, go, ir, np, pyFDN, target_rt):
    rt_measured, f_centre = pyFDN.estimate_rt_bands(ir, fs)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=f_centre,
            y=target_rt[1:9],  # the same eight octave bands
            mode="lines+markers",
            name="target",
            line={"dash": "dash"},
        )
    )
    _fig.add_trace(
        go.Scatter(x=f_centre, y=rt_measured, mode="lines+markers", name="measured")
    )
    _fig.update_layout(
        title="T60: asked for vs. delivered",
        xaxis={"title": "Frequency [Hz]", "type": "log"},
        yaxis={"title": "T60 [s]", "range": [0, None]},
        template="plotly_white",
        height=380,
    )
    _fig.show()

    print(f"target   [s]: {np.round(target_rt[1:9], 2)}")
    print(f"measured [s]: {np.round(rt_measured, 2)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse — and listen

    Three views of the same impulse response, and then your ears. The spectrogram
    shows the absorption as a tilt: the top of the picture empties out first,
    because that is where the target $T_{60}$ was shortest.
    """)
    return


@app.cell
def _(fs, ir, pyFDN):
    pyFDN.plot_spectrogram(ir, fs, title="Impulse response — time-frequency energy")
    return


@app.cell
def _(fs, ir, mo, pyFDN):
    _mixing_time, _ = pyFDN.echo_density(ir, n=1024, fs=fs, hop=256)
    print(f"mixing time: {_mixing_time:.0f} ms")

    mo.vstack(
        [
            pyFDN.plot_impulse_response(ir, fs=fs, title="Impulse response"),
            pyFDN.labeled_audio("impulse response", pyFDN.peak_normalize(ir), fs=fs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run audio through it

    `process_fdn` is the same recursion, driven by a signal instead of an impulse.
    The absorption filters go into the `post_delay` hook — the point in the loop
    just after the delay outputs, which is where `build_to_impz` put them too.

    Pad the input with silence, or the tail is cut off where the signal ends.
    """)
    return


@app.cell
def _(A, B, C, D, absorption, delays, fs, mo, np, pyFDN, td):
    dry, _ = pyFDN.load_audio("synth_dry", fs=fs)
    x = np.pad(dry, (0, 2 * fs))  # room for the tail

    wet = pyFDN.process_fdn(
        x,
        delays,
        A,
        B,
        C,
        D,
        post_delay=td.SOSBank(absorption),  # absorption inside the loop
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("dry", dry, fs=fs),
            pyFDN.labeled_audio("wet", pyFDN.peak_normalize(wet), fs=fs),
        ]
    )
    return wet, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three hooks, one loop

    `process_fdn` takes a filter at three points, and each one is an entire family
    of reverbs:

    | hook | where it sits | what it buys |
    | --- | --- | --- |
    | `post_delay` | after the delay outputs | absorption — used above |
    | `post_matrix` | after the feedback matrix | time variation, non-linearity |
    | `post_output` | on the wet signal | output EQ, voicing |

    A hook is any object with a `.filter(block)` method, so your own DSP drops
    straight in. The cell below is the same FDN with a moving matrix in the loop:
    `TimeVaryingMatrix` stays orthogonal at every sample, so the decay is
    unchanged — but the modes never sit still, and the metallic ringing of a
    static FDN cannot build up.
    """)
    return


@app.cell
def _(A, B, C, D, absorption, delays, fs, mo, np, pyFDN, td, wet, x):
    np.random.seed(11)  # TimeVaryingMatrix draws its phases from the global stream
    wet_moving = pyFDN.process_fdn(
        x,
        delays,
        A,
        B,
        C,
        D,
        post_delay=td.SOSBank(absorption),
        post_matrix=td.TimeVaryingMatrix(len(delays), 10.0, 1.1, fs, 0.7),
    )

    # Try this:
    #   post_matrix=td.AbsoluteValue(len(delays))
    #       -> a rectifier in the loop. The harmonics were never in the input;
    #          the FDN is generating them. No transfer function exists any more.
    #   post_output=td.SOSBank(...)   -> voice the wet signal with pyFDN.design_geq
    mo.hstack(
        [
            pyFDN.labeled_audio("static matrix", pyFDN.peak_normalize(wet), fs=fs),
            pyFDN.labeled_audio(
                "time-varying matrix", pyFDN.peak_normalize(wet_moving), fs=fs
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint

    Get an FDN you like the sound of. Then break it on purpose — make it metallic,
    make it flutter, make it dark — and say which knob did it.

    Where to go from here:

    - `example_vanilla_FDN` — the same FDN through FLAMO, the differentiable path
    - `example_absorption_geq` — the absorption design on its own, in detail
    - `example_delay_matrix_density` — buying echo density with delays in the
      feedback path
    - `example_fdn_gallery` — every feedback matrix in the gallery, side by side
    - `example_time_varying_fdn`, `example_scattering_fdn` — beyond the vanilla FDN
    - `example_train_colorless_FDN` — when there is no closed form left, use
      gradients
    """)
    return


if __name__ == "__main__":
    app.run()
