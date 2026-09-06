# gallery_category: Getting Started
# gallery_title: Train an FDN to match a measured room
# gallery_description: Hands-on walk-through of fitting an FDN to a measured room impulse response by gradient descent - a generic 1 s reverberator in, decay and output EQ trained out - with the objective and the parametrization each a switch you can turn. Every step has experiments to try.
# references: Concert_Hall_Impulse_Responses

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
    # Train an FDN to match a measured room

    **Convert a room impulse response into an FDN** designs a reverberator
    analytically: measure the decay, measure the level, turn both into filters.
    This notebook does the opposite. It starts from a **generic 1 s reverberator
    that knows nothing about the room**, hands every parameter to an optimizer,
    and lets a loss function find them.

    | | | |
    | --- | --- | --- |
    | the target | a measured concert hall | what the loss compares against |
    | the start | a flat 1 s FDN | what the optimizer is given |
    | the knobs | decay, output EQ, matrix, gains | what has a gradient |
    | the exam | octave-band $T_{60}$ and level | measured *after*, fed to nothing |

    Three things have to be right before an optimizer is worth reaching for, and
    each is a step below: a **parametrization** the decay cannot escape from, a
    **module** that can change the colour, and a **loss** that can see a decay at
    all. The obvious choice of loss gets the last one confidently wrong, and
    section 4 shows the table where it does.

    Each code cell ends with a **Try this** block: change a number, and marimo
    re-runs everything downstream — plots, metrics and audio included.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import torch

    import pyFDN

    return go, np, pyFDN, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings

    The fit is 300 gradient steps through a $2^{17}$-point frequency grid, on the
    CPU in `float32` — a couple of minutes. `float64` is the reference here: a
    cumulative-energy loss reads the quietest samples in the buffer, so it is the
    one objective where round-off at the end of a render could plausibly matter.
    On *this* fit it does not — the two precisions land within 0.2 % of the same
    loss — so `float32` is what runs, and the run is half as long.

    Both filters the FDN needs — the in-loop attenuation that sets the decay, and
    the output EQ that colours it — are **first-order shelves**: two numbers and
    one biquad each. Nothing downstream names a design class, so switching the
    whole notebook to a ten-band graphic EQ is switching one word, and the Try
    this block below spells it out. The shelf is what runs because it is the
    cheaper fit by a wide margin — eleven biquads per delay line instead of one
    is roughly five times the wall clock for the same 300 steps — and because on
    this room it is *not* the worse answer. Both results are tabulated further
    down, so the comparison is readable without paying for it.
    """)
    return


@app.cell
def _(np, torch):
    fs = 48000
    device, dtype = "cpu", torch.float32

    # Everything the EQ design decides, in one place: the name, how many biquads
    # it costs, and where on the frequency axis its parameters sit (the design's
    # own band layout, not the estimator's -- they coincide at the octave centres
    # and nowhere else).
    design = "first_order_shelf"
    n_sections = 1
    param_frequencies = np.array([1.0, fs / 2])

    print(f"device:      {device} ({torch.get_num_threads()} CPU threads available)")
    print(f"dtype:       {str(dtype).removeprefix('torch.')}")
    print(f"design:      {design}  —  {n_sections} biquad")
    print(f"parameters sit at (Hz): {param_frequencies.round(0)}")

    # Try this: swap in the ten-band graphic EQ -> the whole notebook re-fits and
    # re-measures. Ten numbers reach a 35% lower loss than two and a *worse* mean
    # RT error, by being wrong in completely different bands. Budget five times
    # the wall clock for it.
    #   from pyFDN.eq import CENTER_FREQUENCIES
    #   design, n_sections = "graphic_eq", 11
    #   param_frequencies = np.concatenate(([1.0], CENTER_FREQUENCIES, [fs / 2]))
    return design, device, dtype, fs, n_sections, param_frequencies


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    ## Step 1 — the target

    The Promenadikeskus concert hall in Pori, Finland, published at
    {pyFDN.paper_link("Concert_Hall_Impulse_Responses")}. Trimmed to the direct
    sound and normalized to unit energy, exactly as in **Convert a room impulse
    response into an FDN**, so the two notebooks are comparable.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    rir, _file_fs = pyFDN.load_audio("s3_r4_o", fs=fs)
    rir = rir[int(np.argmax(np.abs(rir))) :]  # trim to the direct sound
    rir = rir / np.linalg.norm(rir)  # unit energy
    rir_len = len(rir)

    print(f"target RIR: {rir_len} samples ({rir_len / fs:.2f} s) at {fs} Hz")

    # Try this: any other packaged response is a one-word change --
    #   pyFDN.load_audio("s3_r1_o")  -> the same hall, receiver closer to the source
    # and the whole notebook re-fits to it.
    return rir, rir_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The yardstick — measured, then put away

    Octave-band $T_{60}$ and initial level of the *target*, by Schroeder backward
    integration. In the analytic notebook these numbers **are** the design. Here
    they are the exam paper: computed once, fed to nothing, and compared against
    at the end.

    Note the shape — 2.8 s at the bottom, 1.2 s at 8 kHz, and a fall in between
    that never reverses, with a plateau across 63/125 Hz and another across
    500 Hz/1 kHz. The monotone part is what a shelf can reach; the plateaus are
    what it cannot.
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    est_rt, f_centre = pyFDN.estimate_rt_bands(rir, fs)
    est_level, _ = pyFDN.estimate_initial_level_bands(rir, est_rt, fs)

    print(f"bands (Hz):   {f_centre.round(0)}")
    print(f"measured RT:  {est_rt.round(2)}")
    return est_level, est_rt, f_centre


@app.cell
def _(fs, mo, pyFDN, rir):
    mo.vstack(
        [
            pyFDN.plot_spectrogram(rir, fs, title="The target: a measured hall"),
            pyFDN.labeled_audio("the room", pyFDN.peak_normalize(rir), fs=fs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — a reverberator that knows nothing about the room

    `fdn_build_gallery` builds the whole starting point in one call: a random
    orthogonal feedback matrix, sixteen normalized gains, no dry path, and
    per-delay absorption for a **flat 1 s decay in every band**. Only the delays
    are sampled separately, so that `distribution="geometric"` and `coprime=True`
    can be asked for.

    `init_build` is a complete FDN, not a scaffold — the untrained render further
    down is just this build, so nothing has to be reconstructed by hand to say
    what the optimizer started from.

    The room is 2.8 s at 63 Hz falling to 1.2 s at 8 kHz. A flat 1 s start is
    therefore wrong by a factor of three at the bottom of the spectrum and by a
    fifth at the top — and wrong in the *opposite direction* at the two ends,
    which is the part no single scalar can fix.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    num_delays = 16
    delays = pyFDN.sample_delay_lengths(
        num_delays,
        (700, 2500),  # samples: about 15-52 ms at 48 kHz
        distribution="geometric",
        coprime=True,
        sort=True,
        rng=1,
    )

    # rng=2 is not arbitrary. The orthogonal training parametrization lives on
    # SO(N), so it hands a det<0 matrix back with its last column flipped: not
    # the matrix you asked for. This seed lands in SO(N), and the assertion
    # below is what says so.
    init_build = pyFDN.fdn_build_gallery(
        N=num_delays,
        fs=fs,
        delay_range=(700, 2500),
        delay_distribution="geometric",
        coprime=True,
        sort_delays=True,
        io_type="normalized",
        direct_gain=0.0,
        rt=1.0,  # flat 1 s: the entire prior knowledge of the room
        rt_nyquist=1.0,
        rng=2,
    )
    delays = init_build.delays
    assert np.linalg.det(init_build.A) > 0, "feedback matrix is not in SO(N)"

    print(f"delays (samples): {init_build.delays}")
    print(f"absorption:       {init_build.post_delay.shape} — flat 1 s")

    # Try this:
    #   num_delays = 8    -> half the matrix, half the density. Does the fit
    #                        still reach the room's decay? (it does) Does it
    #                        sound like it? (less so -- decay is not density)
    #   (2000, 6000)      -> a much longer delay range: sparser, and the early
    #                        part of the tail stops resembling the target
    #   rt=2.5            -> start near the answer instead of far from it, and
    #                        watch the loss curve start an order of magnitude down
    return delays, init_build


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — make it trainable

    `trainable_from_build` wraps the build into a FLAMO model. Two of the hooks
    get a **module with a parameter** rather than a baked filter:

    | what | which parameter | how |
    |---|---|---|
    | **decay** | `post_delay` hook | trained — as reverberation time in seconds, via `pyFDN.AttenuationFilter` |
    | **colour** | `post_output` hook | trained — as gain in dB, via `pyFDN.OutputEQ` |
    | fine structure of $\lvert H \rvert$ | feedback matrix $A$ | trained — on $SO(N)$ |
    | level | gains $b$, $c$ | trained |
    | dry path | $D$ | trained |
    | when the echoes fall | delays | **fixed** — integer sample counts, no gradient to take |

    Both filters are parametrized by the **quantity you would plot**, not by
    biquad coefficients, and that is what makes the fit converge at all. Trained
    as raw SOS coefficients the decay diverges within fifty steps at any learning
    rate: nothing holds the poles inside the unit circle, and more loop gain is
    the cheapest direction any loss can offer. `AttenuationFilter` instead maps
    one reverberation time per band through $-60 d_i / (\mathrm{RT} f_s)$ dB per
    round trip, so **every** value the parameter can take is a contractive loop,
    with a softplus floor to keep a band that dips through zero from turning the
    attenuation into a gain. `OutputEQ` is the same move on the one module that
    can place band *levels* rather than band decays — it sits outside the
    recursion, so it starts flat and needs no bound at all.
    """)
    return


@app.cell
def _(design, device, dtype, fs, init_build, np, pyFDN, rir, rir_len, torch):
    nfft = 2**17  # 2.73 s at 48 kHz — long enough for the loss and the metrics

    model = pyFDN.trainable_from_build(
        init_build,
        # every gain with a gradient: A, b, c and D
        trainable=pyFDN.Trainable(direct=True),
        # the decay, as a reverberation time rather than as coefficients
        post_delay=pyFDN.AttenuationFilter(
            1.0,
            init_build.delays,
            fs,
            design=design,
            nfft=nfft,
            device=device,
            dtype=dtype,
        ),
        # the output EQ, starting flat, as a gain in dB
        post_output=pyFDN.OutputEQ(
            0.0,
            1,
            fs,
            design=design,
            nfft=nfft,
            device=device,
            dtype=dtype,
        ),
        nfft=nfft,
        device=device,
        dtype=dtype,
    )

    # One excitation, reused by every render. Building it explicitly is what
    # keeps this working on a GPU: the default one is made on the CPU.
    excitation = pyFDN.impulse_excitation(1, nfft, device=device, dtype=dtype)

    def render(m):
        """The FDN's impulse response, straight out of the FLAMO model."""
        h = pyFDN.model_response(m, excitation).h.detach().cpu()
        return np.asarray(h, dtype=np.float64).reshape(-1)[:rir_len]

    def read(name):
        """A parameter and the filter it designs, as numpy."""
        p = pyFDN.param(model, name)
        return (
            p.raw().detach().cpu().numpy().copy().ravel(),
            p.value().detach().cpu().numpy().copy(),
        )

    # the only thing the target tells the initial model: how loud it is
    energy_gain = float(np.linalg.norm(rir[:nfft]) / np.linalg.norm(render(model)))
    with torch.no_grad():
        pyFDN.param(model, "output_gain").raw().mul_(energy_gain)

    # the untrained FDN, before the optimizer touches it. Taken here, in this
    # same cell, because train_fdn steps `model` in place -- once the next cell
    # has run there is no "before" left.
    ir_init = render(model)
    _, init_sos = read("post_delay")

    for _p in pyFDN.params(model):
        print(f"{_p}  raw {tuple(_p.raw().shape)}")
    print(f"\nenergy match: output gain x {energy_gain:.2f}")

    # Try this: comment out the energy match and re-run. The fit spends its
    # first steps on a volume knob, and the loss is normalized by the target's
    # energy, so an FDN two decades too quiet starts on the flat part of the
    # compression curve where there is little gradient to follow. What the
    # scalar cannot do is say *when* that energy arrives, which is the problem.
    return init_sos, ir_init, model, read, render


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — a loss that can see a decay

    This is the step worth dwelling on, because the obvious objective gets it
    confidently wrong. `pyFDN.MatchCumulativeEnergy` compares the short-time
    energy of the two responses, **cumulated twice** — backwards in time, and
    along frequency:

    $$E[f, t] \;=\; \sum_{t' \ge t} \; \sum_{f' \ge f} \big| S[f', t'] \big|^2$$

    The time direction is Schroeder backward integration — the decay itself. The
    frequency direction does the job that **splitting into octave bands** usually
    does, and does it without band edges: a cumulative sum compares every bin
    against every wider band that contains it, all at once, and is monotone and
    smooth in both axes, which is worth a great deal to a gradient. Read down the
    $t = 0$ edge and the surface is the integrated spectrum; read across the
    $f = 0$ edge and it is the full-band energy decay curve; the interior ties
    the two together. One loss, both quantities, no bands.

    There is one term and no weight to tune, because there is nothing to weigh it
    against: the surface carries decay and colour together, so a second term
    would only be a second opinion about the same data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Why not a spectrogram distance — the table where it fails": mo.md(r"""
    Freeze everything except the decay, scale a measured RT by a constant, and
    score the result on a mel multi-resolution spectrogram distance alone:

    | RT scale | 0.4 | 0.6 | **1.0** | 1.3 | 1.6 |
    |---|---|---|---|---|---|
    | mel MSS ($\times 10^{-5}$) | 1.819 | **1.804** | 1.855 | 1.924 | 2.009 |
    | energy decay at 1 s | -67 dB | -47 dB | **-31 dB** | -25 dB | -21 dB |
    | the room, at 1 s | | | **-29 dB** | | |

    The minimum is at 0.6 — an FDN whose tail is 18 dB below the room's at one
    second scores *better* than the one that tracks it to within 2 dB. That is
    not a bug in the loss, it is what a magnitude distance does: two rooms with
    the same decay still have uncorrelated fine structure, and against detail you
    cannot predict, silence is a better guess than the right amount of the wrong
    detail.
    """),
            "power=0.5 — compression, not decibels": mo.md(r"""
    A cumulated energy surface spans the entire dynamic range of the decay, so a
    plain MSE on it sees the first few frames and nothing else. `power`
    compresses that range — the surface is normalized by the target's total
    energy and raised to a fractional power, 0.5 here, so that what is compared
    is amplitude rather than energy. Lower values compress harder and move weight
    onto the quiet end.

    A logarithm is the obvious alternative and is worse: it turns the silence
    *below* the response into an unbounded penalty, so whichever bin is nearest
    zero owns the gradient. A power keeps the compressed surface bounded and its
    gradient finite, and leaves 0 an ordinary value to predict.
    """),
            'frequency="both" — which way the cumulation runs is not a detail': mo.md(r"""
    Cumulating from high to low puts every bin's energy into the rows *below* it,
    so an error in a low band moves only the largest values on the surface — the
    ones compression weights least — while a high band gets rows of its own. On
    this fit, with everything else held fixed at 300 steps, that asymmetry is
    worth more than the compression exponent:

    | `frequency` | `power` | mean RT error | level shape | trained RT at 63 Hz |
    |---|---|---|---|---|
    | `"descending"` (high → low) | 0.5 | 16.8 % | 2.56 dB | 0.26 s |
    | `"descending"` | 0.25 | 19.0 % | 1.46 dB | 0.44 s |
    | `"ascending"` | 0.5 | 12.8 % | 1.12 dB | 2.35 s |
    | **`"both"`** | **0.5** | **10.3 %** | **0.88 dB** | **2.17 s** |

    (the room is 2.8 s at 63 Hz). Cumulating downwards alone leaves the bottom
    octave with essentially no gradient and the fit abandons it; `"both"` scores
    the two directions separately — each normalized and compressed on its own,
    then averaged — and the low bands come back. Lengthening the analysis window
    does *not* fix it (15.4 % at `window=4096`, 63 Hz still at 0.20 s), which is
    the evidence that the problem is weighting rather than frequency resolution.

    Those rows were measured at a different feedback-matrix seed and a shorter
    `nfft` than this notebook now uses. Compare the rows against each other, not
    against the result further down: the gaps between them are far larger than
    the offset.
    """),
        }
    )
    return


@app.cell
def _(pyFDN, rir):
    loss = pyFDN.MatchCumulativeEnergy(rir, window=1024, power=0.5, frequency="both")
    # loss = pyFDN.MatchEnergyDecay(rir, window=4096)   # bands, not cumulation
    # Try this — each one is a row of the tables above, run for yourself:
    #   frequency="descending"  -> the bottom octave is abandoned; look at the
    #                              trained RT curve at 63 Hz
    #   power=0.25              -> harder compression, more weight on the tail
    #   window=4096             -> longer analysis window; slower, and no better
    #
    # Or replace the objective outright and watch the decay go wrong:
    #   loss = pyFDN.MatchMelSpectrogram(rir)
    #   loss = pyFDN.MatchEnergyDecay(rir, window=4096)   # bands, not cumulation
    #   loss = pyFDN.MatchCumulativeEnergy(rir) + 0.1 * pyFDN.MatchMelSpectrogram(rir)
    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — train

    300 Adam steps at `lr=3e-2`. On a CPU that is a minute or two; watch the loss
    curve below rather than the clock.

    The trained response is rendered at the end of the same cell, out of the same
    model, through the same `render` the untrained one went through: two FDNs
    being compared on a metric should not be reaching it by two different routes.
    """)
    return


@app.cell
def _(device, dtype, loss, model, pyFDN, read, render):
    log = pyFDN.train_fdn(
        model,
        loss,
        optimizer="adam",
        max_steps=300,
        lr=3e-2,
        patience=100,
        device=device,
        dtype=dtype,
        rng=0,
    )

    trained_rt, trained_sos = read("post_delay")
    trained_eq_db, trained_eq_sos = read("post_output")
    ir_trained = render(model)

    print(
        f"ran {log.steps_run} steps, loss {log.train_loss[0]:.4g} -> "
        f"{log.train_loss[-1]:.4g} "
        f"({100 * (1 - log.train_loss[-1] / log.train_loss[0]):.0f}% down)"
    )
    print(f"\ntrained RT (s):         {trained_rt.round(2)}")
    print(f"trained output EQ (dB): {trained_eq_db.round(1)}")
    print(f"\nin-loop filter: {trained_sos.shape} — per delay line")
    print(f"output filter:  {trained_eq_sos.shape}")

    # Try this — break it on purpose, and say from the loss curve what failed:
    #   max_steps=20     -> stopped, not converged
    #   lr=1.0           -> the step size overshoots every minimum it finds
    #   optimizer="lbfgs"
    #   rng=1, rng=2     -> a different random start. The mean RT error moves by
    #                       a few tenths of a point, not by points: the result is
    #                       a property of the fit, not of one lucky matrix.
    return (
        ir_trained,
        log,
        trained_eq_db,
        trained_eq_sos,
        trained_rt,
        trained_sos,
    )


@app.cell
def _(go, log):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(y=log.train_loss, mode="lines"))
    _fig.update_layout(
        title="Training loss",
        xaxis={"title": "step"},
        yaxis={"title": "cumulative-energy RMS error", "type": "log"},
        template="plotly_white",
        height=380,
    )
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 — the two filters are the answer

    Both trained quantities are things you would plot anyway — a reverberation
    time in seconds and an output EQ in dB — and neither needs an estimator to
    read. But the thing to look at is the **curve** each set of numbers designs,
    because that is what the FDN runs. The parameters are drawn on top of it as
    markers, at the frequencies they actually sit at.

    The decay is plotted as reverberation time against frequency, read off the
    fitted filter's gain per sample,
    $\mathrm{RT}(f) = -60 / (f_s \cdot 20\log_{10} g(f))$, which is
    delay-line-independent because the decay is homogeneous.

    Read the ends of either curve with care. The outermost parameters sit at DC
    and at **24 kHz** — an octave and a half above the highest band the estimator
    reports, and well into where the recording has nothing left — so they are
    free to go anywhere. The shelf's Nyquist endpoint settles around 0.2 s; the
    graphic EQ's goes negative outright, and the parametrization floors it.
    Neither is a claim about 8 kHz: what the FDN does there is the *curve* at
    8 kHz. At the edges of a design, read the filter, not the parameter.
    """)
    return


@app.cell
def _(
    delays,
    est_rt,
    f_centre,
    fs,
    go,
    init_sos,
    np,
    param_frequencies,
    pyFDN,
    trained_rt,
    trained_sos,
):
    def _rt_curve(sos):
        """Reverberation time vs frequency implied by a homogeneous decay filter."""
        angles, magnitude = pyFDN.sos_gain_per_sample_curves(sos, delays, 512)
        freqs = angles / np.pi * (fs / 2)
        return freqs, -60.0 / (fs * 20.0 * np.log10(magnitude[:, 0]))

    curve_f, rt_curve = _rt_curve(trained_sos)
    _, _rt_init_curve = _rt_curve(init_sos)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=curve_f, y=_rt_init_curve, name="initial (flat 1 s)", line={"dash": "dot"}
        )
    )
    _fig.add_trace(go.Scatter(x=curve_f, y=rt_curve, name="trained filter"))
    _fig.add_trace(
        go.Scatter(
            x=param_frequencies,
            y=trained_rt,
            name="the trained parameters",
            mode="markers",
            marker={"size": 11, "symbol": "diamond"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=f_centre,
            y=est_rt,
            name="measured in the RIR (never seen by the fit)",
            mode="lines+markers",
            line={"dash": "dash"},
        )
    )
    _fig.update_layout(
        title="The absorption parameter: reverberation time vs frequency",
        xaxis={
            "title": "Frequency (Hz)",
            "type": "log",
            "range": [0, np.log10(fs / 2)],
        },
        yaxis={"title": "RT (s)", "rangemode": "tozero"},
        template="plotly_white",
        height=380,
    )
    _fig.show()

    print(
        f"filter RT at the octave centres (s): "
        f"{np.interp(f_centre, curve_f, rt_curve).round(2)}"
    )
    print(f"measured, octave bands (s):          {est_rt.round(2)}")
    return (rt_curve,)


@app.cell
def _(fs, go, np, param_frequencies, pyFDN, trained_eq_db, trained_eq_sos):
    _probe = np.logspace(0, np.log10(fs / 2), 400)
    # probe_sos evaluates each biquad on its own, so the cascade is their sum in dB
    _db, _, _ = pyFDN.probe_sos(trained_eq_sos[:, :, 0], _probe, 2**14, fs)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_probe,
            y=np.zeros_like(_probe),
            name="initial (flat)",
            line={"dash": "dot"},
        )
    )
    _fig.add_trace(go.Scatter(x=_probe, y=_db.sum(axis=1), name="trained filter"))
    _fig.add_trace(
        go.Scatter(
            x=param_frequencies,
            y=trained_eq_db,
            name="the trained parameters",
            mode="markers",
            marker={"size": 11, "symbol": "diamond"},
        )
    )
    _fig.update_layout(
        title="The output EQ parameter: gain",
        xaxis={
            "title": "Frequency (Hz)",
            "type": "log",
            "range": [0, np.log10(fs / 2)],
        },
        yaxis={"title": "Gain (dB)"},
        template="plotly_white",
        height=380,
    )
    _fig.show()

    print(f"trained output EQ (dB): {trained_eq_db.round(1)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7 — did you get what you asked for?

    The same two estimators, now applied to both rendered FDNs — against a
    measurement neither of them saw. This is the loop that matters and it is the
    same one the analytic notebook runs: *state a target, render, measure,
    compare.*

    Two things are worth being clear about before reading the numbers. The level
    **offset** is not an achievement: the energy match set it before the
    optimizer ran, and the fit merely kept it. The level **shape** is, because
    nothing in the FDN proper can bend it — that is the trained output EQ, and it
    is the only part of this that the analytic pipeline would have had to design.
    """)
    return


@app.cell
def _(fs, ir_init, ir_trained, pyFDN):
    rt_init, _ = pyFDN.estimate_rt_bands(ir_init, fs)
    rt_trained, _ = pyFDN.estimate_rt_bands(ir_trained, fs)
    level_init, _ = pyFDN.estimate_initial_level_bands(ir_init, rt_init, fs)
    level_trained, _ = pyFDN.estimate_initial_level_bands(ir_trained, rt_trained, fs)
    return level_init, level_trained, rt_init, rt_trained


@app.cell
def _(
    est_level,
    est_rt,
    f_centre,
    go,
    level_init,
    level_trained,
    np,
    pyFDN,
    rt_init,
    rt_trained,
):
    from plotly.subplots import make_subplots

    _fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Reverberation time", "Initial level")
    )
    for _name, _rt, _lv, _dash in (
        ("Target RIR", est_rt, est_level, None),
        ("FDN, untrained", rt_init, level_init, "dot"),
        ("FDN, trained", rt_trained, level_trained, None),
    ):
        _style = {"dash": _dash} if _dash else {}
        _fig.add_trace(
            go.Scatter(
                x=f_centre,
                y=_rt,
                name=_name,
                mode="lines+markers",
                line=_style,
                legendgroup=_name,
            ),
            row=1,
            col=1,
        )
        _fig.add_trace(
            go.Scatter(
                x=f_centre,
                y=pyFDN.lin_to_db(_lv),
                name=_name,
                mode="lines+markers",
                line=_style,
                legendgroup=_name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    _fig.update_xaxes(title_text="Frequency (Hz)", type="log")
    _fig.update_yaxes(title_text="RT (s)", rangemode="tozero", row=1, col=1)
    _fig.update_yaxes(title_text="Initial level (dB)", row=1, col=2)
    _fig.update_layout(template="plotly_white", height=400)
    _fig.show()

    def _report(name, rt, level):
        _err = pyFDN.lin_to_db(level) - pyFDN.lin_to_db(est_level)
        print(
            f"{name:16s} RT error {100 * np.abs(rt / est_rt - 1).mean():4.1f}%   "
            f"level offset {_err.mean():+5.1f} dB   "
            f"level shape {np.abs(_err - _err.mean()).mean():4.2f} dB"
        )

    _report("FDN, untrained", rt_init, level_init)
    _report("FDN, trained", rt_trained, level_trained)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "What ten times the parameters actually buys — both designs, side by side": mo.md(r"""
    From a flat 1 s decay and a flat EQ, one energy match and 300 gradient steps,
    with the EQ design in each of its two settings and everything else identical:

    | | mean RT error | level offset | level shape | final loss |
    |---|---|---|---|---|
    | untrained (flat 1 s) | 52.4 % | -1.3 dB | 1.71 dB | 0.0932 |
    | trained, **first-order shelf** (2 numbers) | 9.8 % | +0.4 dB | 0.79 dB | 0.00393 |
    | trained, **ten-band graphic EQ** (10 numbers) | 10.6 % | +0.2 dB | 0.71 dB | 0.00256 |

    Read the two trained rows against each other and the headline is not the one
    you would expect. Five times the parameters and eleven times the biquads
    reach a **lower loss** — 35 % lower — and a better level shape, and a
    *higher* RT error. The optimizer did its job; the extra freedom went
    somewhere the exam paper does not award marks for.

    Per band it is obvious where:

    | | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
    |---|---|---|---|---|---|---|---|---|
    | RT error, shelf | 10 % | 7 % | 13 % | 5 % | 0 % | 5 % | 14 % | **25 %** |
    | RT error, graphic EQ | **30 %** | 8 % | 13 % | 4 % | 7 % | 5 % | **9 %** | **9 %** |

    The shelf carries its error at the two ends, which is what a monotone tilt
    pinned at two endpoints has to do: the room holds a 2.8 s plateau across 63
    and 125 Hz and has already dropped to 2.5 s by 250 Hz, and a shelf cannot
    hold a plateau and then step down. The graphic EQ spends its extra freedom
    exactly where you would expect — the top two octaves go from 14 % and 25 % to
    9 % and 9 % — and then throws it all away in the bottom octave, which goes
    from 10 % to 30 %.

    That is not a defect of the design. It is the `frequency` table restated:
    cumulating the energy leaves the bottom octave with the least gradient of any
    band, and `"both"` improves that rather than curing it. The shelf gets 63 Hz
    nearly right by *not being free* — its low plateau is pinned by the whole
    midrange, which happens to sit near the room's bottom-octave value — while
    the graphic EQ, free to move that band on its own, is moved by a loss that
    barely sees it. **Extra parameters are only worth what the objective can
    supervise**, and a mean over eight octave bands is not the thing the
    optimizer was minimizing.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy decay and spectrograms

    The band metrics above are a summary; the surfaces the loss actually compares
    are below. The energy decay curves are the $f = 0$ edge of that surface, and
    the spectrograms are what it integrates.
    """)
    return


@app.cell
def _(fs, ir_init, ir_trained, pyFDN, rir):
    pyFDN.plot_edc(
        rir,
        ir_init,
        ir_trained,
        fs=fs,
        labels=["Target RIR", "FDN, untrained", "FDN, trained"],
        title="Energy decay curve",
    ).show()
    return


@app.cell
def _(fs, ir_init, pyFDN):
    pyFDN.plot_spectrogram(ir_init, fs, title="FDN, untrained")
    return


@app.cell
def _(fs, ir_trained, pyFDN):
    pyFDN.plot_spectrogram(ir_trained, fs, title="FDN, trained")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listen

    All three peak-normalized, so the A/B compares timbre rather than level.
    """)
    return


@app.cell
def _(fs, ir_init, ir_trained, mo, pyFDN, rir):
    mo.hstack(
        [
            pyFDN.labeled_audio("Target RIR", pyFDN.peak_normalize(rir), fs=fs),
            pyFDN.labeled_audio("Untrained", pyFDN.peak_normalize(ir_init), fs=fs),
            pyFDN.labeled_audio("Trained", pyFDN.peak_normalize(ir_trained), fs=fs),
        ],
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The deliverable — back to plain NumPy

    `extract_build` reads the trained model back out as an `FDNBuild` — no torch
    — with the two trained filters baked into the `post_delay` and `post_output`
    hooks as ordinary SOS banks. That is the thing `build_to_impz`,
    `build_to_flamo` or `process_fdn` would take, and it is what makes the
    assertions below a test of the FDN you would ship rather than of a torch
    graph.

    Nothing about a build remembers which design produced it, which is the point:
    `n_sections` biquads are `n_sections` biquads however they were chosen.
    """)
    return


@app.cell
def _(model, np, pyFDN, trained_eq_sos, trained_sos):
    trained_build = pyFDN.extract_build(model)

    print(f"post_delay:  {trained_build.post_delay.shape}")
    print(f"post_output: {trained_build.post_output.shape}")
    np.testing.assert_allclose(trained_build.post_delay, trained_sos, atol=1e-12)
    np.testing.assert_allclose(trained_build.post_output, trained_eq_sos, atol=1e-12)

    # Try this: run dry audio through the reverb you just trained.
    #   b = trained_build
    #   dry, _ = pyFDN.load_audio("synth_dry", fs=b.fs)
    #   wet = pyFDN.process_fdn(np.pad(dry, (0, 3 * int(b.fs))), b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test — the fit stays stable, and finds most of the room's decay from 1 s

    Six assertions, and none of them names a design. The first three are what the
    RT parametrization exists for: the extracted FDN still renders, the decay it
    settled on is a contractive filter at **every** frequency rather than only at
    the parameter points, and the filter has the number of sections its design
    promised. The last three are the fit itself — a decay that started flat and
    knew nothing about the room ends up substantially closer to it, in the mean
    and in every band the loss can resolve.

    Note that the stability check is on the *curve*, not on the parameters. A
    design's parameters are allowed to go negative — on the graphic EQ the
    Nyquist one does; what has to stay positive is the RT the designed filter
    actually realizes.
    """)
    return


@app.cell
def _(est_rt, n_sections, np, rt_curve, rt_init, rt_trained, trained_sos):
    assert np.all(np.isfinite(rt_trained)), "trained FDN did not render"
    assert np.all(np.isfinite(rt_curve)) and np.all(rt_curve > 0), (
        "the decay filter is not contractive at every frequency"
    )
    assert trained_sos.shape[0] == n_sections, "the design changed its mind"

    _err_init = np.abs(rt_init / est_rt - 1)
    _err = np.abs(rt_trained / est_rt - 1)
    print(f"RT error per band, untrained: {_err_init.round(3)}")
    print(f"RT error per band, trained:   {_err.round(3)}")
    assert _err.mean() < 0.3 * _err_init.mean(), "the fit barely moved the decay"
    assert _err.mean() < 0.15, "the trained decay is not close to the measurement"
    assert _err.max() < 0.35, "one band's decay is far off the measurement"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint

    Get the mean RT error below 9 %. Then say which of the three things you
    changed did it — the start, the objective, or the parametrization — and which
    band paid for it.

    Then break it on purpose, and read the failure off the loss curve rather than
    off the audio: `max_steps=20`, `lr=1.0`, `frequency="descending"`,
    `loss = pyFDN.MatchMelSpectrogram(rir)`. Each fails differently, and each
    failure is one of the four sections above.

    Where to go from here:

    - `example_process_fdn` — the same room, matched by hand with no gradients
    - `example_rir_to_fdn` — the same room, designed analytically from its
      measured decay and level. Both notebooks and this one end with the same
      three plots, on purpose
    - `example_train_colorless_FDN` — a different objective on the same
      machinery: flatness instead of a target response
    - `example_multislope_rir_to_fdn` — a coupled room, whose decay is not one
      exponential and cannot be fitted as one
    - `example_fdn_to_faust` — take the build you just trained to real time
    """)
    return


if __name__ == "__main__":
    app.run()
