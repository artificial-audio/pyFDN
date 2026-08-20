# gallery_category: FDN Design & Analysis
# gallery_title: Train an FDN to a room with four numbers
# gallery_description: The same gradient fit as "Train an FDN to match a room impulse response", with one first-order shelf for the decay and one for the output EQ -- four trained filter parameters instead of twenty.
# references: Concert_Hall_Impulse_Responses

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    # Training an FDN to a room with first-order filters

    **Train an FDN to match a room impulse response** fits the same measurement
    with a ten-band graphic EQ for the decay and another for the output EQ:
    twenty trained filter parameters. This notebook keeps everything else
    identical -- the same target, the same generic 1 s starting point, the same
    doubly-cumulated energy loss, the same 300 Adam steps -- and replaces both
    filters with **one first-order shelf each**.

    That is four numbers: a reverberation time at DC and one at Nyquist, and an
    output gain at DC and one at Nyquist. The target is again the Promenadikeskus
    concert hall in Pori, Finland, published at
    {pyFDN.paper_link("Concert_Hall_Impulse_Responses")}.

    The interesting part is that it does not do worse. On the metrics the fit
    never saw, four parameters land where twenty do -- and the twenty reach a
    visibly lower *loss* while getting there, which is the part worth thinking
    about.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is being trained

    | what | which parameter | how |
    |---|---|---|
    | **decay** | in-loop absorption | **trained** -- RT at DC and at Nyquist, `absorption_rt=(1.0, 1.0)` |
    | **colour** of the output | post EQ | **trained** -- gain at DC and at Nyquist, `post_eq_db=(0.0, 0.0)` |
    | fine structure of $\lvert H \rvert$, echo build-up | feedback matrix $A$ | **trained** -- on $SO(N)$ |
    | **level** | gains $b$, $c$ | **trained** |
    | dry path | $D$ | **trained** |
    | when the echoes fall | delays | **fixed** -- integer sample counts, no gradient to take |

    `trainable_from_build` picks the filter design from the *length* of what you
    hand it, because the parameter is the design: ten values are the ten bands of
    a graphic EQ, two are the two endpoints of a first-order shelf. Nothing else
    in the call changes.

    ## Two degrees of freedom, and what they cost

    A first-order shelf has exactly two once its crossover is fixed -- its value
    at DC and its value at Nyquist -- and pyFDN's design
    (`pyFDN.first_order_absorption`, Jot 2015) puts the transition at
    $f_s/8$, the midpoint of the bilinear-warped frequency axis. So the trained
    decay is a monotone tilt from one plateau to the other, and it *cannot* be
    anything else. There is no setting of the two numbers that gives the 250 Hz
    octave a longer tail than its neighbours.

    That is a real restriction and it is worth being explicit that it is one. It
    is also, for an absorptive room, most of what there is to say: air and
    material absorption both rise with frequency, so a measured RT curve is
    usually a tilt with a few dB of wobble on it, and the wobble is the part a
    fit is least able to distinguish from the fine structure it cannot predict
    anyway.

    ## Stability is free here, and the floor is about something else

    The graphic-EQ decay needs its parametrization to keep the loop contractive,
    because a fit that wants more energy will happily raise the loop gain past 1.
    The shelf inherits the same guarantee for the same reason -- a positive RT is
    a negative dB attenuation -- and adds one of its own: its pole is

    $$p \;=\; \frac{1 - t/\sqrt{k}}{1 + t/\sqrt{k}}, \qquad t = \tan(2\pi f_c/f_s),$$

    which lies inside the unit circle for **any** pair of endpoint gains, since
    $t > 0$ below $f_s/4$ and $\sqrt{k} > 0$ always. Eleven cascaded biquads
    designed by least squares have no such closed form.

    What the RT still needs a floor for is the *sign*: a gradient step that puts
    an endpoint at or below zero turns $-60 d_i / (\mathrm{RT} f_s)$ from an
    attenuation into a gain. `pyFDN.train.shelf` floors it exactly as
    `pyFDN.train.decay` does -- softplus, one round trip of the longest delay
    line, a knee one floor wide -- so a band that dips across zero still has a
    gradient to come back on. The last cell tests that the trained endpoints came
    out positive.

    ## The loss is the sibling notebook's

    `pyFDN.MatchCumulativeEnergy(rir, window=1024, power=0.5, frequency="both")`:
    short-time energy cumulated backwards in time (Schroeder integration, i.e.
    the decay) and along frequency (which does the job of splitting into octave
    bands, without band edges). Why a spectrogram distance cannot see a decay at
    all, why the compression is a power rather than a logarithm, and why the
    frequency cumulation has to run both ways, are all worked through in **Train
    an FDN to match a room impulse response** and not repeated here. The same
    `dtype=torch.float64` and `alias_decay_db=60` apply, for the same reason:
    backward integration reads the quietest samples in the buffer, and in float32
    those samples are the anti-aliasing reconstruction's rounding noise.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The target

    Trimmed to the onset and normalized to unit energy, exactly as in the two
    sibling notebooks so all three are comparable.
    """)
    return


@app.cell
def _(np, pyFDN):
    fs = 48000
    rir, _file_fs = pyFDN.load_audio("s3_r4_o", fs=fs)
    rir = rir[int(np.argmax(np.abs(rir))) :]
    rir = rir / np.linalg.norm(rir)
    rir_len = len(rir)

    print(f"target RIR: {rir_len} samples ({rir_len / fs:.2f} s) at {fs} Hz")
    return fs, rir, rir_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The yardstick

    Octave-band RT and initial level of the *target*, by Schroeder backward
    integration -- computed after the fit and fed to nothing. Note the shape: 2.8 s
    at the bottom, 1.2 s at 8 kHz, and a monotone fall in between apart from a
    single 63/125 Hz plateau. That is a shelf's shape, which is the reason this
    notebook works.
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    est_rt, f_centre = pyFDN.estimate_rt_bands(rir, fs)
    est_level, _ = pyFDN.estimate_initial_level_bands(rir, est_rt, fs)

    print(f"bands (Hz):   {f_centre.round(0)}")
    print(f"measured RT:  {est_rt.round(2)}")
    return est_level, est_rt, f_centre


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- the same reverberator that knows nothing about the room

    `fdn_build_gallery` builds the entire starting FDN in one call -- a random
    orthogonal feedback matrix, sixteen normalized gains, no dry path, and
    per-delay first-order absorption for a **flat 1 s decay**. Its `rt=` /
    `rt_nyquist=` absorption is `pyFDN.first_order_absorption`, the very filter
    this notebook trains, so `init_build` is a complete FDN rather than a
    scaffold: nothing has to be patched onto it afterwards, and the untrained
    render further down is just this build with the energy match applied.

    Only the delays are sampled separately, because the gallery's own delay
    sampling does not expose `distribution="geometric"` or `coprime=True`; they
    are passed straight in.

    The one number this notebook hands the trainer that the ten-band notebook
    spells differently: `init_rt` is `(1.0, 1.0)`, two shelf endpoints, rather
    than `np.full(10, 1.0)`.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    num_delays = 16
    delays = pyFDN.sample_delay_lengths(
        num_delays,
        (700, 2500),
        distribution="geometric",
        coprime=True,
        sort=True,
        rng=1,
    )

    # The whole starting FDN, from one call: a random orthogonal feedback matrix,
    # normalized gains, no dry path, and per-delay first-order absorption for a
    # flat 1 s decay -- which is exactly the filter this notebook then trains.
    #
    # rng=0 is not arbitrary. The orthogonal training parametrization lives on
    # SO(N), so it hands a det<0 matrix back with its last column flipped: not
    # the matrix you asked for. This seed lands in SO(N), and the assertion
    # below is what says so.
    init_build = pyFDN.fdn_build_gallery(
        delays=delays,
        fs=fs,
        io_type="normalized",
        direct_gain=0.0,
        rt=1.0,
        rt_nyquist=1.0,
        rng=0,
    )
    assert np.linalg.det(init_build.A) > 0, "feedback matrix is not in SO(N)"

    init_rt = np.array([1.0, 1.0])  # the two shelf endpoints handed to the trainer

    print(f"delays (samples): {init_build.delays}")
    print(
        f"absorption:       {init_build.filters.shape} -- flat 1 s, one biquad per line"
    )
    return delays, init_build, init_rt, num_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- the model

    `absorption_rt=(1.0, 1.0)` builds the in-loop filter as
    `pyFDN.first_order_absorption` made differentiable in the RT, and
    `post_eq_db=(0.0, 0.0)` builds the output filter as a flat
    `pyFDN.first_order_shelving_eq`. Both map onto a **one-section**
    `(1, 6, N)` SOS bank, which is what `extract_build` reads back out.

    Then the single adjustment the measurement is allowed to make before the
    optimizer starts: **the overall energy**, one scalar on the output gain, so
    the initial FDN and the target hold the same total energy in the training
    window. It is a level match, not a decay match.
    """)
    return


@app.cell
def _(fs, init_build, init_rt, np, pyFDN, rir):
    import torch

    nfft = 2**16  # 1.37 s at 48 kHz -- longer than the decay being fitted

    model = pyFDN.trainable_from_build(
        init_build,
        # everything with a gradient: A, b, c, D, the decay and the output EQ
        trainable=pyFDN.Trainable(absorption=True, direct=True, post_eq=True),
        absorption_rt=init_rt,  # 2 values -> a first-order shelf, not a GEQ
        post_eq_db=(0.0, 0.0),  # likewise: a flat first-order shelf
        nfft=nfft,
        device="cpu",
        # no alias_decay_db: this FDN decays, so its poles are well inside the
        # unit circle and the FFT evaluation is sound -- see the note below
        dtype=torch.float64,
    )

    # the only thing the target tells the initial model: how loud it is
    _ir = np.asarray(pyFDN.model_response(model).h.detach()).reshape(-1)
    energy_gain = float(np.linalg.norm(rir[:nfft]) / np.linalg.norm(_ir))
    with torch.no_grad():
        pyFDN.param(model, "output_gain").raw().mul_(energy_gain)

    # ParamRef.shape is the MAPPED value -- the SOS bank the system runs. What
    # the optimizer steps is .raw(), and for the two filters that is the pair.
    for _p in pyFDN.params(model):
        print(f"{_p}  raw {tuple(_p.raw().shape)}")
    print(f"\nenergy match: output gain x {energy_gain:.2f}")
    return energy_gain, model, nfft, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- train

    Identical to the ten-band notebook, down to the seed.
    """)
    return


@app.cell
def _(model, pyFDN, rir, torch):
    loss = pyFDN.MatchCumulativeEnergy(rir, window=1024, power=0.5, frequency="both")

    log = pyFDN.train_fdn(
        model,
        loss,
        optimizer="adam",
        max_steps=300,
        lr=3e-2,
        patience=100,
        device="cpu",
        dtype=torch.float64,
        rng=0,
    )
    trained_rt = pyFDN.param(model, "absorption").raw().detach().numpy().copy()
    trained_eq_db = pyFDN.param(model, "post_eq").raw().detach().numpy().copy().ravel()
    trained_build = pyFDN.extract_build(model)

    print(
        f"ran {log.steps_run} steps, loss {log.train_loss[0]:.4g} -> "
        f"{log.train_loss[-1]:.4g} "
        f"({100 * (1 - log.train_loss[-1] / log.train_loss[0]):.0f}% down)"
    )
    print(
        f"\ndecay:     RT {trained_rt[0]:.2f} s at DC, {trained_rt[1]:.2f} s at Nyquist"
    )
    print(
        f"output EQ: {trained_eq_db[0]:+.1f} dB at DC, "
        f"{trained_eq_db[1]:+.1f} dB at Nyquist"
    )
    print(
        f"\nin-loop filter: {trained_build.filters.shape} -- one biquad per delay line"
    )
    print(f"output filter:  {trained_build.post_eq.shape} -- one biquad")
    return log, loss, trained_build, trained_eq_db, trained_rt


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
    ## The two filters that are the answer

    Four numbers, but the thing to look at is the **curve** each pair designs,
    because that is what the FDN runs. The decay is plotted as reverberation time
    against frequency -- read off the fitted filter's gain per sample,
    $\mathrm{RT}(f) = -60 / (f_s \cdot 20\log_{10} g(f))$, which is
    delay-line-independent because the decay is homogeneous.

    The shelf's low plateau settles at 2.4 s, which is where the *middle* of the
    spectrum is; the room's 2.8 s bottom octave pulls at it but cannot bend it,
    because a shelf that reached 2.8 s at DC would have to pass through 2.8 s at
    250 Hz too, and the loss would pay for that everywhere. So the trained
    endpoint is a compromise the parametrization forced, and it is the honest one.

    At the top the endpoint reads 0.21 s against a measured 1.2 s at 8 kHz, which
    looks alarming until you notice that the endpoint is at **24 kHz** -- an
    octave and a half above the highest band the estimator reports, and well into
    where the recording has nothing left. What the FDN actually does at 8 kHz is
    the curve, not the endpoint: 0.66 s there, and the rendered FDN measures
    0.89 s in that octave. As in the ten-band notebook: at the edges of the
    design, read the filter, not the parameter.
    """)
    return


@app.cell
def _(delays, est_rt, f_centre, fs, go, init_rt, np, pyFDN, trained_build):
    def _rt_curve(sos):
        """Reverberation time vs frequency implied by a homogeneous decay filter."""
        angles, magnitude = pyFDN.sos_gain_per_sample_curves(sos, delays, 512)
        freqs = angles / np.pi * (fs / 2)
        return freqs, -60.0 / (fs * 20.0 * np.log10(magnitude[:, 0]))

    _f, _rt_trained_curve = _rt_curve(trained_build.filters)
    _, _rt_init_curve = _rt_curve(
        pyFDN.first_order_absorption(init_rt[0], init_rt[1], delays, fs)
    )

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_f, y=_rt_init_curve, name="initial (flat 1 s)", line={"dash": "dot"}
        )
    )
    _fig.add_trace(go.Scatter(x=_f, y=_rt_trained_curve, name="trained shelf"))
    _fig.add_trace(
        go.Scatter(
            x=np.array([1.0, fs / 2]),
            y=np.asarray([_rt_trained_curve[0], _rt_trained_curve[-1]]),
            name="the two trained endpoints",
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
            "range": [1, np.log10(fs / 2)],
        },
        yaxis={"title": "RT (s)", "rangemode": "tozero"},
        template="plotly_white",
        height=380,
    )
    _fig.show()

    print(
        f"shelf RT at the octave centres (s): {np.interp(f_centre, _f, _rt_trained_curve).round(2)}"
    )
    print(f"measured, octave bands (s):         {est_rt.round(2)}")
    return


@app.cell
def _(fs, go, np, pyFDN, trained_build, trained_eq_db):
    _probe = np.logspace(0, np.log10(fs / 2), 400)
    _db, _, _ = pyFDN.probe_sos(trained_build.post_eq[:, :, 0], _probe, 2**14, fs)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_probe,
            y=np.zeros_like(_probe),
            name="initial (flat)",
            line={"dash": "dot"},
        )
    )
    _fig.add_trace(go.Scatter(x=_probe, y=_db[:, 0], name="trained shelf"))
    _fig.add_trace(
        go.Scatter(
            x=np.array([1.0, fs / 2]),
            y=trained_eq_db,
            name="the two trained endpoints",
            mode="markers",
            marker={"size": 11, "symbol": "diamond"},
        )
    )
    _fig.update_layout(
        title="The output EQ parameter: shelving gain",
        xaxis={
            "title": "Frequency (Hz)",
            "type": "log",
            "range": [1, np.log10(fs / 2)],
        },
        yaxis={"title": "Gain (dB)"},
        template="plotly_white",
        height=380,
    )
    _fig.show()

    print(f"trained output EQ (dB): {trained_eq_db.round(1)} at DC and Nyquist")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render both FDNs

    Both down the **same** path -- `build_to_flamo` then `flamo_time_response` at
    `nfft = 2**19` (10.9 s, nearly three times the target, so the FFT
    wrap-around the training render had to be protected against is not a factor
    here) -- and both measured with the same estimators. Two FDNs being compared
    on a metric should not be reaching it through two different renderers.

    Why the round trip through `FDNBuild` rather than just rendering the trained
    model? Because `nfft` is **structural** in FLAMO -- it fixes the frequency
    grid, the delay phase ramps and the alias envelope of every module at
    construction, and there is no setter. The model was deliberately built at
    `nfft = 2**16`, 1.37 s: that is the window the loss looks at, and 2**19 would
    have been eight times the FFT work on every one of the 300 steps. But the
    estimators below have to see the whole 3.9 s tail, and Schroeder integration
    over a 1.37 s window under-reads a 2.4 s decay -- on this FDN by up to 10 %
    in the bottom two octaves, which is larger than the error the table is trying
    to measure. So the render has to happen at a different `nfft`, and that means
    rebuilding.

    `extract_build` -> `build_to_flamo` **is** that rebuild; `FDNBuild` is just
    the plain-numpy description in the middle of it. That it is also the
    deliverable -- the thing `process_fdn` or `build_to_faust` would take -- is
    the bonus: rendering through it is what makes the last cell's assertions a
    test of the FDN you would actually ship, rather than of a torch graph.

    Worth being clear about one thing, because "outside the recursion" invites
    the wrong reading: the output EQ **is** an ordinary member of the FLAMO
    graph. `assemble_fdn_core` wires it in after the output gain as a leaf named
    `output_filter`, the same optimizer steps it as steps the feedback matrix,
    and `extract_build` reads it back out as `build.post_eq`. Outside the
    *recursion* is a statement about where it sits in the signal flow -- which is
    exactly why it can shape the spectrum without touching the decay -- not about
    it being applied separately afterwards.

    What cannot render it is `build_to_impz`, whose `process_fdn` block
    simulation has no output-filter slot; it raises on a build with `post_eq`
    rather than silently dropping it. On this FDN the two renderers agree to
    -69 dB and to three decimals on every band metric below, so using the FLAMO
    one for both costs nothing.
    """)
    return


@app.cell
def _(energy_gain, fs, init_build, np, pyFDN, rir_len, trained_build):
    import dataclasses

    # the untrained FDN: init_build already carries the flat 1 s absorption, so
    # "untrained" is that build plus the energy match -- exactly what the
    # optimizer was handed, with nothing reconstructed by hand
    start_build = dataclasses.replace(init_build, C=init_build.C * energy_gain)

    def _render(build):
        """One render path for both FDNs, output EQ included."""
        model = pyFDN.build_to_flamo(build, nfft=2**19, device="cpu")
        return np.asarray(pyFDN.flamo_time_response(model, fs=fs)).reshape(-1)[:rir_len]

    ir_init = _render(start_build)
    ir_trained = _render(trained_build)

    rt_init, _ = pyFDN.estimate_rt_bands(ir_init, fs)
    rt_trained, _ = pyFDN.estimate_rt_bands(ir_trained, fs)
    level_init, _ = pyFDN.estimate_initial_level_bands(ir_init, rt_init, fs)
    level_trained, _ = pyFDN.estimate_initial_level_bands(ir_trained, rt_trained, fs)
    return (
        dataclasses,
        ir_init,
        ir_trained,
        level_init,
        level_trained,
        rt_init,
        rt_trained,
        start_build,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Four parameters against twenty

    The same two estimators, applied to the rendered FDNs, and the ten-band
    notebook's own row for comparison -- same target, same delays, same matrix,
    same loss, same 300 steps, only the filter design differing:

    | | mean RT error | level offset | level shape | final loss |
    |---|---|---|---|---|
    | untrained | 52.4 % | -1.3 dB | 1.71 dB | 0.1319 |
    | ten-band graphic EQ | 9.6 % | +0.3 dB | 0.77 dB | **0.00373** |
    | first-order shelves | 10.7 % | +0.3 dB | 0.82 dB | 0.00555 |

    **Read those two rows as a tie, not a ranking.** A single pair of numbers
    invites the conclusion that ten bands are worth about a point of RT error,
    and that conclusion does not survive changing one thing that has nothing to
    do with the filters: the random feedback matrix. Re-running both fits with a
    different `rng=` on `fdn_build_gallery`, everything else fixed:

    | matrix seed | shelf RT error | ten-band RT error |
    |---|---|---|
    | 0 (the fit above) | 10.7 % | 9.6 % |
    | 1 | 9.3 % | 12.3 % |
    | 4 | 10.1 % | -- |

    The ordering flips between seed 0 and seed 1, and the shelf's own spread
    across three seeds (9.3-10.7 %) is as wide as its gap from the ten-band fit
    on either one. So the honest statement is that **two numbers per filter reach
    the same accuracy as ten**, not that they beat them.

    What does *not* move with the seed is the **loss**: the graphic EQ lands at
    0.0036-0.0037 on both seeds it was run on and the shelf at 0.0055-0.0074 on
    all three. So the eighteen
    extra parameters reliably fit the training objective better while landing no
    closer to the room. That is the finding worth keeping, and it is the ordinary
    one: the extra freedom goes into the fine structure of one particular
    measurement, which is the part of a room that does not generalize. The shelf
    cannot chase it, so it does not.

    Per band the two designs fail in different places, and that part is
    structural rather than seed noise:

    | | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
    |---|---|---|---|---|---|---|---|---|
    | ten-band RT error | 24 % | 6 % | 14 % | 3 % | 7 % | 5 % | 9 % | 8 % |
    | shelf RT error | 14 % | 9 % | 9 % | 3 % | 4 % | 6 % | 16 % | 26 % |

    The graphic EQ's worst band is the bottom octave, where the cumulated loss
    has the least to say and ten free bands can drift. The shelf's worst is the
    top, where its Nyquist endpoint is extrapolating past anything the recording
    contains -- a restriction, not a drift, and one you can read off the design.

    One thing that is not an achievement in either column: the level *offset* was
    set by the energy match before the optimizer ran. The level *shape* is,
    because nothing in the FDN proper can bend it -- that is the trained output
    EQ, and here it is two numbers rather than ten.

    Where the shelf wins outright is cost. One biquad in the loop instead of
    eleven, and one on the output instead of eleven: on the machine these notes
    were written on, the whole notebook runs in about 80 s against about 300 s
    for the ten-band one, almost all of it the 300 training steps.
    """)
    return


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
    mo.md(r"""
    ## Energy decay and spectrograms

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
def _(fs, pyFDN, rir):
    pyFDN.plot_spectrogram(rir, fs, title="Target RIR")
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
    ## What is left for a designed EQ to fix

    The analytic pipeline ends by designing an output GEQ from the residual
    between the target's band levels and the FDN's. That filter is now two
    trained numbers, so the same residual is a test of it: whatever a `design_geq`
    call would still be asked to correct is what the fit did not manage.

    It comes out at 0.82 dB of shape -- and the ten-band fit's residual is
    0.77 dB, which is the same answer reached from two numbers instead of ten.
    A designed GEQ on what is left would still buy that 0.8 dB in either case,
    and nothing stops you from running one afterwards. What the fit buys is that
    the EQ was chosen *while* the decay and the matrix were still moving, rather
    than as a correction applied to something already fixed.
    """)
    return


@app.cell
def _(est_level, f_centre, level_trained, np, pyFDN, trained_eq_db):
    residual_db = pyFDN.lin_to_db(est_level) - pyFDN.lin_to_db(level_trained)
    print(f"bands (Hz):         {f_centre.round(0)}")
    print(f"residual left (dB): {residual_db.round(1)}")
    print(
        f"\ntrained output EQ: {trained_eq_db.round(1)} dB at DC and Nyquist"
        f"\nresidual: {residual_db.mean():+.1f} dB offset, "
        f"{np.abs(residual_db - residual_db.mean()).mean():.2f} dB of shape"
    )
    return (residual_db,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Listen

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
    ## Test: four numbers, one biquad each, and most of the room's decay

    Six assertions. The first three are what the parametrization exists for: the
    extracted FDN still renders, both RT endpoints stayed positive (a negative
    one is an amplifying loop), and each filter really is a single biquad. The
    last three are the fit -- a decay that started flat and knew nothing about the
    room ends up substantially closer to it, in the mean and in every band the
    loss can resolve.
    """)
    return


@app.cell
def _(est_rt, np, rt_init, rt_trained, trained_build, trained_rt):
    assert np.all(np.isfinite(rt_trained)), "trained FDN did not render"
    assert np.all(trained_rt > 0), "an RT endpoint left the stable region"
    assert trained_build.filters.shape[0] == 1, "the decay is not one biquad"
    assert trained_build.post_eq.shape[0] == 1, "the output EQ is not one biquad"

    _err_init = np.abs(rt_init / est_rt - 1)
    _err = np.abs(rt_trained / est_rt - 1)
    print(f"RT error per band, untrained: {_err_init.round(3)}")
    print(f"RT error per band, trained:   {_err.round(3)}")
    assert _err.mean() < 0.3 * _err_init.mean(), "the fit barely moved the decay"
    assert _err.mean() < 0.15, "the trained decay is not close to the measurement"
    assert _err.max() < 0.35, "one band's decay is far off the measurement"
    return


if __name__ == "__main__":
    app.run()
