# gallery_category: FDN Design & Analysis
# gallery_title: Train an FDN to a room with four numbers
# gallery_description: Fit a 16-line FDN to a measured concert hall by gradient descent, with one first-order shelf for the decay and one for the output EQ -- four trained filter parameters, one biquad each.
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

    A 16-line feedback delay network is fitted to a measured room by gradient
    descent: the feedback matrix, the input and output gains, the dry path, the
    in-loop absorption and the output EQ all move together under a single loss
    on the doubly-cumulated energy. Both filters are **one first-order shelf**.

    That is four numbers for everything the FDN does with frequency: a
    reverberation time at DC and one at Nyquist, and an output gain at DC and
    one at Nyquist. The starting point knows nothing about the room -- a flat
    1 s decay and a random orthogonal matrix -- and the target is the
    Promenadikeskus concert hall in Pori, Finland, published at
    {pyFDN.paper_link("Concert_Hall_Impulse_Responses")}.

    The interesting part is how little those four numbers give up. The decay
    they can describe is a monotone tilt and nothing else, which is a real
    restriction; it is also most of what an absorptive room has to say.
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
    attenuation into a gain. `pyFDN.train.filters` floors it the same way
    for every design -- softplus, one round trip of the longest delay
    line, a knee one floor wide -- so a band that dips across zero still has a
    gradient to come back on. The last cell tests that the trained endpoints came
    out positive.

    ## The loss

    `pyFDN.MatchCumulativeEnergy(rir, window=1024, power=0.5, frequency="both")`
    takes the short-time energy of both signals and integrates it twice:
    backwards in time, which is Schroeder integration and therefore the decay,
    and along frequency, which does the job of splitting into octave bands
    without having to put edges anywhere. A plain spectrogram distance will not
    do, because two rooms with the same decay have uncorrelated fine structure:
    against detail a fit cannot predict, silence scores better than the right
    amount of the wrong detail. Cumulating closes that trap in both axes.

    `power=0.5` compresses a surface that spans the whole dynamic range of the
    decay, by comparing amplitudes rather than energies. A logarithm is the
    obvious alternative and is worse here -- it turns the silence *below* the
    response into an unbounded penalty. `frequency="both"` averages the two
    cumulation directions; cumulating one way only leaves the bottom octave with
    almost no gradient, and the fit abandons it.

    `dtype=torch.float64` throughout, because backward integration reads the
    quietest samples in the buffer, and in float32 those samples are rounding
    noise rather than signal.
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

    Trimmed to the onset and normalized to unit energy.
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
    at the bottom, 1.2 s at 8 kHz, and a fall in between that never reverses,
    with a plateau across 63/125 Hz and another across 500 Hz/1 kHz. That is a
    shelf's shape, which is the reason this notebook works.
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
    render below is this FDN with only the energy match applied.

    Only the delays are sampled separately, because the gallery's own delay
    sampling does not expose `distribution="geometric"` or `coprime=True`; they
    are passed straight in.

    `init_rt` is `(1.0, 1.0)` -- the two shelf endpoints, both at 1 s, which is
    that same flat decay written in the form the trainer takes.
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
    `(1, 6, N)` SOS bank -- that mapped value is the filter the FDN actually
    runs, and `pyFDN.param(model, ...).value()` reads it back out.

    `nfft = 2**17` is 2.73 s at 48 kHz, and it is chosen once and used for
    everything. Both jobs it has to do put a floor under it. The loss compares
    this window against the target, so it has to hold the decay being fitted;
    and the *same* render is what the octave-band estimators at the bottom
    measure, where Schroeder integration over a window shorter than the decay
    under-reads it. 2.73 s clears both: the target has only -72 dB of its energy
    left after it, and the band RTs come out equal to three decimals against a
    render four times as long.

    That is what lets the trained model be measured directly, rather than
    exported to an `FDNBuild` and re-rendered at some larger `nfft`. The reason
    such a round trip is otherwise needed is that `nfft` is **structural** in
    FLAMO -- it fixes the frequency grid, the delay phase ramps and the alias
    envelope of every module at construction, and there is no setter -- so a
    render at a different length means rebuilding. Sizing it correctly once
    costs a factor of two on every training step and removes the rebuild.

    Then the single adjustment the measurement is allowed to make before the
    optimizer starts: **the overall energy**, one scalar on the output gain, so
    the initial FDN and the target hold the same total energy in the training
    window. It is a level match, not a decay match. The untrained render is
    taken immediately after it, in this same cell, because `train_fdn` steps
    `model` in place -- once the next cell has run there is no "before" left.
    """)
    return


@app.cell
def _(fs, init_build, init_rt, np, pyFDN, rir, rir_len):
    import torch

    nfft = 2**17  # 2.73 s at 48 kHz -- long enough for the loss and the metrics

    model = pyFDN.trainable_from_build(
        init_build,
        # everything with a gradient: A, b, c, D, the decay and the output EQ
        trainable=pyFDN.Trainable(absorption=True, direct=True, post_eq=True),
        absorption_rt=init_rt,  # 2 values -> a first-order shelf, not a GEQ
        post_eq_db=(0.0, 0.0),  # likewise: a flat first-order shelf
        nfft=nfft,
        device="cpu",
        # no alias_decay_db: this FDN decays, so its poles are well inside the
        # unit circle and the FFT evaluation is sound
        dtype=torch.float64,
    )

    def render(m):
        """The FDN's impulse response, straight out of the FLAMO model."""
        return np.asarray(pyFDN.model_response(m).h.detach()).reshape(-1)[:rir_len]

    # the only thing the target tells the initial model: how loud it is
    energy_gain = float(np.linalg.norm(rir[:nfft]) / np.linalg.norm(render(model)))
    with torch.no_grad():
        pyFDN.param(model, "output_gain").raw().mul_(energy_gain)

    # the untrained FDN, before the optimizer touches it -- and the flat 1 s
    # absorption it starts from, as the SOS bank the model runs
    ir_init = render(model)
    init_sos = pyFDN.param(model, "absorption").value().detach().numpy().copy()

    # ParamRef.shape is the MAPPED value -- the SOS bank the system runs. What
    # the optimizer steps is .raw(), and for the two filters that is the pair.
    for _p in pyFDN.params(model):
        print(f"{_p}  raw {tuple(_p.raw().shape)}")
    print(f"\nenergy match: output gain x {energy_gain:.2f}")
    return energy_gain, init_sos, ir_init, model, nfft, render, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- train

    300 Adam steps at `lr=3e-2`. The trained response is rendered at the end of
    the same cell, out of the same model, through the same `render` the
    untrained one went through: two FDNs being compared on a metric should not
    be reaching it by two different routes.
    """)
    return


@app.cell
def _(model, pyFDN, render, rir, torch):
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
    trained_sos = pyFDN.param(model, "absorption").value().detach().numpy().copy()
    trained_eq_sos = pyFDN.param(model, "post_eq").value().detach().numpy().copy()
    ir_trained = render(model)

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
    print(f"\nin-loop filter: {trained_sos.shape} -- one biquad per delay line")
    print(f"output filter:  {trained_eq_sos.shape} -- one biquad")
    return (
        ir_trained,
        log,
        loss,
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
    ## The two filters that are the answer

    Four numbers, but the thing to look at is the **curve** each pair designs,
    because that is what the FDN runs. The decay is plotted as reverberation time
    against frequency -- read off the fitted filter's gain per sample,
    $\mathrm{RT}(f) = -60 / (f_s \cdot 20\log_{10} g(f))$, which is
    delay-line-independent because the decay is homogeneous.

    The shelf's low plateau settles at 2.5 s, which is where the *middle* of the
    spectrum is; the room's 2.8 s bottom octave pulls at it but cannot bend it,
    because a shelf that reached 2.8 s at DC would have to pass through 2.8 s at
    250 Hz too, and the loss would pay for that everywhere. So the trained
    endpoint is a compromise the parametrization forced, and it is the honest one.

    At the top the endpoint reads 0.20 s against a measured 1.2 s at 8 kHz, which
    looks alarming until you notice that the endpoint is at **24 kHz** -- an
    octave and a half above the highest band the estimator reports, and well into
    where the recording has nothing left. What the FDN actually does at 8 kHz is
    the curve, not the endpoint: 0.65 s there, and the rendered FDN measures
    0.89 s in that octave. At the edges of the design, read the filter, not the
    parameter.
    """)
    return


@app.cell
def _(delays, est_rt, f_centre, fs, go, init_sos, np, pyFDN, trained_sos):
    def _rt_curve(sos):
        """Reverberation time vs frequency implied by a homogeneous decay filter."""
        angles, magnitude = pyFDN.sos_gain_per_sample_curves(sos, delays, 512)
        freqs = angles / np.pi * (fs / 2)
        return freqs, -60.0 / (fs * 20.0 * np.log10(magnitude[:, 0]))

    _f, _rt_trained_curve = _rt_curve(trained_sos)
    _, _rt_init_curve = _rt_curve(init_sos)

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
def _(fs, go, np, pyFDN, trained_eq_db, trained_eq_sos):
    _probe = np.logspace(0, np.log10(fs / 2), 400)
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
    ## Measuring both FDNs

    Both responses already exist: `ir_init` from the model before the optimizer
    ran, `ir_trained` from the same model after. Both left FLAMO through the
    same `render` at the same `nfft`, so what follows compares two FDNs rather
    than two renderers.

    Worth being clear about one thing, because "outside the recursion" invites
    the wrong reading: the output EQ **is** an ordinary member of the FLAMO
    graph. `assemble_fdn_core` wires it in after the output gain as a leaf named
    `output_filter`, and the same optimizer steps it as steps the feedback
    matrix -- so it is in the render above with everything else. Outside the
    *recursion* is a statement about where it sits in the signal flow, which is
    exactly why it can shape the spectrum without touching the decay, not about
    it being applied separately afterwards.
    """)
    return


@app.cell
def _(fs, ir_init, ir_trained, pyFDN):
    rt_init, _ = pyFDN.estimate_rt_bands(ir_init, fs)
    rt_trained, _ = pyFDN.estimate_rt_bands(ir_trained, fs)
    level_init, _ = pyFDN.estimate_initial_level_bands(ir_init, rt_init, fs)
    level_trained, _ = pyFDN.estimate_initial_level_bands(ir_trained, rt_trained, fs)
    return level_init, level_trained, rt_init, rt_trained


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How close four numbers get

    The two estimators applied to the rendered FDNs, against the measurement
    neither of them saw:

    | | mean RT error | level offset | level shape | final loss |
    |---|---|---|---|---|
    | untrained (flat 1 s) | 52.4 % | -1.3 dB | 1.71 dB | 0.0932 |
    | trained | **9.9 %** | +0.3 dB | 0.80 dB | **0.00392** |

    That figure is a property of the fit rather than of one lucky matrix:
    re-running with a different seed on the random orthogonal feedback matrix,
    everything else fixed, gives 9.9 %, 10.2 % and 10.0 % on seeds 0, 1 and 4 of
    `fdn_build_gallery`.

    Per band it fails where the design says it must:

    | | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
    |---|---|---|---|---|---|---|---|---|
    | RT error | 10 % | 6 % | 13 % | 4 % | 0 % | 5 % | 15 % | 26 % |

    The middle comes out almost exact and both ends carry the error, which is
    what a monotone tilt pinned at two endpoints has to do. At the bottom the
    room holds a 2.8 s plateau across 63 and 125 Hz and has already dropped to
    2.5 s by 250 Hz; a shelf cannot hold a plateau and then step down, so it
    splits the difference. The top octave is worse, for a different reason: its
    Nyquist endpoint is extrapolating an octave and a half past the highest band
    the estimator reports, into a part of the spectrum the recording barely
    contains. Both are restrictions you can read off the design rather than
    drift the optimizer fell into.

    One thing in that table which is not an achievement: the level *offset* was
    set by the energy match before the optimizer ran. The level *shape* is,
    because nothing in the FDN proper can bend it -- that is the trained output
    EQ, and here it is two numbers.

    The cost side is where a shelf is unambiguously ahead. One biquad in the
    loop per delay line and one on the output, designed in closed form from two
    numbers each, against the eleven-section least-squares design a ten-band
    graphic EQ needs in the same slots.
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

    It comes out at 0.80 dB of shape -- what a two-parameter shelf cannot
    reach, and what a designed GEQ on top would still buy. Nothing stops you
    from running one afterwards. What the fit buys is that the EQ was chosen
    *while* the decay and the matrix were still moving, rather than as a
    correction applied to something already fixed.
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
    trained FDN still renders, both RT endpoints stayed positive (a negative one
    is an amplifying loop), and each filter really is a single biquad. The
    last three are the fit -- a decay that started flat and knew nothing about the
    room ends up substantially closer to it, in the mean and in every band the
    loss can resolve.
    """)
    return


@app.cell
def _(est_rt, np, rt_init, rt_trained, trained_eq_sos, trained_rt, trained_sos):
    assert np.all(np.isfinite(rt_trained)), "trained FDN did not render"
    assert np.all(trained_rt > 0), "an RT endpoint left the stable region"
    assert trained_sos.shape[0] == 1, "the decay is not one biquad"
    assert trained_eq_sos.shape[0] == 1, "the output EQ is not one biquad"

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
