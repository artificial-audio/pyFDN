# gallery_category: FDN Design & Analysis
# gallery_title: Train an FDN to match a room impulse response
# gallery_description: Fit every parameter of an FDN -- the decay included -- to a measured RIR by gradient descent on a mel multi-resolution spectral loss.
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
    # Training an FDN to match a measured room

    **Convert a room impulse response into an FDN** designs the whole reverberator analytically: octave-band RT and level are estimated from the measurement and turned into filters. This notebook keeps that design only as a starting point and **fits every parameter by gradient descent** -- the decay included -- against the same measured RIR, the Promenadikeskus concert hall in Pori, Finland, published at {pyFDN.paper_link("Concert_Hall_Impulse_Responses")}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is being trained

    | what | which parameter | how |
    |---|---|---|
    | **decay**, per octave band | in-loop absorption | **trained** -- as reverberation time in seconds, `absorption_rt=` |
    | **colour** -- fine structure of $\lvert H \rvert$, echo build-up | feedback matrix $A$ | **trained** -- on $SO(N)$ |
    | **level** and its coarse tilt | gains $b$, $c$ | **trained** |
    | dry path | $D$ | **trained** |
    | when the echoes fall | delays | **fixed** -- integer sample counts, no gradient to take |

    Everything with a gradient is in the fit. Putting the decay in there took three things that a colour-only fit does not need, and each of them is a measurement, not a preference.

    ## 1. A parametrization the decay cannot escape

    Training the absorption filter's *coefficients* does not work, and not for want of tuning. The untrained FDN starts far too quiet, and the cheapest direction for any loss is more loop gain -- so a raw SOS cascade, with nothing holding its poles inside the unit circle, walks straight out of it. At `lr=3e-2` and at `lr=1e-3` alike, the fit below diverges within fifty steps, the loss ends four orders of magnitude *above* where it started, and the extracted FDN renders as `nan`.

    `absorption_rt=` replaces the filter with the same graphic-EQ design (`pyFDN.absorption_geq`, Schlecht and Habets 2017) rewritten as a differentiable function of the reverberation time per band:

    $$\mathrm{RT}_k \;\longrightarrow\; \underbrace{-60\,d_i / (\mathrm{RT}_k f_s)}_\text{dB per round trip} \;\longrightarrow\; \text{GEQ command gains} \;\longrightarrow\; \text{biquads}$$

    A positive RT means a negative dB attenuation, which means a contractive loop -- for **every** value the parameter can take. The least-squares fit that `design_geq` runs at each call is linear in its target, so it collapses into one constant matrix and the whole chain is closed-form differentiable; no iterative filter design inside the training loop. The trained number is then the reverberation time itself, which is also the number worth plotting.

    ## 2. A loss that can see the decay -- which a spectrogram distance cannot

    This is the one worth dwelling on, because the obvious objective gets it confidently wrong. Freeze everything except the decay, scale the measured RT by a constant, and score the result on the mel spectrogram distance alone:

    | RT scale | 0.4 | 0.6 | **1.0** | 1.3 | 1.6 |
    |---|---|---|---|---|---|
    | mel MSS ($\times 10^{-5}$) | 1.819 | **1.804** | 1.855 | 1.924 | 2.009 |
    | energy decay at 1 s | -67 dB | -47 dB | **-31 dB** | -25 dB | -21 dB |
    | the room, at 1 s | | | **-29 dB** | | |

    The minimum is at 0.6 -- an FDN whose tail is 18 dB below the room's at one second scores *better* than the one that tracks it to within 2 dB. That is not a bug in the loss, it is what a magnitude distance does: two rooms with the same decay still have uncorrelated fine structure, and against detail you cannot predict, silence is a better guess than the right amount of the wrong detail. Turning on the log term does not change the minimum.

    `pyFDN.MatchEnergyDecay` compares the quantity that actually is the decay: the Schroeder backward integral per octave band, each normalized to its own value at $t=0$ so the term reads decay and nothing else. Its minimum sits on the measurement. The objective is therefore a sum -- colour from the spectrogram, decay from the energy curves.

    ## 3. Enough accuracy at the bottom of the buffer

    Backward integration starts at the *end* of the response, so it reads the quietest samples in the buffer -- and in float32 those samples are not the FDN. `alias_decay_db=60` evaluates the system on a circle of radius $\gamma<1$ and the output layer divides the $\gamma^n$ envelope back out, which in single precision amplifies rounding into a -45 dB noise floor over the last eighth of the buffer. A spectrogram distance never notices. The energy decay curve integrates that floor into every earlier frame, and reports 10.1 dB of error on an FDN whose true error is 2.3 dB. `dtype=torch.float64` reproduces the exact time-domain render to six digits, and costs about twice the wall clock.

    So the pipeline is:

    1. `estimate_rt_bands` -- octave-band RT from the measurement, as the **starting point**.
    2. `pyFDN.trainable_from_build(..., absorption_rt=rt, trainable=Trainable(absorption=True, direct=True))`.
    3. `pyFDN.train_fdn(model, MatchMelSpectrogram(rir) + w * MatchEnergyDecay(rir))`.
    4. `pyFDN.extract_build` -- read the fitted FDN back out and render it.
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

    Trimmed to the onset and normalized to unit energy, exactly as in **Convert a room impulse response into an FDN** so the two notebooks are comparable.
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
    ## Step 1 -- where the fit starts

    Octave-band RT by Schroeder backward integration, extended to the ten GEQ design bands (DC, 63 Hz … 8 kHz, Nyquist). In the analytic notebook this *is* the answer; here it is the initial value of a trainable parameter, and the interesting question at the bottom of the notebook is how far the fit moves away from it.
    """)
    return


@app.cell
def _(fs, np, pyFDN, rir):
    est_rt, f_centre = pyFDN.estimate_rt_bands(rir, fs)
    est_level, _ = pyFDN.estimate_initial_level_bands(rir, est_rt, fs)

    # The GEQ bands extend the octave estimates by one band at each end.
    init_rt = np.concatenate(([est_rt[0]], est_rt, [est_rt[-1]]))

    num_delays = 16
    delays = pyFDN.sample_delay_lengths(
        num_delays,
        (700, 2500),
        distribution="geometric",
        coprime=True,
        sort=True,
        rng=3,
    )

    print(f"initial RT (s):    {est_rt.round(2)}")
    print(f"target level (dB): {pyFDN.lin_to_db(est_level).round(1)}")
    print(f"delays (samples):  {delays}")
    return delays, est_level, est_rt, f_centre, init_rt, num_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- the starting FDN

    A random orthogonal feedback matrix, normalized input/output gains, no dry path, and the absorption filters implied by `init_rt`. This is a complete, working FDN already -- it is the "before" of every comparison below.

    Note that `FDNBuild.filters` is left empty: `absorption_rt` builds the in-loop filter from the reverberation time instead, so the decay enters the model as a parameter rather than as fixed coefficients. `pyFDN.absorption_geq(init_rt, delays, fs)` produces the very same filters at step 0, to within 0.05 dB.
    """)
    return


@app.cell
def _(delays, fs, np, num_delays, pyFDN):
    init_build = pyFDN.FDNBuild(
        A=pyFDN.fdn_build_gallery(
            num_delays, fs=fs, delay_range=(700, 2500), rt=None, rng=3
        ).A,
        B=np.ones((num_delays, 1)) / np.sqrt(num_delays),
        C=np.ones((1, num_delays)) / np.sqrt(num_delays),
        D=np.zeros((1, 1)),
        delays=delays,
        fs=float(fs),
    )
    return (init_build,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- the objective

    ```python
    loss = pyFDN.MatchMelSpectrogram(rir, nfft=(256, 512, 1024, 2048)) + 1e-5 * pyFDN.MatchEnergyDecay(rir)
    ```

    Two terms, one per thing being fitted.

    **`MatchMelSpectrogram`** is a mel-scaled multi-resolution STFT distance -- the **colour** term. Two rooms with the same decay still have completely different modes, so a loss that compares $\lvert H \rvert$ bin by bin is mostly reading an irreducible mismatch. Mel bands average across frequency, which leaves the loss measuring the envelope in time and frequency, the part an FDN can actually reproduce.

    **`MatchEnergyDecay`** is the **decay** term, for the reason measured in section 2 above. Its value is the RMS error of the octave-band Schroeder curves, in dB.

    ### The weight is not decoration

    The two terms differ by six orders of magnitude: the spectrogram distance is about $2\times10^{-5}$ and the decay error about 2.3 dB. `1e-5` is chosen to put them in the same range at step 0, so neither term is decorative. Loss weights are never scale-free -- read `TrainLog.loss_log`, which stores every term *unweighted*, before trusting any weight, including a preset's. (For the same reason `train_fdn(model, "match_mel_spectrogram")` is wrong here: its `0.2 * Sparsity(...)` is calibrated for the colorless objective, whose terms are of order $10^{-1}$, and would outweigh this fit by four orders of magnitude.)

    ### Three more numbers that need justifying

    * **`nfft = 2**16`** -- 1.37 s, the window the loss sees of both signals. The decay has to fit inside it now that it is being fitted: the target is 38 dB down by the end of this window, more range than the 30 dB a $T_{30}$ estimate uses. Doubling it to `2**17` sharpens the decay term's optimum from a 10% underestimate to none at all, and doubles the wall clock.
    * **`dtype=torch.float64`** -- section 3.
    * **`alias_decay_db=60`** -- the FDN is still well above the numerical floor at the end of the window, and the FFT-domain render wraps everything after it back onto the start. The anti-aliasing decay suppresses that wrap-around by 60 dB; the `"time"` output layer removes the resulting $\gamma^n$ envelope again, so the loss sees the true impulse response and the envelope never reaches the extracted build.
    """)
    return


@app.cell
def _(init_build, init_rt, pyFDN, rir):
    import torch

    nfft = 2**16  # 1.37 s at 48 kHz -- longer than the decay being fitted

    model = pyFDN.trainable_from_build(
        init_build,
        # everything with a gradient: A, b, c, D, and the decay
        trainable=pyFDN.Trainable(absorption=True, direct=True),
        absorption_rt=init_rt,
        nfft=nfft,
        alias_decay_db=60.0,
        device="cpu",
        # the decay term integrates from the end of the buffer, where float32
        # holds only the anti-aliasing reconstruction's rounding noise
        dtype=torch.float64,
    )

    loss = pyFDN.MatchMelSpectrogram(
        rir, nfft=(256, 512, 1024, 2048)
    ) + 1e-5 * pyFDN.MatchEnergyDecay(rir)

    log = pyFDN.train_fdn(
        model,
        loss,
        optimizer="adam",
        max_steps=100,
        lr=3e-2,
        patience=50,
        device="cpu",
        dtype=torch.float64,
        rng=0,
    )
    trained_rt = pyFDN.param(model, "absorption").raw().detach().numpy().copy()
    trained_build = pyFDN.extract_build(model)

    print(
        f"ran {log.steps_run} steps, loss {log.train_loss[0]:.4g} -> "
        f"{log.train_loss[-1]:.4g} "
        f"({100 * (1 - log.train_loss[-1] / log.train_loss[0]):.0f}% down)"
    )
    for _name, _history in log.loss_log.items():
        print(f"  {_name:22s} {_history[0]:.4g} -> {_history[-1]:.4g}")
    return log, loss, nfft, torch, trained_build, trained_rt


@app.cell
def _(go, log):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(y=log.train_loss, mode="lines", name="mel MSS"))
    _fig.update_layout(
        title="Training loss",
        xaxis={"title": "step"},
        yaxis={"title": "mel multi-resolution STFT distance", "type": "log"},
        template="plotly_white",
        height=380,
    )
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The parameter that is the decay

    Because the decay is parametrized by reverberation time, the trained decay can be read straight off the parameter -- no rendering, no estimator. Below it is plotted against the Schroeder estimate it started from.

    These two curves are measured in different ways and there is no reason for them to coincide exactly. The parameter is the RT the *filter cascade* is designed for; the estimate is what a backward integration of one realized impulse response returns. The two outermost points -- DC and Nyquist -- are the GEQ's shelving bands, extrapolated from the octave estimates rather than measured, and the Nyquist point in particular is free to go anywhere: there is no content up there in the recording to hold it. Read the eight octave bands in between.
    """)
    return


@app.cell
def _(est_rt, f_centre, go, init_rt, np, trained_rt):
    _geq_f = np.concatenate(([1.0], f_centre, [24000.0]))

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_geq_f,
            y=init_rt,
            name="initial (from the RIR)",
            mode="lines+markers",
            line={"dash": "dot"},
        )
    )
    _fig.add_trace(
        go.Scatter(x=_geq_f, y=trained_rt, name="trained", mode="lines+markers")
    )
    _fig.update_layout(
        title="The absorption parameter, before and after",
        xaxis={"title": "Frequency (Hz)", "type": "log"},
        yaxis={"title": "RT parameter (s)", "rangemode": "tozero"},
        template="plotly_white",
        height=380,
    )
    _fig.show()

    print(f"initial RT (s): {init_rt.round(2)}")
    print(f"trained RT (s): {trained_rt.round(2)}")
    print(f"measured, octave bands (s): {est_rt.round(2)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render both FDNs

    `build_to_impz` runs the extracted build in the time domain, so the full tail comes out without the FFT wrap-around the training render had to be protected against. Both FDNs are rendered to the target's length and measured with the same estimators that produced the starting values.
    """)
    return


@app.cell
def _(fs, init_build, init_rt, pyFDN, rir_len, trained_build):
    import dataclasses

    # the untrained FDN is init_build plus the decay the fit started from
    start_build = dataclasses.replace(
        init_build,
        filters=pyFDN.absorption_geq(init_rt, init_build.delays, fs),
    )

    ir_init = pyFDN.build_to_impz(start_build, rir_len).squeeze()
    ir_trained = pyFDN.build_to_impz(trained_build, rir_len).squeeze()

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
    ## What the fit moved

    The two panels below are the same measurement applied to both FDNs: octave-band RT and initial level, estimated from the rendered impulse responses rather than read off the parameters.

    **Reverberation time** is now the fit's business, and it lands on the measurement to a few percent per band. It is worth being clear about what that does and does not show: the decay term is what put it there, and the decay term was built from this same RIR. The claim is not that the decay came for free -- it is that a decay left entirely free, initialized from a Schroeder estimate and then optimized against the recording, stays where the estimate put it instead of drifting off. Without the decay term, section 2's table is what the optimizer would be following instead.

    **Initial level** moves a long way. Most of it is a single number: the untrained FDN is uniformly too quiet, because normalized gains of $1/\sqrt{N}$ are an arbitrary starting level with no relation to the measurement, and the fit corrects that offset outright. The *shape* of the curve moves less. That is structural -- `b` and `c` are one frequency-flat scalar per delay line, so no setting of them is a filter, and the decay parameter can tilt the tail but not the onset. Matching band levels properly is a filter-design problem, which the last section of this notebook solves the way the analytic one does, with `design_geq` on the output.
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
    ### Is it more than a gain?

    A fair question, given how much of the level plot is one offset. Scale the untrained FDN by that offset -- one number, no optimizer -- and score it on the same loss. Whatever gap is left between that and the trained model is what the gradient steps bought beyond a volume knob.
    """)
    return


@app.cell
def _(
    dataclasses,
    est_level,
    init_rt,
    level_init,
    log,
    loss,
    nfft,
    pyFDN,
    start_build,
    torch,
):
    # the exact scalar the level plot is off by, applied to the input gain
    _offset_db = (pyFDN.lin_to_db(est_level) - pyFDN.lin_to_db(level_init)).mean()
    _matched = dataclasses.replace(
        start_build, B=start_build.B * 10 ** (_offset_db / 20)
    )

    def _score(build):
        _model = pyFDN.trainable_from_build(
            build,
            absorption_rt=init_rt,
            nfft=nfft,
            alias_decay_db=60.0,
            device="cpu",
            dtype=torch.float64,
        )
        return float(loss(pyFDN.model_response(_model)).detach())

    loss_matched = _score(_matched)
    print(f"untrained          {log.train_loss[0]:.4g}")
    print(f"untrained + {_offset_db:+.1f} dB  {loss_matched:.4g}")
    print(f"trained            {log.train_loss[-1]:.4g}")
    print(
        f"-> the scalar alone explains "
        f"{100 * (log.train_loss[0] - loss_matched) / (log.train_loss[0] - log.train_loss[-1]):.0f}%"
        " of the improvement; the fit is "
        f"{100 * (1 - log.train_loss[-1] / loss_matched):.0f}% below the gain-matched FDN."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy decay and spectrograms

    The band metrics above are blind to everything the fit is actually spending its capacity on, so look at the responses themselves. The energy decay curves show the tail; the spectrograms show the time-frequency envelope the mel loss compares.
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
    ## Finish the job: the band levels get a filter

    The one thing the fit could only nibble at is the *shape* of the initial-level curve, and it needs no optimizer -- it needs an equalizer. This is the self-correcting output GEQ of **Convert a room impulse response into an FDN**, designed here from the *trained* FDN's residual: the dB difference between the target's band levels and the ones just measured, which is by construction exactly the correction the model still needs. Being a residual is what makes it self-correcting, and it is also why it belongs after training rather than before -- the fit has already taken care of the offset, so the GEQ is left with the shape alone.

    The result stores on the same `FDNBuild` as `post_eq`. Output EQ lives outside the recursion, so `build_to_impz` declines to render it -- use the FLAMO path (`build_to_flamo` -> `flamo_time_response`) with an `nfft` long enough to hold the whole tail.
    """)
    return


@app.cell
def _(
    dataclasses,
    est_level,
    est_rt,
    f_centre,
    fs,
    go,
    level_trained,
    np,
    pyFDN,
    rir_len,
    trained_build,
):
    # what the trained FDN is still missing, per band, in dB
    _residual_db = pyFDN.lin_to_db(est_level) - pyFDN.lin_to_db(level_trained)
    geq_target_db = np.concatenate(
        ([_residual_db[0]], _residual_db, [_residual_db[-1]])
    )

    _sos_eq, _ = pyFDN.design_geq(geq_target_db, fs=fs)
    _sos_eq = _sos_eq / _sos_eq[:, 3:4]  # a0 = 1
    eq_build = dataclasses.replace(trained_build, post_eq=_sos_eq[:, :, np.newaxis])

    # post_eq sits outside the recursion, so render through FLAMO; 2**18 is
    # 5.5 s, well past the end of the decay, so nothing wraps around.
    _model = pyFDN.build_to_flamo(eq_build, nfft=2**18, device="cpu")
    ir_eq = np.asarray(pyFDN.flamo_time_response(_model, fs=fs)).reshape(-1)[:rir_len]

    rt_eq, _ = pyFDN.estimate_rt_bands(ir_eq, fs)
    level_eq, _ = pyFDN.estimate_initial_level_bands(ir_eq, rt_eq, fs)
    _err = pyFDN.lin_to_db(level_eq) - pyFDN.lin_to_db(est_level)
    print(f"output GEQ target (dB): {geq_target_db.round(1)}")
    print(
        f"{'trained + EQ':16s} RT error {100 * np.abs(rt_eq / est_rt - 1).mean():4.1f}%   "
        f"level offset {_err.mean():+5.1f} dB   "
        f"level shape {np.abs(_err - _err.mean()).mean():4.2f} dB"
    )

    _fig = go.Figure()
    for _name, _lv in (
        ("Target RIR", est_level),
        ("FDN, trained", level_trained),
        ("FDN, trained + EQ", level_eq),
    ):
        _fig.add_trace(
            go.Scatter(
                x=f_centre, y=pyFDN.lin_to_db(_lv), name=_name, mode="lines+markers"
            )
        )
    _fig.update_layout(
        title="Initial level after output equalization",
        xaxis={"title": "Frequency (Hz)", "type": "log"},
        yaxis={"title": "Initial level (dB)"},
        template="plotly_white",
        height=380,
    )
    _fig.show()
    return ir_eq, level_eq, rt_eq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Listen

    All four peak-normalized, so the A/B compares timbre rather than the level the fit corrected.
    """)
    return


@app.cell
def _(fs, ir_eq, ir_init, ir_trained, mo, pyFDN, rir):
    mo.hstack(
        [
            pyFDN.labeled_audio("Target RIR", pyFDN.peak_normalize(rir), fs=fs),
            pyFDN.labeled_audio("Untrained", pyFDN.peak_normalize(ir_init), fs=fs),
            pyFDN.labeled_audio("Trained", pyFDN.peak_normalize(ir_trained), fs=fs),
            pyFDN.labeled_audio("Trained + EQ", pyFDN.peak_normalize(ir_eq), fs=fs),
        ],
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: the fit stays stable, and finds the room's decay

    Three assertions. The first two are what the RT parametrization exists for: whatever the optimizer did, the parameter never left the stable region, and the FDN it left behind still renders. The third is that a decay trained as a free parameter ends up on the measured one.
    """)
    return


@app.cell
def _(est_rt, np, rt_trained, trained_rt):
    assert np.all(np.isfinite(rt_trained)), "trained FDN did not render"
    assert np.all(trained_rt > 0), "absorption parameter left the stable region"

    _err = np.abs(rt_trained / est_rt - 1)
    print(f"RT error per band: {_err.round(3)}")
    assert _err.mean() < 0.15, "trained decay drifted from the measurement"
    assert np.all(_err < 0.35), "one band's decay drifted from the measurement"
    return


if __name__ == "__main__":
    app.run()
