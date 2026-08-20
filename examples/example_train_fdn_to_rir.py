# gallery_category: FDN Design & Analysis
# gallery_title: Train an FDN to match a room impulse response
# gallery_description: Fit every parameter of an FDN -- decay and output EQ included -- to a measured RIR, starting from a generic 1 s reverberator, on a doubly-cumulated energy loss.
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

    **Convert a room impulse response into an FDN** designs the whole reverberator analytically: octave-band RT and level are estimated from the measurement and turned into filters. This notebook does the opposite. It starts from a **generic 1 s reverberator that knows nothing about the room**, and fits every parameter it has by gradient descent against the measurement -- the decay and the output EQ included. The target is the Promenadikeskus concert hall in Pori, Finland, published at {pyFDN.paper_link("Concert_Hall_Impulse_Responses")}.

    The estimators (`estimate_rt_bands`, `estimate_initial_level_bands`) still appear below, but only ever as the **yardstick**: nothing they return is fed to the model. What they measure at the end is a test the fit either passes or does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is being trained

    | what | which parameter | how |
    |---|---|---|
    | **decay**, per octave band | in-loop absorption | **trained** -- as reverberation time in seconds, `absorption_rt=` |
    | **colour** of the output | post EQ | **trained** -- as band gain in dB, `Trainable(post_eq=True)` |
    | fine structure of $\lvert H \rvert$, echo build-up | feedback matrix $A$ | **trained** -- on $SO(N)$ |
    | **level** | gains $b$, $c$ | **trained** |
    | dry path | $D$ | **trained** |
    | when the echoes fall | delays | **fixed** -- integer sample counts, no gradient to take |

    Every parameter with a gradient is in the fit, and the whole starting point is one number: a flat reverberation time of 1 s. Four things make that work, and each of them is a measurement rather than a preference.

    ## 1. A parametrization the decay cannot escape

    Training the absorption filter's *coefficients* does not work, and not for want of tuning. A too-quiet FDN offers any loss the same cheap direction -- more loop gain -- so a raw SOS cascade, with nothing holding its poles inside the unit circle, walks straight out of it. At `lr=3e-2` and at `lr=1e-3` alike the fit diverges within fifty steps, the loss ends four orders of magnitude *above* where it started, and the extracted FDN renders as `nan`.

    `absorption_rt=` replaces the filter with the same graphic-EQ design (`pyFDN.absorption_geq`, Schlecht and Habets 2017) rewritten as a differentiable function of the reverberation time per band:

    $$\mathrm{RT}_k \;\longrightarrow\; \underbrace{-60\,d_i / (\mathrm{RT}_k f_s)}_\text{dB per round trip} \;\longrightarrow\; \text{GEQ command gains} \;\longrightarrow\; \text{biquads}$$

    A positive RT means a negative dB attenuation, which means a contractive loop -- for **every** value the parameter can take. The least-squares fit that `design_geq` runs at each call is linear in its target, so it collapses into one constant matrix and the chain is closed-form differentiable; no iterative filter design inside the training loop. One RT per band is shared by all $N$ delay lines, and what differs between them is only the round-trip length $d_i$ -- exactly the homogeneous decay an FDN is designed for.

    ## 2. Something in the model that can change the colour

    The gains $b$ and $c$ are one frequency-flat number per delay line, so no setting of them is a filter: on its own, an FDN can place its band *decays* but not its band *levels*. `Trainable(post_eq=True)` adds the one module that can -- the output EQ, which sits outside the recursion -- and parametrizes it the same way the decay is parametrized, by ten band gains in dB rather than by 66 free biquad coefficients. It starts flat. Being outside the loop it constrains nothing, so unlike the decay it needs no floor and no bound.

    In the analytic pipeline this same filter is *designed*, once, from the residual between the target's band levels and the FDN's. Here it is simply another parameter, and the last section checks what residual is left for a designed EQ to fix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A loss that can see the decay -- which a spectrogram distance cannot

    This is the one worth dwelling on, because the obvious objective gets it confidently wrong. Freeze everything except the decay, scale a measured RT by a constant, and score the result on a mel multi-resolution spectrogram distance alone:

    | RT scale | 0.4 | 0.6 | **1.0** | 1.3 | 1.6 |
    |---|---|---|---|---|---|
    | mel MSS ($\times 10^{-5}$) | 1.819 | **1.804** | 1.855 | 1.924 | 2.009 |
    | energy decay at 1 s | -67 dB | -47 dB | **-31 dB** | -25 dB | -21 dB |
    | the room, at 1 s | | | **-29 dB** | | |

    The minimum is at 0.6 -- an FDN whose tail is 18 dB below the room's at one second scores *better* than the one that tracks it to within 2 dB. That is not a bug in the loss, it is what a magnitude distance does: two rooms with the same decay still have uncorrelated fine structure, and against detail you cannot predict, silence is a better guess than the right amount of the wrong detail.

    `pyFDN.MatchCumulativeEnergy` compares something a fit cannot cheat: the short-time energy, **cumulated twice**, backwards in time and along frequency.

    $$E[f, t] \;=\; \sum_{t' \ge t} \; \sum_{f' \ge f} \big| S[f', t'] \big|^2$$

    The time direction is Schroeder backward integration -- the decay itself. The frequency direction does the job that **splitting into octave bands** usually does, and does it without band edges: a cumulative sum compares every bin against every wider band that contains it, all at once, and is monotone and smooth in both axes, which is worth a great deal to a gradient. Read down the $t = 0$ edge and the surface is the integrated spectrum; read across the $f = 0$ edge and it is the full-band energy decay curve; the interior ties the two together. One loss, both quantities, no bands.

    ### Compression, not decibels

    A cumulated energy surface spans the entire dynamic range of the decay, so a plain MSE on it sees the first few frames and nothing else. `power` compresses that range -- the surface is normalized by the target's total energy and raised to a fractional power, 0.5 here, so that what is compared is amplitude rather than energy. Lower values compress harder and move weight onto the quiet end.

    A logarithm is the obvious alternative and is worse: it turns the silence *below* the response into an unbounded penalty, so whichever bin is nearest zero owns the gradient. A power keeps the compressed surface bounded and its gradient finite, and leaves 0 an ordinary value to predict.

    ### Which way the frequency cumulation runs is not a detail

    Cumulating from high to low puts every bin's energy into the rows *below* it, so an error in a low band moves only the largest values on the surface -- the ones compression weights least -- while a high band gets rows of its own. On this fit, with everything else held fixed at 300 steps, that asymmetry is worth more than the compression exponent:

    | `frequency` | `power` | mean RT error | level shape | trained RT at 63 Hz |
    |---|---|---|---|---|
    | `"descending"` (high → low) | 0.5 | 16.8 % | 2.56 dB | 0.26 s |
    | `"descending"` | 0.25 | 19.0 % | 1.46 dB | 0.44 s |
    | `"ascending"` | 0.5 | 12.8 % | 1.12 dB | 2.35 s |
    | **`"both"`** | **0.5** | **10.3 %** | **0.88 dB** | **2.17 s** |

    (the room is 2.8 s at 63 Hz). Cumulating downwards alone leaves the bottom octave with essentially no gradient and the fit abandons it; `"both"` scores the two directions separately -- each normalized and compressed on its own, then averaged -- and the low bands come back. Lengthening the analysis window does *not* fix it (15.4 % at `window=4096`, 63 Hz still at 0.20 s), which is the evidence that the problem is weighting rather than frequency resolution.

    **On the numbers in that table**: they were measured at an earlier configuration of this notebook -- a different feedback-matrix seed, and `alias_decay_db=60` (section 4) -- so its `"both"` row reads 10.3 % where the fit below lands at 9.6 %. Compare the rows against each other, not against the result further down; the gaps between them are far larger than the offset, but they have not been re-measured on the current setup.

    ## 4. Enough accuracy at the bottom of the buffer

    Backward integration starts at the *end* of the response, so this loss reads the quietest samples in the buffer -- the ones a render is least accurate about. Two settings decide how accurate they are, and only one of them is needed here.

    Evaluating an FDN as $(I - A\,D(z))^{-1}$ on the DFT grid renders one period of a *periodic* signal, so whatever the true response still has beyond `nfft` wraps back around to the start of the buffer. `alias_decay_db` suppresses that: it evaluates the system on a circle of radius $\gamma<1$ and the `"time"` output layer divides the $\gamma^n$ envelope back out, leaving the true response with its wrap-around attenuated by exactly that many dB.

    A **lossless** FDN cannot do without it: its poles sit exactly on the unit circle, where the inverse is near-singular and the response comes out wrong rather than merely wrapped. This one is not lossless. By the end of the fit its 2.4 s tail is 52 dB down at the end of the 1.37 s training window, and the wrap-around that leaks back measures **-39 dB** relative to the response, peaking at -50 dB. Turning the suppression on moves the fitted result by less than the matrix seed does, so this notebook leaves it at the default and does not pass the argument at all.

    It is still worth knowing what the setting is for -- the same objective on a near-lossless FDN would need it -- and worth knowing that in float32 it is actively *harmful*: the $\gamma^n$ reconstruction amplifies round-off at the end of the buffer by the same factor it suppresses aliasing, which is exactly where backward integration reads.

    `dtype=torch.float64` does stay, and costs about twice the wall clock. It is the cheaper insurance of the two: a cumulative energy integrates whatever floor sits at the end of the buffer into every earlier frame.

    So the whole pipeline is:

    1. an FDN with a flat 1 s decay and a flat output EQ, scaled once to the target's energy.
    2. `pyFDN.trainable_from_build(..., absorption_rt=rt, trainable=Trainable(absorption=True, direct=True, post_eq=True))`.
    3. `pyFDN.train_fdn(model, MatchCumulativeEnergy(rir, power=0.5, frequency="both"))`.
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
    ## The yardstick

    Octave-band RT and initial level of the *target*, by Schroeder backward integration. In the analytic notebook these are the design; here they are the exam paper, computed after the fit and fed to nothing.
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
    ## Step 1 -- a reverberator that knows nothing about the room

    `fdn_build_gallery` builds the whole thing in one call: a random orthogonal feedback matrix, sixteen normalized gains, no dry path, a flat output EQ, and a **flat 1 s decay in every band**. Only the delays are sampled separately, because the gallery's own delay sampling does not expose `distribution="geometric"` or `coprime=True`; they are passed straight in.

    So `init_build` is a complete FDN, not a scaffold -- the untrained render further down is just this build with the energy match applied, and nothing has to be reconstructed by hand to say what the optimizer started from.

    The measured room is 2.8 s at 63 Hz falling to 1.2 s at 8 kHz, so this starting point is wrong by a factor of three at the bottom of the spectrum and by a fifth at the top -- and wrong in the wrong direction at both ends.
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
    # normalized gains, no dry path, and per-delay absorption for a flat 1 s
    # decay. The gallery's absorption is a first-order shelf and the filter
    # trained below is a ten-band GEQ, but *flat* they are the same filter to
    # 2e-15, so this build is an exact description of the optimizer's start.
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

    init_rt = np.full(10, 1.0)  # the 10 GEQ design bands: DC, 63 Hz … 8 kHz, Nyquist
    return delays, init_build, init_rt, num_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- the model, and the one thing the room is allowed to set

    `trainable_from_build` with `absorption_rt=` (the decay as a parameter) and `Trainable(post_eq=True)` (the output EQ as a parameter, starting flat).

    Then the single adjustment the measurement is allowed to make before the optimizer starts: **the overall energy**. One scalar on the output gain, so that the initial FDN and the target hold the same total energy in the training window. Without it the fit spends its first steps on a volume knob, and the loss below is normalized by the target's energy, so an FDN two decades too quiet starts on the flat part of the compression curve where there is little gradient to follow. It is a level match, not a decay match: what the scalar cannot do is tell the model *when* that energy arrives, which is the entire problem.
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
        absorption_rt=init_rt,
        nfft=nfft,
        device="cpu",
        # no alias_decay_db: this FDN decays, so its poles are well inside the
        # unit circle and the FFT evaluation is sound -- see section 4
        dtype=torch.float64,
    )

    # the only thing the target tells the initial model: how loud it is
    _ir = np.asarray(pyFDN.model_response(model).h.detach()).reshape(-1)
    energy_gain = float(np.linalg.norm(rir[:nfft]) / np.linalg.norm(_ir))
    with torch.no_grad():
        pyFDN.param(model, "output_gain").raw().mul_(energy_gain)

    for _p in pyFDN.params(model):
        print(_p)
    print(f"\nenergy match: output gain x {energy_gain:.2f}")
    return energy_gain, model, nfft, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- the objective, and the numbers that need justifying

    One term. There is no weight to tune, because there is nothing to weigh it against: the surface carries the decay and the colour together, so a second term would only be a second opinion about the same data.

    * **`frequency="both"`** and **`power=0.5`** -- section 3's table.
    * **`window=1024`** (21 ms) -- the analysis window. `MatchEnergyDecay` needs 4096 to resolve the 63 Hz octave, and this loss does not, because it never splits into octaves; at `window=4096` the fit is no better (11.1 % against 10.3 %, on the earlier configuration noted in section 3) and takes longer.
    * **`nfft = 2**16`** -- 1.37 s, the window of both signals the loss sees. The decay has to fit inside it now that it is being fitted from scratch: the target is 38 dB down by the end of it.
    * **`dtype=torch.float64`** and **`alias_decay_db=60`** -- section 4.
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
    ## The two parameters that are the answer

    Both trained quantities are the things you would plot anyway: a reverberation time in seconds, and an output EQ in dB. Neither needs rendering or an estimator to read.

    The RT is plotted against the flat 1 s it started from and against the Schroeder estimate of the target -- which the fit never saw. From 1 s in every band it finds 2.1-2.5 s across the middle of the spectrum, against a measurement of 2.4-2.8 s.

    Two caveats on how to read the ends of that curve. The outermost points, DC and Nyquist, are the GEQ's *shelving* bands: the estimate has nothing to say up or down there, and neither does the recording, so they are free to go anywhere and the Nyquist one duly goes negative (the parametrization floors it; section 1). And the 8 kHz parameter reads 0.60 s against a measured 1.2 s, yet the rendered FDN's 8 kHz octave comes out within 8 % of the measurement -- because what decays that octave is the *designed filter* around 5.6-11.3 kHz, which the peaking band and the high shelf set together. At the edges of the band layout, read the filter, not the parameter.
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
            name="initial (flat 1 s)",
            mode="lines+markers",
            line={"dash": "dot"},
        )
    )
    _fig.add_trace(
        go.Scatter(x=_geq_f, y=trained_rt, name="trained", mode="lines+markers")
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
        title="The absorption parameter: reverberation time per band",
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


@app.cell
def _(f_centre, go, np, trained_eq_db):
    _geq_f = np.concatenate(([1.0], f_centre, [24000.0]))

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_geq_f,
            y=np.zeros_like(trained_eq_db),
            name="initial (flat)",
            mode="lines",
            line={"dash": "dot"},
        )
    )
    _fig.add_trace(
        go.Scatter(x=_geq_f, y=trained_eq_db, name="trained", mode="lines+markers")
    )
    _fig.update_layout(
        title="The output EQ parameter: band gain",
        xaxis={"title": "Frequency (Hz)", "type": "log"},
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
    ## Render both FDNs

    Both down the **same** path -- `build_to_flamo` then `flamo_time_response` at `nfft = 2**19` (10.9 s, nearly three times the target, so the FFT wrap-around the training render had to be protected against is not a factor here) -- and both measured with the same estimators. Two FDNs being compared on a metric should not be reaching it through two different renderers.

    Why the round trip through `FDNBuild` rather than just rendering the trained model? Because `nfft` is **structural** in FLAMO -- it fixes the frequency grid, the delay phase ramps and the alias envelope of every module at construction, and there is no setter. The model was deliberately built at `nfft = 2**16`, 1.37 s: that is the window the loss looks at, and 2**19 would have been eight times the FFT work on every one of the 300 steps. But the estimators below have to see the whole 3.9 s tail, and Schroeder integration over a 1.37 s window under-reads a 2.4 s decay -- on this FDN by up to 10 % in the bottom two octaves, which is larger than the error the table is trying to measure. So the render has to happen at a different `nfft`, and that means rebuilding.

    `extract_build` -> `build_to_flamo` **is** that rebuild; `FDNBuild` is just the plain-numpy description in the middle of it. That it is also the deliverable -- the thing `process_fdn` or `build_to_faust` would take -- is the bonus: rendering through it is what makes the last cell's assertions a test of the FDN you would actually ship, rather than of a torch graph.

    Worth being clear about one thing, because "outside the recursion" invites the wrong reading: the output EQ **is** an ordinary member of the FLAMO graph. `assemble_fdn_core` wires it in after the output gain as a leaf named `output_filter`, the same optimizer steps it as steps the feedback matrix, and `extract_build` reads it back out as `build.post_eq`. Outside the *recursion* is a statement about where it sits in the signal flow -- which is exactly why it can shape the spectrum without touching the decay -- not about it being applied separately afterwards.

    What cannot render it is `build_to_impz`, whose `process_fdn` block simulation has no output-filter slot; it raises on a build with `post_eq` rather than silently dropping it. On this FDN the two renderers agree to -69 dB and to three decimals on every band metric below, so using the FLAMO one for both costs nothing.
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
    ## What the fit moved

    The same two estimators applied to both rendered FDNs. From a flat 1 s decay and a flat EQ, one energy match and 300 gradient steps:

    | | mean RT error | level offset | level shape |
    |---|---|---|---|
    | untrained | 52.4 % | -1.3 dB | 1.71 dB |
    | **trained** | **9.6 %** | **+0.3 dB** | **0.77 dB** |

    Per band the trained decay is within 10 % in six of the eight octaves, 14 % at 250 Hz and 24 % at 63 Hz -- the bottom octave staying hardest even with `frequency="both"`.

    Two things are worth being clear about. The level *offset* is not an achievement: the energy match set it before the optimizer ran, and the fit merely kept it. The level *shape* is, because nothing in the FDN proper can bend it -- that is the trained output EQ, and it is the only part of this that the analytic pipeline would have had to design.

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

    The band metrics above are a summary; the surfaces the loss actually compares are below. The energy decay curves are the $f = 0$ edge of that surface, and the spectrograms are what it integrates.
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

    The analytic pipeline ends by designing an output GEQ from the residual between the target's band levels and the FDN's. That filter is now a trained parameter, so the same residual is a test of it: whatever a `design_geq` call would still be asked to correct is what the fit did not manage.

    What came out is a gentle curve -- about +1.3 dB across the middle, -2.6 dB at 8 kHz -- and it takes the band-level shape error from 1.71 dB to 0.77 dB. So the answer is "most of it, not all of it": a designed GEQ on the residual would still buy the remaining 0.8 dB, and nothing stops you from running one afterwards. What the fit does buy is that the EQ was chosen *while* the decay and the matrix were still moving, rather than as a correction applied to something already fixed.
    """)
    return


@app.cell
def _(est_level, f_centre, level_trained, np, pyFDN, trained_eq_db):
    residual_db = pyFDN.lin_to_db(est_level) - pyFDN.lin_to_db(level_trained)
    print(f"bands (Hz):            {f_centre.round(0)}")
    print(f"trained output EQ (dB):{trained_eq_db[1:-1].round(1)}")
    print(f"residual left (dB):    {residual_db.round(1)}")
    print(
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
    ## Test: the fit stays stable, and finds most of the room's decay from 1 s

    Four assertions. The first two are what the RT parametrization exists for: whatever the optimizer did, the extracted FDN still renders, and the decay never left the stable region. The third and fourth are the fit itself -- a decay that started flat and knew nothing about the room ends up substantially closer to it, in the mean and in every band the loss can resolve.
    """)
    return


@app.cell
def _(est_rt, np, rt_init, rt_trained, trained_rt):
    assert np.all(np.isfinite(rt_trained)), "trained FDN did not render"
    assert np.all(trained_rt[1:-1] > 0), "absorption parameter left the stable region"

    _err_init = np.abs(rt_init / est_rt - 1)
    _err = np.abs(rt_trained / est_rt - 1)
    print(f"RT error per band, untrained: {_err_init.round(3)}")
    print(f"RT error per band, trained:   {_err.round(3)}")
    assert _err.mean() < 0.3 * _err_init.mean(), "the fit barely moved the decay"
    assert _err.mean() < 0.15, "the trained decay is not close to the measurement"
    assert _err.max() < 0.30, "one band's decay is far off the measurement"
    return


if __name__ == "__main__":
    app.run()
