# gallery_category: FDN Design & Analysis
# gallery_title: Train an FDN to match a room impulse response
# gallery_description: Fit every parameter of an FDN -- decay and output EQ included -- to a measured RIR, starting from a generic 1 s reverberator, with the filter design a one-line switch between a ten-band graphic EQ and a first-order shelf.
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

    The two filters the FDN needs -- the in-loop absorption that sets the decay, and the output EQ that colours it -- are built from a `pyFDN.eq.EQDesign`, and **which design is a switch, two sections down**. A ten-band graphic EQ spends ten numbers and eleven biquads on each; a first-order shelf spends two and one. The notebook runs either. On this room the two land within half a point of each other on mean RT error -- and get there by being wrong in completely different places, which turns out to say more about the loss than about the filters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is being trained

    | what | which parameter | how |
    |---|---|---|
    | **decay** | `post_delay` hook | **trained** -- as reverberation time in seconds, via `pyFDN.DecayFilter` |
    | **colour** of the output | `post_output` hook | **trained** -- as gain in dB, via `pyFDN.OutputEQ` |
    | fine structure of $\lvert H \rvert$, echo build-up | feedback matrix $A$ | **trained** -- on $SO(N)$ |
    | **level** | gains $b$, $c$ | **trained** |
    | dry path | $D$ | **trained** |
    | when the echoes fall | delays | **fixed** -- integer sample counts, no gradient to take |

    Every parameter with a gradient is in the fit, and the whole starting point is one number: a flat reverberation time of 1 s. Four things make that work, and each of them is a measurement rather than a preference.
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
    ## 1. A parametrization the decay cannot escape

    Training the absorption filter's *coefficients* does not work, and not for want of tuning. A too-quiet FDN offers any loss the same cheap direction -- more loop gain -- so a raw SOS cascade, with nothing holding its poles inside the unit circle, walks straight out of it. At `lr=3e-2` and at `lr=1e-3` alike the fit diverges within fifty steps, the loss ends four orders of magnitude *above* where it started, and the extracted FDN renders as `nan`.

    `pyFDN.DecayFilter` rewrites the filter as a differentiable function of the **reverberation time**:

    $$\mathrm{RT}_k \;\longrightarrow\; \underbrace{-60\,d_i / (\mathrm{RT}_k f_s)}_\text{dB per round trip} \;\longrightarrow\; \text{design} \;\longrightarrow\; \text{biquads}$$

    A positive RT means a negative dB attenuation, which means a contractive loop -- for **every** value the parameter can take. One RT per band is shared by all $N$ delay lines, and what differs between them is only the round-trip length $d_i$: exactly the homogeneous decay an FDN is designed for.

    What the RT still needs a floor for is the *sign*. A gradient step that puts a band at or below zero turns $-60 d_i / (\mathrm{RT} f_s)$ from an attenuation into a gain. `DecayFilter` floors it the same way whatever the design -- softplus, one round trip of the longest delay line, a knee one floor wide -- so a band that dips across zero still has a gradient to come back on.

    ## 2. Something in the model that can change the colour

    The gains $b$ and $c$ are one frequency-flat number per delay line, so no setting of them is a filter: on its own, an FDN can place its band *decays* but not its band *levels*. `pyFDN.OutputEQ` adds the one module that can -- the output EQ, which sits outside the recursion -- and parametrizes it the same way the decay is parametrized, by gain in dB rather than by free biquad coefficients. It starts flat. Being outside the loop it constrains nothing, so unlike the decay it needs no floor and no bound.

    In the analytic pipeline this same filter is *designed*, once, from the residual between the target's band levels and the FDN's. Here it is simply another parameter, and the last section checks what residual is left for a designed EQ to fix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The switch: which design the two filters use

    Both filters are built from an `EQDesign`, and a design carries **its own target**. So switching the whole notebook between a ten-band graphic EQ and a first-order shelf is switching one name: a scalar target spreads across however many parameters that design has, and `DecayFilter` / `OutputEQ` take it from there.

    ```python
    decay_design = design(1.0)   # a flat 1 s decay:  10 numbers, or 2
    eq_design    = design(0.0)   # a flat output EQ:  10 numbers, or 2
    ```

    Nothing downstream names either class again. The cells below read the design through the interface -- `n_params`, `n_sections`, and the SOS bank it maps onto -- so the model, the training call, the plots and the assertions are all written once.

    The shelf is the default because it is the cheaper run by a wide margin: eleven biquads per delay line instead of one puts the graphic EQ at roughly six times the wall clock for the same 300 steps. Both results are tabulated further down, so you can read the comparison without paying for it.
    """)
    return


@app.cell
def _(mo, np, pyFDN):
    from pyFDN.eq.design_geq import CENTER_FREQUENCIES

    fs = 48000

    # Everything that differs between the two runs of this notebook, in one
    # place: the design class, and where on the frequency axis each of its
    # parameters sits (which is the design's own band layout, not the
    # estimator's -- they coincide at the octave centres and nowhere else).
    _designs = {
        "Ten-band graphic EQ -- 10 numbers, 11 biquads": (
            pyFDN.GraphicEQ,
            np.concatenate(([1.0], CENTER_FREQUENCIES, [fs / 2])),
        ),
        "First-order shelf -- 2 numbers, 1 biquad": (
            pyFDN.FirstOrderShelf,
            np.array([1.0, fs / 2]),
        ),
    }
    design_choice = mo.ui.dropdown(
        options=_designs,
        value="First-order shelf -- 2 numbers, 1 biquad",
        label="EQ design",
    )
    mo.vstack([design_choice])
    return design_choice, fs


@app.cell
def _(design_choice):
    design, param_frequencies = design_choice.value

    print(f"design:      {design.__name__}")
    print(f"n_params:    {design.n_params}")
    print(f"n_sections:  {design.n_sections}")
    print(f"parameters sit at (Hz): {param_frequencies.round(0)}")
    return design, param_frequencies


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What each design can and cannot say

    A **ten-band graphic EQ** (`pyFDN.GraphicEQ`, Schlecht and Habets 2017) places a peaking filter at each octave centre from 63 Hz to 8 kHz plus a shelf at either end, and can therefore describe an RT curve of essentially any shape the octave grid can resolve. The least-squares fit that designs it is linear in its target, so it collapses into one constant matrix and the chain stays closed-form differentiable -- no iterative filter design inside the training loop.

    A **first-order shelf** (`pyFDN.FirstOrderShelf`, Jot 2015) has exactly two degrees of freedom once its crossover is fixed at $f_s/8$: its value at DC and its value at Nyquist. The curve it designs is a monotone tilt from one plateau to the other and it *cannot* be anything else. There is no setting of the two numbers that gives the 250 Hz octave a longer tail than its neighbours.

    That is a real restriction and it is worth being explicit that it is one. It is also, for an absorptive room, most of what there is to say: air and material absorption both rise with frequency, so a measured RT curve is usually a tilt with a few dB of wobble on it, and the wobble is the part a fit is least able to distinguish from the fine structure it cannot predict anyway.

    The shelf throws in a stability guarantee the graphic EQ has no closed form for. Its pole is

    $$p \;=\; \frac{1 - t/\sqrt{k}}{1 + t/\sqrt{k}}, \qquad t = \tan(2\pi f_c/f_s),$$

    which lies inside the unit circle for **any** pair of endpoint gains, since $t > 0$ below $f_s/4$ and $\sqrt{k} > 0$ always. Eleven cascaded biquads designed by least squares offer nothing equivalent; there, the RT parametrization of section 1 is the whole of the argument.
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

    Those rows were measured on the graphic-EQ setting, at a different feedback-matrix seed and a shorter `nfft` than this notebook now uses, so the `"both"` row reads 10.3 % where the fit below lands elsewhere. Compare the rows against each other, not against the result further down: the gaps between them are far larger than the offset.

    ## 4. Enough accuracy at the bottom of the buffer

    Backward integration starts at the *end* of the response, so this loss reads the quietest samples in the buffer -- the ones a render is least accurate about. Two settings decide how accurate they are, and only one of them is needed here.

    Evaluating an FDN as $(I - A\,D(z))^{-1}$ on the DFT grid renders one period of a *periodic* signal, so whatever the true response still has beyond `nfft` wraps back around to the start of the buffer. `alias_decay_db` suppresses that: it evaluates the system on a circle of radius $\gamma<1$ and the `"time"` output layer divides the $\gamma^n$ envelope back out, leaving the true response with its wrap-around attenuated by exactly that many dB.

    A **lossless** FDN cannot do without it: its poles sit exactly on the unit circle, where the inverse is near-singular and the response comes out wrong rather than merely wrapped. This one is not lossless -- by the end of the fit its tail is far enough down at the end of the training window that the wrap-around which leaks back is inaudible against it -- so this notebook leaves the setting at its default and does not pass the argument at all.

    It is still worth knowing what it is for -- the same objective on a near-lossless FDN would need it -- and worth knowing that in float32 it is actively *harmful*: the $\gamma^n$ reconstruction amplifies round-off at the end of the buffer by the same factor it suppresses aliasing, which is exactly where backward integration reads.

    `dtype=torch.float64` does stay, and costs about twice the wall clock. It is the cheaper insurance of the two: a cumulative energy integrates whatever floor sits at the end of the buffer into every earlier frame.

    So the whole pipeline is:

    1. an FDN with a flat 1 s decay and a flat output EQ, scaled once to the target's energy.
    2. `pyFDN.trainable_from_build(..., trainable=Trainable(direct=True), post_delay=DecayFilter(design(1.0), ...), post_output=OutputEQ(design(0.0), ...))` -- `Trainable` names the gains that train; each filter module carries its own gradient flag.
    3. `pyFDN.train_fdn(model, MatchCumulativeEnergy(rir, power=0.5, frequency="both"))`.
    4. read the two filters back out, and measure the render.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The target

    Trimmed to the onset and normalized to unit energy, exactly as in **Convert a room impulse response into an FDN** so the two notebooks are comparable.
    """)
    return


@app.cell
def _(fs, np, pyFDN):
    rir, _file_fs = pyFDN.load_audio("s3_r4_o", fs=fs)
    rir = rir[int(np.argmax(np.abs(rir))) :]
    rir = rir / np.linalg.norm(rir)
    rir_len = len(rir)

    print(f"target RIR: {rir_len} samples ({rir_len / fs:.2f} s) at {fs} Hz")
    return rir, rir_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The yardstick

    Octave-band RT and initial level of the *target*, by Schroeder backward integration. In the analytic notebook these are the design; here they are the exam paper, computed after the fit and fed to nothing.

    Note the shape: 2.8 s at the bottom, 1.2 s at 8 kHz, and a fall in between that never reverses, with a plateau across 63/125 Hz and another across 500 Hz/1 kHz. The monotone part is what a shelf can reach; the plateaus are what it cannot.
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

    `fdn_build_gallery` builds the whole thing in one call: a random orthogonal feedback matrix, sixteen normalized gains, no dry path, and per-delay absorption for a **flat 1 s decay in every band**. Only the delays are sampled separately, because the gallery's own delay sampling does not expose `distribution="geometric"` or `coprime=True`; they are passed straight in.

    So `init_build` is a complete FDN, not a scaffold -- the untrained render further down is just this build with the energy match applied, and nothing has to be reconstructed by hand to say what the optimizer started from. The gallery's absorption is a first-order shelf, but *flat* every design in `pyFDN.eq` is the same filter to numerical precision, so this build describes the optimizer's starting point exactly on either setting of the switch.

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

    print(f"delays (samples): {init_build.delays}")
    print(f"absorption:       {init_build.post_delay.shape} -- flat 1 s")
    return delays, init_build, num_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- the model, and the one thing the room is allowed to set

    `trainable_from_build` with a `DecayFilter` in the `post_delay` hook (the decay as a parameter) and an `OutputEQ` in the `post_output` hook (starting flat). Each is handed `design(...)`, so the design and the number of values it takes arrive together and nothing is inferred from how long a target happens to be.

    `nfft = 2**17` is 2.73 s at 48 kHz, and it is chosen once and used for everything. Both jobs it has to do put a floor under it. The loss compares this window against the target, so it has to hold the decay being fitted; and the *same* render is what the octave-band estimators at the bottom measure, where Schroeder integration over a window shorter than the decay under-reads it. 2.73 s clears both: the target has only -72 dB of its energy left after it, and the band RTs come out equal to three decimals against a render four times as long.

    That is what lets the trained model be measured directly, rather than exported to an `FDNBuild` and re-rendered at some larger `nfft`. The reason such a round trip is otherwise needed is that `nfft` is **structural** in FLAMO -- it fixes the frequency grid, the delay phase ramps and the alias envelope of every module at construction, and there is no setter -- so a render at a different length means rebuilding. Sizing it correctly once costs a factor of two on every training step and removes the rebuild.

    Then the single adjustment the measurement is allowed to make before the optimizer starts: **the overall energy**. One scalar on the output gain, so that the initial FDN and the target hold the same total energy in the training window. Without it the fit spends its first steps on a volume knob, and the loss below is normalized by the target's energy, so an FDN two decades too quiet starts on the flat part of the compression curve where there is little gradient to follow. It is a level match, not a decay match: what the scalar cannot do is tell the model *when* that energy arrives, which is the entire problem.

    The untrained render is taken immediately after it, in this same cell, because `train_fdn` steps `model` in place -- once the next cell has run there is no "before" left.
    """)
    return


@app.cell
def _(design, fs, init_build, np, pyFDN, rir, rir_len):
    import torch

    nfft = 2**17  # 2.73 s at 48 kHz -- long enough for the loss and the metrics

    model = pyFDN.trainable_from_build(
        init_build,
        # every gain with a gradient: A, b, c and D
        trainable=pyFDN.Trainable(direct=True),
        # the decay, as a reverberation time rather than as coefficients -- a
        # DecayFilter trains its own parameter unless told requires_grad=False
        post_delay=pyFDN.DecayFilter(
            design(1.0),
            init_build.delays,
            fs,
            nfft=nfft,
            device="cpu",
            dtype=torch.float64,
        ),
        # the output EQ, starting flat, as a gain in dB
        post_output=pyFDN.OutputEQ(
            design(0.0),
            1,
            fs,
            nfft=nfft,
            device="cpu",
            dtype=torch.float64,
        ),
        nfft=nfft,
        device="cpu",
        # no alias_decay_db: this FDN decays, so its poles are well inside the
        # unit circle and the FFT evaluation is sound -- see section 4
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
    init_sos = pyFDN.param(model, "post_delay").value().detach().numpy().copy()

    # ParamRef.shape is the MAPPED value -- the SOS bank the system runs. What
    # the optimizer steps is .raw(), and for the two filters that is the target.
    for _p in pyFDN.params(model):
        print(f"{_p}  raw {tuple(_p.raw().shape)}")
    print(f"\nenergy match: output gain x {energy_gain:.2f}")
    return energy_gain, init_sos, ir_init, model, nfft, render, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- the objective, and the numbers that need justifying

    One term. There is no weight to tune, because there is nothing to weigh it against: the surface carries the decay and the colour together, so a second term would only be a second opinion about the same data.

    * **`frequency="both"`** and **`power=0.5`** -- section 3's table.
    * **`window=1024`** (21 ms) -- the analysis window. `MatchEnergyDecay` needs 4096 to resolve the 63 Hz octave, and this loss does not, because it never splits into octaves; at `window=4096` the fit is no better and takes longer.
    * **`nfft = 2**17`** -- 2.73 s, the window of both signals the loss sees. The decay has to fit inside it now that it is being fitted from scratch.
    * **`dtype=torch.float64`** -- section 4.

    300 Adam steps at `lr=3e-2`. The trained response is rendered at the end of the same cell, out of the same model, through the same `render` the untrained one went through: two FDNs being compared on a metric should not be reaching it by two different routes.
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

    _decay = pyFDN.param(model, "post_delay")
    _eq = pyFDN.param(model, "post_output")
    trained_rt = _decay.raw().detach().numpy().copy().ravel()
    trained_sos = _decay.value().detach().numpy().copy()
    trained_eq_db = _eq.raw().detach().numpy().copy().ravel()
    trained_eq_sos = _eq.value().detach().numpy().copy()
    ir_trained = render(model)

    print(
        f"ran {log.steps_run} steps, loss {log.train_loss[0]:.4g} -> "
        f"{log.train_loss[-1]:.4g} "
        f"({100 * (1 - log.train_loss[-1] / log.train_loss[0]):.0f}% down)"
    )
    print(f"\ntrained RT (s):         {trained_rt.round(2)}")
    print(f"trained output EQ (dB): {trained_eq_db.round(1)}")
    print(f"\nin-loop filter: {trained_sos.shape} -- per delay line")
    print(f"output filter:  {trained_eq_sos.shape}")
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

    Both trained quantities are things you would plot anyway -- a reverberation time in seconds and an output EQ in dB -- and neither needs an estimator to read. But the thing to look at is the **curve** each set of numbers designs, because that is what the FDN runs. The parameters are drawn on top of it as markers, at the frequencies they actually sit at.

    The decay is plotted as reverberation time against frequency, read off the fitted filter's gain per sample, $\mathrm{RT}(f) = -60 / (f_s \cdot 20\log_{10} g(f))$, which is delay-line-independent because the decay is homogeneous.

    Read the ends of either curve with care. The outermost parameters sit at DC and at **24 kHz** -- an octave and a half above the highest band the estimator reports, and well into where the recording has nothing left -- so they are free to go anywhere. The shelf's Nyquist endpoint settles around 0.2 s; the graphic EQ's goes negative outright, and the parametrization floors it (section 1). Neither is a claim about 8 kHz: what the FDN does there is the curve at 8 kHz, which is 0.65 s on the shelf and 0.73 s on the graphic EQ. At the edges of the design, read the filter, not the parameter.
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
    return curve_f, rt_curve


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
    ## Measuring both FDNs

    Both responses already exist: `ir_init` from the model before the optimizer ran, `ir_trained` from the same model after. Both left FLAMO through the same `render` at the same `nfft`, so what follows compares two FDNs rather than two renderers.

    Worth being clear about one thing, because "outside the recursion" invites the wrong reading: the output EQ **is** an ordinary member of the FLAMO graph. `assemble_fdn_core` wires it in after the output gain as a leaf named `post_output`, the same optimizer steps it as steps the feedback matrix, and it is in the render above with everything else. Outside the *recursion* is a statement about where it sits in the signal flow -- which is exactly why it can shape the spectrum without touching the decay -- not about it being applied separately afterwards.
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
    ## What the fit moved, on both designs

    The same two estimators applied to both rendered FDNs, against a measurement neither of them saw. From a flat 1 s decay and a flat EQ, one energy match and 300 gradient steps -- and with the switch in each of its two positions, everything else identical:

    | | mean RT error | level offset | level shape | final loss |
    |---|---|---|---|---|
    | untrained (flat 1 s) | 52.4 % | -1.3 dB | 1.71 dB | 0.0932 |
    | trained, **first-order shelf** (2 numbers) | 9.9 % | +0.3 dB | 0.80 dB | 0.00392 |
    | trained, **ten-band graphic EQ** (10 numbers) | 9.4 % | +0.2 dB | 0.73 dB | 0.00263 |

    Five times the parameters and eleven times the biquads buy half a point of RT error. That is the headline, and it is a statement about the *room* more than about the designs: an absorptive hall's RT curve is close enough to a monotone tilt that the extra eight degrees of freedom have little left to describe.

    The shelf figure is a property of the fit rather than of one lucky matrix -- re-running with a different seed on the random orthogonal feedback matrix, everything else fixed, gives 9.9 %, 10.2 % and 10.0 % on seeds 0, 1 and 4 of `fdn_build_gallery`.

    Per band, though, the two designs are not close at all. They are wrong in different places, and the averages hide it:

    | | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
    |---|---|---|---|---|---|---|---|---|
    | RT error, shelf | 10 % | 6 % | 14 % | 4 % | 0 % | 5 % | 15 % | 26 % |
    | RT error, graphic EQ | **27 %** | 6 % | 14 % | 1 % | 4 % | 5 % | **8 %** | **9 %** |

    The shelf carries its error at both ends, which is what a monotone tilt pinned at two endpoints has to do. At the bottom the room holds a 2.8 s plateau across 63 and 125 Hz and has already dropped to 2.5 s by 250 Hz; a shelf cannot hold a plateau and then step down, so it splits the difference. The top is worse and for a different reason: its Nyquist endpoint extrapolates an octave and a half past the highest band the estimator reports. Both are restrictions you can read off the design before running anything.

    The graphic EQ spends its extra freedom exactly where you would expect -- the top two octaves go from 15 % and 26 % to 8 % and 9 % -- and then loses it all again in the bottom octave, which goes from 10 % to 27 %. The room is 2.8 s at 63 Hz; the shelf lands on 2.5 s there and the graphic EQ on 1.8 s, despite having a parameter for that band and the shelf not having one.

    That is worth sitting with, because it is not a defect of the design. It is section 3's table restated: cumulating the energy leaves the bottom octave with the least gradient of any band, and `frequency="both"` improves that rather than curing it. The shelf gets 63 Hz nearly right by *not being free* -- its low plateau is pinned by the whole midrange, which happens to sit near the room's bottom-octave value -- while the graphic EQ, free to move that band on its own, is moved by a loss that barely sees it. Extra parameters are only worth what the objective can supervise, and the honest reading of the two rows is that the graphic EQ is the better fit everywhere the loss has traction and the worse one everywhere it does not.

    Two things are worth being clear about on either setting. The level *offset* is not an achievement: the energy match set it before the optimizer ran, and the fit merely kept it. The level *shape* is, because nothing in the FDN proper can bend it -- that is the trained output EQ, and it is the only part of this that the analytic pipeline would have had to design.
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

    It takes the band-level shape error from 1.71 dB down to 0.73 dB on the graphic EQ and 0.80 dB on the shelf. So the answer is "most of it, not all of it": a designed GEQ on the residual would still buy the remainder, and nothing stops you from running one afterwards. What the fit does buy is that the EQ was chosen *while* the decay and the matrix were still moving, rather than as a correction applied to something already fixed.
    """)
    return


@app.cell
def _(est_level, f_centre, level_trained, np, pyFDN, trained_eq_db):
    residual_db = pyFDN.lin_to_db(est_level) - pyFDN.lin_to_db(level_trained)
    print(f"bands (Hz):         {f_centre.round(0)}")
    print(f"residual left (dB): {residual_db.round(1)}")
    print(
        f"\ntrained output EQ: {trained_eq_db.round(1)} dB"
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
    ## The deliverable

    `extract_build` reads the trained model back out as an `FDNBuild` -- plain numpy, no torch -- with the two trained filters baked into the `post_delay` and `post_output` hooks as ordinary SOS banks. That is the thing `process_fdn`, `build_to_impz` or `build_to_faust` would take, and it is what makes the assertions below a test of the FDN you would ship rather than of a torch graph.

    Nothing about it remembers which design produced it, which is the point: a build is baked, and `n_sections` biquads are `n_sections` biquads however they were chosen.
    """)
    return


@app.cell
def _(model, np, pyFDN, trained_eq_sos, trained_sos):
    trained_build = pyFDN.extract_build(model)

    print(f"post_delay:  {trained_build.post_delay.shape}")
    print(f"post_output: {trained_build.post_output.shape}")
    np.testing.assert_allclose(trained_build.post_delay, trained_sos, atol=1e-12)
    np.testing.assert_allclose(trained_build.post_output, trained_eq_sos, atol=1e-12)
    return (trained_build,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: the fit stays stable, and finds most of the room's decay from 1 s

    Six assertions, and none of them names a design. The first three are what the RT parametrization exists for: the extracted FDN still renders, the decay it settled on is a contractive filter at **every** frequency rather than only at the parameter points, and the filter has the number of sections its design promised. The last three are the fit itself -- a decay that started flat and knew nothing about the room ends up substantially closer to it, in the mean and in every band the loss can resolve.

    Note that the stability check is on the *curve*, not on the parameters. On the graphic EQ the Nyquist parameter is allowed to go negative, and does; what has to stay positive is the RT the designed filter actually realizes.
    """)
    return


@app.cell
def _(design, est_rt, np, rt_curve, rt_init, rt_trained, trained_sos):
    assert np.all(np.isfinite(rt_trained)), "trained FDN did not render"
    assert np.all(np.isfinite(rt_curve)) and np.all(rt_curve > 0), (
        "the decay filter is not contractive at every frequency"
    )
    assert trained_sos.shape[0] == design.n_sections, "the design changed its mind"

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
