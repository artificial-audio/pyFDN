# gallery_category: FDN Design & Analysis

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
    # Time-domain graph engine vs FLAMO

    `pyFDN.td` renders an arbitrary FLAMO model **structure** directly in the
    time domain — no torch, no FFT. It walks the same Shell / Series / Parallel /
    Recursion / leaf tree that FLAMO builds and maps each node to a stateful
    NumPy operator, then streams the signal through block by block. The feedback
    `Recursion` is processed in blocks no larger than the shortest loop delay, so
    its read-before-write delay line lines up sample-for-sample with FLAMO's
    frequency-domain render.

    This notebook does two things:

    1. **Cross-check** — build one FDN, render it with both `td.process`
       (time domain) and FLAMO (frequency domain), and confirm the impulse
       responses match to numerical precision.
    2. **Time-varying matrix** — drop a `td.TimeVaryingMatrix` onto the feedback
       path. The loop becomes genuinely time-varying and has no static transfer
       function, so it is a render only the time-domain engine can produce.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import torch

    import pyFDN
    from pyFDN import td
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix

    pio.renderers.default = "sphinx_gallery"
    return TimeVaryingMatrix, go, np, pyFDN, td, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameters

    A small lossless-mixing FDN (random orthogonal feedback matrix) with
    per-delay-line first-order absorption setting a frequency-dependent
    reverberation time.
    """)
    return


@app.cell
def _(np, pyFDN):
    fs = 48_000
    delays = np.array([373, 421, 547, 661])
    num_delays = delays.size
    ir_len = fs // 2  # 0.5 s

    np.random.seed(5)
    A = pyFDN.random_orthogonal(num_delays)
    B = np.ones((num_delays, 1)) / num_delays
    C = np.ones((1, num_delays))
    D = np.zeros((1, 1))

    # Frequency-dependent decay: 0.8 s at DC, 0.3 s at Nyquist.
    absorption_sos = pyFDN.first_order_absorption(0.8, 0.3, delays, fs, None)

    print(f"Delays: {delays}")
    print(f"Absorption SOS shape: {absorption_sos.shape}")
    return A, B, C, D, absorption_sos, delays, fs, ir_len, num_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render with both engines

    `dss_to_flamo` builds the FLAMO model. `td.process` compiles that same model
    into a time-domain operator tree and runs it; `flamo_time_response` renders
    it in the frequency domain. The two impulse responses must overlap.
    """)
    return


@app.cell
def _(A, B, C, D, absorption_sos, delays, fs, ir_len, np, pyFDN, td, torch):
    impulse = np.zeros(ir_len)
    impulse[0] = 1.0

    model = pyFDN.dss_to_flamo(
        A,
        B,
        C,
        D,
        delays,
        fs,
        nfft=2**17,
        sos_filter=absorption_sos,
        dtype=torch.float64,
    )

    ir_td = td.process(model, impulse)
    ir_flamo = pyFDN.flamo_time_response(model).squeeze().astype(np.float64)[:ir_len]

    max_deviation = np.max(np.abs(ir_td - ir_flamo))
    print(f"Max |IR_td - IR_flamo| = {max_deviation:.3e}")
    assert max_deviation < 1e-6
    return impulse, ir_flamo, ir_td, model


@app.cell
def _(ir_flamo, ir_td, pyFDN):
    pyFDN.plot_impulse_response(
        ir_td,
        ir_flamo,
        labels=["td.process (time domain)", "FLAMO (frequency domain)"],
        title="Impulse response: pyFDN.td vs FLAMO",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Difference over time

    The residual is FLAMO's frequency sampling of the IIR absorption versus the
    exact time-domain `sosfilt`, plus any circular-convolution tail wrap — both
    far below audibility.
    """)
    return


@app.cell
def _(fs, go, ir_flamo, ir_td, np, pyFDN):
    diff = ir_td - ir_flamo
    fig_err = go.Figure()
    fig_err.add_trace(
        go.Scatter(
            x=np.arange(len(diff)) / fs,
            y=pyFDN.lin_to_db(diff),
            mode="lines",
            name="|IR_td - IR_flamo|",
            line={"width": 0.8},
        )
    )
    fig_err.update_layout(
        title="Difference between the two engines",
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Error (dB)"},
        template="plotly_white",
        height=340,
    )
    fig_err.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time-varying feedback matrix (td only)

    The same FDN, but the feedback path is now `Series([Gain(A),
    TimeVaryingMatrix])`: after the static mixing matrix `A`, each adjacent
    channel pair is rotated by a sinusoidally modulated angle that changes every
    sample. Building the operator tree by hand shows how `td` composes — the
    `Recursion` takes any forward and feedback operator.

    Because the loop modulates over time it has no static transfer function, so
    FLAMO cannot render it; the time-domain engine can.
    """)
    return


@app.cell
def _(A, B, C, TimeVaryingMatrix, absorption_sos, delays, fs, np, td):
    np.random.seed(2)
    tvm = TimeVaryingMatrix(
        N=delays.size,
        cycles_per_second=1.5,
        amplitude=0.35,
        fs=fs,
        spread=0.1,
    )

    # Hand-built forward path: delay -> absorption; feedback path: A -> tvm.
    # Each tree gets its own forward chain so they never share leaf state.
    forward = td.Series([td.Delay(delays), td.SOSBank(absorption_sos)])
    static_tree = td.Series([td.Gain(B), td.Recursion(forward, td.Gain(A)), td.Gain(C)])
    forward_tv = td.Series([td.Delay(delays), td.SOSBank(absorption_sos)])
    varying_tree = td.Series(
        [
            td.Gain(B),
            td.Recursion(
                forward_tv, td.Series([td.Gain(A), td.TimeVaryingMatrix(tvm)])
            ),
            td.Gain(C),
        ]
    )
    return static_tree, varying_tree


@app.cell
def _(impulse, pyFDN, static_tree, varying_tree):
    ir_static = static_tree.process(impulse).squeeze()
    ir_varying = varying_tree.process(impulse).squeeze()

    pyFDN.plot_impulse_response(
        ir_static,
        ir_varying,
        labels=["static feedback A", "A + time-varying matrix"],
        title="Static vs time-varying feedback (td)",
    )
    return ir_static, ir_varying


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two responses share their early reflections (the loop has not engaged
    yet) and diverge once the modulated feedback starts smearing the modal
    structure — the hallmark of a time-varying FDN that suppresses metallic
    ringing.
    """)
    return


@app.cell
def _(ir_static, ir_varying, np):
    divergence = np.max(np.abs(ir_static - ir_varying))
    print(f"Peak static-vs-varying deviation: {divergence:.3e}")
    assert divergence > 1e-3
    return


if __name__ == "__main__":
    app.run()
