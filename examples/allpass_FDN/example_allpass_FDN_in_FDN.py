import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Allpass FDN embedded in a larger FDN

    Single input, stereo output. The signal flow is:

    **gain (1→N)** → **input delay** → **Recursion(** feedforward = **allpass MIMO FDN** + **attenuation filter**, feedback = **main delays** **)** → **output delay** → **gain (N→2)**

    The inner **allpass MIMO FDN** is a **homogeneous allpass FDN** (random delays, diagonal gain G, orthogonal U, A = G @ U, completed with **complete_fdn**). It is placed in the feedforward path of the recursion, followed by an attenuation filter; the feedback path is the main delay lines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML
    from collections import OrderedDict
    from flamo.processor import dsp, system
    import pyFDN
    from pyFDN.auxiliary.flamo import delay_module, gain_module, sos_filter_module

    np.random.seed(10)
    Fs = 48000
    nfft = 2**17
    N = 4  # number of delay lines (1→N input gain, N→2 output gain)
    return (
        Fs,
        N,
        OrderedDict,
        delay_module,
        dsp,
        gain_module,
        go,
        nfft,
        np,
        pyFDN,
        sos_filter_module,
        system,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. MIMO allpass FDN (inner block)

    Build a **homogeneous allpass FDN** (like in `example_allpass_FDN_homogeneous_mimo`): random delays, diagonal gain **G**, random orthogonal **U**, **A = G @ U**, then **complete_fdn** to get B, C, D. Get the FLAMO core (no Shell) to embed in the recursion feedforward.
    """)
    return


@app.cell
def _(Fs, N, nfft, np, pyFDN):
    # Homogeneous MIMO allpass FDN: delays, G = diag(g^delays), A = G @ U, complete_fdn
    delays_sch = np.random.randint(51, 300, size=N)
    _g = pyFDN.rt_to_gain_per_sample(0.07, Fs)
    G = np.diag(_g**delays_sch)
    U = pyFDN.random_orthogonal(N)
    A_sch = G @ U
    B_sch, C_sch, D_sch, X = pyFDN.complete_fdn(A_sch, N, N)

    # FLAMO core (N→N), no Shell — for use inside the recursion
    allpass_fdn_core = pyFDN.dss_to_flamo(
        A_sch, B_sch, C_sch, D_sch, delays_sch, Fs, nfft=nfft, shell=False
    )
    return (allpass_fdn_core,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Delays and attenuation

    Main delays (feedback path), input delay, output delay (all in seconds). Attenuation = diagonal gain in the feedforward path after the allpass FDN.
    """)
    return


@app.cell
def _(Fs, N, delay_module, nfft, np, pyFDN, sos_filter_module):
    # Main delays (feedback path), input and output delays — in seconds
    main_delay_sec = np.random.uniform(0.02, 0.04, size=N)
    input_delay_sec = np.linspace(0.01, 0.02 * N, N) + np.random.uniform(
        0, 0.001, size=N
    )
    output_delay_sec = np.linspace(0.01, 0.02, N) + np.random.uniform(0, 0.001, size=N)

    main_delays = delay_module(main_delay_sec, nfft, Fs=Fs)
    input_delays = delay_module(input_delay_sec, nfft, Fs=Fs)
    output_delays = delay_module(output_delay_sec, nfft, Fs=Fs)

    # Attenuation: one-pole absorption (SOS shape must be (n_sections, 6, n_channels))
    main_delay_smp = np.round(main_delay_sec * Fs).astype(float)
    rt_dc, rt_ny = 1.4, 0.3
    sos = pyFDN.one_pole_absorption(rt_dc, rt_ny, main_delay_smp, fs=Fs)
    sos = sos[np.newaxis, :, :]  # (1, 6, N)
    attenuation = sos_filter_module(sos, nfft)
    return attenuation, input_delays, main_delays, output_delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Full chain: gain (1→N) → input delay → recursion → output delay → gain (N→2)

    Recursion: **feedforward** = allpass MIMO FDN + attenuation, **feedback** = main delays. Then wrap in Shell for FFT/iFFT and get the impulse response.
    """)
    return


@app.cell
def _(
    N,
    OrderedDict,
    allpass_fdn_core,
    attenuation,
    dsp,
    gain_module,
    input_delays,
    main_delays,
    nfft,
    np,
    output_delays,
    system,
):
    # Input gain 1→N, output gain N→2 (stereo: normalize rows for equal energy)
    B_in = np.ones((N, 1))
    C_out = np.random.randn(2, N)
    C_out = C_out / np.linalg.norm(C_out, axis=1, keepdims=True)

    gain_B_in = gain_module(B_in, nfft)
    gain_C_out = gain_module(C_out, nfft)

    # Recursion: fF = allpass FDN → attenuation, fB = main delays
    feedforward = system.Series(
        OrderedDict(
            {
                "allpass_fdn": allpass_fdn_core,
                "attenuation": attenuation,
            }
        )
    )
    feedback_loop = system.Recursion(fF=feedforward, fB=main_delays)

    # Full chain
    core_chain = system.Series(
        OrderedDict(
            {
                "input_gain": gain_B_in,
                "input_delay": input_delays,
                "feedback_loop": feedback_loop,
                "output_delay": output_delays,
                "output_gain": gain_C_out,
            }
        )
    )

    model = system.Shell(
        core=core_chain,
        input_layer=dsp.FFT(nfft),
        output_layer=dsp.iFFT(nfft),
    )

    ir = model.get_time_response()
    ir_stereo = np.asarray(ir).squeeze()
    if ir_stereo.ndim == 3:
        ir_stereo = ir_stereo[0]
    return ir_stereo, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize signal flow
    """)
    return


@app.cell
def _(model, pyFDN):
    _g = pyFDN.draw_flamo_graph(model)
    _g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Plot and play
    """)
    return


@app.cell
def _(Fs, go, ir_stereo, mo, np, pyFDN):
    t = np.arange(ir_stereo.shape[0]) / Fs

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=pyFDN.mulaw_encode(ir_stereo[:, 0]), mode="lines", name="L")
    )
    fig.add_trace(
        go.Scatter(x=t, y=pyFDN.mulaw_encode(ir_stereo[:, 1]), mode="lines", name="R")
    )
    fig.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Amplitude [mu-law]",
        title="Allpass FDN in FDN — stereo IR",
        height=300,
        margin=dict(t=50, b=50, l=60, r=40),
        # xaxis=dict(range=[0, 0.2]),
        showlegend=True,
    )
    fig.show()

    mo.vstack([mo.audio(ir_stereo.T, Fs)])
    return


if __name__ == "__main__":
    app.run()
