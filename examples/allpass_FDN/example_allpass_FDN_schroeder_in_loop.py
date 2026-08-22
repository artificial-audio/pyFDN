# gallery_category: Allpass FDN Examples
# gallery_title: Schroeder allpass in a feedback loop
# gallery_description: Place a Schroeder allpass cascade inside a recursive loop and examine the resulting reverberator.

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo, pyFDN):
    mo.md(f"""
    # FDN with Schroeder allpass filters in the loop

    Schroeder allpass filters can be placed **behind the delays** in the FDN loop to increase echo density. The allpass cascade on its own is rendered in the time domain; the recursive networks are rendered with **FLAMO** (gain and delay modules).

    Steps:
    1. Build a **MIMO parallel Schroeder allpass** (block-diagonal).
    2. Build a **vanilla FDN (SISO)**.
    3. Place the **Schroeder allpass behind the delays** of the FDN and render.

    > Reference: *{pyFDN.paper_link("Vaananen1997EfficientParametricReverberator")}*.

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
    import plotly.io as pio

    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML

    import pyFDN

    np.random.seed(6)
    Fs = 48000
    nfft = 2**16
    return Fs, nfft, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. MIMO parallel Schroeder allpass

    Build N parallel SISO Schroeder allpasses (block-diagonal DSS), render the impulse response in the time domain with **dss_to_impz**, and play it.
    """)
    return


@app.cell
def _(np, pyFDN):
    N = 4
    sections_per_channel = 2
    ir_len_schroeder = 2**10

    allpass_delays = np.random.randint(30, 200, size=(N, sections_per_channel))
    allpass_gains = np.full(allpass_delays.shape, 0.7)

    A_list, B_list, C_list, D_list, delays_list = [], [], [], [], []
    for i in range(N):
        Ai, bi, ci, di = pyFDN.series_allpass(allpass_gains[i])
        A_list.append(Ai)
        B_list.append(bi)
        C_list.append(ci)
        D_list.append(di)
        delays_list.append(allpass_delays[i])

    from scipy.linalg import block_diag

    A_schroeder = block_diag(*A_list)
    B_schroeder = block_diag(*B_list)
    C_schroeder = block_diag(*C_list)
    D_schroeder = block_diag(*D_list)
    delays_schroeder = np.concatenate(delays_list)

    ir_schroeder = pyFDN.dss_to_impz(
        ir_len_schroeder,
        delays_schroeder,
        A_schroeder,
        B_schroeder,
        C_schroeder,
        D_schroeder,
    )
    return (
        A_schroeder,
        B_schroeder,
        C_schroeder,
        D_schroeder,
        N,
        delays_schroeder,
        ir_schroeder,
    )


@app.cell
def _(
    A_schroeder,
    B_schroeder,
    C_schroeder,
    D_schroeder,
    delays_schroeder,
    ir_schroeder,
    pyFDN,
):
    print(ir_schroeder.shape)
    print(delays_schroeder)
    pyFDN.plot_system_matrix(A_schroeder, B_schroeder, C_schroeder, D_schroeder)
    return


@app.cell
def _(Fs, ir_schroeder, mo, np, pyFDN):
    ir_schroeder_channel = ir_schroeder[:, 0, 0]
    _fig = pyFDN.plot_impulse_response(
        ir_schroeder_channel,
        fs=Fs,
        title="MIMO parallel Schroeder allpass — impulse response (in0→out0)",
    )

    mo.vstack([_fig, mo.audio(np.asarray(ir_schroeder_channel), Fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Vanilla FDN (SISO)

    Let **fdn_build_gallery** assemble the whole thing -- random orthogonal feedback matrix, delays, B/C/D, and the per-delay-line absorption matching the requested reverberation time -- into a single **FDNBuild**. Render it with **build_to_flamo** and play.
    """)
    return


@app.cell
def _(Fs, N, nfft, pyFDN):
    fdn_build = pyFDN.fdn_build_gallery(
        N,
        fs=Fs,
        delay_range=(600, 3900),
        rt=2.0,
        rt_nyquist=0.7,
        rng=6,
    )

    model_fdn = pyFDN.build_to_flamo(fdn_build, nfft=nfft)
    ir_fdn = pyFDN.flamo_time_response(model_fdn).squeeze()
    return fdn_build, ir_fdn


@app.cell
def _(Fs, ir_fdn, mo, np, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        ir_fdn,
        fs=Fs,
        title="Vanilla FDN (SISO) — impulse response",
    )

    mo.vstack([_fig, mo.audio(np.asarray(ir_fdn), Fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. FDN with Schroeder allpass behind the delays

    Reuse the two models above: take the **vanilla FDN** (the `fdn_build`) and the **MIMO Schroeder allpass** (A_schroeder, B_schroeder, C_schroeder, D_schroeder, delays_schroeder). Build the Schroeder as a core with **dss_to_flamo(..., shell=False)** and pass it in the **post_delay** hook of **build_to_flamo** -- extra hook modules are appended after whatever the build already carries, so the loop becomes delay -> absorption -> Schroeder.
    """)
    return


@app.cell
def _(
    A_schroeder,
    B_schroeder,
    C_schroeder,
    D_schroeder,
    Fs,
    delays_schroeder,
    fdn_build,
    nfft,
    pyFDN,
):
    # Schroeder core (4-in, 4-out) from section 1; append to FDN forward path
    schroeder_core = pyFDN.dss_to_flamo(
        A_schroeder,
        B_schroeder,
        C_schroeder,
        D_schroeder,
        delays_schroeder,
        Fs,
        nfft=nfft,
        shell=False,
    )
    model_fdn_allpass = pyFDN.build_to_flamo(
        fdn_build,
        nfft=nfft,
        post_delay=schroeder_core,
    )
    ir_fdn_allpass = pyFDN.flamo_time_response(model_fdn_allpass).squeeze()

    pyFDN.plot_flamo_graph(model_fdn_allpass)
    return (ir_fdn_allpass,)


@app.cell
def _(Fs, ir_fdn_allpass, mo, np, pyFDN):
    _fig = pyFDN.plot_impulse_response(
        ir_fdn_allpass,
        fs=Fs,
        title="FDN with Schroeder allpass behind delays — impulse response",
    )

    mo.vstack([_fig, mo.audio(np.asarray(ir_fdn_allpass), Fs)])
    return


if __name__ == "__main__":
    app.run()
