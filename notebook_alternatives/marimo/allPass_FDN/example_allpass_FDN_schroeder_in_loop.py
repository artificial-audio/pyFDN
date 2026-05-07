import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FDN with Schroeder allpass filters in the loop

    Schroeder allpass filters can be placed **behind the delays** in the FDN loop to increase echo density. The rendering is done with **FLAMO** (gain and delay modules).

    Steps:
    1. Build a **MIMO parallel Schröder allpass** (block-diagonal).
    2. Build a **vanilla FDN (SISO)**.
    3. Place the **Schröder allpass behind the delays** of the FDN and render.

    > Reference: Väänänen, R., Välimäki, V., Huopaniemi, J. & Karjalainen, M. Efficient and Parametric Reverberator for Room Acoustics Modeling. 200–203 (1997).


    — Original MATLAB: Sebastian J. Schlecht, 29 Dec 2019
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
    from IPython.display import Audio, display
    import pyFDN

    np.random.seed(6)
    Fs = 48000
    nfft = 2**17
    irLen = nfft
    return Audio, Fs, display, go, nfft, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. MIMO parallel Schröder allpass

    Build N parallel SISO Schröder allpasses (block-diagonal DSS), convert to FLAMO with **dss_to_flamo**, render and play.
    """)
    return


@app.cell
def _(Fs, nfft, np, pyFDN):
    N = 4
    sections_per = 2
    gain_per_sample_sch = pyFDN.rt_to_gain_per_sample(0.2, Fs)
    delays_per = np.random.randint(30, 200, size=(N, sections_per))
    g_per = gain_per_sample_sch ** delays_per * 0 + 0.7

    A_list, B_list, C_list, D_list, delays_list = [], [], [], [], []
    for i in range(N):
        Ai, bi, ci, di = pyFDN.series_allpass(g_per[i])
        A_list.append(Ai)
        B_list.append(bi)
        C_list.append(ci)
        D_list.append(di)
        delays_list.append(delays_per[i])  # 1 sample per section

    from scipy.linalg import block_diag
    A_sch = block_diag(*A_list)
    B_sch = block_diag(*B_list)
    C_sch = block_diag(*C_list)
    D_sch = block_diag(*D_list)
    delays_sch = np.concatenate(delays_list)

    model_sch = pyFDN.dss_to_flamo(A_sch, B_sch, C_sch, D_sch, delays_sch, Fs, nfft=nfft)
    ir_sch = model_sch.get_time_response()
    return A_sch, B_sch, C_sch, D_sch, N, delays_sch, ir_sch


@app.cell
def _(A_sch, B_sch, C_sch, D_sch, delays_sch, ir_sch, pyFDN):
    print(ir_sch.shape)
    print(delays_sch)
    pyFDN.plot_system_matrix(A_sch, B_sch, C_sch, D_sch)
    return


@app.cell
def _(Audio, Fs, display, go, ir_sch, np, pyFDN):
    ir_sch_channel = ir_sch[0, :, 1].squeeze()
    _t = np.arange(len(ir_sch_channel)) / Fs
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_t, y=pyFDN.mulaw_encode(ir_sch_channel), mode='lines'))
    _fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Amplitude [mu-law]',
        title='MIMO parallel Schröder allpass — impulse response (in0→out0)',
        height=300,
        margin=dict(t=50, b=50, l=60, r=40),
        showlegend=False,
        xaxis=dict(range=[0, 0.1])
    )
    _fig.show()
    display(Audio(ir_sch_channel, rate=Fs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Vanilla FDN (SISO)

    Random orthogonal feedback matrix, gains from delays, B/C/D identity-like. Build FLAMO with **dss_to_flamo**, render and play.
    """)
    return


@app.cell
def _(Fs, N, nfft, np, pyFDN):
    delays_fdn = np.random.randint(1200, 7900, size=N)
    feedback_matrix = pyFDN.random_orthogonal(N)

    B_fdn = np.ones((N, 1))
    C_fdn = np.ones((1, N))
    D_fdn = np.ones((1, 1))

    rt_dc = 2.0
    rt_ny = 0.7
    sos = pyFDN.one_pole_absorption(rt_dc, rt_ny, delays_fdn, fs=Fs)[np.newaxis, :]

    model_fdn = pyFDN.dss_to_flamo(feedback_matrix, B_fdn, C_fdn, D_fdn, delays_fdn, Fs, nfft=nfft, sos_filter=sos)
    ir_fdn = model_fdn.get_time_response().squeeze()
    print(ir_fdn.shape)
    print(delays_fdn)
    print(sos.shape)
    return B_fdn, C_fdn, D_fdn, delays_fdn, feedback_matrix, ir_fdn, sos


@app.cell
def _(delays_fdn):
    delays_fdn
    return


@app.cell
def _(Audio, Fs, display, go, ir_fdn, np, pyFDN):
    _t = np.arange(len(ir_fdn)) / Fs
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_t, y=pyFDN.mulaw_encode(ir_fdn), mode='lines'))
    _fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Amplitude [mu-law]',
        title='Vanilla FDN (SISO) — impulse response',
        height=300,
        margin=dict(t=50, b=50, l=60, r=40),
        showlegend=False,
    )
    _fig.show()
    display(Audio(ir_fdn, rate=Fs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. FDN with Schröder allpass behind the delays

    Reuse the two models above: take the **vanilla FDN** (feedback_matrix, B_fdn, C_fdn, D_fdn, delays_fdn) and the **MIMO Schröder allpass** (A_sch, B_sch, C_sch, D_sch, delays_sch). Build the Schröder as a core with **dss_to_flamo(..., shell=False)** and pass it as **post_delay_module** to append it to the FDN's forward path (after the FDN delays).
    """)
    return


@app.cell
def _(
    A_sch,
    B_fdn,
    B_sch,
    C_fdn,
    C_sch,
    D_fdn,
    D_sch,
    Fs,
    delays_fdn,
    delays_sch,
    feedback_matrix,
    nfft,
    pyFDN,
    sos,
):
    # Schröder core (4-in, 4-out) from section 1; append to FDN forward path
    schroeder_core = pyFDN.dss_to_flamo(
        A_sch, B_sch, C_sch, D_sch, delays_sch, Fs, nfft=nfft,
        shell=False,
    )
    model_fdn_ap = pyFDN.dss_to_flamo(
        feedback_matrix, B_fdn, C_fdn, D_fdn, delays_fdn, Fs, nfft=nfft,
        sos_filter=sos,
        post_delay_module=schroeder_core,
    )
    ir_fdn_ap = model_fdn_ap.get_time_response().squeeze()
    return ir_fdn_ap, model_fdn_ap


@app.cell
def _(Audio, Fs, display, go, ir_fdn_ap, np, pyFDN):
    _t = np.arange(len(ir_fdn_ap)) / Fs
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_t, y=pyFDN.mulaw_encode(ir_fdn_ap), mode='lines'))
    _fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Amplitude [mu-law]',
        title='FDN with Schröder allpass behind delays — impulse response',
        height=300,
        margin=dict(t=50, b=50, l=60, r=40),
        showlegend=False,
    )
    _fig.show()
    display(Audio(ir_fdn_ap, rate=Fs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the resulting signal flow
    """)
    return


@app.cell
def _(display, model_fdn_ap, pyFDN):
    # List all nodes with paths
    root = pyFDN.flamo_model_to_nodes(model_fdn_ap)
    flat = pyFDN.flamo_nodes_flat(root)
    for n in flat:
        print(n["path"], n["type"], n["name"])

    # Draw flowchart
    g = pyFDN.draw_flamo_graph(model_fdn_ap)
    display(g)
    return


if __name__ == "__main__":
    app.run()
