import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scattering Delay Network (SDN) Demo

    This example uses the **Scattering Delay Network (SDN)** to compute room acoustics coefficients (delays, feedback matrix, wall filters) from geometry and wall absorption, then builds a **FLAMO** model to render the impulse response.

    **Pipeline:** SDN (room, source/receiver, wall filters) → `sdn.compute()` → `sdn.sdn_to_flamo(nfft)` → FLAMO model → `get_time_response()` → IR.

    Original code of SDN: Enzo de Sena, Orchisama Das
    Python Translation: Sebastian J. Schlecht, Friday, 20. February 2026
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
    from scipy.signal import butter, tf2sos
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML
    from IPython.display import Audio, display

    import pyFDN

    return Audio, butter, display, go, np, pyFDN, tf2sos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Room and wall filters

    Define room size, source and receiver positions, and sample rate. Wall filters are second-order sections (SOS); here we use a lowpass per wall with optional attenuation.
    """)
    return


@app.cell
def _(butter, np, tf2sos):
    room_size = np.array([5, 5, 5])
    source_pos = np.array([0.3, 0.5, 0.9]) 
    receiver_pos = np.array([0.4, 0.1, 0.4]) 

    # More natural setup
    room_size = np.array([7, 9, 5])
    source_pos = np.array([0.3, 0.5, 0.9]) * room_size
    receiver_pos = np.array([0.4, 0.1, 0.4]) * room_size
    Fs = 44100

    # Wall filters: 6 walls, 5 outputs each (SOS format)
    wall_attenuation = 0.9
    filt_order = 4
    b, a = butter(filt_order, 15000.0 / (Fs / 2), btype="low")
    b = b * wall_attenuation
    sos = tf2sos(np.asarray(b), np.asarray(a))
    wall_filters = [[sos for _ in range(5)] for _ in range(6)]
    return Fs, receiver_pos, room_size, source_pos, wall_filters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SDN compute

    Build the SDN and run `compute()` to get delays, routing, scattering, and wall filter coefficients. No time-domain simulation is run here—only coefficient extraction.
    """)
    return


@app.cell
def _(Fs, pyFDN, receiver_pos, room_size, source_pos, wall_filters):
    sdn = pyFDN.SDN(
        room_size=room_size,
        source_pos=source_pos,
        receiver_pos=receiver_pos,
        Fs=Fs,
        wall_filters=wall_filters,
    )
    result = sdn.compute()
    return (sdn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D visualization

    Optional: visualize the room, source, and receiver.
    """)
    return


@app.cell
def _(sdn):
    sdn.visualize(show=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLAMO model and impulse response

    Convert the SDN result into a FLAMO model (feedback delays, feedback matrix, absorption filters). Then obtain the impulse response via `get_time_response()`.
    """)
    return


@app.cell
def _(np, sdn):
    nfft = 2**17
    model, sdn_result = sdn.sdn_to_flamo(nfft=nfft)

    ir = np.asarray(model.get_time_response().squeeze())
    return (ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse response
    """)
    return


@app.cell
def _(Audio, Fs, display, go, ir, np):
    t = np.arange(len(ir)) / Fs
    fig = go.Figure(
        data=[go.Scatter(x=t, y=ir, mode="lines", line=dict(color="rgb(31, 119, 180)", width=1))],
        layout=dict(
            xaxis_title="Time [s]",
            yaxis_title="Amplitude",
            title="SDN impulse response (FLAMO)",
            template="plotly_white",
            height=320,
        ),
    )
    fig.show()

    display(Audio(ir, rate=Fs))
    return


if __name__ == "__main__":
    app.run()
