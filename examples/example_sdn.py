# gallery_category: Special FDNs
# gallery_title: Scattering delay network from room geometry
# gallery_description: Derive a scattering delay network from room geometry and wall absorption, then render its impulse response with FLAMO.

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
    # Scattering Delay Network (SDN) Demo

    This example uses the **Scattering Delay Network (SDN)** to compute room acoustics coefficients (delays, feedback matrix, wall filters) from geometry and wall absorption, then builds a **FLAMO** model to render the impulse response.

    **Pipeline:** `pyFDN.SDN(room, source, receiver, wall_filters=...)` → `pyFDN.sdn_to_flamo(sdn)` → FLAMO model → `pyFDN.flamo_to_impz()` → NumPy IR.

    Original code of SDN: Enzo de Sena, Orchisama Das
    Python Translation: Sebastian J. Schlecht, Friday, 20. February 2026
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Room and wall filters

    Define room size, source and receiver positions, and sample rate. The wall filters are an SOS bank with one first-order shelving EQ per wall, with separate gains at DC and Nyquist.
    """)
    return


@app.cell
def _(np, pyFDN):
    room_size = np.array([7, 9, 5])
    source_pos = np.array([0.3, 0.5, 0.9]) * room_size
    receiver_pos = np.array([0.4, 0.1, 0.4]) * room_size
    fs = 44100

    # One first-order shelving EQ per wall (floor, ceiling, left, right, front,
    # back); each wall filters the five delay lines it sends into.
    wall_filters = pyFDN.gain_to_first_order_shelf(
        np.full(6, -1.0), np.full(6, -6.0), fs, crossover=8000.0
    )
    return fs, receiver_pos, room_size, source_pos, wall_filters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SDN

    Build the SDN: its delays (in samples), scattering matrix `A`, source and receiver gains `B` and `C`, direct path `D`, and wall filters come out of the geometry. No time-domain simulation is run here.
    """)
    return


@app.cell
def _(fs, pyFDN, receiver_pos, room_size, source_pos, wall_filters):
    sdn = pyFDN.SDN(
        room_size, source_pos, receiver_pos, fs=fs, wall_filters=wall_filters
    )
    return (sdn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D visualization

    Optional: visualize the room, source, and receiver.
    """)
    return


@app.cell
def _(pyFDN, sdn):
    pyFDN.plot_sdn(sdn)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLAMO model and impulse response

    Convert the SDN result into a FLAMO model (feedback delays, feedback matrix, absorption filters).
    """)
    return


@app.cell
def _(pyFDN, sdn):
    model = pyFDN.sdn_to_flamo(sdn, nfft=2**17, device="cpu")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN signal-flow graph

    The SDN is a 30-line FDN with input/output routing. `pyFDN.plot_flamo_graph` draws the full FLAMO signal flow: source→wall input path, the scattering/permutation feedback loop with wall filters, the wall→receiver output path, and the parallel direct path.
    """)
    return


@app.cell
def _(model, pyFDN):
    pyFDN.plot_flamo_graph(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot impulse response
    Obtain the impulse response as a NumPy array via `pyFDN.flamo_to_impz()`.
    """)
    return


@app.cell
def _(fs, mo, model, pyFDN):
    ir = pyFDN.flamo_to_impz(model).squeeze()
    fig = pyFDN.plot_impulse_response(
        ir,
        fs=fs,
        mulaw=False,
        title="SDN impulse response (FLAMO)",
    )
    mo.vstack([fig, mo.audio(ir, fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameter overview

    The 30-line core drawn as FDN parameters: the scattering matrix `A`, the source and receiver gains summed over the walls, the direct-path gain, and the wall filters as the per-line attenuation. The source, receiver and direct-path delays are not part of this picture.
    """)
    return


@app.cell
def _(pyFDN, sdn):
    pyFDN.plot_fdn_parameter(
        sdn.delays,
        sdn.A,
        sdn.B.sum(axis=1, keepdims=True),
        sdn.C.sum(axis=0, keepdims=True),
        sdn.D,
        post_delay=sdn.wall_filters,
        fs=sdn.fs,
        title="SDN FDN parameters",
    )
    return


if __name__ == "__main__":
    app.run()
