# gallery_category: Special FDNs

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

    **Pipeline:** SDN (room, source/receiver, wall filters) → `sdn.compute()` → `sdn.sdn_to_flamo(nfft)` → FLAMO model → `pyFDN.flamo_time_response()` → NumPy IR.

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
    import plotly.io as pio
    from scipy.linalg import block_diag

    pio.renderers.default = "sphinx_gallery"  # interactive in Jupyter + docs HTML

    import pyFDN

    return block_diag, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Room and wall filters

    Define room size, source and receiver positions, and sample rate. Wall filters are first-order shelving EQs in second-order-section (SOS) format, with separate gains at DC and Nyquist.
    """)
    return


@app.cell
def _(np, pyFDN):
    room_size = np.array([7, 9, 5])
    source_pos = np.array([0.3, 0.5, 0.9]) * room_size
    receiver_pos = np.array([0.4, 0.1, 0.4]) * room_size
    Fs = 44100

    # Wall filters: one first-order shelving EQ per wall, shared by its 5 outputs.
    wall_sos = pyFDN.first_order_shelving_eq(
        db_dc=np.full(6, -1.0),
        db_nyquist=np.full(6, -6.0),
        fs=Fs,
        crossover_frequency=8000.0,
    )
    wall_filters = [[wall_sos[:, :, wall] for _ in range(5)] for wall in range(6)]
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
    _ = sdn.compute()  # mutates sdn in place; assign to suppress the result-dict output
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

    Convert the SDN result into a FLAMO model (feedback delays, feedback matrix, absorption filters).
    """)
    return


@app.cell
def _(sdn):
    nfft = 2**17
    model, sdn_result = sdn.sdn_to_flamo(nfft=nfft)
    return model, sdn_result


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
    Obtain the impulse response as a NumPy array via `pyFDN.flamo_time_response()`.
    """)
    return


@app.cell
def _(Fs, mo, model, pyFDN):
    ir = pyFDN.flamo_time_response(model).squeeze()
    fig = pyFDN.plot_impulse_response(
        ir,
        fs=Fs,
        mulaw=False,
        title="SDN impulse response (FLAMO)",
    )
    mo.vstack([fig, mo.audio(ir, Fs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameter overview

    The core feedback loop reconstructed as standard FDN parameters: feedback matrix `A = blockdiag(scattering) @ permutation`, net per-line input/output gains, the direct-path gain, and the wall filters as the per-line attenuation.
    """)
    return


@app.cell
def _(block_diag, np, pyFDN, sdn_result):
    A = (
        block_diag(*sdn_result["scattering_matrices"])
        @ sdn_result["permutation_matrix"]
    )
    delays_smp = np.rint(
        np.asarray(sdn_result["delay_lengths_flat"]) * sdn_result["Fs"]
    ).astype(int)
    B = sdn_result["input_matrix"] @ np.asarray(sdn_result["input_gains"]).reshape(
        -1, 1
    )
    C = (
        np.asarray(sdn_result["output_gains"]).reshape(1, -1)
        @ sdn_result["output_matrix"]
    )
    D = np.array([[sdn_result["direct_path_gain"]]])

    pyFDN.plot_fdn_parameter(
        delays_smp,
        A,
        B,
        C,
        D,
        attenuation_sos=sdn_result["wall_filters_sos"],
        fs=sdn_result["Fs"],
        title="SDN FDN parameters",
    )
    return


if __name__ == "__main__":
    app.run()
