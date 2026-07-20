# gallery_category: Special FDNs

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from docs.references import paper_link

    return mo, paper_link


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    # Coupled Rooms FDN Example

    This example models **two acoustically coupled rooms** with a Feedback Delay Network (FDN): one small room with a short reverberation time (RT), one large room with a long RT. Each room is an independent FDN with **frequency-dependent RT** (first-order shelving absorption) and its own **output EQ**.

    The coupled FDN is assembled by simple **concatenation** of the two single-room FDNs (block-diagonal feedback matrix, stacked delays and absorption filters) plus a single **mixing matrix** that couples the two rooms.

    - **Sound source:** in the first room (short RT) only.
    - **Receivers:** two outputs, one per room — channel 1 is room 1 (short RT), channel 2 is room 2 (long RT).

    Translation of `example_coupledRooms.m` to Python using FLAMO.

    Based on:
    > *{paper_link("Delay_Network_Architectures_for_Room_and_Coupled_Space_Modeling")}*.

    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.linalg import block_diag

    import pyFDN

    return block_diag, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single-room FDNs

    Build each room with `pyFDN.fdn_build_gallery`. Both rooms have
    frequency-dependent reverberation (RT at DC and at Nyquist, realised as
    per-line first-order shelving absorption) and a different per-room output EQ.
    """)
    return


@app.cell
def _(pyFDN):
    fs = 48000
    nfft = 2**17

    # Equal room sizes keep the coupling matrix a simple block rotation.
    N1 = N2 = 12
    N = N1 + N2
    ix1 = slice(0, N1)  # delay lines of room 1
    ix2 = slice(N1, N)  # delay lines of room 2
    num_input = 1
    num_output = 2

    # Room 1: small, short RT, brighter (high frequencies decay faster); dark EQ.
    room1 = pyFDN.fdn_build_gallery(
        N1,
        fs=fs,
        delay_range=(400, 800),
        rt=0.825,
        rt_nyquist=0.6,
        rt_crossover=1000.0,
        post_eq_db_dc=0.0,
        post_eq_db_nyquist=-8.0,
        post_eq_crossover=1000.0,
        io_type="ones",
        rng=5,
    )
    # Room 2: large, long RT; different (brighter) EQ.
    room2 = pyFDN.fdn_build_gallery(
        N2,
        fs=fs,
        delay_range=(1100, 2600),
        rt=4.2,
        rt_nyquist=1.5,
        rt_crossover=2000.0,
        post_eq_db_dc=2.0,
        post_eq_db_nyquist=-12.0,
        post_eq_crossover=2000.0,
        io_type="ones",
        rng=6,
    )
    return N, fs, ix1, ix2, nfft, num_input, num_output, room1, room2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concatenate and couple

    The coupled FDN is the concatenation of the two rooms — block-diagonal
    feedback matrix, stacked delays and absorption filters, source in room 1,
    one output per room — composed with a single orthogonal **mixing matrix**
    that couples the rooms by a coupling angle.
    """)
    return


@app.cell
def _(N, block_diag, ix1, ix2, np, num_input, num_output, room1, room2):
    coupling = 0.2  # coupling angle between the two rooms (0 = uncoupled)

    # Concatenate the two single-room FDNs.
    delays = np.concatenate([room1.delays, room2.delays])
    A_rooms = block_diag(room1.A, room2.A)
    attenuation_sos = np.concatenate([room1.filters, room2.filters], axis=2)
    post_eq_sos = np.concatenate([room1.post_eq, room2.post_eq], axis=2)

    # Mixing matrix: orthogonal block rotation coupling room 1 with room 2.
    eye = np.eye(room1.delays.size)
    mixing_matrix = np.block(
        [
            [np.cos(coupling) * eye, np.sin(coupling) * eye],
            [-np.sin(coupling) * eye, np.cos(coupling) * eye],
        ]
    )
    A = mixing_matrix @ A_rooms

    # Source in room 1 only; one receiver per room.
    B = np.zeros((N, num_input))
    B[ix1] = room1.B
    C = np.zeros((num_output, N))
    C[0, ix1] = room1.C[0]
    C[1, ix2] = room2.C[0]
    D = np.zeros((num_output, num_input))
    return A, B, C, D, attenuation_sos, delays, post_eq_sos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build FDN (FLAMO)
    """)
    return


@app.cell
def _(A, B, C, D, attenuation_sos, delays, fs, nfft, post_eq_sos, pyFDN):
    model = pyFDN.dss_to_flamo(
        A,
        B,
        C,
        D,
        delays,
        fs,
        nfft=nfft,
        sos_filter=attenuation_sos,
        output_filter=post_eq_sos,
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response
    """)
    return


@app.cell
def _(model, pyFDN):
    ir = pyFDN.flamo_time_response(model).squeeze()
    return (ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN signal-flow graph
    """)
    return


@app.cell
def _(model, pyFDN):
    pyFDN.plot_flamo_graph(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameters
    """)
    return


@app.cell
def _(A, B, C, D, attenuation_sos, delays, fs, post_eq_sos, pyFDN):
    pyFDN.plot_fdn_parameter(
        delays,
        A,
        B,
        C,
        D,
        attenuation_sos=attenuation_sos,
        post_eq_sos=post_eq_sos,
        fs=fs,
        title="Coupled Rooms FDN Parameters",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots and audio
    """)
    return


@app.cell
def _(fs, ir, pyFDN):
    pyFDN.plot_impulse_response(
        ir[:, 0],
        ir[:, 1],
        fs=fs,
        labels=["Room 1 (source)", "Room 2"],
        title="Coupled Rooms Impulse Response",
    )

    fig_edc = pyFDN.plot_edc(
        ir[:, 0],
        ir[:, 1],
        fs=fs,
        labels=["Room 1", "Room 2"],
        title="Energy decay curves",
    )
    fig_edc.update_xaxes(range=[0, min(2, len(ir) / fs)])
    fig_edc.update_yaxes(range=[-40, 15])
    return


@app.cell
def _(fs, ir, mo):
    mo.vstack(
        [
            mo.md("Room 1 RIR:"),
            mo.audio(ir[:, 0], rate=fs),
            mo.md("Room 2 RIR:"),
            mo.audio(ir[:, 1], rate=fs),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
