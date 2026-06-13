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
    # Coupled Rooms FDN Example

    This example models **two acoustically coupled rooms** with a Feedback Delay Network (FDN): one room with shorter reverb time (RT), one with longer RT.

    - **Sound source:** in the first room (shorter RT). The input is injected only into the first room’s delay lines.
    - **Receivers:** two outputs, one receiver in each room — channel 1 is the first room (short RT), channel 2 is the second room (long RT).

    Translation of `example_coupledRooms.m` to Python using FLAMO.

    Based on:
    > Das, O., Abel, J. S. & Canfield-Dafilou, E. K. *Delay Network Architectures For Room And Coupled Space Modeling*. DAFx2020 (2020).

    **Original MATLAB:** (c) Sebastian Jiro Schlecht, 2020. **Python:** Facundo Franchino, 2025.
    """)
    return


@app.cell
def _():
    from collections import OrderedDict

    import numpy as np
    import torch
    from flamo.processor import dsp, system

    import pyFDN

    return OrderedDict, dsp, np, pyFDN, system, torch


@app.cell
def _(np, torch):
    # Reproducibility (matches MATLAB rng(5))
    torch.manual_seed(5)
    np.random.seed(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN Configuration
    """)
    return


@app.cell
def _(pyFDN, torch):
    fs = 48000
    nfft = 2**17

    N1 = 6
    N2 = 6
    small_room = pyFDN.fdn_build_gallery(
        N1, fs=fs, delay_range=(400, 800), rt60=0.525, io_type="ones", rng=5
    )
    large_room = pyFDN.fdn_build_gallery(
        N2, fs=fs, delay_range=(1100, 2600), rt60=4.2, io_type="ones", rng=6
    )
    N = N1 + N2
    ix1 = slice(0, N1)  # indices for room 1
    ix2 = slice(N1, N)  # indices for room 2

    num_input = 1
    num_output = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}, FDN: {N} delay lines (room 1: {N1}, room 2: {N2})")
    return (
        N,
        N1,
        device,
        fs,
        ix1,
        ix2,
        large_room,
        nfft,
        num_input,
        num_output,
        small_room,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Delay lines

    Random short and long room delays from the gallery presets.
    """)
    return


@app.cell
def _(large_room, small_room, torch):
    delays_room1 = torch.as_tensor(small_room.delays, dtype=torch.float32)
    delays_room2 = torch.as_tensor(large_room.delays, dtype=torch.float32)
    delay_lengths = torch.cat([delays_room1, delays_room2])

    print(f"Room 1 delays: {delays_room1.tolist()} samples")
    print(f"Room 2 delays: {delays_room2.tolist()} samples")
    return delay_lengths, delays_room1, delays_room2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feedback matrix

    Couple the orthogonal feedback matrices of the small-room and large-room
    gallery builds.
    """)
    return


@app.cell
def _(N, ix1, ix2, large_room, pyFDN, small_room, torch):
    coupling = 0.3
    A1 = torch.as_tensor(small_room.A, dtype=torch.float32)
    A2 = torch.as_tensor(large_room.A, dtype=torch.float32)
    A1_sqrt = pyFDN.matrix_sqrt(A1)
    A2_sqrt = pyFDN.matrix_sqrt(A2)

    cos_c = torch.cos(torch.tensor(coupling))
    sin_c = torch.sin(torch.tensor(coupling))
    feedback_matrix = torch.zeros(N, N)
    feedback_matrix[ix1, ix1] = cos_c * A1
    feedback_matrix[ix1, ix2] = sin_c * torch.matmul(A1_sqrt, A2_sqrt)
    feedback_matrix[ix2, ix1] = -sin_c * torch.matmul(A2_sqrt, A1_sqrt)
    feedback_matrix[ix2, ix2] = cos_c * A2
    return (feedback_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build FDN (FLAMO)

    Source in room 1 only; one output per room.
    """)
    return


@app.cell
def _(
    N,
    device,
    dsp,
    ix1,
    ix2,
    large_room,
    nfft,
    num_input,
    num_output,
    small_room,
    torch,
):
    # Source in room 1 only
    input_gain = dsp.Gain(
        size=(N, num_input), nfft=nfft, requires_grad=False, device=device
    )
    input_gain_values = torch.zeros(N, num_input)
    input_gain_values[ix1, :] = torch.as_tensor(
        small_room.B, dtype=input_gain_values.dtype
    )
    input_gain.assign_value(input_gain_values)

    # One receiver per room (out 1 = room 1, out 2 = room 2)
    output_gain = dsp.Gain(
        size=(num_output, N), nfft=nfft, requires_grad=False, device=device
    )
    output_gain_values = torch.zeros(num_output, N)
    output_gain_values[0, ix1] = torch.as_tensor(
        small_room.C[0], dtype=output_gain_values.dtype
    )
    output_gain_values[1, ix2] = torch.as_tensor(
        large_room.C[0], dtype=output_gain_values.dtype
    )
    output_gain.assign_value(output_gain_values)
    return input_gain, output_gain


@app.cell
def _(N, delay_lengths, device, dsp, feedback_matrix, nfft):
    delays = dsp.parallelDelay(
        size=(N,),
        max_len=int(delay_lengths.max()),
        nfft=nfft,
        isint=True,
        requires_grad=False,
        device=device,
    )
    delays.assign_value(delays.sample2s(delay_lengths.int()))

    mixing_matrix = dsp.Matrix(
        size=(N, N),
        nfft=nfft,
        matrix_type="random",
        requires_grad=False,
        device=device,
    )
    mixing_matrix.assign_value(feedback_matrix)
    return delays, mixing_matrix


@app.cell
def _(N, device, dsp, large_room, nfft, np, small_room, torch):
    def _dc_gain(filters):
        numerator = np.prod(np.sum(filters[:, :3, :], axis=1), axis=0)
        denominator = np.prod(np.sum(filters[:, 3:, :], axis=1), axis=0)
        return numerator / denominator

    attenuation_values = torch.as_tensor(
        np.concatenate([_dc_gain(small_room.filters), _dc_gain(large_room.filters)]),
        dtype=torch.float32,
    )

    attenuation = dsp.parallelGain(
        size=(N,), nfft=nfft, requires_grad=False, device=device
    )
    attenuation.assign_value(attenuation_values)
    print("Using small-room and large-room gallery absorption")
    return (attenuation,)


@app.cell
def _(
    OrderedDict,
    attenuation,
    delays,
    dsp,
    input_gain,
    mixing_matrix,
    nfft,
    output_gain,
    system,
):
    feedback = system.Series(
        OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
    )
    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    fdn = system.Series(
        OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )
    model = system.Shell(
        core=fdn, input_layer=dsp.FFT(nfft), output_layer=dsp.iFFT(nfft)
    )
    print("FDN built.")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response
    """)
    return


@app.cell
def _(fs, model, pyFDN):
    ir = pyFDN.flamo_time_response(model).squeeze()
    print(f"IR shape: {ir.shape}, duration: {ir.shape[0] / fs:.2f} s")
    return (ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots and audio
    """)
    return


@app.cell
def _(fs, ir, pyFDN):
    fig_ir = pyFDN.plot_impulse_response(
        ir[:, 0],
        ir[:, 1],
        fs=fs,
        labels=["Room 1 (source)", "Room 2"],
        title="Coupled Rooms Impulse Response",
    )
    return (fig_ir,)


@app.cell
def _(
    fig_ir,
    fs,
    ir,
    mo,
    model,
    pyFDN,
):
    parameters = pyFDN.flamo_model_to_fdn_parameters(model)

    fig_parameters = pyFDN.plot_fdn_parameter(
        parameters.delays,
        parameters.A,
        parameters.B,
        parameters.C,
        parameters.D,
        attenuation_sos=parameters.attenuation_sos,
        post_eq_sos=parameters.post_eq_sos,
        fs=parameters.fs or fs,
        title="Coupled Rooms FDN Parameters",
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

    mo.vstack(
        [
            fig_ir,
            fig_parameters,
            fig_edc,
            mo.md("Room 1 RIR:"),
            mo.audio(ir[:, 0], rate=fs),
            mo.md("Room 2 RIR:"),
            mo.audio(ir[:, 1], rate=fs),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
