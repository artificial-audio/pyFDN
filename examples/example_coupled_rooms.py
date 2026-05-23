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

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from flamo.processor import dsp, system
    import pyFDN

    return OrderedDict, dsp, np, plt, pyFDN, system, torch


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
def _(torch):
    fs = 48000
    nfft = 2**17

    # Delay lines per room (defines the split; change only here to resize rooms)
    N1 = 6
    N2 = 6
    N = N1 + N2
    ix1 = slice(0, N1)  # indices for room 1
    ix2 = slice(N1, N)  # indices for room 2

    num_input = 1
    num_output = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}, FDN: {N} delay lines (room 1: {N1}, room 2: {N2})")
    return N, N1, N2, device, fs, ix1, ix2, nfft, num_input, num_output


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Delay lines

    Per-room delay lengths (samples); same seed as reference MATLAB.
    """)
    return


@app.cell
def _(torch):
    # Per-room delay lengths (samples); lengths must match N1 and N2
    delays_room1 = torch.tensor([411, 736, 403, 760, 544, 606], dtype=torch.float32)
    delays_room2 = torch.tensor([2532, 2037, 1593, 1375, 1161, 2477], dtype=torch.float32)
    delay_lengths = torch.cat([delays_room1, delays_room2])

    print(f"Room 1 delays: {delays_room1.tolist()} samples")
    print(f"Room 2 delays: {delays_room2.tolist()} samples")
    return (delay_lengths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feedback matrix

    Coupled orthogonal matrix from `tiny_rotation_matrix` per room and a coupling parameter.
    """)
    return


@app.cell
def _(N, N1, N2, ix1, ix2, pyFDN, torch):
    coupling = 0.3
    A1 = pyFDN.tiny_rotation_matrix(N1, 12).float()
    A2 = pyFDN.tiny_rotation_matrix(N2, 12).float()
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
def _(N, device, dsp, ix1, ix2, nfft, num_input, num_output, torch):
    # Source in room 1 only
    input_gain = dsp.Gain(size=(N, num_input), nfft=nfft, requires_grad=False, device=device)
    input_gain_values = torch.zeros(N, num_input)
    input_gain_values[ix1, :] = 1.0
    input_gain.assign_value(input_gain_values)

    # One receiver per room (out 1 = room 1, out 2 = room 2)
    output_gain = dsp.Gain(size=(num_output, N), nfft=nfft, requires_grad=False, device=device)
    output_gain_values = torch.zeros(num_output, N)
    output_gain_values[0, ix1] = 1.0
    output_gain_values[1, ix2] = 1.0
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
def _(N, delay_lengths, device, dsp, fs, ix1, ix2, nfft, pyFDN, torch):
    # Per-room RT (1 kHz band); frequency-independent attenuation
    short_rt = torch.tensor([0.5, 0.5, 0.55, 0.575, 0.525, 0.375, 0.275, 0.2, 0.175, 0.175])
    long_rt = torch.tensor([4.0, 4.0, 4.4, 4.6, 4.2, 3.0, 2.2, 1.6, 1.4, 1.4])
    short_rt_1khz = short_rt[4].item()
    long_rt_1khz = long_rt[4].item()

    attenuation_values = torch.zeros(N)
    g_short = pyFDN.rt_to_gain_per_sample(short_rt_1khz, fs)
    g_long = pyFDN.rt_to_gain_per_sample(long_rt_1khz, fs)
    attenuation_values[ix1] = g_short ** delay_lengths[ix1]
    attenuation_values[ix2] = g_long ** delay_lengths[ix2]

    attenuation = dsp.parallelGain(size=(N,), nfft=nfft, requires_grad=False, device=device)
    attenuation.assign_value(attenuation_values)
    print(f"Room 1 RT (1 kHz): {short_rt_1khz:.2f} s, Room 2: {long_rt_1khz:.2f} s")
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
    feedback = system.Series(OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation}))
    feedback_loop = system.Recursion(fF=delays, fB=feedback)
    fdn = system.Series(OrderedDict(
            {
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }
        )
    )
    model = system.Shell(core=fdn, input_layer=dsp.FFT(nfft), output_layer=dsp.iFFT(nfft))
    print("FDN built.")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response
    """)
    return


@app.cell
def _(fs, model, np):
    ir = np.asarray(model.get_time_response().squeeze())
    print(f"IR shape: {ir.shape}, duration: {ir.shape[0] / fs:.2f} s")
    return (ir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots and audio
    """)
    return


@app.cell
def _(N1, feedback_matrix, fs, ir, mo, np, plt, pyFDN):
    fig = plt.figure(figsize=(15, 5))


    # Plot 1: Impulse responses
    ax1 = plt.subplot(1, 3, 1)
    samples = np.arange(len(ir))
    ax1.plot(
        samples,
        pyFDN.mulaw_encode(ir[:, 0], 1),
        label="Room 1 (source)",
        alpha=0.7,
        linewidth=0.5,
    )
    ax1.plot(
        samples,
        pyFDN.mulaw_encode(ir[:, 1], 1) - 2,
        label="Room 2",
        alpha=0.7,
        linewidth=0.5,
    )
    ax1.set_xlabel("Time [samples]")
    ax1.set_ylabel("Amplitude [mu-law]")
    ax1.set_title("Coupled Rooms Impulse Response")
    ax1.legend()

    ax1.grid(True, alpha=0.3)
    ax1.set_xlim((0, len(ir)))

    # Plot 2: Feedback matrix
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(feedback_matrix.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("Feedback Matrix")
    ax2.axhline(y=N1 - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.axvline(x=N1 - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)

    # Plot 3: Energy decay curves
    ax3 = plt.subplot(1, 3, 3)
    t = np.arange(len(ir)) / fs
    edc_db = pyFDN.sq_to_db(pyFDN.edc(ir))
    ax3.plot(t, edc_db[:, 0], label="Room 1", alpha=0.8)
    ax3.plot(t, edc_db[:, 1], label="Room 2", alpha=0.8)

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Energy [dB]")
    ax3.set_title("Energy Decay Curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim((0, min(2, len(ir) / fs)))
    ax3.set_ylim((-40, 15))

    plt.tight_layout()
    plt.show()

    # add play widget
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
