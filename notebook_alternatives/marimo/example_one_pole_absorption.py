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
    # One-pole absorption FDN (FLAMO integration)

    This example builds an FDN with **one-pole absorption filters** (designed with pyFDN) and runs it in **FLAMO** for the impulse response. Results are compared to a MATLAB reference when the reference file is available.

    **What it does:**
    - Loads reference parameters from a MATLAB export (or uses the path below).
    - Designs one-pole absorption coefficients with `pyFDN.one_pole_absorption` (T60 at DC and Nyquist).
    - Assembles the FDN in FLAMO (delays, absorption, feedback matrix, I/O gains) and computes the IR.
    - Plots IR vs reference, filter responses, spectrogram, RT60 curve, and coefficients.

    *(Run from repo root or `examples/` so `tests/reference/example_onePoleAbsorption.mat` is found.)*
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    from pathlib import Path

    from scipy.io import loadmat
    from scipy import signal
    from flamo.processor import dsp, system

    import pyFDN

    return (
        OrderedDict,
        Path,
        dsp,
        loadmat,
        np,
        plt,
        pyFDN,
        signal,
        system,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load reference data

    Load parameters and MATLAB reference IR from the regression .mat file (run from repo root or `examples/`).
    """)
    return


@app.cell
def _(Path, loadmat, np, torch):
    np.random.seed(1)
    torch.manual_seed(1)

    # Resolve repo root (nbsphinx runs with cwd = docs/examples/, so go up until we find tests/)
    _repo = Path.cwd()
    while _repo.name in ("examples", "docs"):
        _repo = _repo.parent

    # ref_path = _repo / "tests" / "reference" / "example_onePoleAbsorption.mat"
    ref_path = _repo.parent.parent / "tests" / "reference" / "example_onePoleAbsorption.mat"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = loadmat(str(ref_path), simplify_cells=True)

    rt_dc = float(ref["RT_DC"])
    rt_ny = float(ref["RT_NY"])
    delays = np.asarray(ref["delays"]).flatten()
    fs = int(ref["fs"])
    feedback_matrix = np.asarray(ref["feedbackMatrix"])
    ir_matlab = np.asarray(ref["irTimeDomain"]).flatten()
    n = int(ref["N"])

    print(f"Loaded: N={n}, fs={fs}, RT60 DC={rt_dc}s Nyquist={rt_ny}s")
    return delays, feedback_matrix, fs, ir_matlab, n, rt_dc, rt_ny


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One-pole absorption coefficients (pyFDN)

    Design SOS coefficients for each delay line from target T60 at DC and Nyquist.
    """)
    return


@app.cell
def _(delays, fs, pyFDN, rt_dc, rt_ny):
    sos = pyFDN.one_pole_absorption(rt_dc, rt_ny, delays, fs)
    print("SOS shape (6, N):", sos.shape)
    return (sos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build FDN in FLAMO

    Wire delays, one-pole absorption, feedback matrix, and I/O gains; then wrap in FFT/iFFT shell for time response.
    """)
    return


@app.cell
def _(
    OrderedDict,
    delays,
    dsp,
    feedback_matrix,
    fs,
    n,
    np,
    sos,
    system,
    torch,
):
    device = "cpu"
    n_fft = 2**16
    num_input = 1
    num_output = 1
    impulse_response_length = fs

    delays_torch = torch.tensor(delays, dtype=torch.float32)
    feedback_matrix_torch = torch.tensor(feedback_matrix, dtype=torch.float32)
    absorption_coeff = torch.tensor(sos[np.newaxis, ...], dtype=torch.float32)

    input_gain = dsp.Gain(size=(n, num_input), nfft=n_fft, device=device)
    input_gain.assign_value(torch.ones(n, num_input))

    output_gain = dsp.Gain(size=(num_output, n), nfft=n_fft, device=device)
    output_gain.assign_value(torch.ones(num_output, n))

    delay_module = dsp.parallelDelay(
        size=(n,),
        max_len=int(delays_torch.max()),
        nfft=n_fft,
        isint=True,
        unit=1,
        device=device,
    )
    delay_module.assign_value(delay_module.sample2s(delays_torch))

    mixing_matrix = dsp.Matrix(size=(n, n), nfft=n_fft, matrix_type="random", device=device)
    mixing_matrix.assign_value(feedback_matrix_torch)

    absorption = dsp.parallelSOSFilter(size=(n,), n_sections=1, nfft=n_fft, device=device)
    absorption.assign_value(absorption_coeff)

    attenuated_delay = system.Series(
        OrderedDict({"delay": delay_module, "absorption": absorption})
    )
    feedback_loop = system.Recursion(fF=attenuated_delay, fB=mixing_matrix)
    fdn = system.Series(
        OrderedDict({
            "input_gain": input_gain,
            "feedback_loop": feedback_loop,
            "output_gain": output_gain,
        })
    )

    direct_gain = dsp.Gain(size=(num_output, num_input), nfft=n_fft, device=device)
    direct_gain.assign_value(torch.ones(num_output, num_input))

    complete_system = system.Parallel(brA=direct_gain, brB=fdn, sum_output=True)
    model = system.Shell(
        core=complete_system,
        input_layer=dsp.FFT(n_fft),
        output_layer=dsp.iFFT(n_fft),
    )
    return impulse_response_length, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate and compare IR
    """)
    return


@app.cell
def _(impulse_response_length, ir_matlab, model, np):
    ir_python = model.get_time_response().flatten()
    ir_python = ir_python[:impulse_response_length]
    ir_matlab_1 = ir_matlab[:impulse_response_length]
    correlation = np.corrcoef(ir_python, ir_matlab_1)[0, 1]
    print(f'Correlation (FLAMO vs MATLAB): {correlation:.6f}')
    return ir_matlab_1, ir_python


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots

    IR vs reference, one-pole filter responses, spectrogram, theoretical RT60, and filter coefficients.
    """)
    return


@app.cell
def _(
    delays,
    fs,
    ir_matlab_1,
    ir_python,
    n,
    np,
    plt,
    pyFDN,
    rt_dc,
    rt_ny,
    signal,
    sos,
):
    t = np.arange(len(ir_python)) / fs
    freqs = np.logspace(1, np.log10(fs / 2), 100)
    omega = 2 * np.pi * freqs / fs
    f_spec, t_spec, Sxx = signal.spectrogram(ir_python, fs, nperseg=512, noverlap=384)
    rt_theory = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        omega_f = 2 * np.pi * f / fs
        h_avg = np.mean([np.abs(sos[0, j] / (1 + sos[4, j] * np.exp(-1j * omega_f))) for j in range(n)])
        if h_avg > 0:
            slope = 20 * np.log10(h_avg) / np.mean(delays) * fs
            rt_theory[i] = -60 / slope if slope < 0 else 10
    plt.figure(figsize=(8, 3))

    # Impulse response
    plt.plot(t, pyFDN.mulaw_encode(ir_python), 'b-', alpha=0.7, linewidth=0.5, label='FLAMO (Python)')
    plt.plot(t, pyFDN.mulaw_encode(ir_matlab_1), 'r--', alpha=0.7, linewidth=0.5, label='MATLAB reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude [mu-law]')
    plt.title('Impulse response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 3))
    plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')

    # Spectrogram
    plt.xlabel('Time (s)')
    plt.title('Spectrogram (FLAMO)')
    plt.ylim([0, fs / 2])
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 4))
    for i in range(n):
        H = sos[0, i] / (1 + sos[4, i] * np.exp(-1j * omega))
        plt.semilogx(freqs, 20 * np.log10(np.abs(H)), label=f'Delay {i + 1} ({delays[i]} smp)')

    # One-pole filter responses
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('One-pole filter responses')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim([20, fs / 2])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(6, 4))
    plt.plot(delays, sos[0, :], 'bo', label='b0')
    plt.plot(delays, sos[4, :], 'ro', label='a1')
    plt.xlabel('Delay (samples)')
    plt.ylabel('Coefficient value')

    # One-pole coefficients
    plt.title('One-pole filter coefficients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.semilogx(freqs, rt_theory, 'b-', linewidth=2, label='Theoretical')
    plt.axhline(y=rt_dc, color='g', linestyle='--', label=f'Target DC: {rt_dc}s')
    plt.axhline(y=rt_ny, color='r', linestyle='--', label=f'Target Nyquist: {rt_ny}s')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RT60 (s)')

    # RT60 vs frequency
    plt.title('Reverberation time vs frequency')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim([20, fs / 2])
    plt.ylim([0, max(rt_dc * 1.2, 4)])
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
