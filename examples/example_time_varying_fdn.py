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
    # Time Varying FDN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Example for time-varying matrices.<br/>
    Process a musical sound with a time-varying FDN reverberation. Different
    options include slow and fast time-variation.


    Reference: *Schlecht and Habets 2015 : "Practical Considerations of Time-Varying
    Feedback Delay Networks"* <br/>
    Reference: *Schlecht and Habets 2015 : "Time-varying feedback matrices in feedback delay networks
    and their application in artificial reverberation"*

    Original MATLAB: Sebastian J. Schlecht, Saturday, 28 December 2019
    """)
    return


@app.cell
def _():
    import numpy as np
    import scipy.linalg as la
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from pyFDN.auxiliary.acoustics import one_pole_absorption
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix
    from pyFDN.generate.random_orthogonal import random_orthogonal
    from pyFDN.process import process_fdn
    import pyFDN

    return (
        TimeVaryingMatrix,
        la,
        np,
        one_pole_absorption,
        plt,
        process_fdn,
        pyFDN,
        random_orthogonal,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Object Initialization & Audio Loading
    """)
    return


@app.cell
def _(mo, np, pyFDN):
    np.random.seed(1)

    # init source signal
    mode = 'melody'
    _output = None

    if mode == 'sine':
        fs = 48000
        time = np.linspace(0, 4, 4 * fs)[:, None]

        synth1 = 0.5 * np.sin(time * 440 * 2 * np.pi)
        synth2 = 0.5 * np.sin(time * 660 * 2 * np.pi)

        # Concatenate columns horizontally
        synth = np.hstack((synth1, synth2))

    elif mode == 'melody':
        synth, fs = pyFDN.load_audio("synth_dry.wav")
        print(f"Loaded {len(synth)} samples at {fs} Hz ({len(synth) / fs:.2f} s)")

        samples = np.arange(len(synth))
        time = ((samples / fs) * 1000 * 1000)

        _output = mo.vstack([mo.audio(synth, fs)])

    _output
    return fs, mode, synth, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define FDN: Signal Dimensionality & Formatting
    """)
    return


@app.cell
def _(la, mode, np, random_orthogonal):
    N = 8
    num_input = 1 if mode == 'melody' else 2
    num_output = 2

    input_gain = la.orth(np.random.randn(N, num_input))

    random_matrix = np.random.randn(num_output, N)
    output_gain = la.orth(random_matrix.T).T

    direct = np.zeros((num_output,num_input))
    delays = np.random.randint(750, 2001, size=N)[None, :]

    feedback_matrix = random_orthogonal(N)
    return N, delays, direct, feedback_matrix, input_gain, output_gain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate Absorption Ailter
    """)
    return


@app.cell
def _(delays, fs, one_pole_absorption):
    RT_DC = 4 # seconds
    RT_NY = 1 # seconds

    coeffs = one_pole_absorption(RT_DC, RT_NY, delays, fs)
    return (coeffs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time Varying Matrix Generation & Reverberation Processing Across Matrix Variations
    """)
    return


@app.cell
def _(
    N,
    TimeVaryingMatrix,
    coeffs,
    delays,
    direct,
    feedback_matrix,
    fs,
    input_gain,
    output_gain,
    process_fdn,
    synth,
):
    matrix_types = ['no_variation', 'slow_variation','fast_variation']

    reverbed_synth = {}

    for matrix_type in matrix_types:
        if matrix_type == 'no_variation':
            modulation_frequency = 0  # hz
            modulation_amplitude = 0.0
            spread = 0

        elif matrix_type == 'slow_variation':
            modulation_frequency = 1  # hz
            modulation_amplitude = 0.9
            spread = 0.3

        elif matrix_type == 'fast_variation':
            modulation_frequency = 10  # hz
            modulation_amplitude = 0.1
            spread = 0.7

        tv_matrix = TimeVaryingMatrix(
            N, modulation_frequency, modulation_amplitude, fs, spread
        )

        reverbed_synth[matrix_type] = process_fdn(
            synth,
            delays,
            feedback_matrix,
            input_gain,
            output_gain,
            direct,
            absorption_filters=coeffs,
            extra_matrix=tv_matrix,
        )
    return matrix_types, reverbed_synth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Output Visualization
    """)
    return


@app.cell
def _(matrix_types, plt, reverbed_synth, time):
    plt.figure(figsize=(10, 6))
    plt.title("Time-Varying FDN Output")
    plt.grid(True, linestyle='--', alpha=0.6) 

    # Plot each matrix type with distinct styles
    for it, name in enumerate(matrix_types):
        plt.plot(
            time, 
            reverbed_synth[name][:, 0] + it * 1.5,
            label=name,
            linewidth=1.5
        )

    plt.legend(loc='upper right', title="Matrix Types")
    plt.xlabel("Time [seconds]")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Audio Playback
    """)
    return


@app.cell
def _(fs, matrix_types, mo, reverbed_synth):
    mo.vstack([
        mo.vstack([
            mo.md(f"**{name}**"),
            mo.audio(src=reverbed_synth[name].T, rate=fs)
        ])
        for name in matrix_types
    ])
    return


if __name__ == "__main__":
    app.run()
