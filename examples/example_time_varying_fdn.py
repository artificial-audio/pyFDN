# gallery_category: FDN Design & Analysis

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from docs.references import paper_link

    return mo, paper_link


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time Varying FDN
    """)
    return


@app.cell
def _(mo, paper_link):
    mo.md(f"""
    Example for time-varying matrices. <br/>
    Process a musical sound with a time-varying FDN reverberation. Different
    options include slow and fast time-variation.

    Reference: *{paper_link("Schlecht2015PracticalConsiderationsTimevarying")}*. <br/>
    Reference: *{paper_link("Schlecht2015TimevaryingFeedbackMatrices")}*.

    """)
    return


@app.cell
def _():
    import numpy as np
    import scipy.linalg as la

    import pyFDN
    from pyFDN.auxiliary.acoustics import one_pole_absorption
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix
    from pyFDN.generate.random_orthogonal import random_orthogonal
    from pyFDN.process import process_fdn

    return (
        TimeVaryingMatrix,
        la,
        np,
        one_pole_absorption,
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
def _(mo):
    sound_selection = mo.ui.dropdown(
        options=["sine", "melody"],
        value="melody",
        label="Sound",
    )
    mo.output.replace(sound_selection)
    return (sound_selection,)


@app.cell
def _(mo, np, pyFDN, sound_selection):
    np.random.seed(1)

    # init source signal
    mode = sound_selection.value

    if mode == "sine":
        fs = 48000
        duration = 4
        time = np.linspace(0, duration, duration * fs)[:, None]

        synth1 = 0.5 * np.sin(time * 440 * 2 * np.pi)
        synth2 = 0.5 * np.sin(time * 660 * 2 * np.pi)

        # Concatenate columns horizontally
        synth = synth1 + synth2
        synth[-2 * fs :, :] = 0.0

    elif mode == "melody":
        synth, fs = pyFDN.load_audio("synth_dry.wav")
        print(f"Loaded {len(synth)} samples at {fs} Hz ({len(synth) / fs:.2f} s)")

        samples = np.arange(len(synth))
        time = (samples / fs) * 1000 * 1000

    _audio_src = synth.T if synth.ndim == 2 else synth
    mo.vstack([mo.audio(_audio_src, fs)])
    return fs, synth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define FDN: Signal Dimensionality & Formatting
    """)
    return


@app.cell
def _(la, np, random_orthogonal):
    N = 8
    num_input = 1
    num_output = 2

    input_gain = la.orth(np.random.randn(N, num_input))

    random_matrix = np.random.randn(num_output, N)
    output_gain = la.orth(random_matrix.T).T

    direct = np.zeros((num_output, num_input))
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
def _(N, delays, fs, one_pole_absorption, pyFDN):
    RT_DC = 4  # seconds
    RT_NY = 1  # seconds

    coeffs = one_pole_absorption(RT_DC, RT_NY, delays, fs)

    # Constract the absorption
    absorption = pyFDN.SOSFilterBank(coeffs, N)
    return (absorption,)


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
    absorption,
    delays,
    direct,
    feedback_matrix,
    fs,
    input_gain,
    output_gain,
    process_fdn,
    synth,
):
    matrix_types = ["no_variation", "slow_variation", "fast_variation"]

    reverbed_synth = {}

    for matrix_type in matrix_types:
        if matrix_type == "no_variation":
            modulation_frequency = 0  # hz
            modulation_amplitude = 0.0
            spread = 0

        elif matrix_type == "slow_variation":
            modulation_frequency = 1.0  # hz
            modulation_amplitude = 3.0
            spread = 0.3

        elif matrix_type == "fast_variation":
            modulation_frequency = 10  # hz
            modulation_amplitude = 1.1
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
            post_delay=absorption,
            post_matrix=tv_matrix,
        )
    return matrix_types, reverbed_synth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Output Visualization
    """)
    return


@app.cell
def _(fs, matrix_types, mo, pyFDN, reverbed_synth):
    mo.vstack(
        [
            pyFDN.plot_spectrogram(
                reverbed_synth[name][:, 0],
                fs,
                nperseg=2048 * 8,
                noverlap=2048 * 1,
                title=f"{name} — spectrogram",
                colorscale="Magma",
                height=350,
            )
            for name in matrix_types
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Audio Playback
    """)
    return


@app.cell
def _(fs, matrix_types, mo, reverbed_synth):
    mo.vstack(
        [
            mo.vstack(
                [mo.md(f"**{name}**"), mo.audio(src=reverbed_synth[name].T, rate=fs)]
            )
            for name in matrix_types
        ]
    )
    return


if __name__ == "__main__":
    app.run()
