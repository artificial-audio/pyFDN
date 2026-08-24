# gallery_category: Special FDNs
# gallery_description: Process music through an FDN whose orthogonal feedback matrix changes over time at selectable modulation rates.

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
    # Time-varying FDN

    A static FDN has fixed modes, and a sustained tone excites the same few of them for as long as it lasts — which is what "metallic" or "ringing" describes. Modulating the feedback matrix moves the modes while the sound decays, so no single one is driven long enough to stand out.

    The modulation costs something too: move the matrix too fast and the shifting pitch becomes audible as chorusing. This notebook runs one musical phrase through three settings — none, slow, fast — so the trade is audible in the same signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    Reference: *{pyFDN.paper_link("Schlecht2015PracticalConsiderationsTimevarying")}*. <br/>
    Reference: *{pyFDN.paper_link("Schlecht2015TimevaryingFeedbackMatrices")}*.

    """)
    return


@app.cell
def _():
    import numpy as np
    import scipy.linalg as la

    import pyFDN
    from pyFDN import td
    from pyFDN.generate.random_orthogonal import random_orthogonal
    from pyFDN.process import process_fdn

    return (
        la,
        np,
        process_fdn,
        pyFDN,
        random_orthogonal,
        td,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The dry signal

    Either a two-tone sine that stops two seconds before the end — the tail is then all reverb, which is where modulation artefacts are easiest to hear — or a synth phrase.
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
        synth, fs = pyFDN.load_audio("synth_dry")

        print(f"Loaded {len(synth)} samples at {fs} Hz ({len(synth) / fs:.2f} s)")

        samples = np.arange(len(synth))
        time = (samples / fs) * 1000 * 1000

    _audio_src = synth.T if synth.ndim == 2 else synth
    mo.audio(_audio_src, fs)
    return fs, synth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The FDN

    Eight delay lines, one input and two outputs for a stereo return. The input and output gains are orthonormalised so neither adds any colouration of its own, leaving the loop responsible for everything that is heard.
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
    ## Absorption

    A one-pole filter per delay line, giving 4 s of reverberation at DC falling to 1 s at Nyquist. Absorption is what makes the decay finite; the modulation below changes *which* modes decay, not how fast.
    """)
    return


@app.cell
def _(delays, fs, pyFDN, td):
    RT_DC = 4  # seconds
    RT_NY = 1  # seconds

    coeffs = pyFDN.decay_to_one_pole(RT_DC, RT_NY, delays, fs)

    absorption = td.SOSBank(coeffs)
    return (absorption,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three modulation rates

    `td.TimeVaryingMatrix` rotates the feedback matrix continuously, and takes a modulation frequency, an amplitude and a spread over the delay lines. The same phrase is rendered three times: static, 1 Hz, and 10 Hz.
    """)
    return


@app.cell
def _(
    N,
    absorption,
    delays,
    direct,
    feedback_matrix,
    fs,
    input_gain,
    output_gain,
    process_fdn,
    synth,
    td,
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

        tv_matrix = td.TimeVaryingMatrix(
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
    ## What the modulation does to the spectrum

    Read the three spectrograms downwards. The static FDN leaves horizontal ridges where individual modes sustain; slow modulation smears them; fast modulation broadens them into bands, which is the point at which the movement starts to become audible in its own right.
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
    ## Listen

    The ringing of the static version and the chorusing of the fast one are both clearest in the tail, once the dry signal has stopped.
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
