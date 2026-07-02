# gallery_category: Absorption & Filters

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from docs.references import paper_link

    return mo, paper_link


@app.cell(hide_code=True)
def _(mo, paper_link):
    mo.md(f"""
    # Converting a room impulse response into an FDN

    Estimates the frequency-dependent decay of a measured room impulse
    response and designs an FDN to match it:

    1. Estimate RT and initial level in octave bands from the RIR.
    2. Design per-delay-line GEQ absorption filters matching the decay.
    3. Design an output GEQ matching the initial spectral level.
    4. Compare the FDN impulse response with the target RIR.

    The impulse response is from the Promenadikeskus concert hall in Pori,
    Finland, published at
    {paper_link("Concert_Hall_Impulse_Responses")}.

    Instead of DecayFitNet (used in the MATLAB original), decay parameters are
    estimated with `estimate_rt_bands` (Schroeder backward integration per
    octave band) and `estimate_initial_level_bands` (band energy matched to an
    exponential decay model).  The output EQ is designed from the *difference*
    between the target and the unequalized FDN band levels, which makes the
    level match self-correcting.

    Original MATLAB: `example_RIR2FDN.m`, Sebastian J. Schlecht, 28 January 2023.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return go, np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load room impulse response

    Trim to the onset (direct sound) and normalize the energy.
    """)
    return


@app.cell
def _(np, pyFDN):
    rir, fs = pyFDN.load_audio("s3_r4_o.wav")
    _onset = int(np.argmax(np.abs(rir)))
    rir = rir[_onset:]
    rir = rir / np.linalg.norm(rir)
    rir_len = len(rir)

    print(f"RIR: {rir_len} samples ({rir_len / fs:.2f} s) at {fs} Hz")
    return fs, rir, rir_len


@app.cell
def _(fs, mo, pyFDN, rir):
    mo.audio(pyFDN.peak_normalize(rir), fs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimate decay parameters

    RT and initial level per octave band (63 Hz – 8 kHz).
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    est_rt, f_centre = pyFDN.estimate_rt_bands(rir, fs)
    est_level, _ = pyFDN.estimate_initial_level_bands(rir, est_rt, fs)

    print(f"Bands (Hz):  {f_centre}")
    print(f"RT (s):    {est_rt.round(2)}")
    print(f"Level (dB):  {pyFDN.lin_to_db(est_level).round(1)}")
    return est_level, est_rt, f_centre


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define FDN and absorption filters

    A 16-delay FDN with a random orthogonal feedback matrix.  The target RT
    at the 10 GEQ design bands (DC, 63 Hz … 8 kHz, Nyquist) extends the octave
    band estimates, shortening the lowest and the two highest bands (air and
    boundary absorption shortens the decay at the spectral edges).
    """)
    return


@app.cell
def _(est_rt, fs, np, pyFDN):
    np.random.seed(5)
    num_delays = 16
    build = pyFDN.fdn_build_gallery(
        num_delays,
        fs=fs,
        delay_range=(500, 3500),
        io_type="ones",
        direct_gain=0.0,
        rt=None,
        rng=5,
    )
    delays = build.delays
    feedback_matrix = build.A
    input_gain, output_gain, direct_gain = build.B, build.C, build.D

    target_rt = np.concatenate(([est_rt[0]], est_rt, [est_rt[-1]]))
    target_rt = target_rt * np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.5])
    sos_absorption = pyFDN.absorption_geq(target_rt, delays, fs)

    print(f"Delays: {delays}")
    print(f"Target RT at GEQ bands (s): {target_rt.round(2)}")
    return (
        delays,
        direct_gain,
        feedback_matrix,
        input_gain,
        output_gain,
        sos_absorption,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute the unequalized FDN impulse response

    The absorption filters sit in the recursion loop of a FLAMO model
    (input → B → [delays → SOS → A] → C → output).  This first model has no
    output equalizer yet; its impulse response provides the reference level
    for the EQ design.
    """)
    return


@app.cell
def _(
    delays,
    direct_gain,
    feedback_matrix,
    fs,
    input_gain,
    np,
    output_gain,
    pyFDN,
    rir_len,
    sos_absorption,
):
    nfft = int(2 ** np.ceil(np.log2(rir_len)))
    _model = pyFDN.dss_to_flamo(
        feedback_matrix,
        input_gain,
        output_gain,
        direct_gain,
        delays,
        fs,
        nfft=nfft,
        sos_filter=sos_absorption,
        shell=True,
    )
    ir_unequalized = pyFDN.flamo_time_response(_model).squeeze()[:rir_len]
    print(f"Unequalized FDN IR computed: {rir_len} samples")
    return ir_unequalized, nfft


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Output equalization

    The initial level of the unequalized FDN is roughly flat; an output GEQ
    shapes it to the spectral envelope of the target RIR.  The GEQ target is
    the band-wise dB difference between target and FDN initial levels, with
    extra attenuation at the extrapolated DC and Nyquist bands.

    The equalizer is placed at the end of the FLAMO graph (`output_filter`),
    so the final model renders the complete RIR in one pass:
    input → B → [delays → SOS → A] → C → GEQ → output.
    """)
    return


@app.cell
def _(
    delays,
    direct_gain,
    est_level,
    feedback_matrix,
    fs,
    input_gain,
    ir_unequalized,
    nfft,
    np,
    output_gain,
    pyFDN,
    rir_len,
    sos_absorption,
):
    fdn0_rt, _ = pyFDN.estimate_rt_bands(ir_unequalized, fs)
    fdn0_level, _ = pyFDN.estimate_initial_level_bands(ir_unequalized, fdn0_rt, fs)

    _diff_db = pyFDN.lin_to_db(est_level) - pyFDN.lin_to_db(fdn0_level)
    target_level_db = np.concatenate(([_diff_db[0]], _diff_db, [_diff_db[-1]]))
    target_level_db = target_level_db - np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 30])

    equalization_sos, _ = pyFDN.design_geq(target_level_db, fs=fs)
    equalization_sos = equalization_sos / equalization_sos[:, 3:4]  # a0 = 1

    model_eq = pyFDN.dss_to_flamo(
        feedback_matrix,
        input_gain,
        output_gain,
        direct_gain,
        delays,
        fs,
        nfft=nfft,
        sos_filter=sos_absorption,
        output_filter=equalization_sos[:, :, np.newaxis],
        shell=True,
    )
    ir_fdn = pyFDN.flamo_time_response(model_eq).squeeze()[:rir_len]

    print(f"GEQ target (dB): {target_level_db.round(1)}")
    return equalization_sos, ir_fdn, model_eq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize DSP graph
    """)
    return


@app.cell
def _(
    delays,
    direct_gain,
    equalization_sos,
    feedback_matrix,
    fs,
    input_gain,
    model_eq,
    output_gain,
    pyFDN,
    sos_absorption,
):
    pyFDN.plot_flamo_graph(model_eq)
    pyFDN.plot_fdn_parameter(
        delays,
        feedback_matrix,
        input_gain,
        output_gain,
        direct_gain,
        attenuation_sos=sos_absorption,
        post_eq_sos=equalization_sos,
        fs=fs,
    )
    return


@app.cell
def _(fs, ir_fdn, mo, pyFDN):
    mo.audio(pyFDN.peak_normalize(ir_fdn), fs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrograms

    Target RIR (top) and FDN impulse response (bottom).
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    pyFDN.plot_spectrogram(rir, fs, title="Target RIR")
    return


@app.cell
def _(fs, ir_fdn, pyFDN):
    pyFDN.plot_spectrogram(ir_fdn, fs, title="FDN impulse response")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reverberation time and initial level match

    Estimate the decay parameters of the FDN impulse response with the same
    estimator and compare with the target RIR.
    """)
    return


@app.cell
def _(fs, ir_fdn, pyFDN):
    fdn_rt, _ = pyFDN.estimate_rt_bands(ir_fdn, fs)
    fdn_level, _ = pyFDN.estimate_initial_level_bands(ir_fdn, fdn_rt, fs)
    return fdn_level, fdn_rt


@app.cell
def _(est_rt, f_centre, fdn_rt, go):
    fig_rt = go.Figure()
    fig_rt.add_trace(
        go.Scatter(x=f_centre, y=est_rt, mode="lines+markers", name="Target RIR")
    )
    fig_rt.add_trace(go.Scatter(x=f_centre, y=fdn_rt, mode="lines+markers", name="FDN"))
    fig_rt.update_layout(
        title="Reverberation time",
        xaxis={"title": "Frequency (Hz)", "type": "log"},
        yaxis={"title": "Reverberation time (s)", "rangemode": "tozero"},
        template="plotly_white",
        height=380,
    )
    fig_rt.show()
    return


@app.cell
def _(est_level, f_centre, fdn_level, go, pyFDN):
    fig_level = go.Figure()
    fig_level.add_trace(
        go.Scatter(
            x=f_centre,
            y=pyFDN.lin_to_db(est_level),
            mode="lines+markers",
            name="Target RIR",
        )
    )
    fig_level.add_trace(
        go.Scatter(
            x=f_centre,
            y=pyFDN.lin_to_db(fdn_level),
            mode="lines+markers",
            name="FDN",
        )
    )
    fig_level.update_layout(
        title="Initial level",
        xaxis={"title": "Frequency (Hz)", "type": "log"},
        yaxis={"title": "Initial level (dB)"},
        template="plotly_white",
        height=380,
    )
    fig_level.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: reverberation time accuracy

    The FDN reverberation time should be within 20% of the target in every
    octave band (the two highest bands were deliberately shortened by design).
    """)
    return


@app.cell
def _(est_rt, fdn_rt, np):
    rt_error = np.abs(est_rt / fdn_rt - 1)
    print(f"RT error per band: {rt_error.round(3)}")
    assert np.all(rt_error < 0.2), "FDN reverberation time deviates more than 20%"
    return


if __name__ == "__main__":
    app.run()
