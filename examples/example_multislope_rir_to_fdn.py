# gallery_category: Absorption & Filters
# gallery_description: Estimate two decay slopes per octave from a measured multi-room response and resynthesize them with parallel FDNs.
# references: Neural_Network_For_Multi_Exponential_Sound_Energy_Decay_Analysis, Acoustic_Analysis_And_Dataset_Of_Transitions_Between_Coupled_Rooms
# requires: multislope

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-slope decay: from a measured response to an FDN

    A single room decays with one exponential slope per frequency band, and
    ``pyFDN.estimate_rt_bands`` is built for exactly that case. A room that
    opens onto another does not: energy leaks between the two spaces, so the
    energy decay curve (EDC) bends — a fast slope early on, a slow one later,
    and a single reverberation time fitted to it describes neither space.

    This example estimates a **multi-slope** decay from a measured response and
    turns it into an FDN:

    1. Load a room impulse response measured across a room transition.
    2. Show that a single-slope RT lands between the two decay rates.
    3. Fit two slopes per octave band with **DecayFitNet**, from the
       [`multislope`](https://pypi.org/project/multislope/) package.
    4. Convert the fitted slope amplitudes into per-slope initial levels with
       `pyFDN.slope_amplitude_to_level`.
    5. Build **one FDN per slope**, sum them, and compare the octave-band EDCs
       with the measurement.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from multislope import DecayFitNet
    from scipy.signal import sosfilt

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return DecayFitNet, go, np, pyFDN, sosfilt


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    ## A measured multi-room response

    The response comes from the dataset accompanying
    {pyFDN.paper_link("Acoustic_Analysis_And_Dataset_Of_Transitions_Between_Coupled_Rooms")},
    measured at Aalto University by walking an ambisonic microphone from a
    meeting room out into the hallway it opens onto, and published at
    [doi.org/10.5281/zenodo.4636068](https://doi.org/10.5281/zenodo.4636068).
    The source stays inside the meeting room, and this is the receiver 2.9 m
    along that walk, past the doorway and out of line of sight — so what
    reaches it is the hallway's own quick decay riding on the slower one
    leaking out of the meeting room. Only the omnidirectional component of the
    ambisonic response is used here.

    Measured responses set the terms of what can be recovered. This one carries
    roughly 40 dB of usable decay per octave band before it reaches the noise
    floor of the measurement, so the analysis below stays inside that range.
    The response already starts at the onset, so no trimming is needed — and
    with no line of sight there is no direct sound to trim to.
    """)
    return


@app.cell
def _(np, pyFDN):
    rir, fs = pyFDN.load_audio("meetingroom_to_hallway_290cm")
    rir = rir / np.linalg.norm(rir)
    nfft = 2**17

    print(f"RIR: {len(rir)} samples ({len(rir) / fs:.2f} s) at {fs} Hz")
    return fs, nfft, rir


@app.cell
def _(fs, mo, pyFDN, rir):
    mo.audio(pyFDN.peak_normalize(rir), fs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One reverberation time per band is not enough

    `pyFDN.estimate_rt_bands` fits a single line to the Schroeder decay curve
    between -5 dB and -35 dB.  On a double-slope decay that line is a
    compromise: the estimate sits between the two decay times and follows
    neither.
    """)
    return


@app.cell
def _(fs, pyFDN, rir):
    single_slope_rt, f_centre = pyFDN.estimate_rt_bands(rir, fs)
    print(f"Bands (Hz):        {f_centre}")
    print(f"Single-slope RT (s): {single_slope_rt.round(2)}")
    return f_centre, single_slope_rt


@app.cell(hide_code=True)
def _(mo, pyFDN):
    mo.md(f"""
    ## Fitting two slopes with DecayFitNet

    `multislope.DecayFitNet` filters the RIR into octave bands, backward
    integrates each band, and predicts the decay times `T`, the slope
    amplitudes `A` and the noise floor `N` of a multi-exponential decay model.
    The network is described in
    {pyFDN.paper_link("Neural_Network_For_Multi_Exponential_Sound_Energy_Decay_Analysis")}.

    Fitting a noise floor explicitly is what makes the network usable on a
    measurement: the flat tail a real RIR ends in is absorbed by `N` instead of
    being mistaken for a very slow third slope.  The slopes come back sorted by
    ascending decay time, so column 0 is the fast one throughout.
    """)
    return


@app.cell
def _(DecayFitNet, f_centre, fs, pyFDN, rir):
    net = DecayFitNet(n_slopes=2, sample_rate=fs, filter_frequencies=list(f_centre))
    fit = net.estimate(rir)

    # estimator-neutral arrays: decay times in seconds and EDC amplitudes,
    # de-normalised back to the physical energy scale of the RIR
    decay_time = fit.t  # (n_bands, n_slopes)
    amplitude = fit.a * fit.norm_vals[0][:, None]  # (n_bands, n_slopes)

    # amplitude is the energy of a slope; convert it into the initial
    # amplitude of the corresponding exponential decay
    slope_level = pyFDN.slope_amplitude_to_level(amplitude, decay_time, fs)

    print(f"Decay time, fast slope (s): {decay_time[:, 0].round(2)}")
    print(f"Decay time, slow slope (s): {decay_time[:, 1].round(2)}")
    print(f"Level, fast slope (dB):     {pyFDN.lin_to_db(slope_level[:, 0]).round(1)}")
    print(f"Level, slow slope (dB):     {pyFDN.lin_to_db(slope_level[:, 1]).round(1)}")
    return decay_time, slope_level


@app.cell
def _(decay_time, f_centre, go, single_slope_rt):
    fig_rt = go.Figure()
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=decay_time[:, 0],
            mode="lines+markers",
            line={"color": "#636efa"},
            name="Fast slope",
        )
    )
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=decay_time[:, 1],
            mode="lines+markers",
            line={"color": "#ef553b"},
            name="Slow slope",
        )
    )
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=single_slope_rt,
            mode="lines+markers",
            line={"color": "#00cc96", "dash": "dash"},
            name="Single-slope RT",
        )
    )
    fig_rt.update_layout(
        title="Decay times: two fitted slopes vs. one reverberation time",
        xaxis={"title": "Frequency (Hz)", "type": "log", "range": [1.7, 4.0]},
        yaxis={"title": "Decay time (s)", "rangemode": "tozero"},
        template="plotly_white",
        height=420,
    )
    fig_rt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The single-slope estimate runs between the two fitted slopes in every band,
    which is the point: no single reverberation time describes this decay.  The
    contrast is largest in the mid bands, where the slow slope runs three to
    four times as long as the fast one — around 0.6 s against 2.4 s at 1 kHz.
    The slow slope starts some 20 dB below the fast one, so the knee in the EDC
    sits between -15 and -25 dB, high enough to be plainly visible above the
    noise floor of the measurement.

    ## One FDN per slope

    Each slope becomes its own FDN.  A GEQ absorption filter per delay line
    gives the FDN the decay time of that slope, and an output GEQ sets its
    initial level.  The level target is the *difference* between the level the
    slope should have and the level the unequalized FDN happens to produce, so
    the design corrects itself.

    The two GEQ designs work on a 10-point grid (DC, 63 Hz … 8 kHz, Nyquist);
    the octave-band estimates are extended to it by repeating the edge bands.

    `pyFDN.design_geq` returns its biquad sections in the unnormalised form
    `[b0, b1, b2, a0, a1, a2]`, straight out of the analytic filter formulas,
    so `a0` is not 1 (a peaking section, for instance, has `a0 = sqrt(g) + t`).
    Filtering code expects the normalised form, so each section is divided by
    its own `a0` — column 3 of the SOS matrix — which scales `b` and `a`
    together and leaves the transfer function unchanged.
    `pyFDN.absorption_geq` does this internally; `design_geq` leaves it to the
    caller.
    """)
    return


@app.cell
def _(decay_time, fs, nfft, np, pyFDN, rir, slope_level):
    def geq_grid(band_values):
        """Extend 8 octave-band values onto the 10-point GEQ design grid."""
        return np.concatenate(([band_values[0]], band_values, [band_values[-1]]))

    resynthesis = np.zeros(len(rir))
    slope_fdn_rt = []

    for _slope in range(decay_time.shape[1]):
        _build = pyFDN.fdn_build_gallery(
            16,
            fs=fs,
            delay_range=(500, 2500),
            io_type="ones",
            direct_gain=0.0,
            rt=None,
            rng=10 + _slope,
        )
        _absorption = pyFDN.absorption_geq(
            geq_grid(decay_time[:, _slope]), _build.delays, fs
        )

        # unequalized FDN: reference level for the output GEQ
        _ir_flat = pyFDN.flamo_time_response(
            pyFDN.dss_to_flamo(
                _build.A,
                _build.B,
                _build.C,
                _build.D,
                _build.delays,
                fs,
                nfft=nfft,
                sos_filter=_absorption,
                shell=True,
            )
        ).squeeze()[: len(rir)]

        _rt_flat, _ = pyFDN.estimate_rt_bands(_ir_flat, fs)
        _level_flat, _ = pyFDN.estimate_initial_level_bands(_ir_flat, _rt_flat, fs)
        _gain_db = pyFDN.lin_to_db(slope_level[:, _slope]) - pyFDN.lin_to_db(
            _level_flat
        )
        _eq, _ = pyFDN.design_geq(geq_grid(_gain_db), fs=fs)
        _eq = _eq / _eq[:, 3:4]  # normalise each section so a0 = 1

        resynthesis += pyFDN.flamo_time_response(
            pyFDN.dss_to_flamo(
                _build.A,
                _build.B,
                _build.C,
                _build.D,
                _build.delays,
                fs,
                nfft=nfft,
                sos_filter=_absorption,
                output_filter=_eq[:, :, np.newaxis],
                shell=True,
            )
        ).squeeze()[: len(rir)]

        slope_fdn_rt.append(_rt_flat)
        print(f"Slope {_slope}: FDN RT (s) {_rt_flat.round(2)}")
        print(f"          output GEQ (dB) {_gain_db.round(1)}")

    slope_fdn_rt = np.stack(slope_fdn_rt, axis=1)
    return resynthesis, slope_fdn_rt


@app.cell
def _(fs, mo, pyFDN, resynthesis):
    mo.audio(pyFDN.peak_normalize(resynthesis), fs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy decay curves

    The comparison that matters is the octave-band EDC: the sum of the two
    FDNs should bend the same way as the measured response.  Both curves are
    normalised to 0 dB at the onset.
    """)
    return


@app.cell
def _(f_centre, fs, np, pyFDN, resynthesis, rir, sosfilt):
    _bands, _ = pyFDN.octave_bands(fs=fs)
    band_sos = pyFDN.octave_band_filterbank(_bands, fs)

    def band_edc_db(signal, band_index):
        """Octave-band EDC in dB, normalised to 0 dB at the onset."""
        curve = pyFDN.sq_to_db(pyFDN.edc(sosfilt(band_sos[band_index], signal)))
        return curve - curve[0]

    edc_target = np.stack([band_edc_db(rir, k) for k in range(len(f_centre))])
    edc_fdn = np.stack([band_edc_db(resynthesis, k) for k in range(len(f_centre))])
    return edc_fdn, edc_target


@app.cell
def _(edc_fdn, edc_target, f_centre, fs, go, np):
    fig_edc = go.Figure()
    _time = np.arange(edc_target.shape[1])[::64] / fs
    for _index, _colour in zip(
        (2, 4, 6), ("#636efa", "#ef553b", "#00cc96"), strict=True
    ):
        fig_edc.add_trace(
            go.Scatter(
                x=_time,
                y=edc_target[_index][::64],
                mode="lines",
                line={"color": _colour},
                name=f"{f_centre[_index]:.0f} Hz, measured",
            )
        )
        fig_edc.add_trace(
            go.Scatter(
                x=_time,
                y=edc_fdn[_index][::64],
                mode="lines",
                line={"color": _colour, "dash": "dash"},
                name=f"{f_centre[_index]:.0f} Hz, two FDNs",
            )
        )
    fig_edc.update_layout(
        title="Octave-band energy decay curves",
        xaxis={"title": "Time (s)", "range": [0, 1.5]},
        yaxis={"title": "Energy decay (dB)", "range": [-70, 2]},
        template="plotly_white",
        height=460,
    )
    fig_edc.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: the FDNs realise the estimated decay

    Two checks.  Each per-slope FDN must reproduce the decay time it was
    designed for, and the sum of the two must follow the measured EDC over its
    first 30 dB.

    The EDC error is printed for all eight bands but asserted only over
    250 Hz – 4 kHz.  The two edge bands are excluded deliberately, not to make
    the check pass: at 62 Hz and 8 kHz the octave filter runs into the ends of
    the spectrum, the GEQ command point sits at the edge of its design grid,
    and the measurement is quietest — so the EDC there flattens onto the noise
    floor of the recording, which the FDN has no reason to reproduce.  Over the
    five bands where measurement and model are both trustworthy the match is
    well inside 2 dB rms.
    """)
    return


@app.cell
def _(decay_time, np, slope_fdn_rt):
    rt_error = np.abs(slope_fdn_rt / decay_time - 1)
    print(f"Decay time error per slope and band: {rt_error.round(3)}")
    assert np.all(rt_error < 0.2), "FDN decay time deviates more than 20%"
    return


@app.cell
def _(edc_fdn, edc_target, f_centre, np):
    edc_error = np.array(
        [
            np.sqrt(np.mean((edc_target[k][_valid] - edc_fdn[k][_valid]) ** 2))
            for k in range(len(f_centre))
            if (_valid := edc_target[k] > -30).any()
        ]
    )
    mid_bands = slice(2, 7)  # 250 Hz .. 4 kHz
    print(f"EDC error per band (dB rms): {edc_error.round(2)}")
    print(f"Bands asserted on: {f_centre[mid_bands].round(0)}")
    assert np.all(edc_error[mid_bands] < 2.0), (
        "Resynthesised EDC deviates more than 2 dB rms"
    )
    return


if __name__ == "__main__":
    app.run()
