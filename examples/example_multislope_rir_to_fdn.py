# gallery_category: Absorption & Filters
# gallery_description: Estimate two decay slopes per octave from a coupled-room response and resynthesize them with parallel FDNs.
# references: Neural_Network_For_Multi_Exponential_Sound_Energy_Decay_Analysis
# requires: multislope

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
    # Multi-slope decay: from a coupled-space RIR to an FDN

    A single room decays with one exponential slope per frequency band, and ``pyFDN.estimate_rt_bands`` is built for exactly that case. Two coupled rooms do not: energy leaks from the small room into the large one, so the energy decay curve (EDC) bends — a fast slope early on, a slow one later, and a single reverberation time fitted to it describes neither room.

    This example estimates a **multi-slope** decay and turns it into an FDN:

    1. Render a coupled-rooms RIR (two FDNs joined by a mixing matrix).
    2. Show that a single-slope RT lands between the two true decay times.
    3. Fit two slopes per octave band with **DecayFitNet**, from the [`multislope`](https://pypi.org/project/multislope/) package.
    4. Convert the fitted slope amplitudes into per-slope initial levels with `pyFDN.slope_amplitude_to_level`.
    5. Build **one FDN per slope**, sum them, and compare the octave-band EDCs with the original.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from multislope import DecayFitNet
    from scipy.linalg import block_diag
    from scipy.signal import sosfilt

    import pyFDN

    pio.renderers.default = "sphinx_gallery"
    return DecayFitNet, block_diag, go, np, pyFDN, sosfilt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A coupled-space impulse response

    Two single-room FDNs — a small bright room (RT 0.7 s at DC) and a large reverberant one (RT 3.2 s) — are concatenated into one FDN with a block-diagonal feedback matrix and coupled by anorthogonal block rotation, as in the *Coupled Rooms* example.  The source sits in the small room, and so does the receiver: it picks up the small room directly and the large room only weakly, which is the geometry that produces a pronounced double-slope decay.
    """)
    return


@app.cell
def _(block_diag, np, pyFDN):
    fs = 48000
    nfft = 2**18  # 5.5 s of impulse response

    num_delays_room = 12
    room_small = pyFDN.fdn_build_gallery(
        num_delays_room,
        fs=fs,
        delay_range=(400, 900),
        rt=0.7,
        rt_nyquist=0.5,
        rt_crossover=1000.0,
        io_type="ones",
        rng=5,
    )
    room_large = pyFDN.fdn_build_gallery(
        num_delays_room,
        fs=fs,
        delay_range=(1100, 2600),
        rt=3.2,
        rt_nyquist=1.6,
        rt_crossover=1000.0,
        io_type="ones",
        rng=6,
    )

    coupling = 0.15  # coupling angle between the rooms (0 = uncoupled)
    _eye = np.eye(num_delays_room)
    _mixing = np.block(
        [
            [np.cos(coupling) * _eye, np.sin(coupling) * _eye],
            [-np.sin(coupling) * _eye, np.cos(coupling) * _eye],
        ]
    )

    room_A = _mixing @ block_diag(room_small.A, room_large.A)
    room_delays = np.concatenate([room_small.delays, room_large.delays])
    room_absorption = np.concatenate(
        [room_small.post_delay, room_large.post_delay], axis=2
    )

    # source in the small room, receiver in the small room with weak coupling
    room_B = np.zeros((2 * num_delays_room, 1))
    room_B[:num_delays_room] = room_small.B
    room_C = np.zeros((1, 2 * num_delays_room))
    room_C[0, :num_delays_room] = room_small.C[0]
    room_C[0, num_delays_room:] = 0.2 * room_large.C[0]

    rir = pyFDN.flamo_time_response(
        pyFDN.dss_to_flamo(
            room_A,
            room_B,
            room_C,
            np.zeros((1, 1)),
            room_delays,
            fs,
            nfft=nfft,
            post_delay=room_absorption,
            shell=True,
        )
    ).squeeze()

    print(f"RIR: {len(rir)} samples ({len(rir) / fs:.2f} s) at {fs} Hz")
    return fs, nfft, rir, room_large, room_small


@app.cell
def _(fs, mo, pyFDN, rir):
    mo.audio(pyFDN.peak_normalize(rir), fs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The decay times of the two rooms *in isolation* are known here, because the rooms were designed: the per-sample gain of each room's absorption filters gives its reverberation time as a function of frequency. They are the reference below — but note that the coupled system does not decay at exactly these rates, as the fit will show.
    """)
    return


@app.cell
def _(fs, np, pyFDN, room_large, room_small):
    def room_reference_rt(build):
        """Reverberation time per frequency from a room's absorption filters."""
        angles, magnitude = pyFDN.sos_gain_per_sample_curves(
            build.post_delay, build.delays, nfft=256
        )
        frequency = angles / np.pi * fs / 2
        return frequency, pyFDN.slope_to_rt(pyFDN.lin_to_db(magnitude.mean(axis=1)), fs)

    reference_frequency, reference_rt_small = room_reference_rt(room_small)
    _, reference_rt_large = room_reference_rt(room_large)
    return reference_frequency, reference_rt_large, reference_rt_small


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One reverberation time per band is not enough

    `pyFDN.estimate_rt_bands` fits a single line to the Schroeder decay curve between -5 dB and -35 dB. On a double-slope decay that line is a compromise: the estimate sits between the two true decay times and follows neither.
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

    `multislope.DecayFitNet` filters the RIR into octave bands, backward integrates each band, and predicts the decay times `T`, the slope amplitudes `A` and the noise floor `N` of a multi-exponential decay model.
    The network is described in {pyFDN.paper_link("Neural_Network_For_Multi_Exponential_Sound_Energy_Decay_Analysis")}.

    The network resamples every EDC to a fixed length, so the analysis window sets the time resolution of the fit: a 0.6 s slope inside a 5.5 s window occupies only a handful of samples and gets smeared into the late decay.
    Trimming the RIR to roughly the range that carries useful decay — here 2 s, below which the FDN response has fallen past -60 dB — keeps both slopes resolvable.
    """)
    return


@app.cell
def _(DecayFitNet, f_centre, fs, pyFDN, rir):
    analysis_length = int(2.0 * fs)

    net = DecayFitNet(n_slopes=2, sample_rate=fs, filter_frequencies=list(f_centre))
    fit = net.estimate(rir[:analysis_length])

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
def _(
    decay_time,
    f_centre,
    go,
    reference_frequency,
    reference_rt_large,
    reference_rt_small,
    single_slope_rt,
):
    fig_rt = go.Figure()
    fig_rt.add_trace(
        go.Scatter(
            x=reference_frequency,
            y=reference_rt_small,
            mode="lines",
            line={"color": "#636efa", "dash": "dot"},
            name="Small room (design)",
        )
    )
    fig_rt.add_trace(
        go.Scatter(
            x=reference_frequency,
            y=reference_rt_large,
            mode="lines",
            line={"color": "#ef553b", "dash": "dot"},
            name="Large room (design)",
        )
    )
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=decay_time[:, 0],
            mode="lines+markers",
            line={"color": "#636efa"},
            name="Fast slope (estimated)",
        )
    )
    fig_rt.add_trace(
        go.Scatter(
            x=f_centre,
            y=decay_time[:, 1],
            mode="lines+markers",
            line={"color": "#ef553b"},
            name="Slow slope (estimated)",
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
        title="Decay times: two fitted slopes vs. the two rooms",
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
    The single-slope estimate runs between the two rooms, while the fitted slopes track them individually.
    The slow slope lands below the large room's dotted curve, and that is the coupling rather than an estimation error: the dotted curves are the decay times the two rooms would have in isolation, but the coupling rotation mixes their feedback loops, so the decay rates of the *coupled* system are pulled towards each other. The slow decay time of the coupled space therefore sits somewhere between the two isolated ones, and moves further from the large room the larger the coupling angle.

    ## One FDN per slope

    Each slope becomes its own FDN. A GEQ absorption filter per delay line gives the FDN the decay time of that slope, and an output GEQ sets its initial level. The level target is the difference* between the level the slope should have and the level the unequalized FDN happens to produce, so the design corrects itself.

    The two GEQ designs work on a 10-point grid (DC, 63 Hz … 8 kHz, Nyquist);
    the octave-band estimates are extended to it by repeating the edge bands.

    `pyFDN.design_geq` returns its biquad sections in the unnormalised form `[b0, b1, b2, a0, a1, a2]`, straight out of the analytic filter formulas, so `a0` is not 1 (a peaking section, for instance, has `a0 = sqrt(g) + t`).

    Filtering code expects the normalised form, so each section is divided by its own `a0` — column 3 of the SOS matrix — which scales `b` and `a` together and leaves the transfer function unchanged. `pyFDN.absorption_geq` does this internally; `design_geq` leaves it to the caller.
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
                post_delay=_absorption,
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
                post_delay=_absorption,
                post_output=_eq[:, :, np.newaxis],
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

    The comparison that matters is the octave-band EDC: the sum of the two FDNs should bend the same way as the coupled-rooms response. Both curves are normalised to 0 dB at the onset.
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
        (1, 4, 6), ("#636efa", "#ef553b", "#00cc96"), strict=True
    ):
        fig_edc.add_trace(
            go.Scatter(
                x=_time,
                y=edc_target[_index][::64],
                mode="lines",
                line={"color": _colour},
                name=f"{f_centre[_index]:.0f} Hz, coupled rooms",
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
        xaxis={"title": "Time (s)", "range": [0, 3]},
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

    Two checks. Each per-slope FDN must reproduce the decay time it was designed for, and the sum of the two must follow the coupled-rooms EDC over its first 40 dB — the range the two-slope model describes.
    """)
    return


@app.cell
def _(decay_time, np, slope_fdn_rt):
    rt_error = np.abs(slope_fdn_rt / decay_time - 1)
    print(f"Decay time error per slope and band: {rt_error.round(3)}")
    assert np.all(rt_error < 0.15), "FDN decay time deviates more than 15%"
    return


@app.cell
def _(edc_fdn, edc_target, f_centre, np):
    edc_error = np.array(
        [
            np.sqrt(np.mean((edc_target[k][_valid] - edc_fdn[k][_valid]) ** 2))
            for k in range(len(f_centre))
            if (_valid := edc_target[k] > -40).any()
        ]
    )
    print(f"EDC error per band (dB rms): {edc_error.round(2)}")
    assert np.all(edc_error < 3.0), "Resynthesised EDC deviates more than 3 dB rms"
    return


if __name__ == "__main__":
    app.run()
