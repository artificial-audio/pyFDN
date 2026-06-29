# gallery_category: Special FDNs

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
    # Reverberation enhancement with a time-varying FDN

    A **reverberation enhancement system** (RES) makes a room sound more
    reverberant electroacoustically: microphones pick up the room, a reverberator
    processes the signal, and loudspeakers play it back — adding energy to the
    reverberant field. The catch is that the loudspeakers leak back into the
    microphones, so the reverberator sits *inside* an acoustic feedback loop. Too
    much loop gain and the system colours (rings) or howls; the usable gain before
    that happens is the **maximum stable gain** (MSG).

    A **time-varying FDN** raises the MSG: by continuously modulating the feedback
    matrix it stops any single loop mode from building up, so the same enhancement
    can be driven harder before it rings.

    This example wires up a real RES:

    * `pyroomacoustics` places a performer, a listener, **6 microphones** over the
      stage and **6 loudspeakers** over the audience, and computes every room
      impulse response — including the loudspeaker→microphone coupling that closes
      the loop.
    * the reverberator is a 6-in/6-out FDN built from `pyFDN.td` operators, with
      an optional `td.TimeVaryingMatrix` on its feedback path.
    * a block time-domain simulator runs the closed electroacoustic loop.

    We then (1) confirm the RES enhances reverberation and (2) show the
    time-varying FDN stays stable at a loop gain where the static one already
    rings.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from scipy.signal import fftconvolve

    import pyFDN
    from pyFDN import td
    from pyFDN.dsp.dfilt_matrix import FIRMatrixFilter
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix

    pio.renderers.default = "sphinx_gallery"
    return FIRMatrixFilter, TimeVaryingMatrix, fftconvolve, go, np, pyFDN, td


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Room, stage and audience layout

    A 24 × 18 × 9 m hall (≈0.6 s natural reverberation). The performer is at the
    front of the stage and the listener sits in the audience. The 6 microphones
    hang low over the stage apron; the 6 loudspeakers are high over the audience —
    a separation that keeps the loudspeaker→microphone coupling modest, as a real
    install would.
    """)
    return


@app.cell
def _(go, np):
    import pyroomacoustics as pra

    fs = 48_000
    room_dim = [24.0, 18.0, 9.0]

    performer = [12.0, 3.0, 1.7]  # front of stage
    listener = [12.0, 13.0, 1.2]  # in the audience
    mics = np.array(  # over the stage apron, low
        [
            [7, 4, 2.2],
            [12, 4, 2.2],
            [17, 4, 2.2],
            [9, 6, 2.2],
            [12, 6, 2.2],
            [15, 6, 2.2],
        ],
        dtype=float,
    )
    speakers = np.array(  # over the audience, high
        [
            [4, 10, 7.5],
            [12, 10, 7.5],
            [20, 10, 7.5],
            [4, 16, 7.5],
            [12, 16, 7.5],
            [20, 16, 7.5],
        ],
        dtype=float,
    )

    e_absorption, max_order = pra.inverse_sabine(0.6, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=min(max_order, 8),
    )
    for position in [performer, *speakers]:
        room.add_source(position)
    room.add_microphone_array(pra.MicrophoneArray(np.vstack([mics, listener]).T, fs))
    room.compute_rir()

    # Top-view layout.
    fig_room = go.Figure()
    fig_room.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=room_dim[0],
        y1=room_dim[1],
        line={"color": "#444"},
        fillcolor="rgba(0,0,0,0)",
    )
    fig_room.add_trace(
        go.Scatter(
            x=mics[:, 0],
            y=mics[:, 1],
            mode="markers",
            name="microphones",
            marker={"size": 11, "symbol": "circle", "color": "#4f8a5e"},
        )
    )
    fig_room.add_trace(
        go.Scatter(
            x=speakers[:, 0],
            y=speakers[:, 1],
            mode="markers",
            name="loudspeakers",
            marker={"size": 13, "symbol": "square", "color": "#3d6d9e"},
        )
    )
    fig_room.add_trace(
        go.Scatter(
            x=[performer[0]],
            y=[performer[1]],
            mode="markers",
            name="performer",
            marker={"size": 15, "symbol": "star", "color": "#c0392b"},
        )
    )
    fig_room.add_trace(
        go.Scatter(
            x=[listener[0]],
            y=[listener[1]],
            mode="markers",
            name="listener",
            marker={"size": 15, "symbol": "diamond", "color": "#7b5ea7"},
        )
    )
    fig_room.update_layout(
        title="RES layout (top view): stage at front, audience behind",
        xaxis={"title": "x (m)", "range": [-1, 25]},
        yaxis={"title": "y (m)", "range": [-1, 19], "scaleanchor": "x"},
        template="plotly_white",
        height=460,
    )
    fig_room.show()
    return fs, room


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract the room transfer paths

    From the impulse responses we pull out the paths the simulator needs:
    performer→microphones and performer→listener (the dry excitation),
    loudspeaker→listener (what the RES delivers), and the 6 × 6
    loudspeaker→microphone **coupling** that closes the loop. The coupling is
    truncated to its first ~43 ms — the early part that dominates feedback
    colouration — so the in-loop convolution stays cheap.
    """)
    return


@app.cell
def _(fftconvolve, fs, np, room):
    rir = room.rir
    coupling_taps = 2048  # ~43 ms of loudspeaker -> mic coupling

    def _pad(h, n):
        h = np.asarray(h, dtype=float)
        return np.pad(h, (0, max(0, n - len(h))))[:n]

    # source index 0 = performer, 1..6 = loudspeakers; mic index 0..5, 6 = listener.
    coupling = np.stack(
        [[_pad(rir[m][l + 1], coupling_taps) for l in range(6)] for m in range(6)]
    )  # (6 mics, 6 speakers, taps)
    h_speaker_listener = [rir[6][l + 1] for l in range(6)]
    h_source_listener = np.asarray(rir[6][0], dtype=float)

    sig_len = int(0.8 * fs)
    impulse = np.zeros(sig_len)
    impulse[0] = 1.0
    src_to_mic = np.stack(
        [fftconvolve(impulse, rir[m][0])[:sig_len] for m in range(6)], axis=1
    )  # (T, 6)
    dry_listener = fftconvolve(impulse, h_source_listener)[:sig_len]

    peak_coupling = np.abs(coupling).max()
    print(f"Peak loudspeaker->mic coupling: {peak_coupling:.3f}")
    print(f"Coupling length: {coupling_taps} taps ({1000 * coupling_taps / fs:.0f} ms)")
    return coupling, dry_listener, h_speaker_listener, impulse, sig_len, src_to_mic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The reverberator: a 6×6 FDN

    The reverberator maps the 6 microphones to the 6 loudspeakers through an
    8-line FDN with frequency-dependent absorption (its own ~1.2 s decay). The
    feedback path is either the static mixing matrix `A`, or `Series([Gain(A),
    TimeVaryingMatrix])` — the only change needed to make the loop time-varying.
    """)
    return


@app.cell
def _(TimeVaryingMatrix, fs, np, pyFDN, td):
    n_lines = 8
    fdn_delays = np.array([557, 619, 691, 757, 821, 887, 953, 1021])
    np.random.seed(1)
    A = pyFDN.random_orthogonal(n_lines)
    gen = np.random.default_rng(1)
    B_fdn = gen.standard_normal((n_lines, 6)) / np.sqrt(n_lines)  # mics -> lines
    C_fdn = gen.standard_normal((6, n_lines)) / np.sqrt(
        n_lines
    )  # lines -> loudspeakers
    fdn_absorption = pyFDN.first_order_absorption(1.2, 0.6, fdn_delays, fs, None)

    def make_reverberator(time_varying):
        """Build the 6-in/6-out FDN operator tree; optionally time-varying."""
        forward = td.Series([td.Delay(fdn_delays), td.SOSBank(fdn_absorption)])
        if time_varying:
            np.random.seed(3)  # deterministic modulation across rebuilds
            tvm = TimeVaryingMatrix(
                N=n_lines, cycles_per_second=1.2, amplitude=0.7, fs=fs, spread=0.2
            )
            feedback = td.Series([td.Gain(A), td.TimeVaryingMatrix(tvm)])
        else:
            feedback = td.Gain(A)
        return td.Series(
            [td.Gain(B_fdn), td.Recursion(forward, feedback), td.Gain(C_fdn)]
        )

    return (make_reverberator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closed-loop RES simulator

    Block by block: the microphones hear the dry source plus the room coupling of
    the previous block's loudspeaker output (a one-block ≈ 5 ms processing
    latency, as real RES hardware has); the reverberator turns the microphone
    signal into the next loudspeaker output, scaled by the loop gain `g`. The
    listener finally hears the dry source plus every loudspeaker through its
    room response.
    """)
    return


@app.cell
def _(
    FIRMatrixFilter,
    coupling,
    dry_listener,
    fftconvolve,
    h_speaker_listener,
    np,
    sig_len,
    src_to_mic,
):
    block = 256  # processing latency / loop block (~5.3 ms)

    def run_res(reverberator, g):
        """Run the closed electroacoustic loop; return the listener signal."""
        length = sig_len - (sig_len % block)
        coupling_filter = FIRMatrixFilter(coupling)  # 6x6 FIR, stateful
        loudspeakers = np.zeros((length, 6))
        previous = np.zeros((block, 6))
        start = 0
        while start < length:
            acoustic_feedback = coupling_filter.filter(previous)  # (block, 6 mics)
            mic = src_to_mic[start : start + block] + acoustic_feedback
            speaker_out = g * reverberator.process(mic)  # (block, 6)
            loudspeakers[start : start + block] = speaker_out
            previous = speaker_out
            start += block

        listener = dry_listener[:length].copy()
        for ls in range(6):
            listener += fftconvolve(loudspeakers[:, ls], h_speaker_listener[ls])[
                :length
            ]
        return listener

    return block, run_res


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reverberation enhancement

    At a comfortable loop gain the RES adds a long reverberant tail to the dry
    response — the energy decay curve at the listener decays far more slowly with
    the system on.
    """)
    return


@app.cell
def _(dry_listener, fs, go, make_reverberator, np, run_res):
    def edc_db(x):
        energy = np.cumsum(x[::-1] ** 2)[::-1]
        return 10 * np.log10(energy / energy[0] + 1e-20)

    g_operating = 0.8
    enhanced = run_res(make_reverberator(time_varying=True), g_operating)
    dry = dry_listener[: len(enhanced)]

    late = slice(int(0.4 * fs), None)
    enhancement_ratio = np.sqrt((enhanced[late] ** 2).mean()) / np.sqrt(
        (dry[late] ** 2).mean()
    )
    print(f"Late-tail RMS gain (enhanced / dry): {enhancement_ratio:.1f}x")
    assert np.isfinite(enhanced).all()
    assert enhancement_ratio > 5.0

    t = np.arange(len(dry)) / fs
    fig_edc = go.Figure()
    fig_edc.add_trace(
        go.Scatter(x=t, y=edc_db(dry), name="dry room", line={"color": "#888"})
    )
    fig_edc.add_trace(
        go.Scatter(
            x=t,
            y=edc_db(enhanced),
            name=f"RES on (g={g_operating})",
            line={"color": "#7b5ea7"},
        )
    )
    fig_edc.update_layout(
        title="Energy decay at the listener: RES extends the reverberation",
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Energy decay (dB)", "range": [-60, 2]},
        template="plotly_white",
        height=380,
    )
    fig_edc.show()
    return (edc_db,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Maximum stable gain: static vs time-varying

    Now push the loop gain up. We track the short-time energy envelope of the
    listener response: a stable (well-enhanced) system **decays**, an
    over-driven one **grows** as a loop mode regenerates. At the same high gain
    the static FDN is already growing (ringing), while the time-varying FDN still
    decays — its modulation breaks up the runaway mode.
    """)
    return


@app.cell
def _(fs, go, make_reverberator, np, run_res):
    def envelope_db(x, win=2048, hop=512):
        starts = np.arange(0, len(x) - win, hop)
        env = np.array(
            [10 * np.log10((x[i : i + win] ** 2).mean() + 1e-20) for i in starts]
        )
        return starts / fs, env

    g_challenge = 1.6
    rec_static = run_res(make_reverberator(time_varying=False), g_challenge)
    rec_varying = run_res(make_reverberator(time_varying=True), g_challenge)

    def growth(x):  # last-quarter energy / first-quarter energy; >1 means growing
        q = len(x) // 4
        return (x[-q:] ** 2).mean() / ((x[:q] ** 2).mean() + 1e-30)

    growth_static, growth_varying = growth(rec_static), growth(rec_varying)
    print(
        f"g={g_challenge}: growth static={growth_static:.2f}  varying={growth_varying:.2f}"
    )
    assert np.isfinite(rec_static).all() and np.isfinite(rec_varying).all()
    assert growth_static > 1.0  # static loop is regenerating (unstable/colouring)
    assert growth_varying < 1.0  # time-varying loop still decays
    assert growth_varying < growth_static

    fig_msg = go.Figure()
    ts, es = envelope_db(rec_static)
    tv, ev = envelope_db(rec_varying)
    fig_msg.add_trace(
        go.Scatter(x=ts, y=es, name="static FDN (rings)", line={"color": "#c0392b"})
    )
    fig_msg.add_trace(
        go.Scatter(
            x=tv, y=ev, name="time-varying FDN (stable)", line={"color": "#7b5ea7"}
        )
    )
    fig_msg.update_layout(
        title=f"Short-time energy at g={g_challenge}: static grows, time-varying decays",
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Energy (dB)"},
        template="plotly_white",
        height=380,
    )
    fig_msg.show()
    return (g_challenge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The stability margin

    Sweeping the loop gain makes the gain in maximum stable gain explicit: the
    growth ratio crosses 1 (the stability boundary) at a higher gain for the
    time-varying FDN. The horizontal distance between the two crossings is the
    extra gain — a few dB — that time variation buys.
    """)
    return


@app.cell
def _(go, make_reverberator, np, run_res):
    gains = np.array([1.0, 1.4, 1.8, 2.2, 2.6])

    def growth_ratio(x):
        q = len(x) // 4
        return (x[-q:] ** 2).mean() / ((x[:q] ** 2).mean() + 1e-30)

    ratios_static, ratios_varying = [], []
    for g in gains:
        ratios_static.append(growth_ratio(run_res(make_reverberator(False), g)))
        ratios_varying.append(growth_ratio(run_res(make_reverberator(True), g)))

    fig_sweep = go.Figure()
    fig_sweep.add_trace(
        go.Scatter(
            x=gains,
            y=ratios_static,
            name="static FDN",
            mode="lines+markers",
            line={"color": "#c0392b"},
        )
    )
    fig_sweep.add_trace(
        go.Scatter(
            x=gains,
            y=ratios_varying,
            name="time-varying FDN",
            mode="lines+markers",
            line={"color": "#7b5ea7"},
        )
    )
    fig_sweep.add_hline(
        y=1.0,
        line={"dash": "dash", "color": "#444"},
        annotation_text="stability boundary",
    )
    fig_sweep.update_layout(
        title="Tail growth vs loop gain: time variation raises the maximum stable gain",
        xaxis={"title": "Loop gain g"},
        yaxis={"title": "Tail growth ratio", "type": "log"},
        template="plotly_white",
        height=400,
    )
    fig_sweep.show()
    return


if __name__ == "__main__":
    app.run()
