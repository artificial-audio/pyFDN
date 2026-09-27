# gallery_category: Special FDNs
# gallery_description: Reverberation enhancement system built with the pyFDN.td graph: an FDN inside an acoustic feedback loop, with a time-varying feedback matrix raising the stable gain before feedback.
# requires: pyroomacoustics

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

    A **reverberation enhancement system** (RES) makes a room sound more reverberant electroacoustically: microphones pick up the room, a reverberator processes the signal, and loudspeakers play it back — adding energy to the reverberant field. The catch is that the loudspeakers leak back into the microphones, so the reverberator sits *inside* an acoustic feedback loop. Too much loop gain and the system colours (rings) or howls; the usable gain before that happens is the **maximum stable gain** (MSG).

    A **time-varying FDN** raises the MSG: by continuously modulating the feedback matrix it stops any single loop mode from building up, so the same enhancement can be driven harder before it rings.

    This example wires up a real RES:

    * `pyroomacoustics` places a performer, a listener, **6 microphones** over the stage and **6 loudspeakers** over the audience, and computes every room impulse response — including the loudspeaker→microphone coupling that closes the loop.
    * the reverberator is a 6-in/6-out FDN built from `pyFDN.td` operators, with an optional `td.TimeVaryingMatrix` on its feedback path.
    * the **entire** system — room paths, coupling and FDN — is assembled as a single `td` operator tree and run by one `.process_signal(source)` call. The whole electroacoustic feedback loop is just a `td.Recursion` whose feedback path is the room coupling and whose forward path is the FDN.

    We then (1) confirm the RES enhances reverberation and (2) show the time-varying FDN stays stable at a loop gain where the static one already rings.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import pyFDN
    from pyFDN import td

    return np, plt, pyFDN, td


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Room, stage and audience layout

    A 24 × 18 × 9 m hall (≈0.6 s natural reverberation). The performer is at the front of the stage and the listener sits in the audience. The 6 microphone hang low over the stage apron; the 6 loudspeakers are high over the audience — a separation that keeps the loudspeaker→microphone coupling modest, as a real install would.
    """)
    return


@app.cell
def _(mo, np, plt):
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
    fig_room, _ax = plt.subplots(figsize=(7, 5.4))
    _ax.add_patch(
        plt.Rectangle((0, 0), room_dim[0], room_dim[1], fill=False, edgecolor="#444")
    )
    _ax.scatter(
        mics[:, 0], mics[:, 1], s=80, marker="o", color="#4f8a5e", label="microphones"
    )
    _ax.scatter(
        speakers[:, 0],
        speakers[:, 1],
        s=100,
        marker="s",
        color="#3d6d9e",
        label="loudspeakers",
    )
    _ax.scatter(
        [performer[0]],
        [performer[1]],
        s=200,
        marker="*",
        color="#c0392b",
        label="performer",
    )
    _ax.scatter(
        [listener[0]],
        [listener[1]],
        s=120,
        marker="D",
        color="#7b5ea7",
        label="listener",
    )
    _ax.set_xlim(-1, 25)
    _ax.set_ylim(-1, 19)
    _ax.set_aspect("equal")
    _ax.set_title("RES layout (top view): stage at front, audience behind")
    _ax.set_xlabel("x (m)")
    _ax.set_ylabel("y (m)")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig_room.tight_layout()
    mo.output.replace(fig_room)
    return fs, room


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract the room transfer paths

    From the impulse responses we pull out four filter matrices the tree needs: performer→microphones (the excitation), performer→listener (dry direct), loudspeaker→listener (what the RES delivers), and the 6 × 6 loudspeaker→microphone **coupling** that closes the loop. Each becomes a `td.MatrixConvolver`. The coupling is truncated to its first ~43 ms — the early part that dominates feedback colouration — because it runs inside the loop, once per block; the rest use the full room responses.
    """)
    return


@app.cell
def _(fs, np, room):
    rir = room.rir
    coupling_taps = 2048  # ~43 ms of loudspeaker -> mic coupling (early, in-loop)
    room_taps = max(len(rir[o][s]) for o in range(len(rir)) for s in range(len(rir[0])))

    def _pad(h, n):
        h = np.asarray(h, dtype=float)
        return np.pad(h, (0, max(0, n - len(h))))[:n]

    def _matrix(out_rows, in_cols, taps):
        """Stack chosen RIRs into a (n_out, n_in, taps) filter matrix."""
        return np.stack([[_pad(rir[o][i], taps) for i in in_cols] for o in out_rows])

    # source index 0 = performer, 1..6 = loudspeakers; mic index 0..5, 6 = listener.
    _mic, _spk, _src, _lis = range(6), range(1, 7), [0], [6]
    source_to_mic = _matrix(_mic, _src, room_taps)  # (6, 1, L): excitation
    coupling = _matrix(_mic, _spk, coupling_taps)  # (6, 6, taps): closes the loop
    speaker_to_listener = _matrix(_lis, _spk, room_taps)  # (1, 6, L): delivered
    source_to_listener = _matrix(_lis, _src, room_taps)  # (1, 1, L): dry direct

    sig_len = int(0.8 * fs)
    print(f"Peak loudspeaker->mic coupling: {np.abs(coupling).max():.3f}")
    print(f"Room RIRs: {room_taps} taps; coupling truncated to {coupling_taps} taps")
    return (
        coupling,
        sig_len,
        source_to_listener,
        source_to_mic,
        speaker_to_listener,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The reverberator: a 6×6 FDN

    The reverberator maps the 6 microphones to the 6 loudspeakers through an 8-line FDN with frequency-dependent absorption (its own ~1.2 s decay). The feedback path is either the static mixing matrix `A`, or `Series([Gain(A), td.TimeVaryingMatrix(...)])` — the only change needed to make the loop time-varying.
    """)
    return


@app.cell
def _(fs, np, pyFDN, td):
    n_lines = 8
    fdn_delays = np.array([557, 619, 691, 757, 821, 887, 953, 1021])
    np.random.seed(1)
    A = pyFDN.random_orthogonal(n_lines)
    gen = np.random.default_rng(1)
    B_fdn = gen.standard_normal((n_lines, 6)) / np.sqrt(n_lines)  # mics -> lines
    C_fdn = gen.standard_normal((6, n_lines)) / np.sqrt(
        n_lines
    )  # lines -> loudspeakers
    fdn_absorption = pyFDN.decay_to_first_order_shelf(1.2, 0.6, None, fdn_delays, fs)
    fdn_block = 64  # Recursion block size; must be at most half the shortest delay

    def make_reverberator(time_varying, g):
        """6-in/6-out FDN operator tree.

        The loop gain ``g`` is folded into the output gain ``C``, and the
        feedback path is optionally made time-varying.
        """
        # The Recursion inserts fdn_block samples of delay into the loop, so the
        # delay lines are shortened by that much to keep the loop delays exact.
        forward = td.Series(
            [td.Delay(fdn_delays - fdn_block), td.SOSBank(fdn_absorption)]
        )
        if time_varying:
            np.random.seed(3)  # deterministic modulation across rebuilds
            tvm = td.TimeVaryingMatrix(
                N=n_lines, cycles_per_second=1.2, amplitude=0.7, fs=fs, spread=0.2
            )
            feedback = td.Series([td.Gain(A), tvm])
        else:
            feedback = td.Gain(A)
        return td.Series(
            [
                td.Gain(B_fdn),
                td.Recursion(forward, feedback, block_size=fdn_block),
                td.Gain(g * C_fdn),
            ]
        )

    return (make_reverberator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The entire RES as one `td` tree

    Everything now assembles into a single operator tree:

    ```
    Parallel(
        Series(
            MatrixConvolver(source → mics),        # excitation
            Recursion(                             # the electroacoustic loop
                fF = Series(Delay(latency), FDN),  #   mics → loudspeakers
                fB = MatrixConvolver(coupling),    #   loudspeakers → mics
                block_size = loop_block,
            ),
            MatrixConvolver(loudspeakers → listener),
        ),
        MatrixConvolver(source → listener),        # dry direct path
    )
    ```

    The feedback loop `loudspeaker → room → mic → FDN → loudspeaker` is literally a `td.Recursion`. A short `Delay` (the RES processing latency, ~5 ms) leads its forward path, which is what lets the block recursion break the loop — the same role the FDN's own delays play inside the FDN. A `Recursion` processes `block_size` samples at a time and so inserts that many samples of delay into
    its loop, which is why the latency delay (and, inside the FDN, the delay lines) is shortened by exactly one block. One `.process_signal(source)` runs the whole system.
    """)
    return


@app.cell
def _(
    coupling,
    make_reverberator,
    np,
    sig_len,
    source_to_listener,
    source_to_mic,
    speaker_to_listener,
    td,
):
    latency = 256  # RES processing latency (~5.3 ms); breaks the electroacoustic loop
    loop_block = 64  # Recursion block size of the electroacoustic loop

    def build_res(time_varying, g):
        """Assemble the whole reverberation enhancement system as a td tree."""
        electroacoustic_loop = td.Recursion(
            td.Series(
                [
                    # shortened by loop_block, which the Recursion adds back
                    td.Delay(np.full(6, latency - loop_block)),
                    make_reverberator(time_varying, g),
                ]
            ),
            td.MatrixConvolver(coupling),  # loudspeakers -> mics (closes the loop)
            block_size=loop_block,
        )
        res_path = td.Series(
            [
                td.MatrixConvolver(source_to_mic),  # source -> 6 mics
                electroacoustic_loop,  # 6 mics -> 6 loudspeakers
                td.MatrixConvolver(speaker_to_listener),  # 6 loudspeakers -> listener
            ]
        )
        return td.Parallel(
            [res_path, td.MatrixConvolver(source_to_listener)], sum_output=True
        )

    impulse = np.zeros(sig_len)
    impulse[0] = 1.0

    def render(time_varying, g):
        """Listener impulse response of the RES at loop gain ``g``."""
        return build_res(time_varying, g).process_signal(impulse).squeeze()

    return (render,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reverberation enhancement

    At a comfortable loop gain the RES adds a long reverberant tail to the dry response — the energy decay curve at the listener decays far more slowly with the system on.
    """)
    return


@app.cell
def _(fs, np, plt, render):
    def edc_db(x):
        energy = np.cumsum(x[::-1] ** 2)[::-1]
        return 10 * np.log10(energy / energy[0] + 1e-20)

    g_operating = 0.8
    enhanced = render(time_varying=True, g=g_operating)
    dry = render(time_varying=False, g=0.0)  # g=0 -> only the dry direct path

    late = slice(int(0.4 * fs), None)
    enhancement_ratio = np.sqrt((enhanced[late] ** 2).mean()) / np.sqrt(
        (dry[late] ** 2).mean()
    )
    print(f"Late-tail RMS gain (enhanced / dry): {enhancement_ratio:.1f}x")
    assert np.isfinite(enhanced).all()
    assert enhancement_ratio > 5.0

    t = np.arange(len(dry)) / fs
    fig_edc, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.plot(t, edc_db(dry), color="#888", label="dry room")
    _ax.plot(t, edc_db(enhanced), color="#7b5ea7", label=f"RES on (g={g_operating})")
    _ax.set_ylim(-60, 2)
    _ax.set_title("Energy decay at the listener: RES extends the reverberation")
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Energy decay (dB)")
    _ax.legend()
    fig_edc.tight_layout()
    fig_edc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Maximum stable gain: static vs time-varying

    Now push the loop gain up. We track the short-time energy envelope of the listener response: a stable (well-enhanced) system **decays**, an over-driven one **grows** as a loop mode regenerates. At the same high gain the static FDN is already growing (ringing), while the time-varying FDN still decays — its modulation breaks up the runaway mode.
    """)
    return


@app.cell
def _(fs, np, plt, render):
    def envelope_db(x, win=2048, hop=512):
        starts = np.arange(0, len(x) - win, hop)
        env = np.array(
            [10 * np.log10((x[i : i + win] ** 2).mean() + 1e-20) for i in starts]
        )
        return starts / fs, env

    g_challenge = 1.6
    rec_static = render(time_varying=False, g=g_challenge)
    rec_varying = render(time_varying=True, g=g_challenge)

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

    ts, es = envelope_db(rec_static)
    tv, ev = envelope_db(rec_varying)
    fig_msg, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.plot(ts, es, color="#c0392b", label="static FDN (rings)")
    _ax.plot(tv, ev, color="#7b5ea7", label="time-varying FDN (stable)")
    _ax.set_title(
        f"Short-time energy at g={g_challenge}: static grows, time-varying decays"
    )
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Energy (dB)")
    _ax.legend()
    fig_msg.tight_layout()
    fig_msg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The stability margin

    Sweeping the loop gain makes the gain in maximum stable gain explicit: the growth ratio crosses 1 (the stability boundary) at a higher gain for the time-varying FDN. The horizontal distance between the two crossings is the extra gain — a few dB — that time variation buys.
    """)
    return


@app.cell
def _(np, plt, render):
    gains = np.array([1.0, 1.4, 1.8, 2.2, 2.6])

    def growth_ratio(x):
        q = len(x) // 4
        return (x[-q:] ** 2).mean() / ((x[:q] ** 2).mean() + 1e-30)

    ratios_static, ratios_varying = [], []
    for g in gains:
        ratios_static.append(growth_ratio(render(time_varying=False, g=g)))
        ratios_varying.append(growth_ratio(render(time_varying=True, g=g)))

    fig_sweep, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(gains, ratios_static, "o-", color="#c0392b", label="static FDN")
    _ax.plot(gains, ratios_varying, "o-", color="#7b5ea7", label="time-varying FDN")
    _ax.axhline(1.0, linestyle="--", color="#444")
    _ax.annotate(
        "stability boundary", (gains[-1], 1.0), ha="right", va="bottom", color="#444"
    )
    _ax.set_yscale("log")
    _ax.set_title(
        "Tail growth vs loop gain: time variation raises the maximum stable gain"
    )
    _ax.set_xlabel("Loop gain g")
    _ax.set_ylabel("Tail growth ratio")
    _ax.legend()
    fig_sweep.tight_layout()
    fig_sweep
    return


if __name__ == "__main__":
    app.run()
