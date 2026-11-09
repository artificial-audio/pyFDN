# gallery_category: Analysis & Verification
# gallery_title: Tracking the modes of a reverberation enhancement system
# gallery_description: Follow the dominant poles of an FDN-based RES as its loop gain rises past the maximum stable gain, and see which channels carry the mode that goes unstable.

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
    # Tracking the modes of a reverberation enhancement system

    A **reverberation enhancement system** (RES) feeds microphone signals through a reverberator to loudspeakers in the same room, so the loudspeakers reach the microphones again: an acoustic feedback loop around an FDN. Its stability is decided by a few modes. As the electronic loop gain $g$ rises they move towards the unit circle, ring longer, and at the **maximum stable gain** (MSG) the first one crosses.

    The whole loop has tens of thousands of poles, so a full modal decomposition at every gain is out of reach. Instead we

    1. find the few **dominant** poles at a high gain with `pyFDN.seed_poles`,
    2. follow them down the gain with `pyFDN.track_poles`, a continuation method after Bai and Schlecht (ICASSP 2026),
    3. read off each mode's decay rate, the MSG, and which channels carry the critical mode.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from flamo.processor import system
    from scipy.signal import lfilter

    import pyFDN

    return lfilter, np, plt, pyFDN, system, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The system

    An FDN with four delay lines and a 1.5 s reverberation time, four microphones and four loudspeakers. The loudspeaker → microphone coupling $R(z)$ is a $4 \times 4$ FIR matrix of 1024 taps: a direct path $1/r$ at the propagation delay and a decaying, low-passed noise tail for the early room response.
    """)
    return


@app.cell
def _(lfilter, np, pyFDN):
    fs = 48_000.0
    rng = np.random.default_rng(2)
    n_lines = n_channels = 4
    delays = np.array([997, 1153, 1327, 1559])
    line_gains = 10 ** (-3 * delays / (1.5 * fs))  # 1.5 s reverberation time
    A = pyFDN.random_orthogonal(n_lines, rng=rng) * line_gains[None, :]
    B = rng.standard_normal((n_lines, n_channels)) / n_lines  # mics -> FDN
    C = rng.standard_normal((n_channels, n_lines)) / n_lines  # FDN -> speakers

    _taps = 1024
    _n = np.arange(_taps)
    _distance = rng.uniform(2.0, 5.0, (n_channels, n_channels))  # metres
    _lag = np.round(_distance / 343 * fs).astype(int)
    coupling = np.zeros((n_channels, n_channels, _taps))
    for _i in range(n_channels):
        for _j in range(n_channels):
            _tail = np.where(_n > _lag[_i, _j], np.exp(-(_n - _lag[_i, _j]) / 600), 0)
            _noise = lfilter([0.3], [1, -0.7], 0.05 * rng.standard_normal(_taps))
            coupling[_i, _j] = _tail * _noise
            coupling[_i, _j, _lag[_i, _j]] += 1 / _distance[_i, _j]
    return A, B, C, coupling, delays, fs, n_channels, n_lines


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One flat loop

    Pole tracking needs the RES as **one** FLAMO `Recursion`, not the FDN's loop nested inside the acoustic one: a nested loop puts the FDN's own poles into $\det P(z)$, right next to the closed-loop poles being tracked. With the delay-line outputs $x$ and the microphone signals $m$ as the loop state, the characteristic matrix is

    $$
    P(z) = I - \begin{bmatrix} D_m(z) & 0 \\ 0 & R(z) \end{bmatrix}
    \begin{bmatrix} A & B \\ g\,C & 0 \end{bmatrix},
    $$

    with $D_m(z)$ the delays. The feedforward path is block-diagonal: FLAMO's `Parallel` feeds both branches the same input, so a fixed selection gain in front of each picks its part. The feedback path is the mixing matrix followed by a diagonal gain that holds $g$ on the loudspeaker rows, which is the parameter we move.
    """)
    return


@app.cell
def _(A, B, C, coupling, delays, fs, n_channels, n_lines, np, pyFDN, system, torch):
    _nfft = 2**10  # not used for probing; FLAMO needs one to build modules
    _opt = {"device": "cpu", "dtype": torch.float64}  # pole tracking needs float64
    _select_x = np.hstack([np.eye(n_lines), np.zeros((n_lines, n_channels))])
    _select_s = np.hstack([np.zeros((n_channels, n_lines)), np.eye(n_channels)])
    _feedforward = system.Parallel(
        system.Series(
            pyFDN.gain_module(_select_x, _nfft, **_opt),
            pyFDN.delay_module(delays, _nfft, fs=fs, **_opt),
        ),
        system.Series(
            pyFDN.gain_module(_select_s, _nfft, **_opt),
            pyFDN.fir_matrix_module(coupling, _nfft, **_opt),
        ),
        sum_output=False,
    )
    _mixing = np.block([[A, B], [C, np.zeros((n_channels, n_channels))]])
    _feedback = system.Series(
        pyFDN.gain_module(_mixing, _nfft, **_opt),
        pyFDN.gain_module(np.eye(n_lines + n_channels), _nfft, **_opt),
    )
    loop = system.Recursion(fF=_feedforward, fB=_feedback)
    # degree of det P(z): the delays plus the coupling FIR on every microphone
    order = int(delays.sum() + n_channels * (coupling.shape[-1] - 1))
    print("parameters:", [_name for _name, _ in loop.named_parameters()])
    print("order:", order)
    return loop, order


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gain path and dominant poles

    The path moves only `feedback.1.param`, the diagonal gain, from −20 dB to −5 dB in steps uniform in dB (for a delay loop, $\log z$ moves linearly in $\log g$, so the predictor is nearly exact). We seed the eight largest poles at the **top** of the range, where stability is decided, and track them **backwards**: the modes that matter are the ones dominant near the MSG, not at low gain.
    """)
    return


@app.cell
def _(loop, n_channels, n_lines, np, order, pyFDN, torch):
    low_db, high_db = -20.0, -5.0

    def path(t):
        gain = 10 ** ((low_db + (high_db - low_db) * t) / 20)
        ones = torch.ones(n_lines, dtype=torch.float64)
        return {
            "feedback.1.param": torch.diag(
                torch.cat([ones, gain * ones.new_ones(n_channels)])
            )
        }

    seeds = pyFDN.seed_poles(loop, 8, path=path, t=1.0, order=order)
    tracks = pyFDN.track_poles(
        loop, path, seeds, t0=1.0, t1=0.0, order=order, max_step=0.04
    )
    gain_db = low_db + (high_db - low_db) * tracks.t
    print(
        f"{len(tracks.t)} steps, tracks lost: {int(np.isnan(tracks.poles[-1]).sum())}"
    )
    return gain_db, high_db, low_db, path, tracks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decay rate of the dominant modes

    A pole $z_k$ decays (or grows) at $20 f_s \log_{10}|z_k|$ dB per second; its $T_{60}$ is $60$ divided by the decay rate. Each mode's critical gain, where its rate crosses zero, is interpolated between the tracked steps (about half a dB apart). The MSG is the lowest of them.
    """)
    return


@app.cell
def _(fs, gain_db, np, plt, tracks):
    rate = 20 * fs * np.log10(np.abs(tracks.poles))  # dB/s, > 0 grows
    frequency = np.angle(tracks.poles) * fs / (2 * np.pi)
    _order = np.argsort(gain_db)
    critical = np.array(  # NaN: the mode does not cross within the range
        [
            np.interp(0.0, rate[_order, _k], gain_db[_order])
            if rate[:, _k].min() < 0 < rate[:, _k].max()
            else np.nan
            for _k in range(rate.shape[1])
        ]
    )
    critical_mode = int(np.nanargmin(critical))
    msg_db = float(critical[critical_mode])

    _fig, (_ax_rate, _ax_freq) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for _k in range(rate.shape[1]):
        _style = (
            {"lw": 2.5, "color": "C3"}
            if _k == critical_mode
            else {"lw": 1, "color": "0.5"}
        )
        _ax_rate.plot(gain_db, rate[:, _k], **_style)
        _ax_freq.plot(gain_db, frequency[:, _k] / 1000, **_style)
    for _ax in (_ax_rate, _ax_freq):
        _ax.axvline(msg_db, color="C3", ls="--", lw=1)
        _ax.set_xlabel("loop gain (dB)")
    _ax_rate.axhline(0, color="k", lw=0.8)
    _ax_rate.set_ylabel("decay rate (dB/s, > 0 grows)")
    _ax_rate.set_title(f"MSG = {msg_db:.2f} dB")
    _ax_freq.set_ylabel("mode frequency (kHz)")
    _ax_freq.set_title("the modes barely move in frequency")
    _fig.tight_layout()
    print(f"critical mode at {frequency[0, critical_mode]:.0f} Hz, MSG {msg_db:.2f} dB")
    print(
        f"T60 of that mode 3 dB below the MSG: {60 / -np.interp(msg_db - 3, gain_db[_order], rate[_order, critical_mode]):.1f} s"
    )
    _fig
    return critical_mode, msg_db


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check: no untracked mode crosses first

    Tracking eight modes can only find the MSG if none of the other poles crosses earlier. Seeding again directly at the MSG, from scratch, must give the critical mode as the largest pole, on the unit circle: a decay rate of zero.
    """)
    return


@app.cell
def _(fs, high_db, loop, low_db, msg_db, np, order, path, pyFDN):
    _t = (msg_db - low_db) / (high_db - low_db)
    _largest = pyFDN.seed_poles(loop, 1, path=path, t=_t, order=order)[0]
    _rate = 20 * fs * np.log10(abs(_largest))
    print(
        f"largest pole at the MSG: {np.angle(_largest) * fs / (2 * np.pi):.0f} Hz, {_rate:+.3f} dB/s"
    )
    assert abs(_rate) < 0.1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which channels carry the critical mode

    At a pole, the right singular vector $r$ of $P$ satisfies $r = F B r$: it is the loop state of the mode, the delay-line outputs and the microphone signals. Its magnitudes show where the mode lives: which microphones pick up the feedback most, i.e. where a notch or a gain change would help.
    """)
    return


@app.cell
def _(critical_mode, gain_db, msg_db, n_channels, n_lines, np, plt, tracks):
    _step = int(np.argmin(np.abs(gain_db - msg_db)))
    _weight = np.abs(tracks.right[_step, critical_mode])
    _labels = [f"line {_i + 1}" for _i in range(n_lines)] + [
        f"mic {_i + 1}" for _i in range(n_channels)
    ]
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.bar(
        _labels, _weight / _weight.max(), color=["0.6"] * n_lines + ["C0"] * n_channels
    )
    _ax.set_ylabel("|r| (normalised)")
    _ax.set_title("loop state of the critical mode at the MSG")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
