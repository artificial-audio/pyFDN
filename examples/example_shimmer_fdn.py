# gallery_category: Effects
# gallery_title: Shimmer reverberation with nonlinear FDNs
# gallery_description: Drop five nonlinear and pitch-shifting operators into the feedback loop of an FDN and hear how each one turns a plain reverb into a shimmer effect.

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Shimmer reverberation with nonlinear feedback delay networks

    The classic recipe of a shimmer reverb uses a pitch shifter in series to a
     reverb. Here the nonlinearity goes *inside* the FDN's feedback loop instead:
    it is placed right after the delay lines, so its output is first mixed by `A` and
    then redistributed across every delay line on the next round trip, and the effect
    compounds with every recirculation rather than being applied once.

    Any operator dropped into that loop has to roughly preserve energy, or the
    reverb either dies out early or blows up. This notebook works through
    five such operators, all part of `pyFDN.td`, applied to a handful of channels
    at a time so you can hear each one against the same plain FDN:

    | operator | what it does | main knob |
    | --- | --- | --- |
    | `ControllableFullWaveRect` | $y[n] = g_\text{cfwr}\big((1-\alpha)x[n] + \alpha\lvert x[n]\rvert\big)$ — a tunable rectifier | `alpha` |
    | `SDFD` | splits the signal into positive/negative parts and delays them by a different, signal-dependent amount | `d` |
    | `RingModulator` | $y[n] = g_\text{rm}\, x[n]\sin(2\pi f_\text{rm} n / f_s)$ — amplitude modulation by a sine | `mod_freq` |
    | `PitchShift` | dual-read-head pitch shifter reading a circular buffer at a different rate than it's written | `transpose_cents` |
    | `GranularPitchShift` | the same idea, done with short overlapping grains instead of two continuous read heads | `transpose_cents`, `grain_dur_samps` |

    Each is a `post_delay` hook for `process_dss`.


    Reference: "Shimmer Reverberation with Nonlinear Feedback Delay Networks",
    Gloria Dal Santo, Xiaojie Pi, Karolina Prawda, Sebastian J. Schlecht, Vesa Välimäki.
    Proceedings of the 29th International Conference on Digital Audio Effects (DAFx26), Cambridge, MA, USA, 1–4 September 2026
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN
    from pyFDN import td

    np.random.seed(0)
    return np, pyFDN, td


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The plain FDN

    An 8-line FDN with a random orthogonal feedback matrix and a frequency-dependent
    decay. This is the reverb the nonlinearities below will be
    dropped into. Everything that follows changes only the `post_delay` hook.
    """)
    return


@app.cell
def _(np, pyFDN):
    fs = 48_000  # sampling rate in Hz
    N = 8  # number of delay lines
    ir_len_seconds = 3.0

    # delay line lengths
    delays = pyFDN.sample_delay_lengths(
        N,
        delay_range=(1000, 3000),  # samples: about 21-62 ms at 48 kHz
        distribution="geometric",
        coprime=True,  # avoid coinciding echoes
        rng=2,
    )

    A = pyFDN.fdn_matrix_gallery(N, "orthogonal")
    B = np.random.randn(N, 1) / np.sqrt(N)  # (N, num_inputs)
    C = np.random.randn(1, N) / np.sqrt(N)  # (num_outputs, N)
    D = np.zeros((1, 1))  # (num_outputs, num_inputs) — wet only

    target_rt = np.array([4.4, 4.4, 4.3, 4.1, 3.8, 3.4, 3.0, 1.7, 1.5, 0.5])

    absorption = pyFDN.decay_to_geq(target_rt, delays, fs)  # (n_sections, 6, N)

    build = pyFDN.FDNBuild(
        A=A, B=B, C=C, D=D, delays=delays, fs=fs, post_delay=absorption
    )
    ir = pyFDN.build_to_impz(build, int(ir_len_seconds * fs)).squeeze()
    return A, B, C, D, N, absorption, delays, fs, ir, target_rt


@app.cell
def _(fs, ir, mo, pyFDN):
    mo.vstack(
        [
            # pyFDN.plot_FDN_build(
            #     build,
            #     title="The complete FDN"
            #     ),
            # pyFDN.plot_impulse_response(
            #     ir, fs=fs, title="Linear (Vanilla) FDN"
            # ),
            # pyFDN.plot_spectrogram(ir, fs, title="Impulse response — time-frequency energy"),
            pyFDN.labeled_audio(
                "Linear (Vanilla) FDN", pyFDN.peak_normalize(ir), fs=fs
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The reference: plain FDN, no nonlinearity

    `process_dss` runs the same recursion as above, driven by a dry synth signal
    instead of an impulse. This "plain FDN" render is the baseline every
    nonlinear variant below gets compared against — same input, same delays, same
    decay, nothing else in the loop yet.
    """)
    return


@app.cell
def _(A, B, C, D, absorption, delays, fs, mo, np, pyFDN, td):
    dry, _ = pyFDN.load_audio("synth_dry", fs=fs)
    x = np.pad(dry, (0, 2 * fs))  # room for the tail

    wet = pyFDN.process_dss(
        x,
        delays,
        A,
        B,
        C,
        D,
        post_delay=td.SOSBank(absorption),  # absorption inside the loop
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("dry", dry, fs=fs),
            pyFDN.labeled_audio("wet", pyFDN.peak_normalize(wet), fs=fs),
        ]
    )
    return wet, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Controllable full-wave rectifier

    A tunable rectifier: at `alpha = 0` the nonlinearity drops out, at
    `alpha = 1` it's a full-wave rectifier ($y = g_\text{cfwr}\lvert x\rvert$),
    and in between it blends the two. Rectification folds the negative half of the waveform onto
    the positive one, generating even harmonics. The operator is followed internally
    by a DC blocker with slow gain compensation. This prevent any DC offset from
    building up over thousands of trips around the loop. Placed in the feedback path, each
    recirculation adds another layer of harmonics on top of the last, so the
    distortion thickens the longer the tail rings on. `active_channels` picks
    which delay lines get the operator, here the two channels with the
    longest delays, so the effect builds in gradually behind the clean lines.
    """)
    return


@app.cell
def _(A, B, C, D, absorption, delays, fs, mo, pyFDN, td, wet, x):
    wet_cfwr = pyFDN.process_dss(
        x,
        delays,
        A,
        B,
        C,
        D,
        post_delay=td.Series(
            [
                td.SOSBank(absorption),  # absorption inside the loop
                td.ControllableFullWaveRect(
                    len(delays), alpha=0.25, active_channels=[-1, -3]
                ),
            ]
        ),
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("plain FDN", pyFDN.peak_normalize(wet), fs=fs),
            pyFDN.labeled_audio(
                "Controllable Full-Wave Rectifier FDN",
                pyFDN.peak_normalize(wet_cfwr),
                fs=fs,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal-dependent fractional delay (SDFD)

    In the SDFD, the input signal is split into its positive and negative parts,
    and each part is delayed by a slightly different, sub-sample amount that depends
    on `d` (roughly `1 + d` samples for the positive part, `1 - d` for the negative
    one). Recombining two copies of the same signal offset by a fraction of a
    sample smears the zero crossings, which is heard as a soft, asymmetric
    distortion rather than a hard clip. It's milder and less
    harmonically dense than the rectifier above, so it reads more as
    "coloration" than "distortion" once it's circulating in the loop.
    """)
    return


@app.cell
def _(A, B, C, D, absorption, delays, fs, mo, pyFDN, td, wet, x):
    wet_sdfd = pyFDN.process_dss(
        x,
        delays,
        A,
        B,
        C,
        D,
        post_delay=td.Series(
            [
                td.SOSBank(absorption),  # absorption inside the loop
                td.SDFD(len(delays), d=0.5, active_channels=[4, 5, 6, 7]),
            ]
        ),
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("plain FDN", pyFDN.peak_normalize(wet), fs=fs),
            pyFDN.labeled_audio("SDFD FDN", pyFDN.peak_normalize(wet_sdfd), fs=fs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ring modulation

    A ring modulator multiplies the signal by a sine wave at `mod_freq`, which
    shifts its spectrum up and down by that frequency rather than adding
    harmonics on top of it. At the 10 Hz used below it reads as a tremolo, a slow
    pulsing of the reverb tail. Push `mod_freq` into the audible range instead and the
    tremolo turns into a genuine timbral shift: the reverb takes on an
    inharmonic, bell- or metal-like character, because the sidebands it creates
    are no longer harmonically related to the original spectrum. `mod_amp`
    compensates for the fact that a unit sine only has half the average power
    of the signal it multiplies.
    """)
    return


@app.cell
def _(A, B, C, D, absorption, delays, fs, mo, np, pyFDN, td, wet, x):
    wet_rm = pyFDN.process_dss(
        x,
        delays,
        A,
        B,
        C,
        D,
        post_delay=td.Series(
            [
                td.SOSBank(absorption),  # absorption inside the loop
                td.RingModulator(
                    len(delays),
                    mod_freq=10,
                    mod_amp=np.sqrt(2),
                    fs=fs,
                    active_channels=[4, 5, 6, 7],
                ),
            ]
        ),
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("plain FDN", pyFDN.peak_normalize(wet), fs=fs),
            pyFDN.labeled_audio(
                "Ring Modulator FDN", pyFDN.peak_normalize(wet_rm), fs=fs
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pitch shift

    This is the operator that gives the effect closest to conventional shimmer
    reverbs. A pitch shifter by `transpose_cents` cents (700 cents ≈ a perfect fifth, 1200
    would be a full octave). Each channel writes into a circular buffer and reads it back
    with two read heads spaced half a window apart, moving faster or slower
    than the write head to compress or stretch time, which is heard as a
    pitch change. The two heads cross-fade into each other so that whichever
    one is about to wrap around the buffer is faded out first, avoiding a
    click. Because this sits in the feedback path, every trip around the loop
    shifts the signal a little further: the reverb tail climbs in pitch the
    longer it rings, rather than being shifted once and left alone. Only
    channel 1 is shifted, so you can hear the shifted line rise out of the
    otherwise-unchanged reverb around it.

    This effect works better with longer delays, so let's sample a new set.
    """)
    return


@app.cell
def _(A, B, C, D, N, fs, mo, pyFDN, target_rt, td, wet, x):
    delays_2 = pyFDN.sample_delay_lengths(
        N,
        delay_range=(1000, 6000),  # samples: about 21-125 ms at 48 kHz
        distribution="geometric",
        coprime=True,  # avoid coinciding echoes
        rng=2,
    )
    absorption_2 = pyFDN.decay_to_geq(target_rt, delays_2, fs)  # (n_sections, 6, N)

    window_size = 2048
    wet_ps = pyFDN.process_dss(
        x,
        delays_2,
        A,
        B,
        C,
        D,
        post_delay=td.Series(
            [
                td.SOSBank(absorption_2),  # absorption inside the loop
                td.PitchShift(
                    len(delays_2),
                    max_delay_samps=window_size * 2,
                    window_size=window_size,
                    transpose_cents=-700,
                    fs=fs,
                    active_channels=[-1, -2],
                ),
            ]
        ),
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("plain FDN", pyFDN.peak_normalize(wet), fs=fs),
            pyFDN.labeled_audio("Pitch Shift FDN", pyFDN.peak_normalize(wet_ps), fs=fs),
        ]
    )
    return absorption_2, delays_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Granular pitch shift

    Same idea as `PitchShift`, but done with short, randomly placed grains instead of two
    continuous read heads. Every `grain_dur_samps` samples, each of the two
    active grains jumps to a new random position inside the buffer and starts
    reading from there, windowed by a raised-cosine envelope so the jump is
    inaudible as a click. The randomness trades the smooth, continuous pitch
    glide of `PitchShift` for a more diffuse, granular texture. Shorter
    grains sound more textured and "clouded," longer ones preserve more of the
    original timbre.
    """)
    return


@app.cell
def _(A, B, C, D, absorption_2, delays_2, fs, mo, pyFDN, td, wet, x):
    grain_dur_samps = 1024
    wet_gps = pyFDN.process_dss(
        x,
        delays_2,
        A,
        B,
        C,
        D,
        post_delay=td.Series(
            [
                td.SOSBank(absorption_2),  # absorption inside the loop
                td.GranularPitchShift(
                    len(delays_2),
                    max_delay_samps=grain_dur_samps * 4,
                    grain_dur_samps=grain_dur_samps,
                    transpose_cents=700,
                    active_channels=[
                        -1,
                    ],
                    seed=0,  # grain positions are random; fix them for the docs
                ),
            ]
        ),
    )

    mo.hstack(
        [
            pyFDN.labeled_audio("plain FDN", pyFDN.peak_normalize(wet), fs=fs),
            pyFDN.labeled_audio(
                "Granular Pitch Shift FDN", pyFDN.peak_normalize(wet_gps), fs=fs
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
