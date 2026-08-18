# gallery_category: FDN Design & Analysis
# gallery_title: Train a colorless FDN
# gallery_description: Optimize an FDN for a flat lossless magnitude response, extract its build, and add decay for listening.

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo, pyFDN):
    mo.md(f"""
    # Colorless FDN, trained in-notebook

    The companion to **Colorless FDN**, which loads pre-optimized, build-ready JSON presets. Here we run the optimization ourselves with `pyFDN`'s training API, following *{pyFDN.paper_link("Differentiable_FDN_For_Colorless_Reverberation")}* (and its "tiny colorless FDN" follow-up):

    1. `pyFDN.build_fdn` -- a standard FDN skeleton with random orthogonal feedback matrix.
    2. `pyFDN.train_fdn(model, "colorless")` -- optimize the feedback matrix and gains for a flat magnitude (magnitude MSE + a feedback-matrix sparsity penalty), in place.
    3. `pyFDN.extract_build` -- read both the initial and optimized FDNs back out.

    Delays stay fixed. We then add homogeneous decay so the result is audible.

    Two details make or break this fit, both to do with the fact that a **lossless** FDN with an orthogonal feedback matrix has every pole *exactly* on the unit circle, so $|H(\\omega)|$ is unbounded:

    * **Training.** `build_fdn(rt=None)` therefore defaults to `alias_decay_db=`{pyFDN.LOSSLESS_ALIAS_DECAY_DB}, flamo's $\\gamma^n$ envelope, which evaluates the system just *inside* the unit circle. Without it the MSE is dominated by whichever handful of bins land nearest a pole, and the optimizer flattens nothing -- it just scales `b` and `c` down. The envelope never reaches the extracted build: it enters flamo's `get_freq_response`, not the parameter map.
    * **Measuring.** For the same reason, spectral flatness of the lossless $|H|$ is not a usable number -- its arithmetic mean is set by a couple of pole spikes, so it swings by orders of magnitude with the FFT size. We report flatness of the **decayed** response instead. Homogeneous decay does not change colouration, so this measures the thing we optimized, on a grid fine enough to resolve the modes.

    - pyFDN training pipeline: Jeremy B. Bai, 2026-06-19
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    # Measure the colour on the DECAYED response, on a grid fine enough to
    # resolve the modes: the lossless |H| has poles on the unit circle, so its
    # flatness is dominated by a few spikes and depends on the FFT size.
    EVAL_NFFT = 2**16  # > 2 * sum(delays), so every mode gets its own bins
    EVAL_RT = 1.0  # homogeneous decay does not change colouration

    def flatness(magnitude):
        # Spectral flatness (geometric/arithmetic mean of power, DC excluded);
        # 1.0 is perfectly flat. Inlined here so the example needs no metrics API.
        power = np.abs(magnitude).ravel()[1:] ** 2
        power = power[power > 0]
        if power.size == 0:
            return 0.0
        return float(np.exp(np.mean(np.log(power))) / np.mean(power))

    def decayed_magnitude(build):
        """|H| of `build` with homogeneous decay, on the fine evaluation grid."""
        model = pyFDN.trainable_from_build(
            pyFDN.build_set_decay(build, EVAL_RT), nfft=EVAL_NFFT, output="magnitude"
        )
        return np.abs(pyFDN.flamo_freq_response(model).squeeze())

    return EVAL_NFFT, decayed_magnitude, flatness, np, pyFDN


@app.cell
def _(decayed_magnitude, flatness, pyFDN):
    fs = 48000
    nfft = 2**12

    # 1. build a small "tiny colorless" lossless skeleton. rt=None also switches
    #    on the default anti-aliasing decay that keeps |H| bounded for training.
    delays = pyFDN.sample_delay_lengths(
        8, (600, 1200), distribution="geometric", coprime=True, rng=2
    )
    model = pyFDN.build_fdn(
        delays=delays, rt=None, nfft=nfft, output="magnitude", device="cpu", rng=2
    )
    init_build = pyFDN.extract_build(model)  # random init, before training

    # 2. train in place for a flat ("colorless") magnitude; then extract.
    #    Adam, not L-BFGS: the magnitude objective is nonconvex and densely
    #    modal, and L-BFGS's line search settles into the nearest stationary
    #    point within a few dozen steps.
    log = pyFDN.train_fdn(
        model,
        "colorless",
        optimizer="adam",
        max_steps=2000,
        lr=1e-2,
        device="cpu",
        rng=1,
    )
    opt_build = pyFDN.extract_build(model)

    # 3. measure on the decayed response (see above), not on the lossless |H|.
    mag_init = decayed_magnitude(init_build)
    mag_opt = decayed_magnitude(opt_build)

    print(
        f"ran {log.steps_run} steps, loss {log.train_loss[0]:.3f} -> {log.train_loss[-1]:.3f}"
    )
    print(
        f"spectral flatness  init {flatness(mag_init):.4f}"
        f"   colorless {flatness(mag_opt):.4f}"
    )
    return fs, init_build, log, mag_init, mag_opt, opt_build


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Magnitude response and training loss

    The magnitude response of each FDN *with decay applied* -- the lossless one is a forest of poles on the unit circle and plots as noise. Training pulls the peaks and notches in toward the mean; the flatness in the legend is the number that matters.
    """)
    return


@app.cell
def _(EVAL_NFFT, flatness, fs, log, mag_init, mag_opt, mo, np, pyFDN):
    import matplotlib.pyplot as plt

    _freqs = np.fft.rfftfreq(EVAL_NFFT, 1.0 / fs)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.2))
    _axes[0].plot(
        _freqs,
        pyFDN.lin_to_db(mag_init),
        alpha=0.5,
        label=f"init ({flatness(mag_init):.3f})",
    )
    _axes[0].plot(
        _freqs,
        pyFDN.lin_to_db(mag_opt),
        label=f"colorless ({flatness(mag_opt):.3f})",
    )
    _axes[0].set(
        xlabel="frequency [Hz]",
        ylabel="magnitude [dB]",
        xscale="log",
        title="Magnitude response with decay (flatness in legend)",
    )
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(log.train_loss, lw=1)
    _axes[1].set(xlabel="step", ylabel="loss", yscale="log", title="Training loss")
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FDN parameters: random init vs colorless

    The stored `FDNBuild` parameters side by side -- delay lengths, the feedback matrix `A`, and the input/output/direct gains `b`, `c`, `d`, on a shared color scale. Training reshapes `A` and the gains (the delays stay fixed) to flatten the magnitude response.
    """)
    return


@app.cell
def _(init_build, mo, opt_build, pyFDN):
    _build_init = pyFDN.plot_FDN_build(init_build, title="Random init")
    _build_opt = pyFDN.plot_FDN_build(opt_build, title="Colorless")
    mo.hstack([mo.as_html(_build_init), mo.as_html(_build_opt)], gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Listen: random init vs colorless

    Two renderings of each FDN, built end to end through the render API (`pyFDN.build_set_decay` -> `pyFDN.build_to_impz`), peak-normalized with `pyFDN.peak_normalize` so the A/B compares *timbre*, not level:

    * **Long tail (hear the colour)** -- a very long reverberation time, so the FDN rings with almost no decay and you hear its colour directly: the random init is tonal/metallic (sharp modal resonances), the colorless one is noise-like (flat spectrum). `build_to_impz` renders in the time domain (`pyFDN.process_fdn`), so the long ring is captured faithfully without the FFT wrap-around an nfft-length frequency-domain render would alias back onto the start.
    * **Reverb tail** -- a short homogeneous $T_{60}$, giving an audible decay.
    """)
    return


@app.cell
def _(fs, init_build, opt_build, pyFDN):
    # A long "ring" RT keeps the colour audible with little decay; a short RT gives
    # an audible reverb tail.
    rt_ring, rt_rev = 60.0, 2.0
    n_samples = int(2.0 * fs)

    def render(build, rt):
        """build (with homogeneous decay) -> peak-normalized 1-D impulse response."""
        ir = pyFDN.build_to_impz(pyFDN.build_set_decay(build, rt), n_samples).squeeze()
        # peak-normalize so the A/B compares timbre at matched level, not loudness.
        return pyFDN.fade_out(pyFDN.peak_normalize(ir), 2048)

    # Long "ring": the tail is still loud at the buffer end, so fade it out to
    # avoid a click on the abrupt cutoff.
    init_noise = render(init_build, rt_ring)
    opt_noise = render(opt_build, rt_ring)

    # Reverb: a short homogeneous T60 gives an audible decaying tail.
    init_decay = render(init_build, rt_rev)
    opt_decay = render(opt_build, rt_rev)
    return init_decay, init_noise, opt_decay, opt_noise


@app.cell
def _(fs, init_decay, init_noise, mo, opt_decay, opt_noise, pyFDN):
    _plot = pyFDN.plot_impulse_response(
        opt_decay, init_decay, fs=fs, labels=["Colorless", "Random init"]
    )
    _audio = mo.hstack(
        [
            mo.vstack(
                [
                    mo.Html("<b>Long tail (hear the colour)</b>").style(
                        {"font-size": "1.2em"}
                    ),
                    pyFDN.labeled_audio("Random init", init_noise, fs=fs),
                    pyFDN.labeled_audio("Colorless", opt_noise, fs=fs),
                ],
                gap=1,
            ),
            mo.vstack(
                [
                    mo.Html("<b>Reverb tail</b>").style({"font-size": "1.2em"}),
                    pyFDN.labeled_audio("Random init", init_decay, fs=fs),
                    pyFDN.labeled_audio("Colorless", opt_decay, fs=fs),
                ],
                gap=1,
            ),
        ],
        gap=2,
    )

    mo.vstack([_plot, _audio], gap=3)
    return


if __name__ == "__main__":
    app.run()
