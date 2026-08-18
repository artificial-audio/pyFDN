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
    2. `pyFDN.train_fdn(model, loss)` -- optimize the feedback matrix and gains toward an objective we write out in full, in place.
    3. `pyFDN.extract_build` -- read both the initial and optimized FDNs back out.

    Delays stay fixed. We then add homogeneous decay so the result is audible.

    ## The objective, spelled out

    Every loss is a function of the model's **impulse response**, so an objective is a sum of named parts rather than a mode to select:

    ```python
    loss = pyFDN.FlatMagnitude() + 0.2 * pyFDN.Sparsity(pyFDN.param(model, "feedback"))
    ```

    * `pyFDN.FlatMagnitude()` reads `pyFDN.Response.magnitude` -- the `rfft` of the impulse response -- and fits it to a constant. That is the colorless objective.
    * `pyFDN.Sparsity(...)` is a cost on a *model parameter* rather than on the response, so it has to name one: `pyFDN.param(model, "feedback")` resolves the feedback matrix in this model's graph, and `pyFDN.params(model)` lists everything else on offer. It rewards a dense matrix, i.e. good mixing.

    `train_fdn(model, "colorless")` is shorthand for exactly this sum -- see `pyFDN.train.presets`.

    ## Two details that make or break the fit

    * **Rendering.** A **lossless** FDN has every pole *exactly* on the unit circle, where the FFT-domain evaluation is near-singular and the impulse response comes out wrong. `build_fdn(rt=None)` therefore defaults to `alias_decay_db=`{pyFDN.LOSSLESS_ALIAS_DECAY_DB}: flamo evaluates the system on a slightly smaller circle, and the `"time"` output layer removes that $\\gamma^n$ envelope again. What a loss sees is the impulse response *itself*, accurate to {pyFDN.LOSSLESS_ALIAS_DECAY_DB:.0f} dB -- which is all `alias_decay_db` means. It never reaches the extracted build.
    * **Reading the result.** The numbers below are the loss terms themselves: `TrainLog.loss_log` keeps each term's history *unweighted*, so `FlatMagnitude` is exactly the mean squared deviation of $|H|$ from 1 that was optimized, and `Sparsity` the density of the feedback matrix. They compare across runs at the same `nfft` -- but not across different `nfft`, since truncating to a longer window widens the peak-to-median range of $|H|$ and the same FDN then scores a larger MSE.

    - pyFDN training pipeline: Jeremy B. Bai, 2026-06-19
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    return np, pyFDN


@app.cell
def _(pyFDN):
    fs = 48000
    # The objective's frequency resolution is this nfft and nothing else: it is
    # the rfft of an nfft-long impulse response. More of it resolves the modes
    # more finely and fits them better, at proportionally more time per step.
    nfft = 2**14

    # 1. build a small "tiny colorless" lossless skeleton. rt=None also switches
    #    on the default anti-aliasing decay, without which a lossless FDN's
    #    impulse response cannot be rendered at all.
    delays = pyFDN.sample_delay_lengths(
        8, (200, 600), distribution="geometric", coprime=True, sort=True, rng=2
    )
    model = pyFDN.build_fdn(delays=delays, rt=None, nfft=nfft, device="cpu", rng=2)
    init_build = pyFDN.extract_build(model)  # random init, before training

    # |H| exactly as the loss sees it: the rfft of the model's impulse response.
    mag_init = pyFDN.model_response(model).magnitude.detach().numpy().squeeze()

    # 2. write the objective out: a flat magnitude response, plus a density
    #    reward on this model's feedback matrix.
    loss = pyFDN.FlatMagnitude() + 0.2 * pyFDN.Sparsity(pyFDN.param(model, "feedback"))
    # loss = pyFDN.FlatSpectrogram(nfft=(256, 512, 1024, 2048)) + 0.2 * pyFDN.Sparsity(
    # pyFDN.param(model, "feedback")
    # )

    # 3. train in place; then extract. Adam, not L-BFGS: the magnitude objective
    #    is nonconvex and densely modal, and L-BFGS's line search settles into
    #    the nearest stationary point within a few dozen steps.
    log = pyFDN.train_fdn(
        model,
        loss,
        optimizer="adam",
        max_steps=2000,
        lr=1e-2,
        # This objective crosses long flat stretches before improving again;
        # the default patience of 10 stops inside one of them (at nfft=2**14
        # that ends the fit after 24 steps, at more than twice the loss).
        patience=100,
        device="cpu",
        rng=1,
    )
    opt_build = pyFDN.extract_build(model)
    mag_opt = pyFDN.model_response(model).magnitude.detach().numpy().squeeze()

    # 4. report the loss itself -- the total, then each term unweighted.
    print(
        f"ran {log.steps_run} steps, total loss "
        f"{log.train_loss[0]:.4f} -> {log.train_loss[-1]:.4f}"
    )
    for _term, _history in log.loss_log.items():
        print(f"  {_term:18} {_history[0]:.4f} -> {_history[-1]:.4f}")
    return fs, init_build, log, mag_init, mag_opt, nfft, opt_build


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Magnitude response and training loss

    The magnitude response the loss actually saw -- `pyFDN.Response.magnitude`, the rfft of the model's `nfft`-long impulse response -- before and after training. `FlatMagnitude` fits this curve to the dashed line at 0 dB, and the peaks and notches pull in toward it.
    """)
    return


@app.cell
def _(fs, log, mag_init, mag_opt, mo, nfft, np, pyFDN):
    import matplotlib.pyplot as plt

    _freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.2))
    _axes[0].plot(_freqs, pyFDN.lin_to_db(mag_init), alpha=0.5, label="init")
    _axes[0].plot(_freqs, pyFDN.lin_to_db(mag_opt), label="colorless")
    _axes[0].axhline(0.0, color="k", ls="--", lw=0.8, label="target |H| = 1")
    _axes[0].set(
        xlabel="frequency [Hz]",
        ylabel="magnitude [dB]",
        xscale="log",
        title="Magnitude response as the loss sees it",
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
