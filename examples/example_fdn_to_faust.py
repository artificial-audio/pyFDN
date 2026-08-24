# gallery_category: Export & Deployment
# gallery_title: Compile an FDN to FAUST
# gallery_description: Compile a pyFDN design through FLAMO and adac into certified FAUST source for browser, offline, and plugin deployment.
# references: Franchino2026ADAC
# requires: adac

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from pyFDN import paper_link

    return mo, paper_link


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FDN to FAUST: compiling a pyFDN design to real-time DSP

    A pyFDN design lives as NumPy arrays and, through `dss_to_flamo`, as a differentiable FLAMO graph. Neither runs in a DAW. [`adac`](https://github.com/cucuwritescode/adac) closes that gap: it walks the FLAMO graph, extracts every delay, gain, matrix and filter into a JSON intermediate representation, and emits [FAUST](https://faust.grame.fr) source that compiles to a plugin.

    **The pipeline:**

    1. `pyFDN.fdn_build_gallery` + `pyFDN.dss_to_flamo` — design the FDN;
    2. `adac.flamo_to_json` — extract parameters into a JSON IR;
    3. `adac.certify` — small-gain stability certificate for the feedback loop;
    4. `adac.json_to_faust` — emit FAUST, with optional `rt60` / `dry_wet` macro knobs;
    5. run it: one click in the browser IDE, or `faust2juce` / `adac.export_juce` for a VST3.

    The last section renders the *compiled* FAUST and overlays it on the FLAMO reference, so the translation is checked, not assumed.
    """)
    return


@app.cell(hide_code=True)
def _(mo, paper_link):
    mo.md(f"""
    Reference: *{paper_link("Franchino2026ADAC")}*.

    Install: `pip install adac` (NumPy only).
    """)
    return


@app.cell
def _():
    import base64
    import importlib.util
    import json
    import shutil
    import subprocess
    import tempfile
    import urllib.parse
    from pathlib import Path

    import adac
    import numpy as np
    import torch

    import pyFDN

    return (
        Path,
        adac,
        base64,
        importlib,
        json,
        np,
        pyFDN,
        shutil,
        subprocess,
        tempfile,
        torch,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Design the FDN

    A six-line FDN from the gallery, mono in and stereo out: random orthogonal feedback, one-pole absorption per delay line giving 1.8 s at DC and 0.4 s at Nyquist, unity direct path.

    The output matrix `C` is `(2, 6)` with random gains, so the two channels are decorrelated mixes of the same delay lines — with `io_type="ones"` both columns would be identical and the result would be dual mono.

    Nothing here is FAUST-specific: this is the ordinary pyFDN design loop, and the rest of the notebook never assumes a channel count.
    """)
    return


@app.cell
def _(pyFDN, torch):
    torch.manual_seed(42)
    n = 6
    fs = 48000

    build = pyFDN.fdn_build_gallery(
        n,
        fs=fs,
        num_inputs=1,
        num_outputs=2,
        io_type="random",
        direct_gain=1.0,
        rt=1.8,
        rt_nyquist=0.4,
        rng=42,
    )
    model = pyFDN.dss_to_flamo(
        build.A,
        build.B,
        build.C,
        build.D,
        build.delays,
        build.fs,
        nfft=2**18,
        post_delay=build.post_delay,
        post_output=build.post_output,
    )
    # (1 input, n_samples, 2 outputs) -> (n_samples, 2), the shape pyFDN plots take.
    ir_flamo = pyFDN.flamo_time_response(model)[0]
    print(f"{n} delay lines: {build.delays.astype(int)} samples")
    print(f"{build.B.shape[1]} in, {build.C.shape[0]} out -> {ir_flamo.shape}")
    return fs, ir_flamo, model


@app.cell
def _(model, pyFDN):
    # The graph adac is about to traverse: B -> (delays -> absorption) ~ A -> C, plus D.
    pyFDN.plot_flamo_graph(model, name="FDN")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Extract the parameters

    `flamo_to_json` traverses the FLAMO graph and detaches every parameter into a plain dict. Extraction is map-aware: a matrix with a non-identity parametrisation (orthogonal, Householder, Hadamard) serialises the *effective* matrix the model applies, so what FAUST gets is what FLAMO ran.
    """)
    return


@app.cell
def _(adac, fs, json, mo, model):
    config = adac.flamo_to_json(model, fs, name="PyFDNReverb")
    _pretty = json.dumps(config, indent=1)
    mo.accordion(
        {
            f"JSON intermediate representation ({len(_pretty)} characters)": mo.md(
                f"```json\n{_pretty[:1500]}\n...\n```"
            )
        }
    )
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Certify stability

    Before anything is emitted, `certify` bounds the loop gain around every feedback path: the product of per-element spectral norms, evaluated at the parameter values *as they will be written into the FAUST source* (single precision, ten significant figures).

    Below one at every frequency is a sufficient condition for BIBO stability, so a `certified-stable` verdict rules out a plugin that blows up after the rounding a code generator does.
    """)
    return


@app.cell
def _(adac, config, mo):
    cert = adac.certify(config)
    loop = cert["loops"][0]

    mo.md(f"""
    **Verdict: `{cert["verdict"]}`**

    | quantity | value |
    |---|---|
    | loop gain bound (max over frequency) | {loop["loop_gain_bound_max"]:.4f} |
    | loop gain bound (min over frequency) | {loop["loop_gain_bound_min"]:.4f} |
    | feedback matrix spectral radius | {loop["feedback_spectral_radius"]:.6f} |
    | implied RT60 range | {loop["rt60_estimate_s"]["min"]:.2f}–{loop["rt60_estimate_s"]["max"]:.2f} s |
    | absorption sections stable | {cert["filters"]["all_sections_stable"]} |

    `adac.write_certificate(config, "PyFDNReverb.dsp")` drops this next to the
    generated source as `PyFDNReverb.cert.json`; `export_juce` refuses to build
    an uncertified model unless `strict=False`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Generate FAUST

    Two variants of the same network:

    - **plain** — a faithful translation of the trained model, used for the
      equivalence check in section 6;
    - **with macro controls** — `rt60` and `dry_wet` sliders layered on top.
      These are performance knobs, not trained parameters: the `rt60` slider
      rescales the per-line attenuation against each delay length, leaving the
      feedback matrix and absorption filters untouched.
    """)
    return


@app.cell
def _(adac, config):
    faust_plain = adac.json_to_faust(config)
    faust_code = adac.json_to_faust(config, controls={"rt60": True, "dry_wet": True})
    print(f"plain: {len(faust_plain)} characters, with controls: {len(faust_code)}")
    return faust_code, faust_plain


@app.cell(hide_code=True)
def _(faust_code, mo):
    mo.md(f"```faust\n{faust_code}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The whole reverberator is one `process` line. The FDN core is FAUST's `~` feedback operator wrapped around `delays : absorption`, with the feedback matrix hoisted into a sum-of-products function `fB`. Note the delay lengths:
    `~` inserts one implicit sample of delay, and adac subtracts it from every line so the network keeps the exact delays pyFDN designed.
    """)
    return


@app.cell
def _(Path, faust_code, faust_plain, mo, tempfile):
    work_dir = Path(tempfile.mkdtemp(prefix="pyfdn_faust_"))
    dsp_path = work_dir / "PyFDNReverb.dsp"
    dsp_path.write_text(faust_code)
    (work_dir / "PyFDNReverb_plain.dsp").write_text(faust_plain)

    mo.vstack(
        [
            mo.md(f"Written to `{dsp_path}`"),
            mo.download(
                faust_code.encode(),
                filename="PyFDNReverb.dsp",
                label="Download PyFDNReverb.dsp",
            ),
        ]
    )
    return dsp_path, work_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Run it

    The shortest path to a running artifact needs no toolchain at all: the FAUST web IDE accepts a whole program as a base64 `inline=` parameter, compiles it to WebAssembly in the browser and starts it. The link below carries this exact FDN, sliders and all — open it, allow audio, and the `rt60` and `dry/wet` knobs are live.
    """)
    return


@app.cell
def _(base64, faust_code, mo, urllib):
    ide_url = "https://faustide.grame.fr/?" + urllib.parse.urlencode(
        {
            "autorun": "1",
            "voices": "0",
            "name": "PyFDNReverb",
            "inline": base64.b64encode(faust_code.encode()).decode(),
        }
    )

    mo.md(f"""
    [**Open PyFDNReverb in the FAUST web IDE →**]({ide_url})

    ({len(ide_url)} characters of URL — the entire reverb travels in the link.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Native targets

    With the FAUST distribution installed, the same file goes to C++, a JUCE plugin project, or a standalone app. The cell below runs the compiler if it is on `PATH` and otherwise just lists the commands.
    """)
    return


@app.cell
def _(dsp_path, mo, shutil, subprocess, work_dir):
    _commands = f"""
    cd {work_dir}
    faust -o PyFDNReverb.cpp {dsp_path.name}   # portable C++
    faust2juce -vst3 {dsp_path.name}           # JUCE plugin project
    faust2caqt {dsp_path.name}                 # standalone macOS app
    """

    if shutil.which("faust") is None:
        _blocks = [
            mo.md(
                "`faust` is not on `PATH` — install it from https://faust.grame.fr "
                f"to run:\n```bash\n{_commands}\n```"
            )
        ]
    else:
        _cpp = work_dir / "PyFDNReverb.cpp"
        subprocess.run(
            ["faust", "-o", str(_cpp), str(dsp_path)],
            check=True,
            capture_output=True,
        )
        _head = "\n".join(_cpp.read_text().splitlines()[:20])
        _blocks = [
            mo.md(
                f"`faust -o PyFDNReverb.cpp` produced "
                f"{_cpp.stat().st_size} bytes of C++:"
            ),
            mo.md(f"```cpp\n{_head}\n...\n```"),
        ]

    mo.vstack(_blocks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And a full VST3/AU, built and installed in one call (needs FAUST + JUCE):

    ```python
    adac.export_juce(
        config,
        "exported/",
        name="PyFDNReverb",
        controls={"rt60": True, "dry_wet": True, "pre_delay": True},
        juce_modules="~/JUCE/modules",
        build=True,
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Does the plugin match the design?

    Emitting code is only useful if the compiled DSP is the network pyFDN designed. [DawDreamer](https://github.com/DBraun/DawDreamer) embeds libfaust, so the generated `.dsp` can be compiled and rendered offline right here and compared sample by sample against FLAMO's frequency-domain response.

    Optional: `pip install dawdreamer`. Without it the rest of the notebook still runs; this section reports as skipped.
    """)
    return


@app.cell
def _(importlib, np):
    HAS_DAWDREAMER = importlib.util.find_spec("dawdreamer") is not None

    def render_faust(dsp_file, x, fs):
        """Compile ``dsp_file`` with libfaust and render the mono signal ``x``.

        Returns the output as ``(n_samples, n_channels)`` — for this stereo FDN,
        one column per output channel.
        """
        import dawdreamer

        engine = dawdreamer.RenderEngine(int(fs), 128)
        faust = engine.make_faust_processor("faust")
        faust.set_dsp(str(dsp_file))
        playback = engine.make_playback_processor(
            "input", np.asarray(x, dtype=np.float32)[None, :]
        )
        engine.load_graph([(playback, []), (faust, ["input"])])
        engine.render(len(x) / fs)
        return engine.get_audio().T

    return HAS_DAWDREAMER, render_faust


@app.cell
def _(HAS_DAWDREAMER, fs, ir_flamo, np, render_faust, work_dir):
    if HAS_DAWDREAMER:
        _impulse = np.zeros(len(ir_flamo), dtype=np.float32)
        _impulse[0] = 1.0
        ir_faust = render_faust(work_dir / "PyFDNReverb_plain.dsp", _impulse, fs)
    else:
        ir_faust = None
        print("dawdreamer not installed — skipping the compiled-FAUST comparison")
    return (ir_faust,)


@app.cell
def _(fs, ir_faust, ir_flamo, mo, np, pyFDN):
    if ir_faust is None:
        _blocks = [mo.md("_Install `dawdreamer` to render the compiled FAUST here._")]
    else:
        _m = min(len(ir_flamo), len(ir_faust))
        _a, _b = np.asarray(ir_flamo)[:_m], np.asarray(ir_faust)[:_m]
        # Both are (n_samples, 2): compare the stereo pair channel by channel.
        _err = np.max(np.abs(_a - _b), axis=0)
        _err_db = 20 * np.log10(_err / np.max(np.abs(_a), axis=0))
        _traces = [_a[:, 0], _b[:, 0], _a[:, 1], _b[:, 1]]
        _labels = ["FLAMO L", "FAUST L", "FLAMO R", "FAUST R"]

        _blocks = [
            mo.md(f"""
            Peak difference between the FLAMO reference and the compiled FAUST
            plugin: **{_err[0]:.2e}** left ({_err_db[0]:.1f} dB below the channel
            peak) and **{_err[1]:.2e}** right ({_err_db[1]:.1f} dB), i.e.
            single-precision arithmetic noise plus FLAMO's FFT time-aliasing
            floor. Each channel lies on top of its reference, and so do the
            energy decay curves.
            """),
            pyFDN.plot_impulse_response(
                *_traces,
                fs=fs,
                labels=_labels,
                title="Impulse response: reference vs compiled plugin",
            ),
            pyFDN.plot_edc(
                *_traces,
                fs=fs,
                labels=_labels,
                title="Energy decay curve",
            ),
        ]

    mo.vstack(_blocks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listen

    A dry synth phrase through the FLAMO model and, when DawDreamer is available, through the compiled FAUST plugin. Same network, two runtimes. The phrase is trimmed to what fits in one `nfft` block alongside the 2 s tail, so both runtimes get the same excerpt.
    """)
    return


@app.cell
def _(HAS_DAWDREAMER, fs, mo, model, np, pyFDN, render_faust, work_dir):
    _tail = 2 * fs
    # FLAMO convolves inside a single nfft block, so only nfft minus the tail
    # is usable input. Trim the phrase to that window: otherwise flamo_process
    # silently drops whatever does not fit and the two runtimes are compared on
    # different excerpts.
    _usable = int(model.get_inputLayer().nfft) - _tail
    dry, _ = pyFDN.load_audio("synth_dry.wav", fs=fs)
    dry = np.asanyarray(dry, dtype=np.float32)[:_usable]
    _n = len(dry) + _tail
    wet_flamo = np.asarray(
        pyFDN.flamo_process(model, dry, fs=fs, tail_seconds=_tail / fs)
    )[:_n]

    # mo.audio wants (n_channels, n_samples), the transpose of the plotting shape.
    _players = [
        pyFDN.labeled_audio("Dry", dry, fs=fs),
        pyFDN.labeled_audio("Wet — FLAMO", wet_flamo.T, fs=fs),
    ]

    if HAS_DAWDREAMER:
        # Same trailing silence as flamo_process, so the tail is not cut.
        _padded = np.zeros(_n, dtype=np.float32)
        _padded[: len(dry)] = dry
        wet_faust = render_faust(work_dir / "PyFDNReverb_plain.dsp", _padded, fs)
        _players.append(pyFDN.labeled_audio("Wet — compiled FAUST", wet_faust.T, fs=fs))

    mo.hstack(_players)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where this lands

    The design stays in Python — gallery matrices, absorption fitting, optimisation, the analysis tools in the rest of this gallery — and the deployment artifact is generated, not re-implemented by hand.

    A colourless FDN trained with `pyFDN.train`, or an FDN fitted to a measured RIR by `example_rir_to_fdn`, compiles through exactly the same three calls, because the compiler reads the FLAMO graph rather than any particular design recipe.

    `adac.HotReload` closes the loop further: it republishes the model to a running FAUST plugin during training, so the optimisation is audible while it runs.
    """)
    return


if __name__ == "__main__":
    app.run()
