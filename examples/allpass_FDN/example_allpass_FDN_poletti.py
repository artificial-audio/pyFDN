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
    # Poletti's Allpass FDN (MIMO)

    Example for Poletti's unitary reverberator: a multi-input multi-output (MIMO) allpass feedback delay network with reduced colouration.

    **Reference:** Poletti, M. A. (1995). *A unitary reverberator for reduced colouration in assisted reverberation systems.* INTER-NOISE and NOISE-CON, 5, 1223–1232.

    See also: *Allpass Feedback Delay Networks*, Sebastian J. Schlecht (IEEE Trans. Signal Processing).

    — Original MATLAB: Sebastian J. Schlecht, 26 Dec 2020
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import pyFDN
    from pyFDN.auxiliary.allpass import (
        is_allpass,
        is_paraunitary,
        is_uniallpass,
        poletti_allpass,
    )

    np.random.seed(42)
    return (
        is_allpass,
        is_paraunitary,
        is_uniallpass,
        np,
        plt,
        poletti_allpass,
        pyFDN,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build Poletti allpass FDN

    Use a random orthogonal matrix **U** and gain **g** to form the delay state-space **(A, B, C, D)**.
    """)
    return


@app.cell
def _(np, poletti_allpass, pyFDN):
    N = 4
    U = pyFDN.random_orthogonal(N)
    Fs = 48000
    rt = 1
    delays = np.random.randint(200, 1000, size=N)
    g = pyFDN.rt_to_gain_per_sample(rt, Fs)

    average_gain = g ** np.mean(delays)
    A, B, C, D = poletti_allpass(average_gain, U)
    return A, B, C, D, Fs, N, delays, rt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the system matrix
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    _fig = pyFDN.plot_system_matrix(A, B, C, D)
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: uniallpass

    Check that the FDN is uniallpass (lossless with diagonal Lyapunov matrix).
    """)
    return


@app.cell
def _(A, B, C, D, is_uniallpass):
    _is_a, P = is_uniallpass(A, B, C, D)
    assert _is_a, "Expected uniallpass"
    print("Uniallpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: determinant allpass

    Check that the determinant transfer function is allpass (numerator = reversed denominator).
    """)
    return


@app.cell
def _(A, B, C, D, N, is_allpass, np):
    test_delays = 2 ** np.arange(N)
    _is_a, den, num = is_allpass(A, B, C, D, test_delays)
    assert _is_a, "Expected allpass"
    print("Allpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: impulse response is paraunitary

    Compute the MIMO impulse response and verify it is paraunitary (lossless).
    """)
    return


@app.cell
def _(A, B, C, D, Fs, delays, is_paraunitary, pyFDN, rt):
    ir_len = int(rt * Fs * 5)
    impulse_response = pyFDN.dss_to_impz(ir_len, delays, A, B, C, D)
    # Shape: (ir_len, n_out, n_in)

    is_p, test_matrix, max_off = is_paraunitary(impulse_response)
    assert is_p, "Expected paraunitary impulse response"
    print("Paraunitary: OK")
    print("Max off-diagonal in correlation matrix:", max_off)
    return impulse_response, ir_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Impulse response matrix

    Plot each input→output channel impulse response in a grid (rows = outputs, cols = inputs). One consequence of the allpass design here is that the direct sound dominates on the main diagonal.
    """)
    return


@app.cell
def _(Fs, impulse_response, ir_len, np, plt, pyFDN):
    fig, _, _ = pyFDN.plot_impulse_response_matrix(
        np.arange(ir_len),
        pyFDN.mulaw_encode(impulse_response),
        xlabel="Time [samples]",
        ylabel="Amplitude [mu-law]",
        title="Poletti allpass FDN — MIMO impulse response",
        xlim=(-1000, Fs / 2),
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrogram (one channel)

    Spectrogram of a single input→output channel (e.g. output 0, input 0). The modal decay is not homogeneous in the Poletti allpass which leads to uneven decay.
    """)
    return


@app.cell
def _(Fs, impulse_response, mo, pyFDN):
    # Pick one channel: output 1, input 0
    channel_ir = impulse_response[:, 1, 0]
    _fig = pyFDN.plot_spectrogram(channel_ir, Fs, xlim=(0, 2))
    _fig.show()

    mo.vstack([mo.audio(channel_ir, Fs)])
    return


if __name__ == "__main__":
    app.run()
