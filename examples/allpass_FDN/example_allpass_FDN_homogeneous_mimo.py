import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homogeneous allpass FDN (MIMO)

    Example for an allpass FDN with **homogeneous decay** so that all poles have the same decay rate. Compared to the SISO case, the MIMO has considerably more degrees of freedom.

    See *Allpass Feedback Delay Networks*, Sebastian J. Schlecht (IEEE Trans. Signal Processing).

    — Original MATLAB: Sebastian J. Schlecht, 10 June 2020
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
    import numpy as np
    import pyFDN

    np.random.seed(1)
    return np, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build homogeneous allpass FDN

    Random delays, gain matrix **G**, random mixing matrix **U**, combined to the feedback matrix **A**. The remaining coefficients are reconstructed by completing the allpass FDN.
    """)
    return


@app.cell
def _(np, pyFDN):
    Fs = 48000
    N = 8
    numio = N

    delays = np.random.randint(800, 1800, size=N)  # delays in samples, 1..30
    g = pyFDN.rt_to_gain_per_sample(0.6, Fs)
    G = np.diag(g**delays)  # gain matrix
    U = pyFDN.random_orthogonal(N)
    A = G @ U

    B, C, D, X = pyFDN.complete_fdn(A, N, str(numio))
    return A, B, C, D, Fs, delays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test: uniallpass

    Check that the system is uniallpass, i.e., allpass for any delays.
    """)
    return


@app.cell
def _(A, B, C, D, pyFDN):
    is_a, _ = pyFDN.is_uniallpass(A, B, C, D, tol=1e-7)
    assert is_a, "Expected allpass for homogeneous FDN with these delays"
    print("is_allpass: OK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot system matrix

    Visualize system matrix as 2×2 block heatmaps.
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
    ## Impulse response

    Compute the impulse response with **dss_to_impz** (MIMO).
    """)
    return


@app.cell
def _(A, B, C, D, Fs, delays, pyFDN):
    ir_len = Fs  # 2 seconds
    impulse_response = pyFDN.dss_to_impz(ir_len, delays, A, B, C, D)
    return impulse_response, ir_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time domain and play

    Plot the impulse response in the time domain and use the audio widget to play it.
    """)
    return


@app.cell
def _(Fs, impulse_response, ir_len, mo, np):
    import matplotlib.pyplot as plt

    ir_channel = impulse_response[:, 2, 1]
    t = np.arange(ir_len) / Fs
    _fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, ir_channel, color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Homogeneous allpass FDN — impulse response (time domain)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    mo.vstack([mo.audio(ir_channel, Fs)])
    return


if __name__ == "__main__":
    app.run()
