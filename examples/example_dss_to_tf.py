# gallery_category: Representations
# gallery_description: Convert a delay state-space FDN into a matrix transfer function and verify the result in the time domain.

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
    # Delay state-space to transfer function

    An FDN in delay state-space form keeps the delays separate from the feedback matrix. `dss_to_tf` collapses that structure into a matrix transfer function: a shared denominator polynomial `tfA` — the loop's characteristic polynomial — and a numerator polynomial `tfB` per input/output pair.

    The two forms describe the same system, so they must produce the same impulse response. Rendering the delay recursion directly (`dss_to_impz`) and evaluating the polynomials (`mtf_to_impz`) is the check, and it is exact to machine precision because no approximation is involved — only a change of representation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A MIMO FDN

    Four delay lines with a random orthogonal feedback matrix, three inputs and two outputs, so the transfer function is a 2x3 matrix of polynomials over one common denominator.
    """)
    return


@app.cell
def _():
    import numpy as np

    import pyFDN

    np.random.seed(5)
    fs = 48000
    impulse_response_length = fs // 100

    build = pyFDN.fdn_build_gallery(
        4,
        fs=fs,
        delay_range=(50, 101),
        num_inputs=3,
        num_outputs=2,
        io_type="identity",
        direct_gain=None,
        rt=None,
        rng=5,
    )
    A, B, C, D, delays = (
        build.A,
        build.B,
        build.C,
        build.D,
        build.delays,
    )
    return A, B, C, D, delays, impulse_response_length, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Through the transfer function

    `dss_to_tf` returns `(tfB, tfA)` in ascending powers of $z^{-1}$; `mtf_to_impz` runs the resulting polynomial ratio out to `impulse_response_length` samples.
    """)
    return


@app.cell
def _(A, B, C, D, delays, impulse_response_length, pyFDN):
    tfB, tfA = pyFDN.dss_to_tf(delays, A, B, C, D)
    ir_tf = pyFDN.mtf_to_impz(tfB, tfA, impulse_response_length)
    return (ir_tf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Through the delay recursion

    The same system run as a time-domain recursion over the delay lines, which is the reference the transfer function has to reproduce.
    """)
    return


@app.cell
def _(A, B, C, D, delays, impulse_response_length, pyFDN):
    ir_dss = pyFDN.dss_to_impz(impulse_response_length, delays, A, B, C, D)
    return (ir_dss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Both impulse response matrices

    One panel per input/output pair, transfer function above and delay recursion below. The two grids are indistinguishable.
    """)
    return


@app.cell
def _(ir_dss, ir_tf, mo, pyFDN):
    fig1 = pyFDN.plot_impulse_response_matrix(t=None, ir=ir_tf)
    fig2 = pyFDN.plot_impulse_response_matrix(t=None, ir=ir_dss)

    mo.vstack([fig1, fig2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification

    Indistinguishable by eye is not a check; this one is to $10^{-10}$.
    """)
    return


@app.cell
def _(ir_dss, ir_tf, pyFDN):
    assert pyFDN.is_almost_zero(ir_dss - ir_tf, tol=1e-10), (
        "IR from TF and DSS should match"
    )
    return


if __name__ == "__main__":
    app.run()
