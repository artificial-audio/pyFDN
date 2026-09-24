# gallery_category: Representations
# gallery_description: Convert delay state-space FDN parameters into a conventional state-space model and verify matching impulse responses.

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
    # Delay state-space to state-space

    Delay state-space form describes an FDN with $N$ delay lines as $N$ delay lengths plus an $N \times N$ feedback matrix — a compact description that says nothing about how many states the system actually has. `pyFDN.dss_to_ss` expands it into an ordinary state-space system by giving every sample inside every delay line its own state, so the $N \times N$ feedback matrix becomes a $\sum_i m_i$ square matrix of mostly shift structure.

    That expansion is what lets any standard state-space tool work on an FDN — here `scipy.signal.dimpulse`, but equally a controllability or balancing routine. The price is size: three delays of 13, 19 and 23 samples already need 55 states, and a realistic FDN needs tens of thousands. The delays are kept deliberately tiny for that reason.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A three-delay FDN

    Three short delays with a random orthogonal feedback matrix and random input/output vectors. The per-line attenuation that `fdn_build_gallery` returns as a separate `post_delay` gain is folded into the feedback matrix, so the whole loop is the single matrix `A` that `dss_to_ss` expects.
    """)
    return


@app.cell
def _():
    import dataclasses

    import numpy as np
    from scipy.signal import dimpulse, dlti

    import pyFDN

    np.random.seed(1)
    impulse_response_length = 1000

    delays = np.array([13, 19, 23])
    build = pyFDN.fdn_build_gallery(
        delays=delays,
        io_type="random",
        direct_gain=None,
        rt=0.02,
        rng=1,
    )
    build = dataclasses.replace(build, A=np.diag(build.post_delay[0, 0, :]) @ build.A)
    A, b, c, d = build.A, build.B, build.C, build.D
    return (
        A,
        b,
        build,
        c,
        d,
        delays,
        dimpulse,
        dlti,
        impulse_response_length,
        np,
        pyFDN,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expand and compare

    `dss_to_ss` returns the expanded matrices; `scipy.signal.dimpulse` then treats the FDN as any other discrete-time system. Running the delay recursion with `dss_to_impz` gives the reference.
    """)
    return


@app.cell
def _(A, b, c, d, delays, dimpulse, dlti, impulse_response_length, np, pyFDN):
    aa, bb, cc, dd = pyFDN.dss_to_ss(delays, A, b, c, d)

    system = dlti(aa, bb, cc, dd, dt=1.0)
    _, ir_state_space = dimpulse(system, n=impulse_response_length)
    ir_state_space = np.squeeze(ir_state_space)

    ir_delay_state_space = pyFDN.dss_to_impz(
        impulse_response_length, delays, A, b, c, d
    )
    ir_delay_state_space = np.asarray(ir_delay_state_space).squeeze()

    assert pyFDN.is_almost_zero(ir_state_space - ir_delay_state_space, tol=0.001)

    pyFDN.plot_impulse_response(
        ir_state_space,
        ir_delay_state_space,
        labels=["State space", "Delay state space"],
    )
    return aa, bb, cc, dd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The FDN it started from

    Feedback matrix, delays, input/output vectors and magnitude response of the delay state-space description.
    """)
    return


@app.cell
def _(build, pyFDN):
    pyFDN.plot_FDN_build(build, title="FDN parameters")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The expanded system

    The same FDN as one 55-state system. The feedback matrix is almost entirely the sub-diagonal that shifts each delay line along by one sample; the original $3 \times 3$ mixing survives only in the few entries where a delay line ends and feeds the others.
    """)
    return


@app.cell
def _(aa, bb, cc, dd, pyFDN):
    pyFDN.plot_system_matrix(aa, bb, cc, dd, title="State-space system")
    return


if __name__ == "__main__":
    app.run()
