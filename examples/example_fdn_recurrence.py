# gallery_category: Feedback Matrices
# gallery_description: Run a tiny lossless FDN for millions of samples to see whether it unmixes back into a sparse response, and how the feedback matrix decides that.

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
    # Can an FDN unmix itself?

    A lossless FDN is an orthogonal map on its $M = \sum_i m_i$ delay-line samples: energy is never lost, only shuffled. Poincaré recurrence then says the state must come back arbitrarily close to where it started, infinitely often. If it does, the impulse response should briefly collapse from dense noise back into a few sparse echoes.

    The catch is *how long* that takes. The state moves on a torus spanned by the eigenphases of the $M \times M$ state-transition matrix. Poles that are exact roots of unity are periodic and come back on their own; only the rationally independent rest have to line up by chance, and the waiting time grows roughly like $(1/\varepsilon)^{d}$ in the number $d$ of those free modes. So the question becomes: how many free modes does a given feedback matrix leave?

    Four short delays keep $M = 26$, small enough to diagonalise exactly and evaluate millions of samples in closed form.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.linalg import hadamard, schur

    import pyFDN

    delays = np.array([3, 5, 7, 11])
    n = len(delays)
    horizon = 1_000_000
    return delays, hadamard, horizon, n, np, plt, pyFDN, schur


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it for a million samples

    `pyFDN.dss_to_ss` expands the FDN into its $M$-state transition matrix $\mathbf{T}$. Because $\mathbf{T}$ is orthogonal, its complex Schur form is diagonal: $\mathbf{T} = \mathbf{Z}\boldsymbol{\Lambda}\mathbf{Z}^H$, so the state after $k$ samples is $\mathbf{x}(k) = \mathbf{Z}\boldsymbol{\Lambda}^k\mathbf{Z}^H\mathbf{x}(0)$ — no sample-by-sample loop, and no accumulated rounding error.

    Sparsity of the state is measured by its participation ratio $\|\mathbf{x}\|_2^4 / \|\mathbf{x}\|_4^4$, the effective number of occupied delay-line taps. The impulse enters all four lines equally, so it starts at 4; a fully mixed, Gaussian-like state sits near $M/3 \approx 9$; a value near 1 means all the energy is back in a single tap. A pole counts as *periodic* when $\lambda^k = 1$ for some $k \le 5000$.
    """)
    return


@app.cell
def _(n, np, pyFDN, schur):
    def recurrence(delays, A, horizon, chunk=100_000):
        """Participation ratio, output and periodic poles of a lossless FDN."""
        T, b, c, _ = pyFDN.dss_to_ss(
            delays, A, np.ones((n, 1)) / np.sqrt(n), np.ones((1, n)), np.zeros((1, 1))
        )
        S, Z = schur(T, output="complex")
        lam = np.diag(S)
        modes = Z.conj().T @ b[:, 0]
        pr = np.empty(horizon)
        y = np.empty(horizon)
        for start in range(0, horizon, chunk):
            k = np.arange(start, min(start + chunk, horizon))
            x = (Z @ (modes[:, None] * lam[:, None] ** k)).real
            pr[k] = (x**2).sum(0) ** 2 / (x**4).sum(0)
            y[k] = c[0] @ x
        orders = np.arange(1, 5001)
        hits = np.abs(lam[:, None] ** orders - 1) < 1e-8
        periodic = hits.any(1)
        period = int(np.lcm.reduce(orders[hits[periodic].argmax(1)], initial=1))
        return pr, y, int(periodic.sum()), period

    return (recurrence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hadamard against random orthogonal

    The $4 \times 4$ Hadamard matrix against a batch of Haar-random orthogonal matrices, all on the same delays. The Householder reflection $\mathbf{I} - \tfrac{1}{2}\mathbf{1}\mathbf{1}^\top$ is a useful control: like the Hadamard matrix (which it equals up to signs and a row permutation) its eigenvalues are only $\pm 1$, so if eigenvalue structure of the matrix alone were what mattered, the two should behave alike.
    """)
    return


@app.cell
def _(delays, hadamard, horizon, n, np, pyFDN, recurrence):
    rng = np.random.default_rng(0)
    matrices = {
        "Hadamard": hadamard(n) / np.sqrt(n),
        "Householder": np.eye(n) - 2 / n,
    }
    matrices |= {f"random {i}": pyFDN.random_orthogonal(n, rng) for i in range(12)}
    results = {name: recurrence(delays, A, horizon) for name, A in matrices.items()}
    return matrices, results


@app.cell
def _(delays, mo, np, results):
    _settle = 1000  # skip the initial transient, where the state is still sparse
    mo.ui.table(
        [
            {
                "matrix": name,
                "periodic poles": periodic,
                "free poles": int(delays.sum()) - periodic,
                "common period": period if periodic else None,
                "min participation": round(float(pr[_settle:].min()), 2),
                "at sample": int(np.argmin(pr[_settle:]) + _settle),
            }
            for name, (pr, _y, periodic, period) in results.items()
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Over time

    The running minimum of the participation ratio in blocks of 2000 samples. Every curve falls below the starting value of 4 at some point — partial unmixing happens for any matrix — but only the Hadamard FDN gets close to a single tap, and the Householder FDN barely unmixes at all.
    """)
    return


@app.cell
def _(horizon, np, plt, results):
    _block = 2000
    _t = np.arange(0, horizon, _block) + _block / 2
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _highlight = {"Hadamard": "C3", "Householder": "C0"}
    for _name, (_pr, *_) in results.items():
        _env = _pr.reshape(-1, _block).min(1)
        if _name in _highlight:
            _ax.plot(_t, _env, color=_highlight[_name], lw=1.2, label=_name, zorder=3)
        else:
            _ax.plot(_t, _env, color="0.6", lw=0.5)
    _ax.plot([], [], color="0.6", lw=0.5, label="random orthogonal")
    _ax.axhline(4, color="k", ls="--", lw=0.8, label="initial state")
    _ax.set_xlabel("Time (samples)")
    _ax.set_ylabel("Min participation ratio per block")
    _ax.legend(loc="upper right")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The response at its sparsest

    The output around the Hadamard FDN's sparsest moment, next to a random matrix at *its* sparsest. Sample 0 marks the minimum.
    """)
    return


@app.cell
def _(np, plt, results):
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    for _ax, _name in zip(_axes, ["Hadamard", "random 0"], strict=True):
        _pr, _y, *_ = results[_name]
        _k = int(np.argmin(_pr[1000:]) + 1000)
        _lags = np.arange(-60, 61)
        _ax.stem(_lags, _y[_k + _lags], basefmt=" ")
        _ax.set_title(f"{_name}: around sample {_k} (participation {_pr[_k]:.2f})")
        _ax.set_ylabel("Output")
    _axes[-1].set_xlabel("Samples relative to the sparsest moment")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other delays

One delay set proves little, so the same comparison on three more, over a shorter horizon of 300 000 samples. Each row is one delay set: the minimum participation ratio reached by the Hadamard and Householder FDNs, and the best and median of eight random orthogonal matrices.
    """)
    return


@app.cell
def _(hadamard, matrices, mo, n, np, pyFDN, recurrence):
    _rng = np.random.default_rng(1)
    _rows = []
    for _delays in ([3, 5, 7, 11], [5, 7, 11, 13], [4, 6, 9, 13], [7, 11, 13, 17]):
        _min = {
            _name: recurrence(_delays, matrices[_name], 300_000)[0][1000:].min()
            for _name in ("Hadamard", "Householder")
        }
        _rand = [
            recurrence(_delays, pyFDN.random_orthogonal(n, _rng), 300_000)[0][
                1000:
            ].min()
            for _ in range(8)
        ]
        _rows.append(
            {
                "delays": str(_delays),
                "M": sum(_delays),
                "Hadamard": round(float(_min["Hadamard"]), 2),
                "Householder": round(float(_min["Householder"]), 2),
                "random best": round(float(min(_rand)), 2),
                "random median": round(float(np.median(_rand)), 2),
            }
        )
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Hadamard

    The Hadamard FDN gets sparsest on every delay set, ahead of the best random matrix each time. The structure of the matrix clearly matters. Its eigenvalues are not the reason, though. The Householder matrix has the same $\pm 1$ eigenvalues and the same entry magnitudes, and it unmixes *less* than a typical random matrix. Only the sign pattern differs.

    Two observations point at the mechanism, without settling it:

    - **Focusing.** One sample before the Hadamard FDN's sparsest moment, almost all the energy sits in four taps of nearly equal magnitude reaching the matrix together. The Sylvester Hadamard matrix maps an equal-magnitude vector with the right signs (any of its columns) onto a single line, so a near-return of that pattern collapses into one echo. The Householder matrix only flips the sign of $\mathbf{1}$, its own eigenvector, so the same moment stays four echoes wide.
    - **Periodic poles.** On the first delay set, 12 of the Hadamard FDN's 26 poles are exact roots of unity with a common period of 42 samples. They come back on their own, which leaves fewer free modes that have to line up by chance. That cannot be the whole story: on other delays the Hadamard FDN has few periodic poles and still wins.

    Two caveats keep this a curiosity rather than an audible effect. First, it scales badly: the minima creep up as $M$ grows, even over these few tens of states. A practical FDN has thousands of free modes, and recurrence time grows exponentially with their number. Second, a real FDN is lossy: by the time even this tiny network unmixes, an RT of a second has pushed the response hundreds of decibels below the start.
    """)
    return


if __name__ == "__main__":
    app.run()
