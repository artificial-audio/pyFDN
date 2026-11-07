# gallery_category: Feedback Matrices
# gallery_description: Run a lossless FDN for millions of samples to see whether it unmixes back into a sparse response, and how the feedback matrix and size decide that.

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

    A lossless FDN is an orthogonal map on its $M = \sum_i m_i$ delay-line samples: energy is never lost, only shuffled. A finite-dimensional orthogonal map has recurrent trajectories: the state returns arbitrarily close to its starting point. If it does, the impulse response should briefly collapse from dense noise back into a few sparse echoes.

    The catch is *how long* that takes. The state moves quasi-periodically on a torus spanned by the eigenphases of the $M \times M$ state-transition matrix, and aligning many independent phases can require very long waiting times. Four delay lines with $M = 1000$ states are long enough to watch the mixing happen, and short enough to run for a minute of audio.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.linalg import hadamard

    import pyFDN

    delays = np.array([227, 241, 251, 281])  # M = 1000
    n = len(delays)
    horizon = 2_000_000
    return delays, hadamard, horizon, n, np, plt, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it for two million samples

    The state of the FDN at time $k$ is the last $m_i$ samples written into each delay line $i$. Writing $s_i(k)$ for the signal entering line $i$, the lossless recursion is $\mathbf{s}(k) = \mathbf{A}\,[s_1(k - m_1), \dots, s_N(k - m_N)]^\top$, with the impulse $\mathbf{s}(0) = \mathbf{1}/\sqrt{N}$. No line reads anything younger than the shortest delay, so whole blocks of that length are computed with one matrix product.

    Sparsity of the state is measured by its participation ratio $\|\mathbf{x}\|_2^4 / \|\mathbf{x}\|_4^4$, the effective number of occupied delay-line taps. Both norms are sums of $s_i^2$ and $s_i^4$ over a sliding window of length $m_i$ per line, which cumulative sums give at every sample. The impulse enters all four lines equally, so the ratio starts at 4; a fully mixed, Gaussian-like state sits near $M/3 \approx 333$; a value of 1 means all the energy is in a single tap.
    """)
    return


@app.cell
def _(np):
    def recurrence(delays, A, horizon):
        """Delay-line input signals and state participation ratio of a lossless FDN."""
        delays = np.asarray(delays)
        n = len(delays)
        pad = delays.max()
        s = np.zeros((pad + horizon, n))
        s[pad] = 1 / np.sqrt(n)
        for start in range(pad + 1, pad + horizon, delays.min()):
            stop = min(start + delays.min(), pad + horizon)
            heads = [s[start - m : stop - m, i] for i, m in enumerate(delays)]
            s[start:stop] = np.stack(heads, 1) @ A.T
        t = np.arange(pad, pad + horizon) + 1  # state after writing sample t - 1
        moments = []
        power = s
        for _ in range(2):  # second and fourth moments
            power = np.square(power)
            c = np.cumsum(np.vstack([np.zeros(n), power]), 0)
            moments.append(sum(c[t, i] - c[t - m, i] for i, m in enumerate(delays)))
        e2, e4 = moments
        return s[pad:], e2**2 / e4

    return (recurrence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output is the sum of the four delay-line outputs. It matches `pyFDN.dss_to_impz` over the first 5000 samples, which checks the block recursion.
    """)
    return


@app.cell
def _(delays, hadamard, n, np, pyFDN, recurrence):
    def output(s):
        return sum(np.r_[np.zeros(m), s[:-m, i]] for i, m in enumerate(delays))

    _A = hadamard(n) / np.sqrt(n)
    _ref = pyFDN.dss_to_impz(
        delays,
        _A,
        np.ones((n, 1)) / np.sqrt(n),
        np.ones((1, n)),
        np.zeros((1, 1)),
        5000,
    )[:, 0, 0]
    assert np.allclose(output(recurrence(delays, _A, 5000)[0]), _ref)

    # Check the sliding-window state metric against the processing engine.
    _delays = np.array([3, 5, 7, 11])
    _graph = pyFDN.dss_to_td(
        _delays, _A, np.ones((n, 1)) / np.sqrt(n), np.ones((1, n)), [[0.0]]
    )
    _expected_pr = recurrence(_delays, _A, 64)[1]
    for _k in range(64):
        _graph.process_signal(np.array([float(_k == 0)]))
        _buffer = _graph.ops[0].ops[1].state.buffer
        _energy = np.sum(_buffer**2)
        assert np.isclose(_energy, 1.0)
        assert np.isclose(_expected_pr[_k], _energy**2 / np.sum(_buffer**4))
    return (output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hadamard against random orthogonal

    The $4 \times 4$ Hadamard matrix against a batch of Haar-random orthogonal matrices, all on the same delays. The Householder reflection $\mathbf{I} - \tfrac{1}{2}\mathbf{1}\mathbf{1}^\top$ is a useful control: it equals the Hadamard matrix up to signs and a row permutation, and has the same entry magnitudes. Both have eigenvalues $\pm 1$, but their multiplicities differ: Hadamard has two of each, while Householder has three $+1$ and one $-1$. This is not a comparison at a fixed spectrum.
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
    results = {}
    for _name, _A in matrices.items():
        _s, _pr = recurrence(delays, _A, horizon)
        results[_name] = _pr
        if _name == "Hadamard":
            hadamard_signal = _s
    return hadamard_signal, matrices, results


@app.cell
def _(mo, np, results):
    _settle = 100_000  # past the initial mixing
    mo.ui.table(
        [
            {
                "matrix": name,
                "after 1000": round(float(pr[1000]), 1),
                "after 5000": round(float(pr[5000]), 1),
                "late median": round(float(np.median(pr[_settle:])), 1),
                "late min": round(float(pr[_settle:].min()), 1),
                "at sample": int(np.argmin(pr[_settle:]) + _settle),
            }
            for name, pr in results.items()
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mixing, then nothing

    Left: the first 50 000 samples at full resolution, as the participation ratio climbs from 4 towards the fully mixed plateau. Right: the minimum per block of 4000 samples over the whole run. It fluctuates below the plateau but never comes back down towards 4: over 42 seconds at 48 kHz, no matrix gets below 190 taps.

    The Hadamard FDN behaves differently from the start. It reaches its plateau fastest, within about 5000 samples, but that plateau sits lower: its median of about 300 taps is below every random matrix, so its participation ratio is lower in this run. The Householder FDN mixes almost as fast and then sits right at $M/3$.
    """)
    return


@app.cell
def _(horizon, n, np, plt, results):
    _highlight = {"Hadamard": "C3", "Householder": "C0"}
    _block = 4000
    _fig, (_early, _late) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for _name, _pr in results.items():
        _style = {"color": _highlight.get(_name, "0.6"), "lw": 0.5}
        if _name in _highlight:
            _style |= {"label": _name, "lw": 1.0, "zorder": 3}
        _early.plot(_pr[:50_000], **_style)
        _late.plot(
            np.arange(0, horizon, _block) + _block / 2,
            _pr.reshape(-1, _block).min(1),
            **_style,
        )
    _early.plot([], [], color="0.6", lw=0.5, label="random orthogonal")
    for _ax in (_early, _late):
        _ax.axhline(n, color="k", ls="--", lw=0.8)
        _ax.set_xlabel("Time (samples)")
        _ax.set_yscale("log")
    _early.set_ylabel("Participation ratio (taps)")
    _late.set_ylabel("Min per block")
    _early.legend(loc="lower right")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The response at its sparsest

    The output of the Hadamard FDN at the start, and around its sparsest moment after mixing. Even that moment looks like dense noise.
    """)
    return


@app.cell
def _(hadamard_signal, np, output, plt, results):
    _pr = results["Hadamard"]
    _y = output(hadamard_signal)
    _k = int(np.argmin(_pr[100_000:]) + 100_000)
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    _axes[0].plot(_y[:2000], lw=0.7)
    _axes[0].set_title("Start of the response")
    _axes[1].plot(_y[_k - 1000 : _k + 1000], lw=0.7)
    _axes[1].set_title(f"Around sample {_k} (participation {_pr[_k]:.0f})")
    for _ax in _axes:
        _ax.set_ylabel("Output")
    _axes[-1].set_xlabel("Samples in window")
    _fig.tight_layout()
    _fig.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From 26 to 1000 states

    Small networks do unmix. The same comparison over a range of sizes, one million samples each, shows the deepest dip after mixing as a fraction of the fully mixed $M/3$: Hadamard, Householder, and the best and median of eight random orthogonal matrices.
    """)
    return


@app.cell
def _(matrices, mo, n, np, pyFDN, recurrence):
    _rng = np.random.default_rng(1)
    _rows = []
    for _delays in (
        [3, 5, 7, 11],
        [7, 11, 13, 17],
        [17, 23, 29, 31],
        [53, 59, 67, 71],
        [227, 241, 251, 281],
    ):

        def _dip(A, d=_delays):
            pr = recurrence(d, A, 1_000_000)[1]
            return pr[100 * sum(d) :].min() / (sum(d) / 3)

        _rand = [_dip(pyFDN.random_orthogonal(n, _rng)) for _ in range(8)]
        _rows.append(
            {
                "delays": str(_delays),
                "M": sum(_delays),
                "Hadamard": round(float(_dip(matrices["Hadamard"])), 3),
                "Householder": round(float(_dip(matrices["Householder"])), 3),
                "random best": round(float(min(_rand)), 3),
                "random median": round(float(np.median(_rand)), 3),
            }
        )
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What decides it

    **Size.** With 26 states, the Hadamard FDN gets back to almost a single tap within a million samples (12% of $M/3$, about one tap). Across these five delay sets, the deepest observed dip rises towards the Gaussian reference: by $M = 1000$, no matrix gets below about 58% of it. These finite runs do not establish a recurrence-time bound for larger networks. Absorption further reduces the absolute level of any late recurrence; audibility is not tested here.

    **Matrix.** In these tested delay sets and random draws, the Hadamard FDN dips deepest. Householder has the same entry magnitudes but a different eigenvalue multiplicity, and usually dips less than the median random matrix. This experiment does not separate the effects of sign pattern, feedback spectrum, delay assignment, and the full state-transition spectrum.

    A plausible mechanism, seen at $M = 26$: just before the sparsest moment, the energy sits in four nearly equal taps reaching the matrix together, and the Sylvester Hadamard matrix maps any of its $\pm 1$ columns onto a single line. Householder maps $\mathbf{1}$ to $-\mathbf{1}$, preserving four equal taps for that particular input pattern. Focusing is a hypothesis here; it is not isolated by this comparison.
    """)
    return


if __name__ == "__main__":
    app.run()
