# gallery_category: Feedback Matrices
# gallery_description: A tiny lossless FDN spreads an impulse over its delay lines, then briefly concentrates it into almost one tap again.

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.linalg import hadamard

    import pyFDN

    return hadamard, mo, np, plt, pyFDN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A lossless FDN can unmix itself

    Here is a small curiosity: four delay lines of lengths $[3, 5, 7, 11]$, coupled by a normalized Hadamard matrix. An impulse initially occupies four taps, spreads through the 26 stored samples, and later concentrates into almost one tap again.

    We measure the effective number of occupied taps with the participation ratio $\|x\|_2^4 / \|x\|_4^4$. A single occupied tap gives 1; equal energy in four taps gives 4. Energy stays constant throughout: the low values below show concentration, not decay. This illustrates what is possible in a tiny lossless network; it makes no claim about recurrence times or audibility in practical reverberators.
    """)
    return


@app.cell
def _(hadamard, np, pyFDN):
    delays = np.array([3, 5, 7, 11])
    horizon = 20_000
    _graph = pyFDN.dss_to_td(
        delays, hadamard(4) / 2, np.ones((4, 1)) / 2, np.ones((1, 4)), [[0.0]]
    )
    _state = _graph.ops[0].ops[1].state
    states = np.empty((horizon, delays.sum()))
    for _k in range(horizon):
        _graph.process_signal(np.array([float(_k == 0)]))
        states[_k] = np.concatenate(
            [_state.buffer[i, :m] for i, m in enumerate(delays)]
        )
    energy = np.sum(states**2, axis=1)
    participation = energy**2 / np.sum(states**4, axis=1)
    sparse_at = 1000 + int(np.argmin(participation[1000:]))

    assert np.allclose(energy, 1.0)
    assert np.isclose(participation[0], 4.0)
    assert participation.max() > 10
    assert participation[sparse_at] < 1.5
    return participation, sparse_at, states


@app.cell
def _(participation, plt, sparse_at):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.plot(participation, lw=0.6)
    _ax.scatter(sparse_at, participation[sparse_at], color="C3", zorder=3)
    _ax.set(xlabel="Time (samples)", ylabel="Effective occupied taps")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, participation, plt, sparse_at, states):
    _mixed_at = int(np.argmax(participation[:1000]))
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for _ax, _k, _title in zip(
        _axes,
        [0, _mixed_at, sparse_at],
        ["Initial", "Spread out", "Concentrated again"],
        strict=False,
    ):
        _ax.bar(np.arange(states.shape[1]), states[_k] ** 2)
        _ax.set_title(f"{_title}\nSample {_k}, participation {participation[_k]:.2f}")
        _ax.set_xlabel("Stored tap")
    _axes[0].set_ylabel("Tap energy")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
