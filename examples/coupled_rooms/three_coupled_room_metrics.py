"""Standalone marimo notebook for three-room GFDN sweeps, plots and metrics.

Run with ``marimo edit examples/coupled_rooms/three_coupled_room_metrics.py``.
The original physics notebook retains its single-configuration analyses.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import json
    import tempfile
    import soundfile as sf
    import matplotlib.pyplot as plt
    import pyFDN
    import csv
    import io

    from numpy.typing import NDArray, ArrayLike
    from scipy.linalg import block_diag, expm
    from scipy.signal import resample_poly
    from pathlib import Path

    from pyFDN.build import FDNBuild
    from pyFDN.auxiliary.physics_based_coupling import (
        create_lossless_coupling_matrix, create_state_transition_matrix,
        trajectory, match_delay_totals_to_volumes, get_coupling_angles, get_coupling_matrix,
        get_feedback_matrix, get_room_time_varying_matrix, gfdn_energy_ledger,
    )


    return (
        ArrayLike,
        NDArray,
        Path,
        FDNBuild,
        block_diag,
        create_lossless_coupling_matrix,
        create_state_transition_matrix,
        csv,
        expm,
        match_delay_totals_to_volumes,
        get_coupling_angles,
        get_coupling_matrix,
        get_feedback_matrix,
        gfdn_energy_ledger,
        io,
        json,
        mo,
        np,
        plt,
        pyFDN,
        resample_poly,
        get_room_time_varying_matrix,
        sf,
        tempfile,
        trajectory,
    )


@app.cell
def _(get_room_time_varying_matrix, np):
    def make_room_modulation(coupling, lines_per_group, fs, random_state):
        """Reuse identical rotation parameters for different fixed couplings."""
        saved_rng = np.random.get_state()
        try:
            np.random.set_state(random_state)
            return get_room_time_varying_matrix(coupling, lines_per_group, fs=fs)
        finally:
            np.random.set_state(saved_rng)
    return (make_room_modulation,)


@app.cell
def _(mo):
    V1 = mo.ui.number(30, 400, value=160, label="Room 1 volume (m³)")
    V2 = mo.ui.number(30, 400, value=80, label="Room 2 volume (m³)")
    V3 = mo.ui.number(30, 400, value=120, label="Room 3 volume (m³)")
    S12 = mo.ui.number(0.1, 6, value=2.0, step=0.1, label="Aperture 1–2 (m²)")
    S23 = mo.ui.number(0.1, 6, value=1.5, step=0.1, label="Aperture 2–3 (m²)")
    S13 = mo.ui.number(0.1, 6, value=0.5, step=0.1, label="Aperture 1–3, complete topology (m²)")
    dur = mo.ui.number(1, 60, value=5, label="Histogram sweep duration (s)")
    run_sweeps = mo.ui.run_button(label="Generate histogram and distribution plots")
    mo.vstack([
        mo.md(r"""
        # Three-room GFDN metrics and parameter sweeps

        $M=\operatorname{blkdiag}(M_1,M_2,M_3)$ contains the orthogonal room mixers.
        $F=\Phi\otimes I_K$ is fixed coupling and $G$ is diagonal per-line
        amplitude absorption. Each build contains `A = F @ M @ G` (lossless: $G=I$).
        The `get_room_time_varying_matrix` hook $F R(n)F^\top$ changes the effective
        feedback to $F R(n)M G$. Thus only $M_i(n)=R_i(n)M_i$ varies.
        All audio and sweep renders below use `pyFDN.process_fdn`.
        """),
        V1, V2, V3, S12, S23, S13, dur, run_sweeps,
        mo.md("Plots are saved in the sibling `GroupedFDN/figures` directory. "
              "The comparison table below has its own duration and run button."),
    ])
    return S12, S13, S23, V1, V2, V3, dur, run_sweeps


@app.cell
def _(
    S12,
    S13,
    S23,
    V1,
    V2,
    V3,
    create_lossless_coupling_matrix,
    create_state_transition_matrix,
    dur,
    np,
    pyFDN,
    trajectory,
):
    fs = 48000
    num_rooms = 3
    V = np.array([V1.value, V2.value, V3.value])
    _chain = np.array([[0, S12.value, 0], [S12.value, 0, S23.value], [0, S23.value, 0]])
    _complete = _chain.copy()
    _complete[0, 2] = _complete[2, 0] = S13.value
    beta_chain = create_lossless_coupling_matrix(_chain, V)
    beta_complete = create_lossless_coupling_matrix(_complete, V)
    _source = np.array([1., 0., 0.])
    _, Eref_chain = trajectory(
        create_state_transition_matrix(beta_chain), _source, np.eye(3),
        fs, int(dur.value * fs))
    _, Eref_complete = trajectory(
        create_state_transition_matrix(beta_complete), _source, np.eye(3),
        fs, int(dur.value * fs))

    def sample_delay_line_lengths(avg_tau, num_del_per_group):
        return np.concatenate([
            pyFDN.sample_delay_lengths(
                num_del_per_group,
                (int(0.5 * avg_tau * volume / V[0]),
                 int(1.5 * avg_tau * volume / V[0])),
                coprime=True, rng=331 + 10 * room,
            )
            for room, volume in enumerate(V)
        ]).astype(int)

    return (
        Eref_chain,
        Eref_complete,
        V,
        beta_chain,
        beta_complete,
        fs,
        num_rooms,
        sample_delay_line_lengths,
    )


@app.cell
def _(np):
    def db(values, is_energy_signal=True):
        logarithm = np.log10(np.abs(values) + 1e-10)
        return 10 * logarithm if is_energy_signal else 20 * logarithm


    return (db,)


@app.cell
def _(Path, db, np, num_rooms, plt):
    def delay_line_kl_divergence(_E_ss) -> float:
        """
        KL divergence of the delay-line steady-state energy
        distribution from the uniform distribution.
        """

        _p = _E_ss / np.sum(_E_ss)
        _N = len(_p)

        _D_KL = np.sum(
            _p * np.log(_p * _N)
        )

        return _D_KL

    def plot_gfdn_delay_element_distributions(
        _Y,
        _delays,
        _Nroom,
        _steady_state_start:int,
        _steady_state_stop:int,
    ):
        """
        Plot the distribution of mean delay-element energies.
        """

        # squared delay-line states
        _S2 = _Y**2

        # Select steady-state samples
        _S2_ss = _S2[_steady_state_start:_steady_state_stop, :]

        # Average over input/realisation and time
        _E_ss = _S2_ss.mean(axis=0)
        _E_ss_per_group = np.zeros(num_rooms, dtype=np.float32)
        for _k in range(num_rooms):
            _E_ss_per_group[_k] = _E_ss[_k*_Nroom:(_k+1)*_Nroom].mean()
        _E_ss_group_variance = np.std(_E_ss_per_group)/ np.mean(_E_ss_per_group)


        _fig, _ax = plt.subplots(figsize=(6, 4))

        _ax.bar(
            np.arange(len(_E_ss)),
            _E_ss,
        )

        kl_div = delay_line_kl_divergence(_E_ss)

        _ax.axhline(
            1.0 / _delays.sum(),
            linestyle="--",
            color='k',
            label=r"equipartition $p^2 = 1 / \sum_j m_j$",
        )

        _ax.set_xlabel("Delay line $j$")
        _ax.set_ylabel(r"$\mathbb{{E}}[s^{j}(m_j + \infty)^2]$")
        _ax.set_title(rf"$D_{{KL}}(\mathrm{{unif.}}= {kl_div:.4f}, \text{{Var}} \left(\mathbb{{E}}(\sum_{{i \in \text{{FDN_i}}}}s_i^2(\infty)) \right) = {db(_E_ss_group_variance):.4f}$")
        _ax.legend()

        return _E_ss, _ax, _fig


    fig_path = Path(__file__).resolve().parents[3] / 'GroupedFDN' / 'figures'
    return fig_path, plot_gfdn_delay_element_distributions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loop over $K_i$ and $\tau_i$ and plot the delay line energy distribution and histograms
    """)
    return


@app.cell
def _(
    ArrayLike,
    NDArray,
    FDNBuild,
    make_room_modulation,
    beta_chain,
    beta_complete,
    block_diag,
    db,
    dur,
    expm,
    fs,
    get_coupling_angles,
    get_coupling_matrix,
    get_feedback_matrix,
    gfdn_energy_ledger,
    np,
    num_rooms,
    pyFDN,
    sample_delay_line_lengths,
):
    def run_gfdn_for_Nroom(_avg_tau: float, _Nroom: int, use_chain_topology:bool=True) -> NDArray:
        """Run GFDN for different delay line lengths and numbers"""

        N1 = N2 = N3 = _Nroom
        Ntot = num_rooms * _Nroom

        # ------------------------------------------------------------
        # Delay lengths
        # -----------------------------------------------------------
        delays = sample_delay_line_lengths(_avg_tau, _Nroom)

        M = np.array([
            delays[:N1].sum(),
            delays[N1:N1 + N2].sum(),
            delays[N1 + N2:].sum(),
        ])

        dt_i = M / (_Nroom * fs)

        # ------------------------------------------------------------
        # Coupling matrix
        # ------------------------------------------------------------
        if use_chain_topology:
            theta = get_coupling_angles(beta_chain, dt_i)
        else:
            theta = get_coupling_angles(beta_complete, dt_i)
        K = get_coupling_matrix(theta)
        R_room = expm(K)

        # ------------------------------------------------------------
        # Internal room mixers
        # ------------------------------------------------------------
        np.random.seed(1)

        Qblocks = block_diag(
            pyFDN.random_orthogonal(N1),
            pyFDN.random_orthogonal(N2),
            pyFDN.random_orthogonal(N3),
        )

        A = get_feedback_matrix(
            R_room,
            Qblocks,
            _Nroom,
        )

        # ------------------------------------------------------------
        # Input
        # ------------------------------------------------------------
        B = np.zeros((Ntot, num_rooms))

        B[:N1, 0] = 1.0 / np.sqrt(N1)
        B[N1:N1 + N2, 1] = 1.0 / np.sqrt(N2)
        B[N1 + N2:, 2] = 1.0 / np.sqrt(N3)

        C_lines = np.eye(Ntot)
        n_samp = int(dur.value * fs)

        # A = F @ M; the hook inserts room rotations before the fixed F.
        modulation = make_room_modulation(R_room, _Nroom, fs, np.random.get_state())
        build = FDNBuild(
            A=A, B=B, C=C_lines, D=np.zeros((Ntot, num_rooms)),
            delays=delays, fs=fs, post_matrix=modulation)
        Y = np.zeros((num_rooms, n_samp, Ntot))
        for source in range(num_rooms):
            signal = np.zeros((n_samp, num_rooms))
            signal[0, source] = 1
            modulation.reset()
            Y[source] = pyFDN.process_fdn(signal, build)

        return Y, delays

    def get_energy_error(_Y:NDArray, _delays: ArrayLike, _Nroom:int,
                         _tau_avg:float,  Eref: NDArray, _src:int=0):
        """Calculate the MAE between the GFDN trajectory and Markov trajectory (Eref)"""

        _E_ex, _n_ex = gfdn_energy_ledger(_Y, num_rooms, _Nroom, _delays)
        _tail = slice(int(0.85 * _n_ex), _n_ex)

        _transient_ind = slice(0, min(int(0.2 * fs), _n_ex))
        _err_transient = _E_ex[_src, _transient_ind] - Eref[_transient_ind]
        _mae_transient = 100 * np.mean(np.abs(_err_transient), axis=0)

        _err = _E_ex[_src] - Eref[:_n_ex]
        _err_tail = _E_ex[_src, _tail] - Eref[_tail]
        _mae = 100 * np.mean(np.abs(_err), axis=0)
        _mae_tail = 100 * np.mean(np.abs(_err_tail), axis=0)

        print(rf"Error for $K_i$={_Nroom}, $\bar{{m}}$={_tau_avg}: Total MAE={np.round(_mae,3)} pp, Late MAE={np.round(_mae_tail, 3)}, Transient MAE=_{np.round(_mae_transient, 3)}")
        return (_mae, _mae_tail, _mae_transient)


    return get_energy_error, run_gfdn_for_Nroom


@app.cell
def _(
    Eref_chain,
    Eref_complete,
    db,
    fig_path,
    get_energy_error,
    mo,
    np,
    plot_gfdn_delay_element_distributions,
    plt,
    run_gfdn_for_Nroom,
    run_sweeps,
):
    mo.stop(not run_sweeps.value, mo.md("Run the histogram sweeps to generate plots."))
    fig_path.mkdir(parents=True, exist_ok=True)
    Nroom_values = [8, 16, 24, 32]
    avg_delay_values = [1000, 2000, 4000, 8000]
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _distribution_plots = []
    use_chain_topology = True
    E_ref = Eref_chain if use_chain_topology else Eref_complete

    for _i, _Nroom in enumerate(Nroom_values):

        print(f"Running Nroom = {_Nroom}...")

        _Y, _delays = run_gfdn_for_Nroom(2000, _Nroom, use_chain_topology=use_chain_topology)
        _error = get_energy_error(_Y, _delays, _Nroom, 2000, E_ref)

        # --------------------------------------------------------
        # Use the valid portion of the signal
        # --------------------------------------------------------
        _n_ex = _Y.shape[1] - _delays.max() - 1
        _, _distribution_ax, _distribution_fig = plot_gfdn_delay_element_distributions(
            _Y[0], _delays, _Nroom, int(0.85 * _n_ex), _n_ex)
        _distribution_fig.savefig(
            fig_path / f'delay_line_energy_dist_del_len=2000_Ngrp={_Nroom}_tv.png',
            dpi=300)
        _distribution_plots.append(mo.mpl.interactive(_distribution_fig))
        _ind_slice = np.arange(int(0.85 * _n_ex), _n_ex, dtype=np.int32)
        _S2_ss = _Y[0, _ind_slice, :]**2

        _E_ss = _S2_ss.flatten()

        _Ntot = _Y.shape[-1]
        _E_expected = _S2_ss.sum()
        _to_plot = _E_ss / _E_expected

        # --------------------------------------------------------
        # Histogram
        # --------------------------------------------------------
        _counts, _bins = np.histogram(
            _to_plot,
            bins=50000,
            range=(0, 0.1),
            density=True,
        )

        _ax.stairs(
            np.log10(_counts + np.finfo(np.float32).eps),
            _bins,
            label=rf"$K_i={_Nroom}, \sigma = {db(np.std(_to_plot)):.3f}$ dB",
        )


    _ax.set_xlabel(
        r"Normalised delay-line energy"
    )
    _ax.set_ylabel("Num delay elements (log10)")
    _ax.set_xlim(0, 5*1e-5)
    _ax.set_ylim(0, 6)
    _ax.legend()

    _fig.tight_layout()

    _fig.savefig(fig_path / 'delay_line_energy_histogram_del_len=2000_Ngrp_sweep_tv.png', dpi=300)
    mo.vstack([mo.mpl.interactive(_fig), *_distribution_plots])
    return E_ref, Nroom_values, avg_delay_values, use_chain_topology


@app.cell
def _(
    E_ref,
    Nroom_values,
    avg_delay_values,
    db,
    fig_path,
    fs,
    get_energy_error,
    mo,
    np,
    plot_gfdn_delay_element_distributions,
    plt,
    run_gfdn_for_Nroom,
    use_chain_topology,
):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _distribution_plots = []

    for _i, _avg_tau in enumerate(avg_delay_values):

        print(f"Running delay line length = {_avg_tau}...")

        _Y, _delays = run_gfdn_for_Nroom(_avg_tau, Nroom_values[-2], use_chain_topology=use_chain_topology)
        _error = get_energy_error(_Y, _delays, Nroom_values[-2], _avg_tau, E_ref)


        # --------------------------------------------------------
        # Use the valid portion of the signal
        # --------------------------------------------------------
        _n_ex = _Y.shape[1] - _delays.max() - 1
        _, _distribution_ax, _distribution_fig = plot_gfdn_delay_element_distributions(
            _Y[0], _delays, Nroom_values[-2], int(0.85 * _n_ex), _n_ex)
        _distribution_fig.savefig(
            fig_path / f'delay_line_energy_dist_del_len={_avg_tau}_Ngrp={Nroom_values[-2]}_tv.png',
            dpi=300)
        _distribution_plots.append(mo.mpl.interactive(_distribution_fig))
        _ind_slice = np.arange(int(0.85 * _n_ex), _n_ex, dtype=np.int32)
        _S2_ss = _Y[0, _ind_slice, :]**2

        _E_ss = _S2_ss.flatten()

        _Ntot = _Y.shape[-1]
        _E_expected = _S2_ss.sum()
        _to_plot = _E_ss / _E_expected

        # --------------------------------------------------------
        # Histogram
        # --------------------------------------------------------
        _counts, _bins = np.histogram(
            _to_plot,
            bins=50000,
            range=(0, 0.1),
            density=True,
        )

        _ax.stairs(
            np.log10(_counts + np.finfo(np.float32).eps),
            _bins,
            label=rf"$\bar{{m}}={_avg_tau / fs * 1e3:.3f}ms, \sigma = {db(np.std(_to_plot)):.3f}$ dB",
        )

    _ax.set_xlabel(
        r"Normalised delay-line energy"
    )
    _ax.set_ylabel("Num delay elements (log10)")
    _ax.set_xlim(0, 5*1e-5)
    _ax.set_ylim(0, 6)
    _ax.legend()

    _fig.tight_layout()

    _fig.savefig(fig_path / f'delay_line_energy_histogram_Ngrp={Nroom_values[-2]}_avg_delay_line_length_sweep.png', dpi=300)
    mo.vstack([mo.mpl.interactive(_fig), *_distribution_plots])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate a table of steady state and transient MAE as a function of $K_i$ and $\tau_$ for 1) fixed feedback matrix 2) time varying feedback matrix 3) co-prime delay line lengths 4) exact delay line lengths (matches volume ratio exactly)
    """)
    return


@app.cell
def _(
    FDNBuild,
    make_room_modulation,
    match_delay_totals_to_volumes,
    block_diag,
    create_state_transition_matrix,
    expm,
    get_coupling_angles,
    get_coupling_matrix,
    np,
    pyFDN,
):

    def measure_lossless_energy_errors(
        line_outputs, line_delays, lines_per_group, sa_reference, volumes, fs,
    ):
        """Return linear-energy MAEs in percentage points, plus signed biases.

        Unit-energy impulse, source in room 1, C=I. Reconstruct the same
        forward-window cell-energy ledger used elsewhere in this notebook.
        Use the last 15% of valid samples, taking absolute values BEFORE
        averaging over time and rooms. This is not an error in decibels.
        """
        line_delays = np.asarray(line_delays, dtype=int)
        num_groups = len(volumes)
        valid_stop = len(line_outputs) - int(line_delays.max()) - 1
        if valid_stop <= 0:
            raise ValueError("Simulation must exceed the longest delay.")
        cs = np.vstack([
            np.zeros((1, len(line_delays))),
            np.cumsum(line_outputs ** 2, axis=0),
        ])
        ledger = np.zeros((valid_stop, num_groups))
        indices = np.arange(valid_stop)
        for j, delay in enumerate(line_delays):
            ledger[:, j // lines_per_group] += (
                cs[indices + 1 + delay, j] - cs[indices + 1, j])
        late = slice(int(0.85 * valid_stop), valid_stop)
        totals = line_delays.reshape(num_groups, lines_per_group).sum(axis=1)
        equipartition = totals / totals.sum()
        volume_target = np.asarray(volumes) / np.sum(volumes)
        sa_difference = ledger[late] - sa_reference[late]
        equip_difference = ledger[late] - equipartition
        transient_stop = min(int(0.2 * fs), valid_stop)
        densities = ledger[late].mean(axis=0) / totals
        return {
            "sa_mae_pp": float(100 * np.abs(sa_difference).mean()),
            "equipartition_mae_pp": float(100 * np.abs(equip_difference).mean()),
            "allocation_mae_pp": float(100 * np.abs(equipartition - volume_target).mean()),
            "sa_mean_bias_pp": float(100 * np.abs(sa_difference.mean(axis=0)).mean()),
            "equipartition_mean_bias_pp": float(100 * np.abs(equip_difference.mean(axis=0)).mean()),
            "transient_mae_pp": float(100 * np.abs(
                ledger[:transient_stop] - sa_reference[:transient_stop]).mean()),
            "density_cv_percent": float(100 * np.std(densities) / np.mean(densities)),
            "energy_conservation_error": float(np.max(np.abs(ledger.sum(axis=1) - 1))),
        }

    def run_steady_state_comparison(
        volumes, topology_betas, sample_delays, fs, duration=10.0,
        matrix_seed=1, configurations=None, progress=None,
    ):
        """Paired fixed/modulated sweeps for coprime and exact-volume delays.

        Fresh modulation starts at n=0 for every render. Both delay designs
        use identical internal mixers and modulation parameters. Coupling is
        recalculated from each design's actual mean delay lengths.
        """
        if configurations is None:
            configurations = [(k, 10000) for k in (8, 16, 24, 32)] + [
                (32, length) for length in (1000, 2000, 4000, 8000)]
        n_samples = int(duration * fs)
        num_groups = len(volumes)
        source = np.eye(num_groups)[0]
        results = []
        for topology, beta in topology_betas.items():
            generator = create_state_transition_matrix(beta)
            eigenvalues, eigenvectors = np.linalg.eig(generator)
            coefficients = np.linalg.solve(eigenvectors, source)
            reference = ((np.exp(
                np.arange(n_samples)[:, None] / fs * eigenvalues
            ) * coefficients) @ eigenvectors.T).real
            for lines_per_group, length_setting in configurations:
                base = sample_delays(length_setting, lines_per_group)
                exact = match_delay_totals_to_volumes(base, lines_per_group, volumes)
                channels = num_groups * lines_per_group
                # Match the existing sweep's legacy RNG sequence without
                # changing the RNG state of other notebook cells.
                saved_rng = np.random.get_state()
                try:
                    np.random.seed(matrix_seed)
                    mixers = block_diag(*[
                        pyFDN.random_orthogonal(lines_per_group)
                        for _ in range(num_groups)])
                    modulation_rng = np.random.get_state()
                finally:
                    np.random.set_state(saved_rng)
                input_gains = np.zeros((channels, 1))
                input_gains[:lines_per_group, 0] = 1 / np.sqrt(lines_per_group)
                impulse = np.zeros((n_samples, 1))
                impulse[0, 0] = 1
                for design, lengths in (("coprime", base), ("exact volume", exact)):
                    totals = lengths.reshape(num_groups, lines_per_group).sum(axis=1)
                    angles = get_coupling_angles(beta, totals / (lines_per_group * fs))
                    coupling = expm(get_coupling_matrix(angles))
                    F = np.kron(coupling, np.eye(lines_per_group))
                    M = mixers
                    feedback = F @ M  # G = I for this lossless comparison.
                    modulation = make_room_modulation(coupling, lines_per_group, fs, modulation_rng)
                    for varying in (False, True):
                        modulation.reset()
                        build = FDNBuild(
                            A=feedback, B=input_gains, C=np.eye(channels),
                            D=np.zeros((channels, 1)), delays=lengths, fs=fs,
                            post_matrix=modulation if varying else None)
                        outputs = pyFDN.process_fdn(impulse, build)
                        metrics = measure_lossless_energy_errors(
                            outputs, lengths, lines_per_group, reference, volumes, fs)
                        results.append({
                            "topology": topology, "delay_design": design,
                            "lines_per_group": lines_per_group,
                            "delay_setting_samples": length_setting,
                            "modulated": varying, "duration_s": duration,
                            "matrix_seed": matrix_seed,
                            "modulation_scope": "room_internal_fixed_coupling", **metrics,
                        })
                        del outputs
                if progress is not None:
                    progress(topology, lines_per_group, length_setting)
        return results

    return match_delay_totals_to_volumes, run_steady_state_comparison


@app.cell
def _(mo):
    comparison_duration = mo.ui.number(1, 60, value=10, label="Comparison duration (s)")
    comparison_seed = mo.ui.number(0, 100000, value=1, label="Internal matrix seed")
    run_comparison = mo.ui.run_button(label="Generate steady-state comparison tables")
    mo.vstack([
        mo.md(r"""
        ### Steady-state error: fixed versus varying, coprime versus exact volume

        For unit input energy, the reported SA MAE in percentage points is
        $100\,\operatorname{mean}_{n\in W,i}|E_i^{\rm GFDN}(n)-E_i^{\rm SA}(n)|$.
        $W$ is the last 15% after trimming the longest delay from the end.
        The reference is **total room energy**, with source in room 1, computed
        from the column-conservative SA generator. No logarithm is taken.

        Equipartition MAE replaces SA by $\tau_i/\sum_j\tau_j$; allocation MAE
        compares this target with $V_i/\sum_jV_j$. The CSV also reports mean
        biases (average before taking absolute values), which separate a
        persistent offset from finite-window fluctuations.

        This runs 64 renders: both topologies, eight configurations, two delay
        designs and fixed/varying feedback. The duration is independent of the
        histogram sweep duration. Exact-volume designs relax coprimality
        while retaining independently sampled delay spreads in each group.
        """),
        comparison_duration, comparison_seed, run_comparison,
    ])
    return comparison_duration, comparison_seed, run_comparison


@app.cell
def _(
    V,
    beta_chain,
    beta_complete,
    comparison_duration,
    comparison_seed,
    csv,
    fs,
    io,
    mo,
    run_comparison,
    run_steady_state_comparison,
    sample_delay_line_lengths,
):
    mo.stop(not run_comparison.value, mo.md("Run the comparison to populate the tables."))
    comparison_results = run_steady_state_comparison(
        V, {"chain": beta_chain, "complete": beta_complete},
        sample_delay_line_lengths, fs, duration=comparison_duration.value,
        matrix_seed=int(comparison_seed.value),
        progress=lambda topology, lines, length: print(
            f"Finished {topology}: K={lines}, delay setting={length}"),
    )
    _table_rows = []
    for _fixed, _varying in zip(comparison_results[::2], comparison_results[1::2]):
        _table_rows.append({
            "Topology": _fixed["topology"], "Delays": _fixed["delay_design"],
            "K": _fixed["lines_per_group"],
            "Delay setting (samples)": _fixed["delay_setting_samples"],
            "SA MAE fixed (pp)": _fixed["sa_mae_pp"],
            "SA MAE varying (pp)": _varying["sa_mae_pp"],
            "Equipartition MAE fixed (pp)": _fixed["equipartition_mae_pp"],
            "Equipartition MAE varying (pp)": _varying["equipartition_mae_pp"],
            "Allocation MAE (pp)": _fixed["allocation_mae_pp"],
        })
    _buffer = io.StringIO()
    _writer = csv.DictWriter(_buffer, fieldnames=comparison_results[0].keys())
    _writer.writeheader()
    _writer.writerows(comparison_results)
    mo.vstack([
        mo.ui.table(_table_rows, selection=None),
        mo.download(
            data=_buffer.getvalue().encode(), filename="gfdn_steady_state_comparison.csv",
            mimetype="text/csv", label="Download all comparison metrics"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create sound examples for GFDN with $K_i = 16, \tau_i = 2000$ samples with exact vs co-prime delay lengths and fixed vs TV matrix
    """)
    return


@app.cell
def _(
    Path,
    FDNBuild,
    make_room_modulation,
    block_diag,
    expm,
    get_coupling_angles,
    get_coupling_matrix,
    json,
    match_delay_totals_to_volumes,
    np,
    pyFDN,
    resample_poly,
    sf,
    tempfile,
):
    def render_direct_wet_signals(
        stimulus_dir, output_dir, volumes, topology_betas, sample_delays, fs,
        absorption=(0.1, 0.3, 0.2), tail_seconds=10.0, matrix_seed=1,
        progress=None,
    ):
        """Process stimuli directly through 16-line GFDNs, including modulation.

        Source in R1; separate receiver-room files; independent identical
        systems for stereo channels, with synchronised modulation clocks.
        Add tail_seconds of zeros and keep the modulation running throughout.
        One gain per stimulus is shared by all designs, topologies and rooms.
        """
        stimulus_dir, output_dir = Path(stimulus_dir), Path(output_dir)
        stimuli = sorted(stimulus_dir.glob("*.wav"))
        if not stimuli:
            raise ValueError(f"No WAV stimuli found in {stimulus_dir}")
        lines, mean_delay = 16, 2000
        volumes = np.asarray(volumes, dtype=float)
        groups, channels = len(volumes), len(volumes) * lines
        base = sample_delays(mean_delay * volumes[0] / volumes.mean(), lines)
        exact = match_delay_totals_to_volumes(
            base, lines, volumes, target_total=mean_delay * channels)
        gamma = 343 * 6 * np.cbrt(volumes) ** 2 * np.asarray(absorption) / (4 * volumes)
        saved_rng = np.random.get_state()
        try:
            np.random.seed(matrix_seed)
            mixers = block_diag(*[pyFDN.random_orthogonal(lines) for _ in volumes])
            modulation_rng = np.random.get_state()
        finally:
            np.random.set_state(saved_rng)
        B = np.zeros((channels, 1))
        B[:lines, 0] = 1 / np.sqrt(lines)
        C = np.kron(np.eye(groups), np.ones((1, lines)) / np.sqrt(lines))
        networks = []
        for design, lengths in (("coprime", base), ("exact_volume", exact)):
            totals = lengths.reshape(groups, lines).sum(axis=1)
            attenuation = np.exp(-np.repeat(gamma, lines) * lengths / (2 * fs))
            for topology, beta in topology_betas.items():
                angles = get_coupling_angles(beta, totals / (lines * fs))
                coupling = expm(get_coupling_matrix(angles))
                F = np.kron(coupling, np.eye(lines))
                M = mixers
                G = np.diag(attenuation)
                feedback = F @ M
                modulation = make_room_modulation(coupling, lines, fs, modulation_rng)
                for lossy in (False, True):
                    for varying in (False, True):
                        name = (f"{topology}_K16_mean2000_{design}_"
                                f"{'lossy' if lossy else 'lossless'}_"
                                f"{'varying' if varying else 'fixed'}_srcR1")
                        build = FDNBuild(
                            A=feedback @ G if lossy else feedback, B=B, C=C,
                            D=np.zeros((groups, 1)), delays=lengths, fs=fs,
                            post_matrix=modulation if varying else None)
                        networks.append((name, lengths, build, varying))
        names = [f"{p.stem}__{name}_recR{r+1}.wav"
                 for p in stimuli for name, _, _, _ in networks for r in range(groups)]
    
        if any((output_dir / name).exists() for name in names):
            raise FileExistsError("Wet files already exist; choose a new output folder.")
        output_dir.mkdir(parents=True, exist_ok=True)
    
        records, gains = [], {}
        for stimulus in stimuli:
            dry, input_fs = sf.read(stimulus, always_2d=True)
            if input_fs != fs:
                factor = np.gcd(input_fs, fs)
                dry = resample_poly(dry, fs // factor, input_fs // factor, axis=0)
            dry_frames = len(dry)
            padded = np.pad(dry, ((0, int(tail_seconds * fs)), (0, 0)))
            largest_peak, pending = 0.0, []
            with tempfile.TemporaryDirectory(prefix="raw_direct_wet_", dir=output_dir) as scratch:
                for name, lengths, build, varying in networks:
                    wet = np.empty((len(padded), groups, dry.shape[1]))
                    for channel in range(dry.shape[1]):
                        if build.post_matrix is not None:
                            build.post_matrix.reset()
                        wet[:, :, channel] = pyFDN.process_fdn(
                            padded[:, channel:channel+1], build)
                    if not np.isfinite(wet).all():
                        raise ValueError(f"Non-finite output in {name}")
                    for receiver in range(groups):
                        audio = wet[:, receiver, :]
                        peak = float(np.max(np.abs(audio)))
                        if peak <= 0:
                            raise ValueError(f"Silent output in {name}, room {receiver+1}")
                        largest_peak = max(largest_peak, peak)
                        filename = f"{stimulus.stem}__{name}_recR{receiver+1}.wav"
                        sf.write(Path(scratch) / filename, audio, fs, subtype="FLOAT")
                        pending.append({
                            "file": filename, "stimulus": str(stimulus.resolve()),
                            "network": name, "receiver_room": receiver + 1,
                            "input_sample_rate": input_fs, "sample_rate": fs,
                            "frames": len(audio), "dry_frames_at_output_rate": dry_frames,
                            "channels": dry.shape[1], "raw_peak": peak,
                            "actual_mean_delay_samples": float(lengths.mean()),
                            "group_delay_totals_samples": lengths.reshape(groups, lines).sum(axis=1).tolist(),
                            "delay_lengths_samples": lengths.tolist(),
                        })
                    if progress is not None:
                        progress(stimulus.name, name)
                gain = 0.95 / largest_peak
                gains[stimulus.name] = gain
                for record in pending:
                    audio, _ = sf.read(Path(scratch) / record["file"], always_2d=True)
                    sf.write(output_dir / record["file"], audio * gain, fs, subtype="PCM_24")
                    record["listening_gain"] = gain
                records.extend(pending)
        manifest = {
            "processing": "Direct pyFDN.process_fdn; zero initial state; wet only; no convolution.",
            "lines_per_group": lines, "target_global_mean_delay_samples": mean_delay,
            "sample_rate": fs, "tail_seconds": tail_seconds, "matrix_seed": matrix_seed,
            "volumes_m3": volumes.tolist(), "absorption_coefficients": list(absorption),
            "energy_absorption_rates_per_s": gamma.tolist(),
            "topology_betas": {name: beta.tolist() for name, beta in topology_betas.items()},
            "modulation": {"scope": "room_internal_fixed_coupling",
                           "frequency_hz": 1.0, "amplitude_radians": 3.0, "spread": 0.3,
                           "clock": "Reset at stimulus start; continues through the zero-input tail."},
            "source_and_receiver_weights": "1/sqrt(K) per line in source/receiver room",
            "gain_by_stimulus": gains, "files": records,
        }
        (output_dir / "wet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output_dir / "README.txt").write_text(
            "Direct time-domain GFDN outputs: 16 lines PER GROUP, target global mean 2000 samples.\n"
            "Both topologies, coprime/exact-volume delays, lossless/lossy, fixed/varying.\n"
            "Source R1; separate receiver R1/R2/R3 files. Wet only; no dry mix or convolution.\n"
            "48 kHz PCM-24, input channels preserved, one gain per stimulus shared by all variants.\n"
            "Ten seconds of zero-input tail by default; lossless tails are truncated, no fades.\n"
            "Stereo uses independent identical networks with synchronised modulation.\n"
            "Internal room mixers vary; inter-room coupling remains fixed.\n"
            "See wet_manifest.json for actual delays, parameters and gains.\n")
        return manifest

    return (render_direct_wet_signals,)


@app.cell
def _(Path, mo):
    _grouped_fdn = Path(__file__).resolve().parents[3] / "GroupedFDN"
    wet_stimulus_directory = mo.ui.text(
        value=str(_grouped_fdn / "DiffGFDN/audio/sound_examples/stimulus"), label="Stimulus folder")
    wet_output_directory = mo.ui.text(
        value=str(_grouped_fdn / "audio/sa2gfdn/wet"), label="Wet output folder")
    wet_tail = mo.ui.number(1, 60, value=10, label="Zero-input tail (seconds)")
    wet_absorption = mo.ui.array([
        mo.ui.number(0, 1, value=_a, step=0.01, label=f"Room {_i+1} absorption")
        for _i, _a in enumerate((0.1, 0.3, 0.2))])
    run_wet_render = mo.ui.run_button(label="Render 16-line wet signals directly through GFDNs")
    mo.vstack([
        mo.md("### Direct GFDN wet audio: K=16, target global mean delay=2000 samples\n"
              "Uses the room volumes/apertures above and the comparison matrix seed. "
              "Includes both topologies and all fixed/varying, lossless/lossy and delay designs. "
              "Only the internal room mixers vary; coupling stays fixed. "
              "Modulation runs throughout each stimulus and its tail."),
        wet_stimulus_directory, wet_output_directory, wet_tail, wet_absorption, run_wet_render,
    ])
    return (
        run_wet_render,
        wet_absorption,
        wet_output_directory,
        wet_stimulus_directory,
        wet_tail,
    )


@app.cell
def _(
    V,
    beta_chain,
    beta_complete,
    comparison_seed,
    fs,
    mo,
    render_direct_wet_signals,
    run_wet_render,
    sample_delay_line_lengths,
    wet_absorption,
    wet_output_directory,
    wet_stimulus_directory,
    wet_tail,
):
    mo.stop(not run_wet_render.value, mo.md("Run the direct wet render to save audio files."))
    wet_render_manifest = render_direct_wet_signals(
        wet_stimulus_directory.value, wet_output_directory.value, V,
        {"chain": beta_chain, "complete": beta_complete}, sample_delay_line_lengths, fs,
        absorption=wet_absorption.value, tail_seconds=wet_tail.value,
        matrix_seed=int(comparison_seed.value),
        progress=lambda stimulus, network: print(f"{stimulus}: {network}"),
    )
    mo.md(f"Saved {len(wet_render_manifest['files'])} wet WAVs in `{wet_output_directory.value}`.")
    return


if __name__ == "__main__":
    app.run()
