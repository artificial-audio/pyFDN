import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    import pickle
    from pathlib import Path
    from scipy.linalg import block_diag, expm
    from typing import Optional, Tuple, List, Union
    from numpy.typing import NDArray, ArrayLike

    import pyFDN
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix
    from pyFDN.auxiliary.geometry import (get_plane_area, 
                                          get_room_volume, 
                                          get_room_surface_area, 
                                          get_room_absorptive_area, 
                                          point_in_room, 
                                          solid_angle_quad)
    from pyFDN.auxiliary.physics_based_coupling import (make_beta, make_gamma, make_Q,
                                                        trajectory, trajectory_with_delays,
                                                        make_theta, make_K, get_decay_matrix, get_feedback_matrix,
                                                        run_gfdn, gfdn_ledger)

    return (
        ArrayLike,
        NDArray,
        Path,
        TimeVaryingMatrix,
        Tuple,
        Union,
        block_diag,
        get_decay_matrix,
        get_feedback_matrix,
        get_plane_area,
        get_room_absorptive_area,
        get_room_volume,
        gfdn_ledger,
        make_K,
        make_Q,
        make_beta,
        make_gamma,
        make_theta,
        mo,
        np,
        pickle,
        plt,
        point_in_room,
        pyFDN,
        run_gfdn,
        solid_angle_quad,
        trajectory,
        trajectory_with_delays,
    )


@app.cell
def _(mo):
    mo.md(rf"""
    In this notebook, we will parse the ERTD dataset and see if it matches Cremer-Muller's statistical theory. Then, we will
    generate a GFDN to match statistical theory and see how well it aligns with data
    """)
    return


@app.cell
def _(
    ArrayLike,
    NDArray,
    Path,
    ROOM1_DIMS,
    ROOM1_START,
    ROOM2_DIMS,
    ROOM2_START,
    Tuple,
    Union,
    mo,
    np,
    pickle,
):
    def _select_along_receiver_axis(rirs: NDArray[np.float64],
                                    selected_indices: NDArray[np.integer],
                                    n_receivers: int) -> NDArray[np.float64]:
        """Select receivers from `rirs` by detecting the receiver axis."""
        rirs = np.asarray(rirs, dtype=np.float64)
        receiver_axes = [
            axis for axis, size in enumerate(rirs.shape) if size == n_receivers
        ]
        if not receiver_axes:
            raise ValueError(
                f"No axis in {repr(rirs)} matches number of receivers ({n_receivers}). "
                f"rirs.shape={rirs.shape}")
        return np.take(rirs, selected_indices, axis=receiver_axes[0])


    def _infer_axis_spacing(values: NDArray[np.float64]) -> float:
        """Infer uniform spacing from sorted coordinate levels along one axis."""
        levels = np.unique(np.round(values, decimals=9))
        diffs = np.diff(levels)
        if diffs.size == 0:
            raise ValueError("Could not infer spacing from axis values.")
        return float(np.min(diffs))


    def dataset_to_sim_mic_pos(
            mic_pos: Union[ArrayLike, NDArray]) -> NDArray[np.float64]:
        """Map ERTD dataset mic position to simulation coordinates."""
        mic_pos = np.asarray(mic_pos, dtype=np.float64)
        if mic_pos.ndim == 1:
            mic_pos = mic_pos[np.newaxis, :]
        sim_mic_pos = np.zeros_like(mic_pos)

        for k in range(mic_pos.shape[0]):
            corr_x = ROOM1_DIMS[0] + ROOM2_DIMS[0] - mic_pos[k, 0]
            sim_mic_pos[k, :] = np.array([corr_x, mic_pos[k, 1], mic_pos[k, 2]],
                                         dtype=np.float64)
        return sim_mic_pos


    def read_ertd(pkl_path: Path) -> Tuple[NDArray, NDArray, ArrayLike]:
        """Read the ERTD dataset and return the RIRs, receiver positions and source position"""
        with open(pkl_path.resolve(), "rb") as file:
            data_dict = pickle.load(file)

        # get source position
        src_pos = np.asarray(data_dict["srcPos"], dtype=np.float64).squeeze()
        # Dataset contract: recPos is always shape (3, n).
        rec_pos = np.asarray(data_dict["recPos"], dtype=np.float64).T
        rirs = np.asarray(data_dict["rirs"], dtype=np.float64)

        return rirs, rec_pos, src_pos


    def parse_ertd(
        rirs: NDArray,
        rec_pos: NDArray, 
        src_pos: ArrayLike,
        line_spacing_m: float = 0.5,
        target_y_m: float = 10.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
        """
        Parse ERTD and pick receiver positions on the y-line closest to `target_y_m`,
        sampled every `line_spacing_m` along x.
        """
        assert line_spacing_m > 0, "Line spacing must be positive"

        # Select exactly one y-line: closest value at/below target if available.
        unique_y = np.unique(np.round(rec_pos[:, 1], decimals=6))
        below_or_equal = unique_y[unique_y <= float(target_y_m)]
        selected_y = float(
            np.max(below_or_equal)) if below_or_equal.size > 0 else float(
                np.min(unique_y))

        y_line_indices = np.flatnonzero(
            np.isclose(rec_pos[:, 1], selected_y, rtol=0.0, atol=1e-9))
        y_line_points = rec_pos[y_line_indices]

        x_spacing_m = _infer_axis_spacing(y_line_points[:, 0])
        stride = max(1, int(round(line_spacing_m / x_spacing_m)))
        x_origin = float(np.min(y_line_points[:, 0]))
        x_grid_idx = np.rint(
            (y_line_points[:, 0] - x_origin) / x_spacing_m).astype(int)
        selected_indices = y_line_indices[(x_grid_idx % stride) == 0]
        if selected_indices.size == 0:
            selected_indices = y_line_indices[:1]

        # Ensure the receiver path is spatially continuous along x.
        # Without this, indices can be in dataset order (not x-sorted), causing a jump in the last frame.
        sort_order = np.argsort(rec_pos[selected_indices, 0])
        selected_indices = selected_indices[sort_order]

        selected_receivers = rec_pos[selected_indices]
        selected_rirs = _select_along_receiver_axis(rirs, selected_indices,
                                                    len(rec_pos))
        return src_pos, selected_receivers, selected_rirs, selected_indices


    def room_energy_from_rirs(
        rirs: np.ndarray,
        receiver_positions: np.ndarray,
    ):
        """
        Estimate the spatially integrated room energy from a dense set of RIRs.

        Args:
            rirs : (Nrec, Nt)
                Pressure RIRs.
            receiver_positions : (Nrec,3)
                Receiver coordinates.
            room_bounds :
                [(xmin,ymin,zmin),(xmax,ymax,zmax)] for each room.

        Returns:
            room_energy : (Nroom,Nt)
                Estimated room energies.
            room_masks : List[np.ndarray]
                Boolean mask of receivers belonging to each room.
        """
        room_bounds = []
        room_bounds.append((np.array(ROOM1_START), np.array(ROOM1_START)+np.array(ROOM1_DIMS)))
        room_bounds.append((np.array(ROOM2_START), np.array(ROOM2_START)+ np.array(ROOM2_DIMS)))
        n_rooms = len(room_bounds)
        n_samples = rirs.shape[1]
        room_energy = np.zeros((n_rooms, n_samples))
        room_masks = []

        pressure2 = rirs**2
        for room_idx, (xyz_min, xyz_max) in enumerate(room_bounds):
            mask = np.all(
                (receiver_positions >= xyz_min)
                & (receiver_positions <= xyz_max),
                axis=1,
            )

            room_masks.append(mask)
            room_energy[room_idx] = pressure2[mask].mean(axis=0)

        return room_energy, room_masks

    mo.md(rf"""### ERTD parsing only

          Room 1 is reverberant hallway and Room 2 is coupled meeting room (drier).
          Source is in room 2.""")
    return dataset_to_sim_mic_pos, parse_ertd, read_ertd, room_energy_from_rirs


@app.cell
def _(get_plane_area, np):
    # depth, length, height
    ROOM1_DIMS = [4.5, 18.0, 2.8]
    # ERTD recPos x spans up to 9.3m, so ROOM1_DIMS[0] + ROOM2_DIMS[0] must be >= 9.3
    # to keep mirrored positions inside the simulated geometry.
    ROOM2_DIMS = [4.8, 6.6, 2.8]
    ROOM1_START = [0, 0, 0]
    ROOM2_START = [4.5, 7.0, 0.0]
    SOURCE_POS = [5.9, 12.5, 1.5]
    ROOM_DIMS = [ROOM1_DIMS, ROOM2_DIMS]
    ROOM_START = [ROOM1_START, ROOM2_START]
    ROOM1_ABS = [0.042 for i in range(6)]
    ROOM2_ABS = [0.12 for i in range(6)]

    # define aperture coordinates
    aperture_start = [4.5, 9.5, 0]
    aperture_z_len = 2.0  # Approximation (door height not specified in dataset docs).
    aperture_y_len = 1.0
    aperture_coords = [
        [aperture_start[0], aperture_start[1], aperture_start[2]],
        [aperture_start[0], aperture_start[1] + aperture_y_len, aperture_start[2]],
        [aperture_start[0],aperture_start[1] + aperture_y_len,aperture_start[2] + aperture_z_len],
        [aperture_start[0], aperture_start[1], aperture_start[2] + aperture_z_len]]
    aperture_area = get_plane_area(aperture_coords)
    aperture_center = np.mean(np.asarray(aperture_coords), axis=0)
    return (
        ROOM1_ABS,
        ROOM1_DIMS,
        ROOM1_START,
        ROOM2_ABS,
        ROOM2_DIMS,
        ROOM2_START,
        SOURCE_POS,
        aperture_area,
        aperture_center,
        aperture_coords,
    )


@app.cell
def _(
    ROOM1_ABS,
    ROOM1_DIMS,
    ROOM2_ABS,
    ROOM2_DIMS,
    aperture_area,
    get_room_absorptive_area,
    get_room_volume,
    make_Q,
    make_beta,
    make_gamma,
    mo,
    np,
):
    S = np.array([[0, aperture_area], [aperture_area, 0]])
    V = np.array([get_room_volume(ROOM1_DIMS), get_room_volume(ROOM2_DIMS)])
    absorp_area = np.array([get_room_absorptive_area(ROOM1_DIMS, ROOM1_ABS), get_room_absorptive_area(ROOM2_DIMS, ROOM2_ABS)])
    num_rooms = 2

    beta = make_beta(S, V)
    gamma = make_gamma(absorp_area, V)
    Q = make_Q(beta)

    mo.md(rf"""
    ### Physical generators

    $\mathbf Q$ :
    $$\mathbf Q = \begin{{pmatrix}}{Q[0,0]:.4f}&{Q[0,1]:.4f}\\{Q[1,0]:.4f}&{Q[1,1]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    Both have zero column sums (pure exchange, no absorption in this notebook) and are reversible w.r.t.
    $\pi_i\propto V_i$ — target long‑run split $V_1{{:}}V_2{{:}}= {V[0]:.3f}{{:}}{V[1]:.3f}$.

    $\mathbf \Gamma$ :
    $$\mathbf \Gamma = \begin{{pmatrix}}{gamma[0,0]:.4f}&{gamma[0,1]:.4f}\\{gamma[1,0]:.4f}&{gamma[1,1]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$
    """)
    return Q, V, beta, gamma, num_rooms


@app.cell
def _(
    ArrayLike,
    Path,
    ROOM1_DIMS,
    ROOM1_START,
    ROOM2_DIMS,
    ROOM2_START,
    aperture_coords,
    mo,
    np,
    num_rooms,
    parse_ertd,
    point_in_room,
    read_ertd,
    solid_angle_quad,
):
    fs = 44100
    c = 343
    rate_s = 1.0 / fs
    n_samp = int(fs * 0.5)
    tsec = np.arange(n_samp) / fs
    file_path = Path('../GroupedFDN/DiffGFDN/resources/ERTD_dataset/ertd1.pkl')
    all_rirs, all_rec_pos, src_pos = read_ertd(file_path)
    _, sel_rec, sel_rirs, sel_idx = parse_ertd(all_rirs, all_rec_pos, src_pos)

    def find_room(point: ArrayLike) -> int:
        """Find which room point is in"""
        in_rooms = [
            point_in_room(ROOM1_START, ROOM1_DIMS, point),
            point_in_room(ROOM2_START, ROOM2_DIMS, point)
        ]
        point_in_which_room = in_rooms.index(True) if True in in_rooms else None
        return point_in_which_room

    def set_point_weight(point: ArrayLike) -> ArrayLike:
        """Decide which room a point is in and set its weight according to the solid angle it subtends to the aperture"""
        point_in_which_room = find_room(point)

        _omega = solid_angle_quad(np.array(aperture_coords), point)
        _weight = np.ones(num_rooms) * (_omega / (4*np.pi))
        _weight[point_in_which_room] = (1 -(_omega / (4 * np.pi))) / (num_rooms - 1)
        return _weight


    mo.md("### Markov trajectory helpers")
    return (
        all_rec_pos,
        all_rirs,
        c,
        find_room,
        fs,
        n_samp,
        sel_rec,
        sel_rirs,
        set_point_weight,
        tsec,
    )


@app.cell
def _(
    Q,
    SOURCE_POS,
    aperture_center,
    c,
    dataset_to_sim_mic_pos,
    find_room,
    fs,
    gamma,
    mo,
    n_samp,
    np,
    num_rooms,
    sel_rec,
    set_point_weight,
    trajectory,
    trajectory_with_delays,
):
    # get source weighting
    src_weight = set_point_weight(np.asarray(SOURCE_POS))
    src_room = [find_room(SOURCE_POS)]

    # for now select two receivers and plot their ledger
    rec_idx = [2, 5, 8, 13]
    rec_pos = dataset_to_sim_mic_pos(sel_rec[rec_idx])
    rec_weight = np.asarray([set_point_weight(rec_pos[k]) for k in range(len(rec_idx))])
    rec_room = [find_room(rec_pos[k]) for k in range(len(rec_idx))]

    # propagation delays (samples)
    src_delay = np.round(
        np.linalg.norm(SOURCE_POS - aperture_center[None, :], axis=1)
        / c * fs
    ).astype(int)

    rec_delay = np.round(
        np.linalg.norm(rec_pos - aperture_center[None, :], axis=1)
        / c * fs
    ).astype(int)

    # trajectory for a particular source and receiver position
    common_decay, Emarkov_src_rec = trajectory_with_delays(Q, src_weight[None,:], 
                                                         rec_weight, src_delay, rec_delay, 
                                                         src_room, rec_room, fs, n_samp,
                                                         gamma)
    Emarkov_src_rec = Emarkov_src_rec.squeeze()
    common_t60= np.log(1.0 / 1000) / common_decay

    # trajectory independent of source and receiver position
    _, Emarkov_lossy = trajectory(Q, np.array([0, 1]), np.eye(num_rooms), fs, n_samp, gamma)

    mo.md(rf"""
    ### Markov ledger

    The common decay times are {np.round(common_t60, 3)}s

    The source weights are [{src_weight[0]:.3f}, {src_weight[1]:.3f}].

    The receiver weights are {np.round(rec_weight, 3)}
    """)
    return (
        Emarkov_lossy,
        Emarkov_src_rec,
        rec_idx,
        rec_pos,
        rec_weight,
        src_weight,
    )


@app.cell
def _(
    all_rec_pos,
    all_rirs,
    dataset_to_sim_mic_pos,
    mo,
    np,
    pyFDN,
    room_energy_from_rirs,
):
    fdtd_room_energy, room_masks = room_energy_from_rirs(all_rirs, dataset_to_sim_mic_pos(all_rec_pos))
    fdtd_energy_decay = pyFDN.auxiliary.acoustics.edc(np.sqrt(fdtd_room_energy) / np.sqrt(np.sum(fdtd_room_energy)) , axis=-1)

    mo.md(rf"""### FDTD integrated energy ledger (mean over receivers in each room)

    We are plotting the EDC of the averaged room-integrated FDTD ledger, calculated as,

    $E_{{i_\text{{FDTD}}}}(t) = \int_{{V_i}} p^2(\mathbf{{x}}, t) dV \approx \frac{{1}}{{N}} \sum_i p_i^2(t)$, 

    vs the predicted energy update from Cremer-Muller equations when we excite room 2 with $b_S = [0, 1]$ and weigh the outputs
    equally from both roons $c_R = \begin{{pmatrix}}1 & 0 \\ 0 & 1 \end{{pmatrix}}$.
    """)
    return (fdtd_energy_decay,)


@app.cell
def _(mo):
    mo.md("""
    ## Design approrpiate GFDN to match the Markov ledger
    """)
    return


@app.cell
def _(V, fs, mo, np, pyFDN):
    Nroom = 16
    N1 = N2 = Nroom
    Ntot = N1 + N2
    M_avg = 1000

    gscale = V[0] / V[1]
    d2 = pyFDN.sample_delay_lengths(N2, (M_avg * 0.5, M_avg * 1.5), coprime=True, rng=122)
    d1 = pyFDN.sample_delay_lengths(
        N1, (int(M_avg * 0.5 * gscale), int(M_avg * 1.5 * gscale)), coprime=True, rng=331
    )
    delays = np.concatenate([d1, d2]).astype(int)
    delays_per_fdn = np.asarray([delays[:N1], delays[N1:]], dtype=np.int32)

    M1, M2 = int(d1.sum()), int(d2.sum())
    M = [M1, M2]
    ratio_mismatch = (V[0] / V[1]) / (M1 / M2)

    k1 = Nroom / np.sum(delays[:N1])
    k2 = Nroom / np.sum(delays[N1:])
    dt_i = 1.0 / (np.array([k1, k2], dtype=np.float32)  * fs)

    mo.md(rf"""
    ## 1 · Geometry — $M_i \propto V_i$, equal $N_i$ (so $\mathbf D=\mathbf I$, no weighting needed)

    | | room 1 | room 2 |
    |---|---|---|
    | $N_i$ | {N1} | {N2} |
    | $M_i$ | {M[0]} | {M[1]} |
    | Ratio mismatch| {ratio_mismatch:.4f} |
    | $\Delta t_i=M_i/(N_if_s)$ | {dt_i[0]*1000:.3f} ms | {dt_i[1]*1000:.3f} ms | 
    """)
    return N1, N2, Ntot, delays, delays_per_fdn, dt_i


@app.cell
def _(beta, dt_i, make_K, make_theta, mo, np, num_rooms):
    theta = make_theta(beta, dt_i)
    K = make_K(theta)
    R_room = np.array([[np.cos(theta[0,1]), np.sin(theta[0, 1])],[-np.sin(theta[1, 0]), np.cos(theta[1, 0])]])

    # sanity check: beta_ij*dt_i should equal beta_ji*dt_j (design consistency)
    _check = []
    for _i in range(num_rooms):
        for _j in range(_i + 1, num_rooms):
            _check.append(
                (beta[_i, _j] * dt_i[_i], beta[_j, _i] * dt_i[_j]))

    mo.md(rf"""
    ## 2 · Pairwise angles $\theta_{{ij}}$ (rate‑matched, $\sin^2\theta_{{ij}}=\beta_{{ij}}\Delta t_i$)

    | edge | $\theta_{{ij}}$|
    |---|---|
    | 1-2 | {K[0,1]:.4f} | 
    | 2-1 | {K[1, 0]: .4f} |


    Consistency check ($\beta_{{ij}}\Delta t_i$ vs. $\beta_{{ji}}\Delta t_j$, should match by the $\Delta t_i\propto V_i$
    design): edge 1–2: ${_check[0][0]:.5f}$ vs ${_check[0][1]:.5f}$;
    """)
    return (R_room,)


@app.cell
def _(
    N1,
    N2,
    Ntot,
    R_room,
    TimeVaryingMatrix,
    block_diag,
    delays_per_fdn,
    fs,
    gamma,
    get_decay_matrix,
    get_feedback_matrix,
    mo,
    np,
    pyFDN,
):
    # --- Lift into the full GFDN feedback matrix --------------------------------
    np.random.seed(12343)
    # time varying matrix for faster mixing
    modulation_frequency = 1.0  # hz
    modulation_amplitude = 3.0
    spread = 0.3
    tv_matrix = TimeVaryingMatrix(
            Ntot, modulation_frequency, modulation_amplitude, fs, spread
        )

    Qblocks = block_diag(pyFDN.random_orthogonal(N1),
                         pyFDN.random_orthogonal(N2))
    Gamma = get_decay_matrix(gamma, delays_per_fdn, fs)
    A = get_feedback_matrix(R_room, Qblocks, N1)
    A_lossy = A @ Gamma 
    _ok = pyFDN.is_unilossless(A)

    mo.md(rf"""
    ## 3 · The two GFDN feedback matrices

    Same internal room mixers $\mathbf Q_1,\mathbf Q_2$ for both — only the room‑coupling
    $\mathbf R_{{\rm room}}$ differs. `is_unilossless`: = **{_ok}**
    (both should be `True` — pure exchange, no absorption).
    """)
    return A_lossy, tv_matrix


@app.cell
def _(
    A_lossy,
    N1,
    N2,
    Ntot,
    delays,
    gfdn_ledger,
    mo,
    n_samp,
    np,
    num_rooms,
    run_gfdn,
    tv_matrix,
):
    # Markov ledger matching - excite room 2 only and take output equally from rooms 1 and 2
    B = np.zeros((Ntot, num_rooms))
    B[N1:, 1] = 1.0 / np.sqrt(N2)

    # output taken from all rooms
    C_lines = np.eye(Ntot)
    Y_lossy = run_gfdn(A_lossy, B, C_lines, delays, n_samp, _src=1, tv_matrix=tv_matrix)
    E_ex_lossy, n_ex = gfdn_ledger(Y_lossy, num_rooms, N1, delays)

    mo.md("""### Get the GFDN ledger comparable to the Markov ledger""")
    return E_ex_lossy, n_ex


@app.cell
def _(
    A_lossy,
    N1,
    N2,
    Ntot,
    delays,
    mo,
    n_samp,
    np,
    rec_pos,
    rec_weight,
    run_gfdn,
    src_weight,
    tv_matrix,
):
    # position dependent EDC matching - excite the particular source-receiver position
    # size is num_delay_lines x num_sources
    num_src = 1
    B_src = np.zeros((Ntot, num_src))
    B_src[:N1, 0] = src_weight[0] / np.sqrt(N1)
    B_src[N1:, 0] = src_weight[1] / np.sqrt(N2)

    # size is num_receivers x num_delay_lines
    num_rec = rec_pos.shape[0]
    C_rec = np.zeros((num_rec, Ntot))
    offset = 0
    for _room, _M in enumerate([N1, N2]):
        C_rec[:, offset:offset+_M] = (
            rec_weight[:, _room][:, None] / np.sqrt(_M)
        )
        offset += _M

    Y_src_rec = run_gfdn(A_lossy, B_src, C_rec, delays, n_samp, tv_matrix=tv_matrix)

    mo.md("### Get the GFDN RIRs corresponding to the desired source and receiver positions")
    return Y_src_rec, num_rec


@app.cell
def _(
    E_ex_lossy,
    Emarkov_lossy,
    V,
    fdtd_energy_decay,
    mo,
    n_ex,
    np,
    num_rooms,
    plt,
    tsec,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _labels = ["Room 1 (hallway)", "Room 2 (meeting)"]
    _colors = ["C0", "C1"]
    _targets = V / V.sum()

    def db(_Y, is_energy_signal:bool=True):
        tmp = np.log10(np.abs(_Y) + 1e-10)
        return 10*tmp if is_energy_signal else 20*tmp

    def ms_to_samp(time_ms: float, fs:float):
        return int(np.round(time_ms * 1e-3 * fs))

    for room in range(num_rooms):

        ax = _axs[room]
        ax.plot(
            tsec,
            db(fdtd_energy_decay[room, :len(tsec)]),
            color=_colors[room],
            lw=1.5,
            label="FDTD",
        )
        ax.plot(
            tsec,
            db(Emarkov_lossy[:, room]),
            "--",
            color=_colors[room],
            lw=1.5,
            label="Cremer-Muller",
        )
        ax.plot(
            tsec[:n_ex], 
            db(E_ex_lossy[1, :, room]), 
               '-.', 
               color = _colors[room], 
               lw=1.2, 
               label='GFDN'
        )
        # ax.axhline(
        #     db(np.array([_targets[room]]))[0],
        #     color=_colors[room],
        #     ls=":",
        #     lw=1.0,
        #     label="Target",
        # )
        ax.set_title(_labels[room])
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-60, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _axs[0].set_ylabel("Energy (dB)")
    _fig.suptitle(
        "Room-integrated energy: FDTD vs. Markov model vs. GFDN",
        fontsize=10,
    )
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return db, ms_to_samp


@app.cell
def _(mo):
    mo.md("""
    ### Plot the EDC for source-receiver pairs
    """)
    return


@app.cell
def _(
    Emarkov_src_rec,
    Y_src_rec,
    db,
    fs,
    mo,
    ms_to_samp,
    n_samp,
    np,
    num_rec,
    plt,
    pyFDN,
    rec_idx,
    rec_pos,
    sel_rec,
    sel_rirs,
    tsec,
):
    _fig, _axs = plt.subplots(1, num_rec, figsize=(10.5, 3.5), sharex=True, sharey=True)
    _tmax =  tsec[-1]
    _nmax = int(_tmax * len(tsec) / tsec[-1])
    _src = 1

    _start_time = ms_to_samp(50, fs)
    edc_ref = pyFDN.auxiliary.acoustics.edc(sel_rirs[rec_idx,_start_time:_nmax] / np.sqrt(np.sum(sel_rirs[rec_idx, _start_time:_nmax]**2)), axis=-1)
    edc_fdn = pyFDN.auxiliary.acoustics.edc(Y_src_rec[:,_start_time:_nmax, :].squeeze() / np.sqrt(np.sum(Y_src_rec[:, _start_time:_nmax,:].squeeze()**2)), axis=0)

    # shape noise to generate RIR from Markov ledger
    noise = np.random.normal(0, 1.0, n_samp)
    noise *= np.sqrt(n_samp / np.sum(noise**2))
    rir_markov = np.einsum('tk, t -> tk', np.sqrt(Emarkov_src_rec), noise)
    edc_markov = pyFDN.auxiliary.acoustics.edc(rir_markov / np.sqrt(np.sum(rir_markov**2)), axis=0)

    for _rec, _ax in zip(range(len(rec_pos)), _axs):
        _ax.plot(tsec[_start_time:_nmax] * 1000, db(edc_ref[_rec, :]), lw=1.0, color="C0")
        _ax.plot(tsec[:_nmax] * 1000, db(edc_markov[:, _rec]), lw=1.0, color="C2")
        _ax.plot(tsec[_start_time:_nmax] * 1000, db(edc_fdn[:, _rec]), lw=1.0, color="C3")
        _ax.set_title(f"receiver:{np.round(sel_rec[rec_idx[_rec]], 2)}", fontsize=9)
        _ax.set_xlabel("time (ms)")
        _ax.set_ylabel("$EDC (db)$")
        _ax.grid(True, alpha=0.3)

    _axs[0].set_ylim(-60, 0)
    _axs[0].legend(['Reference', 'Cremer-Muller', 'GFDN'])
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
