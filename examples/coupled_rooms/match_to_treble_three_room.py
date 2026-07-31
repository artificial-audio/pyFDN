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
    from scipy.signal import butter, sosfilt, fftconvolve
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    from typing import Optional, Tuple, List, Union
    from numpy.typing import NDArray, ArrayLike

    import pyFDN
    from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix
    from pyFDN.auxiliary.geometry import (get_plane_area, 
                                          get_room_volume, 
                                          get_room_surface_area, 
                                          get_room_absorptive_area, 
                                          point_in_room,
                                          find_room,
                                          get_point_to_room_weights,
                                          Aperture
                                        )
    from pyFDN.auxiliary.physics_based_coupling import (make_beta, make_gamma, make_Q,
                                                        trajectory, trajectory_with_delays,
                                                        make_theta, make_K, get_decay_matrix, get_feedback_matrix,
                                                        run_gfdn, gfdn_ledger, room_energy_ledger_from_rirs)

    from dependencies.DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
    from sklearn.cluster import KMeans

    return (
        Aperture,
        ArrayLike,
        DecayFitNetToolbox,
        KMeans,
        List,
        NDArray,
        Path,
        TimeVaryingMatrix,
        Tuple,
        Union,
        block_diag,
        butter,
        cKDTree,
        expm,
        find_room,
        get_decay_matrix,
        get_feedback_matrix,
        get_point_to_room_weights,
        get_room_absorptive_area,
        get_room_volume,
        gfdn_ledger,
        least_squares,
        make_K,
        make_Q,
        make_beta,
        make_gamma,
        make_theta,
        mo,
        np,
        pickle,
        plt,
        pyFDN,
        room_energy_ledger_from_rirs,
        run_gfdn,
        sosfilt,
        trajectory,
        trajectory_with_delays,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we will parse the Treble's three room dataset and see if it matches Cremer-Muller's statistical theory. Then, we will
    generate a GFDN to match statistical theory and see how well it aligns with data
    """)
    return


@app.cell
def _(
    ArrayLike,
    List,
    NDArray,
    Path,
    ROOM1_DIMS,
    ROOM2_DIMS,
    ROOM2_START,
    Tuple,
    Union,
    cKDTree,
    mo,
    np,
    pickle,
):
    def dataset_to_sim_mic_pos(mic_pos: Union[ArrayLike, NDArray]) -> NDArray:
        """
        Map Treble three dataset mic position to simulation coordinates.
        mic_pos contains coordinates in (y, x, z) order
        """
        mic_pos = np.asarray(mic_pos, dtype=np.float64)
        if mic_pos.ndim == 1:
            mic_pos = mic_pos[np.newaxis, :]
        sim_mic_pos = np.zeros_like(mic_pos)

        for k in range(mic_pos.shape[0]):
            corr_x = mic_pos[k, 1] + ROOM2_START[0]
            corr_y = ROOM1_DIMS[1] + ROOM2_DIMS[1] - mic_pos[k, 0]
            sim_mic_pos[k, :] = np.array([corr_x, corr_y, mic_pos[k, 2]],
                                         dtype=np.float64)
        return sim_mic_pos


    def sample_receivers_along_line(
        receiver_positions: NDArray[np.float64],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        line_spacing_m: float,
    ) -> Tuple[List, NDArray]:
        """
        Sample the closest receiver positions along a line in the x-y plane.
        Args:
            receiver_positions (NDArray): shape (N, 3) containing all receiver positions.
            start_x, start_y (float, float): start point of the line.
            end_x, end_y (float, float): end point of the line.
            line_spacing_m (float): desired spacing between consecutive samples.
        Returns:
            List, NDArray: unique receiver positions closest to the sampled line 
                           and their indices in the dataset.
        """

        length = np.hypot(end_x - start_x, end_y - start_y)

        # Distances along the line, ensuring the endpoint is included
        if np.isclose(length, 0.0):
            query_points = np.array([[start_x, start_y]], dtype=np.float64)
        else:
            distances = np.arange(0.0, length, line_spacing_m)
            if distances.size == 0 or distances[-1] < length:
                distances = np.append(distances, length)

            t = distances / length

            query_points = np.column_stack((
                start_x + t * (end_x - start_x),
                start_y + t * (end_y - start_y),
            ))

        # Build KD-tree in 2D since height of receiver is fixed
        tree = cKDTree(receiver_positions[:, :2])

        # Find nearest receiver for each query point
        _, indices = tree.query(query_points)

        # Remove duplicate receivers while preserving order
        _, first = np.unique(indices, return_index=True)
        indices = indices[np.sort(first)]

        return indices, receiver_positions[indices]


    def create_linear_trajectory(
        receiver_positions: NDArray,
        line_spacing_m: float = 0.3,
    ) -> Tuple[ArrayLike, NDArray]:
        """
        Create a linear trajectory of a listener moving from Room 2 to room 1 to room 3.
        Returns a number array of shape (N, 3) containing listener positions along a trajectory
        and the indices of receivers in that trajectoty
        """
        assert line_spacing_m >= 0.3, "The dataset contains receivers every 0.3m on an x-y plane"

        # moving from room 2 to room 1
        idx1, traj1 = sample_receivers_along_line(
            receiver_positions,
            start_x=0.5,
            start_y=3.5,
            end_x=9.0,
            end_y=3.5,
            line_spacing_m=line_spacing_m,
        )

        # moving from room 1 to room 3
        idx2, traj2 = sample_receivers_along_line(
            receiver_positions,
            start_x=9.1,
            start_y=3.5,
            end_x=9.0,
            end_y=12.0,
            line_spacing_m=line_spacing_m,
        )

        # Avoid duplicating the corner point if both segments share it
        if np.allclose(traj1[-1], traj2[0]):
            traj2 = traj2[1:]
            idx2 = idx2[1:]

        return np.r_[idx1, idx2], np.vstack((traj1, traj2))

    def read_treble_data(pkl_path: Path) -> Tuple[NDArray, NDArray, ArrayLike, float]:
        """Read the Treble dataset and return the RIRs, receiver positions and source position"""
        with open(pkl_path.resolve(), "rb") as file:
            data_dict = pickle.load(file)
            sample_rate = data_dict['fs']
            source_position = np.asarray(data_dict['srcPos'],
                                         dtype=np.float64).squeeze()
            receiver_positions = data_dict['rcvPos'].T
            rirs = np.squeeze(data_dict['srirs'])

        return rirs, receiver_positions, source_position, sample_rate

    def parse_treble_three_room(
        pkl_path: Path,
        line_spacing_m: float = 0.3,
    ) -> Tuple[NDArray, NDArray, NDArray, List[int]]:
        """
        Parse the three room dataset and pick receiver positions on the y-line closest to `target_y_m`,
        sampled every `line_spacing_m` along x.
        Returns:
            Tuple: containing the sample rate, receivers in a trajectory, corresponding RIRs and the receiver indices
        """
        rirs, receiver_positions, source_position, sample_rate = read_treble_data(pkl_path)

        selected_indices, selected_receivers = create_linear_trajectory(
            receiver_positions, line_spacing_m)
        selected_rirs = rirs[selected_indices]
        return source_position, selected_receivers, selected_rirs, selected_indices


    mo.md("### Treble data parsing only")
    return dataset_to_sim_mic_pos, parse_treble_three_room, read_treble_data


@app.cell
def _(DecayFitNetToolbox, KMeans, NDArray, mo, np, pyFDN):
    # --- Stage 1: per-receiver multi-slope fits, aggregated per room -----------
    def estimate_room_slopes(rirs_in_room: NDArray, fs: float, n_slopes: int=3):
        """
        Estimate the T60s from the RIRs using k-means clustering
        Args:
            rirs_in_room : (n_rec_in_room, n_samples) raw RIRs for receivers in one room
        Returns: 
            T60s (s) and amplitudes for up to n_slopes, averaged across
        receivers and sorted slowest-first (most physically identifiable slope
        first, matching the paper's point that dominant/late slopes are best-
        supported by the data).
        """
        decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=fs)
        edcs_in_room = pyFDN.auxiliary.acoustics.edc(rirs_in_room, axis=-1)

        Ts, As = [], []
        for edc in edcs_in_room:
            estimated_parameters, _ = decayfitnet.estimate_parameters(edc.copy(), input_is_edc=True)
            T, A, N = estimated_parameters  # T: decay times, A: amplitudes, N: noise floor
            T, A = np.asarray(T).flatten(), np.asarray(A).flatten()
            order = np.argsort(-T)  # slowest (largest T60) first
            Ts.append(T)
            As.append(A)

        model_order = 1
        kmeans = KMeans(n_clusters=model_order)
        # Ts should be of shape n_samples, n_feature
        Ts = np.asarray(Ts)
        kmeans.fit(Ts)
        cluster_labels = kmeans.labels_.reshape(rirs_in_room.shape[0])
        cluster_centers = np.squeeze(kmeans.cluster_centers_)
        return cluster_centers

    mo.md("### Common slopes analysis from data")
    return (estimate_room_slopes,)


@app.cell
def _(Aperture, mo, np):
    # depth, length, height (middle room is room 1, first room from left is 2 and room at right is 3)
    ROOM1_DIMS = [3.0, 6.0, 3.0]
    ROOM2_DIMS = [8.0, 4.0, 3.0]
    ROOM3_DIMS = [8.0, 4.0, 3.0]
    ROOM1_START = [0, 0, 0]
    ROOM2_START = [-2.0, 6.0, 0]
    ROOM3_START = [3.0, 0, 0]
    # Dataset source is [2.0, 2.0, 1.5] in the same (y, x, z) convention as rcvPos.
    # Applying _dataset_to_sim_mic_pos gives simulation coordinates [0.0, 8.0, 1.5].
    SOURCE_POS = [0.0, 8.0, 1.5]
    # Treble/FDTD boundary labels in room_geometry2.pdf are absorption coefficients.
    ROOM1_ALPHA = 0.01
    ROOM2_ALPHA = 0.2
    ROOM3_ALPHA = 0.1
    ROOM1_ABS = [ROOM1_ALPHA for i in range(6)]
    ROOM2_ABS = [ROOM2_ALPHA for i in range(6)]
    ROOM3_ABS = [ROOM3_ALPHA for i in range(6)]

    ROOM_DIMS = [ROOM1_DIMS, ROOM2_DIMS, ROOM3_DIMS]
    ROOM_START = [ROOM1_START, ROOM2_START, ROOM3_START]

    APERTURE1_COORDS = [
        [0.75, 6, 0],
        [0.75, 6, 3],
        [2.25, 6, 3],
        [2.25, 6, 0],
    ]
    APERTURE2_COORDS = [
        [3, 0, 0],
        [3, 0, 3],
        [3, 1.5, 3],
        [3, 1.5, 0],
    ]
    aperture_12 = Aperture(np.asarray(APERTURE1_COORDS), 0, 1)
    aperture_13 = Aperture(np.asarray(APERTURE2_COORDS), 0, 2)
    apertures = [aperture_12, aperture_13]
    aperture_12_area = aperture_12.area
    aperture_13_area = aperture_13.area

    mo.md("### Room and aperture geometry")
    return (
        ROOM1_ABS,
        ROOM1_DIMS,
        ROOM2_ABS,
        ROOM2_DIMS,
        ROOM2_START,
        ROOM3_ABS,
        ROOM3_DIMS,
        ROOM_DIMS,
        ROOM_START,
        SOURCE_POS,
        aperture_12_area,
        aperture_13_area,
        apertures,
    )


@app.cell
def _(
    ROOM_DIMS,
    ROOM_START,
    all_rec_pos,
    all_rirs,
    dataset_to_sim_mic_pos,
    fs,
    mo,
    pyFDN,
    room_energy_ledger_from_rirs,
):
    fdtd_room_energy, room_masks = room_energy_ledger_from_rirs(all_rirs, dataset_to_sim_mic_pos(all_rec_pos), 
                                                                ROOM_START, ROOM_DIMS)

    fdtd_energy_env = pyFDN.auxiliary.acoustics.calculate_energy_envelope(fdtd_room_energy, 
                                                                          fs, smooth_time_ms=50, time_axis=-1)

    mo.md(rf"""### FDTD integrated energy ledger (mean over receivers in each room)

    We are plotting the EDC of the averaged room-integrated FDTD ledger, calculated as,

    $E_{{i_\text{{FDTD}}}}(t) = \int_{{V_i}} p^2(\mathbf{{x}}, t) dV \approx \frac{{1}}{{N}} \sum_i p_i^2(t)$, 

    vs the predicted energy update from Cremer-Muller equations when we excite room 2 with $b_S = [0, 1]$ and weigh the outputs
    equally from both roons $c_R = \begin{{pmatrix}}1 & 0 \\ 0 & 1 \end{{pmatrix}}$.
    """)
    return (fdtd_energy_env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupled room ODEs and solutions
    """)
    return


@app.cell
def _(
    ROOM1_ABS,
    ROOM1_DIMS,
    ROOM2_ABS,
    ROOM2_DIMS,
    ROOM3_ABS,
    ROOM3_DIMS,
    aperture_12_area,
    aperture_13_area,
    get_room_absorptive_area,
    get_room_volume,
    make_Q,
    make_beta,
    make_gamma,
    mo,
    np,
):
    S = np.array([[0, aperture_12_area, aperture_13_area], [aperture_12_area, 0, 0], [aperture_13_area, 0, 0]])
    V = np.array([get_room_volume(ROOM1_DIMS), get_room_volume(ROOM2_DIMS), get_room_volume(ROOM3_DIMS)])
    absorp_area = np.array([get_room_absorptive_area(ROOM1_DIMS, ROOM1_ABS), get_room_absorptive_area(ROOM2_DIMS, ROOM2_ABS), get_room_absorptive_area(ROOM3_DIMS, ROOM3_ABS)])
    num_rooms = 3

    beta = make_beta(S, V)
    gamma = make_gamma(absorp_area, V)
    Q = make_Q(beta)

    mo.md(rf"""
    ### Physical generators

    $$\mathbf \beta = \begin{{pmatrix}}{beta[0,0]:.4f}&{beta[0,1]:.4f}&{beta[0, 2]:.4f}\\{beta[1,0]:.4f}&{beta[1,1]:.4f}&{beta[1, 2]:.4f}\\
    {beta[2,0]:.4f}&{beta[2,1]:.4f}&{beta[2,2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    $$\mathbf Q = \begin{{pmatrix}}{Q[0,0]:.4f}&{Q[0,1]:.4f}&{Q[0, 2]:.4f}\\{Q[1,0]:.4f}&{Q[1,1]:.4f}&{Q[1, 2]:.4f}\\
    {Q[2,0]:.4f}&{Q[2,1]:.4f}&{Q[2,2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    Both have zero column sums (pure exchange, no absorption in this notebook) and are reversible w.r.t.
    $\pi_i\propto V_i$ — target long‑run split $V_1{{:}}V_2{{:}}V_3 = {V[0]:.0f}{{:}}{V[1]:.0f}{{:}}{V[2]:.0f}$.

    $$\mathbf \Gamma = \begin{{pmatrix}}{gamma[0,0]:.4f}&{gamma[0,1]:.4f}&{gamma[0, 2]:.4f}\\{gamma[1,0]:.4f}&{gamma[1,1]:.4f}&{gamma[1,2]:.4f}\\{gamma[2,0]:.4f}&{gamma[2,1]:.4f}&{gamma[2, 2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$
    """)
    return Q, V, beta, gamma, num_rooms


@app.cell
def _(
    Path,
    ROOM_DIMS,
    ROOM_START,
    SOURCE_POS,
    apertures,
    dataset_to_sim_mic_pos,
    find_room,
    get_point_to_room_weights,
    mo,
    np,
    parse_treble_three_room,
    read_treble_data,
):
    file_path = Path('../GroupedFDN/DiffGFDN/resources/Georg_3room_FDTD/srirs.pkl')
    all_rirs, all_rec_pos, src_pos, fs = read_treble_data(file_path)
    _, sel_rec, sel_rirs, sel_idx = parse_treble_three_room(file_path, line_spacing_m=0.5)

    c = 343
    rate_s = 1.0 / fs
    n_samp = int(fs * 2.0)
    tsec = np.arange(n_samp) / fs

    # get source weighting
    src_weight, src_delay = get_point_to_room_weights(np.asarray(SOURCE_POS), 
                                                      ROOM_DIMS, ROOM_START, apertures, return_delays=True, fs=fs)
    src_room = [find_room(ROOM_DIMS, ROOM_START, SOURCE_POS)]

    # for now select a few receivers and plot their ledger
    rec_idx = [2, 8, 13, 21, 30]
    rec_pos = dataset_to_sim_mic_pos(sel_rec[rec_idx])
    rec_params = np.asarray([get_point_to_room_weights(rec_pos[k], ROOM_DIMS, ROOM_START, 
                                                                  apertures, return_delays=True, fs=fs) 
                             for k in range(len(rec_idx))])
    rec_weight = np.asarray([rec_params[k][0] for k in range(len(rec_idx))])
    rec_delay = np.asarray([rec_params[k][1] for k in range(len(rec_idx))])

    rec_room = [find_room(ROOM_DIMS, ROOM_START, rec_pos[k]) for k in range(len(rec_idx))]

    mo.md("### Read data and calculate source and receiver weights and delays")
    return (
        all_rec_pos,
        all_rirs,
        fs,
        n_samp,
        rec_delay,
        rec_idx,
        rec_pos,
        rec_room,
        rec_weight,
        sel_rec,
        sel_rirs,
        src_delay,
        src_room,
        src_weight,
        tsec,
    )


@app.cell
def _(
    Q,
    all_rirs,
    estimate_room_slopes,
    fs,
    gamma,
    mo,
    n_samp,
    np,
    num_rooms,
    rec_delay,
    rec_room,
    rec_weight,
    src_delay,
    src_room,
    src_weight,
    trajectory,
    trajectory_with_delays,
):
    # trajectory for a particular source and receiver position
    common_decay, Emarkov_src_rec = trajectory_with_delays(Q, src_weight[None,:], 
                                                         rec_weight, src_delay[None, :], rec_delay, 
                                                         src_room, rec_room, fs, n_samp,
                                                         gamma)

    Emarkov_src_rec = Emarkov_src_rec.squeeze()
    common_t60= np.log(1e-6) / common_decay

    # trajectory independent of source and receiver position
    _, Emarkov_lossy = trajectory(Q, np.array([0, 1, 0]), np.eye(num_rooms), fs, n_samp, gamma)

    # without the 0.5, the decay rates appear to be twice as long
    common_t60_ref = 0.5 * estimate_room_slopes(all_rirs, int(fs), n_slopes=num_rooms)

    mo.md(rf"""
    ### Markov trajectory

    The common decay times from solving ODEs are {np.round(common_t60, 3)}s.

    The common decay times from kmeans are {np.round(common_t60_ref, 3)}s.

    The estimated source weights are [{src_weight[0]:.3f}, {src_weight[1]:.3f}, {src_weight[2]:.3f}]

    The estimated receiver weights are {np.round(rec_weight, 3)}
    """)
    return common_t60, common_t60_ref


@app.cell
def _(
    List,
    NDArray,
    beta,
    common_t60_ref,
    gamma,
    least_squares,
    make_Q,
    mo,
    np,
):
    # Fit the eta correction against the DecayFitNet-identified rates ---
    def fit_eta_to_decayfitnet(common_T60s: List,
                               beta :NDArray,
                               gamma_diag :NDArray,
                               db_target=60.0):
        """
        common_T60s : list of length num_rooms, each element containing estimated T60s
                      for that room's DecayFitNet fit.

        Strategy: use only the SLOWEST (best-identified) rate from each room as
        the calibration target, since faster/secondary slopes are typically
        weakly supported by the data (cf. the paper's note that Room 2's rate
        had "a platykurtic posterior... only weakly supported"). If the model is
        behaving physically, these per-room slowest rates should roughly agree
        with each other (same shared system eigenvalue seen from different
        rooms)
        """
        # Longest T60 and corresponding room
        idx_slow = np.argmax(np.asarray(common_T60s))
        T60_slow = common_T60s[idx_slow]

        ln_target = np.log(10**(-db_target / 10))  # e.g. ln(1e-6) for 60 dB
        dominant_rate = np.array([ln_target / -T60_slow])  # 1/s, positive
        # dominant_rate = np.array([ln_target / -common_T60s])  # 1/s, positive

        def residual(eta):
            eta = np.abs(eta)
            # eta[i] = 1 : no correction in room i
            # eta[i] > 1 : stronger decay than Sabine predicts
            # eta[i] < 1 : weaker decay than Sabine predicts
            gamma_c = gamma_diag.copy()
            # Correct ONLY the absorption of the room
            # associated with the longest T60
            gamma_c[idx_slow] *= eta[0]
            # Correct absorption of all rooms
            # gamma_c = gamma_c * eta
            Q = make_Q(beta)
            A_c = Q - gamma_c
            w = np.sort(np.abs(np.linalg.eig(A_c)[0].real))
            # match the model's SLOWEST eigenvalue to the consensus dominant rate --
            # the best-identified quantity on both sides of the comparison
            return np.sum(np.abs(np.array([w[0] - dominant_rate])))

        result = least_squares(residual,
                               x0=np.ones_like(gamma_diag.shape[0]),
                               bounds=(0.1, 5.0))
        print(f'Mismatch in rates is {result.fun}s')
        return np.abs(result.x)


    eta_fit = fit_eta_to_decayfitnet(common_t60_ref, beta, gamma)
    # gamma_c  = gamma * eta_fit

    idx_slow = np.argmax(np.asarray(common_t60_ref))
    gamma_c = gamma.copy()
    gamma_c[idx_slow, idx_slow] *= eta_fit[0]

    mo.md(rf"""# Fix mismatch in decay rates between observation and theory

    Use the common decay times calculated from the data and use that to find the correct
    absorption matrix that would reproduce the same decay rates.

    Original absorption:
    $\mathbf \Gamma$ :
    $$\mathbf \Gamma = \begin{{pmatrix}}{gamma[0,0]:.4f}&{gamma[0,1]:.4f}&{gamma[0, 2]:.4f}\\{gamma[1,0]:.4f}&{gamma[1,1]:.4f}&{gamma[1,2]:.4f}\\{gamma[2,0]:.4f}&{gamma[2,1]:.4f}&{gamma[2, 2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    Corrected absorption:
    $\mathbf \Gamma_c$ :
    $$\mathbf \Gamma_c = \begin{{pmatrix}}{gamma_c[0,0]:.4f}&{gamma_c[0,1]:.4f}&{gamma_c[0, 2]:.4f}\\{gamma_c[1,0]:.4f}&{gamma_c[1,1]:.4f}&{gamma_c[1,2]:.4f}\\{gamma_c[2,0]:.4f}&{gamma_c[2,1]:.4f}&{gamma_c[2, 2]:.4f}\end{{pmatrix}}\ \mathrm{{s}}^{{-1}}$$

    """)
    return (gamma_c,)


@app.cell
def _(
    Q,
    common_t60,
    common_t60_ref,
    fs,
    gamma_c,
    mo,
    n_samp,
    np,
    num_rooms,
    rec_delay,
    rec_room,
    rec_weight,
    src_delay,
    src_room,
    src_weight,
    trajectory,
    trajectory_with_delays,
):
    # trajectory for a particular source and receiver position
    new_common_decay, Emarkov_src_rec_new = trajectory_with_delays(Q, src_weight[None,:], 
                                                             rec_weight, src_delay[None, :], rec_delay, 
                                                             src_room, rec_room, fs, n_samp,
                                                             gamma_c)
    Emarkov_src_rec_new = Emarkov_src_rec_new.squeeze()
    new_common_t60= np.log(1e-6) / new_common_decay

    # trajectory independent of source and receiver position
    _, Emarkov_lossy_new = trajectory(Q, np.array([0, 1, 0]), np.eye(num_rooms), fs, n_samp, gamma_c)

    mo.md(rf"""
    The old common decay times from solving ODEs are {np.round(common_t60, 3)}s.

    The common decay times from kmeans are {np.round(common_t60_ref, 3)}s.

    The new common decay times from solving ODEs are {np.round(new_common_t60, 3)}s.
    """)
    return Emarkov_lossy_new, Emarkov_src_rec_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design approrpiate GFDN to match the Markov trajectory
    """)
    return


@app.cell
def _(V, fs, mo, np, num_rooms, pyFDN):
    def ms_to_samps(time_ms: float, fs:float):
        return int(np.round(time_ms * 1e-3 * fs))

    Nroom = 24
    avg_delay_ms = 30
    avg_delays = ms_to_samps(avg_delay_ms, fs)
    N1 = N2 = N3 = Nroom
    Ntot = N1 + N2 + N3

    gscale = V / V[0]
    delay_min = int(np.sqrt(0.5) * avg_delays)
    delay_max = int(np.sqrt(2) * avg_delays)
    delays = []

    for _i in range(num_rooms):
        delays.append(pyFDN.sample_delay_lengths(
            Nroom, (int(delay_min * gscale[_i]), int(delay_max * gscale[_i])), coprime=True, rng=331 + 10*_i))

    delays = np.concatenate(delays).astype(int)
    M1, M2, M3 = int(delays[:N1].sum()), int(delays[N1:N1+N2].sum()), int(delays[N1+N2:].sum())
    delays_per_fdn = np.asarray([delays[:N1], delays[N1:N1+N2], delays[N1+N2:]], dtype=np.int32)
    M = np.asarray([M1, M2, M3])
    dt_i = M / (Nroom * fs)

    mo.md(rf"""
    ## 1 · Geometry — $M_i \propto V_i$, equal $N_i$ (so $\mathbf D=\mathbf I$, no weighting needed)

    | | room 1 | room 2 | room 3 |
    |---|---|---|---|
    | $V_i$ | {V[0]} | {V[1]} | {V[2]}| 
    | $N_i$ | {N1} | {N2} | {N3} |
    | actual $\sum d_j$ | {int(delays[:N1].sum())} | {int(delays[N1:N1+N2].sum())} | {int(delays[N1+N2:].sum())} |
    | $\Delta t_i=M_i/(N_if_s)$ | {dt_i[0]*1000:.3f} ms | {dt_i[1]*1000:.3f} ms | {dt_i[2]*1000:.3f} ms |
    """)
    return N1, N2, N3, Ntot, delays, delays_per_fdn, dt_i, ms_to_samps


@app.cell
def _(beta, dt_i, expm, make_K, make_theta, mo, num_rooms):
    theta = make_theta(beta, dt_i)
    K = make_K(theta)
    R_room = expm(K)

    # sanity check: beta_ij*dt_i should equal beta_ji*dt_j (design consistency)
    _check = []
    for _i in range(num_rooms):
        for _j in range(_i + 1, num_rooms):
            _check.append(
                (beta[_i, _j] * dt_i[_i], beta[_j, _i] * dt_i[_j]))

    mo.md(rf"""
    ## 2 · Pairwise angles $\theta_{{ij}}$ (rate‑matched, $\sin^2\theta_{{ij}}=\beta_{{ij}}\Delta t_i$)

    | edge | $\theta_{{ij}}$ |
    |---|---|
    | 1-2 | {K[0,1]:.4f} | 
    | 2-1 | {K[1,0]:.4f} |
    | 1-3 | {K[0,2]:.4f} |
    | 3-1 | {K[2,0]:.4f} |
    | 2-3 | {K[1,2]:.4f} | 
    | 3-2 | {K[2,1]:.4f} |

    Consistency check ($\beta_{{ij}}\Delta t_i$ vs. $\beta_{{ji}}\Delta t_j$, should match by the $\Delta t_i\propto V_i$
    design): edge 1–2: ${_check[0][0]:.5f}$ vs ${_check[0][1]:.5f}$; edge 1–3: ${_check[1][0]:.5f}$ vs
    ${_check[1][1]:.5f}$; edge 2–3: ${_check[2][0]:.5f}$ vs ${_check[2][1]:.5f}$.

    Note -- if the delays are too large, then 0.5(beta / dt_i) approaches 1, and the coupling angle caps at pi/2.
    """)
    return (R_room,)


@app.cell
def _(
    N1,
    N2,
    N3,
    Ntot,
    R_room,
    TimeVaryingMatrix,
    block_diag,
    delays_per_fdn,
    fs,
    gamma_c,
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
                         pyFDN.random_orthogonal(N2),
                         pyFDN.random_orthogonal(N3))
    Gamma = get_decay_matrix(gamma_c, delays_per_fdn, fs)
    A = get_feedback_matrix(R_room, Qblocks, N1)
    A_lossy = A @ Gamma 
    _ok = pyFDN.is_unilossless(A)

    mo.md(rf"""
    ## 3 · The two GFDN feedback matrices

    Same internal room mixers $\mathbf Q_1,\mathbf Q_2, \mathbf Q_3$ for both — only the room‑coupling
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
    # Markov ledger matching - excite room 2 only and take output equally from rooms 1, 2 and 3
    B = np.zeros((Ntot, num_rooms))
    B[N1:N1+N2, 1] = 1.0 / np.sqrt(N2)

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
    N3,
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
    B_src[N1:N1+N2, 0] = src_weight[1] / np.sqrt(N2)
    B_src[N1+N2:, 0] = src_weight[2] / np.sqrt(N3)

    # size is num_receivers x num_delay_lines
    num_rec = rec_pos.shape[0]
    C_rec = np.zeros((num_rec, Ntot))
    offset = 0
    for _room, _M in enumerate([N1, N2, N3]):
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
    Emarkov_lossy_new,
    V,
    fdtd_energy_env,
    mo,
    n_ex,
    np,
    num_rooms,
    plt,
    tsec,
):
    _fig, _axs = plt.subplots(1, 3, figsize=(10.5, 4.2))
    _labels = ["Room 1 (middle)", "Room 2 (left)", "Room 3 (right)"]
    _colors = ["C0", "C1", "C2"]
    _targets = V / V.sum()

    def db(_Y, is_energy_signal:bool=True):
        tmp = np.log10(np.abs(_Y) + 1e-10)
        return 10*tmp if is_energy_signal else 20*tmp

    for room in range(num_rooms):

        ax = _axs[room]
        ax.plot(
            tsec,
            db(fdtd_energy_env[room, :len(tsec)] / fdtd_energy_env[1, 0]),
            color=_colors[room],
            lw=1.5,
            label="FDTD",
        )
        ax.plot(
            tsec,
            db(Emarkov_lossy_new[:, room]),
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
    return (db,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot the EDC for source-receiver pairs
    """)
    return


@app.cell
def _(
    Emarkov_src_rec_new,
    Y_src_rec,
    butter,
    db,
    fs,
    mo,
    ms_to_samps,
    n_samp,
    np,
    num_rec,
    plt,
    pyFDN,
    rec_idx,
    rec_pos,
    sel_rec,
    sel_rirs,
    sosfilt,
    tsec,
):
    _fig, _axs = plt.subplots(1, num_rec, figsize=(12.5, 3.5), sharex=True, sharey=True)
    _tmax =  tsec[-1]
    _nmax = int(_tmax * len(tsec) / tsec[-1])
    _src = 1

    _start_time = ms_to_samps(50, fs)
    _highpass_cutoff_hz = 750.0
    _highpass_sos = butter(
        8,
        _highpass_cutoff_hz,
        btype="highpass",
        fs=int(fs),
        output="sos",
    )
    _ref_rirs = sosfilt(_highpass_sos, sel_rirs[rec_idx], axis=-1)
    edc_ref = pyFDN.auxiliary.acoustics.edc(_ref_rirs[:,_start_time:_nmax], axis=-1, normalize=True)
    edc_fdn = pyFDN.auxiliary.acoustics.edc(Y_src_rec[:,_start_time:_nmax, :].squeeze(), axis=0, normalize=True)

    # shape noise to generate RIR from Markov ledger
    noise = np.random.normal(0, 1.0, n_samp)
    noise *= np.sqrt(n_samp / np.sum(noise**2))
    rir_markov = np.einsum('tk, t -> tk', np.sqrt(Emarkov_src_rec_new), noise)
    edc_markov = pyFDN.auxiliary.acoustics.edc(rir_markov, axis=0, normalize=True)

    for _rec, _ax in zip(range(len(rec_pos)), _axs):
        _ax.plot(tsec[_start_time:_nmax] * 1000, db(edc_ref[_rec, :]), lw=1.0, color="C0")
        _ax.plot(tsec[:_nmax] * 1000, db(edc_markov[:, _rec]), lw=1.0, color="C2")
        _ax.plot(tsec[_start_time:_nmax] * 1000, db(edc_fdn[:, _rec]), lw=1.0, color="C3")
        _ax.set_title(f"receiver:{np.round(sel_rec[rec_idx[_rec]], 2)}", fontsize=9)
        _ax.set_xlabel("time (ms)")
        _ax.set_ylabel("$EDC (db)$")
        _ax.grid(True, alpha=0.3)

    _axs[0].set_ylim(-60, 10)
    _axs[0].legend(['Reference', 'Cremer-Muller', 'GFDN'])
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
