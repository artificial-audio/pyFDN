"""Helper functions needed for physics-based coupling"""
import numpy as np
from scipy.linalg import block_diag, expm
from typing import Optional, Tuple, List, Union
from numpy.typing import NDArray, ArrayLike

import pyFDN
from pyFDN.dsp.time_varying_matrix import TimeVaryingMatrix


def make_beta(aperture_area: NDArray, volume: ArrayLike, c: float = 343):
    """Create the coupling coefficient from Cremer-Muller theory"""
    beta = c * aperture_area / (4.0 * volume[:, None])
    np.fill_diagonal(beta, 0.0)
    return beta


def make_gamma(absorp_area: List, volume: List, c: float = 343):
    """Create the diagonal absorption matrix from Cremer-Muller theory"""
    gamma = (c * absorp_area) / (4 * volume)
    return np.diag(gamma)


def make_Q(beta: NDArray):
    """Create the state transition matrix of the CT system from Cremer-Muller theory"""
    Q = beta.T.copy()
    np.fill_diagonal(Q, -beta.sum(axis=1))
    return Q


def trajectory(Q: NDArray,
               src_weight: Union[ArrayLike, NDArray],
               mic_weight: Union[ArrayLike, NDArray],
               fs: float,
               n_samp: int,
               gamma: Optional[NDArray] = None):
    r"""
    True physical trajectory e(t) = c_R^\top expm(Q t) b_S = c_R^\top V exp(\Lambda t) V^{-1} b_S
    Args:
        Q (NDarray): state transition matrix of size Nrooms x Nrooms
        _src_weight (ArrayLike): source weightings for each room of size num_src x Nrooms
        _mic_weight (NDArray, ArrayLike): receiver weightings for each room of size num_rec x Nrooms
        fs (float): sampling frequency
        n_samp (int): number of time samples
        _gamma (NDArray, optional): diagonal absorption matrix
    Returns:
        NDArray: the common decays and the energy decay trajectory of size num_src x num_time_samps x num_rec
    """

    if gamma is None:
        gamma = np.zeros_like(Q)
    w, Vv = np.linalg.eig(Q - gamma)
    # Q is reversible w.r.t. pi ~ V, so spectrum is real
    w, Vv = w.real, Vv.real
    Vinv = np.linalg.inv(Vv)
    coeffs = Vinv @ src_weight
    tsec = np.arange(n_samp) / fs
    modes = np.exp(np.outer(tsec, w))
    T = np.einsum("ik,tk->ti", Vv, modes * coeffs[None, :])
    return w, np.einsum("ti, mi -> tm", T, mic_weight)


def trajectory_with_delays(
    Q: NDArray,
    src_weight: ArrayLike,
    rec_weight: ArrayLike,
    src_delay: List[int],
    rec_delay: List[int],
    src_room: List[int],
    rec_room: List[int],
    fs: float,
    n_samp: int,
    gamma: Optional[NDArray] = None,
) -> NDArray:
    """
    Continuous-time Markov model with propagation delays between the
    source/receiver and the aperture.

    Args:
        src_weight : (n_src, n_room)
        rec_weight : (n_rec, n_room)
        src_room   : (n_src,) room index of each source
        rec_room   : (n_rec,) room index of each receiver
        src_delay  : (n_src,) aperture->source delay (samples)
        rec_delay  : (n_rec,) aperture->receiver delay (samples)
    Returns:
        NDArray: the common decays and the energy decay trajectory
"""

    if gamma is None:
        gamma = np.zeros_like(Q)

    A = Q - gamma

    # eigendecomposition
    w, V = np.linalg.eig(A)
    w = w.real
    V = V.real
    Vinv = np.linalg.inv(V)

    t = np.arange(n_samp) / fs
    modes = np.exp(np.outer(t, w))

    n_src = src_weight.shape[0]
    n_rec = rec_weight.shape[0]
    n_room = src_weight.shape[1]

    out = np.zeros((n_samp, n_src, n_rec))

    for s in range(n_src):

        # Source operator
        coeff = Vinv @ src_weight[s]
        room_energy = (V @ (modes * coeff[None, :]).T).T

        # delay only the rooms different from the source room
        Ls = int(src_delay[s])
        if Ls > 0:
            room_energy_delayed = room_energy.copy()

            for room in range(n_room):
                if room == src_room[s]:
                    continue

                room_energy_delayed[:, room] = 0.0
                room_energy_delayed[Ls:, room] = room_energy[:-Ls, room]

            room_energy = room_energy_delayed

        # Receiver operator
        for r in range(n_rec):
            Lr = int(rec_delay[r])
            room_energy_obs = room_energy.copy()

            if Lr > 0:
                for room in range(n_room):
                    if room == rec_room[r]:
                        continue

                    room_energy_obs[:, room] = 0.0
                    room_energy_obs[Lr:, room] = room_energy[:-Lr, room]

            out[:, s, r] = room_energy_obs @ rec_weight[r]

    return w, out


def make_theta(beta: NDArray, dt_i: List) -> NDArray:
    """
    Get the coupling angles from the coupling coefficients
    and mean free path lengths for all FDNs
    """
    n = len(dt_i)
    theta = np.zeros((n, n))
    for _i in range(n):
        for _j in range(_i + 1, n):
            val = 0.5 * (beta[_i, _j] * dt_i[_i] + beta[_j, _i] * dt_i[_j])
            theta[_i,
                  _j] = theta[_j,
                              _i] = np.arcsin(np.sqrt(val)) if val > 0 else 0.0
    return theta


def make_K(theta: NDArray):
    """Construct a skew symmetric matrix from the coupling coefficients"""
    tri = np.triu(theta, 1)
    return tri - tri.T


def lift(R_room: NDArray, Nroom: int):
    """
    List R_room from num_rooms x num_rooms to num_rooms x Nroom, num_rooms x Nroom.
    Nroom is the number of delay lines per room (same for each room)
    """
    # Simpler lifting with kroneckers
    num_rooms = R_room.shape[0]
    mat = np.zeros((num_rooms * Nroom, num_rooms * Nroom))
    for _i in range(num_rooms):
        for _j in range(num_rooms):
            mat[_i * Nroom:(_i + 1) * Nroom,
                _j * Nroom:(_j + 1) * Nroom] = R_room[_i, _j] * np.eye(Nroom)
    return mat


def get_decay(_gamma: ArrayLike, _M: ArrayLike, fs: float):
    """Get the diagonal decay matrix" of size num_rooms x num_rooms"""
    return np.diag(np.exp(-_gamma * _M / (2 * fs)))


def get_decay_matrix(gamma: NDArray, delays_per_fdn: NDArray, fs: float):
    """
    Get the block diagonal decay matrix
    Args:
        gamma (NDArray): num_rooms x num_rooms absorption matrix derived from physics
        delays_per_fdn (List): num_rooms x Nroom delay line lengths
        fs (float): sampling frequency
    """
    num_rooms = gamma.shape[0]
    Gamma = []
    for _i in range(num_rooms):
        Gamma.append(get_decay(gamma[_i, _i], delays_per_fdn[_i], fs))
    return block_diag(*Gamma)


def get_feedback_matrix(R_room: NDArray,
                        Qblocks: NDArray,
                        Nroom: int,
                        D: Optional[NDArray] = None):
    """
    Get the GFDN orthonormal feedback matrix.
    Args:
        R_room (NDArray): num_rooms x num_rooms coupling matrix
        Qblocks (NDArray: block diagonal per-room feedback matrix of shape num_rooms x Nroom, num_rooms x Nroom
        D (NDArray, optional): compensating matrix if the delay line ratio does not match the volume ratio exactly
    """
    R_room_lift = lift(R_room, Nroom)
    if D is None:
        return R_room_lift @ Qblocks
    else:
        return D @ R_room_lift @ np.linalg.inv(D) @ Qblocks


def run_gfdn(A,
             _B: NDArray,
             _C: NDArray,
             delays: ArrayLike,
             n_samp: int,
             _src: Union[int, List] = 0,
             tv_matrix: Optional[TimeVaryingMatrix] = None):
    """
    Generic wrapper around process_fdn.
    Args:
        A (NDArray): Ntot x Ntot feedback matrix (with losses)
        _B (NDArray): (Ntot, num_inputs) input matrix
        _C (NDArray) : (num_outputs, Ntot) output matrix
        delays (ArrayLike): delay line lengths in samples
        _src (int, list): room where the source is
        n_samp (int): number of time samples
        tv_matrix (Optional): if using a time varying matrix for increased mixing
    Returns:
        NDArray : GFDN output of size (num_inputs, n_samp, num_outputs)
    """

    num_inputs = _B.shape[1]
    num_outputs = _C.shape[0]

    Y = np.zeros((num_inputs, n_samp, num_outputs))
    for n_src in range(num_inputs):
        x = np.zeros((n_samp, num_inputs))
        k = _src[n_src] if isinstance(_src, list) else _src
        x[0, k] = 1.0

        Y[n_src] = pyFDN.process_fdn(
            x,
            delays,
            A,
            _B,
            _C,
            np.zeros((num_outputs, num_inputs)),
            extra_matrix=tv_matrix,
        )
    return Y


def gfdn_ledger(Y: NDArray, num_rooms: int, Nroom: int, delays: ArrayLike):
    """
    Calculate the GFDN energy ledger
    Args:
        Y (NDArray): of size num_inputs x n_samp x num_outputs
        num_rooms (int): number of rooms in GFDN
        Nroom (int): number of delay lines per room
        delays (ArrayLike): delay line lengths in samples
    Returns:
        NDArray: energy ledger of shape num_rooms x time samples x num_rooms
    """
    Ntot = num_rooms * Nroom
    # cumulative sum of the energy
    cs = np.concatenate(
        [np.zeros((num_rooms, 1, Ntot)),
         np.cumsum(Y**2, axis=1)], axis=1)
    n_samp = Y.shape[1]
    n_ex = n_samp - int(delays.max()) - 1
    idx = np.arange(n_ex)
    E = np.zeros((num_rooms, n_ex, num_rooms))
    bounds = [i * Nroom for i in range(num_rooms)]
    for _j, _m in enumerate(delays):
        # current room
        _room = next((i - 1 for i, b in enumerate(bounds) if _j < b),
                     len(bounds) - 1)
        E[:, :, _room] += cs[:, idx + 1 + _m, _j] - cs[:, idx + 1, _j]
    return E, n_ex
