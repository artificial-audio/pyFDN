#!/usr/bin/env python3
"""
FDN allpass / orthogonal completions with optional diagonal similarity pre-scaling.

Implements:
  1) Full MIMO (k = N): Halmos/Julia dilation completion (balanced, V orthogonal/unitary)
  2) General MIMO (k <= N): CS/SVD defect-subspace completion (balanced)
  3) SISO is just the k=1 special case of (2)
  4) Optional diagonal similarity preprocessing: find diagonal X ≻ 0 and scale
       A_tilde = X^{-1/2} A X^{1/2}
     then complete A_tilde in balanced form, and map back:
       B = X^{1/2} B_tilde
       C = C_tilde X^{-1/2}
       D = D_tilde

Dependencies:
  - numpy
  - scipy (recommended) for stable matrix square roots in full MIMO.
    If scipy not available, a fallback eigen-sqrt for symmetric/Hermitian matrices is used.

Notes:
  - "Orthogonal" below means real-orthogonal; for complex matrices, it means unitary.
  - The general (k<=N) completion is exact if A has (approximately) N-k singular values at 1
    and k singular values strictly < 1 (within tolerance). Otherwise we still build a
    defect-subspace completion using the k smallest singular values, but exact orthogonality
    may be lost; checks will report the deviation.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import scipy.linalg as sla
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import pyFDN
# ---------------------------- Utilities -------------------------------------

def _I(n: int, dtype) -> np.ndarray:
    return np.eye(n, dtype=dtype)

def _adj(A: np.ndarray) -> np.ndarray:
    # conjugate-transpose
    return A.conj().T

def hermitize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + _adj(M))

def eig_sqrt_psd(M: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Matrix square root for Hermitian PSD matrix (fallback if SciPy is unavailable).
    Clips eigenvalues below eps to eps (use eps=0 for pure PSD).
    """
    M = hermitize(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V * np.sqrt(w)) @ _adj(V)

def sqrtm_psd(M: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Square root for Hermitian PSD matrix using SciPy if available; otherwise eigen fallback.
    Clips eigenvalues below eps to eps.
    """
    M = hermitize(M)
    if _HAVE_SCIPY:
        # SciPy sqrtm can introduce tiny imaginary parts; we hermitize after.
        S = sla.sqrtm(M)
        S = hermitize(S)
        # Clip if requested by projecting via eigen-sqrt if eps>0
        if eps > 0:
            return eig_sqrt_psd(M, eps=eps)
        return S
    else:
        return eig_sqrt_psd(M, eps=eps)

def diag_sqrt(x: np.ndarray) -> np.ndarray:
    return np.diag(np.sqrt(x))

def diag_inv_sqrt(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return np.diag(1.0 / np.sqrt(np.maximum(x, eps)))

def orth_error(V: np.ndarray) -> float:
    """|| V^* V - I ||_F"""
    n = V.shape[1]
    E = _adj(V) @ V - _I(n, V.dtype)
    return float(np.linalg.norm(E, ord="fro"))

def block_matrix(A, B, C, D) -> np.ndarray:
    return np.block([[A, B],
                     [C, D]])


# ------------------ Diagonal similarity preprocessing ------------------------

def diagonal_similarity_from_abs2_lyapunov(
    A: np.ndarray,
    q: float | np.ndarray = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Heuristic diagonal X = diag(x) ≻ 0 by enforcing the *diagonal* Lyapunov-like equation
        x - diag(A diag(x) A^*) = q
    which is linear in x using M = |A|^2 elementwise:
        (I - M) x = q,  M_ij = |A_ij|^2

    This is a sufficient-condition style preprocessing; it does NOT guarantee existence for
    arbitrary A. If (I - M) is ill-conditioned or yields nonpositive entries, this fails.

    Returns:
        x : (N,) positive vector, X = diag(x)
    """
    A = np.asarray(A)
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    M = np.abs(A) ** 2  # elementwise
    rhs = np.full(N, float(q), dtype=float) if np.isscalar(q) else np.asarray(q, dtype=float)
    if rhs.shape != (N,):
        raise ValueError("q must be scalar or shape (N,)")

    K = np.eye(N) - M
    # Solve (I - M) x = q
    try:
        x = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Failed to solve (I - |A|^2) x = q; matrix may be singular.") from e

    if np.any(~np.isfinite(x)):
        raise RuntimeError("Non-finite entries in x from diagonal similarity solve.")
    if np.min(x) <= eps:
        raise RuntimeError(
            f"Diagonal similarity failed: min(x)={np.min(x):.3e} <= {eps}. "
            "Try a different A, reduce gain, or skip preprocessing."
        )
    return x


def apply_diagonal_similarity(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    A_tilde = X^{-1/2} A X^{1/2} with X = diag(x)
    """
    A = np.asarray(A)
    x = np.asarray(x, dtype=float)
    Dp = diag_sqrt(x)
    Dm = diag_inv_sqrt(x, eps=0.0)
    return Dm @ A @ Dp


def map_back_from_similarity(Bt: np.ndarray, Ct: np.ndarray, Dt: np.ndarray, x: np.ndarray):
    """
    Map balanced completion (A_tilde, Bt, Ct, Dt) back to original coordinates:
        B = X^{1/2} Bt
        C = Ct X^{-1/2}
        D = Dt
    """
    x = np.asarray(x, dtype=float)
    Dp = diag_sqrt(x)
    Dm = diag_inv_sqrt(x, eps=0.0)
    B = Dp @ Bt
    C = Ct @ Dm
    D = Dt
    return B, C, D


# ------------------------ Completion methods ---------------------------------

def complete_full_mimo_halmos(A: np.ndarray, psd_clip: float = 0.0):
    """
    Full MIMO (k=N) Halmos/Julia dilation.

    For A with ||A||_2 <= 1 (contraction), the block matrix
        V = [[A,  (I - A A^*)^{1/2}],
             [(I - A^* A)^{1/2},  -A^*]]
    is unitary/orthogonal.

    Returns: B, C, D (all NxN)
    """
    A = np.asarray(A)
    N = A.shape[0]
    if A.shape != (N, N):
        raise ValueError("A must be square.")

    dtype = A.dtype
    I = _I(N, dtype)

    BB = I - A @ _adj(A)
    CC = I - _adj(A) @ A

    B = sqrtm_psd(BB, eps=psd_clip)
    C = sqrtm_psd(CC, eps=psd_clip)
    D = -_adj(A)
    return B, C, D


def complete_general_mimo_svd(A: np.ndarray, k: int, tol_one: float = 1e-8, choose_smallest_if_needed: bool = True):
    """
    General MIMO completion via defect subspace (CS/SVD-based).

    Given A (NxN) and k <= N, we aim for a balanced orthogonal/unitary block::

        V = [[A, B],
             [C, D]]

    with B in Nxk, C in kxN, D in kxk.

    Exact orthogonality is guaranteed when A has:

    - N-k singular values equal to 1 (within tol_one)
    - k singular values strictly < 1

    If the singular value pattern doesn't match, we can still build the completion using
    the k smallest singular values (choose_smallest_if_needed=True), but V may deviate from unitary.

    Construction (in the singular vector basis)::

        A = U1 S V1^* + U2 I V2^*
        B = U1 sqrt(I - S^2)
        C = sqrt(I - S^2) V1^*
        D = -S

    Returns: B (Nxk), C (kxN), D (kxk)
    """
    A = np.asarray(A)
    N = A.shape[0]
    if A.shape != (N, N):
        raise ValueError("A must be square.")
    if not (1 <= k <= N):
        raise ValueError("k must satisfy 1 <= k <= N")

    # SVD: A = U diag(s) Vh
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    V = _adj(Vh)

    # Identify "defect" indices: s < 1 - tol
    # defect = np.where(s < 1.0 - tol_one)[0]
    # if defect.size >= k:
    #     idx = defect[:k]
    # else:
    #     if not choose_smallest_if_needed:
    #         raise RuntimeError(
    #             f"Singular values do not provide enough defect directions: found {defect.size}, need {k}."
    #         )
    
    # fall back to k smallest singular values
    idx = np.argsort(s)[:k]

    # Build U1,V1 for selected singular values
    U1 = U[:, idx]               # Nxk
    V1 = V[:, idx]               # Nxk
    S = np.diag(s[idx])          # kxk
    # sqrt(I - S^2)
    G = np.diag(np.sqrt(np.maximum(1.0 - (s[idx] ** 2), 0.0)))  # kxk, real nonnegative

    B = U1 @ G                   # Nxk
    C = G @ _adj(V1)             # kxN
    D = -S                       # kxk

    return B, C, D



# ------------------ Diagonal similarity parameterization ---------------------

def scaled_A_from_u(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    A_tilde(u) = D^{-1} A D, with D = diag(exp(u)).
    Implemented without forming D matrices explicitly.
    """
    d = np.exp(u)
    # Right-multiply columns by d, left-multiply rows by 1/d:
    return (A * d[None, :]) / d[:, None]

def G_from_u(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    G(u) = I - A_tilde^* A_tilde, Hermitian by construction (hermitized).
    """
    At = scaled_A_from_u(A, u)
    N = A.shape[0]
    G = np.eye(N, dtype=At.dtype) - _adj(At) @ At
    return hermitize(G)


# ------------------ Objective: PSD + rank-defect k ---------------------------

def softplus(x: np.ndarray, tau: float) -> np.ndarray:
    """
    Smooth approximation of max(0, x) for vector x.
    tau: temperature (smaller -> sharper).
    """
    # stable softplus: tau*log(1+exp(x/tau))
    z = x / tau
    # avoid overflow
    z = np.clip(z, -60.0, 60.0)
    return tau * np.log1p(np.exp(z))

def objective_rank_defect(
    u: np.ndarray,
    A: np.ndarray,
    k: int,
    tau: float = 1e-3,
    alpha_tail: float = 1.0,
    beta_psd: float = 10.0,
    gamma_top: float = 1e-4,
    delta_top: float = 1e-3,
    reg_u: float = 1e-6,
) -> float:
    """
    Objective encouraging:
      - PSD: all eigenvalues of G(u) >= 0 (penalize negatives)
      - rank-defect k: smallest N-k eigenvalues -> 0 (tail penalty)
      - avoid rank < k: largest k eigenvalues not too small (top-k barrier)
      - mild regularization on u (scale gauge)

    Eigenvalues are in ascending order from eigh.
    """
    G = G_from_u(A, u)
    lam = np.linalg.eigvalsh(G)  # ascending: lam[0] ... lam[N-1]
    N = lam.size

    # PSD penalty: penalize negative eigenvalues
    neg_pen = np.sum(softplus(-lam, tau=tau) ** 2)

    # Tail-to-zero penalty: for rank k, we want N-k eigenvalues near 0.
    # Since lam ascending, the "near-zero" ones are lam[0 : N-k]
    tail = lam[: max(N - k, 0)]
    tail_pen = np.sum(tail ** 2)

    # Top-k away-from-zero: enforce lam[N-k:] >= delta_top
    top = lam[max(N - k, 0):]
    top_barrier = np.sum(softplus(delta_top - top, tau=tau) ** 2)

    # Regularize u (and remove gauge by centering)
    u0 = u - np.mean(u)
    reg = reg_u * float(np.dot(u0, u0))
    reg = 0
    
    return float(alpha_tail * tail_pen + beta_psd * neg_pen + gamma_top * top_barrier + reg)

# ------------------ Initialization helpers -----------------------------------

def init_u_abs2_heuristic(A: np.ndarray, q: float = 1.0) -> np.ndarray:
    """
    Same heuristic idea as (I - |A|^2) x = q, then u = 0.5 log x.
    This is a decent initializer for the nonconvex optimization.
    """
    A = np.asarray(A)
    N = A.shape[0]
    M = np.abs(A) ** 2
    rhs = np.full(N, float(q), dtype=float)
    K = np.eye(N) - M
    x = np.linalg.solve(K, rhs)
    x = np.maximum(x, 1e-12)
    u = 0.5 * np.log(x)
    u = u - np.mean(u)
    return u


# ------------------ Main optimizer -------------------------------------------

import scipy.linalg as sla
from scipy.optimize import minimize

def optimize_diagonal_similarity_rank_defect_scipy(
    A: np.ndarray,
    k: int,
    u0: np.ndarray | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 300,
    tau: float = 1e-7,
    alpha_tail: float = 1.0,
    beta_psd: float = 10.0,
    gamma_top: float = 1e-4,
    delta_top: float = 1e-3,
    reg_u: float = 1e-6,
    verbose: bool = True,
):
    """
    Optimize u (log-diagonal scaling) so that G(u) is PSD with rank ~ k.

    Returns:
      u_opt, info dict including achieved eigenvalues and success flag.
    """
    A = np.asarray(A)
    N = A.shape[0]
    if A.shape != (N, N):
        raise ValueError("A must be square.")
    if not (1 <= k <= N):
        raise ValueError("k must satisfy 1 <= k <= N")

    if u0 is None:
        # heuristic initializer; if it fails, fall back to zeros
        try:
            u0 = init_u_abs2_heuristic(A, q=1.0)
        except Exception:
            u0 = np.zeros(N, dtype=float)

    u0 = np.asarray(u0, dtype=float)
    if u0.shape != (N,):
        raise ValueError("u0 must have shape (N,)")

    def fun(u):
        u = np.asarray(u, dtype=float)
        # enforce gauge (scale invariance) by centering u inside objective
        uc = u - np.mean(u)
        return objective_rank_defect(
            uc, A, k,
            tau=tau,
            alpha_tail=alpha_tail,
            beta_psd=beta_psd,
            gamma_top=gamma_top,
            delta_top=delta_top,
            reg_u=reg_u,
        )

    # Optional: mild bounds to avoid extreme scaling (helps numerics)
    # Feel free to widen if you need more freedom.
    bounds = [(-10.0, 10.0)] * N  # start moderate; widen only if needed

    res = minimize(
        fun,
        x0=u0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 5000,
            "maxfun": 200000,
            "ftol": 1e-14,
            "gtol": 1e-12,
            "eps": 1e-12,
            "maxls": 80,
            "disp": True,
        },
    )

    u_opt = np.asarray(res.x, dtype=float)
    u_opt = u_opt - np.mean(u_opt)

    # Diagnostics
    G = G_from_u(A, u_opt)
    lam = np.linalg.eigvalsh(G)  # ascending
    info = {
        "success": bool(res.success),
        "status": int(res.status) if hasattr(res, "status") else None,
        "message": str(res.message),
        "nit": int(res.nit) if hasattr(res, "nit") else None,
        "fun": float(res.fun),
        "eig_min": float(lam[0]),
        "eig_max": float(lam[-1]),
        "eig_tail_norm": float(np.linalg.norm(lam[: max(N - k, 0)])),
        "eig_top_min": float(np.min(lam[max(N - k, 0):])) if k > 0 else None,
        "eigvals": lam,
    }
    if verbose:
        print("Optimization:", info["success"], info["message"])
        print(f"  objective={info['fun']:.3e}, nit={info['nit']}")
        print(f"  eig_min={info['eig_min']:.3e}, eig_top_min={info['eig_top_min']:.3e}, tail_norm={info['eig_tail_norm']:.3e}")

    return u_opt, info


# ------------------------------ High-level API -------------------------------

def complete_fdn(
    A: np.ndarray,
    k: int | None = None,
    preprocessing: str = "optimize",
    q: float | np.ndarray = 1.0,
    tol_one: float = 1e-8,
    psd_clip: float = 0.0,
    return_similarity: bool = True,
):
    """
    Compute a completion (B,C,D) for given A.

    Args:
      A: (N,N) feedback matrix
      k: number of IO channels. If None -> full MIMO (k=N).
      preprocessing:
         "none"  -> X = I
         "optimize" -> find diagonal similarity X=diag(x) that makes G(u) PSD with rank ~ k.
      q: RHS for diagonal similarity solve (scalar or (N,) vector).
      tol_one: tolerance for 'singular value equals 1' decisions in general MIMO.
      psd_clip: eigenvalue clip (>=0) for PSD square roots in full MIMO completion.
      return_similarity: if True, returns (B,C,D,x). else (B,C,D).

    Returns:
      B, C, D, x (if return_similarity), where x is diagonal of X used (x=ones if none).
    """
    A = np.asarray(A)
    N = A.shape[0]
    if A.shape != (N, N):
        raise ValueError("A must be square.")
    if k is None:
        k = N
    if not (1 <= k <= N):
        raise ValueError("k must satisfy 1 <= k <= N")

    # Preprocessing: compute x and A_tilde
    if preprocessing == "none" or k == N:
        x = np.ones(N, dtype=float)
        At = A
    elif preprocessing == "optimize":
        tail_norm_threshold = 1e-8  # can be parameterized as needed
        max_retries = 10
        for attempt in range(max_retries):
            u = np.random.rand(N)
            u, info = optimize_diagonal_similarity_rank_defect_scipy(
                A, k=k, u0=u,
                verbose=True,
                maxiter=2000,
                tau=1e-10,
                alpha_tail=1.0,
                beta_psd=1e-10,
                gamma_top=1e-10,
                delta_top=1e-10,
                reg_u=1e-6,
            )
            if info.get('eig_tail_norm', np.inf) < tail_norm_threshold:
                break
            if attempt == max_retries - 1:
                warnings.warn("eig_tail_norm did not go under threshold after max retries.")
        x = np.exp(u) ** 2
        At = apply_diagonal_similarity(A, x)
    else:
        raise ValueError("preprocessing must be 'none' or 'optimize'")

    # Completion in balanced coordinates (X = I)
    if k == N:
        Bt, Ct, Dt = complete_full_mimo_halmos(At, psd_clip=psd_clip)
    else:
        Bt, Ct, Dt = complete_general_mimo_svd(At, k=k, tol_one=tol_one)

    
    # Random orthogonal transformation to mix B, C, and D
    UU = pyFDN.random_orthogonal(k)
    Bt = Bt @ UU
    Dt = Dt @ UU
    VV = pyFDN.random_orthogonal(k)
    Ct = VV @ Ct
    Dt = VV @ Dt

    # Map back
    if preprocessing == "none":
        B, C, D = Bt, Ct, Dt
    else:
        B, C, D = map_back_from_similarity(Bt, Ct, Dt, x)

    if return_similarity:
        return B, C, D, x
    return B, C, D


def check_completion(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> dict:
    """
    Check balanced orthogonality/unitarity of V = [[A,B],[C,D]]:
      V^* V should be identity.

    Returns:
      dict with Frobenius norm error and max-abs error.
    """
    V = block_matrix(A, B, C, D)
    n = V.shape[1]
    E = _adj(V) @ V - _I(n, V.dtype)
    return {
        "fro_err": float(np.linalg.norm(E, ord="fro")),
        "max_abs_err": float(np.max(np.abs(E))),
    }


__all__ = [
    "block_matrix",
    "check_completion",
    "complete_fdn",
    "complete_full_mimo_halmos",
    "complete_general_mimo_svd",
    "diagonal_similarity_from_abs2_lyapunov",
    "apply_diagonal_similarity",
    "map_back_from_similarity",
    "diag_sqrt",
    "diag_inv_sqrt",
    "eig_sqrt_psd",
    "hermitize",
    "orth_error",
    "sqrtm_psd",
]