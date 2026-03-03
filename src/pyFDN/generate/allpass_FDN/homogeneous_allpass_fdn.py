import numpy as np
from numpy.typing import ArrayLike

def homogeneous_allpass_fdn(
    G: ArrayLike,
    X: ArrayLike,
    *,
    verbose: bool = False,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate allpass FDN with homogeneous decay: V = [A,b;c,d] uniallpass, A = U @ G.

    Assumptions (as in the Cauchy/CORDIC-style construction you are using):
      - G is diagonal (N,N), real
      - X is diagonal (N,N), real, positive
      - Define R = G^2 X (diagonal). Build a Cauchy-like K_{ij} = 1/(p_i - r_j),
        with p = diag(X), r = diag(R)
      - Recover rank-1 factors beta, alpha such that
            inv(K) = diag(beta) @ K.T @ diag(alpha)
        and then set
            U = diag(sqrt(beta)) @ K @ diag(sqrt(alpha))
        (up to global sign conventions)

    Returns
    -------
    A : (N, N) ndarray
    b : (N, 1) ndarray
    c : (1, N) ndarray
    d : (1, 1) ndarray
    U : (N, N) ndarray
    """
    G = np.asarray(G, dtype=float)
    X = np.asarray(X, dtype=float)

    if G.ndim != 2 or X.ndim != 2 or G.shape[0] != G.shape[1] or X.shape[0] != X.shape[1]:
        raise ValueError("G and X must be square matrices.")
    if G.shape != X.shape:
        raise ValueError("G and X must have the same shape.")
    N = G.shape[0]

    # basic diagonal checks (soft)
    if np.linalg.norm(G - np.diag(np.diag(G)), ord="fro") > 1e-10:
        raise ValueError("G must be diagonal for this construction.")
    if np.linalg.norm(X - np.diag(np.diag(X)), ord="fro") > 1e-10:
        raise ValueError("X must be diagonal for this construction.")
    if np.any(np.diag(X) <= 0):
        raise ValueError("X must be positive diagonal.")

    xx = np.diag(X).copy()
    rr = np.diag((G @ G @ X)).copy()   # = diag(G^2 X)

    # Cauchy-like orthogonal U: K_ij = 1/(p_i - r_j)
    denom = xx.reshape(-1, 1) - rr.reshape(1, -1)
    if np.min(np.abs(denom)) < tol:
        raise ValueError("Cauchy denominator p_i - r_j too small; construction ill-conditioned.")
    K = 1.0 / denom

    # beta_alpha should be rank-1: beta_alpha = inv(K) ./ K.T  (elementwise division)
    invK = np.linalg.inv(K)
    beta_alpha = invK / K.T

    # Extract rank-1 factors from leading singular triplet:
    # beta_alpha ≈ (sqrt(sigma) * u) (sqrt(sigma) * v)^T
    U_svd, s_svd, Vh_svd = np.linalg.svd(beta_alpha, full_matrices=False)
    sigma = s_svd[0]
    alpha = np.sqrt(sigma) * U_svd[:, 0]       # (N,)
    beta = np.sqrt(sigma) * Vh_svd[0, :]     # (N,) since Vh row is v^T

    # Fix global sign ambiguity for consistency
    alpha = np.abs(alpha)
    beta = np.abs(beta)

    # Build U via diagonal scalings
    U = np.diag(np.sqrt(beta)) @ K @ np.diag(np.sqrt(alpha))
    A = U @ G

    # Choose d (sign convention varies; keep your original intent)
    # For orthogonal U, det(U)=±1, det(A)=det(U)*prod(diag(G)).
    d = np.array([[((-1) ** N) * np.linalg.det(A)]])

    b = np.sqrt(beta).reshape(-1, 1)
    b_hat = np.sqrt(alpha).reshape(-1, 1)

    # Solve for ambiguity between beta and alpha
    factor = b_hat / (U.T @ b)
    factor = np.sqrt(np.mean(factor))
    b = b * factor
    b_hat = b_hat / factor

    # Coupling equation for X-uniallpass (scalar output):
    c = (-np.linalg.inv(X) @ np.linalg.inv(A) @ b * d).T

    if verbose:
        from pyFDN.auxiliary.allpass import is_uniallpass, is_allpass
        print("||U U^T - I||_F =", np.linalg.norm(U @ U.T - np.eye(N)))
        print("||X U - U R - b b^T U||_F =", np.linalg.norm(X @ U - U @ (G @ G @ X) - b @ b.T @ U))
        is_a1, XX = is_uniallpass(A, b, c, d)
        print("isUniallpass:", is_a1, "XX=\n", XX)
        delays = 2 ** np.arange(N)
        is_a2, den, num = is_allpass(A, b, c, d, delays)
        print("isAllpass (delays 2^0..2^(N-1)):", is_a2)

    return A, b, c, d, U
