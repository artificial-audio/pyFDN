import numpy as np
from scipy.linalg import block_diag

def dss2ss(m, A, b=None, c=None, d=None):
    """
    Convert delay state-space (DSS) FDN to standard state-space.
    
    Parameters
    ----------
    m : list or array
        Vector of delays in samples (min 3 samples).
    A : ndarray
        Feedback matrix (NxN).
    b : ndarray, optional
        Input gains (Nx1). Defaults to ones(N,1).
    c : ndarray, optional
        Output gains (1xN). Defaults to ones(1,N).
    d : ndarray, optional
        Direct gains (1x1). Defaults to np.ones((1,1)).
    
    Returns
    -------
    AA : ndarray
        State-space transition matrix.
    bb : ndarray
        State-space input gains.
    cc : ndarray
        State-space output gains.
    dd : ndarray
        State-space direct gains.
    """
    A = np.asarray(A)
    N = A.shape[0]
    
    # Default gains
    if b is None:
        b = np.ones((N, 1))
    if c is None:
        c = np.ones((1, N))
    if d is None:
        d = np.ones((1, 1))
    
    U_blocks = []
    P = np.zeros((N, 0))  # start with 0 columns
    R = np.zeros((0, N))  # start with 0 rows
    
    for it in range(N):
        # U_j: (m[it]-3) x (m[it]-3) with 1's on first superdiagonal
        size_Uj = m[it] - 3
        if size_Uj > 0:
            U_j = np.diag(np.ones(size_Uj), 1)
        else:
            U_j = np.zeros((0,0))
        U_blocks.append(U_j)
        
        # R_j: (m[it]-2) x N, last row = 1 at column it
        R_j = np.zeros((m[it]-2, N))
        R_j[-1, it] = 1
        R = np.vstack([R, R_j]) if R.size else R_j
        
        # P_j: N x (m[it]-2), first column = 1 at row it
        P_j = np.zeros((N, m[it]-2))
        if m[it]-2 > 0:
            P_j[it, 0] = 1
        P = np.hstack([P, P_j]) if P.size else P_j
    
    # Block diagonal U
    if U_blocks:
        U = block_diag(*U_blocks)
    else:
        U = np.zeros((0,0))
    
    # Construct AA
    top = np.hstack([U, np.zeros_like(R), R])
    middle = np.hstack([P, np.zeros((N, 2*N))])
    bottom = np.hstack([np.zeros_like(P), A, np.zeros_like(A)])
    AA = np.vstack([top, middle, bottom])
    
    NN = AA.shape[0]
    
    # Construct bb
    bb = np.zeros((NN, 1))
    bb[-N:] = b
    
    # Construct cc
    cc = np.zeros((1, NN))
    cc[0, -2*N:-N] = c
    
    # Direct gain
    dd = d
    
    return AA, bb, cc, dd
