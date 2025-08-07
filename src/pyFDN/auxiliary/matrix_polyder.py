import numpy as np


def matrix_polyder(B, A, var='z^-1'):
    """
    Wrapper function for polynomial derivative of filter matrices.
    B: shape (order, N, M)
    A: shape (order, N, M)
    var: 'z^1' or 'z^-1'
    Returns:
        Q, P: shape (order, N, M)
    """
    order, N, M = B.shape
    Q = np.zeros((order, N, M), dtype=B.dtype)
    P = np.zeros((order, N, M), dtype=B.dtype)
    for it1 in range(N):
        for it2 in range(M):
            if var == 'z^1':
                q, p = polyder(B[:, it1, it2], A[:, it1, it2])
                Q[-len(q):, it1, it2] = q
                P[-len(p):, it1, it2] = p
            elif var == 'z^-1':
                q, p = negpolyder(B[:, it1, it2], A[:, it1, it2])
                Q[:len(q), it1, it2] = q
                P[:len(p), it1, it2] = p
            else:
                raise ValueError("var must be 'z^1' or 'z^-1'")
    return Q, P

def negpolyder(b, a):
    """
    Derivative of rational function b(z^-1)/a(z^-1) with respect to z^-1.
    This is done by reversing the coefficients, applying the derivative, and reversing back.
    """
    # Reverse coefficients for 'z^-1' convention
    b_rev = b[::-1]
    a_rev = a[::-1]
    # Use standard polyder for 'z^1'
    num, den = polyder(b_rev, a_rev)
    # Reverse back
    num = num[::-1]
    den = den[::-1]
    return num, den

def polyder(b, a):
    """
    Derivative of rational function b(z)/a(z) with respect to z.
    Returns numerator and denominator of the derivative.
    """
    # (b/a)' = (b'*a - b*a') / a^2
    db = np.polyder(b)
    da = np.polyder(a)
    num = np.polysub(np.polymul(db, a), np.polymul(b, da))
    den = np.polymul(a, a)
    return num, den
