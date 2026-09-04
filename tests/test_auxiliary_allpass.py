"""Tests for auxiliary.allpass module."""

import numpy as np
import pytest

from pyFDN.auxiliary.allpass import is_uniallpass, poletti_allpass
from pyFDN.generate.allpass_in_fdn import allpass_in_fdn
from pyFDN.generate.random_orthogonal import random_orthogonal


def test_is_uniallpass_poletti():
    A, B, C, D = poletti_allpass(0.7, random_orthogonal(8))
    is_ua, P = is_uniallpass(A, B, C, D)
    assert is_ua
    np.testing.assert_allclose(P, np.diag(np.diag(P)), atol=1e-9)


def test_is_uniallpass_rejects_lossless_feedback_matrix():
    """A lossless A has no finite Lyapunov solution; report False, don't solve.

    The allpass-in-FDN feedback matrix has all eigenvalues on the unit circle,
    which makes ``A P A' - P = -B B'`` singular.  Solving it anyway is a
    coin flip between an ill-conditioned answer and a LinAlgError.
    """
    N = 4
    rng = np.random.default_rng(0)
    g = rng.uniform(-0.8, 0.8, N)
    b = rng.standard_normal((N, 1)) / np.sqrt(N)
    c = rng.standard_normal((1, N)) / np.sqrt(N)
    A, B, C, D = allpass_in_fdn(g, random_orthogonal(N), b, c, 0.0)

    assert np.max(np.abs(np.linalg.eigvals(A))) == pytest.approx(1.0)
    with pytest.warns(UserWarning, match="spectral radius"):
        is_ua, P = is_uniallpass(A, B, C, D)
    assert not is_ua
    assert np.isnan(P).all()
