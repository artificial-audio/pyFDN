import numpy as np
from types import SimpleNamespace
from pyFDN.auxiliary.pole_boundaries import pole_boundaries

def test_pole_boundaries_basic():
    delays = np.array([1, 2])
    b = np.ones((2, 1, 4))
    a = np.ones((2, 1, 4))
    absorption = SimpleNamespace(b=b, a=a)
    feedback_matrix = np.ones((2, 2, 4))
    fs = 48000
    MinCurve, MaxCurve, f = pole_boundaries(delays, absorption, feedback_matrix, fs, nfft=8)
    assert MinCurve.shape == (8,)
    assert MaxCurve.shape == (8,)
    assert f.shape == (8,)