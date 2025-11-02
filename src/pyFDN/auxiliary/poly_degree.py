from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.mag2db import mag2db

def poly_degree(polynomial: ArrayLike, var: str, tol: float | None = None) -> int:
    """Return the polynomial degree, matching ``polyDegree.m`` semantics."""

    coeffs = np.asarray(polynomial)
    if coeffs.ndim != 1:
        coeffs = np.ravel(coeffs)
    if coeffs.size == 0:
        return 0

    if tol is None:
        tol = mag2db(np.finfo(float).eps)

    coeff_db = mag2db(coeffs)
    max_coeff = np.max(coeff_db)
    mask = coeff_db - max_coeff > tol
    active = np.nonzero(mask)[0]
    if active.size == 0:
        return 0

    if var == "z^-1":
        return int(active[-1])
    if var == "z^1":
        return int(len(coeffs) - 1 - active[0])
    raise ValueError("var must be 'z^-1' or 'z^1'")
