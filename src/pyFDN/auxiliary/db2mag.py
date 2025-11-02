from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

def db2mag(db: ArrayLike) -> np.ndarray:
    """Convert decibel values to linear magnitude."""

    db_arr = np.asarray(db, dtype=float)
    return np.power(10.0, db_arr / 20.0)