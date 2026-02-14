"""Auxiliary DSP and numerical helpers."""

from .acoustics import (
    absorption_filters,
    absorption_to_t60,
    one_pole_absorption,
    rt60_to_slope,
    slope_to_rt60,
)
from .delay import matrix_delay_approximation, mgrpdelay, ms2smp
from .filters import TFMatrix, ZFIR, ZFilter, ZScalar, ZSOS, ZTF
from .math import (
    det_polynomial,
    matrix_convolution,
    matrix_polyder,
    matrix_polyval,
    negpolyder,
    outer_sum_approximation,
    poly_degree,
    polyder_rational,
    polydiag,
)
from .utils import (
    db2mag,
    ensure_3d,
    hertz2unit,
    is_bounding_curve,
    last_nonzero_indices,
    mag2db,
    pole_boundaries,
)

__all__ = [
    "TFMatrix",
    "ZFIR",
    "ZFilter",
    "ZScalar",
    "ZSOS",
    "ZTF",
    "absorption_filters",
    "absorption_to_t60",
    "db2mag",
    "det_polynomial",
    "ensure_3d",
    "hertz2unit",
    "is_bounding_curve",
    "last_nonzero_indices",
    "mag2db",
    "matrix_convolution",
    "matrix_delay_approximation",
    "matrix_polyder",
    "matrix_polyval",
    "mgrpdelay",
    "ms2smp",
    "negpolyder",
    "one_pole_absorption",
    "outer_sum_approximation",
    "pole_boundaries",
    "poly_degree",
    "polyder_rational",
    "polydiag",
    "rt60_to_slope",
    "slope_to_rt60",
]
