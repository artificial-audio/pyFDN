"""Top-level package for pyFDN."""

__author__ = "Facundo Franchino"
__version__ = "0.1.0"

__all__ = [
    # filter classes
    "DFiltMatrix",
    "FeedbackDelay",
    "FilterMatrix",
    "TFMatrix",
    "ZFIR",
    "ZFilter",
    "ZSOS",
    "ZScalar",
    "ZTF",
    # acoustics
    "absorption_filters",
    "absorption_to_t60",
    "one_pole_absorption",
    "rt60_to_slope",
    "slope_to_rt60",
    # delay utilities
    "matrix_delay_approximation",
    "mgrpdelay",
    "ms_to_smp",
    # matrix generators
    "construct_cascaded_paraunitary_matrix",
    "construct_velvet_feedback_matrix",
    "is_almost_zero",
    "random_matrix_shift",
    "random_orthogonal",
    "shift_matrix",
    "shift_matrix_distribute",
    "vanilla_FDN",
    # polynomial and matrix maths
    "det_polynomial",
    "matrix_convolution",
    "matrix_polyder",
    "matrix_polyval",
    "negpolyder",
    "outer_sum_approximation",
    "poly_degree",
    "polyder_rational",
    "polydiag",
    # general utilities
    "db_to_mag",
    "ensure_3d",
    "hertz_to_unit",
    "is_bounding_curve",
    "last_nonzero_indices",
    "mag_to_db",
    "mulaw_decode",
    "mulaw_encode",
    "peak_normalize",
    "pole_boundaries",
    "skew",
    # state-space translators
    "dss_to_impz",
    "dss_to_ss",
    # fdn processing
    "process_fdn",
]

#acoustics and absorption
from .auxiliary.acoustics import (
    absorption_filters,
    absorption_to_t60,
    one_pole_absorption,
    rt60_to_slope,
    slope_to_rt60,
)

#delay utilities
from .auxiliary.delay import matrix_delay_approximation, mgrpdelay, ms_to_smp

# filter classes
from .auxiliary.filters import TFMatrix, ZFIR, ZFilter, ZScalar, ZSOS, ZTF

#polynomial and matrix maths
from .auxiliary.math import (
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

#general utilities
from .auxiliary.utils import (
    db_to_mag,
    ensure_3d,
    hertz_to_unit,
    is_bounding_curve,
    last_nonzero_indices,
    mag_to_db,
    mulaw_decode,
    mulaw_encode,
    peak_normalize,
    pole_boundaries,
    skew,
)

#matrix generators
from .generate.random_orthogonal import random_orthogonal
from .generate.random_matrix_shift import random_matrix_shift
from .generate.shift_matrix import shift_matrix
from .generate.shift_matrix_distribute import shift_matrix_distribute
from .generate.construct_cascaded_paraunitary_matrix import (
    construct_cascaded_paraunitary_matrix,
)
from .generate.construct_velvet_feedback_matrix import construct_velvet_feedback_matrix
from .generate.is_almost_zero import is_almost_zero
from .generate.vanilla_FDN import vanilla_FDN

#state-space translators
from .translate.dss_to_ss import dss_to_ss
from .translate.dss_to_impz import dss_to_impz

#fdn processing
from .process import process_fdn

#dsp components
from .dsp.filter_matrix import FilterMatrix
from .dsp.feedback_delay import FeedbackDelay
from .dsp.dfiltmatrix import DFiltMatrix
