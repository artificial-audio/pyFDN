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
    "ms2smp",
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
    "skew",
    "db2mag",
    "ensure_3d",
    "hertz2unit",
    "is_bounding_curve",
    "last_nonzero_indices",
    "mag2db",
    "mulaw_decode",
    "mulaw_encode",
    "peak_normalize",
    "pole_boundaries",
    # state-space translators
    "dss2impz",
    "dss2ss",
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
from .auxiliary.delay import matrix_delay_approximation, mgrpdelay, ms2smp

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
    skew,
    db2mag,
    ensure_3d,
    hertz2unit,
    is_bounding_curve,
    last_nonzero_indices,
    mag2db,
    mulaw_decode,
    mulaw_encode,
    peak_normalize,
    pole_boundaries,
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
from .translate.dss2ss import dss2ss
from .translate.dss2impz import dss2impz

#fdn processing
from .process import process_fdn

#dsp components
from .dsp.filter_matrix import FilterMatrix
from .dsp.feedback_delay import FeedbackDelay
from .dsp.dfiltmatrix import DFiltMatrix
