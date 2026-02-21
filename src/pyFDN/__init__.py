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
    "echo_density",
    "absorption_to_rt",
    "edc",
    "one_pole_absorption",
    "rt_to_gain_per_sample",
    "rt_to_slope",
    "slope_to_rt",
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
    "tiny_rotation_matrix",
    "vanilla_FDN",
    # polynomial and matrix maths
    "det_polynomial",
    "interpolate_orthogonal",
    "is_orthogonal",
    "matrix_convolution",
    "matrix_polyder",
    "matrix_polyval",
    "matrix_sqrt",
    "negpolyder",
    "outer_sum_approximation",
    "poly_degree",
    "polyder_rational",
    "polydiag",
    # general utilities
    "db_to_lin",
    "db_to_sq",
    "ensure_3d",
    "hertz_to_unit",
    "is_bounding_curve",
    "last_nonzero_indices",
    "lin_to_db",
    "sq_to_db",
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
    # plotting
    "plot_impulse_response_matrix",
    "plot_system_matrix",
    "plot_spectrogram",
    # SDN (scattering delay network)
    "SDN",
]

#acoustics and absorption
from .auxiliary.acoustics import (
    absorption_filters,
    echo_density,
    absorption_to_rt,
    edc,
    one_pole_absorption,
    rt_to_gain_per_sample,
    rt_to_slope,
    slope_to_rt,
)

#delay utilities
from .auxiliary.delay import matrix_delay_approximation, mgrpdelay, ms_to_smp

# filter classes
from .auxiliary.filters import TFMatrix, ZFIR, ZFilter, ZScalar, ZSOS, ZTF

#tiny rotation matrix
from .auxiliary.tiny_rotation_matrix import tiny_rotation_matrix

#polynomial and matrix maths
from .auxiliary.math import (
    det_polynomial,
    interpolate_orthogonal,
    is_orthogonal,
    matrix_convolution,
    matrix_polyder,
    matrix_polyval,
    matrix_sqrt,
    negpolyder,
    outer_sum_approximation,
    poly_degree,
    polyder_rational,
    polydiag,
)

#general utilities
from .auxiliary.utils import (
    db_to_lin,
    db_to_sq,
    ensure_3d,
    hertz_to_unit,
    is_bounding_curve,
    last_nonzero_indices,
    lin_to_db,
    mulaw_decode,
    mulaw_encode,
    peak_normalize,
    pole_boundaries,
    sq_to_db,
    skew,
)

#plotting
from .auxiliary.plot import (
    plot_impulse_response_matrix,
    plot_system_matrix,
    plot_spectrogram,
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
from .generate.SDN import SDN

#state-space translators
from .translate.dss_to_ss import dss_to_ss
from .translate.dss_to_impz import dss_to_impz

#fdn processing
from .process import process_fdn


#dsp components
from .dsp.filter_matrix import FilterMatrix
from .dsp.feedback_delay import FeedbackDelay
from .dsp.dfiltmatrix import DFiltMatrix
