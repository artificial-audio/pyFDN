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
    "general_char_poly",
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
    "dss_to_tf",
    "mtf_to_impz",
    # fdn processing
    "process_fdn",
    # plotting
    "plot_impulse_response_matrix",
    "plot_system_matrix",
    "plot_spectrogram",
    # SDN (scattering delay network)
    "SDN",
    # allpass FDN
    "allpass",
    "allpass_completion",
    "apply_diagonal_similarity",
    "block_matrix",
    "check_completion",
    "complete_allpass_fdn",
    "complete_fdn",
    "complete_full_mimo_halmos",
    "complete_general_mimo_svd",
    "complete_orthogonal",
    "diagonal_similarity_from_abs2_lyapunov",
    "diag_inv_sqrt",
    "diag_sqrt",
    "eig_sqrt_psd",
    "hermitize",
    "homogeneous_allpass_fdn",
    "map_back_from_similarity",
    "rand_admissible_homogeneous_allpass",
    "orth_error",
    "sqrtm_psd",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
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
    general_char_poly,
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

from .auxiliary.allpass import (
    is_uniallpass,
    is_allpass,
    is_paraunitary,
    poletti_allpass,
    series_allpass,
    nested_allpass,
)

from .generate.allpass_FDN import allpass_completion
from .generate.allpass_FDN.complete_orthogonal import complete_orthogonal
from .generate.allpass_FDN.complete_allpass_fdn import complete_allpass_fdn
from .generate.allpass_FDN.homogeneous_allpass_fdn import homogeneous_allpass_fdn
from .generate.allpass_FDN.rand_admissible_homogeneous_allpass import rand_admissible_homogeneous_allpass
from .generate.allpass_FDN.allpass_completion import (
    apply_diagonal_similarity,
    block_matrix,
    check_completion,
    complete_fdn,
    complete_full_mimo_halmos,
    complete_general_mimo_svd,
    diagonal_similarity_from_abs2_lyapunov,
    diag_inv_sqrt,
    diag_sqrt,
    eig_sqrt_psd,
    hermitize,
    map_back_from_similarity,
    orth_error,
    sqrtm_psd,
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
from .translate.dss_to_tf import dss_to_tf
from .translate.mtf_to_impz import mtf_to_impz

#fdn processing
from .process import process_fdn


#dsp components
from .dsp.filter_matrix import FilterMatrix
from .dsp.feedback_delay import FeedbackDelay
from .dsp.dfiltmatrix import DFiltMatrix

# allpass FDN (import last so they are not shadowed; expose allpass submodule and its names)
# from . import auxiliary
# allpass = auxiliary.allpass
# # Re-export from allpass so IDE "Go to definition" resolves to auxiliary/allpass.py
# is_uniallpass = allpass.is_uniallpass
# is_allpass = allpass.is_allpass
# is_paraunitary = allpass.is_paraunitary
# poletti_allpass = allpass.poletti_allpass
# series_allpass = allpass.series_allpass
# nested_allpass = allpass.nested_allpass
