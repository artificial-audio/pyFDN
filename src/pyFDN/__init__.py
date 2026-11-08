"""Top-level package for pyFDN."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__author__ = "Facundo Franchino"
__version__ = "0.6.0"

__all__ = [
    # acoustics
    "echo_density",
    "estimate_initial_level_bands",
    "estimate_rt_bands",
    "edc",
    "octave_band_filterbank",
    "octave_bands",
    "rt_to_gain_per_sample",
    "rt_to_slope",
    "slope_amplitude_to_level",
    "slope_to_rt",
    "sos_gain_per_sample_curves",
    # delay utilities
    "matrix_delay_approximation",
    "matrix_group_delay",
    "ms_to_samples",
    "flamo_delay_feedback_matrix",
    "swap_flamo_recursion_paths",
    "flamo_to_impz",
    "flamo_process",
    # building a FLAMO FDN graph from numpy values
    "assemble_fdn_core",
    "wrap_fdn_shell",
    "gain_module",
    "delay_module",
    "matrix_module",
    "fir_matrix_module",
    "sos_filter_module",
    "AttenuationFilter",
    "OutputEQ",
    "audio_metadata",
    "available_audio",
    "load_audio",
    # packaged references and presets, saving
    "paper_link",
    "paper_reference",
    "available_fdn_presets",
    "load_fdn_preset",
    "save_fdn",
    "load_fdn",
    "resave_fdn",
    # matrix generators
    "allpass_in_fdn",
    "anderson_matrix",
    "complete_orthogonal",
    "cascaded_paraunitary_matrix",
    "paraunitary_from_elementals",
    "fdn_matrix_gallery",
    "fdn_dss_gallery",
    "fdn_system_gallery",
    "filter_matrix_gallery",
    "DSS",
    "FEEDBACK_MATRIX_TYPES",
    "FILTER_MATRIX_TYPES",
    "SYSTEM_TYPES",
    "householder_matrix",
    "kronecker_angles",
    "kronecker_matrix",
    "kronecker_transform",
    "nearest_orthogonal",
    "nearest_sign_agnostic_orthogonal",
    "random_orthogonal",
    "reflection_kernel",
    "rotation_kernel",
    "rotation_matrix_from_angles",
    "sample_delay_lengths",
    "schroeder_reverberator",
    "tiny_rotation_matrix",
    # eq
    "EQDesign",
    "decay_to_first_order_shelf",
    "decay_to_graphic_eq",
    "decay_to_one_pole",
    "decay_to_two_stage",
    "gain_to_bounded_graphic_eq",
    "gain_to_first_order_shelf",
    "gain_to_graphic_eq",
    "gain_to_one_pole",
    "gain_to_two_stage",
    "graphic_eq_design_matrix",
    "probe_sos",
    # polynomial and matrix maths
    "adjugate_polynomial",
    "adjugate",
    "det_polynomial",
    "general_char_poly",
    "interpolate_orthogonal",
    "is_orthogonal",
    "is_unilossless",
    "loop_transfer_function",
    "matrix_convolution",
    "outer_sum_approximation",
    # general utilities
    "fade_out",
    "hertz_to_unit",
    "hertz_to_rad",
    "rad_to_hertz",
    "is_bounding_curve",
    "lin_to_db",
    "max_corr",
    "sq_to_db",
    "mulaw_encode",
    "peak_normalize",
    "pole_boundaries",
    # state-space translators
    "dss_to_flamo",
    "flamo_to_dss",
    "dss_to_impz",
    "dss_to_pr",
    "flamo_to_pr",
    "flamo_decompose_for_pr",
    "FlamoDecompositionForPR",
    "dss_to_ss",
    "dss_to_td",
    "dss_to_tf",
    "impz_to_res",
    "tf_to_impz",
    "pr_to_impz",
    "PoleResidue",
    "seed_poles",
    "track_poles",
    "PoleTracks",
    # fdn processing
    "process_dss",
    # training
    "trainable_fdn",
    "trainable_from_dss",
    "LOSSLESS_ALIAS_DECAY_DB",
    "Trainable",
    "train_fdn",
    "TrainLog",
    # training: what a loss sees
    "Response",
    "model_response",
    "impulse_excitation",
    "param",
    "params",
    "ParamRef",
    # training: losses
    "Loss",
    "ResponseLoss",
    "ParameterLoss",
    "Match",
    "FlatMagnitude",
    "AsymmetricFlatMagnitude",
    "FlatSpectrogram",
    "SpectralFlatness",
    "MatchMagnitude",
    "MatchPhase",
    "MatchPhaseSpectrogram",
    "MatchMelMagnitude",
    "MatchSpectrogram",
    "MatchMelSpectrogram",
    "MatchImpulseResponse",
    "MatchEnergyDecay",
    "MatchCumulativeEnergy",
    "MatchMelEnergyDecayRelief",
    "MatchEchoDensity",
    "MatchEnvelope",
    "Energy",
    "Sparsity",
    "L1",
    "L2",
    # features
    "mimo_rir_eigenvalues_per_frequency",
    # plotting
    "animate",
    "plot_db_per_sample",
    "plot_edc",
    "plot_fdn_parameter",
    "plot_impulse_response",
    "plot_impulse_response_matrix",
    "plot_matrix",
    "plot_matrix_grid",
    "plot_system_matrix",
    "plot_spectrogram",
    # notebook display (marimo)
    "labeled_audio",
    # FLAMO graph
    "plot_flamo_graph",
    # time-domain graph engine
    "td",
    # SDN (scattering delay network)
    "SDN",
    "plot_sdn",
    "sdn_to_flamo",
    # allpass FDN
    "complete_fdn",
    "complete_full_mimo_halmos",
    "complete_general_mimo_svd",
    "homogeneous_allpass_fdn",
    "random_admissible_homogeneous_allpass",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
]

# acoustics and absorption
# Time-domain graph engine (pyFDN.td operators and connectors).
from . import td  # noqa: E402
from .auxiliary.acoustics import (
    echo_density,
    edc,
    estimate_initial_level_bands,
    estimate_rt_bands,
    octave_band_filterbank,
    octave_bands,
    rt_to_gain_per_sample,
    rt_to_slope,
    slope_amplitude_to_level,
    slope_to_rt,
    sos_gain_per_sample_curves,
)
from .auxiliary.allpass import (
    is_allpass,
    is_paraunitary,
    is_uniallpass,
)
from .auxiliary.audio import audio_metadata, available_audio, load_audio

# delay utilities
from .auxiliary.delay import (
    matrix_delay_approximation,
    matrix_group_delay,
    ms_to_samples,
)
from .auxiliary.flamo import (
    assemble_fdn_core,
    delay_module,
    fir_matrix_module,
    flamo_delay_feedback_matrix,
    flamo_process,
    flamo_to_impz,
    gain_module,
    matrix_module,
    sos_filter_module,
    swap_flamo_recursion_paths,
    wrap_fdn_shell,
)
from .auxiliary.flamo_graph import (
    plot_flamo_graph,
)
from .auxiliary.marimo_utils import labeled_audio

# polynomial and matrix maths
from .auxiliary.math import (
    adjugate,
    adjugate_polynomial,
    det_polynomial,
    general_char_poly,
    interpolate_orthogonal,
    is_orthogonal,
    is_unilossless,
    loop_transfer_function,
    matrix_convolution,
    outer_sum_approximation,
)

# plotting
from .auxiliary.plot import (
    animate,
    plot_db_per_sample,
    plot_edc,
    plot_fdn_parameter,
    plot_impulse_response,
    plot_impulse_response_matrix,
    plot_matrix,
    plot_matrix_grid,
    plot_spectrogram,
    plot_system_matrix,
)
from .auxiliary.poles import is_bounding_curve, pole_boundaries

# general utilities
from .auxiliary.utils import (
    fade_out,
    hertz_to_rad,
    hertz_to_unit,
    lin_to_db,
    max_corr,
    mulaw_encode,
    peak_normalize,
    rad_to_hertz,
    sq_to_db,
)
from .dss import DSS
from .eq import (
    EQDesign,
    decay_to_first_order_shelf,
    decay_to_graphic_eq,
    decay_to_one_pole,
    decay_to_two_stage,
    gain_to_bounded_graphic_eq,
    gain_to_first_order_shelf,
    gain_to_graphic_eq,
    gain_to_one_pole,
    gain_to_two_stage,
    graphic_eq_design_matrix,
    probe_sos,
)
from .generate.allpass_fdn.allpass_completion import (
    complete_fdn,
    complete_full_mimo_halmos,
    complete_general_mimo_svd,
)
from .generate.allpass_fdn.homogeneous_allpass_fdn import homogeneous_allpass_fdn
from .generate.allpass_fdn.random_admissible_homogeneous_allpass import (
    random_admissible_homogeneous_allpass,
)
from .generate.fdn_dss_gallery import fdn_dss_gallery
from .generate.fdn_matrix_gallery import (
    FEEDBACK_MATRIX_TYPES,
    FILTER_MATRIX_TYPES,
    SYSTEM_TYPES,
    fdn_matrix_gallery,
    fdn_system_gallery,
    filter_matrix_gallery,
)
from .generate.kronecker_matrix import (
    kronecker_angles,
    kronecker_matrix,
    kronecker_transform,
    reflection_kernel,
    rotation_kernel,
)

# matrix generators
from .generate.orthogonal import (
    anderson_matrix,
    complete_orthogonal,
    householder_matrix,
    nearest_orthogonal,
    nearest_sign_agnostic_orthogonal,
    random_orthogonal,
    rotation_matrix_from_angles,
    tiny_rotation_matrix,
)
from .generate.paraunitary import (
    cascaded_paraunitary_matrix,
    paraunitary_from_elementals,
)
from .generate.sample_delay_lengths import sample_delay_lengths
from .generate.sdn import SDN, plot_sdn, sdn_to_flamo
from .generate.structures import (
    allpass_in_fdn,
    nested_allpass,
    poletti_allpass,
    schroeder_reverberator,
    series_allpass,
)

# fdn processing
from .process import process_dss
from .references import paper_link, paper_reference
from .save import (
    available_fdn_presets,
    load_fdn,
    load_fdn_preset,
    resave_fdn,
    save_fdn,
)

# state-space translators
from .translate.dss_to_flamo import dss_to_flamo
from .translate.dss_to_impz import dss_to_impz
from .translate.dss_to_pr import dss_to_pr
from .translate.dss_to_ss import dss_to_ss
from .translate.dss_to_td import dss_to_td
from .translate.dss_to_tf import dss_to_tf
from .translate.flamo_to_dss import flamo_to_dss
from .translate.impz_to_res import impz_to_res
from .translate.pr_to_impz import PoleResidue, pr_to_impz
from .translate.tf_to_impz import tf_to_impz

# The training API and the FLAMO pole finder need torch and flamo. They are
# imported on first access, so ``import pyFDN`` stays light for NumPy users.
if TYPE_CHECKING:
    from .train import (
        L1,
        L2,
        LOSSLESS_ALIAS_DECAY_DB,
        AsymmetricFlatMagnitude,
        AttenuationFilter,
        Energy,
        FlatMagnitude,
        FlatSpectrogram,
        Loss,
        Match,
        MatchCumulativeEnergy,
        MatchEchoDensity,
        MatchEnergyDecay,
        MatchEnvelope,
        MatchImpulseResponse,
        MatchMagnitude,
        MatchMelEnergyDecayRelief,
        MatchMelMagnitude,
        MatchMelSpectrogram,
        MatchPhase,
        MatchPhaseSpectrogram,
        MatchSpectrogram,
        OutputEQ,
        ParameterLoss,
        ParamRef,
        Response,
        ResponseLoss,
        Sparsity,
        SpectralFlatness,
        Trainable,
        TrainLog,
        impulse_excitation,
        mimo_rir_eigenvalues_per_frequency,
        model_response,
        param,
        params,
        train_fdn,
        trainable_fdn,
        trainable_from_dss,
    )
    from .translate.flamo_to_pr import (
        FlamoDecompositionForPR,
        flamo_decompose_for_pr,
        flamo_to_pr,
    )
    from .translate.track_poles import PoleTracks, seed_poles, track_poles

_LAZY_MODULES = {
    "L1": ".train",
    "L2": ".train",
    "LOSSLESS_ALIAS_DECAY_DB": ".train",
    "AsymmetricFlatMagnitude": ".train",
    "AttenuationFilter": ".train",
    "Energy": ".train",
    "FlatMagnitude": ".train",
    "FlatSpectrogram": ".train",
    "Loss": ".train",
    "Match": ".train",
    "MatchCumulativeEnergy": ".train",
    "MatchEchoDensity": ".train",
    "MatchEnergyDecay": ".train",
    "MatchEnvelope": ".train",
    "MatchImpulseResponse": ".train",
    "MatchMagnitude": ".train",
    "MatchMelEnergyDecayRelief": ".train",
    "MatchMelMagnitude": ".train",
    "MatchMelSpectrogram": ".train",
    "MatchPhase": ".train",
    "MatchPhaseSpectrogram": ".train",
    "MatchSpectrogram": ".train",
    "OutputEQ": ".train",
    "ParameterLoss": ".train",
    "ParamRef": ".train",
    "Response": ".train",
    "ResponseLoss": ".train",
    "Sparsity": ".train",
    "SpectralFlatness": ".train",
    "Trainable": ".train",
    "TrainLog": ".train",
    "trainable_fdn": ".train",
    "impulse_excitation": ".train",
    "mimo_rir_eigenvalues_per_frequency": ".train",
    "model_response": ".train",
    "param": ".train",
    "params": ".train",
    "train_fdn": ".train",
    "trainable_from_dss": ".train",
    "FlamoDecompositionForPR": ".translate.flamo_to_pr",
    "flamo_decompose_for_pr": ".translate.flamo_to_pr",
    "flamo_to_pr": ".translate.flamo_to_pr",
    "PoleTracks": ".translate.track_poles",
    "seed_poles": ".translate.track_poles",
    "track_poles": ".translate.track_poles",
}


def __getattr__(name: str) -> Any:
    module = _LAZY_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_MODULES))
