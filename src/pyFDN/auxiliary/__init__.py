"""Auxiliary modules (utils, acoustics, allpass, flamo wrappers, etc.)."""

from .allpass import (
    is_allpass,
    is_paraunitary,
    is_uniallpass,
    nested_allpass,
    poletti_allpass,
    series_allpass,
)
from .audio import load_audio
from .flamo import (
    delay_module,
    flamo_process,
    flamo_time_response,
    gain_module,
    sos_filter_module,
)
from .flamo_graph import (
    FlamoFDNParameters,
    flamo_model_to_fdn_parameters,
    flamo_model_to_nodes,
    flamo_nodes_flat,
    plot_flamo_graph,
)
from .poles import reduce_conjugate_pairs
from .utils import skew

__all__ = [
    "skew",
    "gain_module",
    "delay_module",
    "sos_filter_module",
    "flamo_time_response",
    "flamo_process",
    "load_audio",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
    "flamo_model_to_nodes",
    "flamo_model_to_fdn_parameters",
    "flamo_nodes_flat",
    "FlamoFDNParameters",
    "plot_flamo_graph",
    "reduce_conjugate_pairs",
]
