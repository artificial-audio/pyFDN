"""Public API manifest for :mod:`pyFDN`."""

from __future__ import annotations

_EXPORT_ITEMS: tuple[tuple[str, str], ...] = (
    ("Biquads", "pyFDN.recursive.biquads"),
    ("DFiltMatrix", "pyFDN.dsp.dfiltmatrix"),
    ("Delay", "pyFDN.recursive.delay_lines"),
    ("DelayRead", "pyFDN.recursive.delay_lines"),
    ("DelayWrite", "pyFDN.recursive.delay_lines"),
    ("FeedbackDelay", "pyFDN.dsp.feedback_delay"),
    ("FeedbackMix", "pyFDN.recursive.feedback_mix"),
    ("FilterMatrix", "pyFDN.dsp.filter_matrix"),
    ("IIRFilterState", "pyFDN.dsp.filter_matrix"),
    ("InputTap", "pyFDN.recursive.input_tap"),
    ("OutputTap", "pyFDN.recursive.output_tap"),
    ("RecursionCore", "pyFDN.recursive.core"),
    ("SOSFilterState", "pyFDN.dsp.filter_matrix"),
    ("Stage", "pyFDN.recursive.stage"),
    ("TFMatrix", "pyFDN.auxiliary.filters"),
    ("ZFIR", "pyFDN.auxiliary.filters"),
    ("ZFilter", "pyFDN.auxiliary.filters"),
    ("ZSOS", "pyFDN.auxiliary.filters"),
    ("ZScalar", "pyFDN.auxiliary.filters"),
    ("ZTF", "pyFDN.auxiliary.filters"),
    ("absorption_filters", "pyFDN.auxiliary.acoustics"),
    ("absorption_to_t60", "pyFDN.auxiliary.acoustics"),
    (
        "construct_cascaded_paraunitary_matrix",
        "pyFDN.generate.construct_cascaded_paraunitary_matrix",
    ),
    (
        "construct_velvet_feedback_matrix",
        "pyFDN.generate.construct_velvet_feedback_matrix",
    ),
    ("db2mag", "pyFDN.auxiliary.utils"),
    ("det_polynomial", "pyFDN.auxiliary.math"),
    ("dss2impz", "pyFDN.translate.dss2impz"),
    ("dss2ss", "pyFDN.translate.dss2ss"),
    ("ensure_3d", "pyFDN.auxiliary.utils"),
    ("hertz2unit", "pyFDN.auxiliary.utils"),
    ("is_almost_zero", "pyFDN.generate.is_almost_zero"),
    ("is_bounding_curve", "pyFDN.auxiliary.utils"),
    ("last_nonzero_indices", "pyFDN.auxiliary.utils"),
    ("mag2db", "pyFDN.auxiliary.utils"),
    ("matrix_convolution", "pyFDN.auxiliary.math"),
    ("matrix_delay_approximation", "pyFDN.auxiliary.delay"),
    ("matrix_polyder", "pyFDN.auxiliary.math"),
    ("matrix_polyval", "pyFDN.auxiliary.math"),
    ("mgrpdelay", "pyFDN.auxiliary.delay"),
    ("ms2smp", "pyFDN.auxiliary.delay"),
    ("negpolyder", "pyFDN.auxiliary.math"),
    ("one_pole_absorption", "pyFDN.auxiliary.acoustics"),
    ("outer_sum_approximation", "pyFDN.auxiliary.math"),
    ("pole_boundaries", "pyFDN.auxiliary.utils"),
    ("poly_degree", "pyFDN.auxiliary.math"),
    ("polyder_rational", "pyFDN.auxiliary.math"),
    ("polydiag", "pyFDN.auxiliary.math"),
    ("process_fdn", "pyFDN.process"),
    ("random_matrix_shift", "pyFDN.generate.random_matrix_shift"),
    ("random_orthogonal", "pyFDN.generate.random_orthogonal"),
    ("rt60_to_slope", "pyFDN.auxiliary.acoustics"),
    ("shift_matrix", "pyFDN.generate.shift_matrix"),
    ("shift_matrix_distribute", "pyFDN.generate.shift_matrix_distribute"),
    ("slope_to_rt60", "pyFDN.auxiliary.acoustics"),
)

_seen: set[str] = set()
for _name, _module in _EXPORT_ITEMS:
    if _name in _seen:
        raise RuntimeError(f"Duplicate public API symbol '{_name}' in manifest")
    _seen.add(_name)

EXPORT_MAP: dict[str, str] = dict(_EXPORT_ITEMS)
EXPORTS: tuple[str, ...] = tuple(name for name, _ in _EXPORT_ITEMS)
