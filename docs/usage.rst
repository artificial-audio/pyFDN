=====
Usage
=====

All main functions are accessible directly from ``pyFDN``::

    import pyFDN

    feedback = pyFDN.random_orthogonal(4)
    attenuation = pyFDN.decay_to_first_order_shelf(
        1.2, 0.9, None, [100, 150, 200, 250], 48_000
    )
    gain_db = pyFDN.lin_to_db([0.5, 1.0, 2.0])

Or import specific functions::

    from pyFDN import random_orthogonal, decay_to_first_order_shelf, dss_to_ss

    feedback = random_orthogonal(4)

DSS systems and filters
-----------------------

A delay state-space (DSS) system is an FDN's per-line ``delays``, feedback
matrix ``A`` and gains ``B``/``C``/``D`` -- nothing else. :class:`~pyFDN.DSS`
holds exactly these, in the argument order of the ``dss_to_*`` functions, so a
system unpacks straight into them. Filters and the sample rate travel
alongside, as plain keyword arguments::

    dss, filters = pyFDN.fdn_dss_gallery(8, rt=1.2, rt_nyquist=0.8)
    ir = pyFDN.dss_to_td(*dss, **filters).impulse_response(48_000)
    model = pyFDN.dss_to_flamo(*dss, 48_000.0, **filters)

``filters`` maps the three hooks ``post_delay``, ``post_matrix`` and
``post_output`` to SOS banks (or ``None``). :func:`pyFDN.flamo_to_dss` returns
the same pair from a FLAMO model, and :func:`pyFDN.trainable_from_dss` takes
it back: ``trainable_from_dss(*dss, fs, **filters)``. The matrix-only
functions -- ``dss_to_impz``, ``process_dss``, ``dss_to_tf``, ``dss_to_ss``,
``dss_to_pr`` -- take ``*dss`` first and no filters; with filters, build the graph.

A filter (FIR) feedback matrix, such as a paraunitary scattering matrix, is
an operator on the feedback path rather than a DSS matrix. Keep ``A`` the
identity and put it in the ``post_matrix`` hook of either graph::

    I = np.eye(len(delays))
    graph = pyFDN.dss_to_td(delays, I, B, C, D, post_matrix=pyFDN.td.MatrixFIR(fir))
    model = pyFDN.dss_to_flamo(
        delays, I, B, C, D, fs, post_matrix=pyFDN.fir_matrix_module(fir, 2**16)
    )

Packaged tutorial resources
---------------------------

The small resources used by the browser tutorials are available through the
public API, independent of the current working directory::

    dry, fs = pyFDN.load_audio("synth_dry")

    dss = pyFDN.load_fdn_preset("colorless_N8_d1")
    decay = pyFDN.decay_to_first_order_shelf(1.5, 1.5, None, dss.delays, fs)
    wet = pyFDN.dss_to_td(*dss, post_delay=decay).process_signal(dry, squeeze=True)

    citation = pyFDN.paper_link("Allpass_Feedback_Delay_Networks")

Use ``pyFDN.available_audio()`` and ``pyFDN.available_fdn_presets()`` to list
the bundled choices. Attribution and license information for audio is returned
by ``pyFDN.audio_metadata(name)``; the presets are credited in
``pyFDN/resources/ATTRIBUTION.md``.

Saving
------

:func:`pyFDN.save_fdn` writes a :class:`~pyFDN.DSS` as readable JSON -- the
five fields as nested lists, one matrix row per line, plus the pyFDN version
that wrote it. The packaged presets are such files::

    pyFDN.save_fdn(dss, "reverberator.json")
    dss = pyFDN.load_fdn("reverberator.json")

Anything else -- a ``td`` graph, a trained FLAMO model, a ``(dss, filters)``
pair -- is pickled, together with the pyFDN and Python versions that wrote it::

    pyFDN.save_fdn(model, "trained.pkl")
    model = pyFDN.load_fdn("trained.pkl")

There is no format to maintain for pickles. When a refactor breaks one, the
load error names both versions: map a moved class in ``pyFDN.save.RENAMED`` (or
rebuild the object by hand) and rewrite the file with
:func:`pyFDN.resave_fdn`. Loading a pickle runs code from the file, so only
load pickles you or pyFDN wrote. A FLAMO model holds lambdas, which are stored
as Python bytecode, so its file loads only on the same Python minor version.
