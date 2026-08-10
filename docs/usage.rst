=====
Usage
=====

All main functions are accessible directly from ``pyFDN``::

    import pyFDN

    feedback = pyFDN.random_orthogonal(4)
    absorption = pyFDN.first_order_absorption(1.2, 0.9, [100, 150, 200, 250], 48_000)
    gain_db = pyFDN.lin_to_db([0.5, 1.0, 2.0])

Or import specific functions::

    from pyFDN import random_orthogonal, first_order_absorption, dss_to_ss

    feedback = random_orthogonal(4)

Packaged tutorial resources
---------------------------

The small resources used by the browser tutorials are available through the
public API, independent of the current working directory::

    dry, fs = pyFDN.load_audio("synth_dry")

    preset = pyFDN.load_fdn_preset("colorless_N8_d1")
    reverberator = pyFDN.build_set_decay(preset, 1.5)

    citation = pyFDN.paper_link("Allpass_Feedback_Delay_Networks")

Use ``pyFDN.available_audio()`` and ``pyFDN.available_fdn_presets()`` to list
the bundled choices. Attribution and license information for audio is returned
by ``pyFDN.audio_metadata(name)``.
