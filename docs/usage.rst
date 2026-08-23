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

FDN builds can also be exchanged as readable, versioned JSON files::

    pyFDN.save_fdn_build("reverberator.json", reverberator)
    restored = pyFDN.load_fdn_build("reverberator.json")

Preset documents
----------------

An ``FDNPreset`` keeps a baked ``FDNBuild`` together with catalog metadata and
the design choices that cannot be recovered reliably from its numbers. Unknown
design choices are left out rather than guessed::

    preset = pyFDN.FDNPreset(
        build=reverberator,
        metadata={
            "name": "small-room",
            "description": "A short, neutral room",
            "authors": ["A. Author"],
            "license": "CC0-1.0",
            "tags": ["room", "short"],
        },
        design={
            "delays": {"type": "uniform", "coprime": True},
            "feedback_matrix": {"type": "orthogonal"},
            "input_matrix": {"type": "normalised"},
            "output_matrix": {"type": "normalised"},
        },
    )
    pyFDN.save_fdn_preset("small-room.json", preset)
    restored = pyFDN.load_fdn_preset_file("small-room.json")

Metadata is an open dictionary because it is descriptive rather than part of
the sound. By convention, ``tags`` is a list of strings, so selecting presets
does not require a metadata class::

    requested = {"room", "short"}
    matches = requested <= set(restored.metadata.get("tags", []))

The numerical build, including its sample rate, is authoritative. Filter design
records may additionally carry RT or gain targets. In that case
``pyFDN.trainable_from_preset`` restores the meaningful FLAMO parameters and
checks that they reproduce the baked SOS coefficients before building a model.
