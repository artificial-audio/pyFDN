===============================
Training losses: design notes
===============================

Rationale and empirical findings behind the losses in :mod:`pyFDN.train.losses`.
These are measurements from particular runs, kept out of the API docstrings so
those stay short and stable. Treat the numbers as evidence for a design choice,
not as a contract — nothing tests them, and they are FDN- and seed-dependent.


Flatness on the training grid depends on ``nfft``
=================================================

:class:`~pyFDN.FlatMagnitude` measures ``|H|`` at the model's DFT bins, so its
frequency resolution *is* the model's ``nfft``. The truncation to ``nfft``
samples is a rectangular window, so the peak-to-median range of ``|H|`` grows
with ``nfft`` (13 dB at ``2**12``, 30 dB at ``2**15`` for a lossless FDN), and
the mean-squared error weights the tallest modes ever more heavily. On the
8-line lossless FDN of ``example_train_colorless_FDN`` the flatness reached is
0.61 at ``2**12``, 0.67 at ``2**13``, 0.70 at ``2**14``. A multi-resolution
loss (:class:`~pyFDN.FlatSpectrogram`), whose analysis windows set their own
resolution, is the way out: scoring one lossless FDN at ``nfft`` of ``2**12``,
``2**13`` and ``2**14`` spreads ``FlatMagnitude`` by 6.9× and ``FlatSpectrogram``
by 1.5×.

The optimization crosses long plateaus on ``FlatMagnitude``; ``train_fdn``'s
default ``patience=10`` stops inside one. Raise it (~100) for a converged fit.


Asymmetric flatness: peaks over dips
====================================

:class:`~pyFDN.AsymmetricFlatMagnitude` raises the two sides of the deviation
from flat to different powers so peaks cost far more than dips. The exponent,
not a weight, is what makes it bite: at ``peak_power=4`` a peak twice as tall
costs sixteen times as much. Flat stays the unique minimum at every
``peak_power`` (normalizing by RMS forces ``<(1 + d)^2> = 1``), and the loss is
gain-invariant.

It is deliberately **not** in decibels: ``∂dB/∂|H| ∝ 1/|H|``, so in dB the
deepest nulls dominate the gradient however lightly weighted, and a dB version
stalls within ~300 steps at higher peaks than ``FlatMagnitude`` reaches. In
linear magnitude the gradient is ``∝ p·(d⁺)^(p-1)``, largest at the tallest
peaks, and a dip is bounded at ``d = -1``.

Measured on four 8-line lossless FDNs at ``nfft=2**14``, up to 2000 Adam steps
at ``lr=1e-2``, ``patience=400``, then given homogeneous decay to measure
colouration on a fine grid. Tallest mode is in dB above the response's median:

==================  =============  =========
objective           tallest mode   flatness
==================  =============  =========
``FlatMagnitude``   17.5 dB        0.32
``peak_power=2``    17.1 dB        0.32
``peak_power=3``    15.9 dB        0.35
``peak_power=4``    13.7 dB        0.46
``peak_power=6``    14.2 dB        0.36
==================  =============  =========

The reliable claim is about the *peak*; the dip and plain flatness vary a lot
FDN to FDN. The cost is steps and steadiness: the gradient vanishes like
``(d⁺)^(p-1)`` near the optimum (``peak_power=4`` averaged 1600 steps against
640 at 2), and the seed-to-seed spread grows with ``p`` (±2.1 dB at 4 against
±1.1 dB at 2). The advantage is not unconditional — at ``nfft=2**13`` it
disappears. Measure your own case before assuming a higher exponent helps. Loss
values are not comparable across ``peak_power`` or with ``FlatMagnitude``;
compare the responses.


Multi-resolution flatness: average over frames first
====================================================

:class:`~pyFDN.FlatSpectrogram` averages the short-time magnitudes over frames
into a Welch estimate *before* measuring flatness. Asking each individual frame
to be flat instead is actively harmful: an isolated echo inside a short frame
already has a flat frame spectrum, so that objective rewards an impulsive,
comb-filtered IR — trained on the 8-line FDN of ``example_train_colorless_FDN``,
per-frame flatness drives the result's spectral flatness *below* its random
start, while the time-averaged form matches ``FlatMagnitude``'s flatness with a
denser feedback matrix.


Energy decay vs. spectrogram distance
=====================================

:class:`~pyFDN.MatchEnergyDecay` compares octave-band Schroeder curves. A
magnitude spectrogram distance is not a substitute for fitting a decay: it
compares two signals frame by frame, and two rooms with identical decay have
uncorrelated fine structure, so predicting *silence* scores better there than
predicting the right amount of the wrong detail. On a 16-line FDN fitted to a
2.4 s hall, the mel spectrogram distance is minimized by an FDN whose RT is 40 %
short of the measurement; ``MatchEnergyDecay`` is minimized within a few percent.


Doubly-cumulated energy
=======================

:class:`~pyFDN.MatchCumulativeEnergy` integrates the short-time power spectrum
twice — backwards in time and along frequency — and scores the RMS difference of
the two surfaces after a compressive power.

**Why cumulate twice.** The time direction is Schroeder backward integration
(the reason ``MatchEnergyDecay`` exists). The frequency direction replaces
splitting into octave bands: band edges are an arbitrary quantization a fit can
satisfy on average while getting the shape wrong, whereas a cumulative sum is
the limit of ever-finer bands and is monotone and smooth in both axes, which a
gradient values. Read down the ``t = 0`` edge and you have the integrated
spectrum (colour); read across the ``f = 0`` edge and you have the full-band
decay; the interior ties them together band by band.

**Compression, not decibels.** The surface spans the whole dynamic range of the
decay (six orders of magnitude), so a plain MSE would see only the first frames.
Raising the normalized surface to a fractional ``power`` (0.5 default) compresses
that range while staying bounded, its gradient ``∝ x^(p-1)`` finite everywhere
the floor allows. A logarithm is worse: it turns the silence *below* the response
into an unbounded penalty dominated by whichever bin is nearest zero.

**Frequency direction outweighs power.** Fitting the 16-line FDN of
``example_train_fdn_to_rir`` to a 2.4 s hall (2.8 s at 63 Hz) for 300 Adam
steps, from a flat 1 s decay, scoring the render against the unseen measurement:

==================  =========  ==============  ============  ============
``frequency``       ``power``  mean RT error   level shape   RT at 63 Hz
==================  =========  ==============  ============  ============
``"descending"``    0.5        16.8 %          2.56 dB       0.26 s
``"descending"``    0.25       19.0 %          1.46 dB       0.44 s
``"ascending"``     0.5        12.8 %          1.12 dB       2.35 s
``"both"``          0.5        10.3 %          0.88 dB       2.17 s
==================  =========  ==============  ============  ============

Cumulating downwards alone leaves the bottom octave with almost no gradient and
the fit abandons it; ``"both"`` recovers it and is best overall. A longer
analysis window does not help (15.4 % at ``window=4096``), which identifies the
cause as weighting rather than frequency resolution. The default is
``"descending"`` (the plain reading of "energy above this frequency"), but reach
for ``"both"`` when the fit has to find a decay it was not given.
