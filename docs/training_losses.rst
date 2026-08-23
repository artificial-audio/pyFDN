===============================
Training losses: design notes
===============================

Why the losses in :mod:`pyFDN.train.losses` are shaped the way they are. The
reasoning is qualitative on purpose: what is stated here is a design choice and
its rationale, not a benchmark. Behaviour is FDN- and seed-dependent, so measure
your own case before assuming any of it transfers.


Flatness on the training grid depends on ``nfft``
=================================================

:class:`~pyFDN.FlatMagnitude` measures ``|H|`` at the model's DFT bins, so its
frequency resolution *is* the model's ``nfft``. The truncation to ``nfft``
samples is a rectangular window, so the peak-to-median range of ``|H|`` grows
with ``nfft``, and the mean-squared error weights the tallest modes ever more
heavily. Loss values are therefore comparable across runs at one ``nfft`` and
not across different ones. A multi-resolution loss
(:class:`~pyFDN.FlatSpectrogram`), whose analysis windows set their own
resolution, is the way out.

The optimization crosses long plateaus on ``FlatMagnitude``; ``train_fdn``'s
default ``patience=10`` stops inside one. Raise it for a converged fit.


Asymmetric flatness: peaks over dips
====================================

:class:`~pyFDN.AsymmetricFlatMagnitude` raises the two sides of the deviation
from flat to different powers so peaks cost more than dips. The exponent, not a
weight, is what makes it bite: at ``peak_power=4`` a peak twice as tall costs
sixteen times as much. Flat stays the unique minimum at every ``peak_power``
(normalizing by RMS forces ``<(1 + d)^2> = 1``), and the loss is gain-invariant.

It is deliberately **not** in decibels: ``∂dB/∂|H| ∝ 1/|H|``, so in dB the
deepest nulls dominate the gradient however lightly weighted. In linear
magnitude the gradient is ``∝ p·(d⁺)^(p-1)``, largest at the tallest peaks,
while a dip is bounded at ``d = -1``.

The cost is steps and steadiness: that same gradient vanishes near the optimum,
so a higher exponent takes longer to converge and varies more from seed to seed.
The advantage over ``FlatMagnitude`` is not unconditional. Loss values are not
comparable across ``peak_power``, or with ``FlatMagnitude``; compare the
responses.


Multi-resolution flatness: average over frames first
====================================================

:class:`~pyFDN.FlatSpectrogram` averages the short-time magnitudes over frames
into a Welch estimate *before* measuring flatness. Asking each individual frame
to be flat instead is actively harmful: an isolated echo inside a short frame
already has a flat frame spectrum, so that objective rewards an impulsive,
comb-filtered IR.


Energy decay vs. spectrogram distance
=====================================

:class:`~pyFDN.MatchEnergyDecay` compares octave-band Schroeder curves. A
magnitude spectrogram distance is not a substitute for fitting a decay: it
compares two signals frame by frame, and two rooms with identical decay have
uncorrelated fine structure, so predicting *silence* scores better there than
predicting the right amount of the wrong detail. Fitted to a measured room, a
spectrogram distance is minimized by an FDN whose RT falls well short of the
measurement, where an energy-decay loss is not.


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
decay, so a plain MSE would see only the first frames. Raising the normalized
surface to a fractional ``power`` (0.5 default) compresses that range while
staying bounded, its gradient ``∝ x^(p-1)`` finite everywhere the floor allows.
A logarithm is worse: it turns the silence *below* the response into an
unbounded penalty dominated by whichever bin is nearest zero.

**Which way the frequency cumulation runs.** Cumulating downwards alone — the
default, and the plain reading of "energy above this frequency" — leaves the
bottom octave with almost no gradient, and a fit that has to *find* a decay
tends to abandon it there. ``frequency="both"`` scores both directions and
averages, which recovers the low end; it is the setting to reach for when the
decay is not given. This is a matter of weighting rather than frequency
resolution, so a longer analysis window is not the fix.
