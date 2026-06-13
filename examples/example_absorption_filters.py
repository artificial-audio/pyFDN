# gallery_category: Absorption & Filters

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Absorption filters example

    This example designs **frequency-dependent absorption filters** for an FDN (Feedback Delay Network) reverb. Each filter approximates a target RT (reverberation time) curve over frequency for a given delay line.

    **What it does:**
    - Defines target RT values at a few frequencies for two delay lines (e.g. 20 ms and 40 ms).
    - Uses `pyFDN.absorption_filters` to compute FIR filter coefficients that realize these targets.
    - Uses `pyFDN.absorption_to_rt` to evaluate the achieved RT vs frequency.
    - Plots the filter impulse responses and the target vs approximated RT curves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Setup:** `target_frequency` and `target_rt` define the desired RT (seconds) at each frequency for each delay. `filter_order` is chosen from the desired filter length in samples (here 10 ms). The first figure shows the impulse responses of the two absorption filters; the second compares the target RT curves (markers) with the approximation (lines).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import pyFDN

    return np, plt, pyFDN


@app.cell
def _(np, plt, pyFDN):
    fs = 48000
    target_frequency = np.array([0, 500, 5000, 8000, fs / 2])
    # Define two separate RT curves, then merge into one array
    rt_curve_1 = np.array([2, 2, 2, 1, 1])  # seconds
    rt_curve_2 = np.array([2, 2, 1, 0.5, 0.5])  # seconds
    target_rt = np.column_stack([rt_curve_1, rt_curve_2])

    delays_ms = np.array([20, 40])  # ms
    delays = pyFDN.ms_to_smp(delays_ms, fs)

    filter_length = 10  # ms
    filter_order = int(pyFDN.ms_to_smp(filter_length, fs))

    coeffs = pyFDN.absorption_filters(
        target_frequency, target_rt, filter_order, delays, fs
    )
    rt, rt_f = pyFDN.absorption_to_rt(coeffs, np.array(delays), 2**10, fs)

    # Plot filters
    plt.figure()
    plt.plot(pyFDN.mulaw_encode(coeffs.T))
    plt.legend(["Filter 1", "Filter 2"])
    plt.xlabel("Time [samples]")
    plt.ylabel("Amplitude [mu-law]")
    plt.grid()

    # Plot target vs approximation
    plt.figure()
    plt.plot(target_frequency, target_rt, "o-")
    plt.plot(rt_f, rt)
    plt.legend(["Target 1", "Target 2", "Approx 1", "Approx 2"])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Reverberation time [s]")
    plt.xlim([0, fs / 2])
    plt.ylim(bottom=0)
    plt.grid()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
