"""Likelihood-derived diagnostics without a false-alarm calibration."""

import numpy as np


def signal_to_noise(depth, variance):
    """Return fitted depth divided by its formal standard error."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(depth) / np.sqrt(variance)


def standardized_power(log_likelihood):
    """Standardize finite likelihood scores; invalid trials remain NaN.

    This diagnostic is not a calibrated detection probability or a
    detrended signal detection efficiency statistic.
    """
    scores = np.asarray(log_likelihood, dtype=np.float64)
    finite = np.isfinite(scores)
    power = np.full_like(scores, np.nan)
    if finite.any():
        scale = np.std(scores[finite])
        power[finite] = (scores[finite] - np.mean(scores[finite])) / scale if scale else 0.0
    return power
