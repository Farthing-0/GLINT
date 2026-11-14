"""Slow NumPy oracles for tiny numerical cases, never used by the package."""

import numpy as np


def single_events(time, residual, weight, durations, epochs, shape, min_points):
    """Fit each event directly against all original observations."""
    size = (len(durations), len(epochs))
    likelihood, depth, variance = np.full(size, np.nan), np.full(size, np.nan), np.full(size, np.inf)
    for row, duration in enumerate(durations):
        for column, epoch in enumerate(epochs):
            model = np.interp((time - epoch) / duration,
                              np.linspace(-0.5, 0.5, len(shape)), shape, left=0, right=0)
            selected = (weight > 0) & (model > 0)
            if selected.sum() < min_points:
                continue
            w, y, s = weight[selected], residual[selected], model[selected]
            precision = np.sum(w * s * s)
            fitted = np.sum(w * y * s) / precision
            likelihood[row, column] = 0.5 * np.sum(w * (y * y - (y - fitted * s) ** 2))
            depth[row, column], variance[row, column] = fitted, 1 / precision
    return likelihood, depth, variance


def periodic_events(likelihood, depth, variance, periods, durations, step, min_transits,
                    max_duration_fraction, depth_min=0):
    """Enumerate candidates using the direct two-pass likelihood definition."""
    scores, indices = [], []
    for period in periods:
        best, winner = -np.inf, (-1, -1)
        spacing = period / step
        for row, duration in enumerate(durations):
            if duration > max_duration_fraction * period:
                continue
            for epoch in range(min(int(np.ceil(spacing)), depth.shape[1])):
                event = np.arange(int(np.ceil(depth.shape[1] / spacing)) + 1)
                selected = epoch + np.rint(spacing * event).astype(int)
                selected = selected[selected < depth.shape[1]]
                selected = selected[np.isfinite(variance[row, selected])]
                if selected.size < min_transits:
                    continue
                d, v = depth[row, selected].astype(float), variance[row, selected].astype(float)
                fitted = np.sum(d / v) / np.sum(1 / v)
                if fitted <= depth_min:
                    continue
                score = np.sum(likelihood[row, selected].astype(float) - 0.5 * (d - fitted) ** 2 / v)
                if score > best:
                    best, winner = score, (row, epoch)
        scores.append(best if np.isfinite(best) else np.nan)
        indices.append(winner)
    return np.asarray(scores), np.asarray(indices)
