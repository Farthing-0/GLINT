"""Small time-domain utilities for examples and candidate inspection."""

import numpy as np

from .validate import positive


def fold(time, period, epoch):
    """Return centered time offsets in [-period/2, period/2)."""
    period = positive(period, "period")
    if not np.isfinite(epoch):
        raise ValueError("epoch must be finite")
    return (np.asarray(time, dtype=np.float64) - epoch + period / 2) % period - period / 2


def transit_mask(time, period, epoch, duration):
    """Identify observations inside a periodic transit window."""
    duration = positive(duration, "duration")
    period = positive(period, "period")
    if duration >= period:
        raise ValueError("duration must be shorter than period")
    return np.abs(fold(time, period, epoch)) <= duration / 2
