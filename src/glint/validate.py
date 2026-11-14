"""Small validation helpers shared by the public interfaces."""

import numpy as np


def positive(value, name, *, allow_zero=False):
    """Return a finite scalar in the requested positive domain."""
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        domain = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {domain}")
    return value


def integer(value, name, minimum=1):
    """Reject fractional counts and booleans rather than truncating them."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def vector(values, name, *, minimum=1):
    """Copy a one-dimensional numeric input into contiguous host memory."""
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or result.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional array of length >= {minimum}")
    return result


def positive_grid(values, name):
    """Return sorted unique finite positive grid values."""
    result = vector(values, name)
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError(f"{name} must contain only finite positive values")
    return np.unique(result)


def readonly(values, dtype=None):
    """Return an owned, read-only array for a public data object."""
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result
