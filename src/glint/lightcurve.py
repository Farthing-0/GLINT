"""Validated host-side light curves without a CUDA context."""

from dataclasses import dataclass

import numpy as np

from .validate import readonly, vector


@dataclass(frozen=True, init=False)
class LightCurve:
    """Sorted, normalized observations with optional zero-weight gaps.

    Time is measured in days and flux is relative to an out-of-transit level
    of one. Infinite positive uncertainties mask observations. Missing flux
    is allowed only at masked observations. Duplicate times are retained as
    independent measurements. Detrending and normalization are caller tasks.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray

    def __init__(self, time, flux, flux_error=None):
        time, flux = vector(time, "time", minimum=3), vector(flux, "flux", minimum=3)
        if time.shape != flux.shape or not np.all(np.isfinite(time)):
            raise ValueError("time and flux must have equal lengths; time must be finite")
        if flux_error is None:
            finite_flux = flux[np.isfinite(flux)]
            scatter = np.std(finite_flux) if finite_flux.size else np.nan
            if not np.isfinite(scatter) or scatter <= 0:
                raise ValueError("Provide flux_error when flux has no measurable scatter")
            flux_error = np.full(time.size, scatter)
        elif np.ndim(flux_error) == 0:
            flux_error = np.full(time.size, float(flux_error))
        error = vector(flux_error, "flux_error", minimum=3)
        if error.shape != time.shape or np.any(np.isnan(error) | (error <= 0)):
            raise ValueError("flux_error must match time and contain positive values or infinity")
        valid = np.isfinite(error)
        if valid.sum() < 3 or not np.all(np.isfinite(flux[valid])):
            raise ValueError("At least three finite observations with finite errors are required")
        # Masked endpoints cannot enlarge the usable observing baseline.
        first, last = np.min(time[valid]), np.max(time[valid])
        if first == last:
            raise ValueError("Observations must span a positive time interval")
        keep = (time >= first) & (time <= last)
        order = np.flatnonzero(keep)[np.argsort(time[keep], kind="stable")]
        for name, values in (("time", time), ("flux", flux), ("flux_error", error)):
            object.__setattr__(self, name, readonly(values[order]))

    @property
    def baseline(self):
        return float(self.time[-1] - self.time[0])

    @property
    def reference_time(self):
        return float(self.time[0])

    @property
    def offset_time(self):
        return self.time - self.reference_time

    @property
    def residual(self):
        return np.where(np.isfinite(self.flux_error), 1.0 - self.flux, 0.0)

    @property
    def weight(self):
        return np.square(1.0 / self.flux_error)

    def mask_transit(self, period, epoch, duration):
        """Return a new curve with the specified transit windows masked."""
        from .helpers import transit_mask

        errors = self.flux_error.copy()
        errors[transit_mask(self.time, period, epoch, duration)] = np.inf
        return LightCurve(self.time, self.flux, errors)
