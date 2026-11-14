"""Immutable settings for grid construction and GPU execution."""

from dataclasses import dataclass

import numpy as np

from .constants import MAX_GRID_AXIS, MAX_INT32
from .validate import integer, positive


@dataclass(frozen=True)
class SearchConfig:
    """Search settings; all time quantities are in days.

    Stellar masses and radii are in solar units. Execution settings change
    batching only; they do not discard epochs, durations, or trial periods.
    """

    period_min: float = 0.0
    period_max: float = np.inf
    duration_min: float = 0.02
    duration_max: float = 0.5
    duration_step: float = 1.1
    epoch_step_fraction: float = 0.1
    oversampling: float = 3.0
    min_transits: int = 2
    min_points: int = 1
    depth_min: float = 0.0
    stellar_radius: float = 1.0
    stellar_mass: float = 1.0
    radius_min: float = 0.13
    radius_max: float = 3.5
    mass_min: float = 0.1
    mass_max: float = 1.0
    physical_duration_limits: bool = True
    max_duration_fraction: float = 0.12
    device_id: int = 0
    batch_size: int = 256
    threads: int = 128
    memory_fraction: float = 0.7
    mode: str = "moments"

    def __post_init__(self):
        for name in (
            "duration_min", "duration_max", "duration_step", "epoch_step_fraction",
            "oversampling", "stellar_radius", "stellar_mass", "radius_min",
            "radius_max", "mass_min", "mass_max", "max_duration_fraction",
            "memory_fraction",
        ):
            positive(getattr(self, name), name)
        positive(self.period_min, "period_min", allow_zero=True)
        positive(self.depth_min, "depth_min", allow_zero=True)
        if np.isnan(self.period_max) or self.period_max <= self.period_min:
            raise ValueError("period_max must exceed period_min")
        for low, high in (("duration_min", "duration_max"), ("radius_min", "radius_max"),
                          ("mass_min", "mass_max")):
            if getattr(self, low) > getattr(self, high):
                raise ValueError(f"{low} must not exceed {high}")
        if self.duration_step <= 1:
            raise ValueError("duration_step must exceed one")
        if self.epoch_step_fraction > 1 or self.max_duration_fraction >= 1:
            raise ValueError("epoch_step_fraction must be <= 1 and max_duration_fraction < 1")
        if self.memory_fraction >= 1:
            raise ValueError("memory_fraction must be less than one")
        integer(self.min_transits, "min_transits", 2)
        integer(self.min_points, "min_points")
        if self.min_transits > MAX_INT32 or self.min_points > MAX_INT32:
            raise ValueError("Event and observation thresholds exceed the supported index range")
        integer(self.device_id, "device_id", 0)
        integer(self.batch_size, "batch_size")
        integer(self.threads, "threads")
        if self.batch_size > MAX_GRID_AXIS or self.threads not in (128, 256, 512):
            raise ValueError("batch_size must be <= 65535; threads must be 128, 256, or 512")
        if not isinstance(self.physical_duration_limits, (bool, np.bool_)):
            raise ValueError("physical_duration_limits must be a boolean")
        if self.mode not in ("moments", "two_pass"):
            raise ValueError("mode must be 'moments' or 'two_pass'")
