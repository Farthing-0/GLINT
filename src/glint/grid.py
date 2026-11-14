"""Period, duration, and epoch grids for exhaustive discrete searches."""

import numpy as np

from .constants import GRAVITATIONAL_CONSTANT as G
from .constants import MAX_GRID_AXIS, MAX_INT32, SECONDS_PER_DAY, SOLAR_MASS, SOLAR_RADIUS
from .validate import positive, positive_grid


def duration_grid(config, durations=None):
    """Use an explicit grid or logarithmically spaced durations."""
    if durations is not None:
        result = positive_grid(durations, "durations")
    else:
        count = int(np.floor((np.log(config.duration_max) - np.log(config.duration_min))
                             / np.log(config.duration_step))) + 1
        if count > MAX_GRID_AXIS:
            raise ValueError("The duration count exceeds the CUDA grid limit")
        result = config.duration_min * config.duration_step ** np.arange(count)
    if result.size > MAX_GRID_AXIS:
        raise ValueError("The duration count exceeds the CUDA grid limit")
    return result


def period_grid(baseline, config, periods=None):
    """Space trial frequencies uniformly in their cube root.

    Automatic periods extend to baseline / min_transits. Explicit periods
    override the automatic range; event coverage is still checked on the GPU.
    """
    if periods is not None:
        return positive_grid(periods, "periods")
    span = positive(baseline, "baseline") * SECONDS_PER_DAY
    radius = config.stellar_radius * SOLAR_RADIUS
    mass = config.stellar_mass * SOLAR_MASS
    slow = config.min_transits / span
    fast = np.sqrt(G * mass / (3 * radius) ** 3) / (2 * np.pi)
    step = (2 * np.pi) ** (2 / 3) * radius / (
        3 * np.pi * np.cbrt(G * mass) * span * config.oversampling
    )
    upper = min(config.period_max, baseline / config.min_transits)
    lower = max(config.period_min, 1 / (fast * SECONDS_PER_DAY))
    if lower > upper:
        raise ValueError("The requested bounds leave no automatic trial periods")
    origin = np.cbrt(slow)
    start = max(0, int(np.ceil((np.cbrt(1 / (upper * SECONDS_PER_DAY)) - origin) / step)))
    stop = int(np.floor((np.cbrt(1 / (lower * SECONDS_PER_DAY)) - origin) / step))
    if stop < start:
        raise ValueError("The automatic period grid is empty; provide explicit periods")
    frequencies = (origin + step * np.arange(start, stop + 1)) ** 3
    result = (1 / frequencies / SECONDS_PER_DAY)[::-1].copy()
    result = result[(result >= lower) & (result <= upper)]
    if result.size == 0:
        raise ValueError("The automatic period grid is empty; provide explicit periods")
    return result


def epoch_grid(baseline, durations, fraction):
    """Return relative epochs, including partially observed edge transits."""
    step = positive(durations[0] * fraction, "epoch step")
    origin = -0.5 * float(durations[-1])
    intervals = (baseline + durations[-1]) / step
    if not np.isfinite(intervals) or intervals > MAX_INT32 - 513:
        raise ValueError("The epoch grid exceeds the supported index range")
    count = int(np.ceil(intervals)) + 1
    return origin + np.arange(count, dtype=np.float64) * step, step


def duration_limits(periods, config):
    """Bound circular-orbit durations using stellar and planet size ranges.

    The shortest orbit uses impact parameter 0.9 and a 0.01 solar-radius
    planet. The longest uses a central transit and a 0.2 solar-radius planet.
    Orbits must remain at least three stellar radii from the star.
    """
    periods = np.asarray(periods, dtype=np.float64)
    lower = np.zeros_like(periods)
    upper = periods * config.max_duration_fraction
    if not config.physical_duration_limits:
        return lower, upper

    scale = G * SOLAR_MASS * (periods * SECONDS_PER_DAY) ** 2 / (4 * np.pi**2)
    outer = np.cbrt(scale * config.mass_max) / (config.radius_min * SOLAR_RADIUS)
    inner = np.cbrt(scale * config.mass_min) / (config.radius_max * SOLAR_RADIUS)
    large_radius = np.where(inner >= 3, config.radius_max, config.radius_min)
    inner = np.maximum(inner, 3)
    short_chord = ((1 + 0.01 / config.radius_min) ** 2 - 0.9**2)
    short_sine = np.sqrt(short_chord / np.maximum(outer**2 - 0.9**2, 1e-30))
    long_sine = (1 + 0.2 / large_radius) / inner
    lower = periods / np.pi * np.arcsin(np.clip(short_sine, 0, 1))
    upper = np.minimum(upper, periods / np.pi * np.arcsin(np.clip(long_sine, 0, 1)))
    invalid = outer < 3
    lower[invalid], upper[invalid] = np.inf, -np.inf
    return lower, upper


def duration_indices(periods, durations, config):
    """Return inclusive duration bounds; first > last means no valid duration."""
    lower, upper = duration_limits(periods, config)
    # The moment likelihood requires disjoint event windows after rounding.
    step = durations[0] * config.epoch_step_fraction
    separation = np.floor(periods / step) * step
    upper = np.minimum(upper, np.nextafter(separation, 0.0))
    first = np.searchsorted(durations, lower, side="left").astype(np.int32)
    last = (np.searchsorted(durations, upper, side="right") - 1).astype(np.int32)
    return first, last
