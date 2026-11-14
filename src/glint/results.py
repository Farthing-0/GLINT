"""Host-side search results that remain usable after GPU buffers are released."""

from dataclasses import asdict, dataclass
import json

import numpy as np

from .stats import signal_to_noise, standardized_power
from .validate import readonly


@dataclass(frozen=True)
class Transit:
    """A fitted event or periodic candidate; depth is a positive flux deficit."""

    period: float | None
    epoch: float
    duration: float
    depth: float
    depth_error: float
    log_likelihood: float
    n_transits: int

    @property
    def snr(self):
        return self.depth / self.depth_error

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LinearResult:
    """Single-event maps indexed by (duration, epoch)."""

    epochs: np.ndarray
    durations: np.ndarray
    log_likelihood: np.ndarray
    depth: np.ndarray
    depth_variance: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, readonly(getattr(self, name)))

    @property
    def best(self):
        valid = np.isfinite(self.log_likelihood) & (self.depth > 0)
        if not valid.any():
            return None
        index = np.argmax(np.where(valid, self.log_likelihood, -np.inf))
        duration, epoch = np.unravel_index(index, self.depth.shape)
        return Transit(None, float(self.epochs[epoch]), float(self.durations[duration]),
                       float(self.depth[duration, epoch]),
                       float(np.sqrt(self.depth_variance[duration, epoch])),
                       float(self.log_likelihood[duration, epoch]), 1)


@dataclass(frozen=True)
class SearchResult:
    """One winning candidate per period, sorted by increasing period.

    Invalid trials contain NaN scores and -1 indices. Equal scores select
    the shortest duration, then the earliest epoch. The global best uses
    the shortest period when scores tie.
    """

    periods: np.ndarray
    log_likelihood: np.ndarray
    depth: np.ndarray
    depth_variance: np.ndarray
    epoch_indices: np.ndarray
    duration_indices: np.ndarray
    n_transits: np.ndarray
    epoch_origin: float
    epoch_step: float
    durations: np.ndarray

    def __post_init__(self):
        for name in ("periods", "log_likelihood", "depth", "depth_variance",
                     "epoch_indices", "duration_indices", "n_transits", "durations"):
            object.__setattr__(self, name, readonly(getattr(self, name)))

    @property
    def power(self):
        return standardized_power(self.log_likelihood)

    @property
    def snr(self):
        return signal_to_noise(self.depth, self.depth_variance)

    def candidate(self, index):
        """Return one candidate, or None when its period has no valid fit."""
        if not np.isfinite(self.log_likelihood[index]) or self.duration_indices[index] < 0:
            return None
        return Transit(
            period=float(self.periods[index]),
            epoch=float(self.epoch_origin + self.epoch_step * self.epoch_indices[index]),
            duration=float(self.durations[self.duration_indices[index]]),
            depth=float(self.depth[index]),
            depth_error=float(np.sqrt(self.depth_variance[index])),
            log_likelihood=float(self.log_likelihood[index]),
            n_transits=int(self.n_transits[index]),
        )

    @property
    def best(self):
        if not np.isfinite(self.log_likelihood).any():
            return None
        return self.candidate(int(np.nanargmax(self.log_likelihood)))

    def save(self, path):
        """Save numeric arrays and a JSON candidate without pickled objects."""
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["best_json"] = json.dumps(self.best.to_dict() if self.best else None)
        np.savez_compressed(path, **payload)
