"""Public search orchestration, independent of CUDA implementation details."""

from .config import SearchConfig
from .core import SearchWorkspace
from .grid import duration_grid, epoch_grid, period_grid
from .lightcurve import LightCurve
from .transit import TransitTemplate
from .validate import readonly


class GLINT:
    """Search normalized observations with a reusable two-stage GPU workspace.

    Construction validates host data only. Use ``power`` for a full search,
    or ``linear_search`` and ``period_search`` to inspect the stages. A context
    manager releases owned GPU arrays while preserving returned host results.
    """

    def __init__(self, time, flux=None, flux_error=None, *, config=None,
                 durations=None, template="limb_darkened"):
        if isinstance(time, LightCurve):
            if flux is not None or flux_error is not None:
                raise ValueError("Do not supply flux arrays together with a LightCurve")
            curve = time
        else:
            curve = LightCurve(time, flux, flux_error)
        if config is not None and not isinstance(config, SearchConfig):
            raise TypeError("config must be a SearchConfig")
        self._config = config if config is not None else SearchConfig()
        self._light_curve = curve
        self._template = template if isinstance(template, TransitTemplate) else TransitTemplate.build(template)
        self._durations = readonly(duration_grid(self.config, durations))
        epochs, step = epoch_grid(curve.baseline, self.durations, self.config.epoch_step_fraction)
        self._epochs, self._epoch_step = readonly(epochs), step
        self._workspace = None
        self.linear_result = None
        self.result = None

    @property
    def config(self):
        return self._config

    @property
    def light_curve(self):
        return self._light_curve

    @property
    def template(self):
        return self._template

    @property
    def durations(self):
        return self._durations

    def _ensure_linear(self):
        if self._workspace is None:
            workspace = SearchWorkspace(self.light_curve, self.template, self.durations,
                                        self._epochs, self._epoch_step, self.config)
            try:
                workspace.linear_search()
            except Exception:
                workspace.close()
                raise
            self._workspace = workspace
        return self._workspace

    def linear_search(self):
        """Return host copies of the single-event likelihood, depth, and variance."""
        if self.linear_result is None:
            self.linear_result = self._ensure_linear().linear_result()
        return self.linear_result

    def period_search(self, periods=None):
        """Search explicit or automatic periods, computing the linear stage as needed."""
        trials = period_grid(self.light_curve.baseline, self.config, periods)
        self.result = self._ensure_linear().period_search(trials)
        return self.result

    def power(self, periods=None):
        """Run both search stages and return a host-side SearchResult."""
        return self.period_search(periods)

    def close(self):
        """Release device buffers; a later search recreates them if necessary."""
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
