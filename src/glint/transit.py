"""Self-contained transit templates and model evaluation."""

from dataclasses import dataclass

import numpy as np

from .helpers import fold
from .validate import integer, positive, readonly, vector


@dataclass(frozen=True, init=False)
class TransitTemplate:
    """A unit-depth deficit sampled uniformly from first to fourth contact.

    Samples describe a positive flux deficit, not normalized flux. The shape
    is zero outside [-0.5, 0.5] and linearly interpolated inside that interval.
    """

    values: np.ndarray
    name: str

    def __init__(self, values, name="custom"):
        values = vector(values, "template", minimum=3)
        if not np.all(np.isfinite(values) & (values >= 0)) or values.max() == 0:
            raise ValueError("Template samples must be finite, nonnegative, and nonzero")
        object.__setattr__(self, "values", readonly(values / values.max(), np.float32))
        object.__setattr__(self, "name", str(name))

    @classmethod
    def build(cls, name="limb_darkened", samples=1025):
        """Generate a limb-darkened, grazing, box, or trapezoid template.

        Physical shapes use a circular orbit with radius ratio 0.03,
        a/R = 20, and quadratic limb darkening (0.4804, 0.1867).
        """
        integer(samples, "samples", 3)
        phase = np.linspace(-0.5, 0.5, samples)
        if name == "box":
            return cls(np.ones(samples), name)
        if name == "trapezoid":
            return cls(np.clip((0.5 - np.abs(phase)) / 0.15, 0, 1), name)
        if name not in ("limb_darkened", "grazing"):
            raise ValueError("Unknown template; choose limb_darkened, grazing, box, or trapezoid")

        import batman

        impact = 0.32 if name == "limb_darkened" else 0.93
        parameters = batman.TransitParams()
        parameters.t0, parameters.per = 0.0, 10.0
        parameters.rp, parameters.a = 0.03, 20.0
        parameters.inc = np.degrees(np.arccos(impact / parameters.a))
        parameters.ecc, parameters.w = 0.0, 90.0
        parameters.u, parameters.limb_dark = [0.4804, 0.1867], "quadratic"
        duration = parameters.per / np.pi * np.arcsin(np.sqrt(
            ((1 + parameters.rp) ** 2 - impact**2) / (parameters.a**2 - impact**2)
        ))
        model = batman.TransitModel(parameters, phase * duration)
        return cls(np.maximum(1.0 - model.light_curve(parameters), 0.0), name)

    def evaluate(self, phase):
        """Evaluate the deficit at times expressed in transit durations."""
        return np.interp(phase, np.linspace(-0.5, 0.5, self.values.size),
                         self.values, left=0.0, right=0.0)

    def model(self, time, transit):
        """Evaluate a fitted transit on arbitrary observation times."""
        positive(transit.duration, "duration")
        offset = (np.asarray(time) - transit.epoch if transit.period is None
                  else fold(time, transit.period, transit.epoch))
        return 1.0 - transit.depth * self.evaluate(offset / transit.duration)
