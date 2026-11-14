"""Input and configuration contracts without a CUDA dependency."""

import numpy as np
import pytest

from glint import GLINT, LightCurve, SearchConfig, TransitTemplate
from glint.grid import duration_grid, period_grid


def test_unsorted_observations_and_masked_gaps():
    curve = LightCurve([3, 1, 2, 1], [1, 0.99, np.nan, 1.01], [0.1, 0.1, np.inf, 0.2])
    np.testing.assert_array_equal(curve.time, [1, 1, 2, 3])
    assert curve.weight[2] == 0 and curve.residual[2] == 0
    assert not curve.time.flags.writeable
    with pytest.raises(ValueError):
        curve.flux[0] = 0


@pytest.mark.parametrize("error", [0, -1, np.nan, [0.1, np.inf, np.inf]])
def test_invalid_uncertainty(error):
    with pytest.raises(ValueError):
        LightCurve([0, 1, 2], [1, 1, 1], error)


def test_constant_flux_requires_explicit_uncertainty():
    with pytest.raises(ValueError, match="flux_error"):
        LightCurve([0, 1, 2], [1, 1, 1])
    assert LightCurve([0, 1, 2], [1, 1, 1], 0.01).baseline == 2


@pytest.mark.parametrize("settings", [
    {"duration_step": 1}, {"period_min": 2, "period_max": 1},
    {"epoch_step_fraction": 0}, {"threads": 129}, {"batch_size": 0},
    {"min_transits": 1}, {"min_transits": 2.5}, {"memory_fraction": 1},
])
def test_invalid_search_settings(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)


def test_grids_are_ordered_and_respect_bounds():
    config = SearchConfig(period_min=1, period_max=5, duration_min=0.1, duration_max=0.4)
    periods = period_grid(30, config)
    durations = duration_grid(config)
    assert periods.size > 1 and np.all(np.diff(periods) > 0)
    assert periods[0] >= 1 and periods[-1] <= 5
    assert durations[0] == 0.1 and durations[-1] <= 0.4
    np.testing.assert_array_equal(period_grid(30, config, [9, 7, 9]), [7, 9])


def test_host_construction_never_loads_the_gpu(monkeypatch):
    import glint.backend as backend
    import glint.core as core

    def forbidden():
        raise AssertionError("Host construction attempted to load CuPy")

    monkeypatch.setattr(backend, "cupy", forbidden)
    monkeypatch.setattr(core, "cupy", forbidden)
    with GLINT([0, 1, 2], [1, 1, 1], 0.01, template="box") as search:
        assert search.result is None
        np.testing.assert_array_equal(search.template.evaluate([-1, 0, 1]), [0, 1, 0])


def test_custom_template_is_normalized_and_owned():
    values = np.array([0, 2, 0], dtype=float)
    template = TransitTemplate(values)
    values[1] = 0
    np.testing.assert_array_equal(template.evaluate([-0.5, 0, 0.5]), [0, 1, 0])
