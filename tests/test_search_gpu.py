"""Compare small GPU searches with independent, exhaustive NumPy fits."""

import numpy as np
import pytest

from glint import GLINT, SearchConfig, TransitTemplate
from .reference import periodic_events, single_events


@pytest.mark.gpu
@pytest.mark.parametrize("mode,batch_size,threads", [("moments", 1, 128), ("two_pass", 3, 256)])
def test_full_search_matches_direct_fits(gpu_device, mode, batch_size, threads):
    rng = np.random.default_rng(817)
    time = np.sort(rng.uniform(0, 5, 250)) + 2457000.0
    error = rng.uniform(0.001, 0.002, time.size)
    error[70:90] = np.inf
    phase = (time - time[0] - 0.4 + 0.65) % 1.3 - 0.65
    flux = 1 - 0.006 * np.clip((0.1 - np.abs(phase)) / 0.03, 0, 1)
    flux += rng.normal(0, 0.001, time.size)
    config = SearchConfig(device_id=gpu_device, physical_duration_limits=False,
                          max_duration_fraction=0.4, epoch_step_fraction=0.25,
                          min_points=2, mode=mode, batch_size=batch_size, threads=threads)
    template = TransitTemplate.build("trapezoid")
    periods = np.array([0.9, 1.3, 1.71, 2.3])
    with GLINT(time, flux, error, config=config, durations=[0.16, 0.24], template=template) as search:
        linear = search.linear_search()
        # Mirror storage precision only; all fitting and enumeration are independent.
        expected = single_events(search.light_curve.offset_time,
                                 search.light_curve.residual.astype(np.float32).astype(float),
                                 search.light_curve.weight, search.durations, search._epochs,
                                 template.values, config.min_points)
        for actual, reference in zip((linear.log_likelihood, linear.depth, linear.depth_variance), expected):
            np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=1e-9)
        expected_scores, expected_indices = periodic_events(
            linear.log_likelihood, linear.depth, linear.depth_variance,
            periods, search.durations, search._epoch_step, config.min_transits,
            config.max_duration_fraction,
        )
        result = search.power(periods)
        np.testing.assert_allclose(result.log_likelihood, expected_scores, rtol=1e-10, atol=1e-9)
        np.testing.assert_array_equal(result.duration_indices, expected_indices[:, 0])
        np.testing.assert_array_equal(result.epoch_indices, expected_indices[:, 1])
    assert result.best is not None
    assert np.isfinite(result.best.epoch)


@pytest.mark.gpu
def test_insufficient_events_has_no_candidate(gpu_device):
    config = SearchConfig(device_id=gpu_device, physical_duration_limits=False, min_transits=3)
    with GLINT(np.linspace(0, 1, 30), np.ones(30), 0.01, config=config,
               durations=[0.1], template="box") as search:
        result = search.power([3.0])
    assert result.best is None
    assert np.isnan(result.log_likelihood).all()
    assert (result.epoch_indices == -1).all()
    assert (result.duration_indices == -1).all()
    assert (result.n_transits == 0).all()
