# GLINT

GLINT searches normalized stellar light curves for periodic transits on NVIDIA
GPUs. It is a standalone Python project built around CuPy, with packaged CUDA
kernels and locally generated transit templates.

**Status:** initial framework implementation. Tests, CUDA compilation, examples,
and performance measurements have not been run. Numerical behavior and recovery
performance still need validation before scientific use.

## Design

- Fit a transit template at every trial duration and epoch once.
- Combine single-event fits into periodic candidates using weighted moments.
- Evaluate the complete configured grid in memory-bounded period batches.
- Use original observation timestamps, including irregular sampling and gaps.
- Return host-side results with explicit likelihoods, uncertainties, and event counts.
- Keep GPU initialization lazy and device selection explicit.

The project owns its public API, data structures, grids, templates, kernels,
tests, and examples. It has no runtime imports, inheritance, path injection,
or data-file dependencies on other transit-search projects.

## Installation

Requirements: Python 3.10 or newer, an NVIDIA GPU, and a CuPy distribution
compatible with the installed CUDA environment. Kernel compilation uses NVRTC;
the environment must provide the matching CUDA runtime and headers.

From this directory, install exactly one CUDA extra:

```bash
python -m pip install -e '.[cuda12,test,examples]'
# For a CUDA 13 environment, use cuda13 instead of cuda12.
```

If CuPy is already configured, install only the project:

```bash
python -m pip install -e .
```

The distribution name is `glint-transit`; the Python import is `glint`.
NumPy handles host-side data and grids. `batman-package` generates physical
transit shapes. CuPy manages all device arrays and search execution. There is
no CPU search fallback.

## Quick start

```python
from glint import GLINT, SearchConfig

# time is in days; flux is detrended and normalized to one.
config = SearchConfig(
    period_min=1.0,
    period_max=12.0,
    duration_min=0.02,
    duration_max=0.5,
    device_id=0,
)

with GLINT(time, flux, flux_error, config=config) as search:
    result = search.power()

if result.best is not None:
    print(result.best.to_dict())
result.save("periodogram.npz")
```

For a complete runnable data-generation example, see
[examples/synthetic_search.py](examples/synthetic_search.py).

## Search stages and templates

```python
from glint import GLINT, LightCurve, TransitTemplate

curve = LightCurve(time, flux, flux_error)
template = TransitTemplate.build("limb_darkened")

with GLINT(curve, template=template, durations=[0.05, 0.1, 0.2]) as search:
    events = search.linear_search()
    result = search.period_search(periods=[2.1, 2.2, 2.3])
    candidate = result.best
    if candidate is not None:
        model_flux = search.template.model(curve.time, candidate)
```

`linear_search()` returns maps with shape `(number of durations, number of epochs)`.
It explicitly copies those maps to the host. `power()` and `period_search()` keep
the maps on the device and transfer only the period results. Repeated period
searches on one instance reuse the maps. `close()` releases the instance's
device arrays to CuPy's allocator; it does not clear the shared memory pool.
Returned results remain usable after closing the search.

Built-in shapes are `limb_darkened`, `grazing`, `box`, and `trapezoid`.
`TransitTemplate(values)` accepts a nonnegative deficit sampled uniformly across
the transit and normalizes it to unit depth. Physical shapes are generated on
the host; box, trapezoid, and custom shapes need no physical-model initialization.

## Configuration and conventions

All times and durations are in days. Stellar masses and radii are in solar
units. Depth is the fractional flux deficit, and epoch is a mid-transit time
in the input time system. Each candidate's epoch originates in the first
trial-period interval of the padded epoch grid and may precede the first observation.

| Setting | Default | Meaning |
| --- | --- | --- |
| `period_min`, `period_max` | `0`, infinity | Automatic period bounds |
| `duration_min`, `duration_max` | `0.02`, `0.5` | Automatic duration bounds |
| `duration_step` | `1.1` | Multiplicative duration spacing |
| `epoch_step_fraction` | `0.1` | Epoch spacing divided by the shortest duration |
| `oversampling` | `3` | Automatic frequency-grid oversampling |
| `min_transits` | `2` | Minimum number of observed events |
| `min_points` | `1` | Minimum weighted points in an individual event |
| `depth_min` | `0` | Strict lower bound on joint transit depth |
| `physical_duration_limits` | `True` | Apply circular-orbit duration bounds |
| `max_duration_fraction` | `0.12` | Maximum duration divided by period |
| `stellar_radius`, `stellar_mass` | `1`, `1` | Automatic period-grid parameters |
| `radius_min`, `radius_max` | `0.13`, `3.5` | Duration-bound stellar radii |
| `mass_min`, `mass_max` | `0.1`, `1` | Duration-bound stellar masses |
| `device_id` | `0` | Logical CUDA device, respecting device visibility |
| `batch_size` | `256` | Upper bound on periods per batch |
| `threads` | `128` | CUDA block size: 128, 256, or 512 |
| `memory_fraction` | `0.7` | Fraction of available memory used for allocations |
| `mode` | `"moments"` | Weighted moments or the `"two_pass"` diagnostic mode |

Explicit `periods` and `durations` replace their automatic grids. Periods and
durations are sorted and deduplicated; duration filtering and event-coverage
requirements still apply. A custom period may exceed the automatic baseline
limit, but it cannot yield a candidate without enough observed events.

Input observations are sorted, without resampling. Positive infinite uncertainty
masks a point, including a missing flux value. A scalar uncertainty is accepted.
When uncertainty is omitted, the finite flux standard deviation is used; a
constant curve requires explicit uncertainties. Data must already be normalized
and detrended.

`SearchResult.log_likelihood` is the improvement over a constant unit-flux model.
`power` standardizes its finite scores, and `snr` is depth divided by its formal
error. Neither quantity is a calibrated false-alarm probability. Invalid periods
have NaN fit values, `-1` indices, and zero event counts; `best` is `None` if all
periods are invalid. Ties prefer shorter durations, earlier epochs, then shorter
periods. The candidate model uses exact periodic times; the search samples the
nearest event-grid entries, so its fitted score is a discretized approximation.

## Project layout

```text
GLINT/
  pyproject.toml
  README.md
  LICENSE
  src/glint/
    main.py          # Public search lifecycle
    config.py        # Immutable validated settings
    lightcurve.py    # Host-side observation model
    validate.py      # Input validation helpers
    grid.py          # Period, duration, and epoch grids
    transit.py       # Generated and custom templates
    backend.py       # Lazy CuPy and NVRTC loading
    core.py          # GPU workspace and batch orchestration
    kernels.cu       # Single-event fits and periodic reductions
    results.py       # Candidates, periodograms, and serialization
    stats.py         # Likelihood and uncertainty diagnostics
    helpers.py       # Folding and transit masks
    constants.py
    version.py
  docs/architecture.md
  examples/          # Synthetic and CSV examples
  tests/             # Host contracts and independent GPU oracles
```

## Development

The [test suite](tests/README.md) and [examples](examples/README.md) have their
own synthetic inputs and do not need an external repository or dataset.
They are prepared for the next validation phase and have not been executed.

```bash
python -m pytest -m "not gpu"
GLINT_TEST_DEVICE=0 python -m pytest -m gpu
python examples/synthetic_search.py --plot
```

See [the architecture notes](docs/architecture.md) for likelihood equations,
memory ownership, precision choices, and the current scope.

## Acknowledgements

GLINT acknowledges [CETRA](https://github.com/leigh2/cetra) for the two-stage
transit-search approach, [TLS](https://github.com/hippke/tls) for transit-template
search and adaptive period-grid ideas, and [GTLS](https://github.com/Farthing-0/GTLS)
for its GPU search architecture and period-batching design. These projects
informed the design; GLINT provides its own implementation and API.

## License

GLINT is distributed under the [MIT License](LICENSE).
