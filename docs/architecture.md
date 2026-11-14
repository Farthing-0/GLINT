# Architecture

GLINT separates the public search lifecycle from numerical planning and GPU
execution. Package import, input validation, template generation, and grid
construction do not create a CUDA context. Only an actual search loads CuPy
and compiles the packaged CUDA source with NVRTC.

## Data flow

```text
LightCurve + TransitTemplate + SearchConfig
                    |
          duration and epoch grids
                    |
       SearchWorkspace on one device
                    |
      single-event weighted template fits
                    |
        likelihood / depth / variance maps
                    |
     period grid and duration restrictions
                    |
       exhaustive batched periodic search
                    |
       one candidate per period -> SearchResult
```

`main.py` owns orchestration and caches one workspace. `core.py` owns device
buffers and launches kernels. `backend.py` handles optional CuPy imports and
device-specific module compilation. `grid.py`, `transit.py`, and `lightcurve.py`
define the numerical inputs independently of execution. `results.py` contains
host arrays only, so callers can inspect or save results after releasing the GPU.

## Observation and template model

Observations remain at their original sorted timestamps. A binary search finds
the start of each trial window, and GPU threads fit only points in that window.
No interpolated observations are added across gaps. Duplicate timestamps remain
independent measurements. Missing observations have zero weight.

Times are shifted by the first usable timestamp before upload and represented
as float64. The epoch grid extends half the longest duration beyond both ends
of the data to include partially observed edge transits. It has one uniform
spacing: the shortest duration times `epoch_step_fraction`.

Templates store a nonnegative unit-depth deficit over normalized time
`[-0.5, 0.5]`. Template interpolation is linear; outside support the deficit is
zero. Limb-darkened and grazing templates are generated with a physical transit
model, without bundled reference-project assets. Template shape is fixed during
a search; only duration, depth, and epoch vary.

Exposure integration, stellar variability removal, and uncertainty estimation
beyond the optional global scatter estimate are outside this framework.

## Single-event likelihood

For a trial event, let `r_i = 1 - flux_i`, `s_i` be its template deficit,
and `w_i = 1 / error_i^2`. Define

```text
A = sum(w_i * s_i * r_i)
B = sum(w_i * s_i^2)
depth = A / B
variance = 1 / B
log_likelihood = A^2 / (2 * B)
```

The linear kernel evaluates these quantities for each duration and epoch.
Only observations with positive weight and positive template support count
toward `min_points`. Invalid events have NaN depth and likelihood, and infinite
variance. Single-event depths may be negative; this avoids selecting only
downward noise before fitting a common periodic depth.

Residuals and template samples use float32; observation weights and time
coordinates use float64. The residual is formed on the host in float64 before
conversion. Reductions use float64, then the three persistent maps use float32
to control their memory cost. Nonrepresentable fits become invalid events.

## Periodic likelihood

For each trial period `P`, duration, and initial epoch index `k`, event `j`
uses index `k + round_to_even(j * P / epoch_step)`. Every initial epoch in the
first trial-period interval is searched, bounded by the available epoch map.
The original, unrounded period is retained in the output. The period grid and
nearest-event sampling are discrete approximations; output templates evaluated
at exact periodic epochs can therefore differ slightly from the search score.

For each valid event fit `(d_j, v_j, L_j)`, accumulate

```text
W = sum(1 / v_j)
D = sum(d_j / v_j)
C = sum(L_j - d_j^2 / (2 * v_j))
joint_depth = D / W
joint_variance = 1 / W
joint_log_likelihood = C + D^2 / (2 * W)
```

The correction `C` retains differences between the stored likelihood and the
likelihood reconstructed from rounded depth and variance. The optional
`two_pass` mode instead evaluates

```text
sum(L_j) - 0.5 * sum((d_j - joint_depth)^2 / v_j)
```

Both modes use the same events and grids, require `min_transits` valid events,
and require joint depth strictly above `depth_min`. The event count measures
coverage, not the number of individually significant dips. Both modes accumulate
in float64. Algebraic equivalence does not imply bitwise equality.

Period duration bounds use circular orbits and configurable stellar size and
mass limits, with three stellar radii as the minimum orbital separation.
`max_duration_fraction` is applied even when physical filtering is disabled.
Duration windows are also kept disjoint after event-grid rounding.

## GPU execution and ownership

One linear block fits one duration and epoch, then strides over additional
epochs. Periodic blocks evaluate contiguous epoch tiles for one period and
duration. A second reduction selects the best tile for each period. Equal
scores use duration and epoch order, making tie behavior explicit.

The resident linear maps require `12 * duration_count * epoch_count` bytes,
excluding input arrays and allocator overhead. They must fit in memory; there
is no implicit coarsening or host spill. Allocation errors report the required
size so callers can choose a smaller explicit grid.

Periodic scratch buffers are allocated once and reused across batches. The
requested `batch_size` is an upper limit, reduced to fit the configured memory
fraction. The driver also checks CUDA dimension and index limits. Available
memory includes blocks held unused by CuPy's default allocator, so the pool can
be reused by subsequent searches. The estimate is conservative but does not
reserve memory against concurrent work in other processes.

Public operations synchronize their current stream before returning. An
instance is serial and bound to one logical device. `close()` releases its
array references without clearing process-wide allocator or module caches.
Concurrent use of the same instance, external context resets, and custom
memory allocator budgeting are not supported.

## Validation boundary

The initial implementation includes independent host tests, tiny NumPy
least-squares and exhaustive-period oracles, and GPU test cases. None have
been run during framework construction. CUDA compilation and package building
have also not been exercised.

The next validation phase should establish kernel correctness, memory-budget
behavior, tie handling, large-time precision, missing-data behavior, and
injection recovery before profiling performance. A few strong-signal examples
would not establish detection sensitivity or false-alarm calibration.

The present scope is a single-device two-stage search with fixed templates.
It does not include detrending, eccentric duration bounds, multi-device
scheduling, peak refinement, iterative multi-planet extraction, or a benchmark
comparison framework. Additional features should preserve the separation
between data objects, grid planning, GPU execution, and result analysis.
