"""GPU workspace and memory-aware drivers for the two search stages."""

import numpy as np

from .backend import available_memory, cupy, load_module
from .constants import MAX_GRID_AXIS, MAX_INT32
from .grid import duration_indices
from .results import LinearResult, SearchResult


class SearchWorkspace:
    """Own the device arrays of one light curve on one CUDA device.

    The linear maps stay on the device across period searches. The workspace
    uses the current CuPy stream; a completed public search synchronizes it.
    Instances are intended for serial use and must not be shared across threads.
    """

    def __init__(self, curve, template, durations, epochs, epoch_step, config):
        self.curve, self.config = curve, config
        self.durations, self.epochs, self.epoch_step = durations, epochs, epoch_step
        self.maps = None
        cp = cupy()
        if curve.time.size > MAX_INT32 - config.threads or durations.size > MAX_GRID_AXIS:
            raise ValueError("The observation or duration count exceeds the CUDA index limits")
        shape = (durations.size, epochs.size)
        required = 12 * int(np.prod(shape)) + 20 * curve.time.size + template.values.nbytes
        required += durations.nbytes
        residual, weight = curve.residual.astype(np.float32), curve.weight
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(weight)):
            raise ValueError("Flux residuals or inverse variances exceed the supported numeric range")
        if not np.any(weight > 0):
            raise ValueError("All observation weights underflow to zero")
        with cp.cuda.Device(config.device_id):
            budget = int(available_memory(cp) * config.memory_fraction)
            if required > budget:
                raise MemoryError(
                    f"The linear workspace needs {required / 2**20:.1f} MiB; "
                    f"the configured budget is {budget / 2**20:.1f} MiB. "
                    "Use fewer durations or a larger epoch_step_fraction."
                )
            self.module = load_module(config.device_id)
            self.time = cp.asarray(curve.offset_time)
            self.residual = cp.asarray(residual)
            self.weight = cp.asarray(weight)
            self.template = cp.asarray(template.values)
            self.device_durations = cp.asarray(durations)
            self.maps = tuple(cp.empty(shape, dtype=cp.float32) for _ in range(3))

    def linear_search(self):
        """Evaluate every duration and epoch once, leaving the maps on the GPU."""
        cp, cfg = cupy(), self.config
        with cp.cuda.Device(cfg.device_id):
            self.module.get_function("linear_search")(
                (min(self.epochs.size, MAX_GRID_AXIS), self.durations.size), (cfg.threads,),
                (self.time, self.residual, self.weight, np.int32(self.time.size),
                 self.template, np.int32(self.template.size), self.device_durations,
                 np.float64(self.epochs[0]), np.float64(self.epoch_step),
                 np.int32(self.epochs.size), np.int32(cfg.min_points), *self.maps),
                shared_mem=20 * cfg.threads,
            )
            cp.cuda.get_current_stream().synchronize()

    def linear_result(self):
        """Explicitly copy the potentially large single-event maps to the host."""
        cp = cupy()
        with cp.cuda.Device(self.config.device_id):
            arrays = [cp.asnumpy(array) for array in self.maps]
        return LinearResult(self.epochs + self.curve.reference_time, self.durations, *arrays)

    def period_search(self, periods):
        """Batch all requested periods without removing any grid candidates."""
        cp, cfg = cupy(), self.config
        nduration, nepoch = self.durations.size, self.epochs.size
        spacing = periods / self.epoch_step
        if not np.all(np.isfinite(spacing) & (spacing > 0)):
            raise ValueError("Period-to-epoch-step ratios exceed the supported numeric range")
        first, last = duration_indices(periods, self.durations, cfg)
        phase_counts = np.ceil(np.minimum(spacing, nepoch)).astype(np.int64)
        max_tiles = int((phase_counts.max() + cfg.threads - 1) // cfg.threads)
        if nduration * max_tiles >= MAX_INT32:
            raise ValueError("The period reduction exceeds the supported index range")
        dtypes = (np.float64, np.float64, np.float64, np.int32, np.int32, np.int32)
        output = [np.empty(periods.size, dtype=dtype) for dtype in dtypes]
        with cp.cuda.Device(cfg.device_id):
            # Five tile arrays, six result arrays, and three metadata arrays.
            per_period = 32 * nduration * max_tiles + 36 + 16
            budget = int(available_memory(cp) * cfg.memory_fraction)
            capacity = min(cfg.batch_size, periods.size, budget // per_period)
            if capacity < 1:
                raise MemoryError("Insufficient GPU memory for even one period batch")
            tile_size = capacity * nduration * max_tiles
            scratch = [cp.empty(tile_size, dtype=dtype)
                       for dtype in (np.float64, np.float64, np.float64, np.int32, np.int32)]
            reduced = [cp.empty(capacity, dtype=dtype) for dtype in dtypes]
            device_spacing = cp.empty(capacity, dtype=cp.float64)
            device_first = cp.empty(capacity, dtype=cp.int32)
            device_last = cp.empty(capacity, dtype=cp.int32)
            search = self.module.get_function("periodic_search")
            reduce = self.module.get_function("reduce_periods")
            for start in range(0, periods.size, capacity):
                stop = min(start + capacity, periods.size)
                count = stop - start
                tiles = int((phase_counts[start:stop].max() + cfg.threads - 1) // cfg.threads)
                device_spacing[:count].set(spacing[start:stop])
                device_first[:count].set(first[start:stop])
                device_last[:count].set(last[start:stop])
                search(
                    (tiles, count, nduration), (cfg.threads,),
                    (*self.maps, device_spacing, device_first, device_last,
                     np.int32(nepoch), np.int32(nduration), np.int32(tiles),
                     np.int32(cfg.min_transits), np.float64(cfg.depth_min),
                     np.int32(cfg.mode == "two_pass"), *scratch),
                    shared_mem=12 * cfg.threads,
                )
                reduce(
                    (count,), (cfg.threads,),
                    (*scratch, np.int32(nduration), np.int32(tiles), *reduced),
                    shared_mem=12 * cfg.threads,
                )
                for target, source in zip(output, reduced):
                    target[start:stop] = cp.asnumpy(source[:count])
            cp.cuda.get_current_stream().synchronize()
        return SearchResult(periods, *output, self.curve.reference_time + self.epochs[0],
                            self.epoch_step, self.durations)

    def close(self):
        """Release owned arrays to CuPy without clearing its process-wide pool."""
        self.maps = None
        self.time = self.residual = self.weight = self.template = self.device_durations = None
