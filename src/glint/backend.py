"""Lazy CuPy loading and device-specific CUDA module compilation."""

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def cupy():
    """Import the GPU backend only when a search is requested."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GLINT requires CuPy for searches. Install the CuPy distribution "
            "matching your CUDA runtime, or use the cuda12/cuda13 installation extra."
        ) from exc
    return cp


@lru_cache(maxsize=None)
def load_module(device_id):
    """Compile packaged CUDA source with NVRTC on the selected device."""
    cp = cupy()
    with cp.cuda.Device(device_id):
        source = files("glint").joinpath("kernels.cu").read_text(encoding="utf-8")
        module = cp.RawModule(code=source, options=("--std=c++11",))
        module.compile()
    return module


def available_memory(cp):
    """Include unused blocks already held by CuPy's default allocator."""
    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
