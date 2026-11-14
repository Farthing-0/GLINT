"""GPU availability is checked only by tests that request the GPU fixture."""

import os

import pytest


@pytest.fixture
def gpu_device():
    cp = pytest.importorskip("cupy")
    device = int(os.environ.get("GLINT_TEST_DEVICE", "0"))
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    if device < 0 or device >= count:
        pytest.skip(f"CUDA device {device} is unavailable")
    with cp.cuda.Device(device):
        yield device
