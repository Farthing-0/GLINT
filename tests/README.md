# Tests

These tests use locally generated arrays and a small NumPy reference solver.
They require no external repositories, downloaded light curves, or comparison
engines. NumPy reference routines are test code and are not a CPU search backend.

Install the project with its `test` extra before running the suite. GPU tests
also require CuPy, CUDA headers, and an NVIDIA device. Select the device with
`GLINT_TEST_DEVICE`; its default is zero.

```bash
python -m pytest -m "not gpu"
GLINT_TEST_DEVICE=0 python -m pytest -m gpu
```

The host tests cover input handling, configuration, grid bounds, and templates.
The GPU tests compare single-event fits and exhaustive periodic enumeration
against direct weighted least squares, including gaps and fractional period
strides. They also cover periods with insufficient observed events.

The initial framework has not been tested. These tests are supplied for the
subsequent validation phase; no numerical accuracy or runtime claim follows
from their presence.
