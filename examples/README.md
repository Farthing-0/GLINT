# Examples

Install GLINT and a compatible CuPy distribution before running these scripts
from the project root. Plotting additionally uses the `examples` extra.

```bash
python examples/synthetic_search.py --device 0
python examples/synthetic_search.py --device 0 --plot
python examples/csv_search.py lightcurve.csv --period-min 1 --period-max 12
```

The synthetic example generates its own observations, transit injection,
noise, and observing gap with a fixed random seed. It needs no network access
or downloaded data. The CSV example reads caller-provided observations with
the header `time,flux,flux_error`. Time is in days, flux is normalized to one,
and uncertainties use the same units as flux.

Both scripts write a compressed NumPy result to `outputs/` by default. Load
it with `np.load(path, allow_pickle=False)`. The examples have not been run
during the initial framework implementation.
