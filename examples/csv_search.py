"""Search a CSV containing time, flux, and flux_error columns."""

import argparse
from pathlib import Path

import numpy as np

from glint import GLINT, SearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--period-min", type=float, default=1.0)
    parser.add_argument("--period-max", type=float, default=12.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/search.npz"))
    args = parser.parse_args()
    data = np.genfromtxt(args.input, delimiter=",", names=True, ndmin=1)
    required = {"time", "flux", "flux_error"}
    if not required.issubset(data.dtype.names or ()):
        parser.error("The CSV must contain time, flux, and flux_error columns")
    config = SearchConfig(period_min=args.period_min, period_max=args.period_max,
                          device_id=args.device)
    with GLINT(data["time"], data["flux"], data["flux_error"], config=config) as search:
        result = search.power()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(result.best.to_dict() if result.best else "No valid periodic candidate")


if __name__ == "__main__":
    main()
