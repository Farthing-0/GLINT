"""Search a deterministic synthetic light curve without downloads or data files."""

import argparse
from pathlib import Path

import numpy as np

from glint import GLINT, SearchConfig


def make_light_curve():
    """Inject a trapezoid into gapped observations with Gaussian noise."""
    rng = np.random.default_rng(42)
    time = np.arange(0, 24, 0.02)
    time = time[(time < 9) | (time > 11)]
    offset = (time - 1.25 + 1.6) % 3.2 - 1.6
    deficit = np.clip((0.08 - np.abs(offset)) / 0.025, 0, 1)
    error = np.full(time.size, 0.0005)
    flux = 1.0 - 0.004 * deficit + rng.normal(size=time.size) * error
    return time, flux, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic.npz"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    time, flux, error = make_light_curve()
    config = SearchConfig(period_min=2, period_max=4, duration_min=0.08,
                          duration_max=0.24, device_id=args.device)
    with GLINT(time, flux, error, config=config) as search:
        result = search.power()
        template = search.template
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print("Injected period: 3.2 days")
    print("Best candidate:", result.best.to_dict() if result.best else None)
    if args.plot:
        import matplotlib.pyplot as plt
        from glint.helpers import fold

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(result.periods, result.log_likelihood)
        axes[0].set(xlabel="Period (days)", ylabel="Log-likelihood improvement")
        if result.best is not None:
            candidate = result.best
            phase = fold(time, candidate.period, candidate.epoch)
            axes[1].scatter(phase, flux, s=4, alpha=0.5)
            grid = np.linspace(-candidate.period / 2, candidate.period / 2, 2000)
            axes[1].plot(grid, template.model(grid + candidate.epoch, candidate), color="tab:red")
        axes[1].set(xlabel="Time from transit (days)", ylabel="Normalized flux")
        figure.tight_layout()
        figure.savefig(args.output.with_suffix(".png"), dpi=160)


if __name__ == "__main__":
    main()
