"""Recreate an optimizer figure from its saved MATLAB result file.

Example:
    python plot_optimization_result.py optimization_results/segment_result.mat
    python plot_optimization_result.py result.mat --save-figure result.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optimization_results import load_archive, plot_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path, help="Saved Fourier or segment .mat result.")
    parser.add_argument(
        "--save-figure", type=Path, help="Also save the recreated figure (for example, PNG or PDF)."
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    args = parser.parse_args()

    archive = load_archive(args.mat_file)
    def value(name: str):
        return np.asarray(archive[name]).reshape(-1)[0]

    print(f"Loaded result: {args.mat_file}")
    print(f"Waveform: {value('waveform_kind')}; created: {value('created_at')}")
    print(
        f"Mean directed transport: {float(value('mean_directed_transport_mm')):.6f} mm; "
        f"spread: {float(value('transport_spread_mm')):.6f} mm; "
        f"score: {float(value('score_mm')):.6f} mm"
    )
    print(
        f"Converged: {bool(value('optimizer_success'))}; "
        f"transport minimum met: {bool(value('transport_minimum_met'))}"
    )
    figure = plot_archive(archive)
    if args.save_figure:
        args.save_figure.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_figure, dpi=180)
        print(f"Saved figure: {args.save_figure}")
    if args.no_show:
        import matplotlib.pyplot as plt

        plt.close(figure)
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
