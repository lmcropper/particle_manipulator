"""Optimize a periodic piston-flow waveform for particle transport.

The optimizer minimizes displacement spread across sizes, subject to a minimum
mean signed particle displacement over the requested duration. The waveform is
a zero-mean Fourier series with a fixed peak air speed.

Example:
    python optimize_particle_waveform.py --duration 1 --frequency 10 --peak-speed 2
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

from piston_particle_solver import Particle, PistonFlow, SimulationResult, simulate


def parse_sizes(value: str) -> list[float]:
    try:
        sizes = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated numbers") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("provide at least one particle size")
    if any(not np.isfinite(size) or size < 20.0 or size > 100.0 for size in sizes):
        raise argparse.ArgumentTypeError("particle sizes must be between 20 and 100 µm")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("particle sizes must not contain duplicates")
    return sizes


@dataclass(frozen=True)
class FourierWaveform:
    """Unit-peak, zero-mean waveform with a fixed fundamental sine term."""

    coefficients: np.ndarray
    harmonics: int

    def __post_init__(self) -> None:
        # Normalize on a dense phase grid so the requested peak speed is met.
        phase = np.linspace(0.0, 1.0, 16384, endpoint=False)
        peak = np.max(np.abs(self.raw(phase)))
        object.__setattr__(self, "_normalization", float(peak))

    def raw(self, phase: float | np.ndarray) -> float | np.ndarray:
        phase_array = np.asarray(phase)
        angle = 2.0 * np.pi * phase_array
        values = np.sin(angle)  # Fundamental sets scale and removes degeneracy.
        for harmonic in range(2, self.harmonics + 1):
            sine_coefficient = self.coefficients[2 * (harmonic - 2)]
            cosine_coefficient = self.coefficients[2 * (harmonic - 2) + 1]
            values = values + sine_coefficient * np.sin(harmonic * angle)
            values = values + cosine_coefficient * np.cos(harmonic * angle)
        if np.ndim(phase_array) == 0:
            return float(values)
        return values

    def __call__(self, phase: float) -> float:
        return float(self.raw(phase) / self._normalization)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=10.0, help="Drive frequency [Hz].")
    parser.add_argument("--duration", type=float, default=1.0, help="Optimization duration [s].")
    parser.add_argument("--peak-speed", type=float, default=2.0, help="Maximum air speed [m/s].")
    parser.add_argument(
        "--sizes-um",
        type=parse_sizes,
        default=parse_sizes("20,30,40,50,60,70,80,90,100"),
        help="Comma-separated particle diameters [µm] (20 to 100).",
    )
    parser.add_argument("--particle-density", type=float, default=157.37, help="Glass density [kg/m^3].")
    parser.add_argument("--harmonics", type=int, default=3, help="Number of Fourier harmonics (2 or more).")
    parser.add_argument(
        "--min-displacement-mm",
        type=float,
        default=10.0,
        help="Minimum mean directed displacement across sizes [mm].",
    )
    parser.add_argument(
        "--direction",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Preferred transport direction (+1 or -1 along the solver axis).",
    )
    parser.add_argument("--maxiter", type=int, default=20, help="Differential-evolution iterations.")
    parser.add_argument("--popsize", type=int, default=6, help="Population multiplier for the optimizer.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for repeatable optimization.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_waveform.csv"),
        help="CSV output path for the optimized phase/velocity waveform.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not open result plots.")
    return parser


def write_waveform_csv(path: Path, waveform: FourierWaveform, peak_speed: float) -> None:
    phase = np.linspace(0.0, 1.0, 1001)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("phase", "velocity_m_per_s"))
        writer.writerows((float(p), float(peak_speed * waveform(float(p)))) for p in phase)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frequency <= 0.0 or not np.isfinite(args.frequency):
        parser.error("--frequency must be positive and finite")
    if args.duration <= 0.0 or not np.isfinite(args.duration):
        parser.error("--duration must be positive and finite")
    if args.peak_speed <= 0.0 or not np.isfinite(args.peak_speed):
        parser.error("--peak-speed must be positive and finite")
    if args.particle_density <= 0.0 or not np.isfinite(args.particle_density):
        parser.error("--particle-density must be positive and finite")
    if args.harmonics < 2:
        parser.error("--harmonics must be at least 2")
    if args.min_displacement_mm < 0.0 or not np.isfinite(args.min_displacement_mm):
        parser.error("--min-displacement-mm must be nonnegative and finite")
    if args.maxiter < 1 or args.popsize < 1:
        parser.error("--maxiter and --popsize must be positive integers")

    coefficient_count = 2 * (args.harmonics - 1)
    bounds = [(-1.0, 1.0)] * coefficient_count
    size_particles = [
        (diameter_um, Particle(diameter=diameter_um * 1e-6, density=args.particle_density))
        for diameter_um in args.sizes_um
    ]
    best_progress = {
        "objective": float("inf"),
        "mean_directed_mm": float("nan"),
        "spread_mm": float("nan"),
        "meets_floor": False,
    }
    iteration = 0

    def simulate_candidate(
        coefficients: np.ndarray,
    ) -> tuple[list[SimulationResult], FourierWaveform]:
        waveform = FourierWaveform(np.asarray(coefficients, dtype=float), args.harmonics)
        piston = PistonFlow(
            frequency_hz=args.frequency,
            peak_speed=args.peak_speed,
            waveform=waveform,
        )
        results = [
            simulate(piston, particle=particle, duration=args.duration)
            for _, particle in size_particles
        ]
        return results, waveform

    @lru_cache(maxsize=4096)
    def displacement_values(*coefficients: float) -> tuple[float, ...]:
        # Constraint and objective calls share their simulation results.
        results, _waveform = simulate_candidate(np.asarray(coefficients, dtype=float))
        return tuple(float(result.position[-1] - result.position[0]) for result in results)

    def objective(coefficients: np.ndarray) -> float:
        displacements = np.asarray(displacement_values(*map(float, coefficients)))
        directed_mm = args.direction * displacements * 1e3
        spread_mm = float(np.std(displacements) * 1e3)
        # Once the transport floor is met, only dispersion determines fitness.
        if directed_mm.mean() >= args.min_displacement_mm and spread_mm < best_progress["objective"]:
            best_progress["objective"] = spread_mm
            best_progress["mean_directed_mm"] = float(np.mean(directed_mm))
            best_progress["spread_mm"] = spread_mm
            best_progress["meets_floor"] = True
        return spread_mm

    def transport_constraint(coefficients: np.ndarray) -> float:
        displacements = np.asarray(displacement_values(*map(float, coefficients)))
        directed_mm = args.direction * displacements * 1e3
        return float(np.mean(directed_mm) - args.min_displacement_mm)

    def report_progress(_coefficients: np.ndarray, convergence: float) -> bool:
        nonlocal iteration
        iteration += 1
        floor_status = "met" if best_progress["meets_floor"] else "not met"
        print(
            f"Iteration {iteration}/{args.maxiter}: "
            f"best mean transport {best_progress['mean_directed_mm']:.3f} mm; "
            f"size spread {best_progress['spread_mm']:.3f} mm; "
            f"{args.min_displacement_mm:g} mm floor {floor_status}; "
            f"convergence {convergence:.3g}",
            flush=True,
        )
        return False

    print(
        f"Optimizing {len(size_particles)} sizes over {args.duration:g} s "
        f"({args.frequency:g} Hz, peak speed {args.peak_speed:g} m/s)..."
    )
    optimization = differential_evolution(
        objective,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=True,
        updating="immediate",
        workers=1,
        callback=report_progress,
        constraints=(NonlinearConstraint(transport_constraint, 0.0, np.inf),),
    )

    results, waveform = simulate_candidate(optimization.x)
    displacements = np.asarray([result.position[-1] - result.position[0] for result in results])
    directed_mm = args.direction * displacements * 1e3
    mean_directed_mm = float(np.mean(directed_mm))
    if mean_directed_mm < args.min_displacement_mm - 0.01:
        raise SystemExit(
            "No feasible waveform found: mean directed displacement reached "
            f"{mean_directed_mm:.3f} mm, below the required "
            f"{args.min_displacement_mm:.3f} mm. Try more iterations, more harmonics, "
            "or a higher peak speed."
        )
    print(f"Optimizer converged: {optimization.success} ({optimization.message})")
    print(f"Optimized size-to-size spread: {optimization.fun * 1e3:.6f} mm")
    print(f"Mean directed transport: {np.mean(directed_mm):.6f} mm")
    print(f"Size-to-size standard deviation: {np.std(displacements) * 1e3:.6f} mm")
    print(f"Size-to-size range: {np.ptp(displacements) * 1e3:.6f} mm")
    print(f"Minimum mean directed displacement required: {args.min_displacement_mm:.6f} mm")
    print(f"Saved waveform: {args.output}")
    print(f"{'Diameter [µm]':>14} {'Net displacement [mm]':>23} {'Directed [mm]':>16}")
    for (diameter_um, _particle), result, directed_displacement in zip(
        size_particles, results, directed_mm
    ):
        displacement_mm = (result.position[-1] - result.position[0]) * 1e3
        print(f"{diameter_um:14g} {displacement_mm:23.6f} {directed_displacement:16.6f}")

    write_waveform_csv(args.output, waveform, args.peak_speed)

    if args.no_plot:
        return
    import matplotlib.pyplot as plt

    phase = np.linspace(0.0, 1.0, 1000, endpoint=False)
    figure, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(phase, [args.peak_speed * waveform(float(p)) for p in phase])
    axes[0].set_ylabel("Air velocity [m/s]")
    axes[0].set_xlabel("Waveform phase")
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(results)))
    for color, ((diameter_um, _particle), result) in zip(colors, zip(size_particles, results)):
        elapsed = result.time - result.time[0]
        displacement_mm = (result.position - result.position[0]) * 1e3
        axes[1].plot(elapsed, displacement_mm, color=color, label=f"{diameter_um:g} µm")
    axes[1].set_ylabel("Displacement [mm]")
    axes[1].legend(title="Diameter", ncol=3)
    axes[2].bar([item[0] for item in size_particles], displacements * 1e3, width=6.0)
    axes[2].set_xlabel("Particle diameter [µm]")
    axes[2].set_ylabel("Net displacement [mm]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Optimized particle transport waveform")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
