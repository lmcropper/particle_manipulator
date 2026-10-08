"""Compare glass microsphere motion across a range of particle diameters.

Example:
    python compare_particle_sizes.py --frequency 10 --peak-speed 2
    python compare_particle_sizes.py --sizes-um 20,40,60,80,100 --duration 3
"""

from __future__ import annotations

import argparse

import numpy as np

from piston_particle_solver import Particle, PistonFlow, simulate


def parse_sizes(value: str) -> list[float]:
    """Parse and validate a comma-separated list of diameters in micrometers."""
    try:
        sizes = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated numbers") from exc

    if not sizes:
        raise argparse.ArgumentTypeError("provide at least one particle size")
    if any(not np.isfinite(size) or size <= 0.0 for size in sizes):
        raise argparse.ArgumentTypeError("particle sizes must be positive finite numbers")
    if any(size < 20.0 or size > 100.0 for size in sizes):
        raise argparse.ArgumentTypeError("particle sizes must be between 20 and 100 µm")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("particle sizes must not contain duplicates")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=10.0, help="Drive frequency [Hz].")
    parser.add_argument("--peak-speed", type=float, default=2.0, help="Peak air speed [m/s].")
    parser.add_argument(
        "--second-harmonic",
        type=float,
        default=0.45,
        help="Second-harmonic coefficient for the asymmetric waveform.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration [s] (default: 20 cycles).",
    )
    parser.add_argument(
        "--sizes-um",
        type=parse_sizes,
        default=parse_sizes("20,30,40,50,60,70,80,90,100"),
        help="Comma-separated particle diameters in micrometers (20 to 100).",
    )
    parser.add_argument(
        "--particle-density",
        type=float,
        default=2500.0,
        help="Glass particle density [kg/m^3].",
    )
    parser.add_argument("--csv", help="CSV containing phase and air-velocity columns.")
    parser.add_argument("--no-plot", action="store_true", help="Print results without plotting.")
    args = parser.parse_args()

    if args.frequency <= 0.0:
        parser.error("--frequency must be positive")
    if args.peak_speed <= 0.0:
        parser.error("--peak-speed must be positive")
    if args.particle_density <= 0.0:
        parser.error("--particle-density must be positive")
    duration = args.duration if args.duration is not None else 20.0 / args.frequency
    if duration <= 0.0:
        parser.error("--duration must be positive")

    if args.csv:
        piston = PistonFlow.from_csv(args.csv, frequency_hz=args.frequency)
    else:
        piston = PistonFlow.asymmetric_harmonic(
            frequency_hz=args.frequency,
            peak_speed=args.peak_speed,
            second_harmonic=args.second_harmonic,
        )

    results = []
    for diameter_um in args.sizes_um:
        particle = Particle(
            diameter=diameter_um * 1e-6,
            density=args.particle_density,
        )
        result = simulate(piston, particle=particle, duration=duration)
        results.append((diameter_um, particle, result))

    print(f"Simulated {len(results)} particle sizes for {duration:g} s at {args.frequency:g} Hz")
    print(f"Glass density: {args.particle_density:g} kg/m^3")
    print(f"{'Diameter [µm]':>14} {'Final displacement [mm]':>25} {'Final speed [m/s]':>19} {'Max Re':>10}")
    for diameter_um, _particle, result in results:
        displacement_mm = (result.position[-1] - result.position[0]) * 1e3
        print(
            f"{diameter_um:14g} {displacement_mm:25.6f} "
            f"{result.particle_velocity[-1]:19.6g} {result.reynolds_number.max():10.4g}"
        )

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(results)))
    for color, (diameter_um, _particle, result) in zip(colors, results):
        elapsed = result.time - result.time[0]
        position_mm = (result.position - result.position[0]) * 1e3
        axes[0].plot(elapsed, position_mm, color=color, label=f"{diameter_um:g} µm")
        axes[1].plot(elapsed, result.particle_velocity, color=color)

    axes[0].set_ylabel("Displacement [mm]")
    axes[0].legend(title="Diameter", ncol=3)
    axes[1].set_ylabel("Particle velocity [m/s]")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"Glass microsphere response to asymmetric piston flow ({args.frequency:g} Hz)")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
