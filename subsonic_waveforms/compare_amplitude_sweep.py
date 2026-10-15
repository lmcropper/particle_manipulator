"""Sweep the amplitude of one saved optimized air-displacement waveform.

The waveform shape, frequency, duration, air properties, and particle density
come from the selected .mat file. A multiplier scales both air displacement
and velocity. It is deliberately not clamped to the original optimization
limits, so the full 0.1x to 10x response can be seen.

Example:
    python compare_amplitude_sweep.py optimization_results/result.mat --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from optimization_results import load_archive
from optimize_particle_waveform import FourierWaveform
from optimize_segment_waveform import FilteredSegmentWaveform, SegmentWaveform
from piston_particle_solver import AirProperties, Particle, PistonFlow, simulate


DEFAULT_FACTORS = np.logspace(-1.0, 1.0, 21)


def scalar(archive: dict, key: str):
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"expected a scalar for {key}")
    return value.reshape(-1)[0]


def parse_positive_list(value: str) -> np.ndarray:
    try:
        values = np.asarray([float(part.strip()) for part in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("provide comma-separated numbers") from exc
    if not values.size or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise argparse.ArgumentTypeError("all values must be finite and positive")
    return np.unique(values)


def make_waveform(archive: dict, frequency: float, scale_m: float):
    """Rebuild the saved position shape and its physical velocity at 1x."""
    kind = str(scalar(archive, "waveform_kind"))
    if kind == "fourier":
        shape = FourierWaveform(
            coefficients=np.atleast_1d(np.asarray(archive["fourier_coefficients"], dtype=float)),
            harmonics=int(scalar(archive, "fourier_harmonics")),
        )
        position = lambda phase: scale_m * shape(phase)
        velocity = lambda phase: shape.velocity(phase, frequency, scale_m)
        jump_m_per_s = 0.0
    elif kind == "segment":
        source = SegmentWaveform(
            durations=np.atleast_1d(np.asarray(archive["segment_durations"], dtype=float)),
            vertices=np.atleast_1d(np.asarray(archive["segment_position_vertices"], dtype=float)),
        )
        cutoff = float(scalar(archive, "lowpass_cutoff_hz"))
        shape = FilteredSegmentWaveform(source, frequency, cutoff)
        position = lambda phase: scale_m * shape(phase)
        velocity = lambda phase: shape.velocity(phase, scale_m)
        segment_speeds = scale_m * frequency * source.phase_slopes
        jump_m_per_s = float(np.max(np.abs(segment_speeds - np.roll(segment_speeds, 1))))
    else:
        raise ValueError(f"unsupported waveform kind: {kind}")
    return position, velocity, jump_m_per_s


def select_archive(parser: argparse.ArgumentParser) -> Path:
    directory = Path(__file__).resolve().parent.parent / "optimization_results"
    choices = sorted(directory.glob("*.mat"), key=lambda path: path.stat().st_mtime, reverse=True)[:20]
    if not choices:
        parser.error(f"no .mat files found in {directory}; provide a file path")
    print("Recent optimization results:")
    for number, path in enumerate(choices, 1):
        print(f"  {number:2d}. {path.name}")
    try:
        answer = input("Select a number or enter a .mat file path: ").strip()
    except EOFError:
        parser.error("provide a .mat file path when running without a terminal")
    if answer.isdecimal() and 1 <= int(answer) <= len(choices):
        return choices[int(answer) - 1]
    return Path(answer).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mat_file", type=Path, nargs="?",
        help="Completed optimization .mat file (omit to choose from recent results)",
    )
    parser.add_argument(
        "--factors", type=parse_positive_list, default=DEFAULT_FACTORS,
        help="Comma-separated amplitude multipliers (default: 21 log-spaced values from 0.1 to 10)",
    )
    parser.add_argument(
        "--sizes-um", type=parse_positive_list,
        help="Optional particle diameters in micrometers (default: sizes saved in the .mat)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Evaluation device (auto uses CUDA when available)",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        help="CSV and PNG path prefix (default: beside the .mat file)",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening it")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mat_file is None:
        args.mat_file = select_archive(parser)
    if not args.mat_file.is_file():
        parser.error(f"optimization result not found: {args.mat_file}")
    archive = load_archive(args.mat_file)
    settings = json.loads(str(scalar(archive, "settings_json")))
    frequency = float(scalar(archive, "frequency_hz"))
    duration = float(scalar(archive, "simulation_duration_s"))
    scale_m = float(scalar(archive, "selected_displacement_scale_m"))
    direction = float(scalar(archive, "direction"))
    density = float(scalar(archive, "particle_density_kg_m3"))
    air = AirProperties(
        density=float(scalar(archive, "air_density_kg_m3")),
        dynamic_viscosity=float(scalar(archive, "air_viscosity_pa_s")),
    )
    saved_sizes = np.atleast_1d(np.asarray(archive["particle_sizes_um"], dtype=float))
    sizes_um = saved_sizes if args.sizes_um is None else args.sizes_um
    factors = np.asarray(args.factors, dtype=float)
    position, velocity, base_jump = make_waveform(archive, frequency, scale_m)
    particles = [Particle(diameter=size * 1e-6, density=density) for size in sizes_um]

    # Dense sampling gives the plotted peak values; the reconstructed waveform
    # itself is used for integration. No multiplier is capped or renormalized.
    phase = np.arange(16384, dtype=float) / 16384
    base_peak_position_m = float(np.max(np.abs(position(phase))))
    base_peak_speed = float(np.max(np.abs(velocity(phase))))
    peak_position_mm = factors * base_peak_position_m * 1e3
    peak_speed = factors * base_peak_speed
    jumps = factors * base_jump
    position_limit_mm = float(scalar(archive, "peak_displacement_limit_m")) * 1e3
    speed_limit = float(scalar(archive, "max_air_speed_m_per_s"))
    jump_limit = float(settings.get("max_velocity_jump", np.inf))
    over_limits = (
        (peak_position_mm > position_limit_mm * (1 + 1e-6))
        | (peak_speed > speed_limit * (1 + 1e-6))
        | (jumps > jump_limit * (1 + 1e-6))
    )
    limit_factor = min(
        position_limit_mm / (base_peak_position_m * 1e3) if base_peak_position_m else np.inf,
        speed_limit / base_peak_speed if base_peak_speed else np.inf,
        jump_limit / base_jump if base_jump else np.inf,
    )

    use_cuda = args.device != "cpu"
    integrate_batch = None
    if use_cuda:
        try:
            from cuda_backend import cupy_module, integrate_batch as cuda_integrate_batch

            cupy_module()
            integrate_batch = cuda_integrate_batch
        except RuntimeError as exc:
            if args.device == "cuda":
                parser.error(str(exc))
            use_cuda = False
            print(f"CUDA unavailable; using CPU ({exc})", flush=True)

    backend = "CUDA estimate" if use_cuda else "CPU"
    print(
        f"{args.mat_file.name}: {frequency:g} Hz, {duration:g} s, "
        f"{len(sizes_um)} particle sizes, {len(factors)} amplitudes ({backend}).",
        flush=True,
    )
    if use_cuda:
        assert integrate_batch is not None
        sample_phase = np.arange(8192, dtype=float) / 8192
        base_wave = np.asarray(velocity(sample_phase), dtype=float)
        waves = factors[:, None] * base_wave[None, :]
        final_positions, _steps = integrate_batch(
            waves, frequency, duration,
            np.asarray([particle.diameter for particle in particles]),
            np.asarray([particle.density for particle in particles]),
            air.density, air.dynamic_viscosity,
            int(settings.get("cuda_steps_per_period", 1000)),
        )
        displacements_m = final_positions
    else:
        displacements_m = np.empty((len(factors), len(particles)))
        for row, factor in enumerate(factors):
            piston = PistonFlow(
                frequency_hz=frequency,
                peak_speed=float(factor),
                waveform=velocity,
            )
            for column, particle in enumerate(particles):
                result = simulate(piston, particle=particle, air=air, duration=duration)
                displacements_m[row, column] = result.position[-1] - result.position[0]
            print(f"Evaluated {factor:.4g}x ({row + 1}/{len(factors)})", flush=True)

    directed_mm = direction * displacements_m * 1e3
    mean_mm = np.mean(directed_mm, axis=1)
    spread_mm = np.std(directed_mm, axis=1)
    range_mm = np.ptp(directed_mm, axis=1)
    minimum_mm = float(scalar(archive, "minimum_mean_transport_mm"))
    reward = float(scalar(archive, "transport_reward_weight"))
    score_mm = spread_mm - reward * (mean_mm - minimum_mm)

    print(
        f"{'Factor':>8} {'Peak mm':>9} {'Speed m/s':>10} {'Mean mm':>10} "
        f"{'Spread mm':>10} {'Range mm':>9} {'Score mm':>9} {'Limits':>8}"
    )
    for row, factor in enumerate(factors):
        print(
            f"{factor:8.3f} {peak_position_mm[row]:9.3f} {peak_speed[row]:10.3f} "
            f"{mean_mm[row]:10.3f} {spread_mm[row]:10.3f} {range_mm[row]:9.3f} "
            f"{score_mm[row]:9.3f} {'OVER' if over_limits[row] else 'within':>8}"
        )
    print(
        f"Saved limits: peak air displacement {position_limit_mm:g} mm, "
        f"air speed {speed_limit:g} m/s"
        + (f", unfiltered segment speed jump {jump_limit:g} m/s." if np.isfinite(jump_limit) else ".")
    )
    if np.array_equal(sizes_um, saved_sizes) and np.any(np.isclose(factors, 1.0)):
        row = int(np.argmin(np.abs(factors - 1.0)))
        print(
            f"At 1x: {backend} mean {mean_mm[row]:.3f} mm, spread {spread_mm[row]:.3f} mm; "
            f"saved CPU mean {float(scalar(archive, 'mean_directed_transport_mm')):.3f} mm, "
            f"spread {float(scalar(archive, 'transport_spread_mm')):.3f} mm."
        )

    prefix = args.output_prefix or args.mat_file.with_name(args.mat_file.stem + "_amplitude_sweep")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "factor", "solver", "peak_air_displacement_mm", "peak_air_speed_m_per_s",
            "unfiltered_segment_jump_m_per_s", "exceeds_saved_limits", "mean_directed_mm",
            "spread_mm", "range_mm", "score_mm", "meets_transport_minimum",
            *(f"directed_{size:g}um_mm" for size in sizes_um),
        ))
        for row, factor in enumerate(factors):
            writer.writerow((
                factor, backend, peak_position_mm[row], peak_speed[row], jumps[row],
                bool(over_limits[row]), mean_mm[row], spread_mm[row], range_mm[row],
                score_mm[row], bool(mean_mm[row] >= minimum_mm), *directed_mm[row],
            ))

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for column, size in enumerate(sizes_um):
        label = f"{size:g} µm"
        axes[0, 0].plot(factors, directed_mm[:, column], "o-", label=label)
        axes[0, 1].plot(factors, directed_mm[:, column], "o-", label=label)
    axes[0, 0].set_title("Full amplitude range")
    axes[0, 0].set_ylabel("Directed net transport [mm]")
    axes[0, 0].legend(title="Diameter", ncol=min(4, len(sizes_um)))
    axes[0, 1].set_title("Within saved airflow limits")
    axes[0, 1].set_ylabel("Directed net transport [mm]")
    within = factors <= limit_factor * (1 + 1e-6)
    if np.any(within):
        zoom_max = min(limit_factor, factors[-1])
        axes[0, 1].set_xlim(factors[0], max(zoom_max, factors[0] * 1.01))
        visible = directed_mm[within]
        low, high = float(np.min(visible)), float(np.max(visible))
        padding = max(0.05 * (high - low), 0.05)
        axes[0, 1].set_ylim(low - padding, high + padding)
    else:
        axes[0, 1].text(0.5, 0.5, "No sampled amplitude meets saved limits",
                        transform=axes[0, 1].transAxes, ha="center", va="center")
    axes[1, 0].plot(factors, mean_mm, "o-", color="tab:blue", label="Mean transport")
    axes[1, 0].fill_between(factors, mean_mm - spread_mm, mean_mm + spread_mm,
                            color="tab:blue", alpha=0.18, label="Mean ± spread")
    axes[1, 0].axhline(minimum_mm, color="0.5", linestyle=":", label="Minimum")
    axes[1, 0].set_ylabel("Directed transport [mm]")
    axes[1, 0].legend()
    axes[1, 1].plot(factors, spread_mm, "o-", color="tab:orange")
    axes[1, 1].set_ylabel("Spread across sizes [mm]")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.axvline(1.0, color="tab:red", linestyle="--", alpha=0.8)
        axis.grid(True, which="both", alpha=0.3)
        axis.set_xlabel("Amplitude multiplier")
        if axis is not axes[0, 1] and limit_factor < factors[-1]:
            axis.axvspan(max(limit_factor, factors[0]), factors[-1],
                         color="tab:red", alpha=0.08)
    figure.suptitle(
        f"Amplitude sweep of saved {frequency:g} Hz waveform ({backend})\n"
        "Red shading marks amplitudes beyond the saved optimization limits"
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    print(f"Saved comparison: {csv_path} and {png_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
