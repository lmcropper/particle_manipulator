"""Compare a saved optimized waveform with triangle displacement waves.

The rising fraction is the fraction of a period spent moving from minimum to
maximum air displacement. Values near 0 or 1 are sawtooth-like; 0.5 is a
symmetric triangle. Physical limits and particle properties come from the .mat.

Example:
    python compare_triangle_symmetry.py optimization_results/segment_800Hz.mat
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from optimization_results import load_archive
from optimize_segment_waveform import FilteredSegmentWaveform, SegmentWaveform
from piston_particle_solver import AirProperties, Particle, PistonFlow, simulate


DEFAULT_FRACTIONS = np.r_[0.01, 0.02, np.arange(0.05, 1.0, 0.05), 0.98, 0.99]


def archive_scalar(archive: dict, name: str):
    return np.asarray(archive[name]).reshape(-1)[0]


def parse_fractions(value: str) -> np.ndarray:
    try:
        fractions = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fractions must be comma-separated numbers") from exc
    if not fractions.size or np.any(~np.isfinite(fractions)) or np.any((fractions <= 0) | (fractions >= 1)):
        raise argparse.ArgumentTypeError("each rising fraction must lie strictly between 0 and 1")
    return np.unique(fractions)


def metrics(displacements_m: np.ndarray, direction: float, minimum_mm: float, reward: float):
    directed_mm = direction * np.asarray(displacements_m, dtype=float) * 1e3
    mean_mm = np.mean(directed_mm, axis=-1)
    spread_mm = np.std(directed_mm, axis=-1)
    range_mm = np.ptp(directed_mm, axis=-1)
    score_mm = spread_mm - reward * (mean_mm - minimum_mm)
    return mean_mm, spread_mm, range_mm, score_mm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path, help="Saved optimized Fourier or segment .mat result.")
    parser.add_argument(
        "--fractions", type=parse_fractions, default=DEFAULT_FRACTIONS,
        help="Comma-separated rising fractions (default: 0.01, 0.02, 0.05 to 0.95, 0.98, 0.99).",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Baseline evaluation device (auto uses CUDA if available).",
    )
    parser.add_argument(
        "--verify-best", action="store_true",
        help="Recheck the best feasible triangle with the adaptive CPU solver.",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        help="Output CSV/PNG prefix (default: beside the .mat file).",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    archive = load_archive(args.mat_file)
    settings = json.loads(str(archive_scalar(archive, "settings_json")))
    frequency = float(archive_scalar(archive, "frequency_hz"))
    duration = float(archive_scalar(archive, "simulation_duration_s"))
    sizes_um = np.atleast_1d(np.asarray(archive["particle_sizes_um"], dtype=float))
    direction = float(archive_scalar(archive, "direction"))
    requested_peak_m = float(archive_scalar(archive, "peak_displacement_limit_m"))
    max_air_speed = float(archive_scalar(archive, "max_air_speed_m_per_s"))
    minimum_mm = float(archive_scalar(archive, "minimum_mean_transport_mm"))
    reward = float(archive_scalar(archive, "transport_reward_weight"))
    particle_density = float(archive_scalar(archive, "particle_density_kg_m3"))
    air = AirProperties(
        density=float(archive_scalar(archive, "air_density_kg_m3")),
        dynamic_viscosity=float(archive_scalar(archive, "air_viscosity_pa_s")),
    )
    kind = str(archive_scalar(archive, "waveform_kind"))
    lowpass_hz = float(archive_scalar(archive, "lowpass_cutoff_hz")) if kind == "segment" else 0.0
    max_velocity_jump = float(settings.get("max_velocity_jump", np.inf))
    steps_per_period = int(settings.get("cuda_steps_per_period", 1000))
    fractions = np.asarray(args.fractions, dtype=float)
    particles = [Particle(diameter=size * 1e-6, density=particle_density) for size in sizes_um]

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

    waveforms = []
    amplitudes = []
    peak_speeds = []
    for rising_fraction in fractions:
        source = SegmentWaveform(
            durations=np.asarray([rising_fraction, 1.0 - rising_fraction]),
            vertices=np.asarray([-1.0, 1.0, -1.0]),
        )
        waveform = FilteredSegmentWaveform(source, frequency, lowpass_hz)
        amplitude = waveform.limited_peak_displacement(
            requested_peak_m, max_air_speed, max_velocity_jump
        )
        waveforms.append(waveform)
        amplitudes.append(amplitude)
        peak_speeds.append(amplitude * frequency * waveform.phase_velocity_peak)
    amplitudes = np.asarray(amplitudes)
    peak_speeds = np.asarray(peak_speeds)

    backend = "CUDA estimate" if use_cuda else "CPU"
    print(
        f"Comparing {len(fractions)} triangle symmetries with {args.mat_file.name} "
        f"at {frequency:g} Hz for {duration:g} s ({backend}; low-pass {lowpass_hz:g} Hz).",
        flush=True,
    )
    if use_cuda:
        assert integrate_batch is not None
        phase = np.arange(8192, dtype=float) / 8192
        waves = np.asarray([
            waveform.velocity(phase, amplitude)
            for waveform, amplitude in zip(waveforms, amplitudes)
        ])
        baseline_displacements, _ = integrate_batch(
            waves, frequency, duration,
            np.asarray([particle.diameter for particle in particles]),
            np.asarray([particle.density for particle in particles]),
            air.density, air.dynamic_viscosity, steps_per_period,
        )
        if "cuda_estimate_displacements_m" in archive:
            optimized_displacements = np.atleast_1d(
                np.asarray(archive["cuda_estimate_displacements_m"], dtype=float)
            )
        else:
            saved_phase = np.atleast_1d(np.asarray(archive["waveform_phase"], dtype=float))
            saved_velocity = np.atleast_1d(np.asarray(archive["waveform_velocity_m_per_s"], dtype=float))
            optimized_wave = np.interp(
                phase, np.r_[saved_phase, 1.0], np.r_[saved_velocity, saved_velocity[0]]
            )
            optimized_positions, _ = integrate_batch(
                optimized_wave, frequency, duration,
                np.asarray([particle.diameter for particle in particles]),
                np.asarray([particle.density for particle in particles]),
                air.density, air.dynamic_viscosity, steps_per_period,
            )
            optimized_displacements = optimized_positions[0]
    else:
        baseline_displacements = np.empty((len(fractions), len(particles)))
        for row, (fraction, waveform, amplitude) in enumerate(zip(fractions, waveforms, amplitudes)):
            piston = PistonFlow(
                frequency_hz=frequency, peak_speed=1.0,
                waveform=lambda phase, wave=waveform, scale=amplitude: wave.velocity(phase, scale),
            )
            for column, particle in enumerate(particles):
                result = simulate(piston, particle=particle, air=air, duration=duration)
                baseline_displacements[row, column] = result.position[-1] - result.position[0]
            print(f"Evaluated rising fraction {fraction:.3f} ({row + 1}/{len(fractions)})", flush=True)
        optimized_displacements = np.atleast_1d(np.asarray(archive["net_displacements_m"], dtype=float))

    baseline_mean, baseline_spread, baseline_range, baseline_score = metrics(
        baseline_displacements, direction, minimum_mm, reward
    )
    optimized_mean, optimized_spread, optimized_range, optimized_score = (
        float(value) for value in metrics(optimized_displacements, direction, minimum_mm, reward)
    )
    feasible = baseline_mean >= minimum_mm
    feasible_indices = np.flatnonzero(feasible)
    best_index = int(feasible_indices[np.argmin(baseline_score[feasible_indices])]) if feasible_indices.size else int(np.argmax(baseline_mean))

    print(
        f"Optimized: mean {optimized_mean:.3f} mm; spread {optimized_spread:.3f} mm; "
        f"score {optimized_score:.3f} mm ({backend})."
    )
    print(f"{'Rise':>7} {'Peak mm':>9} {'Speed m/s':>11} {'Mean mm':>10} {'Spread mm':>11} {'Score mm':>10} {'Feasible':>9}")
    for row, fraction in enumerate(fractions):
        print(
            f"{fraction:7.3f} {amplitudes[row] * 1e3:9.3f} "
            f"{peak_speeds[row]:11.3f} {baseline_mean[row]:10.3f} "
            f"{baseline_spread[row]:11.3f} {baseline_score[row]:10.3f} "
            f"{str(bool(feasible[row])):>9}"
        )
    print(
        f"Best {'feasible' if feasible_indices.size else 'available'} triangle: "
        f"rise {fractions[best_index]:.3f}; mean {baseline_mean[best_index]:.3f} mm; "
        f"spread {baseline_spread[best_index]:.3f} mm; score {baseline_score[best_index]:.3f} mm."
    )
    matched_index = int(np.argmin(np.abs(baseline_mean - optimized_mean)))
    print(
        f"Closest mean transport: triangle rise {fractions[matched_index]:.3f}; "
        f"mean {baseline_mean[matched_index]:.3f} mm; "
        f"spread {baseline_spread[matched_index]:.3f} mm."
    )
    if feasible_indices.size and optimized_mean >= minimum_mm:
        winner = "optimized waveform" if optimized_score < baseline_score[best_index] else "triangle"
        print(f"Lower feasible score: {winner}.")
    dominates_optimized = feasible & (baseline_mean >= optimized_mean) & (baseline_spread <= optimized_spread)
    if np.any(dominates_optimized):
        print(f"{np.count_nonzero(dominates_optimized)} triangle wave(s) match or beat optimized mean transport and spread.")
    else:
        print("No swept triangle matches or beats optimized mean transport and spread together.")

    if args.verify_best:
        waveform = waveforms[best_index]
        amplitude = amplitudes[best_index]
        piston = PistonFlow(
            frequency_hz=frequency, peak_speed=1.0,
            waveform=lambda phase: waveform.velocity(phase, amplitude),
        )
        verified_displacements = []
        for particle in particles:
            result = simulate(piston, particle=particle, air=air, duration=duration)
            verified_displacements.append(result.position[-1] - result.position[0])
        verified_displacements = np.asarray(verified_displacements)
        verified_mean, verified_spread, _, verified_score = (
            float(value) for value in metrics(verified_displacements, direction, minimum_mm, reward)
        )
        saved_cpu = np.atleast_1d(np.asarray(archive["net_displacements_m"], dtype=float))
        saved_mean, saved_spread, _, saved_score = (
            float(value) for value in metrics(saved_cpu, direction, minimum_mm, reward)
        )
        print(
            f"CPU check: best triangle mean {verified_mean:.3f} mm, spread {verified_spread:.3f} mm, "
            f"score {verified_score:.3f} mm; optimized mean {saved_mean:.3f} mm, "
            f"spread {saved_spread:.3f} mm, score {saved_score:.3f} mm."
        )

    prefix = args.output_prefix or args.mat_file.with_name(args.mat_file.stem + "_triangle_sweep")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    size_columns = [f"directed_{size:g}um_mm" for size in sizes_um]
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("waveform", "solver", "rising_fraction", "peak_displacement_mm", "peak_air_speed_m_per_s", "mean_directed_mm", "spread_mm", "range_mm", "score_mm", "feasible", *size_columns))
        writer.writerow(("optimized", backend, "", "", "", optimized_mean, optimized_spread, optimized_range, optimized_score, optimized_mean >= minimum_mm, *(direction * optimized_displacements * 1e3)))
        for row, fraction in enumerate(fractions):
            writer.writerow(("triangle", backend, fraction, amplitudes[row] * 1e3, peak_speeds[row], baseline_mean[row], baseline_spread[row], baseline_range[row], baseline_score[row], bool(feasible[row]), *(direction * baseline_displacements[row] * 1e3)))

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(fractions, baseline_mean, "o-")
    axes[0, 0].axhline(optimized_mean, color="tab:red", linestyle="--", label="Optimized")
    axes[0, 0].axhline(minimum_mm, color="0.5", linestyle=":", label="Minimum")
    axes[0, 0].set_ylabel("Mean directed transport [mm]")
    axes[0, 0].legend()
    axes[0, 1].plot(fractions, baseline_spread, "o-")
    axes[0, 1].axhline(optimized_spread, color="tab:red", linestyle="--", label="Optimized")
    axes[0, 1].set_ylabel("Transport spread [mm]")
    axes[0, 1].legend()
    for axis in axes[0]:
        axis.set_xlabel("Rising fraction (0.5 = symmetric triangle)")
    scatter = axes[1, 0].scatter(baseline_mean, baseline_spread, c=fractions, cmap="viridis")
    axes[1, 0].scatter([optimized_mean], [optimized_spread], marker="*", s=180, color="tab:red", label="Optimized")
    axes[1, 0].set_xlabel("Mean directed transport [mm]")
    axes[1, 0].set_ylabel("Transport spread [mm]")
    axes[1, 0].legend()
    figure.colorbar(scatter, ax=axes[1, 0], label="Rising fraction")
    axes[1, 1].plot(sizes_um, direction * optimized_displacements * 1e3, "o-", label="Optimized")
    axes[1, 1].plot(sizes_um, direction * baseline_displacements[best_index] * 1e3, "o-", label=f"Best triangle ({fractions[best_index]:.2f})")
    axes[1, 1].set_xlabel("Particle diameter [um]")
    axes[1, 1].set_ylabel("Directed displacement [mm]")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"Triangle symmetry versus optimized waveform at {frequency:g} Hz ({backend})")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    print(f"Saved comparison: {csv_path} and {png_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
