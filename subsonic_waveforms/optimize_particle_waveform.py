"""Optimize a periodic Fourier air-displacement waveform for particle transport.

The objective minimizes the spread of net transport among particle sizes, with
a small reward for longer mean directed transport and a minimum transport
constraint. Fourier coefficients specify air displacement; air velocity is its
time derivative. Peak displacement and air speed are capped.

Example:
    python optimize_particle_waveform.py --duration 1 --frequency 80 --peak-air-displacement-mm 10
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

from optimization_results import build_archive, plot_archive, save_archive
from piston_particle_solver import AirProperties, Particle, PistonFlow, SimulationResult, simulate


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
    """Unit-peak, zero-mean air-displacement shape with a fixed fundamental."""

    coefficients: np.ndarray
    harmonics: int

    def __post_init__(self) -> None:
        # Use the same dense grid to normalize displacement and bound its
        # derivative, which sets the physical air-velocity scale.
        grid_count = max(16384, int(np.ceil(8.0 * np.pi * self.harmonics)))
        phase = np.arange(grid_count, dtype=float) / grid_count
        # Bernstein's bound for a degree-N trigonometric polynomial gives a
        # conservative maximum between samples from its sampled maximum.
        sample_bound_factor = 1.0 / (1.0 - np.pi * self.harmonics / grid_count)
        peak = float(np.max(np.abs(self.raw(phase)))) * sample_bound_factor
        object.__setattr__(self, "_normalization", peak)
        derivative_peak = (
            float(np.max(np.abs(self.raw_phase_derivative(phase))))
            * sample_bound_factor / peak
        )
        object.__setattr__(self, "_phase_derivative_bound", derivative_peak)

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

    def raw_phase_derivative(self, phase: float | np.ndarray) -> float | np.ndarray:
        """Derivative of the unnormalized position with respect to phase."""
        phase_array = np.asarray(phase)
        angle = 2.0 * np.pi * phase_array
        values = 2.0 * np.pi * np.cos(angle)
        for harmonic in range(2, self.harmonics + 1):
            sine_coefficient = self.coefficients[2 * (harmonic - 2)]
            cosine_coefficient = self.coefficients[2 * (harmonic - 2) + 1]
            values = values + 2.0 * np.pi * harmonic * (
                sine_coefficient * np.cos(harmonic * angle)
                - cosine_coefficient * np.sin(harmonic * angle)
            )
        if phase_array.ndim == 0:
            return float(values)
        return values

    def __call__(self, phase: float | np.ndarray) -> float | np.ndarray:
        """Return normalized air displacement."""
        return self.raw(phase) / self._normalization

    def velocity(
        self, phase: float | np.ndarray, frequency_hz: float, peak_displacement_m: float
    ) -> float | np.ndarray:
        """Return physical air velocity from the displacement derivative."""
        return (
            peak_displacement_m * frequency_hz
            * self.raw_phase_derivative(phase) / self._normalization
        )

    def limited_peak_displacement(
        self, requested_peak_m: float, frequency_hz: float, max_air_speed: float
    ) -> float:
        return min(
            requested_peak_m,
            max_air_speed / (frequency_hz * self._phase_derivative_bound),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=80.0, help="Drive frequency [Hz].")
    parser.add_argument("--duration", type=float, default=1.0, help="Optimization duration [s].")
    parser.add_argument(
        "--peak-air-displacement-mm", type=float, default=10.0,
        help="Maximum peak absolute air displacement about its mean [mm].",
    )
    parser.add_argument(
        "--max-air-speed", type=float, default=5.0,
        help="Maximum absolute air velocity [m/s].",
    )
    parser.add_argument(
        "--sizes-um",
        type=parse_sizes,
        default=parse_sizes("20,30,40,50,60,70,80,90,100"),
        help="Comma-separated particle diameters [µm] (20 to 100).",
    )
    parser.add_argument("--particle-density", type=float, default=157.37, help="Glass density [kg/m^3].")
    parser.add_argument("--harmonics", type=int, default=11, help="Number of Fourier harmonics (2 or more).")
    parser.add_argument(
        "--min-mean-displacement-mm",
        type=float,
        default=5.0,
        help="Minimum mean directed displacement across particle sizes [mm].",
    )
    parser.add_argument(
        "--transport-reward-weight",
        type=float,
        default=0.3,
        help="Score reduction per mm of mean directed transport above the minimum "
        "(default: 0.2 mm per mm; lower score is better).",
    )
    parser.add_argument(
        "--direction",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Preferred transport direction (+1 or -1 along the solver axis).",
    )
    parser.add_argument("--maxiter", type=int, default=100, help="Differential-evolution iterations.")
    parser.add_argument(
        "--exploration-samples",
        type=int,
        default=10000,
        help="Number of random waveform coefficient sets to explore before optimization.",
    )
    parser.add_argument("--popsize", type=int, default=10, help="Population multiplier for the optimizer.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for repeatable optimization.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device (auto uses CUDA when CuPy and a CUDA GPU are available).",
    )
    parser.add_argument(
        "--cuda-steps-per-period",
        type=int,
        default=1000,
        help="Fixed RK4 steps per period for CUDA simulations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_waveform.csv"),
        help="CSV output path for phase, derived air velocity, and air displacement.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("optimization_results"),
        help="Directory for timestamped MATLAB result files.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not open result plots.")
    return parser


def write_waveform_csv(
    path: Path, waveform: FourierWaveform, frequency_hz: float, peak_displacement_m: float
) -> None:
    phase = np.linspace(0.0, 1.0, 1001)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        # Keep velocity in column 2 for PistonFlow.from_csv compatibility.
        writer.writerow(("phase", "velocity_m_per_s", "air_displacement_m"))
        writer.writerows(
            (
                float(p),
                float(waveform.velocity(float(p), frequency_hz, peak_displacement_m)),
                float(peak_displacement_m * waveform(float(p))),
            )
            for p in phase
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frequency <= 0.0 or not np.isfinite(args.frequency):
        parser.error("--frequency must be positive and finite")
    if args.duration <= 0.0 or not np.isfinite(args.duration):
        parser.error("--duration must be positive and finite")
    if args.peak_air_displacement_mm <= 0.0 or not np.isfinite(args.peak_air_displacement_mm):
        parser.error("--peak-air-displacement-mm must be positive and finite")
    if args.max_air_speed <= 0.0 or not np.isfinite(args.max_air_speed):
        parser.error("--max-air-speed must be positive and finite")
    if args.particle_density <= 0.0 or not np.isfinite(args.particle_density):
        parser.error("--particle-density must be positive and finite")
    if args.harmonics < 2:
        parser.error("--harmonics must be at least 2")
    if args.min_mean_displacement_mm < 0.0 or not np.isfinite(args.min_mean_displacement_mm):
        parser.error("--min-mean-displacement-mm must be nonnegative and finite")
    if args.transport_reward_weight < 0.0 or not np.isfinite(args.transport_reward_weight):
        parser.error("--transport-reward-weight must be nonnegative and finite")
    if args.maxiter < 1 or args.popsize < 1:
        parser.error("--maxiter and --popsize must be positive integers")
    if args.exploration_samples < 5:
        parser.error("--exploration-samples must be at least 5")
    if args.cuda_steps_per_period < 4:
        parser.error("--cuda-steps-per-period must be at least 4")

    use_cuda = args.device == "cuda"
    if args.device in {"auto", "cuda"}:
        try:
            try:
                from .cuda_backend import cupy_module, integrate_batch
            except ImportError:
                from cuda_backend import cupy_module, integrate_batch

            cupy_module()
            use_cuda = True
        except RuntimeError as exc:
            if args.device == "cuda":
                parser.error(str(exc))
            use_cuda = False
            print(f"CUDA unavailable; using CPU evaluations ({exc})")

    coefficient_count = 2 * (args.harmonics - 1)
    bounds = [(-1.0, 1.0)] * coefficient_count
    size_particles = [
        (diameter_um, Particle(diameter=diameter_um * 1e-6, density=args.particle_density))
        for diameter_um in args.sizes_um
    ]
    air = AirProperties()
    requested_peak_m = args.peak_air_displacement_mm * 1e-3
    cuda_displacement_cache: dict[tuple[float, ...], tuple[float, ...]] = {}
    iteration = 0

    def effective_peak_displacement(waveform: FourierWaveform) -> float:
        return waveform.limited_peak_displacement(
            requested_peak_m, args.frequency, args.max_air_speed
        )

    def simulate_candidate(
        coefficients: np.ndarray,
    ) -> tuple[list[SimulationResult], FourierWaveform]:
        waveform = FourierWaveform(np.asarray(coefficients, dtype=float), args.harmonics)
        effective_peak_m = effective_peak_displacement(waveform)
        piston = PistonFlow(
            frequency_hz=args.frequency,
            # This waveform already returns physical velocity in m/s.
            peak_speed=1.0,
            waveform=lambda phase: waveform.velocity(
                phase, args.frequency, effective_peak_m
            ),
        )
        results = [
            simulate(piston, particle=particle, duration=args.duration)
            for _, particle in size_particles
        ]
        return results, waveform

    @lru_cache(maxsize=4096)
    def displacement_values(*coefficients: float) -> tuple[float, ...]:
        # Repeated objective calls share their simulation results.
        results, _waveform = simulate_candidate(np.asarray(coefficients, dtype=float))
        return tuple(float(result.position[-1] - result.position[0]) for result in results)

    def cuda_displacement_values(candidates: np.ndarray) -> np.ndarray:
        """Evaluate a matrix of coefficient vectors in one GPU batch."""
        candidates = np.asarray(candidates, dtype=float)
        keys = [tuple(map(float, row)) for row in candidates]
        missing = list(dict.fromkeys(key for key in keys if key not in cuda_displacement_cache))
        for start in range(0, len(missing), 256):
            batch_keys = missing[start : start + 256]
            phase = np.arange(8192, dtype=float) / 8192
            waveforms = []
            for key in batch_keys:
                waveform = FourierWaveform(np.asarray(key), args.harmonics)
                waveforms.append(
                    waveform.velocity(
                        phase, args.frequency, effective_peak_displacement(waveform)
                    )
                )
            final_positions, _steps = integrate_batch(
                np.asarray(waveforms),
                args.frequency,
                args.duration,
                np.asarray([particle.diameter for _, particle in size_particles]),
                np.asarray([particle.density for _, particle in size_particles]),
                air.density,
                air.dynamic_viscosity,
                args.cuda_steps_per_period,
            )
            for key, final in zip(batch_keys, final_positions):
                cuda_displacement_cache[key] = tuple(map(float, final))
        values = np.asarray(
            [cuda_displacement_cache[key] for key in keys], dtype=float
        ).reshape(len(keys), len(size_particles))
        if len(cuda_displacement_cache) > 4096:
            # Keep the newest working set without allowing long searches to grow unbounded.
            for stale_key in list(cuda_displacement_cache)[: len(cuda_displacement_cache) - 2048]:
                del cuda_displacement_cache[stale_key]
        return values

    def unpack_candidates(coefficients: np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(coefficients, dtype=float)
        if values.ndim == 1:
            return values[None, :], True
        return values.T, False

    def displacement_metrics(displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        displacements = np.asarray(displacements, dtype=float)
        spreads_mm = np.std(displacements, axis=1) * 1e3
        means_mm = np.mean(args.direction * displacements, axis=1) * 1e3
        scores_mm = spreads_mm - args.transport_reward_weight * (
            means_mm - args.min_mean_displacement_mm
        )
        return spreads_mm, means_mm, scores_mm

    def objective(coefficients: np.ndarray) -> float | np.ndarray:
        if use_cuda:
            candidates, scalar = unpack_candidates(coefficients)
            displacements = cuda_displacement_values(candidates)
            _, _, scores = displacement_metrics(displacements)
            return float(scores[0]) if scalar else scores
        displacements = np.asarray([displacement_values(*map(float, coefficients))])
        _, _, scores = displacement_metrics(displacements)
        return float(scores[0])

    def transport_constraint(coefficients: np.ndarray) -> float | np.ndarray:
        if use_cuda:
            candidates, scalar = unpack_candidates(coefficients)
            displacements = cuda_displacement_values(candidates)
            _, means, _ = displacement_metrics(displacements)
            margins = means - args.min_mean_displacement_mm
            return float(margins[0]) if scalar else margins[None, :]
        displacements = np.asarray([displacement_values(*map(float, coefficients))])
        _, means, _ = displacement_metrics(displacements)
        return float(means[0] - args.min_mean_displacement_mm)

    def report_progress(coefficients: np.ndarray, convergence: float) -> bool:
        nonlocal iteration
        iteration += 1
        if use_cuda:
            displacements = cuda_displacement_values(np.asarray(coefficients)[None, :])
        else:
            displacements = np.asarray([displacement_values(*map(float, coefficients))])
        spreads, means, scores = displacement_metrics(displacements)
        status = "met" if means[0] >= args.min_mean_displacement_mm else "below"
        print(
            f"Iteration {iteration}/{args.maxiter}: selected score {scores[0]:.3f} mm; "
            f"spread {spreads[0]:.3f} mm; mean transport {means[0]:.3f} mm "
            f"({args.min_mean_displacement_mm:g} mm minimum {status}); "
            f"convergence {convergence:.3g}",
            flush=True,
        )
        return False

    print(
        f"Optimizing {len(size_particles)} sizes over {args.duration:g} s "
        f"({args.frequency:g} Hz, maximum air displacement "
        f"{args.peak_air_displacement_mm:g} mm, maximum air speed "
        f"{args.max_air_speed:g} m/s, "
        f"device {'CUDA' if use_cuda else 'CPU'}). Randomly exploring waveform coefficients..."
    )
    initial_population_size = max(5, args.popsize * coefficient_count)
    exploration_count = max(args.exploration_samples, initial_population_size)
    exploration_rng = np.random.default_rng(args.seed)
    exploration_candidates = exploration_rng.uniform(
        -1.0, 1.0, size=(exploration_count, coefficient_count)
    )
    # Include smooth Fourier shapes as well as unrestricted high-harmonic
    # shapes. The latter often have to be scaled down by the speed cap.
    harmonic_numbers = np.repeat(np.arange(2, args.harmonics + 1), 2)
    decay_orders = exploration_rng.choice(
        (0.0, 1.0, 2.0, 3.0), size=exploration_count
    )
    exploration_candidates *= harmonic_numbers[None, :] ** (-decay_orders[:, None])
    exploration_candidates[0] = 0.0
    for index, (sine, cosine) in enumerate(
        ((0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)), start=1
    ):
        exploration_candidates[index] = 0.0
        exploration_candidates[index, :2] = (sine, cosine)
    exploration_spreads = np.empty(exploration_count)
    exploration_means = np.empty(exploration_count)
    exploration_scores = np.empty(exploration_count)
    report_every = max(1, exploration_count // 20)
    next_report = report_every
    exploration_start_time = time.perf_counter()
    last_report_time = exploration_start_time
    batch_size = 128 if use_cuda else 1
    for start in range(0, exploration_count, batch_size):
        stop = min(start + batch_size, exploration_count)
        if use_cuda:
            displacements = cuda_displacement_values(exploration_candidates[start:stop])
        else:
            displacements = np.asarray([
                displacement_values(*map(float, exploration_candidates[start]))
            ])
        spreads, means, scores = displacement_metrics(displacements)
        exploration_spreads[start:stop] = spreads
        exploration_means[start:stop] = means
        exploration_scores[start:stop] = scores
        now = time.perf_counter()
        if stop >= next_report or now - last_report_time >= 15.0 or stop == exploration_count:
            completed_feasible = np.flatnonzero(
                exploration_means[:stop] >= args.min_mean_displacement_mm
            )
            if completed_feasible.size:
                best = int(completed_feasible[np.argmin(exploration_scores[completed_feasible])])
                best_text = (
                    f"best feasible score {exploration_scores[best]:.3f} mm "
                    f"(spread {exploration_spreads[best]:.3f} mm, "
                    f"mean {exploration_means[best]:.3f} mm)"
                )
            else:
                best_text = f"best mean transport {np.max(exploration_means[:stop]):.3f} mm"
            print(
                f"Exploration {stop}/{exploration_count} "
                f"({100 * stop / exploration_count:.0f}%, "
                f"{now - exploration_start_time:.1f} s): "
                f"{completed_feasible.size} feasible; {best_text}",
                flush=True,
            )
            while next_report <= stop:
                next_report += report_every
            last_report_time = now
    feasible_indices = np.flatnonzero(
        exploration_means >= args.min_mean_displacement_mm
    )
    if feasible_indices.size:
        elite_count = min(feasible_indices.size, max(1, initial_population_size // 4))
        elite_indices = feasible_indices[
            np.argsort(exploration_scores[feasible_indices])[:elite_count]
        ]
    else:
        elite_count = min(exploration_count, max(1, initial_population_size // 4))
        elite_indices = np.argsort(exploration_means)[-elite_count:][::-1]

    selected_indices = list(map(int, elite_indices))
    min_distances = np.full(exploration_count, np.inf)
    for index in selected_indices:
        distance = np.linalg.norm(exploration_candidates - exploration_candidates[index], axis=1)
        min_distances = np.minimum(min_distances, distance)
    while len(selected_indices) < initial_population_size:
        min_distances[selected_indices] = -np.inf
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        distance = np.linalg.norm(
            exploration_candidates - exploration_candidates[next_index], axis=1
        )
        min_distances = np.minimum(min_distances, distance)
    exploration_population = exploration_candidates[selected_indices]

    if feasible_indices.size:
        best_index = int(feasible_indices[np.argmin(exploration_scores[feasible_indices])])
        feasible_text = (
            f"best feasible score {exploration_scores[best_index]:.3f} mm "
            f"(spread {exploration_spreads[best_index]:.3f} mm, "
            f"mean transport {exploration_means[best_index]:.3f} mm)"
        )
    else:
        feasible_text = "no random sample met the mean-transport minimum"
    print(
        f"Explored {exploration_count} random waveforms; {feasible_indices.size} met "
        f"the {args.min_mean_displacement_mm:g} mm minimum. {feasible_text}. "
        f"Seeding optimization with {initial_population_size} diverse candidates."
    )
    print("Starting spread-and-transport optimization...")
    iteration = 0
    optimization = differential_evolution(
        objective,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=False,
        updating="deferred" if use_cuda else "immediate",
        workers=1,
        vectorized=use_cuda,
        callback=report_progress,
        constraints=(NonlinearConstraint(transport_constraint, 0.0, np.inf),),
        init=exploration_population,
    )

    results, waveform = simulate_candidate(optimization.x)
    effective_peak_m = effective_peak_displacement(waveform)
    displacements = np.asarray([result.position[-1] - result.position[0] for result in results])
    directed_mm = args.direction * displacements * 1e3
    spread_mm = float(np.std(directed_mm))
    mean_directed_mm = float(np.mean(directed_mm))
    score_mm = spread_mm - args.transport_reward_weight * (
        mean_directed_mm - args.min_mean_displacement_mm
    )
    transport_met = mean_directed_mm >= args.min_mean_displacement_mm
    print(f"Differential evolution converged: {optimization.success} ({optimization.message})")
    if not transport_met:
        print(
            "Warning: selected waveform did not meet the mean transport minimum "
            f"({mean_directed_mm:.3f} mm versus "
            f"{args.min_mean_displacement_mm:.3f} mm required). "
            "Saving the best available result."
        )
    print(f"Maximum permitted peak air displacement: {args.peak_air_displacement_mm:.6f} mm")
    print(f"Selected peak air displacement: {effective_peak_m * 1e3:.6f} mm")
    print(f"Maximum permitted air speed: {args.max_air_speed:.6f} m/s")
    print(f"Upper bound on peak air velocity: {effective_peak_m * args.frequency * waveform._phase_derivative_bound:.6f} m/s")
    print(f"Minimum mean directed transport: {args.min_mean_displacement_mm:.6f} mm")
    print(f"Transport reward weight: {args.transport_reward_weight:g} mm per mm")
    if use_cuda:
        cuda_final = cuda_displacement_values(np.asarray(optimization.x)[None, :])
        cuda_spreads, cuda_means, cuda_scores = displacement_metrics(cuda_final)
        print(
            "CUDA estimate for selected waveform: "
            f"mean {cuda_means[0]:.6f} mm, "
            f"spread {cuda_spreads[0]:.6f} mm, "
            f"score {cuda_scores[0]:.6f} mm"
        )
    print(f"Mean directed transport: {mean_directed_mm:.6f} mm")
    print(f"Net-transport standard deviation: {spread_mm:.6f} mm")
    print(f"Final score (lower is better): {score_mm:.6f} mm")
    print(f"Net-transport range: {np.ptp(directed_mm):.6f} mm")
    print(f"{'Diameter [um]':>14} {'Net displacement [mm]':>23} {'Directed [mm]':>16}")
    for (diameter_um, _particle), result, directed in zip(
        size_particles, results, directed_mm
    ):
        displacement_mm = (result.position[-1] - result.position[0]) * 1e3
        print(f"{diameter_um:14g} {displacement_mm:23.6f} {directed:16.6f}")

    write_waveform_csv(args.output, waveform, args.frequency, effective_peak_m)
    print(f"Saved waveform: {args.output}")

    plot_status = []
    if not optimization.success:
        plot_status.append("optimizer did not converge")
    if not transport_met:
        plot_status.append("transport minimum not met")
    title = "Optimized Fourier air displacement waveform"
    if plot_status:
        title += f" ({'; '.join(plot_status)})"
    phase = np.arange(4096, dtype=float) / 4096
    archive = build_archive(
        kind="fourier",
        args=args,
        optimization=optimization,
        size_particles=size_particles,
        air=air,
        results=results,
        parameters=optimization.x,
        selected_peak_m=effective_peak_m,
        phase=phase,
        position_m=effective_peak_m * waveform(phase),
        velocity_m_per_s=waveform.velocity(phase, args.frequency, effective_peak_m),
        plot_title=title,
        extra={
            "evaluation_device": "cuda" if use_cuda else "cpu",
            "cuda_steps_per_period": args.cuda_steps_per_period,
            "exploration_samples_evaluated": exploration_count,
            "exploration_feasible_count": int(feasible_indices.size),
            "parameter_bounds": np.asarray(bounds, dtype=float),
            "fourier_coefficients": np.asarray(optimization.x, dtype=float),
            "fourier_harmonics": args.harmonics,
            "fourier_position_normalization": waveform._normalization,
            "fourier_phase_derivative_bound": waveform._phase_derivative_bound,
            **({"cuda_estimate_displacements_m": cuda_final.ravel()} if use_cuda else {}),
        },
    )
    mat_path = save_archive(args.results_dir, archive)
    print(f"Saved MATLAB result: {mat_path}")

    if args.no_plot:
        return
    import matplotlib.pyplot as plt

    plot_archive(archive)
    plt.show()


if __name__ == "__main__":
    main()
