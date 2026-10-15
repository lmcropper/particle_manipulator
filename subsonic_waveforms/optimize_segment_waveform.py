"""Optimize piecewise-linear air displacement waveforms for particle transport.

Each waveform is a periodic sequence of linear segments. Segment durations and
slopes are optimized directly; closure and zero mean are enforced when a
candidate displacement waveform is constructed. Air velocity is the time
derivative of that waveform. The objective rewards low spread among net
particle transports and, to a lesser extent, longer mean directed transport.
A minimum mean directed transport is also enforced.
The segmented displacement is smoothed with a periodic Gaussian low-pass
filter before its velocity is sent to the particle solver.

Example:
    python optimize_segment_waveform.py --segments 4 --min-mean-displacement-mm 5
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
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
        raise argparse.ArgumentTypeError("particle sizes must be between 20 and 100 um")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("particle sizes must not contain duplicates")
    return sizes


@dataclass(frozen=True)
class SegmentWaveform:
    """A continuous, periodic, zero-mean piecewise-linear displacement shape."""

    durations: np.ndarray
    vertices: np.ndarray
    boundaries: np.ndarray = field(init=False, repr=False)
    phase_slopes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        boundaries = np.concatenate(([0.0], np.cumsum(self.durations)))
        boundaries[-1] = 1.0
        object.__setattr__(
            self,
            "boundaries",
            boundaries,
        )
        object.__setattr__(self, "phase_slopes", np.diff(self.vertices) / self.durations)

    @classmethod
    def from_parameters(cls, parameters: np.ndarray, segments: int) -> "SegmentWaveform":
        parameters = np.asarray(parameters, dtype=float)
        expected = 2 * (segments - 1)
        if parameters.shape != (expected,):
            raise ValueError(f"expected {expected} parameters for {segments} segments")

        # The final duration logit is fixed at zero to remove a redundant
        # common offset. Softmax gives positive durations summing to one.
        duration_logits = np.append(parameters[: segments - 1], 0.0)
        duration_weights = np.exp(duration_logits - np.max(duration_logits))
        durations = duration_weights / np.sum(duration_weights)

        # The final slope returns the air displacement to its starting value.
        free_slopes = parameters[segments - 1 :]
        final_slope = -float(np.dot(free_slopes, durations[:-1])) / durations[-1]
        slopes = np.append(free_slopes, final_slope)
        vertices = np.concatenate(([0.0], np.cumsum(slopes * durations)))

        # Remove the arbitrary position offset and normalize peak displacement.
        mean = float(np.sum(0.5 * (vertices[:-1] + vertices[1:]) * durations))
        vertices -= mean
        peak = float(np.max(np.abs(vertices)))
        if peak > 1e-14:
            vertices /= peak
        else:
            vertices[:] = 0.0
        vertices[-1] = vertices[0]
        return cls(durations=durations, vertices=vertices)

    def __call__(self, phase: float | np.ndarray) -> float | np.ndarray:
        """Return dimensionless air displacement at a phase within the period."""
        phase_array = np.asarray(phase, dtype=float)
        wrapped = np.mod(phase_array, 1.0)
        segment = np.searchsorted(self.boundaries[1:], wrapped, side="right")
        fraction = (wrapped - self.boundaries[segment]) / self.durations[segment]
        values = self.vertices[segment] + fraction * (
            self.vertices[segment + 1] - self.vertices[segment]
        )
        if phase_array.ndim == 0:
            return float(values)
        return values

    def velocity(
        self,
        phase: float | np.ndarray,
        frequency_hz: float,
        peak_displacement_m: float,
    ) -> float | np.ndarray:
        """Return the segment derivative of displacement in m/s."""
        phase_array = np.asarray(phase, dtype=float)
        wrapped = np.mod(phase_array, 1.0)
        segment = np.searchsorted(self.boundaries[1:], wrapped, side="right")
        values = peak_displacement_m * frequency_hz * self.phase_slopes[segment]
        if phase_array.ndim == 0:
            return float(values)
        return values

    def limited_peak_displacement(
        self,
        requested_peak_m: float,
        frequency_hz: float,
        max_air_speed: float,
        max_velocity_jump: float,
    ) -> float:
        """Scale displacement so segment speeds and all border jumps stay capped."""
        speeds_per_m = frequency_hz * self.phase_slopes
        peak_speed_per_m = float(np.max(np.abs(speeds_per_m)))
        peak_jump_per_m = float(np.max(np.abs(speeds_per_m - np.roll(speeds_per_m, 1))))
        return min(
            requested_peak_m,
            max_air_speed / peak_speed_per_m if peak_speed_per_m > 0.0 else np.inf,
            max_velocity_jump / peak_jump_per_m if peak_jump_per_m > 0.0 else np.inf,
        )


class FilteredSegmentWaveform:
    """Periodic low-pass displacement with velocity from its smooth derivative."""

    def __init__(
        self, source: SegmentWaveform, frequency_hz: float, cutoff_hz: float
    ) -> None:
        self.source = source
        self.frequency_hz = frequency_hz
        self.cutoff_hz = cutoff_hz
        self.position_spline = None
        self.phase_velocity_spline = None
        if cutoff_hz > 0.0:
            sample_count = 4096
            phase = np.arange(sample_count, dtype=float) / sample_count
            spectrum = np.fft.rfft(source(phase))
            spectrum[0] = 0.0  # Keep the air displacement centered on zero.
            harmonic_frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_count)
            physical_frequencies = harmonic_frequencies * frequency_hz
            response = np.exp(
                -0.5 * np.log(2.0) * (physical_frequencies / cutoff_hz) ** 2
            )
            positions = np.fft.irfft(spectrum * response, n=sample_count)
            periodic_phase = np.append(phase, 1.0)
            periodic_positions = np.append(positions, positions[0])
            self.position_spline = CubicSpline(
                periodic_phase, periodic_positions, bc_type="periodic"
            )
            self.phase_velocity_spline = self.position_spline.derivative()
            # Sample more densely than the spline knots to account for small
            # interpolation extrema when enforcing displacement and speed caps.
            dense_phase = np.arange(16384, dtype=float) / 16384
            self.position_peak = float(np.max(np.abs(self.position_spline(dense_phase))))
            self.phase_velocity_peak = float(
                np.max(np.abs(self.phase_velocity_spline(dense_phase)))
            )
        else:
            self.position_peak = 1.0
            self.phase_velocity_peak = float(np.max(np.abs(source.phase_slopes)))

    def __call__(self, phase: float | np.ndarray) -> float | np.ndarray:
        if self.position_spline is None:
            return self.source(phase)
        values = self.position_spline(np.mod(phase, 1.0))
        return float(values) if np.ndim(phase) == 0 else values

    def velocity(self, phase: float | np.ndarray, peak_displacement_m: float) -> float | np.ndarray:
        if self.phase_velocity_spline is None:
            return self.source.velocity(phase, self.frequency_hz, peak_displacement_m)
        values = (
            peak_displacement_m * self.frequency_hz
            * self.phase_velocity_spline(np.mod(phase, 1.0))
        )
        return float(values) if np.ndim(phase) == 0 else values

    def limited_peak_displacement(
        self, requested_peak_m: float, max_air_speed: float, max_velocity_jump: float
    ) -> float:
        raw_limit = self.source.limited_peak_displacement(
            requested_peak_m, self.frequency_hz, max_air_speed, max_velocity_jump
        )
        if self.position_spline is None:
            return raw_limit
        margin = 1.001
        return min(
            raw_limit,
            requested_peak_m / (self.position_peak * margin)
            if self.position_peak > 0.0 else np.inf,
            max_air_speed / (self.frequency_hz * self.phase_velocity_peak * margin)
            if self.phase_velocity_peak > 0.0 else np.inf,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=800.0, help="Drive frequency [Hz].")

    mm_per_sec_min = 50.0
    duration = 0.02
    parser.add_argument("--duration", type=float, default=duration, help="Simulation duration [s].")
    parser.add_argument(
        "--peak-air-displacement-mm",
        type=float,
        default=1.0,
        help="Maximum peak absolute air displacement about its mean [mm].",
    )
    parser.add_argument(
        "--max-air-speed",
        type=float,
        default=30.0,
        help="Maximum absolute air velocity before and after filtering [m/s].",
    )
    parser.add_argument(
        "--max-velocity-jump",
        type=float,
        default=60.0,
        help="Maximum unfiltered air-velocity change at any segment border, including the period boundary [m/s].",
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=4000.0,
        help="Gaussian zero-phase displacement low-pass cutoff (-3 dB) [Hz]; 0 disables filtering.",
    )
    parser.add_argument("--segments", type=int, default=16, help="Number of linear waveform segments.")
    parser.add_argument(
        "--sizes-um",
        type=parse_sizes,
        default=parse_sizes("30,40,50,60,70,80,90,100"),
        help="Comma-separated particle diameters [um] (20 to 100).",
    )
    parser.add_argument("--particle-density", type=float, default=157.37, help="Glass density [kg/m^3].")
    parser.add_argument(
        "--min-mean-displacement-mm",
        type=float,
        default=mm_per_sec_min * duration,
        help="Minimum mean directed displacement across particle sizes [mm].",
    )
    parser.add_argument(
        "--transport-reward-weight",
        type=float,
        default=0.1,
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
    parser.add_argument("--maxiter", type=int, default=2000, help="Differential-evolution iterations.")
    parser.add_argument(
        "--exploration-samples",
        type=int,
        default=100000,
        help="Number of random segment waveforms to evaluate before optimization.",
    )
    parser.add_argument("--popsize", type=int, default=20, help="Population multiplier for the optimizer.")
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
        default=Path("optimized_segment_waveform.csv"),
        help="CSV output path for filtered and unfiltered air velocity and displacement.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("optimization_results"),
        help="Directory for timestamped MATLAB result files.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not open result plots.")
    return parser


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
    if args.max_velocity_jump <= 0.0 or not np.isfinite(args.max_velocity_jump):
        parser.error("--max-velocity-jump must be positive and finite")
    if args.lowpass_hz < 0.0 or not np.isfinite(args.lowpass_hz):
        parser.error("--lowpass-hz must be nonnegative and finite")
    if args.particle_density <= 0.0 or not np.isfinite(args.particle_density):
        parser.error("--particle-density must be positive and finite")
    if args.segments < 2:
        parser.error("--segments must be at least 2")
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

    integrate_batch = None
    use_cuda = args.device == "cuda"
    if args.device in {"auto", "cuda"}:
        try:
            from cuda_backend import cupy_module, integrate_batch as cuda_integrate_batch

            cupy_module()
            integrate_batch = cuda_integrate_batch
            use_cuda = True
        except RuntimeError as exc:
            if args.device == "cuda":
                parser.error(str(exc))
            use_cuda = False
            print(f"CUDA unavailable; using CPU evaluations ({exc})")

    segments = args.segments
    parameter_count = 2 * (segments - 1)
    # Bounded duration logits control relative segment lengths. Free slopes
    # are scale-free because every resulting waveform is peak-normalized.
    bounds = [(-3.0, 3.0)] * (segments - 1) + [(-1.0, 1.0)] * (segments - 1)
    lower_bounds = np.asarray([bound[0] for bound in bounds])
    upper_bounds = np.asarray([bound[1] for bound in bounds])
    size_particles = [
        (diameter_um, Particle(diameter=diameter_um * 1e-6, density=args.particle_density))
        for diameter_um in args.sizes_um
    ]
    air = AirProperties()
    peak_displacement_m = args.peak_air_displacement_mm * 1e-3
    cpu_cache: dict[tuple[float, ...], tuple[float, ...]] = {}
    cuda_cache: dict[tuple[float, ...], tuple[float, ...]] = {}
    iteration = 0

    def effective_peak_displacement(waveform: FilteredSegmentWaveform) -> float:
        return waveform.limited_peak_displacement(
            peak_displacement_m,
            args.max_air_speed,
            args.max_velocity_jump,
        )

    def make_waveform(parameters: np.ndarray) -> FilteredSegmentWaveform:
        source = SegmentWaveform.from_parameters(parameters, segments)
        return FilteredSegmentWaveform(source, args.frequency, args.lowpass_hz)

    def simulate_candidate(parameters: np.ndarray) -> tuple[list[SimulationResult], FilteredSegmentWaveform]:
        waveform = make_waveform(parameters)
        effective_peak_m = effective_peak_displacement(waveform)
        piston = PistonFlow(
            frequency_hz=args.frequency,
            # PistonFlow multiplies its waveform by peak_speed; this waveform
            # already returns physical air velocity, so its scale factor is 1.
            peak_speed=1.0,
            waveform=lambda phase: waveform.velocity(phase, effective_peak_m),
        )
        results = [
            simulate(piston, particle=particle, duration=args.duration)
            for _, particle in size_particles
        ]
        return results, waveform

    def cpu_displacements(parameters: np.ndarray) -> tuple[float, ...]:
        key = tuple(map(float, parameters))
        if key not in cpu_cache:
            results, _ = simulate_candidate(np.asarray(parameters, dtype=float))
            cpu_cache[key] = tuple(
                float(result.position[-1] - result.position[0]) for result in results
            )
            if len(cpu_cache) > 4096:
                cpu_cache.pop(next(iter(cpu_cache)))
        return cpu_cache[key]

    def cuda_displacements(candidates: np.ndarray) -> np.ndarray:
        assert integrate_batch is not None
        candidates = np.asarray(candidates, dtype=float)
        keys = [tuple(map(float, row)) for row in candidates]
        missing = list(dict.fromkeys(key for key in keys if key not in cuda_cache))
        for start in range(0, len(missing), 256):
            batch_keys = missing[start : start + 256]
            phase = np.arange(8192, dtype=float) / 8192
            waves = []
            for key in batch_keys:
                waveform = make_waveform(np.asarray(key))
                waves.append(
                    waveform.velocity(phase, effective_peak_displacement(waveform))
                )
            waves = np.asarray(waves)
            final_positions, _steps = integrate_batch(
                waves,
                args.frequency,
                args.duration,
                np.asarray([particle.diameter for _, particle in size_particles]),
                np.asarray([particle.density for _, particle in size_particles]),
                air.density,
                air.dynamic_viscosity,
                args.cuda_steps_per_period,
            )
            for key, final in zip(batch_keys, final_positions):
                cuda_cache[key] = tuple(map(float, final))
        if len(cuda_cache) > 8192:
            for old_key in list(cuda_cache)[: len(cuda_cache) - 4096]:
                del cuda_cache[old_key]
        return np.asarray([cuda_cache[key] for key in keys], dtype=float).reshape(
            len(keys), len(size_particles)
        )

    def candidates_from_optimizer(values: np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(values, dtype=float)
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

    def objective(values: np.ndarray) -> float | np.ndarray:
        if use_cuda:
            candidates, scalar = candidates_from_optimizer(values)
            displacements = cuda_displacements(candidates)
            _, _, scores = displacement_metrics(displacements)
            return float(scores[0]) if scalar else scores
        displacements = np.asarray([cpu_displacements(np.asarray(values, dtype=float))])
        _, _, scores = displacement_metrics(displacements)
        return float(scores[0])

    def transport_constraint(values: np.ndarray) -> float | np.ndarray:
        if use_cuda:
            candidates, scalar = candidates_from_optimizer(values)
            displacements = cuda_displacements(candidates)
            _, means, _ = displacement_metrics(displacements)
            margins = means - args.min_mean_displacement_mm
            return float(margins[0]) if scalar else margins[None, :]
        displacements = np.asarray(cpu_displacements(np.asarray(values, dtype=float)))
        mean = float(np.mean(args.direction * displacements * 1e3))
        return float(mean - args.min_mean_displacement_mm)

    def report_progress(candidate: np.ndarray, convergence: float) -> bool:
        nonlocal iteration
        iteration += 1
        if use_cuda:
            displacements = cuda_displacements(np.asarray(candidate)[None, :])[0]
        else:
            displacements = np.asarray(cpu_displacements(candidate))
        spreads, means, scores = displacement_metrics(displacements[None, :])
        mean_mm, spread_mm, score_mm = float(means[0]), float(spreads[0]), float(scores[0])
        status = "met" if mean_mm >= args.min_mean_displacement_mm else "below"
        print(
            f"Iteration {iteration}/{args.maxiter}: selected score {score_mm:.3f} mm; "
            f"spread {spread_mm:.3f} mm; "
            f"selected mean transport {mean_mm:.3f} mm "
            f"({args.min_mean_displacement_mm:g} mm minimum {status}); "
            f"convergence {convergence:.3g}",
            flush=True,
        )
        return False

    population_size = max(5, args.popsize * parameter_count)
    exploration_count = max(args.exploration_samples, population_size)
    print(
        f"Exploring {exploration_count} random {segments}-segment waveforms "
        f"for {len(size_particles)} particle sizes on {'CUDA' if use_cuda else 'CPU'} "
        f"(low-pass cutoff {args.lowpass_hz:g} Hz)...",
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    exploration = rng.uniform(
        lower_bounds, upper_bounds, size=(exploration_count, parameter_count)
    )
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
            displacements = cuda_displacements(exploration[start:stop])
        else:
            displacements = np.asarray([cpu_displacements(exploration[start])])
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
        elite_count = min(feasible_indices.size, max(1, population_size // 4))
        elite_indices = feasible_indices[
            np.argsort(exploration_scores[feasible_indices])[:elite_count]
        ]
    else:
        elite_indices = np.asarray([int(np.argmax(exploration_means))])

    selected = list(map(int, elite_indices))
    normalized = (exploration - lower_bounds) / (upper_bounds - lower_bounds)
    min_distances = np.full(exploration_count, np.inf)
    for index in selected:
        min_distances = np.minimum(
            min_distances,
            np.linalg.norm(normalized - normalized[index], axis=1),
        )
    while len(selected) < population_size:
        min_distances[selected] = -np.inf
        index = int(np.argmax(min_distances))
        selected.append(index)
        min_distances = np.minimum(
            min_distances,
            np.linalg.norm(normalized - normalized[index], axis=1),
        )
    initial_population = exploration[selected]
    if feasible_indices.size:
        best_index = int(feasible_indices[np.argmin(exploration_scores[feasible_indices])])
        feasible_score_text = (
            f"best feasible score {exploration_scores[best_index]:.3f} mm "
            f"(spread {exploration_spreads[best_index]:.3f} mm, "
            f"mean transport {exploration_means[best_index]:.3f} mm)"
        )
    else:
        feasible_score_text = "no random sample met the mean-transport minimum"
    print(
        f"Explored {exploration_count} waveforms; {feasible_indices.size} met the "
        f"{args.min_mean_displacement_mm:g} mm minimum. {feasible_score_text}. "
        f"Starting differential evolution with {population_size} diverse candidates."
    )

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
        init=initial_population,
    )

    parameters = optimization.x
    results, waveform = simulate_candidate(parameters)
    effective_peak_m = effective_peak_displacement(waveform)
    raw_waveform = waveform.source
    displacements = np.asarray([result.position[-1] - result.position[0] for result in results])
    directed_mm = args.direction * displacements * 1e3
    mean_directed_mm = float(np.mean(directed_mm))
    spread_mm = float(np.std(directed_mm))
    score_mm = spread_mm - args.transport_reward_weight * (
        mean_directed_mm - args.min_mean_displacement_mm
    )
    print(f"Optimizer converged: {optimization.success} ({optimization.message})")
    transport_met = mean_directed_mm >= args.min_mean_displacement_mm
    if not transport_met:
        print(
            f"Warning: selected waveform did not meet the mean transport minimum "
            f"({mean_directed_mm:.3f} mm versus "
            f"{args.min_mean_displacement_mm:.3f} mm required). "
            "Saving the best available result."
        )
    print(f"Segments: {segments}")
    if args.lowpass_hz:
        print(f"Periodic Gaussian low-pass cutoff: {args.lowpass_hz:g} Hz")
    else:
        print("Periodic low-pass filter: disabled")
    print(f"Maximum permitted peak air displacement: {args.peak_air_displacement_mm:.6f} mm")
    print(f"Selected filtered peak air displacement: {effective_peak_m * waveform.position_peak * 1e3:.6f} mm")
    print(f"Maximum permitted air speed: {args.max_air_speed:.6f} m/s")
    print(f"Maximum permitted velocity jump: {args.max_velocity_jump:.6f} m/s")
    print(f"Minimum mean directed transport: {args.min_mean_displacement_mm:.6f} mm")
    print(f"Transport reward weight: {args.transport_reward_weight:g} mm per mm")
    if use_cuda:
        cuda_final = cuda_displacements(np.asarray(parameters)[None, :])[0]
        cuda_spreads, cuda_means, cuda_scores = displacement_metrics(cuda_final[None, :])
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

    segment_velocities = raw_waveform.phase_slopes * effective_peak_m * args.frequency
    print(f"Filtered peak absolute air velocity: {effective_peak_m * args.frequency * waveform.phase_velocity_peak:.6f} m/s")
    print(f"Unfiltered peak absolute air velocity: {np.max(np.abs(segment_velocities)):.6f} m/s")
    print(
        "Largest unfiltered segment-border velocity jump: "
        f"{np.max(np.abs(segment_velocities - np.roll(segment_velocities, 1))):.6f} m/s"
    )
    print("Unfiltered segment parameters:")
    print(f"{'Segment':>8} {'Duration [ms]':>15} {'Start [mm]':>14} {'End [mm]':>12} {'Air speed [m/s]':>16}")
    for index, (fraction, velocity) in enumerate(zip(raw_waveform.durations, segment_velocities), start=1):
        duration_ms = fraction / args.frequency * 1e3
        start_position_mm = raw_waveform.vertices[index - 1] * effective_peak_m * 1e3
        end_position_mm = raw_waveform.vertices[index] * effective_peak_m * 1e3
        print(
            f"{index:8d} {duration_ms:15.6f} {start_position_mm:14.6f} "
            f"{end_position_mm:12.6f} {velocity:16.6g}"
        )

    print(f"{'Diameter [um]':>14} {'Net displacement [mm]':>23} {'Directed [mm]':>16}")
    for (diameter_um, _particle), result, directed in zip(size_particles, results, directed_mm):
        displacement_mm = (result.position[-1] - result.position[0]) * 1e3
        print(f"{diameter_um:14g} {displacement_mm:23.6f} {directed:16.6f}")

    phase = np.linspace(0.0, 1.0, 1001)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        # Keep velocity in column 2 for PistonFlow.from_csv compatibility.
        writer.writerow((
            "phase", "velocity_m_per_s", "air_displacement_m",
            "unfiltered_velocity_m_per_s", "unfiltered_air_displacement_m",
        ))
        writer.writerows(
            (
                float(point),
                float(waveform.velocity(float(point), effective_peak_m)),
                float(effective_peak_m * waveform(float(point))),
                float(raw_waveform.velocity(float(point), args.frequency, effective_peak_m)),
                float(effective_peak_m * raw_waveform(float(point))),
            )
            for point in phase
        )
    print(f"Saved waveform: {args.output}")

    plot_status = []
    if not optimization.success:
        plot_status.append("optimizer did not converge")
    if not transport_met:
        plot_status.append("transport minimum not met")
    title = (
        f"Optimized segment air displacement waveform ({args.lowpass_hz:g} Hz low-pass)"
        if args.lowpass_hz else "Optimized unfiltered segment air displacement waveform"
    )
    if plot_status:
        title += f" ({'; '.join(plot_status)})"
    phase = np.arange(4096, dtype=float) / 4096
    archive = build_archive(
        kind="segment",
        args=args,
        optimization=optimization,
        size_particles=size_particles,
        air=air,
        results=results,
        parameters=parameters,
        selected_peak_m=effective_peak_m,
        phase=phase,
        position_m=effective_peak_m * waveform(phase),
        velocity_m_per_s=waveform.velocity(phase, effective_peak_m),
        plot_title=title,
        extra={
            "evaluation_device": "cuda" if use_cuda else "cpu",
            "cuda_steps_per_period": args.cuda_steps_per_period,
            "exploration_samples_evaluated": exploration_count,
            "exploration_feasible_count": int(feasible_indices.size),
            "parameter_bounds": np.asarray(bounds, dtype=float),
            "segment_durations": raw_waveform.durations,
            "segment_phase_boundaries": raw_waveform.boundaries,
            "segment_position_vertices": raw_waveform.vertices,
            "segment_phase_slopes": raw_waveform.phase_slopes,
            "segment_velocities_m_per_s": segment_velocities,
            "lowpass_cutoff_hz": args.lowpass_hz,
            "raw_waveform_position_m": effective_peak_m * raw_waveform(phase),
            "raw_waveform_velocity_m_per_s": raw_waveform.velocity(
                phase, args.frequency, effective_peak_m
            ),
            **({"cuda_estimate_displacements_m": cuda_final} if use_cuda else {}),
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
