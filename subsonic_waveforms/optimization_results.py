"""Save optimization runs as MATLAB files and plot them without rerunning a solver."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from piston_particle_solver import SimulationResult


SCHEMA_VERSION = 1


def _metric_tag(value: float) -> str:
    sign = "m" if value < 0.0 else ""
    return sign + f"{abs(value):.3f}".replace(".", "p")


def _padded_trajectories(results: list[SimulationResult]) -> dict[str, np.ndarray]:
    counts = np.asarray([len(result.time) for result in results], dtype=np.int32)
    width = int(np.max(counts))
    data: dict[str, np.ndarray] = {"trajectory_sample_count": counts}
    for result_field in fields(SimulationResult):
        values = np.full((len(results), width), np.nan, dtype=float)
        for row, result in enumerate(results):
            samples = np.asarray(getattr(result, result_field.name), dtype=float)
            values[row, : len(samples)] = samples
        data[f"trajectory_{result_field.name}"] = values
    return data


def build_archive(
    *,
    kind: str,
    args: Any,
    optimization: Any,
    size_particles: list[tuple[float, Any]],
    air: Any,
    results: list[SimulationResult],
    parameters: np.ndarray,
    selected_peak_m: float,
    phase: np.ndarray,
    position_m: np.ndarray,
    velocity_m_per_s: np.ndarray,
    plot_title: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat MATLAB-friendly record of inputs, outputs, and trajectories."""
    if kind not in {"fourier", "segment"}:
        raise ValueError(f"unknown waveform kind: {kind}")
    displacements_m = np.asarray(
        [result.position[-1] - result.position[0] for result in results], dtype=float
    )
    directed_mm = float(args.direction) * displacements_m * 1e3
    mean_mm = float(np.mean(directed_mm))
    spread_mm = float(np.std(directed_mm))
    score_mm = spread_mm - float(args.transport_reward_weight) * (
        mean_mm - float(args.min_mean_displacement_mm)
    )
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    archive: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "waveform_kind": kind,
        "created_at": datetime.now().astimezone().isoformat(),
        "settings_json": json.dumps(settings, sort_keys=True),
        "plot_title": plot_title,
        "parameters": np.asarray(parameters, dtype=float),
        "optimizer_success": int(bool(optimization.success)),
        "optimizer_message": str(optimization.message),
        "optimizer_iterations": int(optimization.nit),
        "optimizer_evaluations": int(optimization.nfev),
        "optimizer_objective_mm": float(optimization.fun),
        "transport_minimum_met": int(mean_mm >= float(args.min_mean_displacement_mm)),
        "frequency_hz": float(args.frequency),
        "simulation_duration_s": float(args.duration),
        "direction": float(args.direction),
        "particle_sizes_um": np.asarray([diameter for diameter, _ in size_particles]),
        "particle_density_kg_m3": float(args.particle_density),
        "air_density_kg_m3": float(air.density),
        "air_viscosity_pa_s": float(air.dynamic_viscosity),
        "peak_displacement_limit_m": float(args.peak_air_displacement_mm) * 1e-3,
        "selected_displacement_scale_m": float(selected_peak_m),
        "max_air_speed_m_per_s": float(args.max_air_speed),
        "minimum_mean_transport_mm": float(args.min_mean_displacement_mm),
        "transport_reward_weight": float(args.transport_reward_weight),
        "net_displacements_m": displacements_m,
        "directed_displacements_mm": directed_mm,
        "mean_directed_transport_mm": mean_mm,
        "transport_spread_mm": spread_mm,
        "transport_range_mm": float(np.ptp(directed_mm)),
        "score_mm": score_mm,
        "waveform_phase": np.asarray(phase, dtype=float),
        "waveform_position_m": np.asarray(position_m, dtype=float),
        "waveform_velocity_m_per_s": np.asarray(velocity_m_per_s, dtype=float),
    }
    archive.update(_padded_trajectories(results))
    if extra:
        archive.update(extra)
    return archive


def save_archive(directory: Path, archive: dict[str, Any]) -> Path:
    """Save a compressed .mat with a sortable, metric-bearing filename."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kind = str(archive["waveform_kind"])
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    frequency = f"{float(archive['frequency_hz']):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    spread = _metric_tag(float(archive["transport_spread_mm"]))
    mean = _metric_tag(float(archive["mean_directed_transport_mm"]))
    status = "feasible" if int(archive["transport_minimum_met"]) else "below_min"
    path = directory / (
        f"{kind}_{stamp}_f{frequency}Hz_spread{spread}mm_mean{mean}mm_{status}.mat"
    )
    if path.exists():
        stem = path.stem
        suffix = 2
        while path.exists():
            path = directory / f"{stem}_{suffix}.mat"
            suffix += 1
    savemat(path, archive, do_compression=True)
    return path


def load_archive(path: Path) -> dict[str, Any]:
    """Load a saved result and reject unsupported archive versions."""
    archive = loadmat(Path(path), squeeze_me=True, chars_as_strings=True)
    version = int(_scalar(archive, "schema_version"))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported optimization archive version: {version}")
    return archive


def _scalar(archive: dict[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"expected one value for {name}")
    return value.reshape(-1)[0]


def plot_archive(archive: dict[str, Any]):
    """Recreate the four-panel optimization figure from saved arrays."""
    import matplotlib.pyplot as plt

    kind = str(_scalar(archive, "waveform_kind"))
    phase = np.atleast_1d(np.asarray(archive["waveform_phase"], dtype=float))
    position = np.atleast_1d(np.asarray(archive["waveform_position_m"], dtype=float))
    velocity = np.atleast_1d(np.asarray(archive["waveform_velocity_m_per_s"], dtype=float))
    sizes = np.atleast_1d(np.asarray(archive["particle_sizes_um"], dtype=float))
    displacements = np.atleast_1d(np.asarray(archive["net_displacements_m"], dtype=float))
    counts = np.atleast_1d(np.asarray(archive["trajectory_sample_count"], dtype=int))
    times = np.atleast_2d(np.asarray(archive["trajectory_time"], dtype=float))
    trajectories = np.atleast_2d(np.asarray(archive["trajectory_position"], dtype=float))

    figure, axes = plt.subplots(4, 1, figsize=(10, 12))
    if kind == "segment" and "raw_waveform_position_m" in archive:
        raw_position = np.atleast_1d(np.asarray(archive["raw_waveform_position_m"], dtype=float))
        raw_velocity = np.atleast_1d(np.asarray(archive["raw_waveform_velocity_m_per_s"], dtype=float))
        axes[0].plot(phase, raw_position * 1e3, color="0.65", linestyle="--", label="Segment input")
        axes[1].step(phase, raw_velocity, where="post", color="0.65", linestyle="--", label="Segment derivative")
        axes[0].plot(phase, position * 1e3, label="Filtered")
        axes[1].plot(phase, velocity, label="Filtered")
        axes[0].legend()
        axes[1].legend()
    else:
        axes[0].plot(phase, position * 1e3)
        axes[1].plot(phase, velocity)
    axes[0].set_ylabel("Air displacement [mm]")
    axes[0].set_xlabel("Waveform phase")
    axes[1].set_ylabel("Air velocity [m/s]")
    axes[1].set_xlabel("Waveform phase")

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(sizes)))
    for row, (size, color) in enumerate(zip(sizes, colors)):
        count = int(counts[row])
        elapsed = times[row, :count] - times[row, 0]
        displacement_mm = (trajectories[row, :count] - trajectories[row, 0]) * 1e3
        axes[2].plot(elapsed, displacement_mm, color=color, label=f"{size:g} um")
    axes[2].set_ylabel("Particle displacement [mm]")
    axes[2].legend(title="Diameter", ncol=3)

    axes[3].bar(sizes, displacements * 1e3, width=6.0)
    axes[3].axhline(
        float(np.mean(displacements) * 1e3),
        color="tab:red", linestyle="--", label="Mean displacement",
    )
    axes[3].set_xlabel("Particle diameter [um]")
    axes[3].set_ylabel("Net displacement [mm]")
    axes[3].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(str(_scalar(archive, "plot_title")))
    figure.tight_layout()
    return figure
