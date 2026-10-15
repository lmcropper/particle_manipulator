"""Optional CUDA integration helpers backed by CuPy.

CUDA is deliberately optional: importing the regular solver does not require
CuPy or a CUDA-capable machine. The CUDA path uses fixed-step RK4 and sampled
waveforms, while the CPU solver keeps its adaptive SciPy integration.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


_KERNEL = r"""
extern "C" __device__ double flow_at(const double* waves, int wave_stride,
                                     int samples, int candidate, double phase) {
    double wrapped = phase - floor(phase);
    double location = wrapped * samples;
    int left = (int)location;
    if (left >= samples) left = 0;
    int right = left + 1;
    if (right == samples) right = 0;
    double fraction = location - left;
    const double* wave = waves + candidate * wave_stride;
    return wave[left] + fraction * (wave[right] - wave[left]);
}

extern "C" __device__ double acceleration(double velocity, double air_velocity,
        double diameter, double density, double mass, double air_density,
        double viscosity) {
    double relative = air_velocity - velocity;
    double speed = fabs(relative);
    if (speed < 1.0e-12) return 0.0;
    double reynolds = air_density * diameter * speed / viscosity;
    double force;
    if (reynolds < 1.0e-8) {
        force = 3.0 * 3.14159265358979323846 * viscosity * diameter * relative;
    } else {
        double cd = reynolds < 1000.0
            ? 24.0 / reynolds * (1.0 + 0.15 * pow(reynolds, 0.687))
            : 0.44;
        double area = 0.78539816339744830962 * diameter * diameter;
        force = 0.5 * air_density * area * cd * relative * speed;
    }
    return force / mass;
}

extern "C" __global__ void integrate_batch(
        const double* waves, const double* diameters, const double* densities,
        double* final_positions, double* trajectory_positions,
        double* trajectory_velocities, int wave_stride, int samples,
        int candidate_count, int particle_count, int steps, double duration,
        double frequency, double air_density, double viscosity,
        double initial_position, double initial_velocity, int save_trajectory) {
    int item = blockDim.x * blockIdx.x + threadIdx.x;
    int total = candidate_count * particle_count;
    if (item >= total) return;
    int candidate = item / particle_count;
    int particle = item - candidate * particle_count;
    double diameter = diameters[particle];
    double density = densities[particle];
    double mass = density * 3.14159265358979323846 * diameter * diameter * diameter / 6.0;
    double dt = duration / steps;
    double x = initial_position;
    double v = initial_velocity;
    const int trajectory_stride = steps + 1;
    if (save_trajectory) {
        trajectory_positions[item * trajectory_stride] = x;
        trajectory_velocities[item * trajectory_stride] = v;
    }
    for (int i = 0; i < steps; ++i) {
        double t = i * dt;
        double u1 = flow_at(waves, wave_stride, samples, candidate, frequency * t);
        double a1 = acceleration(v, u1, diameter, density, mass, air_density, viscosity);
        double k1x = v;
        double k1v = a1;

        double u2 = flow_at(waves, wave_stride, samples, candidate, frequency * (t + 0.5 * dt));
        double a2 = acceleration(v + 0.5 * dt * k1v, u2, diameter, density, mass, air_density, viscosity);
        double k2x = v + 0.5 * dt * k1v;
        double k2v = a2;

        double a3 = acceleration(v + 0.5 * dt * k2v, u2, diameter, density, mass, air_density, viscosity);
        double k3x = v + 0.5 * dt * k2v;
        double k3v = a3;

        double u4 = flow_at(waves, wave_stride, samples, candidate, frequency * (t + dt));
        double a4 = acceleration(v + dt * k3v, u4, diameter, density, mass, air_density, viscosity);
        double k4x = v + dt * k3v;
        double k4v = a4;

        x += dt * (k1x + 2.0*k2x + 2.0*k3x + k4x) / 6.0;
        v += dt * (k1v + 2.0*k2v + 2.0*k3v + k4v) / 6.0;
        if (save_trajectory) {
            trajectory_positions[item * trajectory_stride + i + 1] = x;
            trajectory_velocities[item * trajectory_stride + i + 1] = v;
        }
    }
    final_positions[item] = x;
}
"""


def cupy_module():
    """Import CuPy and check that a CUDA device is available."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CUDA acceleration requires CuPy. Install a CuPy wheel matching "
            "your CUDA runtime, for example `pip install cupy-cuda12x`."
        ) from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA-capable GPU was detected.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(f"CUDA is unavailable: {exc}") from exc
    return cp


@lru_cache(maxsize=1)
def _kernel(cp):
    return cp.RawKernel(_KERNEL, "integrate_batch", options=("--std=c++11",))


def integrate_batch(
    waveforms: np.ndarray,
    frequency_hz: float,
    duration: float,
    diameters: np.ndarray,
    densities: np.ndarray,
    air_density: float,
    viscosity: float,
    steps_per_period: int,
    initial_position: float = 0.0,
    initial_velocity: float = 0.0,
    save_trajectory: bool = False,
):
    """Integrate candidate waveforms against particles on the active GPU."""
    cp = cupy_module()
    waves = np.ascontiguousarray(waveforms, dtype=np.float64)
    if waves.ndim == 1:
        waves = waves[None, :]
    diameters = np.ascontiguousarray(diameters, dtype=np.float64)
    densities = np.ascontiguousarray(densities, dtype=np.float64)
    if diameters.ndim != 1 or densities.shape != diameters.shape:
        raise ValueError("diameters and densities must be one-dimensional arrays of equal length")
    steps = max(1, int(np.ceil(duration * frequency_hz * steps_per_period)))
    d_waves = cp.asarray(waves)
    d_diameters = cp.asarray(diameters)
    d_densities = cp.asarray(densities)
    candidate_count, samples = waves.shape
    total = candidate_count * len(diameters)
    d_final = cp.empty(total, dtype=cp.float64)
    trajectory_size = total * (steps + 1) if save_trajectory else 1
    d_positions = cp.empty(trajectory_size, dtype=cp.float64)
    d_velocities = cp.empty(trajectory_size, dtype=cp.float64)
    threads = 128
    _kernel(cp)(((total + threads - 1) // threads,), (threads,), (
        d_waves, d_diameters, d_densities, d_final, d_positions, d_velocities,
        np.int32(samples), np.int32(samples), np.int32(candidate_count),
        np.int32(len(diameters)), np.int32(steps), np.float64(duration),
        np.float64(frequency_hz), np.float64(air_density), np.float64(viscosity),
        np.float64(initial_position), np.float64(initial_velocity),
        np.int32(bool(save_trajectory)),
    ))
    final = cp.asnumpy(d_final).reshape(candidate_count, len(diameters))
    if not save_trajectory:
        return final, steps
    positions = cp.asnumpy(d_positions).reshape(candidate_count, len(diameters), steps + 1)
    velocities = cp.asnumpy(d_velocities).reshape(candidate_count, len(diameters), steps + 1)
    return final, steps, positions, velocities


def sample_waveform(waveform, sample_count: int = 8192) -> np.ndarray:
    """Sample any periodic phase waveform for interpolation by the CUDA kernel."""
    phase = np.arange(sample_count, dtype=np.float64) / sample_count
    try:
        values = np.asarray(waveform(phase), dtype=np.float64)
    except (TypeError, ValueError):
        values = np.asarray([waveform(float(point)) for point in phase], dtype=np.float64)
    if values.ndim == 0:
        values = np.full(sample_count, float(values))
    return np.ascontiguousarray(values)
