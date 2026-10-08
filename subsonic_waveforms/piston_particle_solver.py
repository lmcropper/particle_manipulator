"""One-dimensional particle-motion solver for a prescribed piston-flow waveform.

The initial model assumes that the air velocity is spatially uniform and equal
to the piston velocity.  The state is [position, velocity], and the particle
experiences quasi-steady sphere drag based on the instantaneous relative
velocity w = u_air - v_particle.

This is intentionally a reduced model.  The next natural extension is to
replace PistonFlow.velocity(t) with a chamber field velocity(x, t).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


ScalarFunction = Callable[[float], float]
ForceFunction = Callable[[float, float], float]
DragFunction = Callable[[float, "Particle", "AirProperties"], float]


@dataclass(frozen=True)
class AirProperties:
    """Room-temperature air properties in SI units."""

    density: float = 1.20       # kg/m^3
    dynamic_viscosity: float = 1.81e-5  # Pa s


@dataclass(frozen=True)
class Particle:
    """Spherical particle properties in SI units."""

    diameter: float = 49e-6   # m
    density: float = 157.0    # kg/m^3, glass

    @property
    def area(self) -> float:
        return np.pi * self.diameter**2 / 4.0

    @property
    def mass(self) -> float:
        return self.density * np.pi * self.diameter**3 / 6.0

    @property
    def stokes_response_time(self) -> float:
        """Response time m/(3*pi*mu*d) for the supplied air properties."""
        return self.mass / (3.0 * np.pi * AirProperties().dynamic_viscosity * self.diameter)


@dataclass
class PistonFlow:
    """Spatially uniform piston-flow model.

    ``waveform`` must return a dimensionless periodic velocity shape as a
    function of phase in [0, 1).  It should have zero mean if the piston has
    zero net displacement over one period.
    """

    frequency_hz: float
    peak_speed: float
    waveform: ScalarFunction

    def velocity(self, t: float) -> float:
        phase = (self.frequency_hz * t) % 1.0
        return self.peak_speed * float(self.waveform(phase))

    @staticmethod
    def asymmetric_harmonic(
        frequency_hz: float,
        peak_speed: float,
        second_harmonic: float = 0.45,
    ) -> "PistonFlow":
        """Create a smooth, zero-mean asymmetric waveform.

        The shape is sin(2*pi*phase) + a*sin(4*pi*phase), normalized to unit
        peak magnitude.  The second harmonic breaks half-wave symmetry and
        makes the positive and negative portions dynamically different.
        """

        raw = lambda phase: np.sin(2.0 * np.pi * phase) + second_harmonic * np.sin(
            4.0 * np.pi * phase
        )
        samples = np.linspace(0.0, 1.0, 10001, endpoint=False)
        normalization = np.max(np.abs(raw(samples)))
        return PistonFlow(
            frequency_hz=frequency_hz,
            peak_speed=peak_speed,
            waveform=lambda phase: raw(phase) / normalization,
        )

    @staticmethod
    def from_csv(
        path: str | Path,
        frequency_hz: float,
        phase_column: int = 0,
        velocity_column: int = 1,
        delimiter: str = ",",
        skiprows: int = 1,
    ) -> "PistonFlow":
        """Load one period of a measured waveform from a CSV file.

        The phase column should run from 0 to 1.  The velocity column is in
        m/s, so ``peak_speed`` is set to 1 and the interpolator returns the
        physical velocity directly.
        """

        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
        phase = np.asarray(data[:, phase_column], dtype=float)
        velocity = np.asarray(data[:, velocity_column], dtype=float)

        order = np.argsort(phase)
        phase = phase[order]
        velocity = velocity[order]

        if phase[0] < 0.0 or phase[-1] > 1.0:
            raise ValueError("CSV phase values must lie between 0 and 1.")
        if np.any(np.diff(phase) <= 0.0):
            raise ValueError("CSV phase values must be strictly increasing.")

        if phase[0] > 0.0:
            phase = np.insert(phase, 0, 0.0)
            velocity = np.insert(velocity, 0, velocity[-1])
        if phase[-1] < 1.0:
            phase = np.append(phase, 1.0)
            velocity = np.append(velocity, velocity[0])

        interpolator = PchipInterpolator(phase, velocity, extrapolate=False)
        return PistonFlow(
            frequency_hz=frequency_hz,
            peak_speed=1.0,
            waveform=lambda p: float(interpolator(p)),
        )


@dataclass
class SimulationResult:
    time: np.ndarray
    position: np.ndarray
    particle_velocity: np.ndarray
    air_velocity: np.ndarray
    relative_velocity: np.ndarray
    reynolds_number: np.ndarray
    drag_force: np.ndarray

    @property
    def period(self) -> float:
        return self.time[-1] - self.time[0]

    def cycle_displacements(self, frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
        """Return cycle start times and displacement over each completed cycle."""
        T = 1.0 / frequency_hz
        cycle_index = np.floor((self.time - self.time[0]) / T).astype(int)
        starts = []
        displacements = []
        for index in range(cycle_index.max()):
            mask = cycle_index == index
            next_mask = cycle_index == index + 1
            if np.any(mask) and np.any(next_mask):
                starts.append(self.time[mask][0])
                displacements.append(self.position[next_mask][0] - self.position[mask][0])
        return np.asarray(starts), np.asarray(displacements)


def schiller_naumann_cd(reynolds_number: float) -> float:
    """Schiller-Naumann sphere drag coefficient for Re below about 1000."""
    if reynolds_number <= 1e-12:
        return np.inf
    if reynolds_number < 1000.0:
        return 24.0 / reynolds_number * (1.0 + 0.15 * reynolds_number**0.687)
    return 0.44


def sphere_drag_force(
    relative_velocity: float,
    particle: Particle,
    air: AirProperties,
) -> float:
    """Quasi-steady drag force, positive in the positive coordinate direction."""
    w = float(relative_velocity)
    if abs(w) < 1e-12:
        return 0.0

    reynolds_number = air.density * particle.diameter * abs(w) / air.dynamic_viscosity
    if reynolds_number < 1e-8:
        return 3.0 * np.pi * air.dynamic_viscosity * particle.diameter * w

    cd = schiller_naumann_cd(reynolds_number)
    return 0.5 * air.density * particle.area * cd * w * abs(w)


def stokes_drag_force(
    relative_velocity: float,
    particle: Particle,
    air: AirProperties,
) -> float:
    """Strictly linear Stokes drag, useful as a validation model."""
    return 3.0 * np.pi * air.dynamic_viscosity * particle.diameter * relative_velocity


def simulate(
    piston: PistonFlow,
    particle: Particle = Particle(),
    air: AirProperties = AirProperties(),
    duration: float = 2.0,
    initial_position: float = 0.0,
    initial_velocity: float = 0.0,
    axial_force: Optional[ForceFunction] = None,
    max_step_fraction: int = 200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    drag_model: DragFunction = sphere_drag_force,
) -> SimulationResult:
    """Integrate particle motion in a spatially uniform piston flow."""
    if piston.frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    period = 1.0 / piston.frequency_hz
    max_step = period / max_step_fraction

    def rhs(t: float, state: np.ndarray) -> list[float]:
        position, velocity = state
        air_velocity = piston.velocity(t)
        relative_velocity = air_velocity - velocity
        drag = drag_model(relative_velocity, particle, air)
        extra_force = 0.0 if axial_force is None else axial_force(position, velocity)
        return [velocity, (drag + extra_force) / particle.mass]

    solution = solve_ivp(
        rhs,
        (0.0, duration),
        (initial_position, initial_velocity),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    time = solution.t
    position = solution.y[0]
    particle_velocity = solution.y[1]
    air_velocity = np.array([piston.velocity(t) for t in time])
    relative_velocity = air_velocity - particle_velocity
    reynolds_number = air.density * particle.diameter * np.abs(relative_velocity) / air.dynamic_viscosity
    drag_force = np.array([drag_model(w, particle, air) for w in relative_velocity])

    return SimulationResult(
        time=time,
        position=position,
        particle_velocity=particle_velocity,
        air_velocity=air_velocity,
        relative_velocity=relative_velocity,
        reynolds_number=reynolds_number,
        drag_force=drag_force,
    )


def plot_result(result: SimulationResult, frequency_hz: float) -> None:
    """Plot trajectory, velocities, Reynolds number, and drag."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(result.time, result.position * 1e3)
    axes[0].set_ylabel("Position [mm]")

    axes[1].plot(result.time, result.air_velocity, label="air")
    axes[1].plot(result.time, result.particle_velocity, label="particle")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend()

    axes[2].plot(result.time, result.reynolds_number)
    axes[2].set_ylabel("Particle Re")
    axes[2].set_yscale("log")

    axes[3].plot(result.time, result.drag_force * 1e9)
    axes[3].set_ylabel("Drag [nN]")
    axes[3].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"Piston-flow particle simulation: {frequency_hz:g} Hz")
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=10.0, help="Drive frequency [Hz].")
    parser.add_argument("--peak-speed", type=float, default=2.0, help="Peak air speed [m/s].")
    parser.add_argument(
        "--second-harmonic",
        type=float,
        default=0.45,
        help="Second-harmonic coefficient for the built-in waveform.",
    )
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration [s].")
    parser.add_argument(
        "--diameter-um", type=float, default=100.0, help="Particle diameter [micrometers]."
    )
    parser.add_argument("--particle-density", type=float, default=2500.0, help="Particle density [kg/m^3].")
    parser.add_argument("--csv", type=str, default=None, help="CSV containing phase and velocity columns.")
    parser.add_argument("--no-plot", action="store_true", help="Do not open the result plot.")
    args = parser.parse_args()

    frequency = args.frequency
    if args.csv is None:
        piston = PistonFlow.asymmetric_harmonic(
            frequency_hz=frequency,
            peak_speed=args.peak_speed,
            second_harmonic=args.second_harmonic,
        )
    else:
        piston = PistonFlow.from_csv(args.csv, frequency_hz=frequency)

    particle = Particle(
        diameter=args.diameter_um * 1e-6,
        density=args.particle_density,
    )
    duration = args.duration if args.duration is not None else 20.0 / frequency
    result = simulate(piston, particle=particle, duration=duration)

    starts, displacements = result.cycle_displacements(frequency)
    if len(displacements):
        print(f"Final position: {result.position[-1]*1e3:.4f} mm")
        print(f"Final particle velocity: {result.particle_velocity[-1]:.4f} m/s")
        print(f"Maximum particle Re: {result.reynolds_number.max():.3g}")
        print(f"Last cycle displacement: {displacements[-1]*1e3:.5f} mm")

    if not args.no_plot:
        plot_result(result, frequency)


if __name__ == "__main__":
    main()
