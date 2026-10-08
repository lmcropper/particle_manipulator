import unittest

import numpy as np

from piston_particle_solver import (
    AirProperties,
    Particle,
    PistonFlow,
    schiller_naumann_cd,
    simulate,
    stokes_drag_force,
)


class SolverTests(unittest.TestCase):
    def test_particle_defaults_are_reasonable(self):
        particle = Particle()
        self.assertAlmostEqual(particle.mass, 1.309e-9, delta=0.01e-9)
        self.assertGreater(particle.area, 0.0)

    def test_drag_coefficient_reduces_to_stokes_form(self):
        reynolds_number = 1e-4
        cd = schiller_naumann_cd(reynolds_number)
        self.assertAlmostEqual(cd / (24.0 / reynolds_number), 1.0, places=3)

    def test_asymmetric_waveform_has_zero_mean(self):
        piston = PistonFlow.asymmetric_harmonic(10.0, 2.0, 0.45)
        phase = np.linspace(0.0, 1.0, 100000, endpoint=False)
        mean_velocity = np.mean([piston.waveform(p) for p in phase])
        self.assertAlmostEqual(mean_velocity, 0.0, places=5)

    def test_simulation_produces_finite_output(self):
        piston = PistonFlow.asymmetric_harmonic(10.0, 0.5, 0.45)
        result = simulate(piston, duration=0.5)
        self.assertTrue(np.all(np.isfinite(result.position)))
        self.assertTrue(np.all(np.isfinite(result.particle_velocity)))
        self.assertTrue(np.all(result.reynolds_number >= 0.0))

    def test_strict_stokes_model_has_no_mean_velocity_drift(self):
        frequency = 10.0
        piston = PistonFlow.asymmetric_harmonic(frequency, 0.5, 0.45)
        result = simulate(
            piston,
            duration=20.0 / frequency,
            drag_model=stokes_drag_force,
        )
        # The final instantaneous velocity is phase-dependent; average over
        # the last period to test the long-time mean instead.
        mask = result.time > result.time[-1] - 1.0 / frequency
        mean_velocity = np.mean(result.particle_velocity[mask])
        self.assertAlmostEqual(mean_velocity, 0.0, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
