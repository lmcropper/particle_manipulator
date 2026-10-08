# Piston-flow particle solver

This is an initial one-dimensional simulator for a glass microsphere in a
prescribed, spatially uniform piston flow. It is intended as a first model for
a sealed or semi-open chamber driven by a subwoofer cone.

The model integrates

```text
dx/dt = v
m dv/dt = F_drag(u_air - v) + F_axial
```

with quasi-steady spherical drag:

```text
Re = rho_air * d * abs(u_air - v) / mu
F_drag = 0.5 * rho_air * A * Cd(Re) * (u_air - v) * abs(u_air - v)
```

The default drag coefficient is the Schiller-Naumann correlation. The default
particle is a 100 micrometer glass sphere with density 2500 kg/m^3, and the
default air properties are approximately room-temperature air.

## Run the example

```bash
python -m pip install -r requirements.txt
python piston_particle_solver.py --frequency 10 --peak-speed 2
```

The example uses a smooth asymmetric waveform made from a fundamental plus a
second harmonic. It plots particle position, air and particle velocities,
particle Reynolds number, and drag force.

## Use a measured waveform

Create a CSV with one period of data:

```text
phase,velocity_m_per_s
0.0,0.0
0.1,0.8
...
1.0,0.0
```

Then load it with:

```python
from piston_particle_solver import PistonFlow, simulate

piston = PistonFlow.from_csv("waveform.csv", frequency_hz=10.0)
result = simulate(piston, duration=5.0)
```

The same can be run from the command line:

```bash
python piston_particle_solver.py --csv waveform.csv --frequency 10 --duration 5
```

Use `--no-plot` for batch runs. Other useful options include
`--diameter-um`, `--particle-density`, `--peak-speed`, and
`--second-harmonic`.

## Compare particle sizes

Run the same flow waveform against glass microspheres from 20 to 100 µm:

```bash
python compare_particle_sizes.py --frequency 10 --peak-speed 2
```

The comparison defaults to diameters of 20, 30, ..., 100 µm and a glass
density of 2500 kg/m³. It plots each particle's displacement and velocity and
prints final displacement, final speed, and maximum Reynolds number. Customize
the size list with `--sizes-um`:

```bash
python compare_particle_sizes.py --sizes-um 20,40,60,80,100 --duration 3
```

The same comparison can use a measured air-velocity waveform with `--csv
waveform.csv`. Use `--no-plot` to print the summary only.

## Optimize a transport waveform

Search for a zero-mean periodic waveform that transports the particles in a
chosen direction while reducing the spread in their final displacements:

```bash
python optimize_particle_waveform.py --duration 1 --frequency 10 --peak-speed 2
```

The script uses a Fourier waveform with three harmonics by default and
minimizes size-to-size displacement spread, subject to a minimum mean
transport requirement. It does not reward additional transport beyond that
floor. It reports each size's net displacement, plots the optimized waveform
and particle motion, and saves `optimized_waveform.csv` in the current directory. That CSV can be used
by the solver or size-comparison script with `--csv`. By default mean net
displacement across the selected sizes must reach at least 10 mm in the chosen
direction; change the floor with `--min-displacement-mm`. The script reports
failure and does not save a waveform if the search cannot meet that requirement. Use
`--sizes-um`, `--harmonics`, `--direction -1`, `--maxiter`, and `--seed` to
configure the search. The peak air speed remains fixed at the value set by
`--peak-speed`. During the search it prints the best mean transport, size
spread, floor status, and convergence once per iteration.

The velocity column should be the local air velocity in m/s, not the
subwoofer drive voltage. If the input is cone displacement, differentiate it
to obtain cone velocity and add the chamber area/flow model later.

## Important current assumptions

- The air velocity is spatially uniform and equal to the piston velocity.
- The other end is treated as open, so no chamber standing-wave solution is
  included.
- Drag is quasi-steady; added mass, Basset history, acoustic radiation force,
  and acoustic streaming are not yet included.
- The optical force is assumed to have no axial component by default.
- The waveform should be continuous or sufficiently well resolved. For sharp
  waveform corners, use a smaller integration step or integrate each segment
  separately.

The next model extension should replace `piston.velocity(t)` with
`air_velocity(x, t)` so that the open-ended chamber geometry can be included.
