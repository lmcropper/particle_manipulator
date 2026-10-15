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

CUDA acceleration is optional. Install a CuPy wheel that matches the installed
CUDA runtime (for example, `python -m pip install cupy-cuda12x`), then use
`--device cuda`. CUDA uses fixed-step RK4; set `--cuda-steps-per-period` to
adjust its time resolution (default 1000). The direct `simulate()` API also
accepts `backend="cuda"` and `cuda_steps_per_period=1000`. Its CPU backend keeps
the adaptive SciPy solver and remains the default. CUDA currently supports the
built-in sphere drag model and does not support custom axial-force callbacks.

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

Search for a periodic Fourier air-displacement waveform that gives similar net
transport across particle sizes:

```bash
python optimize_particle_waveform.py --duration 1 --frequency 80 --peak-air-displacement-mm 10
```

The Fourier coefficients define zero-mean air displacement. The script
differentiates that curve to obtain the air velocity used by the particle
solver. `--peak-air-displacement-mm` (default 10) is the maximum displacement
amplitude, and `--max-air-speed` (default 5 m/s) reduces that amplitude for a
candidate whose derivative would exceed the speed cap. The score is the
standard deviation of net transport minus `--transport-reward-weight`
(default 0.2) times the mean directed transport above
`--min-mean-displacement-mm` (default 5). The minimum is also a hard
optimization constraint. Set `--direction -1` for transport in the negative
solver direction. Random exploration seeds differential evolution with feasible
and diverse candidates. Its results include the spread, mean transport, score,
and displacement for each particle size. The saved `optimized_waveform.csv`
contains phase, derived air velocity, and air displacement; velocity remains
the second column for the solver's `--csv` loader. The optimizer still reports,
saves, and plots its best available result if it does not converge or misses
the transport minimum. It does not run a local polishing step.

The optimizer accepts `--device auto`, `--device cpu`, or `--device cuda`.
`auto` uses CUDA when CuPy and a CUDA GPU are available and otherwise falls
back to CPU. CUDA evaluates differential-evolution populations as batches of
particle simulations; its fixed-step RK4 resolution is controlled by
`--cuda-steps-per-period` (default 1000). CPU-only installations need no CuPy.
Both waveform optimizers report random-exploration progress, including samples
completed, feasible count, and the best result found so far.

The velocity column should be the local air velocity in m/s, not the
subwoofer drive voltage. If the input is cone displacement, differentiate it
to obtain cone velocity and add the chamber area/flow model later.

## Optimize an air displacement waveform

The segment optimizer parameterizes the periodic **air displacement** as
linear segments with adjustable durations and slopes:

```bash
python optimize_segment_waveform.py --segments 4 --peak-air-displacement-mm 10
```

`--peak-air-displacement-mm` sets the maximum absolute displacement from its
time-mean position. Before simulation, the script applies a periodic,
zero-phase Gaussian low-pass filter to this segmented displacement. The default
`--lowpass-hz 240` is the filter's -3 dB cutoff; set it to 0 to compare against
the unfiltered segments. The solver receives the derivative of the filtered
displacement, so air velocity is continuous across segment borders.
`--max-air-speed` (default 5 m/s) caps both unfiltered and filtered air speed,
and `--max-velocity-jump` (default 8 m/s) caps the unfiltered speed change at
every segment border, including the period boundary. A candidate that would
exceed a cap is simulated at a smaller displacement amplitude. The CSV saves
phase, filtered air velocity and displacement, then unfiltered velocity and
displacement. Filtered velocity remains the second column for the existing
`--csv` loader. This is an air displacement model, not yet a geometric
conversion from measured speaker-cone displacement.

## Saved optimization results

Both optimizers save a compressed MATLAB `.mat` file in
`optimization_results/` after each run, including runs that did not converge
or meet the transport minimum. Use `--results-dir` to choose another directory.
Filenames contain the waveform type, local timestamp, main frequency, transport
spread, mean directed transport, and feasibility status. Each file contains the optimizer
parameters and settings, sampled air displacement and velocity, every particle
trajectory and diagnostic array, and final transport metrics. Segment results
also include durations, vertices, slopes, and the raw segmented waveform;
Fourier results include all Fourier coefficients.

To recreate the same four-panel figure without rerunning the optimizer:

```bash
python plot_optimization_result.py optimization_results/YOUR_RESULT.mat
python plot_optimization_result.py optimization_results/YOUR_RESULT.mat --save-figure result.png --no-show
```

## Compare with triangle and sawtooth waves

`compare_triangle_symmetry.py` sweeps the fraction of each cycle spent rising
from minimum to maximum air displacement. A fraction of 0.5 is a symmetric
triangle; values close to 0 or 1 approach sawtooths in opposite directions.
The script reads frequency, particle sizes, duration, density, low-pass cutoff,
and displacement, speed, and segment-jump limits from an optimized `.mat` file.
It prints mean directed transport and standard deviation across particle sizes,
then saves a CSV and four-panel PNG beside the input file.

```bash
python compare_triangle_symmetry.py optimization_results/YOUR_800Hz_RESULT.mat --no-show --verify-best
```

Use `--fractions 0.01,0.05,0.25,0.5,0.75,0.95,0.99` for a shorter sweep. The
default `--device auto` uses CUDA when available; `--device cpu` runs the
adaptive solver for every triangle. `--verify-best` also checks the best
feasible triangle with the adaptive CPU solver. A near-sawtooth may have a
smaller displacement amplitude because the saved speed and segment-jump caps
apply to every comparison waveform.

## Sweep the amplitude of a saved waveform

Use `compare_amplitude_sweep.py` to keep an optimized waveform's shape fixed
while multiplying its air displacement and velocity by 0.1x through 10x. The
script reads the particle sizes, density, frequency, duration, and air
properties from the selected optimization `.mat` file, then saves a CSV of
per-size net transports and a PNG showing transport and spread versus
amplitude.

```bash
python compare_amplitude_sweep.py optimization_results/YOUR_RESULT.mat --no-show
```

Omit the `.mat` path to choose from the 20 most recent saved results in a
terminal.

The default sweep has 21 logarithmically spaced multipliers, including 1x.
Use `--factors 0.1,0.25,0.5,1,2,5,10` for a custom list or `--sizes-um
30,50,70,100` to choose different diameters. `--device auto` uses CUDA when
available; `--device cpu` uses the adaptive solver. Multipliers are not
clamped to the original optimization limits: the table, CSV, and shaded plot
region flag amplitudes that exceed those limits.

## Important current assumptions

- The prescribed air velocity is spatially uniform; speaker-cone motion is not
  converted through a chamber geometry.
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
