# Acoustic Particle Manipulator

This PlatformIO project drives two ultrasonic driver inputs from an ESP32's
MCPWM peripheral. GPIO 25 is driver A and GPIO 26 is driver B. Connect those
pins only to logic-level driver inputs; they must not power the transducers
directly.

## Firmware

The project targets `esp32dev` and Arduino-ESP32 2.x:

```sh
cd ultrasound_phase
pio run --target upload
pio device monitor --baud 115200
```

The firmware command protocol is newline-terminated at 115200 baud:

| Command | Effect |
| --- | --- |
| `F <1000..80000>` | Set output frequency in Hz. |
| `P <-999..999>` | Set B's phase offset in 1/1000-period ticks. Negative phase values wrap; for example `P -250` is equivalent to 270°. |
| `D <1..99>` | Set duty cycle in percent. |
| `A <0\|1>` | Disable or enable driver A only. |
| `B <0\|1>` | Disable or enable driver B only. |
| `E <0\|1>` | Disable or enable both drivers. |
| `SWEEP <step> <delay_ms>` | Start a non-blocking cyclic phase sweep. `step` is signed, `-100..-1` or `1..100`, and `delay_ms` is `1..500`. Negative steps sweep in reverse. |
| `STOP` | Stop an active sweep. |
| `S` | Print a machine-readable status line. |
| `?` | Print command help. |

Disabling a driver forces its PWM output low. Changing frequency or duty does
not re-enable a disabled driver.

## Python GUI

The GUI provides port selection, waveform settings, independent A/B controls,
positive or negative phase sweeps, status synchronization, a device log, and a
raw-command field for the entire protocol.

```sh
cd ultrasound_phase/gui
python3 -m pip install -r requirements.txt
python3 particle_manipulator_gui.py
```

Use the GUI only after you have verified the wiring and driver supply. The
application does not substitute for a hardware emergency stop.
