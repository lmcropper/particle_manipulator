#!/usr/bin/env python3
"""Desktop serial control panel for the ESP32 acoustic particle manipulator."""

from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import serial
from serial import SerialException
from serial.tools import list_ports


STATUS_RE = re.compile(r"(freq|phase|phase_deg|duty|driver_a|driver_b|sweep|step|delay_ms)=([^\s]+)")


class ManipulatorGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Acoustic Particle Manipulator")
        self.minsize(730, 565)
        self.serial_port: serial.Serial | None = None
        self.serial_lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self.stop_reader = threading.Event()
        self.messages: queue.Queue[str] = queue.Queue()

        self.port_var = tk.StringVar()
        self.connection_var = tk.StringVar(value="Disconnected")
        self.frequency_var = tk.StringVar(value="40000")
        self.phase_var = tk.StringVar(value="500")
        self.phase_degrees_var = tk.StringVar(value="180.0°")
        self.duty_var = tk.StringVar(value="50")
        self.sweep_step_var = tk.StringVar(value="5")
        self.sweep_delay_var = tk.StringVar(value="20")
        self.driver_a_var = tk.BooleanVar(value=True)
        self.driver_b_var = tk.BooleanVar(value=True)
        self.raw_command_var = tk.StringVar()

        self._build_ui()
        self.refresh_ports()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(75, self.drain_messages)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        connection = ttk.LabelFrame(outer, text="Serial connection", padding=8)
        connection.grid(row=0, column=0, sticky="ew")
        connection.columnconfigure(1, weight=1)
        ttk.Label(connection, text="Port").grid(row=0, column=0, padx=(0, 6))
        self.port_menu = ttk.Combobox(connection, textvariable=self.port_var, state="readonly")
        self.port_menu.grid(row=0, column=1, sticky="ew")
        ttk.Button(connection, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=6)
        self.connect_button = ttk.Button(connection, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=3)
        ttk.Label(connection, textvariable=self.connection_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        controls = ttk.Frame(outer)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        output = ttk.LabelFrame(controls, text="Waveform", padding=8)
        output.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        output.columnconfigure(1, weight=1)
        self._entry_row(output, 0, "Frequency (Hz)", self.frequency_var, "Apply", self.apply_frequency)
        self._entry_row(output, 1, "Duty cycle (%)", self.duty_var, "Apply", self.apply_duty)
        ttk.Label(output, text="Phase offset (ticks)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(output, textvariable=self.phase_var, width=10).grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(output, text="Apply", command=self.apply_phase).grid(row=2, column=2, pady=4)
        ttk.Label(output, textvariable=self.phase_degrees_var).grid(row=3, column=0, columnspan=3, sticky="w")
        self.phase_scale = ttk.Scale(output, from_=0, to=999, orient="horizontal", command=self.update_phase_from_scale)
        self.phase_scale.set(500)
        self.phase_scale.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        ttk.Label(output, text="Direct phase accepts -999…999; it wraps to 0…999.").grid(row=5, column=0, columnspan=3, sticky="w", pady=(3, 0))

        drivers = ttk.LabelFrame(controls, text="Drivers and sweep", padding=8)
        drivers.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ttk.Checkbutton(drivers, text="Driver A enabled (GPIO 25)", variable=self.driver_a_var,
                        command=lambda: self.set_driver("A", self.driver_a_var.get())).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(drivers, text="Driver B enabled (GPIO 26)", variable=self.driver_b_var,
                        command=lambda: self.set_driver("B", self.driver_b_var.get())).grid(row=1, column=0, columnspan=2, sticky="w")
        button_row = ttk.Frame(drivers)
        button_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(button_row, text="Enable both", command=lambda: self.send("E 1")).grid(row=0, column=0)
        ttk.Button(button_row, text="Disable both", command=lambda: self.send("E 0")).grid(row=0, column=1, padx=6)
        ttk.Separator(drivers).grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(drivers, text="Signed step (ticks)").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(drivers, textvariable=self.sweep_step_var, width=10).grid(row=4, column=1, sticky="ew", pady=3)
        ttk.Label(drivers, text="Interval (ms)").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(drivers, textvariable=self.sweep_delay_var, width=10).grid(row=5, column=1, sticky="ew", pady=3)
        sweep_buttons = ttk.Frame(drivers)
        sweep_buttons.grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Button(sweep_buttons, text="Start sweep", command=self.start_sweep).grid(row=0, column=0)
        ttk.Button(sweep_buttons, text="Stop", command=lambda: self.send("STOP")).grid(row=0, column=1, padx=6)
        ttk.Label(drivers, text="Positive steps advance; negative steps run in reverse.").grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

        raw = ttk.LabelFrame(outer, text="Advanced command", padding=8)
        raw.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        raw.columnconfigure(0, weight=1)
        command = ttk.Entry(raw, textvariable=self.raw_command_var)
        command.grid(row=0, column=0, sticky="ew")
        command.bind("<Return>", lambda _event: self.send_raw())
        ttk.Button(raw, text="Send", command=self.send_raw).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(raw, text="Request status", command=lambda: self.send("S")).grid(row=0, column=2, padx=(6, 0))

        log_frame = ttk.LabelFrame(outer, text="Device log", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, state="disabled", wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    def _entry_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar,
                   button_text: str, command: object) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(parent, text=button_text, command=command).grid(row=row, column=2, pady=4)

    def refresh_ports(self) -> None:
        ports = [port.device for port in list_ports.comports()]
        self.port_menu["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        if not ports:
            self.port_var.set("")

    def toggle_connection(self) -> None:
        if self.serial_port:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        port = self.port_var.get()
        if not port:
            messagebox.showerror("No serial port", "Select the ESP32 serial port first.")
            return
        try:
            self.serial_port = serial.Serial(port, 115200, timeout=0.15)
        except SerialException as exc:
            messagebox.showerror("Could not connect", str(exc))
            return
        self.stop_reader.clear()
        self.reader = threading.Thread(target=self.read_serial, daemon=True)
        self.reader.start()
        self.connection_var.set(f"Connected to {port} at 115200 baud")
        self.connect_button.configure(text="Disconnect")
        self.send("S")

    def disconnect(self) -> None:
        self.stop_reader.set()
        if self.serial_port:
            with self.serial_lock:
                self.serial_port.close()
        self.serial_port = None
        self.connection_var.set("Disconnected")
        self.connect_button.configure(text="Connect")

    def read_serial(self) -> None:
        while not self.stop_reader.is_set():
            port = self.serial_port
            if not port:
                return
            try:
                line = port.readline().decode("utf-8", errors="replace").strip()
            except (SerialException, OSError) as exc:
                self.messages.put(f"[GUI] Serial read error: {exc}")
                return
            if line:
                self.messages.put(line)

    def send(self, command: str) -> None:
        if not self.serial_port:
            self.append_log("[GUI] Connect to the ESP32 before sending commands.")
            return
        try:
            with self.serial_lock:
                if self.serial_port:
                    self.serial_port.write((command.strip() + "\n").encode("ascii"))
        except (SerialException, OSError) as exc:
            self.append_log(f"[GUI] Serial write error: {exc}")
            self.disconnect()

    def send_raw(self) -> None:
        command = self.raw_command_var.get().strip()
        if command:
            self.send(command)
            self.raw_command_var.set("")

    def apply_frequency(self) -> None:
        self.send_validated("F", self.frequency_var.get(), 1000, 80000, "Frequency")

    def apply_duty(self) -> None:
        self.send_validated("D", self.duty_var.get(), 1, 99, "Duty cycle")

    def apply_phase(self) -> None:
        try:
            phase = int(self.phase_var.get())
        except ValueError:
            messagebox.showerror("Invalid phase", "Phase must be an integer from -999 to 999.")
            return
        if not -999 <= phase <= 999:
            messagebox.showerror("Invalid phase", "Phase must be from -999 to 999 ticks.")
            return
        self.phase_degrees_var.set(f"{phase * 360 / 1000:.1f}° (input)")
        self.send(f"P {phase}")

    def send_validated(self, opcode: str, text: str, minimum: int, maximum: int, label: str) -> None:
        try:
            value = int(text)
        except ValueError:
            messagebox.showerror(f"Invalid {label.lower()}", f"{label} must be a whole number.")
            return
        if not minimum <= value <= maximum:
            messagebox.showerror(f"Invalid {label.lower()}", f"{label} must be {minimum} to {maximum}.")
            return
        self.send(f"{opcode} {value}")

    def set_driver(self, name: str, enabled: bool) -> None:
        self.send(f"{name} {int(enabled)}")

    def start_sweep(self) -> None:
        try:
            step = int(self.sweep_step_var.get())
            delay = int(self.sweep_delay_var.get())
        except ValueError:
            messagebox.showerror("Invalid sweep", "Step and interval must be whole numbers.")
            return
        if step == 0 or not -100 <= step <= 100 or not 1 <= delay <= 500:
            messagebox.showerror("Invalid sweep", "Step must be -100…-1 or 1…100; interval must be 1…500 ms.")
            return
        self.send(f"SWEEP {step} {delay}")

    def update_phase_from_scale(self, value: str) -> None:
        phase = round(float(value))
        self.phase_var.set(str(phase))
        self.phase_degrees_var.set(f"{phase * 360 / 1000:.1f}°")

    def drain_messages(self) -> None:
        try:
            while True:
                line = self.messages.get_nowait()
                self.append_log(line)
                self.process_status(line)
        except queue.Empty:
            pass
        self.after(75, self.drain_messages)

    def process_status(self, line: str) -> None:
        if not line.startswith("[STATUS]"):
            return
        fields = dict(STATUS_RE.findall(line))
        if "freq" in fields:
            self.frequency_var.set(fields["freq"])
        if "phase" in fields:
            self.phase_var.set(fields["phase"])
            self.phase_scale.set(float(fields["phase"]))
        if "phase_deg" in fields:
            self.phase_degrees_var.set(f"{fields['phase_deg']}°")
        if "duty" in fields:
            self.duty_var.set(fields["duty"])
        if "driver_a" in fields:
            self.driver_a_var.set(fields["driver_a"] == "ON")
        if "driver_b" in fields:
            self.driver_b_var.set(fields["driver_b"] == "ON")
        if "step" in fields:
            self.sweep_step_var.set(fields["step"])
        if "delay_ms" in fields:
            self.sweep_delay_var.set(fields["delay_ms"])

    def append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        self.disconnect()
        self.destroy()


if __name__ == "__main__":
    ManipulatorGui().mainloop()
