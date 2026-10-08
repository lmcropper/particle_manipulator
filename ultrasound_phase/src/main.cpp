/*
 * Acoustic particle manipulator firmware for ESP32 (Arduino / PlatformIO).
 *
 * GPIO 25: driver A input        GPIO 26: driver B input
 * Serial: 115200 baud, newline-terminated commands.  See README.md for the
 * command protocol used by the companion Python GUI.
 *
 * This uses the ESP32 Arduino 2.x legacy MCPWM API, therefore platformio.ini
 * pins a compatible espressif32 platform release.
 */

#include <Arduino.h>
#include "driver/mcpwm.h"

static constexpr gpio_num_t PIN_CH_A = GPIO_NUM_25;
static constexpr gpio_num_t PIN_CH_B = GPIO_NUM_26;

static constexpr uint32_t MIN_FREQ = 1000;
static constexpr uint32_t MAX_FREQ = 80000;
static constexpr uint32_t PHASE_PERIOD = 1000;
static constexpr uint32_t DEFAULT_FREQ = 40000;
static constexpr float DEFAULT_DUTY = 50.0f;
static constexpr uint32_t DEFAULT_PHASE = 500;

struct Params {
  uint32_t frequency = DEFAULT_FREQ;
  float duty = DEFAULT_DUTY;
  uint32_t phase = DEFAULT_PHASE;       // Always wrapped: 0 .. 999.
  bool driver_a_enabled = true;
  bool driver_b_enabled = true;
} g;

struct SweepState {
  bool running = false;
  int16_t step = 5;                     // Signed: negative means reverse.
  uint16_t delay_ms = 20;
  uint32_t position = DEFAULT_PHASE;
  uint32_t last_ms = 0;
  uint32_t last_report_ms = 0;
} sweep;

static uint32_t wrap_phase(int32_t phase) {
  phase %= static_cast<int32_t>(PHASE_PERIOD);
  if (phase < 0) phase += PHASE_PERIOD;
  return static_cast<uint32_t>(phase);
}

static void hw_set_phase(uint32_t ticks) {
  mcpwm_sync_config_t config = {
    .sync_sig = MCPWM_SELECT_TIMER0_SYNC,
    .timer_val = ticks,
    .count_direction = MCPWM_TIMER_DIRECTION_UP,
  };
  mcpwm_sync_configure(MCPWM_UNIT_0, MCPWM_TIMER_1, &config);
}

// Configure a channel when enabled, or force it low when disabled. Keeping
// this per-channel is important: adjusting frequency/duty must not re-enable
// a driver that was deliberately switched off.
static void hw_apply_channel(mcpwm_timer_t timer, bool enabled) {
  if (enabled) {
    mcpwm_set_duty(MCPWM_UNIT_0, timer, MCPWM_OPR_A, g.duty);
    mcpwm_set_duty_type(MCPWM_UNIT_0, timer, MCPWM_OPR_A,
                        MCPWM_DUTY_MODE_0);
  } else {
    mcpwm_set_signal_low(MCPWM_UNIT_0, timer, MCPWM_OPR_A);
  }
}

static void hw_apply_outputs() {
  mcpwm_set_frequency(MCPWM_UNIT_0, MCPWM_TIMER_0, g.frequency);
  mcpwm_set_frequency(MCPWM_UNIT_0, MCPWM_TIMER_1, g.frequency);
  hw_apply_channel(MCPWM_TIMER_0, g.driver_a_enabled);
  hw_apply_channel(MCPWM_TIMER_1, g.driver_b_enabled);
  hw_set_phase(g.phase); // Frequency changes the timer period; re-sync B.
}

static void hw_init() {
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, PIN_CH_A);
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM1A, PIN_CH_B);

  mcpwm_config_t config = {
    .frequency = g.frequency,
    .cmpr_a = g.duty,
    .cmpr_b = 0.0f,
    .duty_mode = MCPWM_DUTY_MODE_0,
    .counter_mode = MCPWM_UP_COUNTER,
  };
  mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &config);
  mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_1, &config);
  mcpwm_set_timer_sync_output(MCPWM_UNIT_0, MCPWM_TIMER_0,
                              MCPWM_SWSYNC_SOURCE_TEZ);
  hw_set_phase(g.phase);
}

static const char *on_off(bool value) { return value ? "ON" : "OFF"; }

static void print_status() {
  Serial.printf(
      "[STATUS] freq=%lu phase=%lu phase_deg=%.1f duty=%.1f driver_a=%s "
      "driver_b=%s sweep=%s step=%d delay_ms=%u\n",
      static_cast<unsigned long>(g.frequency),
      static_cast<unsigned long>(g.phase),
      g.phase * 360.0f / PHASE_PERIOD,
      g.duty,
      on_off(g.driver_a_enabled), on_off(g.driver_b_enabled),
      on_off(sweep.running), sweep.step, sweep.delay_ms);
}

static void print_help() {
  Serial.println(F("Commands:"));
  Serial.println(F("  F <1000..80000>        Set frequency in Hz"));
  Serial.println(F("  P <-999..999>          Set B phase in ticks; negative values wrap"));
  Serial.println(F("  D <1..99>              Set duty cycle (%)"));
  Serial.println(F("  A <0|1> / B <0|1>      Disable/enable driver A or B"));
  Serial.println(F("  E <0|1>                Disable/enable both drivers"));
  Serial.println(F("  SWEEP <step> <ms>      Signed step -100..100, excluding 0"));
  Serial.println(F("                           Negative steps sweep phase in reverse"));
  Serial.println(F("  STOP                    Stop a sweep"));
  Serial.println(F("  S                       Print status"));
  Serial.println(F("  ?                       Print this help"));
}

static bool parse_long(const String &text, long &value) {
  const char *begin = text.c_str();
  char *end = nullptr;
  value = strtol(begin, &end, 10);
  if (begin == end) return false;
  while (*end && isspace(static_cast<unsigned char>(*end))) ++end;
  return *end == '\0';
}

static void handle_command(const String &raw) {
  String command = raw;
  command.trim();
  if (!command.length()) return;

  String upper = command;
  upper.toUpperCase();
  Serial.print(F(">> "));
  Serial.println(command);

  if (upper == "S") { print_status(); return; }
  if (upper == "?") { print_help(); return; }
  if (upper == "STOP") {
    sweep.running = false;
    Serial.println(F("[OK] Sweep stopped"));
    print_status();
    return;
  }

  if (upper.startsWith("SWEEP")) {
    String args = command.substring(5);
    args.trim();
    int step = 0;
    int delay = 0;
    char extra = '\0';
    if (sscanf(args.c_str(), "%d %d %c", &step, &delay, &extra) != 2 ||
        step < -100 || step > 100 || step == 0 || delay < 1 || delay > 500) {
      Serial.println(F("[ERR] Usage: SWEEP <step -100..100, nonzero> <delay_ms 1..500>"));
      return;
    }
    sweep.step = static_cast<int16_t>(step);
    sweep.delay_ms = static_cast<uint16_t>(delay);
    sweep.position = g.phase;
    sweep.last_ms = millis();
    sweep.last_report_ms = sweep.last_ms;
    sweep.running = true;
    Serial.printf("[OK] Sweep started: phase=%lu step=%d delay_ms=%d\n",
                  static_cast<unsigned long>(sweep.position), step, delay);
    return;
  }

  const char opcode = toupper(static_cast<unsigned char>(command.charAt(0)));
  String argument = command.substring(1);
  argument.trim();
  long value = 0;
  if (!parse_long(argument, value)) {
    Serial.println(F("[ERR] Expected a numeric argument"));
    return;
  }

  switch (opcode) {
    case 'F':
      if (value < MIN_FREQ || value > MAX_FREQ) {
        Serial.println(F("[ERR] Frequency must be 1000..80000 Hz"));
        return;
      }
      g.frequency = static_cast<uint32_t>(value);
      hw_apply_outputs();
      Serial.printf("[OK] freq=%lu Hz\n", static_cast<unsigned long>(g.frequency));
      return;

    case 'P':
      if (value < -999 || value > 999) {
        Serial.println(F("[ERR] Phase must be -999..999 ticks"));
        return;
      }
      g.phase = wrap_phase(static_cast<int32_t>(value));
      sweep.position = g.phase;
      sweep.running = false; // A direct phase change takes precedence.
      hw_set_phase(g.phase);
      Serial.printf("[OK] phase=%lu (%.1f deg)\n", static_cast<unsigned long>(g.phase),
                    g.phase * 360.0f / PHASE_PERIOD);
      return;

    case 'D':
      if (value < 1 || value > 99) {
        Serial.println(F("[ERR] Duty must be 1..99 percent"));
        return;
      }
      g.duty = static_cast<float>(value);
      hw_apply_outputs();
      Serial.printf("[OK] duty=%.1f%%\n", g.duty);
      return;

    case 'A':
      if (value != 0 && value != 1) {
        Serial.println(F("[ERR] A accepts 0 (off) or 1 (on)"));
        return;
      }
      g.driver_a_enabled = value == 1;
      hw_apply_channel(MCPWM_TIMER_0, g.driver_a_enabled);
      Serial.printf("[OK] driver_a=%s\n", on_off(g.driver_a_enabled));
      return;

    case 'B':
      if (value != 0 && value != 1) {
        Serial.println(F("[ERR] B accepts 0 (off) or 1 (on)"));
        return;
      }
      g.driver_b_enabled = value == 1;
      hw_apply_channel(MCPWM_TIMER_1, g.driver_b_enabled);
      Serial.printf("[OK] driver_b=%s\n", on_off(g.driver_b_enabled));
      return;

    case 'E':
      if (value != 0 && value != 1) {
        Serial.println(F("[ERR] E accepts 0 (off) or 1 (on)"));
        return;
      }
      g.driver_a_enabled = value == 1;
      g.driver_b_enabled = value == 1;
      hw_apply_outputs();
      Serial.printf("[OK] drivers=%s\n", value ? "ON" : "OFF");
      return;
  }

  Serial.printf("[ERR] Unknown command '%s' (? for help)\n", command.c_str());
}

static String command_buffer;

void setup() {
  Serial.begin(115200);
  delay(300);
  hw_init();
  Serial.println(F("\r\n=== Acoustic Particle Manipulator — ESP32 ==="));
  print_status();
  print_help();
}

void loop() {
  while (Serial.available()) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\n' || character == '\r') {
      handle_command(command_buffer);
      command_buffer = "";
    } else if (isprint(static_cast<unsigned char>(character)) && command_buffer.length() < 96) {
      command_buffer += character;
    }
  }

  const uint32_t now = millis();
  if (sweep.running && now - sweep.last_ms >= sweep.delay_ms) {
    sweep.last_ms = now;
    sweep.position = wrap_phase(static_cast<int32_t>(sweep.position) + sweep.step);
    g.phase = sweep.position;
    hw_set_phase(g.phase);

    // Reporting every step would make a 1 ms sweep serial-bound. Limit log
    // traffic so the configured timing is governed by the MCU state machine.
    if (now - sweep.last_report_ms < 200) return;
    sweep.last_report_ms = now;
    Serial.printf("[SWEEP] phase=%lu phase_deg=%.1f\n",
                  static_cast<unsigned long>(g.phase),
                  g.phase * 360.0f / PHASE_PERIOD);
  }
}
