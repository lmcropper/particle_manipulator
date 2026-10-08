/*
 * ═══════════════════════════════════════════════════════════════════════════
 *  Acoustic Particle Manipulator — ESP32 / Arduino Framework
 *
 *  Drives two 40 kHz ultrasonic transducers via the MCPWM peripheral.
 *  Channel B phase is offset relative to Channel A using Timer0's TEZ
 *  (timer-equals-zero) sync output, so the node pattern can be translated
 *  along the tube axis without changing frequency.
 *
 *  Target: arduino-esp32 2.x  (ESP-IDF 4.4)
 *  NOTE:   arduino-esp32 3.x uses a different MCPWM API — see bottom of
 *          file for porting notes.
 *
 *  Hardware connections (to your H-bridge / gate-driver logic inputs):
 *    GPIO 25  →  Driver A  →  Transducer A  (left / end 1)
 *    GPIO 26  →  Driver B  →  Transducer B  (right / end 2)
 *
 *  Serial: 115200 baud  |  LF-terminated commands
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  PHASE MECHANICS
 *  ───────────────
 *  The MCPWM timer counts 0-999 per period.  Timer 0 broadcasts a sync
 *  pulse every time it hits zero.  Timer 1 catches that pulse and snaps
 *  its counter to `phase_ticks`, shifting Channel B's waveform forward by:
 *
 *      Δφ (degrees) = (phase_ticks / 1000) × 360
 *
 *  phase_ticks = 0   → Ch B in-phase    (nodes co-located with A-only pattern)
 *  phase_ticks = 500 → Ch B at 180°     (classic standing-wave trap)
 *  Sweeping 0 → 999 translates trapped particles by one full node spacing
 *  (λ/2 ≈ 4.3 mm at 40 kHz in air).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  COMMAND REFERENCE  (115200 baud, newline-terminated)
 *
 *    F <Hz>              Set frequency            (1000 – 80000 Hz)
 *    P <0-999>           Set Ch B phase offset    (0=0°  500=180°  999≈360°)
 *    D <1-99>            Set duty cycle           (%)
 *    E <0|1>             Disable / enable output
 *    SWEEP <step> <ms>   Non-blocking phase sweep (e.g. SWEEP 5 20)
 *                          step  : 1-100 ticks per advance
 *                          ms    : 1-500 ms between advances
 *    STOP                Abort active sweep
 *    S                   Print current status
 *    ?                   Print this help
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <Arduino.h>
#include "driver/mcpwm.h"
#include "soc/mcpwm_periph.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Pin & default config  — edit here
// ─────────────────────────────────────────────────────────────────────────────

static constexpr gpio_num_t PIN_CH_A = GPIO_NUM_25;   // Transducer A
static constexpr gpio_num_t PIN_CH_B = GPIO_NUM_26;   // Transducer B

static constexpr uint32_t DEF_FREQ  = 40000;  // Hz
static constexpr float    DEF_DUTY  = 50.0f;  // %  (50 = square wave)
static constexpr uint32_t DEF_PHASE = 500;    // ticks  (500 = 180°)


// ─────────────────────────────────────────────────────────────────────────────
//  Runtime state
// ─────────────────────────────────────────────────────────────────────────────

struct Params {
    uint32_t freq    = DEF_FREQ;
    float    duty    = DEF_DUTY;
    uint32_t phase   = DEF_PHASE;
    bool     enabled = true;
} g;

struct SweepState {
    bool     running  = false;
    uint32_t step     = 5;    // ticks per advance
    uint32_t delay_ms = 20;   // ms between advances
    uint32_t pos      = 0;    // current position (0-999)
    uint32_t last_ms  = 0;
} sw;


// ─────────────────────────────────────────────────────────────────────────────
//  MCPWM hardware layer
// ─────────────────────────────────────────────────────────────────────────────

/** Apply phase offset to Channel B. Ticks range 0-999. */
static void hw_phase(uint32_t ticks) {
    mcpwm_sync_config_t cfg = {
        .sync_sig        = MCPWM_SELECT_TIMER0_SYNC,   // slave to Timer 0's TEZ pulse
        .timer_val       = ticks,                       // counter value on sync event
        .count_direction = MCPWM_TIMER_DIRECTION_UP
    };
    mcpwm_sync_configure(MCPWM_UNIT_0, MCPWM_TIMER_1, &cfg);
}

/** Set frequency and duty on both channels. Restores normal PWM mode. */
static void hw_freq_duty(uint32_t freq, float duty) {
    mcpwm_set_frequency(MCPWM_UNIT_0, MCPWM_TIMER_0, freq);
    mcpwm_set_frequency(MCPWM_UNIT_0, MCPWM_TIMER_1, freq);
    mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, duty);
    mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_1, MCPWM_OPR_A, duty);
    // Ensure we're in normal PWM mode (not forced-low from a previous disable)
    mcpwm_set_duty_type(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, MCPWM_DUTY_MODE_0);
    mcpwm_set_duty_type(MCPWM_UNIT_0, MCPWM_TIMER_1, MCPWM_OPR_A, MCPWM_DUTY_MODE_0);
}

/** Enable or disable both output channels. Disable forces both outputs LOW. */
static void hw_enable(bool on) {
    if (on) {
        hw_freq_duty(g.freq, g.duty);
    } else {
        // Safe state: force both outputs low so transducers are not driven
        mcpwm_set_signal_low(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A);
        mcpwm_set_signal_low(MCPWM_UNIT_0, MCPWM_TIMER_1, MCPWM_OPR_A);
    }
}

/** One-time peripheral initialisation. */
static void hw_init() {
    // Bind GPIO pins to MCPWM outputs
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, PIN_CH_A);  // Timer 0 operator A
    mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM1A, PIN_CH_B);  // Timer 1 operator A

    mcpwm_config_t cfg = {
        .frequency    = g.freq,
        .cmpr_a       = g.duty,
        .cmpr_b       = 0.0f,
        .duty_mode    = MCPWM_DUTY_MODE_0,
        .counter_mode = MCPWM_UP_COUNTER
    };
    mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &cfg);
    mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_1, &cfg);

    // Timer 0 broadcasts a sync event each time its counter hits zero
    mcpwm_set_timer_sync_output(MCPWM_UNIT_0, MCPWM_TIMER_0,
                                MCPWM_SWSYNC_SOURCE_TEZ);
    // Apply initial phase to Timer 1
    hw_phase(g.phase);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Serial UI helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_status() {
    float deg = (g.phase / 1000.0f) * 360.0f;
    Serial.printf(
        "[STATUS] freq=%u Hz  phase=%u (%.1f\xC2\xB0)  duty=%.1f%%  output=%s%s\n",
        g.freq, g.phase, deg, g.duty,
        g.enabled ? "ON" : "OFF",
        sw.running ? "  [SWEEPING]" : ""
    );
}

static void print_help() {
    Serial.println();
    Serial.println(F("+-----------------------+---------------------------------+"));
    Serial.println(F("| Command               | Description                     |"));
    Serial.println(F("+-----------------------+---------------------------------+"));
    Serial.println(F("| F <Hz>                | Frequency (1000-80000 Hz)       |"));
    Serial.println(F("| P <0-999>             | Ch B phase offset               |"));
    Serial.println(F("|                       |   0=0deg  250=90deg  500=180deg |"));
    Serial.println(F("| D <1-99>              | Duty cycle (%)                  |"));
    Serial.println(F("| E <0|1>               | Disable / enable output         |"));
    Serial.println(F("| SWEEP <step> <ms>     | Non-blocking phase sweep        |"));
    Serial.println(F("|   e.g. SWEEP 5 20     |   step 1-100  delay 1-500 ms   |"));
    Serial.println(F("| STOP                  | Abort active sweep              |"));
    Serial.println(F("| S                     | Print status                    |"));
    Serial.println(F("| ?                     | Print this help                 |"));
    Serial.println(F("+-----------------------+---------------------------------+"));
}


// ─────────────────────────────────────────────────────────────────────────────
//  Command parser
// ─────────────────────────────────────────────────────────────────────────────

static void handle_cmd(const String &raw) {
    String s = raw;
    s.trim();
    if (!s.length()) return;

    // Echo the command back
    Serial.print(F(">> "));
    Serial.println(s);

    // Upper-case copy for keyword matching
    String u = s;
    u.toUpperCase();

    // ── Keyword commands ─────────────────────────────────────────────────

    if (u == "S") { print_status(); return; }
    if (u == "?") { print_help();   return; }

    if (u == "STOP") {
        if (sw.running) {
            sw.running = false;
            Serial.println(F("[OK] Sweep aborted"));
            print_status();
        } else {
            Serial.println(F("[INFO] No sweep active"));
        }
        return;
    }

    // ── SWEEP <step> <delay_ms> ──────────────────────────────────────────
    if (u.startsWith("SWEEP")) {
        String args = s.substring(5);   // text after "SWEEP"
        args.trim();
        int sp = args.indexOf(' ');
        if (sp < 1) {
            Serial.println(F("[ERR] Usage: SWEEP <step 1-100> <delay_ms 1-500>"));
            return;
        }
        int step = args.substring(0, sp).toInt();
        int dly  = args.substring(sp + 1).toInt();
        if (step < 1 || step > 100 || dly < 1 || dly > 500) {
            Serial.println(F("[ERR] step: 1-100   delay: 1-500 ms"));
            return;
        }
        sw.step     = (uint32_t)step;
        sw.delay_ms = (uint32_t)dly;
        sw.pos      = g.phase;    // begin from current phase position
        sw.last_ms  = millis();
        sw.running  = true;
        if (!g.enabled) {
            g.enabled = true;
            hw_enable(true);
        }
        Serial.printf("[OK] Sweep from phase=%u  step=%d  delay=%d ms  (STOP to abort)\n",
                      sw.pos, step, dly);
        return;
    }

    // ── Single-letter commands with numeric argument ───────────────────────

    char cmd = toupper(s.charAt(0));
    String arg = s.substring(1);
    arg.trim();

    // F — frequency
    if (cmd == 'F') {
        long f = arg.toInt();
        if (f < 1000 || f > 80000) {
            Serial.println(F("[ERR] Frequency must be 1000-80000 Hz"));
            return;
        }
        g.freq = (uint32_t)f;
        if (g.enabled) hw_freq_duty(g.freq, g.duty);
        // Re-apply phase — timer period changed so sync timing shifts
        hw_phase(g.phase);
        Serial.printf("[OK] freq=%u Hz\n", g.freq);
        return;
    }

    // P — phase
    if (cmd == 'P') {
        long p = arg.toInt();
        if (p < 0 || p > 999) {
            Serial.println(F("[ERR] Phase: 0-999 ticks  (500 = 180 deg)"));
            return;
        }
        g.phase    = (uint32_t)p;
        sw.running = false;   // manual phase overrides any active sweep
        hw_phase(g.phase);
        Serial.printf("[OK] phase=%u (%.1f deg)\n",
                      g.phase, g.phase / 1000.0f * 360.0f);
        return;
    }

    // D — duty cycle
    if (cmd == 'D') {
        float d = arg.toFloat();
        if (d < 1.0f || d > 99.0f) {
            Serial.println(F("[ERR] Duty: 1-99 %"));
            return;
        }
        g.duty = d;
        if (g.enabled) hw_freq_duty(g.freq, g.duty);
        Serial.printf("[OK] duty=%.1f%%\n", g.duty);
        return;
    }

    // E — enable / disable
    if (cmd == 'E') {
        g.enabled = (arg.toInt() != 0);
        if (!g.enabled) sw.running = false;   // kill sweep too
        hw_enable(g.enabled);
        Serial.printf("[OK] output=%s\n", g.enabled ? "ON" : "OFF");
        return;
    }

    Serial.printf("[ERR] Unknown command '%s'  (? for help)\n", s.c_str());
}


// ─────────────────────────────────────────────────────────────────────────────
//  Arduino entry points
// ─────────────────────────────────────────────────────────────────────────────

static String cmd_buf;

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println(F("\r\n=== Acoustic Particle Manipulator — ESP32 ==="));
    hw_init();
    print_status();
    print_help();
}

void loop() {
    // ── Serial command reader ─────────────────────────────────────────────
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            handle_cmd(cmd_buf);
            cmd_buf = "";
        } else if (isprint(c)) {
            cmd_buf += c;
        }
    }

    // ── Sweep state machine (non-blocking) ────────────────────────────────
    if (sw.running) {
        uint32_t now = millis();
        if (now - sw.last_ms >= sw.delay_ms) {
            sw.last_ms = now;

            // Advance and wrap
            sw.pos = (sw.pos + sw.step) % 1000;
            g.phase = sw.pos;
            hw_phase(g.phase);

            Serial.printf("[SWEEP] phase=%u (%.1f deg)\n",
                          g.phase, g.phase / 1000.0f * 360.0f);

            // Optional: uncomment to auto-stop at 0 each cycle
            // if (sw.pos < sw.step) { sw.running = false; Serial.println("[SWEEP] Cycle done"); }
        }
    }
}


/*
 * ═══════════════════════════════════════════════════════════════════════════
 *  PORTING TO arduino-esp32 3.x  (ESP-IDF 5.x)
 *  ─────────────────────────────────────────────────────────────────────────
 *  The legacy MCPWM driver used here was deprecated in IDF 5.0 and the
 *  legacy header (driver/mcpwm.h) may still compile with deprecation
 *  warnings, or may be absent.
 *
 *  For IDF 5.x use the new composable API:
 *    #include "driver/mcpwm_timer.h"
 *    #include "driver/mcpwm_oper.h"
 *    #include "driver/mcpwm_cmpr.h"
 *    #include "driver/mcpwm_gen.h"
 *    #include "driver/mcpwm_sync.h"
 *
 *  Key objects to create:
 *    mcpwm_new_timer()        → mcpwm_timer_handle_t  (×2)
 *    mcpwm_new_operator()     → mcpwm_oper_handle_t   (×2)
 *    mcpwm_new_comparator()   → mcpwm_cmpr_handle_t   (×2)
 *    mcpwm_new_generator()    → mcpwm_gen_handle_t    (×2)
 *    mcpwm_new_timer_sync_src()  → sync source on Timer 0 TEZ
 *    (use mcpwm_timer_sync_phase_config_t on Timer 1 to apply offset)
 *
 *  Phase offset in IDF 5.x:
 *    mcpwm_timer_sync_phase_config_t phase_cfg = {
 *        .sync_src   = sync_src_from_timer0,
 *        .count_value = phase_ticks,    // 0 to period_ticks
 *        .direction  = MCPWM_TIMER_DIRECTION_UP
 *    };
 *    mcpwm_timer_set_phase_on_sync(timer1, &phase_cfg);
 * ═══════════════════════════════════════════════════════════════════════════
 */
