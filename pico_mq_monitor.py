"""
MQ Sensor Health Monitor — Raspberry Pi Pico (MicroPython)
===========================================================
Reads raw ADC values from MQ gas sensors, converts to PPM,
classifies air quality, and reports health risks over serial (USB/UART).

No external libraries needed — runs on stock MicroPython.

Wiring:
  MQ135 AOUT → GP26 (ADC0) — VOC / Air Quality
  MQ5   AOUT → GP27 (ADC1) — LPG / Natural Gas
  MQ4   AOUT → GP28 (ADC2) — Methane / CNG

  Sensor VCC  → Pico VBUS (5V from USB) — NOT 3V3!
  Sensor GND  → Pico GND

IMPORTANT:
  - Only GP26, GP27, GP28 can read analog. GP0-GP22 are digital-only!
  - MQ sensors need 5V heater power — connect VCC to VBUS pin.
  - Breakout boards have a built-in load resistor (RL ~10kOhm).
    No extra resistor needed if using a module with 4 pins.
  - FIRST RUN: Keep CALIBRATE_ON_BOOT = True, run in CLEAN outdoor air.
    After calibration saves, set CALIBRATE_ON_BOOT = False.

Usage:
  1. Flash MicroPython onto your Pico
  2. Copy this file to the Pico as main.py
  3. Open serial monitor (115200 baud)
  4. On first run, calibrate in clean outdoor air
  5. Readings print every INTERVAL_SEC seconds
"""

from machine import ADC, Pin
import time
import math
import json

# ============================================================================
# CONFIGURATION — EDIT THESE FOR YOUR SETUP
# ============================================================================

# ADC pin assignments: sensor_name -> GPIO pin number
# Pico ADC pins: GP26 (ADC0), GP27 (ADC1), GP28 (ADC2)
# IMPORTANT: Only GP26, GP27, GP28 can read analog! Other GPIOs are digital-only.
SENSOR_PINS = {
    "MQ135": 26,   # ADC0 — GP26 — VOC / Air Quality
    "MQ5":   27,   # ADC1 — GP27 — LPG / Natural Gas
    "MQ4":   28,   # ADC2 — GP28 — Methane / CNG
}

# Multiplexer control pins (set to None if not using a mux)
# For CD4051: 3 select pins → 8 channels
MUX_PINS = None  # e.g., (10, 11, 12) for GP10, GP11, GP12
# Map sensor to mux channel (only needed if using mux)
MUX_CHANNELS = {
    # "MQ4": 1,
    # "MQ5": 2,
    # "MQ6": 3,
    # "MQ8": 4,
}

# Circuit parameters
VCC = 3.3              # Pico runs at 3.3V (NOT 5V like Arduino)
RL = 10.0              # Load resistor in kOhm (on your MQ breakout board)
ADC_RESOLUTION = 65535  # Pico ADC is 16-bit

# How often to read sensors (seconds)
INTERVAL_SEC = 5

# CALIBRATION: Set True on FIRST RUN in clean air, then set to False
CALIBRATE_ON_BOOT = True
CALIBRATION_SECONDS = 30  # How long to sample during calibration

# Number of ADC samples to average per reading (reduces noise)
SAMPLES_PER_READ = 50

# LED pin for danger alert (onboard LED)
LED_PIN = 25  # Pico onboard LED

# ============================================================================
# SENSOR CALIBRATION DATA
# ============================================================================
# R0 values — you MUST calibrate these in clean air!
# Procedure:
#   1. Power sensors for 24-48 hours (burn-in / preheat)
#   2. Place in clean outdoor air
#   3. Read raw ADC, compute Rs = RL * (VCC - Vsensor) / Vsensor
#   4. Divide Rs by the "clean air ratio" from datasheet:
#      R0 = Rs / clean_air_ratio
#   5. Record R0 below
#
# The values below are DEFAULTS — replace with your measured values!
SENSOR_CALIBRATION = {
    "MQ135": {"r0": 76.63,  "clean_air_ratio": 3.6},
    "MQ2":   {"r0": 9.83,   "clean_air_ratio": 9.8},
    "MQ4":   {"r0": 11.47,  "clean_air_ratio": 4.4},
    "MQ5":   {"r0": 6.5,    "clean_air_ratio": 6.5},
    "MQ6":   {"r0": 10.0,   "clean_air_ratio": 10.0},
    "MQ8":   {"r0": 1.0,    "clean_air_ratio": 70.0},
    "MQ9":   {"r0": 9.6,    "clean_air_ratio": 9.6},
}

# ============================================================================
# PPM CONVERSION — Log-log curves from datasheets
# ============================================================================
# Each sensor has a power-law relationship: PPM = a * (Rs/R0)^b
# These coefficients come from fitting the datasheet sensitivity curves.
# Different gases have different curves; we use the primary target gas.
PPM_CURVES = {
    # MQ135: NH3 / general air quality curve
    "MQ135": {"a": 116.602, "b": -2.769},
    # MQ2: LPG / smoke curve
    "MQ2":   {"a": 574.25,  "b": -2.222},
    # MQ4: Methane (CH4) curve
    "MQ4":   {"a": 1012.7,  "b": -2.786},
    # MQ5: LPG curve
    "MQ5":   {"a": 1163.8,  "b": -3.874},
    # MQ6: LPG / propane curve
    "MQ6":   {"a": 1009.2,  "b": -2.35},
    # MQ8: Hydrogen (H2) curve
    "MQ8":   {"a": 976.97,  "b": -0.688},
    # MQ9: CO curve
    "MQ9":   {"a": 599.65,  "b": -2.244},
}

# ============================================================================
# HEALTH THRESHOLDS (OSHA/WHO/EPA)
# ============================================================================
# (safe_ppm, warning_ppm, danger_ppm)
THRESHOLDS = {
    "MQ135": (150, 350, 500),    # General VOC / air quality
    "MQ2":   (4, 7, 10),         # Combustible gas / smoke
    "MQ4":   (25, 45, 70),       # Methane
    "MQ5":   (7, 12, 18),        # LPG / natural gas
    "MQ6":   (4, 7, 10),         # Isobutane / LPG
    "MQ8":   (40, 150, 400),     # Hydrogen
    "MQ9":   (9, 25, 50),        # Carbon monoxide (OSHA PEL=50)
}

# What each sensor detects (short labels for serial output)
DETECTS = {
    "MQ135": "VOC/NH3/NOx/Benzene",
    "MQ2":   "Smoke/CH4/LPG/H2",
    "MQ4":   "Methane/CNG",
    "MQ5":   "LPG/NaturalGas",
    "MQ6":   "LPG/Isobutane",
    "MQ8":   "Hydrogen",
    "MQ9":   "CO/CH4/LPG",
}

# ============================================================================
# MLP NEURAL NETWORK WEIGHTS (exported from PC training)
# ============================================================================
# To use the trained air quality model on the Pico:
#   1. Run: python export_weights_for_pico.py
#   2. Copy the printed MLP_SCALER_MEAN, MLP_SCALER_SCALE, and MLP_WEIGHTS
#      and paste them below, replacing the None values.
#
# The MLP takes 3 sensor PPM inputs (MQ135, MQ5, MQ4) and predicts
# SAFE vs DANGEROUS air quality based on OSHA/WHO thresholds.
#
# Architecture: input(3) -> dense(16) -> dense(8) -> output(2)
# Set to None to use only threshold-based classification (no ML).

MLP_SCALER_MEAN = [384.636822, 10.298266, 39.730507]
MLP_SCALER_SCALE = [230.923964, 5.401375, 21.7602]

MLP_WEIGHTS = [
    # Layer 1 (3 -> 16)
    (
        [[-0.3241, -0.1834, -0.3627], [0.3235, -0.0109, 0.322], [0.4422, 0.0754, 0.5277], [0.2424, -0.1513, 0.5223], [-0.2653, 0.3145, -0.0826], [-0.1994, -0.4202, -0.5126], [-0.5631, -0.1334, 0.258], [0.4197, -0.0041, -0.1157], [-0.0765, -0.1543, -0.4897], [0.4161, 0.5081, 0.0957], [-0.6046, -0.4417, -0.5683], [0.6975, 0.1517, 0.649], [0.2362, 0.0562, -0.2399], [-0.378, -0.532, -0.0075], [-0.1749, 0.3044, -0.213], [-0.4493, -0.36, 0.2099]],
        [-0.1296, -0.5398, 0.6964, 0.4487, 0.6812, 0.63, 0.1625, 0.5114, -0.4549, -0.1531, -0.6236, -0.0097, -0.2267, -0.3372, 0.5588, 0.0206]
    ),
    # Layer 2 (16 -> 8)
    (
        [[-0.2163, -0.3086, 0.5529, 0.5769, 0.2106, 0.5874, 0.6094, 0.4968, -0.6005, 0.2071, -0.3779, -0.1665, 0.0571, -0.5025, 0.3784, -0.1563], [-0.0378, 0.1644, -0.0179, -0.159, -0.2001, -0.3671, 0.1736, 0.2383, -0.0449, 0.056, -0.4246, -0.1426, -0.0096, -0.0131, -0.2406, 0.0425], [-0.2739, 0.138, -0.142, -0.324, -0.3993, -0.0588, 0.0906, -0.1521, -0.0539, -0.1155, 0.0731, 0.3808, 0.0856, 0.1242, 0.342, 0.2381], [0.2249, 0.1649, -0.3417, 0.151, -0.3136, 0.1857, 0.2847, -0.3117, -0.205, 0.3731, -0.0025, -0.1885, 0.0224, -0.3953, -0.2876, 0.0319], [-0.2567, 0.4571, 0.0005, 0.4503, -0.2785, -0.0829, 0.4926, -0.0834, -0.2754, 0.6492, -0.2996, 0.3605, -0.222, 0.1971, 0.0311, 0.2163], [0.3665, -0.434, -0.2049, -0.0084, 0.0487, -0.4219, -0.3216, -0.1155, -0.158, -0.2574, -0.1857, 0.1538, 0.2245, -0.2196, -0.399, -0.208], [0.2776, -0.2932, 0.04, 0.0815, -0.3689, -0.3114, 0.2449, 0.1297, 0.5506, -0.1911, 0.5713, -0.4509, -0.3672, 0.1894, 0.2429, -0.4173], [-0.2295, -0.1982, 0.3274, 0.1834, 0.1968, -0.1553, 0.2219, 0.5501, -0.2324, -0.0108, -0.3428, 0.4169, -0.1242, -0.3137, 0.5652, 0.5365]],
        [0.5895, -0.0098, -0.1822, -0.1568, 0.4155, 0.3246, 0.2023, 0.4691]
    ),
    # Layer 3 (8 -> 1) — single output, use sigmoid
    (
        [[0.4212, -0.5144, -0.4423, 0.5371, 0.3628, -0.657, -0.544, 0.4567]],
        [-0.6192]
    ),
]
# Model: input(3) -> 16 -> 8 -> 1  |  97.2% acc  |  98.9% CV
# Trained on MQ135/MQ5/MQ4 PPM -> SAFE vs DANGEROUS (OSHA/WHO thresholds)

# Order of sensor inputs the MLP expects
MLP_SENSOR_ORDER = ["MQ135", "MQ5", "MQ4"]


# ============================================================================
# CALIBRATION FUNCTIONS
# ============================================================================

def load_calibration():
    """Load saved R0 calibration from file, if it exists."""
    try:
        with open("calibration.json", "r") as f:
            cal = json.loads(f.read())
        for sensor, r0_val in cal.items():
            if sensor in SENSOR_CALIBRATION:
                SENSOR_CALIBRATION[sensor]["r0"] = r0_val
        print("  Loaded saved calibration from calibration.json")
        for sensor in SENSOR_PINS:
            print("    {} R0 = {:.2f}".format(sensor, SENSOR_CALIBRATION[sensor]["r0"]))
        return True
    except OSError:
        return False


def calibrate_sensors(adcs, duration_sec=CALIBRATION_SECONDS):
    """
    Calibrate R0 for each sensor by reading in CLEAN AIR.
    Must be run outdoors or in well-ventilated area with AQI < 50.
    Saves results to calibration.json on the Pico filesystem.
    """
    print("\n" + "=" * 50)
    print(" AUTO-CALIBRATION MODE")
    print("=" * 50)
    print(" Ensure sensors are in CLEAN outdoor air!")
    print(" Sampling for {} seconds...".format(duration_sec))
    print(" (LED will blink during calibration)\n")

    led = Pin(LED_PIN, Pin.OUT)
    rs_accum = {s: [] for s in adcs}

    start = time.ticks_ms()
    n = 0
    while time.ticks_diff(time.ticks_ms(), start) < duration_sec * 1000:
        n += 1
        led.value(n % 2)  # blink LED
        for sensor in adcs:
            raw = 0
            for _ in range(SAMPLES_PER_READ):
                raw += adcs[sensor].read_u16()
                time.sleep_us(100)
            raw /= SAMPLES_PER_READ
            voltage = (raw / ADC_RESOLUTION) * VCC
            if voltage > 0.05:
                rs = RL * (VCC - voltage) / voltage
                rs_accum[sensor].append(rs)
        # Progress
        elapsed = time.ticks_diff(time.ticks_ms(), start) // 1000
        print("  ... {} / {} sec ({} samples)".format(elapsed, duration_sec, n))
        time.sleep(1)

    led.value(0)

    # Compute R0 = avg(Rs) / clean_air_ratio
    cal_data = {}
    print("\n  Calibration Results:")
    print("  " + "-" * 40)
    for sensor in adcs:
        samples = rs_accum[sensor]
        if len(samples) == 0:
            print("  {} : NO VALID READINGS -- check wiring!".format(sensor))
            continue
        avg_rs = sum(samples) / len(samples)
        clean_ratio = SENSOR_CALIBRATION[sensor]["clean_air_ratio"]
        r0 = avg_rs / clean_ratio
        SENSOR_CALIBRATION[sensor]["r0"] = r0
        cal_data[sensor] = r0
        print("  {} : R0 = {:.2f}  (avg Rs = {:.2f}, {} samples)".format(
            sensor, r0, avg_rs, len(samples)))

    # Save to file
    try:
        with open("calibration.json", "w") as f:
            f.write(json.dumps(cal_data))
        print("\n  Saved to calibration.json")
        print("  >> Set CALIBRATE_ON_BOOT = False to skip next time <<")
    except Exception as e:
        print("\n  WARNING: Could not save: {}".format(e))
        print("  Copy the R0 values above into SENSOR_CALIBRATION manually.")

    print("=" * 50 + "\n")


def check_wiring(adcs):
    """Quick sanity check: warn if ADC reads near 0 or max (not connected)."""
    print("\n  Wiring check:")
    all_ok = True
    for sensor in adcs:
        raw = adcs[sensor].read_u16()
        voltage = (raw / ADC_RESOLUTION) * VCC
        if raw < 500:
            print("    {} (GP{}) : raw={} V={:.3f}V -- VERY LOW, not connected?".format(
                sensor, SENSOR_PINS[sensor], raw, voltage))
            all_ok = False
        elif raw > 64000:
            print("    {} (GP{}) : raw={} V={:.3f}V -- NEAR MAX, check wiring!".format(
                sensor, SENSOR_PINS[sensor], raw, voltage))
            all_ok = False
        else:
            print("    {} (GP{}) : raw={} V={:.3f}V -- OK".format(
                sensor, SENSOR_PINS[sensor], raw, voltage))
    if all_ok:
        print("  All sensors responding.\n")
    else:
        print("  WARNING: Some sensors may have wiring issues!\n")


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def read_adc_avg(adc, n_samples=SAMPLES_PER_READ):
    """Read ADC multiple times and return average (reduces noise)."""
    total = 0
    for _ in range(n_samples):
        total += adc.read_u16()
        time.sleep_us(100)
    return total / n_samples


def set_mux_channel(channel):
    """Set multiplexer select pins to the given channel number."""
    if MUX_PINS is None:
        return
    for i, pin_num in enumerate(MUX_PINS):
        pin = Pin(pin_num, Pin.OUT)
        pin.value((channel >> i) & 1)
    time.sleep_ms(5)  # settling time


def raw_to_ppm(raw_adc, sensor_name):
    """Convert raw 16-bit ADC value to PPM using sensor calibration curve."""
    if raw_adc <= 0:
        return 0.0

    # ADC value → voltage
    voltage = (raw_adc / ADC_RESOLUTION) * VCC

    if voltage <= 0.01:
        return 0.0

    # Voltage → sensor resistance (Rs)
    # Voltage divider: Vsensor = VCC * RL / (Rs + RL)
    # → Rs = RL * (VCC - Vsensor) / Vsensor
    rs = RL * (VCC - voltage) / voltage

    # Rs/R0 ratio
    cal = SENSOR_CALIBRATION.get(sensor_name)
    if cal is None:
        return 0.0
    r0 = cal["r0"]
    if r0 <= 0:
        return 0.0
    ratio = rs / r0

    # Ratio → PPM using power law: PPM = a * ratio^b
    curve = PPM_CURVES.get(sensor_name)
    if curve is None or ratio <= 0:
        return 0.0

    try:
        ppm = curve["a"] * math.pow(ratio, curve["b"])
    except (ValueError, OverflowError):
        ppm = 0.0

    return max(0.0, ppm)


def classify(ppm, sensor_name):
    """Classify a PPM reading as SAFE / WARNING / DANGER."""
    safe, warn, danger = THRESHOLDS.get(sensor_name, (999999, 999999, 999999))
    if ppm >= danger:
        return "DANGER"
    elif ppm >= warn:
        return "WARNING"
    elif ppm >= safe:
        return "ELEVATED"
    return "SAFE"


def relu(x):
    """ReLU activation for MLP forward pass."""
    return max(0.0, x)


def mlp_forward(inputs, weights):
    """
    Run MLP forward pass in pure MicroPython.
    weights = list of (W_matrix, b_vector) tuples per layer.
    For binary classification with single output: returns danger probability.
    For multi-class with 2+ outputs: returns list of class probabilities.
    """
    x = inputs
    for i, (W, b) in enumerate(weights):
        y = []
        for j in range(len(b)):
            s = b[j]
            for k in range(len(x)):
                s += W[j][k] * x[k]
            # ReLU for hidden layers, raw for output
            if i < len(weights) - 1:
                s = relu(s)
            y.append(s)
        x = y

    # Single output neuron -> sigmoid (binary classification)
    if len(x) == 1:
        logit = x[0]
        # Clip to avoid overflow
        if logit > 20:
            prob = 1.0
        elif logit < -20:
            prob = 0.0
        else:
            prob = 1.0 / (1.0 + math.exp(-logit))
        return prob  # probability of "dangerous" class

    # Multiple output neurons -> softmax
    max_x = max(x)
    exps = [math.exp(v - max_x) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


def estimate_copd_risk(readings):
    """
    Estimate respiratory / COPD risk from sensor readings.
    If MLP weights are loaded, uses the trained neural network.
    Otherwise, falls back to rule-based scoring.
    Returns (risk_percent, risk_label, method_used).
    """
    ml_result = None

    # ── Try MLP inference first ──
    if MLP_WEIGHTS is not None and MLP_SCALER_MEAN is not None:
        # Build input vector in the order the model expects
        inputs = []
        for sensor in MLP_SENSOR_ORDER:
            inputs.append(readings.get(sensor, 0.0))

        # StandardScaler transform: (x - mean) / scale
        scaled = []
        for i in range(len(inputs)):
            scaled.append((inputs[i] - MLP_SCALER_MEAN[i]) / MLP_SCALER_SCALE[i])

        # Forward pass through MLP
        result = mlp_forward(scaled, MLP_WEIGHTS)

        # Single output = sigmoid probability of "dangerous"
        if isinstance(result, float):
            danger_prob = result
        else:
            danger_prob = result[1]  # softmax: index 1 = dangerous

        danger_pct = danger_prob * 100

        if danger_pct >= 70:
            ml_label = "HIGH RISK"
        elif danger_pct >= 40:
            ml_label = "MODERATE"
        else:
            ml_label = "LOW RISK"

        ml_result = (danger_pct, ml_label, "ML")

    # ── Rule-based scoring (always computed for comparison) ──
    score = 0
    max_score = 0

    if "MQ135" in readings:
        ppm = readings["MQ135"]
        max_score += 30
        if ppm >= 500:
            score += 30
        elif ppm >= 350:
            score += 20
        elif ppm >= 150:
            score += 10

    if "MQ9" in readings:
        ppm = readings["MQ9"]
        max_score += 25
        if ppm >= 50:
            score += 25
        elif ppm >= 25:
            score += 15
        elif ppm >= 9:
            score += 5

    if "MQ2" in readings:
        ppm = readings["MQ2"]
        max_score += 20
        if ppm >= 10:
            score += 20
        elif ppm >= 7:
            score += 12
        elif ppm >= 4:
            score += 5

    if "MQ4" in readings:
        ppm = readings["MQ4"]
        max_score += 10
        if ppm >= 70:
            score += 10
        elif ppm >= 45:
            score += 5

    if "MQ5" in readings:
        ppm = readings["MQ5"]
        max_score += 15
        if ppm >= 18:
            score += 15
        elif ppm >= 12:
            score += 8
        elif ppm >= 7:
            score += 3

    if "MQ8" in readings:
        ppm = readings["MQ8"]
        max_score += 15
        if ppm >= 400:
            score += 15
        elif ppm >= 150:
            score += 8
        elif ppm >= 40:
            score += 3

    if max_score == 0:
        if ml_result:
            return ml_result
        return 0, "NO DATA", "NONE"

    rule_pct = (score / max_score) * 100
    if rule_pct >= 60:
        rule_label = "HIGH RISK"
    elif rule_pct >= 30:
        rule_label = "MODERATE"
    else:
        rule_label = "LOW RISK"

    # If ML is available, use it as primary and show both
    if ml_result:
        return ml_result
    return rule_pct, rule_label, "RULES"


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("=" * 50)
    print(" MQ SENSOR HEALTH MONITOR — Pico")
    print(" Initializing sensors...")
    print("=" * 50)

    # Initialize ADC objects
    adcs = {}
    for sensor, pin in SENSOR_PINS.items():
        adcs[sensor] = ADC(Pin(pin))
        print("  {} -> GP{} (ADC)".format(sensor, pin))

    # LED for danger indicator
    led = Pin(LED_PIN, Pin.OUT)
    led.value(0)

    # Quick wiring sanity check
    check_wiring(adcs)

    # Preheat warning
    print("  WARNING: MQ sensors need 2-5 min preheat for stable readings.")
    print("  First readings may be inaccurate.\n")

    # Load saved calibration, or run auto-calibration in clean air
    if not load_calibration():
        if CALIBRATE_ON_BOOT:
            print("  No saved calibration found. Running auto-calibration...")
            print("  Make sure you are in CLEAN AIR!")
            calibrate_sensors(adcs)
        else:
            print("  WARNING: No calibration! Using default R0 values.")
            print("  PPM readings will be INACCURATE.")
            print("  Set CALIBRATE_ON_BOOT = True and re-run in clean air.\n")

    # Rolling averages for stability (last 10 readings per sensor)
    history = {s: [] for s in SENSOR_PINS}
    HISTORY_LEN = 10

    reading_num = 0
    while True:
        reading_num += 1
        readings = {}
        levels = {}
        any_danger = False
        any_warning = False

        print("-" * 50)
        print("Reading #{} | {}".format(reading_num, time.ticks_ms()))
        print("-" * 50)

        for sensor in SENSOR_PINS:
            # Set mux channel if needed
            if sensor in MUX_CHANNELS:
                set_mux_channel(MUX_CHANNELS[sensor])

            # Read raw ADC
            raw = read_adc_avg(adcs[sensor])
            voltage = (raw / ADC_RESOLUTION) * VCC

            # Convert to PPM
            ppm = raw_to_ppm(raw, sensor)

            # Rolling average
            history[sensor].append(ppm)
            if len(history[sensor]) > HISTORY_LEN:
                history[sensor].pop(0)
            avg_ppm = sum(history[sensor]) / len(history[sensor])

            # Classify
            level = classify(avg_ppm, sensor)
            readings[sensor] = avg_ppm
            levels[sensor] = level

            if level == "DANGER":
                any_danger = True
            elif level == "WARNING":
                any_warning = True

            # Print sensor reading
            safe, warn, danger = THRESHOLDS.get(sensor, (0, 0, 0))
            print("  {} | raw={:.0f} V={:.3f}V | {:.1f} ppm (avg:{:.1f}) | {} | {}".format(
                sensor, raw, voltage, ppm, avg_ppm, level, DETECTS.get(sensor, "")
            ))
            if level == "DANGER":
                print("    *** DANGER: {:.1f} ppm exceeds {:.0f} ppm limit! ***".format(avg_ppm, danger))
            elif level == "WARNING":
                print("    * Warning: {:.1f} ppm above {:.0f} ppm threshold".format(avg_ppm, warn))

        # COPD risk assessment
        risk_pct, risk_label, risk_method = estimate_copd_risk(readings)
        print("\n  Risk Score: {:.0f}% -> {} [{}]".format(risk_pct, risk_label, risk_method))

        # Overall verdict
        if any_danger:
            verdict = "!!! DANGEROUS — leave area / ventilate immediately !!!"
            led.value(1)
        elif any_warning:
            verdict = "!! WARNING — improve ventilation, limit exposure !!"
            # Blink LED for warning
            led.value(1)
            time.sleep_ms(200)
            led.value(0)
        else:
            verdict = "OK — Air quality acceptable"
            led.value(0)

        print("  VERDICT: {}".format(verdict))

        # JSON output for apps/logging (one line, easy to parse)
        json_parts = []
        for s, ppm in readings.items():
            json_parts.append('"{}": {{"ppm": {:.1f}, "level": "{}"}}'.format(s, ppm, levels[s]))
        print('JSON:{{"reading": {}, "risk": "{}", "risk_pct": {:.0f}, {}}}'.format(
            reading_num, risk_label, risk_pct, ", ".join(json_parts)
        ))

        print()
        time.sleep(INTERVAL_SEC)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
