# VOC-AQI Monitoring System with MQ Series Sensors

## Respiratory Disease Risk Prediction & Real-Time Air Quality Monitoring

A system that combines **CDC COPD epidemiological data** with **live MQ gas sensor readings** on a **Raspberry Pi Pico** to predict respiratory health risks. The project trains multiple ML models on both datasets, selects the best one, and exports its weights for on-device inference — no internet, no cloud, no external libraries needed on the microcontroller.

---

## Project Structure

```
├── Combined_Model_Training.ipynb    # End-to-end: data loading, training, evaluation, weight export
├── pico_mq_monitor.py               # MicroPython — live sensor reading + ML inference on Pico
├── pico_mlp_weights.py              # Auto-generated weights file (output of notebook)
├── mq_health_predictor.py           # Standalone PC trainer — 3 models, reports, CLI
├── export_weights_for_pico.py       # Standalone PC script — lightweight MLP weight export
├── data/
│   ├── MQ_Dataset.xlsx              # 210 readings from 7 MQ sensors (30 per sensor)
│   └── U.S._Chronic_Disease_Indicators__COPD.csv   # 309,215 CDC records
└── README.md
```

---

## Idea & Motivation

Air pollution is a leading environmental risk factor for chronic obstructive pulmonary disease (COPD). The CDC's Chronic Disease Indicators dataset shows that states with worse air quality have measurably higher COPD prevalence. MQ-series gas sensors (MQ135, MQ5, MQ4, etc.) can measure VOC, methane, LPG, and other gases in real time — but raw PPM readings alone don't tell you about health risk.

**This project bridges the gap:**

1. **Train on epidemiological data** — learn the relationship between air quality indicators and COPD prevalence across U.S. states using the CDC dataset.
2. **Train on sensor data** — learn gas concentration thresholds that correspond to dangerous conditions using MQ sensor readings.
3. **Combine both** — build an ensemble that uses COPD-informed risk thresholds to label sensor data, then trains a compact MLP that runs on a $4 microcontroller.
4. **Deploy to Pico** — export the best model's weights and run real-time inference on a Raspberry Pi Pico with zero dependencies.

The result is a portable, offline air quality monitor that understands health context — not just raw gas levels.

---

## How It Works

### Combined Model Training Notebook

`Combined_Model_Training.ipynb` is the single source of truth for the entire pipeline:

#### Step 1 — Load & Explore Both Datasets
- Loads CDC COPD data (309K rows) and MQ sensor readings (210 rows)
- Computes COPD prevalence by state, maps to geographic coordinates
- Pivots MQ sensor data into a feature matrix (MQ135, MQ2, MQ4, MQ5, MQ8, MQ9 PPM)

#### Step 2 — Train Individual Models

| Model | Dataset | Architecture | Test Accuracy | Purpose |
|-------|---------|-------------|---------------|---------|
| **COPD MLP** | CDC | 32→16 neurons | ~67% | COPD risk from geographic/demographic features |
| **VOC Gradient Boosting** | MQ sensors | 50 stages, lr=0.1 | ~100% | VOC danger from MQ135 readings |
| **Multi-Sensor RF** | MQ sensors | 100 trees | ~98% | Compound danger from all 6 sensors |

#### Step 3 — Pico-Compatible MLP
- Trains a lightweight MLP: **input(3) → 16 → 8 → 1** on MQ135 + MQ5 + MQ4 PPM
- Uses COPD-informed thresholds to create health risk labels
- Early stopping with validation monitoring
- **Architecture: 209 parameters (~0.8 KB)** — fits in Pico flash

#### Step 4 — Ensemble & Model Selection
- Stacking ensemble (logistic regression meta-learner on top of all models)
- Compares all models head-to-head and selects the best by cross-validated accuracy
- Exports winner's weights to `pico_mlp_weights.py`

#### Step 5 — Export & Verify
- Writes weights, biases, scaler parameters, and sensor order to Python file
- Runs verification: forward pass on test cases confirming inference matches training
- Generates confusion matrix and feature importance visualizations

### Run on Raspberry Pi Pico

`pico_mq_monitor.py` runs on stock MicroPython with **no external libraries**:

- Reads raw ADC from 3 MQ sensors every 5 seconds
- Converts to PPM using datasheet calibration curves
- Imports weights from `pico_mlp_weights.py`
- **Runs the trained MLP forward pass** (pure math — ReLU activation, matrix multiply)
- Classifies air quality as SAFE / ELEVATED / WARNING / DANGER
- Outputs risk score with `[ML]` tag confirming neural network inference
- Auto-calibrates R0 in clean air on first boot
- LED alert on danger detection

---

## Hardware Setup

### Wiring (Raspberry Pi Pico)

| Sensor | Pico Pin | ADC Channel | Detects |
|--------|----------|-------------|---------|
| MQ135 AOUT | GP26 | ADC0 | VOC / NH3 / NOx / Benzene |
| MQ5 AOUT | GP27 | ADC1 | LPG / Natural Gas |
| MQ4 AOUT | GP28 | ADC2 | Methane / CNG |
| All VCC | **VBUS** (5V) | — | Heater power (not 3V3!) |
| All GND | GND | — | Common ground |

> **Only GP26, GP27, GP28 support analog input.** GP0–GP22 are digital-only.

### First Run (Calibration)

1. Flash MicroPython onto the Pico
2. Upload `pico_mq_monitor.py` as `main.py` and `pico_mlp_weights.py` alongside it
3. Place sensors in **clean outdoor air**
4. Run — auto-calibration samples for 30 seconds, saves R0 values to `calibration.json`
5. After calibration, set `CALIBRATE_ON_BOOT = False` in the script

---

## MQ Sensors — What They Detect

| Sensor | Target Gases | Health Threshold (safe / warning / danger) |
|--------|-------------|---------------------------------------------|
| **MQ-135** | NH3, NOx, Benzene, CO2, Smoke | 150 / 350 / 500 ppm |
| **MQ-4** | Methane (CH4), CNG | 25 / 45 / 70 ppm |
| **MQ-5** | LPG, Natural Gas | 7 / 12 / 18 ppm |
| MQ-2 | LPG, Smoke, Methane, H2 | 4 / 7 / 10 ppm |
| MQ-8 | Hydrogen (H2) | 40 / 150 / 400 ppm |
| MQ-9 | Carbon Monoxide (CO) | 9 / 25 / 50 ppm (OSHA PEL=50) |

Thresholds based on OSHA PEL, WHO, and EPA guidelines.

---

## Datasets

| Dataset | Source | Size |
|---------|--------|------|
| CDC Chronic Disease Indicators (COPD) | [data.cdc.gov](https://data.cdc.gov/resource/hksd-2xuw.csv) | 309,215 rows × 34 cols |
| MQ Sensor Readings | Raspberry Pi + MQ breakout boards | 210 rows (7 sensors × 30 readings) |

---

## Requirements

### PC (training)

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

### Pico (inference)

Stock MicroPython — no packages needed. The neural network runs as pure arithmetic.

---

## Quick Start

```bash
# Option 1: Run the notebook (recommended)
# Open Combined_Model_Training.ipynb and Run All
# → produces pico_mlp_weights.py

# Option 2: Standalone scripts
python export_weights_for_pico.py   # train + export weights
python mq_health_predictor.py       # full analysis with plots

# Deploy to Pico (via Thonny)
# Upload pico_mq_monitor.py as main.py
# Upload pico_mlp_weights.py alongside it
```

---

## Sample Pico Output

```
==================================================
 MQ SENSOR HEALTH MONITOR — Pico
==================================================
  Wiring check:
    MQ135 (GP26) : raw=14500 V=0.730V -- OK
    MQ5   (GP27) : raw=13200 V=0.665V -- OK
    MQ4   (GP28) : raw=13800 V=0.695V -- OK

--------------------------------------------------
Reading #1
--------------------------------------------------
  MQ135 | 24.3 ppm (avg:24.3) | SAFE | VOC/NH3/NOx/Benzene
  MQ5   |  1.2 ppm (avg:1.2)  | SAFE | LPG/NaturalGas
  MQ4   |  8.1 ppm (avg:8.1)  | SAFE | Methane/CNG

  Risk Score: 8% -> LOW RISK [ML]
  VERDICT: OK — Air quality acceptable
```
