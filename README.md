# VOC-AQI Monitoring System with MQ Series Sensors

## Respiratory Disease Risk Prediction & Real-Time Air Quality Monitoring

A system that combines CDC COPD epidemiological data with live MQ gas sensor readings on a **Raspberry Pi Pico** to predict respiratory health risks using a trained neural network.

---

## Project Structure

```
├── pico_mq_monitor.py              # Pico MicroPython — live sensor reading + ML inference
├── mq_health_predictor.py           # PC — trains 3 ML models, analyzes MQ data, generates reports
├── export_weights_for_pico.py       # PC — trains lightweight MLP, exports weights for Pico
├── COPD_and_Low_Air_Quality.ipynb   # Jupyter Notebook — EDA, causal inference, model comparison
├── data/
│   ├── MQ_Dataset.xlsx                                              # 210 readings from 7 MQ sensors
│   └── U.S._Chronic_Disease_Indicators__COPD.csv                   # 309,215 CDC records
└── README.md
```

---

## How It Works

### 1. Train on PC

`mq_health_predictor.py` trains three models on CDC + MQ sensor data:

| Model | Architecture | Accuracy | Purpose |
|-------|-------------|----------|---------|
| **MLP Neural Network** | 64 → 32 → 16 neurons, 50 epochs | 67.2% (CV: 77.4%) | COPD risk from geographic/demographic features |
| **Gradient Boosting** | 100 stages, lr=0.05 | 100% (AUC: 1.0) | VOC risk from MQ135 readings |
| **Random Forest** | 200 trees, max_depth=8 | 98.4% | Multi-sensor compound danger from all MQ sensors |

### 2. Export Weights

`export_weights_for_pico.py` trains a lightweight MLP on sensor PPM data and exports the weights:

- **Architecture:** input(3) → 16 → 8 → 1 (sigmoid)
- **Inputs:** MQ135, MQ5, MQ4 PPM values
- **Output:** Probability of dangerous air quality
- **Accuracy:** 97.2% test, 98.9% cross-validated
- **Size:** 209 parameters (~0.8 KB) — fits in Pico flash

### 3. Run on Raspberry Pi Pico

`pico_mq_monitor.py` runs on stock MicroPython with **no external libraries**:

- Reads raw ADC from 3 MQ sensors every 5 seconds
- Converts to PPM using datasheet calibration curves
- **Runs the trained MLP forward pass** (pure math, no scikit-learn)
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
2. Upload `pico_mq_monitor.py` as `main.py`
3. Place sensors in **clean outdoor air**
4. Run — auto-calibration samples for 30 seconds, saves R0 values to `calibration.json`
5. After calibration, set `CALIBRATE_ON_BOOT = False` in the script

---

## MQ Sensors — What They Detect

| Sensor | Target Gases | Health Threshold (safe / warning / danger) |
|--------|-------------|-----------------|
| **MQ-135** | NH3, NOx, Benzene, CO2, Smoke | 150 / 350 / 500 ppm |
| **MQ-4** | Methane (CH4), CNG | 25 / 45 / 70 ppm |
| **MQ-5** | LPG, Natural Gas | 7 / 12 / 18 ppm |
| MQ-2 | LPG, Smoke, Methane, H2 | 4 / 7 / 10 ppm |
| MQ-8 | Hydrogen (H2) | 40 / 150 / 400 ppm |
| MQ-9 | Carbon Monoxide (CO) | 9 / 25 / 50 ppm (OSHA PEL=50) |

Thresholds based on OSHA PEL, WHO, and EPA guidelines.

---

## COPD Analysis (Notebook)

The Jupyter Notebook explores whether air pollution causes COPD using CDC data:

- **Causal Inference:** IPW estimate shows poor air quality causes **6.83% higher COPD prevalence**
- **Predictive Models:** Geographic location (lat/lon) is the strongest predictor of COPD occurrence
- **Key Finding:** Racial features have minimal predictive power — geography alone suffices

> If the notebook fails to render on GitHub, use [nbviewer](https://nbviewer.org/github/raachelssss/COPD-and-Air-Quality/blob/main/COPD_and_Low_Air_Quality.ipynb).

### Model Performance (Notebook)

| Model | Test Accuracy | ROC-AUC |
|-------|--------------|---------|
| **Gradient Boosting** | **87.6%** | **0.944** |
| Random Forest | 86.0% | 0.924 |
| KNN | 76.0% | 0.830 |
| Neural Network (MLP) | 71.1% | 0.806 |
| SVM (RBF) | 70.2% | 0.787 |

---

## Datasets

| Dataset | Source | Size |
|---------|--------|------|
| CDC Chronic Disease Indicators (COPD) | [data.cdc.gov](https://data.cdc.gov/resource/hksd-2xuw.csv) | 309,215 rows x 34 cols |
| MQ Sensor Readings | Raspberry Pi + MQ breakout boards | 210 rows (7 sensors x 30 readings) |

---

## Requirements

### PC (training)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Pico (inference)

Stock MicroPython — no packages needed. The neural network runs as pure arithmetic.

---

## Quick Start

```bash
# 1. Train and export weights (PC)
python export_weights_for_pico.py

# 2. Full analysis with plots (PC)
python mq_health_predictor.py

# 3. Upload to Pico (via Thonny)
# Copy pico_mq_monitor.py as main.py
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
