# Code Documentation — VOC-AQI Monitoring System

This document explains every file in the project and how they connect to form an end-to-end pipeline: from raw CDC data and MQ sensor readings, through ML model training, to real-time inference on a Raspberry Pi Pico.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PC (Training Side)                          │
│                                                                    │
│  data/                                                             │
│    ├── U.S._Chronic_Disease_Indicators_COPD.csv  (309K CDC rows)   │
│    └── MQ_Dataset.xlsx                           (210 MQ readings) │
│            │                                                       │
│            ▼                                                       │
│  ┌──────────────────────────────────┐                              │
│  │ Combined_Model_Training.ipynb    │ ◄── Single source of truth   │
│  │  (or mq_health_predictor.py /   │     for training pipeline     │
│  │   export_weights_for_pico.py)   │                               │
│  └──────────────┬───────────────────┘                              │
│                 │ exports weights                                   │
│                 ▼                                                   │
│        pico_mlp_weights.py   (auto-generated weights file)         │
│                                                                    │
│  build_notebook.py  ◄── utility to rebuild the .ipynb from code    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  upload via Thonny / USB
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Raspberry Pi Pico (Inference Side)               │
│                                                                    │
│  main.py  (= pico_mq_monitor.py)                                  │
│    imports ──► pico_mlp_weights.py                                 │
│                                                                    │
│  Hardware:                                                         │
│    MQ135 (GP26) ──┐                                                │
│    MQ5  (GP27) ───┤──► ADC read ──► PPM conversion ──► MLP ──► Risk│
│    MQ4  (GP28) ──┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Explanation

### 1. `data/U.S._Chronic_Disease_Indicators__Chronic_Obstructive_Pulmonary_Disease.csv`

**What it is:** CDC Chronic Disease Indicators dataset — 309,215 rows of COPD prevalence data across U.S. states, stratified by race/ethnicity and year.

**Key columns used:**
- `LocationAbbr` — state abbreviation (mapped to latitude/longitude)
- `DataValue` — age-adjusted COPD prevalence (%)
- `Stratification1` — race/ethnicity category
- `YearEnd` — year of the data point

**Role:** Training data for the COPD Risk MLP (Model B). Establishes the epidemiological link between geographic/demographic factors and respiratory disease prevalence.

---

### 2. `Combined_Model_Training.ipynb`

**What it is:** The main Jupyter notebook — the single source of truth for the entire training pipeline. It trains 5 models, compares them, and exports the best Pico-compatible model's weights.

**What it does step-by-step:**

| Step | Section | Action |
|------|---------|--------|
| 1 | Import Libraries | Loads sklearn, pandas, numpy, matplotlib |
| 2 | Load MQ Dataset | Reads `MQ_Dataset.xlsx`, pivots to a feature matrix (rows = reading indices, columns = sensor PPM values) |
| 3 | Load COPD Dataset | Reads CDC CSV, filters for age-adjusted COPD prevalence with race stratification, merges with state lat/lon coordinates |
| 4 | Feature Engineering (MQ) | Defines OSHA/WHO health thresholds for each sensor. Computes a `danger_score` per reading (how many sensors exceed thresholds). Creates binary target: `dangerous = 1` if score >= 2 |
| 5 | Data Augmentation | Expands the 30-row MQ dataset to ~1200+ samples via Gaussian noise injection + synthetic clean/polluted/borderline examples. Labels are recomputed from thresholds after augmentation |
| 6 | Feature Engineering (COPD) | Maps states to lat/lon, encodes race, creates binary target: prevalence > median = HIGH RISK |
| 7 | **Train Model A: Pico Sensor MLP** | Architecture: `input(3) → 16 → 8 → output(2)`. Trains on MQ135 + MQ5 + MQ4 PPM. Uses early stopping (patience=15). This is the model that runs on the Pico |
| 8 | **Train Model B: COPD Risk MLP** | Architecture: `input(4) → 64 → 32 → 16 → output(2)`. Trains on lat, lon, race, year from CDC data. Learns geographic/demographic COPD risk patterns |
| 9 | **Train Model C: VOC Gradient Boosting** | 100 boosting stages on MQ135 time-series features (rolling mean, std, diff, pct_change). Threshold: 300 ppm (WHO/EPA moderate→poor AQ) |
| 10 | **Train Model D: Multi-Sensor RF** | 200-tree Random Forest using all 7 MQ sensors. Danger threshold: score >= 7 (compound risk from multiple sensors) |
| 11 | **Train Model E: Ensemble** | Stacking ensemble: Pico MLP probability + COPD risk score + raw PPM → Logistic Regression meta-learner |
| 12 | Evaluate & Compare | Training curves, confusion matrices, 5-fold cross-validation for all models. Bar chart comparison |
| 13 | Select Best | Picks the best Pico-compatible model (must use only 3 sensors, fit in 264KB RAM) |
| 14 | Export Weights | Writes `MLP_SCALER_MEAN`, `MLP_SCALER_SCALE`, `MLP_WEIGHTS` to `pico_mlp_weights.py` |
| 15 | Verify | Runs manual forward pass on test cases (clean air, moderate, warning, dangerous) to confirm exported weights match training predictions |

**Models trained:**

| Model | Type | Input | Dataset | Output |
|-------|------|-------|---------|--------|
| A — Pico MLP | MLP (3→16→8→2) | MQ135, MQ5, MQ4 PPM | MQ augmented | SAFE / DANGEROUS |
| B — COPD MLP | MLP (4→64→32→16→2) | lat, lon, race, year | CDC 309K rows | LOW / HIGH RISK |
| C — VOC GB | Gradient Boosting (100 stages) | MQ135 + time features | MQ135 augmented | LOW / HIGH VOC |
| D — Multi-Sensor RF | Random Forest (200 trees) | All 7 MQ sensors | MQ augmented | SAFE / DANGEROUS |
| E — Ensemble | Logistic Regression | MLP prob + COPD score + raw PPM | Combined | SAFE / DANGEROUS |

---

### 3. `mq_health_predictor.py`

**What it is:** Standalone PC script that trains the same 3 core models (COPD MLP, VOC Gradient Boosting, Multi-Sensor RF), runs a full sensor analysis, and generates a comprehensive health risk report with plots.

**Key classes:**

#### `COPDRiskModel`
- Loads CDC CSV, filters for COPD prevalence + race stratification
- Merges with state coordinates (lat/lon lookup table)
- Binary target: prevalence > median → HIGH RISK
- MLP architecture: `input(4) → 64 → 32 → 16 → output(2)`
- Features: latitude, longitude, race (encoded), year
- Epoch-by-epoch training with early stopping (patience=10)
- `predict(lat, lon, race, year)` → risk label + probability

#### `VOCRiskModel`
- Takes MQ135 PPM time series
- Augments 30 readings → 450 via Gaussian noise (15x)
- Builds time-series features: rolling mean/std (window=5,15), diff, diff2, pct_change
- Risk threshold: 300 ppm (WHO/EPA moderate→poor boundary)
- Gradient Boosting: 100 stages, lr=0.05, max_depth=4, subsample=0.8
- Tracks staged accuracy using `staged_predict()`
- `predict_single(mq135_ppm)` → risk + probability

#### `MultiSensorModel`
- Pivots MQ dataset so each row has all 7 sensor PPM values
- Danger score based on OSHA thresholds per sensor (weighted: danger=3, warning=2, safe=1)
- Binary target: score >= 7 → DANGEROUS
- Augments with 5% Gaussian noise (20x)
- Random Forest: 200 trees, max_depth=8
- Reports feature importance (which sensor matters most)

**Other functions:**
- `analyze_sensor()` — per-sensor analysis against health thresholds
- `estimate_air_quality()` — estimates PM2.5 and ozone from MQ readings (approximation)
- `classify_aqi()` — EPA AQI category (Good/Moderate/Unhealthy/Hazardous)
- `generate_plots()` — sensor bar charts, VOC risk scatter, AQI estimation, training curves
- `print_report()` — comprehensive health risk report with compound breakdown and actionable recommendations

**CLI usage:**
```bash
python mq_health_predictor.py                        # analyze MQ_Dataset.xlsx
python mq_health_predictor.py --file readings.xlsx   # custom file
python mq_health_predictor.py --live 411.8 23.0 7.0  # live MQ135, MQ9, MQ2 PPM
```

**Connection to other files:**
- Reads: `data/MQ_Dataset.xlsx`, `data/U.S._Chronic_Disease_Indicators_COPD.csv`
- Writes: plots to `output/` directory
- Shares the same health thresholds and sensor specifications as `pico_mq_monitor.py`

---

### 4. `export_weights_for_pico.py`

**What it is:** Lightweight standalone script that trains ONLY the Pico-compatible MLP and prints the weights in copy-paste format.

**What it does:**
1. Loads `MQ_Dataset.xlsx` and pivots to get MQ135 + MQ5 + MQ4 PPM per reading
2. Computes danger labels using OSHA/WHO thresholds (danger_score >= 2 → dangerous)
3. Augments the 30-row dataset:
   - 30 rounds of Gaussian noise (8% and 15% scale)
   - 100 synthetic "clean air" samples (PPM < 50% of safe threshold)
   - 100 synthetic "polluted air" samples (PPM around danger threshold)
   - 100 synthetic "borderline" samples (PPM around warning threshold)
4. Trains MLP: `input(3) → 16 → 8 → output(2)` with early stopping (patience=15)
5. Prints `MLP_SCALER_MEAN`, `MLP_SCALER_SCALE`, and `MLP_WEIGHTS` to stdout

**Connection to other files:**
- Reads: `data/MQ_Dataset.xlsx`
- Output: printed weights meant to be pasted into `pico_mq_monitor.py` (or saved as `pico_mlp_weights.py`)
- Alternative to running the full notebook — same model, simpler script

---

### 5. `pico_mlp_weights.py`

**What it is:** Auto-generated Python file containing the trained MLP weights, biases, and scaler parameters. This is the **output** of the training pipeline and the **input** to the Pico inference code.

**Contents:**
- `MLP_SCALER_MEAN = [384.64, 10.30, 39.73]` — StandardScaler mean per sensor
- `MLP_SCALER_SCALE = [230.92, 5.40, 21.76]` — StandardScaler standard deviation per sensor
- `MLP_WEIGHTS` — nested list of `(weight_matrix, bias_vector)` tuples for 3 layers:
  - Layer 1: 3 inputs → 16 neurons (48 weights + 16 biases)
  - Layer 2: 16 → 8 neurons (128 weights + 8 biases)
  - Layer 3: 8 → 1 output (8 weights + 1 bias) — sigmoid for binary classification
- `MLP_SENSOR_ORDER = ['MQ135', 'MQ5', 'MQ4']` — input order the model expects

**Total parameters:** ~209 (~0.8 KB) — easily fits in Pico's 264KB RAM.

**Connection to other files:**
- Generated by: `Combined_Model_Training.ipynb` or `export_weights_for_pico.py`
- Consumed by: `pico_mq_monitor.py` (values are copy-pasted into the script, or the file is imported)

---

### 6. `pico_mq_monitor.py`

**What it is:** The MicroPython script that runs on the Raspberry Pi Pico. It reads 3 MQ gas sensors, converts raw ADC to PPM, runs ML inference, and outputs real-time health risk assessments over serial.

**No external libraries needed** — runs on stock MicroPython.

**Key sections:**

#### Configuration (top of file)
- `SENSOR_PINS` — MQ135→GP26, MQ5→GP27, MQ4→GP28
- `VCC = 3.3`, `RL = 10.0 kΩ`, `ADC_RESOLUTION = 65535` (16-bit)
- `INTERVAL_SEC = 5` — reading interval
- `CALIBRATE_ON_BOOT = True` — run auto-calibration in clean air on first boot
- `SENSOR_CALIBRATION` — R0 values per sensor (set during calibration or manually)
- `PPM_CURVES` — power-law coefficients (`PPM = a × (Rs/R0)^b`) from datasheets

#### Health Thresholds
- `THRESHOLDS` — `(safe, warning, danger)` PPM tuples per sensor, based on OSHA/WHO/EPA

#### MLP Weights (embedded)
- `MLP_SCALER_MEAN`, `MLP_SCALER_SCALE`, `MLP_WEIGHTS` — copied from `pico_mlp_weights.py`
- `MLP_SENSOR_ORDER = ["MQ135", "MQ5", "MQ4"]`

#### Calibration Functions
- `load_calibration()` — loads saved R0 values from `calibration.json`
- `calibrate_sensors()` — samples ADC for 30 seconds in clean air, computes R0 = avg(Rs) / clean_air_ratio, saves to `calibration.json`
- `check_wiring()` — sanity check: warns if ADC reads near 0 (disconnected) or near max (shorted)

#### Core Functions
- `read_adc_avg(adc, n=50)` — averages 50 ADC samples to reduce noise
- `raw_to_ppm(raw_adc, sensor)` — ADC → voltage → Rs → Rs/R0 ratio → PPM via power law
- `classify(ppm, sensor)` — PPM → SAFE / ELEVATED / WARNING / DANGER using thresholds
- `mlp_forward(inputs, weights)` — pure MicroPython neural network forward pass:
  - Matrix multiply + bias add per layer
  - ReLU activation for hidden layers
  - Sigmoid for single output neuron (binary classification)
  - No numpy, no libraries — just loops and basic math
- `estimate_copd_risk(readings)` — primary risk assessment:
  1. **ML path:** scales sensor PPM using `MLP_SCALER_MEAN/SCALE`, runs `mlp_forward()`, outputs danger probability (0–100%)
  2. **Rule-based fallback:** weighted score from each sensor exceeding thresholds
  3. Returns `(risk_percent, risk_label, method)` — label is "LOW RISK" / "MODERATE" / "HIGH RISK"

#### Main Loop
```
Every 5 seconds:
  1. Read raw ADC from each sensor (50-sample average)
  2. Convert to voltage and PPM
  3. Update rolling average (last 10 readings)
  4. Classify using the ROLLING AVERAGE (not instantaneous):
     - Level label (SAFE/ELEVATED/WARNING/DANGER)
     - Risk score via MLP or rule-based
  5. Print formatted output + JSON line
  6. LED: solid ON for DANGER, blink for WARNING, OFF for SAFE
```

**Important detail — instantaneous vs. average PPM:**

The output line `MQ135 | 2175.3 ppm (avg:252.4) | ELEVATED` shows:
- `2175.3 ppm` — instantaneous reading from current ADC sample (displayed only)
- `252.4 ppm` — rolling average of last 10 readings → **this is what determines the level and risk score**

The rolling average smooths sensor noise and transient spikes.

**Connection to other files:**
- Contains embedded copies of `MLP_SCALER_MEAN`, `MLP_SCALER_SCALE`, `MLP_WEIGHTS` (from `pico_mlp_weights.py`)
- Deployed as `main.py` on the Pico alongside `pico_mlp_weights.py`

---

### 7. `build_notebook.py`

**What it is:** A utility script that programmatically constructs `Combined_Model_Training.ipynb` by generating all the notebook cells (markdown + code) in Python and writing them as a `.ipynb` JSON file.

**Why it exists:** Notebooks are JSON files with a specific schema. This script lets you version-control the notebook content as readable Python code instead of raw JSON, and rebuild the `.ipynb` cleanly.

**What it does:**
1. Opens the existing `Combined_Model_Training.ipynb`
2. Creates cell objects using `make_code_cell()` and `make_md_cell()` helpers
3. Builds all 21 cells in order (title → imports → data loading → training → evaluation → export → verification → summary)
4. Writes the assembled cells back to the notebook JSON

**Connection to other files:**
- Reads/writes: `Combined_Model_Training.ipynb`
- The code inside the generated cells references `data/MQ_Dataset.xlsx`, `data/COPD.csv`, and outputs `pico_mlp_weights.py`

---

## How All Files Connect — The Full Pipeline

### Training Flow (PC side)

```
Step 1: Data Ingestion
  data/COPD.csv ──────────────┐
  data/MQ_Dataset.xlsx ───────┤
                              ▼
Step 2: Model Training    (Combined_Model_Training.ipynb)
                          (or mq_health_predictor.py)
                          (or export_weights_for_pico.py)
  ┌────────────────────────────────────────────────┐
  │ MQ data → pivot → augment → train Pico MLP    │
  │ CDC data → filter → features → train COPD MLP │
  │ MQ135 → time features → train VOC GB          │
  │ All MQ → augment → train Multi-Sensor RF      │
  │ All outputs → train Ensemble                   │
  │ Compare all → select best Pico-compatible      │
  └───────────────────┬────────────────────────────┘
                      │
Step 3: Weight Export │
                      ▼
              pico_mlp_weights.py
              (scaler params + weight matrices)
```

### Deployment Flow (Pico side)

```
Step 4: Upload to Pico via Thonny
  pico_mq_monitor.py ──► renamed to main.py on Pico
  pico_mlp_weights.py ──► copied alongside

Step 5: Runtime on Pico
  Power on → auto-calibrate in clean air (first run)
           → save R0 to calibration.json
  Loop:
    ADC read (GP26, GP27, GP28)
      → voltage = raw / 65535 × 3.3V
      → Rs = 10kΩ × (3.3 - V) / V
      → ratio = Rs / R0
      → PPM = a × ratio^b
      → rolling average (10 readings)
      → classify(avg_ppm) → SAFE/ELEVATED/WARNING/DANGER
      → scale PPM: (ppm - mean) / std
      → mlp_forward(scaled, weights) → danger probability
      → risk label: <40% LOW, 40-70% MODERATE, >70% HIGH
      → print to serial + JSON
      → LED alert if WARNING/DANGER
```

### Shared Constants Across Files

These critical values are kept consistent across all files:

| Constant | Value | Used In |
|----------|-------|---------|
| MQ135 safe/warn/danger | 150 / 350 / 500 ppm | notebook, predictor, pico monitor, export |
| MQ5 safe/warn/danger | 7 / 12 / 18 ppm | notebook, predictor, pico monitor, export |
| MQ4 safe/warn/danger | 25 / 45 / 70 ppm | notebook, predictor, pico monitor, export |
| Pico MLP architecture | 3 → 16 → 8 → 1(or 2) | notebook, export, pico monitor |
| Sensor order | MQ135, MQ5, MQ4 | notebook, export, weights, pico monitor |
| PPM curves (a, b coefficients) | per sensor | pico monitor |
| R0 clean air ratios | per sensor | pico monitor |

### Alternative Training Paths

There are 3 ways to train and export weights — they all produce the same result:

| Path | Script | Models Trained | Output |
|------|--------|---------------|--------|
| **Recommended** | `Combined_Model_Training.ipynb` | All 5 models + comparison | `pico_mlp_weights.py` + plots |
| Quick export | `export_weights_for_pico.py` | Pico MLP only | Prints weights to stdout |
| Full analysis | `mq_health_predictor.py` | 3 models + health report | Plots + terminal report |

---

## Key Algorithms

### PPM Conversion (on Pico)
The MQ sensor breakout boards output an analog voltage proportional to gas concentration. Conversion:

```
Voltage = (ADC_raw / 65535) × 3.3V
Rs = RL × (VCC - V) / V          # sensor resistance
ratio = Rs / R0                    # R0 = calibrated clean-air resistance
PPM = a × ratio^b                  # power-law from datasheet curves
```

### MLP Forward Pass (on Pico)
Pure arithmetic — no libraries:

```
For each layer (W, b):
  For each neuron j:
    output[j] = b[j] + Σ(W[j][k] × input[k])
    if hidden layer: output[j] = max(0, output[j])   # ReLU
  input = output

Final: sigmoid(output[0]) → probability of "dangerous"
```

### Rolling Average
The Pico keeps a buffer of the last 10 PPM readings per sensor. The **average** of this buffer is used for classification and risk scoring, not the instantaneous value. This smooths out ADC noise and transient spikes.

---

## Summary Table

| File | Language | Runs On | Purpose |
|------|----------|---------|---------|
| `Combined_Model_Training.ipynb` | Python (Jupyter) | PC | Train all models, compare, export weights |
| `mq_health_predictor.py` | Python | PC | Standalone 3-model trainer + health report |
| `export_weights_for_pico.py` | Python | PC | Quick Pico MLP training + weight export |
| `build_notebook.py` | Python | PC | Rebuild notebook `.ipynb` from code |
| `pico_mlp_weights.py` | Python | PC/Pico | Auto-generated weights (bridge between PC and Pico) |
| `pico_mq_monitor.py` | MicroPython | Pico | Real-time sensor reading + ML inference |
| `data/COPD.csv` | CSV | PC | CDC COPD prevalence dataset |
| `data/MQ_Dataset.xlsx` | Excel | PC | MQ sensor readings (7 sensors × 30 readings) |
