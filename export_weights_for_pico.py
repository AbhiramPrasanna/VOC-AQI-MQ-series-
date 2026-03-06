"""
Export trained MLP weights for Raspberry Pi Pico
=================================================
Trains a neural network on MQ sensor PPM data to classify air quality
as SAFE vs DANGEROUS, then exports weights in a format the Pico can use.

The model takes 3 sensor PPM readings as input (MQ135, MQ5, MQ4) and
outputs a probability of dangerous air quality.

Usage:
    python export_weights_for_pico.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MQ_FILE = os.path.join(DATA_DIR, "MQ_Dataset.xlsx")

# ── Health thresholds from OSHA/WHO/EPA (same as in pico_mq_monitor.py) ──
THRESHOLDS = {
    "MQ135": {"safe": 150, "warning": 350, "danger": 500},
    "MQ4":   {"safe": 25,  "warning": 45,  "danger": 70},
    "MQ5":   {"safe": 7,   "warning": 12,  "danger": 18},
}

# Sensors available on user's Pico
PICO_SENSORS = ["MQ135", "MQ5", "MQ4"]


def compute_danger_score(row):
    """Score a reading based on how many sensors exceed health thresholds."""
    score = 0
    for sensor, thresh in THRESHOLDS.items():
        if sensor in row.index:
            val = row[sensor]
            if val >= thresh["danger"]:
                score += 3
            elif val >= thresh["warning"]:
                score += 2
            elif val >= thresh["safe"]:
                score += 1
    return score


def train_and_export():
    print("=" * 60)
    print("  SENSOR-BASED MLP TRAINING FOR PICO")
    print("=" * 60)

    # ── Load MQ dataset ──
    mq = pd.read_excel(MQ_FILE)
    print(f"  Loaded {len(mq)} readings from MQ_Dataset.xlsx")

    # Pivot: one row per reading index, columns = sensors
    pivot = mq.pivot_table(index="idx", columns="sensor", values="ppm", aggfunc="mean")
    pivot = pivot.reset_index(drop=True)

    available = [s for s in PICO_SENSORS if s in pivot.columns]
    print(f"  Available sensors in data: {available}")
    print(f"  Readings: {len(pivot)}")

    if len(available) == 0:
        print("  ERROR: No matching sensors found in dataset!")
        return

    # ── Compute danger target ──
    pivot["danger_score"] = pivot.apply(compute_danger_score, axis=1)
    # "Dangerous" if any sensor is at warning or above (score >= 2)
    pivot["dangerous"] = (pivot["danger_score"] >= 2).astype(int)

    print(f"  Target distribution: {pivot['dangerous'].mean()*100:.1f}% dangerous")

    X = pivot[available].values
    y = pivot["dangerous"].values

    # ── Data augmentation (30 readings is tiny) ──
    rng = np.random.RandomState(42)
    X_aug = [X]
    y_aug = [y]

    # Augment with noise + also generate synthetic cross-threshold examples
    for i in range(30):
        noise_scale = 0.08 if i < 20 else 0.15
        noise = rng.normal(0, noise_scale, X.shape) * np.abs(X) + rng.normal(0, 1, X.shape)
        X_noisy = np.maximum(X + noise, 0)  # PPM can't be negative
        X_aug.append(X_noisy)
        # Recompute labels for augmented data
        y_noisy = []
        for row in X_noisy:
            score = 0
            for j, sensor in enumerate(available):
                if sensor in THRESHOLDS:
                    t = THRESHOLDS[sensor]
                    if row[j] >= t["danger"]:
                        score += 3
                    elif row[j] >= t["warning"]:
                        score += 2
                    elif row[j] >= t["safe"]:
                        score += 1
            y_noisy.append(1 if score >= 2 else 0)
        y_aug.append(np.array(y_noisy))

    # Also add synthetic "clean air" and "polluted air" examples
    n_synthetic = 100
    # Clean air
    clean = rng.uniform(0, 1, (n_synthetic, len(available)))
    for j, sensor in enumerate(available):
        clean[:, j] *= THRESHOLDS[sensor]["safe"] * 0.5
    X_aug.append(clean)
    y_aug.append(np.zeros(n_synthetic, dtype=int))

    # Polluted air
    polluted = rng.uniform(0.5, 1.5, (n_synthetic, len(available)))
    for j, sensor in enumerate(available):
        polluted[:, j] *= THRESHOLDS[sensor]["danger"]
    X_aug.append(polluted)
    y_aug.append(np.ones(n_synthetic, dtype=int))

    # Borderline cases (around warning thresholds)
    borderline = rng.uniform(0.7, 1.3, (n_synthetic, len(available)))
    for j, sensor in enumerate(available):
        borderline[:, j] *= THRESHOLDS[sensor]["warning"]
    X_aug.append(borderline)
    y_b = []
    for row in borderline:
        score = 0
        for j, sensor in enumerate(available):
            if sensor in THRESHOLDS:
                t = THRESHOLDS[sensor]
                if row[j] >= t["danger"]:
                    score += 3
                elif row[j] >= t["warning"]:
                    score += 2
                elif row[j] >= t["safe"]:
                    score += 1
        y_b.append(1 if score >= 2 else 0)
    y_aug.append(np.array(y_b))

    X_all = np.vstack(X_aug)
    y_all = np.concatenate(y_aug)

    print(f"  Augmented dataset: {len(X_all)} samples")
    print(f"  Class balance: {y_all.mean()*100:.1f}% dangerous, "
          f"{(1-y_all.mean())*100:.1f}% safe")

    # ── Train/test split ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # ── Architecture: keep small for Pico (264KB RAM) ──
    # input(3) -> dense(16) -> dense(8) -> output(2)
    print(f"\n  Training MLP: input({len(available)}) -> 16 -> 8 -> output(2)")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    n_epochs = 100
    print(f"\n  {'Epoch':>7} | {'Train Acc':>10} | {'Val Acc':>8}")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*8}")

    best_val = 0
    patience = 0
    for epoch in range(1, n_epochs + 1):
        mlp.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        val_acc = accuracy_score(y_test, mlp.predict(X_test))
        if epoch <= 3 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"  {epoch:>7} | {train_acc*100:>9.1f}% | {val_acc*100:>7.1f}%")
        if val_acc > best_val:
            best_val = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print(f"  ** Early stopping at epoch {epoch}")
                break

    # ── Evaluate ──
    y_pred = mlp.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred)
    print(f"\n  Final Test Accuracy: {final_acc*100:.1f}%")
    print(classification_report(y_test, y_pred,
                                target_names=["SAFE", "DANGEROUS"], zero_division=0))

    cv = cross_val_score(
        MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200, random_state=42),
        X_scaled, y_all, cv=5, scoring="accuracy"
    )
    print(f"  5-Fold CV: {cv.mean()*100:.1f}% +/- {cv.std()*100:.1f}%")

    # ── Export weights ──
    total_params = sum(W.size + b.size for W, b in zip(mlp.coefs_, mlp.intercepts_))
    mem_kb = total_params * 4 / 1024

    print(f"\n  Total parameters: {total_params}")
    print(f"  Memory: ~{mem_kb:.1f} KB (fits in Pico flash)")
    print()
    print("=" * 60)
    print("  PASTE THE FOLLOWING INTO pico_mq_monitor.py")
    print("=" * 60)
    print()

    # Scaler params
    print("# ── Sensor MLP Scaler (StandardScaler) ──")
    print("# Sensors: {}".format(available))
    print("MLP_SCALER_MEAN = {}".format(
        [round(float(x), 6) for x in scaler.mean_]))
    print("MLP_SCALER_SCALE = {}".format(
        [round(float(x), 6) for x in scaler.scale_]))
    print()

    # Weights
    print("MLP_WEIGHTS = [")
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        W_list = [[round(float(x), 4) for x in row] for row in W.T]
        b_list = [round(float(x), 4) for x in b]
        print("    # Layer {} ({} -> {})".format(i + 1, W.shape[0], W.shape[1]))
        print("    (")
        print("        {},".format(W_list))
        print("        {}".format(b_list))
        print("    ),")
    print("]")
    print()
    print("# Model info: input={}, hidden=(16,8), output=2".format(len(available)))
    print("# Accuracy: {:.1f}%  |  CV: {:.1f}%".format(final_acc * 100, cv.mean() * 100))
    print("# Threshold: danger_score >= 2 (based on OSHA/WHO limits)")


if __name__ == "__main__":
    train_and_export()
