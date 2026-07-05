"""
anomaly_detection.py
Unsupervised anomaly detection on raw 1Hz telemetry using Isolation
Forest, validated against the ground-truth anomaly labels the simulator
injected (anomaly_events.csv).

IMPORTANT FRAMING: in a real deployment you would NOT have these ground
truth labels -- that's the whole point of using an unsupervised method
here instead of a classifier. We only have labels because we control the
simulator, and we use them ONLY to validate the detector's precision/
recall after the fact, exactly like validate_against_ground_truth.py did
for the risk score in Phase 1. The Isolation Forest itself never sees
these labels.

FEATURE ENGINEERING NOTE (this took two iterations to get right):
An earlier version included raw speed/RPM/gyro/throttle plus their deltas.
It scored 0% recall at any reasonable threshold despite a plausible-looking
ROC-AUC, because normal-but-aggressive driving (harsh braking, high-speed
cornering) produced larger statistical outliers than the actual injected
sensor anomalies -- Isolation Forest was correctly finding outliers, just
not the ones we cared about. Restricting features to SENSOR HEALTH signals
(engine temp, oil pressure, battery voltage, brake efficiency) plus GPS
integrity and a dedicated impact-detection delta improved ROC-AUC from
0.71 to 0.80. This is a real, worth-remembering lesson: unsupervised
anomaly detection is only as good as the feature space's ability to
separate the anomaly you care about from ordinary variation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import joblib

SENSOR_HEALTH_COLS = [
    "engine_temp_c", "oil_pressure_psi", "battery_voltage", "brake_efficiency_pct",
]
SORT_KEYS = ["trip_id", "timestamp"]


def load_and_sort(telemetry_path: str) -> pd.DataFrame:
    """Single canonical sort, done once, so every downstream array/series
    is guaranteed to be in the same row order."""
    telemetry = pd.read_csv(telemetry_path, parse_dates=["timestamp"])
    return telemetry.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects `df` already sorted by (trip_id, timestamp) via load_and_sort."""
    df = df.copy()
    df["is_gps_missing"] = df["gps_lat"].isna().astype(int)
    df["gps_lon_filled"] = df.groupby("trip_id")["gps_lon"].ffill().bfill()

    features = pd.DataFrame(index=df.index)
    for col in SENSOR_HEALTH_COLS:
        features[col] = df[col]
        features[f"{col}_delta"] = df.groupby("trip_id")[col].diff().fillna(0).abs()

    features["is_gps_missing"] = df["is_gps_missing"]
    features["gps_lon_delta"] = df.groupby("trip_id")["gps_lon_filled"].diff().fillna(0).abs()
    # acceleration_x_g delta is kept specifically for impact-event detection:
    # a genuine impact produces a much larger, more sudden deceleration
    # spike than ordinary braking. Other behavioral signals (speed, rpm,
    # gyro, throttle) are deliberately excluded -- see module docstring.
    features["accel_delta"] = df.groupby("trip_id")["acceleration_x_g"].diff().fillna(0).abs()
    return features


def attach_ground_truth(df: pd.DataFrame, anomalies: pd.DataFrame) -> pd.Series:
    """
    Reconstructs each row's position within its trip (0-indexed, matching
    how anomalies.py originally assigned index_in_trip) and left-joins the
    ground truth flag on (trip_id, row_idx_in_trip). Vectorized via merge
    -- matters once telemetry reaches ~1M rows. Expects `df` already
    sorted via load_and_sort so the returned Series aligns positionally
    with anything else built from the same sorted `df`.
    """
    row_idx = df.groupby("trip_id").cumcount()
    relevant = anomalies[anomalies["trip_id"].isin(df["trip_id"].unique())].copy()
    relevant = relevant.rename(columns={"index_in_trip": "row_idx_in_trip"})
    relevant["is_anomaly"] = 1

    key_df = pd.DataFrame({
        "trip_id": df["trip_id"].values, "row_idx_in_trip": row_idx.values,
    })
    merged = key_df.merge(
        relevant[["trip_id", "row_idx_in_trip", "is_anomaly"]],
        on=["trip_id", "row_idx_in_trip"], how="left",
    )
    return merged["is_anomaly"].fillna(0).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=str, default="./data/telemetry_recent.csv")
    parser.add_argument("--anomalies", type=str, default="./data/anomaly_events.csv")
    parser.add_argument("--out", type=str, default="./model_output")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_and_sort(args.telemetry)
    anomalies = pd.read_csv(args.anomalies)

    print(f"Telemetry rows: {len(df):,}")
    relevant = anomalies[anomalies["trip_id"].isin(df["trip_id"].unique())]
    print(f"Ground-truth anomaly events falling within this telemetry sample: {len(relevant)}")
    if len(relevant) < 50:
        print("WARNING: fewer than 50 ground-truth events in this sample -- precision/recall "
              "numbers below will be statistically noisy. Regenerate telemetry with a larger "
              "raw window / more vehicles for a meaningful evaluation.")

    y_true = attach_ground_truth(df, anomalies)
    print(f"True anomaly rate in this sample: {y_true.mean()*100:.3f}%")

    features = build_anomaly_features(df)
    feature_cols = features.columns.tolist()
    X = features[feature_cols].fillna(0).values

    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
    model.fit(X)
    anomaly_score = -model.score_samples(X)  # flip sign: higher = more anomalous

    roc_auc = roc_auc_score(y_true, anomaly_score)
    pr_auc = average_precision_score(y_true, anomaly_score)
    baseline = y_true.mean()
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}  (random-guessing baseline: {baseline:.5f} -- "
          f"{pr_auc/baseline:.1f}x better than random)")

    print("\nOperating points (recall/precision at different review-capacity levels):")
    print(f"{'Top %':>8} {'Rows flagged':>13} {'Recall':>8} {'Precision':>10} {'True positives':>15}")
    for top_pct in [0.001, 0.005, 0.01, 0.02, 0.05]:
        threshold = np.quantile(anomaly_score, 1 - top_pct)
        flagged = anomaly_score >= threshold
        tp = int((flagged & (y_true.values == 1)).sum())
        recall = tp / y_true.sum() if y_true.sum() else 0
        precision = tp / flagged.sum() if flagged.sum() else 0
        print(f"{top_pct*100:>7.1f}% {flagged.sum():>13,} {recall:>8.3f} {precision:>10.4f} {tp:>15}")

    operating_threshold = np.quantile(anomaly_score, 0.98)  # top-2% -- reasonable review volume
    ml_flag = anomaly_score >= operating_threshold

    # HYBRID RULE: GPS dropout is a deterministic, binary condition
    # (gps_lat is null) that occurs in only ~0.02% of rows -- too sparse
    # for a multivariate outlier detector to reliably isolate among 13
    # other continuous features. A simple rule catches it with perfect
    # recall trivially; forcing the ML model to also learn this wastes its
    # capacity on a problem that doesn't need it. Isolation Forest is kept
    # focused on the anomaly types that actually benefit from multivariate
    # outlier detection (sensor spikes, impacts, route deviation).
    gps_missing_flag = features["is_gps_missing"].values.astype(bool)
    predicted_flag = ml_flag | gps_missing_flag

    print(f"\nHybrid rule: GPS-dropout rows are ALWAYS flagged directly (rule-based), "
          f"regardless of ML score. This adds {int(gps_missing_flag.sum())} "
          f"rule-flagged rows on top of the {int(ml_flag.sum()):,} ML-flagged rows.")

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, anomaly_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_curve, precision_curve)
    ax.axhline(baseline, color="red", linestyle="--", label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Isolation Forest Anomaly Detection\nPR-AUC={pr_auc:.4f} ({pr_auc/baseline:.1f}x random)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}/anomaly_pr_curve.png", dpi=120)
    plt.close()

    # Recall by anomaly type at the top-2% operating point -- vectorized via
    # merge rather than row-wise apply (important at ~1M row scale).
    print("\nRecall by anomaly type (at top-2% operating point):")
    row_idx = df.groupby("trip_id").cumcount()
    type_lookup = relevant.rename(columns={"index_in_trip": "row_idx_in_trip"})[
        ["trip_id", "row_idx_in_trip", "anomaly_type"]
    ]
    key_df = pd.DataFrame({
        "trip_id": df["trip_id"].values, "row_idx_in_trip": row_idx.values,
        "predicted_flag": predicted_flag,
    })
    matched = key_df.merge(type_lookup, on=["trip_id", "row_idx_in_trip"], how="inner")
    for atype, group in matched.groupby("anomaly_type"):
        caught = int(group["predicted_flag"].sum())
        total = len(group)
        print(f"  {atype}: {caught}/{total} caught ({caught/total*100:.1f}% recall)")

    joblib.dump(model, f"{args.out}/anomaly_detector.joblib")
    joblib.dump(feature_cols, f"{args.out}/anomaly_feature_names.joblib")
    print(f"\nModel saved to {args.out}/")


if __name__ == "__main__":
    main()
