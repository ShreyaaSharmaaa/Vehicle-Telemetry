"""
maintenance_feature_engineering.py
Builds features for the predictive maintenance classifier.

TWO leakage risks here, not one -- and the second is easy to miss:

1. LABEL LEAKAGE (same as Phase 1): `vehicle_failure_mode` and
   `days_to_failure` are ground truth and must never be model inputs.
   `will_fail_within_30d` is the TARGET, not a feature.

2. TEMPORAL LEAKAGE (new to this phase): a single trip's raw sensor
   snapshot (e.g. one engine_temp_c reading) is noisy and doesn't reflect
   a TREND. Real predictive maintenance needs to see a vehicle's
   trajectory over its recent trips, not one isolated data point. We
   build rolling-window features (mean and slope over the last N trips)
   computed ONLY from that vehicle's own past trips, in time order --
   never using future trips relative to the row being featurized.

   We also NEVER split train/test by random trip -- adjacent trips from
   the same vehicle are highly correlated (nearly identical rolling
   windows), so a random split would leak information between train and
   test. See train_maintenance_model.py for the vehicle-level split this
   forces us to use.
"""

import pandas as pd
import numpy as np

LEAKAGE_COLUMNS = ["vehicle_failure_mode", "days_to_failure"]
TARGET_COLUMN = "will_fail_within_30d"

ROLLING_WINDOW = 10  # trips of trailing history used for trend features

SENSOR_COLUMNS = [
    "avg_engine_temp_c", "min_oil_pressure_psi",
    "min_battery_voltage", "brake_efficiency_pct",
]


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Simple trailing linear trend (points-per-trip slope) over the last
    `window` trips, shifted so the CURRENT trip's own value is not included
    in computing its own trend -- otherwise a single trip's noise would
    leak into its own feature.
    """
    def slope(x):
        if len(x) < 2:
            return 0.0
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    return series.shift(1).rolling(window, min_periods=2).apply(slope, raw=True)


def build_maintenance_features(trips: pd.DataFrame) -> pd.DataFrame:
    df = trips.copy()
    df["trip_start_time"] = pd.to_datetime(df["trip_start_time"])
    df = df.sort_values(["vehicle_id", "trip_start_time"]).reset_index(drop=True)

    feature_frames = []
    for vehicle_id, group in df.groupby("vehicle_id", sort=False):
        group = group.copy()

        for col in SENSOR_COLUMNS:
            # Trailing rolling mean of the PAST window (shift(1) excludes
            # the current trip itself -- causal, no leakage from the row
            # we're trying to predict on).
            group[f"{col}_roll_mean"] = (
                group[col].shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
            )
            group[f"{col}_roll_trend"] = _rolling_slope(group[col], ROLLING_WINDOW)

        feature_frames.append(group)

    result = pd.concat(feature_frames, ignore_index=True)

    # Fill early-life NaNs (a vehicle's first few trips have no rolling
    # history yet) with the current value / zero trend -- a reasonable
    # default representing "no established trend yet".
    for col in SENSOR_COLUMNS:
        result[f"{col}_roll_mean"] = result[f"{col}_roll_mean"].fillna(result[col])
        result[f"{col}_roll_trend"] = result[f"{col}_roll_trend"].fillna(0.0)

    feature_cols = (
        SENSOR_COLUMNS
        + [f"{c}_roll_mean" for c in SENSOR_COLUMNS]
        + [f"{c}_roll_trend" for c in SENSOR_COLUMNS]
        + ["vehicle_age_days"]
    )

    output = result[["trip_id", "vehicle_id", "trip_start_time"] + feature_cols + [TARGET_COLUMN]].copy()
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    args = parser.parse_args()

    trips = pd.read_csv(args.data)
    features = build_maintenance_features(trips)
    print(f"Feature matrix shape: {features.shape}")
    print(f"\nExample: VEH0000's first 10 trips (early-life vs. established trend):")
    veh0 = features[features["vehicle_id"] == "VEH0000"].head(10)
    print(veh0[["trip_start_time", "avg_engine_temp_c", "avg_engine_temp_c_roll_mean",
                "avg_engine_temp_c_roll_trend"]].to_string(index=False))
    print(f"\nTarget positive rate: {features[TARGET_COLUMN].mean() * 100:.2f}%")
