"""
feature_engineering.py
Builds the trip-level feature matrix used by the driver risk-scoring model.

CRITICAL RULE: `persona` and `vehicle_failure_mode` (and anything derived
from them, like `days_to_failure`/`will_fail_within_30d`) are ground-truth
labels that exist ONLY because we control the simulator. A real production
system would never have these at scoring time. They are used elsewhere
(validate_against_ground_truth.py) purely to sanity-check the model's
output -- never as model inputs.
"""

import pandas as pd
import numpy as np

# Columns that must NEVER be used as model features -- ground truth / leakage.
LEAKAGE_COLUMNS = [
    "persona", "vehicle_failure_mode", "days_to_failure", "will_fail_within_30d",
]

ID_COLUMNS = ["trip_id", "driver_id", "vehicle_id", "route_id", "trip_start_time"]


def build_trip_features(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by trip_id with engineered, model-ready
    behavioral features. Rate-based features (per 100km) are used instead of
    raw counts so a 2-minute trip and a 2-hour trip are comparable -- raw
    counts alone would just reflect trip length, not driving behavior.
    """
    df = trips.copy()

    # Avoid divide-by-zero for very short trips
    safe_distance = df["distance_km"].clip(lower=0.1)
    safe_duration_min = (df["duration_sec"] / 60).clip(lower=0.5)

    features = pd.DataFrame(index=df.index)
    features["trip_id"] = df["trip_id"]

    # --- Behavioral rate features (the core risk signal) ---
    features["harsh_braking_per_100km"] = df["harsh_braking_count"] / safe_distance * 100
    features["harsh_accel_per_100km"] = df["harsh_accel_count"] / safe_distance * 100
    features["harsh_cornering_per_100km"] = df["harsh_cornering_count"] / safe_distance * 100
    features["overspeeding_events_per_100km"] = df["overspeeding_events"] / safe_distance * 100
    features["overspeeding_time_ratio"] = df["overspeeding_duration_sec"] / df["duration_sec"].clip(lower=1)

    # --- Speed profile ---
    features["avg_speed_ratio"] = df["avg_speed_kmh"] / df["speed_limit_kmh"]
    features["max_speed_ratio"] = df["max_speed_kmh"] / df["speed_limit_kmh"]
    features["speed_excess"] = (features["avg_speed_ratio"] - 1.0).clip(lower=0)

    # --- Context features (not risk signals themselves, but useful controls) ---
    features["is_night"] = df["is_night"]
    features["idle_ratio"] = df["idle_time_sec"] / df["duration_sec"].clip(lower=1)
    features["trip_distance_km"] = df["distance_km"]
    features["trip_duration_min"] = safe_duration_min

    # --- Road type one-hot (categorical context) ---
    road_dummies = pd.get_dummies(df["road_type"], prefix="road")
    features = pd.concat([features, road_dummies], axis=1)

    features = features.set_index("trip_id")
    return features


def get_feature_columns(features_df: pd.DataFrame) -> list:
    """All columns in the engineered feature set are valid model inputs --
    leakage columns are never added to this DataFrame in the first place."""
    return features_df.columns.tolist()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    args = parser.parse_args()

    trips = pd.read_csv(args.data)
    features = build_trip_features(trips)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature columns: {get_feature_columns(features)}")
    print(features.describe().T)
