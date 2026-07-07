"""
app/feature_engineering.py
Replicates the training-time feature engineering from Phase 1
(feature_engineering.py) and Phase 2 (maintenance_feature_engineering.py)
at serving time. Kept self-contained in the API rather than importing
across the ml_pipeline folder, since the API is a separately deployable
service -- but the LOGIC must stay identical to what the models were
trained on, or predictions will be silently wrong. If you ever change
the training-side feature engineering, mirror the change here too.

DEFENSIVE DESIGN: rather than trusting column order by convention, every
function below reindexes its output against the exact feature-name list
saved alongside the trained model (feature_names.joblib /
maintenance_feature_names.joblib). This means a training-time feature
reordering can't silently corrupt serving-time predictions -- a mismatch
either self-corrects via reindex or produces an explicit column of NaNs
you'd notice immediately, rather than a silent misalignment.
"""

from typing import List

import numpy as np
import pandas as pd

MAINTENANCE_SENSOR_COLUMNS = [
    "avg_engine_temp_c", "min_oil_pressure_psi",
    "min_battery_voltage", "brake_efficiency_pct",
]
MAINTENANCE_ROLLING_WINDOW = 10


def build_single_trip_risk_features(trip: dict, feature_names: List[str]) -> pd.DataFrame:
    """Mirrors feature_engineering.build_trip_features, for exactly one trip."""
    distance = max(trip["distance_km"], 0.1)
    duration_min = max(trip["duration_sec"] / 60, 0.5)

    row = {
        "harsh_braking_per_100km": trip["harsh_braking_count"] / distance * 100,
        "harsh_accel_per_100km": trip["harsh_accel_count"] / distance * 100,
        "harsh_cornering_per_100km": trip["harsh_cornering_count"] / distance * 100,
        "overspeeding_events_per_100km": trip["overspeeding_events"] / distance * 100,
        "overspeeding_time_ratio": trip["overspeeding_duration_sec"] / max(trip["duration_sec"], 1),
        "avg_speed_ratio": trip["avg_speed_kmh"] / trip["speed_limit_kmh"],
        "max_speed_ratio": trip["max_speed_kmh"] / trip["speed_limit_kmh"],
        "is_night": int(trip["is_night"]),
        "idle_ratio": trip["idle_time_sec"] / max(trip["duration_sec"], 1),
        "trip_distance_km": trip["distance_km"],
        "trip_duration_min": duration_min,
        "road_urban": 1 if trip.get("road_type") == "urban" else 0,
        "road_highway": 1 if trip.get("road_type") == "highway" else 0,
        "road_mixed": 1 if trip.get("road_type") == "mixed" else 0,
    }
    row["speed_excess"] = max(row["avg_speed_ratio"] - 1.0, 0)

    df = pd.DataFrame([row])
    # Reindex against the exact training-time column list -- any column the
    # model expects but we didn't compute becomes 0 (safe default for a
    # one-hot/rate feature); any column we computed but the model doesn't
    # expect is dropped.
    return df.reindex(columns=feature_names, fill_value=0)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope(x):
        if len(x) < 2:
            return 0.0
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]
    return series.shift(1).rolling(window, min_periods=2).apply(slope, raw=True)


def build_maintenance_features_from_history(
    trip_history: pd.DataFrame, feature_names: List[str]
) -> pd.DataFrame:
    """
    `trip_history` must be sorted ascending by trip_start_time and contain
    at least MAINTENANCE_SENSOR_COLUMNS + vehicle_age_days, with the trip
    being scored as the LAST row. Returns a single-row feature DataFrame
    for that last row, using only trips strictly before it (shift(1)) for
    rolling context -- identical causal-leakage discipline as training.
    """
    df = trip_history.copy().reset_index(drop=True)

    for col in MAINTENANCE_SENSOR_COLUMNS:
        df[f"{col}_roll_mean"] = (
            df[col].shift(1).rolling(MAINTENANCE_ROLLING_WINDOW, min_periods=1).mean()
        )
        df[f"{col}_roll_trend"] = _rolling_slope(df[col], MAINTENANCE_ROLLING_WINDOW)

    for col in MAINTENANCE_SENSOR_COLUMNS:
        df[f"{col}_roll_mean"] = df[f"{col}_roll_mean"].fillna(df[col])
        df[f"{col}_roll_trend"] = df[f"{col}_roll_trend"].fillna(0.0)

    last_row = df.iloc[[-1]]
    return last_row.reindex(columns=feature_names, fill_value=0)
