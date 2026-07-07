"""
load_data.py
One-time (or re-runnable) ETL: loads Phase 0's trips.csv into PostgreSQL,
and bulk-computes risk scores + maintenance predictions for all historical
trips using the trained models -- far more efficient than calling the API
40,000 times over HTTP.

RUN THIS FROM THE HOST, NOT INSIDE THE DOCKER CONTAINER: it needs to import
feature engineering code from the sibling ml_pipeline/ folder (see
sys.path manipulation below), and connects to Postgres via the port
docker-compose exposes on localhost, not the internal container network.

Usage (run from fleet_api/, with the Postgres container already up):
    python load_data.py --trips ../data/trips.csv --ml-pipeline-dir ../ml_pipeline
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DEFAULT_DATABASE_URL = "postgresql://fleet_user:fleet_pass_dev_only@localhost:5432/fleet_telemetry"


def reset_tables(engine):
    """
    Truncates all data tables before loading, so this script is safe to
    re-run as many times as you want during development (e.g. after
    retraining a model, or just to get a clean slate) without needing to
    manually clean up a half-loaded database first.

    CASCADE on vehicles + drivers is enough -- every other table has a
    foreign key pointing (directly or transitively) back to one of these
    two, so Postgres cascades the truncate through trips, risk_scores,
    maintenance_predictions, anomaly_flags, alerts, and both
    sim_ground_truth tables automatically.
    """
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE fleet.vehicles, fleet.drivers CASCADE"))
    print("Reset: cleared all existing data (safe re-run, not an accumulating load)")


def load_trips_and_dimensions(engine, trips: pd.DataFrame):
    vehicles = pd.DataFrame({"vehicle_id": trips["vehicle_id"].unique()})
    drivers = pd.DataFrame({"driver_id": trips["driver_id"].unique()})

    vehicles.to_sql("vehicles", engine, schema="fleet", if_exists="append", index=False)
    drivers.to_sql("drivers", engine, schema="fleet", if_exists="append", index=False)
    print(f"Loaded {len(vehicles)} vehicles, {len(drivers)} drivers")

    trip_cols = [
        "trip_id", "vehicle_id", "driver_id", "route_id", "road_type", "trip_start_time",
        "duration_sec", "distance_km", "avg_speed_kmh", "max_speed_kmh", "speed_limit_kmh",
        "harsh_braking_count", "harsh_accel_count", "harsh_cornering_count",
        "overspeeding_duration_sec", "overspeeding_events", "idle_time_sec", "is_night",
        "avg_engine_temp_c", "min_oil_pressure_psi", "min_battery_voltage",
        "brake_efficiency_pct", "vehicle_age_days",
    ]
    trips_out = trips[trip_cols].copy()
    trips_out["trip_start_time"] = pd.to_datetime(trips_out["trip_start_time"])
    trips_out["is_night"] = trips_out["is_night"].astype(bool)
    trips_out.to_sql("trips", engine, schema="fleet", if_exists="append", index=False, chunksize=5000)
    print(f"Loaded {len(trips_out):,} trips")


def load_sim_ground_truth(engine, trips: pd.DataFrame):
    """Loads the hidden simulation labels into the SEPARATE sim_ground_truth
    schema -- never joined into the production `fleet` schema tables."""
    driver_personas = trips.groupby("driver_id")["persona"].first().reset_index()
    driver_personas.columns = ["driver_id", "persona"]
    driver_personas.to_sql("driver_personas", engine, schema="sim_ground_truth",
                            if_exists="append", index=False)

    vehicle_modes = trips.groupby("vehicle_id").agg(
        failure_mode=("vehicle_failure_mode", "first"),
    ).reset_index()
    vehicle_modes.to_sql("vehicle_failure_modes", engine, schema="sim_ground_truth",
                          if_exists="append", index=False)
    print("Loaded simulation ground truth (sim_ground_truth schema, validation-only)")


def bulk_score_risk(engine, trips: pd.DataFrame, feature_engineering, risk_scoring, risk_model, feature_names):
    features = feature_engineering.build_trip_features(trips)
    features_aligned = features.reindex(columns=feature_names, fill_value=0)
    scores = risk_model.predict(features_aligned)
    scores = np.clip(scores, 0, 100)
    tiers = risk_scoring.assign_risk_tier(pd.Series(scores, index=features.index))

    out = pd.DataFrame({
        "trip_id": features.index,
        "risk_score": np.round(scores, 2),
        "risk_tier": tiers.values,
    })
    out.to_sql("risk_scores", engine, schema="fleet", if_exists="append", index=False, chunksize=5000)
    print(f"Bulk-scored risk for {len(out):,} trips")


def bulk_score_maintenance(engine, trips: pd.DataFrame, maintenance_feature_engineering, maintenance_model):
    features = maintenance_feature_engineering.build_maintenance_features(trips)
    feature_cols = [c for c in features.columns
                    if c not in ("trip_id", "vehicle_id", "trip_start_time",
                                 maintenance_feature_engineering.TARGET_COLUMN)]
    proba = maintenance_model.predict_proba(features[feature_cols])[:, 1]

    def tier_of(p):
        if p < 0.25: return "Low"
        if p < 0.5: return "Medium"
        if p < 0.75: return "High"
        return "Critical"

    out = pd.DataFrame({
        "vehicle_id": features["vehicle_id"],
        "trip_id": features["trip_id"],
        "failure_probability": np.round(proba, 4),
        "risk_tier": [tier_of(p) for p in proba],
    })
    out.to_sql("maintenance_predictions", engine, schema="fleet", if_exists="append", index=False, chunksize=5000)
    print(f"Bulk-scored maintenance predictions for {len(out):,} trips")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=str, default="../data/trips.csv")
    parser.add_argument("--ml-pipeline-dir", type=str, default="../ml_pipeline")
    parser.add_argument("--database-url", type=str, default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--skip-bulk-scoring", action="store_true",
                         help="Load raw data only, skip risk/maintenance scoring")
    parser.add_argument("--no-reset", action="store_true",
                         help="Skip truncating existing data first (default: reset for a clean re-run)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.ml_pipeline_dir))
    import feature_engineering
    import risk_scoring
    import maintenance_feature_engineering
    import joblib

    engine = create_engine(args.database_url)

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Database connection OK")

    if not args.no_reset:
        reset_tables(engine)

    trips = pd.read_csv(args.trips)
    load_trips_and_dimensions(engine, trips)
    load_sim_ground_truth(engine, trips)

    if not args.skip_bulk_scoring:
        model_dir = os.path.join(args.ml_pipeline_dir, "model_output")
        risk_model = joblib.load(os.path.join(model_dir, "risk_regressor.joblib"))
        risk_feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
        maintenance_model = joblib.load(os.path.join(model_dir, "maintenance_classifier.joblib"))

        bulk_score_risk(engine, trips, feature_engineering, risk_scoring, risk_model, risk_feature_names)
        bulk_score_maintenance(engine, trips, maintenance_feature_engineering, maintenance_model)

    print("\nDone. Try: curl http://localhost:8000/fleet/summary")


if __name__ == "__main__":
    main()
