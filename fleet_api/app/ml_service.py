"""
app/ml_service.py
Loads the trained .joblib models once at application startup (not per
request -- deserializing a model on every call would add unnecessary
latency and is a common rookie mistake in ML-serving code) and exposes
simple prediction functions for the API routes to call.

MODEL_DIR is expected to contain, copied over from the ml_pipeline outputs:
  risk_regressor.joblib, feature_names.joblib
  maintenance_classifier.joblib, maintenance_feature_names.joblib
"""

import os

import joblib
import pandas as pd

from app.feature_engineering import (
    build_single_trip_risk_features,
    build_maintenance_features_from_history,
)

MODEL_DIR = os.getenv("MODEL_DIR", "./model_output")

RISK_TIERS = [(0, 25, "Low"), (25, 50, "Medium"), (50, 75, "High"), (75, 100.01, "Critical")]
MAINTENANCE_TIERS = [(0, 0.25, "Low"), (0.25, 0.5, "Medium"), (0.5, 0.75, "High"), (0.75, 1.01, "Critical")]


def _assign_tier(value: float, tiers: list) -> str:
    for lo, hi, label in tiers:
        if lo <= value < hi:
            return label
    return tiers[-1][2]


class ModelRegistry:
    """Loaded once at app startup, held for the app's lifetime."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.risk_model = joblib.load(os.path.join(model_dir, "risk_regressor.joblib"))
        self.risk_feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))

        self.maintenance_model = joblib.load(os.path.join(model_dir, "maintenance_classifier.joblib"))
        self.maintenance_feature_names = joblib.load(
            os.path.join(model_dir, "maintenance_feature_names.joblib"))

    def score_trip_risk(self, trip: dict) -> dict:
        features = build_single_trip_risk_features(trip, self.risk_feature_names)
        score = float(self.risk_model.predict(features)[0])
        score = max(0.0, min(100.0, score))
        return {"risk_score": round(score, 2), "risk_tier": _assign_tier(score, RISK_TIERS)}

    def predict_maintenance(self, trip_history: pd.DataFrame) -> dict:
        """
        `trip_history` -- see build_maintenance_features_from_history for
        the exact expected shape. Returns failure probability for the LAST
        trip in the history (i.e. "as of the vehicle's most recent trip").
        """
        features = build_maintenance_features_from_history(trip_history, self.maintenance_feature_names)
        proba = float(self.maintenance_model.predict_proba(features)[0, 1])
        return {
            "failure_probability": round(proba, 4),
            "risk_tier": _assign_tier(proba, MAINTENANCE_TIERS),
        }


# Singleton, created once when the module is first imported (at app startup).
registry = ModelRegistry()
