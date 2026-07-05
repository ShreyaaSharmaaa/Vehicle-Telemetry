"""
risk_scoring.py
Computes a transparent, weighted composite Trip Risk Score (0-100), using
the same scorecard methodology real usage-based-insurance telematics
products use (e.g. Progressive Snapshot-style scoring): each risk-relevant
feature is capped, linearly scaled to points, then combined with
domain-assigned weights.

This is deliberately NOT a black box -- given any trip's raw features, you
can compute its score by hand with a calculator. That's the point: it's the
audit-able baseline against which the learned model (train_risk_model.py)
is compared and validated.

Caps are set near the 95th percentile of each metric in the training data,
so the scoring is sensitive across the realistic range of driving behavior
without being dominated by rare, extreme outlier trips.
"""

import pandas as pd
import numpy as np

# (feature, cap, weight) -- weights sum to 1.0
SCORECARD = [
    ("harsh_braking_per_100km", 45.0, 0.22),
    ("harsh_accel_per_100km", 35.0, 0.18),
    ("harsh_cornering_per_100km", 240.0, 0.15),
    ("overspeeding_events_per_100km", 190.0, 0.15),
    ("overspeeding_time_ratio", 0.6, 0.20),
    ("speed_excess", 0.3, 0.10),
]

NIGHT_DRIVING_BONUS_POINTS = 5.0  # small additive risk bump, not part of the 1.0 weight sum

RISK_TIERS = [
    (0, 25, "Low"),
    (25, 50, "Medium"),
    (50, 75, "High"),
    (75, 100.01, "Critical"),
]


def compute_composite_risk_score(features: pd.DataFrame) -> pd.Series:
    """Returns a Series of risk scores in [0, 100], indexed like `features`."""
    score = pd.Series(0.0, index=features.index)

    for col, cap, weight in SCORECARD:
        points = (features[col].clip(lower=0, upper=cap) / cap) * 100
        score += points * weight

    if "is_night" in features.columns:
        score += features["is_night"] * NIGHT_DRIVING_BONUS_POINTS

    return score.clip(upper=100.0)


def assign_risk_tier(scores: pd.Series) -> pd.Series:
    tiers = pd.Series(index=scores.index, dtype=object)
    for lo, hi, label in RISK_TIERS:
        mask = (scores >= lo) & (scores < hi)
        tiers[mask] = label
    return tiers


def score_breakdown(single_trip_features: pd.Series) -> pd.DataFrame:
    """
    Returns a per-feature point contribution breakdown for ONE trip --
    this is what a fleet manager dashboard would show as "why this trip
    scored what it scored", without needing any ML model at all.
    """
    rows = []
    for col, cap, weight in SCORECARD:
        raw_value = single_trip_features[col]
        clipped = min(max(raw_value, 0), cap)
        points = (clipped / cap) * 100
        contribution = points * weight
        rows.append({
            "feature": col, "raw_value": round(raw_value, 3),
            "cap": cap, "weight": weight,
            "points_0_100": round(points, 1),
            "weighted_contribution": round(contribution, 2),
        })
    if "is_night" in single_trip_features.index and single_trip_features["is_night"]:
        rows.append({
            "feature": "is_night", "raw_value": 1, "cap": 1, "weight": None,
            "points_0_100": None, "weighted_contribution": NIGHT_DRIVING_BONUS_POINTS,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from feature_engineering import build_trip_features

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    args = parser.parse_args()

    trips = pd.read_csv(args.data)
    features = build_trip_features(trips)
    scores = compute_composite_risk_score(features)
    tiers = assign_risk_tier(scores)

    print("Composite Risk Score distribution:")
    print(scores.describe().round(2))
    print()
    print("Risk tier distribution:")
    print(tiers.value_counts())
    print()
    print("Example breakdown for the single highest-scoring trip:")
    top_idx = scores.idxmax()
    print(score_breakdown(features.loc[top_idx]).to_string(index=False))
