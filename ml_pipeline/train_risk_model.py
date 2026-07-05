"""
train_risk_model.py
Trains a Gradient Boosting model to predict the composite risk score from
trip features.

WHY TRAIN A MODEL TO PREDICT A FORMULA WE ALREADY COMPUTED?
This is a legitimate and common real-world pattern (sometimes called
"model distillation" or using a heuristic as weak supervision), not a
circular trick:
  1. The composite scorecard is precise but rigid -- it can't capture
     interactions (e.g. harsh braking AND night driving compounding risk
     more than either alone). A tree-based model learns these interactions
     automatically.
  2. Once trained, the model gives per-trip explainability (via feature
     attribution) that a fixed formula doesn't need, but a fleet manager
     dashboard benefits from showing anyway for consistency with future,
     more complex models.
  3. It's the same architecture you'd reuse later if the composite score
     were replaced with a real outcome label (accidents, claims, insurance
     payouts) -- only the training target changes, not the pipeline.

HONESTY NOTE FOR THE README / INTERVIEW: because the target IS a
deterministic function of a subset of the input features, near-perfect R^2
is EXPECTED here and is not evidence of a hard predictive problem solved.
The value of this step is the reusable pipeline + explainability layer,
not the R^2 number. Say this out loud if asked -- claiming otherwise is
the kind of overstatement that falls apart under a single follow-up
question in an interview.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.inspection import permutation_importance
import joblib

from feature_engineering import build_trip_features
from risk_scoring import compute_composite_risk_score, assign_risk_tier


def train_regressor(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 3),
        "r2": round(r2_score(y_test, preds), 4),
    }
    return model, metrics


def train_classifier(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "macro_f1": round(f1_score(y_test, preds, average="macro"), 4),
    }
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    return model, metrics, report, cm, model.classes_


def run_permutation_importance(model, X_test, y_test, feature_names):
    """
    Sandbox-friendly explainability cross-check. The primary explainability
    deliverable is SHAP (explainability.py) -- run that locally where
    `pip install shap` has internet access. Permutation importance is a
    model-agnostic sanity check that should broadly agree with SHAP's
    global feature ranking.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return importance_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    parser.add_argument("--out", type=str, default="./model_output")
    args = parser.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    trips = pd.read_csv(args.data)
    features = build_trip_features(trips)
    scores = compute_composite_risk_score(features)
    tiers = assign_risk_tier(scores)

    feature_names = features.columns.tolist()
    X = features.values
    y_reg = scores.values
    y_clf = tiers.values

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("=" * 70)
    print("REGRESSOR: predicting continuous risk score (0-100)")
    print("=" * 70)
    reg_model, reg_metrics = train_regressor(X_train, y_reg_train, X_test, y_reg_test)
    print(json.dumps(reg_metrics, indent=2))

    print("\n" + "=" * 70)
    print("CLASSIFIER: predicting risk tier (Low/Medium/High/Critical)")
    print("=" * 70)
    clf_model, clf_metrics, report, cm, classes = train_classifier(
        X_train, y_clf_train, X_test, y_clf_test
    )
    print(json.dumps(clf_metrics, indent=2))
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=classes, columns=classes))

    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE (regressor) -- sandbox explainability check")
    print("=" * 70)
    imp_df = run_permutation_importance(reg_model, X_test, y_reg_test, feature_names)
    print(imp_df.to_string(index=False))

    joblib.dump(reg_model, f"{args.out}/risk_regressor.joblib")
    joblib.dump(clf_model, f"{args.out}/risk_classifier.joblib")
    joblib.dump(feature_names, f"{args.out}/feature_names.joblib")
    imp_df.to_csv(f"{args.out}/permutation_importance.csv", index=False)

    print(f"\nModels and importance table saved to {args.out}/")
