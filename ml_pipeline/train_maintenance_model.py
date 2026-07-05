"""
train_maintenance_model.py
Trains a binary classifier to predict `will_fail_within_30d`.

WHY THE TRAIN/TEST SPLIT IS BY VEHICLE, NOT BY TRIP:
Adjacent trips from the same vehicle share nearly identical rolling-window
features (that's the point of a rolling window). A random trip-level split
would put highly-correlated rows on both sides of the split, inflating
test performance in a way that wouldn't hold up on a genuinely new
vehicle. Splitting by vehicle_id means the test set contains vehicles the
model has NEVER seen in any form during training -- this is the honest
question a real deployment needs answered: "does this generalize to a new
car joining the fleet," not "can it memorize this car's history."

WHY SAMPLE WEIGHTING INSTEAD OF RESAMPLING:
With ~21% positive rate, class weighting is enough here (not so extreme
that SMOTE/undersampling is required) and keeps every real observation in
the training set rather than synthesizing or discarding data.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve,
)
from sklearn.inspection import permutation_importance
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maintenance_feature_engineering import build_maintenance_features, TARGET_COLUMN


def vehicle_level_split(trips: pd.DataFrame, test_size=0.3, random_state=42):
    """Splits VEHICLES (not trips) into train/test, stratified by failure
    mode so both splits contain a representative mix of failure types."""
    vehicle_labels = trips.groupby("vehicle_id")["vehicle_failure_mode"].first()
    train_vehicles, test_vehicles = train_test_split(
        vehicle_labels.index, test_size=test_size, random_state=random_state,
        stratify=vehicle_labels.values,
    )
    return set(train_vehicles), set(test_vehicles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    parser.add_argument("--out", type=str, default="./model_output")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    trips = pd.read_csv(args.data)
    features = build_maintenance_features(trips)

    train_vehicles, test_vehicles = vehicle_level_split(trips)
    print(f"Train vehicles: {len(train_vehicles)}, Test vehicles: {len(test_vehicles)}")

    train_mask = features["vehicle_id"].isin(train_vehicles)
    test_mask = features["vehicle_id"].isin(test_vehicles)

    feature_cols = [c for c in features.columns
                    if c not in ("trip_id", "vehicle_id", "trip_start_time", TARGET_COLUMN)]

    X_train, y_train = features.loc[train_mask, feature_cols], features.loc[train_mask, TARGET_COLUMN]
    X_test, y_test = features.loc[test_mask, feature_cols], features.loc[test_mask, TARGET_COLUMN]

    print(f"Train trips: {len(X_train):,} (positive rate {y_train.mean()*100:.1f}%)")
    print(f"Test trips:  {len(X_test):,} (positive rate {y_test.mean()*100:.1f}%)")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print("\n" + "=" * 70)
    print("MAINTENANCE CLASSIFIER RESULTS (evaluated on UNSEEN vehicles)")
    print("=" * 70)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (average precision): {pr_auc:.4f}  <- more informative than ROC-AUC here given class imbalance")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, preds),
                        index=["Actual: No", "Actual: Yes"],
                        columns=["Pred: No", "Pred: Yes"]))

    # Precision-Recall curve -- more useful than ROC for this imbalance level,
    # and shows the operating-point tradeoff a fleet manager would actually
    # need to choose (e.g. "flag more vehicles, accept lower precision").
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (PR-AUC={pr_auc:.3f})\nEvaluated on vehicles unseen during training")
    plt.tight_layout()
    plt.savefig(f"{args.out}/maintenance_pr_curve.png", dpi=120)
    plt.close()

    imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": imp.importances_mean,
    }).sort_values("importance_mean", ascending=False)
    print("\nPermutation importance:")
    print(imp_df.to_string(index=False))

    joblib.dump(model, f"{args.out}/maintenance_classifier.joblib")
    joblib.dump(feature_cols, f"{args.out}/maintenance_feature_names.joblib")
    imp_df.to_csv(f"{args.out}/maintenance_permutation_importance.csv", index=False)

    metrics = {"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
               "train_vehicles": len(train_vehicles), "test_vehicles": len(test_vehicles)}
    with open(f"{args.out}/maintenance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel, importance table, and PR curve saved to {args.out}/")


if __name__ == "__main__":
    main()
