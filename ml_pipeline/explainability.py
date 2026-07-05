"""
explainability.py
Generates SHAP explanations for the trained risk regressor.

NOTE ON RUNNING THIS: SHAP requires `pip install shap`, which needs
internet access. Run this on your own machine, not in a restricted sandbox.

WHAT TO LOOK FOR:
  1. The global summary plot should roughly match permutation_importance.csv
     from train_risk_model.py (harsh_braking_per_100km and
     overspeeding_time_ratio should dominate). If SHAP's ranking looks
     wildly different from permutation importance, something's wrong --
     investigate before trusting either.
  2. The per-trip force/waterfall plot is the actual product feature: this
     is what powers a "why did this trip get flagged as high risk" panel
     in a fleet manager dashboard.
"""

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import shap
except ImportError:
    raise SystemExit(
        "shap is not installed. Run: pip install shap\n"
        "This script must be run on a machine with internet access."
    )

from feature_engineering import build_trip_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    parser.add_argument("--model_dir", type=str, default="./model_output")
    parser.add_argument("--out", type=str, default="./model_output")
    parser.add_argument("--example_trip_index", type=int, default=0,
                         help="Row index (in the feature matrix) to explain individually")
    args = parser.parse_args()

    model = joblib.load(f"{args.model_dir}/risk_regressor.joblib")
    feature_names = joblib.load(f"{args.model_dir}/feature_names.joblib")

    trips = pd.read_csv(args.data)
    features = build_trip_features(trips)
    features = features[feature_names]  # enforce training column order

    # SHAP on tree ensembles is exact and fast via TreeExplainer -- no need
    # for the slower model-agnostic KernelExplainer here.
    explainer = shap.TreeExplainer(model)

    # Use a sample for the summary plot -- full dataset works but is slower
    # and the summary pattern converges well before using all 40k rows.
    sample = features.sample(n=min(3000, len(features)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # --- Global summary: which features drive risk scores overall ---
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{args.out}/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary to {args.out}/shap_summary.png")

    # --- Local explanation: why did ONE specific trip score what it did ---
    single_trip = features.iloc[[args.example_trip_index]]
    single_shap = explainer.shap_values(single_trip)

    # Newer SHAP versions return expected_value as a length-1 array for
    # single-output regressors rather than a plain float -- normalize it
    # here so waterfall_plot's float() cast doesn't choke on it.
    base_value = explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[0]

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=single_shap[0],
            base_values=base_value,
            data=single_trip.iloc[0].values,
            feature_names=feature_names,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{args.out}/shap_waterfall_example_trip.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved per-trip SHAP waterfall to {args.out}/shap_waterfall_example_trip.png")

    predicted_score = model.predict(single_trip)[0]
    print(f"\nExample trip predicted risk score: {predicted_score:.1f}")
    print("Top SHAP contributors for this trip:")
    contributions = pd.Series(single_shap[0], index=feature_names).sort_values(key=abs, ascending=False)
    print(contributions.head(6).round(2))


if __name__ == "__main__":
    main()
