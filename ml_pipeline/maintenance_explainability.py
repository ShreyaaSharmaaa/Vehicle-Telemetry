"""
maintenance_explainability.py
SHAP explanations for the predictive maintenance classifier.

Same rationale as Phase 1's explainability.py: run this locally (needs
`pip install shap`, which needs internet access this sandbox doesn't have).

WHAT TO LOOK FOR:
  - Global summary should show engine-temp and brake-efficiency rolling
    features dominating, consistent with permutation_importance.csv from
    train_maintenance_model.py. If SHAP disagrees strongly with permutation
    importance here, investigate before trusting either.
  - The per-vehicle waterfall plot is the actual product feature: "why did
    this vehicle get flagged as at-risk of failure in the next 30 days" is
    exactly what a fleet manager needs to see to trust and act on an alert.
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

from maintenance_feature_engineering import build_maintenance_features, TARGET_COLUMN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    parser.add_argument("--model_dir", type=str, default="./model_output")
    parser.add_argument("--out", type=str, default="./model_output")
    parser.add_argument("--example_index", type=int, default=None,
                         help="Row index to explain individually. Defaults to the "
                              "highest-predicted-risk trip in the sample.")
    args = parser.parse_args()

    model = joblib.load(f"{args.model_dir}/maintenance_classifier.joblib")
    feature_cols = joblib.load(f"{args.model_dir}/maintenance_feature_names.joblib")

    trips = pd.read_csv(args.data)
    features = build_maintenance_features(trips)
    X = features[feature_cols]

    explainer = shap.TreeExplainer(model)
    sample = X.sample(n=min(3000, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)
    # GradientBoostingClassifier binary case: shap_values may come back as a
    # list [class0, class1] depending on SHAP version -- normalize to the
    # positive-class contributions, which is what we actually care about.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{args.out}/maintenance_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary to {args.out}/maintenance_shap_summary.png")

    if args.example_index is None:
        proba_sample = model.predict_proba(sample)[:, 1]
        example_pos = proba_sample.argmax()
    else:
        example_pos = args.example_index

    single = sample.iloc[[example_pos]]
    single_shap = explainer.shap_values(single)
    if isinstance(single_shap, list):
        single_shap = single_shap[1]

    base_value = explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=single_shap[0], base_values=base_value,
            data=single.iloc[0].values, feature_names=feature_cols,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{args.out}/maintenance_shap_waterfall_example.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved per-vehicle SHAP waterfall to {args.out}/maintenance_shap_waterfall_example.png")

    predicted_proba = model.predict_proba(single)[0, 1]
    print(f"\nExample trip predicted failure-within-30-days probability: {predicted_proba:.3f}")
    print("Top SHAP contributors:")
    contributions = pd.Series(single_shap[0], index=feature_cols).sort_values(key=abs, ascending=False)
    print(contributions.head(6).round(3))


if __name__ == "__main__":
    main()
