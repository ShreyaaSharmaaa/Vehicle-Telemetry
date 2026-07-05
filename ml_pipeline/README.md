# Phase 1: Driver Risk Scoring Pipeline

## What's in this phase

| File | Purpose |
|---|---|
| `eda.py` | Validates the simulator's behavioral and degradation patterns visually |
| `feature_engineering.py` | Builds the leakage-free, model-ready feature set |
| `risk_scoring.py` | Transparent, weighted composite Trip Risk Score (0-100) |
| `validate_against_ground_truth.py` | Checks the score/features against hidden persona labels |
| `train_risk_model.py` | Gradient Boosting regressor + classifier, trained on the composite score |
| `explainability.py` | SHAP-based per-trip and global explanations (**run locally**, needs `pip install shap`) |

## How to run (in order)

```bash
pip install pandas numpy scikit-learn matplotlib joblib shap

python eda.py --data ./data/trips.csv --out ./eda_output
python feature_engineering.py --data ./data/trips.csv
python risk_scoring.py --data ./data/trips.csv
python validate_against_ground_truth.py --data ./data/trips.csv --out ./eda_output
python train_risk_model.py --data ./data/trips.csv --out ./model_output
python explainability.py --data ./data/trips.csv --model_dir ./model_output --out ./model_output
```

## Key design decisions (and why)

**Leakage prevention.** `persona` and `vehicle_failure_mode` (and anything
derived from them) are ground-truth labels that only exist because we
control the simulator. `feature_engineering.py` never includes them as
model inputs — they're used only in `validate_against_ground_truth.py`,
strictly for post-hoc validation.

**Rate-based features, not raw counts.** `harsh_braking_count` alone would
just reflect trip length. `harsh_braking_per_100km` is comparable across a
2-minute trip and a 2-hour trip.

**Composite score as a transparent scorecard, not a black box.** The
weights and caps in `risk_scoring.py` are domain-assigned (similar to
real usage-based-insurance telematics scoring), not learned. Any trip's
score can be recomputed by hand from `score_breakdown()`. This is the
audit trail a fleet manager or regulator would actually want.

**The Gradient Boosting model predicts the composite score, not a
mystery target.** This is intentional (model distillation / weak
supervision) — see the docstring in `train_risk_model.py` for the full
reasoning. The near-perfect R² (0.997) this produces is **expected and
not a meaningful accuracy claim** — say so directly if asked in an
interview. Its value is the reusable pipeline (feature engineering →
training → SHAP explainability) that would carry over unchanged if the
target were replaced with real accident/claims data.

## Honest findings from validation (don't skip this section)

- **Composite score correctly ranks personas** the formula never saw:
  calm (6.2) < average (16.3) < drowsy_distracted (26.2) < aggressive
  (35.6). Drowsy scoring above average despite not being the fastest
  persona is the expected signature of inconsistency-driven risk.
- **Naive unsupervised KMeans clustering failed** to recover persona
  structure (Adjusted Rand Index ≈ 0.015). Diagnosis: 51-72% of trips have
  *zero* harsh events across every category — this is classic zero-inflated
  telematics data, and Euclidean-distance clustering gets dominated by
  continuous features (speed ratios, trip length) rather than the sparse
  events that actually carry the risk signal. This is a known limitation
  in telematics analytics, and is part of *why* domain-weighted scorecards
  remain the industry standard rather than naive unsupervised approaches.
  A documented "future work" item, not a hidden failure.

## What Phase 2 will change

The maintenance/anomaly modules (predictive maintenance classifier,
Isolation Forest anomaly detection) are separate from this risk-scoring
pipeline and haven't been built yet — that's the next phase.
