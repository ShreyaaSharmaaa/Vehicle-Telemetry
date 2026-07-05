# Phase 2: Predictive Maintenance + Anomaly Detection

## What's in this phase

| File | Purpose |
|---|---|
| `maintenance_feature_engineering.py` | Causal rolling-window trend features per vehicle (no temporal leakage) |
| `train_maintenance_model.py` | Gradient Boosting classifier, split by VEHICLE not by trip |
| `anomaly_detection.py` | Isolation Forest + hybrid rule-based GPS-dropout detection |

## How to run

```bash
python maintenance_feature_engineering.py --data ..\data\trips.csv
python train_maintenance_model.py --data ..\data\trips.csv --out .\model_output

python anomaly_detection.py --telemetry ..\data\telemetry_recent.csv --anomalies ..\data\anomaly_events.csv --out .\model_output
```

## IMPORTANT: regenerate a larger telemetry sample before running anomaly_detection.py

Your existing `telemetry_recent.csv` from Phase 0 only contains ~5 ground-truth
anomaly events (it was deliberately size-limited to 8 vehicles / 7 days for a
shippable file). That's too few to evaluate anything meaningfully — precision/
recall on 5 events is statistical noise. Regenerate a larger sample specifically
for this phase (this will produce a ~350MB file, so don't commit it to git):

```powershell
cd ..\fleet_simulator
python generate_dataset.py --vehicles 50 --drivers 30 --days 365 --raw_window_days 30 --out .\data_anomaly_test
cd ..\ml_pipeline
python anomaly_detection.py --telemetry ..\fleet_simulator\data_anomaly_test\telemetry_recent.csv --anomalies ..\fleet_simulator\data_anomaly_test\anomaly_events.csv --out .\model_output
```

This gives ~185 ground-truth events (this is the exact sample the "Honest
results" section below was validated on) — enough for the recall-by-type
breakdown to be meaningful rather than noisy. Expect this to generate a
~500MB telemetry file and take ~2 minutes to run; this is expected and
fine for a one-time validation run, just don't commit that file to git.

## Key design decisions (and why)

**Two different leakage risks, not one.** Phase 1 only had to worry about
label leakage (never feed `persona` to the model). Predictive maintenance
adds TEMPORAL leakage: adjacent trips from the same vehicle share nearly
identical rolling-window features, so a random trip-level train/test split
would leak information between train and test. `train_maintenance_model.py`
splits by **vehicle**, stratified by failure mode — the test set contains
vehicles the model has never seen in any form. This is the honest question
a real deployment needs answered: does this generalize to a new vehicle
joining the fleet.

**Rolling features are causal.** `maintenance_feature_engineering.py` uses
`.shift(1)` before computing rolling means/trends, so a trip's own value
never leaks into its own trend feature — only genuinely past trips.

**PR-AUC over ROC-AUC for reporting.** Both are shown, but with ~21%
(maintenance) and ~0.01% (anomaly) positive rates, PR-AUC is the more
honest metric — ROC-AUC can look deceptively strong under heavy imbalance.

## Honest results

**Predictive maintenance**: ROC-AUC 0.995, PR-AUC 0.993, evaluated on 15
vehicles never seen during training. Engine-temp trend and brake efficiency
dominate feature importance — exactly the two sensors tied to the two most
common failure modes in the simulator. This is a strong result, but it's a
clean synthetic benchmark; real sensor data is noisier, and this number
should be read as "the pipeline works correctly," not "predictive
maintenance is a solved problem."

**Anomaly detection — this one took real debugging, documented here on
purpose:**
1. First attempt included raw speed/RPM/gyro/throttle features. Result:
   0% recall at any usable threshold, despite a plausible-looking ROC-AUC.
   Diagnosis: normal aggressive driving (harsh braking, hard cornering)
   produced larger statistical outliers than the actual injected sensor
   anomalies. Isolation Forest was finding real outliers — just not the
   ones that mattered.
2. Restricting features to sensor-health signals (engine temp, oil
   pressure, battery voltage, brake efficiency) plus GPS integrity and a
   dedicated impact-delta improved ROC-AUC from 0.71 to 0.80.
3. GPS dropout (a deterministic, binary condition occurring in ~0.02% of
   rows) was still caught poorly by the multivariate model — too sparse to
   reliably isolate among 13 other continuous features. Added a hybrid
   rule (`is_gps_missing` → always flag) instead of forcing the ML model
   to learn something a one-line check already solves perfectly. Recall
   went from 4.4% to 100% for that anomaly type specifically.
4. Final per-type recall at a top-2%-of-rows review threshold, validated on
   a larger sample (50 vehicles, 30-day raw window, 185 ground-truth events
   — large enough for the per-type breakdown to be meaningful):
   gps_dropout 100% (rule-based), impact_event 88.9%, sensor_spike 63.6%,
   route_deviation 16.7%. Route deviation is the hardest type by a wide
   margin — it's a gradual GPS drift, closer to "normal variation" than a
   sharp spike, and would likely need a dedicated geofencing/expected-route
   comparison approach rather than general-purpose outlier detection.
   Overall PR-AUC 0.0057 against a random baseline of 0.00013 (44.6x
   better than chance) — the raw PR-AUC number looks unimpressive in
   isolation precisely because true anomalies are ~0.01% of all rows; the
   multiplier over baseline and the per-type recall are the metrics that
   actually mean something here, not the raw PR-AUC value.

**Takeaway worth repeating in an interview**: the debugging process here
(catching that the model was technically "working" but solving the wrong
problem) is more valuable to talk about than the final numbers. Anyone can
report a PR-AUC; fewer people can explain why their first feature set
failed and what that revealed about the problem.

## What Phase 3 will cover

FastAPI backend serving these models, PostgreSQL schema design, and wiring
the trained `.joblib` models into request/response endpoints.
