"""
anomalies.py
Injects sudden, discrete anomalies into an otherwise-normal trip's
telemetry, and returns ground-truth labeled anomaly event records.

These are intentionally distinct from the gradual degradation modeled in
vehicle.py: degradation is slow drift (predictive maintenance target),
anomalies are sudden and rare (anomaly-detection target). Keeping the two
mechanisms separate avoids leaking maintenance labels into the anomaly
detector's evaluation set, which would make the anomaly module look
artificially good.
"""

import numpy as np
from config import ANOMALY_TYPES


def maybe_inject_anomalies(telemetry, trip_id: str, rng: np.random.Generator):
    """
    Mutates `telemetry` in place with injected anomalies (small probability
    per trip, per anomaly type). Returns a list of anomaly event dicts
    (empty if none injected this trip).
    """
    events = []
    n = len(telemetry)
    if n < 10:
        return events

    for anomaly_type, cfg in ANOMALY_TYPES.items():
        if rng.random() >= cfg["prob_per_trip"]:
            continue

        idx = int(rng.integers(5, n - 5))

        if anomaly_type == "gps_dropout":
            span = slice(idx, min(idx + rng.integers(3, 8), n))
            telemetry.loc[telemetry.index[span], "gps_lat"] = np.nan
            telemetry.loc[telemetry.index[span], "gps_lon"] = np.nan

        elif anomaly_type == "sensor_spike":
            sensor = rng.choice(["oil_pressure_psi", "battery_voltage", "engine_temp_c"])
            spike_dir = rng.choice([-1, 1])
            telemetry.loc[telemetry.index[idx], sensor] += spike_dir * rng.uniform(15, 40)

        elif anomaly_type == "impact_event":
            telemetry.loc[telemetry.index[idx], "acceleration_x_g"] = -rng.uniform(0.8, 1.5)
            telemetry.loc[telemetry.index[idx], "speed_kmh"] *= 0.3

        elif anomaly_type == "route_deviation":
            span = slice(idx, min(idx + rng.integers(10, 30), n))
            drift = rng.uniform(0.01, 0.03) * rng.choice([-1, 1])
            telemetry.loc[telemetry.index[span], "gps_lon"] += drift

        events.append({
            "trip_id": trip_id,
            "anomaly_type": anomaly_type,
            "timestamp": telemetry.iloc[idx]["timestamp"],
            "index_in_trip": idx,
        })

    return events
