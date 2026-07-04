"""
vehicle.py
Defines the Vehicle entity and its hidden degradation trajectory.

The degradation curve is the key realism decision for the predictive
maintenance module: it must be NON-LINEAR (slow drift early in life,
accelerating decline near failure) or every downstream model degenerates
into a trivial threshold rule. We use a simple but effective exponential-ish
curve shape, the same qualitative pattern seen in NASA C-MAPSS turbofan
degradation data.
"""

import numpy as np
from config import FAILURE_MODES, FAILURE_MODE_WEIGHTS


class Vehicle:
    def __init__(self, vehicle_id: str, rng: np.random.Generator):
        self.vehicle_id = vehicle_id
        self.rng = rng

        self.failure_mode = rng.choice(
            list(FAILURE_MODE_WEIGHTS.keys()),
            p=list(FAILURE_MODE_WEIGHTS.values()),
        )
        mode_cfg = FAILURE_MODES[self.failure_mode]

        if self.failure_mode == "healthy_control":
            self.lifetime_days = None  # never fails within sim window
        else:
            lo, hi = mode_cfg["lifetime_days_range"]
            self.lifetime_days = int(rng.integers(lo, hi))

        self.affected_sensor = mode_cfg["affected_sensor"]
        self.baseline_value = mode_cfg["baseline"]
        self.max_drift = mode_cfg["max_drift"]

        # Odometer accumulates across the vehicle's simulated life
        self.odometer_km = float(rng.uniform(500, 30000))  # used vehicles, not brand new

    def degradation_fraction(self, age_days: int) -> float:
        """
        Returns a value in [0, 1] representing how far along the vehicle is
        toward failure at the given age. 0 = brand healthy, 1 = failure day.
        Non-linear: stays low for the first ~60% of life, then accelerates.
        """
        if self.lifetime_days is None:
            return 0.0
        t = np.clip(age_days / self.lifetime_days, 0.0, 1.0)
        # Cubic ease-in creates the "slow then fast" degradation shape.
        return float(t ** 3)

    def days_to_failure(self, age_days: int):
        """Ground-truth RUL (Remaining Useful Life) in days. None if healthy control."""
        if self.lifetime_days is None:
            return None
        return max(self.lifetime_days - age_days, 0)

    def will_fail_within(self, age_days: int, horizon_days: int = 30) -> int:
        """Binary MVP maintenance label: 1 if failure occurs within `horizon_days`."""
        dtf = self.days_to_failure(age_days)
        if dtf is None:
            return 0
        return int(dtf <= horizon_days)

    def sensor_drift(self, age_days: int) -> dict:
        """
        Returns the current drifted value of the affected sensor (if any),
        plus a bounded brake_efficiency_pct value (always present, defaults
        to ~100 for non-brake-wear vehicles with minor natural wear noise).
        """
        frac = self.degradation_fraction(age_days)
        result = {}

        if self.affected_sensor == "engine_temp_c":
            result["engine_temp_offset"] = frac * self.max_drift
        else:
            result["engine_temp_offset"] = 0.0

        if self.affected_sensor == "battery_voltage":
            result["battery_voltage_offset"] = frac * self.max_drift
        else:
            result["battery_voltage_offset"] = 0.0

        if self.affected_sensor == "oil_pressure_psi":
            result["oil_pressure_offset"] = frac * self.max_drift
        else:
            result["oil_pressure_offset"] = 0.0

        if self.affected_sensor == "brake_efficiency_pct":
            result["brake_efficiency_pct"] = 100.0 + frac * self.max_drift
        else:
            # gentle natural wear even for non-brake-failure vehicles
            natural_wear = min(age_days / 3000.0, 1.0) * -8.0
            result["brake_efficiency_pct"] = 100.0 + natural_wear

        result["degradation_fraction"] = frac
        return result
