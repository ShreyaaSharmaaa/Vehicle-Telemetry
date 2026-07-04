"""
config.py
Static configuration for the Fleet Telemetry Simulator:
routes, driver personas, failure modes, and global constants.

Design notes:
- Routes are hand-crafted waypoint sequences (lat/lon) around a base city
  coordinate. We don't hit a live maps API (no external network access in
  this environment), but the waypoint spacing/curvature is chosen to mimic
  real urban vs. highway route geometry (frequent turns vs. long straights).
- Personas encode BEHAVIORAL ground truth. This is what makes the dataset
  usable for validating a risk-scoring model later: we know, by
  construction, which driver is "aggressive" vs "calm".
- Failure modes encode VEHICLE ground truth for predictive maintenance,
  modeled loosely on the shape of NASA C-MAPSS degradation curves
  (slow drift, then accelerating decline near end-of-life).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Global simulation constants
# ---------------------------------------------------------------------------
BASE_CITY_LAT = 23.2599   # Bhopal, used only as a plausible anchor point
BASE_CITY_LON = 77.4126

SAMPLING_HZ = 1  # 1 reading per second during a trip

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Routes: (route_id, road_type, speed_limit_kmh, waypoints as (lat, lon) offsets)
# Offsets are in degrees (~0.01 deg ~ 1.1 km) so routes stay geographically
# plausible without needing real map data.
# ---------------------------------------------------------------------------
ROUTES = [
    {
        "route_id": "R1_URBAN_LOOP",
        "road_type": "urban",
        "speed_limit_kmh": 50,
        "waypoints": [
            (0.000, 0.000), (0.004, 0.002), (0.006, 0.006), (0.004, 0.010),
            (0.000, 0.011), (-0.004, 0.009), (-0.006, 0.005), (-0.003, 0.001),
            (0.000, 0.000),
        ],
    },
    {
        "route_id": "R2_URBAN_CROSSTOWN",
        "road_type": "urban",
        "speed_limit_kmh": 45,
        "waypoints": [
            (0.000, 0.000), (0.002, 0.005), (0.001, 0.012), (0.003, 0.018),
            (0.006, 0.022), (0.010, 0.024),
        ],
    },
    {
        "route_id": "R3_HIGHWAY_NORTH",
        "road_type": "highway",
        "speed_limit_kmh": 100,
        "waypoints": [
            (0.000, 0.000), (0.020, 0.001), (0.045, 0.002), (0.070, 0.001),
            (0.095, 0.003), (0.120, 0.002),
        ],
    },
    {
        "route_id": "R4_HIGHWAY_SOUTH",
        "road_type": "highway",
        "speed_limit_kmh": 110,
        "waypoints": [
            (0.000, 0.000), (-0.018, -0.002), (-0.040, -0.001), (-0.065, -0.003),
            (-0.090, -0.002), (-0.115, -0.004),
        ],
    },
    {
        "route_id": "R5_MIXED_AIRPORT",
        "road_type": "mixed",
        "speed_limit_kmh": 80,
        "waypoints": [
            (0.000, 0.000), (0.005, 0.004), (0.015, 0.010), (0.030, 0.020),
            (0.050, 0.028), (0.055, 0.032),
        ],
    },
    {
        "route_id": "R6_URBAN_MARKET",
        "road_type": "urban",
        "speed_limit_kmh": 35,
        "waypoints": [
            (0.000, 0.000), (0.001, 0.003), (-0.001, 0.006), (0.002, 0.009),
            (0.001, 0.013),
        ],
    },
    {
        "route_id": "R7_HIGHWAY_RING",
        "road_type": "highway",
        "speed_limit_kmh": 100,
        "waypoints": [
            (0.000, 0.000), (0.015, 0.015), (0.030, 0.020), (0.040, 0.005),
            (0.030, -0.010), (0.010, -0.012), (0.000, 0.000),
        ],
    },
    {
        "route_id": "R8_MIXED_SUBURB",
        "road_type": "mixed",
        "speed_limit_kmh": 60,
        "waypoints": [
            (0.000, 0.000), (0.006, -0.003), (0.014, -0.005), (0.020, -0.002),
            (0.026, 0.001), (0.030, 0.006),
        ],
    },
]

# ---------------------------------------------------------------------------
# Driver personas
# ---------------------------------------------------------------------------
# harsh_brake_g_threshold: deceleration magnitude (g) beyond which an event
#   is flagged as "harsh braking". mean/std describe how hard THIS persona
#   brakes when they brake hard, and prob_per_km its frequency.
# speeding_factor: multiplier applied to the posted speed limit that this
#   persona tends to cruise at.
# reaction_noise: extra randomness in speed maintenance (proxy for
#   distraction/drowsiness -- inconsistent speed holding).
PERSONAS = {
    "calm": {
        "harsh_brake_prob_per_km": 0.03,
        "harsh_brake_g_mean": 0.18, "harsh_brake_g_std": 0.03,
        "harsh_accel_prob_per_km": 0.02,
        "harsh_accel_g_mean": 0.15, "harsh_accel_g_std": 0.03,
        "cornering_g_mean": 0.12, "cornering_g_std": 0.02,
        "speeding_factor_mean": 0.95, "speeding_factor_std": 0.05,
        "reaction_noise": 0.03,
        "risk_label": 1,  # lowest risk, 1-5 scale ground truth
    },
    "average": {
        "harsh_brake_prob_per_km": 0.08,
        "harsh_brake_g_mean": 0.28, "harsh_brake_g_std": 0.05,
        "harsh_accel_prob_per_km": 0.06,
        "harsh_accel_g_mean": 0.25, "harsh_accel_g_std": 0.05,
        "cornering_g_mean": 0.20, "cornering_g_std": 0.04,
        "speeding_factor_mean": 1.05, "speeding_factor_std": 0.08,
        "reaction_noise": 0.06,
        "risk_label": 2,
    },
    "aggressive": {
        "harsh_brake_prob_per_km": 0.22,
        "harsh_brake_g_mean": 0.45, "harsh_brake_g_std": 0.08,
        "harsh_accel_prob_per_km": 0.20,
        "harsh_accel_g_mean": 0.42, "harsh_accel_g_std": 0.08,
        "cornering_g_mean": 0.38, "cornering_g_std": 0.07,
        "speeding_factor_mean": 1.25, "speeding_factor_std": 0.12,
        "reaction_noise": 0.08,
        "risk_label": 4,
    },
    "drowsy_distracted": {
        "harsh_brake_prob_per_km": 0.15,
        "harsh_brake_g_mean": 0.40, "harsh_brake_g_std": 0.12,
        "harsh_accel_prob_per_km": 0.10,
        "harsh_accel_g_mean": 0.30, "harsh_accel_g_std": 0.10,
        "cornering_g_mean": 0.25, "cornering_g_std": 0.10,
        "speeding_factor_mean": 1.00, "speeding_factor_std": 0.20,  # inconsistent
        "reaction_noise": 0.18,  # high inconsistency is the key signature
        "risk_label": 5,  # highest risk despite not "speeding" on average
    },
}

PERSONA_POPULATION_WEIGHTS = {
    "calm": 0.30,
    "average": 0.45,
    "aggressive": 0.18,
    "drowsy_distracted": 0.07,
}

# ---------------------------------------------------------------------------
# Vehicle failure modes (predictive maintenance ground truth)
# ---------------------------------------------------------------------------
# Each mode defines which sensor(s) drift, the drift shape, and a plausible
# lifetime range (days of active operation before failure).
FAILURE_MODES = {
    "engine_overheat": {
        "affected_sensor": "engine_temp_c",
        "baseline": 92.0,
        "max_drift": 24.0,      # baseline creeps up to ~116C near failure
        "lifetime_days_range": (150, 320),
    },
    "brake_wear": {
        "affected_sensor": "brake_efficiency_pct",  # derived, 100=new
        "baseline": 100.0,
        "max_drift": -55.0,     # efficiency drops toward ~45%
        "lifetime_days_range": (180, 380),
    },
    "battery_degradation": {
        "affected_sensor": "battery_voltage",
        "baseline": 13.8,
        "max_drift": -2.3,      # voltage sags toward ~11.5V
        "lifetime_days_range": (200, 400),
    },
    "oil_pressure_loss": {
        "affected_sensor": "oil_pressure_psi",
        "baseline": 45.0,
        "max_drift": -22.0,
        "lifetime_days_range": (160, 300),
    },
    "healthy_control": {
        # A fraction of the fleet has no injected failure during the sim
        # window -- necessary as a negative-class control group, otherwise
        # every vehicle "fails" and the model can't learn what healthy
        # looks like.
        "affected_sensor": None,
        "baseline": None,
        "max_drift": 0.0,
        "lifetime_days_range": (None, None),
    },
}

FAILURE_MODE_WEIGHTS = {
    "engine_overheat": 0.20,
    "brake_wear": 0.20,
    "battery_degradation": 0.15,
    "oil_pressure_loss": 0.15,
    "healthy_control": 0.30,
}

# ---------------------------------------------------------------------------
# Anomaly injection (sudden, discrete -- distinct from gradual degradation)
# ---------------------------------------------------------------------------
ANOMALY_TYPES = {
    "gps_dropout": {"prob_per_trip": 0.015},
    "sensor_spike": {"prob_per_trip": 0.02},
    "impact_event": {"prob_per_trip": 0.005},
    "route_deviation": {"prob_per_trip": 0.01},
}
