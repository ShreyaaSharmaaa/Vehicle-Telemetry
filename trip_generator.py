"""
trip_generator.py
Generates a single physically-consistent trip: a 1Hz telemetry DataFrame
plus a trip-level summary row.

Key design principle: fields are NOT generated independently. Speed drives
distance/GPS position; speed changes drive acceleration; acceleration/RPM
drive engine load; engine load + vehicle degradation state drive engine
sensor readings. This correlated generation is what makes the dataset
resemble real telemetry rather than independent noisy columns.
"""

import numpy as np
import pandas as pd
from config import PERSONAS, BASE_CITY_LAT

KM_PER_DEG_LAT = 111.0


def _km_per_deg_lon(lat_deg: float) -> float:
    return 111.0 * np.cos(np.radians(lat_deg))


def _route_cumulative_distance_km(waypoints, base_lat):
    """Cumulative distance (km) at each waypoint, using equirectangular approx
    (fine for the small offsets used in config.py)."""
    km_lon = _km_per_deg_lon(base_lat)
    dists = [0.0]
    for i in range(1, len(waypoints)):
        dlat = (waypoints[i][0] - waypoints[i - 1][0]) * KM_PER_DEG_LAT
        dlon = (waypoints[i][1] - waypoints[i - 1][1]) * km_lon
        dists.append(dists[-1] + np.hypot(dlat, dlon))
    return np.array(dists)


def _interpolate_position(cum_dist_km, waypoints, target_dist_km, base_lat):
    """Linear interpolation of lat/lon along the route at a given cumulative distance."""
    total = cum_dist_km[-1]
    target_dist_km = np.clip(target_dist_km, 0, total)
    idx = np.searchsorted(cum_dist_km, target_dist_km, side="right") - 1
    idx = np.clip(idx, 0, len(waypoints) - 2)
    seg_start_d, seg_end_d = cum_dist_km[idx], cum_dist_km[idx + 1]
    frac = np.where(seg_end_d > seg_start_d,
                     (target_dist_km - seg_start_d) / np.maximum(seg_end_d - seg_start_d, 1e-9),
                     0.0)
    lat0, lon0 = waypoints[idx][:, 0], waypoints[idx][:, 1]
    lat1, lon1 = waypoints[idx + 1][:, 0], waypoints[idx + 1][:, 1]
    lat = lat0 + frac * (lat1 - lat0)
    lon = lon0 + frac * (lon1 - lon0)
    return base_lat + lat, BASE_CITY_LAT * 0 + lon  # lon offset applied by caller


def generate_trip(vehicle, driver_id: str, persona_name: str, route: dict,
                   trip_start_time: pd.Timestamp, age_days: int, rng: np.random.Generator,
                   base_lat: float = 23.2599, base_lon: float = 77.4126):
    """
    Returns (telemetry_df, trip_summary_dict).
    """
    persona = PERSONAS[persona_name]
    waypoints = np.array(route["waypoints"], dtype=float)
    cum_dist_km = _route_cumulative_distance_km(waypoints, base_lat)
    total_distance_km = cum_dist_km[-1]
    speed_limit = route["speed_limit_kmh"]

    # ---- Speed profile (1Hz) ----------------------------------------
    target_cruise = speed_limit * rng.normal(
        persona["speeding_factor_mean"], persona["speeding_factor_std"])
    target_cruise = max(target_cruise, 15.0)

    # Rough duration estimate to size the arrays, then we simulate until
    # cumulative distance reaches the route length.
    est_duration_s = int((total_distance_km / max(target_cruise, 10)) * 3600 * 1.3) + 60
    speed = np.zeros(est_duration_s)
    accel_events = []   # (t_idx, type, g_value)
    corner_events = []  # (t_idx, g_value)

    dist_km = 0.0
    v = 0.0  # km/h
    reaction_noise = persona["reaction_noise"]
    t = 0
    while dist_km < total_distance_km and t < est_duration_s - 1:
        # ramp up from a stop, otherwise hover near target cruise with noise
        if v < target_cruise * 0.9 and t < 15:
            v += rng.uniform(3, 6)  # acceleration phase
        else:
            v += rng.normal(0, reaction_noise * target_cruise * 0.15)

        # stochastic harsh events, frequency scaled by distance covered
        if rng.random() < persona["harsh_brake_prob_per_km"] / 3600 * max(v, 20):
            drop = rng.normal(persona["harsh_brake_g_mean"], persona["harsh_brake_g_std"])
            drop = max(drop, 0.05)
            v = max(v - drop * 9.81 * 3.6, 5.0)  # convert g to km/h delta over ~1s
            accel_events.append((t, "harsh_brake", -drop))
        elif rng.random() < persona["harsh_accel_prob_per_km"] / 3600 * max(v, 20):
            surge = rng.normal(persona["harsh_accel_g_mean"], persona["harsh_accel_g_std"])
            surge = max(surge, 0.05)
            v = min(v + surge * 9.81 * 3.6, speed_limit * 1.6)
            accel_events.append((t, "harsh_accel", surge))

        # near end of route, decelerate to stop
        remaining = total_distance_km - dist_km
        if remaining < 0.05:
            v = max(v - 8, 0)

        v = max(v, 0.0)
        speed[t] = v
        dist_km += v / 3600.0  # km covered this second
        t += 1

    n = t  # actual duration in seconds
    speed = speed[:n]
    if n < 5:
        n = 5
        speed = np.full(n, target_cruise * 0.5)

    # ---- Position (GPS) ------------------------------------------------
    cum_traveled = np.cumsum(speed) / 3600.0  # km
    km_lon = _km_per_deg_lon(base_lat)
    idx = np.searchsorted(cum_dist_km, np.clip(cum_traveled, 0, total_distance_km), side="right") - 1
    idx = np.clip(idx, 0, len(waypoints) - 2)
    seg_start_d = cum_dist_km[idx]
    seg_end_d = cum_dist_km[idx + 1]
    frac = np.where(seg_end_d > seg_start_d,
                     (np.clip(cum_traveled, 0, total_distance_km) - seg_start_d) /
                     np.maximum(seg_end_d - seg_start_d, 1e-9), 0.0)
    lat = base_lat + waypoints[idx, 0] + frac * (waypoints[idx + 1, 0] - waypoints[idx, 0])
    lon = base_lon + waypoints[idx, 1] + frac * (waypoints[idx + 1, 1] - waypoints[idx, 1])

    # heading change magnitude -> proxy for cornering intensity
    heading = np.arctan2(np.gradient(lat), np.gradient(lon) + 1e-9)
    heading_change = np.abs(np.gradient(heading))
    corner_g = heading_change * (speed / 50.0) * rng.normal(
        persona["cornering_g_mean"] / 0.2, persona["cornering_g_std"] / 0.2, size=n)
    corner_g = np.clip(corner_g, 0, 1.2)

    # ---- Derived dynamics ------------------------------------------------
    accel_x = np.gradient(speed) / 3.6  # (km/h)/s -> m/s^2 roughly
    accel_x_g = accel_x / 9.81

    rpm = 750 + speed * 38 + rng.normal(0, 60, size=n)
    rpm = np.clip(rpm, 700, 6500)

    throttle = np.clip(30 + accel_x_g * 90 + rng.normal(0, 5, size=n), 0, 100)
    brake_pressure = np.clip(-accel_x_g * 120, 0, 100)

    # ---- Engine/vehicle sensors (degradation-aware) -----------------------
    drift = vehicle.sensor_drift(age_days)
    warmup = np.minimum(np.arange(n) / 120.0, 1.0) * 12.0  # warms up over ~2 min
    engine_temp = 88 + warmup + drift["engine_temp_offset"] + rng.normal(0, 0.6, size=n)
    oil_pressure = 45 - (rpm - 750) / 6500 * 5 + drift["oil_pressure_offset"] + rng.normal(0, 0.8, size=n)
    battery_voltage = 13.8 + drift["battery_voltage_offset"] + rng.normal(0, 0.05, size=n)
    brake_efficiency_pct = np.full(n, drift["brake_efficiency_pct"])

    tire_base = 32.0
    tire_fl = tire_base + rng.normal(0, 0.4, size=n).cumsum() * 0.01
    tire_fr = tire_base + rng.normal(0, 0.4, size=n).cumsum() * 0.01
    tire_rl = tire_base + rng.normal(0, 0.4, size=n).cumsum() * 0.01
    tire_rr = tire_base + rng.normal(0, 0.4, size=n).cumsum() * 0.01

    fuel_consumption_factor = 1.0 + 0.4 * (persona_name == "aggressive")
    fuel_start = rng.uniform(35, 100)
    fuel_level = fuel_start - (cum_traveled[:n] / max(total_distance_km, 0.1)) * \
        rng.uniform(2, 6) * fuel_consumption_factor
    fuel_level = np.clip(fuel_level, 0, 100)

    timestamps = pd.date_range(trip_start_time, periods=n, freq="s")

    telemetry = pd.DataFrame({
        "timestamp": timestamps,
        "gps_lat": lat, "gps_lon": lon,
        "speed_kmh": speed,
        "acceleration_x_g": accel_x_g,
        "gyro_corner_g": corner_g,
        "rpm": rpm,
        "engine_temp_c": engine_temp,
        "oil_pressure_psi": oil_pressure,
        "battery_voltage": battery_voltage,
        "brake_efficiency_pct": brake_efficiency_pct,
        "fuel_level_pct": fuel_level,
        "throttle_position_pct": throttle,
        "brake_pressure": brake_pressure,
        "tire_pressure_fl": tire_fl, "tire_pressure_fr": tire_fr,
        "tire_pressure_rl": tire_rl, "tire_pressure_rr": tire_rr,
    })

    # ---- Trip-level summary -----------------------------------------------
    harsh_brakes = sum(1 for e in accel_events if e[1] == "harsh_brake")
    harsh_accels = sum(1 for e in accel_events if e[1] == "harsh_accel")
    harsh_corners = int(np.sum(corner_g > (persona["cornering_g_mean"] * 1.5)))
    overspeed_mask = speed > speed_limit * 1.10
    hour = trip_start_time.hour

    summary = {
        "driver_id": driver_id,
        "vehicle_id": vehicle.vehicle_id,
        "persona": persona_name,           # ground truth, hidden from model input
        "route_id": route["route_id"],
        "road_type": route["road_type"],
        "trip_start_time": trip_start_time,
        "duration_sec": n,
        "distance_km": round(float(cum_traveled[-1]), 3),
        "avg_speed_kmh": round(float(np.mean(speed)), 2),
        "max_speed_kmh": round(float(np.max(speed)), 2),
        "speed_limit_kmh": speed_limit,
        "harsh_braking_count": harsh_brakes,
        "harsh_accel_count": harsh_accels,
        "harsh_cornering_count": harsh_corners,
        "overspeeding_duration_sec": int(np.sum(overspeed_mask)),
        "overspeeding_events": int(np.sum(np.diff(overspeed_mask.astype(int)) == 1)),
        "idle_time_sec": int(np.sum(speed < 2)),
        "time_of_day_hour": hour,
        "is_night": int(hour < 5 or hour >= 22),
        "avg_engine_temp_c": round(float(np.mean(engine_temp)), 2),
        "min_oil_pressure_psi": round(float(np.min(oil_pressure)), 2),
        "min_battery_voltage": round(float(np.min(battery_voltage)), 2),
        "brake_efficiency_pct": round(float(brake_efficiency_pct[-1]), 2),
        "vehicle_age_days": age_days,
        "vehicle_failure_mode": vehicle.failure_mode,
        "days_to_failure": vehicle.days_to_failure(age_days),
        "will_fail_within_30d": vehicle.will_fail_within(age_days, 30),
    }

    return telemetry, summary
