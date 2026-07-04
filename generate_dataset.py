"""
generate_dataset.py
Orchestrates the full fleet simulation and writes four output datasets:

  trips.csv               - one row per trip, with behavioral + vehicle
                             health aggregates and ground-truth labels
                             (persona, failure mode, days_to_failure).
                             This is the PRIMARY training table for the
                             driver-risk and maintenance-classification
                             models.
  telemetry_recent.csv     - raw 1Hz telemetry, but ONLY for trips within
                             the last RAW_WINDOW_DAYS of the simulation.
                             Mirrors a real platform's "hot storage" design:
                             you don't keep raw 1Hz data forever.
  maintenance_events.csv   - one row per vehicle failure event that
                             actually occurred within the sim window.
  anomaly_events.csv       - ground-truth labels for injected anomalies,
                             used to evaluate the unsupervised anomaly
                             detector.

Usage:
    python generate_dataset.py --vehicles 50 --days 180 --out ./data
"""

import argparse
import os
import uuid

import numpy as np
import pandas as pd

from config import (
    ROUTES, PERSONAS, PERSONA_POPULATION_WEIGHTS, RANDOM_SEED, BASE_CITY_LAT, BASE_CITY_LON,
)
from vehicle import Vehicle
from trip_generator import generate_trip
from anomalies import maybe_inject_anomalies


def build_fleet(num_vehicles, num_drivers, rng):
    vehicles = [Vehicle(f"VEH{str(i).zfill(4)}", rng) for i in range(num_vehicles)]

    driver_personas = rng.choice(
        list(PERSONA_POPULATION_WEIGHTS.keys()),
        size=num_drivers,
        p=list(PERSONA_POPULATION_WEIGHTS.values()),
    )
    drivers = [
        {"driver_id": f"DRV{str(i).zfill(4)}", "persona": driver_personas[i]}
        for i in range(num_drivers)
    ]
    return vehicles, drivers


def simulate(num_vehicles=50, num_drivers=30, sim_days=180, trips_per_day_lambda=2.2,
             raw_window_days=14, raw_storage_vehicle_sample=None, out_dir="./data",
             seed=RANDOM_SEED):
    """
    raw_storage_vehicle_sample: if set (e.g. 12), only this many vehicles have
        their raw 1Hz telemetry retained, mirroring a realistic hot-storage
        tier that wouldn't keep dense raw data for an entire large fleet on a
        dev/demo environment. All vehicles still get full trip-level
        aggregates -- this only limits the raw telemetry export.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    vehicles, drivers = build_fleet(num_vehicles, num_drivers, rng)
    sim_start = pd.Timestamp("2026-01-01")
    raw_cutoff_day = sim_days - raw_window_days

    raw_storage_vehicle_ids = None
    if raw_storage_vehicle_sample is not None:
        chosen = rng.choice(len(vehicles), size=min(raw_storage_vehicle_sample, len(vehicles)),
                             replace=False)
        raw_storage_vehicle_ids = {vehicles[i].vehicle_id for i in chosen}

    trip_rows = []
    telemetry_frames = []
    anomaly_rows = []
    maintenance_rows = []
    logged_failures = set()

    for day in range(sim_days):
        current_date = sim_start + pd.Timedelta(days=day)

        for vehicle in vehicles:
            age_days = day  # simplification: vehicle "born" at sim start
            n_trips_today = rng.poisson(trips_per_day_lambda)
            if n_trips_today == 0:
                continue

            # assign a driver for the day (rental reassignment realism:
            # not always the same driver on the same vehicle)
            driver = drivers[rng.integers(0, len(drivers))]

            for _ in range(n_trips_today):
                route = ROUTES[rng.integers(0, len(ROUTES))]
                hour = int(rng.choice(
                    range(24),
                    p=_hourly_departure_weights()))
                minute = int(rng.integers(0, 60))
                trip_start = current_date + pd.Timedelta(hours=hour, minutes=minute)

                telemetry, summary = generate_trip(
                    vehicle=vehicle, driver_id=driver["driver_id"],
                    persona_name=driver["persona"], route=route,
                    trip_start_time=trip_start, age_days=age_days, rng=rng,
                    base_lat=BASE_CITY_LAT, base_lon=BASE_CITY_LON,
                )

                trip_id = str(uuid.uuid4())
                summary["trip_id"] = trip_id
                trip_rows.append(summary)

                anomaly_events = maybe_inject_anomalies(telemetry, trip_id, rng)
                anomaly_rows.extend(anomaly_events)

                store_raw = day >= raw_cutoff_day and (
                    raw_storage_vehicle_ids is None
                    or vehicle.vehicle_id in raw_storage_vehicle_ids
                )
                if store_raw:
                    telemetry.insert(0, "trip_id", trip_id)
                    telemetry.insert(1, "vehicle_id", vehicle.vehicle_id)
                    telemetry_frames.append(telemetry)

                # log a maintenance event exactly once, the day it "occurs"
                dtf = vehicle.days_to_failure(age_days)
                if (dtf is not None and dtf <= 0
                        and vehicle.vehicle_id not in logged_failures):
                    maintenance_rows.append({
                        "vehicle_id": vehicle.vehicle_id,
                        "failure_mode": vehicle.failure_mode,
                        "event_date": current_date,
                        "vehicle_age_days": age_days,
                        "odometer_km_at_failure": round(vehicle.odometer_km, 1),
                    })
                    logged_failures.add(vehicle.vehicle_id)

                vehicle.odometer_km += summary["distance_km"]

    trips_df = pd.DataFrame(trip_rows)
    anomalies_df = pd.DataFrame(anomaly_rows)
    maintenance_df = pd.DataFrame(maintenance_rows)
    telemetry_df = (pd.concat(telemetry_frames, ignore_index=True)
                     if telemetry_frames else pd.DataFrame())

    trips_df.to_csv(os.path.join(out_dir, "trips.csv"), index=False)
    anomalies_df.to_csv(os.path.join(out_dir, "anomaly_events.csv"), index=False)
    maintenance_df.to_csv(os.path.join(out_dir, "maintenance_events.csv"), index=False)
    telemetry_df.to_csv(os.path.join(out_dir, "telemetry_recent.csv"), index=False)

    return trips_df, telemetry_df, anomalies_df, maintenance_df


def _hourly_departure_weights():
    """Bimodal weighting: more trips during commute-like hours."""
    base = np.ones(24)
    for h in range(6, 10):
        base[h] = 3.0
    for h in range(16, 20):
        base[h] = 3.0
    for h in range(0, 5):
        base[h] = 0.2
    return base / base.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicles", type=int, default=50)
    parser.add_argument("--drivers", type=int, default=30)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--raw_window_days", type=int, default=21)
    parser.add_argument("--raw_vehicle_sample", type=int, default=None,
                         help="If set, only this many vehicles get raw telemetry stored")
    parser.add_argument("--out", type=str, default="./data")
    args = parser.parse_args()

    trips_df, telemetry_df, anomalies_df, maintenance_df = simulate(
        num_vehicles=args.vehicles, num_drivers=args.drivers, sim_days=args.days,
        raw_window_days=args.raw_window_days,
        raw_storage_vehicle_sample=args.raw_vehicle_sample, out_dir=args.out,
    )

    print(f"Trips generated:        {len(trips_df):,}")
    print(f"Raw telemetry rows:     {len(telemetry_df):,} (last {args.raw_window_days} days)")
    print(f"Anomaly events:         {len(anomalies_df):,}")
    print(f"Maintenance events:     {len(maintenance_df):,}")
    print(f"\nPersona distribution in trips:\n{trips_df['persona'].value_counts()}")
    print(f"\nFailure mode distribution (unique vehicles):\n"
          f"{trips_df.groupby('vehicle_id')['vehicle_failure_mode'].first().value_counts()}")
