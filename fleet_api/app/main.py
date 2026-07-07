"""
app/main.py
FastAPI application serving the driver risk scoring and predictive
maintenance models trained in Phases 1-2, backed by PostgreSQL.

Table creation is NOT done here (no Base.metadata.create_all() call) --
schema.sql is the single source of truth for the database structure,
applied automatically by the Postgres container on first startup via
docker-entrypoint-initdb.d. Keeping schema definition in exactly one place
avoids the two-sources-of-truth drift that happens when an ORM's
create_all() and a hand-written schema.sql quietly diverge over time.
"""

from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.ml_service import registry, MAINTENANCE_TIERS

app = FastAPI(
    title="Fleet Telemetry & Analytics Platform API",
    description="Serves driver risk scoring and predictive maintenance models "
                "for the fleet telemetry platform.",
    version="0.1.0",
)


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    db.execute(select(1))  # confirms the DB connection actually works, not just that the app is up
    return {"status": "ok"}


@app.post("/trips", response_model=schemas.TripOut, status_code=201)
def ingest_trip(trip: schemas.TripIngest, db: Session = Depends(get_db)):
    """
    Ingests one completed trip, auto-registering the vehicle/driver on
    first sight (reasonable for a telemetry ingestion entrypoint -- a
    dedicated fleet-onboarding endpoint would be added in a later phase
    for cases where you want to REQUIRE explicit registration first).
    Immediately scores the trip's driver risk (a single-trip prediction
    needs no historical context, unlike maintenance) and persists both
    the trip and its risk score.
    """
    if db.get(models.Trip, trip.trip_id):
        raise HTTPException(status_code=409, detail=f"Trip {trip.trip_id} already exists")

    if not db.get(models.Vehicle, trip.vehicle_id):
        db.add(models.Vehicle(vehicle_id=trip.vehicle_id))
    if not db.get(models.Driver, trip.driver_id):
        db.add(models.Driver(driver_id=trip.driver_id))

    db_trip = models.Trip(**trip.model_dump())
    db.add(db_trip)
    db.flush()  # ensure the trip row exists before we attach a risk_scores FK to it

    prediction = registry.score_trip_risk(trip.model_dump())
    db_risk = models.RiskScore(
        trip_id=trip.trip_id,
        risk_score=prediction["risk_score"],
        risk_tier=prediction["risk_tier"],
    )
    db.add(db_risk)

    db.commit()
    db.refresh(db_trip)
    return db_trip


@app.get("/trips/{trip_id}", response_model=schemas.TripOut)
def get_trip(trip_id: str, db: Session = Depends(get_db)):
    trip = db.get(models.Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


@app.get("/vehicles/{vehicle_id}", response_model=schemas.VehicleOut)
def get_vehicle(vehicle_id: str, db: Session = Depends(get_db)):
    vehicle = db.get(models.Vehicle, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle


@app.get("/vehicles/{vehicle_id}/trips", response_model=List[schemas.TripOut])
def get_vehicle_trips(vehicle_id: str, limit: int = 20, db: Session = Depends(get_db)):
    if not db.get(models.Vehicle, vehicle_id):
        raise HTTPException(status_code=404, detail="Vehicle not found")

    trips = db.execute(
        select(models.Trip)
        .where(models.Trip.vehicle_id == vehicle_id)
        .order_by(models.Trip.trip_start_time.desc())
        .limit(limit)
    ).scalars().all()
    return trips


@app.get("/vehicles/{vehicle_id}/maintenance-prediction", response_model=schemas.MaintenancePredictionOut)
def get_maintenance_prediction(vehicle_id: str, db: Session = Depends(get_db)):
    """
    Predicts failure-within-30-days probability as of the vehicle's most
    recent trip, using its last 11 trips (10-trip rolling window + the
    current trip) for trend context -- same window size the model was
    trained with. Fewer than 2 trips of history isn't enough to compute
    even the most basic trend feature, so we return 400 rather than a
    misleadingly confident prediction from a cold start.
    """
    if not db.get(models.Vehicle, vehicle_id):
        raise HTTPException(status_code=404, detail="Vehicle not found")

    rows = db.execute(
        select(models.Trip)
        .where(models.Trip.vehicle_id == vehicle_id)
        .order_by(models.Trip.trip_start_time.desc())
        .limit(11)
    ).scalars().all()

    if len(rows) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(rows)} trip(s) on record for this vehicle -- "
                    "need at least 2 to compute a trend-based prediction.",
        )

    rows = list(reversed(rows))  # back to chronological order, oldest first
    history_df = pd.DataFrame([{
        "avg_engine_temp_c": float(r.avg_engine_temp_c),
        "min_oil_pressure_psi": float(r.min_oil_pressure_psi),
        "min_battery_voltage": float(r.min_battery_voltage),
        "brake_efficiency_pct": float(r.brake_efficiency_pct),
        "vehicle_age_days": r.vehicle_age_days,
    } for r in rows])

    prediction = registry.predict_maintenance(history_df)

    db_prediction = models.MaintenancePrediction(
        vehicle_id=vehicle_id,
        trip_id=rows[-1].trip_id,
        failure_probability=prediction["failure_probability"],
        risk_tier=prediction["risk_tier"],
    )
    db.add(db_prediction)
    db.commit()

    return schemas.MaintenancePredictionOut(
        vehicle_id=vehicle_id,
        failure_probability=prediction["failure_probability"],
        risk_tier=prediction["risk_tier"],
        model_version="gbm_v1",
        predicted_at=datetime.utcnow(),
        trips_used_for_prediction=len(rows),
    )


@app.get("/fleet/summary", response_model=schemas.FleetSummaryOut)
def fleet_summary(db: Session = Depends(get_db)):
    total_vehicles = db.execute(select(func.count()).select_from(models.Vehicle)).scalar()
    total_scored = db.execute(select(func.count()).select_from(models.RiskScore)).scalar()

    tier_rows = db.execute(
        select(models.RiskScore.risk_tier, func.count())
        .group_by(models.RiskScore.risk_tier)
    ).all()
    tier_counts = {tier: count for tier, count in tier_rows}

    avg_score = db.execute(select(func.avg(models.RiskScore.risk_score))).scalar()

    flagged = db.execute(
        select(func.count(func.distinct(models.MaintenancePrediction.vehicle_id)))
        .where(models.MaintenancePrediction.risk_tier.in_(["High", "Critical"]))
    ).scalar()

    return schemas.FleetSummaryOut(
        total_vehicles=total_vehicles or 0,
        total_trips_scored=total_scored or 0,
        risk_tier_counts=tier_counts,
        vehicles_flagged_for_maintenance=flagged or 0,
        avg_fleet_risk_score=round(float(avg_score), 2) if avg_score else 0.0,
    )
