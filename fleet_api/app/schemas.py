"""
app/schemas.py
Pydantic models defining the API's request/response contracts. Separate
from the SQLAlchemy models in models.py on purpose -- ORM models describe
the database, these describe the wire format, and conflating the two is a
common source of accidentally exposing internal fields to API consumers.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TripIngest(BaseModel):
    """What a telemetry ingestion client sends for one completed trip."""
    trip_id: str
    vehicle_id: str
    driver_id: str
    route_id: Optional[str] = None
    road_type: Optional[str] = None
    trip_start_time: datetime
    duration_sec: int
    distance_km: float
    avg_speed_kmh: float
    max_speed_kmh: float
    speed_limit_kmh: float
    harsh_braking_count: int = 0
    harsh_accel_count: int = 0
    harsh_cornering_count: int = 0
    overspeeding_duration_sec: int = 0
    overspeeding_events: int = 0
    idle_time_sec: int = 0
    is_night: bool = False
    avg_engine_temp_c: float
    min_oil_pressure_psi: float
    min_battery_voltage: float
    brake_efficiency_pct: float
    vehicle_age_days: int


class RiskScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    trip_id: str
    risk_score: float
    risk_tier: str
    model_version: str
    scored_at: datetime


class TripOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    trip_id: str
    vehicle_id: str
    driver_id: str
    trip_start_time: datetime
    distance_km: float
    avg_speed_kmh: float
    harsh_braking_count: int
    risk_score: Optional[RiskScoreOut] = None


class MaintenancePredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    vehicle_id: str
    failure_probability: float
    risk_tier: str
    model_version: str
    predicted_at: datetime
    trips_used_for_prediction: int


class FleetSummaryOut(BaseModel):
    total_vehicles: int
    total_trips_scored: int
    risk_tier_counts: dict
    vehicles_flagged_for_maintenance: int
    avg_fleet_risk_score: float


class VehicleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vehicle_id: str
    odometer_km: float
    added_at: datetime
