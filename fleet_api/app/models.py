"""
app/models.py
SQLAlchemy ORM models mirroring schema.sql. Table args set the `fleet`
schema explicitly since Postgres schemas aren't the SQLAlchemy default.

These models intentionally have NO persona/failure_mode columns -- see the
comment at the top of schema.sql for why. If you find yourself wanting to
add one of those fields here to make an endpoint easier to write, that's
the leakage instinct creeping back in; put it in sim_ground_truth instead
and query it explicitly, never join it into a serving path.
"""

from sqlalchemy import (
    Column, String, Integer, Numeric, Boolean, TIMESTAMP, ForeignKey, Text, func,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"
    __table_args__ = {"schema": "fleet"}

    vehicle_id = Column(String(20), primary_key=True)
    odometer_km = Column(Numeric(10, 1), nullable=False, default=0)
    added_at = Column(TIMESTAMP, server_default=func.now())
    last_updated = Column(TIMESTAMP, server_default=func.now())

    trips = relationship("Trip", back_populates="vehicle")


class Driver(Base):
    __tablename__ = "drivers"
    __table_args__ = {"schema": "fleet"}

    driver_id = Column(String(20), primary_key=True)
    added_at = Column(TIMESTAMP, server_default=func.now())

    trips = relationship("Trip", back_populates="driver")


class Trip(Base):
    __tablename__ = "trips"
    __table_args__ = {"schema": "fleet"}

    trip_id = Column(String(36), primary_key=True)
    vehicle_id = Column(String(20), ForeignKey("fleet.vehicles.vehicle_id"), nullable=False)
    driver_id = Column(String(20), ForeignKey("fleet.drivers.driver_id"), nullable=False)
    route_id = Column(String(30))
    road_type = Column(String(20))
    trip_start_time = Column(TIMESTAMP, nullable=False)
    duration_sec = Column(Integer, nullable=False)
    distance_km = Column(Numeric(8, 3), nullable=False)
    avg_speed_kmh = Column(Numeric(6, 2))
    max_speed_kmh = Column(Numeric(6, 2))
    speed_limit_kmh = Column(Numeric(6, 2))
    harsh_braking_count = Column(Integer, default=0)
    harsh_accel_count = Column(Integer, default=0)
    harsh_cornering_count = Column(Integer, default=0)
    overspeeding_duration_sec = Column(Integer, default=0)
    overspeeding_events = Column(Integer, default=0)
    idle_time_sec = Column(Integer, default=0)
    is_night = Column(Boolean, default=False)
    avg_engine_temp_c = Column(Numeric(6, 2))
    min_oil_pressure_psi = Column(Numeric(6, 2))
    min_battery_voltage = Column(Numeric(5, 2))
    brake_efficiency_pct = Column(Numeric(5, 2))
    vehicle_age_days = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())

    vehicle = relationship("Vehicle", back_populates="trips")
    driver = relationship("Driver", back_populates="trips")
    risk_score = relationship("RiskScore", back_populates="trip", uselist=False)


class RiskScore(Base):
    __tablename__ = "risk_scores"
    __table_args__ = {"schema": "fleet"}

    id = Column(Integer, primary_key=True)
    trip_id = Column(String(36), ForeignKey("fleet.trips.trip_id"), unique=True, nullable=False)
    risk_score = Column(Numeric(5, 2), nullable=False)
    risk_tier = Column(String(10), nullable=False)
    model_version = Column(String(20), default="gbm_v1")
    scored_at = Column(TIMESTAMP, server_default=func.now())

    trip = relationship("Trip", back_populates="risk_score")


class MaintenancePrediction(Base):
    __tablename__ = "maintenance_predictions"
    __table_args__ = {"schema": "fleet"}

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String(20), ForeignKey("fleet.vehicles.vehicle_id"), nullable=False)
    trip_id = Column(String(36), ForeignKey("fleet.trips.trip_id"))
    failure_probability = Column(Numeric(5, 4), nullable=False)
    risk_tier = Column(String(10), nullable=False)
    model_version = Column(String(20), default="gbm_v1")
    predicted_at = Column(TIMESTAMP, server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = {"schema": "fleet"}

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String(20), ForeignKey("fleet.vehicles.vehicle_id"), nullable=False)
    alert_type = Column(String(20), nullable=False)
    severity = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    resolved = Column(Boolean, default=False)
    resolved_at = Column(TIMESTAMP)
