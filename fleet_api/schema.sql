-- schema.sql
-- Fleet Telemetry Platform database schema.
--
-- DESIGN PRINCIPLE: the production tables below contain ONLY what a real
-- fleet system would actually know. `persona` and `vehicle_failure_mode`
-- (the simulator's hidden ground truth) are deliberately NOT columns here --
-- same leakage discipline as the ML pipeline, now enforced at the schema
-- level instead of just in Python. They live in a separate `sim_ground_truth`
-- schema, used only to validate the deployed models against known-truth
-- data, exactly like validate_against_ground_truth.py did in Phase 1.

CREATE SCHEMA IF NOT EXISTS fleet;
CREATE SCHEMA IF NOT EXISTS sim_ground_truth;

-- ============================================================
-- PRODUCTION SCHEMA
-- ============================================================

CREATE TABLE fleet.vehicles (
    vehicle_id      VARCHAR(20) PRIMARY KEY,
    odometer_km     NUMERIC(10, 1) NOT NULL DEFAULT 0,
    added_at        TIMESTAMP NOT NULL DEFAULT now(),
    last_updated    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE fleet.drivers (
    driver_id       VARCHAR(20) PRIMARY KEY,
    added_at        TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE fleet.trips (
    trip_id                     VARCHAR(36) PRIMARY KEY,
    vehicle_id                  VARCHAR(20) NOT NULL REFERENCES fleet.vehicles(vehicle_id),
    driver_id                   VARCHAR(20) NOT NULL REFERENCES fleet.drivers(driver_id),
    route_id                    VARCHAR(30),
    road_type                   VARCHAR(20),
    trip_start_time             TIMESTAMP NOT NULL,
    duration_sec                INTEGER NOT NULL,
    distance_km                 NUMERIC(8, 3) NOT NULL,
    avg_speed_kmh                NUMERIC(6, 2),
    max_speed_kmh                NUMERIC(6, 2),
    speed_limit_kmh              NUMERIC(6, 2),
    harsh_braking_count          INTEGER NOT NULL DEFAULT 0,
    harsh_accel_count            INTEGER NOT NULL DEFAULT 0,
    harsh_cornering_count        INTEGER NOT NULL DEFAULT 0,
    overspeeding_duration_sec    INTEGER NOT NULL DEFAULT 0,
    overspeeding_events          INTEGER NOT NULL DEFAULT 0,
    idle_time_sec                INTEGER NOT NULL DEFAULT 0,
    is_night                     BOOLEAN NOT NULL DEFAULT false,
    avg_engine_temp_c            NUMERIC(6, 2),
    min_oil_pressure_psi         NUMERIC(6, 2),
    min_battery_voltage          NUMERIC(5, 2),
    brake_efficiency_pct         NUMERIC(5, 2),
    vehicle_age_days             INTEGER,
    created_at                   TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX idx_trips_vehicle_time ON fleet.trips (vehicle_id, trip_start_time);
CREATE INDEX idx_trips_driver ON fleet.trips (driver_id);

CREATE TABLE fleet.risk_scores (
    id              SERIAL PRIMARY KEY,
    trip_id         VARCHAR(36) NOT NULL UNIQUE REFERENCES fleet.trips(trip_id),
    risk_score      NUMERIC(5, 2) NOT NULL,
    risk_tier       VARCHAR(10) NOT NULL,
    model_version   VARCHAR(20) NOT NULL DEFAULT 'gbm_v1',
    scored_at       TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX idx_risk_scores_tier ON fleet.risk_scores (risk_tier);

CREATE TABLE fleet.maintenance_predictions (
    id                      SERIAL PRIMARY KEY,
    vehicle_id              VARCHAR(20) NOT NULL REFERENCES fleet.vehicles(vehicle_id),
    trip_id                 VARCHAR(36) REFERENCES fleet.trips(trip_id),
    failure_probability     NUMERIC(5, 4) NOT NULL,
    risk_tier               VARCHAR(10) NOT NULL,
    model_version           VARCHAR(20) NOT NULL DEFAULT 'gbm_v1',
    predicted_at            TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX idx_maintenance_vehicle_time ON fleet.maintenance_predictions (vehicle_id, predicted_at);

CREATE TABLE fleet.anomaly_flags (
    id                  SERIAL PRIMARY KEY,
    trip_id             VARCHAR(36) NOT NULL REFERENCES fleet.trips(trip_id),
    telemetry_timestamp TIMESTAMP NOT NULL,
    anomaly_score       NUMERIC(8, 5),
    anomaly_type_guess  VARCHAR(30),
    detection_method    VARCHAR(20) NOT NULL,  -- 'isolation_forest' | 'rule_gps_dropout'
    detected_at         TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX idx_anomaly_trip ON fleet.anomaly_flags (trip_id);

CREATE TABLE fleet.alerts (
    id              SERIAL PRIMARY KEY,
    vehicle_id      VARCHAR(20) NOT NULL REFERENCES fleet.vehicles(vehicle_id),
    alert_type      VARCHAR(20) NOT NULL,  -- 'risk' | 'maintenance' | 'anomaly'
    severity        VARCHAR(10) NOT NULL,  -- 'low' | 'medium' | 'high' | 'critical'
    message         TEXT NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    resolved        BOOLEAN NOT NULL DEFAULT false,
    resolved_at     TIMESTAMP
);
CREATE INDEX idx_alerts_vehicle_unresolved ON fleet.alerts (vehicle_id, resolved);

-- ============================================================
-- SIMULATION GROUND TRUTH (validation only -- never queried by the API,
-- never joined into anything the model sees. Kept in a separate schema so
-- it's structurally impossible to accidentally leak into a serving path.)
-- ============================================================

CREATE TABLE sim_ground_truth.driver_personas (
    driver_id   VARCHAR(20) PRIMARY KEY REFERENCES fleet.drivers(driver_id),
    persona     VARCHAR(30) NOT NULL
);

CREATE TABLE sim_ground_truth.vehicle_failure_modes (
    vehicle_id      VARCHAR(20) PRIMARY KEY REFERENCES fleet.vehicles(vehicle_id),
    failure_mode    VARCHAR(30) NOT NULL,
    lifetime_days   INTEGER
);
