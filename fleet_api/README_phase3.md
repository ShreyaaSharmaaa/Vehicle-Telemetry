# Phase 3: FastAPI + PostgreSQL Backend

## IMPORTANT: what's actually been tested vs. not

My sandbox has no internet access, so I could not install FastAPI, SQLAlchemy,
or run PostgreSQL/Docker to test this end-to-end the way every previous phase
was tested before being handed to you. Here's the honest breakdown:

**Verified, with real trained models, in the sandbox:**
- `app/feature_engineering.py` — tested against the actual saved
  `feature_names.joblib` / column order, confirmed correct.
- `app/ml_service.py` — loaded the REAL trained `risk_regressor.joblib` and
  `maintenance_classifier.joblib` and got predictions. Ran a calibration
  sanity check: engine temp near the actual training-data failure asymptote
  (~116-122°C) correctly predicts 99.9% failure probability / Critical,
  while a less extreme rising trend correctly predicts Low risk. The model
  is discriminating on realistic value ranges, not just trend direction —
  a good sign it learned something real, not a shortcut.

**Written carefully but NOT executed anywhere:**
- `app/main.py` (the actual FastAPI routes)
- `app/models.py` / `app/database.py` (SQLAlchemy against real Postgres)
- `schema.sql`, `docker-compose.yml`, `Dockerfile`
- `load_data.py`

This code follows standard, stable, well-documented patterns (FastAPI +
SQLAlchemy 2.0 + Pydantic v2), but you'll be the first one to actually run
it. Expect to hit at least one or two errors — that's normal for
first-run backend code, not a sign something's fundamentally wrong. Paste
whatever comes up and we'll debug it together, same as every phase before.

## What's in this phase

| File | Purpose |
|---|---|
| `schema.sql` | PostgreSQL schema — production tables + separate sim-ground-truth schema |
| `docker-compose.yml` | Orchestrates Postgres + the API container together |
| `Dockerfile` | Builds the FastAPI service image |
| `requirements.txt` | Python dependencies |
| `app/database.py` | SQLAlchemy engine/session setup |
| `app/models.py` | ORM models mirroring schema.sql |
| `app/schemas.py` | Pydantic request/response contracts |
| `app/feature_engineering.py` | Serving-time feature engineering (mirrors training-time logic) |
| `app/ml_service.py` | Loads trained models once at startup, exposes prediction functions |
| `app/main.py` | FastAPI routes |
| `load_data.py` | Bulk-loads trips.csv + precomputes scores (run from host, not in Docker) |

## Setup, step by step

### 1. Copy trained models into this folder

The API container only has access to what's inside `fleet_api/` — it doesn't
see your `ml_pipeline/` folder. Copy the four files it needs:

```powershell
mkdir model_output
copy ..\ml_pipeline\model_output\risk_regressor.joblib .\model_output\
copy ..\ml_pipeline\model_output\feature_names.joblib .\model_output\
copy ..\ml_pipeline\model_output\maintenance_classifier.joblib .\model_output\
copy ..\ml_pipeline\model_output\maintenance_feature_names.joblib .\model_output\
```

### 2. Start Postgres + the API

```powershell
docker compose up --build
```

First run will take a few minutes (downloading the Postgres image, building
the API image). Watch the logs — `schema.sql` runs automatically the first
time the Postgres container initializes its data volume. If you need to
reset the schema later (e.g. after editing schema.sql), you'll need to
remove the volume: `docker compose down -v` before `docker compose up` again.

### 3. Confirm the API is up

In a browser: `http://localhost:8000/docs` — FastAPI's automatic interactive
API documentation. If this loads, the API container is running and can talk
to Postgres (health check happens before docs even render).

Or from the terminal: `curl http://localhost:8000/health`

### 4. Load data

Run this from your **host machine** (not inside Docker), from the
`fleet_api` folder, with the containers still running:

```powershell
pip install sqlalchemy psycopg2-binary pandas numpy scikit-learn joblib
python load_data.py --trips ..\data\trips.csv --ml-pipeline-dir ..\ml_pipeline
```

This loads all 40K+ trips, then bulk-computes risk scores and maintenance
predictions for the entire history (much faster than 40,000 individual API
calls). Expect this to take a minute or two.

### 5. Try it out

```powershell
curl http://localhost:8000/fleet/summary
curl http://localhost:8000/vehicles/VEH0000/trips
curl http://localhost:8000/vehicles/VEH0000/maintenance-prediction
```

Or just use `/docs` in the browser — it gives you a clickable UI for every
endpoint.

## Key design decisions (and why)

**Schema-level leakage prevention, not just code-level.** `persona` and
`vehicle_failure_mode` aren't just excluded from Python feature lists (as
in Phases 1-2) — they don't exist as columns in the `fleet` schema at all.
They live in a separate `sim_ground_truth` schema, structurally impossible
to accidentally join into a serving path.

**Maintenance prediction requires querying trip history, not a stateless
call.** Unlike risk scoring (one trip, one prediction, no history needed),
the maintenance model needs rolling-window trend features. The
`/vehicles/{id}/maintenance-prediction` endpoint queries the last 11 trips
from Postgres, rebuilds the exact rolling features used in training, and
predicts on the most recent one. This is *why* the database matters here,
not just as storage — it's part of the prediction pipeline.

**Defensive feature alignment.** Every prediction function reindexes its
computed features against the exact `feature_names.joblib` list saved at
training time, rather than trusting column order by convention. A future
retraining that reorders features can't silently corrupt predictions.

**`load_data.py` runs on the host, not in the container.** Bulk-scoring
40K trips through individual HTTP requests would be slow and pointless —
it reuses the training-side feature engineering directly for a single
efficient batch operation. The live API's single-trip endpoints exist for
real-time ingestion of *new* trips going forward, not backfilling history.

## What Phase 4 will cover

React frontend dashboard consuming this API — fleet summary view, per-vehicle
drill-down, risk/maintenance alert lists.
