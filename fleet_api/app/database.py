"""
app/database.py
SQLAlchemy engine and session factory. DATABASE_URL comes from the
environment (set by docker-compose.yml inside the container). A sensible
localhost default is provided for running the API outside Docker during
development, pointed at a Postgres instance you started separately.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fleet_user:fleet_pass_dev_only@localhost:5432/fleet_telemetry",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
