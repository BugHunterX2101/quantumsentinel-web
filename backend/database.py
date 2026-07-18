"""QuantumSentinel — Database layer (SQLite via SQLAlchemy).

Simplified from the PRD's PostgreSQL 16 + Redis stack for a single-process
web deployment. Swap DATABASE_URL for a Postgres DSN in production.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quantumsentinel.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401  (ensure models are registered)
    Base.metadata.create_all(bind=engine)
