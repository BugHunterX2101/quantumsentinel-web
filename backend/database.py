"""QuantumSentinel — Database layer (SQLite via SQLAlchemy).

Simplified from the PRD's PostgreSQL 16 + Redis stack for a single-process
web deployment. Swap DATABASE_URL for a Postgres DSN in production.

Pool settings target 100k concurrent users behind a load-balancer
(pool_size=20, max_overflow=40 gives 60 live connections per process).
"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL


_is_sqlite = DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    # SQLite: single writer, disable pool (StaticPool handles thread safety)
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
else:
    # PostgreSQL / production: tuned pool for high concurrency
    connect_args = {}
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

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
    # Lightweight compatibility migration for the portable SQLite demo.
    # A production deployment must use versioned migrations (Alembic).
    if _is_sqlite:
        columns = {c["name"] for c in inspect(engine).get_columns("trades")}
        if "stop_price" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE trades ADD COLUMN stop_price NUMERIC"))
        if "time_in_force" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE trades ADD COLUMN time_in_force VARCHAR DEFAULT 'day'"))
        # Watchlist migration — added in v1.1
        user_cols = {c["name"] for c in inspect(engine).get_columns("users")}
        if "watchlist" not in user_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN watchlist JSON"))
        if "preferred_exchanges" not in user_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN preferred_exchanges JSON"))
        if "user_timezone" not in user_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN user_timezone VARCHAR(64)"))
