"""Alembic environment for QuantumSentinel.

Migrations always run on a connection supplied by backend.database
(``config.attributes["connection"]``), which holds the migration lock, so
there is no separate alembic.ini or database URL to keep in sync.
"""
from alembic import context

from backend import models  # noqa: F401  (registers every table on Base.metadata)
from backend.database import Base

connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("run migrations through backend.database / python -m backend.manage")

context.configure(connection=connection, target_metadata=Base.metadata,
                  compare_type=True, compare_server_default=True)
with context.begin_transaction():
    context.run_migrations()
