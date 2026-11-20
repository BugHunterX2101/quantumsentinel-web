"""The AI tier: a worker that holds no secret of the trusted tier.

It runs work on untrusted input (paper metadata from the internet, and later
uploaded documents) as qs_ai_worker, whose privileges reach only the AI
schema (backend/db_roles.py). Nothing under backend.ai may import the
trusted tier: backend.config, backend.database, backend.models,
backend.services, backend.crypto, backend.main, backend.manage,
backend.worker or backend.db_roles. tests/test_ai_tier.py checks this in a
fresh interpreter. The AI tier's settings come from backend.ai.settings.
"""
