"""The AI tier's daily token budget (table ``token_reservations``, migration 0016).

A model request costs tokens before anyone knows how many, so the budget is
reserve-then-settle:

1. ``reserve``, before the request: the counted input tokens and the most
   output tokens the request allows (its max_tokens), refused with
   BudgetExhausted if either would take the day past its limit. Each worker
   reserves under one advisory lock per AI schema, so concurrent workers
   cannot together overspend.
2. Then exactly one of:

   * ``settle``, once the provider has answered: the tokens it reports, as
     they are, even when they exceed the reservation (structured outputs add
     a system prompt the count did not see). Cache writes and reads count as
     input.
   * ``release``, only when the request provably never went out (the stage
     was switched off, the worker stopped before sending). A request whose
     fate is unknown is never released.
   * ``abandon``, when the request's fate is unknown (its job process
     failed or ran out of time): ``orphaned`` at once, counting in full.
   * ``reclaim_orphans``, run by the worker's reaper: a reservation still
     open while its job is no longer running that attempt with that worker
     (the worker died or lost its lease mid-request) becomes ``orphaned``
     and counts in full, as the request may well have been billed.

A day is a UTC day, the reservation's. Settled reservations count what they
used; open and orphaned ones what they reserved; released ones nothing. A
limit of 0, the default, allows no request at all.

``reserve``, ``settle`` and ``release`` run in the caller's transaction,
which must first have made sure the job is still its own (pacing.own); the
caller keeps that transaction short and never holds it across a request.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from psycopg import Connection, sql

LOCK_PREFIX = "qs-ai-token-budget:"
MAX_TOKENS = 10 ** 9
_MODEL = re.compile(r"[a-z0-9][a-z0-9.:@_-]{0,127}")


class BudgetExhausted(Exception):
    """The day's token budget cannot cover the request."""


def _count(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOKENS:
        raise ValueError(f"{name} must be a whole number from 0 to {MAX_TOKENS}")
    return value


@dataclass(frozen=True)
class Limits:
    """The most input and output tokens all AI workers together may use in a UTC day."""
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self):
        _count("input_tokens", self.input_tokens)
        _count("output_tokens", self.output_tokens)


@dataclass(frozen=True)
class Usage:
    """Tokens one answer used, as the provider reports them."""
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self):
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            _count(name, getattr(self, name))

    @property
    def input_total(self) -> int:
        """Every input token, cache writes and reads included."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


def _t(schema: str, name: str) -> sql.Composable:
    return sql.Identifier(schema, name)


_SPENT = """
    SELECT coalesce(sum(CASE WHEN status = 'settled' THEN input_used + cache_creation_used + cache_read_used
                             WHEN status IN ('reserved', 'orphaned') THEN input_reserved ELSE 0 END), 0)::bigint,
           coalesce(sum(CASE WHEN status = 'settled' THEN output_used
                             WHEN status IN ('reserved', 'orphaned') THEN output_reserved ELSE 0 END), 0)::bigint
    FROM {reservations} WHERE day = (now() AT TIME ZONE 'UTC')::date
"""


def spent_today(conn: Connection, schema: str) -> tuple[int, int]:
    """The input and output tokens today's reservations count."""
    row = conn.execute(sql.SQL(_SPENT).format(reservations=_t(schema, "token_reservations"))).fetchone()
    return int(row[0]), int(row[1])


def reserve(conn: Connection, schema: str, *, job_id: str, attempt: int, worker_id: str, model: str,
            input_tokens: int, max_output_tokens: int, limits: Limits) -> str:
    """Reserve tokens for one request; returns the reservation's id, or
    raises BudgetExhausted. Runs in the caller's transaction. A request
    reserves at least one token each way, so a limit of 0 refuses it."""
    for name, value in (("input_tokens", input_tokens), ("max_output_tokens", max_output_tokens)):
        if _count(name, value) < 1:
            raise ValueError(f"{name} must be at least 1")
    if not isinstance(model, str) or not _MODEL.fullmatch(model):
        raise ValueError("model is not a model id")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a whole number from 1")
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (LOCK_PREFIX + schema,))
    spent_input, spent_output = spent_today(conn, schema)
    if spent_input + input_tokens > limits.input_tokens or spent_output + max_output_tokens > limits.output_tokens:
        raise BudgetExhausted(
            f"today's token budget is spent: {spent_input} of {limits.input_tokens} input and {spent_output} of"
            f" {limits.output_tokens} output tokens counted; this request needs {input_tokens} input and up to"
            f" {max_output_tokens} output tokens")
    reservation_id = str(uuid.uuid4())
    conn.execute(sql.SQL(
        "INSERT INTO {} (id, job_id, attempt, worker_id, model, day, input_reserved, output_reserved)"
        " VALUES (%s, %s, %s, %s, %s, (now() AT TIME ZONE 'UTC')::date, %s, %s)"
    ).format(_t(schema, "token_reservations")),
        (reservation_id, job_id, attempt, worker_id, model, input_tokens, max_output_tokens))
    return reservation_id


def settle(conn: Connection, schema: str, reservation_id: str, usage: Usage) -> bool:
    """Record what an answered request used; False if the reservation was
    no longer open (it was orphaned). Runs in the caller's transaction."""
    return conn.execute(sql.SQL(
        "UPDATE {} SET status = 'settled', input_used = %s, cache_creation_used = %s, cache_read_used = %s,"
        " output_used = %s, closed_at = now() WHERE id = %s AND status = 'reserved'"
    ).format(_t(schema, "token_reservations")),
        (usage.input_tokens, usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
         usage.output_tokens, reservation_id)).rowcount == 1


def release(conn: Connection, schema: str, reservation_id: str) -> bool:
    """Free a reservation whose request never went out; False if it was no
    longer open. Runs in the caller's transaction."""
    return conn.execute(sql.SQL(
        "UPDATE {} SET status = 'released', closed_at = now() WHERE id = %s AND status = 'reserved'"
    ).format(_t(schema, "token_reservations")), (reservation_id,)).rowcount == 1


def abandon(conn: Connection, schema: str, reservation_id: str) -> bool:
    """Close the reservation of a request whose fate is unknown (its job
    process failed or ran out of time): ``orphaned``, so it counts in full.
    False if it was no longer open. Runs in the caller's transaction."""
    return conn.execute(sql.SQL(
        "UPDATE {} SET status = 'orphaned', closed_at = now() WHERE id = %s AND status = 'reserved'"
    ).format(_t(schema, "token_reservations")), (reservation_id,)).rowcount == 1


def reclaim_orphans(conn: Connection, schema: str) -> int:
    """Mark ``orphaned`` every open reservation whose job is no longer
    running that attempt with that worker (or is gone); returns how many."""
    return conn.execute(sql.SQL("""
        UPDATE {reservations} r SET status = 'orphaned', closed_at = now()
        WHERE r.status = 'reserved' AND NOT EXISTS (
            SELECT 1 FROM {jobs} j WHERE j.id = r.job_id AND j.status = 'running'
                                     AND j.worker_id = r.worker_id AND j.attempts = r.attempt)
    """).format(reservations=_t(schema, "token_reservations"), jobs=_t(schema, "jobs"))).rowcount
