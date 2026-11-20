"""The AI tier's feature flags (table ``flags`` in the AI schema).

A stage runs only while both the master flag and its own flag are on. A
flag with no row is off, and so is a stage this code does not know, so
every stage starts switched off. Three places check the flags:

* the application, when it enqueues a job (queue.enqueue);
* the AI worker, in its claim query (queue.claim_next);
* the AI worker, while a job runs (queue.renew).

A change takes effect without restarting anything.
"""
from __future__ import annotations

from psycopg import Connection, sql

MASTER = "research_intelligence"
# One flag per stage of the AI tier; later stages add theirs here.
STAGES = frozenset({"selftest"})
NAMES = frozenset({MASTER} | STAGES)


class FlagError(ValueError):
    pass


def table(schema: str) -> sql.Composable:
    return sql.Identifier(schema, "flags")


def enabled(conn: Connection, schema: str, stage: str) -> bool:
    """Whether ``stage`` may run now: the master flag and its own both on."""
    if stage not in STAGES:
        return False
    row = conn.execute(sql.SQL(
        "SELECT count(*) FILTER (WHERE enabled) = 2 FROM {} WHERE name = ANY(%s)").format(table(schema)),
        ([MASTER, stage],)).fetchone()
    return bool(row and row[0])


def listing(conn: Connection, schema: str) -> list[dict]:
    """Every known flag, set or not, with the effective state of each."""
    rows = {name: (on, set_at, set_by, reason) for name, on, set_at, set_by, reason in conn.execute(
        sql.SQL("SELECT name, enabled, set_at, set_by, reason FROM {}").format(table(schema)))}
    master = rows.get(MASTER, (False,))[0]
    out = []
    for name in [MASTER, *sorted(STAGES)]:
        on, set_at, set_by, reason = rows.get(name, (False, None, None, None))
        out.append({"name": name, "enabled": on, "effective": on if name == MASTER else (on and master),
                    "set_at": set_at, "set_by": set_by, "reason": reason})
    return out


def set_flag(conn: Connection, schema: str, name: str, on: bool, set_by: str, reason: str) -> bool:
    """Set a flag; returns its previous value (False if it had no row)."""
    if name not in NAMES:
        raise FlagError(f"unknown flag {name!r}; known: {', '.join(sorted(NAMES))}")
    set_by, reason = (set_by or "").strip(), (reason or "").strip()
    if not set_by or not reason:
        raise FlagError("say who sets the flag and why")
    previous = conn.execute(sql.SQL("SELECT enabled FROM {} WHERE name = %s FOR UPDATE").format(table(schema)),
                            (name,)).fetchone()
    conn.execute(sql.SQL(
        "INSERT INTO {} (name, enabled, set_at, set_by, reason) VALUES (%s, %s, now(), %s, %s)"
        " ON CONFLICT (name) DO UPDATE SET enabled = EXCLUDED.enabled, set_at = EXCLUDED.set_at,"
        " set_by = EXCLUDED.set_by, reason = EXCLUDED.reason").format(table(schema)),
        (name, on, set_by[:320], reason))
    return bool(previous and previous[0])
