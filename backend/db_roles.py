"""Least-privilege PostgreSQL roles.

Connected as the cluster's bootstrap superuser (the role POSTGRES_USER
creates), the application bypasses every grant, row-level security policy and
trigger, and any SQL it runs can drop the schema, read server files or start
programs (COPY ... TO PROGRAM). ``python -m backend.manage provision-roles``,
run as that superuser, hands the schema to seven roles:

    qs_owner            NOLOGIN. Owns the schema and every object in it.
    qs_migrator         LOGIN, a NOINHERIT member of qs_owner: it holds no
                        table privilege of its own. ``database.migrate`` runs
                        migrations after SET LOCAL ROLE qs_owner, so whatever
                        a migration creates is owned by qs_owner.
    qs_app              LOGIN. The API (and an embedded research worker):
                        exactly the table privileges in APP_TABLE_PRIVILEGES.
                        No DDL, no TRUNCATE; it cannot own objects, change
                        roles or switch triggers off.
    qs_research_worker  LOGIN. A research worker run as its own service:
                        exactly WORKER_TABLE_PRIVILEGES, the job queue,
                        research results and the audit trail, and nothing of
                        users, accounts, orders, positions, keys or tokens.
    qs_evaluator        LOGIN. The research evaluator (QS_PROCESS_ROLE
                        research-evaluator), which grades strategies on the
                        holdout: exactly EVALUATOR_TABLE_PRIVILEGES.
    qs_ai_worker        LOGIN. The AI tier (backend/ai): exactly
                        AI_WORKER_TABLE_PRIVILEGES in the AI schema, and no
                        privilege at all, not even USAGE, on the application
                        schema: no job queue, worker registry, audit trail
                        or user data of the trusted tier. Its connection
                        limit and statement, lock and idle-transaction
                        timeouts are set on the role itself.
    qs_backup           LOGIN. pg_read_all_data, for pg_dump.

The AI tier's tables live in a schema of their own beside the application's
(ai_schema_for: ``ai`` beside ``public``), owned by the owner role too. Only
the application role and the AI worker hold USAGE on it; the application
holds APP_AI_TABLE_PRIVILEGES there.

The privilege maps are allow-lists. A table missing from
APP_TABLE_PRIVILEGES gets no privilege at all, and tests/test_db_roles.py
fails until it is listed, so a new table is deny-by-default; one missing from
WORKER_TABLE_PRIVILEGES or EVALUATOR_TABLE_PRIVILEGES is simply out of that
role's reach.
``database.migrate`` reapplies the grants after every migration that changes
the schema revision.

Provisioning is idempotent. Run again, it restores the role attributes,
memberships, ownership and grants above and removes anything extra.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

from psycopg import Connection, errors, sql

DML = ("SELECT", "INSERT", "UPDATE", "DELETE")
TABLE_PRIVILEGES = ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")

# The application role's privileges on each table of the schema.
APP_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    # Read at startup to check the schema revision; only migrations write it.
    "alembic_version": ("SELECT",),
    "api_keys": DML,
    "audit_chain_links": DML,
    "audit_logs": DML,
    "backtests": DML,
    # Append-only (migration 0005's triggers). Operators add a holdout start
    # with ``manage.py set-holdout``, which connects as this role.
    "dataset_holdouts": ("SELECT", "INSERT"),
    # Immutable once written (migration 0004's triggers): no UPDATE or DELETE.
    "dataset_versions": ("SELECT", "INSERT"),
    # Append-only (migration 0011's triggers), written only by the research
    # evaluator. The API counts a family's holdout attempts as trials
    # (trial_matrix.family_state), checks an attempt is unspent before
    # ``manage final-evaluate`` queues one, and shows the owner whether a
    # grade passed.
    "holdout_attempts": ("SELECT",),
    "holdout_evaluations": ("SELECT",),
    "idempotency_records": DML,
    "key_pairs": DML,
    "order_security_records": DML,
    "paper_accounts": DML,
    "positions": DML,
    "refresh_tokens": DML,
    "research_experiments": DML,
    "research_jobs": DML,
    # The trial ledger the Deflated Sharpe discount counts. Only
    # research_trials.record writes it, from the worker, and the API runs an
    # embedded worker under this role, so it needs INSERT. Nothing updates or
    # deletes a trial, so a user's count can only grow.
    "research_trials": ("SELECT", "INSERT"),
    # Append-only (migration 0007's triggers): what each accepted
    # strategy-version run measured, written by the (embedded) worker.
    "research_trial_evaluations": ("SELECT", "INSERT"),
    "research_workers": DML,
    "server_signing_keys": DML,
    "signals": DML,
    "strategies": DML,
    # Append-only (migration 0008's triggers): the validation policy's
    # verdicts and the matrices they judged, written by the (embedded) worker.
    "strategy_validations": ("SELECT", "INSERT"),
    # Append-only (migration 0006's triggers): a strategy version, who holds
    # it and a holder's tags are recorded once and never changed.
    "strategy_versions": ("SELECT", "INSERT"),
    "trades": DML,
    "trial_matrices": ("SELECT", "INSERT"),
    "user_datasets": ("SELECT", "INSERT"),
    "user_strategy_tags": ("SELECT", "INSERT"),
    "user_strategy_versions": ("SELECT", "INSERT"),
    "users": DML,
    # Append-only (migration 0009's triggers): each verdict's signed
    # manifest, written by the (embedded) worker with the verdict.
    "validation_manifests": ("SELECT", "INSERT"),
    "webhooks": DML,
}

# The research worker's privileges (backend/worker.py and the code it runs:
# services/research_jobs.py, the experiment finalizers, research_trials and
# the audit writer). Foreign-key checks run as the table owner, so writing
# rows that reference users needs no privilege on users.
WORKER_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    # The worker waits at startup until the schema is at its code's revision.
    "alembic_version": ("SELECT",),
    # Signed audit events for finished jobs; the chain is append-only here.
    "audit_chain_links": ("SELECT", "INSERT"),
    "audit_logs": ("SELECT", "INSERT"),
    # The dashboard backtest's history row. Its id is made client-side, so
    # the insert returns nothing and needs no SELECT; the API reads it back.
    "backtests": ("INSERT",),
    # The worker reads the holdout starts it seals a job's market data at,
    # and checks the result against them; it never sets one.
    "dataset_holdouts": ("SELECT",),
    # The dataset snapshot finalizer registers the datasets it verified and
    # the user who fetched them; both tables are insert-only.
    "dataset_versions": ("SELECT", "INSERT"),
    "user_datasets": ("SELECT", "INSERT"),
    # A family's holdout attempts count as its trials (trial_matrix.
    # family_state). The grades and their figures (holdout_evaluations) are
    # out of its reach.
    "holdout_attempts": ("SELECT",),
    # Experiment finalizers lock the experiment row and record the outcome.
    "research_experiments": ("SELECT", "UPDATE"),
    # Claim, lease, finish, reap and purge jobs; only the API enqueues them.
    "research_jobs": ("SELECT", "UPDATE", "DELETE"),
    "research_trials": ("SELECT", "INSERT"),
    # A strategy backtest records what its accepted result measured.
    "research_trial_evaluations": ("SELECT", "INSERT"),
    "research_workers": DML,
    # The worker registers the server key it signs audit events with.
    "server_signing_keys": ("SELECT", "INSERT"),
    # A strategy backtest loads the version it runs, which the job's user
    # must hold, before the job process starts; the worker never registers one.
    "strategy_versions": ("SELECT",),
    "user_strategy_versions": ("SELECT",),
    # A strategy validation records the matrix it judged, its verdict and
    # the verdict's signed manifest (registering the signing key with it).
    "strategy_validations": ("SELECT", "INSERT"),
    "trial_matrices": ("SELECT", "INSERT"),
    "validation_manifests": ("SELECT", "INSERT"),
}

# The research evaluator's privileges: the worker loop (backend/worker.py) as
# QS_PROCESS_ROLE research-evaluator, which claims only holdout evaluation
# jobs and does no housekeeping (services/holdout_evaluation.py). Each
# table is here for code it runs.
EVALUATOR_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    # The worker waits at startup until the schema is at its code's revision.
    "alembic_version": ("SELECT",),
    # Signed audit events for finished jobs; the chain is append-only here.
    "audit_chain_links": ("SELECT", "INSERT"),
    "audit_logs": ("SELECT", "INSERT"),
    # The holdout start it grades from, read before it spends the attempt.
    "dataset_holdouts": ("SELECT",),
    # It spends an attempt when it claims the job, then records the grade
    # with its signed manifest. Both tables are append-only.
    "holdout_attempts": ("SELECT", "INSERT"),
    "holdout_evaluations": ("SELECT", "INSERT"),
    # Claim, lease, finish and release its own jobs. It never enqueues a job
    # (the API does) or deletes one (the research worker purges).
    "research_jobs": ("SELECT", "UPDATE"),
    # The evaluation the verdict judged: its datasets, period, capital and
    # return-series hash, which the cut re-run must reproduce.
    "research_trial_evaluations": ("SELECT",),
    # Its heartbeat row, removed when it stops.
    "research_workers": DML,
    # The worker registers the server key it signs audit events and grades
    # with.
    "server_signing_keys": ("SELECT", "INSERT"),
    # The passing verdict it grades and that verdict's signed manifest,
    # which names the holdout start the verdict was reached under.
    "strategy_validations": ("SELECT",),
    "validation_manifests": ("SELECT",),
    # The graded strategy version, which the job's user must hold.
    "strategy_versions": ("SELECT",),
    "user_strategy_versions": ("SELECT",),
}

# The AI tier's tables, in the AI schema (migration 0012). The AI worker runs
# no audit, signing or trading code: it reads the feature flags, claims and
# finishes its own jobs and keeps its heartbeat row (backend/ai/queue.py).
AI_WORKER_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    # Read in the claim query and while a job runs; only operators set them.
    "flags": ("SELECT",),
    # Claim, lease, finish, reap and purge; only the application enqueues.
    "jobs": ("SELECT", "UPDATE", "DELETE"),
    "workers": DML,
    # The document registry (migration 0013): it records harvested papers
    # and marks deleted ones, but never deletes a row.
    "documents": ("SELECT", "INSERT", "UPDATE"),
    # Harvest bookkeeping (migration 0014): it takes rate slots and moves
    # the per-set cursors, and never deletes either.
    "sources": ("SELECT", "UPDATE"),
    "harvests": ("SELECT", "INSERT", "UPDATE"),
    # What OpenAlex and Crossref say of each document (migration 0015).
    "document_metadata": ("SELECT", "INSERT", "UPDATE"),
    # Extracted hypotheses (migration 0016): append-only, so it records an
    # extraction and its hypotheses and never changes either.
    "extractions": ("SELECT", "INSERT"),
    "hypotheses": ("SELECT", "INSERT"),
    # The daily token budget: it reserves, then settles, releases or orphans
    # a reservation, and never deletes one.
    "token_reservations": ("SELECT", "INSERT", "UPDATE"),
}
# The application's privileges in the AI schema: ``manage set-flag`` sets the
# flags; the application enqueues AI jobs and reads their outcome, the
# workers' heartbeats and the document registry. It never claims or finishes
# an AI job, and never writes a document.
APP_AI_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    "flags": ("SELECT", "INSERT", "UPDATE"),
    "jobs": ("SELECT", "INSERT"),
    "workers": ("SELECT",),
    "documents": ("SELECT",),
    "sources": ("SELECT",),
    "harvests": ("SELECT",),
    "document_metadata": ("SELECT",),
    "extractions": ("SELECT",),
    "hypotheses": ("SELECT",),
    "token_reservations": ("SELECT",),
}
# Set on qs_ai_worker itself, so they hold whatever its client sets.
AI_WORKER_CONNECTION_LIMIT = 8
AI_WORKER_SETTINGS = {"statement_timeout": "30s", "lock_timeout": "10s",
                      "idle_in_transaction_session_timeout": "60s"}

_ROLE_ATTRIBUTES = "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
# Objects an extension installed into the schema belong to the extension.
_NOT_EXTENSION_MEMBER = ("NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.objid = {oid}"
                         " AND d.classid = {catalog}::regclass AND d.deptype = 'e')")


@dataclass(frozen=True)
class Roles:
    """The role names; tests use their own prefix, since roles are cluster-wide."""
    prefix: str = "qs"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", self.prefix):
            raise ValueError(f"invalid role prefix {self.prefix!r}")

    @property
    def owner(self) -> str:
        return f"{self.prefix}_owner"

    @property
    def migrator(self) -> str:
        return f"{self.prefix}_migrator"

    @property
    def app(self) -> str:
        return f"{self.prefix}_app"

    @property
    def backup(self) -> str:
        return f"{self.prefix}_backup"

    @property
    def research_worker(self) -> str:
        return f"{self.prefix}_research_worker"

    @property
    def evaluator(self) -> str:
        return f"{self.prefix}_evaluator"

    @property
    def ai_worker(self) -> str:
        return f"{self.prefix}_ai_worker"

    @property
    def logins(self) -> tuple[str, ...]:
        return (self.migrator, self.app, self.research_worker, self.evaluator, self.ai_worker, self.backup)

    def table_privileges(self) -> tuple[tuple[str, dict[str, tuple[str, ...]]], ...]:
        """Each role that holds table privileges in the application schema, with its allow-list."""
        return ((self.app, APP_TABLE_PRIVILEGES), (self.research_worker, WORKER_TABLE_PRIVILEGES),
                (self.evaluator, EVALUATOR_TABLE_PRIVILEGES))

    def ai_table_privileges(self) -> tuple[tuple[str, dict[str, tuple[str, ...]]], ...]:
        """Each role that holds table privileges in the AI schema, with its allow-list."""
        return ((self.app, APP_AI_TABLE_PRIVILEGES), (self.ai_worker, AI_WORKER_TABLE_PRIVILEGES))


ROLES = Roles()


def ai_schema_for(schema: str) -> str:
    """The AI tier's schema beside the application schema ``schema``: ``ai``
    beside ``public``, ``<schema>_ai`` beside any other (the test suite gives
    each test a schema of its own)."""
    return "ai" if schema == "public" else f"{schema}_ai"


def _schema_exists(conn: Connection, schema: str) -> bool:
    return conn.execute("SELECT to_regnamespace(%s) IS NOT NULL", (schema,)).fetchone()[0]


def ensure_ai_schema(conn: Connection, roles: Roles = ROLES) -> None:
    """Create the AI schema, owned by the owner role, if it is missing and
    the connected role may create schemas (the bootstrap superuser does).

    Migrations on a provisioned schema run as the owner role, which may not
    create schemas, so ``database.migrate`` calls this before it becomes the
    owner, and migration 0012 finds the schema in place. A migrator that
    cannot create it gets 0012's error, which says to run provision-roles.
    """
    ai = ai_schema_for(conn.execute("SELECT current_schema()").fetchone()[0])
    if _schema_exists(conn, ai):
        return
    if conn.execute("SELECT has_database_privilege(current_database(), 'CREATE')").fetchone()[0]:
        conn.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
            sql.Identifier(ai), sql.Identifier(roles.owner)))


def owns_current_schema(conn: Connection, roles: Roles = ROLES) -> bool:
    """Whether provisioning has given the connection's schema to the owner role."""
    row = conn.execute(
        "SELECT n.nspowner = r.oid FROM pg_namespace n, pg_roles r"
        " WHERE n.nspname = current_schema() AND r.rolname = %s", (roles.owner,)).fetchone()
    return bool(row and row[0])


def can_become(conn: Connection, role: str) -> bool:
    return conn.execute("SELECT pg_has_role(current_user, %s, 'MEMBER')", (role,)).fetchone()[0]


def become(conn: Connection, role: str) -> None:
    """Act as ``role`` until the current transaction ends."""
    conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))


def provision(conn: Connection, roles: Roles = ROLES, passwords: dict[str, str] | None = None,
              lock_wait_ms: int = 10_000) -> dict:
    """Create or repair the roles and hand them the connection's current schema.

    Runs in the caller's transaction, so it applies completely or not at
    all. ``passwords`` maps a login role to its password; a role that does
    not exist yet needs one, and an existing role keeps its password unless
    one is given. Only a SCRAM verifier, computed here, reaches the server,
    so no password appears in its logs.

    Safe while the application runs: it never waits for a table lock while
    holding another (see _ALL_OR_NOTHING_LOCKS), and it retries while tables
    are in use, for up to ``lock_wait_ms``, before giving up with
    RuntimeError and changing nothing.
    """
    passwords = passwords or {}
    me, superuser = conn.execute(
        "SELECT rolname, rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()
    if not superuser:
        raise PermissionError(f"provisioning roles needs a superuser; connected as {me}")
    if me in (roles.owner, *roles.logins):
        raise ValueError(f"connected as {me}, one of the roles being provisioned")
    schema = conn.execute("SELECT current_schema()").fetchone()[0]
    if schema is None:
        raise RuntimeError("the connection has no current schema (check search_path)")
    deadline = time.monotonic() + lock_wait_ms / 1000
    pause, attempts = _RETRY_FIRST_PAUSE_S, 0
    while True:
        attempts += 1
        conn.execute("SAVEPOINT qs_provision")
        try:
            summary = _provision_attempt(conn, roles, passwords, schema)
        except (errors.LockNotAvailable, errors.DeadlockDetected) as exc:
            # Rolling back to the savepoint releases every lock this attempt took.
            conn.execute("ROLLBACK TO SAVEPOINT qs_provision")
            if time.monotonic() + pause > deadline:
                raise RuntimeError(
                    f"schema {schema} stayed in use for {lock_wait_ms} ms ({attempts} attempts; last: "
                    f"{exc.diag.message_primary}); nothing was changed, run it again when it is quieter"
                ) from exc
            time.sleep(pause * random.uniform(0.5, 1.0))
            pause = min(pause * 2, _RETRY_MAX_PAUSE_S)
            continue
        conn.execute("RELEASE SAVEPOINT qs_provision")
        return {**summary, "attempts": attempts}


# Changing an owner takes the object's ACCESS EXCLUSIVE lock. Taken one table
# at a time, that is a deadlock: provisioning held positions and waited for
# users while an order held users and waited for positions, and PostgreSQL
# broke the cycle by aborting one of them, possibly the order. So each attempt
# first takes the lock of every table it will change in one LOCK ... NOWAIT,
# which waits for nothing: if any table is in use it fails at once, the
# attempt lets go of everything, and provisioning tries again shortly after.
_ALL_OR_NOTHING_LOCKS = "LOCK TABLE {} IN ACCESS EXCLUSIVE MODE NOWAIT"
# LOCK refuses sequences, materialized views and foreign tables, so changing
# one of those may still wait; for less than deadlock_timeout (1 s by
# default), so a session queued behind the attempt is released before
# PostgreSQL would look for a deadlock and abort it.
_ATTEMPT_LOCK_TIMEOUT = "200ms"
_RETRY_FIRST_PAUSE_S, _RETRY_MAX_PAUSE_S = 0.02, 0.25


def _provision_attempt(conn: Connection, roles: Roles, passwords: dict[str, str], schema: str) -> dict:
    conn.execute(sql.SQL("SET LOCAL lock_timeout = {}").format(sql.Literal(_ATTEMPT_LOCK_TIMEOUT)))
    ai = ai_schema_for(schema)
    # A schema migrated before migration 0012 has no AI schema yet. Created
    # here, by the superuser, the owner role's next migration can fill it.
    if not _schema_exists(conn, ai):
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(ai)))
    lockable = [sql.Identifier(nsp, name) for nsp, name in conn.execute(
        "SELECT n.nspname, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"
        " WHERE n.nspname = ANY(%s) AND c.relkind IN ('r', 'p', 'v')"
        " AND c.relowner IS DISTINCT FROM (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
        + _NOT_EXTENSION_MEMBER.format(oid="c.oid", catalog="'pg_class'") + " ORDER BY 1, 2",
        ([schema, ai], roles.owner)).fetchall()]
    if lockable:
        conn.execute(sql.SQL(_ALL_OR_NOTHING_LOCKS).format(sql.SQL(", ").join(lockable)))
    created = _ensure_roles(conn, roles, passwords)
    _ensure_memberships(conn, roles)
    moved = _take_ownership(conn, roles, schema) + _take_ownership(conn, roles, ai)
    logins = sql.SQL(", ").join(map(sql.Identifier, roles.logins))
    # USAGE alone reaches no table; the migrator needs it on the application
    # schema to resolve it at all (without it current_schema() is NULL). The
    # AI worker holds none there, so nothing in it (tables, routines, types)
    # is within its reach.
    for name, users in ((schema, (roles.app, roles.research_worker, roles.evaluator, roles.migrator)),
                        (ai, (roles.app, roles.ai_worker))):
        conn.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC, {}").format(sql.Identifier(name), logins))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
            sql.Identifier(name), sql.SQL(", ").join(map(sql.Identifier, users))))
    unlisted = apply_grants(conn, roles)
    return {"schema": schema, "created": created, "ownership_moved": moved, "unlisted": unlisted}


def apply_grants(conn: Connection, roles: Roles = ROLES) -> list[str]:
    """Make the table privileges in the current schema, and in its AI schema,
    exactly the allow-lists.

    Returns the tables APP_TABLE_PRIVILEGES does not name; the application
    role has no privilege on them.
    """
    schema = conn.execute("SELECT current_schema()").fetchone()[0]
    # A schema provisioned before a login role was added lacks that role
    # until provision-roles runs again; it has nothing to revoke or receive.
    present = {name for (name,) in conn.execute(
        "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)", (list(roles.logins),))}
    tables = _grant_exactly(conn, schema, roles, present, roles.table_privileges())
    ai = ai_schema_for(schema)
    if _schema_exists(conn, ai):
        _grant_exactly(conn, ai, roles, present, roles.ai_table_privileges())
    return [table for table in tables if table not in APP_TABLE_PRIVILEGES]


def _grant_exactly(conn: Connection, schema: str, roles: Roles, present: set[str],
                   allow_lists: tuple[tuple[str, dict[str, tuple[str, ...]]], ...]) -> list[str]:
    """Revoke every table, sequence and routine privilege in ``schema`` from
    PUBLIC and the login roles, then grant the allow-lists; returns its tables."""
    schema_id = sql.Identifier(schema)
    grantees = sql.SQL(", ").join([sql.SQL("PUBLIC"), *(sql.Identifier(role) for role in roles.logins
                                                         if role in present)])
    # PostgreSQL grants EXECUTE on every new routine to PUBLIC. Triggers
    # still fire: EXECUTE is checked when a trigger is created, not when it runs.
    for kind in ("TABLES", "SEQUENCES", "ROUTINES"):
        conn.execute(sql.SQL("REVOKE ALL ON ALL {} IN SCHEMA {} FROM {}").format(
            sql.SQL(kind), schema_id, grantees))
    tables = [name for (name,) in conn.execute(
        "SELECT c.relname FROM pg_class c WHERE c.relnamespace = %s::regnamespace"
        " AND c.relkind IN ('r', 'p', 'v', 'm', 'f') AND "
        + _NOT_EXTENSION_MEMBER.format(oid="c.oid", catalog="'pg_class'") + " ORDER BY 1", (schema,))]
    for role, allowed in allow_lists:
        if role not in present:
            continue
        grantee = sql.Identifier(role)
        for table in tables:
            privileges = allowed.get(table)
            if privileges:
                conn.execute(sql.SQL("GRANT {} ON TABLE {} TO {}").format(
                    sql.SQL(", ").join(map(sql.SQL, privileges)), sql.Identifier(schema, table), grantee))
        # A sequence that feeds a column is usable wherever rows can be inserted.
        for (sequence,) in conn.execute(
                "SELECT s.relname FROM pg_class s JOIN pg_depend d ON d.objid = s.oid"
                " AND d.classid = 'pg_class'::regclass AND d.deptype IN ('a', 'i')"
                " JOIN pg_class t ON t.oid = d.refobjid"
                " WHERE s.relkind = 'S' AND s.relnamespace = %s::regnamespace"
                " AND t.relnamespace = %s::regnamespace AND t.relname = ANY(%s)",
                (schema, schema, [t for t in tables if "INSERT" in allowed.get(t, ())])).fetchall():
            conn.execute(sql.SQL("GRANT USAGE, SELECT ON SEQUENCE {} TO {}").format(
                sql.Identifier(schema, sequence), grantee))
    return tables


def _ensure_roles(conn: Connection, roles: Roles, passwords: dict[str, str]) -> list[str]:
    created = []
    for role, login in ((roles.owner, False), (roles.migrator, True), (roles.app, True),
                        (roles.research_worker, True), (roles.evaluator, True), (roles.ai_worker, True),
                        (roles.backup, True)):
        name = sql.Identifier(role)
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        if not exists:
            if login and not passwords.get(role):
                raise ValueError(f"a password is needed to create the login role {role}")
            conn.execute(sql.SQL("CREATE ROLE {}").format(name))
            created.append(role)
        # The migrator acts as the owner only through SET ROLE; INHERIT would
        # let it create objects owned by itself, which no grant would cover.
        inherit = "NOINHERIT" if role in (roles.owner, roles.migrator) else "INHERIT"
        conn.execute(sql.SQL("ALTER ROLE {} WITH {} {} {}").format(
            name, sql.SQL("LOGIN" if login else "NOLOGIN"), sql.SQL(inherit), sql.SQL(_ROLE_ATTRIBUTES)))
        if login and passwords.get(role):
            verifier = conn.pgconn.encrypt_password(
                passwords[role].encode(), role.encode(), b"scram-sha-256").decode()
            conn.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(name, sql.Literal(verifier)))
        # The AI worker's limits belong to the role, so they hold whatever its
        # client sets; no other role keeps a setting or limit of its own.
        limit = AI_WORKER_CONNECTION_LIMIT if role == roles.ai_worker else -1
        conn.execute(sql.SQL("ALTER ROLE {} CONNECTION LIMIT {}").format(name, sql.Literal(limit)))
        conn.execute(sql.SQL("ALTER ROLE {} RESET ALL").format(name))
        if role == roles.ai_worker:
            for setting, value in AI_WORKER_SETTINGS.items():
                conn.execute(sql.SQL("ALTER ROLE {} SET {} = {}").format(
                    name, sql.Identifier(setting), sql.Literal(value)))
    return created


def _ensure_memberships(conn: Connection, roles: Roles) -> None:
    """Exactly: the migrator in the owner role (no inheritance), the backup
    role in pg_read_all_data, and nothing else for the owner or any login role."""
    pg16 = conn.info.server_version >= 160000
    for member in (roles.owner, *roles.logins):
        for granted, grantor in conn.execute(
                "SELECT g.rolname, gr.rolname FROM pg_auth_members m"
                " JOIN pg_roles g ON g.oid = m.roleid JOIN pg_roles gr ON gr.oid = m.grantor"
                " WHERE m.member = (SELECT oid FROM pg_roles WHERE rolname = %s)", (member,)).fetchall():
            statement = sql.SQL("REVOKE {} FROM {}").format(sql.Identifier(granted), sql.Identifier(member))
            if pg16:  # since 16 a membership can be granted by several grantors
                statement += sql.SQL(" GRANTED BY {}").format(sql.Identifier(grantor))
            conn.execute(statement)
    # Since PostgreSQL 16 inheritance is a property of each grant, fixed when
    # it is made (ALTER ROLE ... NOINHERIT leaves existing grants inheriting);
    # earlier releases use the member role's NOINHERIT.
    conn.execute(sql.SQL("GRANT {} TO {}{}").format(
        sql.Identifier(roles.owner), sql.Identifier(roles.migrator),
        sql.SQL(" WITH INHERIT FALSE, SET TRUE" if pg16 else "")))
    conn.execute(sql.SQL("GRANT pg_read_all_data TO {}").format(sql.Identifier(roles.backup)))


def _take_ownership(conn: Connection, roles: Roles, schema: str) -> int:
    """Give the schema and every object in it to the owner role."""
    target = sql.Identifier(roles.owner)
    moved = 0
    if not conn.execute("SELECT n.nspowner = r.oid FROM pg_namespace n, pg_roles r"
                        " WHERE n.nspname = %s AND r.rolname = %s", (schema, roles.owner)).fetchone()[0]:
        conn.execute(sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(sql.Identifier(schema), target))
        moved += 1
    commands = {"r": "TABLE", "p": "TABLE", "f": "FOREIGN TABLE", "v": "VIEW",
                "m": "MATERIALIZED VIEW", "c": "TYPE", "S": "SEQUENCE"}
    # Tables first: a table's sequences and indexes follow it, so the second
    # pass finds only the free-standing sequences that are left.
    for kinds in (["r", "p", "f", "v", "m", "c"], ["S"]):
        for name, kind in conn.execute(
                "SELECT c.relname, c.relkind FROM pg_class c"
                " WHERE c.relnamespace = %s::regnamespace AND c.relkind = ANY(%s)"
                " AND c.relowner <> (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
                + _NOT_EXTENSION_MEMBER.format(oid="c.oid", catalog="'pg_class'") + " ORDER BY 1",
                (schema, kinds, roles.owner)).fetchall():
            conn.execute(sql.SQL("ALTER {} {} OWNER TO {}").format(
                sql.SQL(commands[kind]), sql.Identifier(schema, name), target))
            moved += 1
    for name, kind in conn.execute(
            "SELECT t.typname, t.typtype FROM pg_type t"
            " WHERE t.typnamespace = %s::regnamespace AND t.typtype IN ('e', 'd', 'r')"
            " AND t.typowner <> (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
            + _NOT_EXTENSION_MEMBER.format(oid="t.oid", catalog="'pg_type'"), (schema, roles.owner)).fetchall():
        conn.execute(sql.SQL("ALTER {} {} OWNER TO {}").format(
            sql.SQL("DOMAIN" if kind == "d" else "TYPE"), sql.Identifier(schema, name), target))
        moved += 1
    for (signature,) in conn.execute(
            "SELECT p.oid::regprocedure::text FROM pg_proc p"
            " WHERE p.pronamespace = %s::regnamespace"
            " AND p.proowner <> (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
            + _NOT_EXTENSION_MEMBER.format(oid="p.oid", catalog="'pg_proc'"), (schema, roles.owner)).fetchall():
        # regprocedure output is already a correctly quoted signature,
        # schema-qualified when the schema is not on the search path.
        conn.execute(sql.SQL("ALTER ROUTINE {} OWNER TO {}").format(sql.SQL(signature), target))
        moved += 1
    return moved
