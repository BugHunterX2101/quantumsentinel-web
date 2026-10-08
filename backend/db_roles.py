"""Least-privilege PostgreSQL roles.

Connected as the cluster's bootstrap superuser (the role POSTGRES_USER
creates), the application bypasses every grant, row-level security policy and
trigger, and any SQL it runs can drop the schema, read server files or start
programs (COPY ... TO PROGRAM). ``python -m backend.manage provision-roles``,
run as that superuser, hands the schema to four roles:

    qs_owner     NOLOGIN. Owns the schema and every object in it.
    qs_migrator  LOGIN, a NOINHERIT member of qs_owner: it holds no table
                 privilege of its own. ``database.migrate`` runs migrations
                 after SET LOCAL ROLE qs_owner, so whatever a migration
                 creates is owned by qs_owner.
    qs_app       LOGIN. The API and research workers: exactly the table
                 privileges in APP_TABLE_PRIVILEGES. No DDL, no TRUNCATE; it
                 cannot own objects, change roles or switch triggers off.
    qs_backup    LOGIN. pg_read_all_data, for pg_dump.

APP_TABLE_PRIVILEGES is an allow-list. A table missing from it gets no
privilege at all, and tests/test_db_roles.py fails until it is listed, so a
new table is deny-by-default. ``database.migrate`` reapplies the grants after
every migration that changes the schema revision.

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
    "idempotency_records": DML,
    "key_pairs": DML,
    "order_security_records": DML,
    "paper_accounts": DML,
    "positions": DML,
    "refresh_tokens": DML,
    "research_experiments": DML,
    "research_jobs": DML,
    "research_trials": DML,
    "research_workers": DML,
    "server_signing_keys": DML,
    "signals": DML,
    "strategies": DML,
    "trades": DML,
    "users": DML,
    "webhooks": DML,
}

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
    def logins(self) -> tuple[str, str, str]:
        return (self.migrator, self.app, self.backup)


ROLES = Roles()


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
    lockable = [sql.Identifier(name) for (name,) in conn.execute(
        "SELECT c.relname FROM pg_class c WHERE c.relnamespace = current_schema()::regnamespace"
        " AND c.relkind IN ('r', 'p', 'v')"
        " AND c.relowner IS DISTINCT FROM (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
        + _NOT_EXTENSION_MEMBER.format(oid="c.oid", catalog="'pg_class'") + " ORDER BY 1",
        (roles.owner,)).fetchall()]
    if lockable:
        conn.execute(sql.SQL(_ALL_OR_NOTHING_LOCKS).format(sql.SQL(", ").join(lockable)))
    created = _ensure_roles(conn, roles, passwords)
    _ensure_memberships(conn, roles)
    moved = _take_ownership(conn, roles, schema)
    schema_id = sql.Identifier(schema)
    conn.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC, {}").format(
        schema_id, sql.SQL(", ").join(map(sql.Identifier, roles.logins))))
    # USAGE alone reaches no table; the migrator needs it to resolve the
    # schema at all (without it current_schema() is NULL).
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}, {}").format(
        schema_id, sql.Identifier(roles.app), sql.Identifier(roles.migrator)))
    unlisted = apply_grants(conn, roles)
    return {"schema": schema, "created": created, "ownership_moved": moved, "unlisted": unlisted}


def apply_grants(conn: Connection, roles: Roles = ROLES) -> list[str]:
    """Make the table privileges in the current schema exactly the allow-list.

    Returns the tables the allow-list does not name; the application role
    has no privilege on them.
    """
    schema = sql.Identifier(conn.execute("SELECT current_schema()").fetchone()[0])
    grantees = sql.SQL(", ").join([sql.SQL("PUBLIC"), *map(sql.Identifier, roles.logins)])
    for kind in ("TABLES", "SEQUENCES"):
        conn.execute(sql.SQL("REVOKE ALL ON ALL {} IN SCHEMA {} FROM {}").format(
            sql.SQL(kind), schema, grantees))
    tables = [name for (name,) in conn.execute(
        "SELECT c.relname FROM pg_class c WHERE c.relnamespace = current_schema()::regnamespace"
        " AND c.relkind IN ('r', 'p', 'v', 'm', 'f') AND "
        + _NOT_EXTENSION_MEMBER.format(oid="c.oid", catalog="'pg_class'") + " ORDER BY 1")]
    app = sql.Identifier(roles.app)
    for table in tables:
        privileges = APP_TABLE_PRIVILEGES.get(table)
        if privileges:
            conn.execute(sql.SQL("GRANT {} ON TABLE {} TO {}").format(
                sql.SQL(", ").join(map(sql.SQL, privileges)), sql.Identifier(table), app))
    # A sequence that feeds a column is usable wherever rows can be inserted.
    for (sequence,) in conn.execute(
            "SELECT s.relname FROM pg_class s JOIN pg_depend d ON d.objid = s.oid"
            " AND d.classid = 'pg_class'::regclass AND d.deptype IN ('a', 'i')"
            " JOIN pg_class t ON t.oid = d.refobjid"
            " WHERE s.relkind = 'S' AND s.relnamespace = current_schema()::regnamespace"
            " AND t.relname = ANY(%s)",
            ([t for t in tables if "INSERT" in APP_TABLE_PRIVILEGES.get(t, ())],)).fetchall():
        conn.execute(sql.SQL("GRANT USAGE, SELECT ON SEQUENCE {} TO {}").format(
            sql.Identifier(sequence), app))
    return [table for table in tables if table not in APP_TABLE_PRIVILEGES]


def _ensure_roles(conn: Connection, roles: Roles, passwords: dict[str, str]) -> list[str]:
    created = []
    for role, login in ((roles.owner, False), (roles.migrator, True), (roles.app, True),
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
    return created


def _ensure_memberships(conn: Connection, roles: Roles) -> None:
    """Exactly: the migrator in the owner role (no inheritance), the backup
    role in pg_read_all_data, and nothing else for any of the four."""
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
    if not owns_current_schema(conn, roles):
        conn.execute(sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(sql.Identifier(schema), target))
        moved += 1
    commands = {"r": "TABLE", "p": "TABLE", "f": "FOREIGN TABLE", "v": "VIEW",
                "m": "MATERIALIZED VIEW", "c": "TYPE", "S": "SEQUENCE"}
    # Tables first: a table's sequences and indexes follow it, so the second
    # pass finds only the free-standing sequences that are left.
    for kinds in (["r", "p", "f", "v", "m", "c"], ["S"]):
        for name, kind in conn.execute(
                "SELECT c.relname, c.relkind FROM pg_class c"
                " WHERE c.relnamespace = current_schema()::regnamespace AND c.relkind = ANY(%s)"
                " AND c.relowner <> (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
                + _NOT_EXTENSION_MEMBER.format(oid="c.oid", catalog="'pg_class'") + " ORDER BY 1",
                (kinds, roles.owner)).fetchall():
            conn.execute(sql.SQL("ALTER {} {} OWNER TO {}").format(
                sql.SQL(commands[kind]), sql.Identifier(name), target))
            moved += 1
    for name, kind in conn.execute(
            "SELECT t.typname, t.typtype FROM pg_type t"
            " WHERE t.typnamespace = current_schema()::regnamespace AND t.typtype IN ('e', 'd', 'r')"
            " AND t.typowner <> (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
            + _NOT_EXTENSION_MEMBER.format(oid="t.oid", catalog="'pg_type'"), (roles.owner,)).fetchall():
        conn.execute(sql.SQL("ALTER {} {} OWNER TO {}").format(
            sql.SQL("DOMAIN" if kind == "d" else "TYPE"), sql.Identifier(name), target))
        moved += 1
    for (signature,) in conn.execute(
            "SELECT p.oid::regprocedure::text FROM pg_proc p"
            " WHERE p.pronamespace = current_schema()::regnamespace"
            " AND p.proowner <> (SELECT oid FROM pg_roles WHERE rolname = %s) AND "
            + _NOT_EXTENSION_MEMBER.format(oid="p.oid", catalog="'pg_proc'"), (roles.owner,)).fetchall():
        # regprocedure output is already a correctly quoted signature.
        conn.execute(sql.SQL("ALTER ROUTINE {} OWNER TO {}").format(sql.SQL(signature), target))
        moved += 1
    return moved
