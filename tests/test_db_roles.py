"""Least-privilege database roles (backend/db_roles.py).

Roles are cluster-wide, so every test provisions its own, under a random
prefix, on a schema of its own, and drops them when it ends. The privilege
checks read the catalog for every table in the schema, so a table nobody
thought about fails them instead of slipping through.
"""
import logging
import secrets
import threading
import time
import uuid

import psycopg
import pytest
from alembic import command
from psycopg import errors
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from backend import db_roles, manage, models, worker
from backend.database import (Base, _alembic_config, _configure_session, current_revision,
                              head_revision, migrate)
from backend.services import research_jobs, security_service
from conftest import _admin, _libpq, schema_url
from research_job_helpers import InlineJobProcess

_RELATIONS = ("SELECT c.relname FROM pg_class c WHERE c.relnamespace = %s::regnamespace"
              " AND c.relkind IN ('r', 'p', 'v', 'm', 'f') ORDER BY 1")


class Cluster:
    """A schema at the latest revision plus a set of roles to provision on it."""

    def __init__(self, engine):
        self.engine = engine
        with engine.connect() as connection:
            self.schema = connection.execute(text("SELECT current_schema()")).scalar()
        self.roles = db_roles.Roles(f"qst{uuid.uuid4().hex[:10]}")
        self.passwords = {role: secrets.token_urlsafe(18) for role in self.roles.logins}
        self._closers = []

    def provision(self, passwords=None) -> dict:
        with self.engine.begin() as connection:
            return db_roles.provision(connection.connection.driver_connection, self.roles,
                                      self.passwords if passwords is None else passwords)

    def url(self, role=None):
        url = schema_url(self.schema)
        return url if role is None else url.set(username=role, password=self.passwords[role])

    def connect(self, role=None) -> psycopg.Connection:
        """Autocommit connection as ``role`` (default: the test superuser)."""
        conn = psycopg.connect(_libpq(self.url(role)), autocommit=True, application_name=self.schema)
        self._closers.append(conn.close)
        return conn

    def engine_as(self, role):
        engine = create_engine(self.url(role), connect_args={"application_name": self.schema})
        event.listen(engine, "connect", _configure_session)
        self._closers.append(engine.dispose)
        return engine

    def migrate_as_owner(self, revision: str) -> None:
        """Move the schema to ``revision`` as a migration would (via the migrator)."""
        with self.engine_as(self.roles.migrator).begin() as connection:
            db_roles.become(connection.connection.driver_connection, self.roles.owner)
            if revision < current_revision(connection):
                command.downgrade(_alembic_config(connection), revision)
            else:
                command.upgrade(_alembic_config(connection), revision)

    def drop(self) -> None:
        for close in reversed(self._closers):
            close()
        names = [self.roles.owner, *self.roles.logins]
        with psycopg.connect(_libpq(self.url()), autocommit=True) as conn:
            existing = [name for (name,) in conn.execute(
                "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)", (names,))]
        if existing:
            quoted = ", ".join(f'"{name}"' for name in existing)
            _admin(f"DROP OWNED BY {quoted} CASCADE")
            _admin(f"DROP ROLE {quoted}")


@pytest.fixture
def cluster(make_engine):
    c = Cluster(make_engine(migrated=True))
    yield c
    c.drop()


@pytest.fixture
def provisioned(cluster):
    cluster.provision()
    return cluster


def table_privileges(conn, schema: str, role: str) -> dict[str, set[str]]:
    held = {name: set() for (name,) in conn.execute(_RELATIONS, (schema,))}
    for name, privilege in conn.execute(
            "SELECT c.relname, p FROM pg_class c, unnest(%s::text[]) p"
            " WHERE c.relnamespace = %s::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')"
            " AND has_table_privilege(%s, c.oid, p)",
            (list(db_roles.TABLE_PRIVILEGES), schema, role)):
        held[name].add(privilege)
    return held


def assert_exactly_provisioned(c: Cluster) -> None:
    roles, su = c.roles, c.connect()
    tables = [name for (name,) in su.execute(_RELATIONS, (c.schema,))]
    assert set(tables) == set(db_roles.APP_TABLE_PRIVILEGES)
    assert table_privileges(su, c.schema, roles.app) == {
        name: set(privileges) for name, privileges in db_roles.APP_TABLE_PRIVILEGES.items()}
    assert table_privileges(su, c.schema, roles.backup) == {name: {"SELECT"} for name in tables}
    for role in (roles.migrator, "public"):
        assert table_privileges(su, c.schema, role) == {name: set() for name in tables}, role
    schema_privileges = {role: tuple(su.execute(
        "SELECT has_schema_privilege(%s, %s, 'USAGE'), has_schema_privilege(%s, %s, 'CREATE')",
        (role, c.schema, role, c.schema)).fetchone()) for role in (*roles.logins, "public")}
    assert schema_privileges == {roles.migrator: (True, False), roles.app: (True, False),
                                 roles.backup: (True, False), "public": (False, False)}
    assert su.execute(
        "SELECT count(*) FROM pg_class c WHERE c.relnamespace = %s::regnamespace"
        " AND c.relowner <> %s::regrole", (c.schema, roles.owner)).fetchone()[0] == 0
    assert su.execute("SELECT nspowner::regrole::text FROM pg_namespace WHERE nspname = %s",
                      (c.schema,)).fetchone()[0] == roles.owner
    attributes = {row[0]: row[1:] for row in su.execute(
        "SELECT rolname, rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls,"
        " rolcanlogin, rolinherit FROM pg_roles WHERE rolname = ANY(%s)",
        ([roles.owner, *roles.logins],))}
    assert attributes == {roles.owner: (False, False, False), roles.migrator: (False, True, False),
                          roles.app: (False, True, True), roles.backup: (False, True, True)}
    names = [roles.owner, *roles.logins]
    memberships = su.execute(
        "SELECT m.member::regrole::text, m.roleid::regrole::text FROM pg_auth_members m"
        " WHERE m.member::regrole::text = ANY(%s) OR m.roleid::regrole::text = ANY(%s)",
        (names, names)).fetchall()
    assert sorted(memberships) == sorted([(roles.migrator, roles.owner),
                                          (roles.backup, "pg_read_all_data")])
    if su.info.server_version >= 160000:
        assert su.execute(
            "SELECT inherit_option FROM pg_auth_members WHERE member = %s::regrole",
            (roles.migrator,)).fetchone() == (False,)


def test_the_allow_list_names_every_table():
    assert set(db_roles.APP_TABLE_PRIVILEGES) == set(Base.metadata.tables) | {"alembic_version"}
    assert all(p and set(p) <= set(db_roles.TABLE_PRIVILEGES)
               for p in db_roles.APP_TABLE_PRIVILEGES.values())


def test_privileges_are_exactly_the_allow_list(provisioned):
    assert_exactly_provisioned(provisioned)


# Each of these needs a privilege the bootstrap superuser has and the
# application role must not.
REFUSED = (
    "TRUNCATE trades",
    "DELETE FROM alembic_version",
    "UPDATE alembic_version SET version_num = '0001'",
    "CREATE TABLE intruder (x int)",
    "CREATE SCHEMA intruder",
    "ALTER TABLE trades ADD COLUMN intruder int",
    "DROP TABLE audit_logs",
    "ALTER TABLE audit_logs DISABLE TRIGGER ALL",
    "SET session_replication_role = replica",
    "COPY (SELECT 1) TO PROGRAM 'echo pwned'",
    "SELECT pg_read_file('postgresql.conf')",
    "CREATE ROLE intruder",
    "ALTER ROLE {app} BYPASSRLS",
    "SET ROLE {owner}",
)


def test_the_app_role_is_refused_everything_but_its_own_rows(provisioned):
    roles = provisioned.roles
    app = provisioned.connect(roles.app)
    for statement in REFUSED:
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute(statement.format(app=roles.app, owner=roles.owner))
    app.execute("INSERT INTO research_workers (id, hostname, pid, started_at, last_seen_at)"
                " VALUES ('w1', 'h', 1, now(), now())")
    app.execute("UPDATE research_workers SET pid = 2 WHERE id = 'w1'")
    assert app.execute("SELECT pid FROM research_workers").fetchall() == [(2,)]
    app.execute("DELETE FROM research_workers WHERE id = 'w1'")
    assert app.execute("SELECT version_num FROM alembic_version").fetchone() == (head_revision(),)


def test_the_application_runs_as_the_app_role(provisioned, monkeypatch):
    """A user row, a signed audit event, a queued job claimed with SKIP
    LOCKED and finished with its own audit event, a full chain verification
    and the retention purge: the code paths of an API process and a worker."""
    monkeypatch.setitem(research_jobs.KINDS, "test_echo", research_jobs.JobKind(
        dict, "research_job_helpers:echo_task", "TEST_JOB"))
    session_factory = sessionmaker(bind=provisioned.engine_as(provisioned.roles.app))
    with session_factory() as db:
        assert db.execute(text("SELECT current_user")).scalar() == provisioned.roles.app
        user = models.User(email="roles@example.com", password_hash="x", role="user")
        db.add(user)
        db.commit()
        security_service.write_audit_log(db, user.id, "USER_REGISTERED", "user", user.id, {})
        job = research_jobs.enqueue(db, user.id, "test_echo", {"n": 1})
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=session_factory,
                           lease_seconds=60, timeout_seconds=60, poll_seconds=0.1)
    assert runner.run_once() is True
    with session_factory() as db:
        assert research_jobs.view(db, db.get(models.ResearchJob, job.id))["status"] == "succeeded"
        status = security_service.audit_chain_status(db, full=True)
        assert status["valid"] and status["links"] == 2, status
        assert research_jobs.purge_finished(db, retention_days=0) == 1


def test_the_backup_role_reads_everything_and_writes_nothing(provisioned):
    backup = provisioned.connect(provisioned.roles.backup)
    for (table,) in backup.execute(_RELATIONS, (provisioned.schema,)).fetchall():
        backup.execute(f'SELECT count(*) FROM "{table}"')
    with pytest.raises(errors.InsufficientPrivilege):
        backup.execute("DELETE FROM users")


def test_the_migrator_holds_nothing_until_it_becomes_the_owner(provisioned):
    roles = provisioned.roles
    migrator = provisioned.connect(roles.migrator)
    with pytest.raises(errors.InsufficientPrivilege):
        migrator.execute("SELECT count(*) FROM users")
    with pytest.raises(errors.InsufficientPrivilege):
        migrator.execute("CREATE TABLE made_by_migrator (x int)")
    with migrator.transaction():
        db_roles.become(migrator, roles.owner)
        migrator.execute("CREATE TABLE made_as_owner (x int)")
    assert migrator.execute("SELECT current_user").fetchone() == (roles.migrator,)
    assert migrator.execute("SELECT relowner::regrole::text FROM pg_class"
                            " WHERE oid = 'made_as_owner'::regclass").fetchone() == (roles.owner,)


@pytest.mark.parametrize("as_role", ["migrator", "superuser"])
def test_migrations_run_as_the_owner_and_restore_exact_grants(provisioned, monkeypatch, caplog, as_role):
    """Whoever migrates a provisioned schema, what the migration creates
    belongs to the owner role; a new table gets exactly its allow-list entry
    (sequence included), or nothing if it has none; and drift is undone."""
    roles = provisioned.roles
    provisioned.migrate_as_owner("0002")
    su = provisioned.connect()
    su.execute(f'GRANT TRUNCATE ON trades TO "{roles.app}"')
    upgrade = command.upgrade

    def upgrade_creating_tables(config, revision):
        connection = config.attributes["connection"]
        connection.execute(text("CREATE TABLE unlisted_table (x int)"))
        connection.execute(text("CREATE TABLE listed_table (id bigserial PRIMARY KEY, x int)"))
        upgrade(config, revision)

    monkeypatch.setattr(command, "upgrade", upgrade_creating_tables)
    monkeypatch.setitem(db_roles.APP_TABLE_PRIVILEGES, "listed_table", ("SELECT", "INSERT"))
    engine = provisioned.engine_as(roles.migrator) if as_role == "migrator" else provisioned.engine
    with caplog.at_level(logging.WARNING, logger="backend.database"):
        assert migrate(engine, roles) == head_revision()
    assert [r.getMessage().rsplit(": ", 1)[1] for r in caplog.records
            if "APP_TABLE_PRIVILEGES" in r.getMessage()] == ["unlisted_table"]
    held = table_privileges(su, provisioned.schema, roles.app)
    assert held.pop("unlisted_table") == set()
    assert held == {name: set(p) for name, p in db_roles.APP_TABLE_PRIVILEGES.items()}
    assert su.execute("SELECT count(*) FROM pg_class WHERE relnamespace = %s::regnamespace"
                      " AND relowner <> %s::regrole", (provisioned.schema, roles.owner)).fetchone() == (0,)
    app = provisioned.connect(roles.app)
    assert app.execute("INSERT INTO listed_table (x) VALUES (1) RETURNING id").fetchone() == (1,)
    with pytest.raises(errors.InsufficientPrivilege):
        app.execute("INSERT INTO unlisted_table VALUES (1)")
    su.execute("DROP TABLE unlisted_table, listed_table")
    monkeypatch.undo()
    assert_exactly_provisioned(provisioned)


def test_the_app_role_may_start_on_a_current_schema_but_not_migrate_one(provisioned):
    roles = provisioned.roles
    app_engine = provisioned.engine_as(roles.app)
    assert migrate(app_engine, roles) == head_revision()
    provisioned.migrate_as_owner("0002")
    with pytest.raises(RuntimeError, match=f"run `python -m backend.manage migrate` as {roles.migrator}"):
        migrate(app_engine, roles)
    with provisioned.engine.connect() as connection:
        assert current_revision(connection) == "0002"


def test_roles_that_do_not_own_the_schema_change_nothing(provisioned, make_engine):
    """On a cluster that has the roles, an unprovisioned schema still
    migrates as whoever connects (the test suite's own schemas rely on it)."""
    other = make_engine(migrated=True)
    assert migrate(other, provisioned.roles) == head_revision()
    with other.connect() as connection:
        assert connection.execute(text(
            "SELECT count(*) FROM pg_class WHERE relnamespace = current_schema()::regnamespace"
            " AND relowner = CAST(:owner AS regrole)"), {"owner": provisioned.roles.owner}).scalar() == 0


def test_provisioning_again_repairs_drift(provisioned):
    roles, su = provisioned.roles, provisioned.connect()
    for statement in (
            f'ALTER ROLE "{roles.app}" CREATEROLE BYPASSRLS',
            f'ALTER ROLE "{roles.migrator}" INHERIT',
            f'GRANT "{roles.owner}" TO "{roles.app}"',
            f'GRANT pg_read_all_data TO "{roles.migrator}"',
            f'GRANT TRUNCATE ON trades TO "{roles.app}"',
            f'GRANT INSERT ON users TO "{roles.backup}"',
            "GRANT SELECT ON users TO PUBLIC",
            f'GRANT CREATE ON SCHEMA "{provisioned.schema}" TO "{roles.app}", PUBLIC',
            "ALTER TABLE users OWNER TO CURRENT_USER"):
        su.execute(statement)
    if su.info.server_version >= 160000:
        su.execute(f'GRANT "{roles.owner}" TO "{roles.migrator}" WITH INHERIT TRUE')
    summary = provisioned.provision(passwords={})
    assert summary["created"] == [] and summary["unlisted"] == []
    assert_exactly_provisioned(provisioned)


def test_new_login_roles_need_passwords_and_nothing_is_left_behind(cluster):
    with pytest.raises(ValueError, match="password"):
        cluster.provision(passwords={})
    su = cluster.connect()
    assert su.execute("SELECT count(*) FROM pg_roles WHERE rolname LIKE %s",
                      (cluster.roles.prefix + "%",)).fetchone() == (0,)
    assert su.execute("SELECT nspowner::regrole::text FROM pg_namespace WHERE nspname = %s",
                      (cluster.schema,)).fetchone() != (cluster.roles.owner,)


def _hold(cluster, *statements) -> psycopg.Connection:
    """Another session, left inside a transaction that has run ``statements``."""
    conn = psycopg.connect(_libpq(cluster.url()), application_name=cluster.schema)
    cluster._closers.append(conn.close)
    for statement in statements:
        conn.execute(statement)
    return conn


def _provision_in_background(cluster, **kwargs):
    outcome = {}

    def run():
        try:
            with cluster.engine.begin() as connection:
                outcome["summary"] = db_roles.provision(connection.connection.driver_connection, cluster.roles,
                                                        cluster.passwords, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc
    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def _finishes_promptly(conn, statement) -> float:
    started = time.monotonic()
    conn.execute(statement)
    return time.monotonic() - started


def test_provisioning_cannot_deadlock_with_the_order_path(cluster):
    """An order locks its user, then reads positions. Provisioning used to take
    positions while waiting for users, a lock cycle PostgreSQL broke by
    aborting one side: the provisioning or the user's order."""
    order = _hold(cluster, "SELECT id FROM users FOR UPDATE")
    thread, outcome = _provision_in_background(cluster)
    time.sleep(0.5)  # provisioning is now retrying around the order's lock
    assert _finishes_promptly(order, "SELECT count(*) FROM positions") < 0.5
    order.commit()
    thread.join(30)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["summary"]["attempts"] > 1
    assert_exactly_provisioned(cluster)


def test_provisioning_never_keeps_a_queue_waiting_long_enough_to_look_for_deadlocks(cluster):
    """A free-standing sequence cannot be LOCKed in advance, so taking it can
    still wait, while the attempt holds its table locks. That wait must end
    before a session queued behind provisioning reaches PostgreSQL's deadlock
    check (1 s), or that session is the one aborted; and it must stay short,
    or every query on every table stalls behind it."""
    cluster.connect().execute("CREATE SEQUENCE free_standing")
    session = _hold(cluster, "SELECT nextval('free_standing')")
    thread, outcome = _provision_in_background(cluster)
    time.sleep(0.5)
    assert _finishes_promptly(cluster.connect(), "SELECT count(*) FROM users") < 0.5  # a bystander
    assert _finishes_promptly(session, "SELECT count(*) FROM users") < 0.9
    session.commit()
    thread.join(30)
    assert "error" not in outcome, outcome.get("error")
    su = cluster.connect()
    assert su.execute("SELECT relowner::regrole::text FROM pg_class WHERE oid = 'free_standing'::regclass"
                      ).fetchone() == (cluster.roles.owner,)
    assert_exactly_provisioned(cluster)


def test_provisioning_gives_up_cleanly_while_a_table_stays_busy(cluster):
    reader = _hold(cluster, "SELECT count(*) FROM trades")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='"trades"'):
        with cluster.engine.begin() as connection:
            db_roles.provision(connection.connection.driver_connection, cluster.roles, cluster.passwords,
                               lock_wait_ms=800)
    assert time.monotonic() - started < 5
    assert _finishes_promptly(reader, "SELECT count(*) FROM users") < 0.5  # never queued behind it
    su = cluster.connect()
    assert su.execute("SELECT count(*) FROM pg_roles WHERE rolname LIKE %s",
                      (cluster.roles.prefix + "%",)).fetchone() == (0,)
    reader.rollback()


def test_provisioning_again_takes_no_table_lock(provisioned):
    """Once the schema is handed over, a re-run (drift repair, a new password)
    changes only grants, which wait for no table lock."""
    reader = _hold(provisioned, "SELECT count(*) FROM trades", "SELECT id FROM users FOR UPDATE")
    with provisioned.engine.begin() as connection:
        summary = db_roles.provision(connection.connection.driver_connection, provisioned.roles,
                                     lock_wait_ms=800)
    assert summary["attempts"] == 1 and summary["ownership_moved"] == 0
    reader.rollback()


def test_only_a_superuser_can_provision(provisioned):
    with provisioned.engine_as(provisioned.roles.app).begin() as connection:
        with pytest.raises(PermissionError):
            db_roles.provision(connection.connection.driver_connection, provisioned.roles)


class _Recorder:
    """A psycopg connection that keeps the text of every statement it sends."""

    def __init__(self, conn):
        self._conn, self.sent = conn, []

    def execute(self, query, params=None):
        self.sent.append(query if isinstance(query, str) else query.as_string(self._conn))
        return self._conn.execute(query, params)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_passwords_reach_the_server_only_as_scram_verifiers(cluster):
    """The server stores a SCRAM verifier either way; what matters is that
    the statement it receives (and may log) never carries the password."""
    with cluster.engine.begin() as connection:
        recorder = _Recorder(connection.connection.driver_connection)
        db_roles.provision(recorder, cluster.roles, cluster.passwords)
    sent = "\n".join(recorder.sent)
    assert sent.count("PASSWORD 'SCRAM-SHA-256$") == len(cluster.roles.logins)
    assert not any(password in sent for password in cluster.passwords.values())
    for role in cluster.roles.logins:  # each verifier accepts its password
        cluster.connect(role).close()


def test_the_provision_roles_command(cluster, monkeypatch, capsys):
    settings = {cluster.roles.migrator: "DB_MIGRATOR_PASSWORD", cluster.roles.app: "DB_APP_PASSWORD",
                cluster.roles.backup: "DB_BACKUP_PASSWORD"}
    for role, name in settings.items():
        monkeypatch.setenv(name, cluster.passwords[role])
    assert manage._provision_roles(cluster.engine, cluster.roles) == 0
    out = capsys.readouterr().out
    assert all(password not in out for password in cluster.passwords.values())
    assert f"roles created: {cluster.roles.owner}, {cluster.roles.migrator}" in out
    assert_exactly_provisioned(cluster)
    for name in settings.values():
        monkeypatch.delenv(name)
    assert manage._provision_roles(cluster.engine, cluster.roles) == 0
    assert "roles created: none" in capsys.readouterr().out
    cluster.connect(cluster.roles.app).close()  # the password set the first time still works
