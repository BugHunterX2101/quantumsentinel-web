"""Least-privilege database roles (backend/db_roles.py).

Roles are cluster-wide, so every test provisions its own, under a random
prefix, on a schema of its own, and drops them when it ends. The privilege
checks read the catalog for every table in the schema, so a table nobody
thought about fails them instead of slipping through.
"""
import json
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

from backend import db_roles, main, manage, models, schemas, worker
from backend.database import (Base, _alembic_config, _configure_session, current_revision,
                              head_revision, migrate)
from backend.services import backtest_service, research_jobs, security_service
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
    assert table_privileges(su, c.schema, roles.research_worker) == {
        name: set(db_roles.WORKER_TABLE_PRIVILEGES.get(name, ())) for name in tables}
    assert table_privileges(su, c.schema, roles.evaluator) == {
        name: set(db_roles.EVALUATOR_TABLE_PRIVILEGES.get(name, ())) for name in tables}
    assert table_privileges(su, c.schema, roles.backup) == {name: {"SELECT"} for name in tables}
    for role in (roles.migrator, roles.ai_worker, "public"):
        assert table_privileges(su, c.schema, role) == {name: set() for name in tables}, role
    ai = db_roles.ai_schema_for(c.schema)
    ai_tables = [name for (name,) in su.execute(_RELATIONS, (ai,))]
    assert set(ai_tables) == set(db_roles.AI_WORKER_TABLE_PRIVILEGES) == set(db_roles.APP_AI_TABLE_PRIVILEGES)
    assert table_privileges(su, ai, roles.app) == {
        name: set(privileges) for name, privileges in db_roles.APP_AI_TABLE_PRIVILEGES.items()}
    assert table_privileges(su, ai, roles.ai_worker) == {
        name: set(privileges) for name, privileges in db_roles.AI_WORKER_TABLE_PRIVILEGES.items()}
    assert table_privileges(su, ai, roles.backup) == {name: {"SELECT"} for name in ai_tables}
    for role in (roles.migrator, roles.research_worker, roles.evaluator, "public"):
        assert table_privileges(su, ai, role) == {name: set() for name in ai_tables}, role

    def schema_privileges(schema):
        return {role: tuple(su.execute(
            "SELECT has_schema_privilege(%s, %s, 'USAGE'), has_schema_privilege(%s, %s, 'CREATE')",
            (role, schema, role, schema)).fetchone()) for role in (*roles.logins, "public")}

    # The backup role's USAGE comes from pg_read_all_data.
    assert schema_privileges(c.schema) == {
        roles.migrator: (True, False), roles.app: (True, False), roles.research_worker: (True, False),
        roles.evaluator: (True, False), roles.ai_worker: (False, False), roles.backup: (True, False),
        "public": (False, False)}
    assert schema_privileges(ai) == {
        roles.migrator: (False, False), roles.app: (True, False), roles.research_worker: (False, False),
        roles.evaluator: (False, False), roles.ai_worker: (True, False), roles.backup: (True, False),
        "public": (False, False)}
    for schema in (c.schema, ai):
        assert su.execute(
            "SELECT count(*) FROM pg_class c WHERE c.relnamespace = %s::regnamespace"
            " AND c.relowner <> %s::regrole", (schema, roles.owner)).fetchone()[0] == 0
        assert su.execute("SELECT nspowner::regrole::text FROM pg_namespace WHERE nspname = %s",
                          (schema,)).fetchone()[0] == roles.owner
        # No login role and not PUBLIC may run a routine of either schema.
        assert su.execute(
            "SELECT count(*) FROM pg_proc p, unnest(%s::text[]) r WHERE p.pronamespace = %s::regnamespace"
            " AND has_function_privilege(r, p.oid, 'EXECUTE')",
            ([*(role for role in roles.logins if role != roles.backup), "public"], schema)).fetchone()[0] == 0
    assert dict(su.execute(
        "SELECT r.rolname, r.rolconnlimit FROM pg_roles r WHERE r.rolname = ANY(%s)",
        ([roles.owner, *roles.logins],)).fetchall()) == {
            role: db_roles.AI_WORKER_CONNECTION_LIMIT if role == roles.ai_worker else -1
            for role in (roles.owner, *roles.logins)}
    settings = su.execute(
        "SELECT s.setrole::regrole::text, s.setdatabase, s.setconfig FROM pg_db_role_setting s"
        " WHERE s.setrole::regrole::text = ANY(%s)", ([roles.owner, *roles.logins],)).fetchall()
    assert [(role, database, sorted(config)) for role, database, config in settings] == [
        (roles.ai_worker, 0, sorted(f"{k}={v}" for k, v in db_roles.AI_WORKER_SETTINGS.items()))]
    attributes = {row[0]: row[1:] for row in su.execute(
        "SELECT rolname, rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls,"
        " rolcanlogin, rolinherit FROM pg_roles WHERE rolname = ANY(%s)",
        ([roles.owner, *roles.logins],))}
    assert attributes == {roles.owner: (False, False, False), roles.migrator: (False, True, False),
                          roles.app: (False, True, True), roles.research_worker: (False, True, True),
                          roles.evaluator: (False, True, True), roles.ai_worker: (False, True, True),
                          roles.backup: (False, True, True)}
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


# Tables with users' identities, money, orders and credentials.
TRADING_AND_IDENTITY = {"users", "paper_accounts", "positions", "trades", "api_keys", "key_pairs",
                        "refresh_tokens", "webhooks", "idempotency_records", "order_security_records"}


def test_the_worker_allow_list_is_a_subset_that_leaves_out_trading_and_identity():
    for table, privileges in db_roles.WORKER_TABLE_PRIVILEGES.items():
        assert set(privileges) <= set(db_roles.APP_TABLE_PRIVILEGES[table]), table
    assert not set(db_roles.WORKER_TABLE_PRIVILEGES) & TRADING_AND_IDENTITY


def test_the_research_worker_runs_every_kind_of_job_as_its_own_role(provisioned, monkeypatch):
    """The API, as qs_app, queues jobs; a worker connected as
    qs_research_worker runs them: an experiment's run, validation and replay
    (their finalizers lock and update the experiment), the dashboard backtest
    (its history row and trial ledger) and a stat-test on that backtest, each
    finished with a signed audit event, plus the worker's own housekeeping."""
    returns = [0.001 * ((i % 7) - 3) for i in range(60)]
    monkeypatch.setattr(backtest_service, "run_moving_average_backtest", lambda *a: {
        "initial_capital": 100_000.0, "final_capital": 101_500.0, "total_return": 0.015,
        "sharpe_ratio": 1.25, "max_drawdown": -0.04, "win_rate": 0.5, "total_trades": 6,
        "daily_returns_net": returns})
    roles = provisioned.roles
    api = sessionmaker(bind=provisioned.engine_as(roles.app))()
    worker_sessions = sessionmaker(bind=provisioned.engine_as(roles.research_worker))
    # The worker registers the signing key itself (worker.py at startup). Reset
    # the process-wide flag so that happens here whichever tests ran before.
    identity = security_service.server_identity
    monkeypatch.setattr(identity, "_registered", False)
    with worker_sessions() as db:
        assert db.execute(text("SELECT current_user")).scalar() == roles.research_worker
        # Its startup check (worker._wait_for_schema) reads the schema revision.
        assert current_revision(db.connection()) == head_revision()
        identity.ensure_registered(db)
        assert db.execute(text("SELECT key_id FROM server_signing_keys WHERE fingerprint = :f"),
                          {"f": identity.fingerprint}).scalar() == identity.key_id
    user = models.User(email="worker-role@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=worker_sessions,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)

    def run(response):
        job_id = json.loads(response.body)["job_id"]
        assert runner.run_once() is True
        api.expire_all()
        view = research_jobs.view(api, api.get(models.ResearchJob, job_id))
        assert view["status"] == "succeeded", view
        return view["result"]

    bars = [{"timestamp": 1_700_000_000 + i * 86400, "open": 100 + 0.5 * i, "high": 102 + 0.5 * i,
             "low": 99 + 0.5 * i, "close": 100.5 + 0.5 * i, "volume": 200_000} for i in range(6)]
    experiment = main.experiment_create({
        "strategy_id": "obi_momentum", "strategy_version": "1.0", "dataset_id": "roles", "dataset": bars,
        "parameters": {"obi_threshold": 0.2, "events_per_bar": 120, "warmup_events": 30,
                       "latency_preset": "cloud"}, "random_seed": 11}, user, api)["experiment_id"]
    run(main.experiment_run(experiment, user, api))
    run(main.experiment_validate(experiment, user, api))
    assert run(main.experiment_replay(experiment, {}, user, api))["result_reproduced"] is True
    backtest = run(main.run_backtest(schemas.BacktestRequest(asset="aapl"), user=user, db=api))
    assert backtest["trials_in_family"] == 1
    run(main.statistical_tests(schemas.StatTestRequest(backtest_id=backtest["id"]), user, api))
    status = security_service.audit_chain_status(api, full=True)
    assert status["valid"] and status["links"] >= 5, status
    with worker_sessions() as db:
        assert research_jobs.purge_finished(db, retention_days=0) == 5
    api.close()


def test_an_embedded_worker_records_trials_as_the_app_role_which_can_never_remove_one(provisioned, monkeypatch):
    """RESEARCH_WORKER_MODE=embedded, the default, runs the worker with the
    API's credentials, so qs_app records trials (research_trials.record reads
    which configs are known, then inserts the rest). It can never change or
    remove one, so a user's trial count only grows."""
    monkeypatch.setattr(backtest_service, "run_moving_average_backtest", lambda *a: {
        "initial_capital": 100_000.0, "final_capital": 101_000.0, "total_return": 0.01,
        "sharpe_ratio": 1.0, "max_drawdown": -0.02, "win_rate": 0.5, "total_trades": 4,
        "daily_returns_net": [0.001 * ((i % 5) - 2) for i in range(30)]})
    monkeypatch.setattr(security_service.server_identity, "_registered", False)
    roles = provisioned.roles
    sessions = sessionmaker(bind=provisioned.engine_as(roles.app))
    api = sessions()
    user = models.User(email="embedded@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=sessions,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    counts = []
    for request in (schemas.BacktestRequest(asset="aapl"), schemas.BacktestRequest(asset="AAPL"),
                    schemas.BacktestRequest(asset="aapl", fast_window=5)):
        job_id = json.loads(main.run_backtest(request, user=user, db=api).body)["job_id"]
        assert runner.run_once() is True
        api.expire_all()
        view = research_jobs.view(api, api.get(models.ResearchJob, job_id))
        assert view["status"] == "succeeded", view
        counts.append(view["result"]["trials_in_family"])
    # The same config again is still one trial; a new one is the second.
    assert counts == [1, 1, 2]
    api.close()
    app = provisioned.connect(roles.app)
    for statement in ("UPDATE research_trials SET config_hash = family_hash",
                      "UPDATE research_trials SET user_id = user_id", "DELETE FROM research_trials",
                      "TRUNCATE research_trials"):
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute(statement)
    assert provisioned.connect().execute("SELECT count(*) FROM research_trials").fetchone()[0] == 2


def test_the_research_worker_cannot_reach_trading_or_identity(provisioned):
    roles = provisioned.roles
    research_worker = provisioned.connect(roles.research_worker)
    for table in sorted(set(db_roles.APP_TABLE_PRIVILEGES) - set(db_roles.WORKER_TABLE_PRIVILEGES)):
        with pytest.raises(errors.InsufficientPrivilege):
            research_worker.execute(f'SELECT 1 FROM "{table}" LIMIT 1')
    for statement in ("UPDATE audit_logs SET action = 'x'", "DELETE FROM audit_chain_links",
                      "INSERT INTO research_jobs (id) VALUES ('x')", "UPDATE backtests SET result_json = NULL",
                      "DELETE FROM research_trials", "DELETE FROM server_signing_keys",
                      *(s.format(app=roles.research_worker, owner=roles.owner) for s in REFUSED)):
        with pytest.raises(errors.InsufficientPrivilege):
            research_worker.execute(statement)


HOLDOUT_TABLES = ("holdout_attempts", "holdout_evaluations")


def test_the_evaluator_allow_list_is_a_subset_that_neither_queues_nor_deletes_jobs():
    for table, privileges in db_roles.EVALUATOR_TABLE_PRIVILEGES.items():
        if table not in HOLDOUT_TABLES:
            assert set(privileges) <= set(db_roles.APP_TABLE_PRIVILEGES[table]), table
    assert not set(db_roles.EVALUATOR_TABLE_PRIVILEGES) & TRADING_AND_IDENTITY
    assert set(db_roles.EVALUATOR_TABLE_PRIVILEGES["research_jobs"]) == {"SELECT", "UPDATE"}



def test_only_the_evaluator_writes_the_holdout_tables_and_the_research_worker_never_sees_a_grade():
    for table in HOLDOUT_TABLES:
        assert db_roles.EVALUATOR_TABLE_PRIVILEGES[table] == ("SELECT", "INSERT"), table
        assert db_roles.APP_TABLE_PRIVILEGES[table] == ("SELECT",), table
    # The worker counts attempts as trials, and reads no grade.
    assert db_roles.WORKER_TABLE_PRIVILEGES["holdout_attempts"] == ("SELECT",)
    assert "holdout_evaluations" not in db_roles.WORKER_TABLE_PRIVILEGES

def test_the_evaluator_runs_its_jobs_as_its_own_role(provisioned, monkeypatch):
    """The research evaluator's worker loop, connected as qs_evaluator: its
    startup (schema revision, signing key), heartbeat, the claim of an
    evaluator job, the job's signed audit event and its removal on stop. It
    leaves the ordinary job queued, and each job counts only the workers
    that can run it."""
    monkeypatch.setitem(research_jobs.KINDS, "test_holdout_grade", research_jobs.JobKind(
        dict, "research_job_helpers:echo_task", "TEST_HOLDOUT_GRADE", evaluator=True))
    monkeypatch.setitem(research_jobs.KINDS, "test_echo", research_jobs.JobKind(
        dict, "research_job_helpers:echo_task", "TEST_JOB"))
    roles = provisioned.roles
    api = sessionmaker(bind=provisioned.engine_as(roles.app))()
    evaluator_sessions = sessionmaker(bind=provisioned.engine_as(roles.evaluator))
    identity = security_service.server_identity
    monkeypatch.setattr(identity, "_registered", False)
    with evaluator_sessions() as db:
        assert db.execute(text("SELECT current_user")).scalar() == roles.evaluator
        assert current_revision(db.connection()) == head_revision()
        identity.ensure_registered(db)
    user = models.User(email="evaluator-role@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    ordinary = research_jobs.enqueue(api, user.id, "test_echo", {"n": 1})
    graded = research_jobs.enqueue(api, user.id, "test_holdout_grade", {"n": 2})
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=evaluator_sessions,
                           lease_seconds=60, timeout_seconds=60, poll_seconds=0.1,
                           role="research-evaluator")
    assert runner.run_once() is True
    assert runner.run_once() is False
    api.expire_all()
    assert research_jobs.view(api, api.get(models.ResearchJob, graded.id))["status"] == "succeeded"
    left = research_jobs.view(api, api.get(models.ResearchJob, ordinary.id))
    assert (left["status"], left["attempts"], left["workers_online"]) == ("queued", 0, 0)
    stats = research_jobs.queue_stats(api)
    assert (stats["workers_online"], stats["evaluators_online"]) == (0, 1)
    assert api.execute(text("SELECT role FROM research_workers")).scalars().all() == ["research-evaluator"]
    status = security_service.audit_chain_status(api, full=True)
    assert status["valid"] and status["links"] == 1, status
    with evaluator_sessions() as db:
        research_jobs.remove_worker(db, runner.id)
    assert api.execute(text("SELECT count(*) FROM research_workers")).scalar() == 0
    api.close()


def test_the_evaluator_cannot_reach_beyond_its_jobs(provisioned):
    """Nothing of users, trading, keys or tokens, no research results, and it
    can neither queue nor delete a job, nor rewrite the audit trail."""
    roles = provisioned.roles
    evaluator = provisioned.connect(roles.evaluator)
    for table in sorted(set(db_roles.APP_TABLE_PRIVILEGES) - set(db_roles.EVALUATOR_TABLE_PRIVILEGES)):
        with pytest.raises(errors.InsufficientPrivilege):
            evaluator.execute(f'SELECT 1 FROM "{table}" LIMIT 1')
    for statement in ("INSERT INTO research_jobs (id) VALUES ('x')", "DELETE FROM research_jobs",
                      "UPDATE audit_logs SET action = 'x'", "DELETE FROM audit_chain_links",
                      "DELETE FROM server_signing_keys", "INSERT INTO dataset_holdouts (family) VALUES ('x')",
                      "INSERT INTO research_trials (id) VALUES ('x')",
                      "INSERT INTO strategy_validations (job_id) VALUES ('x')",
                      "INSERT INTO strategy_versions (version_id) VALUES ('x')",
                      "UPDATE holdout_attempts SET holdout_start = holdout_start", "DELETE FROM holdout_attempts",
                      "UPDATE holdout_evaluations SET passed = passed", "DELETE FROM holdout_evaluations",
                      *(s.format(app=roles.evaluator, owner=roles.owner) for s in REFUSED)):
        with pytest.raises(errors.InsufficientPrivilege):
            evaluator.execute(statement)


def test_the_ai_allow_lists_keep_the_ai_worker_to_its_own_jobs():
    assert set(db_roles.AI_WORKER_TABLE_PRIVILEGES["flags"]) == {"SELECT"}
    assert "INSERT" not in db_roles.AI_WORKER_TABLE_PRIVILEGES["jobs"]
    assert set(db_roles.APP_AI_TABLE_PRIVILEGES["jobs"]) == {"SELECT", "INSERT"}
    assert set(db_roles.APP_AI_TABLE_PRIVILEGES["workers"]) == {"SELECT"}
    for allowed in (db_roles.AI_WORKER_TABLE_PRIVILEGES, db_roles.APP_AI_TABLE_PRIVILEGES):
        assert all(p and set(p) <= set(db_roles.TABLE_PRIVILEGES) for p in allowed.values())


def test_the_ai_worker_reaches_nothing_of_the_trusted_tier(provisioned):
    """No USAGE on the application schema: not its job queue, worker
    registry, audit trail, keys or users, not even by qualified name, and
    none of its routines. In its own schema it can neither queue a job nor
    set a flag, and its role carries its timeouts."""
    roles = provisioned.roles
    ai_worker = provisioned.connect(roles.ai_worker)
    main_schema = provisioned.schema
    for table in sorted(db_roles.APP_TABLE_PRIVILEGES):
        with pytest.raises(errors.InsufficientPrivilege):
            ai_worker.execute(f'SELECT 1 FROM "{main_schema}"."{table}" LIMIT 1')
    for statement in (f"INSERT INTO \"{main_schema}\".research_jobs (id) VALUES ('x')",
                      f'UPDATE "{main_schema}".research_workers SET pid = 0',
                      f'DELETE FROM "{main_schema}".audit_logs',
                      f'CREATE TABLE "{main_schema}".made_by_ai (x int)'):
        with pytest.raises(errors.InsufficientPrivilege):
            ai_worker.execute(statement)
    ai = db_roles.ai_schema_for(main_schema)
    ai_worker.execute(f'SET search_path = "{ai}"')
    assert ai_worker.execute("SELECT count(*) FROM jobs").fetchone() == (0,)
    for statement in ("INSERT INTO jobs (id, kind) VALUES ('x', 'ai_selftest')",
                      "INSERT INTO flags (name, enabled, set_by, reason) VALUES ('selftest', true, 'a', 'b')",
                      "UPDATE flags SET enabled = true", "DELETE FROM flags", "TRUNCATE jobs",
                      f'CREATE TABLE "{ai}".made_by_ai (x int)'):
        with pytest.raises(errors.InsufficientPrivilege):
            ai_worker.execute(statement)
    settings = dict(ai_worker.execute(
        "SELECT name, setting || coalesce(unit, '') FROM pg_settings WHERE name = ANY(%s)",
        (list(db_roles.AI_WORKER_SETTINGS),)).fetchall())
    assert settings == {"statement_timeout": "30000ms", "lock_timeout": "10000ms",
                        "idle_in_transaction_session_timeout": "60000ms"}


def test_only_the_app_and_the_ai_worker_reach_the_ai_schema(provisioned):
    roles = provisioned.roles
    ai = db_roles.ai_schema_for(provisioned.schema)
    for role in (roles.research_worker, roles.evaluator, roles.migrator):
        conn = provisioned.connect(role)
        for table in sorted(db_roles.AI_WORKER_TABLE_PRIVILEGES):
            with pytest.raises(errors.InsufficientPrivilege):
                conn.execute(f'SELECT 1 FROM "{ai}"."{table}" LIMIT 1')
    app = provisioned.connect(roles.app)
    for statement in (f"UPDATE \"{ai}\".jobs SET status = 'failed'", f'DELETE FROM "{ai}".jobs',
                      f'DELETE FROM "{ai}".flags', f"INSERT INTO \"{ai}\".workers (id) VALUES ('x')"):
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute(statement)


def test_the_ai_worker_runs_its_jobs_as_its_own_role(provisioned, capsys):
    """An operator, as the app role, switches the selftest stage on and
    queues a selftest job; the AI worker, connected as qs_ai_worker, claims
    and runs it in a job process of its own, which imports nothing of the
    trusted tier. The operator's actions are in the audit trail."""
    from backend.ai import settings as ai_settings
    from backend.ai.worker import AiWorker

    roles = provisioned.roles
    app_sessions = sessionmaker(bind=provisioned.engine_as(roles.app))
    assert manage._ai_selftest(app_sessions, "ops@example.com", 0) == 1
    assert "off" in capsys.readouterr().err
    for flag in ("research_intelligence", "selftest"):
        assert manage._set_flag(app_sessions, flag, True, "ops@example.com", "role test") == 0
    capsys.readouterr()
    assert manage._ai_selftest(app_sessions, "ops@example.com", 0) == 0
    job_id = capsys.readouterr().out.strip().rsplit(": ", 1)[1]
    ai = db_roles.ai_schema_for(provisioned.schema)
    settings = ai_settings.load({
        "QS_PROCESS_ROLE": "ai-worker",
        "AI_DATABASE_URL": provisioned.url(roles.ai_worker).render_as_string(hide_password=False),
        "AI_DB_SCHEMA": ai, "AI_WORKER_HEARTBEAT_SECONDS": "0.5"})
    runner = AiWorker(settings)
    runner.connect()
    try:
        assert runner.conn.execute("SELECT current_user").fetchone() == (roles.ai_worker,)
        assert runner.schema_ready()
        assert runner.run_once() == "succeeded"
        assert runner.run_once() is None
    finally:
        runner.close()
    assert manage._print_ai_job(app_sessions, job_id) == 0
    job = json.loads(capsys.readouterr().out)
    assert job["status"] == "succeeded", job
    assert job["result"]["trusted_modules"] == []
    assert job["result"]["process_role"] == "ai-job"
    with app_sessions() as db:
        actions = db.execute(text("SELECT action FROM audit_logs ORDER BY created_at, id")).scalars().all()
        assert actions == ["AI_FLAG_SET", "AI_FLAG_SET", "AI_JOB_QUEUED"]
        assert db.execute(text(f'SELECT count(*) FROM "{ai}".workers')).scalar() == 1


def test_the_app_and_the_worker_may_register_datasets_but_never_change_them(provisioned, tmp_path):
    from backend.services import dataset_registry
    from backend.services.dataset_store import DatasetStore
    from dataset_helpers import NOW, snapshot

    roles = provisioned.roles
    provisioned.connect().execute("INSERT INTO users (id, email, password_hash) VALUES "
                                  "('u1', 'u1@example.com', 'h'), ('u2', 'u2@example.com', 'h')")
    store = DatasetStore(tmp_path)
    blob = store.put(snapshot(("AAPL",)).datasets[0])
    dataset = store.get(blob.dataset_id, blob.sha256)
    for role, user_id in ((roles.app, "u1"), (roles.research_worker, "u2")):
        with sessionmaker(bind=provisioned.engine_as(role))() as db:
            dataset_registry.register(db, dataset, blob, user_id=user_id, request={}, provider_client="t",
                                      retrieved_at=NOW)
            db.commit()
            assert [v.dataset_id for v in dataset_registry.list_for_user(db, user_id)] == [blob.dataset_id]
        conn = provisioned.connect(role)
        for statement in ("UPDATE dataset_versions SET blob_size = 1", "DELETE FROM dataset_versions",
                          "TRUNCATE dataset_versions CASCADE", "UPDATE user_datasets SET job_id = 'x'",
                          "DELETE FROM user_datasets"):
            with pytest.raises(errors.InsufficientPrivilege):
                conn.execute(statement)
    assert provisioned.connect().execute("SELECT count(*) FROM user_datasets").fetchone() == (2,)


def test_the_app_may_register_and_tag_strategy_versions_but_never_change_them(provisioned):
    """The research worker may only read versions and who holds them
    (test_the_research_worker_runs_a_strategy_backtest_as_its_own_role); tags
    are out of its reach (test_the_research_worker_cannot_reach_trading_or_identity)."""
    from backend.services import strategy_registry
    from test_strategy_spec import spec

    roles = provisioned.roles
    provisioned.connect().execute("INSERT INTO users (id, email, password_hash) VALUES ('u1', 'u1@example.com', 'h')")
    with sessionmaker(bind=provisioned.engine_as(roles.app))() as db:
        version = strategy_registry.register(db, spec(), user_id="u1")
        strategy_registry.tag(db, "u1", version.version_id, "1.0.0")
        db.commit()
        assert strategy_registry.resolve(db, "u1", "1.0.0").version_id == version.version_id
    app = provisioned.connect(roles.app)
    for statement in ("UPDATE strategy_versions SET family = family", "DELETE FROM strategy_versions",
                      "TRUNCATE strategy_versions CASCADE", "UPDATE user_strategy_versions SET created_at = now()",
                      "DELETE FROM user_strategy_versions", "UPDATE user_strategy_tags SET label = 'moved'",
                      "DELETE FROM user_strategy_tags", "TRUNCATE user_strategy_tags"):
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute(statement)
    owner = provisioned.connect()
    assert [owner.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("strategy_versions", "user_strategy_versions", "user_strategy_tags")] == [1, 1, 1]


def test_the_research_worker_runs_a_strategy_backtest_as_its_own_role(provisioned, monkeypatch, tmp_path):
    """The API registers the version and queues the job as qs_app; the worker
    loads the spec, records the trial and its evaluation and signs the audit
    event as qs_research_worker, which can read versions but never write one,
    and which, like qs_app, can never change or remove an evaluation."""
    import numpy as np
    import pandas as pd

    from market_helpers import fake_market
    from test_trial_evaluations import strategy

    days = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(5)
    fake_market(monkeypatch, tmp_path, pd.DataFrame(
        {("Close", t): 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))) for t in ("AAA", "BBB", "CCC")},
        index=days))
    monkeypatch.setattr(security_service.server_identity, "_registered", False)
    roles = provisioned.roles
    api = sessionmaker(bind=provisioned.engine_as(roles.app))()
    worker_sessions = sessionmaker(bind=provisioned.engine_as(roles.research_worker))
    with worker_sessions() as db:
        security_service.server_identity.ensure_registered(db)
    user = models.User(email="strategy-role@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    version_id = main.register_strategy(schemas.StrategyRegisterRequest(spec=strategy(), tag="v1"),
                                        user=user, db=api)["version_id"]
    response = main.strategy_backtest(schemas.StrategyBacktestRequest(strategy="v1", period="1y"), user=user, db=api)
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=worker_sessions,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    assert runner.run_once() is True
    api.expire_all()
    view = research_jobs.view(api, api.get(models.ResearchJob, json.loads(response.body)["job_id"]))
    assert view["status"] == "succeeded", view
    assert view["result"]["strategy_version_id"] == version_id
    # The worker validates the evaluation as its own role too, and records the
    # matrix and the verdict.
    capital = api.get(models.ResearchJob, view["job_id"]).params_json["initial_capital"]
    # A server key not yet registered: the worker registers it, as its own
    # role, in the transaction that records the verdict and its manifest.
    for name, value in (("dsa_pk", None), ("dsa_sk", None), ("fingerprint", None), ("key_id", None),
                        ("_registered", False)):
        monkeypatch.setattr(security_service.server_identity, name, value)
    validation = main.validate_strategy(schemas.StrategyValidationRequest(
        evaluation=view["job_id"], period="1y", initial_capital=capital), user=user, db=api)
    assert runner.run_once() is True
    api.expire_all()
    validated = research_jobs.view(api, api.get(models.ResearchJob, json.loads(validation.body)["job_id"]))
    assert validated["status"] == "succeeded", validated
    assert validated["result"]["validation"]["passed"] is False
    manifest = main.get_strategy_validation_manifest(validated["job_id"], user=user, db=api)
    assert manifest["signature_valid"] is True and manifest["matches_verdict"] is True
    assert manifest["signature"].startswith("ml-dsa:")
    assert security_service.audit_chain_status(api, full=True)["valid"]
    api.close()
    research_worker = provisioned.connect(roles.research_worker)
    assert research_worker.execute("SELECT count(*) FROM strategy_versions").fetchone() == (1,)
    for statement in ("INSERT INTO strategy_versions (version_id) VALUES ('x')",
                      "INSERT INTO user_strategy_versions (user_id, version_id) VALUES ('u', 'x')",
                      "SELECT 1 FROM user_strategy_tags"):
        with pytest.raises(errors.InsufficientPrivilege):
            research_worker.execute(statement)
    for role in (roles.app, roles.research_worker):
        conn = provisioned.connect(role)
        for statement in ("UPDATE research_trial_evaluations SET sharpe = 9",
                          "DELETE FROM research_trial_evaluations", "TRUNCATE research_trial_evaluations",
                          "UPDATE strategy_validations SET passed = true", "DELETE FROM strategy_validations",
                          "TRUNCATE strategy_validations", "DELETE FROM trial_matrices", "TRUNCATE trial_matrices",
                          "UPDATE validation_manifests SET signature = signature",
                          "DELETE FROM validation_manifests", "TRUNCATE validation_manifests"):
            with pytest.raises(errors.InsufficientPrivilege):
                conn.execute(statement)
    owner = provisioned.connect()
    assert owner.execute("SELECT job_id, strategy_version_id FROM research_trial_evaluations").fetchall() == \
        [(view["job_id"], version_id)]
    assert owner.execute("SELECT job_id, evaluation_job_id, passed FROM strategy_validations").fetchall() == \
        [(validated["job_id"], view["job_id"], False)]
    assert owner.execute("SELECT count(*) FROM trial_matrices").fetchone() == (1,)
    assert owner.execute("SELECT v.job_id, k.fingerprint FROM validation_manifests v"
                         " JOIN server_signing_keys k ON k.key_id = v.signing_key_id").fetchall() == \
        [(validated["job_id"], security_service.server_identity.fingerprint)]


def test_the_evaluator_grades_a_strategy_on_its_holdout_as_its_own_role(provisioned, monkeypatch, tmp_path):
    """The whole holdout evaluation under the production roles: the API sets
    the holdout and queues the jobs (qs_app), the research worker backtests
    and validates (qs_research_worker), the evaluator spends the attempt,
    reads the holdout, grades and signs it (qs_evaluator). The research
    worker counts the attempt but never reads a grade, and what it can read
    of the job and its audit event carries no figure of the holdout."""
    import datetime as dt

    import numpy as np
    import pandas as pd

    from backend.services import dataset_holdouts, holdout_evaluation, trial_matrix, validation_policy
    from market_helpers import fake_market
    from test_trial_evaluations import strategy

    days = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(5)
    fake_market(monkeypatch, tmp_path, pd.DataFrame(
        {("Close", t): 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))) for t in ("AAA", "BBB", "CCC")},
        index=days))
    real = validation_policy.evaluate  # the synthetic market cannot pass the real gates
    monkeypatch.setattr(validation_policy, "evaluate", lambda *a, **k: {**real(*a, **k), "passed": True})
    monkeypatch.setattr(security_service.server_identity, "_registered", False)
    roles = provisioned.roles
    api = sessionmaker(bind=provisioned.engine_as(roles.app))()
    worker_sessions = sessionmaker(bind=provisioned.engine_as(roles.research_worker))
    evaluator_sessions = sessionmaker(bind=provisioned.engine_as(roles.evaluator))
    with worker_sessions() as db:
        security_service.server_identity.ensure_registered(db)
    dataset_holdouts.set_start(api, "EQUITY_EOD", dt.date(2023, 11, 1), set_by="ops", reason="test")
    user = models.User(email="holdout-role@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    main.register_strategy(schemas.StrategyRegisterRequest(spec=strategy(), tag="v1"), user=user, db=api)
    research = worker.Worker(job_process=InlineJobProcess(), session_factory=worker_sessions,
                             lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    backtest = main.strategy_backtest(schemas.StrategyBacktestRequest(strategy="v1", period="1y"), user=user, db=api)
    assert research.run_once() is True
    evaluation = json.loads(backtest.body)["job_id"]
    capital = api.get(models.ResearchJob, evaluation).params_json["initial_capital"]
    validation = main.validate_strategy(schemas.StrategyValidationRequest(
        evaluation=evaluation, period="1y", initial_capital=capital), user=user, db=api)
    assert research.run_once() is True
    validation_id = json.loads(validation.body)["job_id"]
    api.expire_all()
    assert api.get(models.ResearchJob, validation_id).status == "succeeded"

    job = holdout_evaluation.queue(api, validation_id, "1y", capital)
    # The research worker leaves it; the evaluator, as qs_evaluator, runs it.
    assert research.run_once() is False
    graded = worker.Worker(job_process=InlineJobProcess(), session_factory=evaluator_sessions,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1,
                           role=research_jobs.EVALUATOR_ROLE)
    assert graded.run_once() is True
    api.expire_all()
    view = research_jobs.view(api, api.get(models.ResearchJob, job.id))
    assert view["status"] == "succeeded", view
    [row] = api.query(models.HoldoutEvaluation).all()
    assert holdout_evaluation.signature_valid(api, row) and row.grade["sessions"] >= 63
    assert view["result"]["passed"] is row.passed
    attempt = api.get(models.HoldoutAttempt, job.id)
    assert trial_matrix.family_state(api, user.id, attempt.family_hash)["holdout_queries"] == 1
    assert security_service.audit_chain_status(api, full=True)["valid"]
    api.close()

    research_worker = provisioned.connect(roles.research_worker)
    assert research_worker.execute("SELECT count(*) FROM holdout_attempts").fetchone() == (1,)
    with pytest.raises(errors.InsufficientPrivilege):
        research_worker.execute("SELECT 1 FROM holdout_evaluations")
    research_worker = provisioned.connect(roles.research_worker)
    [(result, error)] = research_worker.execute(
        "SELECT result_json::text, error_detail FROM research_jobs WHERE id = %s", (job.id,)).fetchall()
    audits = [r[0] for r in research_worker.execute(
        "SELECT row_to_json(a)::text FROM audit_logs a WHERE action = 'HOLDOUT_EVALUATION'").fetchall()]
    assert len(audits) == 1 and error is None
    for figure in ("sharpe", "sessions", "returns", "first_session"):
        assert figure not in result and figure not in audits[0]
    evaluator = provisioned.connect(roles.evaluator)
    for statement in ("UPDATE holdout_evaluations SET passed = NOT passed", "DELETE FROM holdout_attempts",
                      "TRUNCATE holdout_evaluations"):
        with pytest.raises(errors.InsufficientPrivilege):
            evaluator.execute(statement)

def test_the_research_worker_runs_a_dataset_snapshot_as_its_own_role(provisioned, monkeypatch, tmp_path):
    """The snapshot finalizer reads the blobs back and registers them as
    qs_research_worker, with the user's row written by the API as qs_app."""
    from backend import config
    from backend.services import dataset_providers
    from dataset_helpers import FakeYahoo

    monkeypatch.setattr(dataset_providers, "_yahoo_fetch", FakeYahoo())
    monkeypatch.setattr(dataset_providers, "_yahoo_client", lambda: "yfinance test")
    monkeypatch.setattr(config, "DATASET_ROOT", tmp_path)
    monkeypatch.setattr(security_service.server_identity, "_registered", False)
    roles = provisioned.roles
    api = sessionmaker(bind=provisioned.engine_as(roles.app))()
    worker_sessions = sessionmaker(bind=provisioned.engine_as(roles.research_worker))
    with worker_sessions() as db:
        security_service.server_identity.ensure_registered(db)
    user = models.User(email="snapshot-role@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    response = main.snapshot_dataset(schemas.DatasetSnapshotRequest(
        symbols=["AAPL", "BTC-USD"], start="2024-01-01", end="2024-01-11"), user=user, db=api)
    job_id = json.loads(response.body)["job_id"]
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=worker_sessions,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    assert runner.run_once() is True
    api.expire_all()
    view = research_jobs.view(api, api.get(models.ResearchJob, job_id))
    assert view["status"] == "succeeded", view
    assert len(main.list_datasets(limit=50, user=user, db=api)) == 2
    assert security_service.audit_chain_status(api, full=True)["valid"]
    api.close()


def test_a_research_job_registers_and_reuses_its_market_data_as_the_least_privileged_roles(
        provisioned, monkeypatch, tmp_path):
    """The worker registers a task's snapshot as qs_research_worker; the API
    checks, as qs_app, that a rerun names only datasets its user holds."""
    import numpy as np
    import pandas as pd

    from market_helpers import fake_market

    days = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(3)
    fake_market(monkeypatch, tmp_path, pd.DataFrame(
        {("Close", t): 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))) for t in "ABCD"}, index=days))
    monkeypatch.setattr(security_service.server_identity, "_registered", False)
    roles = provisioned.roles
    api = sessionmaker(bind=provisioned.engine_as(roles.app))()
    worker_sessions = sessionmaker(bind=provisioned.engine_as(roles.research_worker))
    with worker_sessions() as db:
        security_service.server_identity.ensure_registered(db)
    user = models.User(email="market-role@example.com", password_hash="x", role="user")
    api.add(user)
    api.commit()
    runner = worker.Worker(job_process=InlineJobProcess(), session_factory=worker_sessions,
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    results = []
    for rerun in (False, True):
        dataset_ids = results[0]["market_data"]["datasets"] if rerun else None
        response = main.correlation_endpoint(schemas.CorrelationRequest(
            assets=list("ABCD"), period="1y", dataset_ids=dataset_ids), user=user, db=api)
        assert runner.run_once() is True
        api.expire_all()
        view = research_jobs.view(api, api.get(models.ResearchJob, json.loads(response.body)["job_id"]))
        assert view["status"] == "succeeded", view
        results.append(view["result"])
    assert results[1]["market_data"] == {**results[0]["market_data"], "source": "datasets"}
    assert len(main.list_datasets(limit=50, user=user, db=api)) == 1
    assert security_service.audit_chain_status(api, full=True)["valid"]
    api.close()


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


@pytest.mark.parametrize("missing, revision", [("research_worker", "0002"), ("evaluator", "0009"),
                                               ("ai_worker", "0011")])
def test_a_schema_provisioned_before_a_role_existed_still_migrates(provisioned, missing, revision):
    """A deployment provisioned before qs_research_worker (or qs_evaluator) was
    added has no such role. The next release's migration still reapplies the
    grants of the roles it has, and running provision-roles again creates the
    missing one."""
    roles = provisioned.roles
    role = getattr(roles, missing)
    provisioned.migrate_as_owner(revision)
    su = provisioned.connect()
    su.execute(f'DROP OWNED BY "{role}"')
    su.execute(f'DROP ROLE "{role}"')
    su.execute(f'GRANT TRUNCATE ON trades TO "{roles.app}"')
    assert migrate(provisioned.engine_as(roles.migrator), roles) == head_revision()
    assert table_privileges(su, provisioned.schema, roles.app) == {
        name: set(p) for name, p in db_roles.APP_TABLE_PRIVILEGES.items()}
    assert provisioned.provision()["created"] == [role]
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
            "ALTER TABLE users OWNER TO CURRENT_USER",
            f'ALTER ROLE "{roles.ai_worker}" CONNECTION LIMIT -1',
            f'ALTER ROLE "{roles.ai_worker}" RESET statement_timeout',
            f'ALTER ROLE "{roles.app}" SET statement_timeout = 0',
            f'GRANT USAGE ON SCHEMA "{provisioned.schema}" TO "{roles.ai_worker}"',
            f'GRANT SELECT ON "{provisioned.schema}".audit_logs TO "{roles.ai_worker}"',
            f'GRANT INSERT ON "{provisioned.schema}_ai".jobs TO "{roles.ai_worker}"',
            f'GRANT USAGE ON SCHEMA "{provisioned.schema}_ai" TO "{roles.research_worker}"',
            f'ALTER TABLE "{provisioned.schema}_ai".jobs OWNER TO CURRENT_USER'):
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
    with pytest.raises(RuntimeError, match=r'\.trades"'):
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
                cluster.roles.research_worker: "DB_RESEARCH_WORKER_PASSWORD",
                cluster.roles.evaluator: "DB_EVALUATOR_PASSWORD", cluster.roles.ai_worker: "DB_AI_WORKER_PASSWORD",
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
