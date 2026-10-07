"""PostgreSQL layer: migrations, session settings, and database-level races.

Each test below pins a defect that was reproduced against a live PostgreSQL
16 server before it was fixed, or a guarantee the migrations must keep.
"""
import datetime as dt
import hashlib
import sys
import threading
import time

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.orm import sessionmaker

from backend import config, database, models
from backend.crypto import pqc
from backend.database import Base
from backend.services import security_service


def schema_diff(engine):
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True,
                                                         "compare_server_default": True})
        return compare_metadata(context, Base.metadata)


# ── migrations ────────────────────────────────────────────────────────────────

class TestMigrations:
    def test_migrations_build_exactly_the_models_schema(self, make_engine):
        engine = make_engine(migrated=True)
        assert schema_diff(engine) == []
        with engine.connect() as conn:
            assert database.current_revision(conn) == database.head_revision()

    def test_migrating_again_changes_nothing(self, make_engine):
        engine = make_engine(migrated=True)
        assert database.migrate(engine) == database.head_revision()
        assert schema_diff(engine) == []

    def test_a_database_from_an_earlier_release_is_adopted_without_data_loss(self, make_engine):
        """Earlier releases built tables with create_all and patched columns
        in place on SQLite only, so a PostgreSQL deployment can lack columns,
        tables and indexes the code relies on. Migrating must add them and
        keep every existing row."""
        engine = make_engine()  # create_all, as earlier releases did
        with engine.begin() as conn:
            for table, column in (("users", "role"), ("users", "watchlist"), ("trades", "stop_price"),
                                  ("trades", "time_in_force"), ("audit_logs", "signing_key_id"),
                                  ("api_keys", "hmac_secret_encrypted"),
                                  ("research_experiments", "manifest_json")):
                conn.execute(text(f'ALTER TABLE {table} DROP COLUMN "{column}"'))
            conn.execute(text("DROP TABLE research_jobs"))
            conn.execute(text("DROP INDEX ix_webhooks_user_id"))
            conn.execute(text("INSERT INTO users (id, email, password_hash) VALUES ('u1', 'old@example.com', 'h')"))
            conn.execute(text("INSERT INTO trades (id, user_id, asset, side, quantity, status) "
                              "VALUES ('t1', 'u1', 'AAPL', 'buy', 2.5, 'FILLED')"))
        assert schema_diff(engine) != []

        database.migrate(engine)

        assert schema_diff(engine) == []
        with engine.connect() as conn:
            user = conn.execute(text("SELECT email, role FROM users WHERE id = 'u1'")).one()
            trade = conn.execute(text("SELECT asset, quantity FROM trades WHERE id = 't1'")).one()
        assert tuple(user) == ("old@example.com", "user")
        assert trade.asset == "AAPL" and float(trade.quantity) == 2.5

    def test_concurrent_startups_migrate_once_and_all_succeed(self, make_engine):
        """Every gunicorn worker runs init_db at boot; they must not race."""
        engine = make_engine(migrated=True)
        with engine.begin() as conn:  # an empty schema, as on a first deploy
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(text(f'DROP TABLE "{table.name}" CASCADE'))
            conn.execute(text("DROP TABLE alembic_version"))
        results, errors = [], []
        barrier = threading.Barrier(4)

        def start():
            try:
                barrier.wait()
                results.append(database.migrate(engine))
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=start) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert results == [database.head_revision()] * 4
        assert schema_diff(engine) == []

    def test_every_revision_after_the_baseline_downgrades_and_upgrades_cleanly(self, make_engine):
        from alembic import command

        engine = make_engine(migrated=True)
        with engine.begin() as conn:
            command.downgrade(database._alembic_config(conn), "0001")
            assert database.current_revision(conn) == "0001"
        assert schema_diff(engine) != []
        database.migrate(engine)
        assert schema_diff(engine) == []

    def test_init_db_refuses_a_stale_schema_when_not_migrating(self, monkeypatch, make_engine):
        engine = make_engine()  # tables, but no migration history
        monkeypatch.setattr(database, "engine", engine)
        monkeypatch.setattr(database, "DB_MIGRATE_ON_STARTUP", False)
        with pytest.raises(RuntimeError, match="backend.manage migrate"):
            database.init_db()


# ── SQLite import ─────────────────────────────────────────────────────────────

def _legacy_sqlite_database(path, orphan_trade=False):
    """A database as a SQLite release left it: naive UTC timestamps, an audit
    chain hashed over them, and (optionally) a row SQLite let through
    although its foreign key points nowhere."""
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    created = dt.datetime(2026, 9, 20, 17, 16, 5, 383516)
    payload = {"action": "USER_REGISTERED", "user_id": "u1", "resource_type": "user",
               "resource_id": "u1", "metadata": {"email": "old@example.com"}}
    entry_hash = security_service._chain_hash("a1", created.isoformat(), payload, "0" * 64)
    identity = security_service.server_identity
    identity.sign(b"ensure a key")
    with engine.begin() as conn:
        conn.execute(models.User.__table__.insert(), [{"id": "u1", "email": "old@example.com",
                                                        "password_hash": "h", "created_at": created}])
        conn.execute(models.ServerSigningKey.__table__.insert(), [{
            "key_id": "k1", "algorithm": "ML-DSA-65", "public_key": pqc.b64(identity.dsa_pk),
            "fingerprint": hashlib.sha256(identity.dsa_pk).hexdigest(), "status": "active"}])
        conn.execute(models.AuditLog.__table__.insert(), [{
            "id": "a1", "user_id": "u1", "action": "USER_REGISTERED", "resource_type": "user",
            "resource_id": "u1", "metadata_json": {"email": "old@example.com"}, "created_at": created,
            "signing_key_id": "k1"}])
        conn.execute(models.AuditChainLink.__table__.insert(), [{
            "id": "l1", "sequence": 1, "audit_log_id": "a1", "previous_hash": "0" * 64,
            "entry_hash": entry_hash, "signing_key_id": "k1",
            "checkpoint_signature": pqc.b64(identity.sign(entry_hash.encode()))}])
        conn.execute(models.Trade.__table__.insert(), [{
            "id": "t1", "user_id": "ghost" if orphan_trade else "u1", "asset": "AAPL", "side": "buy",
            "quantity": 1.5, "status": "FILLED", "filled_price": 101.25, "submitted_at": created}])
    engine.dispose()
    return path


class TestSqliteImport:
    def test_every_row_is_copied_and_the_audit_chain_still_verifies(self, tmp_path, make_engine, capsys):
        from backend import manage

        target = make_engine(migrated=True)
        assert manage._import_sqlite(_legacy_sqlite_database(tmp_path / "old.db"), target) == 0
        with target.connect() as conn:
            assert conn.execute(text("SELECT email FROM users")).scalar_one() == "old@example.com"
            trade = conn.execute(text("SELECT quantity, filled_price, submitted_at FROM trades")).one()
            assert (float(trade.quantity), float(trade.filled_price)) == (1.5, 101.25)
            # The naive SQLite timestamp was UTC and still names the same instant.
            assert trade.submitted_at == dt.datetime(2026, 9, 20, 17, 16, 5, 383516, tzinfo=dt.timezone.utc)
        with sessionmaker(bind=target)() as db:
            assert security_service.audit_chain_status(db)["valid"]
        assert "audit chain verified (1 links)" in capsys.readouterr().out

    def test_a_database_that_already_has_rows_is_never_merged_into(self, tmp_path, make_engine, capsys):
        from backend import manage

        target = make_engine(migrated=True)
        source = _legacy_sqlite_database(tmp_path / "old.db")
        assert manage._import_sqlite(source, target) == 0
        assert manage._import_sqlite(source, target) == 1
        assert "already has rows" in capsys.readouterr().err
        with target.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM users")).scalar() == 1

    def test_a_failed_import_leaves_postgresql_untouched(self, tmp_path, make_engine):
        from backend import manage

        target = make_engine(migrated=True)
        with pytest.raises(Exception, match="trades_user_id_fkey"):
            manage._import_sqlite(_legacy_sqlite_database(tmp_path / "old.db", orphan_trade=True), target)
        with target.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM users")).scalar() == 0


# ── connection settings ───────────────────────────────────────────────────────

def test_every_session_runs_in_utc_with_server_side_limits(make_engine):
    with make_engine().connect() as conn:
        settings = dict(conn.execute(text(
            "SELECT name, setting FROM pg_settings WHERE name IN ('TimeZone', 'statement_timeout',"
            " 'lock_timeout', 'idle_in_transaction_session_timeout')")).all())
    assert settings == {
        "TimeZone": "UTC",
        "statement_timeout": str(config.DB_STATEMENT_TIMEOUT_MS),
        "lock_timeout": str(config.DB_LOCK_TIMEOUT_MS),
        "idle_in_transaction_session_timeout": str(config.DB_IDLE_IN_TRANSACTION_TIMEOUT_MS),
    }


def test_the_application_engine_is_configured_for_production_load():
    pool = database.engine.pool
    assert pool.size() == config.DB_POOL_SIZE
    assert pool._max_overflow == config.DB_MAX_OVERFLOW
    assert pool._timeout == config.DB_POOL_TIMEOUT_SECONDS
    assert database.engine.dialect.driver == "psycopg"


@pytest.mark.parametrize("failure, retry_after", [
    (lambda: PoolTimeout("QueuePool limit reached"), "1"),
    (lambda: OperationalError("SELECT 1", {}, Exception("server closed")), "5"),
])
def test_an_exhausted_pool_or_unreachable_database_is_a_retryable_503(failure, retry_after):
    from starlette.testclient import TestClient

    from backend import main

    class FailingSession:
        def execute(self, *_args, **_kwargs):
            raise failure()

        def close(self):
            pass

    main.app.dependency_overrides[database.get_db] = lambda: FailingSession()
    try:
        response = TestClient(main.app).get("/health/ready")
    finally:
        main.app.dependency_overrides.pop(database.get_db, None)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == retry_after


def test_the_readiness_probe_answers_from_the_database():
    from starlette.testclient import TestClient

    from backend import main

    response = TestClient(main.app).get("/health/ready")
    assert response.status_code == 200 and response.json()["database"] == "ok"


@pytest.mark.parametrize("url, expected", [
    ("postgresql://u:p@db:5432/qs", "postgresql+psycopg://u:p@db:5432/qs"),
    ("postgres://u:p@db/qs", "postgresql+psycopg://u:p@db/qs"),
    ("postgresql+psycopg://u:p@db/qs", "postgresql+psycopg://u:p@db/qs"),
])
def test_postgres_urls_are_pinned_to_the_installed_driver(url, expected):
    assert config._postgres_url(url) == expected


@pytest.mark.parametrize("url", ["sqlite:///./quantumsentinel.db", "mysql://u:p@db/qs",
                                 "postgresql+psycopg2://u:p@db/qs"])
def test_anything_but_postgres_psycopg_is_refused(url):
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        config._postgres_url(url)


# ── audit chain timestamps ────────────────────────────────────────────────────

class TestAuditChainTimezones:
    def test_the_chain_verifies_whatever_the_session_timezone(self, make_engine):
        """The chain hashes each event's created_at. It used to hash the value
        as the driver rendered it, i.e. in the session's TimeZone, so the same
        untouched history verified under one server setting and 'failed'
        under another."""
        with sessionmaker(bind=make_engine())() as db:
            for i in range(3):
                security_service.write_audit_log(db, None, "EVENT", metadata={"i": i})
            for zone in ("UTC", "Asia/Kolkata", "America/New_York"):
                db.execute(text(f"SET TIME ZONE '{zone}'"))
                db.expire_all()
                assert security_service.audit_chain_status(db)["valid"], zone

    @pytest.mark.parametrize("legacy", ["sqlite_naive_utc", "server_timezone"])
    def test_links_hashed_by_earlier_releases_still_verify(self, make_engine, legacy):
        with sessionmaker(bind=make_engine())() as db:
            self._verify_legacy_link(db, legacy)

    @staticmethod
    def _verify_legacy_link(db, legacy):
        entry = security_service.write_audit_log(db, None, "EVENT", metadata={"n": 1})
        link = db.execute(select(models.AuditChainLink)).scalar_one()
        utc = entry.created_at.astimezone(dt.timezone.utc)
        if legacy == "sqlite_naive_utc":
            rendered = utc.replace(tzinfo=None).isoformat()
        else:
            server_tz = security_service._server_timezone(db)
            rendered = utc.astimezone(server_tz).isoformat()
        payload = {"action": "EVENT", "user_id": None, "resource_type": None,
                   "resource_id": None, "metadata": {"n": 1}}
        link.entry_hash = security_service._chain_hash(entry.id, rendered, payload, "0" * 64)
        link.checkpoint_signature = pqc.b64(security_service.server_identity.sign(link.entry_hash.encode()))
        db.commit()
        assert security_service.audit_chain_status(db)["valid"]
        # ... but the content is still bound: altering the event breaks it.
        db.execute(text("UPDATE audit_logs SET action = 'FORGED'"))
        db.commit()
        db.expire_all()
        assert security_service.audit_chain_status(db)["reason"] == "event content does not match its link hash"


class TestAuditChainVerificationScales:
    """audit_chain_status timed out at 1M audit rows: its unchained-event
    count was a NOT IN (subquery), which PostgreSQL cannot plan as an anti
    join, and it loaded each link's event with its own query."""

    @staticmethod
    def _chain(engine, events: int, unchained: int = 0):
        with sessionmaker(bind=engine)() as db:
            for i in range(events):
                security_service.write_audit_log(db, None, "EVENT", metadata={"i": i})
            for i in range(unchained):
                db.add(models.AuditLog(action="UNCHAINED", metadata_json={"i": i}))
            db.commit()

    @staticmethod
    def _statements(engine, run):
        statements = []

        def capture(_conn, _cursor, statement, *_args):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            with sessionmaker(bind=engine)() as db:
                result = run(db)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        return result, statements

    def test_unchained_events_are_counted_with_an_anti_join(self, make_engine):
        engine = make_engine()
        self._chain(engine, events=3, unchained=2)
        status, statements = self._statements(engine, security_service.audit_chain_status)
        assert (status["valid"], status["links"], status["unchained_events"]) == (True, 3, 2)
        [count] = [s for s in statements if "count(" in s and "audit_logs" in s]
        with engine.connect() as conn:
            plan = "\n".join(row[0] for row in conn.exec_driver_sql("EXPLAIN " + count))
        assert "Anti Join" in plan, plan

    def test_verification_issues_the_same_queries_however_long_the_chain(self, make_engine):
        short, long = make_engine(), make_engine()
        self._chain(short, events=2)
        self._chain(long, events=12)
        short_status, short_statements = self._statements(short, security_service.audit_chain_status)
        long_status, long_statements = self._statements(long, security_service.audit_chain_status)
        assert (short_status["valid"], long_status["valid"]) == (True, True)
        assert long_status["links"] == 12
        assert len(long_statements) == len(short_statements)

    def test_a_tampered_link_is_still_reported_at_its_sequence(self, make_engine):
        engine = make_engine()
        self._chain(engine, events=5)
        with engine.begin() as conn:
            conn.execute(text("UPDATE audit_logs SET action = 'FORGED' WHERE id = "
                              "(SELECT audit_log_id FROM audit_chain_links WHERE sequence = 4)"))
        with sessionmaker(bind=engine)() as db:
            status = security_service.audit_chain_status(db)
        assert (status["valid"], status["first_invalid_sequence"], status["links"]) == (False, 4, 5)
        assert status["reason"] == "event content does not match its link hash"



class TestIncrementalChainVerification:
    """At 1M links a full check takes minutes, most of it ML-DSA verification,
    so the operator endpoint checks only links added since the newest link
    the process already verified. Full checks still find what an incremental
    one cannot: rows edited behind that link."""

    _chain = staticmethod(TestAuditChainVerificationScales._chain)

    @staticmethod
    def _tamper(engine, sequence):
        with engine.begin() as conn:
            conn.execute(text("UPDATE audit_logs SET action = 'FORGED' WHERE id = "
                              "(SELECT audit_log_id FROM audit_chain_links WHERE sequence = :s)"),
                         {"s": sequence})

    @staticmethod
    def _check(engine, full):
        with sessionmaker(bind=engine)() as db:
            return security_service.audit_chain_status(db, full=full)

    def test_only_links_added_since_the_last_check_are_verified(self, make_engine, monkeypatch):
        engine = make_engine()
        self._chain(engine, events=5)
        first = self._check(engine, full=False)  # nothing verified yet: a full check
        assert (first["valid"], first["mode"], first["verified_from_sequence"]) == (True, "full", 1)
        self._chain(engine, events=3)
        verified = []
        real_verify = pqc.dsa_verify
        monkeypatch.setattr(pqc, "dsa_verify",
                            lambda pk, message, sig: verified.append(message) or real_verify(pk, message, sig))
        status = self._check(engine, full=False)
        assert (status["valid"], status["links"], status["mode"]) == (True, 8, "incremental")
        assert status["verified_from_sequence"] == 6 and len(verified) == 3
        assert status["last_full_verification_at"] == first["last_full_verification_at"]

    def test_tampering_with_a_new_link_is_found_and_located(self, make_engine):
        engine = make_engine()
        self._chain(engine, events=3)
        self._check(engine, full=False)
        self._chain(engine, events=3)
        self._tamper(engine, 5)
        status = self._check(engine, full=False)
        assert (status["valid"], status["mode"], status["first_invalid_sequence"]) == (False, "incremental", 5)
        assert status["reason"] == "event content does not match its link hash"

    def test_a_rewritten_head_link_forces_a_full_check(self, make_engine):
        engine = make_engine()
        self._chain(engine, events=4)
        self._check(engine, full=False)
        with engine.begin() as conn:
            conn.execute(text("UPDATE audit_chain_links SET entry_hash = repeat('a', 64) WHERE sequence = 4"))
        status = self._check(engine, full=False)
        assert (status["valid"], status["mode"], status["first_invalid_sequence"]) == (False, "full", 4)

    def test_old_rows_need_a_full_check_and_a_failure_ends_incremental_checks(self, make_engine):
        engine = make_engine()
        self._chain(engine, events=4)
        self._check(engine, full=False)
        self._tamper(engine, 2)
        # The documented limit: the edit is behind the verified head.
        assert self._check(engine, full=False)["valid"] is True
        full = self._check(engine, full=True)
        assert (full["valid"], full["first_invalid_sequence"]) == (False, 2)
        # Once a check has failed, the next ones are full until it passes again.
        after = self._check(engine, full=False)
        assert (after["valid"], after["mode"], after["first_invalid_sequence"]) == (False, "full", 2)

    def test_a_check_running_when_another_fails_does_not_restore_incremental_checks(
            self, make_engine, monkeypatch):
        engine = make_engine()
        self._chain(engine, events=3)
        self._check(engine, full=False)
        self._chain(engine, events=2)
        self._tamper(engine, 2)
        real_failure, nested = security_service._link_failure, []

        def failure_during_a_full_check(*args):
            if not nested:  # while this incremental check runs, a full one fails
                nested.append(None)
                nested[0] = self._check(engine, full=True)
            return real_failure(*args)

        monkeypatch.setattr(security_service, "_link_failure", failure_during_a_full_check)
        running = self._check(engine, full=False)
        monkeypatch.undo()
        assert (running["valid"], running["mode"]) == (True, "incremental")
        assert (nested[0]["valid"], nested[0]["first_invalid_sequence"]) == (False, 2)
        after = self._check(engine, full=False)
        assert (after["valid"], after["mode"], after["first_invalid_sequence"]) == (False, "full", 2)

    def test_the_endpoint_is_incremental_unless_full_is_requested(self, make_engine):
        from backend import main

        engine = make_engine()
        with sessionmaker(bind=engine)() as db:
            admin = models.User(email="ops@example.com", password_hash="x", role="admin")
            db.add(admin)
            db.commit()
            security_service.write_audit_log(db, admin.id, "EVENT")
            assert main.audit_chain(admin, db)["mode"] == "full"
            security_service.write_audit_log(db, admin.id, "EVENT")
            assert main.audit_chain(admin, db)["mode"] == "incremental"
            assert main.audit_chain(admin, db, full=True)["mode"] == "full"
        parameters = main.app.openapi()["paths"]["/api/security/audit-chain"]["get"]["parameters"]
        [full] = [p for p in parameters if p["in"] == "query"]
        assert (full["name"], full["schema"]["type"], full["schema"]["default"]) == ("full", "boolean", False)

    def test_manage_command_checks_every_link_and_fails_on_tampering(self, make_engine, capsys):
        import json

        from backend import manage

        engine = make_engine()
        self._chain(engine, events=3)
        assert manage._verify_audit_chain(sessionmaker(bind=engine)) == 0
        assert json.loads(capsys.readouterr().out)["mode"] == "full"
        self._tamper(engine, 1)
        assert manage._verify_audit_chain(sessionmaker(bind=engine)) == 1
        printed = json.loads(capsys.readouterr().out)
        assert (printed["valid"], printed["first_invalid_sequence"]) == (False, 1)

# ── server key registration ───────────────────────────────────────────────────

def test_processes_registering_the_same_server_key_at_once_all_succeed(make_engine):
    """Every API and research worker registers the deployment's key when it
    starts. A check-then-insert let all but one of them crash on the unique
    fingerprint when they started together."""
    Session = sessionmaker(bind=make_engine())
    pk, sk, _ = pqc.dsa_keygen()
    key_ids, errors = [], []
    barrier = threading.Barrier(6)

    def register():
        identity = security_service.ServerIdentity()
        identity.dsa_pk, identity.dsa_sk = pk, sk
        identity.fingerprint = hashlib.sha256(pk).hexdigest()
        db = Session()
        try:
            barrier.wait()
            key_ids.append(identity.register_in_db(db))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            db.close()

    threads = [threading.Thread(target=register) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(set(key_ids)) == 1
    with Session() as db:
        assert db.query(models.ServerSigningKey).count() == 1


def test_replacing_an_expanded_server_key_with_a_seed_key_keeps_the_chain_verifiable(make_engine):
    """The upgrade path for native signing: a deployment swaps its legacy
    4032-byte server key for a generated seed key and restarts. Everything
    the old key signed must still verify."""
    from dilithium_py.ml_dsa import ML_DSA_65

    Session = sessionmaker(bind=make_engine())

    def restart_with(pk, sk):
        identity = security_service.ServerIdentity()
        identity.dsa_pk, identity.dsa_sk = pk, sk
        identity.fingerprint = hashlib.sha256(pk).hexdigest()
        return identity

    old_pk, old_sk = ML_DSA_65.keygen()
    new_pk, new_sk, _ = pqc.dsa_keygen()
    entries = []
    original = security_service.server_identity
    try:
        for pk, sk in ((old_pk, old_sk), (new_pk, new_sk)):
            security_service.server_identity = restart_with(pk, sk)
            with Session() as db:
                for i in range(2):
                    entries.append(security_service.write_audit_log(db, None, "EVENT", metadata={"i": i}).id)
        with Session() as db:
            status = security_service.audit_chain_status(db)
            assert (status["valid"], status["links"]) == (True, 4)
            assert all(security_service.verify_audit_log(db, entry) for entry in entries)
            assert db.query(models.ServerSigningKey).count() == 2
    finally:
        security_service.server_identity = original


def test_concurrent_first_signers_share_one_lazily_generated_key(make_engine, monkeypatch):
    """Without configured keys, the first signer generates the server key.
    Requests arriving together each generated their own and registered it
    under the last one's key_id: a 500 on server_signing_keys_pkey, and
    signatures that no registered key verifies."""
    real_keygen = pqc.dsa_keygen
    keygens = []

    def keygen_at_reference_speed():
        # dilithium-py's keygen takes ~25 ms; native takes ~0.1 ms, which
        # makes the race real but rare. Hold the window open on every run,
        # finishing concurrent keygens at different times.
        keygens.append(None)
        time.sleep(0.025 * len(keygens))
        return real_keygen()

    monkeypatch.setattr(pqc, "dsa_keygen", keygen_at_reference_speed)
    Session = sessionmaker(bind=make_engine())
    identity = security_service.ServerIdentity()
    assert identity.dsa_sk is None
    results, errors = [], []
    barrier = threading.Barrier(8)

    def first_order(n):
        db = Session()
        try:
            barrier.wait()
            identity.ensure_registered(db)
            message = f"order-{n}".encode()
            results.append((message, identity.sign(message), identity.key_id))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            db.close()

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=first_order, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(previous)
    assert errors == []
    assert len(keygens) == 1
    with Session() as db:
        keys = db.query(models.ServerSigningKey).all()
    assert len(keys) == 1
    assert {key_id for _, _, key_id in results} == {keys[0].key_id}
    registered_pk = pqc.unb64(keys[0].public_key)
    assert all(pqc.dsa_verify(registered_pk, message, signature)
               for message, signature, _ in results)


def test_writers_in_separate_processes_extend_one_unbroken_chain(make_engine, monkeypatch):
    """Writers in different processes share no Python lock: only the advisory
    lock orders them. Each must read the chain head after that lock is
    granted, or two links claim the same predecessor (a unique violation on
    sequence, or a fork that verification rejects)."""
    # A process of its own per append: no shared queue, no group commit.
    monkeypatch.setattr(security_service, "_appender_for", lambda db: security_service._AuditAppender())
    Session = sessionmaker(bind=make_engine())
    with Session() as db:
        security_service.server_identity.ensure_registered(db)
    writers, per_writer = 8, 12
    errors, written = [], []
    barrier = threading.Barrier(writers)

    def write(n):
        with Session() as db:
            barrier.wait()
            for i in range(per_writer):
                try:
                    entry = security_service.write_audit_log(db, None, "EVENT", metadata={"n": n, "i": i})
                    written.append(entry.id)
                except Exception as exc:  # pragma: no cover - reported below
                    errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with Session() as db:
        sequences = db.execute(select(models.AuditChainLink.sequence)
                               .order_by(models.AuditChainLink.sequence)).scalars().all()
        assert sequences == list(range(1, writers * per_writer + 1))
        status = security_service.audit_chain_status(db)
        assert status["valid"] and status["unchained_events"] == 0, status
        assert {e.id for e in db.execute(select(models.AuditLog)).scalars()} == set(written)


class TestAuditGroupCommit:
    """Appends queued while one thread of a process writes the chain are
    written after it in one transaction, every link still signed."""

    @pytest.fixture
    def chain(self, make_engine, monkeypatch):
        Session = sessionmaker(bind=make_engine())
        with Session() as db:
            security_service.server_identity.ensure_registered(db)
            appender = security_service._appender_for(db)
        batches, release = [], threading.Event()
        real = security_service._insert_chain_links

        def first_batch_waits(db, batch):
            if not batches:
                assert release.wait(30), "test never released the first batch"
            batches.append([p.action for p in batch])
            return real(db, batch)

        monkeypatch.setattr(security_service, "_insert_chain_links", first_batch_waits)
        return Session, appender, batches, release

    @staticmethod
    def _write(Session, results, user_id, action, n):
        try:
            with Session() as db:
                entry = security_service.write_audit_log(db, user_id, action, metadata={"n": n})
                results[n] = (entry.id, entry.action, entry.metadata_json)
        except Exception as exc:
            results[n] = exc

    def _run(self, chain, writers, user_ids=None):
        """Writer 0 leads a batch held until every other writer has queued."""
        Session, appender, batches, release = chain
        user_ids = user_ids or {}
        results = {}
        threads = [threading.Thread(target=self._write, args=(Session, results, user_ids.get(n), f"E{n}", n),
                                    daemon=True)
                   for n in range(writers)]
        threads[0].start()
        deadline = time.monotonic() + 10
        while not appender._writing and time.monotonic() < deadline:
            time.sleep(0.002)
        for t in threads[1:]:
            t.start()
        while len(appender._queue) < writers - 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert len(appender._queue) == writers - 1
        release.set()
        for t in threads:
            t.join(30)
        assert not any(t.is_alive() for t in threads), "an append never returned"
        return results

    def _assert_chain(self, Session, links):
        with Session() as db:
            sequences = db.execute(select(models.AuditChainLink.sequence)
                                   .order_by(models.AuditChainLink.sequence)).scalars().all()
            assert sequences == list(range(1, links + 1))
            assert db.execute(select(func.count()).select_from(models.AuditChainLink)
                              .where(models.AuditChainLink.checkpoint_signature.is_(None))).scalar() == 0
            status = security_service.audit_chain_status(db)
            assert status["valid"] and status["unchained_events"] == 0, status

    def test_queued_appends_share_one_transaction(self, chain):
        Session, _, batches, _ = chain
        results = self._run(chain, writers=8)
        assert batches == [["E0"], [f"E{n}" for n in range(1, 8)]]
        for n in range(8):
            assert not isinstance(results[n], Exception), results[n]
            assert results[n][1:] == (f"E{n}", {"n": n})  # each caller gets its own event
        self._assert_chain(Session, 8)

    def test_a_batch_never_exceeds_the_cap(self, chain, monkeypatch):
        Session, _, batches, _ = chain
        monkeypatch.setattr(security_service, "_AUDIT_BATCH_MAX", 3)
        results = self._run(chain, writers=10)
        assert [len(b) for b in batches] == [1, 3, 3, 3]
        assert all(not isinstance(r, Exception) for r in results.values())
        self._assert_chain(Session, 10)

    def test_a_leader_keeps_writing_until_its_own_append_is_written(self, monkeypatch):
        """Waiting threads wake in no fixed order: one whose append is past
        the cap can lead, and must not return before its own is written."""
        monkeypatch.setattr(security_service, "_AUDIT_BATCH_MAX", 2)
        written = []

        def write(db, batch):
            written.append([p.action for p in batch])
            for p in batch:
                p.done = True

        monkeypatch.setattr(security_service, "_write_chain_batch", write)
        appender = security_service._AuditAppender()
        make = lambda action: security_service._PendingAudit(action=action)
        appender._queue.extend(make(f"Q{n}") for n in range(5))  # queued by threads still waiting
        own = make("OWN")
        appender.append(None, own)
        assert own.done
        assert written == [["Q0", "Q1"], ["Q2", "Q3"], ["Q4", "OWN"]]

    def test_one_bad_append_fails_only_its_own_caller(self, chain):
        Session, _, batches, _ = chain
        results = self._run(chain, writers=5, user_ids={3: "no-such-user"})
        assert isinstance(results[3], IntegrityError)
        assert all(not isinstance(results[n], Exception) for n in (0, 1, 2, 4))
        # the batch failed as a whole, then each append went alone
        assert batches == [["E0"], ["E1", "E2", "E3", "E4"], ["E1"], ["E2"], ["E3"], ["E4"]]
        self._assert_chain(Session, 4)
