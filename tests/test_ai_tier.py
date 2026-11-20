"""The AI tier (backend/ai): its boundary, settings, job processes, flags and queue.

The AI tier runs untrusted-input work as qs_ai_worker. These tests hold it
to its contract: it imports nothing of the trusted tier, refuses the trusted
tier's secrets in production, hands its job processes no secret, runs only
the targets it lists, and runs a stage only while that stage and the master
flag are both on, checked when a job is queued, claimed and while it runs.
The role's privileges are tested in tests/test_db_roles.py.
"""
import json
import os
import subprocess
import sys
import time

import psycopg
import pytest

from backend.ai import child, flags, queue, selftest
from backend.ai import settings as ai_settings
from backend.ai.kinds import KINDS, TARGETS, AiKind
from backend.ai.worker import AiWorker
from conftest import _libpq, schema_url
from test_research_boundary import ROOT, _chain, closure

OPS = "ops@example.com"
# Names the operating system adds to every process it starts, whatever the
# parent passed: macOS's CoreFoundation sets its text encoding at exec.
PLATFORM_ADDED = {"__CF_USER_TEXT_ENCODING"} if sys.platform == "darwin" else set()
# Secrets and interpreter settings a job process must never see.
CANARIES = {
    "DATABASE_URL": "postgresql://quantumsentinel:canary@postgres/db",
    "AI_DATABASE_URL": "postgresql://qs_ai_worker:canary@postgres/db",
    "DB_AI_WORKER_PASSWORD": "canary", "DB_APP_PASSWORD": "canary", "PGPASSWORD": "canary",
    "JWT_PRIVATE_KEY": "canary", "SERVER_DSA_PRIVATE_KEY": "canary", "REDIS_URL": "redis://:canary@redis",
    "AI_DATABASE_URL_FILE": "/run/secrets/canary", "PYTHONPATH": "/canary", "PYTHONSTARTUP": "/canary.py",
}


def _minimal_environment(**extra) -> dict:
    """Just enough of this machine's environment for an interpreter to start."""
    keep = {name: value for name, value in os.environ.items()
            if name.upper() in {"PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR"}}
    return {**keep, **extra}


def _python(code: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-E", "-s", "-c", code], cwd=ROOT, env=_minimal_environment(**env),
                          capture_output=True, text=True, timeout=120)


# -- the boundary -------------------------------------------------------------

def test_no_import_statement_leads_out_of_the_ai_tier():
    """The closure of every import statement, module-level or inside a
    function, from the worker, the job process and each job target."""
    roots = ["backend.ai.worker", "backend.ai.child", "backend.ai.settings",
             *(target.partition(":")[0] for target in TARGETS)]
    reached = closure(*roots)
    outside = sorted(m for m in reached if m != "backend.ai" and not m.startswith("backend.ai."))
    assert not outside, "\n".join(_chain(reached, m) for m in outside)


def test_loading_every_ai_module_loads_nothing_of_the_trusted_tier():
    """In a fresh interpreter, under the AI worker's process role."""
    result = _python(
        "import importlib, json, pkgutil, sys\n"
        "import backend.ai\n"
        "for info in pkgutil.walk_packages(backend.ai.__path__, 'backend.ai.'):\n"
        "    importlib.import_module(info.name)\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('backend.')"
        " and not (m == 'backend.ai' or m.startswith('backend.ai.')))))\n",
        QS_PROCESS_ROLE="ai-worker")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("role", ["ai-worker", "ai-job"])
def test_the_trusted_configuration_refuses_to_load_in_the_ai_tier(role):
    result = _python("import backend.config", QS_PROCESS_ROLE=role)
    assert result.returncode != 0
    assert "QS_PROCESS_ROLE must be one of" in result.stderr


def test_every_kind_runs_a_function_of_the_ai_tier_in_a_known_stage():
    import importlib

    assert KINDS and TARGETS == {kind.target for kind in KINDS.values()}
    for name, kind in KINDS.items():
        assert kind.name == name and kind.stage in flags.STAGES and kind.timeout_seconds > 0
        module, _, function = kind.target.partition(":")
        assert module.startswith("backend.ai.")
        assert callable(getattr(importlib.import_module(module), function))
        if kind.driver is not None:
            module, _, function = kind.driver.partition(":")
            assert module.startswith("backend.ai.") and module != "backend.ai.child"
            assert callable(getattr(importlib.import_module(module), function))
        for limit in (kind.child_timeout_seconds, kind.child_output_max_bytes):
            assert limit is None or limit > 0
        assert kind.child_timeout_seconds is None or kind.child_timeout_seconds <= kind.timeout_seconds



@pytest.mark.parametrize("fields", [{"target": "backend.main:app"}, {"driver": "backend.main:app"},
                                    {"driver": "os:system"}, {"stage": "unknown"}])
def test_a_kind_that_runs_code_outside_the_ai_tier_is_refused(fields):
    from backend.ai import kinds

    good = {"name": "x", "target": "backend.ai.selftest:run", "stage": "selftest", "timeout_seconds": 60}
    kinds.check({"x": AiKind(**good)})
    kinds.check({"x": AiKind(**good, driver="backend.ai.harvest:run")})
    with pytest.raises(RuntimeError, match="under backend.ai"):
        kinds.check({"x": AiKind(**{**good, **fields})})


# -- settings -----------------------------------------------------------------

def _settings(**environ) -> dict:
    return {"QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL": "postgresql+psycopg://qs_ai_worker:pw@db/qs",
            **environ}


def test_the_settings_need_the_ai_workers_role_and_url():
    with pytest.raises(ai_settings.SettingsError, match="QS_PROCESS_ROLE"):
        ai_settings.load({"AI_DATABASE_URL": "postgresql://x@db/qs"})
    with pytest.raises(ai_settings.SettingsError, match="QS_PROCESS_ROLE"):
        ai_settings.load(_settings(QS_PROCESS_ROLE="api"))
    with pytest.raises(ai_settings.SettingsError, match="AI_DATABASE_URL is required"):
        ai_settings.load({"QS_PROCESS_ROLE": "ai-worker"})
    with pytest.raises(ai_settings.SettingsError, match="PostgreSQL URL"):
        ai_settings.load(_settings(AI_DATABASE_URL="mysql://x@db/qs"))
    loaded = ai_settings.load(_settings())
    assert loaded.database_url == "postgresql://qs_ai_worker:pw@db/qs"
    assert (loaded.schema, loaded.heartbeat_seconds, loaded.lease_seconds) == ("ai", 5, 60)


@pytest.mark.parametrize("schema", ["ai; DROP SCHEMA public", 'a"i', "1ai", "AI", "a" * 64])
def test_the_ai_schema_must_be_a_plain_name(schema):
    with pytest.raises(ai_settings.SettingsError, match="AI_DB_SCHEMA"):
        ai_settings.load(_settings(AI_DB_SCHEMA=schema))


def test_a_lease_must_outlast_two_heartbeats():
    with pytest.raises(ai_settings.SettingsError, match="twice"):
        ai_settings.load(_settings(AI_WORKER_HEARTBEAT_SECONDS="10", AI_JOB_LEASE_SECONDS="20"))
    with pytest.raises(ai_settings.SettingsError):
        ai_settings.load(_settings(AI_WORKER_HEARTBEAT_SECONDS="soon"))


def test_settings_may_come_from_files(tmp_path):
    secret = tmp_path / "url"
    secret.write_text("postgresql://qs_ai_worker:from-file@db/qs\n", encoding="utf-8")
    loaded = ai_settings.load({"QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL_FILE": str(secret)})
    assert loaded.database_url == "postgresql://qs_ai_worker:from-file@db/qs"


@pytest.mark.parametrize("name", ai_settings.TRUSTED_SECRETS)
def test_in_production_the_ai_worker_refuses_every_trusted_secret(name):
    with pytest.raises(ai_settings.SettingsError, match=name):
        ai_settings.load(_settings(ENVIRONMENT="production", **{name: "x"}))
    with pytest.raises(ai_settings.SettingsError, match=name):
        ai_settings.load(_settings(ENVIRONMENT="production", **{f"{name}_FILE": "/run/secrets/x"}))
    assert ai_settings.load(_settings(**{name: "x"})).environment == "development"


def test_the_worker_will_not_start_in_production_with_a_trusted_secret():
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-m", "backend.ai.worker"], cwd=ROOT, capture_output=True, text=True,
        timeout=120, env=_minimal_environment(
            QS_PROCESS_ROLE="ai-worker", ENVIRONMENT="production", DATABASE_URL="postgresql://x@db/qs",
            AI_DATABASE_URL="postgresql://qs_ai_worker:pw@127.0.0.1:1/qs"))
    assert result.returncode == 2
    assert "DATABASE_URL" in result.stderr


# -- job processes --------------------------------------------------------------

def _run_child(target: str, payload: dict, environ: dict | None = None, timeout: float = 60):
    process = child.start(256 << 20, timeout, environ=environ)
    stdout, stderr = process.communicate(child.request(target, payload), timeout=timeout)
    return process.returncode, stdout, stderr


def test_a_job_process_inherits_only_the_allow_list():
    """Through the real start path: the worker's database URL, passwords,
    keys and PYTHON* settings never reach the job."""
    environ = {**os.environ, **CANARIES, "LC_ALL": "C"}
    code, stdout, stderr = _run_child("backend.ai.selftest:run", {}, environ)
    assert code == 0, stderr
    result = json.loads(stdout)
    names = set(result["environment"])
    locale = {name for name in names if name.upper().startswith("LC_")}
    assert {name.upper() for name in names - locale - PLATFORM_ADDED} <= {*child.CHILD_ENVIRONMENT, "QS_PROCESS_ROLE"}
    assert not {name.upper() for name in names} & set(CANARIES)
    assert b"canary" not in stdout
    assert result["process_role"] == "ai-job"
    assert result["trusted_modules"] == []
    if sys.platform.startswith("linux"):
        assert result["limits"]["RLIMIT_CORE"] == 0
        assert result["limits"]["RLIMIT_AS"] == 256 << 20


def test_a_job_process_runs_only_listed_targets():
    for target in ("backend.config:settings", "os:system", "backend.ai.child:main", "backend.ai.selftest:missing"):
        code, _, stderr = _run_child(target, {})
        assert code == 1
        assert b"PermissionError: not an AI job target" in stderr, stderr
    with pytest.raises(PermissionError):
        child.run_target("os:getcwd", {})


def test_a_job_process_refuses_to_run_outside_its_role():
    result = subprocess.run(child.command(), cwd=ROOT, input=child.request("backend.ai.selftest:run", {}),
                            env=_minimal_environment(QS_PROCESS_ROLE="ai-worker"), capture_output=True,
                            timeout=120)
    assert result.returncode == 1
    assert b"QS_PROCESS_ROLE=ai-job" in result.stderr


def test_a_failing_target_reports_one_line_and_no_result():
    code, stdout, stderr = _run_child("backend.ai.selftest:run", {"hold_seconds": True})
    assert code == 1 and stdout == b""
    assert b"hold_seconds must be a number" in stderr
    with pytest.raises(child.JobFailed, match="hold_seconds must be a number"):
        child.outcome(code, stdout, stderr, 1 << 20)


def test_outcomes():
    assert child.outcome(0, b'{"a": 1}', b"", 100) == {"a": 1}
    with pytest.raises(child.JobFailed, match="exited with status 9"):
        child.outcome(9, b"", b"", 100)
    with pytest.raises(child.JobFailed, match="larger than 4 bytes"):
        child.outcome(0, b'{"a": 1}', b"", 4)
    with pytest.raises(child.JobFailed, match="not JSON"):
        child.outcome(0, b"{", b"", 100)
    with pytest.raises(child.JobFailed, match="not a JSON object"):
        child.outcome(0, b"[1]", b"", 100)


def test_the_selftest_checks_its_payload():
    for hold in (-1, 31, "1", True, None):
        with pytest.raises(ValueError):
            selftest.run({"hold_seconds": hold})
    assert selftest.run({})["pid"] == os.getpid()


# -- flags and the queue, on a migrated schema ----------------------------------

class Ai:
    def __init__(self, engine):
        with engine.connect() as connection:
            self.main = connection.exec_driver_sql("SELECT current_schema()").scalar()
        self.schema = f"{self.main}_ai"
        self.url = schema_url(self.main)
        self.conn = psycopg.connect(_libpq(self.url), autocommit=True)

    def flag(self, name, on):
        with self.conn.transaction():
            flags.set_flag(self.conn, self.schema, name, on, OPS, "test")

    def switch(self, master, stage, name="selftest"):
        self.flag(flags.MASTER, master)
        self.flag(name, stage)

    def enqueue(self, payload=None, kind="ai_selftest"):
        with self.conn.transaction():
            return queue.enqueue(self.conn, self.schema, kind, payload or {}, OPS)

    def claim(self, worker="w1", lease=60):
        with self.conn.transaction():
            return queue.claim_next(self.conn, self.schema, worker, lease)

    def job(self, job_id):
        return queue.get(self.conn, self.schema, job_id)

    def worker(self, **environ) -> AiWorker:
        settings = ai_settings.load({
            "QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL": self.url.render_as_string(hide_password=False),
            "AI_DB_SCHEMA": self.schema, "AI_WORKER_HEARTBEAT_SECONDS": "0.2", "AI_JOB_LEASE_SECONDS": "30",
            **environ})
        runner = AiWorker(settings)
        runner.connect()
        return runner


@pytest.fixture
def ai(make_engine):
    db = Ai(make_engine(migrated=True))
    yield db
    db.conn.close()


def test_every_flag_starts_off_and_a_stage_needs_the_master_flag_too(ai):
    assert [(r["name"], r["enabled"], r["effective"]) for r in flags.listing(ai.conn, ai.schema)] == [
        (flags.MASTER, False, False), ("arxiv_sync", False, False), ("metadata_sync", False, False),
        ("selftest", False, False)]
    with pytest.raises(queue.StageOff):
        ai.enqueue()
    ai.switch(master=False, stage=True)
    assert not flags.enabled(ai.conn, ai.schema, "selftest")
    with pytest.raises(queue.StageOff):
        ai.enqueue()
    ai.switch(master=True, stage=False)
    with pytest.raises(queue.StageOff):
        ai.enqueue()
    assert not flags.enabled(ai.conn, ai.schema, "no_such_stage")
    ai.switch(master=True, stage=True)
    assert ai.job(ai.enqueue())["status"] == "queued"
    listing = {r["name"]: r for r in flags.listing(ai.conn, ai.schema)}
    assert [(listing[n]["effective"], listing[n]["set_by"], listing[n]["reason"]) for n in (flags.MASTER, "selftest")
            ] == [(True, OPS, "test")] * 2
    assert (listing["arxiv_sync"]["enabled"], listing["arxiv_sync"]["effective"]) == (False, False)


def test_flags_are_known_and_set_by_someone_for_a_reason(ai):
    with ai.conn.transaction():
        with pytest.raises(flags.FlagError, match="unknown flag"):
            flags.set_flag(ai.conn, ai.schema, "everything", True, OPS, "why")
        for set_by, reason in (("", "why"), (OPS, "  ")):
            with pytest.raises(flags.FlagError, match="who sets the flag and why"):
                flags.set_flag(ai.conn, ai.schema, "selftest", True, set_by, reason)
        assert flags.set_flag(ai.conn, ai.schema, "selftest", True, OPS, "why") is False
        assert flags.set_flag(ai.conn, ai.schema, "selftest", False, OPS, "why") is True


def test_enqueue_checks_its_job(ai):
    ai.switch(True, True)
    with pytest.raises(queue.EnqueueError, match="unknown AI job kind"):
        ai.enqueue(kind="shell")
    with pytest.raises(queue.EnqueueError, match="at most"):
        ai.enqueue({"x": "y" * queue.MAX_PAYLOAD_BYTES})
    with ai.conn.transaction(), pytest.raises(queue.EnqueueError, match="who requests"):
        queue.enqueue(ai.conn, ai.schema, "ai_selftest", {}, " ")


def test_a_queued_job_waits_while_its_stage_is_off_and_runs_once_it_is_on(ai):
    """No restart: the claim query reads the flags."""
    ai.switch(True, True)
    job_id = ai.enqueue()
    for master, stage in ((True, False), (False, True), (False, False)):
        ai.switch(master, stage)
        assert ai.claim() is None
        assert ai.job(job_id)["status"] == "queued"
    ai.switch(True, True)
    claimed = ai.claim()
    assert (claimed["id"], claimed["attempts"]) == (job_id, 1)
    assert ai.claim() is None


def test_a_job_whose_stage_is_switched_off_while_it_runs_is_stopped_and_handed_back(ai):
    ai.switch(True, True)
    job_id = ai.enqueue({"hold_seconds": 20})
    runner = ai.worker()
    try:
        with runner.conn.transaction():
            job = queue.claim_next(runner.conn, runner.schema, runner.id, 30)
        ai.flag("selftest", False)
        started = time.monotonic()
        assert runner.execute(job) == "released (stage switched off)"
        assert time.monotonic() - started < 10
        released = ai.job(job_id)
        assert (released["status"], released["attempts"], released["started_at"]) == ("queued", 0, None)
    finally:
        runner.close()


def test_the_worker_runs_a_selftest_end_to_end(ai):
    ai.switch(True, True)
    job_id = ai.enqueue()
    runner = ai.worker()
    try:
        assert runner.schema_ready()
        assert runner.run_once() == "succeeded"
        assert runner.run_once() is None
    finally:
        runner.close()
    job = ai.job(job_id)
    assert (job["status"], job["attempts"], job["error"]) == ("succeeded", 1, None)
    assert job["result"]["process_role"] == "ai-job" and job["result"]["trusted_modules"] == []
    assert not {name.upper() for name in job["result"]["environment"]} & set(ai_settings.TRUSTED_SECRETS)


def test_a_job_that_runs_too_long_is_killed_and_fails(ai, monkeypatch):
    monkeypatch.setitem(KINDS, "ai_selftest", AiKind("ai_selftest", "backend.ai.selftest:run", "selftest",
                                                     timeout_seconds=1))
    ai.switch(True, True)
    job_id = ai.enqueue({"hold_seconds": 20})
    runner = ai.worker()
    try:
        started = time.monotonic()
        assert runner.run_once() == "timed out"
        assert time.monotonic() - started < 10
    finally:
        runner.close()
    job = ai.job(job_id)
    assert job["status"] == "failed" and "longer than 1 s" in job["error"]


def test_a_job_that_raises_fails_with_its_reason(ai):
    ai.switch(True, True)
    job_id = ai.enqueue({"hold_seconds": 99})
    runner = ai.worker()
    try:
        assert runner.run_once() == "failed"
    finally:
        runner.close()
    job = ai.job(job_id)
    assert job["status"] == "failed"
    assert job["error"].startswith("The AI job failed: ValueError: hold_seconds must be a number")


def test_a_job_whose_worker_died_is_reaped(ai):
    ai.switch(True, True)
    once = ai.enqueue()
    with ai.conn.transaction():
        twice = queue.enqueue(ai.conn, ai.schema, "ai_selftest", {}, OPS)
        ai.conn.execute(f'UPDATE "{ai.schema}".jobs SET max_attempts = 2 WHERE id = %s', (twice,))
    assert ai.claim("dead", lease=0.01)["id"] == once
    assert ai.claim("dead", lease=0.01)["id"] == twice
    time.sleep(0.1)
    with ai.conn.transaction():
        assert queue.renew(ai.conn, ai.schema, once, "someone-else", 60) == "lost"
        assert queue.reap_expired(ai.conn, ai.schema) == (1, 1)
        assert queue.complete(ai.conn, ai.schema, once, "dead", {}) is False
    assert (ai.job(once)["status"], ai.job(once)["error"]) == ("failed", queue.WORKER_GONE_ERROR)
    assert (ai.job(twice)["status"], ai.job(twice)["attempts"]) == ("queued", 1)


def test_release_and_purge(ai):
    ai.switch(True, True)
    job_id = ai.enqueue()
    ai.claim("w1")
    with ai.conn.transaction():
        assert queue.release(ai.conn, ai.schema, job_id, "w2") is False
        assert queue.release(ai.conn, ai.schema, job_id, "w1") is True
    assert (ai.job(job_id)["status"], ai.job(job_id)["attempts"]) == ("queued", 0)
    ai.claim("w1")
    with ai.conn.transaction():
        assert queue.complete(ai.conn, ai.schema, job_id, "w1", {"ok": True}) is True
        assert queue.purge_finished(ai.conn, ai.schema, 7) == 0
        ai.conn.execute(f"UPDATE \"{ai.schema}\".jobs SET finished_at = now() - interval '8 days'")
        assert queue.purge_finished(ai.conn, ai.schema, 7) == 1
    assert ai.job(job_id) is None


def test_heartbeats_and_unregistering(ai):
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    workers = f'"{ai.schema}".workers'
    with ai.conn.transaction():
        queue.heartbeat(ai.conn, ai.schema, "stale", "h", 1, now)
        ai.conn.execute(f"UPDATE {workers} SET last_seen_at = now() - interval '1 hour'")
        queue.heartbeat(ai.conn, ai.schema, "w1", "h", 2, now)
    assert [r[0] for r in ai.conn.execute(f"SELECT id FROM {workers}")] == ["w1"]
    with ai.conn.transaction():
        queue.unregister(ai.conn, ai.schema, "w1")
    assert ai.conn.execute(f"SELECT count(*) FROM {workers}").fetchone() == (0,)


def test_the_running_worker_stops_cleanly(ai):
    import threading

    runner = ai.worker(AI_WORKER_POLL_SECONDS="0.1")
    runner.close()
    thread = threading.Thread(target=runner.run_forever)
    thread.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not ai.conn.execute(
            f'SELECT count(*) FROM "{ai.schema}".workers').fetchone()[0]:
        time.sleep(0.05)
    assert ai.conn.execute(f'SELECT id FROM "{ai.schema}".workers').fetchone() == (runner.id,)
    runner.stop()
    thread.join(20)
    assert not thread.is_alive()
    assert ai.conn.execute(f'SELECT count(*) FROM "{ai.schema}".workers').fetchone() == (0,)
