"""Research pipelines (R6-1): bounded runs of research jobs, and idempotent
job submission.

A pipeline is a budget of jobs; each job queued in it spends one, whatever
then becomes of the job, and one user may queue at most
RESEARCH_PIPELINE_DAILY_JOBS pipeline jobs per UTC day. A job submitted
under an Idempotency-Key is queued once: the same request again returns it.
The schema comes from the migrations, so migration 0018's CHECKs are the
ones under test.
"""
import contextlib
import datetime as dt
import json
import threading

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend import database, main, models, schemas
from backend.services import research_jobs, research_pipelines, research_trials

KIND = "correlation"


@pytest.fixture
def Session(make_engine):
    return sessionmaker(bind=make_engine(migrated=True))


@pytest.fixture
def db(Session):
    session = Session()
    yield session
    session.close()


def make_user(db, email="researcher@example.com"):
    user = models.User(email=email, password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def alice(db):
    return make_user(db, "alice@example.com")


@pytest.fixture
def bob(db):
    return make_user(db, "bob@example.com")


def body(response):
    return json.loads(response.body)


def new_pipeline(db, user, max_jobs=3, name="sweep"):
    return research_pipelines.create(db, user.id, name, max_jobs)


def enqueue(db, user, params=None, **kwargs):
    return research_jobs.enqueue(db, user.id, KIND, params or {"window": 20}, **kwargs)


def counters(db, pipeline_id, user_id):
    db.expire_all()
    queued = db.get(models.ResearchPipeline, pipeline_id).jobs_queued
    today = db.execute(select(func.coalesce(func.sum(models.ResearchPipelineDay.jobs_queued), 0))
                       .where(models.ResearchPipelineDay.user_id == user_id)).scalar_one()
    return queued, today


def job_count(db, user_id):
    return db.execute(select(func.count()).select_from(models.ResearchJob)
                      .where(models.ResearchJob.user_id == user_id)).scalar_one()


def refused(status_code, call, *args, **kwargs):
    with pytest.raises(research_pipelines.JobRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.status_code == status_code, caught.value.detail
    return caught.value.detail


# ── pipelines ─────────────────────────────────────────────────────────────────

class TestPipelines:
    def test_create_list_get_and_close_through_the_api(self, db, alice):
        created = main.create_research_pipeline(schemas.ResearchPipelineRequest(name="  momentum sweep ", max_jobs=2),
                                                user=alice, db=db)
        assert created["name"] == "momentum sweep" and created["status"] == "open"
        assert (created["max_jobs"], created["jobs_queued"], created["jobs_remaining"]) == (2, 0, 2)
        pid = created["pipeline_id"]
        assert [p["pipeline_id"] for p in main.list_research_pipelines(user=alice, db=db)["pipelines"]] == [pid]
        enqueue(db, alice, pipeline_id=pid, stage="screen")
        detail = main.get_research_pipeline(pid, user=alice, db=db)
        assert detail["jobs_queued"] == 1 and detail["jobs_remaining"] == 1
        [job] = detail["jobs"]
        assert job["kind"] == KIND and job["stage"] == "screen" and job["status"] == "queued"
        closed = main.close_research_pipeline(pid, user=alice, db=db)
        assert closed["status"] == "closed" and closed["closed_at"] and closed["jobs_remaining"] == 0
        assert main.close_research_pipeline(pid, user=alice, db=db)["closed_at"] == closed["closed_at"]
        actions = db.execute(select(models.AuditLog.action).where(models.AuditLog.resource_id == pid)
                             .order_by(models.AuditLog.created_at)).scalars().all()
        assert actions == ["PIPELINE_CREATED", "PIPELINE_CLOSED"]  # closing twice audits once

    @pytest.mark.parametrize("name,max_jobs", [("", 5), ("   ", 5), ("x" * 121, 5), ("ok", 0),
                                               ("ok", 10_001), ("ok", -1)])
    def test_create_refuses_a_bad_name_or_budget(self, db, alice, name, max_jobs):
        with pytest.raises(HTTPException) as caught:
            main.create_research_pipeline(schemas.ResearchPipelineRequest(name=name, max_jobs=max_jobs),
                                          user=alice, db=db)
        assert caught.value.status_code == 422
        assert db.execute(select(func.count()).select_from(models.ResearchPipeline)).scalar_one() == 0

    def test_request_schema_forbids_unknown_fields(self):
        with pytest.raises(Exception):
            schemas.ResearchPipelineRequest(name="x", max_jobs=1, jobs_queued=-5)

    def test_another_users_pipeline_is_not_found(self, db, alice, bob):
        pid = new_pipeline(db, alice).id
        for call in (main.get_research_pipeline, main.close_research_pipeline):
            with pytest.raises(HTTPException) as caught:
                call(pid, user=bob, db=db)
            assert caught.value.status_code == 404
        assert main.list_research_pipelines(user=bob, db=db) == {"pipelines": []}
        assert refused(404, enqueue, db, bob, pipeline_id=pid) == "Research pipeline not found"
        assert refused(404, enqueue, db, bob, pipeline_id="no-such-pipeline")
        assert counters(db, pid, bob.id) == (0, 0) and job_count(db, bob.id) == 0


# ── budgets ───────────────────────────────────────────────────────────────────

class TestBudgets:
    def test_the_pipeline_budget_caps_the_jobs_it_queues(self, db, alice):
        pid = new_pipeline(db, alice, max_jobs=2).id
        enqueue(db, alice, {"window": 1}, pipeline_id=pid)
        enqueue(db, alice, {"window": 2}, pipeline_id=pid)
        assert "all 2 jobs" in refused(409, enqueue, db, alice, {"window": 3}, pipeline_id=pid)
        assert counters(db, pid, alice.id) == (2, 2) and job_count(db, alice.id) == 2
        enqueue(db, alice, {"window": 3})  # outside any pipeline: unbudgeted
        assert counters(db, pid, alice.id) == (2, 2)

    def test_a_closed_pipeline_queues_nothing(self, db, alice):
        pid = new_pipeline(db, alice).id
        research_pipelines.close(db, alice.id, pid)
        assert "closed" in refused(409, enqueue, db, alice, pipeline_id=pid)
        assert counters(db, pid, alice.id) == (0, 0) and job_count(db, alice.id) == 0

    def test_the_daily_budget_spans_pipelines_and_resets_at_utc_midnight(self, db, alice, monkeypatch):
        monkeypatch.setattr(research_jobs, "RESEARCH_PIPELINE_DAILY_JOBS", 2)
        first, second = new_pipeline(db, alice, 5, "a").id, new_pipeline(db, alice, 5, "b").id
        enqueue(db, alice, {"window": 1}, pipeline_id=first)
        enqueue(db, alice, {"window": 2}, pipeline_id=second)
        assert "daily limit" in refused(429, enqueue, db, alice, {"window": 3}, pipeline_id=first)
        assert counters(db, first, alice.id) == (1, 2)
        tomorrow = research_jobs.utcnow() + dt.timedelta(days=1)
        monkeypatch.setattr(research_jobs, "utcnow", lambda: tomorrow)
        enqueue(db, alice, {"window": 3}, pipeline_id=first)
        days = db.execute(select(models.ResearchPipelineDay.jobs_queued)
                          .where(models.ResearchPipelineDay.user_id == alice.id)
                          .order_by(models.ResearchPipelineDay.day)).scalars().all()
        assert days == [2, 1]

    def test_a_daily_budget_of_zero_disables_pipeline_jobs(self, db, alice, monkeypatch):
        monkeypatch.setattr(research_jobs, "RESEARCH_PIPELINE_DAILY_JOBS", 0)
        pid = new_pipeline(db, alice).id
        assert "disabled" in refused(429, enqueue, db, alice, pipeline_id=pid)
        assert counters(db, pid, alice.id) == (0, 0)
        enqueue(db, alice)  # jobs outside a pipeline are unaffected

    def test_the_daily_budget_is_per_user(self, db, alice, bob, monkeypatch):
        monkeypatch.setattr(research_jobs, "RESEARCH_PIPELINE_DAILY_JOBS", 1)
        enqueue(db, alice, pipeline_id=new_pipeline(db, alice).id)
        enqueue(db, bob, pipeline_id=new_pipeline(db, bob).id)

    def test_a_job_the_active_limit_refuses_spends_nothing(self, db, alice, monkeypatch):
        """Only a job that is queued spends budget: one refused because the
        user is at RESEARCH_MAX_ACTIVE_JOBS_PER_USER leaves both counters as
        they were."""
        monkeypatch.setattr(research_jobs, "RESEARCH_MAX_ACTIVE_JOBS_PER_USER", 1)
        pid = new_pipeline(db, alice).id
        enqueue(db, alice, {"window": 1}, pipeline_id=pid)
        with pytest.raises(research_jobs.QueueFull):
            enqueue(db, alice, {"window": 2}, pipeline_id=pid)
        assert counters(db, pid, alice.id) == (1, 1) and job_count(db, alice.id) == 1

    def test_a_spent_job_is_never_handed_back(self, db, alice):
        pid = new_pipeline(db, alice, max_jobs=1).id
        job_id = enqueue(db, alice, pipeline_id=pid, idempotency_key="spent-key-1").id
        assert research_jobs.request_cancel(db, alice.id, job_id).status == "cancelled"
        db.execute(update(models.ResearchJob).where(models.ResearchJob.id == job_id)
                   .values(finished_at=research_jobs.utcnow() - dt.timedelta(days=365)))
        db.commit()
        assert research_jobs.purge_finished(db) >= 1
        db.expire_all()
        assert db.get(models.ResearchJob, job_id) is None
        assert counters(db, pid, alice.id) == (1, 1)
        assert refused(409, enqueue, db, alice, pipeline_id=pid)
        # The purged job freed its key: it now queues a new job (outside the
        # spent pipeline) rather than replaying the old one.
        again = enqueue(db, alice, idempotency_key="spent-key-1")
        assert again.id != job_id

    def test_the_app_role_cannot_delete_budget_rows(self):
        from backend.db_roles import APP_TABLE_PRIVILEGES
        for table in ("research_pipelines", "research_pipeline_days"):
            assert set(APP_TABLE_PRIVILEGES[table]) == {"SELECT", "INSERT", "UPDATE"}

    def test_concurrent_enqueues_never_pass_the_budget(self, Session, db, alice, monkeypatch):
        # Without the in-process lock only the database row locks order the
        # enqueues, as they would across API processes.
        monkeypatch.setattr(research_jobs, "_enqueue_lock", contextlib.nullcontext())
        monkeypatch.setattr(research_jobs, "RESEARCH_MAX_ACTIVE_JOBS_PER_USER", 100)
        pid = new_pipeline(db, alice, max_jobs=3).id
        threads, barrier, outcomes = 12, threading.Barrier(12), []

        def submit(i):
            session = Session()
            try:
                barrier.wait()
                research_jobs.enqueue(session, alice.id, KIND, {"window": i}, pipeline_id=pid)
                outcomes.append("queued")
            except research_pipelines.JobRefused as exc:
                outcomes.append(exc.status_code)
            finally:
                session.close()

        workers = [threading.Thread(target=submit, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(60)
        assert sorted(outcomes, key=str) == sorted(["queued"] * 3 + [409] * 9, key=str)
        assert counters(db, pid, alice.id) == (3, 3) and job_count(db, alice.id) == 3

    def test_a_refusal_releases_the_locks_it_took(self, Session, db, alice):
        """A refused enqueue held the user's and the pipeline's row locks; it
        must let them go at once, not when its session happens to close, or
        every other enqueue of that user waits behind it."""
        pid = new_pipeline(db, alice, max_jobs=1).id
        enqueue(db, alice, {"window": 1}, pipeline_id=pid)
        refusing = Session()
        try:
            refused(409, research_jobs.enqueue, refusing, alice.id, KIND, {"window": 2}, pipeline_id=pid)
            other = Session()
            try:
                # SET LOCAL: the timeout ends with each transaction, so it
                # never follows the connection back into the pool.
                other.execute(text("SET LOCAL lock_timeout = '2s'"))
                assert research_jobs.enqueue(other, alice.id, KIND, {"window": 3}).status == "queued"
                other.execute(text("SET LOCAL lock_timeout = '2s'"))
                other.execute(select(models.ResearchPipeline.id).where(models.ResearchPipeline.id == pid)
                              .with_for_update())
                other.rollback()
            finally:
                other.close()
        finally:
            refusing.close()

    def test_the_budget_is_read_from_the_locked_row_not_a_stale_copy(self, Session, db, alice):
        # A session that loaded the pipeline earlier in its transaction must
        # still see the job another session spent while it waited.
        pid = new_pipeline(db, alice, max_jobs=2).id
        enqueue(db, alice, {"window": 1}, pipeline_id=pid)
        stale = Session()
        try:
            # Held, so the identity map (weakly referenced) keeps this copy.
            loaded = stale.get(models.ResearchPipeline, pid)
            assert loaded.jobs_queued == 1 and stale.in_transaction()
            enqueue(db, alice, {"window": 2}, pipeline_id=pid)               # spends the last job
            assert "all 2 jobs" in refused(409, research_jobs.enqueue, stale, alice.id, KIND,
                                           {"window": 3}, pipeline_id=pid)
        finally:
            stale.close()
        assert counters(db, pid, alice.id) == (2, 2) and job_count(db, alice.id) == 2


# ── idempotency ───────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_the_same_request_again_returns_the_same_job_and_spends_nothing(self, db, alice):
        pid = new_pipeline(db, alice).id
        first = enqueue(db, alice, pipeline_id=pid, idempotency_key="order-0001")
        again = enqueue(db, alice, pipeline_id=pid, idempotency_key="order-0001")
        assert again.id == first.id
        assert counters(db, pid, alice.id) == (1, 1) and job_count(db, alice.id) == 1

    def test_a_replay_returns_the_job_whatever_its_status(self, db, alice, monkeypatch):
        job = enqueue(db, alice, idempotency_key="order-0002")
        research_jobs.request_cancel(db, alice.id, job.id)
        assert enqueue(db, alice, idempotency_key="order-0002").status == "cancelled"
        # Nor do the queue cap or a closed pipeline refuse a replay.
        monkeypatch.setattr(research_jobs, "RESEARCH_MAX_ACTIVE_JOBS_PER_USER", 1)
        pid = new_pipeline(db, alice).id
        queued = enqueue(db, alice, pipeline_id=pid, idempotency_key="order-0003")
        research_pipelines.close(db, alice.id, pid)
        assert enqueue(db, alice, pipeline_id=pid, idempotency_key="order-0003").id == queued.id

    @pytest.mark.parametrize("change", [
        {"params": {"window": 99}},
        {"pipeline": True},
        {"stage": "refine"},
    ])
    def test_a_different_request_under_a_used_key_is_refused(self, db, alice, change):
        pid = new_pipeline(db, alice).id
        enqueue(db, alice, {"window": 20}, pipeline_id=pid, stage="screen", idempotency_key="order-0004")
        other_pid = new_pipeline(db, alice, name="other").id
        kwargs = {"pipeline_id": other_pid if change.get("pipeline") else pid,
                  "stage": change.get("stage", "screen"), "idempotency_key": "order-0004"}
        detail = refused(409, enqueue, db, alice, change.get("params", {"window": 20}), **kwargs)
        assert "Idempotency-Key" in detail
        assert job_count(db, alice.id) == 1 and counters(db, other_pid, alice.id)[0] == 0

    def test_a_different_kind_under_a_used_key_is_refused(self, db, alice):
        research_jobs.enqueue(db, alice.id, KIND, {"window": 20}, idempotency_key="order-0005")
        refused(409, research_jobs.enqueue, db, alice.id, "alpha", {"window": 20},
                idempotency_key="order-0005")

    @pytest.mark.parametrize("key", ["short", "x" * 129, "has space1", "semi;colon", "ünïcode-key", ""])
    def test_an_invalid_key_is_refused(self, db, alice, key):
        assert refused(400, enqueue, db, alice, idempotency_key=key) == "invalid Idempotency-Key"
        assert job_count(db, alice.id) == 0

    def test_keys_are_per_user(self, db, alice, bob):
        assert enqueue(db, alice, idempotency_key="shared-key").id != enqueue(db, bob, idempotency_key="shared-key").id

    def test_concurrent_replays_queue_one_job(self, Session, db, alice, monkeypatch):
        monkeypatch.setattr(research_jobs, "_enqueue_lock", contextlib.nullcontext())
        pid = new_pipeline(db, alice).id
        barrier, ids = threading.Barrier(8), []

        def submit():
            session = Session()
            try:
                barrier.wait()
                ids.append(research_jobs.enqueue(session, alice.id, KIND, {"window": 20}, pipeline_id=pid,
                                                 idempotency_key="racing-key").id)
            finally:
                session.close()

        workers = [threading.Thread(target=submit) for _ in range(8)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(60)
        assert len(ids) == 8 and len(set(ids)) == 1
        assert counters(db, pid, alice.id) == (1, 1)


# ── dedupe and stages ─────────────────────────────────────────────────────────

class TestScoping:
    def test_same_as_dedupes_only_within_one_pipeline(self, db, alice):
        a, b = new_pipeline(db, alice, name="a").id, new_pipeline(db, alice, name="b").id
        params = {"window": 20}
        outside = research_jobs.enqueue(db, alice.id, KIND, params, same_as=("window",))
        in_a = research_jobs.enqueue(db, alice.id, KIND, params, same_as=("window",), pipeline_id=a)
        in_b = research_jobs.enqueue(db, alice.id, KIND, params, same_as=("window",), pipeline_id=b)
        assert len({outside.id, in_a.id, in_b.id}) == 3
        assert research_jobs.enqueue(db, alice.id, KIND, params, same_as=("window",), pipeline_id=a).id == in_a.id
        assert research_jobs.enqueue(db, alice.id, KIND, params, same_as=("window",)).id == outside.id
        assert counters(db, a, alice.id)[0] == 1  # the deduped request spent nothing

    def test_a_stage_needs_a_pipeline_and_a_valid_name(self, db, alice):
        pid = new_pipeline(db, alice).id
        with pytest.raises(ValueError):
            enqueue(db, alice, stage="screen")
        for bad in ("Screen", "1st", "a-b", "x" * 65, ""):
            with pytest.raises(ValueError):
                enqueue(db, alice, pipeline_id=pid, stage=bad)
        view = research_jobs.view(db, enqueue(db, alice, pipeline_id=pid, stage="screen"))
        assert (view["pipeline_id"], view["stage"]) == (pid, "screen")
        assert "pipeline_id" not in research_jobs.view(db, enqueue(db, alice, {"window": 1}))


# ── migration 0018's constraints ──────────────────────────────────────────────

class TestConstraints:
    @pytest.mark.parametrize("values", [
        {"jobs_queued": 4},              # past its budget
        {"jobs_queued": -1},
        {"status": "closed"},            # closed without closed_at
        {"status": "paused"},
        {"name": ""},
    ])
    def test_the_pipeline_row_checks(self, db, alice, values):
        pid = new_pipeline(db, alice).id
        with pytest.raises(IntegrityError):
            db.execute(update(models.ResearchPipeline).where(models.ResearchPipeline.id == pid).values(**values))
            db.commit()
        db.rollback()

    def test_a_stage_without_a_pipeline_is_refused_by_the_database(self, db, alice):
        job = enqueue(db, alice)
        with pytest.raises(IntegrityError):
            db.execute(update(models.ResearchJob).where(models.ResearchJob.id == job.id).values(stage="screen"))
            db.commit()
        db.rollback()

    def test_a_key_is_unique_per_user(self, db, alice):
        job = enqueue(db, alice, idempotency_key="unique-key")
        other = enqueue(db, alice, {"window": 1})
        with pytest.raises(IntegrityError):
            db.execute(update(models.ResearchJob).where(models.ResearchJob.id == other.id)
                       .values(idempotency_key=job.idempotency_key, request_hash=job.request_hash))
            db.commit()
        db.rollback()

    @pytest.mark.parametrize("values", [{"request_hash": None}, {"idempotency_key": None}])
    def test_a_key_and_its_request_hash_are_set_together(self, db, alice, values):
        job = enqueue(db, alice, idempotency_key="paired-key")
        assert job.request_hash is not None
        assert enqueue(db, alice, {"window": 1}).request_hash is None
        with pytest.raises(IntegrityError):
            db.execute(update(models.ResearchJob).where(models.ResearchJob.id == job.id).values(**values))
            db.commit()
        db.rollback()


class TestFingerprint:
    def test_a_json_body_is_compared_canonically(self):
        fp = research_jobs.request_fingerprint
        a = fp("POST", "/api/research/correlation", "", b'{"assets": ["A", "B"], "period": "1y"}')
        assert a == fp("post", "/api/research/correlation", "", b'{ "period":"1y","assets":["A","B"] }')
        assert a != fp("POST", "/api/research/correlation", "", b'{"assets": ["B", "A"], "period": "1y"}')
        assert a != fp("POST", "/api/research/alpha", "", b'{"assets": ["A", "B"], "period": "1y"}')
        assert a != fp("POST", "/api/research/correlation", "x=1", b'{"assets": ["A", "B"], "period": "1y"}')

    def test_a_body_that_is_not_json_is_compared_byte_for_byte(self):
        fp = research_jobs.request_fingerprint
        assert fp("POST", "/p", "", b"\xff\x00") == fp("POST", "/p", "", b"\xff\x00")
        assert fp("POST", "/p", "", b"\xff\x00") != fp("POST", "/p", "", b"\xff\x01")
        assert fp("POST", "/p", "", b"") != fp("POST", "/p", "", b"{}")

    def test_a_job_queued_outside_a_request_is_matched_on_its_kind_and_params(self, db, alice):
        first = enqueue(db, alice, {"window": 3}, idempotency_key="internal-key")
        assert enqueue(db, alice, {"window": 3}, idempotency_key="internal-key").id == first.id
        refused(409, enqueue, db, alice, {"window": 4}, idempotency_key="internal-key")


# ── HTTP headers ──────────────────────────────────────────────────────────────

class TestHeaders:
    @pytest.fixture
    def client(self, Session, alice):
        from fastapi.testclient import TestClient

        def session():
            s = Session()
            try:
                yield s
            finally:
                s.close()
        main.app.dependency_overrides[main.get_current_user] = lambda: alice
        main.app.dependency_overrides[database.get_db] = session
        try:
            yield TestClient(main.app)
        finally:
            main.app.dependency_overrides.pop(main.get_current_user, None)
            main.app.dependency_overrides.pop(database.get_db, None)

    def test_routing_headers_reach_the_queue(self, client, db, alice):
        created = client.post("/api/research/pipelines", json={"name": "http", "max_jobs": 1})
        assert created.status_code == 201, created.text
        pid = created.json()["pipeline_id"]
        headers = {"X-Research-Pipeline": pid, "Idempotency-Key": "http-key-0001"}
        first = client.post("/api/research/correlation", json={}, headers=headers)
        assert first.status_code == 202, first.text
        assert first.json()["pipeline_id"] == pid
        again = client.post("/api/research/correlation", json={}, headers=headers)
        assert again.status_code == 202 and again.json()["job_id"] == first.json()["job_id"]
        spent = client.post("/api/research/correlation", json={}, headers={"X-Research-Pipeline": pid})
        assert spent.status_code == 409, spent.text
        assert client.post("/api/research/correlation", json={},
                           headers={"X-Research-Pipeline": "nope"}).status_code == 404
        assert client.post("/api/research/correlation", json={},
                           headers={"Idempotency-Key": "bad key"}).status_code == 400
        assert client.post("/api/research/correlation", json={}).status_code == 202  # no headers: as before
        assert client.get(f"/api/research/pipelines/{pid}").json()["jobs_queued"] == 1

    def test_a_retry_is_matched_on_what_the_client_sent_not_on_server_state(self, client, db, alice):
        """The stat-test job's params carry the family's trial count, read
        when the request arrives. A trial recorded between a request and its
        retry changes that count, but the retry is still the same request: it
        gets the first job, not a 409."""
        family = "ab" * 32
        request = {"returns": [0.01, -0.02, 0.015, 0.003, -0.004, 0.02], "trial_family": family,
                   "n_bootstrap": 100, "n_permutations": 100}
        headers = {"Idempotency-Key": "stat-retry-0001"}
        first = client.post("/api/research/stat-test", json=request, headers=headers)
        assert first.status_code == 202, first.text
        research_trials.record(db, alice.id, family, [{"window": 5}, {"window": 9}], source="test")
        db.commit()
        again = client.post("/api/research/stat-test", json=request, headers=headers)
        assert again.status_code == 202, again.text
        assert again.json()["job_id"] == first.json()["job_id"]
        # A request that does differ is still refused under the used key.
        changed = client.post("/api/research/stat-test", json={**request, "n_bootstrap": 200}, headers=headers)
        assert changed.status_code == 409, changed.text
        # The same body on another endpoint is another request.
        other = client.post("/api/research/correlation", json={}, headers=headers)
        assert other.status_code == 409, other.text
