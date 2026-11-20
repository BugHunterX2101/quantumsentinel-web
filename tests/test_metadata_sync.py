"""The metadata sync (backend/ai/enrich.py, backend/ai/metadata_fetch.py, migration 0015).

The suite runs the sync against a fake OpenAlex and Crossref on 127.0.0.1
that serve the real answers of api.openalex.org and api.crossref.org kept
under tests/fixtures/metadata, for the papers of the arXiv fixture page
arxiv_page1.xml, or whatever a test scripts. The worker runs each job as it
does in production: the driver in the worker, each lookup in a job process.
"""
import dataclasses
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from psycopg import errors

from backend.ai import child, documents, enrich, metadata_fetch, queue
from backend.ai import settings as ai_settings
from backend.ai.kinds import KINDS
from test_ai_tier import Ai, ai  # noqa: F401 - the ai fixture
from test_arxiv_sync import PROXY_VARIABLES, fixture_page

FIXTURES = Path(__file__).parent / "fixtures" / "metadata"
# Not a real key: only its shape matters here.
TEST_KEY = "test-Key_0123456789"
PAGE = fixture_page("arxiv_page1.xml")
PAPERS = {doc.source_id: doc for doc in PAGE.documents}
WITH_DOI = {doc.source_id for doc in PAGE.documents if doc.doi}


def answer(status: int, headers: dict | None = None, body: bytes = b""):
    return lambda request: (status, headers or {}, body)


def as_json(value) -> bytes:
    return json.dumps(value).encode()


class FakeServices:
    """OpenAlex under /openalex and Crossref under /crossref on 127.0.0.1.
    Each request gets the next item of the script, if any, or else the
    fixture of the work it asks about (404 if there is none)."""

    def __init__(self):
        self.script: list = []
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        by_doi = {}
        for source_id, doc in PAPERS.items():
            name = source_id.replace("/", "_")
            by_doi[("openalex", f"10.48550/arXiv.{source_id}".lower())] = f"openalex_{name}.json"
            if doc.doi:
                by_doi[("crossref", doc.doi.lower())] = f"crossref_{name}.json"
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                parts = urllib.parse.urlsplit(self.path)
                service, _, rest = parts.path.lstrip("/").partition("/")
                doi = urllib.parse.unquote(rest.removeprefix("works/").removeprefix("doi:"))
                request = {"at": time.monotonic(), "service": service, "path": parts.path, "doi": doi,
                           "query": dict(urllib.parse.parse_qsl(parts.query)),
                           "user_agent": self.headers.get("User-Agent"),
                           "authorization": self.headers.get("Authorization")}
                with fake.lock:
                    fake.requests.append(request)
                    item = fake.script.pop(0) if fake.script else None
                if item is None:
                    name = by_doi.get((service, doi.lower()))
                    item = answer(200, {"Content-Type": "application/json"}, (FIXTURES / name).read_bytes()) \
                        if name else answer(404, body=b'{"error": "not found"}')
                status, headers, body = item(request)
                try:
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def serve(self, *items) -> None:
        with self.lock:
            self.script.extend(items)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def services(monkeypatch):
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    server = FakeServices()
    yield server
    server.close()


class Meta:
    def __init__(self, ai: Ai, services: FakeServices):
        self.ai, self.services = ai, services

    def sql(self, statement: str, params=()):
        return self.ai.conn.execute(statement.replace("{ai}", f'"{self.ai.schema}"'), params)

    def worker(self, **environ):
        return self.ai.worker(AI_OPENALEX_URL=f"{self.services.base}/openalex",
                              AI_CROSSREF_URL=f"{self.services.base}/crossref", **environ)

    def run(self, payload: dict, **environ) -> tuple[str, dict]:
        job_id = self.ai.enqueue(payload, kind="metadata_sync")
        runner = self.worker(**environ)
        try:
            ended = runner.run_once()
        finally:
            runner.close()
        return ended, self.ai.job(job_id)

    def rows(self, source: str) -> dict[str, dict]:
        cur = self.sql("SELECT d.source_id, m.* FROM {ai}.document_metadata m"
                       " JOIN {ai}.documents d ON d.id = m.document_id WHERE m.source = %s", (source,))
        names = [column.name for column in cur.description]
        return {row[0]: dict(zip(names, row)) for row in cur.fetchall()}


@pytest.fixture
def meta(ai, services):
    """The papers of arxiv_page1.xml in the registry, and the metadata stage on."""
    with ai.conn.transaction():
        documents.upsert_page(ai.conn, ai.schema, PAGE)
    m = Meta(ai, services)
    m.sql("UPDATE {ai}.sources SET next_allowed_at = now()")
    ai.switch(True, True, "metadata_sync")
    return m


# -- the job process's request ------------------------------------------------------------

@pytest.mark.parametrize("doi", ["10.1/x", "10.1234/a?b", "10.1234/a#b", "10.1234/a b", "10.1234/%2e%2e",
                                 "x10.1234/a", "10.1234/", "10.1234/a\nb", "10.1234/" + "a" * 250, None, 10])
def test_a_doi_that_could_change_the_url_is_refused(doi):
    with pytest.raises(ValueError):
        metadata_fetch.request_url({"source": "openalex", "base_url": "https://api.openalex.org", "doi": doi})


def test_the_doi_is_quoted_whole_so_it_stays_one_path_segment():
    url = metadata_fetch.request_url({"source": "crossref", "base_url": "https://api.crossref.org",
                                      "doi": "10.1234/../../admin;x(1):2"})
    assert url == "https://api.crossref.org/works/10.1234%2F..%2F..%2Fadmin%3Bx%281%29%3A2"
    url = metadata_fetch.request_url({"source": "openalex", "base_url": "https://api.openalex.org",
                                      "doi": "10.48550/arXiv.math/0601001"})
    assert url.startswith("https://api.openalex.org/works/doi:10.48550%2FarXiv.math%2F0601001?select=id,")
    for base in ("ftp://x", "https://x/", "https://x?a=1", "https://x#f", "https://x y", None):
        with pytest.raises(ValueError):
            metadata_fetch.request_url({"source": "openalex", "base_url": base, "doi": "10.1234/a"})
    with pytest.raises(ValueError):
        metadata_fetch.request_url({"source": "datacite", "base_url": "https://x", "doi": "10.1234/a"})


def test_the_api_key_goes_only_to_openalex_in_a_header_it_cannot_break_out_of():
    headers = metadata_fetch.request_headers({"source": "openalex", "api_key": TEST_KEY})
    assert headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert headers["User-Agent"] == "quantumsentinel-ai-harvester/1"
    assert "Authorization" not in metadata_fetch.request_headers({"source": "openalex"})
    with pytest.raises(ValueError, match="only an OpenAlex request"):
        metadata_fetch.request_headers({"source": "crossref", "api_key": TEST_KEY})
    for key in ("short", "a" * 129, "key\r\nX-Evil: 1", "key with space", 12345678, ""):
        with pytest.raises(ValueError) as refused:
            metadata_fetch.request_headers({"source": "openalex", "api_key": key})
        assert key == "" or str(key) not in str(refused.value)


def test_the_answers_are_read_down_to_a_few_bounded_fields():
    work = metadata_fetch.read_work("openalex", (FIXTURES / "openalex_1602.06855.json").read_bytes())
    assert work == {"id": "https://openalex.org/W2280232997", "doi": "https://doi.org/10.1016/j.chaos.2016.02.026",
                    "type": "article", "publication_date": "2016-03-15", "cited_by_count": 42,
                    "is_retracted": False, "venue": "Chaos Solitons & Fractals",
                    "topic": "Statistical Mechanics and Entropy", "field": "Physics and Astronomy"}
    retracted = metadata_fetch.read_work("crossref", (FIXTURES / "crossref_retracted.json").read_bytes())
    assert retracted["update_types"] == ["correction", "retraction"] and retracted["venue"] == "The Lancet"
    assert enrich.clean_work("crossref", retracted, "10.1016/S0140-6736(97)11096-0")[2] is True
    for body in (b"<html>", b"[1, 2]", b"\xff\xfe", b"[" * 100_000):
        with pytest.raises(ValueError):
            metadata_fetch.read_work("crossref", body)


@pytest.mark.parametrize("source,work,asked", [
    ("openalex", {"id": "https://evil.example/W1"}, ""),
    ("openalex", {"id": "https://openalex.org/W1", "cited_by_count": True}, ""),
    ("openalex", {"id": "https://openalex.org/W1", "cited_by_count": -1}, ""),
    ("openalex", {"id": "https://openalex.org/W1", "is_retracted": "no"}, ""),
    ("openalex", {"id": "https://openalex.org/W1", "publication_date": "yesterday"}, ""),
    ("openalex", {"id": "https://openalex.org/W1", "venue": 7}, ""),
    ("crossref", {"doi": "10.1234/other", "licence_urls": [], "update_types": []}, "10.1234/asked"),
    ("crossref", {"doi": "10.1234/a", "licence_urls": ["javascript:alert(1)"], "update_types": []}, "10.1234/a"),
    ("crossref", {"doi": "10.1234/a", "licence_urls": [], "update_types": ["Retraction!"]}, "10.1234/a"),
    ("crossref", {"doi": "10.1234/a", "licence_urls": [], "update_types": [], "published": [99999]}, "10.1234/a"),
    ("crossref", "not a work", "10.1234/a"),
])
def test_the_worker_checks_every_answer_again(source, work, asked):
    with pytest.raises(enrich.WorkError):
        enrich.clean_work(source, work, asked)


def test_the_doi_asked_about():
    assert enrich.lookup_doi("openalex", "2605.02085", None) == "10.48550/arXiv.2605.02085"
    # arXiv's DOIs leave out an old-style identifier's subject class.
    assert enrich.lookup_doi("openalex", "math.GT/0309136", None) == "10.48550/arXiv.math/0309136"
    assert enrich.lookup_doi("crossref", "1602.06855", "10.1016/j.chaos.2016.02.026") == "10.1016/j.chaos.2016.02.026"
    assert enrich.lookup_doi("crossref", "1602.06855", "10.48550/arXiv.1602.06855") is None
    assert enrich.lookup_doi("crossref", "1602.06855", None) is None


@pytest.mark.parametrize("payload", [None, [], {}, {"source": "datacite"}, {"source": "openalex", "url": "x"},
                                     {"source": "openalex", "max_docs": 0}, {"source": "openalex", "max_docs": True},
                                     {"source": "openalex", "max_docs": "5"},
                                     {"source": "openalex", "max_docs": enrich.MAX_DOCS + 1}])
def test_the_payload_is_checked(payload):
    with pytest.raises(ValueError):
        enrich.check_payload(payload)


def test_the_settings_keep_the_key_out_of_reprs_and_the_urls_fixed_in_production(tmp_path):
    base = {"QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL": "postgresql://u:p@h/db"}
    settings = ai_settings.load({**base, "AI_OPENALEX_API_KEY": TEST_KEY})
    assert settings.openalex_api_key == TEST_KEY and TEST_KEY not in repr(settings)
    assert (settings.openalex_url, settings.crossref_url) == ("https://api.openalex.org", "https://api.crossref.org")
    key_file = tmp_path / "key"
    key_file.write_text(TEST_KEY + "\n")
    assert ai_settings.load({**base, "AI_OPENALEX_API_KEY_FILE": str(key_file)}).openalex_api_key == TEST_KEY
    assert ai_settings.load({**base, "AI_OPENALEX_API_KEY": ""}).openalex_api_key is None
    with pytest.raises(ai_settings.SettingsError) as refused:
        ai_settings.load({**base, "AI_OPENALEX_API_KEY": "bad key\nX: 1"})
    assert "bad key" not in str(refused.value)
    for name in ("AI_OPENALEX_URL", "AI_CROSSREF_URL"):
        with pytest.raises(ai_settings.SettingsError, match="may not be changed in production"):
            ai_settings.load({**base, "ENVIRONMENT": "production", name: "http://127.0.0.1:9"})
        with pytest.raises(ai_settings.SettingsError):
            ai_settings.load({**base, name: "https://x/?a=1"})


def test_a_job_process_never_inherits_the_key():
    environ = {"AI_OPENALEX_API_KEY": TEST_KEY, "AI_OPENALEX_API_KEY_FILE": "/run/k", "PATH": "/usr/bin"}
    assert not any("OPENALEX" in name for name in child.child_environment(environ))


def test_the_kind_runs_one_lookup_per_job_process():
    spec = KINDS["metadata_sync"]
    assert (spec.target, spec.driver, spec.stage) == ("backend.ai.metadata_fetch:fetch_work",
                                                      "backend.ai.enrich:run", "metadata_sync")
    assert spec.child_timeout_seconds == 60 and spec.child_output_max_bytes == 64 * 1024


def test_fetch_work_follows_no_redirect_and_reports_the_longer_wait(services):
    def fetch():
        return metadata_fetch.fetch_work({"source": "openalex", "base_url": f"{services.base}/openalex",
                                          "doi": "10.48550/arXiv.2605.02085", "timeout_seconds": 5})
    services.serve(answer(302, {"Location": f"{services.base}/openalex/works/doi:10.48550%2FarXiv.1602.06855"}),
                   answer(429, {"Retry-After": "30", "X-RateLimit-Reset": "4000"}),
                   answer(503, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
                   answer(200, body=b"x" * (metadata_fetch.MAX_BYTES + 1)),
                   answer(200, body=b"<html>"))
    assert fetch() == {"http_status": 302, "retry_after": None}
    assert fetch() == {"http_status": 429, "retry_after": 4000}
    assert fetch() == {"http_status": 503, "retry_after": None}
    assert fetch()["unreadable"] == f"the answer is larger than {metadata_fetch.MAX_BYTES} bytes"
    assert fetch() == {"http_status": 200, "unreadable": "the answer is not JSON"}
    assert len(services.requests) == 5
    assert fetch()["work"]["id"] == "https://openalex.org/W7125608781"


# -- the sync, against the fake services ------------------------------------------------------

def test_an_openalex_sync_looks_up_every_paper_and_records_what_it_says(meta, services):
    ended, job = meta.run({"source": "openalex"}, AI_OPENALEX_API_KEY=TEST_KEY)
    assert ended == "succeeded", job
    result = job["result"]
    assert (result["documents"], result["found"], result["changed"], result["looked_up"]) == (6, 6, 6, 6)
    assert result["caught_up"] is True and result["rate_limited"] is False
    assert [r["service"] for r in services.requests] == ["openalex"] * 6
    assert all(r["authorization"] == f"Bearer {TEST_KEY}" and r["user_agent"] == "quantumsentinel-ai-harvester/1"
               and r["query"]["select"] == metadata_fetch.OPENALEX_SELECT for r in services.requests)
    # The key travels in the header alone, never in the URL.
    assert not any(TEST_KEY in r["path"] or TEST_KEY in json.dumps(r["query"]) for r in services.requests)
    # newest first
    first = services.requests[0]["doi"].removeprefix("10.48550/arXiv.")
    assert PAPERS[first].datestamp == max(doc.datestamp for doc in PAPERS.values())
    gaps = [b["at"] - a["at"] for a, b in zip(services.requests, services.requests[1:])]
    assert min(gaps) >= 0.9  # one request a second
    rows = meta.rows("openalex")
    assert set(rows) == set(PAPERS)
    chaos = rows["1602.06855"]
    assert (chaos["status"], chaos["external_id"], chaos["cited_by_count"], chaos["is_retracted"]) == \
        ("found", "W2280232997", 42, False)
    assert chaos["lookup_doi"] == "10.48550/arXiv.1602.06855" and chaos["checks"] == 1
    assert chaos["fields"]["doi"] == "https://doi.org/10.1016/j.chaos.2016.02.026"
    # The key is in no row the database keeps.
    dump = meta.sql("SELECT string_agg(j::text, ' ') FROM {ai}.jobs j").fetchone()[0] \
        + meta.sql("SELECT string_agg(m::text, ' ') FROM {ai}.document_metadata m").fetchone()[0]
    assert TEST_KEY not in dump
    # Nothing is due now: the next sync asks nothing.
    services.requests.clear()
    ended, job = meta.run({"source": "openalex"})
    assert ended == "succeeded" and job["result"]["documents"] == 0 and job["result"]["caught_up"] is True
    assert services.requests == []


def test_a_crossref_sync_asks_only_about_journal_dois_and_never_sends_the_key(meta, services):
    ended, job = meta.run({"source": "crossref"}, AI_OPENALEX_API_KEY=TEST_KEY)
    assert ended == "succeeded" and job["result"]["found"] == 3, job
    assert all(r["service"] == "crossref" and r["authorization"] is None and r["query"] == {}
               for r in services.requests)
    assert {r["doi"] for r in services.requests} == {PAPERS[s].doi for s in WITH_DOI}
    rows = meta.rows("crossref")
    assert set(rows) == WITH_DOI
    wsc = rows["2605.02085"]
    assert (wsc["external_id"], wsc["is_retracted"], wsc["fields"]["type"]) == \
        ("10.1109/wsc68292.2025.11339108", False, "proceedings-article")


def test_each_answer_is_recorded_for_what_it_is_and_failures_are_retried(meta, services, monkeypatch):
    monkeypatch.setattr(enrich, "RETRY_SECONDS", 1)
    services.serve(answer(404), answer(200, body=b"<html>"), answer(400),
                   answer(503), answer(429, {"Retry-After": "1"}))
    ended, job = meta.run({"source": "openalex"})
    assert ended == "succeeded", job
    result = job["result"]
    assert (result["found"], result["not_found"], result["refused"], result["retries"], result["looked_up"]) == \
        (3, 1, 2, 2, 6)
    assert len(services.requests) == 8
    statuses = {row["status"]: row for row in meta.rows("openalex").values()}
    assert statuses["not_found"]["reason"] is None and statuses["not_found"]["external_id"] is None
    reasons = sorted(row["reason"] for row in meta.rows("openalex").values() if row["status"] == "refused")
    assert reasons == ["its answer is not a work: the answer is not JSON", "openalex refused the DOI (HTTP 400)"]


def test_an_answer_that_fails_the_checks_is_refused_not_stored(meta, services):
    services.serve(answer(200, body=as_json({"id": "https://evil.example/W1", "cited_by_count": 5})))
    ended, job = meta.run({"source": "openalex", "max_docs": 1})
    assert ended == "succeeded" and job["result"]["caught_up"] is False
    (row,) = meta.rows("openalex").values()
    assert row["status"] == "refused" and row["cited_by_count"] is None
    assert row["reason"] == "its answer did not pass the checks: it has no OpenAlex work id"


def test_a_spent_daily_allowance_ends_the_sync_and_holds_the_source_back(meta, services):
    services.serve(answer(200, body=(FIXTURES / "openalex_2605.02085.json").read_bytes()),
                   answer(429, {"X-RateLimit-Reset": "7200"}))
    ended, job = meta.run({"source": "openalex"})
    assert ended == "succeeded" and job["result"]["rate_limited"] is True and job["result"]["documents"] == 1
    wait = meta.sql("SELECT extract(epoch FROM next_allowed_at - now())::float8 FROM {ai}.sources"
                    " WHERE name = 'openalex'").fetchone()[0]
    assert 7100 < wait <= 7200
    assert meta.sql("SELECT next_allowed_at <= now() + interval '2 s' FROM {ai}.sources"
                    " WHERE name = 'crossref'").fetchone()[0]


def test_an_unexpected_status_fails_the_job(meta, services):
    services.serve(answer(403))
    ended, job = meta.run({"source": "openalex"})
    assert ended == "failed" and job["error"] == "The openalex sync failed: openalex answered HTTP 403"
    assert meta.rows("openalex") == {}


def test_too_many_failures_in_a_row_fail_the_job(meta, services, monkeypatch):
    monkeypatch.setattr(enrich, "RETRY_SECONDS", 1)
    monkeypatch.setattr(enrich, "MAX_RETRIES", 2)
    services.serve(answer(500), answer(502), answer(504))
    ended, job = meta.run({"source": "openalex"})
    assert ended == "failed" and job["error"].endswith("openalex answered HTTP 504 (2 retries in a row)")


def test_a_worker_that_lost_its_job_records_nothing(meta, services):
    def steal(request):
        meta.sql("UPDATE {ai}.jobs SET worker_id = 'thief' WHERE kind = 'metadata_sync'")
        return 200, {}, (FIXTURES / "openalex_2605.02085.json").read_bytes()

    services.serve(steal)
    ended, job = meta.run({"source": "openalex"})
    assert ended == "lost (lease lapsed)" and job["status"] == "running"
    assert meta.rows("openalex") == {}


def test_due_papers_are_looked_up_again_and_unchanged_answers_change_nothing(meta, services):
    assert meta.run({"source": "openalex"})[0] == "succeeded"
    # The papers last changed long before any lookup, so only age makes one due.
    meta.sql("UPDATE {ai}.documents SET changed_at = now() - interval '90 days'")
    meta.sql("UPDATE {ai}.document_metadata SET checked_at = now() - interval '31 days'"
             " WHERE document_id = (SELECT id FROM {ai}.documents WHERE source_id = '0709.4467')")
    meta.sql("UPDATE {ai}.documents SET changed_at = now() WHERE source_id = '2509.18837'")
    before = meta.rows("openalex")
    services.requests.clear()
    ended, job = meta.run({"source": "openalex"})
    assert ended == "succeeded" and (job["result"]["documents"], job["result"]["changed"]) == (2, 0)
    assert sorted(r["doi"] for r in services.requests) == ["10.48550/arXiv.0709.4467", "10.48550/arXiv.2509.18837"]
    after = meta.rows("openalex")
    assert after["0709.4467"]["checks"] == 2 and after["0709.4467"]["changed_at"] == before["0709.4467"]["changed_at"]
    assert after["1602.06855"] == before["1602.06855"]
    # A not_found paper is asked again after RECHECK_DAYS, a found one not before FOUND_RECHECK_DAYS.
    meta.sql("UPDATE {ai}.document_metadata SET status = 'not_found', external_id = NULL, cited_by_count = NULL,"
             " is_retracted = NULL, fields = '{}', checked_at = now() - interval '4 days'"
             " WHERE document_id = (SELECT id FROM {ai}.documents WHERE source_id = '1602.06855')")
    meta.sql("UPDATE {ai}.document_metadata SET checked_at = now() - interval '29 days'"
             " WHERE document_id = (SELECT id FROM {ai}.documents WHERE source_id = '2605.02085')")
    services.requests.clear()
    ended, job = meta.run({"source": "openalex"})
    assert [r["doi"] for r in services.requests] == ["10.48550/arXiv.1602.06855"]
    assert job["result"]["changed"] == 1 and meta.rows("openalex")["1602.06855"]["status"] == "found"


def test_a_sync_stops_at_the_kinds_deadline_and_the_next_one_carries_on(meta, services, monkeypatch):
    """One request a second and a one-second deadline: the sync ends before
    its next request would; the next sync asks only about the rest."""
    short = dataclasses.replace(KINDS["metadata_sync"], timeout_seconds=1)
    monkeypatch.setitem(KINDS, "metadata_sync", short)
    ended, job = meta.run({"source": "openalex"})
    assert ended == "succeeded", job
    first = job["result"]["documents"]
    assert job["result"]["caught_up"] is False and 1 <= first < len(PAPERS)
    asked = [r["doi"] for r in services.requests]
    monkeypatch.setitem(KINDS, "metadata_sync", dataclasses.replace(short, timeout_seconds=3600))
    ended, job = meta.run({"source": "openalex"})
    assert ended == "succeeded" and job["result"]["caught_up"] is True
    assert job["result"]["documents"] == len(PAPERS) - first
    later = [r["doi"] for r in services.requests][len(asked):]
    assert not set(asked) & set(later) and len(meta.rows("openalex")) == len(PAPERS)


def test_switching_the_stage_off_hands_the_sync_back_and_it_resumes(meta, services):
    def switch_off(request):
        meta.ai.flag("metadata_sync", False)
        time.sleep(2)  # meanwhile the worker checks in, and stops
        return answer(503)(request)

    services.serve(switch_off)
    ended, job = meta.run({"source": "openalex"})
    assert ended == "released (stage switched off)"
    assert (job["status"], job["attempts"]) == ("queued", 0)
    assert meta.rows("openalex") == {}
    meta.ai.flag("metadata_sync", True)
    runner = meta.worker()
    try:
        assert runner.run_once() == "succeeded"
    finally:
        runner.close()
    assert meta.ai.job(job["id"])["result"]["found"] == len(PAPERS)
    assert len(meta.rows("openalex")) == len(PAPERS)


def test_a_doi_that_cannot_be_asked_about_is_refused_without_a_request(meta, services):
    meta.sql("UPDATE {ai}.documents SET doi = '10.1234/a b' WHERE source_id = '0709.4467'")
    ended, job = meta.run({"source": "crossref"})
    assert ended == "succeeded" and (job["result"]["found"], job["result"]["refused"]) == (2, 1)
    assert len(services.requests) == 2
    row = meta.rows("crossref")["0709.4467"]
    assert (row["status"], row["lookup_doi"], row["reason"]) == \
        ("refused", None, "its DOI is not one crossref can be asked about")


def test_one_sync_per_service_at_a_time(meta):
    meta.ai.enqueue({"source": "openalex"}, kind="metadata_sync")
    with pytest.raises(queue.EnqueueError, match="already queued or running"):
        meta.ai.enqueue({"source": "openalex", "max_docs": 5}, kind="metadata_sync")
    meta.ai.enqueue({"source": "crossref"}, kind="metadata_sync")


@pytest.mark.parametrize("values", ["'found', '10.1/x1', 'why', NULL", "'not_found', '10.1/x1', NULL, 'W1'",
                                    "'refused', NULL, NULL, NULL", "'not_found', NULL, NULL, NULL"])
def test_the_table_refuses_rows_that_contradict_their_status(meta, values):
    doc_id = meta.sql("SELECT id FROM {ai}.documents LIMIT 1").fetchone()[0]
    with pytest.raises(errors.CheckViolation):
        meta.sql("INSERT INTO {ai}.document_metadata (document_id, source, status, lookup_doi, reason,"
                 f" external_id) VALUES (%s, 'openalex', {values})", (doc_id,))


@pytest.mark.skipif(os.environ.get("QS_LIVE_METADATA") != "1", reason="set QS_LIVE_METADATA=1 to ask the real services")
def test_live_lookups_at_openalex_and_crossref(ai):
    """The papers of arxiv_page1.xml, looked up at the real OpenAlex (with
    AI_OPENALEX_API_KEY, if set) and Crossref: nine requests in all."""
    with ai.conn.transaction():
        documents.upsert_page(ai.conn, ai.schema, PAGE)
    ai.switch(True, True, "metadata_sync")
    key = {"AI_OPENALEX_API_KEY": os.environ["AI_OPENALEX_API_KEY"]} if os.environ.get("AI_OPENALEX_API_KEY") else {}
    for source, papers in (("openalex", set(PAPERS)), ("crossref", WITH_DOI)):
        job_id = ai.enqueue({"source": source}, kind="metadata_sync")
        runner = ai.worker(**key)
        try:
            assert runner.run_once() == "succeeded"
        finally:
            runner.close()
        result = ai.job(job_id)["result"]
        assert result["caught_up"] is True and result["documents"] == len(papers), result
        rows = Meta(ai, None).rows(source)
        assert set(rows) == papers
        found = [row for row in rows.values() if row["status"] == "found"]
        assert len(found) >= len(papers) - 1, rows
        assert all(row["cited_by_count"] is not None for row in found)
