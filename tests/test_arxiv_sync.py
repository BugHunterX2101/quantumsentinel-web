"""The arXiv sync (backend/ai/harvest.py, backend/ai/oai_fetch.py, migration 0014).

The suite runs the sync against a fake OAI-PMH server on 127.0.0.1 that
serves the real responses of https://oaipmh.arxiv.org/oai kept under
tests/fixtures/arxiv_oai, with their responseDate and token expiry moved to
now, or whatever a test scripts (errors, Retry-After, redirects, dropped
connections). The worker runs each job as it does in production: the driver
in the worker, each request in a job process.

A live test against arXiv itself runs only with QS_LIVE_ARXIV=1. It keeps
arXiv's rate limit (one request every 3 s) and fetches at most a few pages.
"""
import dataclasses
import datetime as dt
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from backend.ai import documents, harvest, oai, oai_fetch, queue
from backend.ai import settings as ai_settings
from backend.ai.kinds import KINDS
from test_ai_tier import OPS, Ai, ai  # noqa: F401 - the ai fixture

OAI_FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_oai"
TOKEN_1 = ("verb%3DListRecords%26metadataPrefix%3DarXiv%26from%3D2026-06-17%26until%3D2026-10-31"
           "%26set%3Dq-fin%26skip%3D14")
TOKEN_2 = ("verb%3DListRecords%26metadataPrefix%3DarXiv%26from%3D2026-09-15%26until%3D2026-10-31"
           "%26set%3Dq-fin%26skip%3D32")
PAGES = ("arxiv_page1.xml", "arxiv_page2.xml", "arxiv_last_page.xml")
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
DROP = "drop"  # a script item: close the connection without an answer


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def today() -> dt.date:
    return utcnow().date()


def days_ago(days: int) -> dt.date:
    return today() - dt.timedelta(days=days)


def dated(name: str, now: dt.datetime | None = None) -> bytes:
    """A fixture as arXiv would send it at ``now``: its responseDate is
    ``now`` and its resumption token expires a day later."""
    now = now or utcnow()
    text = (OAI_FIXTURES / name).read_text("utf-8")
    text = re.sub(r"<responseDate>[^<]*</responseDate>",
                  f"<responseDate>{now:%Y-%m-%dT%H:%M:%SZ}</responseDate>", text)
    text = re.sub(r"expirationDate='[^']*'",
                  f"expirationDate='{now + dt.timedelta(days=1):%Y-%m-%dT%H:%M:%SZ}'", text)
    return text.encode("utf-8")


def answer(status: int, headers: dict | None = None, body: bytes = b""):
    return lambda request: (status, headers or {}, body)


def xml(name: str, now: dt.datetime | None = None):
    return lambda request: (200, {"Content-Type": "text/xml"}, dated(name, now))


def error_page(code: str) -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f"<responseDate>{utcnow():%Y-%m-%dT%H:%M:%SZ}</responseDate><request>x</request>"
            f"<error code='{code}'>Something is wrong.</error></OAI-PMH>").encode()


class FakeOai:
    """An OAI-PMH server on 127.0.0.1 that answers each request with the
    next item of its script and records every request it receives."""

    def __init__(self):
        self.script: list = []
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                parts = urllib.parse.urlsplit(self.path)
                request = {"at": time.monotonic(), "netloc": parts.netloc, "path": parts.path,
                           "query": dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True)),
                           "user_agent": self.headers.get("User-Agent")}
                with fake.lock:
                    fake.requests.append(request)
                    item = fake.script.pop(0) if fake.script else answer(418, body=b"the script is exhausted")
                if item == DROP:
                    self.close_connection = True
                    return
                if isinstance(item, str):
                    item = xml(item)
                status, headers, body = item(request)
                try:
                    self.send_response(status)
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # the job process was killed while this answer was on its way

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.address = f"127.0.0.1:{self.server.server_address[1]}"
        self.url = f"http://{self.address}/oai"

    def serve(self, *items) -> None:
        with self.lock:
            self.script.extend(items)

    def queries(self) -> list[dict]:
        return [request["query"] for request in self.requests]

    def gaps(self) -> list[float]:
        times = [request["at"] for request in self.requests]
        return [later - earlier for earlier, later in zip(times, times[1:])]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake(monkeypatch):
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    server = FakeOai()
    yield server
    server.close()


def window(set_spec: str, start: dt.date, until: dt.date) -> dict:
    return {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec,
            "from": start.isoformat(), "until": until.isoformat()}


def resume(token: str) -> dict:
    return {"verb": "ListRecords", "resumptionToken": token}


class Harvester:
    def __init__(self, ai: Ai, url: str):
        self.ai, self.url = ai, url

    def sql(self, statement: str, params=()):
        return self.ai.conn.execute(statement.replace("{ai}", f'"{self.ai.schema}"'), params)

    def worker(self):
        return self.ai.worker(AI_ARXIV_OAI_URL=self.url)

    def run(self, payload: dict) -> tuple[str, dict]:
        """Queue an arxiv_sync job and run it with a worker of its own."""
        job_id = self.ai.enqueue(payload, kind="arxiv_sync")
        runner = self.worker()
        try:
            ended = runner.run_once()
        finally:
            runner.close()
        return ended, self.ai.job(job_id)

    def cursor(self, set_spec: str = "q-fin") -> dict | None:
        cur = self.sql("SELECT * FROM {ai}.harvests WHERE source = 'arxiv' AND set_spec = %s", (set_spec,))
        row = cur.fetchone()
        return None if row is None else dict(zip([column.name for column in cur.description], row))

    def documents(self) -> int:
        return self.sql("SELECT count(*) FROM {ai}.documents").fetchone()[0]


@pytest.fixture
def sync(ai, fake):
    """Harvests against the fake server, with arXiv's 3 s between requests
    cut to 1 s."""
    harvester = Harvester(ai, fake.url)
    harvester.sql("UPDATE {ai}.sources SET min_interval_seconds = 1, next_allowed_at = now()")
    ai.switch(True, True, "arxiv_sync")
    return harvester


def fixture_page(name: str) -> documents.Page:
    return documents.clean_page(oai.parse_list_records(dated(name)))


# -- the payload and what a job process reports ------------------------------------

def test_a_payload_names_a_set_and_optionally_where_to_start_and_how_far_to_go():
    day = dt.date(2026, 11, 20)
    assert harvest.check_payload({"set": "q-fin"}, day) == ("q-fin", None, None)
    assert harvest.check_payload({"set": "physics:hep-th", "from": "2026-01-01", "max_pages": 7}, day) \
        == ("physics:hep-th", dt.date(2026, 1, 1), 7)
    assert harvest.check_payload({"set": "q-fin", "from": "1991-01-01"}, day)[1] == harvest.EARLIEST
    assert harvest.check_payload({"set": "q-fin", "from": "2026-11-20"}, day)[1] == day


@pytest.mark.parametrize("payload", [
    None, [], "q-fin", {}, {"set": "q-fin", "url": "http://example.com/oai"},
    {"set": "q fin"}, {"set": "../q-fin"}, {"set": ""}, {"set": 7}, {"set": "q" * 129}, {"set": "q-fin;x"},
    {"set": "q-fin", "from": "2026-13-01"}, {"set": "q-fin", "from": "2026-1-1"},
    {"set": "q-fin", "from": 20261101}, {"set": "q-fin", "from": "2026-11-21"},
    {"set": "q-fin", "from": "1990-12-31"}, {"set": "q-fin", "from": "2026-02-30"},
    {"set": "q-fin", "max_pages": 0}, {"set": "q-fin", "max_pages": harvest.MAX_PAGES + 1},
    {"set": "q-fin", "max_pages": True}, {"set": "q-fin", "max_pages": "5"}, {"set": "q-fin", "max_pages": 1.5},
])
def test_a_payload_that_is_not_one_is_refused(payload):
    with pytest.raises(ValueError):
        harvest.check_payload(payload, dt.date(2026, 11, 20))


def test_the_status_a_job_process_reports_is_checked():
    assert harvest.response_status({"http_status": 200, "page": {}}) == (200, None)
    assert harvest.response_status({"http_status": 503, "retry_after": 20}) == (503, 20)
    for retry_after in (-1, True, "20", 1.5):
        assert harvest.response_status({"http_status": 503, "retry_after": retry_after}) == (503, None)
    for out in (None, [], {}, {"http_status": True}, {"http_status": "200"}, {"http_status": 99},
                {"http_status": 600}, {"http_status": 200.0}):
        with pytest.raises(harvest.SyncError):
            harvest.response_status(out)


# -- the job process's request -----------------------------------------------------

def test_a_request_is_one_listrecords_get_and_nothing_else():
    url = oai_fetch.request_url({"base_url": "https://oaipmh.arxiv.org/oai", "params": resume("a%3Db")})
    assert url == "https://oaipmh.arxiv.org/oai?resumptionToken=a%253Db&verb=ListRecords"
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["resumptionToken"] == ["a%3Db"]
    good = window("q-fin", dt.date(2026, 11, 1), dt.date(2026, 11, 20))
    for payload in (
        None, {"params": good}, {"base_url": "ftp://x/oai", "params": good},
        {"base_url": "file:///etc/passwd", "params": good},
        {"base_url": "https://x/oai?verb=Identify", "params": good}, {"base_url": "https://x/oai#a", "params": good},
        {"base_url": "https://x/ oai", "params": good}, {"base_url": "https://x/oai", "params": {}},
        {"base_url": "https://x/oai", "params": {**good, "verb": "Identify"}},
        {"base_url": "https://x/oai", "params": {"set": "q-fin"}},
        {"base_url": "https://x/oai", "params": {**good, "extra": "1"}},
        {"base_url": "https://x/oai", "params": {**good, "set": "q fin"}},
        {"base_url": "https://x/oai", "params": {**good, "set": ""}},
        {"base_url": "https://x/oai", "params": {**good, "set": 7}},
        {"base_url": "https://x/oai", "params": {**good, "set": "x" * 1025}},
    ):
        with pytest.raises(ValueError):
            oai_fetch.request_url(payload)


def test_the_job_process_fetches_and_parses_a_page(fake):
    fake.serve("arxiv_page1.xml")
    params = window("q-fin", dt.date(2026, 11, 1), dt.date(2026, 11, 20))
    out = oai_fetch.fetch_page({"base_url": fake.url, "params": params})
    assert out["http_status"] == 200 and out["page"]["resumption_token"] == TOKEN_1
    assert len(out["page"]["records"]) == 6
    request, = fake.requests
    assert (request["netloc"], request["path"], request["query"]) == ("", "/oai", params)
    assert request["user_agent"] == oai_fetch.USER_AGENT == "quantumsentinel-ai-harvester/1"


def test_the_job_process_reports_any_other_status_and_follows_no_redirect(fake):
    payload = {"base_url": fake.url, "params": resume("t")}
    fake.serve(answer(503, {"Retry-After": "120"}), answer(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
               answer(302, {"Location": fake.url + "?verb=Identify"}), answer(404), answer(204))
    assert oai_fetch.fetch_page(payload) == {"http_status": 503, "retry_after": 120}
    assert oai_fetch.fetch_page(payload) == {"http_status": 429, "retry_after": None}
    assert oai_fetch.fetch_page(payload) == {"http_status": 302, "retry_after": None}
    assert oai_fetch.fetch_page(payload) == {"http_status": 404, "retry_after": None}
    assert oai_fetch.fetch_page(payload) == {"http_status": 204, "retry_after": None}
    assert len(fake.requests) == 5  # the redirect was not followed


def test_the_job_process_refuses_a_page_that_is_too_large(fake, monkeypatch):
    monkeypatch.setattr(oai, "MAX_BYTES", 1000)
    fake.serve("arxiv_page1.xml")
    with pytest.raises(oai.OaiError, match="larger than 1000 bytes"):
        oai_fetch.fetch_page({"base_url": fake.url, "params": resume("t")})
    for timeout in (0, 121, True, "60"):
        with pytest.raises(ValueError, match="timeout_seconds"):
            oai_fetch.fetch_page({"base_url": fake.url, "params": resume("t"), "timeout_seconds": timeout})


def test_the_job_process_takes_its_proxy_from_the_environment_only(fake, monkeypatch):
    # The system's proxy settings (the Windows registry, macOS's
    # configuration) are never consulted.
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": "http://127.0.0.1:1"})
    fake.serve("arxiv_last_page.xml", "arxiv_last_page.xml")
    assert oai_fetch.fetch_page({"base_url": fake.url, "params": resume("t")})["http_status"] == 200
    assert fake.requests[-1]["netloc"] == ""  # straight to the server
    monkeypatch.setenv("HTTP_PROXY", f"http://{fake.address}")
    out = oai_fetch.fetch_page({"base_url": "http://oaipmh.arxiv.invalid/oai", "params": resume("t")})
    assert out["http_status"] == 200
    assert fake.requests[-1]["netloc"] == "oaipmh.arxiv.invalid"  # through the proxy


# -- whole syncs ---------------------------------------------------------------------

def test_a_sync_walks_its_window_page_by_page_and_a_second_sync_changes_nothing(sync, fake):
    start = days_ago(10)
    fake.serve(*PAGES)
    ended, job = sync.run({"set": "q-fin", "from": start.isoformat()})
    assert ended == "succeeded", job
    assert fake.queries() == [window("q-fin", start, today()), resume(TOKEN_1), resume(TOKEN_2)]
    assert all(request["user_agent"] == oai_fetch.USER_AGENT for request in fake.requests)
    assert min(fake.gaps()) >= 0.95  # one request a second, as the sources row says
    expected = [fixture_page(name) for name in PAGES]
    records = sum(len(page.documents) + len(page.deletions) for page in expected)
    papers = {doc.source_id for page in expected for doc in page.documents}
    result = job["result"]
    assert (result["pages"], result["windows_completed"], result["caught_up"], result["retries"]) == (3, 1, True, 0)
    assert result["last_complete_until"] == today().isoformat()
    assert result["inserted"] == len(papers) == sync.documents() and result["rejected"] == 0
    assert result["records"] + result["deletions"] == records
    cursor = sync.cursor()
    assert (cursor["start_date"], cursor["last_complete_until"]) == (start, today())
    assert (cursor["window_from"], cursor["window_until"], cursor["resumption_token"]) == (None, None, None)
    assert (cursor["pages"], cursor["records"], cursor["last_error"]) == (3, records, None)

    # The next sync starts on the last day it completed; arXiv lists the
    # same papers again, and recording them again changes nothing.
    fake.requests.clear()
    fake.serve("arxiv_last_page.xml")
    ended, job = sync.run({"set": "q-fin", "from": "2001-01-01"})  # ``from`` is ignored now
    assert ended == "succeeded", job
    assert fake.queries() == [window("q-fin", today(), today())]
    assert (job["result"]["inserted"], job["result"]["updated"], job["result"]["unchanged"]) \
        == (0, 0, len(fixture_page("arxiv_last_page.xml").documents))
    assert sync.documents() == len(papers)
    assert sync.cursor()["start_date"] == start


def test_a_long_span_is_walked_in_windows(sync, fake):
    start = days_ago(harvest.WINDOW_DAYS + 9)
    middle = start + dt.timedelta(days=harvest.WINDOW_DAYS)
    fake.serve("arxiv_last_page.xml", "arxiv_no_records.xml")
    ended, job = sync.run({"set": "q-fin", "from": start.isoformat()})
    assert ended == "succeeded", job
    assert fake.queries() == [window("q-fin", start, middle), window("q-fin", middle, today())]
    assert (job["result"]["windows_completed"], job["result"]["caught_up"]) == (2, True)
    assert sync.cursor()["last_complete_until"] == today()


def test_an_empty_window_completes(sync, fake):
    fake.serve("arxiv_no_records.xml")
    ended, job = sync.run({"set": "q-fin", "from": days_ago(3).isoformat()})
    assert ended == "succeeded", job
    assert (job["result"]["pages"], job["result"]["records"], job["result"]["caught_up"]) == (1, 0, True)
    assert sync.cursor()["last_complete_until"] == today()


def test_the_cursor_never_passes_the_day_arxiv_says_it_is(sync, fake):
    yesterday = utcnow() - dt.timedelta(days=1)
    fake.serve(xml("arxiv_last_page.xml", now=yesterday))
    ended, job = sync.run({"set": "q-fin", "from": days_ago(5).isoformat()})
    assert ended == "succeeded" and job["result"]["caught_up"] is True, job
    assert sync.cursor()["last_complete_until"] == yesterday.date()


def test_max_pages_stops_a_sync_and_the_next_one_resumes_with_the_token(sync, fake):
    start = days_ago(10)
    fake.serve("arxiv_page1.xml")
    ended, job = sync.run({"set": "q-fin", "from": start.isoformat(), "max_pages": 1})
    assert ended == "succeeded", job
    assert (job["result"]["pages"], job["result"]["caught_up"], job["result"]["last_complete_until"]) \
        == (1, False, None)
    cursor = sync.cursor()
    assert (cursor["window_from"], cursor["window_until"], cursor["resumption_token"]) == (start, today(), TOKEN_1)
    assert cursor["token_expires"] > utcnow()
    fake.serve("arxiv_page2.xml", "arxiv_last_page.xml")
    ended, job = sync.run({"set": "q-fin"})
    assert ended == "succeeded", job
    assert fake.queries()[1:] == [resume(TOKEN_1), resume(TOKEN_2)]
    assert job["result"]["caught_up"] is True and sync.cursor()["resumption_token"] is None


def test_an_expired_token_restarts_its_window(sync, fake):
    start = days_ago(10)
    fake.serve("arxiv_page1.xml")
    assert sync.run({"set": "q-fin", "from": start.isoformat(), "max_pages": 1})[0] == "succeeded"
    sync.sql("UPDATE {ai}.harvests SET token_expires = now() - interval '1 minute'")
    fake.serve(*PAGES)
    ended, job = sync.run({"set": "q-fin"})
    assert ended == "succeeded", job
    assert fake.queries()[1:] == [window("q-fin", start, today()), resume(TOKEN_1), resume(TOKEN_2)]


def test_a_refused_token_restarts_its_window_but_not_forever(sync, fake, monkeypatch):
    start = days_ago(10)
    fake.serve("arxiv_page1.xml", "arxiv_bad_token.xml", *PAGES)
    ended, job = sync.run({"set": "q-fin", "from": start.isoformat()})
    assert ended == "succeeded", job
    assert fake.queries() == [window("q-fin", start, today()), resume(TOKEN_1),
                              window("q-fin", start, today()), resume(TOKEN_1), resume(TOKEN_2)]
    assert job["result"]["pages"] == 4  # the refusal is not a page

    monkeypatch.setattr(harvest, "MAX_TOKEN_RESTARTS", 1)
    sync.sql("UPDATE {ai}.harvests SET last_complete_until = NULL")
    fake.requests.clear()
    fake.serve("arxiv_page1.xml", "arxiv_bad_token.xml", "arxiv_page1.xml", "arxiv_bad_token.xml")
    ended, job = sync.run({"set": "q-fin"})
    assert ended == "failed" and "refused this window's resumption token too many times" in job["error"], job
    assert len(fake.requests) == 4
    assert "too many times" in sync.cursor()["last_error"]


def test_a_busy_server_is_asked_again_after_it_says_when(sync, fake, monkeypatch):
    monkeypatch.setattr(harvest, "RETRY_SECONDS", 1)
    fake.serve(answer(503, {"Retry-After": "2"}), answer(500), DROP, "arxiv_last_page.xml")
    ended, job = sync.run({"set": "q-fin", "from": days_ago(3).isoformat()})
    assert ended == "succeeded", job
    assert job["result"]["retries"] == 3 and job["result"]["caught_up"] is True
    assert len(fake.requests) == 4
    waited = fake.gaps()
    assert waited[0] >= 1.95 and min(waited) >= 0.95
    assert sync.cursor()["last_error"] is None


def test_a_sync_gives_up_after_too_many_failed_requests_in_a_row(sync, fake, monkeypatch):
    monkeypatch.setattr(harvest, "RETRY_SECONDS", 1)
    monkeypatch.setattr(harvest, "MAX_RETRIES", 2)
    fake.serve(answer(503), answer(429), answer(503))
    ended, job = sync.run({"set": "q-fin", "from": days_ago(3).isoformat()})
    assert ended == "failed", job
    assert job["error"] == "The arXiv sync failed: arXiv answered HTTP 503 (2 retries in a row)"
    assert len(fake.requests) == 3
    assert sync.cursor()["last_error"] == "arXiv answered HTTP 503 (2 retries in a row)"

    monkeypatch.setattr(harvest, "MAX_RETRIES", 0)
    fake.serve(DROP)
    ended, job = sync.run({"set": "q-fin"})
    assert ended == "failed" and job["error"].startswith("The arXiv sync failed: the request failed:"), job


@pytest.mark.parametrize("item, reason", [
    (answer(404), "arXiv answered HTTP 404"),
    (answer(302, {"Location": "http://127.0.0.1:1/oai"}), "arXiv answered HTTP 302"),
    (answer(200, {}, error_page("badArgument")), "arXiv answered badArgument: Something is wrong."),
])
def test_any_other_answer_fails_the_sync_and_is_kept_on_the_cursor(sync, fake, item, reason):
    fake.serve(item)
    ended, job = sync.run({"set": "q-fin", "from": days_ago(3).isoformat()})
    assert ended == "failed" and job["error"] == f"The arXiv sync failed: {reason}", job
    assert len(fake.requests) == 1
    cursor = sync.cursor()
    assert (cursor["last_error"], cursor["pages"], cursor["last_complete_until"]) == (reason, 0, None)
    assert sync.documents() == 0


@pytest.mark.parametrize("out, reason", [
    ({"http_status": 200, "page": {"records": []}}, "arXiv's answer did not pass the checks"),
    ({"http_status": 200, "page": "<xml/>"}, "arXiv's answer did not pass the checks"),
    ({"http_status": "200"}, "the job process returned no valid HTTP status"),
    ([200], "the job process returned no result object"),
])
def test_the_worker_trusts_nothing_a_job_process_returns(sync, out, reason):
    job_id = sync.ai.enqueue({"set": "q-fin", "from": days_ago(3).isoformat()}, kind="arxiv_sync")
    runner = sync.worker()
    runner.run_child = lambda *args: out
    try:
        assert runner.run_once() == "failed"
    finally:
        runner.close()
    job = sync.ai.job(job_id)
    assert job["error"].startswith(f"The arXiv sync failed: {reason}"), job
    assert sync.documents() == 0


def test_a_sync_stops_at_the_kinds_deadline_and_keeps_its_place(sync, fake, monkeypatch):
    monkeypatch.setitem(KINDS, "arxiv_sync", dataclasses.replace(KINDS["arxiv_sync"], timeout_seconds=1))
    fake.serve(*PAGES)
    ended, job = sync.run({"set": "q-fin", "from": days_ago(10).isoformat()})
    assert ended == "succeeded", job
    assert (job["result"]["pages"], job["result"]["caught_up"]) == (1, False)
    assert sync.cursor()["resumption_token"] == TOKEN_1


def test_a_page_is_not_recorded_by_a_worker_that_lost_its_job(sync, fake):
    def steal(request):
        sync.sql("UPDATE {ai}.jobs SET worker_id = 'thief' WHERE kind = 'arxiv_sync'")
        return xml("arxiv_page1.xml")(request)

    fake.serve(steal)
    ended, job = sync.run({"set": "q-fin", "from": days_ago(10).isoformat()})
    assert ended == "lost (lease lapsed)" and job["status"] == "running"
    assert sync.sql("SELECT worker_id FROM {ai}.jobs WHERE id = %s", (job["id"],)).fetchone() == ("thief",)
    cursor = sync.cursor()
    assert (cursor["pages"], cursor["resumption_token"], cursor["window_from"]) == (0, None, None)
    assert sync.documents() == 0


def test_the_ownership_check_and_the_page_share_one_transaction(sync):
    start = days_ago(10)
    job_id = sync.ai.enqueue({"set": "q-fin", "from": start.isoformat()}, kind="arxiv_sync")
    runner = sync.worker()
    try:
        with runner.conn.transaction():
            job = queue.claim_next(runner.conn, runner.schema, runner.id, 30)
        assert job["id"] == job_id
        run = harvest._Sync(runner, job, KINDS["arxiv_sync"])
        cursor = run.open_cursor("q-fin", start)
        cursor.update(window_from=start, window_until=today())
        sync.sql("UPDATE {ai}.jobs SET worker_id = 'thief' WHERE id = %s", (job_id,))
        with pytest.raises(harvest.Interrupted):
            run.record(cursor, fixture_page("arxiv_page1.xml"), today())
        assert cursor["token"] is None and run.pages == 0
    finally:
        runner.close()
    assert sync.cursor()["pages"] == 0 and sync.documents() == 0


def test_switching_the_stage_off_hands_the_sync_back_and_it_resumes(sync, fake):
    def switch_off(request):
        sync.ai.flag("arxiv_sync", False)
        time.sleep(2)  # meanwhile the worker checks in, and stops
        return xml("arxiv_page1.xml")(request)

    start = days_ago(10)
    fake.serve(switch_off)
    ended, job = sync.run({"set": "q-fin", "from": start.isoformat()})
    assert ended == "released (stage switched off)"
    assert (job["status"], job["attempts"]) == ("queued", 0)
    assert sync.cursor()["pages"] == 0
    sync.ai.flag("arxiv_sync", True)
    fake.serve(*PAGES)
    runner = sync.worker()
    try:
        assert runner.run_once() == "succeeded"
    finally:
        runner.close()
    assert sync.ai.job(job["id"])["result"]["caught_up"] is True
    assert fake.queries()[1:] == [window("q-fin", start, today()), resume(TOKEN_1), resume(TOKEN_2)]


def test_one_sync_at_a_time(sync):
    payload = {"set": "q-fin", "from": days_ago(3).isoformat()}
    first = sync.ai.enqueue(payload, kind="arxiv_sync")
    with pytest.raises(queue.EnqueueError, match="already queued or running"):
        sync.ai.enqueue({"set": "math"}, kind="arxiv_sync")
    sync.ai.flag("selftest", True)
    sync.ai.enqueue({}, kind="ai_selftest")  # other kinds are not held back
    assert {sync.ai.claim()["id"], sync.ai.claim()["id"]} >= {first}  # both running now
    with pytest.raises(queue.EnqueueError, match="already queued or running"):
        sync.ai.enqueue({"set": "math"}, kind="arxiv_sync")
    with sync.ai.conn.transaction():
        assert queue.complete(sync.ai.conn, sync.ai.schema, first, "w1", {})
    assert sync.ai.enqueue(payload, kind="arxiv_sync")


@pytest.mark.parametrize("payload, reason", [
    ({"set": "q-fin"}, "set q-fin has never been harvested: give the date to start from"),
    ({"set": "q-fin", "from": "2026-11-01", "url": "http://evil.example/oai"},
     "an arxiv_sync payload names only set, from and max_pages"),
    ({"set": "q-fin", "from": "2999-01-01"}, "from must be between 1991-01-01 and today"),
])
def test_a_sync_that_cannot_start_fails_without_a_request(sync, fake, payload, reason):
    ended, job = sync.run(payload)
    assert ended == "failed" and job["error"] == f"The arXiv sync failed: {reason}", job
    assert fake.requests == [] and sync.cursor() is None


# -- settings and the command ----------------------------------------------------------

def test_the_oai_url_is_arxivs_and_production_may_not_change_it():
    base = {"QS_PROCESS_ROLE": "ai-worker", "AI_DATABASE_URL": "postgresql+psycopg://qs_ai_worker:pw@db/qs"}
    assert ai_settings.load(base).arxiv_oai_url == ai_settings.ARXIV_OAI_URL == "https://oaipmh.arxiv.org/oai"
    assert ai_settings.load({**base, "ENVIRONMENT": "production"}).arxiv_oai_url == ai_settings.ARXIV_OAI_URL
    assert ai_settings.load({**base, "AI_ARXIV_OAI_URL": "http://127.0.0.1:8080/oai"}).arxiv_oai_url \
        == "http://127.0.0.1:8080/oai"
    with pytest.raises(ai_settings.SettingsError, match="may not be changed in production"):
        ai_settings.load({**base, "ENVIRONMENT": "production", "AI_ARXIV_OAI_URL": "http://127.0.0.1:8080/oai"})
    for url in ("ftp://example.com/oai", "https://example.com/oai?verb=Identify", "example.com/oai",
                "https://exa mple.com/oai"):
        with pytest.raises(ai_settings.SettingsError, match="AI_ARXIV_OAI_URL"):
            ai_settings.load({**base, "AI_ARXIV_OAI_URL": url})


def test_the_command_checks_and_queues_a_sync(make_engine, capsys):
    from backend import manage

    engine = make_engine(migrated=True)
    sessions = sessionmaker(bind=engine)
    ai = Ai(engine)
    try:
        assert manage._ai_arxiv_sync(sessions, "q fin", None, None, OPS) == 1
        assert "set must be an arXiv set spec" in capsys.readouterr().err
        assert manage._ai_arxiv_sync(sessions, "q-fin", "2026-11-31", None, OPS) == 1
        assert "from must be a date" in capsys.readouterr().err
        assert manage._ai_arxiv_sync(sessions, "q-fin", None, 0, OPS) == 1
        assert "max_pages" in capsys.readouterr().err
        ai.switch(True, True, "arxiv_sync")
        assert manage._ai_arxiv_sync(sessions, "q-fin", "2026-11-01", 5, OPS) == 0
        job_id = capsys.readouterr().out.strip().rsplit(": ", 1)[1]
        job = ai.job(job_id)
        assert (job["kind"], job["status"], job["payload"]) \
            == ("arxiv_sync", "queued", {"set": "q-fin", "from": "2026-11-01", "max_pages": 5})
        assert manage._ai_arxiv_sync(sessions, "math", None, None, OPS) == 1
        assert "already queued or running" in capsys.readouterr().err
        with engine.connect() as connection:
            metadata = connection.exec_driver_sql(
                "SELECT metadata_json FROM audit_logs WHERE action = 'AI_JOB_QUEUED' AND resource_id = %(id)s",
                {"id": job_id}).scalar()
        assert metadata == {"kind": "arxiv_sync", "queued_by": OPS, "set": "q-fin", "from": "2026-11-01",
                            "max_pages": 5}
    finally:
        ai.conn.close()


# -- arXiv itself ------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("QS_LIVE_ARXIV") != "1", reason="set QS_LIVE_ARXIV=1 to harvest from arXiv")
def test_a_live_sync_from_arxiv(ai):
    """Two pages of q-fin from the last 40 days (so a resumption token or a
    second window), then the same pages again."""
    ai.switch(True, True, "arxiv_sync")
    harvester = Harvester(ai, ai_settings.ARXIV_OAI_URL)
    start = days_ago(harvest.WINDOW_DAYS + 9)
    payload = {"set": "q-fin", "from": start.isoformat(), "max_pages": 2}
    started = time.monotonic()
    ended, job = harvester.run(payload)
    elapsed = time.monotonic() - started
    assert ended == "succeeded", job
    first = job["result"]
    print(f"\nlive arXiv sync of q-fin from {start}: {first} in {elapsed:.1f} s; cursor {harvester.cursor()}")
    assert first["pages"] == 2 and first["rejected"] == 0
    assert first["inserted"] == harvester.documents() > 0
    cursor = harvester.cursor()
    assert cursor["pages"] == first["pages"] and cursor["last_error"] is None
    assert cursor["resumption_token"] is not None or cursor["last_complete_until"] is not None

    harvester.sql("UPDATE {ai}.harvests SET last_complete_until = NULL, window_from = NULL, window_until = NULL,"
                  " resumption_token = NULL, token_expires = NULL")
    ended, job = harvester.run({**payload, "max_pages": first["pages"]})
    assert ended == "succeeded", job
    again = job["result"]
    print(f"the same pages again: {again}")
    assert again["inserted"] == 0 and harvester.documents() == first["inserted"]
