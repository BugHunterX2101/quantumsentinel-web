"""The outermost request middleware: security headers, per-principal rate
limiting and request metrics, on every kind of HTTP response."""
import uuid

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from backend import main

ORIGIN = "http://localhost:8000"


def expected_csp(ws_policy):
    return (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data:; "
        f"connect-src 'self' {ws_policy} https://api.github.com https://api.pwnedpasswords.com; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "form-action 'self'"
    )


SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
    "content-security-policy": expected_csp("wss: ws:"),
}

BIG_BODY = b"0123456789abcdef" * 4096
background_ran = []


@pytest.fixture
def routes():
    """Temporary routes covering response shapes the real API produces."""
    prefix = f"/__mw_{uuid.uuid4().hex[:8]}"

    def items(item_id: int):
        return {"item_id": item_id}

    def denied():
        raise HTTPException(401, "nope")

    def framed():
        return PlainTextResponse("x", headers={"X-Frame-Options": "SAMEORIGIN"})

    def streamed():
        return StreamingResponse(iter([BIG_BODY[i:i + 8192] for i in range(0, len(BIG_BODY), 8192)]),
                                 media_type="application/octet-stream")

    def with_background(tasks: BackgroundTasks):
        tasks.add_task(background_ran.append, prefix)
        return {"ok": True}

    def crash():
        raise RuntimeError("boom")

    added = []
    for path, endpoint in [("/items/{item_id}", items), ("/denied", denied), ("/framed", framed),
                           ("/streamed", streamed), ("/background", with_background), ("/crash", crash)]:
        main.app.add_api_route(prefix + path, endpoint, methods=["GET"])
        # Ahead of the SPA catch-all, which would otherwise match first.
        added.append(main.app.router.routes.pop())
        main.app.router.routes.insert(0, added[-1])
    yield prefix
    for route in added:
        main.app.router.routes.remove(route)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "_redis_client", None)
    return TestClient(main.app)


def assert_security_headers(response, limit=240):
    for name, value in SECURITY_HEADERS.items():
        assert response.headers.get_list(name) == [value], name
    assert response.headers.get_list("x-ratelimit-limit") == [str(limit)]
    assert len(response.headers.get_list("x-ratelimit-remaining")) == 1


def requests_counted(method, path, status):
    return REGISTRY.get_sample_value("quantumsentinel_http_requests_total",
                                     {"method": method, "path": path, "status": status}) or 0.0


def unmatched_api_path():
    return f"/api/no-such-route-{uuid.uuid4().hex}"


def unmatched_label(path):
    # Unknown /api paths are answered 404 by the SPA catch-all when the
    # frontend is served, so that route's template is the metric label.
    return "/{path:path}" if main.FRONTEND_DIR.exists() else path


def latency_observations(method, path):
    return REGISTRY.get_sample_value("quantumsentinel_http_request_duration_seconds_count",
                                     {"method": method, "path": path}) or 0.0


class TestSecurityHeaders:
    def test_success(self, client):
        assert_security_headers(client.get("/health/live"))

    def test_unmatched_route(self, client):
        response = client.get(unmatched_api_path())
        assert response.status_code == 404
        assert_security_headers(response)

    def test_handled_http_exception(self, client, routes):
        response = client.get(f"{routes}/denied")
        assert response.status_code == 401
        assert_security_headers(response)

    def test_rejected_host(self, client):
        response = client.get("/health/live", headers={"host": "evil.example"})
        assert response.status_code == 400
        assert_security_headers(response)

    def test_cors_preflight(self, client):
        response = client.options("/health/live", headers={
            "origin": ORIGIN, "access-control-request-method": "GET"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == ORIGIN
        assert_security_headers(response)

    def test_endpoint_values_are_replaced_not_duplicated(self, client, routes):
        response = client.get(f"{routes}/framed")
        assert response.text == "x"
        assert_security_headers(response)

    def test_production_csp_allows_only_secure_websockets(self, client, monkeypatch):
        monkeypatch.setattr(main, "ENVIRONMENT", "production")
        csp = client.get("/health/live").headers.get_list("content-security-policy")
        assert csp == [expected_csp("wss:")]


class TestResponseBodies:
    def test_streamed_body_arrives_intact_and_compressed(self, client, routes):
        response = client.get(f"{routes}/streamed", headers={"accept-encoding": "gzip"})
        assert response.headers["content-encoding"] == "gzip"
        assert response.content == BIG_BODY
        assert_security_headers(response)

    def test_uncompressed_body(self, client, routes):
        response = client.get(f"{routes}/streamed", headers={"accept-encoding": "identity"})
        assert "content-encoding" not in response.headers
        assert response.content == BIG_BODY

    def test_background_tasks_still_run(self, client, routes):
        assert client.get(f"{routes}/background").json() == {"ok": True}
        assert routes in background_ran

    def test_unhandled_error_is_a_plain_500(self, routes):
        response = TestClient(main.app, raise_server_exceptions=False).get(f"{routes}/crash")
        assert response.status_code == 500
        assert response.text == "Internal Server Error"


class TestRateLimit:
    def test_auth_paths_allow_ten_per_minute(self, client):
        path = f"/api/auth/no-such-{uuid.uuid4().hex}"
        remaining = []
        for _ in range(10):
            response = client.get(path)
            assert response.status_code == 404
            assert response.headers["x-ratelimit-limit"] == "10"
            remaining.append(int(response.headers["x-ratelimit-remaining"]))
        assert remaining == list(range(9, -1, -1))
        blocked = client.get(path)
        assert blocked.status_code == 429
        assert blocked.json() == {"detail": "Rate limit exceeded"}
        assert blocked.headers["retry-after"] == "60"

    def test_buckets_are_per_principal_and_per_path(self, client):
        path = f"/api/auth/no-such-{uuid.uuid4().hex}"
        for _ in range(11):
            client.get(path)
        assert client.get(path).status_code == 429
        assert client.get(path, headers={"authorization": "Bearer someone-else"}).status_code == 404
        assert client.get(path, headers={"x-qs-api-key": "a-key"}).status_code == 404
        assert client.get(path + "-other").status_code == 404

    def test_other_paths_allow_240_per_minute(self, client):
        path = unmatched_api_path()
        first = client.get(path)
        assert first.headers["x-ratelimit-limit"] == "240"
        assert first.headers["x-ratelimit-remaining"] == "239"
        assert client.get(path).headers["x-ratelimit-remaining"] == "238"

    def test_shared_counter_in_redis(self, client, monkeypatch):
        class FakeRedis:
            def __init__(self):
                self.counts, self.expiries = {}, []

            async def incr(self, key):
                self.counts[key] = self.counts.get(key, 0) + 1
                return self.counts[key]

            async def expire(self, key, seconds):
                self.expiries.append((key, seconds))

        fake = FakeRedis()
        monkeypatch.setattr(main, "_redis_client", fake)
        path = unmatched_api_path()
        assert [client.get(path).headers["x-ratelimit-remaining"] for _ in range(3)] == ["239", "238", "237"]
        (key, count), = fake.counts.items()
        assert key.startswith("qs:rate:") and key.endswith(":" + path) and count == 3
        assert fake.expiries == [(key, 60)]

    def test_redis_outage_fails_closed_in_production(self, client, monkeypatch):
        class DownRedis:
            async def incr(self, key):
                raise ConnectionError("redis down")

        monkeypatch.setattr(main, "_redis_client", DownRedis())
        monkeypatch.setattr(main, "ENVIRONMENT", "production")
        response = client.get("/health/live")
        assert response.status_code == 503
        assert response.json() == {"detail": "Rate-limit service unavailable"}

    def test_redis_outage_falls_back_to_memory_outside_production(self, client, monkeypatch):
        class DownRedis:
            async def incr(self, key):
                raise ConnectionError("redis down")

        monkeypatch.setattr(main, "_redis_client", DownRedis())
        path = unmatched_api_path()
        assert client.get(path).headers["x-ratelimit-remaining"] == "239"


class TestMetrics:
    def test_requests_are_labelled_by_route_template(self, client, routes):
        template = f"{routes}/items/{{item_id}}"
        before = requests_counted("GET", template, "200"), latency_observations("GET", template)
        assert client.get(f"{routes}/items/7").json() == {"item_id": 7}
        assert client.get(f"{routes}/items/8").json() == {"item_id": 8}
        assert requests_counted("GET", template, "200") == before[0] + 2
        assert latency_observations("GET", template) == before[1] + 2

    def test_unmatched_paths_and_error_statuses(self, client, routes):
        path = unmatched_api_path()
        before = requests_counted("GET", unmatched_label(path), "404")
        assert client.get(path).status_code == 404
        assert requests_counted("GET", unmatched_label(path), "404") == before + 1
        before = requests_counted("GET", f"{routes}/denied", "401")
        client.get(f"{routes}/denied")
        assert requests_counted("GET", f"{routes}/denied", "401") == before + 1

    def test_rejected_requests_are_not_counted(self, client):
        path = f"/api/auth/no-such-{uuid.uuid4().hex}"
        label = unmatched_label(path)
        before = requests_counted("GET", label, "404"), requests_counted("GET", label, "429")
        for _ in range(11):
            client.get(path)
        assert requests_counted("GET", label, "404") == before[0] + 10
        assert requests_counted("GET", label, "429") == before[1]
        assert requests_counted("GET", path, "429") == 0
