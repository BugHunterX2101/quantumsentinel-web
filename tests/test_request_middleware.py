"""The outermost request middleware: security headers, per-principal rate
limiting and request metrics, on every kind of HTTP response."""
import time
import uuid
from collections import OrderedDict

import pytest
from fastapi import BackgroundTasks, HTTPException, Request
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

    def test_buckets_are_per_path(self, client):
        path = f"/api/auth/no-such-{uuid.uuid4().hex}"
        for _ in range(11):
            client.get(path)
        assert client.get(path).status_code == 429
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


class TestRateLimitBuckets:
    @pytest.fixture(autouse=True)
    def empty_map(self, monkeypatch):
        monkeypatch.setattr(main, "_request_windows", OrderedDict())

    def hit(self, key, now):
        window = main._rate_window(key, now)
        while window and now - window[0] > main._RATE_WINDOW_SECONDS:
            window.popleft()
        window.append(now)
        return len(window)

    def test_counts_hits_inside_the_window(self):
        assert [self.hit("a", t) for t in (0.0, 1.0, 2.0)] == [1, 2, 3]
        assert self.hit("a", 61.5) == 2
        assert self.hit("b", 61.5) == 1

    def test_expired_buckets_are_dropped(self):
        for i in range(5):
            self.hit(f"old-{i}", float(i))
        self.hit("live", 30.0)
        self.hit("new", 64.5)
        assert list(main._request_windows) == ["live", "new"]

    def test_a_bucket_in_use_is_kept_while_older_ones_expire(self):
        self.hit("busy", 0.0)
        self.hit("idle", 1.0)
        self.hit("busy", 50.0)
        self.hit("other", 70.0)
        assert list(main._request_windows) == ["busy", "other"]
        assert self.hit("busy", 100.0) == 2

    def test_cap_evicts_the_least_recently_seen_bucket(self, monkeypatch):
        monkeypatch.setattr(main, "_RATE_MAX_KEYS", 3)
        for i, key in enumerate(("a", "b", "c")):
            self.hit(key, float(i))
        self.hit("a", 3.0)
        self.hit("d", 4.0)
        assert list(main._request_windows) == ["c", "a", "d"]
        assert len(main._request_windows) == 3

    def test_a_full_map_of_live_buckets_stays_cheap(self):
        for i in range(main._RATE_MAX_KEYS):
            self.hit(f"k{i}", i / 1000)
        started = time.perf_counter()
        for i in range(5_000):
            self.hit(f"new{i}", 20.0 + i / 1000)
        elapsed = time.perf_counter() - started
        assert len(main._request_windows) == main._RATE_MAX_KEYS
        # The previous scan-and-sort per call took ~68 s for these 5,000 calls.
        assert elapsed < 1.0, elapsed


class TestRateLimitPrincipal:
    """A bucket is keyed on an identity the client cannot invent: a verified
    access token's user, an API key this process has already accepted, or
    else the client address. Any made-up header value must land in the
    caller's address bucket, never in a fresh one."""

    @pytest.fixture
    def sdk_route(self):
        path = f"/api/sdk/__mw_{uuid.uuid4().hex[:8]}"
        good = f"qs_{uuid.uuid4().hex}"

        def endpoint(request: Request):
            if request.headers.get("x-qs-api-key") != good:
                raise HTTPException(403, "Invalid API key or insufficient scope")
            return {"ok": True}

        main.app.add_api_route(path, endpoint, methods=["GET"])
        route = main.app.router.routes.pop()
        main.app.router.routes.insert(0, route)
        yield path, good
        main.app.router.routes.remove(route)

    @staticmethod
    def remaining(response):
        return int(response.headers["x-ratelimit-remaining"])

    @staticmethod
    def token(user_id):
        return main.auth_service.create_access_token(user_id, "free")

    @pytest.mark.parametrize("headers, cookies", [
        ({"authorization": "Bearer made-up"}, {}),
        ({"x-qs-api-key": "made-up"}, {}),
        ({}, {"qs_access": "made-up"}),
        ({"authorization": "Basic made-up"}, {}),
    ])
    def test_auth_paths_are_keyed_on_the_client_address_only(self, client, headers, cookies):
        path = f"/api/auth/no-such-{uuid.uuid4().hex}"
        for _ in range(10):
            assert client.get(path).status_code == 404
        client.cookies.update(cookies)
        assert client.get(path, headers=headers).status_code == 429

    def test_a_signed_token_does_not_buy_extra_login_attempts(self, client):
        path = f"/api/auth/no-such-{uuid.uuid4().hex}"
        for _ in range(10):
            client.get(path)
        bearer = {"authorization": f"Bearer {self.token(uuid.uuid4().hex)}"}
        assert client.get(path, headers=bearer).status_code == 429

    def test_made_up_credentials_share_the_address_bucket(self, client):
        path = unmatched_api_path()
        seen = [self.remaining(client.get(path))]
        for i in range(3):
            seen.append(self.remaining(client.get(path, headers={"authorization": f"Bearer x{i}"})))
            seen.append(self.remaining(client.get(path, headers={"x-qs-api-key": f"k{i}"})))
        client.cookies.set("qs_access", "made-up")
        seen.append(self.remaining(client.get(path)))
        assert seen == list(range(239, 239 - len(seen), -1))

    def test_signed_tokens_are_keyed_on_their_user(self, client):
        path = unmatched_api_path()
        alice, bob = uuid.uuid4().hex, uuid.uuid4().hex
        first, second = self.token(alice), self.token(alice)
        assert first != second
        assert self.remaining(client.get(path, headers={"authorization": f"Bearer {first}"})) == 239
        assert self.remaining(client.get(path, headers={"authorization": f"Bearer {second}"})) == 238
        assert self.remaining(client.get(path, headers={"authorization": f"Bearer {self.token(bob)}"})) == 239
        assert self.remaining(client.get(path)) == 239

    def test_the_cookie_wins_over_other_headers_as_in_get_current_user(self, client):
        path = unmatched_api_path()
        client.cookies.set("qs_access", self.token(uuid.uuid4().hex))
        seen = [self.remaining(client.get(path, headers={"authorization": f"Bearer x{i}",
                                                          "x-qs-api-key": f"k{i}"}))
                for i in range(3)]
        assert seen == [239, 238, 237]
        client.cookies.clear()
        assert self.remaining(client.get(path)) == 239

    def test_an_api_key_gets_its_own_bucket_once_it_has_been_accepted(self, client, sdk_route):
        path, good = sdk_route
        assert client.get(path, headers={"x-qs-api-key": "made-up-1"}).status_code == 403
        # A rejected key is never learned, however often it is sent.
        assert self.remaining(client.get(path, headers={"x-qs-api-key": "made-up-1"})) == 238
        assert self.remaining(client.get(path, headers={"x-qs-api-key": "made-up-2"})) == 237
        accepted = client.get(path, headers={"x-qs-api-key": good})
        assert accepted.status_code == 200 and self.remaining(accepted) == 236
        assert self.remaining(client.get(path, headers={"x-qs-api-key": good})) == 239
        assert self.remaining(client.get(path, headers={"x-qs-api-key": "made-up-3"})) == 235

    def test_hmac_requests_are_keyed_on_the_address(self, client, sdk_route):
        path, good = sdk_route
        client.get(path, headers={"x-qs-api-key": good})
        hmac = {"x-qs-key-id": "id", "x-qs-timestamp": "1", "x-qs-nonce": "n", "x-qs-signature": "s"}
        # require_api_scope ignores X-QS-API-KEY when all four HMAC headers are present.
        assert self.remaining(client.get(path, headers={**hmac, "x-qs-api-key": good})) == 238

    def test_identity_caches_are_bounded(self, client, monkeypatch):
        monkeypatch.setattr(main, "_RATE_PRINCIPAL_CACHE", 2)
        monkeypatch.setattr(main, "_rate_users", OrderedDict())
        path = unmatched_api_path()
        for _ in range(3):
            client.get(path, headers={"authorization": f"Bearer {self.token(uuid.uuid4().hex)}"})
        assert len(main._rate_users) == 2
