"""QuantumSentinel — FastAPI application.

Single-process web port of the multi-service architecture in the design
docs (API Gateway + Trading Engine + PQC Crypto Service + Signal Engine
collapsed into one deployable app for a portable web demo). All PQC
operations are genuine FIPS 203/204 algorithms (see backend/crypto/pqc.py).

Security hardening (v2):
- HttpOnly cookie-based auth with refresh token rotation (Item 1)
- CSRF double-submit cookie pattern
- HMAC-signed API key requests (Item 7)
- Redis-backed kill switches (Item 6)
- Tightened CSP (script-src 'self'; Three.js is self-hosted)
- WebSocket per-user connection limits, idle timeout, sequence numbers
- Server signing key history endpoint (Item 8)
"""
import datetime as dt
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from typing import Optional, List
import logging
import os
import time
import secrets
import subprocess
import sys
import asyncio
from contextlib import asynccontextmanager
import hashlib
import math
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Header, Request, WebSocket, WebSocketDisconnect, Query, Cookie
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as redis

from . import models, schemas
from .database import engine, get_db, init_db, SessionLocal
from .config import (CORS_ORIGINS, ALLOWED_HOSTS, ENVIRONMENT, REDIS_URL, JWT_EXPIRE_SECONDS,
                     COOKIE_DOMAIN, COOKIE_SECURE, COOKIE_SAMESITE,
                     REFRESH_TOKEN_SECONDS, TRUSTED_SERVER_DSA_FINGERPRINT, OPERATOR_ROLES,
                     PAPER_MAX_POSITION_FRACTION, ORDER_SWEEPER_ENABLED, ORDER_SWEEP_INTERVAL_SECONDS,
                     RESEARCH_WORKER_MODE)
from .crypto import pqc
from .services import auth_service, signal_engine, trading_service, portfolio_service, security_service, integration_service, order_security
from .services import paper_broker, research_jobs, research_trials
from .services import walk_forward as walk_forward_service
from .services import stat_tests as stat_tests_service
from .services import redis_store

log = logging.getLogger(__name__)

# Single source of truth for the product version reported by /api/meta,
# the OpenAPI schema and the compliance evidence bundle.
APP_VERSION = "1.2.0"

# Default watchlist for new users.
#
# This is deliberately identical to signal_engine.PRELOADED_ASSETS: those are
# the only tickers the SBA basket pipeline keeps warm, so a default containing
# anything else would show the user a watchlist entry that can never produce a
# signal. It also has to stay strictly below the 50-asset cap, otherwise a
# brand-new user (whose effective watchlist is this default) would be unable to
# add their first ticker.
DEFAULT_WATCHLIST = list(signal_engine.PRELOADED_ASSETS)
MAX_WATCHLIST_SIZE = 50

# ── Exchange Registry ──────────────────────────────────────────────────────────
EXCHANGE_REGISTRY = {
    "US":     {"name": "NYSE / NASDAQ", "country": "United States", "flag": "US",
               "tz": "America/New_York", "open": "09:30", "close": "16:00",
               "currency": "USD", "description": "World's largest equity market"},
    "NSE":    {"name": "NSE / BSE", "country": "India", "flag": "IN",
               "tz": "Asia/Kolkata", "open": "09:15", "close": "15:30",
               "currency": "INR", "description": "India's premier stock exchanges"},
    "LSE":    {"name": "London Stock Exchange", "country": "United Kingdom", "flag": "GB",
               "tz": "Europe/London", "open": "08:00", "close": "16:30",
               "currency": "GBP", "description": "Europe's largest equity market"},
    "XETRA":  {"name": "Deutsche Börse Xetra", "country": "Germany", "flag": "DE",
               "tz": "Europe/Berlin", "open": "09:00", "close": "17:30",
               "currency": "EUR", "description": "Germany's primary electronic trading platform"},
    "TSE":    {"name": "Tokyo Stock Exchange", "country": "Japan", "flag": "JP",
               "tz": "Asia/Tokyo", "open": "09:00", "close": "15:30",
               "currency": "JPY", "description": "Asia's second-largest stock exchange"},
    "HKEX":   {"name": "Hong Kong Stock Exchange", "country": "Hong Kong", "flag": "HK",
               "tz": "Asia/Hong_Kong", "open": "09:30", "close": "16:00",
               "currency": "HKD", "description": "Gateway to Chinese equity markets"},
    "ASX":    {"name": "Australian Securities Exchange", "country": "Australia", "flag": "AU",
               "tz": "Australia/Sydney", "open": "10:00", "close": "16:00",
               "currency": "AUD", "description": "Australia's primary securities exchange"},
    "TSX":    {"name": "Toronto Stock Exchange", "country": "Canada", "flag": "CA",
               "tz": "America/Toronto", "open": "09:30", "close": "16:00",
               "currency": "CAD", "description": "Canada's largest stock exchange"},
    "CRYPTO": {"name": "Crypto Markets", "country": "Global", "flag": "CRYPTO",
               "tz": "UTC", "open": "00:00", "close": "23:59",
               "currency": "USD", "description": "24/7 digital asset markets — never closes"},
}

def _market_status(exch_key: str) -> dict:
    """Return open/closed/pre/after-hours status for a given exchange."""
    info = EXCHANGE_REGISTRY.get(exch_key, {})
    if exch_key == "CRYPTO":
        return {"status": "open", "label": "24/7 OPEN", "next_event": None}
    tz_name = info.get("tz", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = dt.timezone.utc
    now = dt.datetime.now(tz)
    open_h, open_m = map(int, info.get("open", "09:30").split(":"))
    close_h, close_m = map(int, info.get("close", "16:00").split(":"))
    open_time  = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    close_time = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    # Weekends
    if now.weekday() >= 5:
        return {"status": "closed", "label": "WEEKEND", "local_time": now.strftime("%H:%M")}
    if now < open_time:
        mins = int((open_time - now).total_seconds() // 60)
        return {"status": "pre", "label": "PRE-MARKET", "opens_in_mins": mins,
                "local_time": now.strftime("%H:%M")}
    # FIX M3/B7 parity: strict < so the exchange shows CLOSED at exactly close_time,
    # not one extra minute OPEN after the bell.
    if now < close_time:
        mins = int((close_time - now).total_seconds() // 60)
        return {"status": "open", "label": "OPEN", "closes_in_mins": mins,
                "local_time": now.strftime("%H:%M")}
    return {"status": "closed", "label": "CLOSED", "local_time": now.strftime("%H:%M")}


def _user_watchlist(user: models.User) -> list[str]:
    """Return the user's watchlist, falling back to DEFAULT_WATCHLIST."""
    wl = user.watchlist
    if not wl:
        return DEFAULT_WATCHLIST[:]
    return wl

def _etag(value: str) -> str:
    # Cache validator, not a security control.
    return hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest()[:16]


def _probe_tickers_parallel(tickers: list[str], max_workers: int = 10) -> set[str]:
    """Validate a batch of non-preloaded tickers against yfinance concurrently.

    Probing sequentially (one `yf.Ticker(t).fast_info` HTTP round-trip per
    ticker) turns a 50-asset watchlist save into tens of seconds of serial
    network latency. Each probe is independent I/O, so a small thread pool
    collapses that to roughly the slowest single probe.
    """
    if not tickers:
        return set()
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _probe(t: str) -> str | None:
        try:
            info = yf.Ticker(t).fast_info
            return t if getattr(info, "last_price", None) is not None else None
        except Exception:
            return None

    valid: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as pool:
        futures = {pool.submit(_probe, t): t for t in tickers}
        for future in as_completed(futures, timeout=15):
            try:
                result = future.result()
            except Exception:
                result = None
            if result:
                valid.add(result)
    return valid

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Redis client must be created before _lifespan so the startup coroutine can reference it.
_redis_client = redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """FastAPI lifespan handler — replaces deprecated @app.on_event('startup')."""
    # Publish the running loop so synchronous routes can safely await Redis
    # coroutines on it instead of creating (and closing) throwaway loops.
    redis_store.set_app_loop(asyncio.get_running_loop())
    init_db()
    # Register the server signing key in the DB for audit key history (Item 8)
    _enforce_server_identity_pin()
    db = SessionLocal()
    try:
        security_service.server_identity.register_in_db(db)
    finally:
        db.close()
    if _redis_client:
        try:
            await _redis_client.ping()
        except Exception as exc:
            if ENVIRONMENT == "production":
                raise RuntimeError("Redis is required and unavailable") from exc
    sweeper = asyncio.create_task(_order_sweeper()) if ORDER_SWEEPER_ENABLED else None
    research_worker = _start_embedded_worker() if RESEARCH_WORKER_MODE == "embedded" else None
    try:
        yield  # ── application runs here ──
    finally:
        if sweeper:
            sweeper.cancel()
        if research_worker:
            await asyncio.to_thread(_stop_embedded_worker, research_worker)
        redis_store.set_app_loop(None)


def _start_embedded_worker() -> subprocess.Popen:
    """Run one research worker as a child of this API process.

    It is a separate process, so research never competes with requests for
    this process's threads, GIL or database connections. Each API process
    (each gunicorn worker) starts its own; they share the queue safely. To
    scale workers independently, run them as their own service instead
    (RESEARCH_WORKER_MODE=external).
    """
    log.info("starting embedded research worker")
    # Its stdin is a lifeline: only this process holds the write end, and the
    # OS closes it when this process exits, however it exits (even killed
    # outright, as uvicorn --reload does on Windows), so the worker never
    # outlives the API process.
    return subprocess.Popen([sys.executable, "-m", "backend.worker", "--lifeline"],
                            cwd=str(BASE_DIR), stdin=subprocess.PIPE)


def _stop_embedded_worker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    # Closing the lifeline stops the worker gracefully on every OS (it hands
    # its current job back to the queue); terminate() is a hard kill on Windows.
    proc.stdin.close()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5)


def _enforce_server_identity_pin() -> None:
    """Refuse to start if the signing key is not the pinned one."""
    security_service.enforce_identity_pin(TRUSTED_SERVER_DSA_FINGERPRINT)


async def _order_sweeper() -> None:
    """Fill resting paper orders as the market moves.

    Fills are driven by this clock, never by a client reading its order list.
    Each worker may run one; paper_broker's compare-and-set transitions make
    concurrent sweeps safe.
    """
    while True:
        await asyncio.sleep(ORDER_SWEEP_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(sweep_open_orders_once)
        except Exception:
            log.exception("order sweeper iteration failed")


_SWEEP_LOCK_KEY = 7_316_405_312_119


def sweep_open_orders_once() -> int:
    """One sweep, by at most one process at a time across the deployment.

    Every API process runs a sweeper; whichever takes the advisory lock
    first sweeps and the rest skip this tick, instead of all of them loading
    the same orders and fetching the same prices. Session-level lock on a
    connection of its own: fills commit several times during a sweep.
    """
    with engine.connect() as lock:
        if not lock.execute(select(func.pg_try_advisory_lock(_SWEEP_LOCK_KEY))).scalar():
            return 0
        try:
            db = SessionLocal()
            try:
                settled = 0
                for trade, outcome in paper_broker.process_open_orders(db):
                    if outcome.status == "NOT_OPEN":
                        continue
                    _record_order_outcome(db, trade, outcome.reason)
                    settled += 1
                return settled
            finally:
                db.close()
        finally:
            lock.execute(select(func.pg_advisory_unlock(_SWEEP_LOCK_KEY)))
            lock.commit()


# /docs, /redoc and the raw OpenAPI schema disclose every route, request/
# response model and field name. Fine for local development; an
# unauthenticated map of the entire API is unnecessary exposure once live.
_docs_enabled = ENVIRONMENT != "production"
app = FastAPI(
    title="QuantumSentinel API", version=APP_VERSION, lifespan=_lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-CSRF-Token", "Idempotency-Key",
                   "If-None-Match", "X-QS-API-KEY", "X-QS-Key-ID", "X-QS-Timestamp",
                   "X-QS-Nonce", "X-QS-Signature"],
)


@app.exception_handler(redis_store.RiskStateUnavailable)
async def _risk_state_unavailable(_request, _exc):
    # Security-critical shared state (kill switches) could not be read:
    # refuse the action rather than assume the state is permissive.
    return JSONResponse({"detail": "risk state unavailable"}, status_code=503)


def _non_finite_as_text(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)                     # "nan", "inf", "-inf"
    if isinstance(value, dict):
        return {k: _non_finite_as_text(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_non_finite_as_text(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _invalid_request(_request, exc: RequestValidationError):
    # FastAPI's own 422 response, except that a rejected NaN or Infinity
    # echoed back as the error's input is written as text: JSON cannot carry
    # it, and encoding it turned the 422 into a 500.
    return JSONResponse({"detail": _non_finite_as_text(jsonable_encoder(exc.errors()))}, status_code=422)


@app.exception_handler(SQLAlchemyTimeoutError)
async def _database_busy(_request, _exc):
    # Every pooled connection stayed busy for DB_POOL_TIMEOUT_SECONDS: shed
    # the request with a retryable status instead of a 500.
    log.warning("database connection pool exhausted")
    return JSONResponse({"detail": "database busy, retry shortly"}, status_code=503,
                        headers={"Retry-After": "1"})


@app.exception_handler(OperationalError)
async def _database_unavailable(_request, exc):
    # Server unreachable, connection lost, or a statement/lock timeout fired.
    log.error("database operation failed: %s", exc.orig if exc.orig is not None else exc)
    return JSONResponse({"detail": "database unavailable, retry shortly"}, status_code=503,
                        headers={"Retry-After": "5"})

HTTP_REQUESTS = Counter("quantumsentinel_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("quantumsentinel_http_request_duration_seconds", "HTTP request latency", ["method", "path"])

# Bounded in-memory limiter for the single-process reference deployment. It
# deliberately protects write paths even before a user has authenticated.
#
# Keys are (principal, path) pairs and paths include dynamic segments such as
# /api/price/{ticker}, so the map has to be swept or it grows without bound for
# the lifetime of the process.
#
# Buckets are kept in least-recently-used order (each request moves its key to
# the end), so expired buckets and eviction candidates are always at the front.
# Housekeeping pops from the front: O(1) amortised per request. Scanning and
# sorting the whole map instead cost ~13 ms per request once it held
# _RATE_MAX_KEYS live keys, which any client can cause by requesting distinct
# paths.
_request_windows: OrderedDict[str, deque[float]] = OrderedDict()
_RATE_WINDOW_SECONDS = 60
_RATE_MAX_KEYS = 20_000


def _rate_window(key: str, now: float) -> deque[float]:
    """The bucket for ``key``, after dropping buckets whose window has fully expired."""
    while _request_windows:
        oldest = next(iter(_request_windows.values()))
        if oldest and now - oldest[-1] <= _RATE_WINDOW_SECONDS:
            break
        _request_windows.popitem(last=False)
    window = _request_windows.get(key)
    if window is None:
        # Hard ceiling: evict the least-recently-seen bucket rather than let
        # memory grow without bound.
        if len(_request_windows) >= _RATE_MAX_KEYS:
            _request_windows.popitem(last=False)
        window = _request_windows[key] = deque()
    else:
        _request_windows.move_to_end(key)
    return window


# Bucket keys must name a principal the client cannot invent. Header values are
# client-chosen until verified, so keying on a raw header let a client start a
# fresh bucket on every request with a new made-up value: 30 of 30 logins under
# distinct emails reached the password check, against 10 and then 429 without
# the header. Credentials therefore count only once proven: an access token by
# its signature (keyed on its user, so re-issued tokens share one bucket), and an
# API key once this process has accepted it. Anything else, and every
# /api/auth/ route, is keyed on the client address.
_rate_users: OrderedDict[str, str] = OrderedDict()       # access token -> user id
_rate_api_keys: OrderedDict[str, None] = OrderedDict()   # sha256 of accepted keys
_RATE_PRINCIPAL_CACHE = 10_000
_HMAC_HEADERS = ("x-qs-key-id", "x-qs-timestamp", "x-qs-nonce", "x-qs-signature")


def _remember(cache: OrderedDict, key: str, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _RATE_PRINCIPAL_CACHE:
        cache.popitem(last=False)


def _sdk_api_key_hash(request: Request) -> str | None:
    """Hash of the legacy API key that ``require_api_scope`` would check."""
    if all(request.headers.get(name) for name in _HMAC_HEADERS):
        return None  # the HMAC path ignores X-QS-API-KEY
    raw = request.headers.get("x-qs-api-key")
    return hashlib.sha256(raw.encode()).hexdigest() if raw else None


def _rate_principal(request: Request, client: str, sdk_key: str | None) -> str:
    path = request.url.path
    if path.startswith("/api/auth/"):
        return client
    if path.startswith("/api/sdk/"):
        if sdk_key in _rate_api_keys:
            _rate_api_keys.move_to_end(sdk_key)
            return f"key:{sdk_key[:16]}"
        return client
    # Same precedence as get_current_user: the cookie, then a Bearer header.
    token = request.cookies.get("qs_access")
    if not token:
        authorization = request.headers.get("authorization", "")
        token = authorization.split(" ", 1)[1] if authorization.startswith("Bearer ") else None
    if not token:
        return client
    user_id = _rate_users.get(token)
    if user_id is None:
        payload = auth_service.decode_access_token(token)
        if not payload:
            return client
        user_id = payload["sub"]
        _remember(_rate_users, token, user_id)
    else:
        _rate_users.move_to_end(token)
    return f"user:{user_id}"


class SecurityHeadersAndRateLimit:
    """Rate limiting, request metrics and security headers for every HTTP response.

    Plain ASGI rather than ``@app.middleware("http")``: that form runs each
    request in its own task group behind memory streams, which measured
    ~0.4 ms per request here — about half the time of a trivial endpoint.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        client = request.client.host if request.client else "unknown"
        sdk_key = _sdk_api_key_hash(request) if request.url.path.startswith("/api/sdk/") else None
        key = f"{_rate_principal(request, client, sdk_key)}:{request.url.path}"
        now = time.monotonic()
        limit = 10 if request.url.path.startswith("/api/auth/") else 240
        current = 0
        if _redis_client:
            try:
                redis_key = f"qs:rate:{key}"
                current = int(await _redis_client.incr(redis_key))
                if current == 1:
                    await _redis_client.expire(redis_key, 60)
            except Exception:
                if ENVIRONMENT == "production":
                    await JSONResponse({"detail": "Rate-limit service unavailable"},
                                       status_code=503)(scope, receive, send)
                    return
                current = 0
        if not _redis_client or current == 0:
            window = _rate_window(key, now)
            while window and now - window[0] > _RATE_WINDOW_SECONDS:
                window.popleft()
            current = len(window) + 1
            window.append(now)
        if current > limit:
            await JSONResponse({"detail": "Rate limit exceeded"}, status_code=429,
                               headers={"Retry-After": "60"})(scope, receive, send)
            return
        started = time.perf_counter()

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                if sdk_key and message["status"] < 400 and sdk_key not in _rate_api_keys:
                    _remember(_rate_api_keys, sdk_key, None)
                metric_path = getattr(scope.get("route"), "path", request.url.path)
                HTTP_REQUESTS.labels(request.method, metric_path, str(message["status"])).inc()
                HTTP_LATENCY.labels(request.method, metric_path).observe(time.perf_counter() - started)
                headers = MutableHeaders(scope=message)
                headers["X-RateLimit-Limit"] = str(limit)
                headers["X-RateLimit-Remaining"] = str(max(0, limit - current))
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
                # CSP: scripts only from this origin (Three.js is self-hosted), no
                # unsafe-inline/eval, explicit form-action/object-src.
                #
                # style-src DOES need 'unsafe-inline': the frontend renders its Research
                # and Lab result panels (and a fair amount of index.html itself) with
                # inline `style="..."` attributes rather than a stylesheet — hundreds of
                # them, generated dynamically per request (metric cards, tables, mini
                # charts). Without 'unsafe-inline' here, browsers silently drop every one
                # of those styles, so most of Research/Lab renders unstyled/broken while
                # script-src stays fully locked down (inline styles cannot execute
                # script, so this doesn't reopen the XSS surface script-src closes).
                ws_policy = "wss:" if ENVIRONMENT == "production" else "wss: ws:"
                headers["Content-Security-Policy"] = (
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
            await send(message)

        await self.app(scope, receive, send_with_headers)


# Added last, so it wraps every other middleware: TrustedHost rejections and
# CORS preflights carry the security headers too.
app.add_middleware(SecurityHeadersAndRateLimit)


# Startup logic is in _lifespan() above (FastAPI lifespan context manager).


# --------------------------------------------------------------------------
# Auth dependency — MUST be defined before any route that uses Depends(get_current_user)
# --------------------------------------------------------------------------
def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    """Extract auth token from HttpOnly cookie (browser) or Authorization header (SDK)."""
    token = None
    from_cookie = False
    # Priority 1: HttpOnly cookie (browser sessions)
    cookie_token = request.cookies.get("qs_access")
    if cookie_token:
        token = cookie_token
        from_cookie = True
    # Priority 2: Bearer token (SDK / API clients)
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(401, "Missing authentication")
    payload = auth_service.decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    # Validate CSRF for state-changing browser requests.
    # The token must be bound to *this* session: a merely well-signed token
    # issued to some other account would otherwise satisfy the double-submit
    # check, which defeats the point of the pattern.
    if from_cookie and request.method not in ("GET", "HEAD", "OPTIONS"):
        # The token MUST come from the custom header. Falling back to the
        # qs_csrf cookie would validate the cookie against itself: a browser
        # attaches cookies to cross-site requests automatically, so an
        # attacker's forged form/fetch would satisfy the check with zero
        # knowledge of the token. Only a header — which cross-origin callers
        # cannot set without a passing CORS preflight — proves the request
        # originated from our own JS.
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token or not auth_service.verify_csrf_token(csrf_token, session_id=payload["sub"]):
            raise HTTPException(403, "Invalid or missing CSRF token")
    user = db.get(models.User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or inactive")
    return user


def require_api_scope(scope: str):
    """API key auth: supports both legacy bearer key and HMAC-signed requests (Item 7)."""
    async def dependency(
        request: Request,
        x_qs_api_key: str | None = Header(default=None),
        x_qs_key_id: str | None = Header(default=None),
        x_qs_timestamp: str | None = Header(default=None),
        x_qs_nonce: str | None = Header(default=None),
        x_qs_signature: str | None = Header(default=None),
        db: Session = Depends(get_db),
    ) -> models.ApiKey:
        # Path 1: HMAC-signed request (Item 7)
        if x_qs_key_id and x_qs_timestamp and x_qs_nonce and x_qs_signature:
            body = await request.body()
            key = integration_service.verify_hmac_request(
                db, x_qs_key_id, x_qs_timestamp, x_qs_nonce,
                x_qs_signature, request.method, str(request.url.path), body, scope,
            )
            if not key:
                raise HTTPException(403, "Invalid HMAC signature or insufficient scope")
            # Only an authenticated request may consume its nonce: consuming
            # first would let forged requests burn a client's valid nonces.
            # Redis when configured, otherwise the per-process store (dev).
            if not await redis_store.consume_api_nonce(_redis_client, x_qs_key_id, x_qs_nonce):
                raise HTTPException(409, "API request nonce already used")
            return key
        # Path 2: Legacy bearer API key
        if not x_qs_api_key:
            raise HTTPException(401, "Missing X-QS-API-KEY or HMAC signature headers")
        key = integration_service.verify_api_key(db, x_qs_api_key, scope)
        if not key:
            raise HTTPException(403, "Invalid API key or insufficient scope")
        return key
    return dependency


# --------------------------------------------------------------------------
# Health + metrics
# --------------------------------------------------------------------------
@app.get("/health/live", include_in_schema=False)
def liveness():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def readiness(db: Session = Depends(get_db)):
    # Blocking I/O: run it off the event loop so a slow or saturated database
    # cannot stall every other request this process is serving.
    await run_in_threadpool(db.execute, select(1))
    if _redis_client:
        await _redis_client.ping()
    return {"status": "ready", "database": "ok",
            "redis": "ok" if _redis_client else "not_configured"}


@app.get("/metrics", include_in_schema=False)
def metrics(user: models.User = Depends(get_current_user)):
    """Prometheus metrics — requires a valid bearer token to prevent public exposure."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)




# --------------------------------------------------------------------------
# Auth endpoints
# --------------------------------------------------------------------------
@app.post("/api/auth/register")
def register(req: schemas.RegisterRequest, request: Request, db: Session = Depends(get_db)):
    existing = db.execute(select(models.User).where(models.User.email == req.email)).scalar_one_or_none()

    # Pay the same Argon2id + PQC keygen cost whether or not the email is
    # taken. Login already does the equivalent (hash a dummy value when the
    # user doesn't exist, see the "Constant-time" comment below) so that
    # response latency can't be used to enumerate accounts; before this fix,
    # register's existing-email path returned near-instantly while the
    # new-account path spent ~100ms+ on this work, making the two paths
    # distinguishable purely by timing even if the 409 message were removed.
    password_hash = auth_service.hash_password(req.password)
    kem_pk, kem_sk, kem_ms = pqc.kem_keygen()
    dsa_pk, dsa_sk, dsa_ms = pqc.dsa_keygen()

    if existing:
        raise HTTPException(409, "Email already registered")

    user = models.User(email=req.email, password_hash=password_hash)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Two concurrent registrations for the same email can both pass the
        # SELECT check above before either commits; the DB's unique
        # constraint on users.email is the real guard. Without this handler
        # the second request surfaces as a raw 500 instead of the same 409
        # the first-checked path already returns.
        db.rollback()
        raise HTTPException(409, "Email already registered")
    db.refresh(user)

    # PQC identity keys were already generated above (kem_pk/kem_sk/dsa_pk/dsa_sk).
    db.add(models.KeyPair(user_id=user.id, algorithm="ML-KEM-768",
                           public_key=pqc.b64(kem_pk), private_key=security_service.protect_private_key(pqc.b64(kem_sk))))
    db.add(models.KeyPair(user_id=user.id, algorithm="ML-DSA-65",
                           public_key=pqc.b64(dsa_pk), private_key=security_service.protect_private_key(pqc.b64(dsa_sk))))
    db.commit()

    security_service.write_audit_log(db, user.id, "USER_REGISTERED", "user", user.id,
                                      {"email": user.email})

    # HIBP k-anonymity breach check (non-blocking warning only)
    hibp_count = auth_service.check_hibp(req.password)
    breach_warning = None
    if hibp_count > 0:
        breach_warning = (
            f"Your password has appeared {hibp_count:,} time(s) in known data breaches "
            "(HaveIBeenPwned). We strongly recommend choosing a different password before "
            "your first login."
        )

    return {
        "user_id": user.id, "email": user.email, "tier": user.tier,
        "created_at": user.created_at.isoformat(),
        "keygen_ms": {"ml_kem_768": round(kem_ms, 3), "ml_dsa_65": round(dsa_ms, 3)},
        "breach_warning": breach_warning,
    }


@app.post("/api/auth/login")
def login(req: schemas.LoginRequest, request: Request, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else None

    # Rate limit check — must happen BEFORE password verification
    is_locked, retry_after = auth_service.check_rate_limit(req.email, client_ip)
    if is_locked:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.execute(select(models.User).where(models.User.email == req.email)).scalar_one_or_none()

    # Constant-time: always call verify_password even if user not found
    dummy_hash = "$argon2id$v=19$m=65536,t=3,p=4$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    stored = user.password_hash if user else dummy_hash
    is_valid, needs_rehash = auth_service.verify_password(req.password, stored)

    if not user or not is_valid:
        lockout = auth_service.record_failed_attempt(req.email, client_ip)
        if lockout:
            raise HTTPException(
                status_code=429,
                detail=f"Account temporarily locked after repeated failures. Try again in {lockout} seconds.",
                headers={"Retry-After": str(lockout)},
            )
        raise HTTPException(401, "Invalid credentials")

    # Successful login — clear failure counter
    auth_service.clear_failed_attempts(req.email, client_ip)

    # Transparent hash upgrade: PBKDF2 → Argon2id
    if needs_rehash:
        user.password_hash = auth_service.hash_password(req.password)
        db.commit()

    # Item 1: Issue HttpOnly cookies instead of returning token in body
    access_token = auth_service.create_access_token(user.id, user.tier)
    refresh_raw, family_id = auth_service.create_refresh_token(db, user.id)
    csrf_token = auth_service.generate_csrf_token(user.id)

    security_service.write_audit_log(db, user.id, "USER_LOGIN", "user", user.id,
                                      {"ip": client_ip, "hash_upgraded": needs_rehash})

    # Cache refresh token in Redis for fast lookup. Dispatched onto the
    # application's own event loop (see redis_store.run_sync) rather than a
    # throwaway one — the async redis client's connections are bound to the
    # loop that created them, so running them elsewhere silently fails.
    if _redis_client:
        token_hash = hashlib.sha256(refresh_raw.encode()).hexdigest()
        redis_store.run_sync(
            redis_store.cache_refresh_token(_redis_client, token_hash, user.id, REFRESH_TOKEN_SECONDS)
        )

    response = JSONResponse(content={
        "token_type": "cookie",
        "expires_in": JWT_EXPIRE_SECONDS,
        "csrf_token": csrf_token,
        "user": {"user_id": user.id, "email": user.email, "tier": user.tier,
                 "beginner_mode": user.beginner_mode}
    })
    # Set HttpOnly access token cookie
    response.set_cookie(
        key="qs_access", value=access_token,
        httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN, max_age=JWT_EXPIRE_SECONDS, path="/",
    )
    # Set HttpOnly refresh token cookie
    response.set_cookie(
        key="qs_refresh", value=refresh_raw,
        httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN, max_age=REFRESH_TOKEN_SECONDS, path="/api/auth/",
    )
    # Set CSRF token as a readable cookie (double-submit pattern)
    response.set_cookie(
        key="qs_csrf", value=csrf_token,
        httponly=False, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN, max_age=JWT_EXPIRE_SECONDS, path="/",
    )
    return response


@app.post("/api/auth/refresh")
def refresh_session(request: Request, db: Session = Depends(get_db)):
    """Rotate the refresh token and issue a new access token via HttpOnly cookies."""
    refresh_raw = request.cookies.get("qs_refresh")
    if not refresh_raw:
        raise HTTPException(401, "Missing refresh token")

    result = auth_service.rotate_refresh_token(db, refresh_raw, _redis_client)
    if not result:
        # Clear stale cookies
        response = JSONResponse({"detail": "Invalid or expired refresh token"}, status_code=401)
        response.delete_cookie("qs_access", path="/")
        response.delete_cookie("qs_refresh", path="/api/auth/")
        response.delete_cookie("qs_csrf", path="/")
        return response

    new_refresh_raw, user_id, family_id = result
    user = db.get(models.User, user_id)
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or inactive")

    new_access = auth_service.create_access_token(user.id, user.tier)
    csrf_token = auth_service.generate_csrf_token(user.id)

    # Update Redis cache
    if _redis_client:
        old_hash = hashlib.sha256(refresh_raw.encode()).hexdigest()
        new_hash = hashlib.sha256(new_refresh_raw.encode()).hexdigest()
        redis_store.run_sync(redis_store.invalidate_refresh_token(_redis_client, old_hash))
        redis_store.run_sync(
            redis_store.cache_refresh_token(_redis_client, new_hash, user.id, REFRESH_TOKEN_SECONDS)
        )

    response = JSONResponse(content={
        "token_type": "cookie",
        "expires_in": JWT_EXPIRE_SECONDS,
        "csrf_token": csrf_token,
        "user": {"user_id": user.id, "email": user.email, "tier": user.tier,
                 "beginner_mode": user.beginner_mode},
    })
    response.set_cookie(
        key="qs_access", value=new_access,
        httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN, max_age=JWT_EXPIRE_SECONDS, path="/",
    )
    response.set_cookie(
        key="qs_refresh", value=new_refresh_raw,
        httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN, max_age=REFRESH_TOKEN_SECONDS, path="/api/auth/",
    )
    response.set_cookie(
        key="qs_csrf", value=csrf_token,
        httponly=False, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN, max_age=JWT_EXPIRE_SECONDS, path="/",
    )
    return response


@app.post("/api/auth/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    """Revoke refresh token and clear all auth cookies."""
    refresh_raw = request.cookies.get("qs_refresh")
    if refresh_raw:
        auth_service.revoke_refresh_token(db, refresh_raw)
        if _redis_client:
            token_hash = hashlib.sha256(refresh_raw.encode()).hexdigest()
            redis_store.run_sync(redis_store.invalidate_refresh_token(_redis_client, token_hash))
    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie("qs_access", path="/")
    response.delete_cookie("qs_refresh", path="/api/auth/")
    response.delete_cookie("qs_csrf", path="/")
    return response


@app.post("/api/auth/logout-all")
def logout_all(request: Request, user: models.User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    """Revoke every refresh-token family for this account ("log out everywhere"),
    e.g. after a stolen device or a shared-computer login. Requires the same
    authenticated session (JWT + CSRF) as any other state-changing endpoint —
    unlike plain /api/auth/logout, this also invalidates OTHER active sessions,
    not just the caller's own cookies."""
    client_ip = request.client.host if request.client else None
    revoked = auth_service.revoke_user_refresh_tokens(db, user.id)
    security_service.write_audit_log(db, user.id, "USER_LOGOUT_ALL", "user", user.id,
                                      {"ip": client_ip, "revoked_families": revoked})
    response = JSONResponse({"status": "logged_out_all", "revoked_sessions": revoked})
    response.delete_cookie("qs_access", path="/")
    response.delete_cookie("qs_refresh", path="/api/auth/")
    response.delete_cookie("qs_csrf", path="/")
    return response


@app.post("/api/auth/pqc-handshake")
def pqc_handshake(req: schemas.HandshakeRequest, user: models.User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    result = auth_service.perform_handshake(
        db, user.id, req.x25519_public_key, req.ml_kem_public_key, req.client_nonce,
        redis_client=_redis_client,
    )
    security_service.write_audit_log(db, user.id, "PQC_HANDSHAKE", "session",
                                      result["session_id"], {"kem_ms": result["kem_encapsulate_ms"]})
    return result


# --------------------------------------------------------------------------
# User settings endpoint
# --------------------------------------------------------------------------
@app.patch("/api/user/settings")
def update_user_settings(
    req: schemas.UserSettingsRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partial update of per-user preferences (PATCH semantics — only provided fields updated)."""
    if req.beginner_mode is not None:
        user.beginner_mode = req.beginner_mode
    db.commit()
    return {
        "user_id": user.id,
        "email": user.email,
        "beginner_mode": user.beginner_mode,
    }



# --------------------------------------------------------------------------
# Signal endpoints
# --------------------------------------------------------------------------
@app.get("/api/signals/latest")
def latest_signals(
    request: Request,
    assets: Optional[str] = Query(None, description="Comma-separated tickers to filter. Omit to use user watchlist."),
    user: models.User = Depends(get_current_user),
):
    """Return cached signals, filtered to the user's watchlist (or ?assets= override).
    Supports ETag / 304 Not Modified to minimise bandwidth at scale.
    """
    data = signal_engine.get_cached_signals()

    # Determine filter list: query param > user watchlist > exchange filter > all
    if assets:
        wanted = {t.strip().upper() for t in assets.split(",") if t.strip()}
    else:
        wl = _user_watchlist(user)
        # Also apply exchange filter if user has preferences set.
        # FIX: ASSET_EXCHANGE_MAP only covers the 20 preloaded TRACKED_ASSETS
        # (see get_cached_signals below), so a watchlisted ticker outside that
        # set used to silently resolve to "US" here regardless of its real
        # exchange — infer_exchange() is the same suffix-based fallback
        # search_assets()/compute_single_asset() already use for exactly this.
        preferred_ex = set(user.preferred_exchanges or ["US"])
        exchange_map = signal_engine.ASSET_EXCHANGE_MAP
        wanted = {t for t in wl if (exchange_map.get(t) or signal_engine.infer_exchange(t)) in preferred_ex} or set(wl)

    # FIX: the ETag previously hashed data.get("n_assets") — the SHARED
    # preloaded-cache's asset count, which is always 20 and never changes —
    # not the caller's actual filtered result. The comment above this used to
    # claim it "correctly invalidates... after watchlist edit", but it never
    # did: a browser's fetch() automatically sends If-None-Match with a
    # previously-seen ETag, so after a user edited their watchlist or
    # exchange preferences the server would still recognise the old tag
    # (unchanged, since it never depended on `wanted`) and incorrectly
    # answer 304 Not Modified, silently serving the pre-edit result until the
    # shared cache happened to regenerate on its own ~30-60s cycle. Hashing
    # the resolved `wanted` set alongside the cache generation timestamp
    # makes the tag change exactly when the personalized response would.
    tag = _etag(str(data.get("generated_at", "")) + "|" + ",".join(sorted(wanted)))
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers={"ETag": tag, "Cache-Control": "no-cache"})

    filtered_signals = [s for s in data.get("signals", []) if s.get("asset") in wanted]
    # FIX: get_cached_signals() only ever computes the 20 preloaded
    # TRACKED_ASSETS — any watchlisted ticker outside that set (the entire
    # point of the search/watchlist feature, which explicitly supports any
    # searchable ticker) was silently absent from `data["signals"]` and so
    # could never appear here, vanishing from the dashboard on every reload,
    # poll, or the next WebSocket push even though it was still watchlisted.
    # compute_single_asset() is the same on-demand path the search bar
    # already uses, cached for _ONDEMAND_TTL seconds, so this only pays the
    # live-fetch cost once per ticker per cache window.
    have = {s.get("asset") for s in filtered_signals}
    for ticker in wanted - have:
        extra = signal_engine.compute_single_asset(ticker)
        if extra:
            filtered_signals.append(extra)
    result = dict(data)
    result["signals"] = filtered_signals
    result["n_assets"] = len(filtered_signals)
    result["total_assets"] = data.get("n_assets", len(data.get("signals", [])))
    result["watchlist"] = sorted(wanted)

    return JSONResponse(content=result, headers={
        "ETag": tag,
        "Cache-Control": "no-cache",
        "Vary": "Authorization",
    })


@app.post("/api/signals/refresh")
def refresh_signals(user: models.User = Depends(get_current_user)):
    """Force-refresh (bypasses cache) — used by the dashboard's manual refresh button."""
    signal_engine.invalidate_cache()
    return signal_engine.get_cached_signals()


@app.get("/api/signals/asset/{ticker}")
def get_asset_signal(ticker: str, user: models.User = Depends(get_current_user)):
    """Real-time on-demand signal for a single ticker (fetched live from Yahoo Finance).

    Called when the user searches for a specific asset on the dashboard.
    Results are cached for 30s to prevent hammering Yahoo Finance on each keystroke.
    Returns 404 if the ticker is invalid or has insufficient price history.
    """
    clean = ticker.strip().upper()
    if not clean or len(clean) > 20:
        raise HTTPException(400, "Invalid ticker symbol")
    result = signal_engine.compute_single_asset(clean)
    if result is None:
        raise HTTPException(404, f"No price data found for '{clean}'. "
                            "The symbol may be delisted or incorrectly formatted.")
    return result


@app.get("/api/signals/search")
def search_assets(q: str = "", user: models.User = Depends(get_current_user)):
    """Return matching tickers. Searches preloaded assets first, then includes
    the raw query so users can search any world ticker."""
    q = q.strip().upper()
    # FIX M5: `len(q) < 1` is always False after .strip() when `not q` already
    # handles the empty-string case — the redundant len() check is removed.
    if not q:
        return {"results": [], "total": 0}
    matches = [t for t in signal_engine.TRACKED_ASSETS if q in t.upper()][:10]
    if q not in matches and len(q) >= 1:
        matches.append(q)
    results = []
    for t in matches:
        exch = signal_engine.ASSET_EXCHANGE_MAP.get(t) or signal_engine.infer_exchange(t)
        meta = signal_engine.ASSET_METADATA.get(t, {})
        results.append({
            "ticker": t, "exchange": exch,
            "company_name": meta.get("name", t),
            "sector": meta.get("sector", ""),
        })
    return {"results": results, "total": len(results), "query": q}


@app.get("/api/price/{ticker}")
def get_live_price_endpoint(ticker: str, user: models.User = Depends(get_current_user)):
    """Always-fresh price for any ticker - 5-second micro-cache.
    Used by the order price preview for current bid/ask rather than stale signal price."""
    clean = ticker.strip().upper()
    if not clean or len(clean) > 20:
        raise HTTPException(400, "Invalid ticker symbol")
    result = signal_engine.get_live_price(clean)
    if result is None:
        raise HTTPException(404, f"No live price data found for '{clean}'")
    return result


@app.get("/api/asset/info/{ticker}")
def get_asset_info_endpoint(ticker: str, user: models.User = Depends(get_current_user)):
    """Rich metadata: instrument type, exchange, market open status, trading features.
    Called by the order form on asset change so the form adapts dynamically."""
    clean = ticker.strip().upper()
    if not clean or len(clean) > 20:
        raise HTTPException(400, "Invalid ticker symbol")
    result = signal_engine.get_asset_info(clean)
    if result is None:
        raise HTTPException(404, f"No info available for '{clean}'")
    result.pop("_fetched_at", None)
    return result



# WebSocket limits. The per-user count lives in Redis when configured so the
# limit holds across gunicorn workers; the dict is the single-process fallback.
_ws_connections: dict[str, int] = defaultdict(int)
_WS_MAX_PER_USER = 3
_WS_PUSH_INTERVAL = 30
_WS_MAX_MESSAGE_SIZE = 65536  # 64KB
_WS_COUNTER_TTL = 120         # refreshed while connected; bounds leaks from crashed workers
# asyncio timers can fire up to one clock resolution early (~15.6 ms on
# Windows); close this close to expiry rather than push once more.
_WS_EXPIRY_GRACE = 0.25


async def _ws_acquire(user_id: str) -> tuple[bool, bool]:
    """Reserve a connection slot. Returns (acquired, counted_in_redis)."""
    if _redis_client:
        key = f"qs:ws:conn:{user_id}"
        try:
            count = int(await _redis_client.incr(key))
            await _redis_client.expire(key, _WS_COUNTER_TTL)
            if count > _WS_MAX_PER_USER:
                await _redis_client.decr(key)
                return False, False
            return True, True
        except Exception:
            if ENVIRONMENT == "production":
                return False, False
    if _ws_connections[user_id] >= _WS_MAX_PER_USER:
        return False, False
    _ws_connections[user_id] += 1
    return True, False


async def _ws_release(user_id: str, counted_in_redis: bool) -> None:
    if counted_in_redis:
        try:
            if int(await _redis_client.decr(f"qs:ws:conn:{user_id}")) < 0:
                await _redis_client.set(f"qs:ws:conn:{user_id}", 0, ex=_WS_COUNTER_TTL)
        except Exception:
            log.warning("could not release websocket slot for %s", user_id)
    else:
        _ws_connections[user_id] = max(0, _ws_connections[user_id] - 1)


def _ws_watchlist(user_id: str) -> list[str] | None:
    """The user's watchlist, or None when the account is gone or inactive.

    A session of its own per call: a socket stays open for minutes, and a
    session kept for its lifetime would hold a pooled connection the whole
    time, so a pool's worth of open dashboards left no connection for any
    HTTP request on the worker."""
    with SessionLocal() as db:
        user = db.get(models.User, user_id)
        return _user_watchlist(user) if user and user.is_active else None


@app.websocket("/api/signals/stream")
async def signal_stream(websocket: WebSocket):
    """Authenticated signal stream.

    * Authentication is the HttpOnly ``qs_access`` cookie only; tokens are
      never accepted in the subprotocol (they would land in proxy logs).
    * Origin must be an allowed origin.
    * At most 3 connections per user across all workers (Redis).
    * The socket is closed (4401) when the access token expires or the user
      is deactivated; the client reconnects with its refreshed cookie.
    * Client messages larger than 64KB close the socket (1009).
    * Every push carries a sequence number for gap detection.
    """
    origin = websocket.headers.get("origin")
    token = websocket.cookies.get("qs_access")
    payload = auth_service.decode_access_token(token) if token else None
    origin_ok = ("*" in CORS_ORIGINS) or (origin in CORS_ORIGINS)
    if not origin_ok or not payload:
        await websocket.close(code=4401)
        return
    user_id = payload["sub"]
    # Expiry as a monotonic deadline: the waits below run on the event loop's
    # monotonic clock, so checking against wall-clock time could disagree.
    deadline = time.monotonic() + (float(payload["exp"]) - time.time())
    acquired, in_redis = await _ws_acquire(user_id)
    if not acquired:
        await websocket.close(code=4429)  # too many connections
        return
    offered = [p.strip() for p in websocket.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    try:
        if await asyncio.to_thread(_ws_watchlist, user_id) is None:
            await websocket.close(code=4401)
            return
        await websocket.accept(subprotocol="qs" if "qs" in offered else None)
        seq = 0
        while True:
            if deadline - time.monotonic() <= _WS_EXPIRY_GRACE:
                await websocket.close(code=4401, reason="session expired")
                break
            # Re-read the user every cycle: deactivation and watchlist edits
            # take effect on the next push, not at the next reconnect.
            watchlist = await asyncio.to_thread(_ws_watchlist, user_id)
            if watchlist is None:
                await websocket.close(code=4401, reason="account inactive")
                break
            if in_redis:
                try:
                    await _redis_client.expire(f"qs:ws:conn:{user_id}", _WS_COUNTER_TTL)
                except Exception:
                    pass
            try:
                # get_cached_signals() can block on a yfinance download; keep it
                # off the event loop so other connections stay responsive.
                data = await asyncio.to_thread(signal_engine.get_cached_signals)
                wanted = set(watchlist)
                filtered = [s for s in data.get("signals", []) if s.get("asset") in wanted]
                # Watchlisted tickers outside the preloaded set are computed on
                # demand, as in /api/signals/latest.
                have = {s.get("asset") for s in filtered}
                for ticker in wanted - have:
                    extra = await asyncio.to_thread(signal_engine.compute_single_asset, ticker)
                    if extra:
                        filtered.append(extra)
                ws_payload = dict(data)
                ws_payload["signals"] = filtered
                ws_payload["n_assets"] = len(filtered)
                ws_payload["total_assets"] = data.get("n_assets", len(data.get("signals", [])))
                ws_payload["watchlist"] = sorted(wanted)
                ws_payload["sequence"] = seq
                seq += 1
                await websocket.send_json(ws_payload)
            except Exception:
                break
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=max(0.0, min(_WS_PUSH_INTERVAL, deadline - time.monotonic())),
                )
                if len(message.encode("utf-8")) > _WS_MAX_MESSAGE_SIZE:
                    await websocket.close(code=1009, reason="message too large")
                    break
            except asyncio.TimeoutError:
                pass  # no client message within the push interval: send the next update
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("signal stream error")
    finally:
        await _ws_release(user_id, in_redis)



# --------------------------------------------------------------------------
# Watchlist endpoints
# --------------------------------------------------------------------------
@app.get("/api/watchlist")
def get_watchlist(user: models.User = Depends(get_current_user)):
    """Return the current user's watchlist. Falls back to DEFAULT_WATCHLIST."""
    return {"watchlist": _user_watchlist(user), "default": not bool(user.watchlist)}


@app.put("/api/watchlist")
def set_watchlist(
    body: dict,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Replace the full watchlist. Body: {"watchlist": ["AAPL","MSFT", ...]}

    Accepts any ticker that Yahoo Finance can resolve — not limited to the 20
    preloaded stocks. Unknown tickers are validated via a fast yfinance probe,
    run concurrently across a small thread pool so a full 50-ticker save
    doesn't serialise into tens of seconds of network latency.
    """
    tickers = body.get("watchlist", [])
    if not isinstance(tickers, list):
        raise HTTPException(400, "watchlist must be a list")
    if len(tickers) > MAX_WATCHLIST_SIZE:
        raise HTTPException(400, f"Watchlist limited to {MAX_WATCHLIST_SIZE} assets")

    preloaded_set = set(signal_engine.PRELOADED_ASSETS)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        t = str(raw).upper().strip()
        if t and t not in seen:
            normalized.append(t)
            seen.add(t)

    to_probe = [t for t in normalized if t not in preloaded_set]
    probed_valid = _probe_tickers_parallel(to_probe)

    # Preserve the caller's requested ordering rather than "preloaded first".
    cleaned = [t for t in normalized if t in preloaded_set or t in probed_valid]

    if not cleaned:
        raise HTTPException(400, "No recognized tickers provided")
    
    db_user = db.get(models.User, user.id)
    db_user.watchlist = cleaned
    db.commit()
    return {"watchlist": cleaned, "count": len(cleaned)}


@app.post("/api/watchlist/{ticker}")
def add_to_watchlist(
    ticker: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a single ticker to the user's watchlist.
    
    Accepts any ticker that Yahoo Finance can resolve — not limited to preloaded assets.
    """
    import yfinance as yf
    ticker = ticker.upper().strip()
    current = _user_watchlist(user)
    if ticker in current:
        return {"watchlist": current, "message": f"{ticker} already in watchlist"}
    if len(current) >= MAX_WATCHLIST_SIZE:
        raise HTTPException(400, f"Watchlist limited to {MAX_WATCHLIST_SIZE} assets")
    
    # Validate: preloaded assets are always valid; others are probed via yfinance
    if ticker not in signal_engine.PRELOADED_ASSETS:
        try:
            probe = yf.Ticker(ticker)
            if getattr(probe.fast_info, 'last_price', None) is None:
                raise HTTPException(404, f"{ticker} is not a recognized tradable instrument")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(404, f"{ticker} is not a recognized tradable instrument")
    
    current.append(ticker)
    db_user = db.get(models.User, user.id)
    db_user.watchlist = current
    db.commit()
    return {"watchlist": current, "added": ticker}


@app.delete("/api/watchlist/{ticker}")
def remove_from_watchlist(
    ticker: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a single ticker from the user's watchlist."""
    ticker = ticker.upper()
    current = _user_watchlist(user)
    if ticker not in current:
        raise HTTPException(404, f"{ticker} not in watchlist")
    current = [t for t in current if t != ticker]
    if not current:
        raise HTTPException(400, "Cannot remove the last ticker — watchlist must have at least 1 asset")
    db_user = db.get(models.User, user.id)
    db_user.watchlist = current
    db.commit()
    return {"watchlist": current, "removed": ticker}

# --------------------------------------------------------------------------
# Trading endpoints
# --------------------------------------------------------------------------
@app.post("/api/trading/orders", status_code=201)
def place_order(req: schemas.OrderRequest, user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db),
                 idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    # Pydantic Literal types in schemas.py already enforce: quantity>0,
    # side in (buy/sell), order_type in (market/limit/stop/stop_limit),
    # time_in_force in (day/gtc/ioc) — these checks are now redundant.
    # Cross-field semantic checks (price required for conditional order types)
    # are not expressible in Literal and must stay here.
    # FIX M9/M10: use explicit `is None` — `not price` would reject a
    # valid limit_price of 0.0 (impossible for a real asset but defensive).
    if req.order_type in ("limit", "stop_limit") and req.limit_price is None:
        raise HTTPException(400, "limit_price required for limit orders")
    if req.order_type in ("stop", "stop_limit") and req.stop_price is None:
        raise HTTPException(400, "stop_price required for stop orders")

    # A completed retry is a read-only operation: return it before deriving
    # positions or retrieving a fresh market price. This prevents a previously
    # completed order from becoming unavailable merely because a downstream
    # market-data dependency is unavailable on the retry.
    payload_hash = order_security.request_hash(req.model_dump(exclude={"signature"}))
    cached_response = order_security.get_completed_idempotency_response(
        db, user.id, idempotency_key, payload_hash,
    )
    if cached_response is not None:
        return cached_response

    # One strict price observation drives both the risk checks and the fill
    # decision. Without a real price the order is refused — the paper broker
    # never substitutes a guessed price — and it never executes against a
    # stale one (a closed market's last close).
    quote_stale = False
    try:
        last_price = trading_service.get_last_price(req.asset)
    except trading_service.StaleMarketData as exc:
        if req.order_type == "market":
            raise HTTPException(409, f"{exc}. Market orders need a live market; "
                                     "a limit or stop order can rest until it opens.") from exc
        # A limit or stop order rests until the market trades again, and the
        # sweeper fills it only against a live quote. The stale price feeds
        # nothing but position valuation here: risk uses the order's own price.
        last_price, quote_stale = exc.price, True
    except trading_service.MarketDataUnavailable as exc:
        raise HTTPException(503, f"no current market price for {req.asset}; order not accepted") from exc

    positions = portfolio_service.get_positions_with_pnl(db, user.id)
    position = next((p for p in positions if p["asset"] == req.asset), None)
    held = position["quantity"] if position else 0.0
    buy_reserved, pending_sell = paper_broker.open_order_exposure(db, user.id)
    # Shares already promised to open sell orders cannot be sold twice.
    sellable = held - pending_sell.get(req.asset, 0.0)
    if req.side == "sell" and req.quantity > sellable + 1e-9:
        raise HTTPException(400, "sell quantity exceeds the available paper position "
                                 "(held quantity minus open sell orders)")

    account = paper_broker.account_snapshot(db, user.id, positions)
    reserve = paper_broker.reservation_micros(
        req.side, req.quantity, req.order_type, req.limit_price, req.stop_price, market_price=last_price,
    )
    # Buys are sized at the price they reserve (their worst-case cost).
    if req.limit_price is not None:
        price_for_risk = float(req.limit_price)
    elif req.stop_price is not None:
        price_for_risk = float(req.stop_price)
    else:
        price_for_risk = last_price
    equity = account["equity"]
    if req.side == "buy":
        cap = equity * PAPER_MAX_POSITION_FRACTION
        projected = ((position["market_value"] if position else 0.0)
                     + paper_broker.from_micros(buy_reserved.get(req.asset, 0))
                     + req.quantity * price_for_risk)
        if projected > cap + 1e-6:
            raise HTTPException(
                400,
                f"order would take the {req.asset} position to ${projected:,.2f}, above the "
                f"{PAPER_MAX_POSITION_FRACTION:.0%} per-asset limit (${cap:,.2f} of account equity)",
            )
        if reserve > paper_broker.to_micros(account["available_cash"]):
            raise HTTPException(
                400, f"insufficient available cash: order needs ${paper_broker.from_micros(reserve):,.2f}, "
                     f"${account['available_cash']:,.2f} available",
            )

    # Mandatory boundary between order construction and paper execution.
    # Exposure is today's positions at mark plus cash already committed to
    # open buy orders; passing _redis_client makes kill switches consistent
    # across worker processes.
    order_security.assert_risk_gate(
        user_id=user.id, asset=req.asset, side=req.side, quantity=req.quantity,
        price=price_for_risk, held_quantity=sellable, account_equity=equity,
        current_gross_exposure=account["gross_exposure"] + account["reserved_cash"],
        redis_client=_redis_client,
    )

    # ML-DSA-65 signs the canonical order before it is accepted — persisted for audit.
    supplied_envelope = [req.order_id, req.timestamp, req.expires_at, req.nonce]
    if any(value is not None for value in supplied_envelope) and not all(value is not None for value in supplied_envelope):
        raise HTTPException(400, "order_id, timestamp, expires_at, and nonce must be supplied together")
    if all(value is not None for value in supplied_envelope):
        order_id, timestamp, expires_at, nonce = req.order_id, req.timestamp, req.expires_at, req.nonce
    else:
        order_id = models.gen_uuid()
        timestamp, expires_at, nonce = order_security.make_development_envelope(order_id)
    order_security.validate_envelope(timestamp, expires_at, nonce)
    canonical = order_security.canonical_order(
        order_id=order_id, user_id=user.id, asset=req.asset, side=req.side,
        quantity=req.quantity, order_type=req.order_type, limit_price=req.limit_price,
        stop_price=req.stop_price, time_in_force=req.time_in_force,
        timestamp=timestamp, expires_at=expires_at, nonce=nonce,
    )
    # Idempotency binds the caller's business payload. In development the
    # server creates envelope fields, so hashing the generated nonce/order ID
    # would make an otherwise identical retry look different.
    signer_key_id, signature, signature_mode = order_security.verify_or_attest(
        db, user.id, canonical, req.key_id, req.signature,
    )
    # Verify first: invalid signatures must not consume a retry key.
    cached_response = order_security.reserve_idempotency(db, user.id, idempotency_key, payload_hash)
    if cached_response is not None:
        return cached_response
    if db.get(models.Trade, order_id):
        order_security.release_idempotency(db, user.id, idempotency_key)
        raise HTTPException(409, "order_id was already used")

    trade = models.Trade(
        id=order_id,
        user_id=user.id, asset=req.asset, side=req.side, quantity=req.quantity,
        order_type=req.order_type, limit_price=req.limit_price, time_in_force=req.time_in_force,
        stop_price=req.stop_price,
        status="PENDING", pqc_signature=signature,
    )
    # The cash reservation is an atomic compare-and-set committed together
    # with the order row: two concurrent orders can never both spend the
    # same available cash, and a failed insert rolls the reservation back.
    if not paper_broker.try_reserve(db, user.id, reserve):
        db.rollback()
        order_security.release_idempotency(db, user.id, idempotency_key)
        raise HTTPException(400, "insufficient available cash for this order")
    db.add(trade)
    db.add(models.OrderSecurityRecord(
        trade_id=order_id, user_id=user.id, canonical_order=canonical,
        request_hash=payload_hash, nonce=nonce,
        expires_at=dt.datetime.fromtimestamp(expires_at, tz=dt.timezone.utc),
        signer_key_id=signer_key_id, signature=signature, signature_mode=signature_mode,
    ))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        order_security.release_idempotency(db, user.id, idempotency_key)
        # Only the nonce's own uniqueness constraint means a replayed nonce;
        # anything else is a real failure and must surface as one.
        constraint = getattr(getattr(getattr(exc, "orig", None), "diag", None), "constraint_name", None)
        if isinstance(exc, IntegrityError) and constraint == "uq_order_security_user_nonce":
            raise HTTPException(409, "order nonce has already been used") from exc
        raise
    db.refresh(trade)

    if quote_stale:
        fill = {"status": "ACCEPTED", "filled_price": None}
    else:
        fill = trading_service.simulate_fill(
            trade.asset, trade.side, float(trade.quantity), trade.order_type,
            float(trade.limit_price) if trade.limit_price is not None else None,
            float(trade.stop_price) if trade.stop_price is not None else None,
            last_price=last_price,
        )
    reason = None
    if fill["status"] == "FILLED":
        reason = paper_broker.fill_order(db, trade, fill["filled_price"], reserved=reserve).reason
    elif trade.time_in_force == "ioc":
        paper_broker.close_order(db, trade, "EXPIRED", reserve)
    else:
        if not quote_stale and trade.order_type == "stop_limit" and paper_broker.stop_triggered(
                trade.side, float(trade.stop_price), last_price):
            # Triggered on arrival but not marketable: it rests as a limit order.
            trade.order_type = "limit"
            db.commit()
        paper_broker.accept_order(db, trade)
    db.refresh(trade)

    _record_order_outcome(db, trade, reason)
    response = _serialize_trade(trade)
    order_security.complete_idempotency(db, user.id, idempotency_key, response)
    return response


def _record_order_outcome(db: Session, trade: models.Trade, reason: str | None = None) -> None:
    """Audit-log an order state change and notify subscribed webhooks."""
    metadata = {"asset": trade.asset, "side": trade.side, "quantity": float(trade.quantity)}
    if trade.filled_price is not None:
        metadata["filled_price"] = float(trade.filled_price)
    if reason:
        metadata["reason"] = reason
    security_service.write_audit_log(db, trade.user_id, f"ORDER_{trade.status}", "trade", trade.id, metadata)
    event = {"FILLED": "order.filled", "REJECTED": "order.rejected"}.get(trade.status)
    if event:
        integration_service.emit_webhooks(db, trade.user_id, event, _serialize_trade(trade))


def _serialize_trade(t: models.Trade) -> dict:
    return {
        "order_id": t.id,
        "asset": t.asset,
        "side": t.side,
        "quantity": float(t.quantity),
        "order_type": t.order_type,
        # FIX: time_in_force was missing from the response — frontend showed blank
        "time_in_force": t.time_in_force,
        # FIX M4: use explicit `is not None` — SQLAlchemy Numeric columns return
        # Decimal objects; a falsy check would incorrectly treat 0.0 as null.
        "limit_price": float(t.limit_price) if t.limit_price is not None else None,
        "stop_price": float(t.stop_price) if t.stop_price is not None else None,
        "status": t.status,
        "filled_price": float(t.filled_price) if t.filled_price is not None else None,
        "pqc_signature_preview": (
            (t.pqc_signature or "")[:32] + "..." if t.pqc_signature else None
        ),
        "submitted_at": t.submitted_at.isoformat() if t.submitted_at else None,
        "filled_at": t.filled_at.isoformat() if t.filled_at else None,
    }


@app.get("/api/trading/orders")
def list_orders(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Read-only: resting orders are filled by the background order sweeper.
    trades = db.execute(
        select(models.Trade).where(models.Trade.user_id == user.id).order_by(models.Trade.submitted_at.desc())
    ).scalars().all()
    return [_serialize_trade(t) for t in trades]


@app.delete("/api/trading/orders/{order_id}")
def cancel_order(order_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    trade = db.get(models.Trade, order_id)
    if not trade or trade.user_id != user.id:
        raise HTTPException(404, "Order not found")
    if trade.status == "PENDING":
        raise HTTPException(409, "order is still being placed; retry the cancel")
    if trade.status != "ACCEPTED":
        raise HTTPException(400, f"Cannot cancel order in status {trade.status}")
    # Compare-and-set: loses cleanly to a fill that landed first.
    if not paper_broker.close_order(db, trade, "CANCELLED"):
        db.refresh(trade)
        raise HTTPException(409, f"order is no longer open (status {trade.status})")
    security_service.write_audit_log(db, user.id, "ORDER_CANCELLED", "trade", trade.id, {})
    integration_service.emit_webhooks(db, user.id, "order.cancelled", _serialize_trade(trade))
    return _serialize_trade(trade)


# --------------------------------------------------------------------------
# Enterprise SDK (scoped X-QS-API-KEY authentication)
# --------------------------------------------------------------------------
@app.get("/api/sdk/portfolio")
def sdk_portfolio(key: models.ApiKey = Depends(require_api_scope("read")), db: Session = Depends(get_db)):
    positions = portfolio_service.get_positions_with_pnl(db, key.user_id)
    return {"positions": positions,
            "account": paper_broker.account_snapshot(db, key.user_id, positions),
            "risk_metrics": portfolio_service.risk_metrics(db, key.user_id)}


@app.post("/api/sdk/orders", status_code=201)
def sdk_order(req: schemas.OrderRequest, key: models.ApiKey = Depends(require_api_scope("trade")),
              db: Session = Depends(get_db),
              idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    user = db.get(models.User, key.user_id)
    if not user or not user.is_active:
        raise HTTPException(401, "API-key user is inactive")
    return place_order(req, user, db, idempotency_key)


@app.get("/api/integrations/api-keys")
def list_api_keys(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.execute(select(models.ApiKey).where(models.ApiKey.user_id == user.id)
                      .order_by(models.ApiKey.created_at.desc())).scalars().all()
    return [{"id": k.id, "name": k.name, "prefix": k.key_prefix, "scopes": k.scopes,
             "is_revoked": k.is_revoked, "created_at": k.created_at.isoformat(),
             "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None} for k in keys]


@app.post("/api/integrations/api-keys", status_code=201)
def create_api_key(req: schemas.ApiKeyRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    raw, prefix, digest, hmac_secret = integration_service.generate_api_key()
    hmac_enc = integration_service.encrypt_hmac_secret(hmac_secret)
    key = models.ApiKey(user_id=user.id, name=req.name.strip(), key_prefix=prefix,
                        key_hash=digest, hmac_secret_encrypted=hmac_enc, scopes=req.scopes)
    db.add(key); db.commit(); db.refresh(key)
    security_service.write_audit_log(db, user.id, "API_KEY_CREATED", "api_key", key.id, {"scopes": req.scopes})
    return {"id": key.id, "name": key.name, "prefix": key.key_prefix, "scopes": key.scopes,
            "api_key": raw, "hmac_secret": hmac_secret}


@app.delete("/api/integrations/api-keys/{key_id}")
def revoke_api_key(key_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    key = db.get(models.ApiKey, key_id)
    if not key or key.user_id != user.id:
        raise HTTPException(404, "API key not found")
    key.is_revoked = True; db.commit()
    security_service.write_audit_log(db, user.id, "API_KEY_REVOKED", "api_key", key.id, {})
    return {"id": key.id, "is_revoked": True}


@app.get("/api/integrations/webhooks")
def list_webhooks(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    hooks = db.execute(select(models.Webhook).where(models.Webhook.user_id == user.id)
                       .order_by(models.Webhook.created_at.desc())).scalars().all()
    return [{"id": h.id, "url": h.url, "event_types": h.event_types, "is_active": h.is_active,
             "last_delivery_at": h.last_delivery_at.isoformat() if h.last_delivery_at else None} for h in hooks]


@app.post("/api/integrations/webhooks", status_code=201)
def create_webhook(req: schemas.WebhookRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not integration_service._is_public_https(req.url):
        raise HTTPException(422, "webhook host must resolve to a public HTTPS address")
    secret = secrets.token_urlsafe(32)
    hook = models.Webhook(user_id=user.id, url=req.url, secret_hash=integration_service.encrypt_secret(secret),
                          event_types=req.event_types)
    db.add(hook); db.commit(); db.refresh(hook)
    security_service.write_audit_log(db, user.id, "WEBHOOK_CREATED", "webhook", hook.id, {"events": req.event_types})
    return {"id": hook.id, "url": hook.url, "event_types": hook.event_types, "signing_secret": secret}


@app.delete("/api/integrations/webhooks/{hook_id}")
def delete_webhook(hook_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    hook = db.get(models.Webhook, hook_id)
    if not hook or hook.user_id != user.id:
        raise HTTPException(404, "Webhook not found")
    hook.is_active = False; db.commit()
    security_service.write_audit_log(db, user.id, "WEBHOOK_DISABLED", "webhook", hook.id, {})
    return {"id": hook.id, "is_active": False}


# --------------------------------------------------------------------------
# Portfolio endpoints
# --------------------------------------------------------------------------
@app.get("/api/portfolio/positions")
def positions(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return portfolio_service.get_positions_with_pnl(db, user.id)


@app.get("/api/portfolio/account")
def portfolio_account(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Server-side paper account: cash, reserved cash, buying power, equity, exposure."""
    return paper_broker.account_snapshot(db, user.id, portfolio_service.get_positions_with_pnl(db, user.id))


@app.get("/api/portfolio/risk-metrics")
def risk_metrics(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return portfolio_service.risk_metrics(db, user.id)


# --------------------------------------------------------------------------
# Visual strategy builder and historical backtesting
# --------------------------------------------------------------------------
@app.get("/api/strategies/templates")
def strategy_templates():
    return [{
        "id": "ma-crossover", "name": "Moving-average crossover",
        "description": "Buy when the fast average crosses above the slow average; sell on the reverse cross.",
        "fast_window": 20, "slow_window": 50,
    }]


@app.get("/api/strategies")
def list_strategies(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    strategies = db.execute(select(models.Strategy).where(models.Strategy.user_id == user.id)
                            .order_by(models.Strategy.created_at.desc())).scalars().all()
    return [{"id": s.id, "name": s.name, "assets": s.assets, "config": s.config,
             "is_active": s.is_active, "created_at": s.created_at.isoformat()} for s in strategies]


@app.post("/api/strategies", status_code=201)
def create_strategy(req: schemas.StrategyRequest, user: models.User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    config = {"template": "ma-crossover", "fast_window": req.fast_window, "slow_window": req.slow_window}
    strategy = models.Strategy(user_id=user.id, name=req.name.strip(), assets=[req.asset], config=config)
    db.add(strategy)
    db.commit(); db.refresh(strategy)
    security_service.write_audit_log(db, user.id, "STRATEGY_CREATED", "strategy", strategy.id,
                                     {"asset": req.asset, **config})
    return {"id": strategy.id, "name": strategy.name, "assets": strategy.assets, "config": strategy.config}


@app.post("/api/backtests", status_code=202)
def run_backtest(req: schemas.BacktestRequest, user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """Queue the dashboard moving-average backtest; the finished job's result
    carries the stored backtest ``id``."""
    if req.slow_window <= req.fast_window:
        raise HTTPException(400, "slow_window must be larger than fast_window")
    return _queue_research(db, user, "ma_backtest", req)


@app.get("/api/backtests")
def list_backtests(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(select(models.Backtest).where(models.Backtest.user_id == user.id)
                      .order_by(models.Backtest.created_at.desc()).limit(20)).scalars().all()
    return [{"id": row.id, "created_at": row.created_at.isoformat(), **(row.result_json or {})} for row in rows]


@app.get("/api/portfolio/export")
def export_portfolio(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """CSV export for positions, metrics and audit evidence.

    Values are quoted with double-quotes to prevent CSV injection from asset
    names or float representations containing commas.
    """
    def _csv_row(*values) -> str:
        return ",".join(f'"{str(v)}"' for v in values)

    positions_data = portfolio_service.get_positions_with_pnl(db, user.id)
    metrics = portfolio_service.risk_metrics(db, user.id)
    rows = [_csv_row("section", "asset", "quantity", "avg_entry_price",
                     "current_price", "market_value", "unrealized_pnl")]
    for p in positions_data:
        rows.append(_csv_row(
            "position", p["asset"], p["quantity"], p["avg_entry_price"],
            p["current_price"], p["market_value"], p["unrealized_pnl"],
        ))
    for name, value in metrics.items():
        if name != "equity_curve":
            rows.append(_csv_row("metric", name, value))
    return Response(
        "\n".join(rows) + "\n", media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=quantumsentinel-portfolio.csv"},
    )


# --------------------------------------------------------------------------
# Research Engine (Phase 1) — Advanced backtesting, walk-forward, stat tests
# --------------------------------------------------------------------------

def _queue_job(db: Session, user: models.User, kind: str, params: dict,
               same_as: tuple[str, ...] = ()) -> JSONResponse:
    """Queue a research job and answer 202 with the job to poll.

    Research runs in a worker process (backend/worker.py), never in the API
    process: a heavy computation here would hold a request thread and a
    database connection for its whole duration and stall trading requests.
    """
    try:
        job = research_jobs.enqueue(db, user.id, kind, params, same_as=same_as)
    except research_jobs.QueueFull as exc:
        raise HTTPException(429, str(exc)) from exc
    return JSONResponse(research_jobs.view(db, job), status_code=202,
                        headers={"Location": f"/api/research/jobs/{job.id}"})


def _queue_research(db: Session, user: models.User, kind: str, req) -> JSONResponse:
    return _queue_job(db, user, kind, req.model_dump())


@app.post("/api/research/backtest", status_code=202)
def advanced_backtest(req: schemas.AdvancedBacktestRequest,
                      user: models.User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """Queue an advanced backtest (realistic execution costs, multiple
    strategies, comprehensive risk metrics)."""
    return _queue_research(db, user, "advanced_backtest", req)


@app.post("/api/research/walk-forward", status_code=202)
def walk_forward_validation(req: schemas.WalkForwardRequest,
                            user: models.User = Depends(get_current_user),
                            db: Session = Depends(get_db)):
    """Queue walk-forward validation (rolling/expanding windows,
    out-of-sample aggregation)."""
    return _queue_research(db, user, "walk_forward", req)


@app.post("/api/research/stat-test", status_code=202)
def statistical_tests(req: schemas.StatTestRequest,
                      user: models.User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """Queue statistical tests on strategy returns: t-test, bootstrap,
    permutation, Deflated Sharpe Ratio, and multiple-testing corrections."""
    returns = None
    trial_family = req.trial_family
    if req.returns:
        returns = list(req.returns)
    elif req.backtest_id:
        row = db.get(models.Backtest, req.backtest_id)
        if not row or row.user_id != user.id:
            raise HTTPException(404, "Backtest not found")
        rj = row.result_json or {}
        trial_family = trial_family or rj.get("trial_family")
        if rj.get("daily_returns_net"):
            # Full-resolution daily returns; the stored equity curve is
            # downsampled for charting and has the wrong periodicity.
            returns = list(rj["daily_returns_net"])
        else:
            curve = rj.get("equity_curve_net") or rj.get("equity_curve", [])
            if len(curve) > 2:
                returns = [(curve[i] - curve[i - 1]) / curve[i - 1] if curve[i - 1] else 0
                           for i in range(1, len(curve))]
    if returns is None or len(returns) < 5:
        raise HTTPException(422, "Need at least 5 return observations")
    if len(returns) > schemas.STAT_TEST_MAX_RETURNS:
        raise HTTPException(422, f"At most {schemas.STAT_TEST_MAX_RETURNS:,} return observations "
                                 f"can be tested; got {len(returns):,}")
    if not all(isinstance(r, (int, float)) and math.isfinite(r) for r in returns):
        raise HTTPException(422, "Every return observation must be a finite number")

    # The Deflated Sharpe discount uses at least every configuration the
    # server evaluated for this research family (counted now, when the test
    # is queued), whatever is declared.
    server_counted = research_trials.count(db, user.id, trial_family) if trial_family else 0
    n_trials = max(req.n_strategies_tested, server_counted)
    return _queue_job(db, user, "stat_test", {
        "returns": [float(r) for r in returns], "n_trials": n_trials,
        "declared": req.n_strategies_tested, "server_counted": server_counted,
        "trial_family": trial_family, "n_bootstrap": req.n_bootstrap,
        "n_permutations": req.n_permutations,
    })


# --------------------------------------------------------------------------
# Research Engine Phases 2-4 — queued research jobs
# --------------------------------------------------------------------------

@app.post("/api/research/alpha", status_code=202)
def alpha_research_endpoint(req: schemas.AlphaResearchRequest,
                             user: models.User = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    """Queue alpha research: IC, Rank IC, IC decay, hit rate, quintile
    analysis, factor turnover."""
    return _queue_research(db, user, "alpha", req)


@app.post("/api/research/factor-model", status_code=202)
def factor_model_endpoint(req: schemas.FactorModelRequest,
                           user: models.User = Depends(get_current_user),
                           db: Session = Depends(get_db)):
    """Queue a Fama-MacBeth factor model with Newey-West inference."""
    return _queue_research(db, user, "factor_model", req)


@app.post("/api/research/correlation", status_code=202)
def correlation_endpoint(req: schemas.CorrelationRequest,
                          user: models.User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """Queue multi-method correlation analysis (Pearson, Spearman, EWMA,
    Ledoit-Wolf, OAS, PCA)."""
    return _queue_research(db, user, "correlation", req)


@app.post("/api/research/optimize", status_code=202)
def portfolio_optimize_endpoint(req: schemas.PortfolioOptRequest,
                                  user: models.User = Depends(get_current_user),
                                  db: Session = Depends(get_db)):
    """Queue multi-method portfolio optimisation with efficient frontier."""
    return _queue_research(db, user, "optimize", req)


@app.post("/api/research/event-backtest", status_code=202)
def event_backtest_endpoint(req: schemas.EventBacktestRequest,
                             user: models.User = Depends(get_current_user),
                             db: Session = Depends(get_db)):
    """Queue an event-driven backtest (1-bar execution delay, commissions,
    spread, slippage, borrow costs)."""
    return _queue_research(db, user, "event_backtest", req)


@app.post("/api/research/regime", status_code=202)
def regime_detection_endpoint(req: schemas.RegimeDetectionRequest,
                               user: models.User = Depends(get_current_user),
                               db: Session = Depends(get_db)):
    """Queue market-regime detection (Gaussian HMM, volatility percentile,
    SMA trend)."""
    return _queue_research(db, user, "regime", req)


@app.post("/api/research/neutral-strategy", status_code=202)
def neutral_strategy_endpoint(req: schemas.NeutralStrategyRequest,
                               user: models.User = Depends(get_current_user),
                               db: Session = Depends(get_db)):
    """Queue a dollar-neutral cross-sectional long/short strategy."""
    return _queue_research(db, user, "neutral_strategy", req)


@app.post("/api/research/pairs-trading", status_code=202)
def pairs_trading_endpoint(req: schemas.PairsTradingRequest,
                            user: models.User = Depends(get_current_user),
                            db: Session = Depends(get_db)):
    """Queue pairs trading (Engle-Granger cointegration, Kalman hedge
    ratio, Z-score signals)."""
    if req.asset_y == req.asset_x:
        raise HTTPException(422, "asset_y and asset_x must be different")
    return _queue_research(db, user, "pairs_trading", req)


@app.post("/api/research/latency-benchmark", status_code=202)
def latency_benchmark_endpoint(req: schemas.LatencyBenchmarkRequest,
                                user: models.User = Depends(get_current_user),
                                db: Session = Depends(get_db)):
    """Queue an end-to-end research pipeline latency benchmark."""
    return _queue_research(db, user, "latency_benchmark", req)


@app.post("/api/research/report", status_code=202)
def research_report_endpoint(req: schemas.ReportRequest,
                              user: models.User = Depends(get_current_user),
                              db: Session = Depends(get_db)):
    """Queue the full 7-section quant research report."""
    return _queue_research(db, user, "report", req)


@app.get("/api/research/jobs")
def list_research_jobs(limit: int = Query(20, ge=1, le=100),
                       user: models.User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """The caller's most recent research jobs, newest first (without results)."""
    return [research_jobs.view(db, job, include_result=False)
            for job in research_jobs.list_for_user(db, user.id, limit)]


@app.get("/api/research/jobs/{job_id}")
def get_research_job(job_id: str, user: models.User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    """Status of one of the caller's research jobs; ``result`` once it
    succeeded, ``error`` (status_code, detail) if it failed."""
    job = research_jobs.get_for_user(db, user.id, job_id)
    if job is None:
        raise HTTPException(404, "Research job not found")
    return research_jobs.view(db, job)


@app.post("/api/research/jobs/{job_id}/cancel")
def cancel_research_job(job_id: str, user: models.User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Cancel a queued job, or ask the worker to stop a running one."""
    job = research_jobs.request_cancel(db, user.id, job_id)
    if job is None:
        raise HTTPException(404, "Research job not found")
    return research_jobs.view(db, job, include_result=False)


@app.get("/api/research/queue")
def research_queue_status(user: models.User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """Operator view of the research queue: depth, age and live workers."""
    if not _is_admin(user):
        raise HTTPException(403, "queue status requires an operator role")
    return research_jobs.queue_stats(db)


@app.get("/api/research/cpp-status", status_code=200)
def cpp_status_endpoint(user: models.User = Depends(get_current_user)):
    """Return whether the C++ performance extension is loaded.

    Reports: CPP_AVAILABLE bool, version, and kernel names.
    If CPP_AVAILABLE is False, all kernels fall back to NumPy.
    """
    from .services.cpp_ext import CPP_AVAILABLE
    status = {
        "cpp_available": CPP_AVAILABLE,
        "kernels": ["rolling_corr", "hmm_forward", "backtest_loop"],
        "description": (
            "C++ kernels active (hardware-accelerated)"
            if CPP_AVAILABLE
            else "NumPy fallback active — build cpp/ extension to enable C++ kernels"
        ),
    }
    if CPP_AVAILABLE:
        try:
            import _qs_fast  # type: ignore[import]
            status["extension_file"] = getattr(_qs_fast, "__file__", "unknown")
        except Exception:
            pass
    return status


# --------------------------------------------------------------------------
# Market Microstructure endpoints
# --------------------------------------------------------------------------

def _synthetic_tick(price: float) -> float:
    """Tick grid for a synthetic book seeded at ``price``."""
    return 0.0001 if price < 1 else 0.01


def _seed_bar(price: float, spread_pct: float, volume: float) -> dict:
    # Simulation clocks start at 0 so a given seed reproduces the same result.
    return {"timestamp": 0.0, "open": price, "high": price * (1 + spread_pct),
            "low": price * (1 - spread_pct), "close": price, "volume": volume}


@app.get("/api/microstructure/snapshot/{ticker}", status_code=200)
def microstructure_snapshot(ticker: str, levels: int = Query(10, ge=1, le=50),
                            user: models.User = Depends(get_current_user)):
    """SYNTHETIC order-book snapshot seeded at the latest price.

    The book is generated, not observed venue depth; analytics (OBI,
    microprice, spread, depth) describe that synthetic book.
    """
    from .services.market_microstructure import generate_synthetic_l2, build_snapshot_from_events, snapshot_to_dict
    price_data = signal_engine.get_live_price(ticker)
    if not price_data or not price_data.get("price"):
        raise HTTPException(404, f"No price data for {ticker}")
    price = float(price_data["price"])
    tick = _synthetic_tick(price)
    events = generate_synthetic_l2([_seed_bar(price, 0.002, 50_000)], levels=levels,
                                   seed=int(price * 100), tick_size=tick)
    d = snapshot_to_dict(build_snapshot_from_events(events, levels=levels, tick_size=tick))
    d.update({"symbol": ticker.upper(), "data_source": "synthetic", "reference_price": price, "tick_size": tick})
    return d


@app.post("/api/microstructure/analytics", status_code=200)
def microstructure_analytics(request_body: dict, user: models.User = Depends(get_current_user)):
    """Compute microstructure analytics on provided order-book data.

    Accepts ``bids`` and ``asks`` arrays and returns OBI, microprice,
    spread, depth, and microprice deviation.
    """
    from .services.market_microstructure import (
        OrderBookSnapshot, PriceLevel,
        compute_order_book_imbalance, compute_microprice, compute_spread,
        compute_spread_bps, compute_depth, compute_mid_price,
        compute_microprice_deviation,
    )
    try:
        bids = sorted((PriceLevel(float(b["price"]), float(b["size"])) for b in request_body.get("bids", [])),
                      key=lambda lvl: -lvl.price)
        asks = sorted((PriceLevel(float(a["price"]), float(a["size"])) for a in request_body.get("asks", [])),
                      key=lambda lvl: lvl.price)
        levels = int(request_body.get("levels", 5))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, "bids/asks must be lists of {price, size} numbers") from exc
    if any(lvl.price <= 0 or lvl.size < 0 for lvl in bids + asks):
        raise HTTPException(422, "prices must be positive and sizes non-negative")
    if bids and asks and bids[0].price >= asks[0].price:
        raise HTTPException(422, "book is crossed or locked (best bid >= best ask)")
    snap = OrderBookSnapshot(timestamp=time.time(), bids=bids, asks=asks)
    return {
        "mid_price": compute_mid_price(snap),
        "spread": compute_spread(snap),
        "spread_bps": compute_spread_bps(snap),
        "microprice": compute_microprice(snap),
        "microprice_deviation_bps": compute_microprice_deviation(snap),
        "obi": compute_order_book_imbalance(snap, levels=levels),
        "depth": compute_depth(snap, levels=levels),
    }


@app.post("/api/microstructure/replay", status_code=200)
def microstructure_replay(req: schemas.MicrostructureReplayRequest,
                          user: models.User = Depends(get_current_user)):
    """Replay SYNTHETIC L2 (generated from the given OHLCV bars) through the
    OBI-momentum strategy.

    With ``execute`` the strategy trades through the paper exchange —
    latency preset, queue-aware matching, fills, portfolio and execution
    analytics are returned. Deterministic for a given seed.
    """
    from .services.l2_event_replay import L2EventStream, ReplayConfig, replay_session, obi_momentum_strategy
    from .services.latency_model import LATENCY_PRESETS
    from .services.paper_exchange import PaperExchange
    if req.latency_preset not in LATENCY_PRESETS:
        raise HTTPException(422, f"latency_preset must be one of {sorted(LATENCY_PRESETS)}")
    stream = L2EventStream.from_synthetic([b.model_dump() for b in req.bars], seed=req.seed,
                                          events_per_bar=req.events_per_bar, tick_size=req.tick_size)
    config = ReplayConfig(
        snapshot_interval=req.snapshot_interval, warmup_events=req.warmup_events,
        tick_size=req.tick_size, order_quantity=req.order_quantity, order_style=req.order_style,
        cancel_after_events=req.cancel_after_events,
    )
    exchange = None
    if req.execute:
        exchange = PaperExchange(symbol="SYNTHETIC", tick_size=req.tick_size,
                                 latency_ms=LATENCY_PRESETS[req.latency_preset].total_latency_ms)

    def strategy(snap, trades):
        return obi_momentum_strategy(snap, trades, obi_threshold=req.obi_threshold)

    result = replay_session(stream, strategy_fn=strategy, config=config, exchange=exchange).to_dict()
    result["latency_preset"] = req.latency_preset
    return result


# --------------------------------------------------------------------------
# Paper Exchange (research simulation) endpoints
# --------------------------------------------------------------------------

_SEED_EVENTS = 60


@app.post("/api/exchange/order", status_code=200)
def exchange_submit_order(req: schemas.ExchangeSimulationRequest,
                          user: models.User = Depends(get_current_user)):
    """Simulate one paper order against a SYNTHETIC L2 book.

    A book is generated around the latest price, the order is submitted
    (after ``latency_ms``), and ``flow_events`` further synthetic events are
    replayed so a resting order can fill through its queue. Deterministic
    for a given seed. This is a research simulation: it never touches the
    user's paper account, and starting cash is fixed server-side.
    """
    from .services.market_microstructure import TradeSide, generate_synthetic_l2
    from .services.order_book import OrderType, TimeInForce
    from .services.paper_exchange import PaperExchange, PaperPosition, TradingMode

    price_data = signal_engine.get_live_price(req.symbol)
    if not price_data or not price_data.get("price"):
        raise HTTPException(404, f"No price data for {req.symbol}")
    price = float(price_data["price"])
    tick = req.tick_size or _synthetic_tick(price)
    events = generate_synthetic_l2([_seed_bar(price, 0.003, 100_000)], seed=req.seed,
                                   events_per_bar=_SEED_EVENTS + req.flow_events, tick_size=tick)
    exchange = PaperExchange(symbol=req.symbol, latency_ms=req.latency_ms, tick_size=tick)
    for event in events[:_SEED_EVENTS]:
        exchange.on_market_event(event)
    if req.initial_position > 0:
        basis = exchange.book.mid_price or price
        exchange.positions[req.symbol] = PaperPosition(symbol=req.symbol, quantity=req.initial_position,
                                                       avg_entry_price=basis)
    order = exchange.submit_order(
        side=TradeSide(req.side), quantity=req.quantity, order_type=OrderType(req.order_type),
        limit_price=req.limit_price, stop_price=req.stop_price,
        time_in_force=TimeInForce(req.time_in_force),
    )
    for event in events[_SEED_EVENTS:]:
        exchange.on_market_event(event)
    exchange.advance_to(exchange.current_time + req.latency_ms / 1000.0)
    return {
        "data_source": "synthetic",
        "reference_price": price,
        "tick_size": tick,
        "seed": req.seed,
        "order": order.to_dict(),
        "order_events": [e.to_dict() for e in exchange.event_log if e.order_id == order.order_id],
        "fills": [f.to_dict() for f in exchange.fill_history],
        "portfolio": exchange.portfolio_summary(),
        "exchange_stats": exchange.exchange_stats(),
        "execution": exchange.execution_report(),
        "trading_mode": TradingMode.PAPER.value,
    }


@app.get("/api/exchange/book/{ticker}", status_code=200)
def exchange_book(ticker: str, user: models.User = Depends(get_current_user)):
    """SYNTHETIC order book seeded at the latest price (not venue depth)."""
    from .services.market_microstructure import generate_synthetic_l2, build_snapshot_from_events, snapshot_to_dict
    price_data = signal_engine.get_live_price(ticker)
    if not price_data or not price_data.get("price"):
        raise HTTPException(404, f"No price data for {ticker}")
    price = float(price_data["price"])
    tick = _synthetic_tick(price)
    events = generate_synthetic_l2([_seed_bar(price, 0.003, 100_000)], levels=10,
                                   seed=int(price * 100), events_per_bar=100, tick_size=tick)
    d = snapshot_to_dict(build_snapshot_from_events(events, levels=10, tick_size=tick))
    d.update({"symbol": ticker.upper(), "data_source": "synthetic", "reference_price": price, "tick_size": tick})
    return d


# --------------------------------------------------------------------------
# Execution Analytics endpoints
# --------------------------------------------------------------------------

@app.post("/api/execution/analysis", status_code=200)
def execution_analysis(request_body: dict, user: models.User = Depends(get_current_user)):
    """Compute implementation shortfall decomposition for an order."""
    from .services.execution_analytics import compute_implementation_shortfall
    return compute_implementation_shortfall(
        decision_price=float(request_body.get("decision_price", 0)),
        arrival_price=float(request_body.get("arrival_price", 0)),
        execution_vwap=float(request_body.get("execution_vwap", 0)),
        side=request_body.get("side", "BUY"),
        quantity=float(request_body.get("quantity", 0)),
        spread=float(request_body.get("spread", 0)),
        fees=float(request_body.get("fees", 0)),
    )


@app.post("/api/execution/capacity", status_code=200)
def execution_capacity(request_body: dict, user: models.User = Depends(get_current_user)):
    """Capacity analysis across capital sizes."""
    from .services.execution_analytics import compute_capacity_analysis
    results = request_body.get("results_by_capital", {})
    # Convert string keys to float
    typed_results = {float(k): v for k, v in results.items()}
    return compute_capacity_analysis(typed_results)


# --------------------------------------------------------------------------
# Experiment Registry endpoints
# --------------------------------------------------------------------------

@app.post("/api/experiments/create", status_code=201)
def experiment_create(request_body: dict, user: models.User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """Create a new experiment with full provenance tracking."""
    from .services.experiment_registry import PersistentExperimentRegistry
    registry = PersistentExperimentRegistry(db, user.id)
    exp = registry.create(
        strategy_id=request_body.get("strategy_id", ""),
        strategy_version=request_body.get("strategy_version", "1.0"),
        dataset_id=request_body.get("dataset_id", ""),
        dataset=request_body.get("dataset"),
        parameters=request_body.get("parameters"),
        random_seed=request_body.get("random_seed", 42),
        execution_model=request_body.get("execution_model", "LOB_QUEUE_V2"),
        latency_model=request_body.get("latency_model", "zero"),
    )
    return exp.to_dict()


@app.get("/api/experiments/{experiment_id}", status_code=200)
def experiment_get(experiment_id: str, user: models.User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Get experiment details with signed manifest."""
    from .services.experiment_registry import PersistentExperimentRegistry
    registry = PersistentExperimentRegistry(db, user.id)
    exp = registry.get(experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return exp.to_dict()


@app.post("/api/experiments/{experiment_id}/run", status_code=202)
def experiment_run(experiment_id: str, user: models.User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Queue execution of a platform strategy on the experiment's stored
    inputs; the job then records the result and signs the manifest (once).
    Submitting again while the run is queued or running returns its job."""
    from .services.experiment_registry import ExperimentError, PersistentExperimentRegistry
    try:
        params = PersistentExperimentRegistry(db, user.id).run_job_params(experiment_id)
    except ExperimentError as exc:
        raise HTTPException(422, str(exc)) from exc
    if params is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return _queue_job(db, user, "experiment_run", params, same_as=("experiment_id",))


@app.get("/api/experiments/{experiment_id}/manifest", status_code=200)
def experiment_manifest(experiment_id: str, user: models.User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """The signed manifest exactly as signed, verified against the key that signed it."""
    from .services.experiment_registry import PersistentExperimentRegistry
    manifest = PersistentExperimentRegistry(db, user.id).get_manifest(experiment_id)
    if manifest is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return manifest


@app.post("/api/experiments/{experiment_id}/replay", status_code=202)
def experiment_replay(experiment_id: str, request_body: dict, user: models.User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """Queue deterministic verification of an experiment.

    Always verifies inputs: the (optionally supplied) dataset, parameters and
    seed are hashed and compared with the recorded ones. For a completed
    platform-executable experiment the job then re-executes the strategy from
    the stored inputs and compares the result hash (true replay).
    """
    from .services.experiment_registry import PersistentExperimentRegistry
    params = PersistentExperimentRegistry(db, user.id).replay_job_params(experiment_id, request_body)
    if params is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return _queue_job(db, user, "experiment_replay", params)


@app.post("/api/experiments/{experiment_id}/validate", status_code=202)
def experiment_validate(experiment_id: str, user: models.User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Queue the integrity and deployment gates (re-executing the strategy);
    the experiment becomes VALIDATED only if every gate passes. Submitting
    again while validation is queued or running returns its job."""
    from .services.experiment_registry import ExperimentError, PersistentExperimentRegistry
    try:
        params = PersistentExperimentRegistry(db, user.id).validate_job_params(experiment_id)
    except ExperimentError as exc:
        raise HTTPException(409, str(exc)) from exc
    if params is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return _queue_job(db, user, "experiment_validate", params, same_as=("experiment_id",))


@app.post("/api/experiments/{experiment_id}/approve", status_code=200)
def experiment_approve(experiment_id: str, user: models.User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """Approve a VALIDATED experiment for paper deployment.

    Operator role required; the owner cannot approve their own experiment.
    """
    from .services.experiment_registry import ExperimentError, approve_experiment
    if not _is_admin(user):
        raise HTTPException(403, "approving experiments requires an operator role")
    try:
        exp = approve_experiment(db, experiment_id, user.id)
    except LookupError as exc:
        raise HTTPException(404, f"Experiment {experiment_id} not found") from exc
    except ExperimentError as exc:
        raise HTTPException(409, str(exc)) from exc
    security_service.write_audit_log(db, user.id, "EXPERIMENT_APPROVED", "experiment", experiment_id,
                                     {"result_hash": exp.result_hash})
    return exp.to_dict()


# --------------------------------------------------------------------------
# Latency Model endpoints
# --------------------------------------------------------------------------

@app.get("/api/latency/presets", status_code=200)
def latency_presets(user: models.User = Depends(get_current_user)):
    """List all available latency presets."""
    from .services.latency_model import LATENCY_PRESETS
    return {name: model.to_dict() for name, model in LATENCY_PRESETS.items()}


# --------------------------------------------------------------------------
# Security endpoints
# --------------------------------------------------------------------------
@app.get("/api/security/health")
def security_health(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    health = security_service.key_health(db, user.id)
    n_keys = len(health["keys"])
    n_green = sum(1 for k in health["keys"] if k["status"] == "GREEN")
    # Cast to int: Python's round(x, 0) returns float (e.g. 75.0), which would
    # render as "75.0%" in the frontend animateCounter display.
    quantum_safety_score = int(round(100 * (n_green / n_keys))) if n_keys else 100
    # Make created_at timezone-aware if it was stored as a naive datetime
    # to prevent TypeError when subtracting from an aware datetime.
    created_at = security_service.server_identity.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=dt.timezone.utc)
    key_age_days = (dt.datetime.now(dt.timezone.utc) - created_at).days
    return {
        **health,
        "quantum_safety_score": quantum_safety_score,
        "fips_203_compliant": True,
        "fips_204_compliant": True,
        "execution_venue": "internal_paper_broker",
        "server_dsa_key_age_days": key_age_days,
    }


@app.get("/api/security/audit-log")
def audit_log(limit: int = 50, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    logs = db.execute(
        select(models.AuditLog).where(models.AuditLog.user_id == user.id)
        .order_by(models.AuditLog.created_at.desc()).limit(limit)
    ).scalars().all()

    # FIX M6: N+1 query — build a user-id → email map in ONE query instead of
    # calling db.get(User, uid) inside the loop (= 50 extra DB reads for 50 logs).
    unique_user_ids = {l.user_id for l in logs if l.user_id}
    if unique_user_ids:
        user_rows = db.execute(
            select(models.User.id, models.User.email).where(
                models.User.id.in_(unique_user_ids)
            )
        ).all()
        _uid_to_email: dict[str, str] = {row.id: row.email for row in user_rows}
    else:
        _uid_to_email = {}

    return [{
        "id": l.id, "action": l.action, "resource_type": l.resource_type,
        "resource_id": l.resource_id, "metadata": l.metadata_json,
        "user_email": _uid_to_email.get(l.user_id) if l.user_id else None,
        "signature_preview": (l.pqc_signature or "")[:24] + "...",
        "verified": security_service.verify_audit_log(db, l.id),
        "created_at": l.created_at.isoformat(),
    } for l in logs]


@app.get("/api/security/compliance-report")
def compliance_report(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Machine-readable evidence bundle for a DORA/SEC review workflow."""
    health = security_service.key_health(db, user.id)
    logs = db.execute(select(models.AuditLog).where(models.AuditLog.user_id == user.id)
                      .order_by(models.AuditLog.created_at.desc()).limit(100)).scalars().all()
    verified = sum(security_service.verify_audit_log(db, log.id) for log in logs)
    return {
        "report_type": "QuantumSentinel paper-trading compliance evidence",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frameworks": ["FIPS 203", "FIPS 204", "DORA evidence mapping", "SEC Rule 33-11216 review aid"],
        "scope_notice": "Reference-app evidence only; not a certification or legal compliance determination.",
        "key_health": health,
        "audit_log": {"entries_reviewed": len(logs), "signatures_verified": verified,
                      "all_verified": verified == len(logs)},
    }


@app.get("/api/security/audit-chain")
def audit_chain(user: models.User = Depends(get_current_user), db: Session = Depends(get_db),
                full: bool = False):
    """Verify the tamper-evident audit chain (operators only: it spans every
    user's events). By default only links added since this worker last
    verified the chain are checked; ``?full=true`` checks every link."""
    if not _is_admin(user):
        raise HTTPException(403, "Operator role required")
    status = security_service.audit_chain_status(db, full=full)
    return {**status, "verified_at": dt.datetime.now(dt.timezone.utc).isoformat()}


@app.post("/api/security/rotate-keys")
def rotate_keys(req: schemas.RotateKeysRequest, user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    old_keys = db.execute(
        select(models.KeyPair).where(
            models.KeyPair.user_id == user.id, models.KeyPair.algorithm == req.algorithm,
            models.KeyPair.is_active.is_(True),
        )
    ).scalars().all()

    if req.algorithm == "ML-KEM-768":
        pk, sk, ms = pqc.kem_keygen()
    elif req.algorithm == "ML-DSA-65":
        pk, sk, ms = pqc.dsa_keygen()
    else:
        raise HTTPException(400, "Unsupported algorithm")

    rotation_count = (old_keys[0].rotation_count + 1) if old_keys else 0
    new_key = models.KeyPair(user_id=user.id, algorithm=req.algorithm, public_key=pqc.b64(pk),
                              private_key=security_service.protect_private_key(pqc.b64(sk)), rotation_count=rotation_count)
    db.add(new_key)
    for k in old_keys:
        k.is_active = False
        k.revoked_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(new_key)

    security_service.write_audit_log(db, user.id, "KEY_ROTATED", "key_pair", new_key.id, {
        "algorithm": req.algorithm, "reason": req.reason, "keygen_ms": round(ms, 3),
    })
    integration_service.emit_webhooks(db, user.id, "key.rotated", {
        "algorithm": req.algorithm, "rotation_count": rotation_count, "key_pair_id": new_key.id,
    })

    return {"new_key_pair_id": new_key.id, "algorithm": req.algorithm,
            "rotation_count": rotation_count, "keygen_ms": round(ms, 3)}


# --------------------------------------------------------------------------
# Server signing key history (Item 8)
# --------------------------------------------------------------------------
@app.get("/api/security/server-keys")
def server_signing_keys(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return current and historical server signing key fingerprints with activation times."""
    keys = db.execute(
        select(models.ServerSigningKey).order_by(models.ServerSigningKey.activated_at.desc())
    ).scalars().all()
    return {
        "current_fingerprint": security_service.server_identity.fingerprint,
        "trusted_fingerprint": TRUSTED_SERVER_DSA_FINGERPRINT,
        "keys": [{
            "key_id": k.key_id,
            "algorithm": k.algorithm,
            "fingerprint": k.fingerprint,
            "status": k.status,
            "activated_at": k.activated_at.isoformat() if k.activated_at else None,
            "retired_at": k.retired_at.isoformat() if k.retired_at else None,
        } for k in keys],
    }


# --------------------------------------------------------------------------
# Kill switch admin endpoints (Item 6)
# --------------------------------------------------------------------------
def _is_admin(user: models.User) -> bool:
    """Operator privilege comes from a provisioned role, never from the email."""
    return (user.role or "user") in OPERATOR_ROLES


@app.post("/api/risk/kill-switch")
async def manage_kill_switch(
    body: dict,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Set or clear a trading kill switch.

    Platform-wide (`global`) and per-asset switches halt trading for every
    user, so they are restricted to configured operators. A `user`-scoped
    switch is self-service: anyone may halt their OWN trading, but targeting
    another account is an operator action.
    """
    scope = body.get("scope", "global")
    identifier = body.get("identifier")
    enabled = body.get("enabled", True)
    if scope not in {"global", "user", "asset"}:
        raise HTTPException(400, "scope must be global, user, or asset")
    if scope == "user":
        if identifier is None:
            identifier = user.id
        if identifier != user.id and not _is_admin(user):
            raise HTTPException(403, "not permitted to set a kill switch for another user")
    elif not _is_admin(user):
        raise HTTPException(403, f"{scope} kill switches require operator privileges")
    if _redis_client:
        await order_security.set_kill_switch_async(_redis_client, scope, identifier, enabled)
    else:
        order_security.set_kill_switch(scope, identifier, enabled)
    action = "KILL_SWITCH_SET" if enabled else "KILL_SWITCH_CLEARED"
    security_service.write_audit_log(db, user.id, action, "risk", None,
                                      {"scope": scope, "identifier": identifier})
    return {"scope": scope, "identifier": identifier, "enabled": enabled}


@app.get("/api/risk/kill-switch")
async def list_kill_switches_endpoint(user: models.User = Depends(get_current_user)):
    """List active kill switches.

    Non-operators see only switches that affect them (global, their own
    asset-agnostic user switch); the raw list is withheld because a
    `user`-scoped entry discloses another account's id.
    """
    if _redis_client:
        from .services import redis_store
        switches = await redis_store.list_kill_switches(_redis_client)
    else:
        switches = [{"scope": s, "identifier": i} for s, i in order_security._KILL_SWITCHES]
    if not _is_admin(user):
        switches = [
            s for s in switches
            if s.get("scope") != "user" or s.get("identifier") == user.id
        ]
    return {"kill_switches": switches}


# --------------------------------------------------------------------------
# Enterprise SDK / algorithm registry
# --------------------------------------------------------------------------
@app.get("/api/algorithms")
def algorithms():
    return pqc.ALGORITHM_REGISTRY


@app.get("/api/meta")
def meta():
    exchanges_with_status = {}
    for key, info in EXCHANGE_REGISTRY.items():
        exchanges_with_status[key] = {**info, "market_status": _market_status(key)}
    return {
        "product": "QuantumSentinel", "version": APP_VERSION,
        "fips_standards": ["FIPS 203 (ML-KEM-768)", "FIPS 204 (ML-DSA-65)"],
        "tracked_assets": signal_engine.TRACKED_ASSETS,
        "asset_exchange_map": signal_engine.ASSET_EXCHANGE_MAP,
        "exchanges": exchanges_with_status,
        "execution_venue": "internal_paper_broker",
    }


@app.get("/api/exchanges")
def get_exchanges():
    """Return all supported exchanges with live market-hours status."""
    result = {}
    for key, info in EXCHANGE_REGISTRY.items():
        result[key] = {**info, "market_status": _market_status(key),
                       "asset_count": sum(1 for e in signal_engine.ASSET_EXCHANGE_MAP.values() if e == key)}
    return result


@app.get("/api/preferences")
def get_preferences(user: models.User = Depends(get_current_user)):
    """Return the user's regional preferences (exchanges + timezone)."""
    return {
        "preferred_exchanges": user.preferred_exchanges or ["US"],
        "user_timezone": user.user_timezone or "UTC",
        "watchlist": _user_watchlist(user),
    }


@app.put("/api/preferences")
def set_preferences(
    body: dict,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the user's regional preferences."""
    db_user = db.get(models.User, user.id)
    if "preferred_exchanges" in body:
        exchanges = [str(e).upper() for e in body["preferred_exchanges"] if str(e).upper() in EXCHANGE_REGISTRY]
        if not exchanges:
            raise HTTPException(400, "At least one valid exchange must be selected")
        db_user.preferred_exchanges = exchanges
    if "user_timezone" in body:
        tz_val = str(body["user_timezone"])
        try:
            ZoneInfo(tz_val)  # validate
            db_user.user_timezone = tz_val
        except Exception:
            raise HTTPException(400, f"Invalid timezone: {tz_val}")
    db.commit()
    return {
        "preferred_exchanges": db_user.preferred_exchanges,
        "user_timezone": db_user.user_timezone,
    }


# --------------------------------------------------------------------------
# Frontend static hosting + SPA catch-all
# --------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    # Serve static files with a custom response class that adds cache headers
    from fastapi.responses import HTMLResponse
    from starlette.staticfiles import StaticFiles as _StaticFiles

    class CachedStaticFiles(_StaticFiles):
        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            # Cache CSS/JS for 24 hours in browser; revalidate in between
            if hasattr(response, "headers") and path.endswith((".css", ".js")):
                response.headers["Cache-Control"] = "public, max-age=86400, stale-while-revalidate=3600"
            return response

    app.mount("/assets", CachedStaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    # Assets are cached for a day, so index.html names each one with a hash
    # of its content: a deploy that changes app.js (e.g. to follow an API
    # change) reaches every browser on its next page load, never a day late.
    _VERSIONED_ASSETS = ("styles.css", "vendor/three-global.js", "bg3d.js", "app.js")
    _index_cache: tuple = (None, "")

    def _index_html() -> str:
        global _index_cache
        files = [FRONTEND_DIR / "index.html"] + [FRONTEND_DIR / name for name in _VERSIONED_ASSETS]
        key = tuple(f.stat().st_mtime_ns if f.exists() else None for f in files)
        if _index_cache[0] != key:
            html = files[0].read_text(encoding="utf-8")
            for name, path in zip(_VERSIONED_ASSETS, files[1:]):
                if path.exists():
                    version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
                    html = html.replace(f'"/assets/{name}"', f'"/assets/{name}?v={version}"')
            _index_cache = (key, html)
        return _index_cache[1]

    def _index_response() -> HTMLResponse:
        return HTMLResponse(_index_html(), headers={"Cache-Control": "no-cache"})

    @app.get("/")
    def index():
        return _index_response()

    @app.get("/robots.txt", include_in_schema=False)
    def robots():
        robots_path = FRONTEND_DIR / "robots.txt"
        if robots_path.exists():
            return FileResponse(str(robots_path), media_type="text/plain")
        return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /api/\n")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        ico_path = FRONTEND_DIR / "favicon.ico"
        if ico_path.exists():
            return FileResponse(str(ico_path), media_type="image/x-icon")
        # HTTPException is already imported at the top of the module
        raise HTTPException(404, "Not found")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        """SPA catch-all: any unknown route serves index.html so the frontend
        router handles navigation rather than returning a JSON 404.
        Excludes /api, /assets, /health, /metrics paths which are handled above."""
        excluded = ("api/", "assets/", "health/", "health", "metrics")
        if any(path.startswith(prefix) for prefix in excluded):
            raise HTTPException(404, "Not found")
        return _index_response()
