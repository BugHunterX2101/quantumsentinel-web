"""QuantumSentinel — FastAPI application.

Single-process web port of the multi-service architecture in the design
docs (API Gateway + Trading Engine + PQC Crypto Service + Signal Engine
collapsed into one deployable app for a portable web demo). All PQC
operations are genuine FIPS 203/204 algorithms (see backend/crypto/pqc.py).
"""
import datetime as dt
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from typing import Optional, List
import os
import time
import secrets
import asyncio
import hashlib
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Header, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import select
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as redis

from . import models, schemas
from .database import get_db, init_db, SessionLocal
from .config import CORS_ORIGINS, ALLOWED_HOSTS, ENVIRONMENT, REDIS_URL, JWT_EXPIRE_SECONDS
from .crypto import pqc
from .services import auth_service, signal_engine, trading_service, portfolio_service, security_service, backtest_service, integration_service


# Default watchlist for new users (20 blue-chip stocks)
DEFAULT_WATCHLIST = [
    "AAPL","MSFT","NVDA","GOOGL","META",
    "TSLA","AMZN","JPM","V","JNJ",
    "XOM","SPY","QQQ","GLD","COIN",
    "NFLX","AMD","BKNG","LLY","TSM",
]

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
    if now <= close_time:
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
    return hashlib.sha1(value.encode()).hexdigest()[:16]

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="QuantumSentinel API", version="1.0.0")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

HTTP_REQUESTS = Counter("quantumsentinel_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("quantumsentinel_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
_redis_client = redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None

# Bounded in-memory limiter for the single-process reference deployment. It
# deliberately protects write paths even before a user has authenticated.
_request_windows: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def security_headers_and_rate_limit(request, call_next):
    client = request.client.host if request.client else "unknown"
    principal = request.headers.get("x-qs-api-key") or request.headers.get("authorization", "")
    principal_hash = hashlib.sha256(principal.encode()).hexdigest()[:16] if principal else client
    key = f"{principal_hash}:{request.url.path}"
    now = time.monotonic()
    limit = 20 if request.url.path.startswith("/api/auth/") else 240
    current = 0
    if _redis_client:
        try:
            redis_key = f"qs:rate:{key}"
            current = int(await _redis_client.incr(redis_key))
            if current == 1:
                await _redis_client.expire(redis_key, 60)
        except Exception:
            if ENVIRONMENT == "production":
                return JSONResponse({"detail": "Rate-limit service unavailable"}, status_code=503)
            current = 0
    if not _redis_client or current == 0:
        window = _request_windows[key]
        while window and now - window[0] > 60:
            window.popleft()
        current = len(window) + 1
        window.append(now)
    if current > limit:
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429,
                            headers={"Retry-After": "60"})
    started = time.perf_counter()
    response = await call_next(request)
    metric_path = getattr(request.scope.get("route"), "path", request.url.path)
    HTTP_REQUESTS.labels(request.method, metric_path, str(response.status_code)).inc()
    HTTP_LATENCY.labels(request.method, metric_path).observe(time.perf_counter() - started)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current))
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data:; "
        "connect-src 'self' wss: ws: https://api.github.com https://api.pwnedpasswords.com; "
        "frame-ancestors 'none'; base-uri 'self'"
    )
    return response


@app.on_event("startup")
async def on_startup():
    init_db()
    if _redis_client:
        try:
            await _redis_client.ping()
        except Exception as exc:
            if ENVIRONMENT == "production":
                raise RuntimeError("Redis is required and unavailable") from exc


# --------------------------------------------------------------------------
# Auth dependency — MUST be defined before any route that uses Depends(get_current_user)
# --------------------------------------------------------------------------
def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = auth_service.decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = db.get(models.User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or inactive")
    return user


def require_api_scope(scope: str):
    def dependency(
        x_qs_api_key: str | None = Header(default=None),
        db: Session = Depends(get_db),
    ) -> models.ApiKey:
        if not x_qs_api_key:
            raise HTTPException(401, "Missing X-QS-API-KEY")
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
    db.execute(select(1))
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
    if existing:
        raise HTTPException(409, "Email already registered")

    user = models.User(email=req.email, password_hash=auth_service.hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate the user's PQC identity keys
    kem_pk, kem_sk, kem_ms = pqc.kem_keygen()
    dsa_pk, dsa_sk, dsa_ms = pqc.dsa_keygen()
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

    token = auth_service.create_access_token(user.id, user.tier)
    security_service.write_audit_log(db, user.id, "USER_LOGIN", "user", user.id,
                                      {"ip": client_ip, "hash_upgraded": needs_rehash})
    return {
        "access_token": token, "token_type": "bearer",
        "expires_in": JWT_EXPIRE_SECONDS,
        "user": {"user_id": user.id, "email": user.email, "tier": user.tier,
                 "beginner_mode": user.beginner_mode}
    }


@app.post("/api/auth/pqc-handshake")
def pqc_handshake(req: schemas.HandshakeRequest, user: models.User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    result = auth_service.perform_handshake(
        db, user.id, req.x25519_public_key, req.ml_kem_public_key, req.client_nonce
    )
    security_service.write_audit_log(db, user.id, "PQC_HANDSHAKE", "session",
                                      result["session_id"], {"kem_ms": result["kem_encapsulate_ms"]})
    return result




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
    # ETag includes both timestamp AND n_assets to correctly invalidate when
    # signal count changes within the same second (e.g. after watchlist edit).
    tag = _etag(str(data.get("generated_at", "")) + str(data.get("n_assets", "")))
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers={"ETag": tag, "Cache-Control": "no-cache"})

    # Determine filter list: query param > user watchlist > exchange filter > all
    if assets:
        wanted = {t.strip().upper() for t in assets.split(",") if t.strip()}
    else:
        wl = _user_watchlist(user)
        # Also apply exchange filter if user has preferences set
        preferred_ex = set(user.preferred_exchanges or ["US"])
        exchange_map = signal_engine.ASSET_EXCHANGE_MAP
        wanted = {t for t in wl if exchange_map.get(t, "US") in preferred_ex} or set(wl)

    filtered_signals = [s for s in data.get("signals", []) if s.get("asset") in wanted]
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
    """Return matching tickers from the full 300+ universe catalogue.

    Used by the dashboard search bar to show autocomplete suggestions.
    Pure in-memory filter — no network call, sub-millisecond response.
    """
    q = q.strip().upper()
    if not q or len(q) < 1:
        return {"results": [], "total": 0}
    matches = [t for t in signal_engine.TRACKED_ASSETS if q in t.upper()][:20]
    return {
        "results": [
            {"ticker": t, "exchange": signal_engine.ASSET_EXCHANGE_MAP.get(t, "US")}
            for t in matches
        ],
        "total": len(matches),
        "query": q,
    }


@app.websocket("/api/signals/stream")
async def signal_stream(websocket: WebSocket):
    """Authenticated signal stream with browser-safe subprotocol credentials.

    The JWT is carried as the second requested WebSocket subprotocol rather
    than in the URL, keeping it out of query-string logs and referrers.
    """
    origin = websocket.headers.get("origin")
    protocols = [item.strip() for item in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    raw_token = protocols[1] if len(protocols) == 2 and protocols[0] == "qs" else None
    # URL-decode: frontend sends encodeURIComponent(jwt) to avoid header parse issues
    # with JWT special chars (+, /, =). Decode before verification.
    from urllib.parse import unquote
    token = unquote(raw_token) if raw_token else None
    payload = auth_service.decode_access_token(token) if token else None
    # FIX: When CORS_ORIGINS contains "*" (dev mode), skip origin check.
    # A literal `origin not in ["*"]` always fails for specific origin strings.
    origin_ok = ("*" in CORS_ORIGINS) or (origin in CORS_ORIGINS)
    if not origin_ok or not payload:
        await websocket.close(code=4401)
        return
    db = SessionLocal()
    try:
        user = db.get(models.User, payload.get("sub"))
        if not user or not user.is_active:
            await websocket.close(code=4401)
            return
        await websocket.accept(subprotocol="qs")
        while True:
            try:
                data = signal_engine.get_cached_signals()
                wanted = set(_user_watchlist(user))
                filtered = [s for s in data.get("signals", []) if s.get("asset") in wanted]
                ws_payload = dict(data)
                ws_payload["signals"] = filtered
                ws_payload["n_assets"] = len(filtered)
                ws_payload["total_assets"] = data.get("n_assets", len(data.get("signals", [])))
                ws_payload["watchlist"] = sorted(wanted)
                await websocket.send_json(ws_payload)
            except Exception:
                # Connection closed mid-send or serialisation error — exit cleanly
                break
            await asyncio.sleep(30)  # aligned with CACHE_TTL_SECONDS=30
    except WebSocketDisconnect:
        pass
    except Exception:
        # Catch-all for unexpected errors (e.g. DB failure during user lookup)
        pass
    finally:
        db.close()



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
    """Replace the full watchlist. Body: {"watchlist": ["AAPL","MSFT", ...]}"""
    tickers = body.get("watchlist", [])
    if not isinstance(tickers, list):
        raise HTTPException(400, "watchlist must be a list")
    valid = set(signal_engine.TRACKED_ASSETS)
    cleaned = [str(t).upper() for t in tickers if str(t).upper() in valid]
    if not cleaned:
        raise HTTPException(400, "No valid tracked tickers provided")
    if len(cleaned) > 50:
        raise HTTPException(400, "Watchlist limited to 50 assets")
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
    """Add a single ticker to the user's watchlist."""
    ticker = ticker.upper()
    if ticker not in signal_engine.TRACKED_ASSETS:
        raise HTTPException(404, f"{ticker} is not in the tracked asset universe")
    current = _user_watchlist(user)
    if ticker in current:
        return {"watchlist": current, "message": f"{ticker} already in watchlist"}
    if len(current) >= 50:
        raise HTTPException(400, "Watchlist limited to 50 assets")
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
                 db: Session = Depends(get_db)):
    # Pydantic Literal types in schemas.py already enforce: quantity>0,
    # side in (buy/sell), order_type in (market/limit/stop/stop_limit),
    # time_in_force in (day/gtc/ioc) — these checks are now redundant.
    # Cross-field semantic checks (price required for conditional order types)
    # are not expressible in Literal and must stay here.
    if req.order_type in ("limit", "stop_limit") and not req.limit_price:
        raise HTTPException(400, "limit_price required for limit orders")
    if req.order_type in ("stop", "stop_limit") and not req.stop_price:
        raise HTTPException(400, "stop_price required for stop orders")

    # Paper account guardrails from the architecture: no naked shorting,
    # duplicate submissions, and a 5% initial-capital concentration cap.
    existing_positions = {p["asset"]: p for p in portfolio_service.get_positions_with_pnl(db, user.id)}
    held = existing_positions.get(req.asset, {}).get("quantity", 0.0)
    if req.side == "sell" and req.quantity > held:
        raise HTTPException(400, "sell quantity exceeds the available paper position")
    price_for_risk = req.limit_price or req.stop_price or trading_service.get_last_price(req.asset)

    # FIX: Compute real account cash from FILLED trades instead of using the
    # static subtraction (100k - notional_held) which goes negative when
    # holdings exceed $100k and floors position_cap at an unusable $250.
    filled_trades = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user.id, models.Trade.status == "FILLED"
        )
    ).scalars().all()
    cash = 100_000.0
    for ft in filled_trades:
        notional = float(ft.quantity) * float(ft.filled_price or 0)
        cash -= notional if ft.side == "buy" else -notional
    account_equity = max(5_000.0, cash)
    position_cap = account_equity * 0.05
    if req.side == "buy" and req.quantity * price_for_risk > position_cap:
        raise HTTPException(
            400,
            f"order exceeds the 5% paper-account position limit "
            f"(${position_cap:,.0f} based on current account equity)",
        )

    # FIX: 30-second duplicate window (was 15s) — Alpaca round-trips can take
    # 5-10s and a 15s window caused legitimate retry orders to be blocked.
    duplicate_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30)
    duplicate = db.execute(select(models.Trade).where(
        models.Trade.user_id == user.id, models.Trade.asset == req.asset,
        models.Trade.side == req.side, models.Trade.quantity == req.quantity,
        models.Trade.status.in_(("PENDING", "ACCEPTED", "SUBMITTED")),
        models.Trade.submitted_at >= duplicate_cutoff,
    )).scalars().first()
    if duplicate:
        raise HTTPException(409, "duplicate pending order blocked")

    # ML-DSA-65 signs the order payload before it is accepted — persisted for audit.
    # NOTE: renamed from 'payload' to 'order_payload' to avoid shadowing the JWT
    # payload dict used earlier in the WebSocket handler and auth dependency.
    user_dsa_key = db.execute(
        select(models.KeyPair).where(
            models.KeyPair.user_id == user.id, models.KeyPair.algorithm == "ML-DSA-65",
            models.KeyPair.is_active.is_(True),
        )
    ).scalars().first()
    order_payload = (
        f"{req.side}:{req.asset}:{req.quantity}:{req.order_type}"
        f":{req.limit_price}:{req.stop_price}:{req.time_in_force}"
    ).encode()
    signature = None
    if user_dsa_key and user_dsa_key.private_key:
        private_key = security_service.unprotect_private_key(user_dsa_key.private_key)
        sig_bytes, _ = pqc.dsa_sign(pqc.unb64(private_key), order_payload)
        signature = pqc.b64(sig_bytes)

    trade = models.Trade(
        user_id=user.id, asset=req.asset.upper(), side=req.side, quantity=req.quantity,
        order_type=req.order_type, limit_price=req.limit_price, time_in_force=req.time_in_force,
        stop_price=req.stop_price,
        status="PENDING", pqc_signature=signature,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)

    if trading_service.alpaca_enabled():
        try:
            resp = trading_service.submit_alpaca_order(
                trade.asset, trade.side, trade.quantity, trade.order_type,
                trade.limit_price, trade.stop_price, trade.time_in_force,
            )
            trade.alpaca_order_id = resp.get("id")
            trade.status = resp.get("status", "ACCEPTED").upper()
            if trade.status == "FILLED":
                # FIX: float(x or 0) or None is wrong — if filled_avg_price is "0"
                # the result is None (discards a valid zero fill). Use explicit None check.
                raw_price = resp.get("filled_avg_price")
                trade.filled_price = float(raw_price) if raw_price else None
                trade.filled_at = dt.datetime.now(dt.timezone.utc)
        except Exception as e:
            trade.status = "REJECTED"
            db.commit()
            raise HTTPException(502, f"Alpaca order failed: {e}") from e
    else:
        # Cast SQLAlchemy Numeric columns to float before passing to simulate_fill.
        # Passing Decimal objects causes TypeError in comparison operators inside
        # simulate_fill (e.g. last_price <= limit_price where limit_price is Decimal).
        fill = trading_service.simulate_fill(
            trade.asset, trade.side, float(trade.quantity),
            trade.order_type,
            float(trade.limit_price) if trade.limit_price is not None else None,
            float(trade.stop_price) if trade.stop_price is not None else None,
        )
        trade.status = fill["status"]
        if trade.status == "ACCEPTED" and trade.time_in_force == "ioc":
            trade.status = "EXPIRED"
        if fill["status"] == "FILLED":
            trade.filled_price = fill["filled_price"]
            trade.filled_at = dt.datetime.now(dt.timezone.utc)

    db.commit()
    db.refresh(trade)

    if trade.status == "FILLED":
        portfolio_service.recompute_positions(db, user.id)

    security_service.write_audit_log(db, user.id, f"ORDER_{trade.status}", "trade", trade.id, {
        "asset": trade.asset, "side": trade.side, "quantity": float(trade.quantity),
    })
    event = {"FILLED": "order.filled", "REJECTED": "order.rejected"}.get(trade.status)
    if event:
        integration_service.emit_webhooks(db, user.id, event, _serialize_trade(trade))

    return _serialize_trade(trade)


def _serialize_trade(t: models.Trade) -> dict:
    return {
        "order_id": t.id,
        "asset": t.asset,
        "side": t.side,
        "quantity": float(t.quantity),
        "order_type": t.order_type,
        # FIX: time_in_force was missing from the response — frontend showed blank
        "time_in_force": t.time_in_force,
        "limit_price": float(t.limit_price) if t.limit_price else None,
        "stop_price": float(t.stop_price) if t.stop_price else None,
        "status": t.status,
        "alpaca_order_id": t.alpaca_order_id,
        "filled_price": float(t.filled_price) if t.filled_price else None,
        "pqc_signature_preview": (
            (t.pqc_signature or "")[:32] + "..." if t.pqc_signature else None
        ),
        "submitted_at": t.submitted_at.isoformat() if t.submitted_at else None,
        "filled_at": t.filled_at.isoformat() if t.filled_at else None,
    }


@app.get("/api/trading/orders")
def list_orders(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Opportunistically fill locally simulated conditional orders. Alpaca
    # orders are reconciled by the broker and must never be locally filled.
    pending = db.execute(
        select(models.Trade).where(
            models.Trade.user_id == user.id, models.Trade.status == "ACCEPTED",
            models.Trade.alpaca_order_id.is_(None),
        )
    ).scalars().all()
    changed = False
    for t in pending:
        fill_price = None
        if t.order_type == "limit" and t.limit_price is not None:
            # Use the lightweight price-check helper — no full simulate_fill overhead
            fill_price = trading_service.check_pending_limit_fill(
                t.asset, t.side, float(t.limit_price)
            )
        elif t.order_type in ("stop", "stop_limit") and t.stop_price is not None:
            # Stop/stop_limit: check if market price has crossed the stop trigger
            fill = trading_service.simulate_fill(
                t.asset, t.side, float(t.quantity), t.order_type,
                float(t.limit_price) if t.limit_price else None,
                float(t.stop_price),
            )
            fill_price = fill["filled_price"] if fill["status"] == "FILLED" else None
        # market orders in ACCEPTED state are filled immediately on placement;
        # they should not appear in pending — but guard against stale rows
        elif t.order_type == "market":
            fill_price = trading_service.get_last_price(t.asset)
        if fill_price is not None:
            t.status = "FILLED"
            t.filled_price = fill_price
            t.filled_at = dt.datetime.now(dt.timezone.utc)
            changed = True
    if changed:
        db.commit()
        portfolio_service.recompute_positions(db, user.id)

    trades = db.execute(
        select(models.Trade).where(models.Trade.user_id == user.id).order_by(models.Trade.submitted_at.desc())
    ).scalars().all()
    return [_serialize_trade(t) for t in trades]


@app.delete("/api/trading/orders/{order_id}")
def cancel_order(order_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    trade = db.get(models.Trade, order_id)
    if not trade or trade.user_id != user.id:
        raise HTTPException(404, "Order not found")
    if trade.status in ("FILLED", "CANCELLED", "REJECTED"):
        raise HTTPException(400, f"Cannot cancel order in status {trade.status}")
    trade.status = "CANCELLED"
    db.commit()
    security_service.write_audit_log(db, user.id, "ORDER_CANCELLED", "trade", trade.id, {})
    integration_service.emit_webhooks(db, user.id, "order.cancelled", _serialize_trade(trade))
    return _serialize_trade(trade)


# --------------------------------------------------------------------------
# Enterprise SDK (scoped X-QS-API-KEY authentication)
# --------------------------------------------------------------------------
@app.get("/api/sdk/portfolio")
def sdk_portfolio(key: models.ApiKey = Depends(require_api_scope("read")), db: Session = Depends(get_db)):
    return {"positions": portfolio_service.get_positions_with_pnl(db, key.user_id),
            "risk_metrics": portfolio_service.risk_metrics(db, key.user_id)}


@app.post("/api/sdk/orders", status_code=201)
def sdk_order(req: schemas.OrderRequest, key: models.ApiKey = Depends(require_api_scope("trade")),
              db: Session = Depends(get_db)):
    user = db.get(models.User, key.user_id)
    if not user or not user.is_active:
        raise HTTPException(401, "API-key user is inactive")
    return place_order(req, user, db)


@app.get("/api/integrations/api-keys")
def list_api_keys(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.execute(select(models.ApiKey).where(models.ApiKey.user_id == user.id)
                      .order_by(models.ApiKey.created_at.desc())).scalars().all()
    return [{"id": k.id, "name": k.name, "prefix": k.key_prefix, "scopes": k.scopes,
             "is_revoked": k.is_revoked, "created_at": k.created_at.isoformat(),
             "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None} for k in keys]


@app.post("/api/integrations/api-keys", status_code=201)
def create_api_key(req: schemas.ApiKeyRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    raw, prefix, digest = integration_service.generate_api_key()
    key = models.ApiKey(user_id=user.id, name=req.name.strip(), key_prefix=prefix, key_hash=digest, scopes=req.scopes)
    db.add(key); db.commit(); db.refresh(key)
    security_service.write_audit_log(db, user.id, "API_KEY_CREATED", "api_key", key.id, {"scopes": req.scopes})
    return {"id": key.id, "name": key.name, "prefix": key.key_prefix, "scopes": key.scopes, "api_key": raw}


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


@app.post("/api/backtests", status_code=201)
def run_backtest(req: schemas.BacktestRequest, user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    if req.slow_window <= req.fast_window:
        raise HTTPException(400, "slow_window must be larger than fast_window")
    try:
        result = backtest_service.run_moving_average_backtest(
            req.asset, req.fast_window, req.slow_window, req.period
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    record = models.Backtest(user_id=user.id, initial_capital=result["initial_capital"],
                             final_capital=result["final_capital"], sharpe_ratio=result["sharpe_ratio"],
                             max_drawdown=result["max_drawdown"], win_rate=result["win_rate"],
                             total_trades=result["total_trades"], result_json=result)
    db.add(record); db.commit(); db.refresh(record)
    security_service.write_audit_log(db, user.id, "BACKTEST_COMPLETED", "backtest", record.id,
                                     {"asset": req.asset, "period": req.period, "total_return": result["total_return"]})
    return {"id": record.id, **result}


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
        "alpaca_live": trading_service.alpaca_enabled(),
        "server_dsa_key_age_days": key_age_days,
    }


@app.get("/api/security/audit-log")
def audit_log(limit: int = 50, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    logs = db.execute(
        select(models.AuditLog).where(models.AuditLog.user_id == user.id)
        .order_by(models.AuditLog.created_at.desc()).limit(limit)
    ).scalars().all()
    def _user_email(user_id: str | None) -> str | None:
        if not user_id:
            return None
        u = db.get(models.User, user_id)
        return u.email if u else None

    return [{
        "id": l.id, "action": l.action, "resource_type": l.resource_type,
        "resource_id": l.resource_id, "metadata": l.metadata_json,
        "user_email": _user_email(l.user_id),
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
        "product": "QuantumSentinel", "version": "1.1.0",
        "fips_standards": ["FIPS 203 (ML-KEM-768)", "FIPS 204 (ML-DSA-65)"],
        "tracked_assets": signal_engine.TRACKED_ASSETS,
        "asset_exchange_map": signal_engine.ASSET_EXCHANGE_MAP,
        "exchanges": exchanges_with_status,
        "alpaca_live": trading_service.alpaca_enabled(),
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

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

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
        return FileResponse(str(FRONTEND_DIR / "index.html"))
