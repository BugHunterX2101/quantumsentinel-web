"""QuantumSentinel — FastAPI application.

Single-process web port of the multi-service architecture in the design
docs (API Gateway + Trading Engine + PQC Crypto Service + Signal Engine
collapsed into one deployable app for a portable web demo). All PQC
operations are genuine FIPS 203/204 algorithms (see backend/crypto/pqc.py).
"""
import datetime as dt
import os
import time
import secrets
import asyncio
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from . import models, schemas
from .database import get_db, init_db, SessionLocal
from .config import CORS_ORIGINS, ALLOWED_HOSTS
from .crypto import pqc
from .services import auth_service, signal_engine, trading_service, portfolio_service, security_service, backtest_service, integration_service

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="QuantumSentinel API", version="1.0.0")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Bounded in-memory limiter for the single-process reference deployment. It
# deliberately protects write paths even before a user has authenticated.
_request_windows: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def security_headers_and_rate_limit(request, call_next):
    client = request.client.host if request.client else "unknown"
    key = f"{client}:{request.url.path}"
    now = time.monotonic()
    window = _request_windows[key]
    while window and now - window[0] > 60:
        window.popleft()
    limit = 20 if request.url.path.startswith("/api/auth/") else 240
    if len(window) >= limit:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429,
                            headers={"Retry-After": "60"})
    window.append(now)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self' https://api.github.com; frame-ancestors 'none'; base-uri 'self'"
    )
    return response


@app.on_event("startup")
def on_startup():
    init_db()


# --------------------------------------------------------------------------
# Auth dependency
# --------------------------------------------------------------------------
def get_current_user(authorization: str | None = Header(default=None),
                      db: Session = Depends(get_db)) -> models.User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = auth_service.decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = db.get(models.User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(401, "User not found")
    return user


def require_api_scope(scope: str):
    def dependency(x_qs_api_key: str | None = Header(default=None), db: Session = Depends(get_db)) -> models.ApiKey:
        if not x_qs_api_key:
            raise HTTPException(401, "Missing X-QS-API-KEY")
        key = integration_service.verify_api_key(db, x_qs_api_key, scope)
        if not key:
            raise HTTPException(403, "Invalid API key or insufficient scope")
        return key
    return dependency


# --------------------------------------------------------------------------
# Auth endpoints
# --------------------------------------------------------------------------
@app.post("/api/auth/register")
def register(req: schemas.RegisterRequest, db: Session = Depends(get_db)):
    existing = db.execute(select(models.User).where(models.User.email == req.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(409, "Email already registered")

    user = models.User(email=req.email, password_hash=auth_service.hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate the user's PQC identity keys (demo: server-side; a real
    # mobile client would generate these on-device and only upload the
    # public keys).
    kem_pk, kem_sk, kem_ms = pqc.kem_keygen()
    dsa_pk, dsa_sk, dsa_ms = pqc.dsa_keygen()
    db.add(models.KeyPair(user_id=user.id, algorithm="ML-KEM-768",
                           public_key=pqc.b64(kem_pk), private_key=security_service.protect_private_key(pqc.b64(kem_sk))))
    db.add(models.KeyPair(user_id=user.id, algorithm="ML-DSA-65",
                           public_key=pqc.b64(dsa_pk), private_key=security_service.protect_private_key(pqc.b64(dsa_sk))))
    db.commit()

    security_service.write_audit_log(db, user.id, "USER_REGISTERED", "user", user.id,
                                      {"email": user.email})

    return {
        "user_id": user.id, "email": user.email, "tier": user.tier,
        "created_at": user.created_at.isoformat(),
        "keygen_ms": {"ml_kem_768": round(kem_ms, 3), "ml_dsa_65": round(dsa_ms, 3)},
    }


@app.post("/api/auth/login")
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.execute(select(models.User).where(models.User.email == req.email)).scalar_one_or_none()
    if not user or not auth_service.verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = auth_service.create_access_token(user.id, user.tier)
    security_service.write_audit_log(db, user.id, "USER_LOGIN", "user", user.id, {})
    return {
        "access_token": token, "token_type": "bearer", "expires_in": 3600,
        "user": {"user_id": user.id, "email": user.email, "tier": user.tier,
                  "beginner_mode": user.beginner_mode},
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
def latest_signals(user: models.User = Depends(get_current_user)):
    data = signal_engine.get_cached_signals()
    return data


@app.get("/api/signals/refresh")
def refresh_signals(user: models.User = Depends(get_current_user)):
    """Force-refresh (bypasses cache) — used by the dashboard's manual refresh button."""
    signal_engine._cache["generated_at"] = 0
    return signal_engine.get_cached_signals()


@app.websocket("/api/signals/stream")
async def signal_stream(websocket: WebSocket):
    """Authenticated signal stream with browser-safe subprotocol credentials.

    The JWT is carried as the second requested WebSocket subprotocol rather
    than in the URL, keeping it out of query-string logs and referrers.
    """
    origin = websocket.headers.get("origin")
    protocols = [item.strip() for item in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    token = protocols[1] if len(protocols) == 2 and protocols[0] == "qs" else None
    payload = auth_service.decode_access_token(token) if token else None
    if origin not in CORS_ORIGINS or not payload:
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
            await websocket.send_json(signal_engine.get_cached_signals())
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        db.close()


# --------------------------------------------------------------------------
# Trading endpoints
# --------------------------------------------------------------------------
@app.post("/api/trading/orders", status_code=201)
def place_order(req: schemas.OrderRequest, user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    if req.side not in ("buy", "sell"):
        raise HTTPException(400, "side must be buy or sell")
    if req.order_type not in ("market", "limit", "stop", "stop_limit"):
        raise HTTPException(400, "unsupported order_type")
    if req.time_in_force not in ("day", "gtc", "ioc"):
        raise HTTPException(400, "time_in_force must be day, gtc, or ioc")
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
    if req.side == "buy" and req.quantity * price_for_risk > 5_000:
        raise HTTPException(400, "order exceeds the 5% paper-account position limit ($5,000)")
    duplicate_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=15)
    duplicate = db.execute(select(models.Trade).where(
        models.Trade.user_id == user.id, models.Trade.asset == req.asset,
        models.Trade.side == req.side, models.Trade.quantity == req.quantity,
        models.Trade.status.in_(("PENDING", "ACCEPTED")), models.Trade.submitted_at >= duplicate_cutoff,
    )).scalars().first()
    if duplicate:
        raise HTTPException(409, "duplicate pending order blocked")

    # ML-DSA-65 signs the order payload before it is accepted — persisted for audit.
    user_dsa_key = db.execute(
        select(models.KeyPair).where(
            models.KeyPair.user_id == user.id, models.KeyPair.algorithm == "ML-DSA-65",
            models.KeyPair.is_active.is_(True),
        )
    ).scalars().first()
    payload = f"{req.side}:{req.asset}:{req.quantity}:{req.order_type}:{req.limit_price}:{req.stop_price}:{req.time_in_force}".encode()
    signature = None
    if user_dsa_key and user_dsa_key.private_key:
        private_key = security_service.unprotect_private_key(user_dsa_key.private_key)
        sig_bytes, _ = pqc.dsa_sign(pqc.unb64(private_key), payload)
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
            trade.status = "ACCEPTED"
            if resp.get("status") == "filled":
                trade.status = "FILLED"
                trade.filled_price = float(resp.get("filled_avg_price") or 0) or None
                trade.filled_at = dt.datetime.now(dt.timezone.utc)
        except Exception as e:
            trade.status = "REJECTED"
            db.commit()
            raise HTTPException(502, f"Alpaca order failed: {e}")
    else:
        fill = trading_service.simulate_fill(trade.asset, trade.side, trade.quantity,
                                              trade.order_type, trade.limit_price, trade.stop_price)
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
        "order_id": t.id, "asset": t.asset, "side": t.side, "quantity": float(t.quantity),
        "order_type": t.order_type, "limit_price": float(t.limit_price) if t.limit_price else None,
        "stop_price": float(t.stop_price) if t.stop_price else None,
        "status": t.status, "alpaca_order_id": t.alpaca_order_id,
        "filled_price": float(t.filled_price) if t.filled_price else None,
        "pqc_signature_preview": (t.pqc_signature or "")[:32] + "..." if t.pqc_signature else None,
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
        if t.order_type == "limit":
            fill_price = trading_service.check_pending_limit_fill(t.asset, t.side, float(t.limit_price))
        else:
            fill = trading_service.simulate_fill(
                t.asset, t.side, float(t.quantity), t.order_type,
                float(t.limit_price) if t.limit_price else None,
                float(t.stop_price) if t.stop_price else None,
            )
            fill_price = fill["filled_price"] if fill["status"] == "FILLED" else None
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
    """CSV-friendly compliance export for positions, metrics and audit evidence."""
    from fastapi.responses import Response
    positions_data = portfolio_service.get_positions_with_pnl(db, user.id)
    metrics = portfolio_service.risk_metrics(db, user.id)
    rows = ["section,asset,quantity,avg_entry_price,current_price,market_value,unrealized_pnl"]
    rows.extend("position,{asset},{quantity},{avg_entry_price},{current_price},{market_value},{unrealized_pnl}".format(**p)
                for p in positions_data)
    rows.extend(f"metric,{name},{value}" for name, value in metrics.items() if name != "equity_curve")
    return Response("\n".join(rows) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=quantumsentinel-portfolio.csv"})


# --------------------------------------------------------------------------
# Security endpoints
# --------------------------------------------------------------------------
@app.get("/api/security/health")
def security_health(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    health = security_service.key_health(db, user.id)
    n_keys = len(health["keys"])
    n_green = sum(1 for k in health["keys"] if k["status"] == "GREEN")
    quantum_safety_score = round(100 * (n_green / n_keys), 0) if n_keys else 100
    return {
        **health,
        "quantum_safety_score": quantum_safety_score,
        "fips_203_compliant": True,
        "fips_204_compliant": True,
        "alpaca_live": trading_service.alpaca_enabled(),
        "server_dsa_key_age_days": (dt.datetime.now(dt.timezone.utc) - security_service.server_identity.created_at).days,
    }


@app.get("/api/security/audit-log")
def audit_log(limit: int = 50, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    logs = db.execute(
        select(models.AuditLog).where(models.AuditLog.user_id == user.id)
        .order_by(models.AuditLog.created_at.desc()).limit(limit)
    ).scalars().all()
    return [{
        "id": l.id, "action": l.action, "resource_type": l.resource_type,
        "resource_id": l.resource_id, "metadata": l.metadata_json,
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
    return {
        "product": "QuantumSentinel", "version": "1.0.0",
        "fips_standards": ["FIPS 203 (ML-KEM-768)", "FIPS 204 (ML-DSA-65)"],
        "tracked_assets": signal_engine.TRACKED_ASSETS,
        "alpaca_live": trading_service.alpaca_enabled(),
    }


# --------------------------------------------------------------------------
# Frontend static hosting
# --------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
