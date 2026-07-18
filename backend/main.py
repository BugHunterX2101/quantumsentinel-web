"""QuantumSentinel — FastAPI application.

Single-process web port of the multi-service architecture in the design
docs (API Gateway + Trading Engine + PQC Crypto Service + Signal Engine
collapsed into one deployable app for a portable web demo). All PQC
operations are genuine FIPS 203/204 algorithms (see backend/crypto/pqc.py).
"""
import datetime as dt
import os
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from . import models, schemas
from .database import get_db, init_db
from .config import CORS_ORIGINS
from .crypto import pqc
from .services import auth_service, signal_engine, trading_service, portfolio_service, security_service

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="QuantumSentinel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


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
                           public_key=pqc.b64(kem_pk), private_key=pqc.b64(kem_sk)))
    db.add(models.KeyPair(user_id=user.id, algorithm="ML-DSA-65",
                           public_key=pqc.b64(dsa_pk), private_key=pqc.b64(dsa_sk)))
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
def latest_signals():
    data = signal_engine.get_cached_signals()
    return data


@app.get("/api/signals/refresh")
def refresh_signals(user: models.User = Depends(get_current_user)):
    """Force-refresh (bypasses cache) — used by the dashboard's manual refresh button."""
    signal_engine._cache["generated_at"] = 0
    return signal_engine.get_cached_signals()


# --------------------------------------------------------------------------
# Trading endpoints
# --------------------------------------------------------------------------
@app.post("/api/trading/orders", status_code=201)
def place_order(req: schemas.OrderRequest, user: models.User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    if req.side not in ("buy", "sell"):
        raise HTTPException(400, "side must be buy or sell")
    if req.order_type not in ("market", "limit"):
        raise HTTPException(400, "order_type must be market or limit")
    if req.order_type == "limit" and not req.limit_price:
        raise HTTPException(400, "limit_price required for limit orders")

    # ML-DSA-65 signs the order payload before it is accepted — persisted for audit.
    user_dsa_key = db.execute(
        select(models.KeyPair).where(
            models.KeyPair.user_id == user.id, models.KeyPair.algorithm == "ML-DSA-65",
            models.KeyPair.is_active.is_(True),
        )
    ).scalars().first()
    payload = f"{req.side}:{req.asset}:{req.quantity}:{req.order_type}:{req.limit_price}".encode()
    signature = None
    if user_dsa_key and user_dsa_key.private_key:
        sig_bytes, _ = pqc.dsa_sign(pqc.unb64(user_dsa_key.private_key), payload)
        signature = pqc.b64(sig_bytes)

    trade = models.Trade(
        user_id=user.id, asset=req.asset.upper(), side=req.side, quantity=req.quantity,
        order_type=req.order_type, limit_price=req.limit_price, time_in_force=req.time_in_force,
        status="PENDING", pqc_signature=signature,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)

    if trading_service.alpaca_enabled():
        try:
            resp = trading_service.submit_alpaca_order(
                trade.asset, trade.side, trade.quantity, trade.order_type,
                trade.limit_price, trade.time_in_force,
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
                                              trade.order_type, trade.limit_price)
        trade.status = fill["status"]
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

    return _serialize_trade(trade)


def _serialize_trade(t: models.Trade) -> dict:
    return {
        "order_id": t.id, "asset": t.asset, "side": t.side, "quantity": float(t.quantity),
        "order_type": t.order_type, "limit_price": float(t.limit_price) if t.limit_price else None,
        "status": t.status, "alpaca_order_id": t.alpaca_order_id,
        "filled_price": float(t.filled_price) if t.filled_price else None,
        "pqc_signature_preview": (t.pqc_signature or "")[:32] + "..." if t.pqc_signature else None,
        "submitted_at": t.submitted_at.isoformat() if t.submitted_at else None,
        "filled_at": t.filled_at.isoformat() if t.filled_at else None,
    }


@app.get("/api/trading/orders")
def list_orders(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # opportunistically fill any pending limit orders that have become marketable
    pending = db.execute(
        select(models.Trade).where(models.Trade.user_id == user.id, models.Trade.status == "ACCEPTED")
    ).scalars().all()
    changed = False
    for t in pending:
        fill_price = trading_service.check_pending_limit_fill(t.asset, t.side, float(t.limit_price))
        if fill_price:
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
    return _serialize_trade(trade)


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
                              private_key=pqc.b64(sk), rotation_count=rotation_count)
    db.add(new_key)
    for k in old_keys:
        k.is_active = False
        k.revoked_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(new_key)

    security_service.write_audit_log(db, user.id, "KEY_ROTATED", "key_pair", new_key.id, {
        "algorithm": req.algorithm, "reason": req.reason, "keygen_ms": round(ms, 3),
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
