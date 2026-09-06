"""QuantumSentinel — ORM models (mirrors the PRD PostgreSQL schema, simplified)."""
import uuid
import datetime as dt

from sqlalchemy import (
    Column, String, Boolean, DateTime, Numeric, Integer, ForeignKey, Text, JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    tier = Column(String, default="free")  # free | pro | enterprise
    beginner_mode = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    watchlist = Column(JSON, nullable=True)  # list[str] of ticker symbols; None → use default
    preferred_exchanges = Column(JSON, nullable=True)  # list[str] e.g. ["US","NSE","CRYPTO"]
    user_timezone = Column(String(64), nullable=True)  # IANA tz e.g. "Asia/Kolkata"
    created_at = Column(DateTime(timezone=True), default=utcnow)

    key_pairs = relationship("KeyPair", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    positions = relationship("Position", back_populates="user")


class KeyPair(Base):
    __tablename__ = "key_pairs"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    algorithm = Column(String(50), nullable=False)  # ML-KEM-768 | ML-DSA-65 | X25519
    public_key = Column(Text, nullable=False)  # base64
    private_key = Column(Text, nullable=True)  # base64 — demo only; a real client keeps this on-device
    is_active = Column(Boolean, default=True)
    rotation_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="key_pairs")


class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    assets = Column(JSON, default=list)
    config = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Backtest(Base):
    __tablename__ = "backtests"
    id = Column(String, primary_key=True, default=gen_uuid)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    initial_capital = Column(Numeric, default=100_000)
    final_capital = Column(Numeric, nullable=False)
    sharpe_ratio = Column(Numeric, default=0)
    max_drawdown = Column(Numeric, default=0)
    win_rate = Column(Numeric, default=0)
    total_trades = Column(Integer, default=0)
    status = Column(String, default="COMPLETED")
    result_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(80), nullable=False)
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(128), nullable=False, unique=True, index=True)
    scopes = Column(JSON, default=list)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Webhook(Base):
    __tablename__ = "webhooks"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    url = Column(String(2048), nullable=False)
    secret_hash = Column(String(128), nullable=False)
    event_types = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    last_delivery_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    asset = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy | sell
    quantity = Column(Numeric, nullable=False)
    order_type = Column(String, default="market")  # market | limit
    limit_price = Column(Numeric, nullable=True)
    stop_price = Column(Numeric, nullable=True)
    time_in_force = Column(String, default="day")
    status = Column(String, default="PENDING")
    alpaca_order_id = Column(String, nullable=True)
    filled_price = Column(Numeric, nullable=True)
    pqc_signature = Column(Text, nullable=True)  # ML-DSA-65 signature (base64)
    submitted_at = Column(DateTime(timezone=True), default=utcnow)
    filled_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="trades")


class Position(Base):
    __tablename__ = "positions"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    asset = Column(String, nullable=False)
    quantity = Column(Numeric, default=0)
    avg_entry_price = Column(Numeric, default=0)
    realized_pnl = Column(Numeric, default=0)
    updated_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="positions")


class SignalRecord(Base):
    __tablename__ = "signals"
    id = Column(String, primary_key=True, default=gen_uuid)
    asset = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)  # BUY | SELL | HOLD
    confidence = Column(Numeric, nullable=False)
    features = Column(JSON, default=dict)
    sba_iterations = Column(Integer, default=100)
    engine_version = Column(String, default="1.0.0-py")
    generated_at = Column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    metadata_json = Column(JSON, default=dict)
    pqc_signature = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class OrderSecurityRecord(Base):
    """Immutable security envelope retained separately from broker state."""
    __tablename__ = "order_security_records"
    __table_args__ = (UniqueConstraint("user_id", "nonce", name="uq_order_security_user_nonce"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    trade_id = Column(String, ForeignKey("trades.id"), nullable=False, unique=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    canonical_order = Column(Text, nullable=False)
    request_hash = Column(String(64), nullable=False)
    nonce = Column(String(128), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    signer_key_id = Column(String, nullable=True)
    signature = Column(Text, nullable=False)
    signature_mode = Column(String(32), nullable=False)  # client_mldsa | development_server
    created_at = Column(DateTime(timezone=True), default=utcnow)


class IdempotencyRecord(Base):
    """Stores the first response for a caller supplied idempotency key."""
    __tablename__ = "idempotency_records"
    __table_args__ = (UniqueConstraint("user_id", "idempotency_key", name="uq_idempotency_user_key"),)

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    idempotency_key = Column(String(128), nullable=False)
    request_hash = Column(String(64), nullable=False)
    response_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class AuditChainLink(Base):
    """Hash-chain link for tamper-evident audit history."""
    __tablename__ = "audit_chain_links"

    id = Column(String, primary_key=True, default=gen_uuid)
    sequence = Column(Integer, nullable=False, unique=True)
    audit_log_id = Column(String, ForeignKey("audit_logs.id"), nullable=False, unique=True, index=True)
    previous_hash = Column(String(64), nullable=False)
    entry_hash = Column(String(64), nullable=False, unique=True)
    checkpoint_signature = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
