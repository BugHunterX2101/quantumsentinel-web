from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        if not (re.search(r"[A-Z]", value) and re.search(r"[a-z]", value)
                and re.search(r"\d", value)):
            raise ValueError("password must include upper-case, lower-case, and a number")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class HandshakeRequest(BaseModel):
    x25519_public_key: str
    ml_kem_public_key: str | None = None
    client_nonce: str


class OrderRequest(BaseModel):
    asset: str = Field(min_length=1, max_length=12)
    side: str  # buy | sell
    quantity: float = Field(gt=0, le=1_000_000)
    order_type: str = "market"  # market | limit
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    time_in_force: str = "day"

    @field_validator("asset")
    @classmethod
    def normalized_asset(cls, value: str) -> str:
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z.]{1,12}", value):
            raise ValueError("asset must be a valid ticker symbol")
        return value


class RotateKeysRequest(BaseModel):
    algorithm: str = "ML-DSA-65"
    reason: str = "scheduled_rotation"


class StrategyRequest(BaseModel):
    name: str = Field(min_length=3, max_length=80)
    asset: str = Field(min_length=1, max_length=12)
    fast_window: int = Field(default=20, ge=2, le=100)
    slow_window: int = Field(default=50, ge=5, le=250)

    @field_validator("asset")
    @classmethod
    def normalized_strategy_asset(cls, value: str) -> str:
        return OrderRequest.normalized_asset(value)

    @field_validator("slow_window")
    @classmethod
    def valid_windows(cls, value: int, info) -> int:
        if "fast_window" in info.data and value <= info.data["fast_window"]:
            raise ValueError("slow_window must be larger than fast_window")
        return value


class BacktestRequest(BaseModel):
    asset: str = Field(min_length=1, max_length=12)
    fast_window: int = Field(default=20, ge=2, le=100)
    slow_window: int = Field(default=50, ge=5, le=250)
    period: str = "1y"

    @field_validator("asset")
    @classmethod
    def normalized_backtest_asset(cls, value: str) -> str:
        return OrderRequest.normalized_asset(value)

    @field_validator("period")
    @classmethod
    def valid_period(cls, value: str) -> str:
        if value not in {"6mo", "1y", "2y", "5y"}:
            raise ValueError("period must be one of 6mo, 1y, 2y, 5y")
        return value


class ApiKeyRequest(BaseModel):
    name: str = Field(min_length=3, max_length=80)
    scopes: list[str] = Field(default_factory=lambda: ["read"])

    @field_validator("scopes")
    @classmethod
    def valid_scopes(cls, value: list[str]) -> list[str]:
        allowed = {"read", "trade", "admin"}
        scopes = sorted(set(value))
        if not scopes or not set(scopes).issubset(allowed):
            raise ValueError("scopes must be read, trade, and/or admin")
        return scopes


class WebhookRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    event_types: list[str] = Field(default_factory=lambda: ["order.filled", "key.rotated"])

    @field_validator("url")
    @classmethod
    def safe_webhook_url(cls, value: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("webhook URL must be an absolute HTTPS URL without credentials")
        return value

    @field_validator("event_types")
    @classmethod
    def valid_event_types(cls, value: list[str]) -> list[str]:
        allowed = {"order.filled", "order.rejected", "order.cancelled", "key.rotated"}
        events = sorted(set(value))
        if not events or not set(events).issubset(allowed):
            raise ValueError("unsupported webhook event type")
        return events
