from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class HandshakeRequest(BaseModel):
    x25519_public_key: str
    ml_kem_public_key: str | None = None
    client_nonce: str


class OrderRequest(BaseModel):
    asset: str
    side: str  # buy | sell
    quantity: float
    order_type: str = "market"  # market | limit
    limit_price: float | None = None
    time_in_force: str = "day"


class RotateKeysRequest(BaseModel):
    algorithm: str = "ML-DSA-65"
    reason: str = "scheduled_rotation"
