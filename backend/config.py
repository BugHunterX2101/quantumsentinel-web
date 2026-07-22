import os
import secrets

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
JWT_SECRET = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "900"))
REFRESH_SESSION_SECONDS = int(os.getenv("REFRESH_SESSION_SECONDS", "3600"))
WEBHOOK_ENCRYPTION_KEY = os.getenv("WEBHOOK_ENCRYPTION_KEY")

# Wildcard CORS is convenient for a local demo, but is never acceptable once
# credentials are used in a deployed environment.
CORS_ORIGINS = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]
ALLOWED_HOSTS = [host.strip() for host in os.getenv(
    "ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"
).split(",") if host.strip()]

if ENVIRONMENT == "production":
    if not os.getenv("JWT_SECRET_KEY") or len(JWT_SECRET) < 32:
        raise RuntimeError("JWT_SECRET_KEY of at least 32 characters is required in production")
    if "*" in CORS_ORIGINS:
        raise RuntimeError("CORS_ORIGINS must be explicit in production")
    if not WEBHOOK_ENCRYPTION_KEY:
        raise RuntimeError("WEBHOOK_ENCRYPTION_KEY is required in production")
