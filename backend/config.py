import os
import secrets
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def _setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    file_path = os.getenv(f"{name}_FILE")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return default


ENVIRONMENT = _setting("ENVIRONMENT", "development").lower()
JWT_ALGORITHM = "RS256"
_jwt_private_pem = _setting("JWT_PRIVATE_KEY")
_jwt_public_pem = _setting("JWT_PUBLIC_KEY")
if _jwt_private_pem and _jwt_public_pem:
    JWT_SIGNING_KEY = _jwt_private_pem.replace("\\n", "\n").encode()
    JWT_VERIFY_KEY = _jwt_public_pem.replace("\\n", "\n").encode()
else:
    _jwt_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    JWT_SIGNING_KEY = _jwt_private.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    JWT_VERIFY_KEY = _jwt_private.public_key().public_bytes(serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo)
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "900"))
REFRESH_SESSION_SECONDS = int(os.getenv("REFRESH_SESSION_SECONDS", "3600"))
WEBHOOK_ENCRYPTION_KEY = _setting("WEBHOOK_ENCRYPTION_KEY")
PRIVATE_KEY_ENCRYPTION_KEY = _setting("PRIVATE_KEY_ENCRYPTION_KEY")
SERVER_DSA_PRIVATE_KEY = _setting("SERVER_DSA_PRIVATE_KEY")
SERVER_DSA_PUBLIC_KEY = _setting("SERVER_DSA_PUBLIC_KEY")
SERVER_DSA_CREATED_AT = _setting("SERVER_DSA_CREATED_AT")
DATABASE_URL = _setting("DATABASE_URL", "sqlite:///./quantumsentinel.db")
REDIS_URL = _setting("REDIS_URL")
PQC_PROVIDER = _setting("PQC_PROVIDER", "reference")
PQC_PROVIDER_URL = _setting("PQC_PROVIDER_URL")

# Wildcard CORS is convenient for a local demo, but is never acceptable once
# credentials are used in a deployed environment.
CORS_ORIGINS = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]
ALLOWED_HOSTS = [host.strip() for host in os.getenv(
    "ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"
).split(",") if host.strip()]

if ENVIRONMENT == "production":
    if not _jwt_private_pem or not _jwt_public_pem:
        raise RuntimeError("JWT_PRIVATE_KEY and JWT_PUBLIC_KEY are required in production")
    if "*" in CORS_ORIGINS:
        raise RuntimeError("CORS_ORIGINS must be explicit in production")
    if not WEBHOOK_ENCRYPTION_KEY:
        raise RuntimeError("WEBHOOK_ENCRYPTION_KEY is required in production")
    if not PRIVATE_KEY_ENCRYPTION_KEY:
        raise RuntimeError("PRIVATE_KEY_ENCRYPTION_KEY is required in production")
    if not SERVER_DSA_PRIVATE_KEY or not SERVER_DSA_PUBLIC_KEY:
        raise RuntimeError("SERVER_DSA_PRIVATE_KEY and SERVER_DSA_PUBLIC_KEY are required in production")
    if not DATABASE_URL or not DATABASE_URL.startswith(("postgresql://", "postgresql+")):
        raise RuntimeError("DATABASE_URL must use PostgreSQL in production")
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required in production")
    if PQC_PROVIDER == "reference" or not PQC_PROVIDER_URL:
        raise RuntimeError("Production requires a configured external liboqs/HSM PQC provider")
