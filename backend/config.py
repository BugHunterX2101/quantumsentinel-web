import os
import secrets

JWT_SECRET = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 3600

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
