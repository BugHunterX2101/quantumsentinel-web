# Security Policy

## Cryptography

- **ML-KEM-768** (FIPS 203) and **ML-DSA-65** (FIPS 204) are implemented via
  the pure-Python reference packages `kyber-py` and `dilithium-py`. These
  are spec-compliant but are **not constant-time / side-channel hardened**
  like `liboqs`'s C implementation. Do not use this repo's crypto module
  for a production deployment handling real financial assets without
  swapping to `liboqs` (via `python-oqs`) or a hardware-backed equivalent.
- The hybrid handshake combines a classical X25519 ECDH exchange with
  ML-KEM-768 via HKDF-SHA256, so an attacker must break **both** primitives
  to recover a session key (defense against Harvest-Now-Decrypt-Later).
- Every trade order and every audit-log entry is signed with ML-DSA-65 and
  independently re-verified on read.

## Known simplifications vs. the full architecture spec

- Single FastAPI process instead of 4 microservices (API Gateway / Rust
  Signal Engine / Trading Engine / PQC Crypto Service).
- SQLite instead of PostgreSQL + Redis (swap `DATABASE_URL` for Postgres in
  production; add a real cache/pubsub layer for horizontal scaling).
- The portable web mode encrypts user PQC private keys at rest with
  `PRIVATE_KEY_ENCRYPTION_KEY`. A production mobile client should generate
  private keys on-device and never upload them from a Secure Enclave or
  Android Keystore.
- JWT uses RS256. Production deployments must provide stable `JWT_PRIVATE_KEY`
  and `JWT_PUBLIC_KEY` values and rotate them through a key-management system.
- The reference deployment rate limiter is process-local. Deployments with
  more than one application process must replace it with a shared, atomic
  Redis-backed limiter before being exposed to the internet.
- The server audit-signing identity is generated at process start in this
  demo. Persist it in an HSM/KMS (and retain public-key history) before
  relying on audit verification across restarts.

## Deployment baseline

- Set `ENVIRONMENT=production`, stable RS256 JWT keys, encryption keys for
  private material/webhooks, explicit `CORS_ORIGINS`, and explicit
  `ALLOWED_HOSTS`.
- Terminate TLS 1.3 at a managed reverse proxy; do not expose Uvicorn
  directly to the public internet.
- Use PostgreSQL, Redis-backed rate limiting/replay protection, a managed
  secret store, and a hardened liboqs/HSM crypto service for production.
  This repository remains paper-trading only.
- `docker-compose.production.yml` provides the operational baseline, but
  certificate issuance/renewal, secret-manager integration, alert routing,
  off-host backup replication, penetration testing, and formal compliance
  evidence must be owned by the deploying organization.

## Reporting a vulnerability

Open a GitHub issue with the `security` label, or a private security
advisory if the finding is sensitive. Do not include exploit details in a
public issue.
