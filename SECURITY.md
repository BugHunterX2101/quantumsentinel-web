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
- User PQC private keys are generated and stored server-side for demo
  convenience. In the full mobile architecture, private keys are generated
  on-device and never leave the Secure Enclave / Android Keystore.
- JWT uses HS256 for simplicity; the spec calls for RS256 with rotated keys.

## Reporting a vulnerability

Open a GitHub issue with the `security` label, or a private security
advisory if the finding is sensitive. Do not include exploit details in a
public issue.
