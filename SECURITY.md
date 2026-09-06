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
- **Handshake v2** (`QS-HANDSHAKE-V2`): the server signs a canonical JSON
  transcript of the full handshake (protocol version, all public keys,
  nonces, algorithm identifiers, and session context), not just a partial
  hash. Client nonces are atomically consumed via Redis `SET NX EX 300`
  to prevent replay attacks.
- **Server identity pinning**: the frontend pins the server's ML-DSA-65
  fingerprint (SHA-256 of the public key) in `sessionStorage` and warns
  on unexpected changes. Production deployments can set
  `TRUSTED_SERVER_DSA_FINGERPRINT` for additional verification.
- Production trade orders use a deterministic `QS-ORDER-V1` canonical payload,
  client-held ML-DSA-65 signing key, bounded expiry, and one-time nonce. The
  reference development UI uses clearly labelled server attestation only, so
  it is never confused with client authorisation.
- Order retries use `Idempotency-Key`; reusing a key with a different payload
  is rejected. Audit entries are ML-DSA signed and linked in a signed SHA-256
  hash chain to make modification, deletion, or reordering detectable.
- **Per-message AEAD** (optional): after the PQC handshake establishes a
  session key, `message_protocol.py` provides AES-256-GCM encryption with
  sequence-number replay protection for sensitive operations.

## Authentication & session management

- **HttpOnly cookie-based auth**: JWT access tokens are delivered as
  `HttpOnly; Secure; SameSite=Strict` cookies — never exposed to JavaScript.
  This eliminates XSS-based token theft.
- **Refresh-token rotation with family reuse detection**: refresh tokens are
  stored in PostgreSQL (durable) with a Redis cache (fast lookup). If a
  refresh token is used twice (reuse detection), the entire token family
  is revoked, preventing stolen-token replay.
- **CSRF double-submit**: state-changing requests require an `X-CSRF-Token`
  header matching the `qs_csrf` cookie.
- **HMAC-signed API requests**: SDK/trading clients can authenticate via
  `X-QS-KEY-ID`, `X-QS-TIMESTAMP`, `X-QS-NONCE`, `X-QS-SIGNATURE` headers.
  Signature = `HMAC-SHA256(secret, method||path||timestamp||nonce||SHA256(body))`.
  Nonces are deduplicated via Redis `SET NX EX 300`.

## Audit trail

- **Signing key history**: all server ML-DSA signing keys are recorded in the
  `server_signing_keys` table with fingerprints and activation/retirement
  timestamps. Each audit log entry and chain link records `signing_key_id`
  for historical verification across key rotations.
- **PostgreSQL sequence**: audit chain sequence numbers use `nextval()` on
  PostgreSQL for atomic ordering. SQLite uses `MAX()+1` (safe for
  single-writer development).
- Kill switches are stored in Redis (`qs:risk:kill:*`) for multi-worker
  consistency and audit-logged for compliance.

## Frontend security

- **Zero innerHTML with unsanitized data**: all user/API data is rendered via
  DOM API (`createElement`, `textContent`, `replaceChildren`) or pre-escaped
  through `escapeHtml()`.
- **CSP**: `script-src 'self'` (no `unsafe-inline` or external CDN scripts),
  `style-src 'self' https://fonts.googleapis.com` (no `unsafe-inline`),
  `object-src 'none'`, `form-action 'self'`, `frame-ancestors 'none'`.
- **WebSocket hardening**: per-user connection limit (max 3), idle timeout
  (5 minutes), sequence numbers for gap detection, cookie-based auth
  fallback.

## Container security

- Dockerfile runs as non-root `app` user (UID 10001).
- Production compose: `security_opt: ["no-new-privileges:true"]`,
  `cap_drop: ["ALL"]`, `read_only: true` (with tmpfs for `/tmp`),
  `mem_limit: 2g`.

## CI security pipeline

- **Bandit**: Python SAST (medium+ severity)
- **pip-audit**: dependency CVE scanning
- **Gitleaks**: secret detection across git history
- **Semgrep**: SAST with OWASP Top 10 rules
- **Trivy**: container image vulnerability scanning (CRITICAL/HIGH)

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
  demo. Persist it in an HSM/KMS (and retain public-key history in the
  `server_signing_keys` table) before relying on audit verification across
  restarts.

## Deployment baseline

- Set `ENVIRONMENT=production`, stable RS256 JWT keys, encryption keys for
  private material/webhooks, `REFRESH_TOKEN_SECRET`, `CSRF_SECRET`,
  explicit `CORS_ORIGINS`, and explicit `ALLOWED_HOSTS`.
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
