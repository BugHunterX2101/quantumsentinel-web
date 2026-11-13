# Security Policy

## Cryptography

- **ML-KEM-768** (FIPS 203) is implemented via the pure-Python reference
  package `kyber-py`. **ML-DSA-65** (FIPS 204) uses OpenSSL's
  implementation through `cryptography` (>= 48, whose wheels bundle
  OpenSSL >= 3.5) when the installed build supports it, and the
  pure-Python `dilithium-py` otherwise; server keys in the expanded format
  of earlier releases also sign through `dilithium-py`. The pure-Python
  packages are spec-compliant but are **not constant-time / side-channel
  hardened**. Do not use this repo's crypto module for a production
  deployment handling real financial assets without a reviewed provider
  such as `liboqs` (via `python-oqs`) or a hardware-backed equivalent.
- **The production provider setting is a startup gate only.** With
  `ENVIRONMENT=production` the API and the research worker refuse to start,
  and every PQC operation raises, unless `PQC_PROVIDER` is set to something
  other than `reference` and `PQC_PROVIDER_URL` is set
  (`backend/config.py`, `backend/crypto/pqc.py`). Nothing calls that URL yet:
  key encapsulation and signing still run in-process with the libraries
  above. Integrating a real liboqs/HSM adapter is the deployer's work.
- The hybrid handshake combines a classical X25519 ECDH exchange with
  ML-KEM-768 via HKDF-SHA256, so an attacker must break **both** primitives
  to recover a session key (defense against Harvest-Now-Decrypt-Later).
- **Handshake v2** (`QS-HANDSHAKE-V2`): the server signs a canonical JSON
  transcript of the full handshake (protocol version, all public keys,
  nonces, algorithm identifiers, and session context), not just a partial
  hash. Client nonces are consumed atomically with Redis `SET NX EX 300`
  (`qs:pqc:nonce:<sha256>`); a reused nonce gets 409. In production a Redis
  failure refuses the handshake instead of falling back to a per-process
  check.
- **Server identity pinning**: when `TRUSTED_SERVER_DSA_FINGERPRINT` is set,
  the API and the research worker refuse to start unless the server's
  ML-DSA-65 key has exactly that fingerprint (SHA-256 of the public key).
  The browser additionally pins the first fingerprint it sees in
  `localStorage`; a later mismatch marks the handshake untrusted (hard
  failure, not a warning) until the user deliberately re-pins after a
  verified rotation.
- **Scope of the handshake**: the hybrid X25519 + ML-KEM-768 handshake is a
  research prototype. Key encapsulation, derivation and transcript signing
  run on the server; the browser holds no ML-KEM key and does not verify
  ML-DSA signatures. Browser sessions are protected by TLS and HttpOnly
  cookies, not by the handshake's session key.
- Production trade orders use a deterministic `QS-ORDER-V1` canonical payload,
  client-held ML-DSA-65 signing key, bounded expiry, and one-time nonce. The
  reference development UI uses clearly labelled server attestation only, so
  it is never confused with client authorisation.
- Order retries use `Idempotency-Key`; reusing a key with a different payload
  is rejected. Audit entries are ML-DSA signed and linked in a signed SHA-256
  hash chain to make modification, deletion, or reordering detectable.
- **Per-message AEAD**: `backend/crypto/message_protocol.py` implements
  AES-256-GCM with sequence-number replay protection over a handshake
  session key. It is a library module; no endpoint uses it yet.

## Authentication & session management

- **HttpOnly cookie-based auth**: the access and refresh tokens (JWT, RS256)
  are delivered as `HttpOnly` cookies with `SameSite=Strict` (configurable
  through `COOKIE_SAMESITE`), marked `Secure` in production. JavaScript never
  sees them, which removes XSS-based token theft. Only the `qs_csrf` cookie is
  readable by the page, by design.
- **Refresh-token rotation with family reuse detection**: refresh tokens are
  stored in PostgreSQL (durable) with a Redis cache (fast lookup). If a
  refresh token is used twice (reuse detection), the entire token family
  is revoked, preventing stolen-token replay.
- **CSRF**: state-changing requests authenticated by cookie must carry an
  `X-CSRF-Token` header. The token is an HMAC-SHA256 (keyed with
  `CSRF_SECRET`) over the user id and its issue time, valid for 24 hours, and
  is accepted only for the user whose access token made the request. The
  `qs_csrf` cookie is never accepted in place of the header: a browser
  attaches cookies to cross-site requests, but only same-origin script (or a
  passing CORS preflight) can set the header.
- **HMAC-signed API requests**: SDK/trading clients can authenticate via
  `X-QS-KEY-ID`, `X-QS-TIMESTAMP`, `X-QS-NONCE`, `X-QS-SIGNATURE` headers.
  The signature is the hex HMAC-SHA256, keyed with the API key's secret, of
  `METHOD||path||timestamp||nonce||sha256hex(body)`: the five fields joined by
  a literal `||`, the method upper-cased. The timestamp must be within ±30
  seconds. The nonce is consumed (Redis `SET NX EX 300`) only after the
  signature verifies, so forged requests cannot burn a client's nonces.
- **Rate limiting**: every HTTP request is counted per principal and path in
  a 60-second window held in Redis (`qs:rate:*`), which all API workers
  share. `/api/auth/*` allows 10 requests a minute per client address; every
  other path allows 240 a minute per user, SDK key, or (unauthenticated)
  client address. In production a request is refused with 503 when Redis is
  unreachable; outside production the API falls back to a per-process window.
- **Login lockout**: 5 failed logins for one email from one client address
  within 15 minutes lock that pair out for 15 minutes.
- **Client address**: the API takes the client address (rate limits, lockout,
  audit records) from `X-Forwarded-For`, which it trusts from any peer
  (`FORWARDED_ALLOW_IPS: "*"` in `docker-compose.production.yml`). That is safe
  only because the API port is not published and the bundled nginx replaces
  the header with the address of the connection it accepted. A proxy that
  appends to a client-supplied header instead would let clients choose their
  address. Behind a load balancer or CDN, configure nginx's `set_real_ip_from`
  and `real_ip_header` so that `$remote_addr` is still the client.

## Audit trail

- **Signing key history**: all server ML-DSA signing keys are recorded in the
  `server_signing_keys` table with fingerprints and activation/retirement
  timestamps. Each audit log entry and chain link records `signing_key_id`
  for historical verification across key rotations.
- **Chain ordering**: an append takes a PostgreSQL transaction-scoped advisory
  lock (`pg_advisory_xact_lock`), reads the chain head, and writes the next
  links in sequence after it. Concurrent writers in every process and
  container therefore serialise on one lock, and the lock is released when
  the links commit. Pending events are appended in batches, one lock per
  batch.
- Kill switches are stored in Redis (`qs:risk:kill:*`) for multi-worker
  consistency and audit-logged for compliance.

## Frontend security

- **No unsanitized data reaches innerHTML**: the auth screen, dashboard,
  trading, portfolio and security views render via the DOM API
  (`createElement`, `textContent`, `replaceChildren`) or pre-escape any
  server/API string through `escapeHtml()` before it is interpolated into a
  template string. The Research and Lab result panels build their markup
  with `innerHTML` template strings for layout convenience, but the values
  interpolated there are server-computed numbers/labels (Sharpe ratios,
  p-values, metric names) — never raw user or third-party text — so this is
  a rendering-convenience choice, not an unescaped-user-input path.
- **Content Security Policy (CSP)**, sent by the API on every response:
  - `default-src 'self'` — anything not listed below is same-origin only.
  - `script-src 'self'` — only same-origin scripts; no `unsafe-inline` or
    `unsafe-eval`. Three.js is self-hosted (`frontend/vendor`, npm tarball
    verified against its published sha512 integrity), so no third-party
    script host is trusted.
  - `style-src 'self' 'unsafe-inline' https://fonts.googleapis.com` — the
    Research/Lab result panels and parts of the base markup use inline
    `style="..."` attributes rather than a stylesheet, so `unsafe-inline` is
    required here for those views to render at all. This does not reopen the
    script-injection surface `script-src` closes: inline styles cannot
    execute JavaScript, only affect presentation.
  - `font-src 'self' https://fonts.gstatic.com data:` — Google Fonts only.
  - `connect-src 'self' wss: https://api.github.com https://api.pwnedpasswords.com`
    in production; development also allows `ws:`.
  - `img-src 'self' data:` — inline data URIs for icons/avatars.
  - `object-src 'none'` — blocks plugins.
  - `frame-ancestors 'none'` — prevents clickjacking.
  - `form-action 'self'` — prevents form hijacking.
  - `base-uri 'self'` — prevents `<base>` tag injection.
- **Other response headers**: `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and a
  `Permissions-Policy` that disables camera, microphone and geolocation.
  The bundled nginx accepts TLS 1.3 only and adds
  `Strict-Transport-Security: max-age=63072000; includeSubDomains`.
- **WebSocket hardening** (`/api/signals/stream`):
  - authentication is the HttpOnly `qs_access` cookie only; tokens are never
    accepted in the URL or subprotocol;
  - the `Origin` header must be one of `CORS_ORIGINS`, else the socket is
    closed with 4401;
  - at most 3 connections per user across all workers, counted in Redis
    (`qs:ws:conn:<user>`, expiring after 120 seconds unless a live connection
    refreshes it; a slot is taken in one atomic script); a fourth is closed
    with 4429, and in production so is any connection while Redis is
    unreachable;
  - the socket is closed with 4401 when the access token expires or the
    account is deactivated, and the client reconnects with its refreshed
    cookie;
  - a client message over 64 KB closes the socket with 1009;
  - every push carries a sequence number for gap detection.
  There is no idle timeout: an authenticated connection stays open until its
  token expires.

## Container security

- Dockerfile runs as non-root `app` user (UID 10001).
- Production compose: the application services (`quantumsentinel`,
  `research-worker`, `migrate`, `provision`) share `read_only: true` (with
  tmpfs for `/tmp` and `/var/run`), `security_opt: ["no-new-privileges:true"]`
  and `cap_drop: ["ALL"]`. `postgres`, `redis`, `nginx` and the backup
  services also run with `no-new-privileges` and every capability dropped;
  nginx adds back only `NET_BIND_SERVICE`, `CHOWN`, `SETUID` and `SETGID`.
- Memory limits: `mem_limit: 2g` for the API and `1g` for the research worker.
- The database is on an internal network with no route out, and nginx is not
  on it.

## CI security pipeline

| Scanner       | Purpose                              | Fails the build on |
|---------------|--------------------------------------|--------------------|
| **Bandit**    | Python static analysis (SAST)        | High severity with high confidence (the uploaded report lists medium and above) |
| **pip-audit** | Dependency CVE scanning              | Any known vulnerability |
| **Gitleaks**  | Secret detection across git history  | Any finding |
| **Semgrep**   | SAST (`p/python`, `p/security-audit`, `p/owasp-top-ten`) | Any finding |
| **Trivy**     | Container image vulnerability scan   | Critical/High with a fix available |

## Known simplifications vs. the full architecture spec

- One API service (gunicorn running `WEB_CONCURRENCY` uvicorn workers) and
  one research-worker service, which runs each research job in a child
  process that holds no secrets, instead of 4 microservices (API Gateway /
  Rust Signal Engine / Trading Engine / PQC Crypto Service).
- The portable web mode encrypts user PQC private keys at rest with
  `PRIVATE_KEY_ENCRYPTION_KEY`. A production mobile client should generate
  private keys on-device and never upload them from a Secure Enclave or
  Android Keystore.
- JWT uses RS256. In production the API refuses to start without
  `JWT_PRIVATE_KEY` and `JWT_PUBLIC_KEY`; rotate them through a
  key-management system.
- Login lockout state lives in each API worker's memory, not in Redis: with
  `WEB_CONCURRENCY` workers a client can make up to that many times the
  failed attempts before every worker has locked it out, and a restart
  clears the lockout. The Redis-backed limit of 10 requests a minute on
  `/api/auth/*` applies regardless.
- The server's audit-signing identity comes from `SERVER_DSA_PRIVATE_KEY`
  and `SERVER_DSA_PUBLIC_KEY`, which production requires; outside production
  a process without them generates an ephemeral key at start. The keys are
  environment values: hold them in an HSM/KMS before relying on them for
  real assets.

## Deployment baseline

- Set `ENVIRONMENT=production`. The API then refuses to start without
  `DATABASE_URL` (PostgreSQL only), `REDIS_URL`, `JWT_PRIVATE_KEY` and
  `JWT_PUBLIC_KEY`, `SERVER_DSA_PRIVATE_KEY` and `SERVER_DSA_PUBLIC_KEY`,
  `PRIVATE_KEY_ENCRYPTION_KEY`, `WEBHOOK_ENCRYPTION_KEY`,
  `REFRESH_TOKEN_SECRET`, `CSRF_SECRET`, `PQC_PROVIDER`/`PQC_PROVIDER_URL`
  (see Cryptography), or with `*` in `CORS_ORIGINS`. Also set explicit
  `ALLOWED_HOSTS`.
- Terminate TLS 1.3 at a reverse proxy; do not expose the API port directly
  to the public internet. The bundled nginx does both.
- Use a managed secret store and a hardened liboqs/HSM crypto service for
  production. This repository remains paper-trading only.
- `docker-compose.production.yml` provides the operational baseline, but
  certificate issuance/renewal, secret-manager integration, alert routing,
  off-host backup replication, penetration testing, and formal compliance
  evidence must be owned by the deploying organization.

## Reporting a vulnerability

> **Please do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability, please report it responsibly:

1. **GitHub Security Advisory** (preferred): use the
   [private security advisory](https://github.com/BugHunterX2101/quantumsentinel-web/security/advisories/new)
   feature to report privately.
2. **Email**: contact the maintainer directly at the email listed in the
   GitHub profile.

### What to include

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce (proof-of-concept code or screenshots if applicable).
- Affected version(s) and component(s).
- Suggested fix or mitigation (if you have one).

### Response timeline

| Action                    | Target      |
|---------------------------|-------------|
| Acknowledgement           | 48 hours    |
| Initial triage            | 5 days      |
| Fix released (critical)   | 7 days      |
| Fix released (high)       | 14 days     |
| Fix released (medium/low) | 30 days     |

We will credit reporters in the release notes unless anonymity is requested.
