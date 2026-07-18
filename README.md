# QuantumSentinel

**The world's first open-source, mobile-first, post-quantum secure trading terminal — web edition.**

QuantumSentinel is a working reference implementation of a post-quantum
secure algorithmic-trading web app, built from the QuantumSentinel PRD and
Technical Architecture spec. It runs genuine NIST-standardized cryptography
end-to-end — not mocked byte sizes, actual algorithm execution — combined
with a quantum-inspired signal engine driven by real market data.

![Dashboard](https://gumloop.com/artifacts/LG8YkrmpgXibTs4g92H3Cz?version_id=TGxXfaBBCxKbqm8bRP7D2B)

## What's real here

| Capability | Implementation |
|---|---|
| **ML-KEM-768** (FIPS 203) key encapsulation | [`kyber-py`](https://github.com/GiacomoPope/kyber-py) — pure-Python, spec-exact byte sizes (pk 1184B / sk 2400B / ct 1088B / ss 32B) |
| **ML-DSA-65** (FIPS 204) digital signatures | [`dilithium-py`](https://github.com/GiacomoPope/dilithium-py) — signs every order and every audit-log entry; signatures are re-verified on read |
| **Hybrid handshake** | Real X25519 ECDH (via `cryptography`/OpenSSL) + ML-KEM-768, mixed with HKDF-SHA256, exactly as specified in the architecture doc |
| **Quantum-inspired Simulated Bifurcation Algorithm (SBA)** signal engine | NumPy port of the Rust reference design, computing RSI/MACD/momentum/Bollinger features and a coupling matrix over **real Yahoo Finance data** |
| **Paper trading** | Routes to the real Alpaca paper API if `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` are set, otherwise fills orders against live market prices with a built-in simulated broker |
| **Portfolio analytics** | Sharpe ratio, max drawdown, VaR 95%, equity curve — computed from the actual trade ledger |
| **Security dashboard** | Key-rotation schedule (ML-KEM-768 daily / ML-DSA-65 90-day), Quantum Safety Score, ML-DSA-65-signed + re-verified audit trail |
| **Crypto-agility** | Algorithm registry pattern (`backend/crypto/pqc.py`) — add ML-KEM-1024 / ML-DSA-87 / SLH-DSA by adding a registry entry |
| **Beginner Mode** | Plain-language tooltips throughout the UI |

This is a single-process web port of the full multi-service architecture
described in the design docs (API Gateway + Trading Engine + PQC Crypto
Service + Signal Engine collapsed into one FastAPI app, SQLite instead of
PostgreSQL+Redis) — intentionally simplified for a portable, easy-to-run
web demo rather than a Rust/Docker-Compose microservice mesh. The mobile
React Native client and Rust signal engine from the original spec are out
of scope for this repo; the SBA math and API surface are preserved so a
native client could be built against the same backend.

## Quick start

```bash
git clone <this-repo>
cd quantumsentinel-web
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, register an account, and you're in — real
PQC handshake, live signals, paper trading, portfolio analytics, and a
signed audit log all work with **zero configuration**.

### Docker

```bash
docker compose up --build
```

### Optional: connect real Alpaca paper trading

```bash
cp .env.example .env
# fill in ALPACA_API_KEY / ALPACA_SECRET_KEY from https://alpaca.markets
```

Without these, orders fill against live Yahoo Finance prices via a
built-in paper broker — the whole order lifecycle still works end-to-end.

## Architecture

```
backend/
  main.py                 FastAPI app — all routes
  crypto/pqc.py           ML-KEM-768, ML-DSA-65, hybrid handshake, algorithm registry
  services/
    signal_engine.py      SBA quantum-inspired signal engine (real market data)
    trading_service.py    Paper broker (Alpaca or simulated)
    portfolio_service.py  Positions, Sharpe/drawdown/VaR
    security_service.py   Audit log signing/verification, key rotation
    auth_service.py       JWT + PQC session handshake
  models.py, schemas.py, database.py
frontend/
  index.html / styles.css / app.js   Vanilla-JS single-page dashboard
```

## API

Interactive OpenAPI docs are served at `/docs` once the server is running.
Key endpoints: `POST /api/auth/register`, `POST /api/auth/pqc-handshake`,
`GET /api/signals/latest`, `POST /api/trading/orders`,
`GET /api/portfolio/risk-metrics`, `GET /api/security/audit-log`,
`POST /api/security/rotate-keys`, `GET /api/algorithms`.

## Disclaimer

Paper trading only — no live brokerage or custodial capability. Signals
from the SBA engine are informational/simulation-only, not financial
advice. See [SECURITY.md](SECURITY.md) for the threat model.

## License

Apache-2.0. See [LICENSE](LICENSE).
