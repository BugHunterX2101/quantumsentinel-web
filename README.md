<div align="center">

<img src="frontend/favicon.png" alt="QuantumSentinel Logo" width="80" height="80">

# QuantumSentinel

**Open-source quantitative research and paper-trading platform with post-quantum security primitives**

[![CI](https://github.com/BugHunterX2101/quantumsentinel-web/actions/workflows/ci.yml/badge.svg)](https://github.com/BugHunterX2101/quantumsentinel-web/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-pytest-blue)](tests/)
[![FIPS 203](https://img.shields.io/badge/FIPS_203-ML--KEM--768-cyan)](https://csrc.nist.gov/pubs/fips/203/final)
[![FIPS 204](https://img.shields.io/badge/FIPS_204-ML--DSA--65-purple)](https://csrc.nist.gov/pubs/fips/204/final)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Paper Trading](https://img.shields.io/badge/Trading-Paper_Only-orange)](SECURITY.md)

*Post-quantum security primitives · realistic backtesting · walk-forward validation · factor modelling · optional C++ kernels*

</div>

---

> **Safety boundary:** QuantumSentinel is paper-trading and research software. It is not a live brokerage, custodian, financial adviser, or production-certified cryptographic service. Never connect production financial credentials without completing the hardening steps in [SECURITY.md](SECURITY.md).

---

## What Makes QuantumSentinel Different?

Two things that rarely appear together — in one open-source codebase:

**Post-Quantum Cryptography (research / integration-ready)**
- A **hybrid X25519 + ML-KEM-768** handshake prototype (FIPS 203) runs at sign-in. Encapsulation, key derivation and transcript signing happen server-side; browser sessions themselves are protected by TLS + HttpOnly cookies
- **Handshake V2**: full-transcript ML-DSA-65 signing, atomic nonce replay protection (Redis `SET NX`), server identity pinning (startup-enforced `TRUSTED_SERVER_DSA_FINGERPRINT`; hard-fail browser pin)
- **HttpOnly cookie auth**: access + refresh tokens in `Secure; SameSite=Strict` cookies — never exposed to JavaScript. Refresh-token rotation with family-based reuse detection
- **CSRF double-submit** pattern for all state-changing requests; **HMAC-SHA256 request signing** for SDK/API clients
- Production orders use a canonical `QS-ORDER-V1` payload, client ML-DSA-65 signature, bounded expiry, nonce, and idempotency key
- Audit records are server-signed with ML-DSA and protected by a SHA-256 hash chain with `signing_key_id` for historical verification across key rotations
- **Kill switches** (global/user/asset-level) stored in Redis for multi-worker consistency
- Passwords use Argon2id; locally stored private-key and webhook material uses Fernet authenticated encryption


**Quantitative Research Engine**
- Backtested systematic strategies **under realistic transaction-cost and execution assumptions** — not toy MA crossovers with perfect fills
- Walk-forward validation that actually catches overfitting — not just in-sample Sharpe
- Fama-MacBeth cross-sectional factor regressions with Newey-West corrected t-statistics
- Deflated Sharpe Ratio, block bootstrap, permutation testing — statistical guard rails that quant desks actually use
- Optional C++ kernels (pybind11) for rolling correlation, HMM forward pass, and backtest loop — with transparent NumPy fallback

---

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Browser Client"]
        FE["Vanilla JS SPA\nWeb Crypto API · No build step required"]
    end

    subgraph Gateway["FastAPI Gateway"]
        MW["Security Middleware\nCORS · Rate Limiter · CSP · CSRF"]
        AUTH_MW["HttpOnly Cookie Auth\nRefresh-token rotation · CSRF double-submit"]
        WS["WebSocket Stream\n/api/signals/stream"]
        SPA["SPA Catch-all\n/{path:path} → index.html"]
    end

    subgraph PQC["PQC Crypto Layer (pqc.py)"]
        KEM["ML-KEM-768\nFIPS 203\n1184B pk · 1088B ct · 32B ss"]
        DSA["ML-DSA-65\nFIPS 204\n1952B pk · 3309B sig"]
        X25519["X25519\nRFC 7748\nClassical hybrid leg"]
        HKDF["HKDF-SHA256\nSession key derivation"]
        KEM --- HKDF
        X25519 --- HKDF
    end

    subgraph CoreSvcs["Core Services"]
        AUTH["auth_service\nHttpOnly cookie auth · Argon2id\nPQC Handshake V2 · Refresh rotation\nCSRF · Atomic nonce replay"]
        SIG["signal_engine\nSBA · RSI-14 · MACD 12/26/9\nLive price · 20s/15s cache"]
        TRADE["paper_broker + trading_service\nServer-side cash ledger · Atomic reservations\nOrder sweeper · Fill rules"]
        PORT["portfolio_service\nPositions · Equity curve\nSharpe · VaR · Drawdown"]
        SEC["security_service\nServer ML-DSA identity\nAudit chain · Key history\n90-day rotation · Kill switches"]
        INTG["integration_service\nScoped API keys\nHMAC-SHA256 request signing\nFernet-encrypted secrets"]
    end

    subgraph Research["Quant Research Engine"]
        BACK["backtest_service\nEvent-driven · Execution model\nCommission · spread · slippage"]
        WF["walk_forward\nRolling/Expanding windows\nOOS Sharpe · Overfitting flag"]
        ALPHA["alpha_research\nIC · Rank IC · ICIR\nDecay analysis · Quintile returns"]
        FACTOR["factor_model\nFama-MacBeth regression\nNewey-West t-statistics"]
        CORR["correlation_engine\nShrinkage (Ledoit-Wolf)\nPCA denoising"]
        OPT["portfolio_optimization\nMean-Variance · turnover-aware MVO\nRisk-Parity · Efficient Frontier"]
        REGIME["regime_detection\n2-state HMM · Viterbi\nVol/Trend regime"]
        STAT["stat_tests\nDSR · Bootstrap CI\nPermutation · ADF · Cointegration"]
        NEUTRAL["neutral_strategies\nPairs trading · Kalman filter\nOU half-life"]
        CPP["cpp_ext\nC++ kernels (pybind11)\nNumPy fallback"]
        BENCH["latency_bench\np50/p95/p99/p99.9\nC++ vs Python speedup"]
        REPORT["report_generator\n7-section JSON report\nExecutive → Frontier"]
    end

    subgraph Persistence["Persistence"]
        DB[("PostgreSQL 16\nSQLAlchemy 2.0 · Alembic\nRefresh tokens · Key history")]
        REDIS[("Redis\nRate limiting · Nonce replay\nKill switches · Refresh cache")]
    end

    subgraph External["External APIs"]
        YF["Yahoo Finance\n3-month OHLCV · fast_info"]
    end

    Client <-->|"HTTPS / WSS · HttpOnly cookies"| Gateway
    Gateway --> AUTH_MW --> CoreSvcs
    Gateway --> AUTH_MW --> Research
    Gateway --> WS --> SIG
    Gateway --> SPA

    AUTH --> PQC
    SEC --> PQC
    TRADE --> PQC

    CoreSvcs --> DB
    CoreSvcs --> REDIS

    SIG --> YF
    TRADE --> YF

    Research --> CPP
    REPORT --> WF & FACTOR & REGIME & STAT & OPT & ALPHA
```

---

## Research Pipeline — End to End

```mermaid
flowchart LR
    YF["Yahoo Finance\nOHLCV · fast_info\nAny world ticker"]

    subgraph FE["Feature Engineering"]
        F1["Momentum · Reversal\n20-day / cross-sectional"]
        F2["Volatility · Quality\nSharpe-weighted"]
        F3["SBA Spin · HMM State\nQuantum-inspired + regime"]
    end

    subgraph EXEC["Execution Simulator"]
        E1["1-bar execution delay"]
        E2["Bid-ask spread (bps)"]
        E3["Commission model"]
        E4["Slippage (market impact)"]
        E5["Short selling + borrow cost"]
        E6["Leverage limits · Cash constraints"]
        E7["Partial fills"]
    end

    subgraph WF["Walk-Forward Validation"]
        W1["Rolling window splits"]
        W2["Expanding window splits"]
        W3["Per-fold IS optimization"]
        W4["Per-fold OOS Sharpe"]
        W5["Overfitting detection flag"]
    end

    subgraph FACTOR["Factor & Regime"]
        FA["Fama-MacBeth\nCross-sectional regression"]
        HMM["2-state HMM\nBull/Bear detection"]
        NW["Newey-West\nAutocorrelation-corrected t-stats"]
    end

    subgraph STAT["Statistical Validation"]
        S1["Newey-West t-statistic"]
        S2["Deflated Sharpe Ratio (DSR)"]
        S3["Block Bootstrap CI (95%)"]
        S4["Permutation p-value"]
        S5["Ljung-Box autocorrelation"]
        S6["ADF unit-root test"]
    end

    subgraph RISK["Risk Decomposition"]
        R1["CVaR 1% / 5%"]
        R2["Sortino · Calmar · Omega"]
        R3["Max drawdown · Duration"]
        R4["Turnover analysis"]
    end

    YF --> FE --> EXEC --> WF --> FACTOR --> STAT --> RISK --> RPT

    RPT["7-Section\nResearch Report\nJSON output"]
```

---

## PQC Hybrid Handshake Protocol

```mermaid
sequenceDiagram
    participant BR as Browser (Web Crypto API)
    participant GW as FastAPI Gateway
    participant PQC as pqc.py
    participant DB as Database

    Note over BR,DB: Phase 1 — Password Authentication
    BR->>GW: POST /api/auth/login {email, password}
    GW->>DB: Argon2id password verification
    DB-->>GW: User record
    GW-->>BR: RS256 JWT (15 min TTL) + expires_in

    Note over BR,DB: Phase 2 — Hybrid Post-Quantum Key Exchange
    BR->>BR: Generate X25519 ephemeral keypair
    BR->>GW: POST /api/auth/pqc-handshake {x25519_pub_b64, client_nonce_b64}
    GW->>PQC: kem_keygen() → ML-KEM-768 server key
    GW->>PQC: kem_encapsulate(server_kem_pk) → ciphertext + KEM_secret
    GW->>PQC: x25519_exchange(server_priv, client_x25519_pub) → X25519_secret
    GW->>PQC: HKDF-SHA256(X25519_secret ‖ KEM_secret ‖ nonces) → session_key
    GW->>PQC: dsa_sign(server_dsa_sk, ServerHello_payload) → ML-DSA-65 signature
    GW->>GW: Store short-lived session state
    GW-->>BR: ServerHello {kem_ciphertext, server_x25519_pub, ml_dsa_signature, ...}
    BR->>BR: Compare server fingerprint with the pinned one (mismatch = untrusted)
    Note over BR,GW: Research prototype: the browser holds no ML-KEM key and does not\nverify ML-DSA. Session security is TLS + HttpOnly cookies.
```

---

## Database Schema

```mermaid
erDiagram
    users {
        string id PK
        string email UK
        string password_hash
        string tier
        boolean is_active
        datetime created_at
    }
    trades {
        string id PK
        string user_id FK
        string asset
        string side
        float quantity
        string order_type
        string status
        float filled_price
        string pqc_signature
        datetime submitted_at
    }
    positions {
        string id PK
        string user_id FK
        string asset
        float quantity
        float avg_entry_price
        float realized_pnl
        datetime updated_at
    }
    key_pairs {
        string id PK
        string user_id FK
        string algorithm
        text public_key_b64
        boolean is_active
        int rotation_count
        datetime created_at
    }
    audit_logs {
        string id PK
        string user_id FK
        string action
        string resource_type
        string pqc_signature
        datetime created_at
    }
    order_security_records {
        string trade_id FK
        string nonce UK
        string request_hash
        string signature_mode
        datetime expires_at
    }
    idempotency_records {
        string user_id FK
        string idempotency_key UK
        string request_hash
        json response_json
    }
    audit_chain_links {
        int sequence UK
        string audit_log_id FK
        string previous_hash
        string entry_hash
    }
    api_keys {
        string id PK
        string user_id FK
        string name
        string prefix
        json scopes
        boolean is_revoked
        datetime expires_at
    }
    webhooks {
        string id PK
        string user_id FK
        string url
        json event_types
        boolean is_active
        datetime last_delivery_at
    }

    users ||--o{ trades : "places"
    users ||--o{ positions : "holds"
    users ||--o{ key_pairs : "owns"
    users ||--o{ audit_logs : "generates"
    users ||--o{ order_security_records : "authorises"
    users ||--o{ idempotency_records : "retries"
    audit_logs ||--|| audit_chain_links : "chains"
    users ||--o{ api_keys : "manages"
    users ||--o{ webhooks : "configures"
```

---

## Capabilities at a Glance

### Post-Quantum Cryptography

| What | How | Standard |
|---|---|---|
| Session key exchange | Hybrid X25519 + ML-KEM-768 → HKDF-SHA256 | FIPS 203, RFC 7748 |
| Order authorisation | Canonical `QS-ORDER-V1` payload · ML-DSA-65 client signature in production · nonce/expiry/idempotency | FIPS 204 |
| Audit trail | Server-signed entries · SHA-256 hash chain · ML-DSA checkpoints | FIPS 204 |
| Password hashing | Argon2id with OWASP-oriented parameters; legacy PBKDF2 upgrades on login | Argon2 |
| Keys at rest | Fernet authenticated encryption for local development storage | Fernet specification |
| JWT auth | RS256, 15-min TTL, nonce replay protection | RFC 7518 |
| Key rotation | 90-day policy enforced on ML-KEM + ML-DSA keypairs | FIPS 203/204 |

### Order Security and Execution Boundary

Orders pass through a security/risk boundary before the paper broker or an
external paper-trading adapter is called:

```text
Order request → risk gate → canonical QS-ORDER-V1 payload → signature check
              → nonce/replay check → idempotency check → execution → audit chain
```

- Production requests must include a client-held ML-DSA-65 signature, `order_id`,
  timestamp, expiry, nonce, and `Idempotency-Key`.
- Reusing a nonce is rejected; reusing an idempotency key with the same payload
  returns the initial result, while a changed payload is rejected.
- The risk gate checks kill switches, order notional, position availability, and
  projected gross leverage. Development paper trading uses visibly labelled
  server attestation for compatibility; it is not client order authorisation.

### Quant-Math Correctness Auditing

The research engine has been through repeated, source-verified correctness
passes rather than a single review: every formula is checked against its
primary reference (NIST FIPS text, the original paper, or a textbook
derivation) and, where practical, against an independent numerical
reproduction — not accepted on the strength of a code review alone. Past
rounds have caught and fixed, among others, a Sortino downside-deviation
denominator error, an inconsistent Bollinger Band `ddof` convention, a
sign-flip in capacity-degradation analysis, MacKinnon (2010) finite-sample
critical values for Engle-Granger cointegration, a Deflated Sharpe Ratio
variance-formula error, a risk-free-rate omission in the efficient-frontier
Sharpe calculation, and event-driven cost-basis corruption on position
flips — each with a regression test that is verified to fail against the
pre-fix code before being accepted. Concurrency-safety issues (unlocked
shared-state mutation, TOCTOU races on token rotation, and an audit
hash-chain append race) have received the same treatment. `git log` is the
source of truth for the full history of these fixes.

### Research Reproducibility

`research_metadata.py` provides versioned experiment records and dataset
lineage fields: assets, date range, source, adjustment policy, retrieval time,
feature version, parameters, execution model, random seed, Git commit hash,
data hash, IS/OOS metrics, and statistical results. Yahoo Finance daily-bar
data is explicitly labelled as not survivorship-free or point-in-time validated.

### Research Engine

| Module | Capabilities |
|---|---|
| **Backtest Engine** | Event-driven loop · 1-bar execution delay · commission model · bid-ask spread · slippage · short selling · borrow costs · leverage limits · cash constraints · partial fills |
| **Walk-Forward Validation** | Rolling & expanding windows · per-fold IS parameter optimisation · per-fold OOS Sharpe · overfitting detection flag · IS/OOS degradation analysis |
| **Alpha Research** | IC · Rank IC · ICIR · hit rate · decay analysis (1–20 bar horizon) · quintile/decile returns · signal turnover · long-short spread |
| **Fama-MacBeth Factor Model** | Cross-sectional regression · Newey-West autocorrelation-corrected t-statistics · factor premia per asset · per-period R² · factor significance testing |
| **Correlation Engine** | Sample covariance · Ledoit-Wolf shrinkage · PCA denoising · shrinkage intensity optimisation · rolling 252-day window |
| **Portfolio Optimisation** | Mean-variance (Markowitz) · turnover-aware MVO · Risk-Parity · Min-Volatility · Equal-Weight · efficient frontier computation |
| **HMM Regime Detection** | 2-state Gaussian HMM (bull/bear) · Viterbi sequence decoding · volatility regime · trend regime · transition probability matrix |
| **Statistical Testing** | Newey-West t-test · Deflated Sharpe Ratio (DSR) · block bootstrap CI · permutation p-value · Ljung-Box autocorrelation · ADF unit-root · Durbin-Watson · Engle-Granger cointegration |
| **Stat Arb / Pairs Trading** | Engle-Granger cointegration · Kalman filter hedge ratio · z-score entry/exit signals · Ornstein-Uhlenbeck half-life · spread mean-reversion test |
| **C++ Kernels** | pybind11 extension: `rolling_corr` (T×N×N) · `hmm_forward` (scaled forward algorithm) · `backtest_loop` — numerically identical NumPy fallback |
| **Latency Profiler** | `TimerStats` dataclass · per-stage repeated runs · p50/p95/p99/p99.9 percentiles · C++ vs Python speedup benchmark with numerical equivalence check |
| **Research Report** | 7-section structured JSON output covering the full pipeline from executive summary to efficient frontier |

### Trading Terminal

**Navigation Tabs (9):** Dashboard · Order Desk · Strategies · Research · Lab · Portfolio · Security · Integrations · Open Source

- **Signal Engine** — SBA + RSI-14 + MACD 12/26/9 + 20-day momentum + Bollinger Band Width + AI-generated insight text
- **Live Prices** — 5s micro-cached `fast_info` prices for any world ticker via Yahoo Finance
- **Asset Intelligence** — Instrument type · exchange · market open/closed status · fractional & 24/7 flags
- **Paper Trading** — internal paper broker only (no external broker client): server-side cash ledger with atomic cash reservation for open orders, fills against live Yahoo Finance prices, orders rejected when no current price exists, resting orders filled by a background sweeper
- **Order Types** — Market / Limit / Stop / Stop-Limit · Day / GTC / IOC · 30s duplicate guard · oversell prevention · dynamic 5% cap
- **Global Markets** — 9 exchanges: NYSE/NASDAQ, NSE/BSE, LSE, Xetra, TSE, HKEX, ASX, TSX, Crypto — with live market-hours detection
- **Portfolio Risk** — Mark-to-market positions · unrealised/realised P&L · equity curve · Sharpe · VaR 95/99 · CSV export
- **Lab (Advanced Tools)** — Event-driven backtest · Market regime detection (HMM) · Market-neutral L/S strategy · Pairs trading (Engle-Granger + Kalman filter) · Pipeline latency benchmark — each with full result rendering in-browser
- **Enterprise SDK** — Scoped `X-QS-API-KEY` credentials: `read` / `trade` / `admin` · SHA-256 stored hash
- **Signed Webhooks** — HTTPS-only · HMAC-SHA256 event signatures · Fernet-encrypted secrets at rest
- **Observability** — Prometheus metrics (HTTP request counts, latency histograms) behind auth guard

---

## Repository Structure

```
quantumsentinel-web/
│
├── backend/
│   ├── main.py                          ← Central router · middleware · WebSocket · SPA fallback
│   ├── models.py                        ← SQLAlchemy 2.0 schema (20 tables)
│   ├── schemas.py                       ← Pydantic v2 request/response validation (all endpoints)
│   ├── database.py                      ← PostgreSQL engine · pool · session limits · migrate()
│   ├── migrations/                      ← Alembic schema revisions (applied at startup or via manage.py)
│   ├── config.py                        ← ENV-driven config with production safety constraints
│   ├── worker.py                        ← Research worker: claims jobs · runs each in a killable child process
│   │
│   ├── crypto/
│   │   └── pqc.py                       ← ML-KEM-768 · ML-DSA-65 · X25519 · HKDF · registry
│   │
│   └── services/
│       │   ── Core Services ─────────────────────────────────────────────
│       ├── auth_service.py              ← JWT · Argon2id · PQC handshake · nonce TTL store
│       ├── signal_engine.py             ← SBA · RSI · MACD · live price · asset info · caching
│       ├── paper_broker.py              ← Paper cash ledger · atomic reservations · fills · order sweeper
│       ├── trading_service.py           ← Strict market prices · paper fill rules
│       ├── portfolio_service.py         ← Positions · mark-to-market · equity curve · Sharpe · VaR
│       ├── order_security.py            ← canonical orders · nonce/idempotency · risk gate
│       ├── research_metadata.py         ← experiment manifests · data lineage
│       ├── security_service.py          ← server identity · signed audit hash chain · key health
│       ├── integration_service.py       ← Scoped API keys · SSRF-guarded signed webhooks
│       │
│       │   ── Research Engine ─────────────────────────────────────────
│       ├── research_jobs.py             ← Job queue · leases · per-user limit · cancel · results
│       ├── research_tasks.py            ← The research computations a job runs, one function per kind
│       ├── backtest_service.py          ← Event-driven backtest with full execution cost model
│       ├── execution_model.py           ← Commission · spread · slippage · borrow cost models
│       ├── walk_forward.py              ← Rolling/expanding walk-forward OOS validation
│       ├── alpha_research.py            ← IC · Rank IC · ICIR · decay analysis · quintile returns
│       ├── factor_model.py              ← Fama-MacBeth cross-sectional regression · Newey-West
│       ├── correlation_engine.py        ← Shrinkage (Ledoit-Wolf) · PCA denoising
│       ├── portfolio_optimization.py    ← Mean-variance · turnover-aware MVO · Risk-Parity · frontier
│       ├── regime_detection.py          ← 2-state HMM · Viterbi · vol/trend regime
│       ├── stat_tests.py                ← DSR · bootstrap · permutation · ADF · cointegration
│       ├── neutral_strategies.py        ← Pairs trading · Kalman filter · OU half-life
│       ├── report_generator.py          ← 7-section structured JSON research report
│       ├── cpp_ext.py                   ← C++ kernel wrapper + NumPy fallback (auto-selects)
│       ├── latency_bench.py             ← p50/p95/p99/p99.9 pipeline profiler + C++ speedup
│       ├── event_simulator.py           ← Event-driven backtest with 1-bar delay · realistic fills
│       ├── redis_store.py               ← Redis session/nonce/kill-switch helper (optional fallback)
│       │
│       │   ── Market Microstructure ────────────────────────────────────
│       ├── market_microstructure.py      ← L2 order-book analytics · OBI · microprice · spread
│       ├── l2_event_replay.py           ← Incremental L2 replay · strategy → paper orders → fills · CSV/record loaders
│       ├── order_book.py                ← L2 book on integer ticks · MARKET vs PAPER ownership · FIFO queue position
│       ├── matching_engine.py           ← ADD/CANCEL/MODIFY/TRADE processing · one event per fill · stops · IOC/FOK
│       ├── paper_exchange.py            ← Latency-aware simulator · cash/position checks · automatic execution analytics
│       ├── latency_model.py             ← Configurable latency simulation · 7 presets
│       ├── execution_analytics.py       ← Adverse selection · implementation shortfall · capacity
│       └── experiment_registry.py       ← Experiment tracking · ML-DSA signed manifests · gates
│
├── cpp/                                 ← C++ performance kernels (pybind11)
│   ├── qs_fast.cpp                      ← rolling_corr · hmm_forward · backtest_loop
│   ├── setup.py                         ← Cross-platform build: MinGW (Windows) / GCC / Clang
│   └── target-platform extension built in CI/Docker (not committed)
│
├── frontend/                            ← Vanilla JS SPA (zero build step)
│   ├── index.html                       ← App shell · 9-tab navigation · all forms
│   ├── app.js                           ← ~3500-line SPA: auth · trading · portfolio · research · lab · WS
│   ├── bg3d.js                          ← Three.js 3D particle background engine
│   ├── styles.css                       ← Glassmorphism · micro-animations · mobile-first
│   ├── robots.txt                       ← Search engine crawl policy
│   ├── favicon.ico                      ← Quantum diamond icon (ICO format)
│   └── favicon.png                      ← Quantum diamond icon (PNG format)
│
├── tests/                               ← Pytest suite validated by CI
│   ├── test_core.py                     ← Auth · trading · portfolio · signal engine · core routes
│   ├── test_research_engine.py          ← Backtest engine · execution cost model · walk-forward
│   ├── test_phase2.py                   ← Alpha research · IC/Rank IC · decay · quintile returns
│   ├── test_phase3.py                   ← Stat tests · regime detection · pairs trading · optimisation
│   ├── test_phase4.py                   ← C++ kernels · p50/p99 latency · report generator
│   ├── test_order_security.py           ← Canonical order · nonce/idempotency · risk gate · kill-switch
│   ├── test_research_governance.py      ← Experiment metadata · data lineage · reproducibility
│   ├── test_research_jobs.py            ← Job queue · leases · cancel · timeouts · worker lifecycle
│   ├── test_database.py                 ← Migrations · SQLite import · session limits · DB races
│   ├── test_security_hardening.py       ← PQC handshake · audit chain · key rotation · CSRF
│   ├── test_microstructure.py           ← L2 analytics · OBI · microprice · synthetic L2 replay
│   ├── test_paper_exchange.py           ← Order book · matching engine · paper exchange · FIFO
│   ├── test_execution_analytics.py      ← Implementation shortfall · capacity · latency model
│   ├── test_experiment_registry.py      ← Experiment provenance · signed manifests · gates
│   ├── test_portfolio_service.py        ← recompute_positions() · equity curve · oversell edge cases
│   └── test_walk_forward.py             ← Offline walk-forward fold logic (rolling/expanding, degradation)
│
├── deploy/
│   ├── nginx.conf                       ← TLS 1.3 reverse proxy with HSTS
│   └── tls/                             ← Certificate mount point
│
├── .env.example                         ← Development environment template
├── Dockerfile                           ← Multi-stage Python image
├── docker-compose.yml                   ← Dev: app + PostgreSQL 16
├── docker-compose.production.yml        ← Prod: PostgreSQL + Redis + Nginx + Gunicorn
├── requirements.txt                     ← Pinned dependencies
├── SECURITY.md                          ← Threat model · hardening checklist · responsible disclosure
└── README.md                            ← This file
```

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | Official CPython from [python.org](https://python.org) — tested on 3.12 and 3.13 in CI |
| **Git** | Any recent version |
| **PostgreSQL 16+** | The only supported database. `docker compose up -d postgres` starts one with the default credentials; any reachable PostgreSQL works via `DATABASE_URL` |
| **C++ compiler** *(optional)* | MinGW-W64 GCC 16+ (Windows) · GCC 11+ (Linux) · Clang 14+ (macOS) — only needed for C++ kernel speedup |

### 1 — Clone & Install

```bash
git clone https://github.com/BugHunterX2101/quantumsentinel-web.git
cd quantumsentinel-web

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 — Configure (optional)

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

- Orders settle in the internal paper broker against the latest Yahoo Finance price; each account starts with `PAPER_INITIAL_CASH` (server-side, default $100,000)

### 3 — Run

```bash
docker compose up -d postgres        # or point DATABASE_URL at your own PostgreSQL
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- The schema is created and migrated automatically at startup (Alembic, `backend/migrations`).

- App: **http://127.0.0.1:8000**
- Interactive API docs: **http://127.0.0.1:8000/docs**
- Research runs in a background worker process that the API starts and stops with itself (`RESEARCH_WORKER_MODE=embedded`, the default); nothing else needs to be started.

---

## Build the C++ Extension (Optional Performance Upgrade)

The C++ kernels deliver hardware-accelerated performance for large datasets. Without them, all functionality is preserved via numerically identical NumPy fallbacks — `CPP_AVAILABLE` tells you which mode is active.

```bash
# ── Windows (MinGW-W64) ──────────────────────────────────────────────────────
winget install BrechtSanders.WinLibs.POSIX.UCRT   # Install MinGW
pip install pybind11 setuptools

cd cpp
python setup.py build_ext --inplace --compiler=mingw32

# ── Linux / macOS ────────────────────────────────────────────────────────────
cd cpp && pip install -e .

# ── Verify ───────────────────────────────────────────────────────────────────
python -c "from backend.services.cpp_ext import CPP_AVAILABLE; print('C++ kernels:', CPP_AVAILABLE)"
```

Check extension status at runtime: `GET /api/research/cpp-status`

**C++ Kernels:**
- **`rolling_corr(X, window)`** — Rolling Pearson correlation tensor (T × N × N) — used by correlation engine and SBA signal
- **`hmm_forward(obs, pi, A, mu, sigma)`** — Scaled HMM forward algorithm — used by regime detection
- **`backtest_loop(prices, signals, commission, spread_bps)`** — Full backtest event loop — used by backtest engine

---

## API Reference

### Research Engine Endpoints

Research runs as background jobs, never inside the request. The dashboard backtest (`POST /api/backtests`), every `POST` below and the experiment `run`, `replay` and `validate` endpoints validate their input, queue a job and return **`202 Accepted`** at once with the job and a `Location: /api/research/jobs/{job_id}` header. Poll that URL until `status` is `succeeded` (the job then carries `result`), `failed` (`error` holds `status_code` and `detail`) or `cancelled`. Invalid input is still rejected immediately (`400`/`422`) and nothing is queued. A user may have at most `RESEARCH_MAX_ACTIVE_JOBS_PER_USER` jobs queued or running (default 3); one more is refused with `429`. Jobs and their results are visible only to the user who queued them.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/research/backtest` | Event-driven backtest with commission · spread · slippage · execution delay |
| `POST` | `/api/research/walk-forward` | Rolling/expanding OOS validation with per-fold Sharpe and overfitting detection |
| `POST` | `/api/research/alpha` | IC · Rank IC · ICIR · hit rate · decay analysis · quintile/decile returns |
| `POST` | `/api/research/factor-model` | Fama-MacBeth cross-sectional regression with Newey-West t-statistics |
| `POST` | `/api/research/correlation` | Shrinkage (Ledoit-Wolf) and PCA-denoised correlation matrix estimation |
| `POST` | `/api/research/optimize` | Mean-variance · turnover-aware MVO · Risk-Parity · Min-Vol · efficient frontier |
| `POST` | `/api/research/regime` | 2-state Gaussian HMM · Viterbi decoding · volatility and trend regime |
| `POST` | `/api/research/stat-test` | Newey-West and OLS t-tests · bootstrap Sharpe CI (`n_bootstrap`) · sign-flip permutation p-value (`n_permutations`) · Ljung-Box · Deflated Sharpe Ratio, on `returns` (5 to 10,000 finite values) or a stored backtest's returns |
| `POST` | `/api/research/pairs-trading` | Engle-Granger cointegration · Kalman filter hedge ratio · z-score signal · OU half-life |
| `POST` | `/api/research/event-backtest` | Event-driven backtest with execution costs, leverage, and short-borrow accounting |
| **`POST`** | **`/api/research/report`** | **Full 7-section research report across all pipeline stages** |
| `POST` | `/api/research/latency-benchmark` | Per-stage p50/p99 latency profile + C++ vs Python speedup benchmark |
| `GET`  | `/api/research/cpp-status` | C++ extension load status · kernel names · active mode |
| `GET`  | `/api/research/jobs` | The caller's recent research jobs, newest first, without results (`?limit=`, up to 100) |
| `GET`  | `/api/research/jobs/{job_id}` | One job's status and queue position; its result or error once finished |
| `POST` | `/api/research/jobs/{job_id}/cancel` | Cancel a queued job, or stop a running one (its process is killed) |
| `GET`  | `/api/research/queue` | Operator only: queued and running counts, oldest queued job's age, live workers |

### Market Microstructure & Paper Exchange Endpoints

> **Synthetic L2.** Order-book data in these endpoints is generated from OHLCV bars (`data_source: "synthetic"`), not historical venue depth. The generator guarantees a valid book history (tick-grid prices, never crossed, cancels and trades only against resting liquidity) whose mid follows the bars. External L2 can be loaded with `L2EventStream.from_csv` / `from_records`, which validate market-data invariants. The simulator applies ADD/CANCEL/MODIFY/TRADE events to one incrementally maintained book, keeps paper orders in true FIFO queue position behind market liquidity, delays orders by the configured latency, and derives implementation shortfall, adverse selection and fill/queue metrics from its own fills. Live paper trading (`/api/trading/orders`) is separate: it settles in the server-side paper account at real market prices.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/microstructure/snapshot/{ticker}` | Synthetic order-book snapshot with OBI, microprice, spread analytics |
| `POST` | `/api/microstructure/analytics` | Compute microstructure features from provided bid/ask data |
| `POST` | `/api/microstructure/replay` | Replay synthetic L2 through the OBI-momentum strategy; with `execute` it trades through the paper exchange (latency preset, queue-aware fills, execution analytics) |
| `POST` | `/api/exchange/order` | Simulate one order against a synthetic L2 book (deterministic per seed; starting cash fixed server-side; never touches the paper account) |
| `GET` | `/api/exchange/book/{ticker}` | Synthetic order book seeded at the latest price |
| `POST` | `/api/execution/analysis` | Implementation shortfall decomposition (delay + spread + impact + fees) |
| `POST` | `/api/execution/capacity` | Strategy capacity analysis across capital sizes |
| `POST` | `/api/experiments/create` | Create experiment; records dataset, parameter and strategy-code hashes, code commit and dependency-lock hash |
| `GET` | `/api/experiments/{id}` | Get experiment |
| `POST` | `/api/experiments/{id}/run` | Research job: execute a platform strategy (`obi_momentum`) on the stored inputs; results are recorded once and the manifest is ML-DSA-signed. Submitting again while it is queued or running returns the same job |
| `GET` | `/api/experiments/{id}/manifest` | The signed manifest exactly as signed, verified with the key that signed it (valid after key rotation) |
| `POST` | `/api/experiments/{id}/replay` | Research job: verify inputs by hash; for executable strategies, re-execute from stored inputs and compare the result hash |
| `POST` | `/api/experiments/{id}/validate` | Research job: integrity gates (signature, manifest consistency, inputs unchanged, result reproduced) plus deployment gates. Submitting again while it is queued or running returns the same job |
| `POST` | `/api/experiments/{id}/approve` | Operator-only, four-eyes approval of a validated experiment (audit-logged) |
| `GET` | `/api/latency/presets` | List all latency presets (zero → retail, 0ms → 50ms) |

### Research Report — 7-Section JSON Structure

```json
{
  "executive_summary": {
    "sharpe_ratio": 1.42,
    "max_drawdown": -0.087,
    "alpha": 0.063,
    "information_ratio": 0.91,
    "annualized_return": 0.187,
    "annualized_vol": 0.131,
    "avg_daily_turnover": 0.18
  },
  "walk_forward_table": [
    { "fold": 1, "train_start": "2022-01-03", "test_end": "2022-06-30", "oos_sharpe": 1.21, "degraded": false },
    { "fold": 2, "train_start": "2022-01-03", "test_end": "2022-12-30", "oos_sharpe": 0.98, "degraded": false }
  ],
  "factor_premia_table": [
    { "factor": "momentum", "premium": 0.041, "t_stat": 3.5, "significant_5pct": true },
    { "factor": "volatility", "premium": -0.028, "t_stat": -2.1, "significant_5pct": true }
  ],
  "regime_statistics": {
    "current_state": "bull", "bull_prob_pct": 72.4,
    "n_regime_switches": 14, "avg_bull_duration_days": 32
  },
  "statistical_validation": {
    "newey_west_t": 3.12, "survives_deflation": true,
    "bootstrap_ci_95": [0.85, 1.98], "permutation_p_value": 0.02
  },
  "risk_decomposition": {
    "cvar_1pct": -0.031, "cvar_5pct": -0.019,
    "sortino_ratio": 1.87, "calmar_ratio": 2.1, "omega_ratio": 1.43,
    "skewness": -0.14, "excess_kurtosis": 0.88
  },
  "efficient_frontier": [
    { "vol": 0.10, "return": 0.07 }, { "vol": 0.12, "return": 0.09 }
  ]
}
```

### Core Platform Endpoints

| Area | Method | Endpoint | Auth | Notes |
|---|---|---|---|---|
| **Auth** | `POST` | `/api/auth/register` | — | Argon2id password hashing |
| | `POST` | `/api/auth/login` | — | Returns RS256 JWT |
| | `POST` | `/api/auth/pqc-handshake` | JWT | Hybrid X25519 + ML-KEM-768 |
| **Signals** | `GET` | `/api/signals/latest` | JWT | 20 preloaded assets, 20s cache |
| | `GET` | `/api/signals/asset/{ticker}` | JWT | Any world ticker, 15s cache |
| | `WS` | `/api/signals/stream` | JWT | Live push with exponential backoff |
| **Live Market** | `GET` | `/api/price/{ticker}` | JWT | Always-fresh 5s micro-cache |
| | `GET` | `/api/asset/info/{ticker}` | JWT | Instrument type, exchange, market open/closed |
| **Trading** | `POST` | `/api/trading/orders` | JWT | canonical order · production client signature · nonce/idempotency/risk gate |
| | `GET` | `/api/trading/orders` | JWT | Full order history |
| | `DELETE` | `/api/trading/orders/{id}` | JWT | Cancel pending order |
| **Watchlist** | `GET/PUT` | `/api/watchlist` | JWT | Get / replace full list |
| | `POST/DELETE` | `/api/watchlist/{ticker}` | JWT | Add / remove ticker |
| **Portfolio** | `GET` | `/api/portfolio/positions` | JWT | Mark-to-market positions |
| | `GET` | `/api/portfolio/risk-metrics` | JWT | Sharpe, VaR 95/99 |
| | `GET` | `/api/portfolio/export` | JWT | CSV download |
| **Security** | `GET` | `/api/security/health` | JWT | Key age, rotation status |
| | `GET` | `/api/security/audit-log` | JWT | ML-DSA-65 verified entries |
| | `GET` | `/api/security/audit-chain` | JWT (operator) | Verify the audit hash chain: links added since the last check, or every link with `?full=true` |
| | `POST` | `/api/security/rotate-keys` | JWT | New KEM + DSA keypair |
| | `GET` | `/api/security/compliance-report` | JWT | FIPS 203/204 compliance report |
| **SDK** | `GET` | `/api/sdk/portfolio` | `X-QS-API-KEY` (read) | Machine-to-machine |
| | `POST` | `/api/sdk/orders` | `X-QS-API-KEY` (trade) | Programmatic orders |
| **Integrations** | `GET/POST` | `/api/integrations/api-keys` | JWT | Scoped credentials |
| | `GET/POST` | `/api/integrations/webhooks` | JWT | HMAC-SHA256 signed |
| **Observability** | `GET` | `/metrics` | JWT | Prometheus counters/histograms |
| | `GET` | `/health/live` | — | Liveness probe |
| | `GET` | `/health/ready` | — | Readiness (DB + Redis check) |

---

## Security Architecture

### Cryptographic Specifications

| Algorithm | Role | Key Sizes | Compliance |
|---|---|---|---|
| **ML-KEM-768** | Post-quantum key encapsulation | pk: 1184B · sk: 2400B · ct: 1088B · ss: 32B | FIPS 203 |
| **ML-DSA-65** | Post-quantum digital signatures | pk: 1952B · sk: 4032B · sig: 3309B | FIPS 204 |
| **X25519** | Classical hybrid KEM leg | pk: 32B · ss: 32B | RFC 7748 |
| **HKDF-SHA256** | Session key derivation | 32B output | RFC 5869 |
| **RS256 (RSA-2048)** | JWT signing | 2048-bit | RFC 7518 |
| **Argon2id** | Password hashing | memory-hard KDF | Argon2 |
| **Fernet** | Local private-key / webhook-secret encryption | authenticated symmetric encryption | Fernet specification |

### Production Hardening Checklist

- [ ] Generate and persist `JWT_PRIVATE_KEY` / `JWT_PUBLIC_KEY` (RS256)
- [ ] Set `SERVER_DSA_PRIVATE_KEY` / `SERVER_DSA_PUBLIC_KEY` (audit log integrity across restarts; generate with `python -m backend.manage generate-server-key`)
- [ ] Configure `WEBHOOK_ENCRYPTION_KEY` and `PRIVATE_KEY_ENCRYPTION_KEY` (Fernet)
- [ ] Replace `reference` PQC backend with a reviewed liboqs/HSM adapter
- [ ] Enable PostgreSQL with encrypted connections
- [ ] Connect the application as `qs_app`, never as the PostgreSQL superuser: run `python -m backend.manage provision-roles` once (see [Database roles](#database-roles))
- [ ] Size connection pools: (API processes + research workers) × (`DB_POOL_SIZE` + `DB_MAX_OVERFLOW`) must stay below PostgreSQL's `max_connections`
- [ ] Configure Redis with AUTH password and AOF persistence
- [ ] Set strict `CORS_ORIGINS` and `ALLOWED_HOSTS` — no wildcards in production
- [ ] Place TLS 1.3 certificates at `deploy/tls/fullchain.pem` and `deploy/tls/privkey.pem`
- [ ] Rotate ML-KEM-768 and ML-DSA-65 keys within the 90-day policy enforced by `security_service`

Read [SECURITY.md](SECURITY.md) for the full threat model and responsible disclosure policy.

---

## Configuration Reference

Configuration defaults and production checks live in [`backend/config.py`](backend/config.py).

| Variable | Required | Description |
|---|---|---|
| `ENVIRONMENT` | No | `development` (default) or `production` |
| `DATABASE_URL` | Prod | PostgreSQL URL (`postgresql+psycopg://user:password@host:5432/db`; `postgresql://` and `postgres://` are accepted). Any other database is refused at startup. Default: the `docker-compose.yml` dev database |
| `DB_POOL_SIZE` | No | Connections each process keeps open (default 20) |
| `DB_MAX_OVERFLOW` | No | Extra connections a process may open for bursts; they are closed when returned, so steady load should fit in `DB_POOL_SIZE` (default 0) |
| `DB_POOL_TIMEOUT_SECONDS` | No | Wait for a free connection before answering 503 (default 10) |
| `DB_STATEMENT_TIMEOUT_MS` | No | Server-side limit on any one statement (default 30000) |
| `DB_LOCK_TIMEOUT_MS` | No | Server-side limit on waiting for a lock (default 10000) |
| `DB_IDLE_IN_TRANSACTION_TIMEOUT_MS` | No | A session idle inside an open transaction is ended after this long (default 60000) |
| `DB_MIGRATE_ON_STARTUP` | No | `true` (default): API processes apply pending migrations at startup, serialised by a database lock; `false`: run `python -m backend.manage migrate` as a release step |
| `DB_APP_PASSWORD`, `DB_MIGRATOR_PASSWORD`, `DB_BACKUP_PASSWORD` | `provision-roles` only | Passwords for `qs_app`, `qs_migrator` and `qs_backup`, read only by `python -m backend.manage provision-roles` (each also as `*_FILE`). Needed to create a role; given for an existing role, its password is changed |
| `JWT_PRIVATE_KEY` | Prod | RS256 private key (PEM, `\n`-escaped) |
| `JWT_PUBLIC_KEY` | Prod | RS256 public key (PEM, `\n`-escaped) |
| `WEBHOOK_ENCRYPTION_KEY` | Prod | Fernet key for encrypting webhook signing secrets at rest |
| `PRIVATE_KEY_ENCRYPTION_KEY` | Prod | Fernet key for encrypting user ML-DSA private keys in DB |
| `SERVER_DSA_PRIVATE_KEY` | Prod | Persistent ML-DSA-65 server signing key (base64) |
| `SERVER_DSA_PUBLIC_KEY` | Prod | Corresponding public key for audit log verification |
| `SERVER_DSA_CREATED_AT` | Prod | ISO timestamp of server key creation |
| `PQC_PROVIDER` | Prod | Must be set to non-`reference` value in production |
| `PQC_PROVIDER_URL` | Prod | URL to the external liboqs/HSM PQC adapter |
| `CORS_ORIGINS` | Prod | Comma-separated allowed origins (no wildcards) |
| `ALLOWED_HOSTS` | Prod | Comma-separated allowed hostnames |
| `REDIS_URL` | Prod | For distributed rate limiting and session storage |
| `PAPER_INITIAL_CASH` | No | Starting cash of every paper account (default 100000) |
| `PAPER_MAX_POSITION_FRACTION` | No | Per-asset concentration cap as a fraction of equity (default 0.05) |
| `ORDER_SWEEP_INTERVAL_SECONDS` | No | How often resting orders are checked for fills (default 5) |
| `RESEARCH_WORKER_MODE` | No | `embedded` (default): every API process runs one research worker subprocess and stops it with itself; `external`: the API runs none, start workers with `python -m backend.worker`; `off`: nothing runs queued jobs |
| `RESEARCH_JOB_TIMEOUT_SECONDS` | No | Hard run-time limit of one job; its process is killed when it passes (default 900) |
| `RESEARCH_JOB_LEASE_SECONDS` | No | A running job goes back to the queue when its worker stops renewing it for this long (default 60) |
| `RESEARCH_JOB_MAX_ATTEMPTS` | No | Runs a job gets when its workers are lost mid-run, before it is marked failed (default 2) |
| `RESEARCH_MAX_ACTIVE_JOBS_PER_USER` | No | Queued plus running research jobs one user may have (default 3) |
| `RESEARCH_JOB_RETENTION_DAYS` | No | Finished jobs and their results are deleted after this many days (default 7) |
| `RESEARCH_WORKER_POLL_SECONDS` | No | How often an idle worker checks the queue (default 1) |

---

## Database (PostgreSQL)

PostgreSQL 16 or newer is the only supported database; any other `DATABASE_URL` is refused at startup.

- **Migrations.** The schema is versioned with Alembic in [`backend/migrations`](backend/migrations). Each API process applies pending revisions at startup, serialised across processes by a database lock and atomic thanks to PostgreSQL's transactional DDL. To migrate as a separate release step instead, set `DB_MIGRATE_ON_STARTUP=false` and run `python -m backend.manage migrate`. Research workers wait until the schema is at their code's revision.
- **Moving from SQLite** (releases before 1.3): `python -m backend.manage import-sqlite path/to/quantumsentinel.db` copies every row into the configured, empty PostgreSQL database in one transaction, then verifies the row counts and the audit hash chain before committing. It refuses a database that already has rows.
- **Duplicate positions** (deployments upgraded from earlier releases): two fills for one user at the same moment could leave the `positions` table with more than one row per asset, overstating holdings and equity. Fills now rebuild positions inside their own transaction, and each user's next fill corrects their rows; `python -m backend.manage repair-positions` corrects every affected user at once and is safe to run while the API serves orders.
- **Audit chain verification.** `GET /api/security/audit-chain` checks only the links added since that API process last verified the chain, which took under a second at 1M links on the laptop below; nearly all of that is counting events that have no link. A process's first check, and any check after the chain failed one, verifies every link. A full check verifies an ML-DSA signature per link (about 11 s per 50,000 links on the same laptop), and it is the only check that sees rows edited behind the newest verified link, so run `python -m backend.manage verify-audit-chain` on a schedule; it exits 1 if any link fails. `?full=true` runs a full check from the API.
- **Connection budget.** Every API process and research worker has its own pool: keep (processes) × (`DB_POOL_SIZE` + `DB_MAX_OVERFLOW`) below the server's `max_connections` (100 by default), leaving room for maintenance sessions. The default of 20 per process fits four API processes plus their workers.
- **Large tables.** The indexes added by migration `0002` are created `IF NOT EXISTS`; on a large `trades` or `audit_logs` table, build them first with `CREATE INDEX CONCURRENTLY` under the same names to avoid blocking writes during the upgrade.
- **Connection poolers.** Each session is configured when it connects (UTC, statement/lock/idle timeouts) and the order sweeper uses a session-level advisory lock, so a PgBouncer in front of the database must use session pooling, not transaction pooling.

### Database roles

Connected as the superuser that the `postgres` image creates from `POSTGRES_USER`, the application bypasses every grant, row-level security policy and trigger, and any SQL it runs can drop the schema, read server files or run programs on the database host (`COPY ... TO PROGRAM`). [`backend/db_roles.py`](backend/db_roles.py) defines four roles instead:

| Role | Login | Privileges |
|---|---|---|
| `qs_owner` | No | Owns the schema and every object in it |
| `qs_migrator` | Yes | None of its own. Migrations run after `SET LOCAL ROLE qs_owner`, so everything they create is owned by `qs_owner` |
| `qs_app` | Yes | The API and research workers: exactly the table privileges in `APP_TABLE_PRIVILEGES` (`SELECT` only on `alembic_version`). No DDL or `TRUNCATE`; it cannot change roles, ownership or triggers |
| `qs_backup` | Yes | `pg_read_all_data`, for `pg_dump` |

`APP_TABLE_PRIVILEGES` is an allow-list: a table it does not name gets no privilege, and the test suite fails until the table is listed. Grants are reapplied after every migration that changes the schema revision.

To move a deployment over:

1. As the superuser (`DATABASE_URL` naming it), with `DB_APP_PASSWORD`, `DB_MIGRATOR_PASSWORD` and `DB_BACKUP_PASSWORD` set, run `python -m backend.manage provision-roles`. It migrates the schema to the latest revision first, then creates the roles and hands them the schema in one transaction. Only SCRAM verifiers reach the server, so no password appears in its logs. It can run while the application serves requests. Changing a table's owner needs the table's exclusive lock, and taking those one at a time deadlocked with order traffic in a rehearsal (provisioning held `positions` and waited for `users`, while an order held `users` and waited for `positions`). So the command takes every lock it needs in one `LOCK ... NOWAIT`. If any table is in use, it lets go of everything and tries again moments later; queries wait at most a moment. If the tables stay busy for 10 s, it exits 1 without changing anything. It is safe to run again, and a second run takes no table locks.
2. Point `DATABASE_URL` for the API and workers at `qs_app`. With `DB_MIGRATE_ON_STARTUP=true` they start normally while the schema is current, and refuse to start if a migration is pending.
3. Run each release's migrations as `qs_migrator`: `DATABASE_URL=<qs_migrator URL> python -m backend.manage migrate`.
4. Take backups as `qs_backup`: `pg_dump -U qs_backup`. A dump made this way records `qs_owner` as the owner of everything; create the roles before restoring it, or restore with `--no-owner`.

Running `provision-roles` again restores the intended role attributes, memberships, ownership and grants and removes anything extra. On a schema that has not been provisioned, migrations run as whoever connects, as before.

### Measured capacity

One run of 1,000,000 requests: 1,000 users, a mix of portfolio, orders, watchlist, jobs, audit-log and settings-update requests, against 4 API processes and PostgreSQL 16 holding 1M trades and 1M audit events, all on one 8-core Windows laptop together with the load generator:

| Requests | Succeeded | Throughput | p50 | p90 | p99 | Peak DB connections |
|---|---|---|---|---|---|---|
| 1,000,000 | 99.9963% | 514/s | 101 ms | 158 ms | 257 ms | 76 of 100 |

37 requests did not succeed: 13 answered with a 5xx and 24 ended in a client-side connection error or timeout. Every one of the 13 server-side failures traces to Windows socket-buffer exhaustion (`WinError 10055`) in the event loop during the first minutes of the run, and the server logged no other error, so none traces to application code or the database. The production image runs on Linux.

Requests that write the audit trail (logins, orders, fills) are limited differently: each event is signed with the server's ML-DSA key, and the hash chain signs its checkpoint while holding a lock shared by every process. With a server key generated by `python -m backend.manage generate-server-key` (a 32-byte seed, signed through native OpenSSL ML-DSA), 8 concurrent writers sustained about 115 audited writes per second on the same laptop and PostgreSQL 16. That figure is a ceiling for the whole deployment, not per process, because every writer appends to the one chain under the shared lock; most of the time under the lock is the checkpoint signature (about 2 ms) and the round trips to PostgreSQL. A server key in the 4032-byte expanded form that earlier releases generated still works but signs through the pure-Python reference implementation, which held the same test to 4.3 per second; replace it with a generated seed key to lift that limit (audit entries signed by the old key stay verifiable).

---

## Cache TTL Reference

| Cache | TTL | Description |
|---|---|---|
| Preloaded signals | 20 s | 20 blue-chip assets computed concurrently at startup |
| On-demand signals | 15 s | Any ticker searched live via the dashboard search bar |
| Live price | 5 s | `/api/price/{ticker}` micro-cache — nearly real-time |
| Asset info | 30 s | Instrument type, exchange, market status, trading features |

---

## Docker

### Development (app + PostgreSQL 16)

```bash
docker compose up --build
# → http://localhost:8000
```

### Production (PostgreSQL + Redis + Nginx TLS)

```bash
# 1. Generate RSA keypair for JWT RS256
python -c "
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
k = rsa.generate_private_key(65537, 2048)
print('PRIVATE:', k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()).decode().replace('\n','\\n'))
print('PUBLIC:', k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode().replace('\n','\\n'))
"

# 2. Fill production env
cp .env.production.example .env.production

# 3. Place TLS certs at deploy/tls/

# 4. Deploy
docker compose -f docker-compose.production.yml --env-file .env.production up -d --build
```

> [!NOTE]
> Research jobs run in worker subprocesses that the web containers start themselves (one per gunicorn worker, `WEB_CONCURRENCY`), so no extra service is needed. To scale workers separately, set `RESEARCH_WORKER_MODE=external` on the web service and run `python -m backend.worker` as its own service from the same image, with the same environment.

> [!IMPORTANT]
> Production mode deliberately refuses the bundled pure-Python reference PQC backend.
> Set `PQC_PROVIDER` and `PQC_PROVIDER_URL` only after integrating a reviewed liboqs/HSM adapter.

---

## Verification

```bash
# Run the full test suite against PostgreSQL (CI is the source of truth for the test count).
# TEST_DATABASE_URL names a database the suite may use freely; each test gets its own schema.
docker compose up -d postgres
TEST_DATABASE_URL=postgresql+psycopg://quantumsentinel:quantumsentinel@localhost:5432/quantumsentinel_test pytest tests/ -v

# Compile every backend module and test
python -m compileall -q backend tests

# JavaScript syntax check
node --check frontend/app.js

# C++ extension status
python -c "from backend.services.cpp_ext import CPP_AVAILABLE; print('C++ kernels:', CPP_AVAILABLE)"

# Docker health check
docker compose up --build -d && curl http://localhost:8000/health/ready
```

### Continuous Integration & Security Scanning

Every push and pull request runs the full test matrix — **Ubuntu, Windows, and
macOS**, each on **Python 3.12 and 3.13** — followed by five independent
security-scanning jobs, all defined in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Stage | Tool | Checks |
|---|---|---|
| `security-bandit` | [Bandit](https://bandit.readthedocs.io/) | Python SAST — common Python security anti-patterns |
| `security-pip-audit` | [pip-audit](https://github.com/pypa/pip-audit) | Known CVEs in pinned dependencies |
| `security-gitleaks` | [Gitleaks](https://github.com/gitleaks/gitleaks) | Committed secrets / credential leaks across full history |
| `security-semgrep` | [Semgrep](https://semgrep.dev/) | `p/python` · `p/security-audit` · `p/owasp-top-ten` rule sets |
| `security-trivy` | [Trivy](https://aquasecurity.github.io/trivy/) | Container image vulnerability scan (builds the `Dockerfile` image) |

The C++ extension is also built and its import verified as an optional CI step,
falling back cleanly to NumPy when unavailable.

---



## Contributing

Contributions of all sizes are welcome — from bug fixes and documentation improvements to new research modules and cryptographic integrations.

### Workflow

1. **Fork** the repository and clone your fork locally.
2. **Create a feature branch** from `main` using a descriptive name:
   ```bash
   git checkout -b fix/walk-forward-expanding-window
   git checkout -b feat/add-kalman-filter-regime
   git checkout -b docs/update-api-reference
   ```
3. **Make your changes.** Keep commits atomic and focused on a single concern.
4. **Run the full test suite** before pushing — CI is the source of truth for the current test count:
   ```bash
   pytest tests/ -v
   ```
5. **Add tests** for any new functionality. New research modules should have corresponding tests in `tests/` following the pattern established in `test_phase3.py` and `test_phase4.py`.
6. **Commit** using the [Conventional Commits](https://www.conventionalcommits.org/) format:
   ```
   feat: add Kalman-filter-based regime transition model
   fix: correct Newey-West lag selection for short time series
   docs: document DSR formula and assumptions in stat_tests.py
   test: add walk-forward parity tests for expanding window mode
   refactor: extract execution cost model into execution_model.py
   ```
7. **Push** and open a Pull Request against `main`. Describe *what* changed, *why*, and any design trade-offs you considered.

### Code Standards

- **Python style:** PEP 8. Type annotations on all public functions and class methods. Docstrings on every module, class, and non-trivial function.
- **Pydantic schemas:** Any new API endpoint must have a corresponding request schema in `schemas.py` with field validators and a descriptive docstring.
- **No breaking changes to existing API contracts** without a deprecation path documented in the PR description.
- **Research modules** must return plain Python types (no raw NumPy scalars or arrays) so all API responses are JSON-serialisable.
- **Cryptographic code** must not introduce new dependencies without explicit justification and review against the threat model in [SECURITY.md](SECURITY.md).

### Pull Request Checklist

- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] New functionality is covered by at least one new test
- [ ] Public functions have type annotations and docstrings
- [ ] No raw NumPy types leak into API response payloads
- [ ] `python -m py_compile` passes on all modified modules
- [ ] PR description explains the change and links to any relevant issues

### Security Issues

Do **not** open a public GitHub issue to report security vulnerabilities. Follow the responsible disclosure process documented in [SECURITY.md](SECURITY.md).

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

---

<div align="center">

*QuantumSentinel is research and informational software.*
*It is not financial advice and carries no certification for use in regulated financial systems.*

</div>
