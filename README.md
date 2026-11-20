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

`dataset_format.py` names frozen data by its content. Its typed tables are
daily bars keyed by symbol and session date, corporate actions, and the
instruments (exchange, timezone, quote currency). The
`qs-col-sha256-v1` hash covers the values, not a file. The dataset id
`qs-ds-sha256-v1:<hex>` also covers the provider, licence, adjustment policy and
session boundary, so snapshots that differ in any of them never share an id.
`dataset_store.py` keeps each dataset as one write-once file at
`qs-ds-sha256-v1/<2 hex>/<hex>` and rehashes it on every read. Golden vectors
for other implementations are in
[`tests/fixtures/qs_col_sha256_v1.json`](tests/fixtures/qs_col_sha256_v1.json).

`dataset_providers.py` turns symbols and a date range into datasets, one per
family (equities, indices, crypto, FX, continuous futures). Yahoo (yfinance) is
the only provider and is marked `personal_use_only`. Yahoo serves no raw
prices: its bars and dividends are split-adjusted as of the fetch, so its
datasets say `adjustment: split`. Crypto bars are UTC days of CoinMarketCap's
composite price, and sessions still open at retrieval are left out.
`dataset_registry.py` records each verified dataset in `dataset_versions`
(migration 0004). There, CHECKs tie the id to the stored manifest and triggers
refuse UPDATE, DELETE and TRUNCATE. `user_datasets` lists a dataset only to the
users who fetched it.

`POST /api/research/datasets` queues a snapshot job. The job process fetches
and stores the blobs under `DATASET_ROOT`. The worker then reads each blob back,
rehashes it against its id and the reported file hash, and registers only what
verifies.

Research jobs read prices only through `market_data.py`, never from yfinance
directly. Backtests, walk-forward, event backtests, alpha, factor, correlation,
optimisation, regime, neutral, pairs, latency and report requests take optional
`dataset_ids` (datasets the caller holds); without them the job snapshots its
symbols first. Either way the result's `market_data` and the audit event name
the datasets it was computed from, and the worker registers a snapshot before
it stores the result. Prices keep `yf.download(auto_adjust=True)`'s layout and
meaning: dividends are added back to the split-adjusted bars with
prod(1 − dividend / previous close), Yahoo's own Adj Close formula, except that
pence-quoted London listings get their whole dividend (Yahoo's Adj Close adds
back a hundredth of it). `period` is a trailing window that ends at the data's
last completed session, so a rerun on a result's `market_data.datasets`
computes on the same bars.

A dataset family can have a sealed holdout (`dataset_holdouts`, migration
0005), set by an operator with `python -m backend.manage set-holdout`. Research
jobs then see that family's bars and dividends only from before the holdout
start: `market_data.py` drops the rest before it computes the trailing window,
so a job computes what it would on datasets cut at the start. The worker reads
the starts before each job, and refuses the result (409) if a start changed
while the job ran; the result's `market_data.holdouts` and the audit event name
the starts that applied. A start can only move later, which releases the old
holdout to research; triggers refuse earlier starts and any UPDATE, DELETE or
TRUNCATE. Limits: the held-back rows are still in the blobs the worker can read,
and are kept from jobs by this code path, not by storage permissions; and since
Yahoo's prices are split-adjusted when fetched, a split after the start still
scales earlier price levels (returns are unchanged).

The holdout is read once per strategy, by the holdout evaluation
([`holdout_evaluation.py`](backend/services/holdout_evaluation.py), migration
0011). An operator queues it for a strategy whose validation passed
(`manage final-evaluate`). Only the research evaluator runs it, as
`qs_evaluator`:

- **It spends the attempt first.** Before it reads any data, it writes and
  commits a `holdout_attempts` row. One attempt is allowed per user, strategy
  version, dataset family and holdout start. The job has a single attempt,
  and is never handed back on shutdown, so a crash or a data error still
  spends it.
- **It grades only a strategy chosen without the holdout.** The verdict's
  signed manifest must name the current holdout start. The evaluation must
  then reproduce exactly when re-run on its datasets cut at that start.
- **The holdout run.** The strategy then runs on the uncut data from the
  evaluation's first session, and only the returns from the start on are
  graded.
- **The grade.** The holdout policy (`qs-holdout-policy-v1`) passes a
  strategy if and only if it has at least 63 holdout sessions and an
  annualised Sharpe ratio of at least 0.5. Anything else fails closed.
- **Where the figures go.** The grade, its figures and its signed manifest
  are kept only in `holdout_evaluations`, which the research worker cannot
  read. The job's result, its audit event and the owner's view say only
  whether it passed.
- **The holdout counts as a trial.** Each attempt counts as one more trial
  of the strategy's research family, so a strategy picked after a holdout
  grade is deflated for it.

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
│       ├── dataset_format.py            ← Dataset tables · qs-col-sha256-v1 logical hash · dataset ids · blob format
│       ├── dataset_store.py             ← Content-addressed, write-once dataset files, verified on every read
│       ├── dataset_providers.py         ← Provider interface · yfinance (personal use only) · dataset families
│       ├── dataset_registry.py          ← Immutable dataset_versions rows · who holds which dataset
│       ├── dataset_holdouts.py          ← Sealed holdout start per dataset family · only moves later
│       ├── holdout_evaluation.py        ← One-shot holdout grade of a passing strategy · research evaluator only
│       ├── market_data.py               ← Research prices from frozen datasets · total return · trailing windows
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
| `POST` | `/api/research/datasets` | Snapshot daily bars (`symbols`, `start`, `end` exclusive) as frozen, content-addressed datasets, one per family |
| `GET`  | `/api/research/datasets` | The caller's datasets: identity, licence, coverage, blob hashes (`?limit=`, up to 200) |
| `GET`  | `/api/research/datasets/{dataset_id}` | One of the caller's datasets |
| `POST` | `/api/research/latency-benchmark` | Per-stage p50/p99 latency profile + C++ vs Python speedup benchmark |
| `GET`  | `/api/research/cpp-status` | C++ extension load status · kernel names · active mode |
| `GET`  | `/api/research/holdout-evaluations` | The caller's holdout attempts, newest first, and whether each graded one passed; no figure of the holdout (`?limit=`, up to 100) |
| `GET`  | `/api/research/jobs` | The caller's recent research jobs, newest first, without results (`?limit=`, up to 100) |
| `GET`  | `/api/research/jobs/{job_id}` | One job's status and queue position; its result or error once finished |
| `POST` | `/api/research/jobs/{job_id}/cancel` | Cancel a queued job, or stop a running one (its process is killed) |
| `GET`  | `/api/research/queue` | Operator only: queued and running counts, oldest queued job's age, live research workers (`workers_online`) and live research evaluators (`evaluators_online`) |

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
- [ ] Size connection pools: (API processes + research workers + research evaluators) × (`DB_POOL_SIZE` + `DB_MAX_OVERFLOW`) must stay below PostgreSQL's `max_connections`
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
| `QS_PROCESS_ROLE` | No | Which process this is: `api` (default), `research-worker`, `research-evaluator` or `research-job`. In production the API requires every secret below; a research worker or the research evaluator only `DATABASE_URL`, the `SERVER_DSA_*` key and the PQC provider; a job process nothing. The role also decides which job kinds a worker claims: the API's embedded worker and `research-worker` claim every kind except the holdout evaluator's, `research-evaluator` claims only those (and reaps or purges no other job), and any other role claims nothing. No worker claims a kind its code does not know; that job stays queued. A worker starts its job processes as `research-job` with only the platform's variables (paths, locale, proxies, CA bundles, thread counts) and `ENVIRONMENT`. The API refuses to run under another role, and a worker refuses `research-job`. The AI tier's roles, `ai-worker` and `ai-job`, are not among them: `backend/config.py` refuses to load under them (see [AI tier](#ai-tier)) |
| `DATABASE_URL` | Prod | PostgreSQL URL (`postgresql+psycopg://user:password@host:5432/db`; `postgresql://` and `postgres://` are accepted). Any other database is refused at startup. Default: the `docker-compose.yml` dev database |
| `DB_POOL_SIZE` | No | Connections each process keeps open (default 20) |
| `DB_MAX_OVERFLOW` | No | Extra connections a process may open for bursts; they are closed when returned, so steady load should fit in `DB_POOL_SIZE` (default 0) |
| `DB_POOL_TIMEOUT_SECONDS` | No | Wait for a free connection before answering 503 (default 10) |
| `DB_STATEMENT_TIMEOUT_MS` | No | Server-side limit on any one statement (default 30000) |
| `DB_LOCK_TIMEOUT_MS` | No | Server-side limit on waiting for a lock (default 10000) |
| `DB_IDLE_IN_TRANSACTION_TIMEOUT_MS` | No | A session idle inside an open transaction is ended after this long (default 60000) |
| `DB_MIGRATE_ON_STARTUP` | No | `true` (default): API processes apply pending migrations at startup, serialised by a database lock; `false`: run `python -m backend.manage migrate` as a release step |
| `DB_APP_PASSWORD`, `DB_RESEARCH_WORKER_PASSWORD`, `DB_EVALUATOR_PASSWORD`, `DB_AI_WORKER_PASSWORD`, `DB_MIGRATOR_PASSWORD`, `DB_BACKUP_PASSWORD` | `provision-roles`; Docker production | Passwords for `qs_app`, `qs_research_worker`, `qs_evaluator`, `qs_ai_worker`, `qs_migrator` and `qs_backup`, read by `python -m backend.manage provision-roles` (each also as `*_FILE`). Needed to create a role; given for an existing role, its password is changed. `docker-compose.production.yml` also builds each service's `DATABASE_URL` from them, so they must be URL-safe |
| `CSRF_SECRET` | Prod | Signs CSRF tokens and keys the experiment-manifest HMACs; every API process must share it. Outside production each process makes up its own |
| `REFRESH_TOKEN_SECRET` | Prod | Required by the production startup check |
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
| `DATASET_ROOT` | No | Where dataset blobs are stored. The Docker image sets `/data/datasets`; production compose mounts the `datasets` volume there on the research worker (read-only on the research evaluator and the `backup` service). Default outside Docker: `data/datasets` in the repository |

---

## Database (PostgreSQL)

PostgreSQL 16 or newer is the only supported database; any other `DATABASE_URL` is refused at startup.

- **Migrations.** The schema is versioned with Alembic in [`backend/migrations`](backend/migrations). Each API process applies pending revisions at startup, serialised across processes by a database lock and atomic thanks to PostgreSQL's transactional DDL. To migrate as a separate release step instead, set `DB_MIGRATE_ON_STARTUP=false` and run `python -m backend.manage migrate`. Research workers wait until the schema is at their code's revision.
- **Moving from SQLite** (releases before 1.3): `python -m backend.manage import-sqlite path/to/quantumsentinel.db` copies every row into the configured, empty PostgreSQL database in one transaction, then verifies the row counts and the audit hash chain before committing. It refuses a database that already has rows.
- **Duplicate positions** (deployments upgraded from earlier releases): two fills for one user at the same moment could leave the `positions` table with more than one row per asset, overstating holdings and equity. Fills now rebuild positions inside their own transaction, and each user's next fill corrects their rows; `python -m backend.manage repair-positions` corrects every affected user at once and is safe to run while the API serves orders.
- **Audit chain verification.** `GET /api/security/audit-chain` checks only the links added since that API process last verified the chain, which took under a second at 1M links on the laptop below; nearly all of that is counting events that have no link. A process's first check, and any check after the chain failed one, verifies every link. A full check verifies an ML-DSA signature per link (about 11 s per 50,000 links on the same laptop), and it is the only check that sees rows edited behind the newest verified link, so run `python -m backend.manage verify-audit-chain` on a schedule; it exits 1 if any link fails. `?full=true` runs a full check from the API.
- **Dataset holdouts.** `python -m backend.manage set-holdout EQUITY_EOD 2025-01-01 --by <who> --reason <why>` keeps a dataset family's data from that date on away from research jobs (families: `EQUITY_EOD`, `FUND_NAV_EOD`, `INDEX_EOD`, `CRYPTO_SPOT_EOD`, `FX_SPOT_EOD`, `FUTURES_CONTINUOUS_EOD`). The date can only move later; moving it later releases the old holdout to research. Each change is an audit event (`DATASET_HOLDOUT_SET`). `python -m backend.manage holdouts` prints the current starts. A job that is running when a start is set fails with 409 and can be run again.
- **Holdout evaluation.** Set the family's holdout before the strategy's first backtest. Once the strategy's validation passes, queue its one holdout grade with `python -m backend.manage final-evaluate <validation job id> --period <its evaluation's period> --capital <its initial capital> --by <who> --reason <why>`. The command refuses:
  - a verdict that did not pass;
  - a verdict reached under another holdout start;
  - the wrong period or capital;
  - an attempt already spent.

  The research evaluator must be running. Queueing is an audit event (`HOLDOUT_EVALUATION_QUEUED`), and so is the grade (`HOLDOUT_EVALUATION`, pass or fail only).
- **Connection budget.** Every API process, research worker and research evaluator has its own pool: keep (processes) × (`DB_POOL_SIZE` + `DB_MAX_OVERFLOW`) below the server's `max_connections` (100 by default), leaving room for maintenance sessions. The default of 20 per process fits, for example, two API processes, one research worker and one research evaluator (80 connections).
- **Large tables.** The indexes added by migration `0002` are created `IF NOT EXISTS`; on a large `trades` or `audit_logs` table, build them first with `CREATE INDEX CONCURRENTLY` under the same names to avoid blocking writes during the upgrade.
- **Connection poolers.** Each session is configured when it connects (UTC, statement/lock/idle timeouts) and the order sweeper uses a session-level advisory lock, so a PgBouncer in front of the database must use session pooling, not transaction pooling.

### Database roles

Connected as the superuser that the `postgres` image creates from `POSTGRES_USER`, the application bypasses every grant, row-level security policy and trigger, and any SQL it runs can drop the schema, read server files or run programs on the database host (`COPY ... TO PROGRAM`). [`backend/db_roles.py`](backend/db_roles.py) defines seven roles instead:

| Role | Login | Privileges |
|---|---|---|
| `qs_owner` | No | Owns the schema and every object in it |
| `qs_migrator` | Yes | None of its own. Migrations run after `SET LOCAL ROLE qs_owner`, so everything they create is owned by `qs_owner` |
| `qs_app` | Yes | The API (and the research worker it embeds, outside Docker production): exactly the table privileges in `APP_TABLE_PRIVILEGES` (`SELECT` only on `alembic_version`). No DDL or `TRUNCATE`; it cannot change roles, ownership or triggers |
| `qs_research_worker` | Yes | A research worker run as its own service: exactly `WORKER_TABLE_PRIVILEGES`. That covers the job queue, research results, the server signing key, backtest history (insert only) and the audit trail (read and append only). It has nothing on users, accounts, orders, positions, API keys, user keys, refresh tokens or webhooks |
| `qs_evaluator` | Yes | The research evaluator, which grades strategies on their sealed holdout: exactly `EVALUATOR_TABLE_PRIVILEGES`. It reads and updates jobs (to claim and finish its own) but cannot queue or delete one, registers its heartbeat and the server signing key, and reads and appends to the audit trail. It reads what it grades: the verdict and its manifest, the evaluation, the strategy version and the holdout start. It is the only role that writes `holdout_attempts` and `holdout_evaluations`. It reaches no user, account, order or key table, and writes no strategy, dataset, trial or verdict |
| `qs_ai_worker` | Yes | The AI tier (see [AI tier](#ai-tier)): exactly `AI_WORKER_TABLE_PRIVILEGES` in the AI schema, and no privilege at all on the application schema, not even `USAGE`. It reads the feature flags, claims, finishes and purges AI jobs but cannot queue one, and keeps its own heartbeat. It records harvested papers in the document registry and moves the harvest cursors and rate slots, but cannot delete any of them. It reaches no job queue, worker registry, audit trail, key, user or trading table of the trusted tier. Its role carries a connection limit of 8 and its statement (30 s), lock (10 s) and idle-in-transaction (60 s) timeouts |
| `qs_backup` | Yes | `pg_read_all_data`, for `pg_dump` |

The AI tier's tables live in a schema of their own, `ai`, beside `public`, also owned by `qs_owner`. Only `qs_app` and `qs_ai_worker` hold `USAGE` on it; `qs_app` holds `APP_AI_TABLE_PRIVILEGES` there: it sets flags, queues AI jobs and reads the document registry and the harvest cursors, and never claims or finishes a job or writes a document or a cursor. No login role may run a routine of either schema.

The privilege maps are allow-lists. A table `APP_TABLE_PRIVILEGES` does not name gets no privilege, and the test suite fails until the table is listed; a table `WORKER_TABLE_PRIVILEGES` or `EVALUATOR_TABLE_PRIVILEGES` does not name is out of that role's reach. Tests run every kind of research job through a worker connected as `qs_research_worker`, and a holdout grade through an evaluator connected as `qs_evaluator`, and check that every other table and every privileged statement is refused to each. Grants are reapplied after every migration that changes the schema revision.

To move a deployment over:

1. As the superuser (`DATABASE_URL` naming it), with `DB_APP_PASSWORD`, `DB_RESEARCH_WORKER_PASSWORD`, `DB_EVALUATOR_PASSWORD`, `DB_AI_WORKER_PASSWORD`, `DB_MIGRATOR_PASSWORD` and `DB_BACKUP_PASSWORD` set, run `python -m backend.manage provision-roles`. It migrates the schema to the latest revision first, then creates the roles and hands them both schemas in one transaction. Only SCRAM verifiers reach the server, so no password appears in its logs. It can run while the application serves requests. Changing a table's owner needs the table's exclusive lock, and taking those one at a time deadlocked with order traffic in a rehearsal (provisioning held `positions` and waited for `users`, while an order held `users` and waited for `positions`). So the command takes every lock it needs in one `LOCK ... NOWAIT`. If any table is in use, it lets go of everything and tries again moments later; queries wait at most a moment. If the tables stay busy for 10 s, it exits 1 without changing anything. It is safe to run again, and a second run takes no table locks.
2. Point `DATABASE_URL` for the API and workers at `qs_app`. With `DB_MIGRATE_ON_STARTUP=true` they start normally while the schema is current, and refuse to start if a migration is pending.
3. Run each release's migrations as `qs_migrator`: `DATABASE_URL=<qs_migrator URL> python -m backend.manage migrate`.
4. Take backups as `qs_backup`: `pg_dump -U qs_backup`. A dump made this way records `qs_owner` as the owner of everything; create the roles before restoring it, or restore with `--no-owner`. A dump holds only dataset metadata, so back up the blobs under `DATASET_ROOT` too, after each dump (see [Backups](#backups)).

Running `provision-roles` again restores the intended role attributes, memberships, ownership and grants and removes anything extra. On a schema that has not been provisioned, migrations run as whoever connects, as before.

`docker-compose.production.yml` is wired this way:

- The API connects as `qs_app` and runs no research itself (`RESEARCH_WORKER_MODE=external`).
- The `research-worker` service runs research jobs as `qs_research_worker`, limited to 1 CPU and 1 GB. Run more with `--scale research-worker=N`.
- The worker holds only the secrets it uses (`QS_PROCESS_ROLE=research-worker`): its database URL and the server signing key. The API's other secrets, and their `*_FILE` forms, are blanked for it. Its job processes start with no secret at all.
- The `research-evaluator` service (`QS_PROCESS_ROLE=research-evaluator`) runs only the jobs that grade a strategy on its sealed holdout, as `qs_evaluator`, limited to 1 CPU and 1 GB. It holds the same secrets as the worker, sits only on the `data` network, so it cannot reach the internet, and mounts the `datasets` volume read-only. Research workers never claim its jobs, and it claims no other job.
- The `ai-worker` service runs the AI tier's jobs as `qs_ai_worker` (see [AI tier](#ai-tier)), limited to 1 CPU and 1.5 GB. It loads no env file: its environment is its own database URL, its process role and the proxy, nothing else.
- The `smokescreen` service is the AI tier's only way to the internet: [Stripe's smokescreen](https://github.com/stripe/smokescreen) `v0.1.0`, built from source at a pinned commit ([`deploy/smokescreen`](deploy/smokescreen)).
- A one-off `migrate` service applies migrations as `qs_migrator`; the API, the workers and the evaluator start only once it has succeeded.
- The `backup` service, under its own profile, dumps as `qs_backup` every 6 hours and then copies the new dataset blobs (see [Backups](#backups)).
- A `provision` service, under its own profile, runs `provision-roles` as the superuser.
- Only `postgres` and `provision` receive the superuser's password. Every service that loads `.env.production` blanks every database password variable, so that file cannot leak one into the API or the worker.
- `redis` and the research evaluator are only on the `data` network, which has no route out. So the database cannot reach the internet, and nginx (only on `edge`) cannot reach the database. The API and the research worker join both networks, because they need the internet for market data.
- The AI worker shares no network with the API, redis, nginx or the research services. It reaches `postgres` on `ai_db` and smokescreen on `ai_egress`, and neither network has a route out; `postgres` joins `data` and `ai_db`. Only smokescreen also joins `ai_out`, a network with a route to the internet.

To move an existing Docker deployment over, add `DB_APP_PASSWORD`, `DB_RESEARCH_WORKER_PASSWORD`, `DB_EVALUATOR_PASSWORD`, `DB_AI_WORKER_PASSWORD`, `DB_MIGRATOR_PASSWORD` and `DB_BACKUP_PASSWORD` to `.env.production`, then run, in this order:

```bash
docker compose -f docker-compose.production.yml --env-file .env.production --profile provision run --rm provision
docker compose -f docker-compose.production.yml --env-file .env.production up -d
```

The first command runs while the API keeps serving. The second replaces the API container; in CI the API was back about 18 s later. Run them in this order. `up` before `provision` takes the API down: Compose replaces the API container, then `migrate` cannot log in (`password authentication failed for user "qs_migrator"`), so the new API never starts. Running the two commands then restores it; that took 19 s in CI.

A deployment that already runs on these roles, provisioned before `qs_research_worker` existed, takes the same two commands after `DB_RESEARCH_WORKER_PASSWORD` is added. Without that variable Compose refuses to start; with it but before `provision`, the `research-worker` service cannot log in. It retries every 2 s (`database not reachable yet` in its log), so research jobs wait in the queue and start once `provision` has run.

Upgrading also applies migration `0011`, which adds the append-only holdout tables, and `provision` grants them: `qs_app` reads both, `qs_research_worker` reads only the attempts (it counts them as trials), and only `qs_evaluator` writes them.

The same holds for `qs_evaluator`: a deployment provisioned before it existed adds `DB_EVALUATOR_PASSWORD` to `.env.production` and runs the same two commands, `provision` first. Until then Compose refuses to start, or, with the variable set but before `provision`, the `research-evaluator` service retries its login every 2 s; holdout grades wait in the queue meanwhile, and no other job does. Upgrading also applies migration `0010`, which records each worker's role in `research_workers`, so a job's status counts only the workers that can run it.

Upgrading to migration `0012` (the AI tier) takes the same two commands, `provision` first, after `DB_AI_WORKER_PASSWORD` is added to `.env.production`. `qs_owner` may not create schemas, so `provision` (as the superuser) creates the `ai` schema and `qs_ai_worker`; `migrate` then fills the schema as `qs_owner`. Without the variable Compose refuses to start. Run `up` before `provision` and `migrate` stops with `schema ai does not exist and qs_owner may not create it: run python -m backend.manage provision-roles ...`, so the new API never starts; running `provision` and then `up` restores it.

The CI job `deploy-compose` checks this end to end on every push. It deploys the stack with Docker and writes data as the superuser. It then provisions while the API runs and brings the stack up again. It checks eight things:

- only `qs_app`, `qs_research_worker` and `qs_evaluator` sessions reach the database, and none of the API, the worker, the evaluator and `migrate` holds the superuser password;
- the earlier data is intact, and research jobs run in the `research-worker` container, none in the evaluator, whose heartbeat records its role;
- the worker and the evaluator hold none of the API's secrets, and the worker's live job process has no database URL or key in its environment;
- nginx cannot resolve `postgres`, and neither `postgres` nor the evaluator can reach the internet, while the API and the worker can;
- the evaluator reads a dataset blob the worker stored but cannot write the `datasets` volume;
- the backup is a full dump made as `qs_backup`, and it holds a dataset blob the worker stored, which verifies against its hash, but not a blob the worker was still writing;
- after the `datasets` volume is deleted, `restore-datasets` brings that blob back, and the worker reads it and can write into every restored directory;
- the API, the worker and the evaluator restart as their own roles.

It also deploys a fresh install the same way (`provision`, then `up`), starting the `backup` service before the worker, and checks that the worker can still store a blob.

### AI tier

The AI tier ([`backend/ai`](backend/ai)) is where Research Intelligence work that handles untrusted input will run, such as fetching paper metadata from the internet. This release adds its skeleton: its own database role and schema, feature flags, a job queue, a worker and a self-test job. It holds the trusted tier at arm's length:

- **Its own role and schema.** The AI worker connects as `qs_ai_worker` (`AI_DATABASE_URL`), which has no privilege on the application schema (see [Database roles](#database-roles)). Its tables, `flags`, `jobs`, `workers`, `documents`, `sources` and `harvests`, are in the `ai` schema.
- **No trusted code.** No module of `backend.ai` imports anything else of `backend`; a test follows every import statement to check it, and another loads every module in a fresh interpreter. `backend/config.py`, which loads the API's keys, refuses to load under `QS_PROCESS_ROLE=ai-worker` or `ai-job`, so an accidental import fails at once.
- **No trusted secret.** In production the worker refuses to start if it is given `DATABASE_URL`, any database password but its own, a signing or encryption key, the CSRF or refresh-token secret, or Redis's URL or password, or the `*_FILE` form of any of them.
- **Job processes.** The worker runs each job in a fresh interpreter (`python -E -s`) with `QS_PROCESS_ROLE=ai-job`, none of the worker's file descriptors and an allow-listed environment: paths, locale, proxies, CA bundles and `ENVIRONMENT`, and no database URL or password. On Linux it is also limited in memory (`AI_JOB_MEMORY_BYTES`, default 1 GiB), CPU time, open files and core dumps. It runs only the targets listed in [`backend/ai/kinds.py`](backend/ai/kinds.py), and a job that runs past its kind's timeout is killed and fails.
- **One way out.** In Docker the AI worker has no route to the internet. Its `HTTP_PROXY` and `HTTPS_PROXY` name smokescreen, and its job processes inherit them. Smokescreen forwards a request only to the hosts in [`deploy/smokescreen/acl.yaml`](deploy/smokescreen/acl.yaml): `oaipmh.arxiv.org` (arXiv's OAI-PMH; `export.arxiv.org/oai2` redirects there), `api.openalex.org` and `api.crossref.org`. It refuses every other host with `407`. It also refuses any loopback, private (so every Docker network), link-local (cloud metadata) or otherwise non-public address, even for a listed host. It logs each decision as a `CANONICAL-PROXY-DECISION` JSON line (`docker compose logs smokescreen`). To allow another host, add it to `acl.yaml` and rebuild the `smokescreen` image.
- **Feature flags, off by default.** A stage runs only while the master flag `research_intelligence` and its own flag are both on; a flag that was never set is off. The flags are checked when a job is queued, inside the worker's claim query and every `AI_WORKER_HEARTBEAT_SECONDS` while a job runs. Switching a stage off stops its running jobs and hands them back to the queue, uncounted, and switching it on lets the next poll claim them; nothing restarts.
- **Document registry.** Harvested papers go into `ai.documents` (migration `0013`), one row per paper keyed by source and identifier. A job process parses each OAI-PMH page ([`backend/ai/oai.py`](backend/ai/oai.py)). It accepts UTF-8 only and refuses any response with a DTD before parsing, so no entity can be declared or expanded. The worker then checks every field of the job's result again before writing ([`backend/ai/documents.py`](backend/ai/documents.py)). Identifiers, dates, categories, DOIs and URLs must match their formats. Text loses control and format characters, including bidirectional overrides and zero-width characters, and every field is capped in length. A record that fails a check is dropped and counted; the rest of its page is kept. Harvesting a page again is idempotent: a record replaces the stored one only if its datestamp is later, or the same day with different content (`content_hash`, scheme `qs-doc-sha256-v1`). A paper arXiv reports deleted keeps its row with `deleted_at` set, and no role may delete a row. arXiv's metadata format carries no version number, so a row holds the latest metadata of its paper. Each row also records the paper's licence URL and a reuse class for its full text. `open` (CC BY, CC BY-SA, CC0) may be shown and quoted with attribution. `noncommercial` covers any NC licence. `restricted` covers arXiv's non-exclusive licence and anything unrecognised or missing, and allows only a link. The metadata itself is CC0.
- **arXiv harvest.** An `arxiv_sync` job ([`backend/ai/harvest.py`](backend/ai/harvest.py), stage `arxiv_sync`) fills the registry from one arXiv set over OAI-PMH, metadata only. The worker builds every request itself: the URL is `AI_ARXIV_OAI_URL` (default and, in production, only `https://oaipmh.arxiv.org/oai`), and the job's payload names only the set, the first day and a page limit. Each page is fetched and parsed in a job process of its own ([`backend/ai/oai_fetch.py`](backend/ai/oai_fetch.py)), which follows no redirect, takes its proxy from `HTTP_PROXY`/`HTTPS_PROXY` only and refuses a body over 16 MiB. The job walks the set in windows of 31 days from its cursor (`ai.harvests`, migration `0014`) to today, one page at a time. It records each page and moves the cursor in one transaction that first checks the job is still its own, so a sync stopped at any point resumes where it stopped, and recording a page twice changes nothing. Each window starts on the last day of the one before it, since OAI-PMH datestamps are days, and the cursor never passes the day arXiv says it is. All workers share one rate slot per source (`ai.sources`): at most one request every 3 s, as arXiv asks, and later if arXiv sends `Retry-After`. A failed request (no answer, HTTP 429 or 5xx) is retried after its `Retry-After` (at most 15 min) or 30 s, five times in a row at most. An expired or refused resumption token restarts its window. Anything else fails the job, and the reason stays on the cursor (`last_error`). At most one sync is queued or running at a time. The requests carry the User-Agent `quantumsentinel-ai-harvester/1` and nothing about the operator.
- **Metadata lookups.** A `metadata_sync` job ([`backend/ai/enrich.py`](backend/ai/enrich.py), stage `metadata_sync`) asks OpenAlex or Crossref about the registry's papers and records what each says in `ai.document_metadata` (migration `0015`): its status (`found`, `not_found` or `refused`, with the reason), the DOI asked about, the service's id for the work, its citation count, whether it is retracted, and a few bounded fields (type, venue, publication date, topic and field at OpenAlex; publisher, licence URLs and update types such as `retraction` at Crossref). OpenAlex is asked about arXiv's own DOI for each paper (`10.48550/arXiv.<id>`), which it resolves to the merged work, journal version included; Crossref only about the journal DOI arXiv lists, since arXiv's DOIs are DataCite's. A sync looks up every paper never looked up at that service or changed since, every found paper last checked over 30 days ago and every other over 3 days ago, newest first. The worker builds each URL from `AI_OPENALEX_URL` or `AI_CROSSREF_URL` (fixed in production) and the DOI, which must match Crossref's DOI pattern and is percent-encoded whole, so it stays one path segment; a DOI that does not match is recorded as `refused` without a request. Each lookup runs in a job process of its own ([`backend/ai/metadata_fetch.py`](backend/ai/metadata_fetch.py)) that follows no redirect, refuses a body over 4 MiB and returns only those fields, which the worker checks again before it records them. Each service has its own rate slot, one request a second. HTTP 404 or 410 is recorded as `not_found` and 400 as `refused`; no answer, 5xx, or 429 with a wait of up to 15 min is retried, five times in a row at most. A 429 with a longer wait means the service's daily allowance is spent: the sync ends with `rate_limited` and the service's slot is held back until the allowance resets (at most 24 h). Any other answer fails the job. At most one sync per service is queued or running at a time. OpenAlex lets anyone look up a work by DOI without a key, within a daily allowance of 1,000 requests per address; its free API key (`AI_OPENALEX_API_KEY`) raises that to 10,000. The worker sends the key only to OpenAlex, only in an `Authorization: Bearer` header of the job process that makes the lookup, and keeps it out of every URL, payload the database stores, result, log line and the job process's environment.

Operator commands, each audited (`AI_FLAG_SET`, `AI_JOB_QUEUED`):

```bash
python -m backend.manage flags                                   # every flag, its state and who set it
python -m backend.manage set-flag research_intelligence on --by <who> --reason <why>
python -m backend.manage set-flag selftest on --by <who> --reason <why>
python -m backend.manage ai-selftest --by <who> [--hold <seconds>]   # prints the job id
python -m backend.manage set-flag arxiv_sync on --by <who> --reason <why>
python -m backend.manage ai-arxiv-sync q-fin --by <who> --from 2026-01-01 [--max-pages <n>]   # the first sync of a set
python -m backend.manage ai-arxiv-sync q-fin --by <who>         # later syncs go on from the cursor
python -m backend.manage set-flag metadata_sync on --by <who> --reason <why>
python -m backend.manage ai-metadata-sync openalex --by <who> [--max-docs <n>]   # or crossref
python -m backend.manage ai-job <job id>                         # its state and result
```

The self-test job reports what an AI job process runs with: the names (never the values) of its environment variables, its process role, any module of the trusted tier it imported (there must be none) and its resource limits. `--hold` keeps it running for up to 30 s, so that switching `selftest` off can be seen to stop it.

An arXiv sync's result counts its pages, windows and records (inserted, updated, unchanged, deleted, rejected) and says whether it caught up with today. arXiv updates its metadata once a day, so one sync a day keeps a set current, for example from cron: `17 6 * * * cd /srv/quantumsentinel && docker compose -f docker-compose.production.yml exec -T quantumsentinel python -m backend.manage ai-arxiv-sync q-fin --by cron`. While a sync is queued or running, another is refused, so an overlapping run does nothing. Thank you to arXiv for use of its open access interoperability.

A metadata sync's result counts the papers it looked up (`found`, `not_found`, `refused`, and `changed`: those whose record differs from the last lookup) and its retries, and says whether it caught up or stopped because the service's allowance was spent (`rate_limited`). Run it daily after the arXiv sync, one job per service: `47 6 * * * ... python -m backend.manage ai-metadata-sync openalex --by cron` and the same with `crossref`.

Run the worker with `QS_PROCESS_ROLE=ai-worker AI_DATABASE_URL=postgresql://qs_ai_worker:<password>@<host>:5432/<db> python -m backend.ai.worker`. Its other settings: `AI_DB_SCHEMA` (default `ai`), `AI_WORKER_POLL_SECONDS` (2), `AI_WORKER_HEARTBEAT_SECONDS` (5), `AI_JOB_LEASE_SECONDS` (60, more than twice the heartbeat; a job whose worker stops renewing it is reaped after this), `AI_JOB_RESULT_MAX_BYTES` (1 MiB), `AI_JOB_RETENTION_DAYS` (7), `AI_ARXIV_OAI_URL`, `AI_OPENALEX_URL` and `AI_CROSSREF_URL` (the services' endpoints; they may point elsewhere only outside production, for testing) and `AI_OPENALEX_API_KEY` (optional; in Docker, set it in the `.env` that `docker compose` reads). Each may also be given as a file named by `<NAME>_FILE`. In Docker the `ai-worker` service runs it this way, behind smokescreen (see the `docker-compose.production.yml` wiring under [Database roles](#database-roles)). It starts once `migrate` has succeeded and smokescreen accepts connections. `pg_dump` as `qs_backup` includes the `ai` schema.

The job resource limits are applied on Linux only: macOS refuses to lower `RLIMIT_AS` below the address space an arm64 interpreter has already reserved, so there a job runs without them.

### Backups

The `backup` service ([`deploy/backup.sh`](deploy/backup.sh)) runs a pass every 6 hours. Start it with:

```bash
docker compose -f docker-compose.production.yml --env-file .env.production --profile backup up -d backup
```

Each pass dumps the database as `qs_backup` into the `postgres_backups` volume (`quantumsentinel-<UTC time>.dump`), then copies each dataset blob that the backup lacks into `postgres_backups` under `datasets/`. A copy is checked against its source's SHA-256 before it takes the blob's name. Dumps older than 7 days are deleted once a new dump has succeeded. Blobs are never deleted.

The order is what makes a dump restorable. A dataset is registered in the database only after its blob is stored and read back, and nothing deletes a blob, so blobs copied after a dump include every blob that dump refers to. Every dump, old or new, restores with the current `datasets/` backup. A pass whose dump or copies fail logs `backup failed` and tries again at the next interval.

The service mounts the `datasets` volume read-only. The worker writes each blob to a `.tmp-*` file that only the worker can read, then renames it, and a pass skips those files.

Both volumes are on the Docker host, so a lost host takes the backups with it. Copying `postgres_backups` off the host is the operator's job.

To restore:

1. Restore a dump with `pg_restore`. Any dump will do. It records `qs_owner` as the owner of everything, so create the roles first, or restore with `--no-owner` (see [Database roles](#database-roles)).
2. Bring the blobs back:

   ```bash
   docker compose -f docker-compose.production.yml --env-file .env.production --profile restore run --rm restore-datasets
   ```

   This copies each blob that the `datasets` volume lacks from the backup, checking each copy the same way. It never replaces a blob that is already there. It runs as the image's `app` user with no network, so the worker can write into every directory it creates.
3. Start the stack with `up -d`.

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

# 2. Fill production env (every password URL-safe: python -c "import secrets; print(secrets.token_urlsafe(32))")
cp .env.production.example .env.production

# 3. Place TLS certs at deploy/tls/

# 4. Create the least-privilege database roles (once; again only to change a role's password)
docker compose -f docker-compose.production.yml --env-file .env.production --profile provision run --rm --build provision

# 5. Deploy
docker compose -f docker-compose.production.yml --env-file .env.production up -d --build
```

The API connects as `qs_app`, never as the superuser; see [Database roles](#database-roles). `postgres` and `redis` run as their images' unprivileged users, because every service drops all Linux capabilities. Started as root, their entrypoints need `CHOWN`, `SETUID` and `SETGID` to switch user, and without those capabilities neither starts.

> [!NOTE]
> In `docker-compose.production.yml`, research jobs run in the `research-worker` service as `qs_research_worker`, and the web container runs none (`RESEARCH_WORKER_MODE=external`). Outside it, each API process starts its own worker subprocess by default (`embedded`, one per gunicorn worker), so nothing else needs to be started. Either way, research code cannot reach trading or identity code. [`tests/test_research_boundary.py`](tests/test_research_boundary.py) follows every import statement from the worker and from each job kind's target, including imports inside functions, and fails if any path reaches the API, the operator CLI, authentication, trading, the paper broker, portfolios, integrations or Redis.

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
