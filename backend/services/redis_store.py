"""Centralized Redis operations for distributed state.

Provides atomic operations for:
- PQC nonce dedup (qs:pqc:nonce:<hash>)
- Order nonce dedup (qs:order:nonce:<hash>)
- Kill switches (qs:risk:kill:<scope>:<id>)
- Session state (qs:session:<id>)
- Rate limiting (qs:rate:<key>)
- Refresh token cache (qs:refresh:<hash>)

All operations use NX/EX for atomicity. Graceful fallback to in-memory
stores when Redis is not configured (development mode).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-loop bridge for synchronous (threadpool) routes
# ---------------------------------------------------------------------------
# `redis.asyncio` connections are bound to the event loop that created them.
# Spinning up a throwaway `asyncio.new_event_loop()` per request — as earlier
# revisions did — leaves pooled connections attached to a loop that is then
# closed, so every subsequent Redis call raises "Event loop is closed" or
# "attached to a different loop". Instead, sync routes submit their coroutine
# to the single application loop captured at startup.
_app_loop: asyncio.AbstractEventLoop | None = None


def set_app_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Record the application's running event loop (called from the lifespan)."""
    global _app_loop
    _app_loop = loop


def run_sync(coro, timeout: float = 2.0):
    """Run a Redis coroutine from a synchronous route and return its result.

    Returns None (and closes the coroutine) if no application loop is
    available or the call fails/times out — Redis caching is an optimisation
    here, never the source of truth, so a failure must not break the request.
    """
    loop = _app_loop
    if loop is None or loop.is_closed():
        coro.close()
        return None
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)
    except Exception as exc:  # noqa: BLE001 - best-effort cache path
        _log.warning("Redis operation failed from sync context: %s", exc)
        return None

# In-memory fallback stores for development (single process)
_mem_lock = Lock()
_mem_nonces: dict[str, float] = {}
_mem_kill_switches: set[str] = set()
_mem_sessions: dict[str, dict] = {}
_mem_kv: dict[str, tuple[Any, float]] = {}  # key -> (value, expire_at)


def _cleanup_mem() -> None:
    """Remove expired in-memory entries."""
    now = time.time()
    expired = [k for k, (_, exp) in _mem_kv.items() if exp and exp < now]
    for k in expired:
        _mem_kv.pop(k, None)
    expired_nonces = [k for k, t in _mem_nonces.items() if t < now]
    for k in expired_nonces:
        _mem_nonces.pop(k, None)


# ---------------------------------------------------------------------------
# Atomic SET NX (nonce consumption / dedup)
# ---------------------------------------------------------------------------
async def set_nx_ex(redis_client, key: str, ttl: int = 300) -> bool:
    """Atomic set-if-not-exists with TTL. Returns True if set succeeded (first use).
    Returns False if key already existed (replay)."""
    if redis_client:
        try:
            result = await redis_client.set(key, "1", nx=True, ex=ttl)
            return result is not None
        except Exception as exc:
            _log.warning("Redis SET NX failed for %s: %s", key, exc)
            # In production we must not silently fall through
            from ..config import ENVIRONMENT
            if ENVIRONMENT == "production":
                raise
    # In-memory fallback for dev
    with _mem_lock:
        _cleanup_mem()
        now = time.time()
        if key in _mem_kv and _mem_kv[key][1] > now:
            return False
        _mem_kv[key] = ("1", now + ttl)
        return True


async def exists(redis_client, key: str) -> bool:
    """Check if a key exists."""
    if redis_client:
        try:
            return bool(await redis_client.exists(key))
        except Exception as exc:
            _log.warning("Redis EXISTS failed for %s: %s", key, exc)
            from ..config import ENVIRONMENT
            if ENVIRONMENT == "production":
                raise
    with _mem_lock:
        _cleanup_mem()
        return key in _mem_kv


async def set_key(redis_client, key: str, value: str = "1", ttl: int | None = None) -> None:
    """Set a key with optional TTL."""
    if redis_client:
        try:
            if ttl:
                await redis_client.set(key, value, ex=ttl)
            else:
                await redis_client.set(key, value)
            return
        except Exception as exc:
            _log.warning("Redis SET failed for %s: %s", key, exc)
            from ..config import ENVIRONMENT
            if ENVIRONMENT == "production":
                raise
    with _mem_lock:
        expire_at = time.time() + ttl if ttl else float("inf")
        _mem_kv[key] = (value, expire_at)


async def get_key(redis_client, key: str) -> str | None:
    """Get a value by key."""
    if redis_client:
        try:
            return await redis_client.get(key)
        except Exception as exc:
            _log.warning("Redis GET failed for %s: %s", key, exc)
            from ..config import ENVIRONMENT
            if ENVIRONMENT == "production":
                raise
    with _mem_lock:
        _cleanup_mem()
        entry = _mem_kv.get(key)
        return entry[0] if entry else None


async def delete_key(redis_client, key: str) -> None:
    """Delete a key."""
    if redis_client:
        try:
            await redis_client.delete(key)
            return
        except Exception:
            pass
    with _mem_lock:
        _mem_kv.pop(key, None)


# ---------------------------------------------------------------------------
# PQC nonce operations
# ---------------------------------------------------------------------------
async def consume_pqc_nonce(redis_client, nonce_bytes: bytes) -> bool:
    """Atomically consume a PQC handshake nonce. Returns True if unused (accepted)."""
    nonce_hash = hashlib.sha256(nonce_bytes).hexdigest()
    key = f"qs:pqc:nonce:{nonce_hash}"
    return await set_nx_ex(redis_client, key, ttl=300)


async def consume_order_nonce(redis_client, nonce: str, user_id: str) -> bool:
    """Atomically consume an order nonce. Returns True if unused."""
    nonce_hash = hashlib.sha256(f"{user_id}:{nonce}".encode()).hexdigest()
    key = f"qs:order:nonce:{nonce_hash}"
    return await set_nx_ex(redis_client, key, ttl=300)


# ---------------------------------------------------------------------------
# Kill switch operations
# ---------------------------------------------------------------------------
def _kill_key(scope: str, identifier: str | None) -> str:
    if identifier:
        return f"qs:risk:kill:{scope}:{identifier}"
    return f"qs:risk:kill:{scope}"


async def set_kill_switch(redis_client, scope: str, identifier: str | None = None,
                           enabled: bool = True) -> None:
    """Set or clear a kill switch."""
    key = _kill_key(scope, identifier)
    if enabled:
        await set_key(redis_client, key, "1")
        with _mem_lock:
            _mem_kill_switches.add(key)
    else:
        await delete_key(redis_client, key)
        with _mem_lock:
            _mem_kill_switches.discard(key)


async def is_kill_switch_active(redis_client, scope: str,
                                  identifier: str | None = None) -> bool:
    """Check if a kill switch is active."""
    key = _kill_key(scope, identifier)
    if redis_client:
        try:
            return bool(await redis_client.exists(key))
        except Exception:
            pass
    with _mem_lock:
        return key in _mem_kill_switches


async def list_kill_switches(redis_client) -> list[dict]:
    """List all active kill switches."""
    if redis_client:
        try:
            keys = []
            async for key in redis_client.scan_iter("qs:risk:kill:*"):
                parts = key.split(":")
                if len(parts) >= 4:
                    scope = parts[3]
                    identifier = parts[4] if len(parts) > 4 else None
                    keys.append({"scope": scope, "identifier": identifier})
            return keys
        except Exception:
            pass
    with _mem_lock:
        result = []
        for key in _mem_kill_switches:
            parts = key.split(":")
            if len(parts) >= 4:
                scope = parts[3]
                identifier = parts[4] if len(parts) > 4 else None
                result.append({"scope": scope, "identifier": identifier})
        return result


# ---------------------------------------------------------------------------
# Session state operations
# ---------------------------------------------------------------------------
async def store_session(redis_client, session_id: str, session_data: dict,
                         ttl: int = 3600) -> None:
    """Store a session in Redis (JSON serialized)."""
    import json
    await set_key(redis_client, f"qs:session:{session_id}",
                  json.dumps(session_data), ttl=ttl)


async def get_session(redis_client, session_id: str) -> dict | None:
    """Retrieve a session from Redis."""
    import json
    val = await get_key(redis_client, f"qs:session:{session_id}")
    if val:
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Refresh token cache (dual-write: PostgreSQL primary, Redis cache)
# ---------------------------------------------------------------------------
async def cache_refresh_token(redis_client, token_hash: str, user_id: str,
                                ttl: int = 86400) -> None:
    """Cache a refresh token mapping in Redis for fast lookup."""
    await set_key(redis_client, f"qs:refresh:{token_hash}", user_id, ttl=ttl)


async def get_cached_refresh_user(redis_client, token_hash: str) -> str | None:
    """Fast lookup of user_id from refresh token hash via Redis cache."""
    return await get_key(redis_client, f"qs:refresh:{token_hash}")


async def invalidate_refresh_token(redis_client, token_hash: str) -> None:
    """Remove a refresh token from Redis cache."""
    await delete_key(redis_client, f"qs:refresh:{token_hash}")


# ---------------------------------------------------------------------------
# HMAC nonce dedup for API key requests
# ---------------------------------------------------------------------------
async def consume_api_nonce(redis_client, key_id: str, nonce: str) -> bool:
    """Atomically consume an API request nonce."""
    nonce_hash = hashlib.sha256(f"{key_id}:{nonce}".encode()).hexdigest()
    key = f"qs:api:nonce:{nonce_hash}"
    return await set_nx_ex(redis_client, key, ttl=300)
