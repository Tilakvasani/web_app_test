"""
Redis & In-Memory Async Hybrid Cache Manager.
===============================================

Async Redis caching with auto-reconnect and graceful in-memory fallback.
All cache operations are no-ops when Redis is unavailable, so the
application continues to function without it.

Cache key patterns and TTLs:
    mcp:tool_resp:{name}:{hash}    — 5 minutes   (MCP tool API responses)
    mcp:history:{hash}             — 1 hour       (history compression summaries)
    mcp:sifter:{hash}              — 10 minutes   (smart sifter classifications)
    mcp:doc:chunk:{name}:{idx}     — 24 hours     (vector embeddings)

Public singleton:
    cache  — a pre-constructed ``AsyncHybridCache`` instance ready for import.

Usage::

    from database.redis_cache import cache
    await cache.connect()        # call once in FastAPI startup
    value = await cache.get("mykey")
    await cache.set("mykey", data, ttl=3600)
    await cache.delete("mykey")
    await cache.flush_pattern("mcp:tool_resp:*")
    await cache.disconnect()     # call in FastAPI shutdown
"""

import asyncio
import json
import os
import time
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger("mcp_backend")

# ── TTL Constants (seconds) ──────────────────────────────────────────────────
TTL_TOOL_RESPONSE    = 300      # 5 minutes
TTL_HISTORY_SUMMARY  = 3600     # 1 hour
TTL_SIFTER           = 600      # 10 minutes
TTL_EMBEDDING        = 86400    # 24 hours
TTL_DEFAULT          = 3600     # 1 hour fallback

# ── Key Builders ─────────────────────────────────────────────────────────────
KEY_TOOL_RESPONSE    = lambda name, h: f"mcp:tool_resp:{name}:{h}"
KEY_HISTORY          = lambda h: f"mcp:history:{h}"
KEY_SIFTER           = lambda h: f"mcp:sifter:{h}"
KEY_DOC_CHUNK        = lambda name, idx: f"mcp:doc:chunk:{name}:{idx}"

# Eviction runs every 10 minutes
_EVICTION_INTERVAL = 600


class AsyncHybridCache:
    """
    Async Redis cache with non-blocking auto-reconnect and in-memory fallback.
    Call ``await cache.connect()`` on app startup.

    FIX: reconnect attempts now run as background asyncio tasks so they never
    block a user-facing request (previously caused 8 s stalls per request).
    FIX: periodic background eviction prevents in-memory cache from growing
    unbounded when Redis is unavailable.
    """

    _RECONNECT_COOLDOWN = 30  # seconds between reconnect attempts

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._available = False
        self._url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._last_fail_time = 0.0
        self._reconnect_in_progress = False
        self._memory_cache: dict = {}  # In-memory fallback
        self._eviction_task: Optional[asyncio.Task] = None

    async def connect(self, url: Optional[str] = None) -> bool:
        """
        Try to connect to Redis. Returns True if successful.
        App continues normally if this returns False (uses in-memory fallback).
        Uses a 1 s connect timeout so startup is not delayed.
        """
        if url:
            self._url = url
        try:
            client = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1,   # FIX: was 5 s — fail fast
                socket_timeout=3,
                max_connections=20,
                retry_on_timeout=True
            )
            await client.ping()
            self._redis = client
            self._available = True
            self._last_fail_time = 0.0
            logger.info(f"[CACHE] ✅ Redis connected at {self._url}")
            self._start_eviction_loop()
            return True
        except Exception as e:
            self._available = False
            self._last_fail_time = time.time()
            logger.warning(f"[CACHE] ⚠️  Redis unavailable ({e}) — using in-memory fallback")
            self._start_eviction_loop()
            return False

    def _start_eviction_loop(self):
        """Start the background eviction task if not already running."""
        if self._eviction_task is None or self._eviction_task.done():
            try:
                self._eviction_task = asyncio.get_event_loop().create_task(
                    self._evict_expired_loop()
                )
            except RuntimeError:
                pass  # No event loop yet — will be started on first request

    async def _evict_expired_loop(self):
        """FIX: Background task that purges expired in-memory entries every 10 min."""
        while True:
            await asyncio.sleep(_EVICTION_INTERVAL)
            try:
                now = time.time()
                expired = [
                    k for k, v in self._memory_cache.items()
                    if v.get("expires_at") and now > v["expires_at"]
                ]
                for k in expired:
                    self._memory_cache.pop(k, None)
                if expired:
                    logger.info(f"[CACHE] Evicted {len(expired)} expired in-memory entries.")
            except Exception as e:
                logger.warning(f"[CACHE] Eviction error: {e}")

    def _schedule_reconnect(self):
        """
        FIX: Fire a non-blocking background reconnect attempt.
        Never awaited inline — so it never stalls a request.
        """
        if self._reconnect_in_progress:
            return
        if time.time() - self._last_fail_time < self._RECONNECT_COOLDOWN:
            return
        self._reconnect_in_progress = True

        async def _do_reconnect():
            logger.info("[CACHE] 🔄 Redis auto-reconnect attempt (background)...")
            await self.connect(self._url)
            self._reconnect_in_progress = False

        try:
            asyncio.get_event_loop().create_task(_do_reconnect())
        except RuntimeError:
            self._reconnect_in_progress = False

    async def disconnect(self):
        """Gracefully close Redis connection. Call on app shutdown."""
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            self._available = False
            logger.info("[CACHE] Redis connection closed.")

    @property
    def is_available(self) -> bool:
        """True if Redis is connected and responsive."""
        return self._available

    @property
    def redis_client(self) -> Optional[aioredis.Redis]:
        """Direct access to the async Redis client for vector store operations."""
        return self._redis if self._available else None

    # ── Core Operations ───────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieves and deserializes a value. Tries Redis first, falls back
        to in-memory cache if Redis is unavailable.
        FIX: reconnect is scheduled as a background task — never blocks.
        """
        if self._available:
            try:
                raw = await self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"[CACHE] Redis GET error for '{key}': {e}")
                if "Timeout" in str(e) or "Connection" in str(e):
                    self._available = False
                    self._last_fail_time = time.time()
                    self._schedule_reconnect()
        else:
            self._schedule_reconnect()  # FIX: non-blocking background reconnect

        # In-memory fallback
        entry = self._memory_cache.get(key)
        if entry:
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._memory_cache[key]
                return None
            return json.loads(entry["value"])
        return None

    async def set(self, key: str, value: Any, ttl: int = TTL_DEFAULT) -> bool:
        """
        Serializes and stores a value with TTL. Writes to BOTH Redis and
        in-memory cache for maximum resilience.
        """
        serialized = json.dumps(value, default=str)

        # Always store in-memory as fallback (with TTL tracking)
        self._memory_cache[key] = {
            "value": serialized,
            "expires_at": time.time() + ttl if ttl else None
        }

        # Try Redis
        if self._available:
            try:
                await self._redis.setex(key, ttl, serialized)
                return True
            except Exception as e:
                logger.warning(f"[CACHE] Redis SET error for '{key}': {e}")
                if "Timeout" in str(e) or "Connection" in str(e):
                    self._available = False
                    self._last_fail_time = time.time()
                    self._schedule_reconnect()
        else:
            self._schedule_reconnect()
        return True  # in-memory write always succeeds

    async def delete(self, key: str) -> bool:
        """Delete a single key from both Redis and in-memory cache."""
        was_in_memory = key in self._memory_cache
        self._memory_cache.pop(key, None)

        if self._available:
            try:
                await self._redis.delete(key)
                return True
            except Exception as e:
                logger.warning(f"[CACHE] Redis DEL error for '{key}': {e}")
                if "Timeout" in str(e) or "Connection" in str(e):
                    self._available = False
                    self._last_fail_time = time.time()
                    self._schedule_reconnect()
        return was_in_memory

    async def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern using SCAN cursor (non-blocking).
        Also clears matching keys from in-memory cache.
        """
        import fnmatch
        mem_keys_to_delete = [
            k for k in self._memory_cache
            if fnmatch.fnmatch(k, pattern)
        ]
        for k in mem_keys_to_delete:
            del self._memory_cache[k]

        if not self._available:
            self._schedule_reconnect()
            return len(mem_keys_to_delete)
        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            logger.warning(f"[CACHE] Redis FLUSH error for '{pattern}': {e}")
            return len(mem_keys_to_delete)

    async def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds. -1 if no TTL, -2 if not found."""
        if not self._available:
            entry = self._memory_cache.get(key)
            if entry is None:
                return -2
            if entry["expires_at"] is None:
                return -1
            remaining = int(entry["expires_at"] - time.time())
            if remaining <= 0:
                del self._memory_cache[key]
                return -2
            return remaining
        try:
            return await self._redis.ttl(key)
        except Exception:
            return -2

    async def exists(self, key: str) -> bool:
        """Return True if the key exists in Redis or in-memory cache."""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._memory_cache[key]
                return False
            return True
        if not self._available:
            return False
        try:
            return bool(await self._redis.exists(key))
        except Exception:
            return False

    # ── MCP-Specific Helpers ──────────────────────────────────────────────────

    async def get_tool_response(self, tool_name: str, args_hash: str) -> Optional[Any]:
        """Get a cached tool response."""
        return await self.get(KEY_TOOL_RESPONSE(tool_name, args_hash))

    async def set_tool_response(self, tool_name: str, args_hash: str, data: Any) -> bool:
        """Cache a tool response for 5 minutes."""
        return await self.set(KEY_TOOL_RESPONSE(tool_name, args_hash), data, TTL_TOOL_RESPONSE)

    async def get_history_summary(self, summary_hash: str) -> Optional[str]:
        """Get a cached history compression summary."""
        return await self.get(KEY_HISTORY(summary_hash))

    async def set_history_summary(self, summary_hash: str, summary: str) -> bool:
        """Cache a history summary for 1 hour."""
        return await self.set(KEY_HISTORY(summary_hash), summary, TTL_HISTORY_SUMMARY)

    async def get_sifter(self, sifter_hash: str) -> Optional[dict]:
        """Get a cached sifter classification result."""
        return await self.get(KEY_SIFTER(sifter_hash))

    async def set_sifter(self, sifter_hash: str, decision: dict) -> bool:
        """Cache a sifter decision for 10 minutes."""
        return await self.set(KEY_SIFTER(sifter_hash), decision, TTL_SIFTER)

    async def invalidate_tool_cache(self) -> int:
        """Clear all cached tool responses."""
        return await self.flush_pattern("mcp:tool_resp:*")

    async def invalidate_all(self) -> int:
        """Clear all MCP cache keys."""
        return await self.flush_pattern("mcp:*")

    async def cache_stats(self) -> dict:
        """Return cache health + key counts for each namespace."""
        stats = {
            "available": self._available,
            "memory_keys": len(self._memory_cache),
        }
        if not self._available:
            return stats
        try:
            info = await self._redis.info("server")
            tool_keys    = len([k async for k in self._redis.scan_iter("mcp:tool_resp:*")])
            history_keys = len([k async for k in self._redis.scan_iter("mcp:history:*")])
            sifter_keys  = len([k async for k in self._redis.scan_iter("mcp:sifter:*")])
            doc_keys     = len([k async for k in self._redis.scan_iter("mcp:doc:chunk:*")])
            stats.update({
                "redis_version": info.get("redis_version", "?"),
                "tool_responses": tool_keys,
                "history_summaries": history_keys,
                "sifter_cache": sifter_keys,
                "doc_chunks": doc_keys,
                "total_mcp_keys": tool_keys + history_keys + sifter_keys + doc_keys,
            })
        except Exception as e:
            stats["error"] = str(e)
        return stats


# ── Singleton Instance ────────────────────────────────────────────────────────
cache = AsyncHybridCache()
