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


class AsyncHybridCache:
    """
    Async Redis cache with auto-reconnect and in-memory fallback.
    Call ``await cache.connect()`` on app startup.
    On timeout/connection errors, retries every 30s automatically.
    In-memory fallback keeps the app functional when Redis is down.
    """

    _RECONNECT_COOLDOWN = 30  # seconds between reconnect attempts

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._available = False
        self._url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._last_fail_time = 0.0
        self._memory_cache: dict = {}  # In-memory fallback

    async def connect(self, url: Optional[str] = None) -> bool:
        """
        Try to connect to Redis. Returns True if successful.
        App continues normally if this returns False (uses in-memory fallback).
        """
        if url:
            self._url = url
        try:
            client = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                max_connections=20,
                retry_on_timeout=True
            )
            await client.ping()
            self._redis = client
            self._available = True
            self._last_fail_time = 0.0
            logger.info(f"[CACHE] ✅ Redis connected at {self._url}")
            return True
        except Exception as e:
            self._available = False
            self._last_fail_time = time.time()
            logger.warning(f"[CACHE] ⚠️  Redis unavailable ({e}) — using in-memory fallback")
            return False

    async def _try_reconnect(self) -> bool:
        """Auto-reconnect after cooldown period. Returns True if reconnected."""
        if time.time() - self._last_fail_time < self._RECONNECT_COOLDOWN:
            return False
        logger.info("[CACHE] 🔄 Redis auto-reconnect attempt...")
        return await self.connect(self._url)

    async def disconnect(self):
        """Gracefully close Redis connection. Call on app shutdown."""
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
        """
        # Try Redis
        if self._available:
            try:
                raw = await self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"[CACHE] Redis GET error for '{key}': {e}")
                if "Timeout" in str(e) or "Connection" in str(e):
                    self._last_fail_time = time.time()
                    self._available = False
        else:
            await self._try_reconnect()

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
                    self._last_fail_time = time.time()
                    self._available = False
        else:
            await self._try_reconnect()
        # BUG FIX: in-memory write above always succeeds, so return True
        return True

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
                    self._last_fail_time = time.time()
                    self._available = False
        # BUG FIX: return True if the key existed and was removed from memory cache
        return was_in_memory

    async def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern using SCAN cursor (non-blocking).
        Unlike KEYS, SCAN is safe in production with large keyspaces.
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
            if not await self._try_reconnect():
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
                return -2  # key not found
            if entry["expires_at"] is None:
                # BUG FIX: key exists but has no expiry — return -1, not -2
                return -1
            remaining = int(entry["expires_at"] - time.time())
            if remaining <= 0:
                del self._memory_cache[key]
                return -2  # expired
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