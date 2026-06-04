"""
LangGraph Built-in Memory Manager.
====================================

Fully replaces Redis + AsyncHybridCache using LangGraph's native memory:

  MemorySaver     — per-thread conversation checkpointing for the ReAct agent.
                    LangGraph automatically stores and replays message history
                    per thread_id — no manual history management needed.

  InMemoryStore   — general-purpose namespace K/V store for sifter decisions,
                    tool-response caches, and embedding vectors.

Public singletons (drop-in replacements for the old redis_cache.py exports):
    cache         — LangGraphMemoryManager  (same async API as AsyncHybridCache)
    checkpointer  — MemorySaver instance    (passed to create_agent)

Key differences from Redis version:
  - No network dependency, no connection pool, no reconnect logic.
  - Data lives in process memory; clears on server restart (acceptable for
    caches — they're best-effort anyway).
  - `redis_client` property always returns None; vector_store.py now uses
    its own in-memory dict instead of raw Redis hash commands.
  - TTLs are enforced lazily (on next GET) via an internal expiry index.
"""

import asyncio
import fnmatch
import logging
import time
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger("mcp_backend")

# ── TTL Constants (seconds) — identical to redis_cache.py ─────────────────────
TTL_TOOL_RESPONSE   = 300       # 5 min
TTL_HISTORY_SUMMARY = 3600      # 1 hour
TTL_SIFTER          = 600       # 10 min
TTL_EMBEDDING       = 86400     # 24 hours
TTL_DEFAULT         = 3600      # 1 hour fallback

# ── Key Builders — identical to redis_cache.py so all callers still work ───────
KEY_TOOL_RESPONSE = lambda name, h: f"mcp:tool_resp:{name}:{h}"
KEY_HISTORY       = lambda h:       f"mcp:history:{h}"
KEY_SIFTER        = lambda h:       f"mcp:sifter:{h}"
KEY_DOC_CHUNK     = lambda name, i: f"mcp:doc:chunk:{name}:{i}"

# Namespace used for all cache entries inside InMemoryStore
_CACHE_NS = ("mcp_cache",)


class LangGraphMemoryManager:
    """
    Drop-in async replacement for AsyncHybridCache.

    Uses two LangGraph primitives:
      - MemorySaver  : per-conversation checkpoint store (messages, tool calls,
                       agent state) accessed via thread_id in create_agent.
      - InMemoryStore: flat K/V namespace store for sifter / tool-response /
                       embedding caches.

    The public `cache` singleton and `checkpointer` singleton below should be
    imported wherever the old `from database.redis_cache import cache` was used.
    """

    def __init__(self):
        # ── LangGraph primitives ───────────────────────────────────────────────
        self.checkpointer = MemorySaver()
        self._store = InMemoryStore()

        # Manual TTL index: key → unix expiry timestamp
        self._expiry: dict[str, float] = {}

        # Background cleanup task handle (started on first connect())
        self._cleanup_task: Optional[asyncio.Task] = None

    # ── Startup / Shutdown (no-ops — kept for API parity with AsyncHybridCache) ─

    async def connect(self, url: Optional[str] = None) -> bool:
        logger.info(
            "[MEMORY] ✅ LangGraph InMemoryStore + MemorySaver active "
            "(no external connection needed — Redis removed)."
        )
        # Bug #20 fix: start a background task that proactively evicts expired
        # keys every 5 minutes.  Without this, keys with a TTL that are never
        # re-read stay in _store and _expiry forever, causing unbounded growth.
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                self._cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())
                logger.info("[MEMORY] Background TTL cleanup task started (interval: 5 min).")
            except RuntimeError:
                # No running event loop yet (e.g. called from a sync context at
                # import time).  The task will be started on the first await.
                pass
        return True

    async def _ttl_cleanup_loop(self) -> None:
        """Periodically evict all keys whose TTL has elapsed."""
        while True:
            try:
                await asyncio.sleep(300)   # every 5 minutes
                await self._evict_expired()
            except asyncio.CancelledError:
                logger.info("[MEMORY] TTL cleanup task cancelled.")
                break
            except Exception as e:
                logger.warning(f"[MEMORY] TTL cleanup error: {e}")

    async def _evict_expired(self) -> None:
        """Delete every key whose expiry timestamp is in the past."""
        now = time.time()
        expired = [k for k, exp in list(self._expiry.items()) if exp < now]
        for key in expired:
            await self.delete(key)
        if expired:
            logger.info(f"[MEMORY] TTL sweep evicted {len(expired)} expired key(s).")

    async def disconnect(self) -> None:
        """Cancel the background cleanup task and release resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Always True — in-process memory never goes down."""
        return True

    @property
    def redis_client(self):
        """
        Always None.
        Callers in vector_store.py guard on `cache.redis_client` before using
        raw Redis commands; returning None makes them skip gracefully.
        The vector store now has its own in-memory dict instead.
        """
        return None

    # ── Core K/V Operations ────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value, respecting TTL."""
        # Lazy TTL eviction
        exp = self._expiry.get(key)
        if exp and time.time() > exp:
            await self.delete(key)
            return None
        try:
            item = self._store.get(_CACHE_NS, key)
            return item.value["data"] if item else None
        except Exception as e:
            logger.warning(f"[MEMORY] GET error '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = TTL_DEFAULT) -> bool:
        """Store a value with an optional TTL (seconds)."""
        try:
            self._store.put(_CACHE_NS, key, {"data": value})
            if ttl:
                self._expiry[key] = time.time() + ttl
            else:
                self._expiry.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"[MEMORY] SET error '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Remove a key from the store and the expiry index."""
        try:
            self._store.delete(_CACHE_NS, key)
            self._expiry.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"[MEMORY] DELETE error '{key}': {e}")
            return False

    async def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern (e.g. 'mcp:tool_resp:*')."""
        try:
            items = self._store.search(_CACHE_NS)
            deleted = 0
            for item in items:
                if fnmatch.fnmatch(item.key, pattern):
                    await self.delete(item.key)
                    deleted += 1
            return deleted
        except Exception as e:
            logger.warning(f"[MEMORY] FLUSH error '{pattern}': {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Return True if the key exists and has not expired."""
        return (await self.get(key)) is not None

    async def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds; -1 = no TTL, -2 = key not found."""
        exp = self._expiry.get(key)
        if exp is None:
            item = self._store.get(_CACHE_NS, key)
            return -1 if item else -2
        remaining = int(exp - time.time())
        return remaining if remaining > 0 else -2

    # ── MCP-Specific Helpers — identical signatures to redis_cache.py ──────────

    async def get_tool_response(self, tool_name: str, args_hash: str) -> Optional[Any]:
        return await self.get(KEY_TOOL_RESPONSE(tool_name, args_hash))

    async def set_tool_response(self, tool_name: str, args_hash: str, data: Any) -> bool:
        return await self.set(KEY_TOOL_RESPONSE(tool_name, args_hash), data, TTL_TOOL_RESPONSE)

    async def get_history_summary(self, summary_hash: str) -> Optional[str]:
        return await self.get(KEY_HISTORY(summary_hash))

    async def set_history_summary(self, summary_hash: str, summary: str) -> bool:
        return await self.set(KEY_HISTORY(summary_hash), summary, TTL_HISTORY_SUMMARY)

    async def get_sifter(self, sifter_hash: str) -> Optional[dict]:
        return await self.get(KEY_SIFTER(sifter_hash))

    async def set_sifter(self, sifter_hash: str, decision: dict) -> bool:
        return await self.set(KEY_SIFTER(sifter_hash), decision, TTL_SIFTER)

    async def invalidate_tool_cache(self) -> int:
        return await self.flush_pattern("mcp:tool_resp:*")

    async def invalidate_all(self) -> int:
        return await self.flush_pattern("mcp:*")

    async def cache_stats(self) -> dict:
        """Return health stats for the in-memory store."""
        try:
            all_items = self._store.search(_CACHE_NS)
            now = time.time()
            live_keys = [
                i.key for i in all_items
                if not (self._expiry.get(i.key, now + 1) < now)
            ]
            return {
                "available": True,
                "backend": "LangGraph InMemoryStore + MemorySaver",
                "total_live_keys": len(live_keys),
            }
        except Exception as e:
            return {"available": True, "backend": "LangGraph", "error": str(e)}


# ── Public singletons ──────────────────────────────────────────────────────────
memory_manager = LangGraphMemoryManager()

# `cache`        — drop-in for `from database.redis_cache import cache`
cache = memory_manager

# `checkpointer` — used by react_agent.py in create_agent(checkpointer=...)
checkpointer = memory_manager.checkpointer