"""
redis_cache.py — REPLACED BY LangGraph built-in memory.

All symbols that were previously defined here are now re-exported from
database.langgraph_memory so every existing import continues to work
without any changes in other modules.

    from database.redis_cache import cache              ✅ still works
    from database.redis_cache import KEY_SIFTER, TTL_SIFTER  ✅ still works
    from database import cache                          ✅ still works
"""

from database.langgraph_memory import (
    # Singleton (was AsyncHybridCache, now LangGraphMemoryManager)
    cache,
    checkpointer,
    memory_manager,

    # TTL constants
    TTL_TOOL_RESPONSE,
    TTL_HISTORY_SUMMARY,
    TTL_SIFTER,
    TTL_EMBEDDING,
    TTL_DEFAULT,

    # Key builders
    KEY_TOOL_RESPONSE,
    KEY_HISTORY,
    KEY_SIFTER,
    KEY_DOC_CHUNK,
)

__all__ = [
    "cache",
    "checkpointer",
    "memory_manager",
    "TTL_TOOL_RESPONSE",
    "TTL_HISTORY_SUMMARY",
    "TTL_SIFTER",
    "TTL_EMBEDDING",
    "TTL_DEFAULT",
    "KEY_TOOL_RESPONSE",
    "KEY_HISTORY",
    "KEY_SIFTER",
    "KEY_DOC_CHUNK",
]
