"""
In-Memory Vector Store & Azure OpenAI Embeddings RAG Integration.

Replaces the Redis-backed vector store with a pure in-memory dict.
All Redis client calls have been removed; the store lives in process memory
and is rebuilt on startup (same behaviour as when Redis was unavailable).

Public API is identical to the old redis_store.py so all imports still work.
"""

import logging
import numpy as np
import os
from typing import Any, Dict, List

from langchain_openai import AzureOpenAIEmbeddings
from database.langgraph_memory import cache, KEY_DOC_CHUNK, TTL_EMBEDDING

logger = logging.getLogger("mcp_backend")

# ── In-process vector store ────────────────────────────────────────────────────
# Keyed by KEY_DOC_CHUNK(doc_name, idx) → {"text", "url", "doc_name", "vector"}
_DOC_VECTORS: Dict[str, Dict[str, Any]] = {}


def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """Returns the Azure OpenAI Embeddings client."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "text-embedding-3-large",
    )


async def index_documents_in_redis(loaded_docs: Dict[str, Any]):
    """
    Chunks and embeds loaded documentation into the in-memory vector store.

    The function name is kept as `index_documents_in_redis` so the existing
    import in web_app.py and database/__init__.py continues to work unchanged.
    """
    global _DOC_VECTORS
    _DOC_VECTORS.clear()

    if not loaded_docs:
        logger.info("[VECTOR STORE] No documents to index.")
        return

    embeddings = get_embeddings_model()
    logger.info("[VECTOR STORE] Building in-memory document vector index...")

    for doc_name, doc in loaded_docs.items():
        content = doc.get("content", "")
        url = doc.get("url", "")

        # Character-level chunking (1 000 chars / 200 overlap)
        chunks: List[str] = []
        chunk_size, overlap = 1000, 200
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(content[start:end])
            if end == len(content):
                break
            start += chunk_size - overlap

        logger.info(f"[VECTOR STORE] '{doc_name}' → {len(chunks)} chunks.")

        try:
            chunk_embeddings = await embeddings.aembed_documents(chunks)
            for idx, (chunk, vec) in enumerate(zip(chunks, chunk_embeddings)):
                key = KEY_DOC_CHUNK(doc_name, idx)
                _DOC_VECTORS[key] = {
                    "text": chunk,
                    "url": url,
                    "doc_name": doc_name,
                    "vector": vec,
                }
            logger.info(f"[VECTOR STORE] Indexed {len(chunks)} vectors for '{doc_name}'.")
        except Exception as e:
            logger.error(f"[VECTOR STORE ERROR] Embedding failed for '{doc_name}': {e}")


async def vector_search_redis(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Cosine-similarity search over the in-memory vector store.

    Function name kept as `vector_search_redis` for import compatibility.
    """
    if not _DOC_VECTORS:
        return []

    try:
        embeddings = get_embeddings_model()
        query_vec = await embeddings.aembed_query(query)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []

        candidates = []
        for entry in _DOC_VECTORS.values():
            vec = entry["vector"]
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue
            similarity = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            candidates.append({
                "text":     entry["text"],
                "url":      entry["url"],
                "doc_name": entry["doc_name"],
                "score":    similarity,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    except Exception as e:
        logger.error(f"[VECTOR STORE ERROR] Vector search failed: {e}")
        return []


async def retrieve_relevant_tools(query: str, tools: List[Any], top_k: int = 5) -> List[Any]:
    """
    Semantically filters a tool list down to the top_k most relevant tools
    using Azure OpenAI Embeddings.  Embedding vectors are cached in the
    LangGraph InMemoryStore (TTL = 24 hours) to avoid repeated API calls.
    """
    if not tools:
        return []

    # Only semantic-filter when there are many tools (avoids rate-limit on small sets)
    if len(tools) <= 20:
        return tools

    try:
        import hashlib
        embeddings = get_embeddings_model()
        query_vec  = await embeddings.aembed_query(query)

        tool_data    = []
        missed_idx   = []
        missed_texts = []

        for idx, tool in enumerate(tools):
            name      = getattr(tool, "name", "")
            desc      = getattr(tool, "description", "") or ""
            rep_text  = f"Tool: {name}\nDescription: {desc}"
            rep_hash  = hashlib.sha256(rep_text.encode()).hexdigest()[:32]
            cache_key = f"mcp:tool_emb:{name}:{rep_hash}"

            cached_vec = await cache.get(cache_key)
            if cached_vec:
                tool_data.append({"tool": tool, "vector": cached_vec})
            else:
                tool_data.append({"tool": tool, "vector": None,
                                  "cache_key": cache_key, "text": rep_text})
                missed_idx.append(idx)
                missed_texts.append(rep_text)

        if missed_texts:
            logger.info(f"[TOOL RAG] Generating embeddings for {len(missed_texts)} tools...")
            generated = await embeddings.aembed_documents(missed_texts)
            for idx, vec in zip(missed_idx, generated):
                item = tool_data[idx]
                item["vector"] = vec
                await cache.set(item["cache_key"], vec, ttl=TTL_EMBEDDING)

        q_norm = np.linalg.norm(query_vec)
        scored = []
        for item in tool_data:
            vec = item["vector"]
            if not vec:
                continue
            v_norm = np.linalg.norm(vec)
            if v_norm == 0 or q_norm == 0:
                continue
            sim = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            scored.append((sim, item["tool"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scored[:top_k]]
        logger.info(f"[TOOL RAG] Selected {len(selected)} / {len(tools)} tools semantically.")
        return selected

    except Exception as e:
        logger.error(f"[TOOL RAG ERROR] Semantic filtering failed: {e}. Returning all tools.")
        return tools
