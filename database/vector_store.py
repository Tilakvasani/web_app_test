"""
Redis Vector Store & Azure OpenAI Embeddings RAG Integration.

Implements semantic document chunking, indexing, and vector similarity search
using the Azure OpenAI 'text-embedding-3-large' model.
All Redis operations are async to match the AsyncHybridCache pattern.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List
from langchain_openai import AzureOpenAIEmbeddings
from database.redis_cache import cache

logger = logging.getLogger("mcp_backend")


def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """
    Initializes and returns the Azure OpenAI Embeddings client
    configured to use the 'text-embedding-3-large' deployment.
    """
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "text-embedding-3-large",
    )


async def index_documents_in_redis(loaded_docs: Dict[str, Any]):
    """
    Chunks and embeds loaded documentation, storing vectors in Redis.
    If Redis is offline, skips index creation gracefully.
    """
    if not cache.is_available or not cache.redis_client:
        logger.info("[VECTOR STORE] Redis is offline. Skipping vector indexing.")
        return

    client = cache.redis_client
    embeddings = get_embeddings_model()

    logger.info("[VECTOR STORE] Beginning document vector indexing...")

    # Clear existing vectors to rebuild
    try:
        cursor = 0
        while True:
            # BUG FIX: use correct "mcp:doc:chunk:*" prefix to match KEY_DOC_CHUNK constant
            cursor, keys = await client.scan(cursor, match="mcp:doc:chunk:*", count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
    except Exception as e:
        logger.error(f"[VECTOR STORE ERROR] Failed to clear keys: {e}")

    for doc_name, doc in loaded_docs.items():
        content = doc.get("content", "")
        url = doc.get("url", "")
        
        # Simple character chunking (1000 chars per chunk with 200 char overlap)
        chunks = []
        chunk_size = 1000
        overlap = 200
        
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(content[start:end])
            if end == len(content):
                break
            start += chunk_size - overlap

        logger.info(f"[VECTOR STORE] Chunked '{doc_name}' into {len(chunks)} fragments.")

        try:
            # Generate embeddings for all chunks in a single batched API call
            chunk_embeddings = await embeddings.aembed_documents(chunks)
            
            # Store chunks and vectors in Redis using pipeline
            pipeline = client.pipeline()
            for idx, (chunk, vec) in enumerate(zip(chunks, chunk_embeddings)):
                # BUG FIX: use "mcp:doc:chunk:..." prefix to match KEY_DOC_CHUNK constant
                key = f"mcp:doc:chunk:{doc_name}:{idx}"
                pipeline.hset(key, mapping={
                    "text": chunk,
                    "url": url,
                    "doc_name": doc_name,
                    "vector": json.dumps(vec)
                })
            await pipeline.execute()
            logger.info(f"[VECTOR STORE] Successfully indexed {len(chunks)} vectors for '{doc_name}'.")
        except Exception as embed_err:
            logger.error(f"[VECTOR STORE ERROR] Embedding generation failed for '{doc_name}': {embed_err}")


async def vector_search_redis(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Generates query embedding and performs a vector similarity search over Redis documents.
    Returns the top-K matching document chunks.
    """
    if not cache.is_available or not cache.redis_client:
        return []

    client = cache.redis_client
    
    try:
        embeddings = get_embeddings_model()
        # Generate query embedding (async)
        query_vec = await embeddings.aembed_query(query)
        
        # Fetch all chunk keys using async SCAN
        keys = []
        cursor = 0
        while True:
            # BUG FIX: use correct "mcp:doc:chunk:*" prefix to match KEY_DOC_CHUNK constant
            cursor, batch = await client.scan(cursor, match="mcp:doc:chunk:*", count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        if not keys:
            return []

        # BUG FIX: use a pipeline to batch all hgetall calls — avoids N+1 Redis round-trips
        pipeline = client.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        all_data = await pipeline.execute()

        candidates = []
        query_norm = np.linalg.norm(query_vec)
        for data in all_data:
            if not data or "vector" not in data:
                continue

            vec = json.loads(data["vector"])
            vec_norm = np.linalg.norm(vec)
            # BUG FIX: guard against division by zero when either vector has zero norm
            if query_norm == 0 or vec_norm == 0:
                continue
            similarity = np.dot(query_vec, vec) / (query_norm * vec_norm)
            candidates.append({
                "text": data.get("text", ""),
                "url": data.get("url", ""),
                "doc_name": data.get("doc_name", ""),
                "score": float(similarity)
            })
            
        # Sort candidates by similarity score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
    except Exception as search_err:
        logger.error(f"[VECTOR STORE ERROR] Vector search failed: {search_err}")
        return []


async def retrieve_relevant_tools(query: str, tools: List[Any], top_k: int = 5) -> List[Any]:
    """
    Semantically filters list of LangChain tools down to the top_k most relevant ones
    using Azure OpenAI Embeddings and cached embedding vectors.
    """
    if not tools:
        return []
        
    # To prevent rate-limits (HTTP 429) and network overhead on small/medium toolsets, 
    # only perform semantic filtering if the total tool count is larger than 15.
    if len(tools) <= 20:
        return tools

    try:
        import hashlib
        embeddings = get_embeddings_model()
        
        # 1. Generate query embedding (async)
        query_vec = await embeddings.aembed_query(query)
        
        # 2. Compile text representations and calculate keys
        tool_data = []
        missed_indices = []
        missed_texts = []
        
        for idx, tool in enumerate(tools):
            name = getattr(tool, "name", "")
            desc = getattr(tool, "description", "") or ""
            rep_text = f"Tool: {name}\nDescription: {desc}"
            rep_hash = hashlib.sha256(rep_text.encode("utf-8")).hexdigest()[:32]
            cache_key = f"mcp:tool_emb:{name}:{rep_hash}"
            
            # Check cache
            cached_vec = await cache.get(cache_key)
            if cached_vec:
                tool_data.append({
                    "tool": tool,
                    "vector": cached_vec
                })
            else:
                tool_data.append({
                    "tool": tool,
                    "vector": None,
                    "cache_key": cache_key,
                    "text": rep_text
                })
                missed_indices.append(idx)
                missed_texts.append(rep_text)
                
        # 3. Handle cache misses (batch embed them asynchronously)
        if missed_texts:
            logger.info(f"[TOOL RAG] Cache miss for {len(missed_texts)} tools. Generating embeddings...")
            generated_vectors = await embeddings.aembed_documents(missed_texts)
            for idx, vec in zip(missed_indices, generated_vectors):
                item = tool_data[idx]
                item["vector"] = vec
                # Cache the generated vector for future calls (expire in 24 hours)
                await cache.set(item["cache_key"], vec, ttl=86400)
                
        # 4. Compute similarity and sort
        scored_tools = []
        for item in tool_data:
            vec = item["vector"]
            if not vec:
                continue
            # Cosine similarity
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            scored_tools.append((similarity, item["tool"]))
            
        # Sort descending by score
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        
        selected_tools = [tool for score, tool in scored_tools[:top_k]]
        logger.info(f"[TOOL RAG] Semantically selected {len(selected_tools)} tools from {len(tools)} loaded options.")
        return selected_tools
        
    except Exception as err:
        logger.error(f"[TOOL RAG ERROR] Semantic tool retrieval failed: {err}. Gracefully falling back to all tools.")
        return tools