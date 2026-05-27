"""
Main FastAPI Server Application Entrypoint.

Starts the universal MCP backend engine, registers modular routers,
configures CORS middleware policies, and initializes logging.
"""

import asyncio
import logging
import logging.handlers
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import state_router, oauth_router, chat_router
from database import cache

import os
os.makedirs(".logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(".logs", "app.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("mcp_backend")
logger.info("Initializing Modular FastAPI Server...")


async def _token_refresh_loop():
    """
    FIX: Proactive token refresh runs in a background asyncio task every 55 minutes.
    Previously this ran inline at the top of every /api/chat request, adding
    400-700 ms latency when a refresh was due. Moving it here means zero chat
    request overhead — tokens are always fresh in the background.
    """
    # Import here to avoid circular import at module level
    from routes.oauth_routes import proactively_refresh_server_tokens

    # Initial delay so the server finishes startup before the first refresh attempt
    await asyncio.sleep(10)

    while True:
        try:
            await proactively_refresh_server_tokens()
        except Exception as ex:
            logger.error(f"[TOKEN REFRESH BACKGROUND] Error: {ex}")
        # Refresh every 55 minutes (tokens typically expire at 60 min)
        await asyncio.sleep(55 * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan handler (replaces deprecated on_event).
    Manages Redis connection, vector store bootstrapping, and background tasks.
    """
    # ── Startup ──
    logger.info("[STARTUP] Bootstrapping FastAPI Server...")

    await cache.connect()

    try:
        from database import load_state, index_documents_in_redis
        if cache.is_available:
            state = load_state()
            loaded_docs = state.get("loaded_docs", {})
            if loaded_docs:
                logger.info(f"[STARTUP] Redis active. Indexing {len(loaded_docs)} documents in vector store...")
                await index_documents_in_redis(loaded_docs)
            else:
                logger.info("[STARTUP] No documents to index in vector store.")
    except Exception as e:
        logger.error(f"[STARTUP ERROR] Vector store bootstrapping failed: {e}")

    # FIX: Start background token refresh task — removed from chat route handler
    refresh_task = asyncio.create_task(_token_refresh_loop())
    logger.info("[STARTUP] Background token refresh task started (every 55 min).")

    logger.info("[STARTUP] Server ready.")
    yield

    # ── Shutdown ──
    logger.info("[SHUTDOWN] Gracefully shutting down...")
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    await cache.disconnect()
    logger.info("[SHUTDOWN] Complete.")


app = FastAPI(
    title="⬡ Universal MCP Backend Server",
    description="Refactored Modular API Engine for Model Context Protocol and dynamic sessions.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(state_router)
app.include_router(oauth_router)
app.include_router(chat_router)

logger.info("FastAPI Backend routes successfully loaded.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=["*.log", "mcp_state.json", "oauth_flows.json", "uploaded_files/*", "uploaded_files"]
    )