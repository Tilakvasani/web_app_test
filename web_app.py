"""
Main FastAPI Server Application Entrypoint.

Starts the universal MCP backend engine, registers modular routers,
configures CORS middleware policies, and initializes logging.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import state_router, oauth_router, chat_router
from database import cache

# Setup logger configuration to format app.log cleanly
logger = logging.getLogger("mcp_backend")
logger.info("Initializing Modular FastAPI Server...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan handler (replaces deprecated on_event).
    Manages Redis connection and vector store bootstrapping.
    """
    # ── Startup ──
    logger.info("[STARTUP] Bootstrapping FastAPI Server...")
    
    # Connect to Redis (falls back to in-memory if unavailable)
    await cache.connect()
    
    # Index documents in vector store if Redis is active
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
    
    logger.info("[STARTUP] Server ready.")
    yield
    
    # ── Shutdown ──
    logger.info("[SHUTDOWN] Gracefully shutting down...")
    await cache.disconnect()
    logger.info("[SHUTDOWN] Complete.")


app = FastAPI(
    title="⬡ Universal MCP Backend Server", 
    description="Refactored Modular API Engine for Model Context Protocol and dynamic sessions.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local Streamlit calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Modular Routers
app.include_router(state_router)
app.include_router(oauth_router)
app.include_router(chat_router)

logger.info("FastAPI Backend routes successfully loaded.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)