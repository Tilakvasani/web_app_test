import os
import json
import time
import logging
from typing import Dict, Any

from config import STATE_FILE, OAUTH_FLOWS_FILE

logger = logging.getLogger("mcp_backend")

# FIX: Cache state in memory and only re-read from disk when the file changes.
# Previously load_state() opened + parsed the JSON on every single request,
# including high-frequency GET /api/state polls from the Streamlit frontend.
_state_cache: Dict[str, Any] = {}
_state_mtime: float = 0.0


def load_state() -> Dict[str, Any]:
    """
    Loads universal state dictionary from mcp_state.json.
    Uses an mtime check so disk reads only happen when the file actually changed.
    """
    global _state_cache, _state_mtime

    defaults = {
        "mcp_servers": {},
        "loaded_docs": {},
        "registered_clients": {},
        "user_name": "Tilak",
        "uploaded_files": []
    }

    if not os.path.exists(STATE_FILE):
        return dict(defaults)

    try:
        current_mtime = os.path.getmtime(STATE_FILE)
        if current_mtime == _state_mtime and _state_cache:
            return dict(_state_cache)  # return a shallow copy to avoid mutation

        with open(STATE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        _state_cache = {**defaults, **loaded}
        _state_mtime = current_mtime
        return dict(_state_cache)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return dict(_state_cache) if _state_cache else dict(defaults)


def save_state(state: Dict[str, Any]):
    """
    Persists universal state dictionary to mcp_state.json.
    Also refreshes the in-memory cache and mtime so the next load_state()
    call does not do an unnecessary disk re-read.
    """
    global _state_cache, _state_mtime

    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        _state_mtime = os.path.getmtime(STATE_FILE)
        _state_cache = dict(state)
        logger.info(f"Saved state to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_oauth_flows() -> Dict[str, Any]:
    """Loads temporary OAuth flow authorization states from oauth_flows.json."""
    if os.path.exists(OAUTH_FLOWS_FILE):
        try:
            with open(OAUTH_FLOWS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load oauth flows: {e}")
    return {}


def save_oauth_flows(flows: Dict[str, Any]):
    """Persists temporary OAuth flow authorization states to oauth_flows.json."""
    try:
        with open(OAUTH_FLOWS_FILE, "w", encoding="utf-8") as f:
            json.dump(flows, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save oauth flows: {e}")
