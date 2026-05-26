import os
import json
import logging
from typing import Dict, Any

from config import STATE_FILE, OAUTH_FLOWS_FILE

logger = logging.getLogger("mcp_backend")

def load_state() -> Dict[str, Any]:
    """Loads universal state dictionary from mcp_state.json."""
    state = {
        "mcp_servers": {}, 
        "loaded_docs": {}, 
        "registered_clients": {}, 
        "user_name": "Tilak", 
        "uploaded_files": []
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    return state

def save_state(state: Dict[str, Any]):
    """Persists universal state dictionary to mcp_state.json."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
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
