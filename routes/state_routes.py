import os
import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from config import UPLOAD_DIR
from database import load_state, save_state

logger = logging.getLogger("mcp_backend")
router = APIRouter()

# Schemas
class UserNameUpdate(BaseModel):
    user_name: str

def unpack_exception(e: Exception) -> str:
    """Recursively unpack exceptions and construct user-friendly error messages."""
    if hasattr(e, "exceptions") and e.exceptions:
        unpacked_msgs = []
        for child in e.exceptions:
            unpacked_msgs.append(unpack_exception(child))
        return " | ".join(unpacked_msgs)
        
    err_msg = str(e)
    if "401 Unauthorized" in err_msg or "401" in err_msg or "Unauthorized" in err_msg:
        return (
            "Authentication Error (401 Unauthorized): The access token or API credentials for the server "
            "are invalid, revoked, or expired. Please reconnect the server in your sidebar with the correct authentication credentials!"
        )
    if "403 Forbidden" in err_msg or "403" in err_msg:
        return "Access Forbidden (403 Forbidden): Your credentials do not have the required permissions or scopes to execute this action."
    return err_msg

# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/api/state")
async def get_state():
    """Returns the state dictionary."""
    return load_state()

@router.post("/api/user_name")
async def update_user_name(body: UserNameUpdate):
    """Updates client name in the state."""
    state = load_state()
    state["user_name"] = body.user_name.strip()
    save_state(state)
    logger.info(f"[STATE UPDATE] Updated client identity to: {state['user_name']}")
    return {"status": "success", "user_name": state["user_name"]}

@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a local document to the secure directory and registers it in state."""
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        size = len(content)
        state = load_state()
        if "uploaded_files" not in state:
            state["uploaded_files"] = []
            
        # Clear existing duplicate entries
        existing = next((f for f in state["uploaded_files"] if f["name"] == safe_filename), None)
        file_meta = {
            "name": safe_filename,
            "path": os.path.abspath(file_path),
            "size": size,
            "uploaded_at": time.time()
        }
        if existing:
            state["uploaded_files"].remove(existing)
            
        state["uploaded_files"].append(file_meta)
        save_state(state)
        
        logger.info(f"[FILE UPLOAD] File '{safe_filename}' uploaded successfully. Size: {size} bytes.")
        return {"status": "success", "file": file_meta}
    except Exception as e:
        logger.error(f"[FILE UPLOAD ERROR] Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

@router.delete("/api/upload/{filename}")
async def delete_uploaded_file(filename: str):
    """Removes a registered file from the secure local workspace and state."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    state = load_state()
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"[FILE DELETE] Removed local file: {safe_filename}")
        except Exception as e:
            logger.error(f"[FILE DELETE ERROR] Failed to remove file '{safe_filename}': {e}")
            
    if "uploaded_files" in state:
        existing = next((f for f in state["uploaded_files"] if f["name"] == safe_filename), None)
        if existing:
            state["uploaded_files"].remove(existing)
            save_state(state)
            return {"status": "success"}
            
    raise HTTPException(status_code=404, detail="File not found")

@router.get("/api/logs")
async def get_logs(limit: int = 50):
    """Retrieves standard diagnostic logs from app.log."""
    if os.path.exists("app.log"):
        try:
            with open("app.log", "r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
                return {"status": "success", "logs": "".join(lines)}
        except Exception as e:
            return {"status": "error", "message": f"Error reading log file: {e}"}
    return {"status": "error", "message": "Log file app.log does not exist yet."}
