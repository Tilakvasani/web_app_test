"""
OAuth2 & Client Connection Routes Module.

Implements auto-discovery of endpoints via RFC 8414/9470, dynamic client registration
(RFC 7591), standard manual OAuth2 setup, server resource connections, and proactive
in-memory credential token refreshing.
"""

import os
import secrets
import hashlib
import base64
import time
import logging
import httpx
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode, quote, unquote
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from config import _make_cfg
from database import load_state, save_state, load_oauth_flows, save_oauth_flows

logger = logging.getLogger("mcp_backend")
router = APIRouter()

# ── Dynamic Pydantic Schema ──
class ConnectRequest(BaseModel):
    url: str
    name: Optional[str] = ""
    auth_option: str
    bearer_val: Optional[str] = ""
    api_header: Optional[str] = "X-API-Key"
    api_val: Optional[str] = ""
    oauth_client_id: Optional[str] = ""
    oauth_client_secret: Optional[str] = ""
    oauth_auth_endpoint: Optional[str] = ""
    oauth_token_endpoint: Optional[str] = ""
    oauth_scope: Optional[str] = ""

# ── Known OAuth Servers Registry ─────────────────────────────────────────────
GOOGLE_OAUTH_BASE = {
    "mode": "Manual OAuth2",
    "auth_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
}

KNOWN_OAUTH_SERVERS = {
    "gmailmcp.googleapis.com": {
        **GOOGLE_OAUTH_BASE,
        "scope": "https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.readonly"
    },
    "drivemcp.googleapis.com": {
        **GOOGLE_OAUTH_BASE,
        "scope": "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file"
    },
    "calendarmcp.googleapis.com": {
        **GOOGLE_OAUTH_BASE,
        "scope": "https://www.googleapis.com/auth/calendar"
    },
    "chatmcp.googleapis.com": {
        **GOOGLE_OAUTH_BASE,
        "scope": "https://www.googleapis.com/auth/chat.spaces.readonly"
    },
    "googleapis.com": {
        **GOOGLE_OAUTH_BASE,
        "scope": "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/chat.spaces.readonly"
    },

    "microsoft.com": {
        "mode": "Manual OAuth2",
        "auth_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "scope": "User.Read User.ReadBasic.All People.Read email profile openid offline_access"
    },
    "microsoftonline.com": {
        "mode": "Manual OAuth2",
        "auth_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "scope": "User.Read User.ReadBasic.All People.Read email profile openid offline_access"
    },
    "office.com": {
        "mode": "Manual OAuth2",
        "auth_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "scope": "User.Read User.ReadBasic.All People.Read email profile openid offline_access"
    },
    "cal.com": {
        "mode": "Auto-Discover OAuth (RFC 9470)"
    },
    "notion.com": {
        "mode": "Auto-Discover OAuth (RFC 9470)"
    },
    "zoho.com": {
        "mode": "Auto-Discover OAuth (RFC 9470)"
    },
    "zohomcp.in": {
        "mode": "Auto-Discover OAuth (RFC 9470)"
    },
    "indeed.com": {
        "mode": "Manual OAuth2",
        "auth_endpoint": "https://secure.indeed.com/oauth/v2/authorize",
        "token_endpoint": "https://apis.indeed.com/oauth/v2/tokens",
        "scope": "employer_access"
    },
    "apis.indeed.com": {
        "mode": "Manual OAuth2",
        "auth_endpoint": "https://secure.indeed.com/oauth/v2/authorize",
        "token_endpoint": "https://apis.indeed.com/oauth/v2/tokens",
        "scope": "employer_access"
    }
}

# Helper utilities

async def _try_connect_mcp(server_name: str, url: str, auth_type: str, auth_value: str, extra_headers: Optional[Dict[str, str]] = None):
    """Try connecting to an MCP server with transport fallback: streamable_http -> sse."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    for transport in ["streamable_http", "sse"]:
        try:
            cfg = _make_cfg(url, auth_type, auth_value, transport=transport)
            if extra_headers:
                cfg["headers"] = {**cfg.get("headers", {}), **extra_headers}
            client = MultiServerMCPClient({server_name: cfg})
            tools = await client.get_tools()
            logger.info(f"[MCP CONNECT] Connected to '{server_name}' via {transport} transport ({len(tools)} tools).")
            return tools, transport
        except Exception as e:
            logger.info(f"[MCP CONNECT] {transport} failed for '{server_name}': {e}")
            continue
    raise RuntimeError(f"Could not connect to MCP server '{server_name}' via any transport (tried streamable_http, sse).")

def _get_unique_server_name(base_name: str, current_url: str, state: Dict[str, Any]) -> str:
    """
    Generates a unique identifier for a newly registered MCP server.

    If a server with the same URL already exists, returns its active name.
    Otherwise, handles name collisions by appending an incremental numeric suffix.

    Args:
        base_name (str): Requested user-defined or domain-derived name for the server.
        current_url (str): The active MCP connection endpoint URL.
        state (Dict[str, Any]): Universal state directory container.

    Returns:
        str: A unique server name identifier that does not collide in state.
    """
    for name, s in state["mcp_servers"].items():
        if s.get("url") == current_url:
            return name
    if base_name not in state["mcp_servers"]:
        return base_name
    counter = 2
    while f"{base_name} ({counter})" in state["mcp_servers"]:
        counter += 1
    return f"{base_name} ({counter})"

async def _discover_mcp_endpoint(url: str) -> Optional[Dict[str, str]]:
    """
    Attempts to discover an active MCP JSON spec descriptor at the standard well-known location.

    Probes '{domain}/.well-known/mcp.json' to dynamically resolve the remote server's
    name and final connection endpoint.

    Args:
        url (str): Base URL of the target server.

    Returns:
        Optional[Dict[str, str]]: Discovered endpoint meta information (keys: 'name', 'endpoint') or None.
    """
    try:
        parsed = httpx.URL(url)
        origin = f"{parsed.scheme}://{parsed.host}"
        if parsed.port:
            origin += f":{parsed.port}"
        
        discovery_url = f"{origin}/.well-known/mcp.json"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(discovery_url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if "endpoint" in data:
                    return {
                        "name": data.get("name", parsed.host),
                        "endpoint": data["endpoint"]
                    }
    except Exception as e:
        logger.error(f"mcp.json discovery failed for {url}: {e}")
    return None

async def _discover_oauth(mcp_url: str) -> Optional[Dict[str, Any]]:
    """
    Dynamically discovers OAuth2 server metadata using RFC 8414/9470 protocols.

    Queries the target's '.well-known/oauth-protected-resource' to locate the active 
    authorization server, then pulls the auth/token endpoints from the server's metadata.

    Args:
        mcp_url (str): Active connection URL of the protected MCP resource.

    Returns:
        Optional[Dict[str, Any]]: Discovered OAuth authorization server metadata endpoints or None.
    """
    try:
        parsed = httpx.URL(mcp_url)
        origin = f"{parsed.scheme}://{parsed.host}"
        if parsed.port:
            origin += f":{parsed.port}"
            
        resource_url = f"{origin}/.well-known/oauth-protected-resource"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(resource_url, timeout=5.0)
            if resp.status_code != 200:
                return None
            
            resource_data = resp.json()
            auth_servers = resource_data.get("authorization_servers")
            if not auth_servers or not isinstance(auth_servers, list):
                return None
            
            auth_server = auth_servers[0].rstrip("/")
            metadata_url = f"{auth_server}/.well-known/oauth-authorization-server"
            
            resp_meta = await client.get(metadata_url, timeout=5.0)
            if resp_meta.status_code != 200:
                return None
                
            meta_data = resp_meta.json()
            if "authorization_endpoint" in meta_data and "token_endpoint" in meta_data:
                return meta_data
    except Exception as e:
        logger.error(f"OAuth discovery failed for {mcp_url}: {e}")
    return None

async def _discover_llms_txt(url: str) -> Optional[Dict[str, str]]:
    """
    Probes and extracts documentation details or llms.txt specifications from a given URL.

    Supports reading raw .txt/.md files directly, auto-discovering `/llms.txt` at the host root,
    or fallback-scraping an HTML page and cleaning its markup into structured plain text.

    Args:
        url (str): The resource URL containing technical manuals or API specs.

    Returns:
        Optional[Dict[str, str]]: Extracted documentation information (keys: 'name', 'url', 'content', 'type') or None.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            if "llms.txt" in url or url.endswith(".txt") or url.endswith(".md"):
                resp = await client.get(url, timeout=10.0)
                if resp.status_code == 200:
                    return {
                        "name": url.split("/")[-1] or "Documentation",
                        "url": url,
                        "content": resp.text,
                        "type": "txt"
                    }
            
            parsed = httpx.URL(url)
            origin = f"{parsed.scheme}://{parsed.host}"
            if parsed.port:
                origin += f":{parsed.port}"
            
            llms_url = f"{origin}/llms.txt"
            resp = await client.get(llms_url, timeout=5.0)
            if resp.status_code == 200:
                if not resp.text.strip().startswith("<!DOCTYPE") and not "<html" in resp.text.lower():
                    return {
                        "name": f"{parsed.host} llms.txt Index",
                        "url": llms_url,
                        "content": resp.text,
                        "type": "llms"
                    }
 
            resp = await client.get(url, timeout=10.0)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "html" in content_type:
                    title_match = re.search(r"<title>(.*?)</title>", resp.text, re.IGNORECASE)
                    title = title_match.group(1).strip() if title_match else f"{parsed.host} Docs"
                    
                    text = re.sub(r"<(script|style).*?>.*?</\1>", "", resp.text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<h[1-6].*?>(.*?)</h[1-6]>", r"\n\n# \1\n", text, flags=re.IGNORECASE)
                    text = re.sub(r"<p.*?>(.*?)</p>", r"\n\1\n", text, flags=re.IGNORECASE)
                    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
                    text = re.sub(r"<.*?>", "", text)
                    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
                    
                    return {
                        "name": title,
                        "url": url,
                        "content": text,
                        "type": "html"
                    }
                else:
                    return {
                        "name": url.split("/")[-1] or f"{parsed.host} Doc",
                        "url": url,
                        "content": resp.text,
                        "type": "txt"
                    }
    except Exception as e:
        logger.error(f"llms.txt/doc discovery failed for {url}: {e}")
    return None

async def _register_client(metadata: Dict[str, Any], redirect_uri: str, client_name: str) -> Dict[str, Any]:
    """
    Executes dynamic client registration (RFC 7591) with the discovered OAuth2 server.
    """
    reg_endpoint = metadata.get("registration_endpoint")
    if not reg_endpoint:
        raise ValueError("Server does not support dynamic client registration")
        
    registration_request = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            reg_endpoint,
            json=registration_request,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=10.0
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Client registration failed ({resp.status_code}): {resp.text}")
        return resp.json()


def _build_pkce_state(
    name: str,
    mcp_url: str,
    oauth_meta: Dict[str, Any],
    client_id: str,
    client_secret: Optional[str],
    redirect_uri: str,
) -> tuple:
    """
    Generate a PKCE code_verifier + code_challenge pair, store the OAuth flow
    state, and return (state_str, code_challenge, code_verifier).

    Extracted from the auto-discover and manual OAuth init paths to eliminate
    the duplicated 15-line PKCE block that previously existed in both.
    """
    code_verifier = secrets.token_urlsafe(64)
    hashed = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(hashed).decode("utf-8").replace("=", "")
    state_str = secrets.token_urlsafe(16)

    oauth_flows = load_oauth_flows()
    oauth_flows[state_str] = {
        "name": name,
        "url": mcp_url,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "endpoints": oauth_meta,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    save_oauth_flows(oauth_flows)
    return state_str, code_challenge, code_verifier


async def _exchange_code_for_token(
    token_endpoint: str,
    code: str,
    redirect_uri: str,
    client_id: str,
    code_verifier: str,
    client_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform the Authorization Code → Token exchange (RFC 6749 §4.1.3).

    Extracted from oauth_callback so the identical POST logic is not
    copy-pasted across auto-discover and manual OAuth flows.
    """
    data: Dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            token_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")
        return resp.json()

# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/api/mcp/connect")
async def connect_resource(body: ConnectRequest):
    url = body.url.strip()
    c_name = body.name.strip() if body.name else ""
    auth_option = body.auth_option
    
    # Normalizations
    if "{tenantid}" in url.lower():
        tenant_id = os.getenv("MICROSOFT_TENANT_ID", "common").strip() or "common"
        url = re.sub(r"\{tenantId\}", tenant_id, url, flags=re.IGNORECASE)
    if "cloud.microsoftcommon" in url.lower():
        url = url.replace("cloud.microsoftcommon", "cloud.microsoft.com/common")
    elif "cloud.microsoft/" in url.lower() and not "cloud.microsoft.com/" in url.lower():
        url = url.replace("cloud.microsoft/", "cloud.microsoft.com/")
        
    body.url = url
    lower_url = url.lower()
    
    # Auto-Upgrade check
    for domain, spec in KNOWN_OAUTH_SERVERS.items():
        if domain in lower_url:
            target_mode = spec["mode"]
            if auth_option != target_mode:
                logger.info(f"[AUTH UPGRADE] Known OAuth domain '{domain}' detected. Upgrading '{auth_option}' -> '{target_mode}'")
                auth_option = target_mode
                if target_mode == "Manual OAuth2":
                    body.oauth_auth_endpoint = spec["auth_endpoint"]
                    body.oauth_token_endpoint = spec["token_endpoint"]
                    body.oauth_scope = spec["scope"]
                    if "google" in lower_url:
                        body.oauth_client_id = body.oauth_client_id or os.getenv("GOOGLE_CLIENT_ID", "")
                        body.oauth_client_secret = body.oauth_client_secret or os.getenv("GOOGLE_CLIENT_SECRET", "")
                    elif "hubspot" in lower_url or "hubapi" in lower_url:
                        body.oauth_client_id = body.oauth_client_id or os.getenv("HUBSPOT_CLIENT_ID", "")
                        body.oauth_client_secret = body.oauth_client_secret or os.getenv("HUBSPOT_CLIENT_SECRET", "")
                    elif "microsoft" in lower_url or "office" in lower_url or "microsoftonline" in lower_url:
                        body.oauth_client_id = body.oauth_client_id or os.getenv("MICROSOFT_CLIENT_ID", "")
                        body.oauth_client_secret = body.oauth_client_secret or os.getenv("MICROSOFT_CLIENT_SECRET", "")
                    elif "indeed" in lower_url:
                        body.oauth_client_id = body.oauth_client_id or os.getenv("INDEED_CLIENT_ID", "")
                        body.oauth_client_secret = body.oauth_client_secret or os.getenv("INDEED_CLIENT_SECRET", "")
                    
    if auth_option == "Manual OAuth2" and (not body.oauth_client_id or not body.oauth_client_id.strip()):
        provider = (
            "Google" if "google" in lower_url else
            "HubSpot" if ("hubspot" in lower_url or "hubapi" in lower_url) else
            "Microsoft" if ("microsoft" in lower_url or "office" in lower_url or "microsoftonline" in lower_url) else
            "Indeed" if "indeed" in lower_url else
            "Manual OAuth2"
        )
        env_var_id = (
            "GOOGLE_CLIENT_ID" if provider == "Google" else
            "HUBSPOT_CLIENT_ID" if provider == "HubSpot" else
            "MICROSOFT_CLIENT_ID" if provider == "Microsoft" else
            "INDEED_CLIENT_ID" if provider == "Indeed" else
            "CLIENT_ID"
        )
        env_var_secret = (
            "GOOGLE_CLIENT_SECRET" if provider == "Google" else
            "HUBSPOT_CLIENT_SECRET" if provider == "HubSpot" else
            "MICROSOFT_CLIENT_SECRET" if provider == "Microsoft" else
            "INDEED_CLIENT_SECRET" if provider == "Indeed" else
            "CLIENT_SECRET"
        )
        raise HTTPException(
            status_code=400,
            detail=f"{provider} OAuth credentials are missing. Please define '{env_var_id}' and '{env_var_secret}' in your backend '.env' file to authorize {provider} API services."
        )

    state = load_state()
    logger.info(f"[UI CONNECT] Connecting URL: '{url}' (Auth Mode: '{auth_option}')")
    
    mcp_info = await _discover_mcp_endpoint(url)
    if not mcp_info:
        try:
            final_auth_type = "none"
            final_auth_val = ""
            if auth_option == "Bearer Token":
                final_auth_type = "bearer"
                final_auth_val = body.bearer_val
            elif auth_option == "API Key":
                final_auth_type = "api_key"
                final_auth_val = body.api_val
            
            extra_hdrs = None
            if final_auth_type == "api_key" and body.api_header:
                extra_hdrs = {body.api_header: body.api_val}
                
            tools, _ = await _try_connect_mcp("fallback_test", url, final_auth_type, final_auth_val, extra_headers=extra_hdrs)
            
            mcp_info = {
                "name": httpx.URL(url).host or "Direct MCP",
                "endpoint": url
            }
            logger.info(f"[DISCOVER SUCCESS] Verified direct MCP endpoint at {url} with {len(tools)} tools.")
        except Exception as direct_err:
            logger.info(f"Direct MCP endpoint probe failed: {direct_err}")
            
    if not mcp_info and (auth_option != "None" or any(keyword in url.lower() for keyword in ["oauth", "mcp", "message", "sse", "http"])):
        mcp_info = {
            "name": httpx.URL(url).host or "Remote MCP",
            "endpoint": url
        }
        
    if mcp_info:
        mcp_url = mcp_info["endpoint"]
        detected_name = c_name or mcp_info["name"]
        
        is_oauth_autodiscover = (auth_option == "Auto-Discover OAuth (RFC 9470)")
        is_oauth_implicit = (auth_option == "None" and await _discover_oauth(mcp_url) is not None)
        
        if is_oauth_autodiscover or is_oauth_implicit:
            oauth_meta = await _discover_oauth(mcp_url)
            if not oauth_meta:
                raise HTTPException(status_code=400, detail="Server does not support auto-discovered OAuth.")
                
            try:
                client_name = state.get("user_name", "Tilak") or f"{detected_name} Client"
                redirect_uri = "http://localhost:8000/api/mcp/oauth/callback"
                
                registered = state.get("registered_clients", {}).get(mcp_url)
                if not registered:
                    try:
                        registered = await _register_client(oauth_meta, redirect_uri, client_name)
                        state["registered_clients"][mcp_url] = registered
                        save_state(state)
                    except Exception as reg_err:
                        logger.info(f"[OAUTH] Dynamic registration failed: {reg_err}. Trying .env credentials...")
                        # Fallback: use pre-registered credentials from .env
                        env_client_id = None
                        env_client_secret = None
                        lower_mcp = mcp_url.lower()
                        if "hubspot" in lower_mcp or "hubapi" in lower_mcp:
                            env_client_id = os.getenv("HUBSPOT_CLIENT_ID", "").strip()
                            env_client_secret = os.getenv("HUBSPOT_CLIENT_SECRET", "").strip()
                        elif "google" in lower_mcp:
                            env_client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
                            env_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
                        elif "microsoft" in lower_mcp or "office" in lower_mcp:
                            env_client_id = os.getenv("MICROSOFT_CLIENT_ID", "").strip()
                            env_client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", "").strip()
                        
                        if not env_client_id:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Server does not support dynamic client registration. Please add the appropriate CLIENT_ID and CLIENT_SECRET to your .env file."
                            )
                        registered = {"client_id": env_client_id, "client_secret": env_client_secret}
                    
                client_id = registered["client_id"]
                client_secret = registered.get("client_secret")

                redirect_uri = "http://localhost:8000/api/mcp/oauth/callback"
                state_str, code_challenge, _ = _build_pkce_state(
                    detected_name, mcp_url, oauth_meta,
                    client_id, client_secret, redirect_uri
                )
                
                params = {
                    "response_type": "code",
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "state": state_str,
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256"
                }
                # Include discovered scopes if available
                discovered_scopes = oauth_meta.get("scopes_supported", [])
                if discovered_scopes:
                    params["scope"] = " ".join(discovered_scopes)
                # Google-specific params for refresh tokens
                if "google" in mcp_url.lower():
                    params["access_type"] = "offline"
                    params["prompt"] = "consent"
                auth_url = f"{oauth_meta['authorization_endpoint']}?{urlencode(params)}"
                return {"status": "pending", "type": "oauth", "auth_url": auth_url}
            except Exception as e:
                logger.error(f"[OAUTH INIT FAILED] Dynamic OAuth setup failed: {e}")
                raise HTTPException(status_code=500, detail=f"OAuth setup failed: {e}")
                
        elif auth_option == "Manual OAuth2":
            try:
                redirect_uri = "http://localhost:8000/api/mcp/oauth/callback"
                oauth_meta = {
                    "authorization_endpoint": body.oauth_auth_endpoint,
                    "token_endpoint": body.oauth_token_endpoint
                }

                state_str, code_challenge, _ = _build_pkce_state(
                    detected_name, mcp_url, oauth_meta,
                    body.oauth_client_id, body.oauth_client_secret, redirect_uri
                )

                params = {
                    "response_type": "code",
                    "client_id": body.oauth_client_id,
                    "redirect_uri": redirect_uri,
                    "state": state_str,
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                    "access_type": "offline",
                    "prompt": "consent"
                }
                if body.oauth_scope:
                    params["scope"] = body.oauth_scope
                    
                auth_url = f"{body.oauth_auth_endpoint}?{urlencode(params)}"
                return {"status": "pending", "type": "oauth", "auth_url": auth_url}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Manual OAuth setup failed: {e}")
                
        else:
            final_auth_type = "none"
            final_auth_val = ""
            if auth_option == "Bearer Token":
                final_auth_type = "bearer"
                final_auth_val = body.bearer_val
            elif auth_option == "API Key":
                final_auth_type = "api_key"
                final_auth_val = body.api_val
                
            resolved_name = _get_unique_server_name(detected_name, mcp_url, state)
            try:
                extra_hdrs = None
                if final_auth_type == "api_key" and body.api_header:
                    extra_hdrs = {body.api_header: body.api_val}
                    
                tools, working_transport = await _try_connect_mcp(resolved_name, mcp_url, final_auth_type, final_auth_val, extra_headers=extra_hdrs)
                tool_names = [t.name for t in tools]
                
                state["mcp_servers"][resolved_name] = {
                    "url": mcp_url,
                    "auth_type": final_auth_type,
                    "auth_value": final_auth_val,
                    "transport": working_transport,
                    "tools": tool_names
                }
                if final_auth_type == "api_key" and body.api_header:
                    state["mcp_servers"][resolved_name]["api_header"] = body.api_header
                    
                save_state(state)
                return {"status": "success", "type": "server", "name": resolved_name, "tools": tool_names}
            except Exception as add_err:
                raise HTTPException(status_code=400, detail=f"Failed to fetch tools: {add_err}")
                
    else:
        doc_info = await _discover_llms_txt(url)
        if doc_info:
            resolved_name = doc_info["name"]
            if resolved_name in state["loaded_docs"]:
                resolved_name = f"{resolved_name} ({int(time.time())})"
                
            state["loaded_docs"][resolved_name] = {
                "url": doc_info["url"],
                "content": doc_info["content"],
                "type": doc_info["type"]
            }
            save_state(state)
            
            # Asynchronously index documents in Redis
            try:
                from database import index_documents_in_redis
                await index_documents_in_redis(state["loaded_docs"])
            except Exception as index_err:
                logger.error(f"[VECTOR STORE ERROR] Failed to index documents after discovery: {index_err}")
                
            return {"status": "success", "type": "doc", "name": resolved_name}
            
    raise HTTPException(status_code=400, detail="Failed to auto-discover resource.")

@router.get("/api/mcp/oauth/callback")
async def oauth_callback(code: str, state: str):
    logger.info(f"[OAUTH CALLBACK] Received callback for state parameter: '{state}'")
    oauth_flows = load_oauth_flows()
    flow = oauth_flows.get(state)
    
    if not flow:
        return RedirectResponse(url="http://localhost:8501/?oauth_error=session_expired")
        
    try:
        endpoints = flow["endpoints"]
        token_endpoint = endpoints["token_endpoint"]

        # Use the shared helper — no more copy-pasted POST boilerplate here
        tokens = await _exchange_code_for_token(
            token_endpoint=token_endpoint,
            code=code,
            redirect_uri=flow["redirect_uri"],
            client_id=flow["client_id"],
            code_verifier=flow["code_verifier"],
            client_secret=flow.get("client_secret"),
        )

        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")
        expires_in = tokens.get("expires_in", 3600)

        server_url = flow["url"]
        state_dict = load_state()
        server_name = _get_unique_server_name(flow["name"], server_url, state_dict)

        # Fetch tools to verify connection works
        tools, working_transport = await _try_connect_mcp(server_name, server_url, "bearer", access_token)
        tool_names = [t.name for t in tools]

        state_dict["mcp_servers"][server_name] = {
            "url": server_url,
            "auth_type": "bearer",
            "auth_value": access_token,
            "refresh_token": refresh_token,
            "expires_at": time.time() + expires_in,
            "transport": working_transport,
            "oauth_endpoints": endpoints,
            "client_credentials": {
                "client_id": flow["client_id"],
                "client_secret": flow.get("client_secret")
            },
            "tools": tool_names
        }
        save_state(state_dict)

        oauth_flows.pop(state, None)
        save_oauth_flows(oauth_flows)

        logger.info(f"[OAUTH SUCCESS] Successfully connected server '{server_name}'.")
        return RedirectResponse(url=f"http://localhost:8501/?oauth_success=1&server_name={quote(server_name)}")
    except Exception as e:
        logger.error(f"[OAUTH CALLBACK ERROR] OAuth callback failed: {e}")
        return RedirectResponse(url=f"http://localhost:8501/?oauth_error={quote(str(e))}")

@router.delete("/api/mcp/server/{name}")
async def delete_server(name: str):
    state = load_state()
    if name in state.get("mcp_servers", {}):
        state["mcp_servers"].pop(name)
        save_state(state)
        logger.info(f"[SERVER REMOVE] Removed server '{name}'")
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Server not found")

@router.delete("/api/mcp/doc/{name}")
async def delete_doc(name: str):
    state = load_state()
    if name in state.get("loaded_docs", {}):
        state["loaded_docs"].pop(name)
        save_state(state)
        logger.info(f"[DOC REMOVE] Removed document '{name}'")
        
        # Asynchronously rebuild document indices in Redis
        try:
            from database import index_documents_in_redis
            await index_documents_in_redis(state["loaded_docs"])
        except Exception as index_err:
            logger.error(f"[VECTOR STORE ERROR] Failed to rebuild index after deletion: {index_err}")
            
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Documentation not found")

# Proactive Token Checking Utility Function
async def proactively_refresh_server_tokens():
    """Timer/cron utility called to keep dynamic tokens refreshed."""
    now = time.time()
    state = load_state()
    modified = False
    
    for name, s in list(state.get("mcp_servers", {}).items()):
        if s.get("auth_type") == "bearer" and "expires_at" in s and "refresh_token" in s:
            time_left = s["expires_at"] - now

            # FIX: Force-refresh Google tokens on first startup after the scope fix.
            # Old tokens were issued without full Drive scopes — a forced refresh
            # gets a new token with drive + drive.file + drive.readonly included.
            # _scope_refreshed flag prevents redundant refreshes on subsequent startups.
            server_url = s.get("url", "")
            is_google = any(d in server_url.lower() for d in KNOWN_OAUTH_SERVERS.keys())

            # If scopes changed since last login, reset the flag to force a fresh token.
            expected_scope = next((spec.get("scope", "") for dk, spec in KNOWN_OAUTH_SERVERS.items() if dk in server_url.lower()), "")
            if is_google and s.get("_last_scope", "") != expected_scope:
                s["_scope_refreshed"] = False

            force_refresh = is_google and not s.get("_scope_refreshed", False)

            if force_refresh or time_left < 300: # 5 min limit
                logger.info(f"[PROACTIVE REFRESH] Refreshing token for '{name}'...")
                try:
                    endpoints = s["oauth_endpoints"]
                    client_credentials = s.get("client_credentials", {})
                    
                    params = {
                        "grant_type": "refresh_token",
                        "refresh_token": s["refresh_token"],
                        "client_id": client_credentials.get("client_id"),
                    }
                    if client_credentials.get("client_secret"):
                        params["client_secret"] = client_credentials["client_secret"]

                    # FIX: Google requires scope in refresh requests for drivemcp.
                    # Without it, the new access token is issued with reduced/no Drive
                    # permissions and every tool call returns 403 Forbidden.
                    for domain_key, spec in KNOWN_OAUTH_SERVERS.items():
                        if domain_key in server_url.lower():
                            params["scope"] = spec.get("scope", "")
                            break
                        
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            endpoints["token_endpoint"],
                            data=params,
                            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                            timeout=10.0
                        )
                        if resp.status_code == 200:
                            tokens = resp.json()
                            s["auth_value"] = tokens["access_token"]
                            if "refresh_token" in tokens:
                                s["refresh_token"] = tokens["refresh_token"]
                            expires_in = tokens.get("expires_in", 3600)
                            s["expires_at"] = time.time() + expires_in
                            s["_scope_refreshed"] = True  # don't force-refresh again next startup
                            s["_last_scope"] = expected_scope  # track scope version
                            modified = True
                            logger.info(f"[PROACTIVE REFRESH SUCCESS] Refreshed token for '{name}'.")
                        else:
                            logger.error(f"[PROACTIVE REFRESH ERROR] Failed for '{name}': {resp.text}")
                except Exception as e:
                    logger.error(f"[PROACTIVE REFRESH ERROR] Exception for '{name}': {e}")
    if modified:
        save_state(state)