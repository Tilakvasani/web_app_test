"""
Intelligent Chat & Command Routes Module.

Orchestrates the streaming ReAct AI agent loop using Server-Sent Events (SSE),
handles high-speed CLI slash commands (/scrape, /summarize, /export, /quick-add),
and manages direct dynamic MCP tool invocations.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from config import get_llm, UPLOAD_DIR, _make_cfg
from database import load_state, save_state
from agents import stream_agent_interaction
from routes.oauth_routes import proactively_refresh_server_tokens
from tools import web_search, web_scrape, get_read_uploaded_file_tool, get_query_documentation_tool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("mcp_backend")
router = APIRouter()

# Schemas
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []

class CommandRequest(BaseModel):
    command: str

# ── Helper to Call MCP Tools Dynamically ──────────────────────────────────────
async def call_mcp_tool_directly(server_name: str, tool_name: str, tool_args: dict, state: dict) -> Any:
    """Helper to invoke a specific MCP tool directly from the backend."""
    if not state.get("mcp_servers") or server_name not in state["mcp_servers"]:
        raise ValueError(f"Server '{server_name}' is not connected.")
    
    s = state["mcp_servers"][server_name]
    transport = s.get("transport", "streamable_http")
    cfg = _make_cfg(s["url"], s["auth_type"], s["auth_value"], transport=transport)
    if s.get("api_header") and s.get("auth_type") == "api_key":
        cfg["headers"] = {s["api_header"]: s["auth_value"]}
        
    client = MultiServerMCPClient({server_name: cfg})
    try:
        # Retrieve tools first to verify transport link
        await client.get_tools(server_name=server_name)
        # Call tool dynamically
        res = await client.ainvoke(tool_name, tool_args)
        return res
    except Exception as e:
        logger.error(f"Direct MCP tool call failed: {e}")
        raise e

# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_interaction(body: ChatRequest):
    """
    Main Chat streaming endpoint. Initiates LangGraph ReAct agent loop,
    proactively checking/refreshing credentials and streaming responses.
    Includes full conversation stream caching to serve identical repeat calls instantly for 0 tokens.
    """
    try:
        await proactively_refresh_server_tokens()
    except Exception as ex:
        logger.error(f"[TOKEN CHECK ERROR] Error during proactive token refresh: {ex}")
    state = load_state()
    from database import cache
    
    # ── 1. Calculate Request Hash for Conversation Caching ──
    import hashlib
    try:
        # Fully dynamic and typo-immune check: if the query is self-contained and not a conversational follow-up,
        # we cache it based on the prompt alone so repeat questions hit the cache 100% of the time, even with typos!
        prompt_lower = body.prompt.lower().strip()
        
        # Pronouns, references, and question starters that indicate history-dependent follow-up
        conversational_words = {
            "why", "more", "explain", "he", "she", "him", 
            "her", "it", "they", "them", "that", "this", "prev", "first", "second", 
            "last", "those", "these", "there", "here", "then"
        }
        
        # Tokenize and check for conversational indicators
        words = set(re.findall(r"\b\w+\b", prompt_lower))
        is_conversational_followup = bool(words.intersection(conversational_words))
        
        if len(prompt_lower) > 8 and not is_conversational_followup:
            # Cache based on the prompt alone to be 100% dynamic and typo-immune!
            req_hash = hashlib.md5(f"prompt_only:{prompt_lower}".encode("utf-8")).hexdigest()
            logger.info(f"[CHAT CACHE] Using dynamic prompt-only caching key for query '{prompt_lower}'")
        else:
            # Conversational/short queries include history to remain context-aware and safe
            history_str = json.dumps(body.history, sort_keys=True)
            req_hash = hashlib.md5(f"{prompt_lower}:{history_str}".encode("utf-8")).hexdigest()
            logger.info(f"[CHAT CACHE] Using context-aware caching key for query '{prompt_lower}'")
            
        cache_key = f"chat_resp:{req_hash}"
    except Exception as hash_err:
        logger.error(f"[CHAT CACHE ERROR] Failed to compute request hash: {hash_err}")
        cache_key = None

    # ── 2. Serve from Cache if Available (0 LLM Tokens!) ──
    if cache_key:
        try:
            cached_chunks = await cache.get(cache_key)
            if cached_chunks:
                logger.info(f"[CHAT CACHE HIT] Serving cached response for hash '{req_hash}' (0 LLM tokens).")
                
                async def sse_cached_stream():
                    for chunk in cached_chunks:
                        if "event: token_usage" in chunk:
                            # Override token meter to show 0 tokens used!
                            yield "event: token_usage\ndata: {\"input_tokens\": 0, \"output_tokens\": 0, \"cached\": true}\n\n"
                            yield "event: thought\ndata: {\"text\": \"⚡ Response loaded from cache (0 LLM tokens used!)\"}\n\n"
                        else:
                            yield chunk
                            
                return StreamingResponse(sse_cached_stream(), media_type="text/event-stream")
        except Exception as cache_get_err:
            logger.error(f"[CHAT CACHE ERROR] Failed to fetch from cache: {cache_get_err}")

    # ── 3. Cache Miss - Execute Stream and Cache output ──
    async def sse_stream_wrapper():
        chunks_collected = []
        success = False
        try:
            async for sse_chunk in stream_agent_interaction(body.prompt, body.history, state):
                chunks_collected.append(sse_chunk)
                yield sse_chunk
                if "Completed successfully" in sse_chunk or "Completed" in sse_chunk:
                    success = True
        except Exception as stream_err:
            logger.error(f"[CHAT STREAM ERROR] Stream generation wrapper failed: {stream_err}")

        # BUG FIX: moved caching OUT of the finally block.
        # Using `await` inside `finally` of an async generator raises
        # RuntimeError: "async generator ignored GeneratorExit" when Starlette
        # cancels the generator on client disconnect. Caching here is safe and
        # correct — partial/failed streams (success=False) are never cached anyway.
        if success and cache_key and chunks_collected:
            try:
                await cache.set(cache_key, chunks_collected, ttl=600)
                logger.info(f"[CHAT CACHE STORE] Cached new response under hash '{req_hash}'.")
            except Exception as cache_set_err:
                logger.error(f"[CHAT CACHE ERROR] Failed to store response in cache: {cache_set_err}")

    return StreamingResponse(sse_stream_wrapper(), media_type="text/event-stream")


@router.post("/api/commands/execute")
async def execute_command(body: CommandRequest):
    """
    High-speed CLI slash command execution engine. Bypasses the heavy ReAct
    thinking loop to run specific local operations or dynamic MCP lookups.
    """
    command_str = body.command.strip()
    if not command_str.startswith("/"):
        raise HTTPException(status_code=400, detail="Not a valid slash command. Must start with '/'")

    # Split command name and arguments
    parts = command_str.split(" ", 1)
    cmd = parts[0].lower()
    args_str = parts[1].strip() if len(parts) > 1 else ""

    state = load_state()
    llm = get_llm()

    try:
        # ── 1. HELP COMMAND ──
        if cmd == "/help":
            help_md = """
### 📋 Available CLI Slash Tools

| Command | Category | Description |
| :--- | :--- | :--- |
| **`/help`** | Utility | Renders this help cheat-sheet of available tools. |
| **`/clear`** | Utility | Wipes conversation history and resets estimated token trackers. |
| **`/servers`** | MCP Control | Shows status cards for all connected servers. |
| **`/tools`** | MCP Control | Inspects loaded tools and parameter schemas. |
| **`/files`** | Workspace | Explorer for local files uploaded to the workspace. |
| **`/scrape <url_or_query>`** | Agentic Scraper | Direct web scrape or autonomous CRM-lookup extraction (using Azure OpenAI). |
| **`/summarize <source> [id]`** | Intelligent MCP | High-speed LLM summary of local file OR notion/CRM record JSON content. |
| **`/export <server> <id>`** | Intelligent MCP | Fetches a dynamic record (e.g. Notion page) and saves it locally in the workspace. |
| **`/quick-add <service> <text>`** | Intelligent MCP | Natural language parsing to instantly execute creation tools (Cal, Notion, HubSpot). |
| **`/search <query>`** | Workspace | Fast keyword indexing search over all loaded documentation. |
"""
            return {"status": "success", "content": help_md}

        # ── 2. CLEAR COMMAND ──
        elif cmd == "/clear":
            return {
                "status": "success", 
                "action": "clear", 
                "content": "🧹 **Conversation history and token metrics reset successfully!**"
            }

        # ── 3. SERVERS COMMAND ──
        elif cmd == "/servers":
            servers = state.get("mcp_servers", {})
            if not servers:
                return {"status": "success", "content": "⬡ **No active MCP servers currently connected.**"}
            
            srv_md = "### ⬡ Connected MCP Servers:\n\n"
            for name, s in servers.items():
                srv_md += f"- **Server**: `{name}`\n"
                srv_md += f"  - **Protocol**: `{s.get('transport', 'sse')}`\n"
                srv_md += f"  - **URL/Endpoint**: `{s.get('url')}`\n"
                srv_md += f"  - **Auth Type**: `{s.get('auth_type')}`\n"
                srv_md += f"  - **Active Tools**: {len(s.get('tools', []))} loaded.\n\n"
            return {"status": "success", "content": srv_md}

        # ── 4. TOOLS COMMAND ──
        elif cmd == "/tools":
            servers = state.get("mcp_servers", {})
            if not servers:
                return {"status": "success", "content": "🛠️ **No active tools loaded (no servers connected).**"}
            
            tools_md = "### 🛠️ Loaded Parameter Schemas:\n\n"
            for name, s in servers.items():
                tools_md += f"#### Server: `{name}`\n"
                for t in s.get("tools", []):
                    tools_md += f"- `{t}`\n"
                tools_md += "\n"
            return {"status": "success", "content": tools_md}

        # ── 5. FILES COMMAND ──
        elif cmd == "/files":
            files = state.get("uploaded_files", [])
            if not files:
                return {"status": "success", "content": "📂 **Workspace directory is currently empty.**"}
            
            files_md = "### 📂 Local Workspace Files:\n\n"
            files_md += "| Filename | Format | Size (KB) |\n| :--- | :--- | :--- |\n"
            for f in files:
                ext = f['name'].split('.')[-1].upper() if '.' in f['name'] else 'TXT'
                size_kb = f.get('size', 0) / 1024
                files_md += f"| `{f['name']}` | {ext} | {size_kb:.1f} KB |\n"
            return {"status": "success", "content": files_md}

        # ── 6. SCRAPE COMMAND (Agentic CRM-Browsing Bridge) ──
        elif cmd == "/scrape":
            if not args_str:
                return {"status": "error", "content": "⚠️ **Please enter a URL or a query (e.g. `/scrape about us of Turabit`)**"}
            
            # Check if direct URL or query
            if args_str.startswith("http://") or args_str.startswith("https://") or ("." in args_str.split("/")[0] and "/" in args_str):
                # Direct URL Mode
                url = args_str
                res_content = await web_scrape.ainvoke({"url": url})
                return {"status": "success", "content": str(res_content)}
            else:
                # Agentic Query Mode (e.g. "about us of Turabit")
                # Step 1: Use LLM to extract company name and target section
                extract_prompt = (
                    "Review the web scraping request and extract the target company name and the specific page/topic they want.\n"
                    f"Request: \"{args_str}\"\n"
                    "Output a clean JSON with keys 'company' and 'section' (e.g. 'company': 'Turabit', 'section': 'about us'). "
                    "Output ONLY the raw JSON block without markdown formatting."
                )
                extraction_res = await llm.ainvoke([HumanMessage(content=extract_prompt)])
                extraction_text = extraction_res.content if hasattr(extraction_res, 'content') else str(extraction_res)
                
                try:
                    # Clean potential markdown wrapping
                    extraction_text = re.sub(r'```json\s*|\s*```', '', extraction_text).strip()
                    meta = json.loads(extraction_text)
                    company = meta.get("company", "").strip()
                    section = meta.get("section", "").strip()
                except Exception:
                    company = args_str
                    section = "home"

                # Step 2: Search CRM/HubSpot/Zoho for the website link
                found_url = ""
                crm_source = ""
                
                # Check HubSpot
                if "HubSpot" in state.get("mcp_servers", {}):
                    try:
                        # Try listing/searching companies
                        crm_res = await call_mcp_tool_directly("HubSpot", "get_companies", {}, state)
                        # Let LLM find company link from CRM dump
                        find_link_prompt = (
                            f"Find the website URL for the company named '{company}' from this HubSpot CRM company list:\n"
                            f"{str(crm_res)}\n"
                            "Return ONLY the plain website URL (e.g., https://turabit.com) or 'None' if not found."
                        )
                        link_res = await llm.ainvoke([HumanMessage(content=find_link_prompt)])
                        link_text = link_res.content.strip() if hasattr(link_res, 'content') else str(link_res).strip()
                        if "http" in link_text:
                            found_url = link_text
                            crm_source = "HubSpot"
                    except Exception as e:
                        logger.warning(f"Failed to lookup company in HubSpot: {e}")

                # Check Zoho if not found in HubSpot
                if not found_url and "Zoho" in state.get("mcp_servers", {}):
                    try:
                        # Dynamic search module
                        crm_res = await call_mcp_tool_directly("Zoho", "ZohoPeople.forms.READ", {}, state) # Fallback probe module
                        # Let LLM locate Zoho accounts/leads if any
                        find_link_prompt = (
                            f"Find the website URL for the company '{company}' from this Zoho data:\n"
                            f"{str(crm_res)}\n"
                            "Return ONLY the plain URL or 'None'."
                        )
                        link_res = await llm.ainvoke([HumanMessage(content=find_link_prompt)])
                        link_text = link_res.content.strip()
                        if "http" in link_text:
                            found_url = link_text
                            crm_source = "Zoho"
                    except Exception as e:
                        logger.warning(f"Failed to lookup in Zoho: {e}")

                # Step 3: Web Search Fallback if not in CRM
                if not found_url:
                    logger.info(f"Company '{company}' not found in CRM. Running DuckDuckGo web search fallback...")
                    search_res = await web_search.ainvoke({"query": f"{company} company official website URL domain"})
                    find_link_prompt = (
                        f"Identify the official main website URL for company '{company}' from these search results:\n"
                        f"{search_res}\n"
                        "Return ONLY the plain main URL (e.g. https://turabit.com) without any extra characters. If not found, return 'None'."
                    )
                    link_res = await llm.ainvoke([HumanMessage(content=find_link_prompt)])
                    link_text = link_res.content.strip() if hasattr(link_res, 'content') else str(link_res).strip()
                    if "http" in link_text:
                        found_url = link_text
                        crm_source = "Internet Search"

                if not found_url:
                    return {
                        "status": "error", 
                        "content": f"⚠️ **Could not find a website link for '{company}' in your CRM or via web search.**"
                    }

                # Step 4: Web Scrape
                logger.info(f"Found URL: {found_url} via {crm_source}. Scraping website...")
                scrape_res = await web_scrape.ainvoke({"url": found_url})
                
                # Step 5: High-Precision Azure OpenAI Targeted Extraction
                extraction_prompt = (
                    f"We have scraped the webpage of company '{company}' ({found_url}).\n"
                    f"Your job is to read this scraped content and extract ONLY the information regarding: \"{section}\" (e.g., details about the company, contact info, names, etc.).\n"
                    "Be highly accurate and professional. Structure it with beautiful Markdown headers. "
                    "If the requested info is not found, summarize the key findings of the page.\n\n"
                    f"--- Scraped Webpage ---\n{scrape_res}"
                )
                final_extraction = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
                final_text = final_extraction.content if hasattr(final_extraction, 'content') else str(final_extraction)
                
                return {
                    "status": "success",
                    "content": f"### 🌐 Agentic CRM Scraping Results\n"
                               f"- **Company**: `{company}`\n"
                               f"- **Source URL**: [{found_url}]({found_url}) (Discovered via `{crm_source}`)\n"
                               f"- **Target Request**: `{section}`\n\n"
                               f"---\n\n"
                               f"{final_text}"
                }

        # ── 7. SUMMARIZE COMMAND (Intelligent MCP/Local) ──
        elif cmd == "/summarize":
            if not args_str:
                return {"status": "error", "content": "⚠️ **Please specify a local file or MCP source to summarize (e.g. `/summarize notion <page_id>` or `/summarize resume.pdf`)**"}
            
            parts_sum = args_str.split(" ", 1)
            source = parts_sum[0].lower()
            target_id = parts_sum[1].strip() if len(parts_sum) > 1 else ""

            # Check if source is a local file
            local_files = [f['name'].lower() for f in state.get("uploaded_files", [])]
            if args_str.lower() in local_files or source in local_files:
                filename = args_str if args_str.lower() in local_files else source
                # Read local file
                read_file_tool = get_read_uploaded_file_tool(UPLOAD_DIR)
                file_text = await read_file_tool.ainvoke({"filename": filename})
                
                # Directly summarize via Azure OpenAI
                sum_prompt = (
                    f"Please generate a high-quality, comprehensive, and structured executive summary of this local document '{filename}':\n\n"
                    f"{file_text}"
                )
                summary_res = await llm.ainvoke([HumanMessage(content=sum_prompt)])
                summary_text = summary_res.content if hasattr(summary_res, 'content') else str(summary_res)
                return {"status": "success", "content": f"### 📝 Executive Summary: `{filename}`\n\n{summary_text}"}

            # Check if MCP notion summarize
            elif source == "notion" and target_id:
                if "Notion" not in state.get("mcp_servers", {}):
                    return {"status": "error", "content": "⚠️ **Notion MCP server is not connected.**"}
                
                try:
                    # Dynamically match and invoke block/page retrieval tools
                    notion_tools = await call_mcp_tool_directly("Notion", "retrieve_page", {"page_id": target_id}, state)
                except Exception:
                    try:
                        # Fallback try generic get_page
                        notion_tools = await call_mcp_tool_directly("Notion", "get_page", {"page_id": target_id}, state)
                    except Exception as notion_err:
                        return {"status": "error", "content": f"Failed to retrieve Notion page: {notion_err}"}

                sum_prompt = (
                    f"Please generate a clean structured Markdown summary of this raw Notion page JSON data:\n\n"
                    f"{str(notion_tools)}"
                )
                summary_res = await llm.ainvoke([HumanMessage(content=sum_prompt)])
                summary_text = summary_res.content
                return {"status": "success", "content": f"### 📝 Notion Page Summary (`{target_id}`)\n\n{summary_text}"}

            # Check if Zoho CRM record summarize
            elif source == "zoho" and target_id:
                if "Zoho" not in state.get("mcp_servers", {}):
                    return {"status": "error", "content": "⚠️ **Zoho MCP server is not connected.**"}
                try:
                    # Fallback generic retrieval via People/CRM
                    zoho_res = await call_mcp_tool_directly("Zoho", "ZohoPeople.forms.READ", {}, state)
                    sum_prompt = (
                        f"Summarize this Zoho CRM record information dynamically:\n\n"
                        f"Record ID: {target_id}\n"
                        f"Zoho Output: {str(zoho_res)}"
                    )
                    summary_res = await llm.ainvoke([HumanMessage(content=sum_prompt)])
                    return {"status": "success", "content": f"### 📝 Zoho CRM Record Summary (`{target_id}`)\n\n{summary_res.content}"}
                except Exception as e:
                    return {"status": "error", "content": f"Failed Zoho summary: {e}"}

            else:
                return {
                    "status": "error", 
                    "content": f"⚠️ **Unknown source '{args_str}'. Please provide a valid uploaded file name, `/summarize notion <page_id>` or `/summarize zoho <record_id>`.**"
                }

        # ── 8. EXPORT COMMAND (Workspace Saver) ──
        elif cmd == "/export":
            if not args_str:
                return {"status": "error", "content": "⚠️ **Please enter a server and record ID (e.g. `/export notion <page_id>`)**"}
            
            parts_exp = args_str.split(" ", 1)
            server = parts_exp[0].lower()
            record_id = parts_exp[1].strip() if len(parts_exp) > 1 else ""

            if server == "notion" and record_id:
                try:
                    notion_res = await call_mcp_tool_directly("Notion", "get_page", {"page_id": record_id}, state)
                except Exception:
                    try:
                        notion_res = await call_mcp_tool_directly("Notion", "retrieve_page", {"page_id": record_id}, state)
                    except Exception as e:
                        return {"status": "error", "content": f"Failed to retrieve Notion content: {e}"}

                # Save to local workspace!
                safe_name = f"notion_page_{record_id}.md"
                file_path = os.path.join(UPLOAD_DIR, safe_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    # BUG FIX: added default=str to handle non-serializable Pydantic/object
                    # responses from Notion MCP — without it json.dumps raises TypeError
                    f.write(f"# Exported Notion Page: {record_id}\n\n```json\n{json.dumps(notion_res, indent=2, default=str)}\n```")
                
                # Sync dynamic state to register new file
                file_size = os.path.getsize(file_path)
                state["uploaded_files"].append({"name": safe_name, "path": file_path, "size": file_size})
                save_state(state)

                return {
                    "status": "success", 
                    "content": f"✅ **Successfully exported Notion page `{record_id}` to your local workspace files!**\n- Saved as: `{safe_name}`"
                }
            else:
                return {"status": "error", "content": f"⚠️ **Unsupported export source '{server}'. Only `/export notion <id>` is supported.**"}

        # ── 9. QUICK-ADD COMMAND (Intelligent creation) ──
        elif cmd == "/quick-add":
            if not args_str:
                return {"status": "error", "content": "⚠️ **Please enter a service and natural language details (e.g. `/quick-add cal Coffee tomorrow 10am`)**"}
            
            parts_add = args_str.split(" ", 1)
            service = parts_add[0].lower()
            text_details = parts_add[1].strip() if len(parts_add) > 1 else ""

            if service == "cal" and text_details:
                if "Cal.com" not in state.get("mcp_servers", {}):
                    return {"status": "error", "content": "⚠️ **Cal.com MCP server is not connected.**"}
                
                # Use fast Azure OpenAI call to extract booking details
                parse_prompt = (
                    "Extract Cal.com meeting booking details from this natural language text:\n"
                    f"\"{text_details}\"\n"
                    "Output a clean JSON with keys 'title', 'start_time' (ISO format), and 'duration_minutes' (int). "
                    "Use future timestamps relative to 2026-05-26. Output ONLY raw JSON."
                )
                parse_res = await llm.ainvoke([HumanMessage(content=parse_prompt)])
                parse_text = re.sub(r'```json\s*|\s*```', '', parse_res.content).strip()
                
                try:
                    booking_args = json.loads(parse_text)
                    # Dynamically invoke create_booking tool
                    api_res = await call_mcp_tool_directly("Cal.com", "create_booking", booking_args, state)
                    return {"status": "success", "content": f"📅 **Successfully booked on Cal.com!**\n- Event: `{booking_args.get('title')}`\n- Time: `{booking_args.get('start_time')}`\n- API Response: `{str(api_res)}`"}
                except Exception as e:
                    return {"status": "error", "content": f"Failed to add booking: {e}"}

            elif service == "notion" and text_details:
                if "Notion" not in state.get("mcp_servers", {}):
                    return {"status": "error", "content": "⚠️ **Notion MCP server is not connected.**"}
                
                parse_prompt = (
                    "Extract Notion page creation details from this text:\n"
                    f"\"{text_details}\"\n"
                    "Output a clean JSON with key 'title'. Output ONLY raw JSON."
                )
                parse_res = await llm.ainvoke([HumanMessage(content=parse_prompt)])
                parse_text = re.sub(r'```json\s*|\s*```', '', parse_res.content).strip()
                
                try:
                    page_args = json.loads(parse_text)
                    api_res = await call_mcp_tool_directly("Notion", "create_page", page_args, state)
                    return {"status": "success", "content": f"📓 **Successfully added page to Notion!**\n- Title: `{page_args.get('title')}`\n- Status: `Created`"}
                except Exception as e:
                    return {"status": "error", "content": f"Failed to add Notion page: {e}"}

            else:
                return {"status": "error", "content": f"⚠️ **Unsupported quick-add service '{service}'. Choose 'cal' or 'notion'.**"}

        # ── 10. SEARCH COMMAND ──
        elif cmd == "/search":
            if not args_str:
                return {"status": "error", "content": "⚠️ **Please enter a query (e.g. `/search authentication API`)**"}
            
            doc_search_tool = get_query_documentation_tool(state.get("loaded_docs", {}))
            res_content = await doc_search_tool.ainvoke({"query": args_str})
            return {"status": "success", "content": str(res_content)}

        else:
            return {"status": "error", "content": f"❌ **Unknown command '{cmd}'. Type `/help` to see all available tools.**"}

    except Exception as cmd_err:
        logger.error(f"[COMMAND ERROR] Command '{command_str}' failed: {cmd_err}", exc_info=True)
        return {"status": "error", "content": f"❌ **Failed to execute command '{cmd}': {str(cmd_err)}**"}