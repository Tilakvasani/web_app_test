"""
LangGraph ReAct Agent Execution Module.

Constructs and executes the streaming multi-server LangGraph ReAct agent.
Integrates dynamic MCP clients, binds local search/workspace helper tools,
and coordinates pre-execution strategic planners.
"""

import hashlib
import json
import logging
import os
import re
import tiktoken
from typing import Dict, Any, List, AsyncGenerator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import get_llm, UPLOAD_DIR, _make_cfg
from database import cache
from prompts import build_system_prompt
from tools import (
    wrap_tool_with_coercion, 
    get_read_uploaded_file_tool, 
    get_query_documentation_tool,
    web_search,
    web_scrape
)

logger = logging.getLogger("mcp_backend")

def count_tokens(text: str) -> int:
    """Helper to count token length of text using the cl100k_base tokenizer."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4

async def compress_history_sandwich(
    history: List[Dict[str, str]], 
    llm_inst
) -> List[Dict[str, str]]:
    """
    Compresses chat history using the Sandwich Method:
    - Filters out noisy messages (e.g. simple greetings, "ok", "ok bro").
    - If total history size is <= 8 messages, returns it directly.
    - If > 8 messages:
      - Keeps the first 3 messages.
      - Keeps the last 3 messages.
      - Condenses all middle messages into a single summarized AIMessage/HumanMessage context.
    """
    noise_patterns = [
        r"^(hi|hello|hey|greetings|howdy|ok|ok bro|thanks|thank you|ok thanks|cool)$"
    ]
    cleaned = []
    for msg in history:
        content = msg.get("content", "").strip()
        is_noise = False
        for pat in noise_patterns:
            if re.match(pat, content, re.IGNORECASE):
                is_noise = True
                break
        
        # Don't filter out if it is an important detailed request
        if is_noise and len(content) < 30:
            logger.info(f"[HISTORY COMPRESS] Pruned noise message: '{content}'")
            continue
        cleaned.append(msg)
        
    if len(cleaned) <= 8:
        return cleaned

    # Extract sandwich slices
    first_slice = cleaned[:3]
    last_slice = cleaned[-3:]
    middle_slice = cleaned[3:-3]

    # Summarize middle slice
    middle_text = ""
    for msg in middle_slice:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        middle_text += f"{role}: {content}\n"

    # Compute a unique key for the middle slice using MD5
    middle_hash = hashlib.md5(middle_text.encode("utf-8")).hexdigest()
    cache_key = f"summary:{len(middle_slice)}:{middle_hash}"
    cached_summary = await cache.get(cache_key)
    
    if cached_summary:
        logger.info(f"[HISTORY COMPRESS] Middle summary found in cache (Key: {cache_key})!")
        summary_paragraph = cached_summary
    else:
        logger.info(f"[HISTORY COMPRESS] Summary cache miss. Requesting LLM summarization (Key: {cache_key})...")
        summary_prompt = (
            "Generate a brief, high-level summary of the following conversation segment, highlighting the key questions asked, decisions made, and actions performed.\n"
            "Be extremely concise, wrapping everything in a single, dense paragraph.\n\n"
            f"--- Conversation Segment ---\n{middle_text}"
        )
        
        try:
            summary_res = await llm_inst.ainvoke([SystemMessage(content=summary_prompt)])
            summary_paragraph = summary_res.content if hasattr(summary_res, "content") else str(summary_res)
            # Store summary in cache for 1 hour (3600 seconds)
            await cache.set(cache_key, summary_paragraph.strip(), ttl=3600)
        except Exception as e:
            logger.error(f"[HISTORY COMPRESS ERROR] Summarization failed: {e}")
            summary_paragraph = "Middle conversation segment summarized."

    # Build sandwich
    sandwich_history = []
    sandwich_history.extend(first_slice)
    
    # Inject summary as an assistant summary context
    sandwich_history.append({
        "role": "assistant",
        "content": f"[System Context: High-level summary of middle conversation: {summary_paragraph.strip()}]"
    })
    
    sandwich_history.extend(last_slice)
    logger.info(f"[HISTORY COMPRESS] Sandwich compiled successfully! Messages reduced from {len(history)} to {len(sandwich_history)}.")
    return sandwich_history

# Global in-memory cache for pre-fetched tools to eliminate redundant network fetches
_MCP_TOOLS_CACHE = {}

async def stream_agent_interaction(
    prompt: str, 
    history: List[Dict[str, str]], 
    state: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Asynchronously streams the ReAct agent's thoughts, tool invocations, and 
    final responses back to the frontend client using Server-Sent Events (SSE).
    """
    llm_inst = get_llm()
    
    # ── 0. Compress Chat History using the Sandwich Method ──
    compressed_history = await compress_history_sandwich(history, llm_inst)

    # ── 1. Reassemble Dynamic LangChain Messages ──
    msgs = []
    for h in compressed_history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=prompt))

    # ── 1.5. Pre-fetch All Connected MCP Tools Dynamically (with global in-memory cache) ──
    global _MCP_TOOLS_CACHE
    all_mcp_tools = {}
    
    if state.get("mcp_servers"):
        servers_to_fetch = {}
        for name, s in state["mcp_servers"].items():
            # Key uniquely identifies this server instance (handles config changes safely!)
            srv_key = f"{name}:{s.get('url')}:{s.get('auth_value')}"
            if srv_key in _MCP_TOOLS_CACHE:
                all_mcp_tools[name] = _MCP_TOOLS_CACHE[srv_key]
            else:
                servers_to_fetch[name] = (s, srv_key)
                
        if servers_to_fetch:
            logger.info(f"[CHAT] Cache miss for servers {list(servers_to_fetch.keys())}. Pre-fetching tool definitions dynamically...")
            configs = {}
            for name, (s, srv_key) in servers_to_fetch.items():
                transport = s.get("transport", "streamable_http")
                cfg = _make_cfg(s["url"], s["auth_type"], s["auth_value"], transport=transport)
                if s.get("api_header") and s.get("auth_type") == "api_key":
                    cfg["headers"] = {s["api_header"]: s["auth_value"]}
                configs[name] = cfg
                
            if configs:
                client = MultiServerMCPClient(configs)
                for name in configs.keys():
                    try:
                        mcp_tools = await client.get_tools(server_name=name)
                        srv_key = servers_to_fetch[name][1]
                        _MCP_TOOLS_CACHE[srv_key] = mcp_tools
                        all_mcp_tools[name] = mcp_tools
                        logger.info(f"[CHAT] Successfully pre-fetched and cached {len(mcp_tools)} tools from server '{name}'.")
                    except Exception as ex:
                        logger.error(f"[CHAT ERROR] Failed to pre-fetch tools for '{name}': {ex}")

    # ── 2. Smart Planning & Dynamic Tool Sifter ──
    total_input_tokens = 0
    total_output_tokens = 0
    needs_plan = True
    needs_tools = True
    tool_groups = []
    plan_text = ""
    
    try:
        server_descriptions = ""
        for name, mcp_tools in all_mcp_tools.items():
            tool_names = [getattr(t, "name", "") for t in mcp_tools]
            tools_summary = ", ".join(tool_names[:6])
            if len(tool_names) > 6:
                tools_summary += f", and {len(tool_names) - 6} more"
            server_descriptions += f"- '{name}': Handles operations utilizing tools: {tools_summary}.\n"

        classification_prompt = (
            "Analyze the following user query and conversation history, and classify it into one of three execution categories:\n\n"
            "1. 'CONVERSATIONAL': Greetings, small talk, general chit-chat, capabilities questions, general info, or general statements (e.g. 'hi', 'how are you', 'tell me a joke', 'who are you', 'thank you').\n"
            "2. 'DIRECT_TOOL': Simple single-step requests that require executing tools but do NOT need a complex multi-step sequential plan (e.g. 'give me list of companies', 'search files for X', 'scrape website Y', 'get Notion page Z').\n"
            "3. 'COMPLEX': Multi-step workflows requiring sequential tool executions and coordination (e.g. 'look up company X in HubSpot, scrape website, and write to Notion').\n\n"
            "Also, identify which tool groups (connected server names or Local) are required to solve the query. The available tool groups are:\n"
            f"{server_descriptions}"
            "- 'Local': For local workspace files, documentation search, web searching, or web scraping.\n\n"
            f"User Query: \"{prompt}\"\n\n"
            "Output your decision as a raw JSON block with keys 'category' (string) and 'tool_groups' (list of strings). "
            "Only output the JSON block, nothing else."
        )
        
        class_msgs = [SystemMessage(content=classification_prompt)]
        class_res = await llm_inst.ainvoke(class_msgs)
        
        # Track sifter token usage
        if getattr(class_res, "usage_metadata", None):
            total_input_tokens += class_res.usage_metadata.get("input_tokens", 0)
            total_output_tokens += class_res.usage_metadata.get("output_tokens", 0)
        else:
            total_input_tokens += count_tokens(classification_prompt)
            total_output_tokens += count_tokens(class_res.content)
            
        class_decision_text = class_res.content.strip()
        class_decision_text = re.sub(r"```json\s*|\s*```", "", class_decision_text).strip()
        
        try:
            meta = json.loads(class_decision_text)
            category = meta.get("category", "COMPLEX").strip().upper()
            tool_groups = [g.strip() for g in meta.get("tool_groups", [])]
        except Exception:
            category = "COMPLEX"
            tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]
            
        logger.info(f"[PLANNER] Sifter decision: {category} | Groups: {tool_groups}")
        
        if "CONVERSATIONAL" in category:
            needs_plan = False
            needs_tools = False
            plan_text = "Pure conversational small talk."
        elif "DIRECT_TOOL" in category:
            needs_plan = False
            needs_tools = True
            plan_text = "Direct response or direct tool execution."
        else:
            needs_plan = True
            needs_tools = True
    except Exception as sifter_err:
        logger.error(f"[PLANNER ERROR] Smart sifter classification failed: {sifter_err}")
        needs_plan = True
        needs_tools = True
        tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]

    # ── 3. Resolve Connected MCP & Local Tools (Only if Needed!) ──
    tools = []
    
    if needs_tools:
        for name in tool_groups:
            # Match case-insensitively with active servers
            matched_name = None
            for server_name in all_mcp_tools.keys():
                if name.lower() in server_name.lower() or server_name.lower() in name.lower():
                    matched_name = server_name
                    break
            if matched_name:
                tools.extend([wrap_tool_with_coercion(t) for t in all_mcp_tools[matched_name]])
                logger.info(f"[CHAT] Loaded {len(all_mcp_tools[matched_name])} tools from server '{matched_name}'.")

        # Bind secure file reader tool (only if 'Local' is requested)
        if "Local" in tool_groups and state.get("uploaded_files"):
            read_file_tool = get_read_uploaded_file_tool(UPLOAD_DIR)
            tools.append(read_file_tool)
            logger.info("[CHAT] Equipped agent with file reading tool 'read_uploaded_file'.")
            
        # Bind indexed documentation search tool (only if 'Local' is requested)
        if "Local" in tool_groups and state.get("loaded_docs"):
            doc_search_tool = get_query_documentation_tool(state["loaded_docs"])
            tools.append(doc_search_tool)
            logger.info("[CHAT] Equipped agent with doc query tool 'query_documentation'.")

        # Bind web search and scrape tools (only if 'Local' is requested)
        if "Local" in tool_groups:
            tools.append(web_search)
            tools.append(web_scrape)
            logger.info("[CHAT] Equipped agent with web_search and web_scrape tools.")

        # ── 3.5. Semantic Tool RAG Filtering ──
        if tools:
            from database import retrieve_relevant_tools
            # Filter tools down to the top 7 most relevant ones matching the prompt
            tools = await retrieve_relevant_tools(prompt, tools, top_k=7)

    # ── 4. Strategic Planning Phase (One Main LLM Call if Required) ──
    if needs_plan:
        logger.info("[PLANNER] Running pre-execution strategic planning call...")
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Creating strategic execution plan...'})}\n\n"
        
        try:
            planning_system_prompt = (
                "You are the Strategic Planning Engine for an advanced multi-tool AI sales and productivity assistant.\n"
                "Your job is to analyze the user's request and the conversation history, and generate a precise, step-by-step TODO list "
                "showing exactly how the query should be solved using the available resources (HubSpot CRM, Zoho CRM, Notion Workspace, Cal.com, "
                "Google Drive, Gmail, web search, web scrape, local files, and docs search).\n"
                "Focus heavily on workflow dependencies (e.g. '1. Search HubSpot for X... 2. Get website... 3. Scrape and extract details... 4. Enrich CRM').\n"
                "Output the plan clearly. Do NOT call any tools yet—just outline the strategy. Keep it structured and action-oriented."
            )
            
            planning_msgs = [SystemMessage(content=planning_system_prompt)]
            for h in compressed_history:
                role = h.get("role")
                content = h.get("content", "")
                if role == "user":
                    planning_msgs.append(HumanMessage(content=content))
                elif role == "assistant":
                    planning_msgs.append(AIMessage(content=content))
            planning_msgs.append(HumanMessage(content=f"Generate a strategic execution plan (TODO list) for: '{prompt}'"))
            
            plan_response = await llm_inst.ainvoke(planning_msgs)
            
            # Track planning token usage
            if getattr(plan_response, "usage_metadata", None):
                total_input_tokens += plan_response.usage_metadata.get("input_tokens", 0)
                total_output_tokens += plan_response.usage_metadata.get("output_tokens", 0)
            else:
                total_input_tokens += sum(count_tokens(msg.content) for msg in planning_msgs)
                total_output_tokens += count_tokens(plan_response.content)
                
            plan_text = plan_response.content if hasattr(plan_response, "content") else str(plan_response)
            
            # Stream the strategic plan immediately to the Streamlit UI!
            yield f"event: thought\ndata: {json.dumps({'text': f'📋 **Strategic Execution Plan Created**:\\n{plan_text}'})}\n\n"
        except Exception as plan_err:
            logger.error(f"[PLANNER ERROR] Strategic planning call failed: {plan_err}")
            plan_text = "Step 1: Directly process request using available tools."

    # ── 5. Generate Dynamic System Prompt ──
    base_prompt = build_system_prompt(state, active_groups=tool_groups)
    if needs_plan:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"### STRATEGIC PLAN TO FOLLOW:\n"
            f"You must strictly follow this sequential execution plan to solve the user's request:\n"
            f"{plan_text}\n\n"
            f"Use your tools (like web_search, web_scrape, HubSpot, Zoho, Notion, Cal) as dictated by the plan. "
            f"When scraping websites for companies, use the LLM context to cleanly extract contact details, addresses, and business profiles—do not use regex. "
            f"Always enrich HubSpot/Zoho creations with these discovered internet details!"
        )
    elif needs_tools:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"Respond to the user directly or execute any required tools to answer their query immediately. "
            f"If tools are needed, call them dynamically. Otherwise, provide a clear and helpful direct response."
        )
    else:
        # Conversational fallback (pure small talk - no tools at all!)
        system_prompt = (
            "You are a friendly, helpful, and highly capable AI productivity assistant. "
            "Respond directly to the user's greeting or conversational chat in a friendly and professional manner. "
            "Keep the response brief and pleasant."
        )
    
    # SSE Stream Generator
    logger.info("[AGENT EXECUTE] Initializing LangGraph ReAct agent stream...")
    if needs_plan:
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Starting plan execution with dynamic tools...'})}\n\n"
    
    try:
        if tools:
            agent = create_react_agent(llm_inst, tools, prompt=system_prompt)
            async for chunk in agent.astream({"messages": msgs}):
                if "agent" in chunk:
                    for msg in chunk["agent"]["messages"]:
                        # Track token usage from agent messages
                        if getattr(msg, "usage_metadata", None):
                            total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
                            total_output_tokens += msg.usage_metadata.get("output_tokens", 0)
                        
                        # Tool call event detection
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                logger.info(f"[AGENT THOUGHT] Calling tool: {tc['name']}")
                                yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'args': tc['args']})}\n\n"
                                
                        # Streaming content logs
                        if hasattr(msg, "content") and msg.content:
                            logger.info(f"[AGENT CHUNK] Assistant text chunk returned.")
                            yield f"event: content\ndata: {json.dumps({'text': msg.content})}\n\n"
                            
                elif "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        logger.info(f"[TOOL RESPONSE] Tool response captured for: {msg.name}")
                        yield f"event: tool_output\ndata: {json.dumps({'name': msg.name, 'content': str(msg.content)})}\n\n"
        else:
            # Fallback streaming without active tools (pure conversational greeting!)
            logger.info("[CHAT] No tools loaded. Streaming chat response directly from LLM...")
            full_msgs = [SystemMessage(content=system_prompt)] + msgs
            total_input_tokens += sum(count_tokens(m.content) for m in full_msgs)
            
            full_response = ""
            async for token_chunk in llm_inst.astream(full_msgs):
                if token_chunk.content:
                    full_response += token_chunk.content
                    yield f"event: content\ndata: {json.dumps({'text': token_chunk.content})}\n\n"
            total_output_tokens += count_tokens(full_response)
                    
        # Yield total tracked tokens to frontend
        yield f"event: token_usage\ndata: {json.dumps({'input_tokens': total_input_tokens, 'output_tokens': total_output_tokens})}\n\n"
        yield f"event: thought\ndata: {json.dumps({'text': '✅ Completed successfully'})}\n\n"
        
    except Exception as e:
        logger.error(f"[CHAT ERROR] Streaming execution encountered error: {e}", exc_info=True)
        from routes.state_routes import unpack_exception
        clean_msg = unpack_exception(e)
        yield f"event: error\ndata: {json.dumps({'message': clean_msg})}\n\n"
