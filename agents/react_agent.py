"""
LangGraph ReAct Agent Execution Module.

Constructs and executes the streaming multi-server LangGraph ReAct agent.
Integrates dynamic MCP clients, binds local search/workspace helper tools,
and coordinates pre-execution strategic planners.

Memory strategy (post-Redis migration):
  - LangGraph MemorySaver is passed as `checkpointer` to create_react_agent.
  - Each conversation is identified by a `thread_id`; LangGraph manages the
    full message state per thread automatically.
  - On the FIRST call for a thread the caller provides prior `history` to seed
    the checkpoint.  On subsequent calls only the new message is passed — 
    LangGraph appends it to the existing checkpoint state.
  - compress_history_sandwich() is still called for the initial seed to prune
    noise and stay within token limits; it uses the LangGraph InMemoryStore
    cache (no Redis).
"""

import asyncio
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
from database import cache, checkpointer
from database.langgraph_memory import KEY_SIFTER, TTL_SIFTER
from prompts import build_system_prompt
from tools import (
    wrap_tool_with_coercion,
    get_read_uploaded_file_tool,
    get_prepare_file_for_upload_tool,
    get_query_documentation_tool,
    web_search,
    web_scrape
)

logger = logging.getLogger("mcp_backend")

# ── Tiktoken encoder (module-level singleton) ──────────────────────────────────
_TIKTOKEN_ENC = None

def _get_encoder():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    return _TIKTOKEN_ENC


def count_tokens(text: str) -> int:
    try:
        enc = _get_encoder()
        if enc:
            return len(enc.encode(text))
    except Exception:
        pass
    return len(text) // 4


# ── Thread tracking ────────────────────────────────────────────────────────────
# Tracks which thread_ids have already been seeded into the MemorySaver.
# On the first call for a thread we pass the full (compressed) history.
# On every subsequent call we pass only the new user message — LangGraph
# automatically loads and appends to the existing checkpoint state.
_INITIALIZED_THREADS: set = set()


async def compress_history_sandwich(
    history: List[Dict[str, str]],
    llm_inst
) -> List[Dict[str, str]]:
    """
    Compresses chat history for the initial thread seed:
      - Prunes trivial noise messages.
      - If <= 8 messages, returns as-is.
      - Otherwise keeps first 3 + last 3 and summarises the middle via LLM.
        The summary is cached in LangGraph InMemoryStore (1 h TTL) so repeat
        sessions with the same middle slice skip the LLM round-trip.
    """
    noise_patterns = [
        r"^(hi|hello|hey|greetings|howdy|ok|ok bro|thanks|thank you|ok thanks|cool)$"
    ]
    cleaned = []
    for msg in history:
        content = msg.get("content", "").strip()
        is_noise = any(re.match(p, content, re.IGNORECASE) for p in noise_patterns)
        if is_noise and len(content) < 30:
            logger.info(f"[HISTORY COMPRESS] Pruned noise: '{content}'")
            continue
        cleaned.append(msg)

    if len(cleaned) <= 8:
        return cleaned

    first_slice  = cleaned[:3]
    last_slice   = cleaned[-3:]
    middle_slice = cleaned[3:-3]

    middle_text = "\n".join(
        f"{m.get('role','user').upper()}: {m.get('content','')}"
        for m in middle_slice
    )
    middle_hash = hashlib.sha256(middle_text.encode()).hexdigest()[:32]

    cached_summary = await cache.get_history_summary(middle_hash)
    if cached_summary:
        logger.info(f"[HISTORY COMPRESS] Summary cache hit (hash={middle_hash})")
        summary_paragraph = cached_summary
    else:
        logger.info("[HISTORY COMPRESS] Summarising middle slice via LLM...")
        summary_prompt = (
            "Generate a brief, high-level summary of the following conversation segment, "
            "highlighting key questions, decisions, and actions. "
            "One dense paragraph only.\n\n"
            f"--- Conversation ---\n{middle_text}"
        )
        try:
            res = await llm_inst.ainvoke([SystemMessage(content=summary_prompt)])
            summary_paragraph = res.content if hasattr(res, "content") else str(res)
            await cache.set_history_summary(middle_hash, summary_paragraph.strip())
        except Exception as e:
            logger.error(f"[HISTORY COMPRESS ERROR] {e}")
            summary_paragraph = "Middle conversation segment summarised."

    return [
        *first_slice,
        {"role": "assistant",
         "content": f"[System Context: Summary of prior conversation: {summary_paragraph.strip()}]"},
        *last_slice,
    ]


# ── MCP tool cache (keyed by server name + url, NOT token) ────────────────────
_MCP_TOOLS_CACHE: Dict[str, list] = {}


async def stream_agent_interaction(
    prompt: str,
    history: List[Dict[str, str]],
    state: Dict[str, Any],
    thread_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Streams the ReAct agent's thoughts, tool invocations, and final response
    back to the frontend via Server-Sent Events (SSE).

    thread_id  — identifies the conversation in LangGraph's MemorySaver.
                 First call for a thread seeds it with compressed history.
                 Subsequent calls pass only the new message.
    """
    llm_inst = get_llm()

    # ── 0. Decide input messages based on thread state ─────────────────────────
    is_new_thread = thread_id not in _INITIALIZED_THREADS

    if is_new_thread:
        # First time: compress + seed the checkpoint with prior context
        compressed_history = await compress_history_sandwich(history, llm_inst)
        msgs = []
        for h in compressed_history:
            role    = h.get("role")
            content = h.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        msgs.append(HumanMessage(content=prompt))
        _INITIALIZED_THREADS.add(thread_id)
        logger.info(f"[AGENT] New thread '{thread_id}' — seeding with {len(msgs)} messages.")
    else:
        # Existing thread: LangGraph loads full history from MemorySaver
        msgs = [HumanMessage(content=prompt)]
        logger.info(f"[AGENT] Existing thread '{thread_id}' — passing new message only.")

    # ── 1. Pre-fetch MCP tools (token-rotation-safe cache) ────────────────────
    global _MCP_TOOLS_CACHE
    all_mcp_tools: Dict[str, list] = {}

    if state.get("mcp_servers"):
        servers_to_fetch = {}
        for name, s in state["mcp_servers"].items():
            srv_key = f"{name}:{s.get('url')}"
            if srv_key in _MCP_TOOLS_CACHE:
                all_mcp_tools[name] = _MCP_TOOLS_CACHE[srv_key]
            else:
                servers_to_fetch[name] = (s, srv_key)

        if servers_to_fetch:
            logger.info(f"[CHAT] Tool cache miss for {list(servers_to_fetch.keys())}. Fetching...")
            configs = {}
            for name, (s, srv_key) in servers_to_fetch.items():
                transport = s.get("transport", "streamable_http")
                cfg = _make_cfg(s["url"], s["auth_type"], s["auth_value"], transport=transport)
                if s.get("api_header") and s.get("auth_type") == "api_key":
                    cfg["headers"] = {s["api_header"]: s["auth_value"]}
                configs[name] = cfg

            async def _fetch_one(name: str):
                try:
                    single_client = MultiServerMCPClient({name: configs[name]})
                    mcp_tools = await single_client.get_tools()
                    srv_key = servers_to_fetch[name][1]
                    _MCP_TOOLS_CACHE[srv_key] = mcp_tools
                    all_mcp_tools[name] = mcp_tools
                    logger.info(f"[CHAT] Fetched {len(mcp_tools)} tools from '{name}'.")
                except Exception as ex:
                    logger.error(f"[CHAT ERROR] Tool fetch failed for '{name}': {ex}")

            await asyncio.gather(*[_fetch_one(n) for n in configs.keys()])

    # ── 2. Smart Sifter (cached in LangGraph InMemoryStore) ───────────────────
    total_input_tokens  = 0
    total_output_tokens = 0
    needs_plan  = True
    needs_tools = True
    tool_groups = []
    plan_text   = ""

    try:
        server_descriptions = ""
        for name, mcp_tools in all_mcp_tools.items():
            tool_names    = [getattr(t, "name", "") for t in mcp_tools]
            tools_summary = ", ".join(tool_names[:6])
            if len(tool_names) > 6:
                tools_summary += f", and {len(tool_names) - 6} more"
            server_descriptions += f"- '{name}': {tools_summary}.\n"

        for srv_name in state.get("mcp_servers", {}).keys():
            if srv_name not in all_mcp_tools:
                server_descriptions += f"- '{srv_name}': Connected server (tools loading).\n"

        classification_prompt = (
            "Analyze the following user query and conversation history, and classify it:\n\n"
            "1. 'CONVERSATIONAL': Greetings, small talk, general info, capabilities.\n"
            "2. 'DIRECT_TOOL': Single-step tool requests (list companies, search files, etc).\n"
            "3. 'COMPLEX': Multi-step workflows needing sequential tool coordination.\n\n"
            "Also identify required tool groups from:\n"
            f"{server_descriptions}"
            "- 'Local': files, docs, web search, web scraping.\n\n"
            f"User Query: \"{prompt}\"\n\n"
            "Output ONLY a raw JSON block: {\"category\": \"...\", \"tool_groups\": [...]}"
        )

        sifter_hash = hashlib.sha256(
            f"{prompt.lower().strip()}:{server_descriptions}".encode()
        ).hexdigest()[:32]

        cached_sifter = await cache.get_sifter(sifter_hash)
        if cached_sifter:
            category    = cached_sifter.get("category", "COMPLEX")
            tool_groups = cached_sifter.get("tool_groups", [])
            logger.info(f"[PLANNER] Sifter cache hit: {category} | {tool_groups}")
        else:
            class_res = await llm_inst.ainvoke([SystemMessage(content=classification_prompt)])
            if getattr(class_res, "usage_metadata", None):
                total_input_tokens  += class_res.usage_metadata.get("input_tokens", 0)
                total_output_tokens += class_res.usage_metadata.get("output_tokens", 0)
            else:
                total_input_tokens  += count_tokens(classification_prompt)
                total_output_tokens += count_tokens(class_res.content)

            cleaned_text = re.sub(r"```json\s*|\s*```", "", class_res.content.strip()).strip()
            try:
                meta        = json.loads(cleaned_text)
                category    = meta.get("category", "COMPLEX").strip().upper()
                tool_groups = [g.strip() for g in meta.get("tool_groups", [])]
                await cache.set_sifter(sifter_hash, {"category": category, "tool_groups": tool_groups})
            except Exception:
                category    = "COMPLEX"
                tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]

            logger.info(f"[PLANNER] Sifter decision: {category} | {tool_groups}")

        if "CONVERSATIONAL" in category:
            needs_plan = needs_tools = False
            plan_text = "Pure conversational small talk."
        elif "DIRECT_TOOL" in category:
            needs_plan  = False
            needs_tools = True
            plan_text = "Direct tool execution."
        else:
            needs_plan = needs_tools = True

    except Exception as sifter_err:
        logger.error(f"[PLANNER ERROR] Sifter failed: {sifter_err}")
        needs_plan = needs_tools = True
        tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]

    # ── 3. Resolve tools ───────────────────────────────────────────────────────
    tools = []

    if needs_tools:
        for name in tool_groups:
            for server_name in all_mcp_tools:
                if name.lower() in server_name.lower() or server_name.lower() in name.lower():
                    tools.extend([wrap_tool_with_coercion(t) for t in all_mcp_tools[server_name]])
                    logger.info(f"[CHAT] Loaded {len(all_mcp_tools[server_name])} tools from '{server_name}'.")
                    break

        if "Local" in tool_groups and state.get("uploaded_files"):
            tools.append(get_read_uploaded_file_tool(UPLOAD_DIR))
            tools.append(get_prepare_file_for_upload_tool(UPLOAD_DIR))

        if "Local" in tool_groups and state.get("loaded_docs"):
            tools.append(get_query_documentation_tool(state["loaded_docs"]))

        if "Local" in tool_groups:
            tools.append(web_search)
            tools.append(web_scrape)

        if tools:
            from database import retrieve_relevant_tools
            tools = await retrieve_relevant_tools(prompt, tools, top_k=5)

    # ── 4. Strategic Planning Phase ───────────────────────────────────────────
    if needs_plan:
        logger.info("[PLANNER] Running pre-execution strategic planning...")
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Creating strategic execution plan...'})}\\n\\n"

        try:
            planning_system_prompt = (
                "You are the Strategic Planning Engine for an advanced multi-tool AI assistant.\n"
                "Analyse the user's request and generate a precise step-by-step TODO list showing "
                "how to solve it using available tools (HubSpot CRM, Zoho CRM, Notion, Cal.com, "
                "Google Drive, Gmail, web search/scrape, local files, docs search).\n"
                "Do NOT call tools yet — just outline the strategy. Keep it structured."
            )
            planning_msgs = [SystemMessage(content=planning_system_prompt)]
            for h in (compressed_history if is_new_thread else history[-6:]):
                r = h.get("role"); c = h.get("content", "")
                if r == "user":      planning_msgs.append(HumanMessage(content=c))
                elif r == "assistant": planning_msgs.append(AIMessage(content=c))
            planning_msgs.append(HumanMessage(
                content=f"Generate a strategic execution plan (TODO list) for: '{prompt}'"
            ))

            plan_response = await llm_inst.ainvoke(planning_msgs)
            if getattr(plan_response, "usage_metadata", None):
                total_input_tokens  += plan_response.usage_metadata.get("input_tokens", 0)
                total_output_tokens += plan_response.usage_metadata.get("output_tokens", 0)

            plan_text = plan_response.content if hasattr(plan_response, "content") else str(plan_response)
            yield f"event: thought\ndata: {json.dumps({'text': f'📋 **Strategic Plan**:\\n{plan_text}'})}\\n\\n"
        except Exception as plan_err:
            logger.error(f"[PLANNER ERROR] {plan_err}")
            plan_text = "Step 1: Directly process request using available tools."

    # ── 5. Build system prompt ─────────────────────────────────────────────────
    base_prompt = build_system_prompt(state, active_groups=tool_groups)
    if needs_plan:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"### STRATEGIC PLAN TO FOLLOW:\n{plan_text}\n\n"
            "Use your tools as dictated by the plan. When scraping websites, extract details "
            "via LLM context—do not use regex. Enrich CRM with discovered data."
        )
    elif needs_tools:
        system_prompt = (
            f"{base_prompt}\n\n"
            "Respond directly or execute required tools immediately. "
            "If tools are needed, call them. Otherwise give a clear direct response."
        )
    else:
        system_prompt = (
            "You are a friendly, helpful AI productivity assistant. "
            "Respond to the user's greeting or chat in a brief and pleasant manner."
        )

    # ── 6. Execute agent with LangGraph MemorySaver ───────────────────────────
    logger.info(f"[AGENT] Streaming with thread_id='{thread_id}', tools={len(tools)}")
    if needs_plan:
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Executing plan with dynamic tools...'})}\\n\\n"

    thread_config = {"configurable": {"thread_id": thread_id}}

    try:
        if tools:
            # MemorySaver passed as checkpointer — LangGraph handles history per thread_id
            agent = create_react_agent(
                llm_inst,
                tools,
                prompt=system_prompt,
                checkpointer=checkpointer,
            )
            async for chunk in agent.astream({"messages": msgs}, config=thread_config):
                if "agent" in chunk:
                    for msg in chunk["agent"]["messages"]:
                        if getattr(msg, "usage_metadata", None):
                            total_input_tokens  += msg.usage_metadata.get("input_tokens", 0)
                            total_output_tokens += msg.usage_metadata.get("output_tokens", 0)

                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                logger.info(f"[AGENT THOUGHT] Tool call: {tc['name']}")
                                yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'args': tc['args']})}\\n\\n"

                        if hasattr(msg, "content") and msg.content:
                            yield f"event: content\ndata: {json.dumps({'text': msg.content})}\\n\\n"

                elif "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        logger.info(f"[TOOL RESPONSE] {msg.name}")
                        yield f"event: tool_output\ndata: {json.dumps({'name': msg.name, 'content': str(msg.content)})}\\n\\n"
        else:
            # Conversational path — no tools, no checkpointer needed
            logger.info("[CHAT] No tools. Streaming direct LLM response...")
            full_msgs = [SystemMessage(content=system_prompt)] + msgs
            total_input_tokens += sum(count_tokens(m.content) for m in full_msgs)

            full_response = ""
            async for token_chunk in llm_inst.astream(full_msgs):
                if token_chunk.content:
                    full_response += token_chunk.content
                    yield f"event: content\ndata: {json.dumps({'text': token_chunk.content})}\\n\\n"
            total_output_tokens += count_tokens(full_response)

        yield f"event: token_usage\ndata: {json.dumps({'input_tokens': total_input_tokens, 'output_tokens': total_output_tokens})}\\n\\n"
        yield f"event: thought\ndata: {json.dumps({'text': '✅ Completed successfully'})}\\n\\n"

    except Exception as e:
        logger.error(f"[CHAT ERROR] {e}", exc_info=True)
        from routes.state_routes import unpack_exception
        clean_msg = unpack_exception(e)
        yield f"event: error\ndata: {json.dumps({'message': clean_msg})}\\n\\n"
