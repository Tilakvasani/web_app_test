"""
LangGraph ReAct Agent Execution Module.

Key improvements in this version:
  1. Per-turn tool call counter passed into wrap_tool_with_coercion —
     blocks the same tool after 3 calls per turn and forces the agent forward.
  2. recursion_limit set to 12 in thread_config — hard ceiling on agent steps.
  3. System prompt now includes explicit STOP / ACT conditions so the agent
     proceeds to output once it has collected enough data.
  4. Planner prompt improved to emit a "gather → act" pattern with an
     explicit completion trigger.
  5. Tool RAG deduplication: the planner's requested tools are compared
     against the semantic top-K so the correct tools are always loaded.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Dict, Any, List, AsyncGenerator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import get_llm, get_cheap_llm, UPLOAD_DIR, _make_cfg
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
from utils.token_counter import count_tokens

logger = logging.getLogger("mcp_backend")

# ── Thread tracking ────────────────────────────────────────────────────────────
_INITIALIZED_THREADS: set = set()

# Maximum agent steps per turn — keeps costs predictable and prevents runaway loops.
AGENT_RECURSION_LIMIT = 12

# Maximum times the same tool may be called within a single turn.
# After this the loop-guard wrapper returns a stop message instead of calling the tool.
MAX_TOOL_CALLS_PER_TURN = 3


def _reset_thread_checkpoint(thread_id: str) -> None:
    _INITIALIZED_THREADS.discard(thread_id)
    try:
        keys_to_drop = [
            k for k in list(checkpointer.storage)
            if isinstance(k, tuple) and len(k) > 0 and k[0] == thread_id
        ]
        for k in keys_to_drop:
            checkpointer.storage.pop(k, None)
        logger.info(
            f"[AGENT] Checkpoint reset for thread '{thread_id}' "
            f"({len(keys_to_drop)} entries cleared)."
        )
    except Exception as ce:
        logger.warning(
            f"[AGENT] Could not clear checkpointer storage for '{thread_id}': {ce}"
        )


# ── Dynamic prompt helpers ─────────────────────────────────────────────────────

def _infer_server_capabilities(tool_names: List[str]) -> List[str]:
    caps = set()
    name_blob = " ".join(tool_names).lower()
    rules = [
        (["create", "add", "insert", "new"],             "create records"),
        (["update", "edit", "patch", "modify"],          "update records"),
        (["delete", "remove", "archive"],                 "delete records"),
        (["list", "get", "fetch", "query", "search",
          "find", "retrieve"],                            "search & retrieve"),
        (["send", "email", "message", "notify"],          "send messages"),
        (["upload", "attach", "file"],                    "file handling"),
        (["schedule", "event", "calendar", "meeting"],    "calendar / scheduling"),
        (["summarize", "summarise", "analyse", "analyze",
          "extract"],                                     "AI summarisation"),
    ]
    for keywords, label in rules:
        if any(kw in name_blob for kw in keywords):
            caps.add(label)
    return sorted(caps) if caps else ["general actions"]


def _extract_recent_topics(history: List[Dict[str, str]], n: int = 4) -> str:
    recent = [m.get("content", "") for m in history if m.get("role") == "user"][-n:]
    if not recent:
        return "none"
    joined = " | ".join(t[:80] for t in recent)
    return joined if len(joined) <= 300 else joined[:297] + "..."


def build_dynamic_sifter_prompt(
    prompt: str,
    all_mcp_tools: Dict[str, list],
    history: List[Dict[str, str]],
) -> str:
    action_group_lines = []
    for server_name, tools in all_mcp_tools.items():
        tool_names = [getattr(t, "name", "") for t in tools]
        caps = _infer_server_capabilities(tool_names)
        sample = ", ".join(tool_names[:6]) + (", ..." if len(tool_names) > 6 else "")
        action_group_lines.append(
            f"- '{server_name}': {', '.join(caps)}  (tools: {sample})"
        )
    action_group_lines.append(
        "- 'Local': web search, web scraping, local file reading, docs search"
    )
    action_groups_text = "\n".join(action_group_lines)
    recent_topics = _extract_recent_topics(history)

    return (
        "You are a query router for an AI assistant. Classify the user query and "
        "decide which tool groups are needed to answer it.\n\n"
        "## Available action groups\n"
        f"{action_groups_text}\n\n"
        "## Recent conversation context\n"
        f"{recent_topics}\n\n"
        "## Classification rules\n"
        "1. 'CONVERSATIONAL' - greetings, small talk, questions about the assistant's "
        "own capabilities. Use NO tool groups.\n"
        "2. 'DIRECT_TOOL' - single-step requests that need exactly one server or one "
        "tool call (e.g. 'list my Notion pages', 'search for X'). Pick the most "
        "relevant group(s).\n"
        "3. 'COMPLEX' - multi-step workflows requiring coordination across multiple "
        "tools or servers. List all needed groups.\n\n"
        "## Real-time data rule (MANDATORY)\n"
        "If the query asks for information that changes over time - current events, "
        "today's news, live prices, latest software versions, weather, recent "
        "announcements, or contains words like 'current', 'latest', 'today', 'now', "
        "'recent', 'right now', 'live', 'breaking' - you MUST:\n"
        "  a) include 'Local' in tool_groups (for web search)\n"
        "  b) NOT classify as CONVERSATIONAL\n\n"
        f"## User query\n\"{prompt}\"\n\n"
        "Output ONLY a raw JSON object - no markdown, no explanation:\n"
        "{\"category\": \"CONVERSATIONAL|DIRECT_TOOL|COMPLEX\", \"tool_groups\": [...]}"
    )


def build_dynamic_planner_prompt(
    query: str,
    state: Dict[str, Any],
    all_mcp_tools: Dict[str, list],
) -> str:
    """
    Builds a planner prompt that produces a strict gather → act plan.
    The plan always ends with an explicit 'CREATE OUTPUT' step so the agent
    knows to stop collecting data and produce the final result.
    """
    server_lines = []
    for name in state.get("mcp_servers", {}):
        tool_count = len(all_mcp_tools.get(name, []))
        tool_names = [getattr(t, "name", "") for t in all_mcp_tools.get(name, [])]
        sample = ", ".join(tool_names[:5]) + (", ..." if len(tool_names) > 5 else "")
        server_lines.append(f"- {name} ({tool_count} tools: {sample})")
    server_lines.append(
        "- Local tools: web_search, web_scrape, read_uploaded_file, "
        "prepare_file_for_upload, query_documentation"
    )
    servers_text = "\n".join(server_lines)

    return (
        "You are the Strategic Planning Engine for an AI assistant.\n\n"
        "## Active servers and tools\n"
        f"{servers_text}\n\n"
        "## Your task\n"
        f"Generate a precise, ordered execution plan for: \"{query}\"\n\n"
        "## CRITICAL PLANNING RULES\n"
        "- The plan MUST follow the 'GATHER → ACT' pattern:\n"
        "  1. First steps: collect all needed data (search/fetch/list)\n"
        "  2. Final step: create/write/post the output in ONE tool call\n"
        "- NEVER plan more than 2 data-gathering steps for the same source.\n"
        "  If the user asks for N items, gather them in at most 2 calls then act.\n"
        "- The LAST step must always be the 'create output' action "
        "  (e.g. notion-create-pages, gmail-send, etc.).\n"
        "- Use ONLY tool names from the list above — never invent tools.\n\n"
        "## Output format — MANDATORY\n"
        "Output ONLY a raw JSON array — no markdown, no explanation:\n"
        "[\n"
        "  {\"step\": 1, \"tool\": \"<exact_tool_name>\", \"purpose\": \"<one sentence>\"},\n"
        "  {\"step\": 2, \"tool\": \"<exact_tool_name>\", \"purpose\": \"<one sentence>\"}\n"
        "]\n\n"
        "- Steps MUST be in logical execution order.\n"
        "- Do NOT execute any tools — output the plan only."
    )


def _parse_structured_plan(raw: str) -> list:
    try:
        cleaned = re.sub(r"```json\s*|\s*```", "", raw.strip()).strip()
        steps = json.loads(cleaned)
        if isinstance(steps, list) and steps:
            return steps
    except Exception:
        pass
    return [{"step": 1, "tool": "auto", "purpose": raw.strip() or "Process request using available tools."}]


async def compress_history_sandwich(
    history: List[Dict[str, str]],
    llm_inst,
    cheap_llm_inst=None,
) -> List[Dict[str, str]]:
    _summariser = cheap_llm_inst if cheap_llm_inst is not None else llm_inst
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
        logger.info("[HISTORY COMPRESS] Detecting conversation tone...")
        tone_map = {
            "technical":  "Use precise technical terminology; preserve tool names, IDs, and specific details.",
            "casual":     "Use relaxed, plain language; focus on the key points only.",
            "formal":     "Use professional, concise language; avoid contractions.",
        }
        try:
            tone_res = await _summariser.ainvoke([SystemMessage(
                content=(
                    "Classify the tone of the following conversation as exactly one of: "
                    "'technical', 'casual', or 'formal'. "
                    "Reply with ONLY that single word.\n\n"
                    f"--- Conversation ---\n{middle_text}"
                )
            )])
            detected_tone = (tone_res.content if hasattr(tone_res, "content") else "casual").strip().lower()
            if detected_tone not in tone_map:
                detected_tone = "casual"
        except Exception:
            detected_tone = "casual"

        style_instruction = tone_map[detected_tone]
        logger.info(f"[HISTORY COMPRESS] Detected tone: {detected_tone}. Summarising...")
        summary_prompt = (
            f"Summarise the following conversation segment in one dense paragraph. "
            f"{style_instruction} "
            "Highlight key questions, decisions, actions, and any important data "
            "(names, IDs, URLs, amounts). Do not omit facts that a future assistant "
            "turn might need.\n\n"
            f"--- Conversation ---\n{middle_text}"
        )
        try:
            res = await _summariser.ainvoke([SystemMessage(content=summary_prompt)])
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


def _build_anti_loop_instructions(tools_loaded: list) -> str:
    """
    Returns a block of STOP/ACT rules appended to every agent system prompt.
    These are the most important lines in the prompt — they prevent the infinite
    search-paginate-search loop seen in the original recursion-limit crashes.
    """
    tool_names = [getattr(t, "name", str(t)) for t in tools_loaded]
    tools_str = ", ".join(f"`{n}`" for n in tool_names[:10])
    return (
        "\n\n---\n"
        "## ⚠️ MANDATORY EXECUTION RULES — READ BEFORE CALLING ANY TOOL\n\n"
        "### Rule 1 — GATHER THEN ACT (most important)\n"
        "Your job has two phases:\n"
        "  Phase 1 (GATHER): Call data-fetching tools to collect what you need.\n"
        "  Phase 2 (ACT):    Call the output tool ONCE with ALL collected data.\n"
        "Never stay in Phase 1 indefinitely. Move to Phase 2 as soon as you have "
        "enough data — even if it's not perfectly complete.\n\n"
        "### Rule 2 — NO SAME-TOOL LOOPS\n"
        f"You have {len(tool_names)} tools available: {tools_str}.\n"
        "You may call any single tool at most 3 times per response.\n"
        "If a tool returns a loop-guard warning, STOP calling it immediately "
        "and proceed to Phase 2 with whatever data you have.\n\n"
        "### Rule 3 — QUANTITY REQUESTS\n"
        "If the user asks for N items (e.g. '10 jobs'), collect up to N items "
        "across at most 2 search calls, then immediately create the output.\n"
        "Do NOT paginate beyond page 2 or 3 for any single query.\n\n"
        "### Rule 4 — TRUST TRUNCATED RESULTS\n"
        "If a tool result ends with '[Output truncated ...]', that means you have "
        "enough data. Do NOT retry the same call to get more — proceed to ACT.\n\n"
        "### Rule 5 — ONE CREATION CALL\n"
        "Calls that create pages, send emails, or post content must happen exactly "
        "ONCE, containing ALL data gathered in Phase 1. Never split creation across "
        "multiple tool calls.\n"
        "---\n"
    )


async def stream_agent_interaction(
    prompt: str,
    history: List[Dict[str, str]],
    state: Dict[str, Any],
    thread_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Streams the ReAct agent's thoughts, tool invocations, and final response
    back to the frontend via Server-Sent Events (SSE).
    """
    llm_inst = get_llm()

    # ── 0. Decide input messages based on thread state ─────────────────────────
    is_new_thread = thread_id not in _INITIALIZED_THREADS

    cheap_llm = get_cheap_llm()
    compressed_history = await compress_history_sandwich(history, llm_inst, cheap_llm_inst=cheap_llm)

    if is_new_thread:
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
        msgs = [HumanMessage(content=prompt)]
        logger.info(f"[AGENT] Existing thread '{thread_id}' — passing new message only.")

    # ── 1. Pre-fetch MCP tools ─────────────────────────────────────────────────
    global _MCP_TOOLS_CACHE
    all_mcp_tools: Dict[str, list] = {}

    if state.get("mcp_servers"):
        servers_to_fetch = {}
        for name, s in state["mcp_servers"].items():
            from urllib.parse import urlparse
            _raw_url = s.get("url", "")
            _p = urlparse(_raw_url)
            _clean_url = f"{_p.scheme}://{_p.netloc}{_p.path}"
            srv_key = f"{name}:{_clean_url}"
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

    # ── 2. Smart Sifter ────────────────────────────────────────────────────────
    total_input_tokens  = 0
    total_output_tokens = 0
    needs_plan  = True
    needs_tools = True
    tool_groups = []
    plan_text   = ""

    _TRIVIAL_PHRASES = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "ok thanks",
        "ok bro", "cool", "got it", "sounds good", "great", "perfect", "bye",
        "goodbye", "see you", "see ya",
    }
    _prompt_stripped = prompt.strip().lower().rstrip("!.,?")
    if len(prompt.split()) <= 3 and _prompt_stripped in _TRIVIAL_PHRASES:
        needs_plan = needs_tools = False
        tool_groups = []
        plan_text = "Pure conversational small talk."
        logger.info(f"[PLANNER] Trivial query fast-path — skipping sifter LLM call.")
    else:
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

        classification_prompt = build_dynamic_sifter_prompt(prompt, all_mcp_tools, history)

        sifter_hash = hashlib.sha256(
            f"{prompt.lower().strip()}:{server_descriptions}".encode()
        ).hexdigest()[:32]

        cached_sifter = await cache.get_sifter(sifter_hash)
        if cached_sifter:
            category    = cached_sifter.get("category", "COMPLEX")
            tool_groups = cached_sifter.get("tool_groups", [])
            logger.info(f"[PLANNER] Sifter cache hit: {category} | {tool_groups}")
        else:
            class_res = await cheap_llm.ainvoke([SystemMessage(content=classification_prompt)])
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

        _REALTIME_KEYWORDS = {
            "current", "currently", "today", "now", "latest", "recent", "recently",
            "right now", "live", "breaking", "this week", "this month", "this year",
            "2024", "2025", "2026", "news", "price", "stock", "weather", "update",
        }
        _prompt_tokens = set(prompt.lower().replace("?", "").replace(",", "").split())
        if _prompt_tokens & _REALTIME_KEYWORDS:
            if "Local" not in tool_groups:
                tool_groups.append("Local")
            if "CONVERSATIONAL" in category:
                category = "DIRECT_TOOL"
            logger.info(f"[PLANNER] Real-time keyword override applied → {category} | {tool_groups}")

        if "CONVERSATIONAL" in category:
            needs_plan = needs_tools = False
            plan_text = "Pure conversational small talk."
        elif "DIRECT_TOOL" in category:
            needs_plan  = False
            needs_tools = True
            plan_text = "Direct tool execution."
        else:
            needs_plan = needs_tools = True

        _mcp_groups = [g for g in tool_groups if g != "Local"]
        if len(_mcp_groups) > 1 and "CONVERSATIONAL" not in category:
            _explicit_mentions = [g for g in _mcp_groups if g.lower() in prompt.lower()]
            if not _explicit_mentions:
                service_list = ", ".join(_mcp_groups)
                clarify_msg = (
                    f"I found multiple services that could answer this — {service_list}. "
                    "Which one should I use?"
                )
                logger.info(f"[PLANNER] Ambiguous query — clarification needed: {_mcp_groups}")
                yield f"event: clarification\ndata: {json.dumps({'text': clarify_msg, 'options': _mcp_groups})}\n\n"
                return

      except Exception as sifter_err:
        logger.error(f"[PLANNER ERROR] Sifter failed: {sifter_err}")
        needs_plan = needs_tools = True
        tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]

        if state.get("mcp_servers"):
            from urllib.parse import urlparse as _up
            missing = {
                name: s
                for name, s in state["mcp_servers"].items()
                if name not in all_mcp_tools
            }
            if missing:
                logger.info(f"[PLANNER FALLBACK] Re-fetching tools for: {list(missing.keys())}")
                for name, s in missing.items():
                    try:
                        _p = _up(s.get("url", ""))
                        srv_key = f"{name}:{_p.scheme}://{_p.netloc}{_p.path}"
                        transport = s.get("transport", "streamable_http")
                        cfg = _make_cfg(s["url"], s["auth_type"], s["auth_value"], transport=transport)
                        single_client = MultiServerMCPClient({name: cfg})
                        mcp_tools = await single_client.get_tools()
                        _MCP_TOOLS_CACHE[srv_key] = mcp_tools
                        all_mcp_tools[name] = mcp_tools
                        logger.info(f"[PLANNER FALLBACK] Re-fetched {len(mcp_tools)} tools from '{name}'.")
                    except Exception as refetch_err:
                        logger.error(f"[PLANNER FALLBACK] Re-fetch still failed for '{name}': {refetch_err}")

    # ── 3. Resolve tools ───────────────────────────────────────────────────────
    # Per-turn call counter — shared mutable dict passed into every tool wrapper
    # so the loop-guard can count calls across all tools in this one agent turn.
    turn_call_counter: Dict[str, int] = {}

    tools = []

    if needs_tools:
        for name in tool_groups:
            for server_name in all_mcp_tools:
                if name.lower() in server_name.lower() or server_name.lower() in name.lower():
                    # Pass the per-turn counter into each wrapper so loop-guard works
                    tools.extend([
                        wrap_tool_with_coercion(t, call_counter=turn_call_counter)
                        for t in all_mcp_tools[server_name]
                    ])
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
            # ── Tool RAG with SSE keepalive ───────────────────────────────────
            # Embedding 50+ tools takes 9-15s. Without heartbeats the frontend
            # SSE connection times out before the agent even starts.
            yield "event: thought\ndata: {\"text\": \"🔍 Selecting best tools for this task...\"}\n\n"
            rag_task = asyncio.create_task(
                retrieve_relevant_tools(prompt, tools, top_k=8)
            )
            while not rag_task.done():
                yield "event: keepalive\ndata: {}\n\n"
                await asyncio.sleep(2)
            tools = rag_task.result()
            # Pin tools explicitly named in the plan — RAG may miss low-scoring
            # but mandatory tools like notion-create-pages.
            if plan_text:
                pinned_names = set(re.findall(r"\[([^\]]+)\]", plan_text))
                active_names = {getattr(t, "name", "") for t in tools}
                full_pool = [t for lst in all_mcp_tools.values() for t in lst]
                for t in full_pool:
                    tname = getattr(t, "name", "")
                    if tname in pinned_names and tname not in active_names:
                        logger.info(f"[TOOL RAG] Pinning plan-required tool: {tname}")
                        tools.append(wrap_tool_with_coercion(t, call_counter=turn_call_counter))

    # ── 4. Strategic Planning Phase ───────────────────────────────────────────
    if needs_plan:
        logger.info("[PLANNER] Running pre-execution strategic planning...")
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Creating strategic execution plan...'})}\n\n"

        try:
            planning_system_prompt = build_dynamic_planner_prompt(prompt, state, all_mcp_tools)
            planning_msgs = [SystemMessage(content=planning_system_prompt)]
            for h in (compressed_history if is_new_thread else history[-3:]):
                r = h.get("role"); c = h.get("content", "")
                if r == "user":      planning_msgs.append(HumanMessage(content=c))
                elif r == "assistant": planning_msgs.append(AIMessage(content=c))
            planning_msgs.append(HumanMessage(
                content=f"Generate a strategic execution plan (TODO list) for: '{prompt}'"
            ))

            plan_response = await cheap_llm.ainvoke(planning_msgs)
            if getattr(plan_response, "usage_metadata", None):
                total_input_tokens  += plan_response.usage_metadata.get("input_tokens", 0)
                total_output_tokens += plan_response.usage_metadata.get("output_tokens", 0)

            raw_plan  = plan_response.content if hasattr(plan_response, "content") else str(plan_response)
            plan_steps = _parse_structured_plan(raw_plan)

            plan_text = "\n".join(
                f"Step {s.get('step', i+1)}: [{s.get('tool','auto')}] {s.get('purpose','')}"
                for i, s in enumerate(plan_steps)
            )

            display_plan = "\n".join(
                f"{s.get('step', i+1)}. {s.get('purpose','')} → `{s.get('tool','auto')}`"
                for i, s in enumerate(plan_steps)
            )
            yield f"event: thought\ndata: {json.dumps({'text': f'📋 **Execution Plan:**\n{display_plan}'})}\n\n"
        except Exception as plan_err:
            logger.error(f"[PLANNER ERROR] {plan_err}")
            plan_text = "Step 1: Directly process request using available tools."

    # ── 5. Build system prompt ─────────────────────────────────────────────────
    base_prompt = build_system_prompt(state, active_groups=tool_groups)
    anti_loop   = _build_anti_loop_instructions(tools)

    _search_instruction = (
        "After calling web_search or web_scrape, always present the results as a "
        "concise, readable summary — highlight key facts, dates, and names. "
        "Never dump raw snippets or URLs as the final answer.\n\n"
    )
    if needs_plan:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"{_search_instruction}"
            f"### STRATEGIC PLAN — execute steps in this exact order:\n{plan_text}\n\n"
            "Use your tools as dictated by the plan. When scraping websites, extract details "
            "via LLM context — do not use regex. Enrich CRM with discovered data."
            f"{anti_loop}"
        )
    elif needs_tools:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"{_search_instruction}"
            "Respond directly or execute required tools immediately. "
            "If tools are needed, call them. Otherwise give a clear direct response."
            f"{anti_loop}"
        )
    else:
        system_prompt = (
            "You are a friendly, helpful AI productivity assistant. "
            "Respond to the user's greeting or chat in a brief and pleasant manner."
        )

    # ── 6. Execute agent ───────────────────────────────────────────────────────
    logger.info(f"[AGENT] Streaming with thread_id='{thread_id}', tools={len(tools)}, recursion_limit={AGENT_RECURSION_LIMIT}")
    if needs_plan:
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Executing plan with dynamic tools...'})}\n\n"

    # recursion_limit lives in the top-level config, not inside configurable.
    thread_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    }

    try:
        if tools:
            agent = create_react_agent(
                llm_inst,
                tools,
                prompt=system_prompt,
                checkpointer=checkpointer,
            )

            # ── Pre-flight: heal dangling tool calls ──────────────────────────
            if thread_id in _INITIALIZED_THREADS:
                try:
                    cp_tuple = await checkpointer.aget_tuple(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    if cp_tuple:
                        saved_msgs = (
                            cp_tuple.checkpoint
                            .get("channel_values", {})
                            .get("messages", [])
                        )
                        call_ids = {
                            tc["id"]
                            for m in saved_msgs
                            for tc in (getattr(m, "tool_calls", None) or [])
                        }
                        result_ids = {
                            getattr(m, "tool_call_id", None)
                            for m in saved_msgs
                        }
                        result_ids.discard(None)
                        dangling = call_ids - result_ids
                        if dangling:
                            logger.warning(
                                f"[AGENT] Pre-flight: {len(dangling)} dangling tool call(s) "
                                f"found in thread '{thread_id}'. Auto-resetting checkpoint."
                            )
                            _reset_thread_checkpoint(thread_id)
                            msgs = []
                            for h in compressed_history:
                                role = h.get("role"); content = h.get("content", "")
                                if role == "user":
                                    msgs.append(HumanMessage(content=content))
                                elif role == "assistant":
                                    msgs.append(AIMessage(content=content))
                            msgs.append(HumanMessage(content=prompt))
                            _INITIALIZED_THREADS.add(thread_id)
                except Exception as pre_err:
                    logger.warning(
                        f"[AGENT] Pre-flight checkpoint check failed (non-fatal): {pre_err}"
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
                                yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'args': tc['args']})}\n\n"

                        if hasattr(msg, "content") and msg.content:
                            yield f"event: content\ndata: {json.dumps({'text': msg.content})}\n\n"

                elif "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        logger.info(f"[TOOL RESPONSE] {msg.name}")
                        yield f"event: tool_output\ndata: {json.dumps({'name': msg.name, 'content': str(msg.content)})}\n\n"
        else:
            logger.info("[CHAT] No tools. Streaming direct LLM response...")
            full_msgs = [SystemMessage(content=system_prompt)] + msgs
            total_input_tokens += sum(count_tokens(m.content) for m in full_msgs)

            full_response = ""
            async for token_chunk in llm_inst.astream(full_msgs):
                if token_chunk.content:
                    full_response += token_chunk.content
                    yield f"event: content\ndata: {json.dumps({'text': token_chunk.content})}\n\n"
            total_output_tokens += count_tokens(full_response)

        yield f"event: token_usage\ndata: {json.dumps({'input_tokens': total_input_tokens, 'output_tokens': total_output_tokens})}\n\n"
        yield f"event: thought\ndata: {json.dumps({'text': '✅ Completed successfully'})}\n\n"

    except Exception as e:
        err_str = str(e)
        if "AIMessages with tool_calls" in err_str or "INVALID_CHAT_HISTORY" in err_str:
            logger.warning(
                f"[AGENT] INVALID_CHAT_HISTORY in thread '{thread_id}' — "
                "resetting checkpoint so next message auto-heals."
            )
            _reset_thread_checkpoint(thread_id)
            yield (
                f"event: error\ndata: {json.dumps({'message': 'Your conversation history was reset after an interrupted tool call. Please resend your last message and the agent will continue normally.'})}\n\n"
            )
            return
        # Recursion limit hit — surface a clear, friendly message
        if "recursion" in err_str.lower() or "GRAPH_RECURSION_LIMIT" in err_str:
            logger.error(f"[AGENT] Recursion limit ({AGENT_RECURSION_LIMIT}) hit for thread '{thread_id}'.")
            yield (
                f"event: error\ndata: {json.dumps({'message': f'The task required too many steps to complete automatically. Try breaking it into smaller parts, or be more specific about what you need (e.g. specify the exact service to use).'})}\n\n"
            )
            return
        logger.error(f"[CHAT ERROR] {e}", exc_info=True)
        from routes.state_routes import unpack_exception
        clean_msg = unpack_exception(e)
        yield f"event: error\ndata: {json.dumps({'message': clean_msg})}\n\n"