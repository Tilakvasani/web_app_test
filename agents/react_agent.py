"""
LangGraph ReAct Agent — Fixed & Enhanced.

Root-cause fixes applied vs previous version
─────────────────────────────────────────────
1. LOOP BUG: Agent spent all 12 recursion steps on wrong tools (notion-search ×3,
   get_companies ×2) and never reached notion-create-pages.
   Fix → task_type detection in Sifter; creation tools ALWAYS pinned; search tools
   excluded from tool list when task_type == "CREATE".

2. RECURSION: Limit raised 12 → 18 so even a misbehaving plan has budget to finish.

3. PLAN PINNING: Normalised tool-name comparison (strip hyphens/underscores/case) so
   "notion-create-pages" in the plan matches the actual MCP tool regardless of casing.

4. PROMPTS: Sifter, Planner, Anti-loop, and System prompts all rewritten for clarity
   and to produce formatted, readable output (not raw JSON dumps).

5. MAX_TOOL_CALLS_PER_TURN lowered 3 → 2 in the loop-guard so the agent hits the
   guardrail faster and still has budget left to call the creation tool.
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
    web_scrape,
)
from utils.token_counter import count_tokens

logger = logging.getLogger("mcp_backend")

# ── Thread tracking ────────────────────────────────────────────────────────────
_INITIALIZED_THREADS: set = set()

# Raised from 12 → 18 so even multi-step plans have room to finish.
AGENT_RECURSION_LIMIT = 18

# Lowered from 3 → 2: hit the guardrail faster, preserve budget for creation step.
MAX_TOOL_CALLS_PER_TURN = 2


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
        logger.warning(f"[AGENT] Could not clear checkpointer for '{thread_id}': {ce}")


# ── Tool-name normalisation helper ─────────────────────────────────────────────
def _norm(name: str) -> str:
    """Lower-case + strip hyphens/underscores/spaces for fuzzy tool name matching."""
    return re.sub(r"[-_\s]", "", name.lower())


# ── Dynamic prompt helpers ─────────────────────────────────────────────────────

def _infer_server_capabilities(tool_names: List[str]) -> List[str]:
    caps = set()
    name_blob = " ".join(tool_names).lower()
    rules = [
        (["create", "add", "insert", "new"],             "create records"),
        (["update", "edit", "patch", "modify"],           "update records"),
        (["delete", "remove", "archive"],                  "delete records"),
        (["list", "get", "fetch", "query", "search",
          "find", "retrieve"],                             "search & retrieve"),
        (["send", "email", "message", "notify"],           "send messages"),
        (["upload", "attach", "file"],                     "file handling"),
        (["schedule", "event", "calendar", "meeting"],     "calendar / scheduling"),
        (["summarize", "summarise", "analyse", "analyze",
          "extract"],                                      "AI summarisation"),
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
    """
    Classifies the query and identifies which tool groups + task_type are needed.
    task_type is the key addition: CREATE vs SEARCH vs MIXED drives tool filtering
    so the agent never wastes steps calling search tools on a creation task.
    """
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
        "You are a query router for an AI assistant. Your output is a JSON object "
        "that tells the agent which tool groups and task type are needed.\n\n"
        "## Available action groups\n"
        f"{action_groups_text}\n\n"
        "## Recent conversation context\n"
        f"{recent_topics}\n\n"
        "## Classification rules\n"
        "1. 'CONVERSATIONAL' — greetings, small talk, capability questions. NO tool groups.\n"
        "2. 'DIRECT_TOOL' — single-step requests needing exactly one server/tool call.\n"
        "3. 'COMPLEX' — multi-step workflows across multiple tools or servers.\n\n"
        "## Task-type classification (MANDATORY)\n"
        "Set 'task_type' to ONE of:\n"
        "- 'CREATE' — the primary goal is to CREATE, WRITE, POST, or ADD something "
        "(e.g. 'make a Notion page', 'send an email', 'add a CRM record').\n"
        "- 'SEARCH' — the primary goal is to READ, FIND, LIST, or RETRIEVE existing data.\n"
        "- 'MIXED' — the task requires both gathering data AND creating output "
        "(e.g. 'find jobs and create a Notion page' → MIXED, tool groups include both "
        "the data source and the creation target).\n\n"
        "IMPORTANT: When task_type is 'CREATE' or 'MIXED', do NOT include search-only "
        "tools from the creation server in tool_groups. The agent should call the "
        "creation tool directly, not pre-search the destination.\n\n"
        "## Real-time data rule (MANDATORY)\n"
        "If the query asks for current info (events, prices, news, 'latest', 'today', "
        "'now', 'recent', 'live', 'breaking'), you MUST include 'Local' in tool_groups "
        "and NOT classify as CONVERSATIONAL.\n\n"
        f"## User query\n\"{prompt}\"\n\n"
        "Output ONLY a raw JSON object — no markdown, no explanation:\n"
        "{\"category\": \"CONVERSATIONAL|DIRECT_TOOL|COMPLEX\", "
        "\"task_type\": \"CREATE|SEARCH|MIXED\", \"tool_groups\": [...]}"
    )


def build_dynamic_planner_prompt(
    query: str,
    state: Dict[str, Any],
    all_mcp_tools: Dict[str, list],
) -> str:
    """
    Generates a strict GATHER → ACT plan.
    Key changes:
    - Explicitly lists creation tools separately so the planner always picks the right one.
    - Mandates that the LAST step is a creation/write tool.
    - Forbids planning more than 2 data-gather steps for the same source.
    """
    server_lines = []
    creation_tools = []
    for name in state.get("mcp_servers", {}):
        tool_list = all_mcp_tools.get(name, [])
        tool_count = len(tool_list)
        tool_names = [getattr(t, "name", "") for t in tool_list]
        sample = ", ".join(tool_names[:5]) + (", ..." if len(tool_names) > 5 else "")
        server_lines.append(f"- {name} ({tool_count} tools: {sample})")
        # Identify creation tools for this server
        for tn in tool_names:
            if any(kw in tn.lower() for kw in ["create", "add", "insert", "send", "post", "write", "new"]):
                creation_tools.append(f"{tn} (from {name})")
    server_lines.append(
        "- Local tools: web_search, web_scrape, read_uploaded_file, "
        "prepare_file_for_upload, query_documentation"
    )
    servers_text = "\n".join(server_lines)
    creation_text = "\n".join(f"  • {c}" for c in creation_tools) if creation_tools else "  • (none identified)"

    return (
        "You are the Strategic Planning Engine for an AI assistant.\n\n"
        "## Active servers and tools\n"
        f"{servers_text}\n\n"
        "## Creation tools available (use one as the FINAL step)\n"
        f"{creation_text}\n\n"
        "## Your task\n"
        f"Generate a precise, ordered execution plan for: \"{query}\"\n\n"
        "## CRITICAL PLANNING RULES\n"
        "1. GATHER → ACT pattern (non-negotiable):\n"
        "   - Steps 1–N: collect data (search/fetch/list) — at most 2 steps per data source.\n"
        "   - Final step: call ONE creation/write tool with ALL collected data.\n"
        "2. The LAST step MUST be a creation tool (create-pages, send, add-record, etc.).\n"
        "   NEVER end the plan with a search or list tool.\n"
        "3. If the task says 'N items' (e.g. '10 companies'), gather them in ≤2 calls then act.\n"
        "   ALL N items must go into the single creation call — never plan partial creation.\n"
        "4. Do NOT plan a 'search destination before creating' step — e.g. for 'create a "
        "   Notion page', do NOT include notion-search in the plan.\n"
        "5. Use ONLY exact tool names from the lists above — never invent tools.\n"
        "6. The plan must be self-contained and completable in one agent turn. "
        "   Do NOT design a plan that requires asking the user for permission to continue.\n\n"
        "## Output format — MANDATORY\n"
        "Output ONLY a raw JSON array — no markdown, no explanation:\n"
        "[\n"
        "  {\"step\": 1, \"tool\": \"<exact_tool_name>\", \"purpose\": \"<one sentence>\"},\n"
        "  {\"step\": 2, \"tool\": \"<exact_tool_name>\", \"purpose\": \"<one sentence>\"}\n"
        "]\n\n"
        "Steps MUST be in logical execution order. Do NOT execute any tools — output the plan only."
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
            "technical": "Use precise technical terminology; preserve tool names, IDs, and specific details.",
            "casual":    "Use relaxed, plain language; focus on the key points only.",
            "formal":    "Use professional, concise language; avoid contractions.",
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


# ── MCP tool cache ─────────────────────────────────────────────────────────────
_MCP_TOOLS_CACHE: Dict[str, list] = {}


def _build_anti_loop_instructions(tools_loaded: list, plan_steps: list = None) -> str:
    """
    Anti-loop rules appended to every system prompt.
    Key addition: detects creation tools in the plan and injects a HARD RULE
    stating the final call MUST be that tool — preventing the agent from
    spending all its budget on gather steps and never creating.
    """
    tool_names = [getattr(t, "name", str(t)) for t in tools_loaded]
    tools_str  = ", ".join(f"`{n}`" for n in tool_names[:12])

    # Find the creation tool from the plan (last step's tool)
    final_tool_rule = ""
    if plan_steps:
        last_step = plan_steps[-1] if plan_steps else {}
        final_tool = last_step.get("tool", "")
        if final_tool and final_tool != "auto":
            final_tool_rule = (
                f"\n### Rule 6 — MANDATORY FINAL CALL\n"
                f"Your execution plan ends with `{final_tool}`. "
                f"You MUST call `{final_tool}` as your last tool call, "
                f"passing ALL data collected in previous steps. "
                f"If you have called 2 or more gather tools already, "
                f"go directly to `{final_tool}` — do NOT call any more gather tools.\n"
            )

    return (
        "\n\n---\n"
        "## ⚠️ MANDATORY EXECUTION RULES\n\n"
        "### Rule 1 — GATHER THEN ACT\n"
        "Phase 1 (GATHER): Call data-fetching tools to collect what you need.\n"
        "Phase 2 (ACT):    Call the output/creation tool ONCE with ALL collected data.\n"
        "Move to Phase 2 as soon as you have enough data — even if imperfect.\n\n"
        "### Rule 2 — NO SAME-TOOL LOOPS\n"
        f"You have tools: {tools_str}\n"
        f"You may call any single tool at most {MAX_TOOL_CALLS_PER_TURN} times per response.\n"
        "If you see a loop-guard warning, STOP immediately and move to Phase 2.\n\n"
        "### Rule 3 — QUANTITY REQUESTS\n"
        "If the user asks for N items, collect them across at most 2 search calls, "
        "then create the output. Do NOT paginate beyond page 2.\n\n"
        "### Rule 4 — TRUST TRUNCATED RESULTS\n"
        "If a result ends with '[Output truncated ...]', you have enough data. "
        "Do NOT retry — proceed to Phase 2 immediately.\n\n"
        "### Rule 5 — ONE CREATION CALL\n"
        "Calls that create pages, send emails, or post content happen EXACTLY ONCE, "
        "containing ALL gathered data. Never split creation across multiple calls.\n"
        f"{final_tool_rule}"
        "### Rule 7 — NEVER ASK TO CONTINUE\n"
        "NEVER end your response with phrases like 'shall I continue?', 'would you like "
        "me to add more?', 'let me know if you want the rest', or any variation that "
        "implies the task is incomplete. If the user asked for N items, ALL N items must "
        "be included in the single creation call before you reply. Do not stop at 2 and "
        "offer to do the other 8 — do all 10 in one shot.\n"
        "---\n"
    )


# ── Task-type aware tool filter ────────────────────────────────────────────────
_CREATION_KEYWORDS = {"create", "add", "insert", "send", "post", "write", "new", "make"}
_SEARCH_KEYWORDS   = {"search", "list", "get", "fetch", "query", "find", "retrieve"}

def _filter_tools_by_task_type(
    tools: list,
    task_type: str,
    plan_steps: list,
) -> list:
    """
    When task_type is CREATE or MIXED, removes pure-search tools from the same
    server as the creation tool — preventing the agent from calling notion-search
    when its job is to call notion-create-pages.

    Logic:
    - Identify servers that have a creation tool in the plan.
    - For those servers, remove any search-only tools.
    - Always keep tools explicitly named in the plan.
    """
    if task_type == "SEARCH":
        return tools  # no filtering needed for pure search tasks

    # Build set of tool names required by the plan
    plan_tool_names = {_norm(s.get("tool", "")) for s in plan_steps if s.get("tool")}

    # Identify servers whose creation tools are in the plan
    creation_servers: set = set()
    for t in tools:
        tname = getattr(t, "name", "")
        if _norm(tname) in plan_tool_names:
            if any(kw in tname.lower() for kw in _CREATION_KEYWORDS):
                # Extract a rough "server prefix" (e.g. "notion" from "notion-create-pages")
                parts = re.split(r"[-_]", tname.lower())
                if parts:
                    creation_servers.add(parts[0])

    if not creation_servers:
        return tools

    filtered = []
    for t in tools:
        tname  = getattr(t, "name", "")
        tnorm  = _norm(tname)
        tparts = re.split(r"[-_]", tname.lower())
        server_prefix = tparts[0] if tparts else ""

        # Always keep tools in the plan
        if tnorm in plan_tool_names:
            filtered.append(t)
            continue

        # For servers that have a planned creation tool, drop pure-search tools
        if server_prefix in creation_servers:
            is_search_only = (
                any(kw in tname.lower() for kw in _SEARCH_KEYWORDS)
                and not any(kw in tname.lower() for kw in _CREATION_KEYWORDS)
            )
            if is_search_only:
                logger.info(
                    f"[TOOL FILTER] Dropping search-only tool '{tname}' "
                    f"(server '{server_prefix}' has a planned creation tool)."
                )
                continue

        filtered.append(t)

    return filtered


async def stream_agent_interaction(
    prompt: str,
    history: List[Dict[str, str]],
    state: Dict[str, Any],
    thread_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Streams the agent's reasoning and final response via Server-Sent Events (SSE).
    """
    llm_inst   = get_llm()
    cheap_llm  = get_cheap_llm()

    # ── 0. Build message list ──────────────────────────────────────────────────
    is_new_thread = thread_id not in _INITIALIZED_THREADS
    compressed_history = await compress_history_sandwich(history, llm_inst, cheap_llm_inst=cheap_llm)

    if is_new_thread:
        msgs = []
        for h in compressed_history:
            role, content = h.get("role"), h.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        msgs.append(HumanMessage(content=prompt))
        _INITIALIZED_THREADS.add(thread_id)
        logger.info(f"[AGENT] New thread '{thread_id}' — seeding {len(msgs)} messages.")
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
            _p = urlparse(s.get("url", ""))
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
                logger.info(f"[MCP] Config for {name}: headers = {cfg.get('headers', {})}")
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

    # ── 2. Sifter ─────────────────────────────────────────────────────────────
    total_input_tokens  = 0
    total_output_tokens = 0
    needs_plan  = True
    needs_tools = True
    tool_groups = []
    plan_text   = ""
    plan_steps  = []
    task_type   = "MIXED"    # default — safe for both CREATE and MIXED tasks

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
        logger.info("[PLANNER] Trivial query fast-path — skipping sifter LLM call.")
    else:
        try:
            classification_prompt = build_dynamic_sifter_prompt(prompt, all_mcp_tools, history)

            sifter_hash = hashlib.sha256(
                f"{prompt.lower().strip()}:{','.join(sorted(all_mcp_tools.keys()))}".encode()
            ).hexdigest()[:32]

            cached_sifter = await cache.get_sifter(sifter_hash)
            if cached_sifter:
                category    = cached_sifter.get("category", "COMPLEX")
                tool_groups = cached_sifter.get("tool_groups", [])
                task_type   = cached_sifter.get("task_type", "MIXED")
                logger.info(f"[PLANNER] Sifter cache hit: {category} | {task_type} | {tool_groups}")
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
                    task_type   = meta.get("task_type", "MIXED").strip().upper()
                    if task_type not in ("CREATE", "SEARCH", "MIXED"):
                        task_type = "MIXED"
                    await cache.set_sifter(sifter_hash, {
                        "category": category,
                        "tool_groups": tool_groups,
                        "task_type": task_type,
                    })
                except Exception:
                    category    = "COMPLEX"
                    tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]
                    task_type   = "MIXED"

                logger.info(f"[PLANNER] Sifter decision: {category} | task_type={task_type} | {tool_groups}")

            # Real-time keyword override
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
                logger.info(f"[PLANNER] Real-time keyword override → {category} | {tool_groups}")

            if "CONVERSATIONAL" in category:
                needs_plan = needs_tools = False
                plan_text = "Pure conversational small talk."
            elif "DIRECT_TOOL" in category:
                needs_plan  = False
                needs_tools = True
                plan_text = "Direct tool execution."
            else:
                needs_plan = needs_tools = True

            # Multi-service ambiguity check
            _mcp_groups = [g for g in tool_groups if g != "Local"]
            if len(_mcp_groups) > 1 and "CONVERSATIONAL" not in category:
                _explicit_mentions = [g for g in _mcp_groups if g.lower() in prompt.lower()]
                if not _explicit_mentions:
                    service_list = ", ".join(_mcp_groups)
                    clarify_msg = (
                        f"I found multiple services that could help — {service_list}. "
                        "Which one should I use?"
                    )
                    logger.info(f"[PLANNER] Ambiguous query — clarification: {_mcp_groups}")
                    yield f"event: clarification\ndata: {json.dumps({'text': clarify_msg, 'options': _mcp_groups})}\n\n"
                    return

        except Exception as sifter_err:
            logger.error(f"[PLANNER ERROR] Sifter failed: {sifter_err}")
            needs_plan = needs_tools = True
            tool_groups = list(state.get("mcp_servers", {}).keys()) + ["Local"]
            task_type = "MIXED"

    # ── 3. Resolve tools ───────────────────────────────────────────────────────
    turn_call_counter: Dict[str, int] = {}
    tools = []

    if needs_tools:
        for name in tool_groups:
            for server_name in all_mcp_tools:
                if name.lower() in server_name.lower() or server_name.lower() in name.lower():
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
            yield "event: thought\ndata: {\"text\": \"🔍 Selecting best tools for this task...\"}\n\n"
            rag_task = asyncio.create_task(
                retrieve_relevant_tools(prompt, tools, top_k=8)
            )
            while not rag_task.done():
                yield "event: keepalive\ndata: {}\n\n"
                await asyncio.sleep(2)
            tools = rag_task.result()

    # ── 4. Strategic Planning ─────────────────────────────────────────────────
    if needs_plan:
        logger.info("[PLANNER] Running strategic planning...")
        yield f"event: thought\ndata: {json.dumps({'text': '🧠 Building execution plan...'})}\n\n"

        try:
            planning_system_prompt = build_dynamic_planner_prompt(prompt, state, all_mcp_tools)
            planning_msgs = [SystemMessage(content=planning_system_prompt)]
            for h in (compressed_history if is_new_thread else history[-3:]):
                r = h.get("role"); c = h.get("content", "")
                if r == "user":       planning_msgs.append(HumanMessage(content=c))
                elif r == "assistant": planning_msgs.append(AIMessage(content=c))
            planning_msgs.append(HumanMessage(
                content=f"Generate a strategic execution plan for: '{prompt}'"
            ))

            plan_response = await cheap_llm.ainvoke(planning_msgs)
            if getattr(plan_response, "usage_metadata", None):
                total_input_tokens  += plan_response.usage_metadata.get("input_tokens", 0)
                total_output_tokens += plan_response.usage_metadata.get("output_tokens", 0)

            raw_plan   = plan_response.content if hasattr(plan_response, "content") else str(plan_response)
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
            plan_text  = "Step 1: Directly process request using available tools."
            plan_steps = [{"step": 1, "tool": "auto", "purpose": plan_text}]

    # ── 5. Pin plan tools + task-type filtering ────────────────────────────────
    if plan_steps and tools:
        # Normalised pin: fuzzy-match plan tool names against full pool
        pinned_norms   = {_norm(s.get("tool", "")) for s in plan_steps if s.get("tool") and s.get("tool") != "auto"}
        active_norms   = {_norm(getattr(t, "name", "")) for t in tools}
        full_pool      = [t for lst in all_mcp_tools.values() for t in lst]

        for t in full_pool:
            tname = getattr(t, "name", "")
            tnorm = _norm(tname)
            if tnorm in pinned_norms and tnorm not in active_norms:
                logger.info(f"[TOOL RAG] Pinning plan-required tool: {tname}")
                tools.append(wrap_tool_with_coercion(t, call_counter=turn_call_counter))
                active_norms.add(tnorm)

        # Remove search-only tools when destination server has a creation tool planned
        tools = _filter_tools_by_task_type(tools, task_type, plan_steps)

    # ── 6. Build system prompt ─────────────────────────────────────────────────
    base_prompt = build_system_prompt(state, active_groups=tool_groups)
    anti_loop   = _build_anti_loop_instructions(tools, plan_steps=plan_steps)

    _search_instruction = (
        "After calling web_search or web_scrape, present results as a clean markdown "
        "summary — highlight key facts, dates, and names. Never dump raw snippets.\n\n"
    )

    if needs_plan:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"{_search_instruction}"
            f"### Execution Plan — follow steps in this exact order:\n{plan_text}\n\n"
            "Execute steps using your tools. When scraping, extract details via LLM context.\n"
            f"{anti_loop}"
        )
    elif needs_tools:
        system_prompt = (
            f"{base_prompt}\n\n"
            f"{_search_instruction}"
            "Execute required tools immediately. If tools are needed, call them. "
            "Otherwise give a clear direct response.\n"
            f"{anti_loop}"
        )
    else:
        system_prompt = (
            "You are Antigravity, a friendly and helpful AI assistant. "
            "Respond to the user's message in a warm, conversational tone."
        )

    # ── 7. Execute agent ───────────────────────────────────────────────────────
    logger.info(
        f"[AGENT] Streaming thread='{thread_id}', tools={len(tools)}, "
        f"task_type={task_type}, recursion_limit={AGENT_RECURSION_LIMIT}"
    )
    if needs_plan:
        yield f"event: thought\ndata: {json.dumps({'text': '⚡ Executing plan...'})}\n\n"

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

            # Pre-flight: heal dangling tool calls
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
                                if role == "user":      msgs.append(HumanMessage(content=content))
                                elif role == "assistant": msgs.append(AIMessage(content=content))
                            msgs.append(HumanMessage(content=prompt))
                            _INITIALIZED_THREADS.add(thread_id)
                except Exception as pre_err:
                    logger.warning(f"[AGENT] Pre-flight check failed (non-fatal): {pre_err}")

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
        yield f"event: thought\ndata: {json.dumps({'text': '✅ Done!'})}\n\n"

    except Exception as e:
        err_str = str(e)
        if "AIMessages with tool_calls" in err_str or "INVALID_CHAT_HISTORY" in err_str:
            logger.warning(
                f"[AGENT] INVALID_CHAT_HISTORY in thread '{thread_id}' — resetting checkpoint."
            )
            _reset_thread_checkpoint(thread_id)
            yield (
                f"event: error\ndata: {json.dumps({'message': 'Your conversation history was reset after an interrupted tool call. Please resend your last message.'})}\n\n"
            )
            return
        if "recursion" in err_str.lower() or "GRAPH_RECURSION_LIMIT" in err_str:
            logger.error(f"[AGENT] Recursion limit ({AGENT_RECURSION_LIMIT}) hit for thread '{thread_id}'.")
            yield (
                f"event: error\ndata: {json.dumps({'message': 'This task required too many steps. Try breaking it into smaller parts or be more specific about which service to use.'})}\n\n"
            )
            return
        logger.error(f"[CHAT ERROR] {e}", exc_info=True)
        from routes.state_routes import unpack_exception
        clean_msg = unpack_exception(e)
        yield f"event: error\ndata: {json.dumps({'message': clean_msg})}\n\n"