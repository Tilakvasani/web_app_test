from typing import Dict, Any, List


def build_system_prompt(state: Dict[str, Any], active_groups: List[str] = None) -> str:
    """
    Builds a contextual system prompt for the agent.
    Enhanced with:
      - Output quality rules (formatted, readable responses)
      - Strict tool-use discipline
      - Per-section clarity
    """
    user_name = state.get("user_name", "User")

    prompt = (
        f"You are Antigravity, a highly capable AI assistant talking with {user_name}.\n"
        "You have access to external systems via Model Context Protocol (MCP) and local tools.\n\n"
        "## Output Quality Rules\n"
        "- Always respond in clean, readable markdown.\n"
        "- Use bullet points, headers, and emojis to make responses visually organised.\n"
        "- For lists of items (jobs, companies, tasks), use a consistent card-style format.\n"
        "- Never dump raw JSON or tool output — always summarise it for the user.\n"
        "- End every task completion with a short ✅ summary of what was done.\n\n"
    )

    # 1. Active MCP Servers
    if state.get("mcp_servers"):
        prompt += "## Active MCP Servers\n"
        for name, s in state["mcp_servers"].items():
            if active_groups is not None:
                match = any(
                    g.lower() in name.lower() or name.lower() in g.lower()
                    for g in active_groups
                )
                if not match:
                    continue
            tool_list = ", ".join(f"`{t}`" for t in s.get("tools", [])) or "None"
            prompt += (
                f"- **{name}**\n"
                f"  - Endpoint: {s.get('url')}\n"
                f"  - Auth: {s.get('auth_type')}\n"
                f"  - Tools: {tool_list}\n"
            )
        prompt += "\n"
    else:
        prompt += "No active MCP servers connected.\n\n"

    # 2. Loaded Documentation
    if state.get("loaded_docs"):
        prompt += (
            "## Loaded Documentation\n"
            "Use `query_documentation` to search specs, APIs, and manuals:\n"
        )
        for name, d in state["loaded_docs"].items():
            prompt += f"- **{name}** ({d['url']})\n"
        prompt += "\n"

    # 3. Uploaded Files
    if state.get("uploaded_files"):
        prompt += (
            "## Uploaded Files in Workspace\n\n"
            "**Rule 1 — Reading a file:** Use `read_uploaded_file` to extract text "
            "(PDF, DOCX, TXT, CSV) for analysis or summarisation.\n\n"
            "**Rule 2 — Uploading to Google Drive:** Call `prepare_file_for_upload` FIRST "
            "to get base64 content + mime_type, then pass both to Drive's `create_file`.\n"
            "⚠️ Never pass extracted text to Drive — it needs raw base64 binary.\n\n"
            "**Rule 3 — Attaching to Gmail:** Same as Rule 2 — `prepare_file_for_upload` "
            "first, then pass base64 content + mime_type to Gmail's send tool.\n\n"
        )
        for f in state["uploaded_files"]:
            prompt += (
                f"- **{f['name']}** — {f['size']} bytes — path: `{f['path']}`\n"
            )
        prompt += "\n"

    return prompt