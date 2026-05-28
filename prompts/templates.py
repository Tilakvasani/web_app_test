from typing import Dict, Any, List

def build_system_prompt(state: Dict[str, Any], active_groups: List[str] = None) -> str:
    """
    Compiles a highly contextual, dynamic system prompt instructing the agent on
    active MCP servers, loaded documentation resources, and uploaded files.
    """
    user_name = state.get("user_name", "Tilak")
    
    prompt = (
        f"You are Antigravity, a highly capable AI assistant talking with {user_name}.\n"
        "You are equipped with access to external systems via Model Context Protocol (MCP).\n"
        "Below is the list of currently connected and active MCP servers, their endpoints, and their tools. "
        "When the user requests an action, use the corresponding tool from the appropriate server.\n\n"
    )
    
    # 1. Active MCP Servers Section
    if state.get("mcp_servers"):
        prompt += "### Active MCP Servers:\n"
        for name, s in state["mcp_servers"].items():
            if active_groups is not None:
                server_match = False
                for group in active_groups:
                    if group.lower() in name.lower() or name.lower() in group.lower():
                        server_match = True
                        break
                if not server_match:
                    continue  # Skip describing unconnected/skipped servers to save prompt tokens
            
            tool_list = ", ".join([f"`{t}`" for t in s.get("tools", [])]) or "None"
            prompt += (
                f"- **Server Name**: {name}\n"
                f"  - **Endpoint/URL**: {s.get('url')}\n"
                f"  - **Authentication**: {s.get('auth_type')}\n"
                f"  - **Registered Tools**: {tool_list}\n"
            )
        prompt += "\n"
    else:
        prompt += "No active MCP servers are currently connected.\n\n"

    # 2. Loaded Documentation Section
    if state.get("loaded_docs"):
        prompt += (
            "### Loaded Documentation Resources:\n"
            "You have access to loaded documentation resources. "
            "Use the `query_documentation` tool to search for detailed specifications, APIs, and manuals:\n"
        )
        for name, d in state["loaded_docs"].items():
            prompt += f"- **{name}** (URL: {d['url']})\n"
        prompt += "\n"

    # 3. Uploaded Files Section
    if state.get("uploaded_files"):
        prompt += "### Uploaded Local Documents:\n"
        prompt += (
            "You have access to the following documents uploaded by the user to the local server workspace.\n\n"

            "**RULE 1 — Reading / Analysing a file:**\n"
            "Use `read_uploaded_file` to extract and read text from a file "
            "(PDF, DOCX, TXT, CSV, etc.) for analysis, summarisation, or answering questions.\n\n"

            "**RULE 2 — Uploading a file to Google Drive:**\n"
            "ALWAYS call `prepare_file_for_upload` FIRST. It returns JSON with:\n"
            "  • `content`   — base64-encoded binary (for PDF/DOCX/images) or plain text (for .txt/.csv)\n"
            "  • `mime_type` — the correct MIME type Google Drive requires\n"
            "  • `encoding`  — either 'base64' or 'text'\n"
            "Then pass those values directly to Google Drive's `create_file` tool.\n"
            "⚠️ NEVER call `read_uploaded_file` and pass its extracted text to Google Drive's "
            "`create_file` — that causes 'invalid document' errors because Google Drive expects "
            "raw binary content (base64), NOT extracted text.\n\n"

            "**RULE 3 — Attaching a file to Gmail:**\n"
            "Call `prepare_file_for_upload` first to get the base64 content and mime_type, "
            "then pass them as the attachment fields in Gmail's send tool.\n"
        )
        for f in state["uploaded_files"]:
            prompt += (
                f"- **Filename**: {f['name']}\n"
                f"  - **Path**: {f['path']}\n"
                f"  - **Size**: {f['size']} bytes\n"
            )
        prompt += "\n"
        
    return prompt