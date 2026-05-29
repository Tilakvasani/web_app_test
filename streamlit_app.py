"""
Streamlit Client Application Interface.

Implements the universal Nordic glassmorphic chat interface, connects dynamic tools,
manages session state metrics, and intercepts CLI slash command autocompletion logic.
"""

import json
import uuid
import httpx
import streamlit as st
from urllib.parse import quote

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="MCP Chat",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Using default Streamlit theme styling

BACKEND_URL = "http://localhost:8000"
query_params = st.query_params

# ── State Synchronization with FastAPI ─────────────────────────────────────────
def sync_state():
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/state", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.mcp_servers = data.get("mcp_servers", {})
            st.session_state.loaded_docs = data.get("loaded_docs", {})
            st.session_state.user_name = data.get("user_name", "Tilak")
            st.session_state.uploaded_files = data.get("uploaded_files", [])
        else:
            st.error(f"Backend state error ({resp.status_code}): {resp.text}")
    except Exception as e:
        st.error(f"⚠️ FastAPI Backend Server is Offline! Make sure web_app.py is running on {BACKEND_URL}. Error: {e}")

def load_session_data():
    """Initializes dynamic token metrics for the local conversation in-memory."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_usage" not in st.session_state or not isinstance(st.session_state.token_usage, dict) or "session_total_tokens" not in st.session_state.token_usage:
        st.session_state.token_usage = {
            "session_total_tokens": 0,
            "session_input_tokens": 0,
            "session_output_tokens": 0,
            "last_turn_total_tokens": 0,
            "last_turn_input_tokens": 0,
            "last_turn_output_tokens": 0,
            "backend_input_tokens": 0,
            "backend_output_tokens": 0,
            "optimization_count": 0
        }

# Initialize local session states
if "mcp_servers" not in st.session_state:
    st.session_state.mcp_servers = {}
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "Tilak"
if "history" not in st.session_state:
    st.session_state.history = []
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "session_total_tokens": 0,
        "session_input_tokens": 0,
        "session_output_tokens": 0,
        "last_turn_total_tokens": 0,
        "last_turn_input_tokens": 0,
        "last_turn_output_tokens": 0,
        "backend_input_tokens": 0,
        "backend_output_tokens": 0,
        "optimization_count": 0
    }
if "oauth_flow_pending" not in st.session_state:
    st.session_state.oauth_flow_pending = None

# Stable conversation identity for LangGraph MemorySaver.
# Generated once per browser session; regenerated after /clear so the next
# conversation starts with a clean checkpoint.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Proactively sync active sessions
sync_state()
load_session_data()

# ── OAuth Redirect Callback Alert Handler ────────────────────────────────────
if "oauth_success" in query_params:
    st.toast("✅ Successfully authorized and connected dynamic server!")
    st.query_params.clear()
    sync_state()
    load_session_data()
    st.rerun()
elif "oauth_error" in query_params:
    st.error(f"OAuth Flow Failed: {query_params['oauth_error']}")
    st.query_params.clear()
    st.rerun()

# ── Sidebar Resource Manager ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ MCP Chat")
    st.caption("Universal Model Context Protocol Platform")
    
    # Real-time User Name Editor
    old_user_name = st.session_state.user_name
    new_user_name = st.text_input("Your Name / Client Name", value=st.session_state.user_name)
    if new_user_name != old_user_name:
        try:
            resp = httpx.post(f"{BACKEND_URL}/api/user_name", json={"user_name": new_user_name.strip()}, timeout=5.0)
            if resp.status_code == 200:
                st.session_state.user_name = new_user_name.strip()
                st.toast(f"Updated user identity to: {new_user_name}")
            else:
                st.error("Failed to update user identity on backend.")
        except Exception as e:
            st.error(f"Backend offline: {e}")
        
    st.divider()
    
    # Render Token Tracker Stats
    token_stats = st.session_state.get("token_usage", {})
    session_total = token_stats.get("session_total_tokens", 0)
    last_turn_total = token_stats.get("last_turn_total_tokens", 0)
    optimization_count = token_stats.get("optimization_count", 0)
    
    st.markdown("**Estimated Token Usage**")
    cols_tokens = st.columns(2)
    with cols_tokens[0]:
        st.metric("Last Turn", f"{last_turn_total:,}", help="LLM tokens used in the most recent turn (input + output).")
    with cols_tokens[1]:
        st.metric("Session Total", f"{session_total:,}", help="Cumulative LLM tokens used during this entire session.")
        
    st.caption("This tracks LLM tokens by one input with also full session LLM tokens.")
    st.caption(
        f"Session: Input {token_stats.get('session_input_tokens', 0):,} | Output {token_stats.get('session_output_tokens', 0):,}\n\n"
        f"Last Turn: Input {token_stats.get('last_turn_input_tokens', 0):,} | Output {token_stats.get('last_turn_output_tokens', 0):,}"
    )
    if optimization_count > 0:
        st.warning(f"⚙️ Context Auto-Pruned ({optimization_count} times)")
    
    # Calculate status counts
    server_count = len(st.session_state.mcp_servers)
    doc_count = len(st.session_state.loaded_docs)
    total_tools = sum(len(s.get("tools", [])) for s in st.session_state.mcp_servers.values())
    
    # Active status badge
    if server_count + doc_count > 0:
        badge_text = f"● {server_count} Active Server{'s' if server_count!=1 else ''} ({total_tools} Tools)"
        if doc_count > 0:
            badge_text += f" | {doc_count} Doc{'s' if doc_count!=1 else ''}"
        st.success(badge_text)
    else:
        st.info("No resources connected yet")
        
    # Render Connected Servers
    st.subheader("Connected Resources")
    
    if server_count == 0 and doc_count == 0:
        st.info("No servers or documentation pages connected yet. Add one below to unlock tools!")
        
    for name, s in list(st.session_state.mcp_servers.items()):
        with st.expander(f"⬡ {name}", expanded=False):
            st.markdown(f"**URL**: `{s['url']}`")
            st.markdown(f"**Tools ({len(s.get('tools', []))})**:")
            for t in s.get('tools', []):
                st.markdown(f"- `{t}`")
            if st.button(f"🗑️ Remove {name}", key=f"del_srv_{name}", use_container_width=True):
                try:
                    resp = httpx.delete(f"{BACKEND_URL}/api/mcp/server/{quote(name)}", timeout=5.0)
                    if resp.status_code == 200:
                        st.toast(f"Removed server {name}")
                        sync_state()
                        st.rerun()
                    else:
                        st.error(f"Failed to delete server: {resp.text}")
                except Exception as e:
                    st.error(f"Backend error: {e}")
                
    for name, d in list(st.session_state.loaded_docs.items()):
        with st.expander(f"📄 {name}", expanded=False):
            st.markdown(f"**URL**: `{d['url']}`")
            st.markdown("*Active & Searchable*")
            if st.button(f"🗑️ Remove {name}", key=f"del_doc_{name}", use_container_width=True):
                try:
                    resp = httpx.delete(f"{BACKEND_URL}/api/mcp/doc/{quote(name)}", timeout=5.0)
                    if resp.status_code == 200:
                        st.toast(f"Removed document {name}")
                        sync_state()
                        st.rerun()
                    else:
                        st.error(f"Failed to delete document: {resp.text}")
                except Exception as e:
                    st.error(f"Backend error: {e}")
                
    st.divider()
    
    # ── Local Documents Workspace (Uploader) ──────────────────────────────────
    st.subheader("📂 Local Workspace")
    
    uploaded_file = st.file_uploader(
        "Upload local document for analysis",
        type=["txt", "md", "json", "csv", "pdf", "docx"],
        label_visibility="collapsed",
        help="Upload files (CVs, resumes, schemas) to make them searchable and analyzable by the agent."
    )
    if uploaded_file is not None:
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
            with st.spinner(f"📤 Uploading '{uploaded_file.name}' to server..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = httpx.post(f"{BACKEND_URL}/api/upload", files=files, timeout=30.0)
                    if resp.status_code == 200:
                        st.session_state.last_uploaded_name = uploaded_file.name
                        st.toast(f"✅ Successfully uploaded '{uploaded_file.name}'!")
                        sync_state()
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with upload backend: {e}")
                    
    # Render Uploaded Files list
    uploaded_files = st.session_state.get("uploaded_files", [])
    if uploaded_files:
        for f in uploaded_files:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"📄 `{f['name']}` ({f['size'] / 1024:.1f} KB)")
            with cols[1]:
                if st.button("🗑️", key=f"del_file_{f['name']}", help=f"Remove {f['name']}"):
                    try:
                        resp = httpx.delete(f"{BACKEND_URL}/api/upload/{quote(f['name'])}", timeout=5.0)
                        if resp.status_code == 200:
                            st.toast(f"Removed document {f['name']}")
                            if "last_uploaded_name" in st.session_state and st.session_state.last_uploaded_name == f['name']:
                                st.session_state.pop("last_uploaded_name")
                            sync_state()
                            st.rerun()
                        else:
                            st.error(f"Failed to delete document: {resp.text}")
                    except Exception as e:
                        st.error(f"Backend error: {e}")
    else:
        st.caption("No files uploaded to workspace yet.")

    st.divider()
    
    # ── Quick-Connect Service Grid ────────────────────────────────────────────
    st.markdown('<div class="sidebar-subheader" style="font-family: \'Syne\', sans-serif; font-weight: 700; color: #7c6af7; font-size: 14px; margin-bottom: 4px;">⚡ Instant Connect</div>', unsafe_allow_html=True)
    st.caption("Click any app symbol below to authorize and connect its MCP tools dynamically")
    
    # 3-Column Balanced App Grid (3x3 Layout)
    services = [
        {"icon": "📓", "name": "Notion", "url": "https://mcp.notion.com/mcp"},
        {"icon": "📅", "name": "Cal.com", "url": "https://mcp.cal.com/mcp"},
        {"icon": "📁", "name": "G-Drive", "url": "https://drivemcp.googleapis.com/mcp/v1"},
        {"icon": "📧", "name": "Gmail", "url": "https://gmailmcp.googleapis.com/mcp/v1"},
        {"icon": "💬", "name": "G-Chat", "url": "https://chatmcp.googleapis.com/mcp/v1"},
        {"icon": "🗓️", "name": "G-Cal", "url": "https://calendarmcp.googleapis.com/mcp/v1"},
        {"icon": "🟠", "name": "HubSpot", "url": "https://mcp.hubspot.com"},
        {"icon": "🦁", "name": "Zoho", "url": "https://mcpweb-60071985865.zohomcp.in/mcp/a6f901b40a8dbfe4a1ff06b04fc16bdf/message"},
        {"icon": "🟦", "name": "MS Work IQ", "url": "https://mcp.svc.cloud.microsoft/enterprise"}
    ]
    
    cols = st.columns(3)
    for idx, s in enumerate(services):
        col = cols[idx % 3]
        button_label = f"{s['icon']} {s['name']}"
        
        is_connected = s["name"] in st.session_state.mcp_servers
        
        if col.button(button_label, key=f"btn_connect_{s['name']}", use_container_width=True, disabled=is_connected):
            with st.spinner(f"Connecting to {s['name']}..."):
                try:
                    payload = {
                        "url": s["url"],
                        "name": s["name"],
                        "auth_option": "Auto-Discover OAuth (RFC 9470)"
                    }
                    resp = httpx.post(f"{BACKEND_URL}/api/mcp/connect", json=payload, timeout=20.0)
                    if resp.status_code == 200:
                        res_data = resp.json()
                        res_type = res_data.get("type")
                        
                        if res_type == "oauth":
                            auth_url = res_data["auth_url"]
                            st.session_state.oauth_flow_pending = {
                                "url": auth_url,
                                "name": s["name"],
                                "source": "grid"
                            }
                            st.toast(f"🔑 Redirecting to {s['name']} authorization page!")
                            st.rerun()
                        elif res_type == "server":
                            st.toast(f"✅ Connected {s['name']} successfully!")
                            sync_state()
                            st.rerun()
                    else:
                        st.error(f"Failed to connect: {resp.json().get('detail', resp.text)}")
                except Exception as err:
                    st.error(f"Connection failed: {err}")
                    
    # Render pending authorization card under Instant Connect grid (only if source is 'grid')
    if st.session_state.oauth_flow_pending and st.session_state.oauth_flow_pending.get("source") == "grid":
        pending = st.session_state.oauth_flow_pending
        auth_url = pending["url"]
        st.warning(f"🔑 Authorization required to connect secure **{pending['name']}** account.")
        st.link_button(f"⚡ Authorize with {pending['name']}", auth_url, use_container_width=True)
        if st.button("Cancel Connection Flow", key="cancel_oauth_grid", use_container_width=True):
            st.session_state.oauth_flow_pending = None
            st.rerun()

    st.divider()
    
    # ── Custom Resource Connector (Manual Connect) ───────────────────────────
    st.markdown('<div class="sidebar-subheader" style="font-family: \'Syne\', sans-serif; font-weight: 700; color: #7c6af7; font-size: 14px; margin-bottom: 4px;">🔗 Custom Connector</div>', unsafe_allow_html=True)
    st.caption("Manually connect any custom MCP server or documentation link")
    
    custom_name = st.text_input("Resource Name", placeholder="e.g. Custom Server", key="custom_srv_name")
    custom_url = st.text_input("Resource Link / URL", placeholder="https://...", key="custom_srv_url")
    
    if st.button("🔌 Connect Resource", use_container_width=True):
        if not custom_url.strip():
            st.error("Please enter a valid Resource URL!")
        else:
            with st.spinner("Probing and connecting resource..."):
                try:
                    payload = {
                        "url": custom_url.strip(),
                        "name": custom_name.strip() if custom_name.strip() else "",
                        "auth_option": "Auto-Discover OAuth (RFC 9470)"
                    }
                    resp = httpx.post(f"{BACKEND_URL}/api/mcp/connect", json=payload, timeout=25.0)
                    if resp.status_code == 200:
                        res_data = resp.json()
                        res_type = res_data.get("type")
                        
                        if res_type == "oauth":
                            auth_url = res_data["auth_url"]
                            st.session_state.oauth_flow_pending = {
                                "url": auth_url,
                                "name": custom_name.strip() if custom_name.strip() else "Custom Server",
                                "source": "custom"
                            }
                            st.toast("🔑 Redirecting to authorization page!")
                            st.rerun()
                        elif res_type == "server":
                            st.toast("✅ Connected MCP server successfully!")
                            sync_state()
                            st.rerun()
                        elif res_type == "doc":
                            st.toast("✅ Loaded documentation successfully!")
                            sync_state()
                            st.rerun()
                    else:
                        st.error(f"Failed to connect: {resp.json().get('detail', resp.text)}")
                except Exception as err:
                    st.error(f"Connection failed: {err}")
                    
    # Render pending authorization card under Custom Connector (only if source is 'custom')
    if st.session_state.oauth_flow_pending and st.session_state.oauth_flow_pending.get("source") == "custom":
        pending = st.session_state.oauth_flow_pending
        auth_url = pending["url"]
        st.warning(f"🔑 Authorization required to connect secure **{pending['name']}** account.")
        st.link_button(f"⚡ Authorize with {pending['name']}", auth_url, use_container_width=True)
        if st.button("Cancel Connection Flow", key="cancel_oauth_custom", use_container_width=True):
            st.session_state.oauth_flow_pending = None
            st.rerun()

# ── Main Chat Interface ──────────────────────────────────────────────────────
st.title("⬡ Assistant")
st.caption("Ready to execute tasks utilizing loaded server tools and searchable documentation")

# Render active chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Real-time Glassmorphic Slash Command Autocomplete Injection ─────────────
import streamlit.components.v1 as components

components.html("""
<script>
    (function() {
        const commands = [
            { name: "/help", desc: "Show complete guide of available commands" },
            { name: "/clear", desc: "Instantly wipe chat history & reset tokens" },
            { name: "/servers", desc: "Inspect active MCP servers & status" },
            { name: "/tools", desc: "List loaded parameter schemas" },
            { name: "/files", desc: "Explore local workspace files & metadata" },
            { name: "/scrape", desc: "Web scrape or agentic CRM website lookup" },
            { name: "/summarize", desc: "Summarize local document or Notion page" },
            { name: "/export", desc: "Export dynamic Notion page to workspace" },
            { name: "/quick-add", desc: "Natural language schedule/page addition" },
            { name: "/search", desc: "Keyword lookup across loaded documentation" }
        ];

        let activeIndex = 0;
        let visibleCommands = [...commands];
        let userQuery = '';

        function injectAutocompleteMenu() {
            // Traverse parent document since Streamlit runs inside an iframe
            const chatInputContainer = window.parent.document.querySelector('div[data-testid="stChatInput"]');
            const textarea = window.parent.document.querySelector('div[data-testid="stChatInput"] textarea');
            
            if (!chatInputContainer || !textarea) return;
            
            // Check if our menu already exists in the parent container
            let menu = window.parent.document.getElementById('mcp-slash-menu');
            if (!menu) {
                menu = window.parent.document.createElement('div');
                menu.id = 'mcp-slash-menu';
                
                // Add absolute positioning stylesheet
                const style = window.parent.document.createElement('style');
                style.innerText = `
                    #mcp-slash-menu {
                        position: absolute;
                        bottom: calc(100% + 8px);
                        left: 12px;
                        right: 12px;
                        background: rgba(10, 10, 15, 0.96) !important;
                        backdrop-filter: blur(12px) !important;
                        border: 1px solid rgba(124, 106, 247, 0.4) !important;
                        border-radius: 12px !important;
                        box-shadow: 0 -15px 30px rgba(0, 0, 0, 0.6), 0 5px 15px rgba(124, 106, 247, 0.1) !important;
                        z-index: 999999 !important;
                        display: none;
                        max-height: 280px;
                        overflow-y: auto;
                        font-family: 'JetBrains Mono', monospace !important;
                    }
                    .mcp-menu-item {
                        padding: 10px 16px;
                        cursor: pointer;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        border-bottom: 1px solid rgba(30, 30, 46, 0.3);
                        transition: all 0.15s ease;
                    }
                    .mcp-menu-item.active {
                        background: rgba(124, 106, 247, 0.16) !important;
                        border-left: 3px solid #7c6af7 !important;
                        padding-left: 13px;
                    }
                `;
                window.parent.document.head.appendChild(style);
                
                chatInputContainer.style.position = 'relative';
                chatInputContainer.appendChild(menu);
            }

            function renderMenu(filterText) {
                const query = filterText.toLowerCase();
                visibleCommands = query === "/" 
                    ? commands 
                    : commands.filter(c => c.name.startsWith(query));
                
                if (visibleCommands.length === 0) {
                    menu.style.display = 'none';
                    return;
                }
                
                // Keep active index in bounds
                if (activeIndex >= visibleCommands.length) {
                    activeIndex = 0;
                } else if (activeIndex < 0) {
                    activeIndex = visibleCommands.length - 1;
                }
                
                menu.innerHTML = '';
                visibleCommands.forEach((cmd, idx) => {
                    const item = window.parent.document.createElement('div');
                    item.className = 'mcp-menu-item' + (idx === activeIndex ? ' active' : '');
                    
                    const nameSpan = window.parent.document.createElement('span');
                    nameSpan.innerText = cmd.name;
                    nameSpan.style.cssText = 'color: #7c6af7; font-weight: 700; font-size: 11px;';
                    
                    const descSpan = window.parent.document.createElement('span');
                    descSpan.innerText = cmd.desc;
                    descSpan.style.cssText = 'color: #8888a0; font-size: 9px; margin-left: 10px; text-align: right;';
                    
                    item.appendChild(nameSpan);
                    item.appendChild(descSpan);
                    
                    // Mouse selection
                    item.onmouseenter = () => {
                        activeIndex = idx;
                        const allItems = menu.querySelectorAll('.mcp-menu-item');
                        allItems.forEach((el, elIdx) => {
                            if (elIdx === idx) el.classList.add('active');
                            else el.classList.remove('active');
                        });
                    };
                    
                    item.onmousedown = (e) => {
                        e.preventDefault(); // Prevents text box from losing focus
                        e.stopPropagation(); // Prevents outside dismiss trigger
                        selectCommand(cmd.name);
                    };
                    
                    menu.appendChild(item);

                    // Ensure navigated items are smoothly scrolled into view inside the glassmorphic menu
                    if (idx === activeIndex) {
                        setTimeout(() => {
                            item.scrollIntoView({ block: 'nearest' });
                        }, 0);
                    }
                });
                
                menu.style.display = 'block';
            }

            function selectCommand(cmdName) {
                const needsArgs = ["/scrape", "/summarize", "/export", "/quick-add", "/search"].includes(cmdName);
                textarea.value = cmdName + (needsArgs ? " " : "");
                
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                textarea.focus();
                menu.style.display = 'none';
            }

            // Real-time input listener
            textarea.addEventListener('input', function() {
                const val = textarea.value.trim();
                userQuery = val;
                activeIndex = 0;
                
                if (val.startsWith("/")) {
                    renderMenu(val);
                } else {
                    menu.style.display = 'none';
                }
            });

            // Keyboard navigation listener (Up, Down, Enter, Escape)
            textarea.addEventListener('keydown', function(e) {
                if (menu.style.display === 'block') {
                    if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        activeIndex = (activeIndex + 1) % visibleCommands.length;
                        renderMenu(userQuery);
                    } else if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        activeIndex = (activeIndex - 1 + visibleCommands.length) % visibleCommands.length;
                        renderMenu(userQuery);
                    } else if (e.key === 'Enter') {
                        if (visibleCommands.length > 0) {
                            e.preventDefault();
                            selectCommand(visibleCommands[activeIndex].name);
                        }
                    } else if (e.key === 'Escape') {
                        menu.style.display = 'none';
                    }
                }
            });

            // Dismiss menu on outside clicks
            window.parent.document.addEventListener('click', function(e) {
                if (!chatInputContainer.contains(e.target)) {
                    menu.style.display = 'none';
                }
            });
        }

        // Periodically run check to capture Streamlit renders without delays
        setInterval(injectAutocompleteMenu, 1000);
    })();
</script>
""", height=0, width=0)

# User Chat Input
prompt = st.chat_input("Ask a question, request an action, or call a tool...")

if prompt:
    # ── 1. Intercept CLI Slash Commands (Direct sub-second execution) ─────────
    if prompt.startswith("/"):
        # Check for /clear shortcut
        if prompt.strip().lower() == "/clear":
            # Wipe backend LangGraph checkpoint so old memory is truly gone
            try:
                httpx.delete(
                    f"{BACKEND_URL}/api/clear_thread/{st.session_state.thread_id}",
                    timeout=5.0,
                )
            except Exception:
                pass  # Best-effort; frontend clear still proceeds
            # Issue a fresh thread_id so the next message starts a clean slate
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.session_state.token_usage = {
                "session_total_tokens": 0,
                "session_input_tokens": 0,
                "session_output_tokens": 0,
                "last_turn_total_tokens": 0,
                "last_turn_input_tokens": 0,
                "last_turn_output_tokens": 0,
                "backend_input_tokens": 0,
                "backend_output_tokens": 0,
                "optimization_count": 0
            }
            st.toast("🧹 Conversation history and token meters cleared!")
            st.rerun()
            
        # Add command to local history
        st.session_state.history.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        # Render assistant loader and execute call
        with st.chat_message("assistant"):
            status_ph = st.empty()
            with status_ph.status("⚡ Executing CLI Command...", expanded=True) as status_box:
                try:
                    resp = httpx.post(f"{BACKEND_URL}/api/commands/execute", json={"command": prompt}, timeout=60.0)
                    if resp.status_code == 200:
                        res_data = resp.json()
                        content = res_data.get("content", "")
                        
                        # Handle backend-triggered clear actions
                        if res_data.get("action") == "clear":
                            st.session_state.history = []
                            st.session_state.token_usage = {
                                "session_total_tokens": 0,
                                "session_input_tokens": 0,
                                "session_output_tokens": 0,
                                "last_turn_total_tokens": 0,
                                "last_turn_input_tokens": 0,
                                "last_turn_output_tokens": 0,
                                "backend_input_tokens": 0,
                                "backend_output_tokens": 0,
                                "optimization_count": 0
                            }
                            st.toast("🧹 Conversation and token metrics cleared!")
                            st.rerun()
                            
                        # Append backend output to history
                        st.session_state.history.append({"role": "assistant", "content": content})
                        status_box.update(label="✅ Command Completed", state="complete")
                        st.markdown(content)
                    else:
                        err_msg = resp.json().get("detail", resp.text)
                        st.error(f"Command execution failed: {err_msg}")
                        status_box.update(label="❌ Command Failed", state="error")
                except Exception as cmd_err:
                    st.error(f"Error communicating with backend command route: {cmd_err}")
                    status_box.update(label="❌ Connection Offline", state="error")
            
            st.rerun()

    # ── 2. Standard Conversational Queries (Plan-and-Execute Flow) ───────────
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        
        # Calculate dynamic input tokens for this turn
        turn_input_tokens = (len(prompt) + sum(len(m["content"]) for m in st.session_state.history[:-1])) // 4
        st.session_state.token_usage["last_turn_input_tokens"] = turn_input_tokens
        st.session_state.token_usage["session_input_tokens"] += turn_input_tokens
        
        # Render user prompt
        st.chat_message("user").markdown(prompt)
        
        # Render assistant container
        with st.chat_message("assistant"):
            plan_placeholder = st.empty()
            status_placeholder = st.empty()
            text_placeholder = st.empty()
            
            full_response = ""
            active_status = None
            
            try:
                payload = {
                    "prompt": prompt,
                    "history": st.session_state.history[:-1],
                    "thread_id": st.session_state.thread_id,
                }
                
                # Initiate HTTP/SSE streaming connection to FastAPI
                with httpx.stream("POST", f"{BACKEND_URL}/api/chat", json=payload, timeout=60.0) as r:
                    current_event = None
                    
                    for line in r.iter_lines():
                        if not line.strip():
                            continue
                            
                        if line.startswith("event:"):
                            current_event = line.replace("event:", "").strip()
                        elif line.startswith("data:") and current_event:
                            data_str = line.replace("data:", "").strip()
                            try:
                                data_obj = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                                
                            # Process SSE chunks
                            if current_event == "thought":
                                text = data_obj.get("text", "")
                                
                                # Intercept and render the strategic planning TODO plan
                                if "📋 **Strategic Execution Plan Created**:" in text:
                                    if not active_status:
                                        active_status = status_placeholder.status("🧠 Agent is thinking...")
                                    active_status.write("📋 Strategic execution plan compiled successfully.")
                                elif "Completed successfully" in text:
                                    if active_status:
                                        active_status.update(label="✅ Completed successfully", state="complete")
                                        active_status = None
                                else:
                                    if not active_status:
                                        active_status = status_placeholder.status("🧠 Agent is thinking...")
                                    active_status.write(f"💭 {text}")
                                    
                            elif current_event == "tool_call":
                                name = data_obj.get("name", "")
                                args = data_obj.get("args", {})
                                if not active_status:
                                    active_status = status_placeholder.status("🧠 Agent is thinking...")
                                active_status.write(f"🛠️ **Calling tool**: `{name}`")
                                active_status.code(json.dumps(args, indent=2))
                                
                            elif current_event == "tool_output":
                                name = data_obj.get("name", "")
                                content = data_obj.get("content", "")
                                if not active_status:
                                    active_status = status_placeholder.status("🧠 Agent is thinking...")
                                active_status.write(f"📥 **Tool output** (`{name}`):")
                                active_status.code(str(content)[:1200])
                                
                            elif current_event == "content":
                                text = data_obj.get("text", "")
                                if active_status:
                                    active_status.update(label="✅ Thought process completed", state="complete")
                                    active_status = None
                                
                                full_response += text
                                text_placeholder.markdown(full_response)
                                
                            elif current_event == "token_usage":
                                backend_input = data_obj.get("input_tokens", 0)
                                backend_output = data_obj.get("output_tokens", 0)
                                st.session_state.token_usage["backend_input_tokens"] = backend_input
                                st.session_state.token_usage["backend_output_tokens"] = backend_output
                                
                            elif current_event == "error":
                                err_msg = data_obj.get("message", "")
                                st.error(f"Error executing chat request: {err_msg}")
                                if active_status:
                                    active_status.update(label="❌ Encountered error during execution", state="error")
                                    
                # Append final results to history
                st.session_state.history.append({"role": "assistant", "content": full_response})
                
                backend_input = st.session_state.token_usage.get("backend_input_tokens", 0)
                backend_output = st.session_state.token_usage.get("backend_output_tokens", 0)
                
                if backend_input > 0 or backend_output > 0:
                    # Correct the preliminary session input estimate by replacing it with actuals
                    st.session_state.token_usage["session_input_tokens"] -= st.session_state.token_usage["last_turn_input_tokens"]
                    
                    st.session_state.token_usage["last_turn_input_tokens"] = backend_input
                    st.session_state.token_usage["last_turn_output_tokens"] = backend_output
                    st.session_state.token_usage["last_turn_total_tokens"] = backend_input + backend_output
                    
                    st.session_state.token_usage["session_input_tokens"] += backend_input
                    st.session_state.token_usage["session_output_tokens"] += backend_output
                    st.session_state.token_usage["session_total_tokens"] = st.session_state.token_usage["session_input_tokens"] + st.session_state.token_usage["session_output_tokens"]
                else:
                    # Fallback to estimate if backend did not report tokens
                    turn_output_tokens = len(full_response) // 4
                    st.session_state.token_usage["last_turn_output_tokens"] = turn_output_tokens
                    st.session_state.token_usage["session_output_tokens"] += turn_output_tokens
                    st.session_state.token_usage["last_turn_total_tokens"] = st.session_state.token_usage["last_turn_input_tokens"] + turn_output_tokens
                    st.session_state.token_usage["session_total_tokens"] = st.session_state.token_usage["session_input_tokens"] + st.session_state.token_usage["session_output_tokens"]
                
                # Clear temporary backend counters
                st.session_state.token_usage["backend_input_tokens"] = 0
                st.session_state.token_usage["backend_output_tokens"] = 0
                st.rerun()
                
            except Exception as chat_err:
                st.error(f"Error communicating with assistant backend: {chat_err}")
                if active_status:
                    active_status.update(label="❌ Encountered error during execution", state="error")